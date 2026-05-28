"""
Alignment dataset: loads trajectory features and a text string per clip for
contrastive alignment training.

traj_type options:
  "trajectory"     →  rot6D(6) + rel_trans(3) = 9D × N frames
  "velocity"       →  trans_vel(3) + rot_vel(3) = 6D × (N-1) steps
  "direction"      →  trans_dir(3) + rot_dir(3) = 6D × (N-1) steps
  "speed"          →  [log_trans_speed, log_rot_speed] = 2D × (N-1) steps
  "direction+speed"→  direction(6) + log_speeds(2) = 8D × (N-1) steps

text_type options:
  "motion"               →  camera motion caption (from caption_cam+rot/*.txt)
  "logline_script"       →  full scene description
  "macro_type"           →  spatial openness label
  "setting_class"        →  environment type label
  "subject_composition"  →  what is filmed label
  "genre_vibe"           →  emotional tone label

Each item returns:
  traj_feat:   (max_vel_len, D)   padded trajectory features
  seq_len:     int                actual velocity length (= clip_frames - 1)
  text:        str                text string for this clip
  clip_id:     str
  dataset:     str
"""

import json
import numpy as np
from pathlib import Path
from typing import List

import torch
from torch.utils.data import Dataset, Sampler

from cinegen.utils.pose_utils import np_matrices_to_velocity


LOG_SPEED_EPS = 1e-6

ASPECT_KEYS = [
    "logline_script",
    "macro_type",
    "setting_class",
    "subject_composition",
    "genre_vibe",
]

# Human-readable labels used when building composite text strings
ASPECT_LABEL = {
    "setting_class":       "Setting",
    "subject_composition": "Subject",
    "genre_vibe":          "Tone",
    "macro_type":          "Space",
    "logline_script":      "",        # leading field — no prefix
}

VALID_TEXT_TYPES  = frozenset(["motion"] + ASPECT_KEYS)


def is_valid_text_type(text_type: str) -> bool:
    """Accept single types OR '+'-joined composites (may include 'motion')."""
    if text_type in VALID_TEXT_TYPES:
        return True
    parts = text_type.split("+")
    return all(p in ASPECT_KEYS or p == "motion" for p in parts)


def build_composite_text(text_type: str, aspects: dict) -> str:
    """
    Compose a single text string from one or more aspect fields.

    Single types (e.g. 'motion', 'logline_script') are returned as-is.
    Composite types (e.g. 'logline_script+setting_class') are assembled as:
        "<logline>. Setting: <setting_class>. Subject: <subject_composition>."

    Empty fields are silently skipped so missing metadata doesn't pollute the text.
    """
    parts = text_type.split("+")
    segments = []
    for part in parts:
        val = aspects.get(part, "").strip()
        if not val:
            continue
        label = ASPECT_LABEL.get(part, part)
        segments.append(f"{label}: {val}" if label else val.rstrip("."))
    return ". ".join(segments)
VALID_TRAJ_TYPES  = frozenset([
    "trajectory", "velocity", "direction", "speed", "direction+speed"
])

TRAJ_DIM = {
    "trajectory":      9,
    "velocity":        6,
    "direction":       6,
    "speed":           2,
    "direction+speed": 8,
}
AVAILABLE_DATASETS = ["cinetechbench", "movieshots", "shotbench", "vadb", "condensedmovies"]


def _extract_aspects(cinematic_data: dict) -> dict:
    ctx = cinematic_data.get("spatial_context", {})
    return {
        "logline_script":      cinematic_data.get("logline_script", ""),
        "macro_type":          ctx.get("macro_type", ""),
        "setting_class":       ctx.get("setting_class", ""),
        "subject_composition": cinematic_data.get("subject_composition", ""),
        "genre_vibe":          cinematic_data.get("genre_vibe", ""),
    }


# Logline keywords that indicate movieclips.com website UI / ad clips
_AD_KEYWORDS = frozenset([
    "movieclips", "website interface", "grid of", "thumbnail",
    "click to", "movieclips.com", "digital interface", "collage of",
])

def _is_ad_clip(logline: str) -> bool:
    """Return True if logline indicates a movieclips.com UI / ad clip."""
    ll = logline.lower()
    return any(kw in ll for kw in _AD_KEYWORDS)


class AlignDataset(Dataset):
    """
    Args:
        root:          Path to DIY_movies root directory.
        datasets:      Sub-dataset names to include.
        split:         'train' or 'val'.
        val_fraction:  Fraction held out for validation.
        max_seq_len:   Maximum number of frames (velocities = max_seq_len - 1).
        traj_type:     Which trajectory feature to load ("direction" or "speed").
        text_type:     Which text to pair with ("motion" or a cinematic aspect key).
        seed:          Random seed for stable train/val split.
    """

    def __init__(
        self,
        root:         str,
        datasets:     List[str]  = AVAILABLE_DATASETS,
        split:        str        = "train",
        val_fraction: float      = 0.1,
        max_seq_len:  int        = 196,
        traj_type:    str        = "direction",
        text_type:    str        = "motion",
        seed:         int        = 42,
        min_aspect_ratio: float  = 1.2,
    ):
        assert traj_type in VALID_TRAJ_TYPES, \
            f"traj_type must be one of {VALID_TRAJ_TYPES}"
        assert is_valid_text_type(text_type), \
            f"text_type '{text_type}' is not valid. Use a single type or '+'-joined aspects."

        self.root      = Path(root)
        self.traj_type = traj_type
        self.text_type = text_type
        # trajectory uses N absolute frames; all velocity-based use N-1 steps
        self.max_len   = max_seq_len if traj_type == "trajectory" else max_seq_len - 1

        # Load aspect ratio cache for horizontal filtering
        aspect_cache = {}
        ar_cache_path = Path(root) / "aspect_ratio_cache.json"
        if ar_cache_path.exists():
            with open(ar_cache_path) as f:
                aspect_cache = json.load(f)

        # Index: list of (dataset_name, clip_id)
        self.items          = []
        self.motion_captions = {}   # clip_id → str
        self.aspects         = {}   # clip_id → dict[aspect_key → str]
        n_ads_filtered = 0
        n_aspect_filtered = 0

        for ds in datasets:
            pose_dir   = self.root / "filtered_pose" / ds
            cap_dir    = self.root / "vipe_results" / ds / "caption_cam+rot"
            jsonl_path = self.root / "captions" / f"{ds}_captions.jsonl"

            if not pose_dir.exists():
                print(f"[AlignDataset] Skipping {ds}: no pose dir at {pose_dir}")
                continue

            # Build cinematic aspects map from JSONL
            aspect_map = {}
            if jsonl_path.exists():
                with open(jsonl_path) as f:
                    for line in f:
                        entry   = json.loads(line)
                        clip_id = Path(entry.get("video_path", "")).stem
                        aspect_map[clip_id] = _extract_aspects(
                            entry.get("cinematic_data", {})
                        )
            else:
                print(f"[AlignDataset] Warning: no captions JSONL for {ds}")

            for npz_path in sorted(pose_dir.glob("*.npz")):
                clip_id  = npz_path.stem
                txt_path = cap_dir / f"{clip_id}.txt"
                if not txt_path.exists():
                    continue

                # Filter ad / UI clips (movieclips.com website overlays)
                aspects = aspect_map.get(clip_id, {k: "" for k in ASPECT_KEYS})
                if _is_ad_clip(aspects.get("logline_script", "")):
                    n_ads_filtered += 1
                    continue

                # Filter non-horizontal clips (vertical / near-square)
                if clip_id in aspect_cache:
                    w, h = aspect_cache[clip_id]
                    if h > 0 and w / h < min_aspect_ratio:
                        n_aspect_filtered += 1
                        continue

                self.items.append((ds, clip_id))
                self.motion_captions[clip_id] = txt_path.read_text().strip()
                self.aspects[clip_id] = aspects

        if n_ads_filtered > 0:
            print(f"[AlignDataset] Filtered {n_ads_filtered} ad/UI clips")
        if n_aspect_filtered > 0:
            print(f"[AlignDataset] Filtered {n_aspect_filtered} non-horizontal clips (aspect ratio < {min_aspect_ratio})")

        # Train / val split (same seed as main dataset for consistency)
        rng   = np.random.default_rng(seed)
        idx   = rng.permutation(len(self.items))
        n_val = max(1, int(len(self.items) * val_fraction))
        idx   = idx[:n_val] if split == "val" else idx[n_val:]
        self.items = [self.items[i] for i in idx]

        print(f"[AlignDataset] {split} | traj={traj_type} text={text_type} "
              f"| {len(self.items)} clips from {datasets}")

    def __len__(self) -> int:
        return len(self.items)

    @property
    def unique_labels(self) -> list:
        """Returns sorted list of unique text labels in this split."""
        return sorted(set(self._get_text(cid) for _, cid in self.items))

    def _load_traj_feat(self, ds: str, clip_id: str) -> tuple[np.ndarray, int]:
        """Load pose NPZ → compute requested representation → padded feature."""
        path     = self.root / "filtered_pose" / ds / f"{clip_id}.npz"
        matrices = np.load(path)["data"]   # (N, 4, 4)

        if self.traj_type == "trajectory":
            R         = matrices[:, :3, :3]
            rot6d     = R[:, :, :2].transpose(0, 2, 1).reshape(-1, 6)  # (N, 6)
            trans     = matrices[:, :3, 3]
            rel_trans = trans - trans[0:1]                              # (N, 3)
            raw        = np.concatenate([rot6d, rel_trans], axis=-1)   # (N, 9)
            actual_len = min(len(raw), self.max_len)
            raw        = raw[:actual_len]
        else:
            td, rd, ts, rs, _ = np_matrices_to_velocity(matrices)
            actual_len = min(len(td), self.max_len)
            if self.traj_type == "velocity":
                tv  = td[:actual_len] * ts[:actual_len, None]
                rv  = rd[:actual_len] * rs[:actual_len, None]
                raw = np.concatenate([tv, rv], axis=-1)                 # (L, 6)
            elif self.traj_type == "direction":
                raw = np.concatenate(
                    [td[:actual_len], rd[:actual_len]], axis=-1)        # (L, 6)
            elif self.traj_type == "speed":
                ts_log = np.log(ts[:actual_len] + LOG_SPEED_EPS)
                rs_log = np.log(rs[:actual_len] + LOG_SPEED_EPS)
                raw    = np.stack([ts_log, rs_log], axis=-1)            # (L, 2)
            else:  # direction+speed
                ts_log = np.log(ts[:actual_len] + LOG_SPEED_EPS)
                rs_log = np.log(rs[:actual_len] + LOG_SPEED_EPS)
                raw    = np.concatenate([
                    td[:actual_len], rd[:actual_len],
                    ts_log[:, None], rs_log[:, None],
                ], axis=-1)                                              # (L, 8)

        D       = TRAJ_DIM[self.traj_type]
        pad_len = self.max_len - actual_len
        feat    = np.concatenate(
            [raw, np.zeros((pad_len, D), dtype=np.float32)], axis=0
        )
        return feat.astype(np.float32), actual_len

    def _get_text(self, clip_id: str) -> str:
        if self.text_type == "motion":
            return self.motion_captions[clip_id]
        # Handle composite types that include 'motion' (e.g. 'motion+logline_script')
        parts = self.text_type.split("+")
        if "motion" in parts:
            segments = []
            for p in parts:
                if p == "motion":
                    mc = self.motion_captions.get(clip_id, "").strip()
                    if mc:
                        segments.append(f"Camera motion: {mc.rstrip('.')}")
                else:
                    val = self.aspects[clip_id].get(p, "").strip()
                    if val:
                        label = ASPECT_LABEL.get(p, p)
                        segments.append(f"{label}: {val}" if label else val.rstrip("."))
            return ". ".join(segments)
        return build_composite_text(self.text_type, self.aspects[clip_id])

    def __getitem__(self, idx: int) -> dict:
        ds, clip_id = self.items[idx]

        feat, seq_len = self._load_traj_feat(ds, clip_id)
        text          = self._get_text(clip_id)

        return {
            "traj_feat": torch.from_numpy(feat),   # (max_vel_len, D)
            "seq_len":   seq_len,                  # int
            "text":      text,                     # str
            "clip_id":   clip_id,
            "dataset":   ds,
        }


def collate_fn(batch: list) -> dict:
    traj_feats = torch.stack([b["traj_feat"] for b in batch])   # (B, T, D)
    seq_lens   = torch.tensor([b["seq_len"]  for b in batch], dtype=torch.long)
    texts      = [b["text"]    for b in batch]
    clip_ids   = [b["clip_id"] for b in batch]
    datasets   = [b["dataset"] for b in batch]
    return {
        "traj_feat": traj_feats,
        "seq_len":   seq_lens,
        "text":      texts,
        "clip_id":   clip_ids,
        "dataset":   datasets,
    }


class ClassBalancedBatchSampler(Sampler):
    """
    Each batch contains exactly one item per unique text class (or
    `batch_size` classes if there are more classes than batch_size).

    This guarantees every off-diagonal pair in the batch is a genuine
    negative — no false-negative masking needed.

    Used automatically when the number of unique labels is small (categorical).
    """

    def __init__(self, dataset: AlignDataset, batch_size: int, seed: int = 42):
        labels = [dataset._get_text(cid) for _, cid in dataset.items]

        # Group item indices by label
        self._class_to_idxs: dict = {}
        for i, lbl in enumerate(labels):
            self._class_to_idxs.setdefault(lbl, []).append(i)

        self._classes    = list(self._class_to_idxs.keys())
        self._n_classes  = len(self._classes)
        self._batch_size = min(batch_size, self._n_classes)
        self._n_items    = len(dataset)
        self._seed       = seed

    @property
    def effective_batch_size(self) -> int:
        return self._batch_size

    def __len__(self) -> int:
        return self._n_items // self._batch_size

    def __iter__(self):
        rng = np.random.default_rng(self._seed)

        # Shuffle item lists within each class at the start of each epoch
        queues = {c: list(rng.permutation(idxs))
                  for c, idxs in self._class_to_idxs.items()}

        for _ in range(len(self)):
            # Pick batch_size classes at random (without replacement per batch)
            chosen = rng.choice(self._classes, size=self._batch_size, replace=False)
            batch  = []
            for c in chosen:
                if not queues[c]:
                    queues[c] = list(rng.permutation(self._class_to_idxs[c]))
                batch.append(queues[c].pop())
            yield batch
