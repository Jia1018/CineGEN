"""
Classification dataset: loads 4 trajectory input representations and a
categorical label per clip for classifier training.

traj_type options:
  "trajectory"  →  rot6D (6) + relative translation (3) = 9D × N frames
  "velocity"    →  trans_vel (3) + rot_vel (3)           = 6D × (N-1) steps
  "direction"   →  trans_dir (3) + rot_dir (3)           = 6D × (N-1) steps
  "speed"       →  [log_trans_speed, log_rot_speed]      = 2D × (N-1) steps

text_type options (categorical aspect labels from cinematic_data JSONL):
  "macro_type"           →  spatial openness label
  "setting_class"        →  environment type label
  "subject_composition"  →  what is filmed label
  "genre_vibe"           →  emotional tone label

Each item returns:
  feat:      (max_len, D)   float32, padded trajectory feature
  seq_len:   int            actual valid length
  label:     int            class index (-1 if unseen in training split)
  clip_id:   str
  dataset:   str
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional

import torch
from torch.utils.data import Dataset

from cinegen.utils.pose_utils import np_matrices_to_velocity


LOG_SPEED_EPS = 1e-6

VALID_TRAJ_TYPES = frozenset([
    "trajectory", "velocity", "direction", "speed", "direction+speed"
])
VALID_TEXT_TYPES = frozenset([
    "macro_type", "setting_class", "subject_composition", "genre_vibe"
])
AVAILABLE_DATASETS = ["cinetechbench", "movieshots", "shotbench", "vadb", "condensedmovies"]

# Feature dimensionality for each traj_type
TRAJ_DIM: Dict[str, int] = {
    "trajectory":      9,
    "velocity":        6,
    "direction":       6,
    "speed":           2,
    "direction+speed": 8,   # trans_dir(3) + rot_dir(3) + log_ts(1) + log_rs(1)
}


def _extract_aspects(cinematic_data: dict) -> dict:
    ctx = cinematic_data.get("spatial_context", {})
    return {
        "macro_type":          ctx.get("macro_type", ""),
        "setting_class":       ctx.get("setting_class", ""),
        "subject_composition": cinematic_data.get("subject_composition", ""),
        "genre_vibe":          cinematic_data.get("genre_vibe", ""),
    }


class ClfDataset(Dataset):
    """
    Args:
        root:          Path to DIY_movies root directory.
        datasets:      Sub-dataset names to include.
        split:         'train' or 'val'.
        val_fraction:  Fraction held out for validation.
        max_seq_len:   Maximum number of frames. For trajectory: N frames capped
                       at max_seq_len. For velocity/direction/speed: N-1 steps
                       capped at max_seq_len-1.
        traj_type:     Which trajectory representation to load (one of 4 types).
        text_type:     Which cinematic aspect to use as classification label.
        seed:          Random seed for stable train/val split.
        label2idx:     Optional pre-built label→int mapping (used for val split
                       so it shares the training split's label vocabulary).
    """

    def __init__(
        self,
        root:         str,
        datasets:     List[str]         = AVAILABLE_DATASETS,
        split:        str               = "train",
        val_fraction: float             = 0.1,
        max_seq_len:  int               = 300,
        traj_type:    str               = "direction",
        text_type:    str               = "macro_type",
        seed:         int               = 42,
        label2idx:    Optional[dict]    = None,
    ):
        assert traj_type in VALID_TRAJ_TYPES, \
            f"traj_type must be one of {sorted(VALID_TRAJ_TYPES)}"
        assert text_type in VALID_TEXT_TYPES, \
            f"text_type must be one of {VALID_TEXT_TYPES}"

        self.root        = Path(root)
        self.traj_type   = traj_type
        self.text_type   = text_type
        self.max_seq_len = max_seq_len
        self.max_len     = max_seq_len if traj_type == "trajectory" else max_seq_len - 1
        self.feat_dim    = TRAJ_DIM[traj_type]

        # Index: list of (dataset_name, clip_id)
        self._all_items: List[tuple] = []
        self._aspects:   Dict[str, dict] = {}   # clip_id → {aspect_key → str}

        for ds in datasets:
            pose_dir   = self.root / "filtered_pose" / ds
            jsonl_path = self.root / "captions" / f"{ds}_captions.jsonl"

            if not pose_dir.exists():
                print(f"[ClfDataset] Skipping {ds}: no pose dir at {pose_dir}")
                continue

            aspect_map: Dict[str, dict] = {}
            if jsonl_path.exists():
                with open(jsonl_path) as f:
                    for line in f:
                        entry   = json.loads(line)
                        clip_id = Path(entry.get("video_path", "")).stem
                        aspect_map[clip_id] = _extract_aspects(
                            entry.get("cinematic_data", {})
                        )
            else:
                print(f"[ClfDataset] Warning: no captions JSONL for {ds}")

            for npz_path in sorted(pose_dir.glob("*.npz")):
                clip_id = npz_path.stem
                self._all_items.append((ds, clip_id))
                self._aspects[clip_id] = aspect_map.get(
                    clip_id, {k: "" for k in VALID_TEXT_TYPES}
                )

        # Train / val split — same logic as AlignDataset
        rng   = np.random.default_rng(seed)
        idx   = rng.permutation(len(self._all_items))
        n_val = max(1, int(len(self._all_items) * val_fraction))
        split_idx   = idx[:n_val] if split == "val" else idx[n_val:]
        self.items  = [self._all_items[i] for i in split_idx]

        # Build label vocabulary from this split, or use provided mapping
        if label2idx is not None:
            self.label2idx = label2idx
        else:
            all_labels = sorted(set(
                self._aspects[cid].get(text_type, "")
                for _, cid in self.items
                if self._aspects[cid].get(text_type, "")
            ))
            self.label2idx = {lbl: i for i, lbl in enumerate(all_labels)}

        print(
            f"[ClfDataset] {split} | traj={traj_type} text={text_type} "
            f"| {len(self.items)} clips | {len(self.label2idx)} classes"
        )

    @property
    def num_classes(self) -> int:
        return len(self.label2idx)

    @property
    def label_names(self) -> List[str]:
        """Class names sorted by their label index."""
        return sorted(self.label2idx.keys(), key=lambda x: self.label2idx[x])

    def _load_feat(self, ds: str, clip_id: str) -> tuple:
        path     = self.root / "filtered_pose" / ds / f"{clip_id}.npz"
        matrices = np.load(path)["data"]   # (N, 4, 4)

        if self.traj_type == "trajectory":
            R         = matrices[:, :3, :3]                           # (N, 3, 3)
            rot6d     = R[:, :, :2].transpose(0, 2, 1).reshape(-1, 6) # (N, 6)
            trans     = matrices[:, :3, 3]                            # (N, 3)
            rel_trans = trans - trans[0:1]                            # (N, 3)
            raw        = np.concatenate([rot6d, rel_trans], axis=-1)  # (N, 9)
            actual_len = min(len(raw), self.max_len)
            raw        = raw[:actual_len]
        else:
            td, rd, ts, rs, _ = np_matrices_to_velocity(matrices)
            actual_len = min(len(td), self.max_len)
            if self.traj_type == "velocity":
                tv  = td[:actual_len] * ts[:actual_len, None]
                rv  = rd[:actual_len] * rs[:actual_len, None]
                raw = np.concatenate([tv, rv], axis=-1)               # (L, 6)
            elif self.traj_type == "direction":
                raw = np.concatenate(
                    [td[:actual_len], rd[:actual_len]], axis=-1)      # (L, 6)
            elif self.traj_type == "speed":
                ts_log = np.log(ts[:actual_len] + LOG_SPEED_EPS)
                rs_log = np.log(rs[:actual_len] + LOG_SPEED_EPS)
                raw    = np.stack([ts_log, rs_log], axis=-1)          # (L, 2)
            else:  # direction+speed
                ts_log = np.log(ts[:actual_len] + LOG_SPEED_EPS)
                rs_log = np.log(rs[:actual_len] + LOG_SPEED_EPS)
                raw    = np.concatenate([
                    td[:actual_len], rd[:actual_len],
                    ts_log[:, None], rs_log[:, None],
                ], axis=-1)                                            # (L, 8)

        pad_len = self.max_len - actual_len
        feat    = np.concatenate(
            [raw, np.zeros((pad_len, self.feat_dim), dtype=np.float32)], axis=0
        )
        return feat.astype(np.float32), actual_len

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict:
        ds, clip_id = self.items[idx]
        feat, seq_len = self._load_feat(ds, clip_id)
        label_str     = self._aspects[clip_id].get(self.text_type, "")
        label         = self.label2idx.get(label_str, -1)
        return {
            "feat":    torch.from_numpy(feat),
            "seq_len": seq_len,
            "label":   label,
            "clip_id": clip_id,
            "dataset": ds,
        }


def collate_fn(batch: list) -> dict:
    feats    = torch.stack([b["feat"]    for b in batch])
    seq_lens = torch.tensor([b["seq_len"] for b in batch], dtype=torch.long)
    labels   = torch.tensor([b["label"]  for b in batch], dtype=torch.long)
    return {
        "feat":    feats,
        "seq_len": seq_lens,
        "label":   labels,
        "clip_id": [b["clip_id"] for b in batch],
        "dataset": [b["dataset"] for b in batch],
    }
