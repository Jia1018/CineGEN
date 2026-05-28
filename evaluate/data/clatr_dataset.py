"""
Dataset for CLaTr alignment model training.

Supports multiple trajectory types and text types:
  - traj_type: "trajectory" (9D), "direction+speed" (8D), "direction" (6D), "speed" (2D)
  - text_type: "motion", "logline_script", "motion+logline_script", etc.

Text: CLIP token-level embeddings (77, clip_dim) extracted on-the-fly and cached.
"""

import json
import os
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from typing import List, Optional
from torch.utils.data import Dataset
from transformers import CLIPTokenizer, CLIPTextModel

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from cinegen.utils.pose_utils import np_matrices_to_velocity


CLIP_MODEL_ID = "openai/clip-vit-large-patch14"   # 768-dim text

LOG_SPEED_EPS = 1e-6

ASPECT_KEYS = [
    "logline_script", "macro_type", "setting_class",
    "subject_composition", "genre_vibe",
]
ASPECT_LABEL = {
    "setting_class": "Setting", "subject_composition": "Subject",
    "genre_vibe": "Tone", "macro_type": "Space", "logline_script": "",
}

TRAJ_DIM = {
    "trajectory": 9, "velocity": 6, "direction": 6,
    "speed": 2, "direction+speed": 8,
}


# ---------------------------------------------------------------------------
# Feature extraction helpers
# ---------------------------------------------------------------------------

def matrices_to_9d(matrices: np.ndarray) -> np.ndarray:
    """
    (N, 4, 4) c2w matrices → (N, 9) features:
      rot6D (6): first two cols of R, row-major flattened
      trans  (3): velocity (Δpos), first frame is absolute position
    """
    R = matrices[:, :3, :3]                             # (N, 3, 3)
    t = matrices[:, :3,  3]                             # (N, 3)

    # rot6D: first two columns of R
    rot6d = R[:, :, :2].transpose(0, 2, 1).reshape(-1, 6)   # (N, 6)

    # translation velocity
    vel = np.zeros_like(t)                              # (N, 3)
    vel[0]  = t[0]                                      # first frame: absolute
    vel[1:] = t[1:] - t[:-1]                           # subsequent: delta

    return np.concatenate([rot6d, vel], axis=-1).astype(np.float32)   # (N, 9)


def matrices_to_feat(matrices: np.ndarray, traj_type: str) -> tuple:
    """Convert (N,4,4) c2w matrices to features for any traj_type.
    Returns (feat, actual_len) where feat is (L, D)."""
    if traj_type == "trajectory":
        feat = matrices_to_9d(matrices)
        return feat, len(feat)

    td, rd, ts, rs, _ = np_matrices_to_velocity(matrices)
    L = len(td)
    if traj_type == "velocity":
        tv = td * ts[:, None]
        rv = rd * rs[:, None]
        return np.concatenate([tv, rv], axis=-1).astype(np.float32), L
    elif traj_type == "direction":
        return np.concatenate([td, rd], axis=-1).astype(np.float32), L
    elif traj_type == "speed":
        ts_log = np.log(ts + LOG_SPEED_EPS)
        rs_log = np.log(rs + LOG_SPEED_EPS)
        return np.stack([ts_log, rs_log], axis=-1).astype(np.float32), L
    else:  # direction+speed
        ts_log = np.log(ts + LOG_SPEED_EPS)
        rs_log = np.log(rs + LOG_SPEED_EPS)
        return np.concatenate([
            td, rd, ts_log[:, None], rs_log[:, None],
        ], axis=-1).astype(np.float32), L


def compute_standardization(npz_paths: List[Path], max_seq_len: int = 196):
    """
    Compute mean/std for standardization over all training clips.
    Returns dict with shift_mean/std (first frame) and norm_mean/std (velocity frames).
    """
    first_frames, vel_frames = [], []

    for p in npz_paths:
        try:
            mats = np.load(p)["data"].astype(np.float32)
            feat = matrices_to_9d(mats)                # (N, 9)
            trans = feat[:, 6:]                        # (N, 3)
            first_frames.append(trans[0])
            if len(trans) > 1:
                vel_frames.append(trans[1:])
        except Exception:
            continue

    first_frames = np.stack(first_frames)              # (K, 3)
    vel_frames   = np.concatenate(vel_frames, axis=0)  # (M, 3)

    return {
        "shift_mean": first_frames.mean(axis=0).tolist(),
        "shift_std":  first_frames.std(axis=0).clip(min=1e-6).tolist(),
        "norm_mean":  vel_frames.mean(axis=0).tolist(),
        "norm_std":   vel_frames.std(axis=0).clip(min=1e-6).tolist(),
    }


# ---------------------------------------------------------------------------
# CLIP text feature extractor (with file-level caching)
# ---------------------------------------------------------------------------

class CLIPTextCache:
    """Extracts and caches CLIP token-level text features."""

    def __init__(
        self,
        cache_dir: str,
        model_id:  str = CLIP_MODEL_ID,
        max_length: int = 77,
        device:    str = "cpu",
    ):
        self.cache_dir  = Path(cache_dir)
        self.max_length = max_length
        self.device     = device
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.tokenizer = CLIPTokenizer.from_pretrained(model_id)
        self.model     = CLIPTextModel.from_pretrained(model_id).to(device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

        self.dim = self.model.config.hidden_size   # 768 for ViT-L/14

    @torch.no_grad()
    def get(self, clip_id: str, text: str) -> np.ndarray:
        """Returns (77, dim) CLIP token features, using cache if available.

        Recomputes if the cache file is missing, empty, or corrupted (truncated)."""
        cache_path = self.cache_dir / f"{clip_id}.npy"
        if cache_path.exists() and cache_path.stat().st_size > 0:
            try:
                return np.load(str(cache_path))
            except (EOFError, ValueError, OSError):
                # Corrupted cache file — fall through and recompute
                cache_path.unlink(missing_ok=True)

        tokens = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        tokens = {k: v.to(self.device) for k, v in tokens.items()}
        feat   = self.model(**tokens).last_hidden_state[0].cpu().numpy()  # (77, dim)
        # Atomic write: per-process unique tmp name to avoid races when
        # multiple DataLoader workers compute the same clip simultaneously.
        # If another worker won the race, our tmp is harmless extra work.
        tmp_path = cache_path.with_suffix(f".npy.tmp.{os.getpid()}")
        try:
            np.save(str(tmp_path), feat)
            tmp_path.replace(cache_path)
        except OSError:
            # Another worker may have completed the rename first; clean up our tmp.
            tmp_path.unlink(missing_ok=True)
        return feat

    def get_token_ids(self, text: str) -> np.ndarray:
        """Returns (77,) token id array for self-similarity filtering."""
        tokens = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        return tokens["input_ids"][0].numpy()   # (77,)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

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


def _build_text(text_type: str, motion_caption: str, aspects: dict) -> str:
    """Build text string from text_type, motion caption, and aspect dict."""
    if text_type == "motion":
        return motion_caption
    parts = text_type.split("+")
    segments = []
    for p in parts:
        if p == "motion":
            mc = motion_caption.strip()
            if mc:
                segments.append(f"Camera motion: {mc.rstrip('.')}")
        else:
            val = aspects.get(p, "").strip()
            if val:
                label = ASPECT_LABEL.get(p, p)
                segments.append(f"{label}: {val}" if label else val.rstrip("."))
    return ". ".join(segments)


class CLaTrDataset(Dataset):
    """
    Returns:
        traj_feat:    (max_len, D)      trajectory features (D depends on traj_type)
        padding_mask: (max_len,)        True = valid frame
        caption_feat: (77, clip_dim)    CLIP token-level features
        sent_token:   (77,)             CLIP token ids (for false-neg filtering)
        clip_id:      str
    """

    def __init__(
        self,
        root:          str,
        datasets:      List[str],
        split:         str = "train",
        val_fraction:  float = 0.1,
        max_seq_len:   int  = 300,
        traj_type:     str  = "trajectory",
        text_type:     str  = "motion",
        standardization: Optional[dict] = None,
        clip_cache_dir: str = "./clip_cache",
        clip_model_id:  str = CLIP_MODEL_ID,
        clip_device:    str = "cpu",
        seed:          int = 42,
        min_aspect_ratio: float = 1.2,
    ):
        self.root        = Path(root)
        self.traj_type   = traj_type
        self.text_type   = text_type
        self.feat_dim    = TRAJ_DIM[traj_type]
        # trajectory uses N absolute frames; velocity-based use N-1 steps
        self.max_len     = max_seq_len if traj_type == "trajectory" else max_seq_len - 1
        self.std         = standardization   # dict or None (only for trajectory/9D)

        # Load aspect ratio cache for horizontal filtering
        aspect_cache = {}
        ar_cache_path = Path(root) / "aspect_ratio_cache.json"
        if ar_cache_path.exists():
            with open(ar_cache_path) as f:
                aspect_cache = json.load(f)

        # Build item list
        self.items           = []   # (ds, clip_id)
        self.motion_captions = {}   # clip_id → str
        self.aspects         = {}   # clip_id → dict
        n_ads_filtered = 0
        n_aspect_filtered = 0

        for ds in datasets:
            pose_dir    = self.root / "filtered_pose" / ds
            cap_dir     = self.root / "vipe_results"  / ds / "caption_cam+rot"
            jsonl_path  = self.root / "captions" / f"{ds}_captions.jsonl"
            if not pose_dir.exists():
                continue

            # Load cinematic aspects
            aspect_map = {}
            if jsonl_path.exists():
                with open(jsonl_path) as f:
                    for line in f:
                        entry = json.loads(line)
                        clip_id = Path(entry.get("video_path", "")).stem
                        aspect_map[clip_id] = _extract_aspects(
                            entry.get("cinematic_data", {}))

            for npz in sorted(pose_dir.glob("*.npz")):
                cid = npz.stem
                txt = cap_dir / f"{cid}.txt"
                if not txt.exists():
                    continue

                # Filter ad / UI clips (movieclips.com website overlays)
                aspects = aspect_map.get(cid, {k: "" for k in ASPECT_KEYS})
                if _is_ad_clip(aspects.get("logline_script", "")):
                    n_ads_filtered += 1
                    continue

                # Filter non-horizontal clips (vertical / near-square)
                if cid in aspect_cache:
                    w, h = aspect_cache[cid]
                    if h > 0 and w / h < min_aspect_ratio:
                        n_aspect_filtered += 1
                        continue

                self.items.append((ds, cid))
                self.motion_captions[cid] = txt.read_text().strip()
                self.aspects[cid] = aspects

        if n_ads_filtered > 0:
            print(f"[CLaTrDataset] Filtered {n_ads_filtered} ad/UI clips")
        if n_aspect_filtered > 0:
            print(f"[CLaTrDataset] Filtered {n_aspect_filtered} non-horizontal clips (aspect ratio < {min_aspect_ratio})")

        # Train / val split
        rng = np.random.default_rng(seed)
        idx = rng.permutation(len(self.items))
        n_val = max(1, int(len(self.items) * val_fraction))
        if split == "val":
            idx = idx[:n_val]
        else:
            idx = idx[n_val:]
        self.items = [self.items[i] for i in idx]

        # Compute standardization from training split if not provided (only for 9D trajectory)
        if self.std is None and split == "train" and traj_type == "trajectory":
            print("[CLaTrDataset] Computing standardization stats...")
            npz_paths = [
                self.root / "filtered_pose" / ds / f"{cid}.npz"
                for ds, cid in self.items
            ]
            self.std = compute_standardization(npz_paths, max_seq_len)
            print(f"[CLaTrDataset]   norm_mean={np.round(self.std['norm_mean'],4)}")

        # CLIP feature cache — keyed by text_type to avoid collisions
        cache_subdir = f"{clip_cache_dir}/{text_type.replace('+', '_')}"
        self.clip = CLIPTextCache(
            cache_dir=cache_subdir,
            model_id=clip_model_id,
            device=clip_device,
        )
        self.clip_dim = self.clip.dim

        print(f"[CLaTrDataset] {split}: {len(self.items)} clips "
              f"(traj={traj_type}/{self.feat_dim}D, text={text_type})")

    def _standardize(self, feat9d: np.ndarray) -> np.ndarray:
        """Apply per-dim standardization matching E.T.'s convention (9D trajectory only)."""
        if self.std is None:
            return feat9d
        feat = feat9d.copy()
        sm = np.array(self.std["shift_mean"], dtype=np.float32)
        ss = np.array(self.std["shift_std"],  dtype=np.float32)
        nm = np.array(self.std["norm_mean"],  dtype=np.float32)
        ns = np.array(self.std["norm_std"],   dtype=np.float32)
        feat[0, 6:] = (feat[0, 6:] - sm) / ss    # first frame: absolute pos
        feat[1:, 6:] = (feat[1:, 6:] - nm) / ns  # rest: velocity
        return feat

    def _get_text(self, cid: str) -> str:
        return _build_text(self.text_type, self.motion_captions[cid], self.aspects[cid])

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx: int) -> dict:
        ds, cid = self.items[idx]
        npz_path = self.root / "filtered_pose" / ds / f"{cid}.npz"

        # ---- Trajectory ----
        mats = np.load(str(npz_path))["data"].astype(np.float32)  # (N, 4, 4)

        if self.traj_type == "trajectory":
            raw = matrices_to_9d(mats)
            raw = self._standardize(raw)
            actual_len = min(len(raw), self.max_len)
            raw = raw[:actual_len]
        else:
            raw, L = matrices_to_feat(mats, self.traj_type)
            actual_len = min(L, self.max_len)
            raw = raw[:actual_len]

        D = self.feat_dim
        traj_feat = np.zeros((self.max_len, D), dtype=np.float32)
        traj_feat[:actual_len] = raw[:actual_len]
        mask = np.zeros(self.max_len, dtype=bool)
        mask[:actual_len] = True

        # ---- Text ----
        text     = self._get_text(cid)
        cap_feat = self.clip.get(cid, text)                         # (77, clip_dim)
        sent_tok = self.clip.get_token_ids(text)                    # (77,)

        return {
            "traj_feat":    torch.from_numpy(traj_feat),
            "padding_mask": torch.from_numpy(mask),
            "caption_feat": torch.from_numpy(cap_feat),
            "sent_token":   torch.from_numpy(sent_tok).long(),
            "clip_id":      cid,
        }


def collate_fn(batch: list) -> dict:
    return {
        "traj_feat":    torch.stack([b["traj_feat"]    for b in batch]),
        "padding_mask": torch.stack([b["padding_mask"] for b in batch]),
        "caption_feat": torch.stack([b["caption_feat"] for b in batch]),
        "sent_token":   torch.stack([b["sent_token"]   for b in batch]),
        "clip_id":      [b["clip_id"] for b in batch],
    }
