"""
Eval-time dataset for CineGen.

The published eval pack (released on HF Hub at `Ziqi1018/CineScript-eval`) has the
following layout:

    <root>/
        index.jsonl                   # one line per clip (metadata)
        matrices/<clip_id>.npz        # real 4×4 c2w GT trajectories

Each line of ``index.jsonl`` looks like::

    {
      "clip_id": "...",
      "motion_caption": "camera dollies forward ...",
      "logline_script": "INT. PARK - DAY - ...",
      "year": 1999,
      "countries": ["USA"],
      "directors": ["..."],
      "genres": ["Drama"],
      "macro_type": "Exterior (Open)",
      "setting_class": "Urban/City",
      "subject_composition": "Single-Character",
      "genre_vibe": "Drama / Romance / Emotion"
    }

This file deliberately ships an **eval-only** loader — it does not implement
the full DiyMoviesDataset used for training. The training pipeline (with VLM
captioning, aspect extraction, multi-source caching, etc.) is part of the
training-code release that comes with paper acceptance.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset


LOG_SPEED_EPS = 1e-6

TRAJ_DIM = {
    "trajectory":      9,
    "direction+speed": 8,
}

FIRST_POSE_DIM = {
    "trajectory":      9,
    "direction+speed": 8,
}

ASPECT_KEYS = [
    "logline_script",
    "macro_type",
    "setting_class",
    "subject_composition",
    "genre_vibe",
]


# ─────────────────────────────────────────────────────────────────────────────
# Velocity decomposition (extracted from the original utils/pose_utils.py)
# ─────────────────────────────────────────────────────────────────────────────

def np_matrices_to_velocity(matrices: np.ndarray):
    """Decompose (N, 4, 4) c2w matrices into per-step velocity descriptors.

    Returns
    -------
    trans_dir   : (N-1, 3)   unit translation direction per step
    rot_dir     : (N-1, 3)   unit rotation axis per step
    trans_speed : (N-1,)     raw translation speed (Euclidean)
    rot_speed   : (N-1,)     raw rotation magnitude (radians)
    first_pose  : (7,)       [tx, ty, tz, qw, qx, qy, qz]  — legacy field
    """
    from scipy.spatial.transform import Rotation as Rot

    N = matrices.shape[0]
    R = matrices[:, :3, :3]
    t = matrices[:, :3, 3]

    dt = t[1:] - t[:-1]                          # (N-1, 3)
    trans_speed = np.linalg.norm(dt, axis=-1)    # (N-1,)
    trans_dir = np.where(
        trans_speed[:, None] > 1e-8,
        dt / np.maximum(trans_speed[:, None], 1e-8),
        0.0,
    ).astype(np.float32)

    R_rel = np.einsum("nij,njk->nik", R[:-1].transpose(0, 2, 1), R[1:])
    aa = Rot.from_matrix(R_rel).as_rotvec()      # (N-1, 3)
    rot_speed = np.linalg.norm(aa, axis=-1)
    rot_dir = np.where(
        rot_speed[:, None] > 1e-8,
        aa / np.maximum(rot_speed[:, None], 1e-8),
        0.0,
    ).astype(np.float32)

    first_pose = np.zeros(7, dtype=np.float32)
    first_pose[:3] = t[0]
    first_pose[3:] = Rot.from_matrix(R[0]).as_quat(scalar_first=True)
    return trans_dir, rot_dir, trans_speed.astype(np.float32), rot_speed.astype(np.float32), first_pose


# ─────────────────────────────────────────────────────────────────────────────
# First-pose extraction in the 8-D dirspd convention
# ─────────────────────────────────────────────────────────────────────────────

def compute_first_pose_dirspd(matrices: np.ndarray) -> np.ndarray:
    """Build the 8-D first_pose CineGen expects, from a (N, 4, 4) trajectory.

    Format: ``[trans_dir(3), rot_dir(3), log_trans_speed(1), log_rot_speed(1)]``
    where the trans/rot direction+speed are derived from the *absolute*
    initial pose ``matrices[0]`` (not the first-step velocity).
    """
    from scipy.spatial.transform import Rotation as Rot

    R0 = matrices[0, :3, :3]
    t0 = matrices[0, :3, 3].astype(np.float32)

    t0_speed = float(np.linalg.norm(t0))
    t0_dir = t0 / (t0_speed + 1e-8) if t0_speed > 1e-8 else np.zeros(3, dtype=np.float32)

    r0_aa = Rot.from_matrix(R0).as_rotvec().astype(np.float32)
    r0_speed = float(np.linalg.norm(r0_aa))
    r0_dir = r0_aa / (r0_speed + 1e-8) if r0_speed > 1e-8 else np.zeros(3, dtype=np.float32)

    return np.concatenate([
        t0_dir, r0_dir,
        [np.log(t0_speed + LOG_SPEED_EPS).astype(np.float32)],
        [np.log(r0_speed + LOG_SPEED_EPS).astype(np.float32)],
    ])


def compute_first_pose_traj(matrices: np.ndarray) -> np.ndarray:
    """First-pose vector for the trajectory representation (9-D rot6D + zero translation)."""
    R0 = matrices[0, :3, :3]
    rot6d = R0[:, :2].T.reshape(6).astype(np.float32)
    return np.concatenate([rot6d, np.zeros(3, dtype=np.float32)])


# ─────────────────────────────────────────────────────────────────────────────
# EvalDataset
# ─────────────────────────────────────────────────────────────────────────────

class EvalDataset(Dataset):
    """Loads the CineGen eval pack from a local directory (downloaded from HF Hub).

    Parameters
    ----------
    root : str | Path
        Directory containing ``index.jsonl`` and ``matrices/`` subfolder.
    traj_type : str
        One of ``"direction+speed"`` or ``"trajectory"``. Controls the format
        of the returned ``first_pose`` (8-D or 9-D).
    max_seq_len : int, default 300
        Hard cap on returned matrix length.
    """

    def __init__(
        self,
        root: str | Path,
        traj_type: str = "direction+speed",
        max_seq_len: int = 300,
    ):
        if traj_type not in TRAJ_DIM:
            raise ValueError(f"Unknown traj_type {traj_type!r}; choose from {sorted(TRAJ_DIM)}")
        self.root = Path(root)
        self.traj_type = traj_type
        self.max_seq_len = max_seq_len

        index_path = self.root / "index.jsonl"
        if not index_path.exists():
            raise FileNotFoundError(
                f"{index_path} not found. Download the eval pack via scripts/download.sh "
                "(see README.md for details)."
            )

        with open(index_path) as f:
            self.entries: List[dict] = [json.loads(line) for line in f if line.strip()]

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int) -> dict:
        entry = self.entries[idx]
        clip_id = entry["clip_id"]

        matrices = np.load(self.root / "matrices" / f"{clip_id}.npz")["data"].astype(np.float32)
        # Some sources store (N, 3, 4) — pad to (N, 4, 4) for downstream code.
        if matrices.ndim == 3 and matrices.shape[1:] == (3, 4):
            T = matrices.shape[0]
            bot = np.tile(np.array([0, 0, 0, 1], np.float32), (T, 1, 1))
            matrices = np.concatenate([matrices, bot], axis=1)

        # Truncate to max_seq_len
        if matrices.shape[0] > self.max_seq_len:
            matrices = matrices[: self.max_seq_len]

        # Compute first_pose in the traj_type-specific format
        if self.traj_type == "direction+speed":
            first_pose = compute_first_pose_dirspd(matrices)
        else:
            first_pose = compute_first_pose_traj(matrices)

        cinematic_aspects = {k: entry.get(k, "") for k in ASPECT_KEYS}

        return {
            "clip_id":           clip_id,
            "motion_caption":    entry.get("motion_caption", ""),
            "cinematic_aspects": cinematic_aspects,
            "first_pose":        torch.from_numpy(first_pose),
            "matrices":          matrices,                      # (T, 4, 4)
            "seq_len":           matrices.shape[0],
            # Attribute labels exposed for the attribute-fidelity eval
            "attr_labels":       {
                "year":      entry.get("year"),
                "countries": entry.get("countries", []),
                "directors": entry.get("directors", []),
                "genres":    entry.get("genres", []),
            },
        }


# ─────────────────────────────────────────────────────────────────────────────
# Batched collate for inference loop
# ─────────────────────────────────────────────────────────────────────────────

def collate_fn(batch: List[dict]) -> dict:
    """Pad ``first_pose`` and stack what we can; everything else stays as lists."""
    out: Dict = {}
    # Tensors that are always shape-compatible
    out["first_pose"] = torch.stack([b["first_pose"] for b in batch])
    out["seq_len"]    = torch.tensor([b["seq_len"] for b in batch], dtype=torch.long)
    # String/list fields
    out["clip_id"]        = [b["clip_id"] for b in batch]
    out["motion_caption"] = [b["motion_caption"] for b in batch]
    # cinematic_aspects: dict[key → list[str]]  (one entry per sample)
    out["cinematic_aspects"] = {
        k: [b["cinematic_aspects"].get(k, "") for b in batch] for k in ASPECT_KEYS
    }
    # Matrices (variable length) kept as a list of numpy arrays
    out["matrices"] = [b["matrices"] for b in batch]
    # Attribute labels stay as a list of dicts
    out["attr_labels"] = [b["attr_labels"] for b in batch]
    return out
