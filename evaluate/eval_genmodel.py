"""
Evaluation of generated camera trajectories against real references.

Three metrics, directly ported from GenDoP/CLaTr:

  1. F1-Score      — geometric: segment each trajectory into motion primitives,
                    compute weighted F1 between pred and ref segments.
                    Requires paired (pred, ref) trajectories via clip_id.

  2. CLaTr-CLIP   — alignment: use the trained alignment model as a text-traj
                    retrieval scorer.  Reports R@1/2/3/5/10 and MedR for both
                    text→traj and traj→text directions.

  3. CLaTr-FID    — distribution: Fréchet Distance between the trajectory
                    embedding distributions of generated vs real clips.

Alignment axes evaluated in one run:

  V1 (separate direction / speed models):
    motion ↔ direction        →  --motion_dir_ckpt
    motion ↔ speed            →  --motion_spd_ckpt
    <any aspect> ↔ direction  →  --content_dir_ckpts  aspect:path [aspect:path …]
    <any aspect> ↔ speed      →  --content_spd_ckpts  aspect:path [aspect:path …]

  V2 (combined direction+speed or full trajectory models):
    any text ↔ any traj_type  →  --ckpts  label:text_type:path [label:text_type:path …]
    text_type is one of: motion, logline_script, macro_type, setting_class,
                         subject_composition, genre_vibe

───────────────────────────────────────────────────────────────────────────────
Usage
─────

# V1 alignment models (separate direction / speed):
python evaluate/eval_genmodel.py \\
    --traj_dir        generated/ \\
    --metadata_jsonl  generated/metadata.jsonl \\
    --motion_dir_ckpt checkpoints/align/direction_motion/best.pt \\
    --motion_spd_ckpt checkpoints/align/speed_motion/best.pt \\
    --content_dir_ckpts \\
        genre_vibe:checkpoints/align/direction_genre_vibe/best.pt \\
    --split val

# V2 alignment models (direction+speed combined, recommended for Stage 2):
python evaluate/eval_genmodel.py \\
    --traj_dir        generated/ \\
    --metadata_jsonl  generated/metadata.jsonl \\
    --ckpts \\
        motion:motion:checkpoints/align_v2/direction_plus_speed_motion/best.pt \\
        all_aspects:logline_script:checkpoints/align_v2/direction_plus_speed_logline_script_plus_setting_class_plus_subject_composition_plus_genre_vibe/best.pt \\
    --split val

# Skip F1 (generated clips not paired 1-to-1 with real clips):
python evaluate/eval_genmodel.py ... --no_f1

# Save per-clip scores to JSONL:
python evaluate/eval_genmodel.py ... --save results.jsonl

───────────────────────────────────────────────────────────────────────────────
metadata.jsonl format (one JSON object per line):
  {
    "npz":                 "clip001.npz",
    "clip_id":             "clip001",
    "motion_caption":      "camera pans left …",
    "genre_vibe":          "Action / Thriller",
    "macro_type":          "Interior (Restricted)",
    "setting_class":       "Domestic/Residential",
    "subject_composition": "Single-Character"
  }
───────────────────────────────────────────────────────────────────────────────
"""

import argparse
import json
import logging
import sys
from itertools import product
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import mode
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluate.data.dataset import AlignDataset, collate_fn, ASPECT_KEYS
from evaluate.models.align_model import build_align_model, AlignModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

LOG_SPEED_EPS = 1e-6


# ─────────────────────────────────────────────────────────────────────────────
# 0.  Shared helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_model(ckpt_path: str, device: torch.device) -> Tuple[AlignModel, dict]:
    ckpt  = torch.load(ckpt_path, map_location=device)
    cfg   = ckpt["cfg"]
    model = build_align_model(
        traj_type         = cfg["traj_type"],
        embed_dim         = cfg["embed_dim"],
        traj_d_model      = cfg["traj_d_model"],
        traj_nhead        = cfg["traj_nhead"],
        traj_num_layers   = cfg["traj_num_layers"],
        max_vel_len       = cfg["max_seq_len"] if cfg["traj_type"] == "trajectory" else cfg["max_seq_len"] - 1,
        dropout           = cfg.get("dropout", 0.1),
        clip_model_id     = cfg.get("clip_model_id", "openai/clip-vit-large-patch14"),
        freeze_clip       = cfg.get("freeze_clip", True),
        init_temperature  = cfg.get("init_temperature", 0.07),
        learn_temperature = cfg.get("learn_temperature", True),
        pooling           = cfg.get("pooling", "mean"),      # back-compat: old ckpts used mean
        pos_enc           = cfg.get("pos_enc", "learned"),    # back-compat: old ckpts used learned
    ).to(device)
    sd = ckpt["model"]
    model_sd = model.state_dict()
    for k in list(sd.keys()):
        if k in model_sd and sd[k].shape != model_sd[k].shape:
            del sd[k]
    model.load_state_dict(sd, strict=False)
    model.eval()
    return model, cfg


def matrices_to_feat(
    matrices: np.ndarray,  # (N, 4, 4)  c2w
    traj_type: str,
    max_len: int,          # max frames (N) for "trajectory", max steps (N-1) otherwise
) -> Tuple[torch.Tensor, int]:
    """Convert c2w matrix sequence → padded feature tensor (max_len, D).

    Mirrors AlignDataset._load_traj_feat so eval uses the same encoding as training.
    """
    from utils.pose_utils import np_matrices_to_velocity

    if traj_type == "trajectory":
        # rot6D (first two cols of R, row-major) + rel_trans (3) = 9D per frame
        R      = matrices[:, :3, :3]
        rot6d  = R[:, :, :2].transpose(0, 2, 1).reshape(-1, 6)  # (N, 6)
        trans  = matrices[:, :3, 3]
        rel_t  = trans - trans[0:1]                              # (N, 3)
        raw    = np.concatenate([rot6d, rel_t], axis=-1)         # (N, 9)
        D      = 9
        actual_len = min(len(raw), max_len)
        raw    = raw[:actual_len]
    else:
        td, rd, ts, rs, _ = np_matrices_to_velocity(matrices)
        actual_len = min(len(td), max_len)
        if traj_type == "direction":
            raw = np.concatenate([td[:actual_len], rd[:actual_len]], axis=-1)   # (L, 6)
            D   = 6
        elif traj_type == "speed":
            ts_log = np.log(ts[:actual_len] + LOG_SPEED_EPS)
            rs_log = np.log(rs[:actual_len] + LOG_SPEED_EPS)
            raw    = np.stack([ts_log, rs_log], axis=-1)                        # (L, 2)
            D      = 2
        elif traj_type == "velocity":
            tv  = td[:actual_len] * ts[:actual_len, None]
            rv  = rd[:actual_len] * rs[:actual_len, None]
            raw = np.concatenate([tv, rv], axis=-1)                             # (L, 6)
            D   = 6
        else:  # direction+speed
            ts_log = np.log(ts[:actual_len] + LOG_SPEED_EPS)
            rs_log = np.log(rs[:actual_len] + LOG_SPEED_EPS)
            raw    = np.concatenate([
                td[:actual_len], rd[:actual_len],
                ts_log[:, None], rs_log[:, None],
            ], axis=-1)                                                          # (L, 8)
            D      = 8

    pad  = np.zeros((max_len - actual_len, D), dtype=np.float32)
    feat = np.concatenate([raw.astype(np.float32), pad], axis=0)
    return torch.from_numpy(feat), actual_len


def load_npz_matrices(path: str) -> np.ndarray:
    """Load (N,4,4) c2w matrices from a NPZ file."""
    npz = np.load(path)
    if "matrices" in npz:
        return npz["matrices"].astype(np.float32)
    if "data" in npz:
        return npz["data"].astype(np.float32)
    raise KeyError(f"NPZ {path} must contain 'matrices' or 'data' key")


# ─────────────────────────────────────────────────────────────────────────────
# 1.  F1-Score  (ported from GenDoP/evaluate/eval/src/metrics/modules/caption.py)
# ─────────────────────────────────────────────────────────────────────────────

def _se3_inverse(T: np.ndarray) -> np.ndarray:
    """Invert a 4×4 SE(3) matrix."""
    R = T[:3, :3]
    t = T[:3,  3]
    T_inv = np.eye(4)
    T_inv[:3, :3] = R.T
    T_inv[:3,  3] = -R.T @ t
    return T_inv


def _to_euler_angles(rot_mats: np.ndarray) -> np.ndarray:
    """Rotation matrices (N,3,3) → rotation vectors (N,3) via evo lie algebra."""
    from evo.core import lie_algebra as lie
    return np.stack([
        lie.sst_rotation_from_matrix(r).as_rotvec()
        for r in rot_mats
    ])


def _compute_relative(f_t: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    ax, ay, az = np.abs(f_t[:, 0]), np.abs(f_t[:, 1]), np.abs(f_t[:, 2])
    mxy = np.maximum(ax, ay);  mxz = np.maximum(ax, az);  myz = np.maximum(ay, az)
    xy = np.divide(ax - ay, mxy, out=np.zeros_like(mxy), where=mxy != 0)
    xz = np.divide(ax - az, mxz, out=np.zeros_like(mxz), where=mxz != 0)
    yz = np.divide(ay - az, myz, out=np.zeros_like(myz), where=myz != 0)
    return xy, xz, yz


def _compute_camera_dynamics(
    c2w_poses: np.ndarray,   # (T, 4, 4)
    fps: float = 30.0,
) -> Tuple:
    """Compute per-frame translational and rotational velocities."""
    # Relative transform between consecutive frames: inv(c2w[t]) @ c2w[t+1]
    # (same computation GenDoP does on w2c matrices – kept identical for comparability)
    c2w_inv = np.stack([_se3_inverse(T) for T in c2w_poses])   # treated as w2c
    rel = np.matmul(c2w_inv[:-1], c2w_poses[1:])               # (T-1, 4, 4)

    t_vel = fps * rel[:, :3, 3]   # (T-1, 3)
    a_vel = _to_euler_angles(rel[:, :3, :3])   # (T-1, 3)

    return t_vel, _compute_relative(t_vel), a_vel, _compute_relative(a_vel)


def _perform_segmentation(
    t_vel:  np.ndarray,  # (T-1, 3)
    xy: np.ndarray, xz: np.ndarray, yz: np.ndarray,
    static_thr: float = 0.02,
    diff_thr:   float = 0.4,
) -> np.ndarray:
    patterns = list(product([0, 1, -1], repeat=3))
    p2i = {p: i for i, p in enumerate(patterns)}
    segs = []
    for i, v in enumerate(t_vel):
        p = (np.abs(v) > static_thr).astype(int)
        # resolve two-axis ties
        if   p.tolist() == [1,1,0]: p = [1,0,0] if xy[i]> diff_thr else ([0,1,0] if xy[i]<-diff_thr else p)
        elif p.tolist() == [1,0,1]: p = [1,0,0] if xz[i]> diff_thr else ([0,0,1] if xz[i]<-diff_thr else p)
        elif p.tolist() == [0,1,1]: p = [0,1,0] if yz[i]> diff_thr else ([0,0,1] if yz[i]<-diff_thr else p)
        elif p.tolist() == [1,1,1]:
            if   xy[i]> diff_thr: p[1] = 0
            elif xy[i]<-diff_thr: p[0] = 0
            if   xz[i]> diff_thr: p[2] = 0
            elif xz[i]<-diff_thr: p[0] = 0
            if   yz[i]> diff_thr: p[2] = 0
            elif yz[i]<-diff_thr: p[1] = 0
        p = np.sign(v) * np.array(p)
        segs.append(p2i[tuple(p.astype(int).tolist())])
    return np.array(segs, dtype=np.int64)


def _perform_angular_segmentation(
    a_vel: np.ndarray,   # (T-1, 3)
    static_thr: float = 0.005,
) -> np.ndarray:
    ang_patterns = [[0,0,0],[1,0,0],[-1,0,0],[0,1,0],[0,-1,0],[0,0,1],[0,0,-1]]
    p2i = {tuple(p): i for i, p in enumerate(ang_patterns)}
    segs = []
    for v in a_vel:
        p = np.zeros(3)
        if np.abs(v).max() > static_thr:
            k = int(np.argmax(np.abs(v)))
            p[k] = 1
        p = np.sign(v) * p
        segs.append(p2i[tuple(p.astype(int).tolist())])
    return np.array(segs, dtype=np.int64)


def _smooth_segments(arr: np.ndarray, window: int) -> np.ndarray:
    arr = arr.copy()
    if len(arr) < window:
        return arr
    hw = window // 2
    for i in range(len(arr)):
        sl = arr[max(0, i-hw): i+hw+1]
        arr[i] = mode(sl, keepdims=False).mode
    return arr


def _find_chunks(arr):
    chunks, s = [], 0
    for i in range(1, len(arr)):
        if arr[i] != arr[i-1]:
            chunks.append((arr[s], s, i-1))
            s = i
    chunks.append((arr[s], s, len(arr)-1))
    return chunks


def _remove_short_chunks(arr: np.ndarray, min_size: int) -> np.ndarray:
    def _remove_one(chunks):
        if len(chunks) == 1:
            return False, chunks
        lens = [e-s+1 for _, s, e in chunks]
        k = int(np.argmin(lens))
        if lens[k] < min_size:
            L = lens[k]
            if k == 0:
                seg, s, e = chunks[k+1]; chunks[k+1] = (seg, s-L, e)
            elif k == len(chunks)-1:
                seg, s, e = chunks[k-1]; chunks[k-1] = (seg, s, e+L)
            else:
                lL = (L+1)//2;  lR = L//2
                s0, s1, e1 = chunks[k-1]; chunks[k-1] = (s0, s1, e1+lL)
                s0, s1, e1 = chunks[k+1]; chunks[k+1] = (s0, s1-lR, e1)
            chunks.pop(k)
            return True, chunks
        return False, chunks

    chunks = _find_chunks(arr)
    again = True
    while again:
        again, chunks = _remove_one(chunks)
    out = []
    for seg, s, e in chunks:
        out.extend([seg] * (e-s+1))
    return np.array(out, dtype=np.int64)


def _count_segments(arr):
    return 1 + int(np.sum(arr[1:] != arr[:-1]))


def segment_trajectory(c2w: np.ndarray) -> np.ndarray:
    """
    Segment a (T,4,4) c2w sequence into combined motion-primitive labels (T-1,).
    """
    t_vel, (t_xy, t_xz, t_yz), a_vel, _ = _compute_camera_dynamics(c2w)
    cam_segs = _perform_segmentation(t_vel, t_xy, t_xz, t_yz)
    ang_segs = _perform_angular_segmentation(a_vel)

    combined = cam_segs * 7 + ang_segs
    # iteratively smooth until ≤ 4 segments or no further change
    sw, mc = 15, 10
    for _ in range(20):
        smoothed = _smooth_segments(combined, sw)
        smoothed = _remove_short_chunks(smoothed, mc)
        if _count_segments(smoothed) <= 4:
            break
        sw += 5; mc += 5
    return smoothed


NUM_COMBINED_CLASSES = 27 * 7   # 189


def compute_f1(pred_segs: List[np.ndarray], ref_segs: List[np.ndarray]) -> dict:
    """Compute weighted precision / recall / F1 over all (pred, ref) segment pairs."""
    try:
        import torchmetrics.functional as TMF
    except ImportError:
        raise ImportError("torchmetrics required for F1 computation")

    # Truncate each pair to the shorter length before concatenating
    truncated_pred, truncated_ref = [], []
    for p, r in zip(pred_segs, ref_segs):
        min_len = min(len(p), len(r))
        truncated_pred.append(p[:min_len])
        truncated_ref.append(r[:min_len])
    all_pred = torch.from_numpy(np.concatenate(truncated_pred))
    all_ref  = torch.from_numpy(np.concatenate(truncated_ref))

    kw = dict(task="multiclass", num_classes=NUM_COMBINED_CLASSES,
              average="weighted", zero_division=0)
    return {
        "precision": float(TMF.precision(all_pred, all_ref, **kw)),
        "recall":    float(TMF.recall   (all_pred, all_ref, **kw)),
        "f1":        float(TMF.f1_score (all_pred, all_ref, **kw)),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 2.  CLaTr-CLIP  (retrieval metrics, ported from GenDoP CLaTr/src/training/metrics.py)
# ─────────────────────────────────────────────────────────────────────────────

def _cols2metrics(cols: np.ndarray, n: int, rounding: int = 2) -> dict:
    m = {}
    for k in [1, 2, 3, 5, 10]:
        m[f"R{str(k).zfill(2)}"] = round(100 * float(np.sum(cols < k)) / n, rounding)
    m["MedR"] = round(float(np.median(cols) + 1), rounding)
    return m


def _break_ties_avg(sorted_dists: np.ndarray, gt_dists: np.ndarray) -> np.ndarray:
    locs   = np.argwhere((sorted_dists - gt_dists) == 0)
    steps  = np.diff(locs[:, 0])
    splits = np.insert(np.nonzero(steps)[0] + 1, 0, 0)
    summed = np.add.reduceat(locs[:, 1], splits)
    counts = np.diff(np.append(splits, locs.shape[0]))
    return summed / counts


def contrastive_metrics(sims: np.ndarray, rounding: int = 2) -> dict:
    """
    Compute retrieval metrics from an (N, N) similarity matrix.
    GT is on the diagonal.  Returns t2m and m2t dicts.
    """
    n = sims.shape[0]
    dists = -sims
    sorted_dists = np.sort(dists, axis=1)
    gt_dists = np.diag(dists)[:, None]

    rows, cols = np.where((sorted_dists - gt_dists) == 0)
    if rows.size > n:
        cols = _break_ties_avg(sorted_dists, gt_dists)

    t2m = _cols2metrics(cols, n, rounding)
    # m2t: transpose similarity matrix
    dists_T = -sims.T
    sorted_T = np.sort(dists_T, axis=1)
    gt_T     = np.diag(dists_T)[:, None]
    rows_T, cols_T = np.where((sorted_T - gt_T) == 0)
    if rows_T.size > n:
        cols_T = _break_ties_avg(sorted_T, gt_T)
    m2t = _cols2metrics(cols_T, n, rounding)

    return {"t2m": t2m, "m2t": m2t, "N": n}


# ─────────────────────────────────────────────────────────────────────────────
# 3.  CLaTr-FID  (ported from GenDoP/evaluate/eval/src/metrics/modules/fcd.py)
# ─────────────────────────────────────────────────────────────────────────────

def _compute_fd(
    mu1: torch.Tensor, sigma1: torch.Tensor,
    mu2: torch.Tensor, sigma2: torch.Tensor,
) -> float:
    a = (mu1 - mu2).square().sum()
    b = sigma1.trace() + sigma2.trace()
    c = torch.linalg.eigvals(sigma1 @ sigma2).sqrt().real.sum()
    return float((a + b - 2 * c).item())


def frechet_distance(real_feats: np.ndarray, gen_feats: np.ndarray) -> float:
    """
    Fréchet Distance between two sets of feature vectors.
    Returns NaN if inputs contain non-finite values or computation fails.
    """
    if not (np.isfinite(real_feats).all() and np.isfinite(gen_feats).all()):
        return float("nan")

    def stats(X):
        X = torch.from_numpy(X).double()
        mu  = X.mean(dim=0)
        X0  = X - mu
        cov = (X0.T @ X0) / (len(X) - 1)
        return mu, cov

    try:
        mu1, cov1 = stats(real_feats)
        mu2, cov2 = stats(gen_feats)
        return _compute_fd(mu1, cov1, mu2, cov2)
    except Exception as e:
        import logging
        logging.getLogger(__name__).warning(f"frechet_distance failed: {e}")
        return float("nan")


def clatr_score(gen_traj_emb: np.ndarray, gen_text_emb: np.ndarray) -> float:
    """
    CLaTr-Score: mean cosine similarity between matched (traj, text) pairs × 100.
    Follows PulpMotion's SimilarityScore metric.
    """
    # Normalize to unit vectors
    t_norm = gen_traj_emb / (np.linalg.norm(gen_traj_emb, axis=1, keepdims=True) + 1e-8)
    x_norm = gen_text_emb / (np.linalg.norm(gen_text_emb, axis=1, keepdims=True) + 1e-8)
    # Per-pair cosine similarity (diagonal of t_norm @ x_norm.T)
    scores = (t_norm * x_norm).sum(axis=1)
    return max(0.0, float(100 * scores.mean()))


def coverage(real_feats: np.ndarray, gen_feats: np.ndarray, k: int = 3, num_splits: int = 5) -> float:
    """
    Coverage with num_splits averaging (matching GenDoP/PulpMotion scheme).
    Splits data into num_splits chunks, computes coverage per chunk, averages.
    """
    from sklearn.metrics import pairwise_distances
    if num_splits <= 1:
        dist_rr = pairwise_distances(real_feats, real_feats)
    np.fill_diagonal(dist_rr, np.inf)
    real_knn = np.sort(dist_rr, axis=1)[:, k - 1]  # k-th nearest neighbor
    # Nearest generated neighbor for each real sample
    dist_rf = pairwise_distances(real_feats, gen_feats)
    nearest_fake = dist_rf.min(axis=1)
    # Coverage = fraction of real samples "covered"
    return float((nearest_fake < real_knn).mean())


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Embedding extraction
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def encode_trajectories(
    model:      AlignModel,
    cfg:        dict,
    matrices_list: List[np.ndarray],   # list of (N,4,4) arrays
    device:     torch.device,
) -> np.ndarray:
    """Encode a list of c2w matrix sequences → (M, embed_dim) array."""
    traj_type = cfg["traj_type"]
    # "trajectory" uses N absolute frames; all velocity-based types use N-1 steps
    max_len = cfg["max_seq_len"] if traj_type == "trajectory" else cfg["max_seq_len"] - 1

    feats, seq_lens = [], []
    for m in matrices_list:
        f, sl = matrices_to_feat(m, traj_type, max_len)
        feats.append(f)
        seq_lens.append(sl)

    traj_input = torch.stack(feats).to(device)                   # (M, T, D)
    seq_tensor = torch.tensor(seq_lens, device=device)
    emb = model.encode_traj(traj_input, seq_tensor)               # (M, embed_dim)
    return F.normalize(emb, dim=-1).cpu().numpy()


@torch.no_grad()
def encode_texts(
    model:  AlignModel,
    texts:  List[str],
    device: torch.device,
    batch:  int = 64,
) -> np.ndarray:
    """Encode a list of text strings → (M, embed_dim) normalised array."""
    embs = []
    for i in range(0, len(texts), batch):
        e = model.encode_text(texts[i:i+batch])
        embs.append(F.normalize(e, dim=-1).cpu())
    return torch.cat(embs).numpy()


# ─────────────────────────────────────────────────────────────────────────────
# 5.  Real-reference data loader
# ─────────────────────────────────────────────────────────────────────────────

def load_real_val(cfg: dict, split: str = "val") -> Tuple[Dict[str, np.ndarray], Dict[str, str], Dict[str, dict]]:
    """
    Load the real val split — uses DiyMoviesDataset (same as generation) to ensure
    the val split matches the generated clips exactly.

    Returns:
        matrices_by_id   clip_id → (N,4,4) c2w matrices
        motion_by_id     clip_id → motion caption string
        aspects_by_id    clip_id → {aspect_key: value}
    """
    from pathlib import Path as P
    sys.path.insert(0, str(P(__file__).resolve().parents[1]))
    from data.dataset import DiyMoviesDataset, AVAILABLE_DATASETS as GEN_DATASETS

    root = P(cfg["root"])
    val_frac = cfg.get("val_fraction", 0.1)
    seed = cfg.get("seed", 42)

    # Use the SAME dataset class as generation to get identical val split
    tmp = DiyMoviesDataset(
        root=str(root), datasets=GEN_DATASETS, split=split,
        val_fraction=val_frac, seed=seed, traj_type="direction+speed",
        load_rgb=False, load_depth=False,
    )

    matrices_by_id, motion_by_id, aspects_by_id = {}, {}, {}
    for i in range(len(tmp)):
        ds, clip_id = tmp.items[i]
        path = root / "filtered_pose" / ds / f"{clip_id}.npz"
        try:
            matrices_by_id[clip_id] = np.load(path)["data"].astype(np.float32)
        except Exception:
            continue
        item = tmp[i]
        motion_by_id[clip_id] = item.get("motion_caption", "")
        aspects_by_id[clip_id] = item.get("cinematic_aspects", {k: "" for k in ASPECT_KEYS})

    log.info(f"Loaded {len(matrices_by_id)} real {split} trajectories")
    return matrices_by_id, motion_by_id, aspects_by_id


# ─────────────────────────────────────────────────────────────────────────────
# 6.  Main evaluation
# ─────────────────────────────────────────────────────────────────────────────

def run_eval(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}")

    # ------------------------------------------------------------------
    # Load generated trajectories + metadata
    # ------------------------------------------------------------------
    with open(args.metadata_jsonl) as f:
        meta = [json.loads(l) for l in f if l.strip()]

    gen_matrices, gen_clip_ids, gen_motion_caps, gen_aspects = [], [], [], []
    skipped = 0
    for entry in meta:
        npz_path = Path(args.traj_dir) / entry["npz"]
        if not npz_path.exists():
            log.warning(f"Missing: {npz_path}")
            skipped += 1
            continue
        try:
            m = load_npz_matrices(str(npz_path))
        except Exception as e:
            log.warning(f"Failed to load {npz_path}: {e}")
            skipped += 1
            continue
        gen_matrices.append(m)
        gen_clip_ids.append(entry.get("clip_id", npz_path.stem))
        gen_motion_caps.append(entry.get("motion_caption", ""))
        asp = {k: entry.get(k, "") for k in ASPECT_KEYS}
        gen_aspects.append(asp)

    N_gen = len(gen_matrices)
    log.info(f"Loaded {N_gen} generated trajectories  (skipped {skipped})")
    if N_gen == 0:
        log.error("No generated trajectories found.")
        sys.exit(1)

    # ------------------------------------------------------------------
    # Collect checkpoints to evaluate
    # ------------------------------------------------------------------
    VALID_TEXT_TYPES = frozenset(["motion", "motion+logline_script"] + list(ASPECT_KEYS))

    checkpoints: Dict[str, Tuple[str, str]] = {}   # label → (ckpt_path, text_type)
    if args.motion_dir_ckpt:
        checkpoints["motion↔direction"] = (args.motion_dir_ckpt, "motion")
    if args.motion_spd_ckpt:
        checkpoints["motion↔speed"]     = (args.motion_spd_ckpt, "motion")

    def _parse_aspect_ckpts(pairs):
        """Parse list of 'aspect:path' strings → [(aspect, path)]."""
        result = []
        for item in (pairs or []):
            if ":" not in item:
                log.error(f"Bad format for content checkpoint: {item!r}  (expected aspect:path)")
                sys.exit(1)
            aspect, path = item.split(":", 1)
            if aspect not in ASPECT_KEYS:
                log.error(f"Unknown aspect {aspect!r}. Valid: {ASPECT_KEYS}")
                sys.exit(1)
            result.append((aspect, path))
        return result

    for aspect, path in _parse_aspect_ckpts(args.content_dir_ckpts):
        checkpoints[f"{aspect}↔direction"] = (path, aspect)
    for aspect, path in _parse_aspect_ckpts(args.content_spd_ckpts):
        checkpoints[f"{aspect}↔speed"] = (path, aspect)

    # V2 generic checkpoints: --ckpts label:text_type:path
    for item in (args.ckpts or []):
        parts = item.split(":", 2)
        if len(parts) != 3:
            log.error(f"Bad format for --ckpts entry: {item!r}  (expected label:text_type:path)")
            sys.exit(1)
        label, text_type, path = parts
        if text_type not in VALID_TEXT_TYPES:
            log.error(f"Unknown text_type {text_type!r} in --ckpts. Valid: {sorted(VALID_TEXT_TYPES)}")
            sys.exit(1)
        if label in checkpoints:
            log.error(f"Duplicate label {label!r} in --ckpts.")
            sys.exit(1)
        checkpoints[label] = (path, text_type)

    if not checkpoints and not args.no_f1:
        log.error("Provide at least one checkpoint or use --no_f1.")
        sys.exit(1)

    # ------------------------------------------------------------------
    # Load real val data (needed for F1 and FID)
    # ------------------------------------------------------------------
    real_matrices, real_motion, real_aspects = {}, {}, {}
    if checkpoints:
        # Use the first checkpoint's cfg to resolve the dataset split
        first_ckpt_path = next(iter(checkpoints.values()))[0]
        cfg0 = torch.load(first_ckpt_path, map_location="cpu")["cfg"]
        real_matrices, real_motion, real_aspects = load_real_val(cfg0, args.split)

    results = {}   # label → metric dict

    # ------------------------------------------------------------------
    # F1-Score  (geometry only, no model needed)
    # ------------------------------------------------------------------
    if not args.no_f1:
        log.info("Computing F1-Score (geometric segmentation)…")
        paired_pred, paired_ref = [], []
        n_paired = 0
        for i, clip_id in enumerate(gen_clip_ids):
            if clip_id not in real_matrices:
                continue
            try:
                pred_segs = segment_trajectory(gen_matrices[i])
                ref_segs  = segment_trajectory(real_matrices[clip_id])
                if len(pred_segs) < 2 or len(ref_segs) < 2:
                    continue
                paired_pred.append(pred_segs)
                paired_ref.append(ref_segs)
                n_paired += 1
            except Exception as e:
                log.warning(f"F1 segmentation failed for {clip_id}: {e}")

        if paired_pred:
            f1_res = compute_f1(paired_pred, paired_ref)
            results["F1-Score"] = f1_res
            log.info(f"F1 computed on {n_paired} paired clips")
        else:
            log.warning("No paired clips found for F1 computation. "
                        "Ensure clip_ids in metadata match the dataset.")

    # ------------------------------------------------------------------
    # Per-alignment-model metrics (CLaTr-CLIP + CLaTr-FID)
    # ------------------------------------------------------------------
    per_clip_scores = {entry.get("clip_id", ""): {} for entry in meta}

    for label, (ckpt_path, text_type) in checkpoints.items():
        log.info(f"\n── {label}  [{ckpt_path}] ──")
        model, cfg = load_model(ckpt_path, device)
        embed_dim = cfg["embed_dim"]

        # Determine text to use for generated clips
        gen_texts = []
        for i in range(N_gen):
            if text_type == "motion":
                gen_texts.append(gen_motion_caps[i])
            elif "+" in text_type:
                # Composite: e.g. "motion+logline_script"
                parts = text_type.split("+")
                segments = []
                for p in parts:
                    if p == "motion":
                        segments.append(f"Camera motion: {gen_motion_caps[i]}")
                    else:
                        val = gen_aspects[i].get(p, "")
                        if val:
                            segments.append(val)
                gen_texts.append(". ".join(segments))
            else:
                gen_texts.append(gen_aspects[i].get(text_type, ""))

        # ---- CLaTr-CLIP: encode generated traj + their texts ----
        log.info("  Encoding generated trajectories…")
        gen_traj_emb = encode_trajectories(model, cfg, gen_matrices, device)
        log.info("  Encoding generated texts…")
        gen_text_emb = encode_texts(model, gen_texts, device)

        # Similarity matrix (N_gen, N_gen) — diagonal = matched pairs
        sims = gen_traj_emb @ gen_text_emb.T
        clip_res = contrastive_metrics(sims)

        # Per-clip alignment score (diagonal of sims)
        diag_scores = sims.diagonal()
        for i, clip_id in enumerate(gen_clip_ids):
            per_clip_scores[clip_id][f"{label}/sim"] = float(diag_scores[i])

        # ---- CLaTr-FID: generated vs real trajectory distributions ----
        real_ids   = list(real_matrices.keys())
        real_mats  = [real_matrices[cid] for cid in real_ids]
        log.info(f"  Encoding {len(real_mats)} real trajectories for FID…")
        real_traj_emb = encode_trajectories(model, cfg, real_mats, device)

        fid = frechet_distance(real_traj_emb, gen_traj_emb)
        cs  = clatr_score(gen_traj_emb, gen_text_emb)
        cov = coverage(real_traj_emb, gen_traj_emb, k=3)

        results[label] = {
            "CLaTr-CLIP": clip_res,
            "CLaTr-FID":  fid,
            "CLaTr-Score": cs,
            "Coverage":    cov,
        }

    # ------------------------------------------------------------------
    # Print summary table
    # ------------------------------------------------------------------
    _print_summary(results, N_gen)

    # ------------------------------------------------------------------
    # Optionally save per-clip scores
    # ------------------------------------------------------------------
    if args.save:
        rows = []
        for entry in meta:
            cid = entry.get("clip_id", "")
            row = {"clip_id": cid}
            row.update(per_clip_scores.get(cid, {}))
            rows.append(row)
        with open(args.save, "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        log.info(f"Per-clip scores saved → {args.save}")


def _print_summary(results: dict, n_gen: int):
    sep = "═" * 72
    print(f"\n{sep}")
    print(f"  Evaluation Summary  ({n_gen} generated clips)")
    print(sep)

    if "F1-Score" in results:
        r = results["F1-Score"]
        print(f"\n  F1-Score (geometric trajectory segmentation)")
        print(f"    Precision : {r['precision']:.4f}")
        print(f"    Recall    : {r['recall']:.4f}")
        print(f"    F1        : {r['f1']:.4f}  ← higher is better")

    for label, res in results.items():
        if label == "F1-Score":
            continue
        clip_res = res["CLaTr-CLIP"]
        fid      = res["CLaTr-FID"]
        t2m = clip_res["t2m"]
        m2t = clip_res["m2t"]
        N   = clip_res["N"]
        cs  = res.get("CLaTr-Score", None)
        cov = res.get("Coverage", None)
        print(f"\n  ┌─ {label}  (N={N})")
        print(f"  │  FDCLaTr     : {fid:.4f}  ← lower is better")
        if cs is not None:
            print(f"  │  CLaTr-Score : {cs:.2f}  ← higher is better")
        if cov is not None:
            print(f"  │  Coverage    : {cov:.4f}  ← higher is better")
        print(f"  │  CLaTr-CLIP  text→traj :")
        print(f"  │    R@1={t2m['R01']}%  R@5={t2m['R05']}%  R@10={t2m['R10']}%  MedR={t2m['MedR']}")
        print(f"  │  CLaTr-CLIP  traj→text :")
        print(f"  └    R@1={m2t['R01']}%  R@5={m2t['R05']}%  R@10={m2t['R10']}%  MedR={m2t['MedR']}")

    print(f"\n{sep}\n")


# ─────────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__,
    )
    p.add_argument("--traj_dir",        required=True,
                   help="Directory containing generated NPZ files")
    p.add_argument("--metadata_jsonl",  required=True,
                   help="JSONL with fields: npz, clip_id, motion_caption, <aspect_keys>")
    p.add_argument("--split",           default="val", choices=["train","val"],
                   help="Dataset split to use as reference (default: val)")

    # Alignment checkpoints (at least one required unless --no_f1)
    p.add_argument("--motion_dir_ckpt",  default=None,
                   help="Checkpoint for motion↔direction alignment")
    p.add_argument("--motion_spd_ckpt",  default=None,
                   help="Checkpoint for motion↔speed alignment")
    p.add_argument("--content_dir_ckpts", nargs="+", default=[],
                   metavar="ASPECT:PATH",
                   help="One or more content↔direction checkpoints, e.g. "
                        "genre_vibe:checkpoints/align/direction_genre_vibe/best.pt "
                        "macro_type:checkpoints/align/direction_macro_type/best.pt")
    p.add_argument("--content_spd_ckpts", nargs="+", default=[],
                   metavar="ASPECT:PATH",
                   help="One or more content↔speed checkpoints, same format as above")
    p.add_argument("--ckpts", nargs="+", default=[],
                   metavar="LABEL:TEXT_TYPE:PATH",
                   help="V2 generic checkpoints (any traj_type). Format: label:text_type:path. "
                        "text_type is one of: motion, logline_script, macro_type, "
                        "setting_class, subject_composition, genre_vibe. "
                        "The model's traj_type is read from its saved cfg.")

    p.add_argument("--no_f1",   action="store_true",
                   help="Skip F1-Score (e.g. when generated clips are not paired with refs)")
    p.add_argument("--save",    default=None,
                   help="Optional path to save per-clip scores as JSONL")

    args = p.parse_args()
    run_eval(args)


if __name__ == "__main__":
    main()
