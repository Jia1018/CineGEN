"""
Postprocess CineGen's generated trajectories for Blender rendering.

This script:
1. Loads a generated ``<clip_id>.npz`` containing 4×4 c2w ``matrices``.
2. Optionally applies temporal smoothing —
   - Gaussian smoothing on translations,
   - log-map smoothing on rotations (axis-angle → Gaussian → back to rotation matrix).
   Large sigmas clean up the per-frame jitter that makes per-step
   autoregressive samples look noisy in renders.
3. Writes a JSON file with the camera intrinsics + per-frame transform matrices
   in the format consumed by ``blender_render.py``.

Usage
-----
::

    python visualize/postprocess.py \\
        --gen_dir results/cinegen-generated \\
        --clip_id <id> \\
        --out_json cases/<id>.json \\
        --smooth_sigma 3.0
"""
import argparse, json
from pathlib import Path
import numpy as np


def gaussian_kernel(sigma, radius=None):
    if radius is None:
        radius = max(1, int(round(sigma * 3)))
    x = np.arange(-radius, radius + 1, dtype=np.float64)
    k = np.exp(-x * x / (2 * sigma * sigma))
    return k / k.sum()


def smooth_1d(x, sigma):
    """Gaussian smooth along axis 0, with edge replication."""
    if sigma <= 0:
        return x
    k = gaussian_kernel(sigma)
    r = (len(k) - 1) // 2
    pad = np.concatenate([np.repeat(x[:1], r, axis=0), x,
                          np.repeat(x[-1:], r, axis=0)], axis=0)
    out = np.zeros_like(x)
    for i, w in enumerate(k):
        out += w * pad[i:i + len(x)]
    return out


def rot_to_axis_angle(R):
    """(T, 3, 3) → (T, 3) axis-angle."""
    tr = np.einsum("tii->t", R)
    cos_th = np.clip((tr - 1) / 2, -1, 1)
    angle = np.arccos(cos_th)
    skew = (R - R.transpose(0, 2, 1)) / 2
    axis = np.stack([skew[:, 2, 1], skew[:, 0, 2], skew[:, 1, 0]], axis=-1)
    sin_th = np.sin(angle)
    safe = sin_th > 1e-8
    axis = np.where(safe[:, None], axis / np.maximum(sin_th[:, None], 1e-8), 0)
    return axis * angle[:, None]


def axis_angle_to_rot(aa):
    """(T, 3) → (T, 3, 3)."""
    angle = np.linalg.norm(aa, axis=-1)
    safe = angle > 1e-8
    axis = np.where(safe[:, None], aa / np.maximum(angle[:, None], 1e-8), 0)
    K = np.zeros((len(aa), 3, 3))
    K[:, 0, 1] = -axis[:, 2]; K[:, 0, 2] = axis[:, 1]
    K[:, 1, 0] = axis[:, 2];  K[:, 1, 2] = -axis[:, 0]
    K[:, 2, 0] = -axis[:, 1]; K[:, 2, 1] = axis[:, 0]
    I = np.eye(3)[None]
    sa = np.sin(angle)[:, None, None]
    ca = (1 - np.cos(angle))[:, None, None]
    return I + sa * K + ca * (K @ K)


def smooth_trajectory(c2ws, sigma):
    """Gaussian smooth translations + log-space smooth rotations (relative)."""
    if sigma <= 0:
        return c2ws
    out = c2ws.copy()
    # Translation: directly Gaussian smooth
    out[:, :3, 3] = smooth_1d(c2ws[:, :3, 3], sigma)
    # Rotation: convert to axis-angle (relative to first frame) → smooth → reapply
    R = c2ws[:, :3, :3]
    R0 = R[0]
    Rrel = R0.T @ R                          # (T, 3, 3) — frame-0-relative rotations
    aa = rot_to_axis_angle(Rrel)              # (T, 3)
    aa_s = smooth_1d(aa, sigma)
    Rrel_s = axis_angle_to_rot(aa_s)
    out[:, :3, :3] = R0 @ Rrel_s
    return out


def npz_to_json(npz_path: Path, out_json: Path, smooth_sigma: float,
                w=512, h=512, fl=512, cx=256, cy=256):
    data = np.load(npz_path)
    if "matrices" in data:
        m = data["matrices"].astype(np.float64)
    elif "data" in data:
        m = data["data"].astype(np.float64)
    else:
        raise KeyError(f"{npz_path}: neither 'matrices' nor 'data'")

    if m.ndim == 3 and m.shape[1:] == (3, 4):
        T = m.shape[0]
        bot = np.tile(np.array([0, 0, 0, 1], np.float64), (T, 1, 1))
        m = np.concatenate([m, bot], axis=1)
    assert m.shape[1:] == (4, 4), f"unexpected shape {m.shape}"

    if smooth_sigma > 0:
        m = smooth_trajectory(m, smooth_sigma)

    obj = {
        "w": w, "h": h, "fl_x": fl, "fl_y": fl, "cx": cx, "cy": cy,
        "frames": [{"transform_matrix": M.tolist()} for M in m],
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(obj, f, indent=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen_dir", required=True)
    ap.add_argument("--clip_id", required=True)
    ap.add_argument("--out_json", required=True)
    ap.add_argument("--smooth_sigma", type=float, default=3.0,
                    help="Gaussian smoothing sigma (frames). 0 disables.")
    args = ap.parse_args()
    npz = Path(args.gen_dir) / f"{args.clip_id}.npz"
    npz_to_json(npz, Path(args.out_json), args.smooth_sigma)
    print(f"Wrote {args.out_json} (smooth_sigma={args.smooth_sigma})", flush=True)


if __name__ == "__main__":
    main()
