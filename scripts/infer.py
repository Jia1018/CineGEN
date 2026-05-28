"""
CineGen inference — generates camera trajectories for the eval set.

Reads the eval pack (see ``cinegen.dataset.EvalDataset``) and writes
``<out_dir>/<clip_id>.npz`` plus ``<out_dir>/metadata.jsonl`` for each input
clip. The output format matches what the evaluation pipeline expects.

Usage
-----
::

    python scripts/infer.py \\
        --ckpt checkpoints/cinegen/best.pt \\
        --data_root data/cinescript-eval \\
        --out_dir results/cinegen-generated \\
        --batch_size 32

The script auto-detects the training config from the checkpoint (no-AE vs AE,
use_logline, use_first_pose, traj_type, …). The default published checkpoint is
a no-AE / sep-logline / first-pose model on direction+speed features.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from cinegen.model import CineGen, IdentityAE
from cinegen.dataset import EvalDataset, collate_fn, TRAJ_DIM

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


# ─────────────────────────────────────────────────────────────────────────────
# Feature → c2w matrix reconstruction
# ─────────────────────────────────────────────────────────────────────────────

LOG_SPEED_EPS = 1e-6


def dirspd_to_matrices(dirspd: np.ndarray, first_pose: np.ndarray | None = None) -> np.ndarray:
    """Reconstruct (T+1, 4, 4) c2w matrices from (T, 8) dir+spd features.

    Format of each row: ``[trans_dir(3), rot_dir(3), log_trans_speed, log_rot_speed]``.
    If ``first_pose`` (the 8-D vector returned by ``EvalDataset``) is provided, the
    initial pose is reconstructed from it; otherwise the trajectory starts at
    identity / origin.
    """
    from scipy.spatial.transform import Rotation as Rot

    T = dirspd.shape[0]
    trans_dir, rot_dir = dirspd[:, :3], dirspd[:, 3:6]
    log_ts, log_rs = dirspd[:, 6], dirspd[:, 7]
    ts = np.maximum(np.exp(log_ts) - LOG_SPEED_EPS, 0)
    rs = np.maximum(np.exp(log_rs) - LOG_SPEED_EPS, 0)
    trans_vel = trans_dir * ts[:, None]
    rot_vel   = rot_dir * rs[:, None]

    matrices = np.zeros((T + 1, 4, 4), dtype=np.float32)
    matrices[:, 3, 3] = 1.0

    R = np.eye(3, dtype=np.float32)
    t = np.zeros(3, dtype=np.float32)
    if first_pose is not None and len(first_pose) >= 8:
        fp_t = first_pose[:3] * max(np.exp(first_pose[6]) - LOG_SPEED_EPS, 0)
        fp_r_aa = first_pose[3:6] * max(np.exp(first_pose[7]) - LOG_SPEED_EPS, 0)
        t = fp_t
        if np.linalg.norm(fp_r_aa) > 1e-8:
            R = Rot.from_rotvec(fp_r_aa).as_matrix().astype(np.float32)

    matrices[0, :3, :3] = R
    matrices[0, :3, 3]  = t
    for i in range(T):
        t = t + trans_vel[i]
        if np.linalg.norm(rot_vel[i]) > 1e-8:
            dR = Rot.from_rotvec(rot_vel[i]).as_matrix().astype(np.float32)
            R = R @ dR
        matrices[i + 1, :3, :3] = R
        matrices[i + 1, :3, 3]  = t
    return matrices


def traj_feat_to_matrices(feat: np.ndarray) -> np.ndarray:
    """Reconstruct (T, 4, 4) c2w matrices from (T, 9) [rot6D(6) + rel_trans(3)] features."""
    T = feat.shape[0]
    rot6d, rel_trans = feat[:, :6], feat[:, 6:]
    matrices = np.zeros((T, 4, 4), dtype=np.float32)
    matrices[:, 3, 3] = 1.0
    for i in range(T):
        col01 = rot6d[i].reshape(2, 3).T
        c0 = col01[:, 0] / (np.linalg.norm(col01[:, 0]) + 1e-8)
        c1 = col01[:, 1] - np.dot(col01[:, 1], c0) * c0
        c1 = c1 / (np.linalg.norm(c1) + 1e-8)
        c2 = np.cross(c0, c1)
        matrices[i, :3, :3] = np.stack([c0, c1, c2], axis=1)
        matrices[i, :3, 3]  = rel_trans[i]
    return matrices


# ─────────────────────────────────────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────────────────────────────────────

def load_model(ckpt_path: str, device: torch.device):
    """Build CineGen from a checkpoint, auto-detecting its training config."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    a = ckpt.get("args", {})
    ae_dim = ckpt.get("ae_dim") or ckpt.get("latent_dim") or 64

    model = CineGen(
        ae_dim=ae_dim,
        seq_latent_dim=a.get("seq_d_model", 512),
        seq_nhead=a.get("seq_nhead", 8),
        seq_num_layers=a.get("seq_num_layers", 1),
        seq_ff_size=a.get("seq_ff_size", 4096),
        diff_model_channels=a.get("diff_d_model", 1024),
        diff_num_res_blocks=a.get("diff_num_res_blocks", 3),
        clip_dim=512,
        cond_drop_prob=0.0,
        use_first_pose=a.get("use_first_pose", False),
        use_aspects=a.get("use_aspects", False),
        use_logline=a.get("use_logline", False),
        logline_emb_dim=a.get("logline_emb_dim", 64),
        ddpm_T=a.get("ddpm_T", 100),
        text_mode=a.get("text_mode", "combined"),
    ).to(device)
    model.load_state_dict(ckpt["model"], strict=False)
    model.eval()

    log.info(f"Loaded CineGen: {ckpt_path} (epoch={ckpt.get('epoch')}, val_loss={ckpt.get('val_loss'):.6f})")
    log.info(
        f"  use_first_pose={a.get('use_first_pose')}, use_aspects={a.get('use_aspects')}, "
        f"use_logline={a.get('use_logline')}, text_mode={a.get('text_mode')}, "
        f"traj_type={a.get('traj_type', ckpt.get('traj_type'))}"
    )
    return model, ckpt


def build_ae(ckpt, traj_type: str, device: torch.device):
    """Build the AE used at inference time.

    For the published no-AE variant the checkpoint has ``no_ae=True``; we return
    a stateless ``IdentityAE``. If you want to run inference with the original
    PulpAE you'll need the AE checkpoint and its ``PulpAE`` class — both will
    ship in the training-code release.
    """
    if ckpt.get("no_ae", False):
        ae = IdentityAE(TRAJ_DIM[traj_type]).to(device)
        log.info(f"  Using IdentityAE (no compression) for input_dim={TRAJ_DIM[traj_type]}")
        return ae
    raise NotImplementedError(
        "This checkpoint expects a learned PulpAE encoder/decoder, which is not "
        "included in this release. The full AE code will be released with the "
        "training pipeline. See README.md."
    )


# ─────────────────────────────────────────────────────────────────────────────
# Main loop
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="Path to the CineGen checkpoint (.pt)")
    ap.add_argument("--data_root", required=True, help="Eval pack directory (must contain index.jsonl + matrices/)")
    ap.add_argument("--out_dir", required=True, help="Where to write generated NPZ + metadata.jsonl")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--ddpm_steps", type=int, default=50)
    ap.add_argument("--cfg_scale", type=float, default=3.5)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    model, ckpt = load_model(args.ckpt, device)
    traj_type = ckpt.get("traj_type") or ckpt["args"].get("traj_type")
    if traj_type is None:
        raise ValueError("Checkpoint does not specify traj_type")

    ae = build_ae(ckpt, traj_type, device)
    if model.sequencer.use_first_pose:
        model.set_ae_encoder(ae.encoder)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    meta_path = out_dir / "metadata.jsonl"
    if meta_path.exists() and not args.overwrite:
        log.info(f"[skip] {out_dir} already contains metadata.jsonl — pass --overwrite to regenerate")
        return

    ds = EvalDataset(args.data_root, traj_type=traj_type)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, collate_fn=collate_fn,
                        pin_memory=True)
    log.info(f"Eval clips: {len(ds)}")

    no_ae = ckpt.get("no_ae", False)

    metadata = []
    with torch.no_grad():
        for batch in tqdm(loader, desc=out_dir.name):
            texts = batch["motion_caption"]
            B = len(texts)
            max_len = int(batch["seq_len"].max().item())
            L_z = max_len if no_ae else (max_len + 3) // 4

            first_pose = batch["first_pose"].to(device) if model.sequencer.use_first_pose else None
            aspects = batch["cinematic_aspects"] if (model.sequencer.use_aspects or model.sequencer.use_logline) else None

            z = model.sample(
                motion_captions=texts,
                cinematic_aspects=aspects,
                L_z=L_z,
                ddpm_steps=args.ddpm_steps,
                cfg_scale=args.cfg_scale,
                device=device,
                first_pose=first_pose,
            )

            decoded = ae.decode(z.permute(0, 2, 1))  # (B, T_out, D)

            for i in range(B):
                clip_id = batch["clip_id"][i]
                sl = int(batch["seq_len"][i].item())
                feat = decoded[i, :sl].cpu().numpy()

                save_dict = {}
                if traj_type == "direction+speed":
                    fp_np = batch["first_pose"][i].numpy()
                    matrices = dirspd_to_matrices(feat, first_pose=fp_np)
                    save_dict["matrices"]      = matrices
                    save_dict["direction_seq"] = feat[:, :6]
                    save_dict["speed_seq"]     = feat[:, 6:]
                else:
                    matrices = traj_feat_to_matrices(feat)
                    save_dict["matrices"] = matrices

                np.savez_compressed(out_dir / f"{clip_id}.npz", **save_dict)

                ca = batch["cinematic_aspects"]
                metadata.append({
                    "clip_id":         clip_id,
                    "npz":             f"{clip_id}.npz",
                    "motion_caption":  batch["motion_caption"][i],
                    "traj_type":       traj_type,
                    "logline_script":  ca.get("logline_script", [""])[i],
                })

    with open(meta_path, "w") as f:
        for m in metadata:
            f.write(json.dumps(m) + "\n")
    log.info(f"Wrote {len(metadata)} samples to {out_dir}")


if __name__ == "__main__":
    main()
