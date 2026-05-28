"""
Random-unmask inference variant — ablates the variance-guided mask selection
in CineGen's autoregressive sampler.

Identical to ``scripts/infer.py`` except: at each AR step, the next batch of
positions to unmask is chosen *uniformly at random* among the still-masked
positions, rather than by lowest variance of the sequencer's hidden state.

Usage
-----
::

    python scripts/infer_random_mask.py \\
        --ckpt checkpoints/cinegen/best.pt \\
        --data_root data/cinegen-eval \\
        --out_dir results/cinegen-generated_randmask
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

from cinegen.dataset import EvalDataset, collate_fn, TRAJ_DIM
from scripts.infer import (
    build_ae,
    dirspd_to_matrices,
    load_model,
    traj_feat_to_matrices,
)

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


@torch.no_grad()
def sample_random_mask(
    model,
    motion_captions,
    cinematic_aspects,
    L_z,
    ddpm_steps,
    cfg_scale,
    device,
    first_pose,
    n_ar_steps=None,
):
    """Variant of CineGen.sample() that picks the next-unmask positions uniformly at random."""
    n_ar_steps = n_ar_steps or model.n_ar_steps
    B = len(motion_captions)
    model.scheduler.to(device)

    texts = model._build_text(motion_captions, cinematic_aspects)
    text_emb = model.encode_text(texts)
    logline_emb = model._build_logline_emb(motion_captions, cinematic_aspects)

    seq, diff = model.ema_sequencer, model.ema_diffuser
    z = seq.mask_latent.expand(B, L_z, -1).clone()

    for step in range(n_ar_steps):
        n_target = round(L_z * (step + 1) / n_ar_steps)
        n_unmasked = round(L_z * step / n_ar_steps)
        n_new = n_target - n_unmasked
        if n_new <= 0:
            continue

        _, cond_hidden = seq(z, text_emb, first_pose=first_pose,
                             aspects=cinematic_aspects, logline_emb=logline_emb)
        _, uncond_hidden = seq(z, text_emb, first_pose=first_pose, force_mask=True)

        is_masked = torch.all(
            torch.abs(z - seq.mask_latent.squeeze()) < 1e-6, dim=-1
        )
        # ── Random selection (vs cond_hidden.var(dim=-1) in the default sampler)
        scores = torch.rand(B, L_z, device=device)
        scores[~is_masked] = float("inf")
        _, idx = scores.topk(n_new, dim=1, largest=False)

        for b in range(B):
            pos = idx[b]
            cond_ctx, uncond_ctx = cond_hidden[b, pos], uncond_hidden[b, pos]
            x = torch.randn(n_new, model.ae_dim, device=device)
            T = model.ddpm_T
            step_size = max(1, T // ddpm_steps)
            for t_val in range(T - 1, -1, -step_size):
                t = torch.full((n_new,), t_val, device=device, dtype=torch.long)
                t_prev = torch.clamp(t - step_size, min=0)
                cond_pred = diff(x, t.float() / T, cond_ctx)
                uncond_pred = diff(x, t.float() / T, uncond_ctx)
                noise_pred = uncond_pred + cfg_scale * (cond_pred - uncond_pred)
                ab = model.scheduler.alpha_bar[t_val]
                x_0_pred = (x - (1 - ab).sqrt() * noise_pred) / ab.sqrt()
                x = model.scheduler.posterior_sample(x, x_0_pred, t, t_prev)
            z[b, pos] = x
    return z


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--out_dir", required=True)
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
    ae = build_ae(ckpt, traj_type, device)
    if model.sequencer.use_first_pose:
        model.set_ae_encoder(ae.encoder)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    meta_path = out_dir / "metadata.jsonl"
    if meta_path.exists() and not args.overwrite:
        log.info(f"[skip] {out_dir} already contains metadata.jsonl — pass --overwrite")
        return

    ds = EvalDataset(args.data_root, traj_type=traj_type)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, collate_fn=collate_fn,
                        pin_memory=True)

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

            z = sample_random_mask(
                model, texts, aspects, L_z,
                ddpm_steps=args.ddpm_steps, cfg_scale=args.cfg_scale,
                device=device, first_pose=first_pose,
            )
            decoded = ae.decode(z.permute(0, 2, 1))

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
