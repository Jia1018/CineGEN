"""
Unified evaluation across alignment spaces.

Evaluates generated trajectories in BOTH dir+spd and trajectory alignment spaces.
For dir+spd models, also computes direct (no reconstruction) metrics.

Usage:
    # Single model
    PYTHONPATH=. python evaluate/eval_unified.py \
        --gen_dir generated/latent_adaln/mar_adaln_dirspd_combined \
        --model_traj_type direction+speed

    # Batch: all models in a directory
    PYTHONPATH=. python evaluate/eval_unified.py --batch --gen_root generated/latent_adaln

    # Specific alignment checkpoints
    PYTHONPATH=. python evaluate/eval_unified.py \
        --gen_dir ... \
        --dirspd_align checkpoints/align_v2/direction+speed_motion+logline_script/best.pt \
        --traj_align checkpoints/align_v2/trajectory_motion+logline_script/best.pt
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluate.eval_genmodel import (
    load_model, encode_trajectories, encode_texts,
    frechet_distance, clatr_score,
    matrices_to_feat, contrastive_metrics, segment_trajectory,
    compute_f1,
)


def coverage(real_feats, gen_feats, k=3, num_splits=5):
    """Coverage with num_splits averaging (matching GenDoP/PulpMotion scheme)."""
    from sklearn.metrics import pairwise_distances
    if num_splits <= 1:
        dist_rr = pairwise_distances(real_feats, real_feats)
        np.fill_diagonal(dist_rr, np.inf)
        real_knn = np.sort(dist_rr, axis=1)[:, k - 1]
        dist_rf = pairwise_distances(real_feats, gen_feats)
        nearest_fake = dist_rf.min(axis=1)
        return float((nearest_fake < real_knn).mean())
    else:
        reals = np.array_split(real_feats, num_splits)
        fakes = np.array_split(gen_feats, num_splits)
        covs = []
        for r, f in zip(reals, fakes):
            dist_rr = pairwise_distances(r, r)
            np.fill_diagonal(dist_rr, np.inf)
            real_knn = np.sort(dist_rr, axis=1)[:, k - 1]
            dist_rf = pairwise_distances(r, f)
            nearest_fake = dist_rf.min(axis=1)
            covs.append(float((nearest_fake < real_knn).mean()))
        return float(np.mean(covs))

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


# ─────────────────────────────────────────────────────────────────────────────
# Conversion helpers
# ─────────────────────────────────────────────────────────────────────────────

def dirspd_to_matrices(dirspd: np.ndarray, first_pose: np.ndarray = None) -> np.ndarray:
    """
    Reconstruct (N+1, 4, 4) c2w matrices from dir+spd features (N, 8).
    dirspd: (T, 8) = [trans_dir(3), rot_dir(3), log_trans_speed(1), log_rot_speed(1)]
    """
    T = dirspd.shape[0]
    LOG_SPEED_EPS = 1e-6

    trans_dir = dirspd[:, :3]
    rot_dir = dirspd[:, 3:6]
    log_ts = dirspd[:, 6]
    log_rs = dirspd[:, 7]

    ts = np.maximum(np.exp(log_ts) - LOG_SPEED_EPS, 0)
    rs = np.maximum(np.exp(log_rs) - LOG_SPEED_EPS, 0)

    trans_vel = trans_dir * ts[:, None]
    rot_vel = rot_dir * rs[:, None]

    matrices = np.zeros((T + 1, 4, 4), dtype=np.float32)
    matrices[:, 3, 3] = 1.0

    R = np.eye(3, dtype=np.float32)
    t = np.zeros(3, dtype=np.float32)

    if first_pose is not None and len(first_pose) >= 8:
        from scipy.spatial.transform import Rotation as Rot
        fp_trans_dir = first_pose[:3]
        fp_rot_dir = first_pose[3:6]
        fp_ts = max(np.exp(first_pose[6]) - LOG_SPEED_EPS, 0)
        fp_rs = max(np.exp(first_pose[7]) - LOG_SPEED_EPS, 0)
        t = fp_trans_dir * fp_ts
        rot_aa = fp_rot_dir * fp_rs
        if np.linalg.norm(rot_aa) > 1e-8:
            R = Rot.from_rotvec(rot_aa).as_matrix().astype(np.float32)

    matrices[0, :3, :3] = R
    matrices[0, :3, 3] = t

    for i in range(T):
        t = t + trans_vel[i]
        if np.linalg.norm(rot_vel[i]) > 1e-8:
            from scipy.spatial.transform import Rotation as Rot
            dR = Rot.from_rotvec(rot_vel[i]).as_matrix().astype(np.float32)
            R = R @ dR
        matrices[i + 1, :3, :3] = R
        matrices[i + 1, :3, 3] = t

    return matrices


# ─────────────────────────────────────────────────────────────────────────────
# Encoding helpers
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def encode_in_space(model, cfg, matrices_list, device, traj_type, max_len=299, raw=False):
    """Encode matrices into a specific alignment space.
    If raw=True, return unnormalized embeddings (for FCD). Else L2-normalized (for CLaTr-Score)."""
    cfg_override = dict(cfg)
    cfg_override["traj_type"] = traj_type
    if traj_type == "trajectory":
        cfg_override["max_seq_len"] = max_len + 1  # trajectory uses max_seq_len frames
    else:
        cfg_override["max_seq_len"] = max_len + 1  # dir+spd uses max_seq_len - 1 steps

    feats, seq_lens = [], []
    for m in matrices_list:
        ml = max_len if traj_type == "direction+speed" else max_len + 1
        f, sl = matrices_to_feat(m, traj_type, ml)
        feats.append(f)
        seq_lens.append(sl)

    traj_input_full = torch.stack(feats)
    seq_tensor_full = torch.tensor(seq_lens)
    # Chunked encoding to avoid OOM when GPU is shared with training.
    chunk = 64
    embs = []
    for i in range(0, len(feats), chunk):
        ti = traj_input_full[i:i+chunk].to(device)
        st = seq_tensor_full[i:i+chunk].to(device)
        if raw and hasattr(model, 'encode_traj_raw'):
            e = model.encode_traj_raw(ti, st)
        else:
            e = model.encode_traj(ti, st)
        if not raw:
            e = F.normalize(e, dim=-1)
        embs.append(e.cpu())
    emb = torch.cat(embs, dim=0)
    return emb.numpy()


@torch.no_grad()
def encode_dirspd_direct(model, dirspd_list, seq_lens_list, device, max_len=299):
    """Encode dir+spd features directly (no matrix reconstruction)."""
    feats = []
    actual_lens = []
    for ds in dirspd_list:
        al = min(len(ds), max_len)
        ds_clipped = ds[:al].astype(np.float32)
        pad = np.zeros((max_len - al, 8), dtype=np.float32)
        feats.append(torch.from_numpy(np.concatenate([ds_clipped, pad], axis=0)))
        actual_lens.append(al)

    traj_input = torch.stack(feats).to(device)
    seq_tensor = torch.tensor(actual_lens, device=device)
    emb = model.encode_traj(traj_input, seq_tensor)
    return F.normalize(emb, dim=-1).cpu().numpy()


# ─────────────────────────────────────────────────────────────────────────────
# Core eval function
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_model(
    gen_dir: Path,
    model_traj_type: str,
    dirspd_model_path: str,
    traj_model_path: str,
    real_matrices: Dict[str, np.ndarray],
    device: torch.device,
    root: str = "/workspace/writeable/datasets/DIY_movies",
) -> dict:
    """
    Evaluate a single generated model in both alignment spaces.

    Returns dict with keys:
      dirspd_recon:  metrics in dir+spd space (via matrices)
      dirspd_direct: metrics in dir+spd space (raw features, dir+spd models only)
      traj_recon:    metrics in trajectory space (via matrices)
    """
    # Load metadata
    meta_path = gen_dir / "metadata.jsonl"
    with open(meta_path) as f:
        meta = [json.loads(l) for l in f if l.strip()]

    gen_matrices = []
    gen_dirspd = []
    gen_seq_lens = []
    gen_clip_ids = []

    for entry in meta:
        npz_path = gen_dir / entry["npz"]
        if not npz_path.exists():
            continue
        data = np.load(npz_path)
        if "matrices" not in data:
            continue
        mat = data["matrices"].astype(np.float32)
        if not np.isfinite(mat).all():
            continue  # skip samples with NaN/inf matrices
        gen_matrices.append(mat)
        gen_clip_ids.append(entry["clip_id"])

        # Raw dir+spd for direct eval
        if model_traj_type == "direction+speed" and "direction_seq" in data and "speed_seq" in data:
            ds = data["direction_seq"]
            ss = data["speed_seq"]
            if ds.shape[-1] == 6 and ss.shape[-1] == 2:
                gen_dirspd.append(np.concatenate([ds, ss], axis=-1))
                gen_seq_lens.append(len(ds))

    N = len(gen_matrices)
    log.info(f"  Loaded {N} generated trajectories")

    results = {}
    real_mats_list = list(real_matrices.values())

    # Build text for each alignment model
    def build_texts(cfg, meta_entries):
        text_type = cfg.get("text_type", "motion")
        texts = []
        for entry in meta_entries:
            if "+" in str(text_type):
                parts = text_type.split("+")
                segs = []
                for p in parts:
                    if p == "motion":
                        segs.append(f"Camera motion: {entry.get('motion_caption', '')}")
                    else:
                        segs.append(entry.get(p, ""))
                texts.append(". ".join(s for s in segs if s))
            elif text_type == "motion":
                texts.append(entry.get("motion_caption", ""))
            else:
                texts.append(entry.get(text_type, ""))
        return texts

    # Build meta_filtered using the same clip_ids that passed the gen_matrices filter
    # (must stay aligned with gen_matrices — same order, same skips)
    meta_by_id = {e["clip_id"]: e for e in meta}
    meta_filtered = [meta_by_id[cid] for cid in gen_clip_ids]

    # ── Dir+spd alignment space ──
    if dirspd_model_path and Path(dirspd_model_path).exists():
        log.info("  Evaluating in dir+spd alignment space...")
        ds_model, ds_cfg = load_model(dirspd_model_path, device)
        ds_texts = build_texts(ds_cfg, meta_filtered)

        # L2-normalized embeddings for CLaTr-Score
        real_emb_ds = encode_in_space(ds_model, ds_cfg, real_mats_list, device, "direction+speed")
        gen_emb_ds = encode_in_space(ds_model, ds_cfg, gen_matrices, device, "direction+speed")
        text_emb_ds = encode_texts(ds_model, ds_texts, device)

        # Raw (unnormalized) embeddings for FDCLaTr (GenDoP-compatible scale)
        real_emb_ds_raw = encode_in_space(ds_model, ds_cfg, real_mats_list, device, "direction+speed", raw=True)
        gen_emb_ds_raw = encode_in_space(ds_model, ds_cfg, gen_matrices, device, "direction+speed", raw=True)

        results["dirspd_recon"] = {
            "FDCLaTr_raw": frechet_distance(real_emb_ds_raw, gen_emb_ds_raw),
            "FDCLaTr_norm": frechet_distance(real_emb_ds, gen_emb_ds),
            "CLaTr-Score": clatr_score(gen_emb_ds, text_emb_ds),
            "Coverage": coverage(real_emb_ds, gen_emb_ds, k=3, num_splits=5),
        }

        # Direct eval for dir+spd models
        if model_traj_type == "direction+speed" and gen_dirspd:
            gen_emb_direct = encode_dirspd_direct(ds_model, gen_dirspd, gen_seq_lens, device)
            results["dirspd_direct"] = {
                "FDCLaTr_raw": frechet_distance(real_emb_ds_raw, gen_emb_ds_raw),
                "FDCLaTr_norm": frechet_distance(real_emb_ds, gen_emb_direct),
                "CLaTr-Score": clatr_score(gen_emb_direct, text_emb_ds),
                "Coverage": coverage(real_emb_ds, gen_emb_direct, k=3, num_splits=5),
            }

        del ds_model
        torch.cuda.empty_cache()

    # ── Trajectory alignment space ──
    if traj_model_path and Path(traj_model_path).exists():
        log.info("  Evaluating in trajectory alignment space...")
        t_model, t_cfg = load_model(traj_model_path, device)
        t_texts = build_texts(t_cfg, meta_filtered)

        real_emb_t = encode_in_space(t_model, t_cfg, real_mats_list, device, "trajectory")
        gen_emb_t = encode_in_space(t_model, t_cfg, gen_matrices, device, "trajectory")
        text_emb_t = encode_texts(t_model, t_texts, device)

        # Raw embeddings for FDCLaTr
        real_emb_t_raw = encode_in_space(t_model, t_cfg, real_mats_list, device, "trajectory", raw=True)
        gen_emb_t_raw = encode_in_space(t_model, t_cfg, gen_matrices, device, "trajectory", raw=True)

        results["traj_recon"] = {
            "FDCLaTr_raw": frechet_distance(real_emb_t_raw, gen_emb_t_raw),
            "FDCLaTr_norm": frechet_distance(real_emb_t, gen_emb_t),
            "CLaTr-Score": clatr_score(gen_emb_t, text_emb_t),
            "Coverage": coverage(real_emb_t, gen_emb_t, k=3, num_splits=5),
        }

        del t_model
        torch.cuda.empty_cache()

    # ── F1 (space-agnostic, geometric segments) ──
    paired_pred, paired_ref = [], []
    for i, cid in enumerate(gen_clip_ids):
        if cid in real_matrices:
            try:
                ps = segment_trajectory(gen_matrices[i])
                rs = segment_trajectory(real_matrices[cid])
                if len(ps) >= 2 and len(rs) >= 2:
                    paired_pred.append(ps)
                    paired_ref.append(rs)
            except:
                pass
    if paired_pred:
        try:
            f1_res = compute_f1(paired_pred, paired_ref)
            results["F1"] = f1_res["f1"]
            log.info(f"  F1={f1_res['f1']:.4f} on {len(paired_pred)} paired clips")
        except Exception as e:
            log.warning(f"  F1 computation failed: {e}")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Batch runner + report
# ─────────────────────────────────────────────────────────────────────────────

def run_batch(args):
    device = torch.device(args.device)

    # Load real val trajectories once
    from data.dataset import DiyMoviesDataset, AVAILABLE_DATASETS
    val_ds = DiyMoviesDataset(
        root=args.root, datasets=AVAILABLE_DATASETS, split="val",
        val_fraction=0.1, seed=42, traj_type="direction+speed",
        load_rgb=False, load_depth=False, max_seq_len=300,
    )
    real_matrices = {}
    for i in range(len(val_ds)):
        ds_name, clip_id = val_ds.items[i]
        pose_path = Path(args.root) / "filtered_pose" / ds_name / f"{clip_id}.npz"
        if pose_path.exists():
            real_matrices[clip_id] = np.load(pose_path)["data"].astype(np.float32)
    log.info(f"Loaded {len(real_matrices)} real val trajectories")

    # Find all generated model dirs
    gen_root = Path(args.gen_root) if args.gen_root else Path(args.gen_dir).parent
    if args.gen_dir:
        gen_dirs = [Path(args.gen_dir)]
    else:
        gen_dirs = sorted(d for d in gen_root.iterdir() if (d / "metadata.jsonl").exists())

    all_results = {}

    # Auto-select alignment model based on generation model's text mode
    def select_align_paths(gen_name):
        """
        Match generation model's text mode to alignment model's text type.
        Returns (dirspd_align_path, traj_align_path).
        """
        # Parse text mode from gen_name
        # e.g. "pulp_direction_speed_motion_base" → motion
        #      "pulp_trajectory_combined_fp_aspects" → combined (→ motion+logline_script)
        if "_motion_" in gen_name or gen_name.endswith("_motion"):
            ds_suffix = "direction+speed_motion"
            t_suffix = "trajectory_motion"
        elif "_logline_" in gen_name or gen_name.endswith("_logline"):
            ds_suffix = "direction+speed_logline_script"
            t_suffix = "trajectory_logline_script"
        elif "_combined_" in gen_name or gen_name.endswith("_combined"):
            ds_suffix = "direction+speed_motion+logline_script"
            t_suffix = "trajectory_motion+logline_script"
        else:
            # Fallback to combined (most general)
            ds_suffix = "direction+speed_motion+logline_script"
            t_suffix = "trajectory_motion+logline_script"

        ds_path = f"checkpoints/align_v2/{ds_suffix}/best.pt"
        t_path = f"checkpoints/align_v2/{t_suffix}/best.pt"
        return ds_path, t_path

    for gen_dir in gen_dirs:
        name = gen_dir.name
        save_path = Path(args.save_dir) / f"{name}.json"

        if save_path.exists() and not args.overwrite:
            log.info(f"[skip] {name} (exists)")
            all_results[name] = json.load(open(save_path))
            continue

        # Auto-detect traj type from name
        if args.model_traj_type:
            traj_type = args.model_traj_type
        elif "dirspd" in name or "direction_speed" in name:
            traj_type = "direction+speed"
        else:
            traj_type = "trajectory"

        # Auto-select alignment checkpoints based on text mode
        # (unless user explicitly overrides via CLI args)
        if args.dirspd_align is None or args.traj_align is None:
            ds_align, t_align = select_align_paths(name)
            log.info(f"  Auto-selected align: ds={Path(ds_align).parent.name}, traj={Path(t_align).parent.name}")
        else:
            ds_align = args.dirspd_align
            t_align = args.traj_align

        log.info(f"\n{'='*60}")
        log.info(f"  {name} ({traj_type})")
        log.info(f"{'='*60}")

        try:
            results = evaluate_model(
                gen_dir, traj_type,
                ds_align, t_align,
                real_matrices, device, args.root,
            )
        except Exception as e:
            log.error(f"[FAILED] {name}: {e}")
            import traceback
            log.error(traceback.format_exc())
            results = {"error": str(e)}
            torch.cuda.empty_cache()

        all_results[name] = results

        # Save individual result
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "w") as f:
            json.dump(results, f, indent=2)

    # ── Print report ──
    print_report(all_results)

    # Save combined results
    combined_path = Path(args.save_dir) / "_all_results.json"
    with open(combined_path, "w") as f:
        json.dump(all_results, f, indent=2)
    log.info(f"\nAll results saved → {combined_path}")


def print_report(all_results: dict):
    """Print a formatted comparison table."""

    def _fmt(v):
        return f"{v:.2f}" if v is not None else "—"

    def _fmt4(v):
        return f"{v:.4f}" if v is not None else "—"

    # Separate models by type
    dirspd_models = {k: v for k, v in all_results.items() if "dirspd" in k or "direction_speed" in k}
    traj_models = {k: v for k, v in all_results.items() if "traj" in k and k not in dirspd_models}

    print(f"\n{'='*140}")
    print(f"  UNIFIED EVALUATION REPORT")
    print(f"  GT baselines: dir+spd CS=54.61 | trajectory CS=34.87")
    print(f"  FDCLaTr_raw = GenDoP-comparable scale | FDCLaTr_norm = unit-sphere scale")
    print(f"{'='*140}")

    # Header
    print(f"\n{'Model':<42} | {'--- dir+spd alignment ---':^50} | {'--- traj alignment ---':^34} | {'F1':>5}")
    print(f"{'':42} | {'CS(d)':>7} {'CS(r)':>7} {'FD_raw':>7} {'FD_nrm':>7} {'Cov':>6} | {'CS(r)':>7} {'FD_raw':>7} {'FD_nrm':>7} {'Cov':>6} | {'':>5}")
    print("-" * 140)

    for name, r in sorted(all_results.items()):
        dd = r.get("dirspd_direct", {})
        dr = r.get("dirspd_recon", {})
        tr = r.get("traj_recon", {})
        f1 = r.get("F1")

        cs_direct = _fmt(dd.get("CLaTr-Score"))
        cs_recon_ds = _fmt(dr.get("CLaTr-Score"))
        fid_ds_raw = _fmt(dr.get("FDCLaTr_raw"))
        fid_ds_norm = _fmt4(dr.get("FDCLaTr_norm"))
        cov_ds = _fmt4(dr.get("Coverage"))

        cs_recon_t = _fmt(tr.get("CLaTr-Score"))
        fid_t_raw = _fmt(tr.get("FDCLaTr_raw"))
        fid_t_norm = _fmt4(tr.get("FDCLaTr_norm"))
        cov_t = _fmt4(tr.get("Coverage"))

        f1_str = f"{f1:.4f}" if f1 is not None else "—"

        print(f"{name:<42} | {cs_direct:>7} {cs_recon_ds:>7} {fid_ds_raw:>7} {fid_ds_norm:>7} {cov_ds:>6} | {cs_recon_t:>7} {fid_t_raw:>7} {fid_t_norm:>7} {cov_t:>6} | {f1_str:>5}")

    print(f"{'='*140}")


def main():
    p = argparse.ArgumentParser()
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--gen_dir", help="Single model directory")
    g.add_argument("--gen_root", help="Root containing multiple model dirs (batch mode)")
    p.add_argument("--batch", action="store_true", help="(deprecated, use --gen_root)")
    p.add_argument("--model_traj_type", choices=["trajectory", "direction+speed"],
                   help="Override auto-detection of traj type")
    p.add_argument("--dirspd_align", default=None,
                   help="Dir+spd alignment model (default: auto-select by gen model's text mode)")
    p.add_argument("--traj_align", default=None,
                   help="Trajectory alignment model (default: auto-select by gen model's text mode)")
    p.add_argument("--root", default="/workspace/writeable/datasets/DIY_movies")
    p.add_argument("--device", default="cuda")
    p.add_argument("--save_dir", default="results/unified")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    if args.gen_dir:
        args.gen_root = None
    run_batch(args)


if __name__ == "__main__":
    main()
