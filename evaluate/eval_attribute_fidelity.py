"""
Attribute fidelity evaluation: do generated trajectories preserve movie-level
attribute information (era, genre, country, director)?

Pipeline:
  1. Load 4 attribute classifiers (per traj_type: dsp_*.pt, traj_*.pt)
  2. For each labeled clip in the val split:
     a. Run classifier on REAL trajectory features → P_real(class)
     b. For each gen model, run classifier on GENERATED features → P_gen(class)
  3. Report per-attribute, per-gen-model:
     - Accuracy of classifier-on-generated against ground-truth label
     - Comparison vs classifier-on-real (ceiling)

Usage:
    PYTHONPATH=. python evaluate/eval_attribute_fidelity.py
"""

import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score, accuracy_score, balanced_accuracy_score, roc_auc_score

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from evaluate.run_missing_experiments import DspWithTrajStats, TrajWithStatsModel
from evaluate.train_multimodal_clf import MultimodalClassifier
from evaluate.eval_genmodel import matrices_to_feat
from evaluate.data.real_label_dataset import (
    year_to_era, ERA_CLASSES, GENRE_COARSE_MAP, COUNTRY_REGION_MAP
)
from cinegen.utils.pose_utils import np_matrices_to_velocity

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

ROOT = Path("/workspace/writeable/datasets/DIY_movies")
CLF_DIR = REPO / "checkpoints/best_clf"
GEN_ROOT = REPO / "generated/pulp"
LOG_SPEED_EPS = 1e-6

# Map our 4 attributes to checkpoints + label types
ATTRIBUTES = {
    "era": {
        "ckpt_dsp": "dsp_era_6L_nf_lr5e4_nosamp.pt",
        "ckpt_traj": "traj_era_6L_nf_lr5e4_nosamp.pt",
    },
    "genre_primary": {
        "ckpt_dsp": "dsp_genre_primary_stats_6L_nf.pt",
        "ckpt_traj": "traj_genre_primary_stats_6L_nf.pt",
    },
    "country_region": {
        "ckpt_dsp": "dsp_country_region_6L_w160_nf_nosamp.pt",
        "ckpt_traj": "traj_country_region_6L_w160_nf_nosamp.pt",
    },
    "director": {
        "ckpt_dsp": "dsp_director_stats_wide_nosamp.pt",
        "ckpt_traj": "traj_director_stats_wide_nosamp.pt",
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# Label derivation (matches real_label_dataset)
# ─────────────────────────────────────────────────────────────────────────────

def derive_label(clip_info: dict, label_type: str) -> Optional[str]:
    info = clip_info.get("movie_info", {})
    if label_type == "era":
        y = info.get("year")
        return year_to_era(y) if y is not None else None
    if label_type == "genre_primary":
        for g in info.get("genres", []):
            if g in GENRE_COARSE_MAP:
                return GENRE_COARSE_MAP[g]
        return None
    if label_type == "country_region":
        for c in info.get("countries", []):
            if c in COUNTRY_REGION_MAP:
                return COUNTRY_REGION_MAP[c]
        return "Other" if info.get("countries") else None
    if label_type == "director":
        directors = info.get("directors", [])
        return directors[0] if directors else None
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Feature extraction
# ─────────────────────────────────────────────────────────────────────────────

def matrices_to_dirspd_feat(matrices: np.ndarray, max_len: int = 299):
    """(N, 4, 4) c2w → padded (max_len, 8) dir+spd features + actual_len."""
    td, rd, ts, rs, _ = np_matrices_to_velocity(matrices)
    actual_len = min(len(td), max_len)
    ts_log = np.log(ts[:actual_len] + LOG_SPEED_EPS).astype(np.float32)
    rs_log = np.log(rs[:actual_len] + LOG_SPEED_EPS).astype(np.float32)
    raw = np.concatenate([
        td[:actual_len].astype(np.float32),
        rd[:actual_len].astype(np.float32),
        ts_log[:, None], rs_log[:, None],
    ], axis=-1)
    pad = np.zeros((max_len - actual_len, 8), dtype=np.float32)
    return np.concatenate([raw, pad], axis=0), actual_len


def matrices_to_traj_feat(matrices: np.ndarray, max_len: int = 300):
    """(N, 4, 4) c2w → padded (max_len, 9) [rot6D + rel_trans] + actual_len."""
    actual_len = min(len(matrices), max_len)
    R = matrices[:actual_len, :3, :3]
    rot6d = R[:, :, :2].transpose(0, 2, 1).reshape(-1, 6).astype(np.float32)
    rel_trans = (matrices[:actual_len, :3, 3] - matrices[0:1, :3, 3]).astype(np.float32)
    raw = np.concatenate([rot6d, rel_trans], axis=-1)
    pad = np.zeros((max_len - actual_len, 9), dtype=np.float32)
    return np.concatenate([raw, pad], axis=0), actual_len


def first_pose_dirspd(matrices: np.ndarray) -> np.ndarray:
    """Compute 8D first_pose for dir+spd: [trans_dir(3), rot_dir(3), log_ts(1), log_rs(1)]."""
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
        [np.log(t0_speed + LOG_SPEED_EPS)],
        [np.log(r0_speed + LOG_SPEED_EPS)],
    ]).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Classifier loading
# ─────────────────────────────────────────────────────────────────────────────

def load_classifier(ckpt_path: Path, traj_type: str, device, fallback_label2idx=None):
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg = ckpt["config"]
    label2idx = ckpt.get("label2idx") or fallback_label2idx
    if label2idx is None:
        raise ValueError(f"No label2idx in {ckpt_path} and no fallback")
    label_names = ckpt.get("label_names", list(label2idx.keys()))
    num_classes = len(label2idx)

    # Detect max_len from pos_embed shape
    state = ckpt["model"]
    max_len = state["pos_embed.weight"].shape[0]

    # Auto-detect architecture from state_dict keys
    has_pose_proj = "pose_proj.0.weight" in state
    has_stat_proj = "stat_proj.0.weight" in state
    has_traj_norm = "traj_norm.weight" in state

    input_dim = 8 if traj_type == "direction+speed" else 9

    if has_traj_norm:
        # MultimodalClassifier (simpler — no traj_stats)
        model = MultimodalClassifier(
            input_dim=input_dim,
            d_model=cfg["d_model"], nhead=cfg["nhead"],
            num_layers=cfg["num_layers"], max_len=max_len,
            num_classes=num_classes,
            depth_size=64, depth_dim=128, pose_dim=64,
            use_depth=cfg.get("use_depth", True),
            use_first_pose=has_pose_proj,
            dropout=cfg.get("dropout", 0.1),
        ).to(device)
        model_kind = "multimodal"
    else:
        # DspWithTrajStats / TrajWithStatsModel (with traj_stats)
        cls = DspWithTrajStats if traj_type == "direction+speed" else TrajWithStatsModel
        model = cls(
            d_model=cfg["d_model"], nhead=cfg["nhead"],
            num_layers=cfg["num_layers"], max_len=max_len,
            num_classes=num_classes,
            pose_dim=64, stat_dim=64, depth_dim=128,
            use_depth=cfg.get("use_depth", True),
            dropout=cfg.get("dropout", 0.1),
        ).to(device)
        model_kind = "stats"

    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, label2idx, label_names, cfg, max_len, model_kind


# ─────────────────────────────────────────────────────────────────────────────
# Run classifier batched
# ─────────────────────────────────────────────────────────────────────────────

def compute_traj_stats(matrices: np.ndarray) -> np.ndarray:
    """13D global trajectory statistics, matching TrajStatsDataset."""
    trans = matrices[:, :3, 3]
    total_disp = trans[-1] - trans[0]
    disp_mag = float(np.linalg.norm(total_disp))
    disp_height = float(total_disp[1]) if len(total_disp) > 1 else 0.0
    if len(matrices) > 1:
        td, rd, ts, rs, _ = np_matrices_to_velocity(matrices)
        mean_ts, std_ts = float(ts.mean()), float(ts.std())
        mean_rs, std_rs = float(rs.mean()), float(rs.std())
        if len(td) > 1:
            dots = np.sum(td[:-1] * td[1:], axis=-1).clip(-1, 1)
            angles = np.arccos(dots)
            mean_curv, max_curv = float(angles.mean()), float(angles.max())
        else:
            mean_curv = max_curv = 0.0
    else:
        mean_ts = std_ts = mean_rs = std_rs = mean_curv = max_curv = 0.0
    return np.array([
        total_disp[0], total_disp[1], total_disp[2],
        mean_ts, std_ts, mean_rs, std_rs,
        mean_curv, max_curv, len(matrices) / 300.0,
        disp_mag, disp_height, np.log(disp_mag + 1e-6),
    ], dtype=np.float32)


@torch.no_grad()
def run_classifier(model, traj_type, matrices_list, depth_list, device, max_len, model_kind="stats", batch_size=32):
    """Run classifier on a list of c2w matrix sequences. Returns logits (N, C)."""
    all_logits = []
    feat_max = max_len  # matches checkpoint's pos_embed length

    for i in range(0, len(matrices_list), batch_size):
        batch_mats = matrices_list[i:i+batch_size]
        batch_depths = depth_list[i:i+batch_size]

        feats, lens, fps, stats = [], [], [], []
        for m in batch_mats:
            if traj_type == "direction+speed":
                f, l = matrices_to_dirspd_feat(m, max_len=feat_max)
                fps.append(first_pose_dirspd(m))
            else:
                f, l = matrices_to_traj_feat(m, max_len=feat_max)
                fps.append(first_pose_dirspd(m))  # traj model also uses 8D first_pose
            feats.append(f)
            lens.append(l)
            stats.append(compute_traj_stats(m))

        feat_t = torch.from_numpy(np.stack(feats)).to(device)
        lens_t = torch.tensor(lens, device=device)
        depth_t = torch.from_numpy(np.stack(batch_depths)).to(device).float() if model.use_depth else None
        fp_t = torch.from_numpy(np.stack(fps)).to(device)
        stat_t = torch.from_numpy(np.stack(stats)).to(device)

        if model_kind == "stats":
            logits = model(feat_t, lens_t, depth=depth_t, first_pose=fp_t, traj_stats=stat_t)
        else:
            # MultimodalClassifier — only takes traj + depth + (first_pose if dsp)
            if hasattr(model, 'use_first_pose') and model.use_first_pose:
                logits = model(feat_t, lens_t, depth=depth_t, first_pose=fp_t)
            else:
                logits = model(feat_t, lens_t, depth=depth_t)
        all_logits.append(logits.cpu().numpy())
    return np.concatenate(all_logits, axis=0)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    device = torch.device("cuda")

    # 1) Load labeled clips
    log.info("Loading labeled clip mapping...")
    with open(ROOT / "labeling/known_movies/clip_movie_mapping.json") as f:
        labeled = json.load(f)

    # Build clip_id → info dict (using filename without .mp4)
    labeled_by_id = {}
    for entry in labeled:
        clip_id = entry["filename"].replace(".mp4", "")
        labeled_by_id[clip_id] = entry
    log.info(f"Total labeled clips: {len(labeled_by_id)}")

    # 2) Identify val split clips (the ones that have generated outputs)
    # Use first gen dir to find val clips
    sample_gen_dir = next(GEN_ROOT.glob("pulp_*"))
    with open(sample_gen_dir / "metadata.jsonl") as f:
        val_clip_ids = [json.loads(l)["clip_id"] for l in f if l.strip()]
    val_set = set(val_clip_ids)
    log.info(f"Val clips with generated output: {len(val_set)}")

    # Intersect: labeled AND in val
    eval_clip_ids = [cid for cid in val_clip_ids if cid in labeled_by_id]
    log.info(f"Eval clips (labeled ∩ val): {len(eval_clip_ids)}")

    # 3) Load real matrices and depth features for eval clips
    log.info("Loading real matrices + depth features for eval clips...")
    real_mats = {}
    depth_feats = {}
    depth_cache_dir = ROOT / "clip_depth_features"

    for cid in eval_clip_ids:
        # Find pose file across datasets
        for ds_name in ["cinetechbench", "movieshots", "shotbench", "vadb", "condensedmovies"]:
            pose_p = ROOT / "filtered_pose" / ds_name / f"{cid}.npz"
            depth_p = depth_cache_dir / ds_name / f"{cid}.npy"
            if pose_p.exists():
                real_mats[cid] = np.load(pose_p)["data"].astype(np.float32)
                if depth_p.exists():
                    depth_feats[cid] = np.load(depth_p).astype(np.float32)
                else:
                    depth_feats[cid] = np.zeros(128, dtype=np.float32)
                break

    eval_clip_ids = [cid for cid in eval_clip_ids if cid in real_mats]
    log.info(f"After matrix/depth filter: {len(eval_clip_ids)}")

    # 4) Derive labels for each attribute
    log.info("Deriving labels...")
    labels_per_attr = {}
    for attr in ATTRIBUTES:
        labels_per_attr[attr] = {
            cid: derive_label(labeled_by_id[cid], attr)
            for cid in eval_clip_ids
        }
        valid = sum(1 for v in labels_per_attr[attr].values() if v is not None)
        log.info(f"  {attr}: {valid}/{len(eval_clip_ids)} have labels")

    # 5) Find all gen models
    gen_dirs = sorted([d for d in GEN_ROOT.glob("pulp_*") if (d / "metadata.jsonl").exists()])
    log.info(f"Found {len(gen_dirs)} gen models")

    # 6) For each (attribute × traj_type) classifier, evaluate on REAL + each GEN model
    results = {}  # results[attr][model_name] = {"acc": ..., "f1": ..., "bal_acc": ...}

    for attr_name, attr_cfg in ATTRIBUTES.items():
        log.info(f"\n=== Attribute: {attr_name} ===")
        results[attr_name] = {}
        shared_label2idx = None

        for traj_type, ckpt_key in [("direction+speed", "ckpt_dsp"), ("trajectory", "ckpt_traj")]:
            ckpt_path = CLF_DIR / attr_cfg[ckpt_key]
            if not ckpt_path.exists():
                log.warning(f"Skip {attr_name}/{traj_type}: no ckpt")
                continue

            log.info(f"  Loading classifier ({traj_type})...")
            clf, label2idx, label_names, cfg, max_len, model_kind = load_classifier(
                ckpt_path, traj_type, device, fallback_label2idx=shared_label2idx)
            if shared_label2idx is None:
                shared_label2idx = label2idx

            # Build true labels (only clips with valid label for this attribute)
            attr_labels = labels_per_attr[attr_name]
            valid_clips = [cid for cid in eval_clip_ids
                           if attr_labels[cid] is not None and attr_labels[cid] in label2idx]
            if not valid_clips:
                log.warning(f"    No valid clips for {attr_name}")
                continue

            y_true = np.array([label2idx[attr_labels[cid]] for cid in valid_clips])
            log.info(f"    {len(valid_clips)} valid clips, {len(label2idx)} classes")

            # ---- Real ----
            real_mats_list = [real_mats[cid] for cid in valid_clips]
            depth_list = [depth_feats[cid] for cid in valid_clips]
            logits_real = run_classifier(clf, traj_type, real_mats_list, depth_list, device, max_len, model_kind)
            y_pred_real = logits_real.argmax(axis=-1)

            real_metrics = {
                "acc": accuracy_score(y_true, y_pred_real),
                "f1": f1_score(y_true, y_pred_real, average="macro", zero_division=0),
                "bal_acc": balanced_accuracy_score(y_true, y_pred_real),
            }
            log.info(f"    [REAL]   F1={real_metrics['f1']:.3f} acc={real_metrics['acc']:.3f}")

            results[attr_name][f"REAL_{traj_type}"] = real_metrics

            # ---- Each gen model ----
            for gen_dir in gen_dirs:
                model_name = gen_dir.name
                gen_mats = []
                gen_depth = []
                for cid in valid_clips:
                    npz_p = gen_dir / f"{cid}.npz"
                    if not npz_p.exists():
                        continue
                    data = np.load(npz_p)
                    if "matrices" not in data:
                        continue
                    m = data["matrices"].astype(np.float32)
                    if not np.isfinite(m).all():
                        continue
                    gen_mats.append(m)
                    gen_depth.append(depth_feats[cid])

                # Need to filter labels to match gen_mats order
                gen_valid_clips = [cid for cid in valid_clips
                                   if (gen_dir / f"{cid}.npz").exists()
                                   and "matrices" in np.load(gen_dir / f"{cid}.npz")
                                   and np.isfinite(np.load(gen_dir / f"{cid}.npz")["matrices"]).all()]
                y_true_gen = np.array([label2idx[attr_labels[cid]] for cid in gen_valid_clips])

                if len(gen_mats) == 0:
                    continue

                logits_gen = run_classifier(clf, traj_type, gen_mats, gen_depth, device, max_len, model_kind)
                y_pred_gen = logits_gen.argmax(axis=-1)

                gen_metrics = {
                    "n": len(gen_mats),
                    "acc": accuracy_score(y_true_gen, y_pred_gen),
                    "f1": f1_score(y_true_gen, y_pred_gen, average="macro", zero_division=0),
                    "bal_acc": balanced_accuracy_score(y_true_gen, y_pred_gen),
                }
                results[attr_name][f"{model_name}_{traj_type}"] = gen_metrics
                log.info(f"    [GEN]    {model_name}/{traj_type}  F1={gen_metrics['f1']:.3f} acc={gen_metrics['acc']:.3f}")

            del clf
            torch.cuda.empty_cache()

    # Save
    out_path = REPO / "results/attribute_fidelity.json"
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    log.info(f"\nResults saved to {out_path}")

    # ─── Print summary table ───
    print("\n" + "=" * 120)
    print("  ATTRIBUTE FIDELITY EVALUATION")
    print("=" * 120)
    for attr in ATTRIBUTES:
        print(f"\n--- {attr} ---")
        for traj_type in ["direction+speed", "trajectory"]:
            real_key = f"REAL_{traj_type}"
            if real_key in results[attr]:
                r = results[attr][real_key]
                print(f"  REAL ({traj_type}): F1={r['f1']:.3f}  acc={r['acc']:.3f}  bal_acc={r['bal_acc']:.3f}")
        # Gen models
        for model_key in sorted(results[attr]):
            if model_key.startswith("REAL_"):
                continue
            r = results[attr][model_key]
            print(f"  {model_key}: F1={r['f1']:.3f}  acc={r['acc']:.3f}")
    print("=" * 120)


if __name__ == "__main__":
    main()
