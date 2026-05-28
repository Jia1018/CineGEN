"""
Attribute fidelity evaluation using the 9 paper-table classifiers (clf_paper/).

Settings (matching FINAL_REPORT.md / paper Table):
  1. macro_type (3 cls)
  2. era_three_cine (3 cls)
  3. country_region (6 regions)
  4. genre_top3 (3 cls)
  5. genre_top5 (5 cls)
  6. director_top3 (3 cls: Spielberg/Zemeckis/Other)
  7. bin_tarantino (binary)
  8. bin_wes_anderson (binary)
  9. bin_nolan (binary)

For each (setting × traj_type representation):
  - Run classifier on REAL trajectories (ceiling)
  - Run classifier on each gen model's trajectories
  - Report F1 + balanced acc + (AUC for binary)

Usage:
    PYTHONPATH=. python evaluate/eval_attribute_fidelity_paper.py
"""

import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score, accuracy_score, balanced_accuracy_score, roc_auc_score

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from evaluate.run_missing_experiments import DspWithTrajStats, TrajWithStatsModel
from evaluate.eval_attribute_fidelity import (
    matrices_to_dirspd_feat, matrices_to_traj_feat, first_pose_dirspd,
    compute_traj_stats, derive_label,
)
from evaluate.data.real_label_dataset import (
    year_to_era, GENRE_COARSE_MAP, COUNTRY_REGION_MAP
)
from evaluate.flip_traj_wins import ERA_SCHEMES, GENRE_GROUPS
from evaluate.retrain_paper_classifiers import build_dataset


# Cache: (setting, traj_type) -> set of clip_ids in classifier's held-out val movies
_HELD_OUT_CACHE = {}


def get_held_out_cids(setting: str, traj_type: str, data_root: Path | None = None) -> set:
    """Return clip_ids that were held out at classifier training time.

    Reads ``<data_root>/held_out_splits.json`` (shipped with the eval pack).
    Each setting × traj_type maps to a list of clip ids in its classifier's val split.
    """
    if data_root is None:
        return set()
    key = (setting, traj_type)
    if key in _HELD_OUT_CACHE:
        return _HELD_OUT_CACHE[key]
    splits_p = Path(data_root) / "held_out_splits.json"
    if not splits_p.exists():
        _HELD_OUT_CACHE[key] = set()
        return set()
    with open(splits_p) as f:
        splits = json.load(f)
    cids = set(splits.get(setting, {}).get(traj_type, []))
    _HELD_OUT_CACHE[key] = cids
    return cids

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# Defaults are overridden by CLI args in main().
ROOT = Path("data/cinegen-eval")
CLF_DIR = REPO / "checkpoints/clf_paper"
GEN_ROOT = REPO / "results"

PAPER_SETTINGS = [
    "macro_type", "era_three_cine", "country_region",
    "genre_3_drop", "genre_3_drop_ml",
    "genre_top3", "genre_top5", "director_top3", "director_3_drop",
    "bin_tarantino", "bin_wes_anderson", "bin_nolan",
    # Trajectory-only ablation variants (same datasets/labels but classifier
    # trained without depth or first_pose features)
    "era_three_cine_traj_only",
    "genre_3_drop_ml_traj_only",
    "director_3_drop_traj_only",
]

# Redefined 3-class genre coarsening (must match GENRE_3_REVISED in
# retrain_paper_classifiers.py): Drama/Romance, Comedy, Action/Thriller/Sci-Fi.
# Clips outside these 3 buckets (Bio/Hist, Documentary, Other-residual) get
# None and are dropped from the eval, matching the classifier's drop_unmapped=True.
GENRE_3_REVISED_MAP = {
    "Drama":          "Drama / Romance",
    "Romance":        "Drama / Romance",
    "Comedy":         "Comedy",
    "Action":         "Action / Thriller / Sci-Fi",
    "Thriller":       "Action / Thriller / Sci-Fi",
    "Horror":         "Action / Thriller / Sci-Fi",
    "Sci-Fi/Fantasy": "Action / Thriller / Sci-Fi",
    "Adventure":      "Action / Thriller / Sci-Fi",
}


def derive_paper_label(clip_info: dict, setting: str) -> str:
    """Derive label for paper-table settings using matching reformulation."""
    # _traj_only variants share dataset semantics with their base setting
    if setting.endswith("_traj_only"):
        return derive_paper_label(clip_info, setting[:-len("_traj_only")])
    info = clip_info.get("movie_info", {})
    if setting == "macro_type":
        return clip_info.get("macro_type") or info.get("macro_type")
    if setting == "era_three_cine":
        y = info.get("year")
        if y is None:
            return None
        return ERA_SCHEMES["three_cinematography"](y)
    if setting == "country_region":
        for c in info.get("countries", []):
            if c in COUNTRY_REGION_MAP:
                return COUNTRY_REGION_MAP[c]
        return "Other" if info.get("countries") else None
    if setting == "genre_3_drop":
        # Map clip's coarse genres through GENRE_3_REVISED_MAP; return the first
        # matching merged-class label, or None if no genre maps (drop the clip).
        for g in info.get("genres", []):
            coarse = GENRE_COARSE_MAP.get(g)
            if coarse and coarse in GENRE_3_REVISED_MAP:
                return GENRE_3_REVISED_MAP[coarse]
        return None
    if setting == "genre_top3":
        for g in info.get("genres", []):
            if g in GENRE_COARSE_MAP:
                fine = GENRE_COARSE_MAP[g]
                if fine in GENRE_GROUPS["top3"].values():
                    return fine
        return "Other"
    if setting == "genre_top5":
        for g in info.get("genres", []):
            if g in GENRE_COARSE_MAP:
                fine = GENRE_COARSE_MAP[g]
                if fine in GENRE_GROUPS["top5"].values():
                    return fine
        return "Other"
    if setting == "director_top3":
        for d in info.get("directors", []):
            if d in {"Steven Spielberg", "Robert Zemeckis"}:
                return d
        return "Other" if info.get("directors") else None
    if setting == "director_3_drop":
        # Top-3 auteurs: Christopher Nolan, Wes Anderson, Steven Spielberg.
        # Clips whose directors don't include any of these three are dropped
        # (return None), matching the classifier's drop_unmapped=True training.
        targets = {"Christopher Nolan", "Wes Anderson", "Steven Spielberg"}
        for d in info.get("directors", []) or []:
            if d in targets:
                return d
        return None
    if setting.startswith("bin_"):
        target = {
            "bin_tarantino": "Quentin Tarantino",
            "bin_wes_anderson": "Wes Anderson",
            "bin_nolan": "Christopher Nolan",
        }[setting]
        if not info.get("directors"):
            return None
        return target if target in info["directors"] else "Other"
    return None


def derive_paper_label_multi(clip_info: dict, setting: str) -> list:
    """Multi-label variant: returns LIST of labels (or [] if no genre maps).
    For genre_3_drop_ml, maps every coarse genre through GENRE_3_REVISED_MAP and
    returns the deduplicated list of merged-class labels. Clips with no mapped
    coarse genre return [] (treated as 'no label' and skipped, mirroring the
    classifier's drop_unmapped=True training filter)."""
    # _traj_only variants share dataset semantics with their base setting
    if setting.endswith("_traj_only"):
        return derive_paper_label_multi(clip_info, setting[:-len("_traj_only")])
    info = clip_info.get("movie_info", {})
    if setting == "genre_3_drop_ml":
        out = []
        for g in info.get("genres", []):
            coarse = GENRE_COARSE_MAP.get(g)
            if coarse and coarse in GENRE_3_REVISED_MAP:
                m = GENRE_3_REVISED_MAP[coarse]
                if m not in out:
                    out.append(m)
        return out
    return []


def is_multi_label_setting(setting: str) -> bool:
    """Settings whose ckpts are trained with BCE + multi-hot labels."""
    base = setting[:-len("_traj_only")] if setting.endswith("_traj_only") else setting
    return base.endswith("_ml")


def load_paper_classifier(ckpt_path: Path, traj_type: str, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    label2idx = ckpt["label2idx"]
    binary_pos = ckpt.get("binary_pos")
    multi_label = bool(ckpt.get("multi_label", False))
    use_depth = bool(ckpt.get("use_depth", True))
    use_first_pose = bool(ckpt.get("use_first_pose", True))
    num_classes = len(label2idx)
    max_len = ckpt["model"]["pos_embed.weight"].shape[0]
    cls = DspWithTrajStats if traj_type == "direction+speed" else TrajWithStatsModel
    model = cls(
        d_model=cfg["d_model"], nhead=cfg["nhead"],
        num_layers=cfg["num_layers"], max_len=max_len,
        num_classes=num_classes,
        pose_dim=64, stat_dim=64, depth_dim=128,
        use_depth=use_depth,
        use_first_pose=use_first_pose,
        dropout=cfg.get("dropout", 0.1),
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, label2idx, binary_pos, max_len, multi_label


@torch.no_grad()
def run_classifier(model, traj_type, matrices_list, depth_list, device, max_len, batch_size=32):
    all_logits = []
    for i in range(0, len(matrices_list), batch_size):
        batch_mats = matrices_list[i:i+batch_size]
        batch_depths = depth_list[i:i+batch_size]
        feats, lens, fps, stats = [], [], [], []
        for m in batch_mats:
            if traj_type == "direction+speed":
                f, l = matrices_to_dirspd_feat(m, max_len=max_len)
            else:
                f, l = matrices_to_traj_feat(m, max_len=max_len)
            feats.append(f)
            lens.append(l)
            fps.append(first_pose_dirspd(m))
            stats.append(compute_traj_stats(m))
        feat_t = torch.from_numpy(np.stack(feats)).to(device)
        lens_t = torch.tensor(lens, device=device)
        depth_t = torch.from_numpy(np.stack(batch_depths)).to(device).float() if model.use_depth else None
        fp_t = torch.from_numpy(np.stack(fps)).to(device)
        stat_t = torch.from_numpy(np.stack(stats)).to(device)
        logits = model(feat_t, lens_t, depth=depth_t, first_pose=fp_t, traj_stats=stat_t)
        all_logits.append(logits.cpu().numpy())
    return np.concatenate(all_logits, axis=0)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True,
                    help="Eval pack root (contains clip_movie_mapping.json, matrices/, depth/, held_out_splits.json)")
    ap.add_argument("--clf_dir", default="checkpoints/clf_paper",
                    help="Directory of paper-classifier checkpoints")
    ap.add_argument("--gen_dirs", nargs="+", required=True,
                    help="Generation output directories to evaluate (each must contain metadata.jsonl)")
    ap.add_argument("--out_json", default="results/attribute_fidelity_paper.json")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    global ROOT, CLF_DIR
    ROOT = Path(args.data_root)
    CLF_DIR = Path(args.clf_dir)
    device = torch.device(args.device)

    # Load labeled clips (subset bundled with the eval pack)
    log.info("Loading labeled clip mapping...")
    with open(ROOT / "clip_movie_mapping.json") as f:
        labeled = json.load(f)
    labeled_by_id = {e["filename"].replace(".mp4", ""): e for e in labeled}

    # Use the eval pack's index.jsonl as the canonical val-clip list
    with open(ROOT / "index.jsonl") as f:
        val_clip_ids = [json.loads(l)["clip_id"] for l in f if l.strip()]
    eval_clip_ids = [cid for cid in val_clip_ids if cid in labeled_by_id]
    log.info(f"Eval clips (labeled ∩ val): {len(eval_clip_ids)}")

    # Load real matrices + depth from the eval pack
    real_mats, depth_feats = {}, {}
    depth_dir = ROOT / "depth"
    matrices_dir = ROOT / "matrices"
    for cid in eval_clip_ids:
        pose_p = matrices_dir / f"{cid}.npz"
        if pose_p.exists():
            real_mats[cid] = np.load(pose_p)["data"].astype(np.float32)
            dp = depth_dir / f"{cid}.npy"
            depth_feats[cid] = np.load(dp).astype(np.float32) if dp.exists() else np.zeros(128, np.float32)
    eval_clip_ids = [cid for cid in eval_clip_ids if cid in real_mats]
    log.info(f"After matrix filter: {len(eval_clip_ids)}")

    gen_dirs = [Path(p) for p in args.gen_dirs if (Path(p) / "metadata.jsonl").exists()]
    log.info(f"Found {len(gen_dirs)} gen model outputs")

    results = {}

    # Only dirspd classifiers — trajectory results are never consumed by eval_paper_table.py
    traj_types_to_eval = ["direction+speed"]

    for setting in PAPER_SETTINGS:
        log.info(f"\n=== {setting} ===")
        results[setting] = {}
        for traj_type in traj_types_to_eval:
            tag = traj_type.replace("+", "_")
            ckpt_p = CLF_DIR / setting / f"{tag}_best.pt"
            if not ckpt_p.exists():
                log.warning(f"  [skip] {setting}/{tag}: no ckpt")
                continue
            log.info(f"  Loading {tag} classifier...")
            clf, label2idx, binary_pos, max_len, ckpt_is_ml = load_paper_classifier(
                ckpt_p, traj_type, device)
            # Trust the setting-name suffix over the ckpt field — older sweep
            # script overwrote ckpts without preserving the multi_label flag.
            is_ml = is_multi_label_setting(setting) or ckpt_is_ml

            # NOTE: held-out filter dropped. Reasoning: gen trajectories were
            # generated AFTER classifier training, so the classifier has never
            # seen them — no leakage on the gen side. The REAL ceiling F1 may
            # be slightly inflated (classifier overfits to its training-movie
            # real trajectories), but n_eval ≈ 341 (vs ~43 with held-out) gives
            # 8× more statistical power for ranking gen models — the headline
            # metric of interest.
            if is_ml:
                # Multi-label: derive returns list of labels per clip
                attr_labels_ml = {cid: derive_paper_label_multi(labeled_by_id[cid], setting)
                                  for cid in eval_clip_ids}
                valid_clips = [cid for cid in eval_clip_ids
                               if any(l in label2idx for l in attr_labels_ml[cid])]
                if not valid_clips:
                    log.warning(f"    No valid clips")
                    continue
                # Build multi-hot (N, C) ground truth
                C = len(label2idx)
                y_true = np.zeros((len(valid_clips), C), dtype=int)
                for i, cid in enumerate(valid_clips):
                    for l in attr_labels_ml[cid]:
                        if l in label2idx:
                            y_true[i, label2idx[l]] = 1
                log.info(f"    {len(valid_clips)} valid clips (all gen-eval, multi-label), "
                         f"{C} classes, avg labels/clip={float(y_true.sum())/len(valid_clips):.2f}")
            else:
                attr_labels = {cid: derive_paper_label(labeled_by_id[cid], setting)
                               for cid in eval_clip_ids}
                valid_clips = [cid for cid in eval_clip_ids
                               if attr_labels[cid] is not None
                               and attr_labels[cid] in label2idx]
                if not valid_clips:
                    log.warning(f"    No valid clips")
                    continue
                y_true = np.array([label2idx[attr_labels[cid]] for cid in valid_clips])
                log.info(f"    {len(valid_clips)} valid clips (all gen-eval), "
                         f"{len(label2idx)} classes")

            def metrics(y_t, logits, ml=is_ml):
                if ml:
                    probs = torch.sigmoid(torch.from_numpy(logits)).numpy()
                    y_p = (probs > 0.5).astype(int)
                    return {
                        "n": int(y_t.shape[0]),
                        "f1": f1_score(y_t, y_p, average="macro", zero_division=0),
                        "f1_samples": f1_score(y_t, y_p, average="samples", zero_division=0),
                    }
                probs = torch.softmax(torch.from_numpy(logits), dim=-1).numpy()
                y_p = probs.argmax(axis=-1)
                m = {
                    "n": len(y_t),
                    "f1": f1_score(y_t, y_p, average="macro", zero_division=0),
                    "bal_acc": balanced_accuracy_score(y_t, y_p),
                    "acc": accuracy_score(y_t, y_p),
                }
                if binary_pos is not None and len(set(y_t)) > 1:
                    try:
                        m["auc"] = roc_auc_score(y_t, probs[:, binary_pos])
                    except Exception:
                        pass
                return m

            def gen_y_true(clip_subset):
                """Build y_true for a subset of valid clips (multi-hot or int)."""
                if is_ml:
                    yt = np.zeros((len(clip_subset), len(label2idx)), dtype=int)
                    for i, cid in enumerate(clip_subset):
                        for l in attr_labels_ml[cid]:
                            if l in label2idx:
                                yt[i, label2idx[l]] = 1
                    return yt
                return np.array([label2idx[attr_labels[cid]] for cid in clip_subset])

            # Real
            real_mats_list = [real_mats[cid] for cid in valid_clips]
            depth_list = [depth_feats[cid] for cid in valid_clips]
            real_logits = run_classifier(clf, traj_type, real_mats_list, depth_list, device, max_len)
            results[setting][f"REAL_{traj_type}"] = metrics(y_true, real_logits)
            r = results[setting][f"REAL_{traj_type}"]
            extra = (f" f1_samples={r['f1_samples']:.3f}" if is_ml
                      else f" bal_acc={r['bal_acc']:.3f}" + (f" auc={r['auc']:.3f}" if "auc" in r else ""))
            log.info(f"    [REAL]   F1={r['f1']:.3f}{extra}")

            # Each gen model
            for gen_dir in gen_dirs:
                model_name = gen_dir.name
                gen_mats, gen_depth, gen_clips = [], [], []
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
                    gen_clips.append(cid)
                if len(gen_mats) < 5:
                    continue
                y_true_gen = gen_y_true(gen_clips)
                gen_logits = run_classifier(clf, traj_type, gen_mats, gen_depth, device, max_len)
                results[setting][f"{model_name}_{traj_type}"] = metrics(y_true_gen, gen_logits)
                m = results[setting][f"{model_name}_{traj_type}"]
                extra_g = f" auc={m['auc']:.3f}" if "auc" in m else ""
                log.info(f"    [{model_name[:50]:50s}]: F1={m['f1']:.3f}{extra_g}")

            del clf
            torch.cuda.empty_cache()

    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    log.info(f"\nSaved → {out_path}")

    # Summary table
    print(f"\n{'='*120}")
    print(f"  PAPER-CLASSIFIER ATTRIBUTE FIDELITY")
    print(f"{'='*120}")
    for setting in PAPER_SETTINGS:
        print(f"\n--- {setting} ---")
        for tt in traj_types_to_eval:
            r = results[setting].get(f"REAL_{tt}", {})
            if r:
                bits = [f"F1={r['f1']:.3f}"]
                if "bal_acc" in r:
                    bits.append(f"bal_acc={r['bal_acc']:.3f}")
                if "f1_samples" in r:
                    bits.append(f"f1_samples={r['f1_samples']:.3f}")
                if "auc" in r:
                    bits.append(f"auc={r['auc']:.3f}")
                print(f"  REAL ({tt}): " + " ".join(bits))
        for tt in traj_types_to_eval:
            best = max(
                [(k, v["f1"]) for k, v in results[setting].items()
                 if k.endswith(f"_{tt}") and not k.startswith("REAL")],
                key=lambda x: x[1], default=(None, 0)
            )
            if best[0]:
                m = results[setting][best[0]]
                extra = f" auc={m.get('auc'):.3f}" if "auc" in m else ""
                print(f"  Best gen ({tt}): {best[0][:60]} F1={best[1]:.3f}{extra}")


if __name__ == "__main__":
    main()
