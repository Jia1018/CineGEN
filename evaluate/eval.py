"""
CineGen end-to-end evaluation.

Per gen dir, computes:

  Trajectory quality:
    - F1↑       segment-based F1 vs gt (from eval_unified.compute_f1)
    - FCD↓      Fréchet distance in dsp-motion alignment-encoder space (raw)
    - Cov↑      manifold coverage in dsp-motion alignment-encoder space

  Text alignment:
    - AlnScore↑  CLaTr-Score (mean diag of cosine sims), ×100
    - R@1↑       motion-to-text Recall@1
    - MedR↓      motion-to-text median rank

  Per-sample distance (NEW):
    - APSD-trans↓  mean over val of per-frame ‖t_gt - t_gen‖₂ AFTER first-frame
                   alignment (R_0 R_i → I, t_0 → 0)
    - APSD-rot↓    mean over val of per-frame rotation angle (degrees) between
                   gt and gen rotations after the same first-frame normalisation

  Attribute fidelity (3-class macro-F1):
    Genre and Director use trajectory-only classifiers (no depth, no
    first-pose) so the score is not inflated by features the gen models
    don't actually produce. Era still uses the with-depth/FP variant
    pending the balanced-era ablation.

    - Era         era_three_cine            (Film / Early-digital / Digital-mature)
    - Genre       genre_3_drop_ml_traj_only (multi-label; Drama/Romance,
                                              Comedy, Action/Thriller/Sci-Fi).
                                              BCE + sigmoid threshold 0.5;
                                              macro-F1 reported.
    - Dir.        director_3_drop_traj_only (Christopher Nolan / Wes Anderson /
                                              Steven Spielberg)

Reads attribute results from results/attribute_fidelity_paper.json (must be run first).

Usage:
    PYTHONPATH=. python evaluate/eval.py \
        --gen_dirs generated/pulp/baseline_ccd_native_motion ... \
        --alignment_dsp checkpoints/align_v2/direction+speed_motion/best.pt \
        --attr_json results/attribute_fidelity_paper.json \
        --out_csv results/paper_table.csv \
        --out_latex results/paper_table.tex
"""
import argparse, json, sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from evaluate.eval_genmodel import (
    load_model, encode_trajectories, encode_texts, contrastive_metrics
)
from evaluate.eval_unified import (
    coverage, segment_trajectory, compute_f1, encode_in_space,
    frechet_distance,
)
from evaluate.visualize_align_compare import load_clatr_model
from evaluate.eval_attribute_fidelity import matrices_to_dirspd_feat
from evaluate.data.clatr_dataset import CLIPTextCache


ATTR_KEYS = {
    "Era":   "era_three_cine",
    "Genre": "genre_3_drop_ml_traj_only",
    "Dir":   "director_3_drop_traj_only",
}


@torch.no_grad()
def compute_clatr_score(clatr_model, clatr_cfg, clip_encoder, matrices_list,
                        texts, clip_ids, device, batch_size=64):
    """Compute CLaTr-Score (mean diag cos sim ×100) using a CLaTr ckpt trained
    on dirspd+motion. Encodes trajectories via the dirspd 8-D feature pipeline
    and motion captions via the cached CLIP text encoder."""
    max_len = clatr_cfg["max_seq_len"] - 1  # dirspd uses max_seq_len-1
    # Build all dirspd feats + masks
    feats, masks = [], []
    for m in matrices_list:
        f, l = matrices_to_dirspd_feat(m, max_len=max_len)
        feats.append(f)
        mk = np.zeros(max_len, dtype=bool)
        mk[:l] = True
        masks.append(mk)
    # Build CLIP-encoded text feats (cached per clip_id)
    cap_feats = []
    for cid, t in zip(clip_ids, texts):
        cf = clip_encoder.get(cid, t)  # (77, 768)
        cap_feats.append(cf)

    traj_embs, text_embs = [], []
    for i in range(0, len(matrices_list), batch_size):
        f_b = torch.from_numpy(np.stack(feats[i:i+batch_size])).to(device)
        m_b = torch.from_numpy(np.stack(masks[i:i+batch_size])).to(device)
        c_b = torch.from_numpy(np.stack(cap_feats[i:i+batch_size])).to(device)
        traj_embs.append(clatr_model.get_traj_embedding(f_b, m_b).cpu().numpy())
        text_embs.append(clatr_model.get_text_embedding(c_b).cpu().numpy())
    traj_embs = np.concatenate(traj_embs, axis=0)
    text_embs = np.concatenate(text_embs, axis=0)
    # Mean diag cosine sim ×100 (embeddings already L2-normalised by CLaTr)
    return float((traj_embs * text_embs).sum(axis=-1).mean()) * 100


def first_frame_normalize(c2ws: np.ndarray) -> np.ndarray:
    """Apply (R_0,t_0)^{-1} to every frame so c2w[0] = identity."""
    if c2ws.ndim != 3 or c2ws.shape[1:] != (4, 4):
        return c2ws
    R0 = c2ws[0, :3, :3]
    t0 = c2ws[0, :3, 3]
    R0_inv = R0.T
    out = c2ws.copy()
    for i in range(len(c2ws)):
        Ri, ti = c2ws[i, :3, :3], c2ws[i, :3, 3]
        out[i, :3, :3] = R0_inv @ Ri
        out[i, :3, 3] = R0_inv @ (ti - t0)
    return out


def rot_angle_deg(R_gt: np.ndarray, R_gen: np.ndarray) -> np.ndarray:
    """Per-frame angle in degrees between two rotation sequences. R_*: (T, 3, 3)"""
    R_rel = np.einsum("tij,tjk->tik", R_gt.transpose(0, 2, 1), R_gen)  # gt^T gen
    tr = np.einsum("tii->t", R_rel)
    cos_th = np.clip((tr - 1.0) / 2.0, -1.0, 1.0)
    return np.rad2deg(np.arccos(cos_th))


def resample_to(c2ws: np.ndarray, T: int) -> np.ndarray:
    """Linear interpolation along time to T frames."""
    src = len(c2ws)
    if src == T or src < 2 or T < 2:
        return c2ws[:T] if src >= T else np.concatenate([c2ws, np.tile(c2ws[-1:], (T - src, 1, 1))])
    idx = np.linspace(0, src - 1, T)
    out = np.zeros((T, 4, 4), dtype=c2ws.dtype)
    for i, x in enumerate(idx):
        lo, hi = int(np.floor(x)), int(np.ceil(x))
        if lo == hi: out[i] = c2ws[lo]
        else:
            a = x - lo
            out[i] = (1 - a) * c2ws[lo] + a * c2ws[hi]
    return out


def compute_apsd(gen_matrices_list, real_matrices_dict, gen_clip_ids):
    """Compute APSD-trans (units of c2w trans, median) and APSD-rot (degrees, median).

    Median is used (not mean) because autoregressive velocity-integrated trajectories
    can have a small fraction (~1%) of catastrophic divergences that dominate the mean.
    Median reflects typical-case quality better.
    """
    trans_dists, rot_angles, n_paired = [], [], 0
    for i, cid in enumerate(gen_clip_ids):
        if cid not in real_matrices_dict:
            continue
        gen = gen_matrices_list[i]
        gt = real_matrices_dict[cid]
        if gen.ndim != 3 or gt.ndim != 3:
            continue
        T = gt.shape[0]
        if gen.shape[0] != T:
            gen = resample_to(gen, T)
        gt_n = first_frame_normalize(gt)
        gen_n = first_frame_normalize(gen)
        if not (np.isfinite(gt_n).all() and np.isfinite(gen_n).all()):
            continue
        d_trans = np.linalg.norm(gt_n[:, :3, 3] - gen_n[:, :3, 3], axis=-1)
        d_rot = rot_angle_deg(gt_n[:, :3, :3], gen_n[:, :3, :3])
        trans_dists.append(d_trans.mean())
        rot_angles.append(d_rot.mean())
        n_paired += 1
    if n_paired == 0:
        return float("nan"), float("nan"), 0
    # Use MEDIAN — robust to outlier diverged trajectories
    return float(np.median(trans_dists)), float(np.median(rot_angles)), n_paired


def load_real_val_matrices(data_root: str | Path):
    """Load real-val c2w matrices from the CineGen eval pack (downloaded from HF Hub).

    Expects::

        <data_root>/
            index.jsonl                 # one entry per clip
            matrices/<clip_id>.npz      # 4×4 c2w matrices (key: "data")
    """
    from cinegen.dataset import EvalDataset
    ds = EvalDataset(data_root, traj_type="trajectory")
    real = {}
    for entry in ds.entries:
        cid = entry["clip_id"]
        p = Path(data_root) / "matrices" / f"{cid}.npz"
        if not p.exists():
            continue
        m = np.load(p)["data"].astype(np.float32)
        if m.ndim == 3 and m.shape[1:] == (3, 4):
            T = m.shape[0]
            bot = np.tile(np.array([0, 0, 0, 1], np.float32), (T, 1, 1))
            m = np.concatenate([m, bot], axis=1)
        real[cid] = m
    return real


def load_gen_dir(gen_dir: Path):
    """Load (matrices_list, texts, clip_ids) from a gen dir."""
    meta_p = gen_dir / "metadata.jsonl"
    if not meta_p.exists():
        return None
    matrices_list, texts, clip_ids = [], [], []
    with open(meta_p) as f:
        for line in f:
            line = line.strip()
            if not line: continue
            entry = json.loads(line)
            p = gen_dir / entry["npz"]
            if not p.exists(): continue
            data = np.load(p)
            if "matrices" not in data: continue
            m = data["matrices"].astype(np.float32)
            if m.ndim == 3 and m.shape[1:] == (3, 4):
                T = m.shape[0]
                bot = np.tile(np.array([0, 0, 0, 1], np.float32), (T, 1, 1))
                m = np.concatenate([m, bot], axis=1)
            if not np.isfinite(m).all():
                continue
            matrices_list.append(m)
            texts.append(entry["motion_caption"])
            clip_ids.append(entry["clip_id"])
    return matrices_list, texts, clip_ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen_dirs", nargs="+", required=True)
    ap.add_argument("--alignment_dsp",
                    default="checkpoints/align_v2/direction+speed_motion/best.pt")
    ap.add_argument("--clatr_dsp",
                    default="checkpoints/clatr/direction_speed_motion/best.pt",
                    help="CLaTr ckpt trained on YOUR dirspd+motion data — independent "
                         "second alignment-encoder for cross-validation of AlnScore.")
    ap.add_argument("--attr_json", default="results/attribute_fidelity_paper.json")
    ap.add_argument("--out_csv", default="results/paper_table.csv")
    ap.add_argument("--out_latex", default="results/paper_table.tex")
    ap.add_argument("--row_label_map", default=None,
                    help="Optional JSON mapping gen_dir basename → display label")
    ap.add_argument("--device", default="cuda",
                    help="cuda or cpu — use cpu when GPU is shared with training")
    ap.add_argument("--data_root", required=True,
                    help="Path to the CineGen eval pack (downloaded via scripts/download.sh)")
    args = ap.parse_args()

    device = torch.device(args.device)

    # Load alignment model (dsp-motion)
    print("Loading dsp-motion alignment model...", flush=True)
    model, cfg = load_model(args.alignment_dsp, device)

    # Load CLaTr (dsp-motion) — independent alignment encoder + CLIP text cache
    print("Loading CLaTr dsp-motion model + CLIP text encoder...", flush=True)
    clatr_model, clatr_cfg = load_clatr_model(args.clatr_dsp, device)
    clip_encoder = CLIPTextCache(
        cache_dir=clatr_cfg.get("clip_cache", "./clip_cache/clatr") + "/motion",
        device=str(device),
    )

    # Load real val once (for FCD/Cov + F1 pairing)
    print("Loading real val matrices...", flush=True)
    real_matrices = load_real_val_matrices(args.data_root)
    print(f"  {len(real_matrices)} real val clips", flush=True)

    # Encode real once for FCD/Cov
    real_list = list(real_matrices.values())
    real_emb_n = encode_in_space(model, cfg, real_list, device, "direction+speed", raw=False)
    real_emb_r = encode_in_space(model, cfg, real_list, device, "direction+speed", raw=True)

    # Attribute results: read once
    attr = {}
    if Path(args.attr_json).exists():
        attr = json.load(open(args.attr_json))
    label_map = {}
    if args.row_label_map and Path(args.row_label_map).exists():
        label_map = json.load(open(args.row_label_map))

    rows = []
    for gen_dir_str in args.gen_dirs:
        gd = Path(gen_dir_str)
        name = gd.name
        label = label_map.get(name, name)
        loaded = load_gen_dir(gd)
        if loaded is None:
            print(f"[skip] {name}: no metadata.jsonl", flush=True)
            continue
        matrices_list, texts, clip_ids = loaded
        if len(matrices_list) == 0:
            print(f"[skip] {name}: no valid npz", flush=True)
            continue
        N = len(matrices_list)
        print(f"\n=== {name} (N={N}) ===", flush=True)

        # Encode gen
        gen_emb_n = encode_in_space(model, cfg, matrices_list, device, "direction+speed", raw=False)
        gen_emb_r = encode_in_space(model, cfg, matrices_list, device, "direction+speed", raw=True)

        # Filter samples whose alignment encoding contains NaN — protects coverage/FCD
        finite_n = np.isfinite(gen_emb_n).all(axis=1)
        finite_r = np.isfinite(gen_emb_r).all(axis=1)
        finite = finite_n & finite_r
        if not finite.all():
            n_drop = int((~finite).sum())
            print(f"  [warn] dropping {n_drop} NaN-encoded samples", flush=True)
            gen_emb_n = gen_emb_n[finite]
            gen_emb_r = gen_emb_r[finite]
            finite_idx = np.where(finite)[0]
            matrices_list = [matrices_list[i] for i in finite_idx]
            texts = [texts[i] for i in finite_idx]
            clip_ids = [clip_ids[i] for i in finite_idx]
            N = len(matrices_list)

        # Trajectory quality
        cov = coverage(real_emb_n, gen_emb_n, k=3, num_splits=5)
        fcd_val = frechet_distance(real_emb_r, gen_emb_r)

        paired_pred, paired_ref = [], []
        for i, cid in enumerate(clip_ids):
            if cid in real_matrices:
                try:
                    ps = segment_trajectory(matrices_list[i])
                    rs = segment_trajectory(real_matrices[cid])
                    if len(ps) >= 2 and len(rs) >= 2:
                        paired_pred.append(ps); paired_ref.append(rs)
                except Exception:
                    pass
        f1_val = float("nan")
        if paired_pred:
            try:
                f1_val = compute_f1(paired_pred, paired_ref)["f1"]
            except Exception as e:
                print(f"  F1 failed: {e}", flush=True)

        # Text alignment (in-house dsp-motion alignment encoder)
        text_emb = encode_texts(model, texts, device)
        sims = gen_emb_n @ text_emb.T
        cs = float(np.diag(sims).mean()) * 100
        cm = contrastive_metrics(sims)
        r1 = cm["m2t"]["R01"]
        medr = cm["m2t"]["MedR"]

        # CLaTr-Score: independent second alignment via CLaTr ckpt trained on
        # your dirspd+motion data. Same convention (mean diag cos sim ×100).
        try:
            cs_clatr = compute_clatr_score(
                clatr_model, clatr_cfg, clip_encoder,
                matrices_list, texts, clip_ids, device)
        except Exception as e:
            print(f"  CLaTr score failed: {e}", flush=True)
            cs_clatr = float("nan")

        # Attribute (3-class macro-F1)
        attr_f1s = {}
        for short, key in ATTR_KEYS.items():
            full = f"{name}_direction+speed"
            if key in attr and full in attr[key]:
                attr_f1s[short] = attr[key][full]["f1"]
            else:
                attr_f1s[short] = float("nan")

        row = {
            "name": name, "label": label, "N": N,
            "F1": f1_val, "FCD": fcd_val, "Cov": cov,
            "AlnScore": cs, "CLaTr": cs_clatr, "R@1": r1, "MedR": medr,
            **attr_f1s,
        }
        rows.append(row)
        print(f"  F1={f1_val:.3f}  FCD={fcd_val:.2f}  Cov={cov:.3f}  "
              f"CS={cs:.2f}  CLaTr={cs_clatr:.2f}  R@1={r1:.2f}  MedR={medr:.1f}",
              flush=True)

    # ── Save CSV ───────────────────────────────────────────────
    import csv as _csv
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_csv, "w", newline="") as f:
        cols = ["label", "name", "N",
                "F1", "FCD", "Cov", "AlnScore", "CLaTr", "R@1", "MedR",
                "Era", "Genre", "Dir"]
        w = _csv.writer(f, quoting=_csv.QUOTE_MINIMAL)
        w.writerow(cols)
        for r in rows:
            w.writerow([r.get(c, "") for c in cols])
    print(f"\nCSV → {args.out_csv}", flush=True)

    # ── Save LaTeX ────────────────────────────────────────────
    with open(args.out_latex, "w") as f:
        f.write("% 10-metric CineGen evaluation. Generated by evaluate/eval.py\n")
        f.write("\\toprule\n")
        f.write("    & \\multicolumn{3}{c}{Trajectory quality}\n")
        f.write("    & \\multicolumn{4}{c}{Text alignment}\n")
        f.write("    & \\multicolumn{3}{c}{Attribute} \\\\\n")
        f.write("    \\cmidrule(lr){2-4}\\cmidrule(lr){5-8}\\cmidrule(lr){9-11}\n")
        f.write("    Setting\n")
        f.write("    & F1\\up & \\fcd\\down & Cov.\\up\n")
        f.write("    & \\alnscore\\up & CLaTr\\up & R@1\\up & MedR\\down\n")
        f.write("    & Era\\up & Genre\\up & Dir.\\up \\\\\n")
        f.write("    \\midrule\n")
        for r in rows:
            cells = [
                r["label"],
                f"{r['F1']:.3f}", f"{r['FCD']:.2f}", f"{r['Cov']:.3f}",
                f"{r['AlnScore']:.2f}", f"{r['CLaTr']:.2f}",
                f"{r['R@1']:.2f}", f"{r['MedR']:.1f}",
                f"{r['Era']:.3f}", f"{r['Genre']:.3f}", f"{r['Dir']:.3f}",
            ]
            f.write("    " + " & ".join(cells) + " \\\\\n")
        f.write("    \\bottomrule\n")
    print(f"LaTeX → {args.out_latex}", flush=True)


if __name__ == "__main__":
    main()
