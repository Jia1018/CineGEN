"""
1x4 joint-space t-SNE comparison: CLaTr vs CineAlign × {dir+spd, trajectory}.

Produces a single figure with 4 panels in one row:
  [CLaTr / dir+spd]   [CineAlign / dir+spd]   [CLaTr / trajectory]   [CineAlign / trajectory]

Each panel shows trajectory (●) and text (★) embeddings projected together
with t-SNE, coloured by K-means clusters of the trajectory embedding space.
Same colour ●★ pairs that stay close = good alignment.

Usage:
    python evaluate/visualize_align_compare.py \
        --out figs/align_compare/joint_motion_4panel.png
"""
import argparse
import logging
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluate.data.dataset import AlignDataset, collate_fn
from evaluate.data.clatr_dataset import CLaTrDataset, collate_fn as clatr_collate
from evaluate.models.align_model import build_align_model
from evaluate.models.clatr import CLaTr


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_align_model(ckpt_path: str, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg  = ckpt["cfg"]
    model = build_align_model(
        traj_type         = cfg["traj_type"],
        embed_dim         = cfg["embed_dim"],
        traj_d_model      = cfg["traj_d_model"],
        traj_nhead        = cfg["traj_nhead"],
        traj_num_layers   = cfg["traj_num_layers"],
        max_vel_len       = cfg["max_seq_len"] if cfg["traj_type"] == "trajectory"
                            else cfg["max_seq_len"] - 1,
        dropout           = cfg.get("dropout", 0.1),
        clip_model_id     = cfg.get("clip_model_id", "openai/clip-vit-large-patch14"),
        freeze_clip       = cfg.get("freeze_clip", True),
        init_temperature  = cfg.get("init_temperature", 0.07),
        learn_temperature = cfg.get("learn_temperature", True),
        pooling           = cfg.get("pooling", "mean"),
        pos_enc           = cfg.get("pos_enc", "learned"),
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, cfg


def load_clatr_model(ckpt_path: str, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg  = ckpt["cfg"]
    # Determine input dims (clip_dim from a probe; traj_input_dim from traj_type)
    TRAJ_DIM = {"trajectory": 9, "velocity": 6, "direction": 6,
                "speed": 2, "direction+speed": 8}
    traj_input_dim = TRAJ_DIM[cfg["traj_type"]]
    text_input_dim = 768   # ViT-L/14 default

    max_seq_len = cfg["max_seq_len"] if cfg["traj_type"] == "trajectory" \
                  else cfg["max_seq_len"] - 1

    model = CLaTr(
        traj_input_dim    = traj_input_dim,
        text_input_dim    = text_input_dim,
        latent_dim        = cfg["latent_dim"],
        ff_size           = cfg["ff_size"],
        num_layers        = cfg["num_layers"],
        num_heads         = cfg["num_heads"],
        dropout           = cfg["dropout"],
        max_seq_len       = max_seq_len,
        temperature       = cfg["temperature"],
        threshold_selfsim = cfg["threshold_selfsim"],
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, cfg


# ---------------------------------------------------------------------------
# Embedding collection
# ---------------------------------------------------------------------------

@torch.no_grad()
def collect_align(model, cfg, device, max_samples=None):
    ds = AlignDataset(
        root         = cfg["root"],
        datasets     = cfg["datasets"],
        split        = "val",
        val_fraction = cfg.get("val_fraction", 0.1),
        max_seq_len  = cfg["max_seq_len"],
        traj_type    = cfg["traj_type"],
        text_type    = cfg["text_type"],
        seed         = cfg.get("seed", 42),
    )
    loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=4,
                        collate_fn=collate_fn)
    all_traj, all_text = [], []
    for batch in loader:
        traj     = batch["traj_feat"].to(device)
        seq_lens = batch["seq_len"].to(device)
        texts    = batch["text"]
        all_traj.append(model.encode_traj(traj, seq_lens).cpu())
        all_text.append(model.encode_text(texts).cpu())
        if max_samples and sum(t.shape[0] for t in all_traj) >= max_samples:
            break
    traj = torch.cat(all_traj).numpy()
    text = torch.cat(all_text).numpy()
    if max_samples:
        traj = traj[:max_samples]
        text = text[:max_samples]
    return traj, text


@torch.no_grad()
def collect_clatr(model, cfg, device, max_samples=None, ckpt_dir=None):
    # Try to load standardization from sidecar JSON (saved by train_clatr.py)
    std = cfg.get("standardization")
    if std is None and ckpt_dir is not None:
        std_path = Path(ckpt_dir) / "standardization.json"
        if std_path.exists():
            import json as _json
            with open(std_path) as f:
                std = _json.load(f)

    ds = CLaTrDataset(
        root            = cfg["root"],
        datasets        = cfg["datasets"],
        split           = "val",
        val_fraction    = cfg.get("val_fraction", 0.1),
        max_seq_len     = cfg["max_seq_len"],
        traj_type       = cfg["traj_type"],
        text_type       = cfg["text_type"],
        standardization = std,
        clip_cache_dir  = cfg.get("clip_cache", "./clip_cache/clatr"),
        seed            = cfg.get("seed", 42),
    )
    loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=4,
                        collate_fn=clatr_collate)
    all_traj, all_text = [], []
    for batch in loader:
        traj  = batch["traj_feat"].to(device)
        mask  = batch["padding_mask"].to(device)
        cap   = batch["caption_feat"].to(device)
        all_traj.append(model.get_traj_embedding(traj, mask).cpu())
        all_text.append(model.get_text_embedding(cap).cpu())
        if max_samples and sum(t.shape[0] for t in all_traj) >= max_samples:
            break
    traj = torch.cat(all_traj).numpy()
    text = torch.cat(all_text).numpy()
    if max_samples:
        traj = traj[:max_samples]
        text = text[:max_samples]
    return traj, text


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def project_joint(traj_emb, text_emb, perplexity=30, seed=42):
    """Project traj+text together with t-SNE."""
    joint = np.concatenate([traj_emb, text_emb], axis=0)
    joint_2d = TSNE(n_components=2, perplexity=perplexity,
                    random_state=seed, init="pca",
                    learning_rate="auto").fit_transform(joint)
    N = traj_emb.shape[0]
    return joint_2d[:N], joint_2d[N:]


def filter_outliers(joint_traj, joint_text, labels, percentile=98.0,
                    mode="knn", labels_text=None, knn_k=8):
    """
    Drop visually isolated points — those near the edge of the cloud with
    no close neighbours.

    mode="knn"      — distance to k-th nearest neighbour in joint 2D space.
                       Isolated edge points have a large k-NN distance.
                       Drop the top (100 - percentile)% by k-NN distance.
                       This is what you usually want for cleaning up t-SNE
                       projection stragglers.
    mode="global"   — distance from joint median (catches only the very
                       farthest single points; misses isolated edge sub-groups).
    mode="cluster"  — distance from per-cluster median; catches whole drifted
                       sub-groups of the same K-means cluster.

    Returns: (keep_traj, keep_text) boolean masks of shape (N,).
    """
    if mode == "global":
        all_pts = np.concatenate([joint_traj, joint_text], axis=0)
        center = np.median(all_pts, axis=0)
        dist = np.linalg.norm(all_pts - center, axis=1)
        cutoff = np.percentile(dist, percentile)
        n_traj = joint_traj.shape[0]
        return dist[:n_traj] <= cutoff, dist[n_traj:] <= cutoff

    if mode == "knn":
        # k-NN distance in joint 2D space — captures "isolated, near edge"
        from scipy.spatial import cKDTree
        all_pts = np.concatenate([joint_traj, joint_text], axis=0)
        tree = cKDTree(all_pts)
        # k+1 because the closest point is itself (dist 0)
        k_eff = min(knn_k, len(all_pts) - 1)
        dists, _ = tree.query(all_pts, k=k_eff + 1)
        knn_dist = dists[:, -1]   # distance to the k-th neighbour
        cutoff = np.percentile(knn_dist, percentile)
        n_traj = joint_traj.shape[0]
        return knn_dist[:n_traj] <= cutoff, knn_dist[n_traj:] <= cutoff

    # ----- per-cluster mode -----
    if labels_text is None:
        labels_text = labels
    keep_traj = np.ones(joint_traj.shape[0], dtype=bool)
    keep_text = np.ones(joint_text.shape[0], dtype=bool)
    for c in np.unique(labels):
        m_t = labels == c
        m_x = labels_text == c
        if not m_t.any() and not m_x.any():
            continue
        cluster_pts = np.concatenate(
            [joint_traj[m_t], joint_text[m_x]], axis=0
        )
        if len(cluster_pts) < 4:
            continue
        center = np.median(cluster_pts, axis=0)
        traj_d = np.linalg.norm(joint_traj[m_t] - center, axis=1)
        text_d = np.linalg.norm(joint_text[m_x] - center, axis=1)
        cutoff = np.percentile(np.concatenate([traj_d, text_d]), percentile)
        traj_idx = np.where(m_t)[0]
        text_idx = np.where(m_x)[0]
        keep_traj[traj_idx[traj_d > cutoff]] = False
        keep_text[text_idx[text_d > cutoff]] = False
    return keep_traj, keep_text


def kmeans_labels(traj_emb, k=8, seed=42):
    km = KMeans(n_clusters=k, random_state=seed, n_init=10)
    return km.fit_predict(traj_emb)


def plot_panel(ax, joint_traj, joint_text, labels_traj, labels_text, k,
               title, subtitle=None, n_pair_lines=200, pair_idx=None,
               keep_traj=None, keep_text=None):
    """Replicate the joint panel from evaluate/visualize_logline.py:_joint_panel.

    Adds 200 thin connecting lines between matched (traj_i, text_i) pairs
    coloured by trajectory cluster. Outlier filtering is applied AFTER pair
    selection: a pair is drawn only if both endpoints survive filtering.
    """
    cmap = plt.get_cmap("tab10" if k <= 10 else "tab20")
    all_labels = np.unique(np.concatenate([labels_traj, labels_text]))
    palette = {lab: cmap(int(lab) % cmap.N) for lab in all_labels}

    # ---- Pair lines (zorder 1, drawn first) ----
    # `pair_idx` indexes into the FULL (pre-filter) arrays, with corresponding
    # `keep_traj` / `keep_text` boolean masks.  We draw only the pairs whose
    # both endpoints survive outlier filtering.
    if pair_idx is not None and keep_traj is not None and keep_text is not None:
        for i in pair_idx:
            if not (keep_traj[i] and keep_text[i]):
                continue
            # We need the FILTERED 2D coords; lookup by re-indexing.
            ti = int(np.searchsorted(np.where(keep_traj)[0], i))
            xi = int(np.searchsorted(np.where(keep_text)[0], i))
            ax.plot(
                [joint_traj[ti, 0], joint_text[xi, 0]],
                [joint_traj[ti, 1], joint_text[xi, 1]],
                color=palette[int(labels_traj[ti])],
                alpha=0.40, linewidth=1.4, zorder=1,
            )

    # ---- Trajectory scatter (zorder 2) ----  pre-shrink size
    traj_colors = [palette[int(l)] for l in labels_traj]
    ax.scatter(
        joint_traj[:, 0], joint_traj[:, 1],
        c=traj_colors,
        s=45, marker="o", alpha=0.75,
        linewidths=0.3, edgecolors="white",
        zorder=2,
    )
    # ---- Text scatter (zorder 3) ----  pre-shrink size
    text_colors = [palette[int(l)] for l in labels_text]
    ax.scatter(
        joint_text[:, 0], joint_text[:, 1],
        c=text_colors,
        s=120, marker="*", alpha=0.85,
        linewidths=0.3, edgecolors="white",
        zorder=3,
    )

    # ---- Inline legend at top-right of every panel (●/★ key) ----
    circle_h = mlines.Line2D([], [], color="grey", marker="o", markersize=9,
                              linestyle="None", label="trajectory")
    star_h   = mlines.Line2D([], [], color="grey", marker="*", markersize=13,
                              linestyle="None", label="text")
    ax.legend(handles=[circle_h, star_h], loc="upper right",
              fontsize=9, frameon=True, framealpha=0.85,
              handletextpad=0.4, borderpad=0.3, labelspacing=0.25)

    ax.set_title(title, fontsize=12, fontweight="bold")
    if subtitle:
        ax.text(0.5, -0.04, subtitle, transform=ax.transAxes,
                ha="center", va="top", fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])
    # Tight padding inside the axes (less white space around point cloud)
    ax.margins(0.02)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

CKPTS = {
    # (col_idx, model, repr): (label, ckpt_path, loader_kind)
    "clatr_dirspd":  ("CLaTr — dir+spd",         "checkpoints/clatr/direction_speed_motion/best.pt",   "clatr"),
    "ours_dirspd":   ("CineAlign (ours) — dir+spd", "checkpoints/align_v2/direction+speed_motion/best.pt",   "align"),
    "clatr_traj":    ("CLaTr — trajectory",      "checkpoints/clatr/trajectory_motion/best.pt",        "clatr"),
    "ours_traj":     ("CineAlign (ours) — trajectory", "checkpoints/align_v2/trajectory_motion/best.pt",       "align"),
}

# Stats from the paper table
STATS = {
    "clatr_dirspd": "R@1=19.7  R@5=30.5  MedR=23",
    "ours_dirspd":  "R@1=25.8  R@5=40.7  MedR=11",
    "clatr_traj":   "R@1=6.9   R@5=12.0  MedR=361",
    "ours_traj":    "R@1=17.8  R@5=26.5  MedR=51",
}

PANEL_ORDER = ["clatr_dirspd", "ours_dirspd", "clatr_traj", "ours_traj"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="figs/align_compare/joint_motion_4panel.png")
    p.add_argument("--k", type=int, default=8, help="KMeans cluster count for coloring")
    p.add_argument("--max_samples", type=int, default=2000,
                   help="Cap val samples per panel for speed")
    p.add_argument("--perplexity", type=float, default=30.0)
    p.add_argument("--outlier_pct", type=float, default=98.0,
                   help="Drop points above this percentile of distance "
                        "from the reference center. Set to 100 to disable.")
    p.add_argument("--outlier_mode", choices=["global", "cluster", "knn"],
                   default="knn",
                   help="'knn' = isolated edge points (large k-NN distance); "
                        "'global' = farthest from joint median; "
                        "'cluster' = farthest within each K-means cluster.")
    p.add_argument("--knn_k", type=int, default=8,
                   help="k for k-NN outlier filter (mode=knn).")
    p.add_argument("--n_pair_lines", type=int, default=200,
                   help="Number of (traj, text) connecting lines per panel. 0 to disable.")
    p.add_argument("--pair_seed", type=int, default=42,
                   help="RNG seed used to sample which matched pairs get a "
                        "connecting line. Change to reroll the line subset.")
    p.add_argument("--per_panel_dir", default=None,
                   help="If set, also save 4 separate per-panel PDFs (no title, "
                        "no subtitle, no legend) plus a standalone legend strip "
                        "into this directory, for LaTeX subfigure inclusion.")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Collect all 4 panels' embeddings
    panels = {}
    for key in PANEL_ORDER:
        title, ckpt, kind = CKPTS[key]
        log.info(f"=== {title} ({ckpt}) ===")
        if kind == "align":
            model, cfg = load_align_model(ckpt, device)
            traj, text = collect_align(model, cfg, device, args.max_samples)
        else:
            model, cfg = load_clatr_model(ckpt, device)
            traj, text = collect_clatr(model, cfg, device, args.max_samples,
                                        ckpt_dir=Path(ckpt).parent)
        log.info(f"  traj={traj.shape}, text={text.shape}, |traj|={np.linalg.norm(traj, axis=1).mean():.3f}")
        panels[key] = (title, traj, text)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Project & cluster
    log.info("Computing t-SNE, then 2D outlier filter, then K-means on survivors...")
    panel_data = {}
    rng = np.random.default_rng(args.pair_seed)
    for key in PANEL_ORDER:
        title, traj, text = panels[key]
        N = traj.shape[0]

        # 1) t-SNE on the full set (joint traj+text projection).  We project
        #    EVERYTHING first so outlier detection happens in the same 2D
        #    space the viewer actually sees — points that look isolated in
        #    the figure are exactly the ones flagged.
        joint_traj, joint_text = project_joint(traj, text, args.perplexity)

        # 2) Outlier filter on the 2D positions.
        dummy_labels = np.zeros(N, dtype=np.int64)
        keep_t, keep_x = filter_outliers(
            joint_traj, joint_text, dummy_labels,
            percentile=args.outlier_pct,
            mode=args.outlier_mode,
            labels_text=dummy_labels,
            knn_k=args.knn_k,
        )
        # 3) Couple paired drops: ★ dropped ⇒ paired ● dropped (and vice-versa).
        coupled = keep_t & keep_x
        n_dropped_pairs = (~coupled).sum()
        n_dropped_t_only = (~keep_t & keep_x).sum()
        n_dropped_x_only = (keep_t & ~keep_x).sum()
        n_dropped_both   = (~keep_t & ~keep_x).sum()
        log.info(f"  [{title}] dropped {n_dropped_pairs} pairs at "
                 f"{args.outlier_pct}th pct ({args.outlier_mode}, k={args.knn_k}) "
                 f"in 2D t-SNE space "
                 f"[traj-only: {n_dropped_t_only}, text-only: {n_dropped_x_only}, "
                 f"both: {n_dropped_both}]")

        # 4) K-means on SURVIVING raw 256D traj embeddings — guarantees k
        #    non-empty, balanced clusters over the visible points.
        traj_kept = traj[coupled]
        N_kept = len(traj_kept)
        if N_kept >= args.k:
            labels_kept = kmeans_labels(traj_kept, k=args.k)
        else:
            labels_kept = np.zeros(N_kept, dtype=np.int64)

        # 5) Pair indices in the SURVIVING set for connecting lines
        pair_idx = rng.choice(N_kept, size=min(args.n_pair_lines, N_kept),
                              replace=False)

        # All-True masks for the (already cleaned) arrays — plot_panel still
        # uses keep masks to draw pair lines (so outliers don't get lines).
        all_keep = np.ones(N_kept, dtype=bool)
        panel_data[key] = (
            title,
            joint_traj[coupled], joint_text[coupled],
            labels_kept, labels_kept,   # text label = paired traj's label
            pair_idx, all_keep, all_keep,
        )

    # ── Combined 1×4 (PNG/PDF) for quick inspection ──
    fig, axes = plt.subplots(1, 4, figsize=(22, 6.0))
    for ax, key in zip(axes, PANEL_ORDER):
        title, jt, jx, lbl_t, lbl_x, pidx, kt, kx = panel_data[key]
        plot_panel(ax, jt, jx, lbl_t, lbl_x, args.k, title,
                   subtitle=STATS[key],
                   n_pair_lines=args.n_pair_lines,
                   pair_idx=pidx, keep_traj=kt, keep_text=kx)

    circle_h = mlines.Line2D([], [], color="grey", marker="o", markersize=8,
                              linestyle="None", label="trajectory (●)")
    star_h   = mlines.Line2D([], [], color="grey", marker="*", markersize=14,
                              linestyle="None", label="text (★)")
    fig.legend(handles=[circle_h, star_h], loc="lower center",
               bbox_to_anchor=(0.5, -0.02), ncol=2, fontsize=11,
               frameon=True)
    plt.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=180, bbox_inches="tight")
    log.info(f"Saved → {out}")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    log.info(f"Saved → {out.with_suffix('.pdf')}")
    plt.close(fig)

    # ── Per-panel PDFs (no title / no subtitle; inline ●/★ legend) ──
    # These are designed for LaTeX subfigure layout where the (a)–(d)
    # caption is rendered in TeX. White-space around the panel is kept
    # minimal via pad_inches; the inline legend lives inside the axes.
    if args.per_panel_dir:
        panel_dir = Path(args.per_panel_dir)
        panel_dir.mkdir(parents=True, exist_ok=True)
        for key in PANEL_ORDER:
            _, jt, jx, lbl_t, lbl_x, pidx, kt, kx = panel_data[key]
            f, ax = plt.subplots(figsize=(5.0, 5.0))
            plot_panel(ax, jt, jx, lbl_t, lbl_x, args.k,
                       title="",       # blank — TeX adds it
                       subtitle=None,  # no R@k line — table has it
                       n_pair_lines=args.n_pair_lines,
                       pair_idx=pidx, keep_traj=kt, keep_text=kx)
            ax.set_title("")  # clear default title
            f.tight_layout(pad=0.0)
            for ext in ("pdf", "png"):
                fp = panel_dir / f"panel_{key}.{ext}"
                f.savefig(fp, bbox_inches="tight", pad_inches=0.02,
                          dpi=180 if ext == "png" else None)
            log.info(f"Saved per-panel → {panel_dir}/panel_{key}.pdf")
            plt.close(f)

        # Standalone legend strip (●/★ key) for placing above the row
        leg_fig, leg_ax = plt.subplots(figsize=(5.0, 0.4))
        leg_ax.axis("off")
        leg_ax.legend(handles=[circle_h, star_h], loc="center", ncol=2,
                      fontsize=11, frameon=False)
        leg_fig.savefig(panel_dir / "legend.pdf", bbox_inches="tight")
        leg_fig.savefig(panel_dir / "legend.png", bbox_inches="tight", dpi=180)
        log.info(f"Saved legend → {panel_dir}/legend.pdf")
        plt.close(leg_fig)


if __name__ == "__main__":
    main()
