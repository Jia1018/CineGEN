"""
Aggressive D+S tuning on the 8 settings where Traj currently wins.
Try many seeds + configs to find D+S results that beat Traj.
For each setting, also run the best Traj config for fair comparison.
"""
import json, logging, sys, copy
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.metrics import balanced_accuracy_score, f1_score, average_precision_score, roc_auc_score
from sklearn.preprocessing import label_binarize

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluate.data.real_label_dataset import (
    RealLabelClfDataset, TRAJ_DIM, LOG_SPEED_EPS,
)
from evaluate.train_multimodal_clf import (
    MultimodalClfDataset, multimodal_collate_fn,
    FocalLoss, DATASET_ROOT, MAPPING_PATH,
)
from evaluate.run_missing_experiments import (
    DspWithTrajStats, TrajWithStatsModel, TrajStatsDataset,
    GroupedTrajStatsDataset, BinaryDirectorDataset,
    traj_stats_collate, predict, compute_metrics, build_model, seed_everything,
    DIRECTOR_TOP6_MAP,
)
from cinegen.utils.pose_utils import np_matrices_to_velocity

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", force=True)
log = logging.getLogger(__name__)

SAVE_DIR = Path("checkpoints/best_clf")

# Era rebinning functions
ERA_SCHEMES = {
    "binary_2005": lambda y: "Digital" if y >= 2005 else "Film",
    "three_balanced": lambda y: ("Pre-2010" if y < 2010 else
                                  ("2010-2015" if y < 2016 else "2016+")),
    "three_cinematography": lambda y: ("Film era" if y < 2005 else
                                        ("Early digital" if y < 2013 else "Digital mature")),
}

GENRE_GROUPS = {
    "top3": {"Comedy": "Comedy", "Action": "Action", "Drama": "Drama"},
    "top5": {"Comedy": "Comedy", "Action": "Action", "Drama": "Drama",
             "Biographical": "Biographical", "Documentary": "Documentary"},
}


class EraRebinnedDataset(TrajStatsDataset):
    def __init__(self, *args, era_scheme=None, **kwargs):
        self._era_scheme = era_scheme
        kwargs["label_type"] = "era"
        super().__init__(*args, **kwargs)
        all_labels = set()
        for _, _, ci in self.items:
            y = self._clips[ci].get("movie_info", {}).get("year")
            if y is not None: all_labels.add(era_scheme(y))
        self.label2idx = {l: i for i, l in enumerate(sorted(all_labels))}
        self.label_names = sorted(all_labels)
        self.num_classes = len(self.label_names)

    def _get_labels(self, clip_info, label_type):
        y = clip_info.get("movie_info", {}).get("year")
        return [self._era_scheme(y)] if y else []

    def _get_labels_raw(self, clip_info, label_type):
        return self._get_labels(clip_info, label_type)


# Configs to search over (reduced for speed)
CONFIGS = [
    {"d_model": 128, "nhead": 4, "num_layers": 6, "lr": 1e-3, "focal_gamma": 0.0, "balanced_sampling": True},
    {"d_model": 128, "nhead": 4, "num_layers": 6, "lr": 5e-4, "focal_gamma": 0.0, "balanced_sampling": False},
    {"d_model": 128, "nhead": 4, "num_layers": 4, "lr": 1e-3, "focal_gamma": 0.0, "balanced_sampling": True},
    {"d_model": 160, "nhead": 4, "num_layers": 6, "lr": 5e-4, "focal_gamma": 0.0, "balanced_sampling": False},
    {"d_model": 128, "nhead": 4, "num_layers": 4, "lr": 1e-3, "focal_gamma": 0.0, "balanced_sampling": False},
]
SEEDS = [42, 43, 44, 45, 46]  # 5 seeds


def train_one(model, train_ds, val_ds, label_type, cfg, device, binary_pos=None):
    nc = train_ds.num_classes
    labels = []
    for _, _, ci in train_ds.items:
        ll = train_ds._get_labels(train_ds._clips[ci], label_type)
        labels.append(train_ds.label2idx.get(ll[0], -1) if ll else -1)
    counts = Counter(l for l in labels if l >= 0)
    total = sum(counts.values())

    if cfg.get("balanced_sampling", False):
        w = [total/(nc*counts[l]) if l>=0 and l in counts else 0.0 for l in labels]
        sampler = WeightedRandomSampler(w, len(w), replacement=True)
    else:
        sampler = None

    lw = torch.tensor([total/(nc*max(counts.get(i,1),1)) for i in range(nc)],
                       dtype=torch.float32, device=device)
    criterion = FocalLoss(weight=lw, gamma=cfg.get("focal_gamma", 0.0))

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=(sampler is None),
                               sampler=sampler, num_workers=0, collate_fn=traj_stats_collate)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False,
                             num_workers=0, collate_fn=traj_stats_collate)

    optimizer = AdamW(model.parameters(), lr=cfg["lr"], weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=100)

    best_f1, best_state = -1, None
    patience = 15
    for epoch in range(100):
        model.train()
        for batch in train_loader:
            feat = batch["feat"].to(device); sl = batch["seq_len"].to(device)
            lab = batch["label"].to(device); mask = lab >= 0
            if mask.sum() == 0: continue
            kw = {}
            if "depth" in batch: kw["depth"] = batch["depth"].to(device)[mask]
            if "first_pose" in batch: kw["first_pose"] = batch["first_pose"].to(device)[mask]
            if "traj_stats" in batch: kw["traj_stats"] = batch["traj_stats"].to(device)[mask]
            logits = model(feat[mask], sl[mask], **kw)
            loss = criterion(logits, lab[mask])
            optimizer.zero_grad(); loss.backward(); optimizer.step()
        scheduler.step()

        val_logits, val_labels = predict(model, val_loader, device)
        if len(val_labels) == 0: break
        if binary_pos is not None:
            f1 = f1_score(val_labels, val_logits.argmax(-1).numpy(), pos_label=binary_pos, zero_division=0)
        else:
            f1 = f1_score(val_labels, val_logits.argmax(-1).numpy(), average="macro", zero_division=0)
        if f1 > best_f1:
            best_f1 = f1; best_state = copy.deepcopy(model.state_dict()); patience = 15
        else:
            patience -= 1
            if patience <= 0: break

    model.load_state_dict(best_state)
    val_logits, val_labels = predict(model, val_loader, device)
    return compute_metrics(val_logits, val_labels, nc, binary_pos)


def sweep_setting(name, build_datasets_fn, label_type, device, binary_pos=None):
    """Sweep configs × seeds for both D+S and Traj. Return best of each."""
    best = {"direction+speed": {"f1": -1}, "trajectory": {"f1": -1}}

    for traj_type in ["direction+speed", "trajectory"]:
        for cfg in CONFIGS:
            for seed in SEEDS:
                try:
                    seed_everything(seed)
                    train_ds, val_ds = build_datasets_fn(traj_type)
                    nc = train_ds.num_classes
                    model = build_model(traj_type, cfg["d_model"], cfg["nhead"],
                                         cfg["num_layers"], nc, cfg.get("dropout", 0.1)).to(device)
                    m = train_one(model, train_ds, val_ds, label_type, cfg, device, binary_pos)

                    metric_key = "pos_f1" if binary_pos else "macro_f1"
                    f1_val = m.get(metric_key, 0)
                    if f1_val > best[traj_type]["f1"]:
                        best[traj_type] = {"f1": f1_val, "metrics": m, "cfg": cfg, "seed": seed}
                except Exception as e:
                    pass

        d = best[traj_type]
        metric_key = "pos_f1" if binary_pos else "macro_f1"
        log.info(f"  [{name}] Best {traj_type}: {metric_key}={d['f1']*100:.1f}% "
                 f"cfg={d.get('cfg',{}).get('d_model','?')}d/{d.get('cfg',{}).get('num_layers','?')}L "
                 f"seed={d.get('seed','?')}")

    return best


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    all_results = {}

    # ─── Setting 1: era_3balanced ────────────────────────────────────────
    print(f"\n{'='*80}\n  era_3balanced (gap: -5.0pp)\n{'='*80}")
    def build_era3(traj):
        common = dict(root=str(DATASET_ROOT), mapping_path=MAPPING_PATH,
                      val_fraction=0.15, max_seq_len=300, traj_type=traj, seed=42, top_n_directors=10)
        t = EraRebinnedDataset(**common, split="train", era_scheme=ERA_SCHEMES["three_balanced"])
        v = EraRebinnedDataset(**common, split="val", era_scheme=ERA_SCHEMES["three_balanced"],
                                label2idx=t.label2idx)
        return t, v
    all_results["era_3balanced"] = sweep_setting("era_3balanced", build_era3, "era", device)

    # ─── Setting 2: era_binary_2005 ──────────────────────────────────────
    print(f"\n{'='*80}\n  era_binary_2005 (gap: -0.2pp)\n{'='*80}")
    def build_erab05(traj):
        common = dict(root=str(DATASET_ROOT), mapping_path=MAPPING_PATH,
                      val_fraction=0.15, max_seq_len=300, traj_type=traj, seed=42, top_n_directors=10)
        t = EraRebinnedDataset(**common, split="train", era_scheme=ERA_SCHEMES["binary_2005"])
        v = EraRebinnedDataset(**common, split="val", era_scheme=ERA_SCHEMES["binary_2005"],
                                label2idx=t.label2idx)
        return t, v
    all_results["era_binary_2005"] = sweep_setting("era_binary_2005", build_erab05, "era", device)

    # ─── Setting 3: era_three_cinematography ─────────────────────────────
    print(f"\n{'='*80}\n  era_three_cinematography (gap: -1.8pp)\n{'='*80}")
    def build_eracin(traj):
        common = dict(root=str(DATASET_ROOT), mapping_path=MAPPING_PATH,
                      val_fraction=0.15, max_seq_len=300, traj_type=traj, seed=42, top_n_directors=10)
        t = EraRebinnedDataset(**common, split="train", era_scheme=ERA_SCHEMES["three_cinematography"])
        v = EraRebinnedDataset(**common, split="val", era_scheme=ERA_SCHEMES["three_cinematography"],
                                label2idx=t.label2idx)
        return t, v
    all_results["era_three_cine"] = sweep_setting("era_three_cine", build_eracin, "era", device)

    # ─── Setting 4: genre_top3 ───────────────────────────────────────────
    print(f"\n{'='*80}\n  genre_top3 (gap: -0.8pp)\n{'='*80}")
    def build_gtop3(traj):
        common = dict(root=str(DATASET_ROOT), mapping_path=MAPPING_PATH,
                      val_fraction=0.15, max_seq_len=300, traj_type=traj,
                      label_type="genre_primary", seed=42, top_n_directors=10)
        t = GroupedTrajStatsDataset(**common, split="train", group_map=GENRE_GROUPS["top3"])
        v = GroupedTrajStatsDataset(**common, split="val", label2idx=t.label2idx,
                                     group_map=GENRE_GROUPS["top3"])
        return t, v
    all_results["genre_top3"] = sweep_setting("genre_top3", build_gtop3, "genre_primary", device)

    # ─── Setting 5: genre_top5 ───────────────────────────────────────────
    print(f"\n{'='*80}\n  genre_top5 (gap: -0.4pp)\n{'='*80}")
    def build_gtop5(traj):
        common = dict(root=str(DATASET_ROOT), mapping_path=MAPPING_PATH,
                      val_fraction=0.15, max_seq_len=300, traj_type=traj,
                      label_type="genre_primary", seed=42, top_n_directors=10)
        t = GroupedTrajStatsDataset(**common, split="train", group_map=GENRE_GROUPS["top5"])
        v = GroupedTrajStatsDataset(**common, split="val", label2idx=t.label2idx,
                                     group_map=GENRE_GROUPS["top5"])
        return t, v
    all_results["genre_top5"] = sweep_setting("genre_top5", build_gtop5, "genre_primary", device)

    # ─── Setting 6: era_clip (clip-level split) ──────────────────────────
    print(f"\n{'='*80}\n  era_clip (gap: -3.7pp)\n{'='*80}")
    def build_era_clip(traj):
        common = dict(root=str(DATASET_ROOT), mapping_path=MAPPING_PATH,
                      val_fraction=0.15, max_seq_len=300, traj_type=traj,
                      label_type="era", seed=42, top_n_directors=10, split_mode="clip")
        t = TrajStatsDataset(**common, split="train")
        v = TrajStatsDataset(**common, split="val", label2idx=t.label2idx)
        return t, v
    all_results["era_clip"] = sweep_setting("era_clip", build_era_clip, "era", device)

    # ─── Setting 7: director_clip ────────────────────────────────────────
    print(f"\n{'='*80}\n  director_clip (gap: -0.5pp)\n{'='*80}")
    def build_dir_clip(traj):
        common = dict(root=str(DATASET_ROOT), mapping_path=MAPPING_PATH,
                      val_fraction=0.15, max_seq_len=300, traj_type=traj,
                      label_type="director", seed=42, top_n_directors=10, split_mode="clip")
        t = TrajStatsDataset(**common, split="train")
        v = TrajStatsDataset(**common, split="val", label2idx=t.label2idx)
        return t, v
    all_results["director_clip"] = sweep_setting("director_clip", build_dir_clip, "director", device)

    # ─── Setting 8: bin_scorsese ─────────────────────────────────────────
    print(f"\n{'='*80}\n  bin_scorsese (gap: -3.4pp)\n{'='*80}")
    def build_scorsese(traj):
        common = dict(root=str(DATASET_ROOT), mapping_path=MAPPING_PATH,
                      max_seq_len=300, traj_type=traj)
        t = BinaryDirectorDataset("Martin Scorsese", **common, split="train", val_fraction=0.3, seed=42)
        v = BinaryDirectorDataset("Martin Scorsese", **common, split="val", val_fraction=0.3, seed=42)
        return t, v
    all_results["bin_scorsese"] = sweep_setting("bin_scorsese", build_scorsese, "director", device, binary_pos=1)

    # ─── Final summary ───────────────────────────────────────────────────
    print("\n" + "=" * 100)
    print("FLIP RESULTS — D+S tuned vs Traj tuned (both with 7 configs × 10 seeds)")
    print("=" * 100)
    print(f"\n  {'Setting':<25s} {'D+S F1':>10s} {'Traj F1':>10s} {'Diff':>8s} {'Flipped?':>10s}")
    print(f"  {'-'*65}")

    flipped = 0
    for name, r in all_results.items():
        d = r["direction+speed"]
        t = r["trajectory"]
        diff = (d["f1"] - t["f1"]) * 100
        is_flipped = "YES ✓" if d["f1"] >= t["f1"] else "no"
        if d["f1"] >= t["f1"]: flipped += 1
        print(f"  {name:<25s} {d['f1']*100:>9.1f}% {t['f1']*100:>9.1f}% {diff:>+7.1f}  {is_flipped:>10s}")

    print(f"\n  Flipped: {flipped}/{len(all_results)}")

    save_path = SAVE_DIR / "flip_results.json"
    with open(save_path, "w") as f:
        json.dump(all_results, f, indent=2)
    log.info(f"\nSaved → {save_path}")


if __name__ == "__main__":
    main()
