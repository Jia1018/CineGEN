"""
Retrain the 9 paper-table classifiers (with checkpoint saving):
  1. Macro type (3 cls)            — use VLM aspect labels
  2. Era (3 cls)                   — era_three_cinematography
  3. Country (6 regions)           — country_region
  4. Genre top-3
  5. Genre top-5
  6. Director top-3                — Spielberg/Zemeckis/Other
  7. Director-Tarantino binary
  8. Director-Wes Anderson binary
  9. Director-Nolan binary

Each is trained for both direction+speed and trajectory representations.
Uses the BEST cfg from prior sweeps (no full re-sweep).
Saves checkpoints to checkpoints/clf_paper/{setting}/{traj_type}_best.pt
"""

import sys
import json
import logging
import copy
import argparse
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.metrics import f1_score, accuracy_score, balanced_accuracy_score, roc_auc_score

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from evaluate.run_missing_experiments import (
    DspWithTrajStats, TrajWithStatsModel, TrajStatsDataset,
    BinaryDirectorDataset, GroupedTrajStatsDataset, traj_stats_collate,
    seed_everything, build_model, predict, FocalLoss, compute_metrics
)
from evaluate.flip_traj_wins import EraRebinnedDataset, ERA_SCHEMES, GENRE_GROUPS
from evaluate.train_multimodal_clf import VIPE_ROOT, AD_KEYWORDS, DEPTH_SIZE
from evaluate.data.real_label_dataset import TRAJ_DIM

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

DATASET_ROOT = Path("/workspace/writeable/datasets/DIY_movies")
MAPPING_PATH = DATASET_ROOT / "labeling/known_movies/clip_movie_mapping.json"
SAVE_DIR = REPO / "checkpoints/clf_paper"

# Best configs per (setting, traj_type) extracted from flip_results.json + grouping_results.json
# Format: (d_model, num_layers, lr, focal_gamma, balanced_sampling, seed)
BEST_CFGS = {
    # (uses default 128-d, 6L, 1e-3, balanced; common winner across runs)
    "default": {"d_model": 128, "nhead": 4, "num_layers": 6, "lr": 1e-3,
                "focal_gamma": 0.0, "balanced_sampling": True, "seed": 42, "dropout": 0.1},
}

# ---------------------------------------------------------------------------
# Genre coarsening (revised): 7 buckets that merge related fine genres.
# Counts (clip-level, full mapping json): Drama/Romance 1641, Comedy 1165,
# Sci-Fi/Adventure 1028, Thriller/Horror 758, Action 705, Bio/Historical 495,
# Documentary 257.
# ---------------------------------------------------------------------------
GENRE_7CLASS = {
    "Drama":          "Drama / Romance",
    "Romance":        "Drama / Romance",
    "Comedy":         "Comedy",
    "Sci-Fi/Fantasy": "Sci-Fi / Adventure",
    "Adventure":      "Sci-Fi / Adventure",
    "Thriller":       "Thriller / Horror",
    "Horror":         "Thriller / Horror",
    "Action":         "Action",
    "Biographical":   "Bio / Historical",
    "War/Historical": "Bio / Historical",
    "Documentary":    "Documentary",
}

# Top-3 (REDEFINED): Drama/Romance, Comedy, and a merged "Action / Thriller /
# Sci-Fi" bucket combining Action + Thriller + Horror + Sci-Fi/Fantasy +
# Adventure. The previous top-3 cutoff was statistically arbitrary because
# Sci-Fi/Adv (705 distinct clips), Action (705), and Thriller/Horror (686)
# were essentially tied for 3rd. Merging them avoids the arbitrary pick.
GENRE_3_REVISED = {
    "Drama":          "Drama / Romance",
    "Romance":        "Drama / Romance",
    "Comedy":         "Comedy",
    "Action":         "Action / Thriller / Sci-Fi",
    "Thriller":       "Action / Thriller / Sci-Fi",
    "Horror":         "Action / Thriller / Sci-Fi",
    "Sci-Fi/Fantasy": "Action / Thriller / Sci-Fi",
    "Adventure":      "Action / Thriller / Sci-Fi",
}

# Top-5 by count: top-3 + Thriller/Horror + Action
GENRE_5_REVISED = {
    "Drama":          "Drama / Romance",
    "Romance":        "Drama / Romance",
    "Comedy":         "Comedy",
    "Sci-Fi/Fantasy": "Sci-Fi / Adventure",
    "Adventure":      "Sci-Fi / Adventure",
    "Thriller":       "Thriller / Horror",
    "Horror":         "Thriller / Horror",
    "Action":         "Action",
}


# ---------------------------------------------------------------------------
# MacroTypeTrajStatsDataset — VLM macro_type (spatial openness) labels with
# the same depth + traj_stats + first_pose feature pipeline as TrajStatsDataset.
# Cannot reuse RealLabelClfDataset's filtering because macro_type is not a
# real-label type; we replicate the movie-level stratified split inline.
# ---------------------------------------------------------------------------

class MacroTypeTrajStatsDataset(TrajStatsDataset):
    _macro_cache = None

    @classmethod
    def _load_macro_captions(cls, root: Path):
        if cls._macro_cache is not None:
            return cls._macro_cache
        out = {}
        for ds_name in ["cinetechbench", "movieshots", "condensedmovies",
                        "shotbench", "vadb"]:
            jsonl = root / "captions" / f"{ds_name}_captions.jsonl"
            if not jsonl.exists():
                continue
            with open(jsonl) as f:
                for line in f:
                    entry = json.loads(line)
                    cid = Path(entry["video_path"]).stem
                    m = (entry.get("cinematic_data", {})
                              .get("spatial_context", {})
                              .get("macro_type", ""))
                    if m:
                        out[cid] = m
        cls._macro_cache = out
        return out

    def __init__(self, root, mapping_path, split="train", val_fraction=0.15,
                 max_seq_len=300, traj_type="direction+speed", seed=42,
                 label2idx=None, top_n_directors=10, **_kwargs):
        # Required attributes (parent init bypassed)
        self.root        = Path(root)
        self.traj_type   = traj_type
        self.label_type  = "macro_type"
        self.max_seq_len = max_seq_len
        self.max_len     = max_seq_len if traj_type == "trajectory" else max_seq_len - 1
        self.feat_dim    = TRAJ_DIM[traj_type]
        self.multi_label = False
        self.depth_size  = DEPTH_SIZE

        # Macro labels and ad blacklist
        self._macro_labels = self._load_macro_captions(self.root)
        ad_ids = set()
        for jsonl in (self.root / "captions").glob("*_captions.jsonl"):
            with open(jsonl) as f:
                for line in f:
                    entry = json.loads(line)
                    cid = Path(entry["video_path"]).stem
                    logline = (entry.get("cinematic_data", {})
                                    .get("logline_script", "") or "").lower()
                    if any(kw in logline for kw in AD_KEYWORDS):
                        ad_ids.add(cid)

        # Filter clips: pose + depth + macro_type + not ad
        with open(mapping_path) as f:
            all_clips = json.load(f)

        self._clips = []
        for c in all_clips:
            ds = c["dataset"]
            cid = Path(c["filename"]).stem
            if cid in ad_ids:
                continue
            if cid not in self._macro_labels:
                continue
            if not (self.root / "filtered_pose" / ds / f"{cid}.npz").exists():
                continue
            if not (VIPE_ROOT / ds / "depth" / f"{cid}.npy").exists():
                continue
            self._clips.append(c)

        # Movie-level stratified split (matches RealLabelClfDataset logic)
        movie_to_clips = defaultdict(list)
        for i, c in enumerate(self._clips):
            mk = c.get("movie_info", {}).get("imdb_id") or c.get("movie_name", f"unk_{i}")
            movie_to_clips[mk].append(i)

        movie_primary_label = {}
        for mk, idxs in movie_to_clips.items():
            cid = Path(self._clips[idxs[0]]["filename"]).stem
            movie_primary_label[mk] = self._macro_labels.get(cid, "__none__")

        label_to_movies = defaultdict(list)
        for mk, lbl in movie_primary_label.items():
            label_to_movies[lbl].append(mk)

        rng = np.random.default_rng(seed)
        train_mk, val_mk = set(), set()
        for lbl, movies in label_to_movies.items():
            ms = list(movies); rng.shuffle(ms)
            total_clips = sum(len(movie_to_clips[m]) for m in ms)
            target_val = int(total_clips * val_fraction)
            if len(ms) >= 2 and target_val == 0:
                target_val = 1
            so_far = 0
            for m in ms:
                n = len(movie_to_clips[m])
                if so_far < target_val:
                    val_mk.add(m); so_far += n
                else:
                    train_mk.add(m)

        keys = val_mk if split == "val" else train_mk
        split_indices = sorted(i for mk in keys for i in movie_to_clips[mk])
        self.items = [(self._clips[i]["dataset"],
                       Path(self._clips[i]["filename"]).stem, i)
                      for i in split_indices]

        if label2idx is not None:
            self.label2idx = label2idx
        else:
            seen = sorted({self._macro_labels[Path(self._clips[i]["filename"]).stem]
                           for i in split_indices})
            self.label2idx = {l: idx for idx, l in enumerate(seen)}
        self.label_names = sorted(self.label2idx.keys(), key=lambda x: self.label2idx[x])
        self.num_classes = len(self.label2idx)

        print(f"[MacroTypeTrajStatsDataset] {split} | traj={traj_type} "
              f"label=macro_type | {len(self.items)} clips from {len(keys)} movies "
              f"| {self.num_classes} classes")

    def _get_labels(self, clip_info, label_type):
        """Override: read macro label from VLM caption keyed by clip filename."""
        cid = Path(clip_info["filename"]).stem
        m = self._macro_labels.get(cid, "")
        return [m] if m else []

    def _get_labels_raw(self, clip_info, label_type):
        return self._get_labels(clip_info, label_type)

    def __getitem__(self, idx):
        ds, clip_id, _ = self.items[idx]
        feat, seq_len = self._load_feat(ds, clip_id)
        depth = self._load_depth(ds, clip_id)
        first_pose = self._load_first_pose(ds, clip_id)
        traj_stats = self._compute_traj_stats(ds, clip_id)
        label_str = self._macro_labels.get(clip_id, "")
        label = self.label2idx.get(label_str, -1)
        return {
            "feat":       torch.from_numpy(feat),
            "seq_len":    seq_len,
            "label":      label,
            "clip_id":    clip_id,
            "dataset":    ds,
            "depth":      torch.from_numpy(depth),
            "first_pose": torch.from_numpy(first_pose),
            "traj_stats": torch.from_numpy(traj_stats),
        }


def train_one_full_multilabel(model, train_ds, val_ds, label_type, cfg, device,
                               epochs=100, patience=15):
    """Multi-label trainer: BCEWithLogitsLoss + per-class macro F1 (threshold 0.5).
    Class-imbalance handled via pos_weight = (N - n_pos) / n_pos."""
    nc = train_ds.num_classes
    # Compute pos_weight from train set
    pos_counts = torch.zeros(nc)
    for _, _, ci in train_ds.items:
        ll = train_ds._get_labels(train_ds._clips[ci], label_type)
        for l in ll:
            if l in train_ds.label2idx:
                pos_counts[train_ds.label2idx[l]] += 1
    n_total = max(len(train_ds.items), 1)
    pos_weight = ((n_total - pos_counts) / pos_counts.clamp(min=1)).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True,
                               num_workers=0, collate_fn=traj_stats_collate)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False,
                             num_workers=0, collate_fn=traj_stats_collate)

    optimizer = AdamW(model.parameters(), lr=cfg["lr"], weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    best_f1, best_state = -1, None
    pat = patience
    for epoch in range(epochs):
        model.train()
        for batch in train_loader:
            feat = batch["feat"].to(device); sl = batch["seq_len"].to(device)
            lab = batch["label"].to(device).float()  # (B, C) multi-hot
            kw = {}
            if "depth" in batch: kw["depth"] = batch["depth"].to(device)
            if "first_pose" in batch: kw["first_pose"] = batch["first_pose"].to(device)
            if "traj_stats" in batch: kw["traj_stats"] = batch["traj_stats"].to(device)
            logits = model(feat, sl, **kw)
            loss = criterion(logits, lab)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
        scheduler.step()

        # Validate: macro F1 across classes at threshold 0.5
        model.eval()
        all_logits, all_labels = [], []
        with torch.no_grad():
            for batch in val_loader:
                feat = batch["feat"].to(device); sl = batch["seq_len"].to(device)
                kw = {}
                if "depth" in batch: kw["depth"] = batch["depth"].to(device)
                if "first_pose" in batch: kw["first_pose"] = batch["first_pose"].to(device)
                if "traj_stats" in batch: kw["traj_stats"] = batch["traj_stats"].to(device)
                logits = model(feat, sl, **kw)
                all_logits.append(logits.cpu())
                all_labels.append(batch["label"])
        if not all_labels:
            break
        all_logits = torch.cat(all_logits)
        all_labels = torch.cat(all_labels).numpy().astype(int)
        all_pred = (torch.sigmoid(all_logits) > 0.5).numpy().astype(int)
        f1 = f1_score(all_labels, all_pred, average="macro", zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_state = copy.deepcopy(model.state_dict())
            pat = patience
        else:
            pat -= 1
            if pat <= 0: break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_f1


def train_one_full(model, train_ds, val_ds, label_type, cfg, device, binary_pos=None,
                    epochs=100, patience=15):
    nc = train_ds.num_classes
    labels = []
    for _, _, ci in train_ds.items:
        ll = train_ds._get_labels(train_ds._clips[ci], label_type)
        labels.append(train_ds.label2idx.get(ll[0], -1) if ll else -1)
    counts = Counter(l for l in labels if l >= 0)
    total = sum(counts.values())

    if cfg.get("balanced_sampling", False):
        w = [total/(nc*counts[l]) if l >= 0 and l in counts else 0.0 for l in labels]
        sampler = WeightedRandomSampler(w, len(w), replacement=True)
    else:
        sampler = None

    lw = torch.tensor([total/(nc*max(counts.get(i, 1), 1)) for i in range(nc)],
                       dtype=torch.float32, device=device)
    criterion = FocalLoss(weight=lw, gamma=cfg.get("focal_gamma", 0.0))

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=(sampler is None),
                               sampler=sampler, num_workers=0, collate_fn=traj_stats_collate)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False,
                             num_workers=0, collate_fn=traj_stats_collate)

    optimizer = AdamW(model.parameters(), lr=cfg["lr"], weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    best_f1, best_state = -1, None
    pat = patience
    for epoch in range(epochs):
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
            best_f1 = f1
            best_state = copy.deepcopy(model.state_dict())
            pat = patience
        else:
            pat -= 1
            if pat <= 0: break

    model.load_state_dict(best_state)
    return model, best_f1


def build_dataset(setting, traj_type, split):
    """Return (train_ds, val_ds) for a given setting; common args inside."""
    common = dict(root=str(DATASET_ROOT), mapping_path=MAPPING_PATH,
                  val_fraction=0.15, max_seq_len=300, traj_type=traj_type,
                  seed=42, top_n_directors=10)

    if setting == "era_three_cine":
        t = EraRebinnedDataset(**common, split="train", label_type="era",
                                era_scheme=ERA_SCHEMES["three_cinematography"])
        v = EraRebinnedDataset(**common, split="val", label_type="era",
                                era_scheme=ERA_SCHEMES["three_cinematography"],
                                label2idx=t.label2idx)
        return t, v, "era", None

    # ── New era binnings (more balanced) ──────────────────────────────────
    # All cuts chosen using actual filtered-train clip counts (n=2930), not
    # uniform-year assumptions — clip distribution is heavily concentrated in
    # 2013-2016 (2016 alone = 26%).
    if setting in ("era_binary_2014", "era_3_balanced"):
        scheme = {
            # Near-50/50 binary at the median year. Counts: 1416 / 1514 (1.07x).
            "era_binary_2014": lambda y: "Pre-2014" if y < 2014 else "2014+",
            # Most-balanced 3-class. Counts: 958 / 842 / 1130 (1.34x).
            # Framing: pre-streaming-era / streaming-transition / streaming-mature.
            "era_3_balanced":  lambda y: ("Pre-streaming" if y < 2013
                                           else ("Transition" if y < 2016
                                                  else "Streaming-mature")),
        }[setting]
        t = EraRebinnedDataset(**common, split="train", label_type="era",
                                era_scheme=scheme)
        v = EraRebinnedDataset(**common, split="val", label_type="era",
                                era_scheme=scheme, label2idx=t.label2idx)
        return t, v, "era", None

    if setting == "country_region":
        t = TrajStatsDataset(**common, split="train", label_type="country_region")
        v = TrajStatsDataset(**common, split="val", label_type="country_region",
                              label2idx=t.label2idx)
        return t, v, "country_region", None

    if setting == "macro_type":
        common_macro = {k: v for k, v in common.items() if k != "top_n_directors"}
        t = MacroTypeTrajStatsDataset(**common_macro, split="train")
        v = MacroTypeTrajStatsDataset(**common_macro, split="val", label2idx=t.label2idx)
        return t, v, "macro_type", None

    # ── Revised 7-class genre coarsening + ablations ─────────────────────
    # Each of the 5 settings below uses GENRE_7CLASS / GENRE_3_REVISED /
    # GENRE_5_REVISED with either (a) drop_unmapped=True (clips outside the
    # subset are dropped) or (b) drop_unmapped=False (Other bucket added).
    if setting == "genre_3_drop":
        t = GroupedTrajStatsDataset(**common, split="train", label_type="genre_primary",
                                     group_map=GENRE_3_REVISED, drop_unmapped=True)
        v = GroupedTrajStatsDataset(**common, split="val", label_type="genre_primary",
                                     group_map=GENRE_3_REVISED, drop_unmapped=True,
                                     label2idx=t.label2idx)
        return t, v, "genre_primary", None

    if setting == "genre_3_other":
        t = GroupedTrajStatsDataset(**common, split="train", label_type="genre_primary",
                                     group_map=GENRE_3_REVISED, drop_unmapped=False)
        v = GroupedTrajStatsDataset(**common, split="val", label_type="genre_primary",
                                     group_map=GENRE_3_REVISED, drop_unmapped=False,
                                     label2idx=t.label2idx)
        return t, v, "genre_primary", None

    if setting == "genre_5_drop":
        t = GroupedTrajStatsDataset(**common, split="train", label_type="genre_primary",
                                     group_map=GENRE_5_REVISED, drop_unmapped=True)
        v = GroupedTrajStatsDataset(**common, split="val", label_type="genre_primary",
                                     group_map=GENRE_5_REVISED, drop_unmapped=True,
                                     label2idx=t.label2idx)
        return t, v, "genre_primary", None

    if setting == "genre_5_other":
        t = GroupedTrajStatsDataset(**common, split="train", label_type="genre_primary",
                                     group_map=GENRE_5_REVISED, drop_unmapped=False)
        v = GroupedTrajStatsDataset(**common, split="val", label_type="genre_primary",
                                     group_map=GENRE_5_REVISED, drop_unmapped=False,
                                     label2idx=t.label2idx)
        return t, v, "genre_primary", None

    if setting == "genre_7":
        # All 11 mapped fine-genres → 7 buckets; nothing falls outside.
        t = GroupedTrajStatsDataset(**common, split="train", label_type="genre_primary",
                                     group_map=GENRE_7CLASS, drop_unmapped=True)
        v = GroupedTrajStatsDataset(**common, split="val", label_type="genre_primary",
                                     group_map=GENRE_7CLASS, drop_unmapped=True,
                                     label2idx=t.label2idx)
        return t, v, "genre_primary", None

    # ── Multi-label genre variants (label_type="genre" + multi_label=True) ───
    # Same group maps as the single-label variants above, but a clip can be
    # assigned to MULTIPLE classes; loss switches to BCEWithLogitsLoss.
    if setting in ("genre_3_drop_ml", "genre_3_other_ml",
                    "genre_5_drop_ml", "genre_5_other_ml", "genre_7_ml"):
        gmap = {"genre_3_drop_ml": GENRE_3_REVISED,
                 "genre_3_other_ml": GENRE_3_REVISED,
                 "genre_5_drop_ml": GENRE_5_REVISED,
                 "genre_5_other_ml": GENRE_5_REVISED,
                 "genre_7_ml": GENRE_7CLASS}[setting]
        drop = setting.endswith("_drop_ml") or setting == "genre_7_ml"
        t = GroupedTrajStatsDataset(**common, split="train", label_type="genre",
                                     multi_label=True, group_map=gmap, drop_unmapped=drop)
        v = GroupedTrajStatsDataset(**common, split="val", label_type="genre",
                                     multi_label=True, group_map=gmap, drop_unmapped=drop,
                                     label2idx=t.label2idx)
        return t, v, "genre", None

    if setting == "genre_top3":
        t = GroupedTrajStatsDataset(**common, split="train", label_type="genre_primary",
                                     group_map=GENRE_GROUPS["top3"])
        v = GroupedTrajStatsDataset(**common, split="val", label_type="genre_primary",
                                     group_map=GENRE_GROUPS["top3"], label2idx=t.label2idx)
        return t, v, "genre_primary", None

    if setting == "genre_top5":
        t = GroupedTrajStatsDataset(**common, split="train", label_type="genre_primary",
                                     group_map=GENRE_GROUPS["top5"])
        v = GroupedTrajStatsDataset(**common, split="val", label_type="genre_primary",
                                     group_map=GENRE_GROUPS["top5"], label2idx=t.label2idx)
        return t, v, "genre_primary", None

    if setting == "director_top3":
        # Top-3 directors = Spielberg / Zemeckis + Other
        DIR_TOP3 = {
            "Steven Spielberg": "Steven Spielberg",
            "Robert Zemeckis": "Robert Zemeckis",
        }  # everything else → Other (default in GroupedTrajStatsDataset)
        t = GroupedTrajStatsDataset(**common, split="train", label_type="director",
                                     group_map=DIR_TOP3)
        v = GroupedTrajStatsDataset(**common, split="val", label_type="director",
                                     group_map=DIR_TOP3, label2idx=t.label2idx)
        return t, v, "director", None

    # ── Director multi-class (drop unmapped) — better-defined alternatives ──
    # Cutoffs picked at natural gaps in the director histogram (pose-filtered
    # clip counts). All clips outside the top-N set are dropped, mirroring
    # the genre_*_drop pattern. Single-label only (4.9% of clips have 2+
    # directors — multi-label not meaningful here).
    # Uses CLIP-level stratified split (split_mode="clip") because individual
    # directors typically have only 1-3 movies in our dataset, so movie-level
    # splitting puts entire directors all-in-train or all-in-val (e.g. Zemeckis
    # 87 clips → 7 train / 80 val under movie-level seed=42). Clip-level split
    # guarantees every class appears in both train and val.
    if setting in ("director_3_drop", "director_5_drop",
                    "director_7_drop", "director_10_drop"):
        DIR_3 = {
            "Christopher Nolan": "Christopher Nolan",
            "Wes Anderson": "Wes Anderson",
            "Steven Spielberg": "Steven Spielberg",
        }
        DIR_5 = {**DIR_3,
                 "Robert Zemeckis": "Robert Zemeckis",
                 "Quentin Tarantino": "Quentin Tarantino"}
        DIR_7 = {**DIR_5,
                 "Tom McCarthy": "Tom McCarthy",
                 "Martin Scorsese": "Martin Scorsese"}
        DIR_10 = {**DIR_7,
                  "Todd Phillips": "Todd Phillips",
                  "James Mangold": "James Mangold",
                  "Zhang Yimou": "Zhang Yimou"}
        gmap = {"director_3_drop": DIR_3,
                "director_5_drop": DIR_5,
                "director_7_drop": DIR_7,
                "director_10_drop": DIR_10}[setting]
        common2 = dict(common); common2["top_n_directors"] = 20
        t = GroupedTrajStatsDataset(**common2, split="train", label_type="director",
                                     group_map=gmap, drop_unmapped=True,
                                     split_mode="clip")
        v = GroupedTrajStatsDataset(**common2, split="val", label_type="director",
                                     group_map=gmap, drop_unmapped=True,
                                     split_mode="clip",
                                     label2idx=t.label2idx)
        return t, v, "director", None

    if setting.startswith("bin_"):
        director = {
            "bin_tarantino": "Quentin Tarantino",
            "bin_wes_anderson": "Wes Anderson",
            "bin_nolan": "Christopher Nolan",
        }[setting]
        common2 = dict(root=str(DATASET_ROOT), mapping_path=MAPPING_PATH,
                       max_seq_len=300, traj_type=traj_type)
        t = BinaryDirectorDataset(director, **common2, split="train", val_fraction=0.3, seed=42)
        v = BinaryDirectorDataset(director, **common2, split="val", val_fraction=0.3, seed=42)
        return t, v, "director", 1

    # ── Architecture-ablation suffix: "_traj_only" ──────────────────────────
    # Strip suffix and dispatch to the underlying dataset. The setting name
    # alone signals the architecture flags (use_depth=False, use_first_pose=
    # False) which are read by main()/sweep — no dataset change needed.
    if setting.endswith("_traj_only"):
        return build_dataset(setting[:-len("_traj_only")], traj_type, split)

    raise ValueError(f"Unknown setting: {setting}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--settings", nargs="+", default=[
        "macro_type", "era_three_cine", "country_region",
        "genre_top3", "genre_top5",
        "director_top3", "bin_tarantino", "bin_wes_anderson", "bin_nolan",
    ])
    ap.add_argument("--traj_types", nargs="+", default=["direction+speed", "trajectory"])
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--patience", type=int, default=15)
    args = ap.parse_args()

    device = torch.device("cuda")
    cfg = BEST_CFGS["default"]

    for setting in args.settings:
        log.info(f"\n{'='*70}\n  Training: {setting}\n{'='*70}")
        for traj_type in args.traj_types:
            tag = traj_type.replace("+", "_")
            save_path = SAVE_DIR / setting / f"{tag}_best.pt"
            if save_path.exists():
                log.info(f"  [skip] {setting}/{tag} (exists)")
                continue
            save_path.parent.mkdir(parents=True, exist_ok=True)

            try:
                seed_everything(cfg["seed"])
                # Architecture-ablation flags driven by setting-name suffix
                use_depth = not setting.endswith("_traj_only")
                use_first_pose = not setting.endswith("_traj_only")
                train_ds, val_ds, label_type, binary_pos = build_dataset(setting, traj_type, None)
                is_ml = bool(getattr(train_ds, "multi_label", False))
                model = build_model(traj_type, cfg["d_model"], cfg["nhead"],
                                     cfg["num_layers"], train_ds.num_classes,
                                     cfg.get("dropout", 0.1),
                                     use_depth=use_depth,
                                     use_first_pose=use_first_pose).to(device)
                if is_ml:
                    model, best_f1 = train_one_full_multilabel(
                        model, train_ds, val_ds, label_type, cfg, device,
                        epochs=args.epochs, patience=args.patience)
                else:
                    model, best_f1 = train_one_full(
                        model, train_ds, val_ds, label_type, cfg, device,
                        binary_pos=binary_pos, epochs=args.epochs, patience=args.patience)

                torch.save({
                    "model": model.state_dict(),
                    "label2idx": train_ds.label2idx,
                    "label_names": list(train_ds.label2idx.keys()),
                    "label_type": label_type,
                    "binary_pos": binary_pos,
                    "multi_label": is_ml,
                    "use_depth": use_depth,
                    "use_first_pose": use_first_pose,
                    "config": cfg,
                    "setting": setting,
                    "traj_type": traj_type,
                    "best_f1": float(best_f1),
                }, save_path)
                log.info(f"  [done] {setting}/{tag}: F1={best_f1:.4f} → {save_path}")
            except Exception as e:
                log.error(f"  [FAIL] {setting}/{tag}: {e}")
                import traceback; traceback.print_exc()


if __name__ == "__main__":
    main()
