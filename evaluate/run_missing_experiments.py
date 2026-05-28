"""
Run the missing experiments:
1. Binary directors with BOTH D+S and Traj (5 top directors)
2. Director top-6 (Nolan/Anderson/Tarantino/Spielberg/Zemeckis + Other)

All with macro_f1 early stopping, full 100 epochs, 3 seeds.
"""
import json, logging, sys, copy
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.metrics import (balanced_accuracy_score, f1_score,
                              average_precision_score, roc_auc_score)
from sklearn.preprocessing import label_binarize

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluate.data.real_label_dataset import (
    RealLabelClfDataset, TRAJ_DIM, GENRE_COARSE_MAP, COARSE_GENRES,
    COUNTRY_REGION_MAP, year_to_era, LOG_SPEED_EPS,
)
from evaluate.train_multimodal_clf import (
    MultimodalClfDataset, multimodal_collate_fn,
    FocalLoss, DATASET_ROOT, MAPPING_PATH,
)
from cinegen.utils.pose_utils import np_matrices_to_velocity

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", force=True)
log = logging.getLogger(__name__)

SAVE_DIR = Path("checkpoints/best_clf")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

TARGET_DIRECTORS = ["Christopher Nolan", "Wes Anderson", "Quentin Tarantino",
                    "Steven Spielberg", "Martin Scorsese"]

DIRECTOR_TOP6_MAP = {
    "Christopher Nolan": "Christopher Nolan",
    "Wes Anderson": "Wes Anderson",
    "Quentin Tarantino": "Quentin Tarantino",
    "Steven Spielberg": "Steven Spielberg",
    "Robert Zemeckis": "Robert Zemeckis",
}


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# DspWithTrajStats model (same as before)
# ---------------------------------------------------------------------------

class DspWithTrajStats(nn.Module):
    def __init__(self, d_model, nhead, num_layers, max_len, num_classes,
                 pose_dim=64, stat_dim=64, depth_dim=128,
                 use_depth=True, use_first_pose=True, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(8, d_model)
        self.pos_embed = nn.Embedding(max_len, d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.use_first_pose = use_first_pose
        if use_first_pose:
            self.pose_proj = nn.Sequential(nn.Linear(8, pose_dim), nn.ReLU())
        self.stat_proj = nn.Sequential(nn.Linear(13, stat_dim), nn.ReLU())
        self.use_depth = use_depth
        if use_depth:
            self.depth_proj = nn.Sequential(nn.Linear(128, depth_dim), nn.ReLU())
        fused = (d_model + stat_dim
                 + (pose_dim if use_first_pose else 0)
                 + (depth_dim if use_depth else 0))
        self.head = nn.Sequential(
            nn.Linear(fused, d_model), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(d_model, num_classes))

    def forward(self, feat, seq_lens, depth=None, first_pose=None, traj_stats=None):
        B, L, _ = feat.shape
        device = feat.device
        idx = torch.arange(L, device=device).unsqueeze(0)
        pad_mask = ~(idx < seq_lens.unsqueeze(1))
        h = self.input_proj(feat) + self.pos_embed(torch.arange(L, device=device))
        h = self.encoder(h, src_key_padding_mask=pad_mask)
        valid = (~pad_mask).float().unsqueeze(-1)
        pooled = (h * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1)
        pooled = self.norm(pooled)
        parts = [pooled]
        if self.use_first_pose and first_pose is not None:
            parts.append(self.pose_proj(first_pose))
        if traj_stats is not None: parts.append(self.stat_proj(traj_stats))
        if self.use_depth and depth is not None: parts.append(self.depth_proj(depth))
        return self.head(torch.cat(parts, dim=-1))


class TrajWithStatsModel(nn.Module):
    """Same as DspWithTrajStats but for 9D trajectory input."""
    def __init__(self, d_model, nhead, num_layers, max_len, num_classes,
                 pose_dim=64, stat_dim=64, depth_dim=128,
                 use_depth=True, use_first_pose=True, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(9, d_model)
        self.pos_embed = nn.Embedding(max_len, d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.use_first_pose = use_first_pose
        if use_first_pose:
            self.pose_proj = nn.Sequential(nn.Linear(8, pose_dim), nn.ReLU())
        self.stat_proj = nn.Sequential(nn.Linear(13, stat_dim), nn.ReLU())
        self.use_depth = use_depth
        if use_depth:
            self.depth_proj = nn.Sequential(nn.Linear(128, depth_dim), nn.ReLU())
        fused = (d_model + stat_dim
                 + (pose_dim if use_first_pose else 0)
                 + (depth_dim if use_depth else 0))
        self.head = nn.Sequential(
            nn.Linear(fused, d_model), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(d_model, num_classes))

    def forward(self, feat, seq_lens, depth=None, first_pose=None, traj_stats=None):
        B, L, _ = feat.shape
        device = feat.device
        idx = torch.arange(L, device=device).unsqueeze(0)
        pad_mask = ~(idx < seq_lens.unsqueeze(1))
        h = self.input_proj(feat) + self.pos_embed(torch.arange(L, device=device))
        h = self.encoder(h, src_key_padding_mask=pad_mask)
        valid = (~pad_mask).float().unsqueeze(-1)
        pooled = (h * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1)
        pooled = self.norm(pooled)
        parts = [pooled]
        if self.use_first_pose and first_pose is not None:
            parts.append(self.pose_proj(first_pose))
        if traj_stats is not None: parts.append(self.stat_proj(traj_stats))
        if self.use_depth and depth is not None: parts.append(self.depth_proj(depth))
        return self.head(torch.cat(parts, dim=-1))


# ---------------------------------------------------------------------------
# TrajStats dataset (adds traj_stats + first_pose + depth to base)
# ---------------------------------------------------------------------------

class TrajStatsDataset(MultimodalClfDataset):
    def _compute_traj_stats(self, ds, clip_id):
        path = self.root / "filtered_pose" / ds / f"{clip_id}.npz"
        matrices = np.load(path)["data"]
        trans = matrices[:, :3, 3]
        total_disp = trans[-1] - trans[0]
        disp_mag = np.linalg.norm(total_disp)
        disp_height = total_disp[1] if len(total_disp) > 1 else 0
        if len(matrices) > 1:
            td, rd, ts, rs, _ = np_matrices_to_velocity(matrices)
            mean_ts, std_ts = ts.mean(), ts.std()
            mean_rs, std_rs = rs.mean(), rs.std()
            if len(td) > 1:
                dots = np.sum(td[:-1] * td[1:], axis=-1).clip(-1, 1)
                angles = np.arccos(dots)
                mean_curv, max_curv = angles.mean(), angles.max()
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

    def __getitem__(self, idx):
        item = super().__getitem__(idx)
        ds, clip_id, _ = self.items[idx]
        item["traj_stats"] = torch.from_numpy(self._compute_traj_stats(ds, clip_id))
        return item


def traj_stats_collate(batch):
    base = multimodal_collate_fn(batch)
    base["traj_stats"] = torch.stack([b["traj_stats"] for b in batch])
    return base


# ---------------------------------------------------------------------------
# Grouped dataset for director top-6
# ---------------------------------------------------------------------------

class GroupedTrajStatsDataset(TrajStatsDataset):
    def __init__(self, *args, group_map=None, drop_unmapped=False, **kwargs):
        super().__init__(*args, **kwargs)
        if group_map is None:
            return
        self._group_map = group_map
        self._drop_unmapped = drop_unmapped
        new_labels = sorted(set(group_map.values()))
        if not drop_unmapped and "Other" not in new_labels:
            new_labels.append("Other")
            new_labels.sort()
        self.label2idx = {l: i for i, l in enumerate(new_labels)}
        self.label_names = new_labels
        self.num_classes = len(new_labels)
        # Filter items. Single-label: keep clip if PRIMARY genre is mapped.
        # Multi-label: keep clip if AT LEAST ONE coarse genre maps.
        is_ml = bool(getattr(self, "multi_label", False))
        new_items = []
        for ds, cid, ci in self.items:
            ll = self._get_labels_orig(self._clips[ci], self.label_type)
            if not ll:
                continue
            if drop_unmapped:
                if is_ml:
                    if not any(g in group_map for g in ll):
                        continue
                else:
                    if ll[0] not in group_map:
                        continue
            new_items.append((ds, cid, ci))
        self.items = new_items

    def _get_labels_orig(self, clip_info, label_type):
        """Get original labels before grouping."""
        return super()._get_labels(clip_info, label_type)

    def _get_labels(self, clip_info, label_type):
        original = self._get_labels_orig(clip_info, label_type)
        if not original:
            return []
        if not (hasattr(self, '_group_map') and self._group_map):
            return original
        is_ml = bool(getattr(self, "multi_label", False))
        if is_ml:
            mapped = []
            for g in original:
                if g in self._group_map:
                    m = self._group_map[g]
                    if m not in mapped:
                        mapped.append(m)
                elif not getattr(self, '_drop_unmapped', False):
                    if "Other" not in mapped:
                        mapped.append("Other")
            return mapped
        # single-label: drop_unmapped or fall back to "Other"
        if getattr(self, '_drop_unmapped', False) and original[0] not in self._group_map:
            return []
        return [self._group_map.get(original[0], "Other")]


# ---------------------------------------------------------------------------
# Binary director dataset (custom split ensuring director in both train/val)
# ---------------------------------------------------------------------------

class BinaryDirectorDataset(TrajStatsDataset):
    def __init__(self, target_director, *args, split="train", val_fraction=0.3, seed=42, **kwargs):
        self._target_director = target_director
        kwargs["label_type"] = "director"
        kwargs.setdefault("top_n_directors", 20)
        super().__init__(*args, split=split, seed=seed, **kwargs)

        # Rebuild items with custom split ensuring target director in both
        target_indices, other_indices = [], []
        for i, c in enumerate(self._clips):
            dirs = c.get("movie_info", {}).get("directors", [])
            if target_director in dirs:
                target_indices.append(i)
            else:
                other_indices.append(i)

        target_movies = defaultdict(list)
        for i in target_indices:
            mk = self._clips[i].get("movie_info", {}).get("imdb_id", f"unk_{i}")
            target_movies[mk].append(i)
        other_movies = defaultdict(list)
        for i in other_indices:
            mk = self._clips[i].get("movie_info", {}).get("imdb_id", f"unk_{i}")
            other_movies[mk].append(i)

        rng = np.random.default_rng(seed)
        tmk = sorted(target_movies.keys()); rng.shuffle(tmk)
        n_val_t = max(1, int(len(tmk) * val_fraction))
        if n_val_t >= len(tmk): n_val_t = len(tmk) - 1
        val_tmk = set(tmk[:n_val_t]); train_tmk = set(tmk[n_val_t:])

        omk = sorted(other_movies.keys()); rng.shuffle(omk)
        n_val_o = max(1, int(len(omk) * 0.15))
        val_omk = set(omk[:n_val_o]); train_omk = set(omk[n_val_o:])

        # Filter: require cached depth features exist
        cache_root = Path(str(DATASET_ROOT)) / "clip_depth_features"
        all_items = []
        for i, c in enumerate(self._clips):
            ds = c["dataset"]; cid = Path(c["filename"]).stem
            if not (cache_root / ds / f"{cid}.npy").exists():
                continue
            mk = c.get("movie_info", {}).get("imdb_id", f"unk_{i}")
            is_t = target_director in c.get("movie_info", {}).get("directors", [])
            if split == "train":
                if (is_t and mk in train_tmk) or (not is_t and mk in train_omk):
                    all_items.append((ds, cid, i))
            else:
                if (is_t and mk in val_tmk) or (not is_t and mk in val_omk):
                    all_items.append((ds, cid, i))

        self.items = all_items
        self.label2idx = {"Other": 0, target_director: 1}
        self.label_names = ["Other", target_director]
        self.num_classes = 2

    def _get_labels(self, clip_info, label_type):
        dirs = clip_info.get("movie_info", {}).get("directors", [])
        return [self._target_director] if self._target_director in dirs else ["Other"]


# ---------------------------------------------------------------------------
# Training + evaluation
# ---------------------------------------------------------------------------

def predict(model, loader, device):
    model.eval()
    logits_list, labels_list = [], []
    with torch.no_grad():
        for batch in loader:
            feat = batch["feat"].to(device); sl = batch["seq_len"].to(device)
            lab = batch["label"]; mask = lab >= 0
            if mask.sum() == 0: continue
            kw = {}
            if "depth" in batch: kw["depth"] = batch["depth"].to(device)[mask]
            if "first_pose" in batch: kw["first_pose"] = batch["first_pose"].to(device)[mask]
            if "traj_stats" in batch: kw["traj_stats"] = batch["traj_stats"].to(device)[mask]
            logits = model(feat[mask], sl[mask], **kw)
            logits_list.append(logits.cpu())
            labels_list.extend(lab[mask].tolist())
    return torch.cat(logits_list), np.array(labels_list)


def compute_metrics(logits, labels, nc, binary_pos=None):
    preds = logits.argmax(-1).numpy()
    probs = torch.softmax(logits, dim=-1).numpy()
    m = {
        "raw_acc": float((preds == labels).mean()),
        "bal_acc": float(balanced_accuracy_score(labels, preds)),
        "macro_f1": float(f1_score(labels, preds, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(labels, preds, average="weighted", zero_division=0)),
    }
    if binary_pos is not None:
        m["pos_f1"] = float(f1_score(labels, preds, pos_label=binary_pos, zero_division=0))
        try:
            m["auc"] = float(roc_auc_score(labels, probs[:, binary_pos]))
            m["ap"] = float(average_precision_score(labels, probs[:, binary_pos]))
        except:
            m["auc"] = m["ap"] = 0
    else:
        aps = []
        for i in range(nc):
            if (labels == i).sum() > 0:
                aps.append(average_precision_score((labels == i).astype(int), probs[:, i]))
        m["macro_ap"] = float(np.mean(aps)) if aps else 0
    return m


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
            f1 = f1_score(val_labels, val_logits.argmax(-1).numpy(),
                          pos_label=binary_pos, zero_division=0)
        else:
            f1 = f1_score(val_labels, val_logits.argmax(-1).numpy(),
                          average="macro", zero_division=0)
        if f1 > best_f1:
            best_f1 = f1; best_state = copy.deepcopy(model.state_dict()); patience = 15
        else:
            patience -= 1
            if patience <= 0: break

    model.load_state_dict(best_state)
    val_logits, val_labels = predict(model, val_loader, device)
    return compute_metrics(val_logits, val_labels, train_ds.num_classes, binary_pos)


def build_model(traj_type, d_model, nhead, num_layers, nc, dropout=0.1,
                use_depth=True, use_first_pose=True):
    max_len = 300 if traj_type == "trajectory" else 299
    if traj_type == "direction+speed":
        return DspWithTrajStats(d_model, nhead, num_layers, max_len, nc,
                                 use_depth=use_depth,
                                 use_first_pose=use_first_pose,
                                 dropout=dropout)
    else:
        return TrajWithStatsModel(d_model, nhead, num_layers, max_len, nc,
                                   use_depth=use_depth,
                                   use_first_pose=use_first_pose,
                                   dropout=dropout)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cfgs = [
        {"d_model": 128, "nhead": 4, "num_layers": 6, "lr": 1e-3, "focal_gamma": 0.0},
        {"d_model": 128, "nhead": 4, "num_layers": 6, "lr": 5e-4, "focal_gamma": 0.0},
        {"d_model": 128, "nhead": 4, "num_layers": 4, "lr": 1e-3, "focal_gamma": 0.0},
    ]
    seeds = [42, 43, 44]

    results = {}

    # ─── Experiment 1: Binary directors × both reps ──────────────────────
    print("\n" + "=" * 100)
    print("EXPERIMENT 1: Per-director binary classification (D+S vs Traj)")
    print("=" * 100)

    for target_dir in TARGET_DIRECTORS:
        print(f"\n  --- {target_dir} ---")
        best = {"direction+speed": {"f1": -1}, "trajectory": {"f1": -1}}

        for traj_type in ["direction+speed", "trajectory"]:
            for cfg in cfgs:
                for seed in seeds:
                    try:
                        seed_everything(seed)
                        common = dict(root=str(DATASET_ROOT), mapping_path=MAPPING_PATH,
                                       max_seq_len=300, traj_type=traj_type)
                        train_ds = BinaryDirectorDataset(target_dir, **common,
                                                          split="train", val_fraction=0.3, seed=42)
                        val_ds = BinaryDirectorDataset(target_dir, **common,
                                                        split="val", val_fraction=0.3, seed=42)

                        n_pos = sum(1 for _, _, ci in train_ds.items
                                     if target_dir in train_ds._clips[ci].get("movie_info", {}).get("directors", []))
                        if n_pos < 2:
                            continue

                        model = build_model(traj_type, cfg["d_model"], cfg["nhead"],
                                             cfg["num_layers"], 2).to(device)
                        m = train_one(model, train_ds, val_ds, "director", cfg, device, binary_pos=1)

                        if m.get("pos_f1", 0) > best[traj_type]["f1"]:
                            best[traj_type] = {"f1": m["pos_f1"], "metrics": m,
                                                "cfg": cfg, "seed": seed}
                        log.info(f"  {traj_type} {cfg['d_model']}d/{cfg['num_layers']}L/s{seed}: "
                                 f"F1={m.get('pos_f1',0)*100:.1f}% AUC={m.get('auc',0)*100:.1f}%")
                    except Exception as e:
                        log.error(f"  FAIL: {e}")

        d = best["direction+speed"]
        t = best["trajectory"]
        key = f"bin_{target_dir.split()[-1].lower()}"
        results[key] = {"dsp": d, "traj": t}
        print(f"  >> D+S: F1={d.get('metrics',{}).get('pos_f1',0)*100:.1f}% "
              f"AUC={d.get('metrics',{}).get('auc',0)*100:.1f}%")
        print(f"  >> Traj: F1={t.get('metrics',{}).get('pos_f1',0)*100:.1f}% "
              f"AUC={t.get('metrics',{}).get('auc',0)*100:.1f}%")

    # ─── Experiment 2: Director top-6 ────────────────────────────────────
    print("\n" + "=" * 100)
    print("EXPERIMENT 2: Director top-6 (Nolan/Anderson/Tarantino/Spielberg/Zemeckis + Other)")
    print("=" * 100)

    best_top6 = {"direction+speed": {"f1": -1}, "trajectory": {"f1": -1}}

    for traj_type in ["direction+speed", "trajectory"]:
        for cfg in cfgs:
            for seed in seeds:
                try:
                    seed_everything(seed)
                    common = dict(root=str(DATASET_ROOT), mapping_path=MAPPING_PATH,
                                   val_fraction=0.15, max_seq_len=300,
                                   traj_type=traj_type, label_type="director",
                                   seed=42, top_n_directors=20)
                    train_ds = GroupedTrajStatsDataset(**common, split="train",
                                                        group_map=DIRECTOR_TOP6_MAP)
                    val_ds = GroupedTrajStatsDataset(**common, split="val",
                                                      label2idx=train_ds.label2idx,
                                                      group_map=DIRECTOR_TOP6_MAP)

                    model = build_model(traj_type, cfg["d_model"], cfg["nhead"],
                                         cfg["num_layers"], train_ds.num_classes).to(device)
                    m = train_one(model, train_ds, val_ds, "director", cfg, device)

                    if m["macro_f1"] > best_top6[traj_type]["f1"]:
                        best_top6[traj_type] = {"f1": m["macro_f1"], "metrics": m,
                                                  "cfg": cfg, "seed": seed}
                    log.info(f"  {traj_type} {cfg['d_model']}d/{cfg['num_layers']}L/s{seed}: "
                             f"F1={m['macro_f1']*100:.1f}% BalAcc={m['bal_acc']*100:.1f}%")
                except Exception as e:
                    log.error(f"  FAIL: {e}")

    results["director_top6"] = {"dsp": best_top6["direction+speed"],
                                  "traj": best_top6["trajectory"]}

    # ─── Summary ─────────────────────────────────────────────────────────
    print("\n" + "=" * 110)
    print("RESULTS SUMMARY")
    print("=" * 110)

    print(f"\n  {'Setting':<25s} {'D+S metric':>12s} {'Traj metric':>13s} {'Diff':>8s} {'Winner':>8s}")
    print(f"  {'-'*70}")

    for key, r in results.items():
        d_m = r["dsp"].get("metrics", {})
        t_m = r["traj"].get("metrics", {})
        if "pos_f1" in d_m:  # binary
            dv = d_m.get("pos_f1", 0) * 100
            tv = t_m.get("pos_f1", 0) * 100
            metric = "pos_F1"
        else:  # multiclass
            dv = d_m.get("macro_f1", 0) * 100
            tv = t_m.get("macro_f1", 0) * 100
            metric = "macro_F1"
        diff = dv - tv
        winner = "D+S" if dv >= tv else "Traj"
        d_auc = f" AUC={d_m.get('auc',0)*100:.0f}%" if "auc" in d_m else ""
        t_auc = f" AUC={t_m.get('auc',0)*100:.0f}%" if "auc" in t_m else ""
        print(f"  {key:<25s} {dv:>6.1f}%{d_auc:>6s} {tv:>6.1f}%{t_auc:>6s} {diff:>+7.1f}  {winner:>7s}")

    # Save
    save_path = SAVE_DIR / "missing_experiments.json"
    with open(save_path, "w") as f:
        json.dump(results, f, indent=2)
    log.info(f"\nSaved → {save_path}")


if __name__ == "__main__":
    main()
