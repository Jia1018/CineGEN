"""
Multimodal classification: trajectory + depth map + first pose.

Inputs:
  - Trajectory sequence (as before)
  - Depth map of first frame (scene geometry without visual content)
  - First pose: rot6D(6) + translation(3) = 9D (dir+spd only, since traj already has it)

Architecture:
  trajectory → TrajEncoder → traj_emb (D)
  depth_map  → SmallCNN → depth_emb (D)
  first_pose → Linear → pose_emb (D)  [dir+spd only]
  concat → MLP head → logits

Usage:
  python evaluate/train_multimodal_clf.py
"""

import json, logging, sys, copy
from pathlib import Path
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluate.data.real_label_dataset import (
    RealLabelClfDataset, TRAJ_DIM, VALID_TRAJ_TYPES,
    GENRE_COARSE_MAP, COARSE_GENRES, ERA_CLASSES, REGION_CLASSES,
    COUNTRY_REGION_MAP, year_to_era, LOG_SPEED_EPS,
)
from evaluate.models.classifier import TrajClassifier
from cinegen.utils.pose_utils import np_matrices_to_velocity

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", force=True)
log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_ROOT = Path("/workspace/writeable/datasets/DIY_movies")
MAPPING_PATH = str(DATASET_ROOT / "labeling/known_movies/clip_movie_mapping.json")
VIPE_ROOT = DATASET_ROOT / "vipe_results"

AD_KEYWORDS = ["movieclips", "website interface", "grid of", "thumbnail",
               "click to", "movieclips.com", "digital interface", "collage of"]
DEPTH_SIZE = 64  # Resize depth to 64x64


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class MultimodalClfDataset(RealLabelClfDataset):
    """Extends RealLabelClfDataset with depth map and first pose."""

    def __init__(self, *args, depth_size=DEPTH_SIZE, **kwargs):
        super().__init__(*args, **kwargs)
        self.depth_size = depth_size

        # Filter to clips that have depth maps
        before = len(self.items)
        self.items = [
            (ds, cid, ci) for ds, cid, ci in self.items
            if (VIPE_ROOT / ds / "depth" / f"{cid}.npy").exists()
        ]
        removed = before - len(self.items)
        if removed > 0:
            print(f"  [MultimodalClfDataset] Removed {removed} clips without depth maps")

        # Filter ad clips
        captions_dir = DATASET_ROOT / "captions"
        ad_ids = set()
        for jsonl in captions_dir.glob("*_captions.jsonl"):
            with open(jsonl) as f:
                for line in f:
                    entry = json.loads(line)
                    clip_id = Path(entry["video_path"]).stem
                    logline = entry.get("cinematic_data", {}).get("logline_script", "").lower()
                    if any(kw in logline for kw in AD_KEYWORDS):
                        ad_ids.add(clip_id)
        before2 = len(self.items)
        self.items = [(d, c, i) for d, c, i in self.items if c not in ad_ids]
        if before2 - len(self.items) > 0:
            print(f"  [MultimodalClfDataset] Removed {before2-len(self.items)} ad clips")

    def _load_depth(self, ds: str, clip_id: str) -> np.ndarray:
        """Load pre-extracted CLIP depth features (128D vector)."""
        cache_path = DATASET_ROOT / "clip_depth_features" / ds / f"{clip_id}.npy"
        if cache_path.exists():
            return np.load(cache_path).astype(np.float32)  # (128,)
        # Fallback: load raw depth (shouldn't happen if cache is complete)
        path = VIPE_ROOT / ds / "depth" / f"{clip_id}.npy"
        depth = np.load(path).astype(np.float32)
        return depth

    def _load_first_pose(self, ds: str, clip_id: str) -> np.ndarray:
        """Load first pose in same format as dir+spd traj features (8D).

        Format: [trans_dir(3), rot_dir(3), log_trans_speed(1), log_rot_speed(1)]
        This matches the main training code (data/dataset.py) convention.
        """
        from scipy.spatial.transform import Rotation
        path = self.root / "filtered_pose" / ds / f"{clip_id}.npz"
        matrices = np.load(path)["data"]  # (N, 4, 4)
        first = matrices[0]

        # Translation: decompose into direction + speed
        t0 = first[:3, 3].astype(np.float32)
        t0_speed = float(np.linalg.norm(t0))
        t0_dir = t0 / (t0_speed + 1e-8) if t0_speed > 1e-8 else np.zeros(3, dtype=np.float32)

        # Rotation: axis-angle decompose into direction + speed
        R0 = first[:3, :3]
        r0_aa = Rotation.from_matrix(R0).as_rotvec().astype(np.float32)
        r0_speed = float(np.linalg.norm(r0_aa))
        r0_dir = r0_aa / (r0_speed + 1e-8) if r0_speed > 1e-8 else np.zeros(3, dtype=np.float32)

        first_pose = np.concatenate([
            t0_dir, r0_dir,
            [np.log(t0_speed + LOG_SPEED_EPS).astype(np.float32)],
            [np.log(r0_speed + LOG_SPEED_EPS).astype(np.float32)],
        ])  # (8,)
        return first_pose.astype(np.float32)

    def __getitem__(self, idx):
        item = super().__getitem__(idx)
        ds, clip_id, _ = self.items[idx]

        # Add depth
        depth = self._load_depth(ds, clip_id)
        item["depth"] = torch.from_numpy(depth)  # (H, W)

        # Add first pose (for dir+spd, this is extra info not in the sequence)
        first_pose = self._load_first_pose(ds, clip_id)
        item["first_pose"] = torch.from_numpy(first_pose)  # (9,)

        return item


def multimodal_collate_fn(batch):
    feats = torch.stack([b["feat"] for b in batch])
    seq_lens = torch.tensor([b["seq_len"] for b in batch], dtype=torch.long)
    if isinstance(batch[0]["label"], torch.Tensor):
        labels = torch.stack([b["label"] for b in batch])
    else:
        labels = torch.tensor([b["label"] for b in batch], dtype=torch.long)

    # Depth features are pre-extracted 128D vectors
    padded_depths = torch.stack([b["depth"] for b in batch])  # (B, 128)

    first_poses = torch.stack([b["first_pose"] for b in batch])  # (B, 8)
    return {
        "feat": feats, "seq_len": seq_lens, "label": labels,
        "depth": padded_depths, "first_pose": first_poses,
        "clip_id": [b["clip_id"] for b in batch],
        "dataset": [b["dataset"] for b in batch],
    }


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class MultimodalClassifier(nn.Module):
    """
    Trajectory encoder + depth encoder + optional first pose → classification.

    For dir+spd: concat(traj_emb, depth_emb, pose_emb) → head
    For trajectory: concat(traj_emb, depth_emb) → head (first pose already in sequence)
    """

    def __init__(self, input_dim, d_model, nhead, num_layers, max_len,
                 num_classes, depth_size=64, depth_dim=128, pose_dim=64,
                 use_depth=True, use_first_pose=False, dropout=0.1):
        super().__init__()

        # Trajectory encoder (same as TrajClassifier internals)
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_embed = nn.Embedding(max_len, d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.traj_norm = nn.LayerNorm(d_model)

        # Depth feature projection (uses pre-extracted CLIP features, 128D → depth_dim)
        self.use_depth = use_depth
        if use_depth:
            self.depth_proj = nn.Sequential(
                nn.Linear(128, depth_dim),
                nn.ReLU(),
            )

        # First pose encoder (only for dir+spd) — 8D: same format as traj features
        # [trans_dir(3), rot_dir(3), log_trans_speed(1), log_rot_speed(1)]
        self.use_first_pose = use_first_pose
        if use_first_pose:
            self.pose_proj = nn.Sequential(
                nn.Linear(8, pose_dim),
                nn.ReLU(),
                nn.Linear(pose_dim, pose_dim),
            )

        # Fusion head
        fused_dim = d_model + (depth_dim if use_depth else 0) + (pose_dim if use_first_pose else 0)
        self.head = nn.Sequential(
            nn.Linear(fused_dim, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes),
        )

    def forward(self, feat, seq_lens, depth=None, first_pose=None):
        B, L, _ = feat.shape
        device = feat.device

        # Trajectory encoding
        idx = torch.arange(L, device=device).unsqueeze(0)
        pad_mask = ~(idx < seq_lens.unsqueeze(1))
        pos = torch.arange(L, device=device)
        h = self.input_proj(feat) + self.pos_embed(pos)
        h = self.encoder(h, src_key_padding_mask=pad_mask)
        valid = (~pad_mask).float().unsqueeze(-1)
        traj_emb = (h * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1)
        traj_emb = self.traj_norm(traj_emb)  # (B, d_model)

        # Fusion
        parts = [traj_emb]
        if self.use_depth and depth is not None:
            parts.append(self.depth_proj(depth))
        if self.use_first_pose and first_pose is not None:
            parts.append(self.pose_proj(first_pose))
        fused = torch.cat(parts, dim=-1)

        return self.head(fused)


class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=2.0):
        super().__init__()
        self.gamma = gamma
        self.weight = weight

    def forward(self, logits, targets):
        ce = nn.functional.cross_entropy(logits, targets, weight=self.weight, reduction="none")
        pt = torch.exp(-ce)
        return (((1 - pt) ** self.gamma) * ce).mean()


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(model, loader, device, criterion, use_depth=False, use_first_pose=False):
    model.eval()
    need_extra = use_depth or use_first_pose
    class_correct, class_total = {}, {}
    for batch in loader:
        feat = batch["feat"].to(device)
        seq_lens = batch["seq_len"].to(device)
        labels = batch["label"].to(device)
        mask = labels >= 0
        if mask.sum() == 0: continue

        if need_extra:
            depth = batch["depth"].to(device) if use_depth else None
            fp = batch["first_pose"].to(device) if use_first_pose else None
            logits = model(feat[mask], seq_lens[mask],
                           depth[mask] if depth is not None else None,
                           fp[mask] if fp is not None else None)
        else:
            logits = model(feat[mask], seq_lens[mask])

        preds = logits.argmax(-1)
        for p, t in zip(preds.cpu().tolist(), labels[mask].cpu().tolist()):
            class_total[t] = class_total.get(t, 0) + 1
            if p == t: class_correct[t] = class_correct.get(t, 0) + 1
    recalls = [class_correct.get(c, 0) / class_total[c] for c in class_total]
    return np.mean(recalls) if recalls else 0


def train_one(traj_type, label_type, mode, cfg, device):
    """
    mode: 'traj_only', 'traj+depth', 'traj+depth+pose'
    traj+depth+pose only valid for dir+spd
    """
    use_depth = "depth" in mode
    use_first_pose = "pose" in mode and traj_type == "direction+speed"
    tag = f"{mode}/{label_type}/{traj_type}"

    common = dict(root=str(DATASET_ROOT), mapping_path=MAPPING_PATH,
                  val_fraction=0.15, max_seq_len=cfg["max_seq_len"],
                  traj_type=traj_type, label_type=label_type, seed=42,
                  top_n_directors=10)

    need_extra = use_depth or use_first_pose
    if need_extra:
        train_ds = MultimodalClfDataset(**common, split="train", depth_size=DEPTH_SIZE)
        val_ds = MultimodalClfDataset(**common, split="val", label2idx=train_ds.label2idx, depth_size=DEPTH_SIZE)
        collate = multimodal_collate_fn
    else:
        train_ds = RealLabelClfDataset(**common, split="train")
        val_ds = RealLabelClfDataset(**common, split="val", label2idx=train_ds.label2idx)
        from evaluate.data.real_label_dataset import collate_fn
        collate = collate_fn

    nc = train_ds.num_classes
    max_len = cfg["max_seq_len"] if traj_type == "trajectory" else cfg["max_seq_len"] - 1

    if need_extra:
        model = MultimodalClassifier(
            input_dim=TRAJ_DIM[traj_type], d_model=cfg["d_model"],
            nhead=cfg["nhead"], num_layers=cfg["num_layers"],
            max_len=max_len, num_classes=nc,
            depth_size=DEPTH_SIZE, depth_dim=cfg.get("depth_dim", 128),
            pose_dim=cfg.get("pose_dim", 64),
            use_depth=use_depth, use_first_pose=use_first_pose,
            dropout=cfg["dropout"],
        ).to(device)
    else:
        model = TrajClassifier(
            input_dim=TRAJ_DIM[traj_type], d_model=cfg["d_model"],
            nhead=cfg["nhead"], num_layers=cfg["num_layers"],
            max_len=max_len, num_classes=nc, dropout=cfg["dropout"],
        ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(f"=== {tag} | params={n_params/1e3:.1f}K | classes={nc} ===")

    # Balanced sampler
    labels = []
    for _, _, ci in train_ds.items:
        ll = train_ds._get_labels(train_ds._clips[ci], label_type)
        labels.append(train_ds.label2idx.get(ll[0], -1) if ll else -1)
    counts = Counter(l for l in labels if l >= 0)
    total = sum(counts.values())
    w = [total / (nc * counts[l]) if l >= 0 and l in counts else 0.0 for l in labels]
    sampler = WeightedRandomSampler(w, len(w), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], sampler=sampler,
                              num_workers=4, collate_fn=collate, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=cfg["batch_size"], shuffle=False,
                            num_workers=4, collate_fn=collate, pin_memory=True)

    loss_weights = torch.tensor(
        [total / (nc * max(counts.get(i, 1), 1)) for i in range(nc)],
        dtype=torch.float32, device=device)
    criterion = FocalLoss(weight=loss_weights, gamma=2.0)

    optimizer = AdamW(model.parameters(), lr=cfg["lr"], weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg["epochs"])

    best_bal_acc = -1.0
    patience_left = cfg["patience"]

    for epoch in range(cfg["epochs"]):
        model.train()
        for batch in train_loader:
            feat = batch["feat"].to(device)
            seq_lens = batch["seq_len"].to(device)
            lab = batch["label"].to(device)
            mask = lab >= 0
            if mask.sum() == 0: continue

            if need_extra:
                depth = batch["depth"].to(device) if use_depth else None
                fp = batch["first_pose"].to(device) if use_first_pose else None
                logits = model(feat[mask], seq_lens[mask],
                               depth[mask] if depth is not None else None,
                               fp[mask] if fp is not None else None)
            else:
                logits = model(feat[mask], seq_lens[mask])

            loss = criterion(logits, lab[mask])
            optimizer.zero_grad(); loss.backward(); optimizer.step()
        scheduler.step()

        bal_acc = evaluate(model, val_loader, device, criterion, use_depth, use_first_pose)
        if bal_acc > best_bal_acc:
            best_bal_acc = bal_acc
            patience_left = cfg["patience"]
        else:
            patience_left -= 1
            if patience_left <= 0:
                break

    log.info(f"[{tag}] Done. best_bal_acc={best_bal_acc*100:.2f}%")
    return best_bal_acc


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    label_types = ["genre_primary", "era", "country_region", "director"]
    traj_types = ["trajectory", "direction+speed"]

    cfg = {
        "d_model": 128, "nhead": 4, "num_layers": 4, "dropout": 0.1,
        "max_seq_len": 300, "batch_size": 32, "lr": 1e-3,  # smaller batch for CLIP memory
        "epochs": 100, "patience": 15,
        "depth_dim": 128, "pose_dim": 64,
    }

    results = {}

    for traj_type in traj_types:
        # Determine which modes to run
        if traj_type == "direction+speed":
            modes = ["traj_only", "traj+depth", "traj+pose", "traj+depth+pose"]
        else:
            modes = ["traj_only", "traj+depth"]

        for mode in modes:
            for label_type in label_types:
                bal_acc = train_one(traj_type, label_type, mode, cfg, device)
                results.setdefault(label_type, {}).setdefault(traj_type, {})[mode] = bal_acc * 100

    # Print results
    print("\n" + "=" * 95)
    print("MULTIMODAL CLASSIFICATION (Balanced Accuracy %)")
    print("=" * 95)

    for label_type in label_types:
        print(f"\n--- {label_type} ---")
        all_modes = set()
        for traj in traj_types:
            all_modes.update(results.get(label_type, {}).get(traj, {}).keys())
        all_modes = sorted(all_modes)

        print(f"  {'Mode':<25s} {'trajectory':>12s} {'dir+spd':>12s}")
        print(f"  {'-'*50}")
        for mode in all_modes:
            row = f"  {mode:<25s}"
            for traj in traj_types:
                val = results.get(label_type, {}).get(traj, {}).get(mode)
                row += f" {val:>11.1f}%" if val is not None else f" {'—':>12s}"
            print(row)

    # Save
    save_path = PROJECT_ROOT / "checkpoints" / "multimodal_clf" / "summary.json"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(results, f, indent=2)
    log.info(f"\nSummary → {save_path}")


if __name__ == "__main__":
    main()
