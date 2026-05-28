"""
Improved real-label classification dataset with:
  1. Movie-level aggregation — pool all clips from a movie, classify once
  2. Trajectory augmentation — time reversal, speed perturbation, random crop
  3. Class-balanced sampling — WeightedRandomSampler for equal class representation

These address the core issues:
  - Movie-level labels + clip-level classification = granularity mismatch
  - Small dataset (~3k clips) benefits from augmentation
  - Heavy class imbalance hurts minority class learning
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import Counter, defaultdict

import torch
from torch.utils.data import Dataset, WeightedRandomSampler

from evaluate.data.real_label_dataset import (
    RealLabelClfDataset, TRAJ_DIM, VALID_TRAJ_TYPES, VALID_LABEL_TYPES,
    GENRE_COARSE_MAP, COARSE_GENRES, ERA_CLASSES, REGION_CLASSES,
    COUNTRY_REGION_MAP, year_to_era, LOG_SPEED_EPS, collate_fn,
)
from cinegen.utils.pose_utils import np_matrices_to_velocity


# ---------------------------------------------------------------------------
# Trajectory Augmentation
# ---------------------------------------------------------------------------

class TrajectoryAugmentor:
    """Augment trajectory sequences for data augmentation."""

    def __init__(self, p_reverse=0.3, p_speed=0.3, p_crop=0.3, p_noise=0.2,
                 speed_range=(0.7, 1.4), crop_min_frac=0.5, noise_std=0.02):
        self.p_reverse = p_reverse
        self.p_speed = p_speed
        self.p_crop = p_crop
        self.p_noise = p_noise
        self.speed_range = speed_range
        self.crop_min_frac = crop_min_frac
        self.noise_std = noise_std

    def __call__(self, feat: np.ndarray, seq_len: int, rng: np.random.Generator):
        """
        Args:
            feat: (max_len, D) padded feature array
            seq_len: actual valid length
            rng: numpy random generator
        Returns:
            augmented feat, new seq_len
        """
        valid = feat[:seq_len].copy()

        # Random temporal crop
        if rng.random() < self.p_crop and seq_len > 10:
            min_len = max(5, int(seq_len * self.crop_min_frac))
            crop_len = rng.integers(min_len, seq_len + 1)
            start = rng.integers(0, seq_len - crop_len + 1)
            valid = valid[start:start + crop_len]
            seq_len = crop_len

        # Time reversal
        if rng.random() < self.p_reverse:
            valid = valid[::-1].copy()

        # Speed perturbation (resample temporal axis)
        if rng.random() < self.p_speed and seq_len > 5:
            factor = rng.uniform(*self.speed_range)
            new_len = max(3, int(seq_len * factor))
            indices = np.linspace(0, seq_len - 1, new_len)
            valid = np.array([
                np.interp(indices, np.arange(seq_len), valid[:, d])
                for d in range(valid.shape[1])
            ]).T
            seq_len = new_len

        # Gaussian noise
        if rng.random() < self.p_noise:
            valid = valid + rng.normal(0, self.noise_std, valid.shape).astype(np.float32)

        # Re-pad to max_len
        max_len = feat.shape[0]
        if seq_len > max_len:
            valid = valid[:max_len]
            seq_len = max_len

        out = np.zeros_like(feat)
        out[:seq_len] = valid
        return out.astype(np.float32), seq_len


# ---------------------------------------------------------------------------
# Movie-Level Dataset
# ---------------------------------------------------------------------------

class MovieLevelClfDataset(Dataset):
    """
    Aggregates all clips from the same movie into one sample.

    Each movie is represented by pooling its clip features:
      - mean pooling: average trajectory features across all clips
      - stats pooling: [mean, std, min, max] of per-clip mean features

    The label is the movie's ground-truth label (genre, era, etc.)
    """

    def __init__(
        self,
        root:            str,
        mapping_path:    str,
        split:           str               = "train",
        val_fraction:    float             = 0.15,
        max_seq_len:     int               = 300,
        traj_type:       str               = "direction+speed",
        label_type:      str               = "genre_primary",
        seed:            int               = 42,
        label2idx:       Optional[dict]    = None,
        pool_mode:       str               = "stats",  # "mean" or "stats"
        augment:         bool              = False,
        top_n_directors: int               = 20,
    ):
        # Build a clip-level dataset first (reuses all the splitting logic)
        self._clip_ds = RealLabelClfDataset(
            root=root, mapping_path=mapping_path, split=split,
            val_fraction=val_fraction, max_seq_len=max_seq_len,
            traj_type=traj_type, label_type=label_type, seed=seed,
            label2idx=label2idx, top_n_directors=top_n_directors,
        )
        self.label2idx = self._clip_ds.label2idx
        self.label_names = self._clip_ds.label_names
        self.num_classes = self._clip_ds.num_classes
        self.label_type = label_type
        self.traj_type = traj_type
        self.pool_mode = pool_mode
        self.feat_dim = TRAJ_DIM[traj_type]
        self.augmentor = TrajectoryAugmentor() if augment else None
        self._rng = np.random.default_rng(seed + 1000)

        # Group clips by movie
        movie_to_items = defaultdict(list)
        for idx, (ds, clip_id, clip_idx) in enumerate(self._clip_ds.items):
            c = self._clip_ds._clips[clip_idx]
            movie_key = c.get("movie_info", {}).get("imdb_id") or c.get("movie_name", "")
            movie_to_items[movie_key].append(idx)

        # Build movie-level items: (movie_key, [clip_indices], label)
        self.items = []
        self._movie_labels = {}
        for movie_key, clip_indices in movie_to_items.items():
            # Get label from first clip (all clips in same movie share label)
            first_clip_idx = self._clip_ds.items[clip_indices[0]][2]
            labels = self._clip_ds._get_labels(
                self._clip_ds._clips[first_clip_idx], label_type)
            if labels:
                label = self._clip_ds.label2idx.get(labels[0], -1)
            else:
                label = -1
            if label >= 0:
                self.items.append((movie_key, clip_indices, label))
                self._movie_labels[movie_key] = label

        print(f"[MovieLevelClfDataset] {split} | {len(self.items)} movies "
              f"({sum(len(ci) for _, ci, _ in self.items)} clips) "
              f"| {self.num_classes} classes | pool={pool_mode}")

    def _get_clip_feature_summary(self, clip_idx: int) -> np.ndarray:
        """Get a fixed-size summary of one clip's trajectory."""
        item = self._clip_ds[clip_idx]
        feat = item["feat"].numpy()      # (max_len, D)
        seq_len = item["seq_len"]

        if self.augmentor is not None:
            feat, seq_len = self.augmentor(feat, seq_len, self._rng)

        valid = feat[:seq_len]
        if len(valid) == 0:
            return np.zeros(self.feat_dim, dtype=np.float32)
        return valid.mean(axis=0)  # (D,)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        movie_key, clip_indices, label = self.items[idx]

        # Collect per-clip summaries
        clip_summaries = []
        for ci in clip_indices:
            summary = self._get_clip_feature_summary(ci)
            clip_summaries.append(summary)

        clip_summaries = np.stack(clip_summaries)  # (N_clips, D)

        if self.pool_mode == "mean":
            feature = clip_summaries.mean(axis=0)  # (D,)
        elif self.pool_mode == "stats":
            # [mean, std, min, max] → (4D,)
            feature = np.concatenate([
                clip_summaries.mean(axis=0),
                clip_summaries.std(axis=0),
                clip_summaries.min(axis=0),
                clip_summaries.max(axis=0),
            ])
        else:
            raise ValueError(f"Unknown pool_mode: {self.pool_mode}")

        return {
            "feat": torch.from_numpy(feature.astype(np.float32)),
            "label": label,
            "movie_key": movie_key,
            "n_clips": len(clip_indices),
        }

    def get_class_weights_sampler(self) -> WeightedRandomSampler:
        """Create a WeightedRandomSampler for class-balanced training."""
        labels = [label for _, _, label in self.items]
        class_counts = Counter(labels)
        total = len(labels)
        # Weight = 1/count for each class
        weights = [total / (self.num_classes * class_counts[l]) for l in labels]
        return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)


def movie_collate_fn(batch: list) -> dict:
    feats = torch.stack([b["feat"] for b in batch])
    labels = torch.tensor([b["label"] for b in batch], dtype=torch.long)
    return {
        "feat": feats,
        "label": labels,
        "movie_key": [b["movie_key"] for b in batch],
        "n_clips": [b["n_clips"] for b in batch],
    }


# ---------------------------------------------------------------------------
# Augmented Clip-Level Dataset (wraps RealLabelClfDataset)
# ---------------------------------------------------------------------------

class AugmentedClfDataset(Dataset):
    """Wraps RealLabelClfDataset with trajectory augmentation + balanced sampling."""

    def __init__(self, base_dataset: RealLabelClfDataset, augment: bool = True, seed: int = 42):
        self.base = base_dataset
        self.augmentor = TrajectoryAugmentor() if augment else None
        self._rng = np.random.default_rng(seed + 2000)
        # Forward attributes
        self.label2idx = base_dataset.label2idx
        self.label_names = base_dataset.label_names
        self.num_classes = base_dataset.num_classes
        self.items = base_dataset.items
        self._clips = base_dataset._clips
        self._get_labels = base_dataset._get_labels

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        item = self.base[idx]
        if self.augmentor is not None and isinstance(item["label"], int) and item["label"] >= 0:
            feat = item["feat"].numpy()
            seq_len = item["seq_len"]
            feat, seq_len = self.augmentor(feat, seq_len, self._rng)
            item["feat"] = torch.from_numpy(feat)
            item["seq_len"] = seq_len
        return item

    def get_class_weights_sampler(self) -> WeightedRandomSampler:
        """Create sampler for class-balanced batches."""
        labels = []
        for ds_name, clip_id, clip_idx in self.base.items:
            lbl_list = self.base._get_labels(self.base._clips[clip_idx], self.base.label_type)
            label = self.base.label2idx.get(lbl_list[0], -1) if lbl_list else -1
            labels.append(label)
        class_counts = Counter(l for l in labels if l >= 0)
        total = sum(class_counts.values())
        weights = []
        for l in labels:
            if l >= 0:
                weights.append(total / (self.num_classes * class_counts[l]))
            else:
                weights.append(0.0)
        return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
