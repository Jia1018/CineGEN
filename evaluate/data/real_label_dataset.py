"""
Dataset for classification using ground-truth movie labels from Wikidata/IMDb.

Key differences from ClfDataset:
  - Labels come from clip_movie_mapping.json (real labels), not VLM captions
  - Train/val split is done **by movie**, not by clip, to prevent leakage
  - Supports both single-label and multi-label (genre) classification
  - Label types: genre, era (binned year), country_region, director

Genre mapping:
  Wikidata provides fine-grained multi-label genres (e.g. "crime drama film").
  We map these to ~10 coarse groups for classification.
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import Counter, defaultdict

import torch
from torch.utils.data import Dataset

from cinegen.utils.pose_utils import np_matrices_to_velocity


LOG_SPEED_EPS = 1e-6

TRAJ_DIM: Dict[str, int] = {
    "trajectory":      9,
    "velocity":        6,
    "direction":       6,
    "speed":           2,
    "direction+speed": 8,
}

VALID_TRAJ_TYPES = frozenset(TRAJ_DIM.keys())

VALID_LABEL_TYPES = frozenset([
    "genre",           # multi-label coarse genre
    "genre_primary",   # single-label: primary coarse genre
    "era",             # binned year
    "country_region",  # grouped country
    "director",        # top-N directors
])

# ---------------------------------------------------------------------------
# Genre mapping: Wikidata fine → coarse
# ---------------------------------------------------------------------------

GENRE_COARSE_MAP = {
    # Drama
    "drama film": "Drama", "comedy drama": "Drama", "crime drama film": "Drama",
    "musical drama film": "Drama", "tragicomedy": "Drama",
    "psychological drama film": "Drama", "historical drama": "Drama",
    "historical drama film": "Drama", "melodrama": "Drama",
    "legal drama": "Drama", "political drama film": "Drama",
    "family drama film": "Drama",
    # Action
    "action film": "Action", "action thriller film": "Action",
    "martial arts film": "Action", "action comedy film": "Action",
    "action/adventure film": "Action", "chase film": "Action",
    "wuxia film": "Action",
    # Comedy
    "comedy film": "Comedy", "romantic comedy": "Comedy",
    "black comedy film": "Comedy", "slapstick film": "Comedy",
    "romantic comedy film": "Comedy", "parody film": "Comedy",
    "screwball comedy film": "Comedy", "satire film": "Comedy",
    "stoner film": "Comedy",
    # Thriller/Mystery
    "thriller film": "Thriller", "crime thriller film": "Thriller",
    "psychological thriller film": "Thriller", "suspense film": "Thriller",
    "neo-noir": "Thriller", "mystery film": "Thriller",
    "crime film": "Thriller", "heist film": "Thriller",
    "gangster film": "Thriller", "police procedural film": "Thriller",
    "spy film": "Thriller", "whodunit": "Thriller",
    "film noir": "Thriller", "political thriller film": "Thriller",
    "detective film": "Thriller",
    # Sci-Fi / Fantasy
    "science fiction film": "Sci-Fi/Fantasy", "fantasy film": "Sci-Fi/Fantasy",
    "dystopian film": "Sci-Fi/Fantasy", "superhero film": "Sci-Fi/Fantasy",
    "post-apocalyptic film": "Sci-Fi/Fantasy",
    "monster film": "Sci-Fi/Fantasy", "kaiju film": "Sci-Fi/Fantasy",
    # Horror
    "horror film": "Horror", "slasher film": "Horror",
    "zombie film": "Horror", "comedy horror film": "Horror",
    "psychological horror film": "Horror", "supernatural horror film": "Horror",
    "creature film": "Horror", "body horror film": "Horror",
    # Romance
    "romance film": "Romance", "romantic drama film": "Romance",
    # War / Historical
    "war film": "War/Historical", "epic film": "War/Historical",
    "historical film": "War/Historical", "period film": "War/Historical",
    "costume drama film": "War/Historical", "sword and sandal": "War/Historical",
    "samurai film": "War/Historical",
    # Documentary
    "documentary film": "Documentary",
    # Adventure
    "adventure film": "Adventure", "survival film": "Adventure",
    "road movie": "Adventure", "Western film": "Adventure",
    # Biographical
    "biographical film": "Biographical", "biographical drama film": "Biographical",
    # Other (catch-all for structural/meta genres)
    "coming-of-age film": "Drama", "teen film": "Drama",
    "sports film": "Drama", "American football film": "Drama",
    "musical film": "Drama", "animated film": "Other",
    "independent film": "Other", "film based on a novel": "Other",
    "film based on book": "Other", "film based on literature": "Other",
    "LGBTQ-related film": "Other", "flashback film": "Other",
    "ensemble film": "Other", "buddy film": "Other",
    "Christmas film": "Other",
}

COARSE_GENRES = sorted(set(GENRE_COARSE_MAP.values()) - {"Other"})
# => ['Action', 'Adventure', 'Biographical', 'Comedy', 'Documentary', 'Drama',
#     'Horror', 'Romance', 'Sci-Fi/Fantasy', 'Thriller', 'War/Historical']

# ---------------------------------------------------------------------------
# Era binning
# ---------------------------------------------------------------------------

def year_to_era(year: int) -> str:
    if year < 1980:
        return "Classic (<1980)"
    elif year < 2000:
        return "New Hollywood (1980-99)"
    elif year < 2010:
        return "2000s"
    elif year < 2020:
        return "2010s"
    else:
        return "2020s+"

ERA_CLASSES = [
    "Classic (<1980)", "New Hollywood (1980-99)", "2000s", "2010s", "2020s+"
]

# ---------------------------------------------------------------------------
# Country grouping
# ---------------------------------------------------------------------------

COUNTRY_REGION_MAP = {
    "United States": "Hollywood",
    "American": "Hollywood",
    "United Kingdom": "European",
    "France": "European",
    "Germany": "European",
    "Italy": "European",
    "Spain": "European",
    "Austria": "European",
    "Hungary": "European",
    "Sweden": "European",
    "Denmark": "European",
    "Norway": "European",
    "Ireland": "European",
    "Netherlands": "European",
    "Belgium": "European",
    "Czech Republic": "European",
    "Poland": "European",
    "Romania": "European",
    "Switzerland": "European",
    "Greece": "European",
    "Finland": "European",
    "Portugal": "European",
    "People's Republic of China": "East Asian",
    "China": "East Asian",
    "Hong Kong": "East Asian",
    "Japan": "East Asian",
    "South Korea": "East Asian",
    "Taiwan": "East Asian",
    "India": "South Asian",
    "Brazil": "Latin American",
    "Argentina": "Latin American",
    "Mexico": "Latin American",
    "Colombia": "Latin American",
    "Canada": "Hollywood",  # typically Hollywood co-productions
    "Australia": "Hollywood",
    "New Zealand": "Hollywood",
}

REGION_CLASSES = [
    "Hollywood", "European", "East Asian", "South Asian", "Latin American", "Other"
]


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class RealLabelClfDataset(Dataset):
    """
    Args:
        root:            Path to DIY_movies root directory
        mapping_path:    Path to clip_movie_mapping.json
        split:           'train' or 'val'
        val_fraction:    Fraction of **movies** held out for validation
        max_seq_len:     Maximum sequence length
        traj_type:       Trajectory representation type
        label_type:      Which real label to use for classification
        seed:            Random seed for movie-level split
        label2idx:       Pre-built label→int mapping (for val split consistency)
        top_n_directors: Number of top directors to keep (rest → "Other")
        multi_label:     If True and label_type="genre", returns multi-hot vector
    """

    def __init__(
        self,
        root:            str,
        mapping_path:    str,
        split:           str               = "train",
        val_fraction:    float             = 0.15,
        max_seq_len:     int               = 300,
        traj_type:       str               = "direction+speed",
        label_type:      str               = "genre",
        seed:            int               = 42,
        label2idx:       Optional[dict]    = None,
        top_n_directors: int               = 20,
        multi_label:     bool              = False,
        split_mode:      str               = "movie",  # "movie" or "clip"
    ):
        assert traj_type in VALID_TRAJ_TYPES
        assert label_type in VALID_LABEL_TYPES

        self.root        = Path(root)
        self.traj_type   = traj_type
        self.label_type  = label_type
        self.max_seq_len = max_seq_len
        self.max_len     = max_seq_len if traj_type == "trajectory" else max_seq_len - 1
        self.feat_dim    = TRAJ_DIM[traj_type]
        self.multi_label = multi_label and label_type == "genre"

        # Load clip-movie mapping
        with open(mapping_path) as f:
            all_clips = json.load(f)

        # Filter to clips that have both pose data and the needed label
        self._clips = []
        for c in all_clips:
            fname = c["filename"]
            ds = c["dataset"]
            pose_path = self.root / "filtered_pose" / ds / (Path(fname).stem + ".npz")
            if not pose_path.exists():
                continue
            info = c.get("movie_info", {})
            if not self._has_label(info, label_type):
                continue
            self._clips.append(c)

        # Group by movie (for movie-level split)
        movie_to_clips = defaultdict(list)
        for i, c in enumerate(self._clips):
            movie_key = c.get("movie_info", {}).get("imdb_id") or c.get("movie_name", f"unk_{i}")
            movie_to_clips[movie_key].append(i)

        # ─── CLIP-LEVEL SPLIT BRANCH ────────────────────────────────────
        # Each clip is independently assigned to train/val (stratified by label).
        # Pro: every class always present in val
        # Con: clips from the same movie can leak across train/val
        if split_mode == "clip":
            clip_labels = {}
            for i, c in enumerate(self._clips):
                ll = self._get_labels_raw(c, label_type)
                clip_labels[i] = ll[0] if ll else "__none__"

            # Group clips by label
            label_to_clip_indices = defaultdict(list)
            for i, lbl in clip_labels.items():
                label_to_clip_indices[lbl].append(i)

            rng = np.random.default_rng(seed)
            train_clip_indices = []
            val_clip_indices = []

            for lbl, indices in label_to_clip_indices.items():
                shuffled = list(indices)
                rng.shuffle(shuffled)
                n_val = int(len(shuffled) * val_fraction)
                if len(shuffled) >= 2 and n_val == 0:
                    n_val = 1
                val_clip_indices.extend(shuffled[:n_val])
                train_clip_indices.extend(shuffled[n_val:])

            split_indices = sorted(val_clip_indices if split == "val" else train_clip_indices)

            self.items = [(self._clips[i]["dataset"], Path(self._clips[i]["filename"]).stem, i)
                          for i in split_indices]

            # Build label vocabulary
            if label_type == "director":
                self._build_director_vocab(top_n_directors, split_indices)

            if label2idx is not None:
                self.label2idx = label2idx
            else:
                self.label2idx = self._build_label2idx(split_indices, label_type, top_n_directors)

            self.label_names = sorted(self.label2idx.keys(), key=lambda x: self.label2idx[x])
            self.num_classes = len(self.label2idx)

            print(
                f"[RealLabelClfDataset] {split} | traj={traj_type} label={label_type} "
                f"| {len(self.items)} clips | {self.num_classes} classes (CLIP-level split)"
            )
            return  # Skip movie-level split logic below

        # ─── MOVIE-LEVEL SPLIT (original) ───────────────────────────────
        # Stratified movie-level split balanced by CLIP COUNT:
        # 1. Assign each movie a primary label
        # 2. Group movies by label
        # 3. Within each label group, fill val with movies until we reach
        #    ~val_fraction of that group's total clips (not movie count)
        movie_primary_label = {}
        for mk, clip_indices in movie_to_clips.items():
            labels = self._get_labels_raw(self._clips[clip_indices[0]], label_type)
            movie_primary_label[mk] = labels[0] if labels else "__none__"

        label_to_movies = defaultdict(list)
        for mk, lbl in movie_primary_label.items():
            label_to_movies[lbl].append(mk)

        rng = np.random.default_rng(seed)
        train_movie_keys = set()
        val_movie_keys = set()

        for lbl, movies in label_to_movies.items():
            movies_shuffled = list(movies)
            rng.shuffle(movies_shuffled)
            # Target: val gets val_fraction of this label's total clips
            # Guarantee: if a class has 2+ movies, at least 1 movie goes to val
            #            if a class has only 1 movie, it goes to train (can't split)
            total_clips = sum(len(movie_to_clips[mk]) for mk in movies_shuffled)
            target_val_clips = int(total_clips * val_fraction)
            if len(movies_shuffled) >= 2 and target_val_clips == 0:
                target_val_clips = 1  # ensure at least 1 movie goes to val
            val_clips_so_far = 0
            for mk in movies_shuffled:
                n_clips = len(movie_to_clips[mk])
                if val_clips_so_far < target_val_clips:
                    val_movie_keys.add(mk)
                    val_clips_so_far += n_clips
                else:
                    train_movie_keys.add(mk)

        if split == "val":
            split_movie_keys = val_movie_keys
        else:
            split_movie_keys = train_movie_keys

        split_indices = []
        for mk in split_movie_keys:
            split_indices.extend(movie_to_clips[mk])
        split_indices.sort()

        self.items = [(self._clips[i]["dataset"], Path(self._clips[i]["filename"]).stem, i)
                      for i in split_indices]

        # Build label vocabulary
        if label_type == "director":
            self._build_director_vocab(top_n_directors, split_indices)

        if label2idx is not None:
            self.label2idx = label2idx
        else:
            self.label2idx = self._build_label2idx(split_indices, label_type, top_n_directors)

        self.label_names = sorted(self.label2idx.keys(), key=lambda x: self.label2idx[x])
        self.num_classes = len(self.label2idx)

        # Stats
        n_movies_split = len(split_movie_keys)
        print(
            f"[RealLabelClfDataset] {split} | traj={traj_type} label={label_type} "
            f"| {len(self.items)} clips from {n_movies_split} movies "
            f"| {self.num_classes} classes"
            f"{' (multi-label)' if self.multi_label else ''}"
        )

    def _get_labels_raw(self, clip_info: dict, label_type: str) -> list:
        """Get labels without needing label2idx (used during split construction)."""
        info = clip_info.get("movie_info", {})
        if label_type in ("genre", "genre_primary"):
            coarse = []
            for g in info.get("genres", []):
                cg = GENRE_COARSE_MAP.get(g)
                if cg and cg != "Other" and cg not in coarse:
                    coarse.append(cg)
            return coarse[:1] if label_type == "genre_primary" else coarse
        elif label_type == "era":
            y = info.get("year")
            return [year_to_era(y)] if y else []
        elif label_type == "country_region":
            regions = []
            for co in info.get("countries", []):
                r = COUNTRY_REGION_MAP.get(co, "Other")
                if r not in regions:
                    regions.append(r)
            return regions[:1]
        elif label_type == "director":
            return info.get("directors", [])[:1] or ["Other"]
        return []

    def _has_label(self, info: dict, label_type: str) -> bool:
        if label_type in ("genre", "genre_primary"):
            return bool(info.get("genres"))
        elif label_type == "era":
            return info.get("year") is not None
        elif label_type == "country_region":
            return bool(info.get("countries"))
        elif label_type == "director":
            return bool(info.get("directors"))
        return False

    def _get_labels(self, clip_info: dict, label_type: str) -> list:
        """Returns list of label strings for a clip."""
        info = clip_info.get("movie_info", {})
        if label_type == "genre" or label_type == "genre_primary":
            coarse = []
            for g in info.get("genres", []):
                cg = GENRE_COARSE_MAP.get(g)
                if cg and cg != "Other" and cg not in coarse:
                    coarse.append(cg)
            if label_type == "genre_primary":
                return coarse[:1] if coarse else []
            return coarse
        elif label_type == "era":
            y = info.get("year")
            return [year_to_era(y)] if y else []
        elif label_type == "country_region":
            regions = []
            for co in info.get("countries", []):
                r = COUNTRY_REGION_MAP.get(co, "Other")
                if r not in regions:
                    regions.append(r)
            return regions[:1]  # primary region
        elif label_type == "director":
            dirs = info.get("directors", [])
            mapped = []
            for d in dirs:
                if d in self._top_directors:
                    mapped.append(d)
            return mapped[:1] if mapped else ["Other"]
        return []

    def _build_director_vocab(self, top_n: int, indices: list):
        """Find top-N directors by clip count in the given split."""
        dir_counter = Counter()
        for i in indices:
            c = self._clips[i]
            for d in c.get("movie_info", {}).get("directors", []):
                dir_counter[d] += 1
        self._top_directors = {d for d, _ in dir_counter.most_common(top_n)}

    def _build_label2idx(self, indices: list, label_type: str, top_n_directors: int) -> dict:
        if label_type == "genre":
            return {g: i for i, g in enumerate(COARSE_GENRES)}
        elif label_type == "genre_primary":
            return {g: i for i, g in enumerate(COARSE_GENRES)}
        elif label_type == "era":
            return {e: i for i, e in enumerate(ERA_CLASSES)}
        elif label_type == "country_region":
            return {r: i for i, r in enumerate(REGION_CLASSES)}
        elif label_type == "director":
            all_labels = sorted(self._top_directors) + ["Other"]
            return {d: i for i, d in enumerate(all_labels)}
        return {}

    def _load_feat(self, ds: str, clip_id: str) -> Tuple[np.ndarray, int]:
        path = self.root / "filtered_pose" / ds / f"{clip_id}.npz"
        matrices = np.load(path)["data"]  # (N, 4, 4)

        if self.traj_type == "trajectory":
            R = matrices[:, :3, :3]
            rot6d = R[:, :, :2].transpose(0, 2, 1).reshape(-1, 6)
            trans = matrices[:, :3, 3]
            rel_trans = trans - trans[0:1]
            raw = np.concatenate([rot6d, rel_trans], axis=-1)
            actual_len = min(len(raw), self.max_len)
            raw = raw[:actual_len]
        else:
            td, rd, ts, rs, _ = np_matrices_to_velocity(matrices)
            actual_len = min(len(td), self.max_len)
            if self.traj_type == "velocity":
                tv = td[:actual_len] * ts[:actual_len, None]
                rv = rd[:actual_len] * rs[:actual_len, None]
                raw = np.concatenate([tv, rv], axis=-1)
            elif self.traj_type == "direction":
                raw = np.concatenate([td[:actual_len], rd[:actual_len]], axis=-1)
            elif self.traj_type == "speed":
                ts_log = np.log(ts[:actual_len] + LOG_SPEED_EPS)
                rs_log = np.log(rs[:actual_len] + LOG_SPEED_EPS)
                raw = np.stack([ts_log, rs_log], axis=-1)
            else:  # direction+speed
                ts_log = np.log(ts[:actual_len] + LOG_SPEED_EPS)
                rs_log = np.log(rs[:actual_len] + LOG_SPEED_EPS)
                raw = np.concatenate([
                    td[:actual_len], rd[:actual_len],
                    ts_log[:, None], rs_log[:, None],
                ], axis=-1)

        pad_len = self.max_len - actual_len
        feat = np.concatenate(
            [raw, np.zeros((pad_len, self.feat_dim), dtype=np.float32)], axis=0
        )
        return feat.astype(np.float32), actual_len

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict:
        ds, clip_id, clip_idx = self.items[idx]
        feat, seq_len = self._load_feat(ds, clip_id)
        labels = self._get_labels(self._clips[clip_idx], self.label_type)

        if self.multi_label:
            # Multi-hot vector
            label_vec = torch.zeros(self.num_classes, dtype=torch.float32)
            for lbl in labels:
                if lbl in self.label2idx:
                    label_vec[self.label2idx[lbl]] = 1.0
            return {
                "feat": torch.from_numpy(feat),
                "seq_len": seq_len,
                "label": label_vec,
                "clip_id": clip_id,
                "dataset": ds,
            }
        else:
            # Single label (first match)
            label = -1
            if labels:
                label = self.label2idx.get(labels[0], -1)
            return {
                "feat": torch.from_numpy(feat),
                "seq_len": seq_len,
                "label": label,
                "clip_id": clip_id,
                "dataset": ds,
            }


VLM_TEXT_TYPES = frozenset([
    "macro_type", "setting_class", "subject_composition", "genre_vibe"
])


class VLMBaselineClfDataset(Dataset):
    """
    VLM pseudo-label classifier on the SAME subset of clips as RealLabelClfDataset.

    Uses VLM captions (macro_type, genre_vibe, etc.) but restricted to only clips
    in clip_movie_mapping.json, with the same movie-level train/val split.
    This enables fair comparison: same clips, same split, different label source.
    """

    def __init__(
        self,
        root:            str,
        mapping_path:    str,
        split:           str               = "train",
        val_fraction:    float             = 0.15,
        max_seq_len:     int               = 300,
        traj_type:       str               = "direction+speed",
        text_type:       str               = "genre_vibe",
        seed:            int               = 42,
        label2idx:       Optional[dict]    = None,
    ):
        assert traj_type in VALID_TRAJ_TYPES
        assert text_type in VLM_TEXT_TYPES

        self.root        = Path(root)
        self.traj_type   = traj_type
        self.text_type   = text_type
        self.max_seq_len = max_seq_len
        self.max_len     = max_seq_len if traj_type == "trajectory" else max_seq_len - 1
        self.feat_dim    = TRAJ_DIM[traj_type]

        # Load clip-movie mapping (same subset as real-label)
        with open(mapping_path) as f:
            all_clips = json.load(f)

        # Load VLM captions
        vlm_captions = {}
        for ds_name in ["cinetechbench", "movieshots", "condensedmovies",
                        "shotbench", "vadb"]:
            jsonl = self.root / "captions" / f"{ds_name}_captions.jsonl"
            if jsonl.exists():
                with open(jsonl) as f:
                    for line in f:
                        entry = json.loads(line)
                        clip_id = Path(entry["video_path"]).stem
                        ctx = entry.get("cinematic_data", {}).get("spatial_context", {})
                        vlm_captions[clip_id] = {
                            "macro_type": ctx.get("macro_type", ""),
                            "setting_class": ctx.get("setting_class", ""),
                            "subject_composition": entry.get("cinematic_data", {}).get(
                                "subject_composition", ""),
                            "genre_vibe": entry.get("cinematic_data", {}).get(
                                "genre_vibe", ""),
                        }

        # Filter to clips with pose + VLM caption
        self._clips = []
        self._vlm_labels = {}
        for c in all_clips:
            fname = c["filename"]
            ds = c["dataset"]
            clip_id = Path(fname).stem
            pose_path = self.root / "filtered_pose" / ds / f"{clip_id}.npz"
            if not pose_path.exists():
                continue
            if clip_id not in vlm_captions:
                continue
            label = vlm_captions[clip_id].get(text_type, "")
            if not label:
                continue
            self._clips.append(c)
            self._vlm_labels[clip_id] = label

        # Stratified movie-level split (same logic as RealLabelClfDataset)
        movie_to_clips = defaultdict(list)
        for i, c in enumerate(self._clips):
            movie_key = c.get("movie_info", {}).get("imdb_id") or c.get(
                "movie_name", f"unk_{i}")
            movie_to_clips[movie_key].append(i)

        # Stratified split balanced by clip count (same logic as RealLabel)
        label_to_movies = defaultdict(list)
        for mk, clip_indices in movie_to_clips.items():
            clip_id = Path(self._clips[clip_indices[0]]["filename"]).stem
            lbl = self._vlm_labels.get(clip_id, "__none__")
            label_to_movies[lbl].append(mk)

        rng = np.random.default_rng(seed)
        train_movie_keys = set()
        val_movie_keys = set()

        for lbl, movies in label_to_movies.items():
            movies_shuffled = list(movies)
            rng.shuffle(movies_shuffled)
            total_clips = sum(len(movie_to_clips[mk]) for mk in movies_shuffled)
            target_val_clips = int(total_clips * val_fraction)
            val_clips_so_far = 0
            for mk in movies_shuffled:
                n_clips = len(movie_to_clips[mk])
                if val_clips_so_far < target_val_clips:
                    val_movie_keys.add(mk)
                    val_clips_so_far += n_clips
                else:
                    train_movie_keys.add(mk)

        if split == "val":
            split_movie_keys = val_movie_keys
        else:
            split_movie_keys = train_movie_keys

        split_indices = []
        for mk in split_movie_keys:
            split_indices.extend(movie_to_clips[mk])
        split_indices.sort()

        self.items = [(self._clips[i]["dataset"], Path(self._clips[i]["filename"]).stem)
                      for i in split_indices]

        # Build label vocab
        if label2idx is not None:
            self.label2idx = label2idx
        else:
            all_labels = sorted(set(
                self._vlm_labels[clip_id]
                for _, clip_id in self.items
                if self._vlm_labels.get(clip_id)
            ))
            self.label2idx = {lbl: i for i, lbl in enumerate(all_labels)}

        self.label_names = sorted(self.label2idx.keys(), key=lambda x: self.label2idx[x])
        self.num_classes = len(self.label2idx)

        print(
            f"[VLMBaselineClfDataset] {split} | traj={traj_type} text={text_type} "
            f"| {len(self.items)} clips from {len(split_movie_keys)} movies "
            f"| {self.num_classes} classes"
        )

    def _load_feat(self, ds: str, clip_id: str):
        path = self.root / "filtered_pose" / ds / f"{clip_id}.npz"
        matrices = np.load(path)["data"]

        if self.traj_type == "trajectory":
            R = matrices[:, :3, :3]
            rot6d = R[:, :, :2].transpose(0, 2, 1).reshape(-1, 6)
            trans = matrices[:, :3, 3]
            rel_trans = trans - trans[0:1]
            raw = np.concatenate([rot6d, rel_trans], axis=-1)
            actual_len = min(len(raw), self.max_len)
            raw = raw[:actual_len]
        else:
            td, rd, ts, rs, _ = np_matrices_to_velocity(matrices)
            actual_len = min(len(td), self.max_len)
            if self.traj_type == "velocity":
                tv = td[:actual_len] * ts[:actual_len, None]
                rv = rd[:actual_len] * rs[:actual_len, None]
                raw = np.concatenate([tv, rv], axis=-1)
            elif self.traj_type == "direction":
                raw = np.concatenate([td[:actual_len], rd[:actual_len]], axis=-1)
            elif self.traj_type == "speed":
                ts_log = np.log(ts[:actual_len] + LOG_SPEED_EPS)
                rs_log = np.log(rs[:actual_len] + LOG_SPEED_EPS)
                raw = np.stack([ts_log, rs_log], axis=-1)
            else:
                ts_log = np.log(ts[:actual_len] + LOG_SPEED_EPS)
                rs_log = np.log(rs[:actual_len] + LOG_SPEED_EPS)
                raw = np.concatenate([
                    td[:actual_len], rd[:actual_len],
                    ts_log[:, None], rs_log[:, None],
                ], axis=-1)

        pad_len = self.max_len - actual_len
        feat = np.concatenate(
            [raw, np.zeros((pad_len, self.feat_dim), dtype=np.float32)], axis=0
        )
        return feat.astype(np.float32), actual_len

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        ds, clip_id = self.items[idx]
        feat, seq_len = self._load_feat(ds, clip_id)
        label_str = self._vlm_labels.get(clip_id, "")
        label = self.label2idx.get(label_str, -1)
        return {
            "feat": torch.from_numpy(feat),
            "seq_len": seq_len,
            "label": label,
            "clip_id": clip_id,
            "dataset": ds,
        }


def collate_fn(batch: list) -> dict:
    feats = torch.stack([b["feat"] for b in batch])
    seq_lens = torch.tensor([b["seq_len"] for b in batch], dtype=torch.long)
    if isinstance(batch[0]["label"], torch.Tensor):
        labels = torch.stack([b["label"] for b in batch])
    else:
        labels = torch.tensor([b["label"] for b in batch], dtype=torch.long)
    return {
        "feat": feats,
        "seq_len": seq_lens,
        "label": labels,
        "clip_id": [b["clip_id"] for b in batch],
        "dataset": [b["dataset"] for b in batch],
    }
