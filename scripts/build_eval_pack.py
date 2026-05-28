"""
Build the CineGen eval pack from the (private) raw training data.

This is an internal script used to package the val-split eval data for upload to
HuggingFace Hub. Users do **not** need to run this — they should call
``scripts/download.sh`` to fetch the prebuilt pack.

Produces this layout::

    <out_dir>/
        index.jsonl                  # one entry per val clip
        matrices/<clip_id>.npz       # real 4×4 c2w GT
        depth/<clip_id>.npy          # 128-D depth-feature vectors (for attribute classifier)
        clip_movie_mapping.json      # raw labeled-clip table (subset, val clips only)
        held_out_splits.json         # per-(setting, traj_type) clip-id subsets for attr eval

Run::

    python scripts/build_eval_pack.py \\
        --raw_root /workspace/writeable/datasets/DIY_movies \\
        --out_dir data/cinescript-eval
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

# We import from the *original* internal code tree (kept outside this repo) only when
# this script is run; the file is intentionally not part of the public package's import
# graph. The path is configurable via --camgen_root.

ASPECT_KEYS = [
    "logline_script",
    "macro_type",
    "setting_class",
    "subject_composition",
    "genre_vibe",
]


def add_internal_paths(camgen_root: Path):
    sys.path.insert(0, str(camgen_root))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_root", required=True, help="Path to the raw DIY_movies tree")
    ap.add_argument("--camgen_root", required=True,
                    help="Path to the private CamGen repo (provides DiyMoviesDataset + "
                         "PAPER_SETTINGS + build_dataset)")
    ap.add_argument("--out_dir", required=True, help="Where to write the eval pack")
    ap.add_argument("--settings", nargs="+", default=None,
                    help="Subset of PAPER_SETTINGS to compute held-out splits for; "
                         "defaults to all settings shipped in the eval.")
    args = ap.parse_args()

    add_internal_paths(Path(args.camgen_root))
    from data.dataset import DiyMoviesDataset, AVAILABLE_DATASETS
    from evaluate.eval_attribute import ATTR_SETTINGS as PAPER_SETTINGS
    from evaluate._classifier_data import build_dataset

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "matrices").mkdir(exist_ok=True)
    (out / "depth").mkdir(exist_ok=True)

    # ── 1. Enumerate val clips ────────────────────────────────────────────
    ds = DiyMoviesDataset(
        root=args.raw_root, datasets=AVAILABLE_DATASETS, split="val",
        val_fraction=0.1, seed=42, traj_type="trajectory",
        load_rgb=False, load_depth=False, max_seq_len=300,
    )
    print(f"[1/4] {len(ds.items)} val clips")

    # ── 2. Gather per-clip data → index.jsonl + matrices/ + depth/ ──────
    index_entries = []
    val_cids = set()
    n_skipped = 0
    depth_root = Path(args.raw_root) / "clip_depth_features"
    for ds_name, clip_id in tqdm(ds.items, desc="copy matrices+depth"):
        pose_path = Path(args.raw_root) / "filtered_pose" / ds_name / f"{clip_id}.npz"
        if not pose_path.exists():
            n_skipped += 1
            continue
        shutil.copy(pose_path, out / "matrices" / f"{clip_id}.npz")
        # Depth (optional — zero-fallback if missing)
        depth_p = depth_root / ds_name / f"{clip_id}.npy"
        if depth_p.exists():
            shutil.copy(depth_p, out / "depth" / f"{clip_id}.npy")

        motion_caption = ds.motion_captions.get(clip_id, "")
        aspects = ds.cinematic_aspects.get(clip_id, {})

        entry = {
            "clip_id":        clip_id,
            "dataset":        ds_name,
            "motion_caption": motion_caption,
            **{k: aspects.get(k, "") for k in ASPECT_KEYS},
        }
        index_entries.append(entry)
        val_cids.add(clip_id)

    if n_skipped:
        print(f"[2/5] skipped {n_skipped} clips with missing matrices")

    with open(out / "index.jsonl", "w") as f:
        for e in index_entries:
            f.write(json.dumps(e) + "\n")
    print(f"[3/5] wrote {len(index_entries)} entries → index.jsonl")

    # ── 3. Subset clip_movie_mapping.json to val clips ──────────────────
    raw_mapping_p = Path(args.raw_root) / "labeling/known_movies/clip_movie_mapping.json"
    if raw_mapping_p.exists():
        with open(raw_mapping_p) as f:
            full_mapping = json.load(f)
        subset = [e for e in full_mapping
                  if e.get("filename", "").replace(".mp4", "") in val_cids]
        with open(out / "clip_movie_mapping.json", "w") as f:
            json.dump(subset, f)
        print(f"[4/5] subset clip_movie_mapping: {len(subset)} entries")
    else:
        print(f"[4/5] no clip_movie_mapping.json at {raw_mapping_p} — attribute eval will skip clips without labels")

    # ── 5. Compute held-out splits per setting × traj_type ──────────────
    settings = args.settings or PAPER_SETTINGS
    held_out: dict[str, dict[str, list[str]]] = {}
    for s in settings:
        held_out[s] = {}
        for tt in ("direction+speed", "trajectory"):
            try:
                _, val_ds, _, _ = build_dataset(s, tt, None)
                held_out[s][tt] = sorted({item[1] for item in val_ds.items})
            except Exception as e:
                print(f"  [warn] could not compute held-out for {s}/{tt}: {e}")
                held_out[s][tt] = []
    with open(out / "held_out_splits.json", "w") as f:
        json.dump(held_out, f)
    print(f"[5/5] wrote held-out splits for {len(settings)} settings")

    print(f"\nDone. Pack at {out.resolve()}")


if __name__ == "__main__":
    main()
