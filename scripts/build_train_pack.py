"""
Build the CineScript train pack from the (private) raw data tree.

Companion to ``build_eval_pack.py``: same layout and the same index fields, for the
train split instead of the val split, so code that reads the eval pack reads this too.
Users do **not** need to run this — download the prebuilt pack from HuggingFace Hub.

The split is the one used to train CineGEN: all clips of the five source datasets,
shuffled with seed 42, first 10% val, the rest train (``DiyMoviesDataset``).

Produces this layout::

    <out_dir>/
        index.jsonl                  # one entry per train clip
        matrices/<clip_id>.npz       # real 4x4 c2w camera trajectories (ViPE)
        depth/<clip_id>.npy          # 128-D depth-feature vectors, where available
        matrices.tar.gz, depth.tar.gz  # the two folders above, as uploaded
        clip_movie_mapping.json      # movie attributes for the metadata-linked train clips
        stats.json                   # counts reported on the dataset card

A Hugging Face directory holds at most 10,000 files, so ``matrices/`` and ``depth/`` are uploaded
as archives; the folders are kept locally.

Run::

    python scripts/build_train_pack.py \\
        --raw_root /workspace/writeable/datasets/DIY_movies \\
        --camgen_root /workspace/writeable/code/CamGen \\
        --out_dir /workspace/writeable/datasets/release/CineScript-train
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tarfile
from collections import Counter
from pathlib import Path

import numpy as np
from tqdm import tqdm

ASPECT_KEYS = [
    "logline_script",
    "macro_type",
    "setting_class",
    "subject_composition",
    "genre_vibe",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_root", required=True, help="Path to the raw DIY_movies tree")
    ap.add_argument("--camgen_root", required=True,
                    help="Path to the private CamGen repo (provides DiyMoviesDataset)")
    ap.add_argument("--out_dir", required=True, help="Where to write the train pack")
    args = ap.parse_args()

    sys.path.insert(0, str(Path(args.camgen_root)))
    from data.dataset import DiyMoviesDataset, AVAILABLE_DATASETS

    raw = Path(args.raw_root)
    out = Path(args.out_dir)
    (out / "matrices").mkdir(parents=True, exist_ok=True)
    (out / "depth").mkdir(exist_ok=True)

    # ── 1. Enumerate train clips (same arguments as train_pulp_mar.py) ───
    ds = DiyMoviesDataset(
        root=args.raw_root, datasets=AVAILABLE_DATASETS, split="train",
        val_fraction=0.1, seed=42, traj_type="trajectory",
        load_rgb=False, load_depth=False, max_seq_len=300,
    )
    print(f"[1/5] {len(ds.items)} train clips")

    # ── 2. index.jsonl + matrices/ + depth/ ──────────────────────────────
    entries, cids, lengths, per_source = [], set(), [], Counter()
    n_depth = 0
    for ds_name, clip_id in tqdm(ds.items, desc="copy matrices+depth"):
        pose_path = raw / "filtered_pose" / ds_name / f"{clip_id}.npz"
        if not pose_path.exists():
            continue
        shutil.copy(pose_path, out / "matrices" / f"{clip_id}.npz")
        lengths.append(int(np.load(pose_path)["data"].shape[0]))
        depth_p = raw / "clip_depth_features" / ds_name / f"{clip_id}.npy"
        if depth_p.exists():
            shutil.copy(depth_p, out / "depth" / f"{clip_id}.npy")
            n_depth += 1
        aspects = ds.cinematic_aspects.get(clip_id, {})
        entries.append({
            "clip_id": clip_id,
            "dataset": ds_name,
            "motion_caption": ds.motion_captions.get(clip_id, ""),
            **{k: aspects.get(k, "") for k in ASPECT_KEYS},
        })
        cids.add(clip_id)
        per_source[ds_name] += 1
    with open(out / "index.jsonl", "w") as f:
        for e in entries:
            f.write(json.dumps(e) + "\n")
    print(f"[2/5] wrote {len(entries)} entries → index.jsonl ({n_depth} with depth features)")

    # ── 3. Movie attributes for the metadata-linked train clips ──────────
    mapping_p = raw / "labeling/known_movies/clip_movie_mapping.json"
    linked = []
    if mapping_p.exists():
        full = json.load(open(mapping_p))
        linked = [e for e in full if e.get("filename", "").replace(".mp4", "") in cids]
        json.dump(linked, open(out / "clip_movie_mapping.json", "w"))
    print(f"[3/5] clip_movie_mapping: {len(linked)} linked train clips")

    # ── 4. Statistics for the dataset card ───────────────────────────────
    L = np.array(lengths)
    films = {e.get("imdb_id") or e.get("title") or e.get("movie") for e in linked} - {None, ""}
    stats = {
        "clips": len(entries),
        "per_source": dict(per_source.most_common()),
        "pose_frames_total": int(L.sum()),
        "frames_per_clip": {"mean": round(float(L.mean()), 1), "median": int(np.median(L)),
                            "min": int(L.min()), "max": int(L.max())},
        "clips_with_depth_features": n_depth,
        "metadata_linked_clips": len(linked),
        "metadata_linked_films": len(films),
    }
    json.dump(stats, open(out / "stats.json", "w"), indent=2)
    print(f"[4/5] stats: {json.dumps(stats)}")

    # ── 5. Archives for upload (HF caps a directory at 10,000 files) ─────
    for name in ("matrices", "depth"):
        files = sorted((out / name).glob("*"))
        with tarfile.open(out / f"{name}.tar.gz", "w:gz") as tf:
            for p in files:
                tf.add(p, arcname=f"{name}/{p.name}")
        print(f"[5/5] {name}.tar.gz: {len(files)} files")
    print(f"\nDone. Pack at {out.resolve()}")


if __name__ == "__main__":
    main()
