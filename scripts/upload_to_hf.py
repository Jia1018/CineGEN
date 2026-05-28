"""
Upload CineGen checkpoints and/or eval data to HuggingFace Hub.

Two repos are created if missing:
- ``Ziqi1018/CineGen-ckpts`` — model checkpoints (cinegen, alignment, CLaTr, classifiers)
- ``Ziqi1018/CineScript-eval`` — eval data pack (dataset repo)

Usage::

    # Upload checkpoints
    python scripts/upload_to_hf.py ckpts \\
        --src checkpoints_to_upload/ \\
        --repo_id Ziqi1018/CineGen-ckpts

    # Upload eval data pack
    python scripts/upload_to_hf.py data \\
        --src data/cinescript-eval \\
        --repo_id Ziqi1018/CineScript-eval

Requires ``huggingface-cli login`` (or HF_TOKEN env var) with **write** permission
to the target repos.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import HfApi, create_repo


CKPT_LAYOUT_README = """\
# CineGen — Checkpoints

This repo bundles all checkpoints used by the CineGen public release.

| Path | Description | Size |
|---|---|---|
| `cinegen/best.pt` | Main CineGen model (no-AE, sep-encoded logline, first_pose, dirspd) | ~477MB |
| `align_dirspd_motion/best.pt` | Alignment encoder for F1 / FCD / Cov / AlnScore | ~515MB |
| `clatr_dirspd_motion/best.pt` | Independent CLaTr-style alignment encoder for CLaTr column | ~195MB |
| `clf_paper/<setting>/{direction_speed,trajectory}_best.pt` | Attribute classifiers | ~322MB total |

License: CC BY-NC 4.0. See the [GitHub repo](https://github.com/Jia1018/CineGEN) for usage instructions.
"""

DATA_LAYOUT_README = """\
# CineGen — Eval Pack

Val-split data used by the CineGen evaluation pipeline.

## Layout

```
index.jsonl                  # one entry per clip (clip_id, motion_caption, logline, aspects)
matrices/<clip_id>.npz       # real 4×4 c2w GT trajectories (key: "data")
depth/<clip_id>.npy          # 128-D depth features (used by attribute classifiers)
clip_movie_mapping.json      # subset of labeled clips for attribute eval
held_out_splits.json         # per-setting held-out clip_id lists
```

Each `index.jsonl` row::

    {
      "clip_id": "...",
      "dataset": "movieshots",
      "motion_caption": "camera dollies forward ...",
      "logline_script": "INT. PARK - DAY - ...",
      "macro_type": "Exterior (Open)",
      ...
    }

License: CC BY-NC 4.0. See the [GitHub repo](https://github.com/Jia1018/CineGEN) for evaluation instructions.

The data is for research use only and derived from the following source datasets:
- ShotBench, CineTechBench, VADB, MovieShots, CondensedMovies.
See each source's terms of use for their respective licenses.
"""


def upload(src: Path, repo_id: str, repo_type: str, readme_body: str):
    src = Path(src).resolve()
    if not src.exists():
        raise FileNotFoundError(f"Source not found: {src}")

    api = HfApi()
    print(f"Creating/ensuring repo {repo_id} ({repo_type})...")
    create_repo(repo_id, repo_type=repo_type, exist_ok=True)

    # Drop a README at the top of the upload (overwrites any existing one)
    readme_p = src / "README.md"
    if not readme_p.exists():
        readme_p.write_text(readme_body)

    print(f"Uploading {src} → {repo_id} ...")
    api.upload_folder(
        folder_path=str(src),
        repo_id=repo_id,
        repo_type=repo_type,
        commit_message="Initial release",
    )
    print(f"Done. https://huggingface.co/{'datasets/' if repo_type=='dataset' else ''}{repo_id}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_ck = sub.add_parser("ckpts", help="Upload model checkpoints (model repo)")
    p_ck.add_argument("--src", required=True, help="Local folder to upload")
    p_ck.add_argument("--repo_id", default="Ziqi1018/CineGen-ckpts")

    p_da = sub.add_parser("data", help="Upload eval data (dataset repo)")
    p_da.add_argument("--src", required=True, help="Local folder to upload")
    p_da.add_argument("--repo_id", default="Ziqi1018/CineScript-eval")

    args = ap.parse_args()
    if args.cmd == "ckpts":
        upload(Path(args.src), args.repo_id, repo_type="model", readme_body=CKPT_LAYOUT_README)
    elif args.cmd == "data":
        upload(Path(args.src), args.repo_id, repo_type="dataset", readme_body=DATA_LAYOUT_README)


if __name__ == "__main__":
    main()
