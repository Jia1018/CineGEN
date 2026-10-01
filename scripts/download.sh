#!/usr/bin/env bash
# Download CineGen checkpoints + eval data from HuggingFace Hub.
#
# Usage:
#   ./scripts/download.sh            # downloads both ckpts + eval data
#   ./scripts/download.sh --ckpts    # only checkpoints (~1.4GB)
#   ./scripts/download.sh --data     # only eval data (~80MB)
#   ./scripts/download.sh --train    # CineScript train split (~185MB), not part of the default

set -euo pipefail

WHAT="all"
while [ $# -gt 0 ]; do
    case "$1" in
        --ckpts) WHAT="ckpts" ;;
        --data)  WHAT="data" ;;
        --train) WHAT="train" ;;
        --help|-h)
            grep '^#' "$0" | sed 's/^# \{0,1\}//'
            exit 0 ;;
        *) echo "Unknown arg: $1" >&2; exit 1 ;;
    esac
    shift
done

ROOT="$(cd "$(dirname "$0")/.."; pwd)"

if [[ "$WHAT" == "ckpts" || "$WHAT" == "all" ]]; then
    echo "=== Downloading checkpoints from Ziqi1018/CineGen-ckpts ==="
    python -c "
from huggingface_hub import snapshot_download
snapshot_download(repo_id='Ziqi1018/CineGen-ckpts',
                  local_dir='${ROOT}/checkpoints',
                  local_dir_use_symlinks=False)
"
    echo "Checkpoints → ${ROOT}/checkpoints/"
fi

if [[ "$WHAT" == "data" || "$WHAT" == "all" ]]; then
    echo "=== Downloading eval data from Ziqi1018/CineScript-eval ==="
    python -c "
from huggingface_hub import snapshot_download
snapshot_download(repo_id='Ziqi1018/CineScript-eval',
                  repo_type='dataset',
                  local_dir='${ROOT}/data/cinescript-eval',
                  local_dir_use_symlinks=False)
"
    echo "Eval data → ${ROOT}/data/cinescript-eval/"
fi

if [[ "$WHAT" == "train" ]]; then
    echo "=== Downloading train data from Ziqi1018/CineScript-train ==="
    python -c "
from huggingface_hub import snapshot_download
snapshot_download(repo_id='Ziqi1018/CineScript-train',
                  repo_type='dataset',
                  local_dir='${ROOT}/data/cinescript-train',
                  local_dir_use_symlinks=False)
"
    for a in matrices depth; do
        tar -xzf "${ROOT}/data/cinescript-train/${a}.tar.gz" -C "${ROOT}/data/cinescript-train"
        rm "${ROOT}/data/cinescript-train/${a}.tar.gz"
    done
    echo "Train data → ${ROOT}/data/cinescript-train/ (matrices/ and depth/ extracted)"
fi

echo "Done."
