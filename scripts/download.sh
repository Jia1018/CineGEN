#!/usr/bin/env bash
# Download CineGen checkpoints + eval data from HuggingFace Hub.
#
# Usage:
#   ./scripts/download.sh            # downloads both ckpts + eval data
#   ./scripts/download.sh --ckpts    # only checkpoints (~1.5GB)
#   ./scripts/download.sh --data     # only eval data (~80MB)

set -euo pipefail

WHAT="all"
while [ $# -gt 0 ]; do
    case "$1" in
        --ckpts) WHAT="ckpts" ;;
        --data)  WHAT="data" ;;
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

echo "Done."
