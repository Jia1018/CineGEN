#!/usr/bin/env bash
# Render one CineGen-generated clip via Blender.
#
# Pipeline:
#   1. Postprocess the NPZ (smooth + axis-canonicalise) → JSON.
#   2. Run Blender on the JSON → write PNG.
#
# Usage:
#     bash visualize/render_clip.sh <CLIP_ID> [GEN_DIR] [OUT_DIR]
#
# Environment variables:
#     BLENDER  — path to Blender 3.6.5 executable (required)
#     SIGMA    — Gaussian smoothing sigma (frames). Default: 3.0. Set 0 to disable.

set -euo pipefail

CLIP_ID="${1:?usage: $0 <CLIP_ID> [GEN_DIR] [OUT_DIR]}"
GEN_DIR="${2:-results/cinegen-generated}"
OUT_DIR="${3:-results/cinegen-renders/${CLIP_ID}}"
SIGMA="${SIGMA:-3.0}"

if [ -z "${BLENDER:-}" ]; then
    echo "ERROR: BLENDER env var must point to your Blender 3.6.5 executable." >&2
    echo "  e.g.  export BLENDER=/path/to/blender-3.6.5-linux-x64/blender" >&2
    exit 1
fi

mkdir -p "${OUT_DIR}"
JSON_PATH="${OUT_DIR}/${CLIP_ID}.json"
PNG_PATH="${OUT_DIR}/${CLIP_ID}.png"

# 1) Postprocess: NPZ → JSON (with smoothing)
python visualize/postprocess.py \
    --gen_dir "${GEN_DIR}" \
    --clip_id "${CLIP_ID}" \
    --out_json "${JSON_PATH}" \
    --smooth_sigma "${SIGMA}"

# 2) Blender render
"${BLENDER}" --background --python visualize/blender_render.py \
    -- --traj_p "${JSON_PATH}" --out_png "${PNG_PATH}"

echo
echo "Wrote ${PNG_PATH}"
