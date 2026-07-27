#!/bin/bash
# Wrapper for `singularity exec` with standard bind mounts.
# The SIF only holds the venv -- code lives on the host at /workspace, so
# edits to src/*.py take effect immediately on the next run.
#
# Usage:
#   container/run.sh src/main.py --help
#   USE_GPU=1 container/run.sh src/encode.py --model all-MiniLM-L12-v2 --task TwentyNewsgroupsClustering
#   container/run.sh -c 'import mteb; print(mteb.__version__)'
#
# Overrides via env:
#   SIF=/other/path/clustgraphs.sif
#   WORK=/other/code/root          # bound to /workspace
#   CACHE=/other/cache/root        # bound to /cache (HF_HOME lives here)
#   USE_GPU=1                      # adds --nv
#   OMP_NUM_THREADS=8              # default 4
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
SIF="${SIF:-$HERE/clustgraphs.sif}"
WORK="${WORK:-$(dirname "$HERE")}"
CACHE="${CACHE:-$HOME/clustering_graphs/cache}"
mkdir -p "$CACHE/hf" "$CACHE/mteb" "$CACHE/embs"

if [[ ! -f "$SIF" ]]; then
    echo "[run.sh] SIF not found: $SIF" >&2
    echo "[run.sh] build first: bash $HERE/build.sh" >&2
    exit 1
fi

NV=""
[[ "${USE_GPU:-0}" == "1" ]] && NV="--nv"

exec singularity exec $NV \
    --bind "$WORK:/workspace" \
    --bind "$CACHE:/cache" \
    --env HF_HOME=/cache/hf \
    --env HUGGINGFACE_HUB_CACHE=/cache/hf/hub \
    --env TOKENIZERS_PARALLELISM=false \
    --env PYTHONPATH=/workspace/src \
    --env OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}" \
    --env OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}" \
    --env MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}" \
    "$SIF" /opt/venv/bin/python "$@"
