#!/bin/bash
#SBATCH --job-name=encode
#SBATCH --array=0
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=clustering_graphs/logs/encode_%A_%a.out
#SBATCH --error=clustering_graphs/logs/encode_%A_%a.err
#
# PHASE 1 — GPU encode on ARNES. VERIFIED on wn202: `--nv` gives cuda=True and
# nvidia-container-cli is present (unlike Vega's flaky nodes). So here we DO use
# the GPU — MiniLM-L6 over all 11 datasets encodes in ~15-30 min.
#
# Array index selects the model (0 = MiniLM-L6). Full grid: sbatch --array=0-4.

set -euo pipefail

MODELS=(
  sentence-transformers/all-MiniLM-L6-v2
  sentence-transformers/all-MiniLM-L12-v2
  sentence-transformers/all-mpnet-base-v2
  BAAI/bge-base-en-v1.5
  thenlper/gte-large
)
MODEL=${MODELS[$SLURM_ARRAY_TASK_ID]}
[ -z "$MODEL" ] && { echo "FATAL: array idx $SLURM_ARRAY_TASK_ID out of range" >&2; exit 1; }

WORKDIR=$HOME/clustering_graphs
SIF=$WORKDIR/container/clustgraphs.sif
mkdir -p "$WORKDIR/logs" "$WORKDIR/cache/hf" "$WORKDIR/cache/embs" "$WORKDIR/results/encode_receipts"

export APPTAINERENV_OMP_NUM_THREADS=8
export APPTAINERENV_OPENBLAS_NUM_THREADS=8
export APPTAINERENV_MKL_NUM_THREADS=8
export APPTAINERENV_HF_HOME=/cache/hf
export APPTAINERENV_HF_HUB_ENABLE_HF_TRANSFER=0

echo "[$(date +%T)] === PHASE 1 encode (GPU): ${MODEL} on $(hostname) ==="
nvidia-smi -L | head -2 || true

singularity exec --nv \
    --bind $WORKDIR:/workspace \
    --bind $WORKDIR/cache:/cache \
    "$SIF" \
    /opt/venv/bin/python /workspace/src/main.py \
        --models "$MODEL" \
        --datasets TwentyNewsgroupsClustering MedrxivClusteringS2S MedrxivClusteringP2P BiorxivClusteringS2S BiorxivClusteringP2P ArxivClusteringS2S ArxivClusteringP2P RedditClustering RedditClusteringP2P StackExchangeClustering StackExchangeClusteringP2P \
        --encode-only \
        --embed-cache /cache/embs \
        --output-dir /workspace/results/encode_receipts \
        --log-level INFO

echo "[$(date +%T)] === encode ${MODEL} complete (exit=$?) ==="
