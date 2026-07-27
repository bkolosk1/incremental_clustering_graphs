#!/bin/bash
#SBATCH --job-name=encode
#SBATCH --array=0
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=1800M
#SBATCH --time=12:00:00
#SBATCH --partition=cpu
#SBATCH --output=clustering_graphs/logs/encode_%A_%a.out
#SBATCH --error=clustering_graphs/logs/encode_%A_%a.err
#
# VERIFIED on Vega cn0022: plain `singularity exec` (no --nv) runs the SIF fine
# on cpu nodes even though they lack nvidia-container-cli. The ONLY thing that
# triggered the earlier FATAL was `--nv`. So encode runs CPU here — reliable on
# any node, ~2-4 h for MiniLM-L6 across all 11 datasets. The transformer forward
# pass likes threads, hence OMP=16 (vs OMP=4 in the spectral phase).
#
# PHASE 1 — warm the embedding cache. CPU-ONLY.
#
# Why not GPU: Vega's GPU nodes are heterogeneous — several lack the
# `nvidia-container-cli` binary, and the site apptainer config force-enables
# nvccli mode that `--nvccli=false` does not reliably override (observed
# FATAL on jobs 39820844 and 39823728). MiniLM-L6 is a 22M-param model;
# encoding the whole benchmark on a 16-core CPU node takes ~1-2 h and NEVER
# touches the NVIDIA container stack. The rest of the pipeline is CPU-bound
# anyway, so the whole run is now GPU-free and node-agnostic.
#
# Encodes every (dataset, partition) for one model into $WORKDIR/cache/embs,
# then exits. Array index selects the model (default 0 = MiniLM-L6). Full grid:
#   sbatch --array=0-4 scripts/encode_vega.sbatch
#
# Unlike the spectral phase, the transformer forward pass DOES benefit from
# many BLAS/OMP threads, so encode uses OMP=16 (not the OMP=4 of phase 2).

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

# Transformer CPU inference likes threads (unlike the AMG solver in phase 2).
export APPTAINERENV_OMP_NUM_THREADS=16
export APPTAINERENV_OPENBLAS_NUM_THREADS=16
export APPTAINERENV_MKL_NUM_THREADS=16
export APPTAINERENV_HF_HOME=/cache/hf
export APPTAINERENV_HF_HUB_ENABLE_HF_TRANSFER=0

echo "[$(date +%T)] === PHASE 1 encode (CPU): ${MODEL} on $(hostname) ==="

singularity exec \
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
