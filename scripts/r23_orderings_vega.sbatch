#!/bin/bash
#SBATCH --job-name=orderings_cpu
#SBATCH --array=0-1%2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=1800M
#SBATCH --time=12:00:00
#SBATCH --partition=cpu
# 2 GB/core cap on Vega cpu nodes. MD with mst=off skips the dense MST distance
# matrix, so peak RSS is lower than the R2.3 sweep: 16 x 2000M = ~31 GB suffices.
#SBATCH --output=clustering_graphs/logs/orderings_%A_%a.out
#SBATCH --error=clustering_graphs/logs/orderings_%A_%a.err
#
# PHASE 2 — R1.2-b + R2.4 ordering sensitivity. CPU-ONLY; needs the embed
# cache warm (scripts/encode_vega.sbatch first, or chain via submit_all_vega.sh).
#
# Array index 0 = centroid ordering (tight-ball worst case, R1.2-b).
# Array index 1 = class ordering (single-class info-leak worst case).
# temporal ordering needs a --dates-file; run separately once prepared.

set -euo pipefail

WORKDIR=$HOME/clustering_graphs
SIF=$WORKDIR/container/clustgraphs.sif
IDX=$SLURM_ARRAY_TASK_ID

case "$IDX" in
  0) ORDER=centroid ;;
  1) ORDER=class ;;
  *) echo "unknown array idx $IDX" >&2; exit 1 ;;
esac

OUT=$WORKDIR/results/r12b_${ORDER}_vega
mkdir -p "$OUT" "$WORKDIR/logs"

export APPTAINERENV_OMP_NUM_THREADS=4
export APPTAINERENV_OPENBLAS_NUM_THREADS=4
export APPTAINERENV_MKL_NUM_THREADS=4
export APPTAINERENV_HF_HOME=/cache/hf
export APPTAINERENV_HF_HUB_ENABLE_HF_TRANSFER=0
export APPTAINER_NVCCLI=false
export SINGULARITY_NVCCLI=false
export APPTAINERENV_NVIDIA_VISIBLE_DEVICES=
export APPTAINERENV_CUDA_VISIBLE_DEVICES=

echo "[$(date +%T)] === PHASE 2 ordering '${ORDER}' on $(hostname) (CPU) ==="

singularity exec \
    --bind $WORKDIR:/workspace \
    --bind $WORKDIR/cache:/cache \
    "$SIF" \
    /opt/venv/bin/python /workspace/src/main.py \
        --models sentence-transformers/all-MiniLM-L6-v2 \
        --datasets TwentyNewsgroupsClustering MedrxivClusteringS2S MedrxivClusteringP2P BiorxivClusteringS2S BiorxivClusteringP2P StackExchangeClustering \
        --methods MD \
        --ks 1 2 3 6 \
        --seeds 0 1 2 3 4 5 6 7 8 9 \
        --ordering "$ORDER" \
        --mst off \
        --mst-mode paper-faithful \
        --output-dir /workspace/results/r12b_${ORDER}_vega \
        --embed-cache /cache/embs \
        --require-cache \
        --heavy-stats-max-nodes 0 \
        --log-level INFO

echo "[$(date +%T)] === ordering ${ORDER} complete ==="
