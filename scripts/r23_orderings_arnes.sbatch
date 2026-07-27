#!/bin/bash
#SBATCH --job-name=orderings
#SBATCH --array=0-1%2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=all
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=clustering_graphs/logs/orderings_%A_%a.out
#SBATCH --error=clustering_graphs/logs/orderings_%A_%a.err
#
# PHASE 2 — R1.2-b ordering sensitivity on ARNES. CPU (partition `all`), plain
# exec, --require-cache. Array 0=centroid, 1=class (temporal needs --dates-file).

set -euo pipefail

WORKDIR=$HOME/clustering_graphs
SIF=$WORKDIR/container/clustgraphs.sif
IDX=$SLURM_ARRAY_TASK_ID
case "$IDX" in
  0) ORDER=centroid ;;
  1) ORDER=class ;;
  *) echo "unknown array idx $IDX" >&2; exit 1 ;;
esac
mkdir -p "$WORKDIR/results/r12b_${ORDER}_arnes" "$WORKDIR/logs"

export APPTAINERENV_OMP_NUM_THREADS=4
export APPTAINERENV_OPENBLAS_NUM_THREADS=4
export APPTAINERENV_MKL_NUM_THREADS=4
export APPTAINERENV_HF_HOME=/cache/hf
export APPTAINERENV_HF_HUB_ENABLE_HF_TRANSFER=0

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
        --output-dir /workspace/results/r12b_${ORDER}_arnes \
        --embed-cache /cache/embs \
        --require-cache \
        --heavy-stats-max-nodes 0 \
        --log-level INFO

echo "[$(date +%T)] === ordering ${ORDER} complete ==="
