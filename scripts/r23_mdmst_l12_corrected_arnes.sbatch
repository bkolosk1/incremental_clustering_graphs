#!/bin/bash
#SBATCH --job-name=mdmst
#SBATCH --array=0-15%16
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=all
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --time=1-00:00:00
#SBATCH --output=clustering_graphs/logs/r23_%A_%a.out
#SBATCH --error=clustering_graphs/logs/r23_%A_%a.err
#
# PHASE 2 — R2.3 NN + MST baseline on ARNES. CPU (partition `all`), plain exec,
# --require-cache. Spectral clustering is CPU-bound; the GPU was only needed for
# phase-1 encode. ARNES `all` nodes: 16 cores / 336 GB, no per-core mem cap.
# Peak RSS ~24 GiB/shard on ArxivP2P → 48 GB is safe headroom.
#
# 16 shards, partition_idx % 16. MiniLM-L6, 11 datasets, 8 ks, 10 seeds,
# NN + MST, mst-mode paper-faithful.

set -euo pipefail

WORKDIR=$HOME/clustering_graphs
SIF=$WORKDIR/container/clustgraphs.sif
NSHARDS=16
SHARD=$SLURM_ARRAY_TASK_ID
mkdir -p "$WORKDIR/results/r23_mdmst_l12_corrected_arnes/shard${SHARD}" "$WORKDIR/logs"

# OMP=4 optimal for the AMG spectral solver (higher oversubscribes).
export APPTAINERENV_OMP_NUM_THREADS=4
export APPTAINERENV_OPENBLAS_NUM_THREADS=4
export APPTAINERENV_MKL_NUM_THREADS=4
export APPTAINERENV_HF_HOME=/cache/hf
export APPTAINERENV_HF_HUB_ENABLE_HF_TRANSFER=0

echo "[$(date +%T)] === PHASE 2 R2.3 shard ${SHARD}/${NSHARDS} on $(hostname) (CPU) ==="

singularity exec \
    --bind $WORKDIR:/workspace \
    --bind $WORKDIR/cache:/cache \
    "$SIF" \
    /opt/venv/bin/python /workspace/src/main.py \
        --models sentence-transformers/all-MiniLM-L12-v2 \
        --datasets TwentyNewsgroupsClustering MedrxivClusteringS2S MedrxivClusteringP2P BiorxivClusteringS2S BiorxivClusteringP2P ArxivClusteringS2S ArxivClusteringP2P RedditClustering RedditClusteringP2P StackExchangeClustering StackExchangeClusteringP2P \
        --methods MD \
        --ks 1 2 3 6 8 10 15 20 \
        --seeds 0 1 2 3 4 5 6 7 8 9 \
        --ordering random \
        --mst on \
        --mst-mode corrected \
        --output-dir /workspace/results/r23_mdmst_l12_corrected_arnes/shard${SHARD} \
        --embed-cache /cache/embs \
        --require-cache \
        --max-partition-nodes 46340 \
        --heavy-stats-max-nodes 0 \
        --num-shards $NSHARDS \
        --shard-id $SHARD \
        --log-level INFO

echo "[$(date +%T)] === shard ${SHARD} complete ==="
