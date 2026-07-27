#!/bin/bash
#SBATCH --job-name=r23_cpu
#SBATCH --array=0-15%16
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem-per-cpu=1800M
#SBATCH --time=24:00:00
#SBATCH --partition=cpu
# Vega's submit plugin rejects >2 GB/core (256 GB / 128-core nodes). Peak RSS
# measured at ~24 GiB per shard on ArxivP2P N=25k (dense MST distance matrix),
# so we buy memory with cores: 24 cores x 2000M = ~47 GB. BLAS still uses only
# OMP=4; the extra cores are the price of the RAM, not a parallelism knob.
#SBATCH --output=clustering_graphs/logs/r23_%A_%a.out
#SBATCH --error=clustering_graphs/logs/r23_%A_%a.err
#
# PHASE 2 — R2.3 NN + MST baseline. CPU-ONLY: requires the embed cache to be
# warm (run scripts/encode_vega.sbatch first, or chain via submit_all_vega.sh).
#
# Why CPU: spectral_embedding(amg) + KMeans + graph prep are all CPU-bound;
# the GPU was only ever used for encoding, which phase 1 already did. Running
# on the cpu partition means (a) no GPU queue contention, (b) 16-way shard
# parallelism, (c) no nvidia-container-cli anywhere — the class of failure
# that killed shard 0 of job 39820844 cannot occur here.
#
# --require-cache makes any cache miss a loud, immediate failure instead of a
# silent multi-minute CPU encode.
#
# Grid: MiniLM-L6 (paper's main backbone) x 11 datasets x 8 ks x 10 seeds x
# NN + MST, mst-mode paper-faithful (commensurable with the paper's frozen
# MD+MST numbers). 16 shards; each handles partition_idx % 16.

set -euo pipefail

WORKDIR=$HOME/clustering_graphs
SIF=$WORKDIR/container/clustgraphs.sif
NSHARDS=16
SHARD=$SLURM_ARRAY_TASK_ID
OUT=$WORKDIR/results/r23_nnmst_minilm_vega/shard${SHARD}

mkdir -p "$OUT" "$WORKDIR/logs"

# OMP=4 proven fastest for this workload (OMP=16 was ~5x slower: AMG/LOBPCG
# thread oversubscription on N ~ 5-25k graphs). cpus-per-task=8 leaves headroom
# for the Python main thread + IO.
export APPTAINERENV_OMP_NUM_THREADS=4
export APPTAINERENV_OPENBLAS_NUM_THREADS=4
export APPTAINERENV_MKL_NUM_THREADS=4
export APPTAINERENV_HF_HOME=/cache/hf
export APPTAINERENV_HF_HUB_ENABLE_HF_TRANSFER=0
# No GPU on these nodes; make sure the cluster's nvccli auto-config stays off.
export APPTAINER_NVCCLI=false
export SINGULARITY_NVCCLI=false
export APPTAINERENV_NVIDIA_VISIBLE_DEVICES=
export APPTAINERENV_CUDA_VISIBLE_DEVICES=

echo "[$(date +%T)] === PHASE 2 R2.3 shard ${SHARD}/${NSHARDS} on $(hostname) (CPU) ==="

singularity exec \
    --bind $WORKDIR:/workspace \
    --bind $WORKDIR/cache:/cache \
    "$SIF" \
    /opt/venv/bin/python /workspace/src/main.py \
        --models sentence-transformers/all-MiniLM-L6-v2 \
        --datasets TwentyNewsgroupsClustering MedrxivClusteringS2S MedrxivClusteringP2P BiorxivClusteringS2S BiorxivClusteringP2P ArxivClusteringS2S ArxivClusteringP2P RedditClustering RedditClusteringP2P StackExchangeClustering StackExchangeClusteringP2P \
        --methods NN \
        --ks 1 2 3 6 8 10 15 20 \
        --seeds 0 1 2 3 4 5 6 7 8 9 \
        --ordering random \
        --mst on \
        --mst-mode paper-faithful \
        --output-dir /workspace/results/r23_nnmst_minilm_vega/shard${SHARD} \
        --embed-cache /cache/embs \
        --require-cache \
        --heavy-stats-max-nodes 0 \
        --num-shards $NSHARDS \
        --shard-id $SHARD \
        --log-level INFO

echo "[$(date +%T)] === shard ${SHARD} complete ==="
