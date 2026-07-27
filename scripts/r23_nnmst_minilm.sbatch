#!/bin/bash
#SBATCH --job-name=r23_nnmst
#SBATCH --partition=e8
#SBATCH --nodelist=kt-gpu4
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --output=/ceph/home/bkoloski/clustering_graphs/results/r23_nnmst_%j.out
#
# R2.3 — the load-bearing reviewer ask.
# Adds the k-NN + MST baseline to Tables 4/5/6, using the paper's main backbone
# (MiniLM-L6-v2) across all 11 MTEB clustering partitions the paper reports.
#
# mst-mode is `paper-faithful` intentionally: the paper's existing MD+MST numbers
# were produced by that combine, so these new NN+MST rows are commensurable with
# them. A supplementary section will document the `corrected`-mode audit (job 81285)
# separately.

set -euo pipefail

ROOT=/ceph/home/bkoloski/clustering_graphs
cd "$ROOT"

mkdir -p results cache/hf cache/embs

# IJS proxy for HF downloads
export HTTP_PROXY=http://www-proxy.ijs.si:8080
export HTTPS_PROXY=$HTTP_PROXY
export NO_PROXY="localhost,127.0.0.1,.ijs.si"

# Apptainer env passthrough
export APPTAINERENV_HTTP_PROXY=$HTTP_PROXY
export APPTAINERENV_HTTPS_PROXY=$HTTP_PROXY
export APPTAINERENV_NO_PROXY=$NO_PROXY
export APPTAINERENV_HF_HOME=/cache/hf
export APPTAINERENV_HF_HUB_ENABLE_HF_TRANSFER=0

# BLAS threads — OMP=4 was empirically fastest for this workload; OMP=16 was 5x slower.
export APPTAINERENV_OMP_NUM_THREADS=4
export APPTAINERENV_OPENBLAS_NUM_THREADS=4
export APPTAINERENV_MKL_NUM_THREADS=4
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export MKL_NUM_THREADS=4

# The 11 partitions the paper reports in Tables 1/4/5/6.
DATASETS=(
  TwentyNewsgroupsClustering
  MedrxivClusteringS2S
  MedrxivClusteringP2P
  BiorxivClusteringS2S
  BiorxivClusteringP2P
  ArxivClusteringS2S
  ArxivClusteringP2P
  RedditClustering
  RedditClusteringP2P
  StackExchangeClustering
  StackExchangeClusteringP2P
)
KS=(1 2 3 6 8 10 15 20)
SEEDS=(0 1 2 3 4 5 6 7 8 9)

OUT="$ROOT/results/r23_nnmst_minilm"
mkdir -p "$OUT"

echo "[$(date +%T)] === R2.3 NN+MST sweep on $(hostname) ==="
nvidia-smi -L || true
echo "datasets: ${DATASETS[*]}"
echo "ks:       ${KS[*]}"
echo "seeds:    ${SEEDS[*]}"

USE_GPU=1 CACHE="$ROOT/cache" "$ROOT/container/run.sh" \
  /workspace/src/main.py \
  --models sentence-transformers/all-MiniLM-L6-v2 \
  --datasets "${DATASETS[@]}" \
  --methods NN \
  --ks "${KS[@]}" \
  --seeds "${SEEDS[@]}" \
  --ordering random \
  --mst on \
  --mst-mode paper-faithful \
  --output-dir /workspace/results/r23_nnmst_minilm \
  --embed-cache /cache/embs \
  --heavy-stats-max-nodes 0 \
  --log-level INFO 2>&1 | tee -a "$OUT/run.log"

echo "[$(date +%T)] === job complete ==="
