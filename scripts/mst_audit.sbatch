#!/bin/bash
#SBATCH --job-name=mst_audit
#SBATCH --partition=e8
#SBATCH --nodelist=kt-gpu4
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/ceph/home/bkoloski/clustering_graphs/results/mst_audit_%j.out
#
# Side-by-side MST-mode audit (R2.3 / self-audit finding).
# Tight version: 1 small dataset (MedrxivClusteringS2S, ~3.7k/partition, 10 partitions),
# full ks/seeds, skip networkx-heavy graph stats, 16 BLAS threads.

set -euo pipefail

ROOT=/ceph/home/bkoloski/clustering_graphs
cd "$ROOT"

mkdir -p results cache/hf cache/embs

# IJS proxy
export HTTP_PROXY=http://www-proxy.ijs.si:8080
export HTTPS_PROXY=$HTTP_PROXY
export NO_PROXY="localhost,127.0.0.1,.ijs.si"

# Apptainer-preferred env passthrough (apptainer 1.x).
export APPTAINERENV_HTTP_PROXY=$HTTP_PROXY
export APPTAINERENV_HTTPS_PROXY=$HTTP_PROXY
export APPTAINERENV_NO_PROXY=$NO_PROXY
export APPTAINERENV_HF_HOME=/cache/hf
export APPTAINERENV_HF_HUB_ENABLE_HF_TRANSFER=0

# BLAS thread count. Tested OMP=16 -> ~5x SLOWER per cell than OMP=4 on N=5000
# (AMG/LOBPCG oversubscription on small matrices). OMP=4 is the sweet spot for
# this workload; the CPU headroom on the node is not usable by more BLAS threads.
export APPTAINERENV_OMP_NUM_THREADS=4
export APPTAINERENV_OPENBLAS_NUM_THREADS=4
export APPTAINERENV_MKL_NUM_THREADS=4
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export MKL_NUM_THREADS=4

# Audit grid: 1 dataset, full ks/seeds, mst=on only.
DATASETS=(MedrxivClusteringS2S)
KS=(1 2 3 6 8 10 15 20)
SEEDS=(0 1 2 3 4 5 6 7 8 9)

echo "[$(date +%T)] === starting MST audit on $(hostname) ==="
nvidia-smi -L || true

for MODE in paper-faithful corrected; do
  OUT="$ROOT/results/mst_audit_${MODE}"
  mkdir -p "$OUT"
  echo "[$(date +%T)] === mode: $MODE -> $OUT ==="
  USE_GPU=1 CACHE="$ROOT/cache" "$ROOT/container/run.sh" \
    /workspace/src/main.py \
    --models sentence-transformers/all-MiniLM-L6-v2 \
    --datasets "${DATASETS[@]}" \
    --methods MD NN \
    --ks "${KS[@]}" \
    --seeds "${SEEDS[@]}" \
    --ordering random \
    --mst on \
    --mst-mode "$MODE" \
    --output-dir "/workspace/results/mst_audit_${MODE}" \
    --embed-cache /cache/embs \
    --heavy-stats-max-nodes 0 \
    --log-level INFO 2>&1 | tee -a "$OUT/run.log"
  echo "[$(date +%T)] === done $MODE ==="
done

echo "[$(date +%T)] === running paired analysis ==="
"$ROOT/container/run.sh" /workspace/scripts/mst_audit.py \
  --paper "/workspace/results/mst_audit_paper-faithful" \
  --corrected "/workspace/results/mst_audit_corrected" \
  --out "/workspace/results/mst_audit_report" 2>&1 | tee "$ROOT/results/mst_audit_report.txt"

echo "[$(date +%T)] === job complete ==="
