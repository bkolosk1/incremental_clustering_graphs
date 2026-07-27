#!/bin/bash
#SBATCH --job-name=r25cost
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=all
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=0-02:00:00
#SBATCH --output=clustering_graphs/logs/r25cost_%j.out
#SBATCH --error=clustering_graphs/logs/r25cost_%j.err
#
# R2.5 FAIR cost benchmark. One singularity exec per (partition, method, mst) so
# peak RSS is per-config-clean. Two partitions x {Ours, Ours+MST, k-NN, k-NN+MST}.
# CPU node, 4 BLAS/OMP threads (matches the r23 clustering runs).

set -euo pipefail
WORKDIR=$HOME/clustering_graphs
SIF=$WORKDIR/container/clustgraphs.sif
CACHE=$WORKDIR/cache/embs
OUT=$WORKDIR/results/r25_cost/cost.jsonl
mkdir -p "$WORKDIR/results/r25_cost" "$WORKDIR/logs"
: > "$OUT"

export APPTAINERENV_OMP_NUM_THREADS=4
export APPTAINERENV_OPENBLAS_NUM_THREADS=4
export APPTAINERENV_MKL_NUM_THREADS=4

# partition | embedding-hash | task | n_clusters
# StackExchange S2S  part 0  N=11057  nc=20   -> eb0ae624828937b4
# Arxiv P2P          part 0  N=25000  nc=12   -> 4d9484be1eaaff3e
run_cfg () {
  local hash=$1 task=$2 part=$3 nc=$4 method=$5 mst=$6
  echo "[$(date +%T)] $task part=$part $method mst=$mst"
  singularity exec --bind "$WORKDIR":/workspace --bind "$WORKDIR/cache":/cache "$SIF" \
    /opt/venv/bin/python /workspace/scripts/cost_benchmark.py \
      --embs /cache/embs/${hash}.npy --task "$task" --partition "$part" --n-clusters "$nc" \
      --method "$method" --mst "$mst" --k 10 --repeats 3 --out /workspace/results/r25_cost/cost.jsonl
}

for spec in "eb0ae624828937b4 StackExchangeClustering 0 20" "4d9484be1eaaff3e ArxivClusteringP2P 0 12"; do
  set -- $spec
  run_cfg "$1" "$2" "$3" "$4" MD 0   # Ours
  run_cfg "$1" "$2" "$3" "$4" MD 1   # Ours+MST
  run_cfg "$1" "$2" "$3" "$4" NN 0   # k-NN
  run_cfg "$1" "$2" "$3" "$4" NN 1   # k-NN+MST
done
echo "[$(date +%T)] === cost benchmark complete ==="
