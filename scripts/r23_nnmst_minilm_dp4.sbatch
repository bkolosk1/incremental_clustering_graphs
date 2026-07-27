#!/bin/bash
#SBATCH --job-name=r23_dp4
#SBATCH --partition=e8
#SBATCH --nodelist=kt-gpu4
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=06:00:00
#SBATCH --output=/ceph/home/bkoloski/clustering_graphs/results/r23_dp4_%j.out
#
# R2.3 — data-parallel across 4 A100s.
# Runs 4 apptainer processes, each pinned to one GPU via CUDA_VISIBLE_DEVICES.
# Each shard processes partition_idx % 4 == shard_id of every dataset, so all
# datasets progress in parallel and the wall time is dominated by the slowest
# quarter of any one task (not the full task).

set -euo pipefail

ROOT=/ceph/home/bkoloski/clustering_graphs
cd "$ROOT"

mkdir -p results cache/hf cache/embs

# IJS proxy
export HTTP_PROXY=http://www-proxy.ijs.si:8080
export HTTPS_PROXY=$HTTP_PROXY
export NO_PROXY="localhost,127.0.0.1,.ijs.si"
export APPTAINERENV_HTTP_PROXY=$HTTP_PROXY
export APPTAINERENV_HTTPS_PROXY=$HTTP_PROXY
export APPTAINERENV_NO_PROXY=$NO_PROXY
export APPTAINERENV_HF_HOME=/cache/hf
export APPTAINERENV_HF_HUB_ENABLE_HF_TRANSFER=0

# Per-shard OMP=4 → 4 shards * 4 threads = 16 CPU threads total (fits in 64 cpus alloc).
export APPTAINERENV_OMP_NUM_THREADS=4
export APPTAINERENV_OPENBLAS_NUM_THREADS=4
export APPTAINERENV_MKL_NUM_THREADS=4
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export MKL_NUM_THREADS=4

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

OUT="$ROOT/results/r23_nnmst_minilm_dp4"
mkdir -p "$OUT"

echo "[$(date +%T)] === R2.3 DP-4 NN+MST sweep on $(hostname) ==="
nvidia-smi -L || true

# Salvage prior single-shard results so they can be merged in later.
if [[ -d "$ROOT/results/r23_nnmst_minilm" && -n "$(ls $ROOT/results/r23_nnmst_minilm/results.*.jsonl 2>/dev/null)" ]]; then
  echo "[$(date +%T)] salvaging prior r23_nnmst_minilm records"
  mv "$ROOT/results/r23_nnmst_minilm" "$ROOT/results/r23_nnmst_minilm_salvage_$(date +%s)"
fi

launch_shard() {
  local GPU=$1
  local SHARD=$2
  local N=$3
  local SHARD_OUT="$OUT/shard${SHARD}"
  mkdir -p "$SHARD_OUT"
  echo "[$(date +%T)] shard $SHARD launching on GPU $GPU"
  CUDA_VISIBLE_DEVICES=$GPU USE_GPU=1 CACHE="$ROOT/cache" \
    "$ROOT/container/run.sh" \
    /workspace/src/main.py \
    --models sentence-transformers/all-MiniLM-L6-v2 \
    --datasets "${DATASETS[@]}" \
    --methods NN \
    --ks "${KS[@]}" \
    --seeds "${SEEDS[@]}" \
    --ordering random \
    --mst on \
    --mst-mode paper-faithful \
    --output-dir "/workspace/results/r23_nnmst_minilm_dp4/shard${SHARD}" \
    --embed-cache /cache/embs \
    --heavy-stats-max-nodes 0 \
    --num-shards $N \
    --shard-id $SHARD \
    --log-level INFO 2>&1 | sed "s/^/[shard${SHARD}] /" > "$SHARD_OUT/run.log" 2>&1 &
  echo $! > "$SHARD_OUT/pid"
}

NSHARDS=2
for i in 0 1; do
  launch_shard $i $i $NSHARDS
done

wait
echo "[$(date +%T)] === all shards done ==="

echo "[$(date +%T)] === merging shard outputs ==="
cat "$OUT"/shard*/results.*.jsonl > "$OUT/all_results.jsonl"
echo "records after merge: $(wc -l < $OUT/all_results.jsonl)"
echo "[$(date +%T)] === job complete ==="
