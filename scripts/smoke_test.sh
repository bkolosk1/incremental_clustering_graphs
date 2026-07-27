#!/usr/bin/env bash
# End-to-end smoke test for the rebuttal driver.
#
# Runs ONE small task with ONE k, ONE seed, ONE model, and --mst both. Should
# finish in ~2 minutes on a CPU box and produce a non-empty results.<pid>.jsonl
# with at least 4 records (MD/NN x mst on/off).
# This is the reviewer-verification target in rebuttal/plan.md  R2.6 step 4.
#
# Usage:
#   scripts/smoke_test.sh                           # CPU
#   USE_GPU=1 scripts/smoke_test.sh                 # with GPU
#   SIF=/path/to/clustgraphs.sif scripts/smoke_test.sh
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/.." && pwd)

# Paths: we use a single relative path so the same string is valid both on the
# host (under $ROOT) and inside the container (under /workspace bind-mount).
REL_OUT="results/smoke"
REL_CACHE=".smoke_cache/embs"
HOST_OUT="$ROOT/$REL_OUT"
HOST_CACHE="$ROOT/$REL_CACHE"
rm -rf "$HOST_OUT"
mkdir -p "$HOST_OUT" "$HOST_CACHE"

echo "[smoke] running through container/run.sh ..."
USE_GPU="${USE_GPU:-0}" "$ROOT/container/run.sh" \
    /workspace/src/main.py \
    --models sentence-transformers/all-MiniLM-L6-v2 \
    --datasets TwentyNewsgroupsClustering \
    --methods MD NN \
    --ks 3 \
    --seeds 0 \
    --ordering random \
    --mst both \
    --mst-mode paper-faithful \
    --limit 200 \
    --output-dir "/workspace/$REL_OUT" \
    --embed-cache "/workspace/$REL_CACHE" \
    --heavy-stats-max-nodes 5000 \
    --log-level INFO

# Sanity: at least one results.<pid>.jsonl must exist, be non-empty, and have at least 4 records (2 methods x 2 mst modes x 1 k x 1 seed = 4).
shopt -s nullglob
RES_FILES=( "$HOST_OUT"/results.*.jsonl )
if [[ ${#RES_FILES[@]} -eq 0 ]]; then
    echo "[smoke] FAIL: no results.*.jsonl found in $HOST_OUT" >&2
    exit 1
fi
RES="${RES_FILES[0]}"
if [[ ! -s "$RES" ]]; then
    echo "[smoke] FAIL: $RES is empty" >&2
    exit 1
fi
N=$(wc -l < "$RES")
if (( N < 4 )); then
    echo "[smoke] FAIL: expected >=4 lines in $RES, got $N" >&2
    exit 1
fi
echo "[smoke] OK -- $N records in $RES"
echo "[smoke] manifest(s):"
cat "$HOST_OUT"/manifest.*.json
echo
echo "[smoke] last record (preds/labels stripped):"
tail -1 "$RES" | python3 -c 'import sys, json; d=json.loads(sys.stdin.read()); d.pop("labels", None); d.pop("preds", None); print(json.dumps(d, indent=2, ensure_ascii=False))'
