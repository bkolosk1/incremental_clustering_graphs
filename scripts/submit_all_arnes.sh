#!/bin/bash
# Two-phase ARNES pipeline: GPU encode → CPU experiments, chained via afterok.
#   PHASE 1  encode_arnes.sbatch       (gpu partition, --nv, ~15-30 min)
#   PHASE 2a r23_nnmst_arnes.sbatch    (all partition, 16 CPU shards)
#   PHASE 2b r23_orderings_arnes.sbatch(all partition, 2 CPU tasks)
#
# Run ON the ARNES login node from ~/clustering_graphs:
#   bash scripts/submit_all_arnes.sh
#   ENCODE_ARRAY=0-4 bash scripts/submit_all_arnes.sh   # all 5 models
#   bash scripts/submit_all_arnes.sh --skip-encode      # cache already warm
set -euo pipefail
cd "$(dirname "$0")/.."
ENCODE_ARRAY="${ENCODE_ARRAY:-0}"

if [[ "${1:-}" == "--skip-encode" ]]; then
  R23=$(sbatch --parsable scripts/r23_nnmst_arnes.sbatch)
  ORD=$(sbatch --parsable scripts/r23_orderings_arnes.sbatch)
  echo "submitted (no dep): r23=${R23} orderings=${ORD}"
  exit 0
fi

echo ">>> PHASE 1 encode (array=${ENCODE_ARRAY})"
ENC=$(sbatch --parsable --array="${ENCODE_ARRAY}" scripts/encode_arnes.sbatch)
echo "    encode: ${ENC}"
echo ">>> PHASE 2 experiments (afterok:${ENC})"
R23=$(sbatch --parsable --dependency=afterok:${ENC} scripts/r23_nnmst_arnes.sbatch)
ORD=$(sbatch --parsable --dependency=afterok:${ENC} scripts/r23_orderings_arnes.sbatch)
echo "    r23:       ${R23} (16 CPU shards)"
echo "    orderings: ${ORD} (2 CPU tasks)"
echo ""
echo "Monitor: squeue -u \$USER"
