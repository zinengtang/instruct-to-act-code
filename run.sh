#!/usr/bin/env bash
# Run Minecraft Instruct-to-Act training.
#
# Usage:
#   ./run.sh [--seed N] [--logdir PATH] [--batch_size N] [--batch_length N] [--mem_fraction F] [extra train.py flags]
#
# Examples:
#   ./run.sh                              # single job, seed 0
#   ./run.sh --seed 1                     # single job, seed 1
#   ./run.sh --mem_fraction 0.50          # single job, full GPU
#   for s in 0 1 2 3 4; do               # 5 parallel seeds
#     ./run.sh --seed $s --mem_fraction 0.15 &
#   done; wait

set -euo pipefail

SEED=3
LOGDIR=""
BATCH_SIZE=64    # 4x base (16→64); 160 caused CUDA_ERROR_LAUNCH_FAILED on H200
BATCH_LENGTH=""  # empty → use YAML default (64)
MEM_FRACTION=0.15  # per-job GPU fraction; 5 jobs × 0.15 ≈ 105 GB / 140 GB on H200
PYTHON="${PYTHON:-python3}"
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --seed)         SEED="$2";         shift 2 ;;
    --logdir)       LOGDIR="$2";       shift 2 ;;
    --batch_size)   BATCH_SIZE="$2";   shift 2 ;;
    --batch_length) BATCH_LENGTH="$2"; shift 2 ;;
    --mem_fraction) MEM_FRACTION="$2"; shift 2 ;;
    *)              EXTRA_ARGS+=("$1"); shift ;;
  esac
done

[[ -z "$LOGDIR" ]] && LOGDIR="./logdir/mc200m_seed${SEED}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

mkdir -p "$LOGDIR"

# Kill zombie Malmo/Java servers from any prior crashed runs
# global Malmo pkill removed: it killed other running evals on the same node
# (see above)

# BFC allocator with configurable pre-allocation fraction.
# Default 0.15 supports up to 5 parallel jobs on H200 (5 × 21 GB = 105 GB).
# Pass --mem_fraction 0.50 for a single-job run that needs more headroom.
export XLA_PYTHON_CLIENT_MEM_FRACTION="${MEM_FRACTION}"

BATCH_ARGS=(--batch_size "$BATCH_SIZE")
[[ -n "$BATCH_LENGTH" ]] && BATCH_ARGS+=(--batch_length "$BATCH_LENGTH")

echo "==> seed=$SEED  logdir=$LOGDIR  batch_size=$BATCH_SIZE  batch_length=${BATCH_LENGTH:-default}  mem_fraction=$MEM_FRACTION"
echo "==> Note: 200M model JIT-compile takes ~10 min; env collection starts after."

exec xvfb-run -a -s '-screen 0 1024x768x24 -ac +extension GLX +render -noreset' \
  "$PYTHON" "$SCRIPT_DIR/train.py" \
  --config configs/base.yaml configs/minecraft.yaml configs/size200m.yaml \
  --logdir "$LOGDIR" \
  --seed "$SEED" \
  "${BATCH_ARGS[@]}" \
  "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}"
