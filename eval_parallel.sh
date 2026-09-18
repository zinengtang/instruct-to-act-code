#!/usr/bin/env bash
# Parallel evaluation: split N episodes across W workers, merge results.
#
# Usage:
#   ./eval_parallel.sh [--workers N] [--episodes N] [--max_steps N] [extra evaluate.py flags]
#
# Examples:
#   ./eval_parallel.sh                                     # 4 workers × 25 episodes
#   ./eval_parallel.sh --workers 2 --episodes 50           # 2 workers × 25 episodes
#   ./eval_parallel.sh --max_steps 9000                    # halve timeout cost

set -euo pipefail

WORKERS=4
EPISODES=100
MAX_STEPS=9000   # half of 18k; cuts timeout episodes from 22 min → 11 min
CHECKPOINT="${CHECKPOINT:-checkpoints/mc-200m/ckpt}"
OUTDIR="results/eval_seed1"
PYTHON="${PYTHON:-python3}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --workers)   WORKERS="$2";   shift 2 ;;
    --episodes)  EPISODES="$2";  shift 2 ;;
    --max_steps) MAX_STEPS="$2"; shift 2 ;;
    --checkpoint) CHECKPOINT="$2"; shift 2 ;;
    --outdir)    OUTDIR="$2";    shift 2 ;;
    *)           EXTRA_ARGS+=("$1"); shift ;;
  esac
done

EPS_PER_WORKER=$(( (EPISODES + WORKERS - 1) / WORKERS ))
XLA_CACHE_DIR="/tmp/jax_eval_cache"
mkdir -p "$OUTDIR" "$XLA_CACHE_DIR"

echo "==> Parallel eval: $WORKERS workers × $EPS_PER_WORKER episodes = $((WORKERS * EPS_PER_WORKER)) total"
echo "==> max_steps=$MAX_STEPS  checkpoint=$(basename $(dirname $CHECKPOINT))"
echo "==> XLA cache: $XLA_CACHE_DIR"

PIDS=()
LOGS=()
for i in $(seq 0 $((WORKERS - 1))); do
  LOG="/tmp/eval_worker${i}.log"
  LOGS+=("$LOG")
  WORKER_OUTDIR="${OUTDIR}/worker${i}"
  mkdir -p "$WORKER_OUTDIR"

  XLA_FLAGS="--xla_gpu_persistent_cache_dir=${XLA_CACHE_DIR}" \
  XLA_PYTHON_CLIENT_MEM_FRACTION=0.15 \
  xvfb-run -a -s '-screen 0 1024x768x24 -ac +extension GLX +render -noreset' \
    "$PYTHON" "$SCRIPT_DIR/evaluate.py" \
    --config configs/base.yaml configs/minecraft.yaml configs/size200m.yaml \
    --checkpoint "$CHECKPOINT" \
    --planner gpt4o \
    --mode online \
    --episodes "$EPS_PER_WORKER" \
    --max_steps "$MAX_STEPS" \
    --seed "$i" \
    --outdir "$WORKER_OUTDIR" \
    "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}" \
    > "$LOG" 2>&1 &

  PIDS+=($!)
  echo "  Worker $i: PID=$!  log=$LOG"
done

echo ""
echo "==> Waiting for all workers..."
FAILED=0
for i in "${!PIDS[@]}"; do
  if wait "${PIDS[$i]}"; then
    echo "  Worker $i done OK"
  else
    echo "  Worker $i FAILED (exit $?)"
    FAILED=1
  fi
done

# Merge results
echo ""
echo "==> Merging results..."
"$PYTHON" - <<'PYEOF'
import json, glob, sys, os, numpy as np
from pathlib import Path

outdir = os.environ.get('OUTDIR', 'results/eval_seed1')
files  = glob.glob(f'{outdir}/worker*/*_episodes.json')
if not files:
    print("No result files found.")
    sys.exit(1)

all_rewards = []
meta = None
for f in sorted(files):
    d = json.load(open(f))
    all_rewards.extend(d['rewards'])
    meta = d

arr = np.array(all_rewards)
summary = {
    'env':      meta['env'],
    'planner':  meta['planner'],
    'mode':     meta['mode'],
    'episodes': len(arr),
    'mean':     float(arr.mean()),
    'std':      float(arr.std()),
    'ci95':     float(1.96 * arr.std() / np.sqrt(len(arr))),
    'min':      float(arr.min()),
    'max':      float(arr.max()),
}
tag = f"{meta['env']}_{meta['planner']}_{meta['mode']}_merged"
with open(f'{outdir}/{tag}_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)
with open(f'{outdir}/{tag}_episodes.json', 'w') as f:
    json.dump({'rewards': all_rewards, **summary}, f, indent=2)
print(f"Merged {len(arr)} episodes: mean={summary['mean']:.2f} ± {summary['std']:.2f}")
print(f"Saved to {outdir}/{tag}_*.json")
PYEOF

exit $FAILED
