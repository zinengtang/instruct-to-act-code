#!/bin/sh
set -e

echo 'JAX/NVIDIA versions:'
pip freeze | grep -E 'jax|nvidia' | head -10
echo
echo GPUs:
nvidia-smi --query-gpu=gpu_name,memory.total,driver_version --format=csv || true
echo

exec xvfb-run -a -s '-screen 0 1024x768x24 -ac +extension GLX +render -noreset' \
  python /app/train.py "$@"
