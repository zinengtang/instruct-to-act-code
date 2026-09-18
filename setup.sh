#!/usr/bin/env bash
# Clones the official DreamerV3 repo and installs all dependencies.
# Run once from the instruct_to_act/ directory.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── 1. Clone official DreamerV3 ──────────────────────────────────────────────
if [ ! -d "$SCRIPT_DIR/dreamerv3" ]; then
  git clone https://github.com/danijar/dreamerv3.git "$SCRIPT_DIR/dreamerv3"
  git -C "$SCRIPT_DIR/dreamerv3" checkout -q e3f02248693a79dc8b0ebd62c93683888ddaccfe
  # small local modifications (per-module grad-norm metrics, agent/transform hooks used by the language agent)
  git -C "$SCRIPT_DIR/dreamerv3" apply "$SCRIPT_DIR/patches/dreamerv3.patch"
else
  echo "dreamerv3/ already exists, skipping clone."
fi

# ── 2. Install all dependencies into the active Python environment ───────────
# Use the embodied conda env: conda activate embodied
pip install -r "$SCRIPT_DIR/requirements.txt"

# ── 3. Add project and dreamerv3 to Python path ──────────────────────────────
PYPATH_LINE="export PYTHONPATH=\"$SCRIPT_DIR:$SCRIPT_DIR/dreamerv3:\$PYTHONPATH\""
if ! grep -qF "$SCRIPT_DIR/dreamerv3" ~/.bashrc; then
  echo "$PYPATH_LINE" >> ~/.bashrc
fi

echo ""
echo "Setup complete. Activate with: source ~/.bashrc"
echo "Use the embodied conda Python directly:"
echo "  python train.py --config configs/base.yaml configs/minecraft.yaml --logdir logdir/minecraft/seed0"
echo "Or: conda run --no-capture-output -n embodied python train.py ..."
