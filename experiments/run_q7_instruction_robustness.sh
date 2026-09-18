#!/usr/bin/env bash
set -euo pipefail

# Reviewer 2B11 Q7:
# Does the controller exploit instruction artifacts, or does it ground language?
#
# Usage:
#   CHECKPOINT=/path/to/checkpoint.pkl bash experiments/run_q7_instruction_robustness.sh
#
# Optional overrides:
#   ENV_NAME=minecraft
#   PLANNER=qwen
#   PLANNER_MODEL=Qwen/Qwen2.5-VL-7B-Instruct
#   PLANNER_DEVICE=cuda
#   EPISODES=50
#   SIZE_CONFIG=configs/size200m.yaml
#   OUTDIR=results/instruction_robustness
#   CONDITIONS="original paraphrase contradictory underspecified adversarial"
#   PYTHON=/scratch/users/terran/conda/envs/embodied/bin/python
#   USE_XVFB=1
#   KILL_MALMO=1
#   MEM_FRACTION=0.15
#
# By default, perturbations and the follow-accuracy proxy are local/rule-based.
# Set USE_API_JUDGE=1 only if you explicitly want GPT-4o paraphrases,
# contradictions, and judging.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_DIR}"

CHECKPOINT="/data/terran/instruct_to_act/mc200m_seed1"
ENV_NAME="${ENV_NAME:-minecraft}"
PLANNER="${PLANNER:-qwen}"
PLANNER_MODEL="${PLANNER_MODEL:-Qwen/Qwen2.5-VL-7B-Instruct}"
PLANNER_DEVICE="${PLANNER_DEVICE:-cuda}"
EPISODES="${EPISODES:-50}"
SIZE_CONFIG="${SIZE_CONFIG:-configs/size200m.yaml}"
OUTDIR="${OUTDIR:-results/instruction_robustness}"
CONDITIONS="${CONDITIONS:-original paraphrase contradictory underspecified adversarial}"
PYTHON="${PYTHON:-$(command -v python)}"
USE_XVFB="${USE_XVFB:-1}"
KILL_MALMO="${KILL_MALMO:-1}"
MEM_FRACTION="${MEM_FRACTION:-0.15}"

if [[ -z "${CHECKPOINT:-}" ]]; then
  echo "ERROR: set CHECKPOINT to a trained controller checkpoint path." >&2
  echo "Example:" >&2
  echo "  CHECKPOINT=logdir/minecraft/seed0/checkpoint.pkl bash experiments/run_q7_instruction_robustness.sh" >&2
  exit 2
fi

if [[ ! -e "${CHECKPOINT}" ]]; then
  echo "ERROR: CHECKPOINT does not exist: ${CHECKPOINT}" >&2
  exit 2
fi

if [[ -d "${CHECKPOINT}" && -d "${CHECKPOINT}/ckpt" ]]; then
  CHECKPOINT="${CHECKPOINT}/ckpt"
fi

if [[ "${PLANNER}" == "gpt4o" || "${USE_API_JUDGE:-0}" == "1" ]]; then
  if [[ -z "${OPENAI_API_KEY:-}" ]]; then
    echo "ERROR: OPENAI_API_KEY is required for GPT-4o planner or USE_API_JUDGE=1." >&2
    exit 2
  fi
fi

mkdir -p "${OUTDIR}"

if [[ "${KILL_MALMO}" == "1" ]]; then
  pkill -9 -u "$(whoami)" -f "MalmoMod.*fat.jar" 2>/dev/null || true
  pkill -9 -u "$(whoami)" -f "launchClient.sh" 2>/dev/null || true
fi

export XLA_PYTHON_CLIENT_MEM_FRACTION="${MEM_FRACTION}"

RUN_CMD=(
  "${PYTHON}" experiments/exp_instruction_robustness.py
  --env "${ENV_NAME}" \
  --checkpoint "${CHECKPOINT}" \
  --config configs/base.yaml "configs/${ENV_NAME}.yaml" "${SIZE_CONFIG}" \
  --planner "${PLANNER}" \
  --planner_model "${PLANNER_MODEL}" \
  --planner_device "${PLANNER_DEVICE}" \
  --conditions ${CONDITIONS} \
  --episodes "${EPISODES}" \
  --logdir "${OUTDIR}"
)

if [[ "${USE_XVFB}" == "1" && "${ENV_NAME}" == "minecraft" ]]; then
  xvfb-run -a -s '-screen 0 1024x768x24 -ac +extension GLX +render -noreset' "${RUN_CMD[@]}"
else
  "${RUN_CMD[@]}"
fi

RESULT_JSON="${OUTDIR}/robustness_${ENV_NAME}_${PLANNER}.json"

"${PYTHON}" experiments/plot_q7_instruction_robustness.py \
  --input "${RESULT_JSON}" \
  --outdir "${OUTDIR}" \
  --title "Reviewer 2B11 Q7: ${ENV_NAME} / ${PLANNER}"

echo
echo "Done."
echo "JSON:  ${RESULT_JSON}"
echo "Plot:  ${OUTDIR}/q7_instruction_robustness.png"
echo "Table: ${OUTDIR}/q7_instruction_robustness_table.csv"
