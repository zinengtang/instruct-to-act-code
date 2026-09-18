#!/bin/bash
#SBATCH -p jsteinhardt
#SBATCH --gres=gpu:A5000:1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 3:00:00
#SBATCH -J i2a_eval
#SBATCH -o /scratch/users/terran/i2a_runs/eval_%j.out

export HF_HOME=/scratch/users/terran/hf_home
export HF_HUB_CACHE=/scratch/users/terran/hf_home/hub
export HUGGINGFACE_HUB_CACHE=/scratch/users/terran/hf_home/hub
export HF_HUB_OFFLINE=1
unset TRANSFORMERS_CACHE

cd /accounts/projects/berkeleynlp/terran/projects/embodied/instruct_to_act
export PYTHONPATH=$PWD:$PWD/dreamerv3
PY=/scratch/users/terran/conda/envs/embodied/bin/python
CKPT=/scratch/users/terran/i2a_runs/crafter_smoke_seed0/ckpt

echo "== internet check (expect 401 = reachable) =="
curl -s -m 8 -o /dev/null -w "%{http_code}\n" https://api.openai.com/v1/models || echo "NO_INTERNET"

echo "== 1. instruction probe =="
$PY probe_instruction.py --checkpoint $CKPT --steps 300 --rollouts 4 \
    --outdir results/smoke_probe

echo "== 2. gpt4o online demo =="
$PY evaluate.py \
    --config configs/base.yaml configs/crafter.yaml configs/crafter_smoke.yaml \
    --checkpoint $CKPT --planner gpt4o --mode online \
    --episodes 2 --max_steps 600 --outdir results/smoke_eval

echo "== done =="
