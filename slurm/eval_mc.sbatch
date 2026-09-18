#!/bin/bash
#SBATCH -p jsteinhardt
#SBATCH --gres=gpu:A5000:1
#SBATCH -c 8
#SBATCH --mem=48G
#SBATCH -t 6:00:00
#SBATCH -J i2a_mc_eval
#SBATCH -o /scratch/users/terran/i2a_runs/mc_eval_%j.out

export HF_HOME=/scratch/users/terran/hf_home
export HF_HUB_CACHE=/scratch/users/terran/hf_home/hub
export HUGGINGFACE_HUB_CACHE=/scratch/users/terran/hf_home/hub
export HF_HUB_OFFLINE=1
unset TRANSFORMERS_CACHE
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.8

cd /accounts/projects/berkeleynlp/terran/projects/embodied/instruct_to_act
export PYTHONPATH=$PWD:$PWD/dreamerv3
PY=/scratch/users/terran/conda/envs/embodied/bin/python
CKPT=/scratch/users/terran/i2a_runs/mc200m_b_seed0/ckpt
CFG="configs/base.yaml configs/minecraft.yaml configs/size200m.yaml configs/minecraft_b.yaml"
XVFB="xvfb-run -a -s"
XARGS='-screen 0 1024x768x24 -ac +extension GLX +render -noreset'

pkill -9 -u "$(whoami)" -f "MalmoMod.*fat.jar" 2>/dev/null || true
pkill -9 -u "$(whoami)" -f "launchClient.sh"   2>/dev/null || true
nvidia-smi -L
echo "== internet check (expect 401) =="
curl -s -m 8 -o /dev/null -w "%{http_code}\n" https://api.openai.com/v1/models || echo NO_INTERNET

echo "== 1. instruction probe (minecraft) =="
$XVFB "$XARGS" $PY probe_instruction.py --checkpoint $CKPT --config $CFG \
    --steps 300 --rollouts 3 --outdir results/mc_probe

echo "== 2. gpt4o online demo (diamond) =="
$XVFB "$XARGS" $PY evaluate.py --config $CFG \
    --checkpoint $CKPT --planner gpt4o --mode online \
    --episodes 2 --max_steps 3000 --outdir results/mc_eval

echo "== done =="
