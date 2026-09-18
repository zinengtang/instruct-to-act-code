#!/bin/bash
#SBATCH -p berkeleynlp
#SBATCH --qos=preemptive_high_suhr
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=48G
#SBATCH -t 6:00:00
#SBATCH -J i2a_stop
#SBATCH --exclude=thidwick,lorax
#SBATCH -o /scratch/users/terran/i2a_stop/collect_%j.out
# Stop-head relabel study: run the converged MC-200M controller with the scripted planner and record
# per-step RSSM feat + inventory + instruction + p_stop (--record). Env: EPISODES, MAXSTEPS, OUTDIR.
export HF_HOME=/scratch/users/terran/hf_home
export HF_HUB_CACHE=/scratch/users/terran/hf_home/hub
export HUGGINGFACE_HUB_CACHE=/scratch/users/terran/hf_home/hub
export HF_HUB_OFFLINE=1
unset TRANSFORMERS_CACHE
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.8 I2A_EXPORT_FEAT=1
export TMPDIR=${TMPDIR:-/scratch/users/terran/tmp}
mkdir -p "$TMPDIR" "${OUTDIR:-/scratch/users/terran/i2a_stop/pilot}"
cd /accounts/projects/berkeleynlp/terran/projects/embodied/instruct_to_act
export PYTHONPATH=$PWD:$PWD/dreamerv3
PY=/scratch/users/terran/conda/envs/embodied/bin/python
CKPT=/scratch/users/terran/i2a_runs/mc200m_b_seed0/ckpt
CFG="configs/base.yaml configs/minecraft.yaml configs/size200m.yaml configs/minecraft_b.yaml"
XVFB="xvfb-run -a -s"
XARGS='-screen 0 1024x768x24 -ac +extension GLX +render -noreset'
# global Malmo pkill removed 2026-09-18: it killed other running evals on the same node
# (see above)
nvidia-smi -L
echo "== stop-head data collection: episodes=${EPISODES:-1} max_steps=${MAXSTEPS:-300} ckpt=$(cat $CKPT/latest) =="
$XVFB "$XARGS" $PY evaluate.py --config $CFG \
    --checkpoint $CKPT --planner scripted --mode online \
    --episodes ${EPISODES:-1} --max_steps ${MAXSTEPS:-300} --record --outdir ${OUTDIR:-/scratch/users/terran/i2a_stop/pilot}
echo "== collect done =="
