#!/bin/bash
#SBATCH -p berkeleynlp
#SBATCH --qos=preemptive_high_suhr
#SBATCH --gres=gpu:H200:1
#SBATCH -c 8
#SBATCH --mem=48G
#SBATCH -t 14:00:00
#SBATCH -J i2a_eval_done
#SBATCH --exclude=thidwick,lorax
#SBATCH -o /scratch/users/terran/i2a_stop/eval_done_%j.out
# Scripted-planner eval of the converged ground-truth-label controller (mc200m_done_seed0) vs the old one (mc200m_b_seed0)
# under three hand-off rules: stop head + timer (both), timer only, stop head only. Pilot gate (2 episodes) first.
export HF_HOME=/scratch/users/terran/hf_home HF_HUB_CACHE=/scratch/users/terran/hf_home/hub HUGGINGFACE_HUB_CACHE=/scratch/users/terran/hf_home/hub HF_HUB_OFFLINE=1
unset TRANSFORMERS_CACHE; export XLA_PYTHON_CLIENT_MEM_FRACTION=0.8 TMPDIR=/scratch/users/terran/tmp; mkdir -p $TMPDIR
cd /accounts/projects/berkeleynlp/terran/projects/embodied/instruct_to_act; export PYTHONPATH=$PWD:$PWD/dreamerv3
PY=/scratch/users/terran/conda/envs/embodied/bin/python; XVFB="xvfb-run -a -s"; XARGS='-screen 0 1024x768x24 -ac +extension GLX +render -noreset'
# global Malmo pkill removed 2026-09-18: it killed other running evals on the same node
nvidia-smi -L
BASECFG="configs/base.yaml configs/minecraft.yaml configs/size200m.yaml"
run () {  # ckptdir tag cond episodes
  echo "== $2 / $3 : $4 episodes on $(cat $1/latest) =="
  $XVFB "$XARGS" $PY evaluate.py --config $BASECFG configs/minecraft_done.yaml configs/eval_$3.yaml --checkpoint $1 --planner scripted --mode online \
      --episodes $4 --max_steps 3000 --seed 0 --outdir results/mc_eval_done/$2_$3 2>&1 | grep -E "episode|mean|p_stop|Traceback|Error" | tail -8
}
# pilot gate
run /scratch/users/terran/i2a_runs/mc200m_done_seed0/ckpt done both 2
S=$(ls results/mc_eval_done/done_both/*summary.json 2>/dev/null | head -1); [ -n "$S" ] || { echo "FATAL: pilot produced no summary"; exit 1; }
$PY -c "import json,sys; d=json.load(open('$S')); print('pilot summary', {k:d[k] for k in ('episodes','mean','std')}); sys.exit(0 if d['episodes']==2 else 1)" || { echo "FATAL: pilot gate"; exit 1; }
rm -rf results/mc_eval_done/done_both; echo "== PILOT GATE PASSED =="
for C in both timer pstop_only; do run /scratch/users/terran/i2a_runs/mc200m_done_seed0/ckpt done $C 30; done
for C in both timer pstop_only; do run /scratch/users/terran/i2a_runs/mc200m_b_seed0/ckpt old $C 30; done
echo "== EVAL DONE =="
