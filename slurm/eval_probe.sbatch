#!/bin/bash
#SBATCH -p berkeleynlp
#SBATCH --qos=preemptive_high_suhr
#SBATCH --gres=gpu:H200:1
#SBATCH -c 8
#SBATCH --mem=48G
#SBATCH -t 8:00:00
#SBATCH -J i2a_eval_probe
#SBATCH --exclude=thidwick,lorax
#SBATCH -o /scratch/users/terran/i2a_stop/eval_probe_%j.out
# OLD controller + offline ground-truth stop probe: probe-only hand-off and probe + 200-step timer. Pilot 2 episodes first.
export HF_HOME=/scratch/users/terran/hf_home HF_HUB_CACHE=/scratch/users/terran/hf_home/hub HUGGINGFACE_HUB_CACHE=/scratch/users/terran/hf_home/hub HF_HUB_OFFLINE=1
unset TRANSFORMERS_CACHE; export XLA_PYTHON_CLIENT_MEM_FRACTION=0.8 TMPDIR=/scratch/users/terran/tmp I2A_EXPORT_FEAT=1; mkdir -p $TMPDIR
cd /accounts/projects/berkeleynlp/terran/projects/embodied/instruct_to_act; export PYTHONPATH=$PWD:$PWD/dreamerv3
PY=/scratch/users/terran/conda/envs/embodied/bin/python; XVFB="xvfb-run -a -s"; XARGS='-screen 0 1024x768x24 -ac +extension GLX +render -noreset'
# global Malmo pkill removed 2026-09-18: it killed other running evals on the same node
PROBE=${PROBE:-/scratch/users/terran/i2a_stop/probe_old_logistic.pkl}
BASECFG="configs/base.yaml configs/minecraft.yaml configs/size200m.yaml configs/minecraft_b.yaml"; CK=/scratch/users/terran/i2a_runs/mc200m_b_seed0/ckpt
run () {  # cond episodes tag
  echo "== old+probe / $1 : $2 episodes ($PROBE) =="
  $XVFB "$XARGS" $PY evaluate.py --config $BASECFG configs/eval_$1.yaml --checkpoint $CK --planner scripted --mode online --episodes $2 --max_steps 3000 --seed 0 \
      --stop_probe $PROBE --record --outdir /scratch/users/terran/i2a_stop/mc_eval_rec/oldprobe_$3_$1 2>&1 | grep -E "^ep |stop_probe|mean|Traceback|Error" | tail -8
}
TAG=$(basename $PROBE .pkl | sed 's/probe_old_//')
run pstop_only 2 pilot; S=$(ls /scratch/users/terran/i2a_stop/mc_eval_rec/oldprobe_pilot_pstop_only/*summary.json 2>/dev/null | head -1); [ -n "$S" ] || { echo "FATAL: pilot"; exit 1; }; rm -rf /scratch/users/terran/i2a_stop/mc_eval_rec/oldprobe_pilot_pstop_only; echo "== PILOT GATE PASSED =="
run pstop_only 30 $TAG; run both 30 $TAG
echo "== EVAL PROBE DONE =="
