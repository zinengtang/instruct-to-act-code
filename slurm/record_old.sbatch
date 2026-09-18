#!/bin/bash
#SBATCH -p berkeleynlp
#SBATCH --qos=preemptive_high_suhr
#SBATCH --gres=gpu:H200:1
#SBATCH -c 8
#SBATCH --mem=48G
#SBATCH -t 6:00:00
#SBATCH -J i2a_record_old
#SBATCH --exclude=thidwick,lorax
#SBATCH -o /scratch/users/terran/i2a_stop/record_old_%j.out
# Record the OLD controller's rollouts (feat / inventory / instruction / p_stop per step) under timer-only hand-off,
# to fit a ground-truth-label stop head offline without changing the controller. Pilot: 2 episodes first.
export HF_HOME=/scratch/users/terran/hf_home HF_HUB_CACHE=/scratch/users/terran/hf_home/hub HUGGINGFACE_HUB_CACHE=/scratch/users/terran/hf_home/hub HF_HUB_OFFLINE=1
unset TRANSFORMERS_CACHE; export XLA_PYTHON_CLIENT_MEM_FRACTION=0.8 TMPDIR=/scratch/users/terran/tmp I2A_EXPORT_FEAT=1; mkdir -p $TMPDIR
cd /accounts/projects/berkeleynlp/terran/projects/embodied/instruct_to_act; export PYTHONPATH=$PWD:$PWD/dreamerv3
PY=/scratch/users/terran/conda/envs/embodied/bin/python; XVFB="xvfb-run -a -s"; XARGS='-screen 0 1024x768x24 -ac +extension GLX +render -noreset'
# global Malmo pkill removed 2026-09-18: it killed other running evals on the same node
CFG="configs/base.yaml configs/minecraft.yaml configs/size200m.yaml configs/minecraft_b.yaml configs/eval_timer.yaml"
CK=/scratch/users/terran/i2a_runs/mc200m_b_seed0/ckpt; OUT=/scratch/users/terran/i2a_stop/record_old_timer
run () { $XVFB "$XARGS" $PY evaluate.py --config $CFG --checkpoint $CK --planner scripted --mode online --episodes $1 --max_steps 3000 --seed $2 --outdir $3 --record 2>&1 | grep -E "^ep |saved|mean|Traceback|Error" | tail -6; }
run 2 100 ${OUT}_pilot
$PY - <<'PYEOF' || { echo "FATAL: pilot gate"; exit 1; }
import numpy as np, glob, sys
fs = sorted(glob.glob('/scratch/users/terran/i2a_stop/record_old_timer_pilot/record_ep*.npz')); print('pilot records', len(fs))
if not fs: sys.exit(1)
d = np.load(fs[0], allow_pickle=True); print({k: getattr(d[k], 'shape', None) for k in d.files})
sys.exit(0 if len(fs) == 2 and d['feat'].shape[0] > 100 else 1)
PYEOF
echo "== PILOT GATE PASSED =="
run 40 0 $OUT
echo "== RECORD OLD DONE =="
