#!/bin/bash
#SBATCH -p berkeleynlp
#SBATCH --qos=preemptive_high_suhr
#SBATCH --gres=gpu:H200:1
#SBATCH -c 8
#SBATCH --mem=48G
#SBATCH -t 6:00:00
#SBATCH -J i2a_eval_ckpt
#SBATCH --exclude=thidwick,lorax
#SBATCH -o /scratch/users/terran/i2a_stop/eval_ckpt_%j.out
# Recorded scripted-planner eval of a checkpoint snapshot: env CKPT (ckpt dir with 'latest'), TAG (output name), CFG (config list).
export HF_HOME=/scratch/users/terran/hf_home HF_HUB_CACHE=/scratch/users/terran/hf_home/hub HUGGINGFACE_HUB_CACHE=/scratch/users/terran/hf_home/hub HF_HUB_OFFLINE=1
unset TRANSFORMERS_CACHE; export XLA_PYTHON_CLIENT_MEM_FRACTION=0.8 TMPDIR=/scratch/users/terran/tmp I2A_EXPORT_FEAT=1; mkdir -p $TMPDIR
cd /accounts/projects/berkeleynlp/terran/projects/embodied/instruct_to_act; export PYTHONPATH=$PWD:$PWD/dreamerv3
PY=/scratch/users/terran/conda/envs/embodied/bin/python; XVFB="xvfb-run -a -s"; XARGS='-screen 0 1024x768x24 -ac +extension GLX +render -noreset'
# global Malmo pkill removed 2026-09-18: it killed other running evals on the same node
: "${CKPT:?}" "${TAG:?}"; CFG=${CFG:-"configs/base.yaml configs/minecraft.yaml configs/size200m.yaml configs/minecraft_done_hybrid.yaml"}
echo "== eval $TAG on $(cat $CKPT/latest) =="
for C in ${CONDS:-timer pstop_only}; do
  $XVFB "$XARGS" $PY evaluate.py --config $CFG configs/eval_$C.yaml --checkpoint $CKPT --planner scripted --mode online --episodes ${EPISODES:-30} --max_steps 3000 --seed 0 --record \
      --outdir /scratch/users/terran/i2a_stop/mc_eval_rec/${TAG}_$C 2>&1 | grep -E "^ep |mean|Traceback|Error" | tail -4
  echo "== followthrough $TAG $C =="; $PY followthrough.py /scratch/users/terran/i2a_stop/mc_eval_rec/${TAG}_$C 2>&1 | tail -1
done
echo "== EVAL CKPT DONE =="
