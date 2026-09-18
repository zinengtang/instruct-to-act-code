#!/bin/bash
#SBATCH -p berkeleynlp
#SBATCH --qos=preemptive_high_suhr
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=48G
#SBATCH -t 8:00:00
#SBATCH -J i2a_done
#SBATCH --exclude=thidwick,lorax
#SBATCH -o /scratch/users/terran/i2a_stop/train_done_%j.out
export HF_HOME=/scratch/users/terran/hf_home HF_HUB_CACHE=/scratch/users/terran/hf_home/hub HUGGINGFACE_HUB_CACHE=/scratch/users/terran/hf_home/hub HF_HUB_OFFLINE=1
unset TRANSFORMERS_CACHE
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.8 TMPDIR=/scratch/users/terran/tmp
cd /accounts/projects/berkeleynlp/terran/projects/embodied/instruct_to_act
export PYTHONPATH=$PWD:$PWD/dreamerv3
PY=/scratch/users/terran/conda/envs/embodied/bin/python
LOGDIR=${LOGDIR:-/scratch/users/terran/i2a_runs/crafter_done_pilot}
CFG="configs/base.yaml configs/crafter.yaml configs/crafter_done.yaml ${EXTRA_CFG:-}"
nvidia-smi -L; echo "annotator v3 marker: $(grep -c "AchievementAnnotator v3" training/annotator.py); optimizer-fix marker: $(grep -c "Rebuild the optimizer" agent/lang_agent.py)"
echo "== train_done: logdir=$LOGDIR cfg=$CFG =="
mkdir -p $LOGDIR; $PY train.py --config $CFG --logdir $LOGDIR --seed 0 --batch_size 8 2>&1 | tee $LOGDIR/train_full.log | grep -E "AchievementAnnotator|Annotator|Traceback|Error|TRAIN" | cut -c1-200
echo "== TRAIN DONE =="
