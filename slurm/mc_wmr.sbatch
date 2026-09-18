#!/bin/bash
#SBATCH -p berkeleynlp
#SBATCH --qos=preemptive_high_suhr
#SBATCH --exclude=thidwick,lorax
#SBATCH --requeue
#SBATCH --gres=gpu:H200:1
#SBATCH -c 12
#SBATCH --mem=64G
#SBATCH -t 4-00:00:00
#SBATCH -J mc_wmr
#SBATCH -o /scratch/users/terran/i2a_stop/mc_wmr_%j.out

export HF_HOME=/scratch/users/terran/hf_home
export HF_HUB_CACHE=/scratch/users/terran/hf_home/hub
export HUGGINGFACE_HUB_CACHE=/scratch/users/terran/hf_home/hub
export HF_HUB_OFFLINE=1
unset TRANSFORMERS_CACHE

cd /accounts/projects/berkeleynlp/terran/projects/embodied/instruct_to_act
nvidia-smi -L; echo "annotator v3 marker: $(grep -c "AchievementAnnotator v3" training/annotator.py); inventory marker: $(grep -c "INVENTORY_ITEMS" training/annotator.py)"
echo "== internet check (expect 401) =="
curl -s -m 8 -o /dev/null -w "%{http_code}\n" https://api.openai.com/v1/models || echo NO_INTERNET

exec ./run.sh --seed 0 \
  --logdir ${LOGDIR:-/scratch/users/terran/i2a_runs/mc200m_wmr_release} \
  --batch_size 32 --mem_fraction 0.8 \
  --config configs/base.yaml configs/minecraft.yaml configs/size200m.yaml configs/minecraft_wmr_release.yaml ${EXTRA_CFG:-}
