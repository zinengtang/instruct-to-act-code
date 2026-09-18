#!/bin/bash
#SBATCH -p jsteinhardt
#SBATCH --gres=gpu:A5000:1
#SBATCH -c 12
#SBATCH --mem=64G
#SBATCH -t 7-00:00:00
#SBATCH -J i2a_mc_b2
#SBATCH -o /scratch/users/terran/i2a_runs/mc_b2_%j.out

export HF_HOME=/scratch/users/terran/hf_home
export HF_HUB_CACHE=/scratch/users/terran/hf_home/hub
export HUGGINGFACE_HUB_CACHE=/scratch/users/terran/hf_home/hub
export HF_HUB_OFFLINE=1
unset TRANSFORMERS_CACHE

cd /accounts/projects/berkeleynlp/terran/projects/embodied/instruct_to_act
nvidia-smi -L
echo "== internet check (expect 401) =="
curl -s -m 8 -o /dev/null -w "%{http_code}\n" https://api.openai.com/v1/models || echo NO_INTERNET

exec ./run.sh --seed 0 \
  --logdir /scratch/users/terran/i2a_runs/mc200m_b_seed0 \
  --batch_size 32 --mem_fraction 0.8 \
  --config configs/base.yaml configs/minecraft.yaml configs/size200m.yaml configs/minecraft_b2.yaml
