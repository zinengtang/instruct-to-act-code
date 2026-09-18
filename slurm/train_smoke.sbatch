#!/bin/bash
#SBATCH -p jsteinhardt
#SBATCH --gres=gpu:A5000:1
#SBATCH -c 8
#SBATCH --mem=48G
#SBATCH -t 12:00:00
#SBATCH -J i2a_smoke
#SBATCH -o /scratch/users/terran/i2a_runs/smoke_%j.out

export HF_HOME=/scratch/users/terran/hf_home
export HF_HUB_CACHE=/scratch/users/terran/hf_home/hub
export HUGGINGFACE_HUB_CACHE=/scratch/users/terran/hf_home/hub
export HF_HUB_OFFLINE=1
unset TRANSFORMERS_CACHE

cd /accounts/projects/berkeleynlp/terran/projects/embodied/instruct_to_act
export PYTHONPATH=$PWD:$PWD/dreamerv3
nvidia-smi -L

exec /scratch/users/terran/conda/envs/embodied/bin/python train.py \
  --config configs/base.yaml configs/crafter.yaml configs/crafter_smoke.yaml \
  --logdir /scratch/users/terran/i2a_runs/crafter_smoke_seed0 \
  --seed 0 --batch_size 8
