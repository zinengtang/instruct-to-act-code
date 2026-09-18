#!/bin/bash
#SBATCH -p berkeleynlp
#SBATCH --qos=preemptive_high_suhr
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 3:00:00
#SBATCH -J i2a_probe
#SBATCH --exclude=thidwick,lorax
#SBATCH -o /scratch/users/terran/i2a_stop/probe_done_%j.out
# usage: sbatch probe_done.sbatch <ckpt_dir> <out.json> [extra config yaml]
export HF_HOME=/scratch/users/terran/hf_home HF_HUB_CACHE=/scratch/users/terran/hf_home/hub HUGGINGFACE_HUB_CACHE=/scratch/users/terran/hf_home/hub HF_HUB_OFFLINE=1
unset TRANSFORMERS_CACHE; export XLA_PYTHON_CLIENT_MEM_FRACTION=0.8 TMPDIR=/scratch/users/terran/tmp I2A_EXPORT_FEAT=1
cd /accounts/projects/berkeleynlp/terran/projects/embodied/instruct_to_act; export PYTHONPATH=$PWD:$PWD/dreamerv3
/scratch/users/terran/conda/envs/embodied/bin/python probe_done.py --checkpoint "$1" --config configs/base.yaml configs/crafter.yaml configs/crafter_done.yaml ${3:-} --steps 300 --rollouts 10 --out "$2" 2>&1 | grep -vE "Gym|gym|migration|numpy 2|Warning"
echo "== PROBE DONE =="
