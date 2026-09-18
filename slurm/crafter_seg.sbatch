#!/bin/bash
#SBATCH -p berkeleynlp
#SBATCH --qos=preemptive_high_suhr
#SBATCH --exclude=thidwick,lorax
#SBATCH --requeue
#SBATCH --gres=gpu:H200:2
#SBATCH -c 16
#SBATCH --mem=96G
#SBATCH -t 12:00:00
#SBATCH -J crafter_seg
#SBATCH -o /scratch/users/terran/i2a_stop/crafter_seg_%j.out
# GPU0: local VLM annotator (Qwen3.5-27B, vision) via vLLM; GPU1: MC-200M training with hybrid annotation.
export HF_HOME=/scratch/users/terran/hf_home HF_HUB_CACHE=/scratch/users/terran/hf_home/hub HUGGINGFACE_HUB_CACHE=/scratch/users/terran/hf_home/hub HF_HUB_OFFLINE=1
unset TRANSFORMERS_CACHE; export TMPDIR=/scratch/users/terran/tmp VLLM_LOGGING_LEVEL=WARNING; mkdir -p $TMPDIR
cd /accounts/projects/berkeleynlp/terran/projects/embodied/instruct_to_act
V=/scratch/users/terran/bench_venvs/vllm; PORT=$((32000 + SLURM_JOB_ID % 1000))
CUDA_VISIBLE_DEVICES=0 nohup $V/bin/vllm serve Qwen/Qwen3.5-27B --port $PORT --max-model-len 16384 --gpu-memory-utilization 0.85 --max-num-seqs 32 --dtype bfloat16 --gdn-prefill-backend triton --limit-mm-per-prompt '{"image": 16}' > /scratch/users/terran/i2a_stop/vllm_seg_${SLURM_JOB_ID}.log 2>&1 &
for i in $(seq 1 240); do curl -sf http://127.0.0.1:$PORT/v1/models >/dev/null && break; sleep 10; done
curl -sf http://127.0.0.1:$PORT/v1/models >/dev/null || { echo "FATAL: vllm did not start"; tail -20 /scratch/users/terran/i2a_stop/vllm_seg_${SLURM_JOB_ID}.log; exit 1; }
export I2A_VLM_BASE_URL=http://127.0.0.1:$PORT/v1 I2A_VLM_MODEL=Qwen/Qwen3.5-27B
echo "segment marker: $(grep -c "_annotate_segments_per_worker" training/annotator.py)"
export CUDA_VISIBLE_DEVICES=1 PYTHONPATH=$PWD:$PWD/dreamerv3 XLA_PYTHON_CLIENT_MEM_FRACTION=0.8
PY=/scratch/users/terran/conda/envs/embodied/bin/python; LOGDIR=${LOGDIR:-/scratch/users/terran/i2a_runs/crafter_seg_pilot}; mkdir -p $LOGDIR
$PY train.py --config configs/base.yaml configs/crafter.yaml configs/crafter_seg.yaml ${EXTRA_CFG:-} --logdir $LOGDIR --seed 0 --batch_size 8 2>&1 | tee $LOGDIR/train_full.log | grep -E "SegmentAnnotator|AchievementAnnotator|Annotator\]|Traceback|Error|Checkpoint" | cut -c1-220
pkill -f "vllm serve"; echo "== TRAIN DONE =="
