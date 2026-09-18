#!/bin/bash
#SBATCH -p berkeleynlp
#SBATCH --qos=preemptive_high_suhr
#SBATCH --gres=gpu:H200:2
#SBATCH -c 12
#SBATCH --mem=80G
#SBATCH -t 4:00:00
#SBATCH -J harness_pilot
#SBATCH --exclude=thidwick,lorax
#SBATCH -o /scratch/users/terran/i2a_stop/harness_pilot_%j.out
# Planner-harness pilot: local Qwen3.5-27B (vLLM, GPU0) as the VLM planner; old controller on GPU1; text state + feedback + anomaly.
export HF_HOME=/scratch/users/terran/hf_home HF_HUB_CACHE=/scratch/users/terran/hf_home/hub HUGGINGFACE_HUB_CACHE=/scratch/users/terran/hf_home/hub HF_HUB_OFFLINE=1
unset TRANSFORMERS_CACHE; export TMPDIR=/scratch/users/terran/tmp VLLM_LOGGING_LEVEL=WARNING I2A_EXPORT_FEAT=1 I2A_DEBUG_PROMPT=1; mkdir -p $TMPDIR
cd /accounts/projects/berkeleynlp/terran/projects/embodied/instruct_to_act; export PYTHONPATH=$PWD:$PWD/dreamerv3
V=/scratch/users/terran/bench_venvs/vllm; PORT=$((34000 + SLURM_JOB_ID % 1000))
CUDA_VISIBLE_DEVICES=0 nohup $V/bin/vllm serve Qwen/Qwen3.5-27B --port $PORT --max-model-len 8192 --gpu-memory-utilization 0.85 --max-num-seqs 16 --dtype bfloat16 --gdn-prefill-backend triton --limit-mm-per-prompt '{"image": 3}' > /scratch/users/terran/i2a_stop/vllm_harness_${SLURM_JOB_ID}.log 2>&1 &
for i in $(seq 1 240); do curl -sf http://127.0.0.1:$PORT/v1/models >/dev/null && break; sleep 10; done
curl -sf http://127.0.0.1:$PORT/v1/models >/dev/null || { echo "FATAL: vllm did not start"; exit 1; }
export I2A_VLM_BASE_URL=http://127.0.0.1:$PORT/v1 I2A_VLM_MODEL=Qwen/Qwen3.5-27B CUDA_VISIBLE_DEVICES=1 XLA_PYTHON_CLIENT_MEM_FRACTION=0.8
PY=/scratch/users/terran/conda/envs/embodied/bin/python; XVFB="xvfb-run -a -s"; XARGS='-screen 0 1024x768x24 -ac +extension GLX +render -noreset'
pkill -9 -u "$(whoami)" -f "MalmoMod.*fat.jar" 2>/dev/null || true; pkill -9 -u "$(whoami)" -f "launchClient.sh" 2>/dev/null || true
CFG="configs/base.yaml configs/minecraft.yaml configs/size200m.yaml configs/minecraft_b.yaml configs/eval_both.yaml configs/eval_harness.yaml"
CK=${CK:-/scratch/users/terran/i2a_runs/mc200m_b_seed0/ckpt}; EP=${EPISODES:-2}
$XVFB "$XARGS" $PY evaluate.py --config $CFG --checkpoint $CK --planner gpt4o --mode online --episodes $EP --max_steps 3000 --seed 0 --record --outdir /scratch/users/terran/i2a_stop/mc_eval_rec/harness_${TAG:-pilot3} 2>&1 | grep -vE "^\s*$" | grep -E "planner prompt|^ep |S\(t'\)|F\(t'\)|M_r|P_n|p_stop_|instrs=|mean|Traceback|Error|GPT4oPlanner|saved" | head -60
echo "== HARNESS PILOT DONE =="
