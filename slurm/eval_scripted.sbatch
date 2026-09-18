#!/bin/bash
#SBATCH -p jsteinhardt
#SBATCH --gres=gpu:A5000:1
#SBATCH -c 8
#SBATCH --mem=48G
#SBATCH -t 6:00:00
#SBATCH -J i2a_scripted
#SBATCH -o /scratch/users/terran/i2a_runs/scripted_eval_%j.out

# Option-1 eval: fixed ScriptedPlanner (no OpenAI API) isolates the CONTROLLER's
# instruction-following on the final converged checkpoint (~6.17M steps).
# Metric: milestones achieved / instructions issued = per-instruction completion.

export HF_HOME=/scratch/users/terran/hf_home
export HF_HUB_CACHE=/scratch/users/terran/hf_home/hub
export HUGGINGFACE_HUB_CACHE=/scratch/users/terran/hf_home/hub
export HF_HUB_OFFLINE=1
unset TRANSFORMERS_CACHE
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.8
# keep Minecraft temp dirs off the full scratch quota if possible
export TMPDIR=${TMPDIR:-/scratch/users/terran/tmp}
mkdir -p "$TMPDIR"

cd /accounts/projects/berkeleynlp/terran/projects/embodied/instruct_to_act
export PYTHONPATH=$PWD:$PWD/dreamerv3
PY=/scratch/users/terran/conda/envs/embodied/bin/python
CKPT=/scratch/users/terran/i2a_runs/mc200m_b_seed0/ckpt
CFG="configs/base.yaml configs/minecraft.yaml configs/size200m.yaml configs/minecraft_b.yaml"
XVFB="xvfb-run -a -s"
XARGS='-screen 0 1024x768x24 -ac +extension GLX +render -noreset'

# global Malmo pkill removed 2026-09-18: it killed other running evals on the same node
# (see above)
nvidia-smi -L
echo "== scripted-planner eval (no API) on $(cat $CKPT/latest) =="
$XVFB "$XARGS" $PY evaluate.py --config $CFG \
    --checkpoint $CKPT --planner scripted --mode online \
    --episodes 30 --max_steps 3000 --outdir results/mc_eval_scripted

echo "== per-instruction accuracy =="
$PY - <<'PYEOF'
import json, glob, os
f = sorted(glob.glob('results/mc_eval_scripted/*episodes*.json'))
if not f:
    print('no episodes json yet'); raise SystemExit
d = json.load(open(f[-1]))
eps = d.get('episode_detail', [])
if not eps:
    print('no per-episode detail in', f[-1]); raise SystemExit
import statistics as st
ach = [e.get('reward', 0) for e in eps]              # milestones reached
iss = [e.get('instructions_issued', 0) for e in eps] # scripted instructions emitted
pst = [e.get('p_stop_mean') for e in eps if e.get('p_stop_mean') is not None]
acc = [min(1.0, a/i) for a, i in zip(ach, iss) if i]
print(f'episodes={len(eps)}')
print(f'milestones/ep: mean={st.mean(ach):.2f} median={st.median(ach):.1f} max={max(ach):.0f}')
print(f'instructions issued/ep: mean={st.mean(iss):.1f}')
if pst: print(f'p_stop_mean: {st.mean(pst):.3f}')
if acc:
    print(f'INSTRUCTION-FOLLOWING ACCURACY (milestones achieved / instructions issued): '
          f'mean={st.mean(acc):.1%} median={st.median(acc):.1%} n={len(acc)}')
PYEOF
echo "== done =="
