# Decoupling Planning and Control for Instructable Agents

Code for **"Decoupling Planning and Control for Instructable Agents" (COLM 2026)**.

A pre-trained VLM **planner** maps observations to natural-language instructions;
a lightweight language-conditioned world-model **controller** (DreamerV3-style
RSSM + instruction-completion head) executes them at control frequency. The
planner is plug-and-play (no fine-tuning); the controller is trained with
post-hoc instruction relabeling of its own rollouts — no expert data.

Project page: https://zinengtang.github.io/instruct-to-act/

## Install

```bash
conda create -n embodied python=3.11 -y && conda activate embodied
bash setup.sh          # clones official dreamerv3/ + installs requirements
# jax with CUDA (adapt to your CUDA version):
pip install "jax[cuda12]"
```

Minecraft (MineRL 0.4.4) additionally needs Java 8 and xvfb, plus the patched
wheel that removes the obsolete gym pin:

```bash
pip install https://github.com/danijar/minerl/releases/download/v0.4.4-patched/minerl_mirror-0.4.4-cp311-cp311-linux_x86_64.whl
```

API keys: put your OpenAI key in a one-line file `.openai` at the repo root
(gitignored), or export `OPENAI_API_KEY`.

## Training

```bash
python train.py \
  --config configs/base.yaml configs/minecraft.yaml configs/size200m.yaml \
  --logdir <logdir> --seed 0 --batch_size 32
```

- `configs/minecraft_b.yaml` shows the budget-guarded annotation setup
  (`annotate_once: true`, `annotate_max_calls`) used for `mc-200m`.
- `configs/crafter_smoke.yaml` is a ~1.5h pipeline-validation run.
- `run.sh` wraps xvfb + zombie-Java cleanup for Minecraft;
  `slurm/` holds example SLURM scripts from our cluster (adapt paths).

## Evaluation

```bash
python evaluate.py --config <configs> --checkpoint <ckpt> \
  --planner gpt4o --mode online --episodes 100        # Algorithm 1
python probe_instruction.py --checkpoint <ckpt> --config <configs>
    # fixed-instruction probes: action-distribution shift + p_stop + GIFs
```

## Citation

```bibtex
@inproceedings{tang2026instructtoact,
  title     = {Decoupling Planning and Control for Instructable Agents},
  author    = {Tang, Zineng and Allen, Kelsey R. and van Steenkiste, Sjoerd
               and Dasgupta, Ishita and Suhr, Alane},
  booktitle = {Conference on Language Modeling (COLM)},
  year      = {2026}
}
```

MIT License.
