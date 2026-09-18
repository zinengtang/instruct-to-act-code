"""
exp_annotator_sensitivity.py — Annotator sensitivity study.

Addresses Reviewer 2B11 Q4:
  "How sensitive are the results to the VLM used for post-hoc annotation?
   GPT-4o is used for annotation in the main setup; would a weaker or
   open-source annotator still work at scale?"

Extends Table 4 (which covers instruction TYPE, not model) to also sweep
over ANNOTATOR MODELS on Minecraft Diamond.

Conditions
──────────
  gpt4o     — GPT-4o (paper default)
  qwen      — Qwen-VL-2.5-72B (open source, strong)
  gemma     — Gemma-3-27B (open source, medium)
  llava     — LLaVA-v1.6-34b (open source, weaker)
  template  — Rule-based templates (no VLM)
  cluster   — k-means cluster labels
  random    — Random strings (lower bound)

For each condition we train a controller on Minecraft Diamond (or load a
pre-trained one), evaluate for 100 episodes with GPT-4o as the inference
planner, and report the Diamond score.

Usage
─────
  python experiments/exp_annotator_sensitivity.py \
      --env minecraft \
      --conditions gpt4o qwen gemma template cluster random \
      --eval_planner qwen \
      --episodes 100 \
      --logdir results/annotator_sensitivity
"""

import argparse, json, subprocess
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


ANNOTATOR_LABELS = {
    'gpt4o':    'GPT-4o (paper default)',
    'qwen':     'Qwen-VL-2.5-72B',
    'gemma':    'Gemma-3-27B',
    'llava':    'LLaVA-v1.6-34b',
    'template': 'Template-based',
    'cluster':  'Clustered action labels',
    'random':   'Random strings',
}

# Expected cost tiers (relative; for discussion)
RELATIVE_COST = {
    'gpt4o': 'high (API)',
    'qwen':  'medium (local 72B)',
    'gemma': 'medium (local 27B)',
    'llava': 'low (local 34B)',
    'template': 'negligible',
    'cluster':  'negligible',
    'random':   'negligible',
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--env',         default='minecraft')
    p.add_argument('--conditions',  nargs='+',
                   default=['gpt4o', 'qwen', 'gemma', 'template', 'cluster', 'random'])
    p.add_argument('--eval_planner', default='qwen')
    p.add_argument('--episodes',    type=int, default=100)
    p.add_argument('--seeds',       type=int, default=3)
    p.add_argument('--logdir',      default='results/annotator_sensitivity')
    p.add_argument('--train',       action='store_true')
    p.add_argument('--from_checkpoints', default=None)
    return p.parse_args()


def train_condition(condition, env, seed, logdir):
    """Train one controller with the given annotator_type."""
    out = Path(logdir) / condition / f'seed{seed}'
    cmd = [
        'python', 'train.py',
        '--config', f'configs/{env}.yaml', 'configs/base.yaml',
        '--annotator_type', condition,
        '--logdir', str(out),
        '--seed',   str(seed),
    ]
    print(f"Training: annotator={condition} seed={seed}")
    subprocess.run(cmd, check=True)
    return out


def evaluate_condition(condition, env, eval_planner, episodes, ckpt_root, seed, out_dir):
    ckpt = Path(ckpt_root) / condition / f'seed{seed}' / 'checkpoint.pkl'
    if not ckpt.exists():
        print(f"  MISSING: {ckpt}")
        return None
    out = Path(out_dir) / condition / f'seed{seed}'
    out.mkdir(parents=True, exist_ok=True)
    cmd = [
        'python', 'evaluate.py',
        '--config', f'configs/{env}.yaml', 'configs/base.yaml',
        '--checkpoint', str(ckpt),
        '--planner',    eval_planner,
        '--episodes',   str(episodes),
        '--seed',       str(seed),
        '--outdir',     str(out),
    ]
    subprocess.run(cmd, check=True)
    result_files = list(out.glob('*_episodes.json'))
    if not result_files:
        return None
    with open(result_files[0]) as f:
        data = json.load(f)
    return data['rewards']


def plot_results(results: dict, outdir: Path, env: str):
    conditions = list(results.keys())
    means = [np.mean(results[c]) for c in conditions]
    stds  = [np.std(results[c])  for c in conditions]
    labels = [ANNOTATOR_LABELS.get(c, c) for c in conditions]

    fig, ax = plt.subplots(figsize=(10, 4))
    x = range(len(conditions))
    ax.bar(x, means, yerr=stds, capsize=5,
           color=['#2196F3' if c == 'gpt4o' else '#78909C' for c in conditions])
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=20, ha='right', fontsize=9)
    ax.set_ylabel('Task Score')
    ax.set_title(f'Annotator Sensitivity — {env} (eval planner: GPT-4o)')
    ax.axhline(means[conditions.index('random')] if 'random' in conditions else 0,
               ls='--', color='red', alpha=0.5, label='Random (lower bound)')
    ax.legend()
    plt.tight_layout()
    fig.savefig(outdir / f'annotator_sensitivity_{env}.pdf')
    fig.savefig(outdir / f'annotator_sensitivity_{env}.png', dpi=150)
    plt.close()
    print(f"Saved plot to {outdir}/annotator_sensitivity_{env}.{{pdf,png}}")


def main():
    args   = parse_args()
    logdir = Path(args.logdir)
    logdir.mkdir(parents=True, exist_ok=True)
    ckpt_root = args.from_checkpoints or str(logdir / 'checkpoints')

    # Train
    if args.train:
        for condition in args.conditions:
            for seed in range(args.seeds):
                train_condition(condition, args.env, seed, ckpt_root)

    # Evaluate
    all_results = {}
    for condition in args.conditions:
        seed_rewards = []
        for seed in range(args.seeds):
            rewards = evaluate_condition(
                condition, args.env, args.eval_planner,
                args.episodes, ckpt_root, seed,
                str(logdir / 'eval')
            )
            if rewards is not None:
                seed_rewards.extend(rewards)
        if seed_rewards:
            all_results[condition] = seed_rewards
            m, s = np.mean(seed_rewards), np.std(seed_rewards)
            cost = RELATIVE_COST.get(condition, '?')
            print(f"  {condition:<12} score={m:.2f}±{s:.2f}  cost={cost}")

    # Save
    out_path = logdir / f'annotator_sensitivity_{args.env}.json'
    with open(out_path, 'w') as f:
        json.dump({k: {'mean': float(np.mean(v)), 'std': float(np.std(v)),
                        'n': len(v), 'rewards': v}
                    for k, v in all_results.items()}, f, indent=2)
    print(f"Saved to {out_path}")

    # Plot
    plot_results(all_results, logdir, args.env)

    # LaTeX table
    print("\n% Extended Table 4 — Annotator model sensitivity")
    print(r"\begin{tabular}{llr}")
    print(r"\toprule Annotator & Type & Score \\ \midrule")
    for c, vals in all_results.items():
        m, s = np.mean(vals), np.std(vals)
        cost = RELATIVE_COST.get(c, '')
        label = ANNOTATOR_LABELS.get(c, c)
        print(f"{label} & {cost} & ${m:.1f} \\pm {s:.1f}$ \\\\")
    print(r"\bottomrule\end{tabular}")


if __name__ == '__main__':
    main()
