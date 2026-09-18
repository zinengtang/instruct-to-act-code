"""
exp_multiseed_ci.py — Multi-seed evaluation with confidence intervals.

Addresses Reviewer 2B11 Q3:
  "Can you report standard deviations or confidence intervals for all main
   results? How many random seeds were used for controller training and evaluation?"

Protocol
────────
• Train one controller per (env, seed) for N_SEEDS seeds.
• Evaluate each for 100 episodes.
• Report mean ± std and 95% bootstrap CI per environment.
• Output a LaTeX table matching Table 1 format with uncertainty columns.

Usage
─────
  # Run from instruct_to_act/
  python experiments/exp_multiseed_ci.py \
      --envs crafter minecraft atari dmlab \
      --planner gpt4o \
      --seeds 5 \
      --logdir results/multiseed \
      --from_checkpoints logdir/           # if already trained
"""

import sys, os, argparse, json, subprocess
from pathlib import Path

import numpy as np
import scipy.stats


# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--envs',   nargs='+',
                   default=['crafter', 'minecraft', 'atari', 'dmlab'])
    p.add_argument('--planner', default='gpt4o')
    p.add_argument('--seeds',   type=int, default=5)
    p.add_argument('--episodes', type=int, default=100)
    p.add_argument('--logdir',  default='results/multiseed')
    p.add_argument('--from_checkpoints', default=None,
                   help='Dir containing logdir/<env>/seed<s>/checkpoint.pkl')
    p.add_argument('--train',   action='store_true',
                   help='If set, train controllers (else assume already trained)')
    p.add_argument('--gpus',    nargs='+', default=['0', '1', '2', '3'])
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Training (one job per seed × env)
# ─────────────────────────────────────────────────────────────────────────────

def train_all(args):
    """Launch training jobs for each (env, seed) combination."""
    import itertools
    jobs = list(itertools.product(args.envs, range(args.seeds)))
    print(f"Launching {len(jobs)} training jobs across {args.gpus} GPUs …")

    procs = []
    for idx, (env, seed) in enumerate(jobs):
        gpu  = args.gpus[idx % len(args.gpus)]
        out  = Path(args.logdir) / env / f'seed{seed}'
        cmd  = [
            'python', 'train.py',
            '--config', f'configs/{env}.yaml', 'configs/base.yaml',
            '--logdir',  str(out),
            '--seed',    str(seed),
        ]
        env_vars = {**os.environ, 'CUDA_VISIBLE_DEVICES': gpu}
        proc = subprocess.Popen(cmd, env=env_vars)
        procs.append((env, seed, proc))
        print(f"  [{gpu}] {env} seed={seed} → {out}")

    for env, seed, proc in procs:
        proc.wait()
        if proc.returncode != 0:
            print(f"  WARNING: {env} seed={seed} exited with {proc.returncode}")


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_checkpoint(env, seed, planner, episodes, ckpt_dir, out_dir):
    """Run evaluate.py for one (env, seed) pair and return rewards list."""
    ckpt = Path(ckpt_dir) / env / f'seed{seed}' / 'checkpoint.pkl'
    if not ckpt.exists():
        print(f"  MISSING checkpoint: {ckpt}")
        return None

    out = Path(out_dir) / env / f'seed{seed}'
    out.mkdir(parents=True, exist_ok=True)

    cmd = [
        'python', 'evaluate.py',
        '--config', f'configs/{env}.yaml', 'configs/base.yaml',
        '--checkpoint', str(ckpt),
        '--planner',    planner,
        '--episodes',   str(episodes),
        '--seed',       str(seed),
        '--outdir',     str(out),
    ]
    subprocess.run(cmd, check=True)

    # Load results
    result_files = list(out.glob('*_episodes.json'))
    if not result_files:
        return None
    with open(result_files[0]) as f:
        data = json.load(f)
    return data['rewards']


# ─────────────────────────────────────────────────────────────────────────────
# Statistics + reporting
# ─────────────────────────────────────────────────────────────────────────────

def bootstrap_ci(data, n_boot=10_000, ci=0.95):
    """Non-parametric bootstrap 95% CI for the mean."""
    arr   = np.array(data)
    means = [np.random.choice(arr, len(arr), replace=True).mean() for _ in range(n_boot)]
    lo    = np.percentile(means, (1 - ci) / 2 * 100)
    hi    = np.percentile(means, (1 + ci) / 2 * 100)
    return arr.mean(), lo, hi


def compute_stats(all_rewards: list) -> dict:
    """
    all_rewards : list of reward lists, one per seed.
    Returns dict with mean, std, sem, ci95_lo, ci95_hi.
    """
    flat = [r for rewards in all_rewards for r in rewards]
    arr  = np.array(flat)
    mean, lo, hi = bootstrap_ci(flat)
    # Also report seed-level stats (variance across seeds)
    seed_means = [np.mean(r) for r in all_rewards]
    return {
        'n_episodes':    len(flat),
        'n_seeds':       len(all_rewards),
        'mean':          float(mean),
        'std':           float(arr.std()),
        'sem':           float(arr.std() / np.sqrt(len(flat))),
        'ci95_lo':       float(lo),
        'ci95_hi':       float(hi),
        'seed_means':    [float(m) for m in seed_means],
        'seed_std':      float(np.std(seed_means)),
    }


def print_latex_table(results: dict):
    """Print a LaTeX table with mean ± std and 95% CI."""
    envs = list(results.keys())
    print("\n% LaTeX table — Table 1 with uncertainty estimates")
    print(r"\begin{table}[h]")
    print(r"\centering")
    print(r"\begin{tabular}{lrrr}")
    print(r"\toprule")
    print(r"Environment & Mean $\pm$ Std & 95\% CI & \#seeds $\times$ \#eps \\")
    print(r"\midrule")
    for env in envs:
        s = results[env]
        print(f"{env:<15} & ${s['mean']:.1f} \\pm {s['std']:.1f}$ "
              f"& $[{s['ci95_lo']:.1f}, {s['ci95_hi']:.1f}]$ "
              f"& ${s['n_seeds']} \\times {s['n_episodes'] // s['n_seeds']}$ \\\\")
    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\caption{Multi-seed evaluation results. Each entry reports mean "
          r"$\pm$ std over all episodes and seeds, with non-parametric bootstrap "
          r"95\% confidence intervals.}")
    print(r"\end{table}")


# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    logdir = Path(args.logdir)
    logdir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = args.from_checkpoints or str(logdir)

    if args.train:
        train_all(args)

    # Evaluate all (env, seed) pairs
    all_results = {}
    for env in args.envs:
        seed_rewards = []
        for seed in range(args.seeds):
            rewards = evaluate_checkpoint(
                env, seed, args.planner, args.episodes, ckpt_dir,
                str(logdir / 'eval')
            )
            if rewards is not None:
                seed_rewards.append(rewards)
                print(f"  {env} seed={seed}: mean={np.mean(rewards):.2f} "
                      f"± {np.std(rewards):.2f}")
        if seed_rewards:
            all_results[env] = compute_stats(seed_rewards)

    # Save JSON
    out_path = logdir / 'multiseed_ci_results.json'
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved results to {out_path}")

    # Print LaTeX
    print_latex_table(all_results)

    # Print summary
    print("\n── Summary ──")
    for env, s in all_results.items():
        print(f"  {env:<15}  {s['mean']:.2f} ± {s['std']:.2f}  "
              f"(CI [{s['ci95_lo']:.2f}, {s['ci95_hi']:.2f}])  "
              f"seed_std={s['seed_std']:.2f}")


if __name__ == '__main__':
    main()
