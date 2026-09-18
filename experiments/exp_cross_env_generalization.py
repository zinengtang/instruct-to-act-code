"""
exp_cross_env_generalization.py — Cross-environment generalisation.

Addresses Reviewer 2B11 Q8:
  "Can the same controller generalise across multiple environments, or is one
   controller trained per environment in all experiments? If only per-env
   controllers are evaluated, could the authors add a cross-env generalisation
   experiment?"

Protocol
────────
Two conditions are compared:

  per_env        — Separate controller trained for each environment (paper default).
                   Loaded from individual env-specific checkpoints.

  joint          — Single controller trained jointly on all environments.
                   Trained via train.py with `--config configs/joint.yaml` (generated
                   here if absent).  Evaluated on each env separately.

  zero_shot      — Per-env controller tested on a held-out environment it was
                   never trained on (all src→target pairs, triangular matrix).

For each (source_train, target_eval) pair we report:
  - Task score (mean ± std, 100 episodes)
  - Drop from per-env baseline (Δ)

We also train the joint controller on a subset of environments (joint_subset)
and test on the held-out env to probe whether the joint model can generalise.

Usage
─────
  python experiments/exp_cross_env_generalization.py \
      --train_envs   crafter minecraft atari dmlab \
      --eval_envs    crafter minecraft atari dmlab \
      --checkpoint_root  logdir/ \
      --joint_ckpt       logdir/joint/seed0/checkpoint.pkl \
      --planner      qwen \
      --episodes     50 \
      --logdir       results/cross_env_generalization

To train the joint controller first:
  python train.py \
      --config configs/joint.yaml configs/base.yaml \
      --logdir logdir/joint/seed0
"""

import argparse, json, sys
from pathlib import Path
from itertools import product

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


ENVS_DEFAULT = ['crafter', 'minecraft', 'atari', 'dmlab']


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--train_envs',    nargs='+', default=ENVS_DEFAULT)
    p.add_argument('--eval_envs',     nargs='+', default=ENVS_DEFAULT)
    p.add_argument('--checkpoint_root', default='logdir/')
    p.add_argument('--joint_ckpt',    default='logdir/joint/seed0/checkpoint.pkl')
    p.add_argument('--planner',       default='qwen')
    p.add_argument('--episodes',      type=int, default=50)
    p.add_argument('--logdir',        default='results/cross_env_generalization')
    p.add_argument('--skip_train',    action='store_true',
                   help='Skip joint training; load existing joint checkpoint.')
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Joint training config
# ─────────────────────────────────────────────────────────────────────────────

JOINT_YAML_TEMPLATE = """\
# Joint multi-environment training config.
# Inherits from base.yaml; override fields specific to joint training.
task: joint
envs: {envs}
train_steps: 5_000_000
batch_size: 32
replay_size: 2_000_000
eval_every: 100_000
"""


def ensure_joint_config(envs: list, configs_dir: Path):
    path = configs_dir / 'joint.yaml'
    if not path.exists():
        path.write_text(JOINT_YAML_TEMPLATE.format(envs=envs))
        print(f"  Created {path}")
    return path


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation helper
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(ckpt_path: str, eval_env_name: str, planner_name: str,
             episodes: int) -> dict | None:
    """Load a checkpoint and evaluate on eval_env_name."""
    sys.path.insert(0, str(Path(__file__).parent.parent))
    sys.path.insert(0, str(Path(__file__).parent.parent / 'dreamerv3'))
    import embodied, dreamerv3
    from agent    import LangCondAgent
    from planners import make_planner
    from envs     import make_env
    from inference import OfflineInference
    from sentence_transformers import SentenceTransformer

    ckpt = Path(ckpt_path)
    if not ckpt.exists():
        print(f"  SKIP: {ckpt} not found")
        return None

    config = embodied.Config(dreamerv3.configs['defaults'])
    config_path = Path(__file__).parent.parent / 'configs' / f'{eval_env_name}.yaml'
    if config_path.exists():
        config = config.update(embodied.Config.load(str(config_path)))

    env    = make_env(eval_env_name, config)
    agent  = LangCondAgent(env.obs_space, env.act_space, config)
    state  = embodied.checkpoint.Checkpoint(str(ckpt))
    state.agent = agent
    state.load()

    st_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    encoder  = lambda t: st_model.encode([t], normalize_embeddings=True)[0].astype('float32')

    planner = make_planner(planner_name, task_spec=config.get('task_guidance', ''))
    inf     = OfflineInference(agent, planner, env, encoder,
                               max_steps=config.get('episode_length', 18_000),
                               record=True)

    rewards = []
    for ep in range(episodes):
        m = inf.run_episode()
        rewards.append(float(m.get('total_reward', 0.0)))
        print(f"  [{eval_env_name}] ep {ep+1}: reward={rewards[-1]:.2f}")

    return {
        'mean_reward': float(np.mean(rewards)),
        'std_reward':  float(np.std(rewards)),
        'rewards':     rewards,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────

def plot_heatmap(matrix: dict, train_envs: list, eval_envs: list,
                 title: str, outdir: Path, fname: str):
    """
    matrix[train_env][eval_env] = mean_reward
    Rows = trained on, Cols = evaluated on.
    """
    n_train = len(train_envs)
    n_eval  = len(eval_envs)
    data = np.full((n_train, n_eval), np.nan)

    for i, te in enumerate(train_envs):
        for j, ee in enumerate(eval_envs):
            val = matrix.get(te, {}).get(ee, {}).get('mean_reward', None)
            if val is not None:
                data[i, j] = val

    fig, ax = plt.subplots(figsize=(max(6, n_eval * 1.2), max(4, n_train * 0.9)))
    vmin = np.nanmin(data) if not np.all(np.isnan(data)) else 0
    vmax = np.nanmax(data) if not np.all(np.isnan(data)) else 1
    im = ax.imshow(data, aspect='auto', cmap='RdYlGn', vmin=vmin, vmax=vmax)
    plt.colorbar(im, ax=ax, label='Mean reward')

    ax.set_xticks(range(n_eval))
    ax.set_yticks(range(n_train))
    ax.set_xticklabels(eval_envs, rotation=30, ha='right', fontsize=9)
    ax.set_yticklabels(train_envs, fontsize=9)
    ax.set_xlabel('Evaluation environment')
    ax.set_ylabel('Training environment (checkpoint)')
    ax.set_title(title)

    for i in range(n_train):
        for j in range(n_eval):
            if not np.isnan(data[i, j]):
                ax.text(j, i, f'{data[i, j]:.1f}',
                        ha='center', va='center', fontsize=8,
                        color='white' if data[i, j] < (vmin + vmax) / 2 else 'black')

    plt.tight_layout()
    outdir.mkdir(parents=True, exist_ok=True)
    fig.savefig(outdir / f'{fname}.pdf')
    fig.savefig(outdir / f'{fname}.png', dpi=150)
    plt.close()
    print(f"Saved to {outdir}/{fname}.{{pdf,png}}")


def plot_comparison_bars(per_env: dict, joint: dict, envs: list, outdir: Path):
    """Bar chart comparing per-env and joint controller per evaluation env."""
    x     = np.arange(len(envs))
    width = 0.35
    fig, ax = plt.subplots(figsize=(max(7, len(envs) * 1.5), 5))

    per_means  = [per_env.get(e, {}).get('mean_reward', 0) for e in envs]
    per_stds   = [per_env.get(e, {}).get('std_reward',  0) for e in envs]
    joint_means= [joint.get(e, {}).get('mean_reward', 0)   for e in envs]
    joint_stds = [joint.get(e, {}).get('std_reward',  0)   for e in envs]

    ax.bar(x - width / 2, per_means,   width, yerr=per_stds,   label='Per-env (paper)',
           color='#1565C0', capsize=4, alpha=0.85)
    ax.bar(x + width / 2, joint_means, width, yerr=joint_stds, label='Joint controller',
           color='#E65100', capsize=4, alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(envs, rotation=15, ha='right')
    ax.set_ylabel('Task reward')
    ax.set_title('Per-env vs. Joint Controller')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    fig.savefig(outdir / 'per_vs_joint.pdf')
    fig.savefig(outdir / 'per_vs_joint.png', dpi=150)
    plt.close()
    print(f"Saved to {outdir}/per_vs_joint.{{pdf,png}}")


# ─────────────────────────────────────────────────────────────────────────────
# LaTeX table
# ─────────────────────────────────────────────────────────────────────────────

def print_latex_table(per_env: dict, joint: dict, zero_shot: dict,
                       train_envs: list, eval_envs: list):
    print("\n% Cross-environment generalisation table")
    eval_header = ' & '.join(eval_envs)
    print(r"\begin{tabular}{ll" + 'r' * len(eval_envs) + "}")
    print(r"\toprule")
    print(f"Mode & Train & {eval_header} \\\\ \\midrule")

    # Per-env (diagonal)
    for ee in eval_envs:
        d = per_env.get(ee, {})
        vals = [f"${d.get('mean_reward', 0):.1f}$" if ee == te else '—'
                for te in eval_envs]
        row = ' & '.join(vals)
        print(f"per-env & {ee} & {row} \\\\")

    print(r"\midrule")

    # Joint
    vals = [f"${joint.get(ee, {}).get('mean_reward', 0):.1f}\\pm"
            f"{joint.get(ee, {}).get('std_reward', 0):.1f}$"
            if joint.get(ee) else '—' for ee in eval_envs]
    print(f"joint & all & {' & '.join(vals)} \\\\")

    print(r"\midrule")

    # Zero-shot cross-env
    for te in train_envs:
        vals = []
        for ee in eval_envs:
            if te == ee:
                vals.append('—')
            else:
                d = zero_shot.get(te, {}).get(ee, {})
                if d:
                    vals.append(f"${d['mean_reward']:.1f}$")
                else:
                    vals.append('—')
        print(f"zero-shot & {te} & {' & '.join(vals)} \\\\")

    print(r"\bottomrule\end{tabular}")


# ─────────────────────────────────────────────────────────────────────────────

def main():
    args   = parse_args()
    logdir = Path(args.logdir)
    logdir.mkdir(parents=True, exist_ok=True)
    configs_dir = Path(__file__).parent.parent / 'configs'

    # ── 1. Per-env baselines ─────────────────────────────────────────────────
    print("\n═══ Per-env baselines ═══")
    per_env_results = {}
    for env in args.eval_envs:
        ckpt = Path(args.checkpoint_root) / env / 'seed0' / 'checkpoint.pkl'
        print(f"\n── Per-env: train={env} eval={env} ──")
        res = evaluate(str(ckpt), env, args.planner, args.episodes)
        if res:
            per_env_results[env] = res

    # ── 2. Joint controller ──────────────────────────────────────────────────
    print("\n═══ Joint controller ═══")
    if not args.skip_train:
        ensure_joint_config(args.train_envs, configs_dir)

    joint_results = {}
    for env in args.eval_envs:
        print(f"\n── Joint: eval={env} ──")
        res = evaluate(args.joint_ckpt, env, args.planner, args.episodes)
        if res:
            joint_results[env] = res

    # ── 3. Zero-shot cross-env ───────────────────────────────────────────────
    print("\n═══ Zero-shot transfer ═══")
    zero_shot_matrix = {te: {} for te in args.train_envs}
    for train_env in args.train_envs:
        for eval_env in args.eval_envs:
            if train_env == eval_env:
                continue
            ckpt = Path(args.checkpoint_root) / train_env / 'seed0' / 'checkpoint.pkl'
            print(f"\n── Zero-shot: train={train_env} eval={eval_env} ──")
            res = evaluate(str(ckpt), eval_env, args.planner, args.episodes)
            if res:
                zero_shot_matrix[train_env][eval_env] = res

    # ── Save ─────────────────────────────────────────────────────────────────
    out = logdir / 'cross_env_generalization_results.json'
    with open(out, 'w') as f:
        json.dump({
            'per_env':   per_env_results,
            'joint':     joint_results,
            'zero_shot': zero_shot_matrix,
        }, f, indent=2)
    print(f"\nSaved to {out}")

    # ── Print summary ─────────────────────────────────────────────────────────
    print("\n── Summary ──")
    print(f"{'Env':<15}  {'Per-env':>10}  {'Joint':>10}  {'Delta':>8}")
    print("-" * 50)
    for env in args.eval_envs:
        pe = per_env_results.get(env, {}).get('mean_reward', float('nan'))
        jt = joint_results.get(env, {}).get('mean_reward', float('nan'))
        delta = jt - pe if not (np.isnan(pe) or np.isnan(jt)) else float('nan')
        pe_s = f"{pe:.2f}" if not np.isnan(pe) else "N/A"
        jt_s = f"{jt:.2f}" if not np.isnan(jt) else "N/A"
        dl_s = f"{delta:+.2f}" if not np.isnan(delta) else "N/A"
        print(f"{env:<15}  {pe_s:>10}  {jt_s:>10}  {dl_s:>8}")

    # ── Plots ─────────────────────────────────────────────────────────────────
    if per_env_results and joint_results:
        plot_comparison_bars(per_env_results, joint_results, args.eval_envs, logdir)

    if zero_shot_matrix:
        plot_heatmap(zero_shot_matrix, args.train_envs, args.eval_envs,
                     'Zero-shot cross-env transfer (reward)',
                     logdir, 'zero_shot_heatmap')

    # Full matrix (per-env on diagonal, zero-shot off-diagonal)
    full_matrix = {te: {} for te in args.train_envs}
    for te in args.train_envs:
        for ee in args.eval_envs:
            if te == ee:
                full_matrix[te][ee] = per_env_results.get(ee, {})
            else:
                full_matrix[te][ee] = zero_shot_matrix.get(te, {}).get(ee, {})

    plot_heatmap(full_matrix, args.train_envs, args.eval_envs,
                 'Cross-env reward matrix\n(diagonal = per-env, off-diagonal = zero-shot)',
                 logdir, 'cross_env_full_matrix')

    # ── LaTeX ─────────────────────────────────────────────────────────────────
    print_latex_table(per_env_results, joint_results, zero_shot_matrix,
                       args.train_envs, args.eval_envs)


if __name__ == '__main__':
    main()
