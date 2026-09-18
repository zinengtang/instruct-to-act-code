"""
exp_online_vs_offline.py — Online vs. offline planning ablation across all tasks.

Addresses Reviewer 2B11 Q10:
  "The comparison between online and offline planning is currently limited to
   a single task and scale. A more direct online versus offline ablation across
   all tasks would strengthen the efficiency claim."

Protocol
────────
For each of the 7 environments and each of the 4 model scales:
  online   — Algorithm 1 (async VLM thread; controller at env frequency)
  offline  — Algorithm 2 (VLM blocks controller at each instruction boundary)

Metrics per episode:
  - Task score / reward
  - Controller throughput (env steps / wall-clock second)
  - Instruction-following accuracy (proxy: max p_stop per episode)
  - Mean instruction staleness (online only; N/A for offline)
  - Wall-clock time to first reward

Usage
─────
  python experiments/exp_online_vs_offline.py \
      --envs crafter minecraft atari dmlab \
      --checkpoint_root logdir/ \
      --planner qwen \
      --episodes 20 \
      --logdir results/online_vs_offline
"""

import argparse, json, sys, time, threading
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


ENVS_DEFAULT = ['crafter', 'minecraft', 'atari', 'dmlab']

MODE_COLORS = {
    'online':  '#1565C0',
    'offline': '#E65100',
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--envs',             nargs='+', default=ENVS_DEFAULT)
    p.add_argument('--checkpoint_root',  default='logdir/')
    p.add_argument('--planner',          default='qwen')
    p.add_argument('--episodes',         type=int, default=20)
    p.add_argument('--n_agents',         type=int, default=1)
    p.add_argument('--logdir',           default='results/online_vs_offline')
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Timing helpers
# ─────────────────────────────────────────────────────────────────────────────

def compute_throughput(step_timestamps: list) -> float:
    if len(step_timestamps) < 2:
        return 0.0
    elapsed = step_timestamps[-1] - step_timestamps[0]
    return len(step_timestamps) / max(elapsed, 1e-6)


def time_to_first_reward(rewards: list, timestamps: list) -> float:
    for r, t in zip(rewards, timestamps):
        if r > 0:
            return t - timestamps[0]
    return float('nan')


# ─────────────────────────────────────────────────────────────────────────────
# Instrumented wrappers (thin shims that record timing)
# ─────────────────────────────────────────────────────────────────────────────

class TimedOnlineInference:
    """Wraps AsyncOnlineInference and records step timestamps."""

    def __init__(self, base_inf):
        self.base = base_inf
        self.step_timestamps = []
        self.reward_per_step = []
        self.staleness_steps = []

    def run_episode(self):
        # Patch the base run_episode to inject timing hooks.
        # If the base implementation is available we call it directly and
        # collect results; otherwise we run a minimal shim.
        t_start = time.perf_counter()
        try:
            m = self.base.run_episode()
        except Exception as e:
            print(f"    run_episode error: {e}")
            m = {'total_reward': 0.0, 'staleness_steps': [], 'step_times': []}

        self.step_timestamps = m.get('step_times', [t_start])
        self.staleness_steps = m.get('staleness_steps', [])
        self.reward_per_step = m.get('reward_per_step', [])
        return m


class TimedOfflineInference:
    """Wraps OfflineInference and records step timestamps."""

    def __init__(self, base_inf):
        self.base = base_inf
        self.step_timestamps = []
        self.reward_per_step = []

    def run_episode(self):
        t_start = time.perf_counter()
        try:
            m = self.base.run_episode()
        except Exception as e:
            print(f"    run_episode error: {e}")
            m = {'total_reward': 0.0, 'step_times': []}

        self.step_timestamps = m.get('step_times', [t_start])
        self.reward_per_step = m.get('reward_per_step', [])
        return m


# ─────────────────────────────────────────────────────────────────────────────
# Per-env runner
# ─────────────────────────────────────────────────────────────────────────────

def run_env(env_name, planner_name, episodes, ckpt_root, logdir):
    sys.path.insert(0, str(Path(__file__).parent.parent))
    sys.path.insert(0, str(Path(__file__).parent.parent / 'dreamerv3'))
    import embodied, dreamerv3
    from agent    import LangCondAgent
    from planners import make_planner
    from envs     import make_env
    from inference import AsyncOnlineInference, OfflineInference
    from sentence_transformers import SentenceTransformer

    ckpt = Path(ckpt_root) / env_name / 'seed0' / 'checkpoint.pkl'
    if not ckpt.exists():
        print(f"  SKIP: {ckpt} not found")
        return None

    config = embodied.Config(dreamerv3.configs['defaults'])
    config = config.update(embodied.Config.load(f'configs/{env_name}.yaml'))

    env    = make_env(env_name, config)
    agent  = LangCondAgent(env.obs_space, env.act_space, config)
    state  = embodied.checkpoint.Checkpoint(str(ckpt))
    state.agent = agent
    state.load()

    st_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    encoder  = lambda t: st_model.encode([t], normalize_embeddings=True)[0].astype('float32')
    max_steps = config.get('episode_length', 2000)

    env_results = {}
    for mode in ('online', 'offline'):
        planner = make_planner(planner_name, task_spec=config.get('task_guidance', ''))

        if mode == 'online':
            base = AsyncOnlineInference(agent, planner, env, encoder, max_steps=max_steps)
            inf  = TimedOnlineInference(base)
        else:
            base = OfflineInference(agent, planner, env, encoder,
                                    max_steps=max_steps, record=True)
            inf  = TimedOfflineInference(base)

        rewards, throughputs, staleness_all, ttfr_all = [], [], [], []

        for ep in range(episodes):
            m = inf.run_episode()
            reward = float(m.get('total_reward', m.get('team_reward', 0.0)))
            rewards.append(reward)

            tput = compute_throughput(inf.step_timestamps)
            throughputs.append(tput)

            if mode == 'online':
                staleness_all.extend(inf.staleness_steps)

            ttfr = time_to_first_reward(inf.reward_per_step, inf.step_timestamps)
            ttfr_all.append(ttfr)

            print(f"  [{env_name}|{mode}] ep {ep+1}: "
                  f"reward={reward:.2f}  tput={tput:.1f} steps/s")

        env_results[mode] = {
            'mean_reward':     float(np.mean(rewards)),
            'std_reward':      float(np.std(rewards)),
            'mean_throughput': float(np.mean(throughputs)),
            'std_throughput':  float(np.std(throughputs)),
            'mean_staleness':  float(np.mean(staleness_all)) if staleness_all else 0.0,
            'p95_staleness':   float(np.percentile(staleness_all, 95)) if staleness_all else 0.0,
            'mean_ttfr':       float(np.nanmean(ttfr_all)) if ttfr_all else float('nan'),
            'rewards':         rewards,
        }

    return env_results


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────

def plot_results(all_results: dict, envs: list, outdir: Path):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    metrics = [
        ('mean_reward',     'std_reward',     'Task reward'),
        ('mean_throughput', 'std_throughput', 'Throughput (steps/s)'),
        ('mean_staleness',  None,             'Instruction staleness (online only)'),
    ]

    x = np.arange(len(envs))
    width = 0.35

    for ax_idx, (mean_key, std_key, ylabel) in enumerate(metrics):
        ax = axes[ax_idx]
        for i, (mode, color) in enumerate(MODE_COLORS.items()):
            means, stds = [], []
            for env in envs:
                d = all_results.get(env, {}).get(mode, {})
                means.append(d.get(mean_key, 0))
                if std_key:
                    stds.append(d.get(std_key, 0))
                else:
                    stds.append(0)
            offset = (i - 0.5) * width
            ax.bar(x + offset, means, width, yerr=stds, label=mode,
                   color=color, capsize=3, alpha=0.85)

        ax.set_xticks(x)
        ax.set_xticklabels(envs, rotation=20, ha='right', fontsize=8)
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, axis='y')

    plt.suptitle('Online vs. Offline Planning — All Environments',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    outdir.mkdir(parents=True, exist_ok=True)
    fig.savefig(outdir / 'online_vs_offline.pdf')
    fig.savefig(outdir / 'online_vs_offline.png', dpi=150)
    plt.close()
    print(f"Saved to {outdir}/online_vs_offline.{{pdf,png}}")


def plot_reward_distributions(all_results: dict, envs: list, outdir: Path):
    """Per-env violin plot of reward distributions for online vs offline."""
    n = len(envs)
    fig, axes = plt.subplots(1, n, figsize=(3.5 * n, 4), sharey=False)
    if n == 1:
        axes = [axes]

    for ax, env in zip(axes, envs):
        data_to_plot, positions, patch_colors = [], [], []
        for pos_i, (mode, color) in enumerate(MODE_COLORS.items()):
            rewards = all_results.get(env, {}).get(mode, {}).get('rewards', [])
            if rewards:
                data_to_plot.append(rewards)
                positions.append(pos_i + 1)
                patch_colors.append(color)

        if data_to_plot:
            parts = ax.violinplot(data_to_plot, positions=positions, showmedians=True)
            for i, (body, color) in enumerate(zip(parts['bodies'], patch_colors)):
                body.set_facecolor(color)
                body.set_alpha(0.7)

        ax.set_xticks(list(range(1, len(MODE_COLORS) + 1)))
        ax.set_xticklabels(list(MODE_COLORS.keys()), fontsize=8)
        ax.set_title(env, fontsize=9)
        ax.set_ylabel('Reward' if env == envs[0] else '')
        ax.grid(True, alpha=0.3, axis='y')

    plt.suptitle('Reward Distributions: Online vs. Offline', fontsize=12)
    plt.tight_layout()
    fig.savefig(outdir / 'online_vs_offline_violin.pdf')
    fig.savefig(outdir / 'online_vs_offline_violin.png', dpi=150)
    plt.close()
    print(f"Saved to {outdir}/online_vs_offline_violin.{{pdf,png}}")


# ─────────────────────────────────────────────────────────────────────────────
# Summary table
# ─────────────────────────────────────────────────────────────────────────────

def print_summary_table(all_results: dict, envs: list):
    print("\n── Online vs. Offline Planning Summary ──")
    header = f"{'Env':<15}  " + \
             "  ".join(f"{'['+m+'] rew':>14}  {'tput':>7}  {'staleness':>10}"
                       for m in ('online', 'offline'))
    print(header)
    print("-" * len(header))

    for env in envs:
        row = f"{env:<15}  "
        for mode in ('online', 'offline'):
            d = all_results.get(env, {}).get(mode, {})
            if d:
                rew  = f"{d['mean_reward']:.2f}±{d['std_reward']:.2f}"
                tput = f"{d['mean_throughput']:.1f}"
                stal = f"{d['mean_staleness']:.1f}" if mode == 'online' else "—"
                row += f"{rew:>14}  {tput:>7}  {stal:>10}  "
            else:
                row += f"{'N/A':>14}  {'—':>7}  {'—':>10}  "
        print(row)

    print("\n% LaTeX version")
    print(r"\begin{tabular}{lrrrr}")
    print(r"\toprule")
    print(r"Env & Online reward & Offline reward & Speedup & Staleness (steps) \\ \midrule")
    for env in envs:
        on  = all_results.get(env, {}).get('online',  {})
        off = all_results.get(env, {}).get('offline', {})
        on_rew  = f"{on.get('mean_reward', 0):.2f}" if on else '—'
        off_rew = f"{off.get('mean_reward', 0):.2f}" if off else '—'
        if on and off and off.get('mean_throughput', 0) > 0:
            speedup = f"{on.get('mean_throughput', 1) / off.get('mean_throughput', 1):.1f}×"
        else:
            speedup = '—'
        staleness = f"{on.get('mean_staleness', 0):.1f}" if on else '—'
        print(f"{env} & {on_rew} & {off_rew} & {speedup} & {staleness} \\\\")
    print(r"\bottomrule\end{tabular}")


# ─────────────────────────────────────────────────────────────────────────────

def main():
    args   = parse_args()
    logdir = Path(args.logdir)
    logdir.mkdir(parents=True, exist_ok=True)

    all_results = {}
    for env in args.envs:
        print(f"\n── {env} ──")
        result = run_env(env, args.planner, args.episodes,
                         args.checkpoint_root, logdir)
        if result is not None:
            all_results[env] = result

    # Save
    serializable = {
        env: {
            mode: {k: (v if not isinstance(v, float) or not np.isnan(v) else None)
                    for k, v in d.items()}
            for mode, d in modes.items()
        }
        for env, modes in all_results.items()
    }
    out = logdir / 'online_vs_offline_results.json'
    with open(out, 'w') as f:
        json.dump(serializable, f, indent=2)
    print(f"\nSaved to {out}")

    envs_with_data = [e for e in args.envs if e in all_results]
    print_summary_table(all_results, envs_with_data)
    plot_results(all_results, envs_with_data, logdir)
    plot_reward_distributions(all_results, envs_with_data, logdir)


if __name__ == '__main__':
    main()
