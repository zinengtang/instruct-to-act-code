"""
exp_instruction_horizon.py — Instruction horizon and length sensitivity.

Addresses Reviewer EkK5 Q5:
  "How does performance change as the instruction horizon varies? Are there
   tasks where very short or very long instructions break the interface?"

And Reviewer 2B11 Q13:
  "Does the model learn to request instructions of appropriate length, or is
   there a mismatch between the horizon the controller expects and the horizon
   the planner generates?"

Two sweeps
──────────

Sweep A — Token length:
  Truncate or pad planner-generated instructions to fixed token budgets:
    tiny   : 4–6 tokens
    short  : 6–12 tokens  (paper default ≈ 9)
    medium : 12–18 tokens
    long   : 18–30 tokens
    free   : no truncation (planner generates freely)

Sweep B — Fixed cadence K (instruction replaces every K controller steps,
  ignoring the p_stop head; tests whether learned stop boundaries matter):
    K=8, K=16, K=32 (paper ≈ adaptive), K=64, K=adaptive (paper default)

For each (env, sweep, condition) we run `--episodes` episodes and report
task reward, instruction-following accuracy (proxy), and mean instruction
length consumed.

Usage
─────
  python experiments/exp_instruction_horizon.py \
      --envs crafter minecraft dmlab \
      --checkpoint_root logdir/ \
      --planner gpt4o \
      --episodes 20 \
      --logdir results/instruction_horizon
"""

import argparse, json, sys, random, re
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


# ─────────────────────────────────────────────────────────────────────────────
# Instruction length manipulation
# ─────────────────────────────────────────────────────────────────────────────

LENGTH_BUCKETS = {
    'tiny':   (4,  6),
    'short':  (6,  12),
    'medium': (12, 18),
    'long':   (18, 30),
    'free':   (None, None),
}

CADENCE_VALUES = {
    'K=8':      8,
    'K=16':     16,
    'K=32':     32,
    'K=64':     64,
    'adaptive': None,   # use p_stop head
}


def truncate_or_pad(instruction: str, min_tokens: int, max_tokens: int) -> str:
    """Truncate to max_tokens words, or pad with 'continue' if too short."""
    if min_tokens is None:
        return instruction
    words = instruction.split()
    if len(words) > max_tokens:
        words = words[:max_tokens]
    while len(words) < min_tokens:
        words.append('and continue')
    return ' '.join(words)


# ─────────────────────────────────────────────────────────────────────────────
# Planner wrappers
# ─────────────────────────────────────────────────────────────────────────────

class LengthConstrainedPlanner:
    """Wraps a base planner and enforces a token-length budget on instructions."""

    def __init__(self, base_planner, length_bucket: str):
        self.base   = base_planner
        self.bucket = length_bucket
        self.min_t, self.max_t = LENGTH_BUCKETS[length_bucket]
        self.issued_lengths = []

    def step(self, obs, plan, memory, p_stop, chat_context=''):
        instr, plan, memory = self.base.step(obs, plan, memory, p_stop, chat_context)
        if instr is not None:
            instr = truncate_or_pad(instr, self.min_t, self.max_t)
            self.issued_lengths.append(len(instr.split()))
        return instr, plan, memory

    def emit(self, obs, plan, memory, chat_context=''):
        instr, plan, memory = self.base.emit(obs, plan, memory, chat_context)
        instr = truncate_or_pad(instr, self.min_t, self.max_t)
        self.issued_lengths.append(len(instr.split()))
        return instr, plan, memory

    def advance(self, obs, plan, memory):
        return self.base.advance(obs, plan, memory)


class FixedCadenceInference:
    """
    Wraps OfflineInference but replaces instructions every K steps
    instead of using the p_stop head.  K=None falls back to p_stop.
    """

    def __init__(self, controller, planner, env, lang_encoder, K, max_steps=2000):
        self.ctrl    = controller
        self.planner = planner
        self.env     = env
        self.encode  = lang_encoder
        self.K       = K
        self.max_steps = max_steps
        self.issued_lengths = []

    def run_episode(self):
        import numpy as np

        ctrl_state = self.ctrl.init_policy(batch_size=1)
        lang_size  = getattr(self.ctrl, '_lang_size', 384)
        null_embed = np.zeros(lang_size, dtype=np.float32)
        cur_embed  = null_embed.copy()

        obs, _  = self.env.reset()
        memory, plan = {}, None
        total_reward = 0.0
        step_rewards = []

        instr, plan, memory = self.planner.emit(obs=obs, plan=plan, memory=memory)
        cur_embed = self.encode(instr)
        self.issued_lengths.append(len(instr.split()))

        for step in range(self.max_steps):
            obs_b = {k: np.expand_dims(v, 0) for k, v in obs.items()}
            obs_b['lang_embed'] = cur_embed[np.newaxis]

            ctrl_state, action, extra = self.ctrl.policy(ctrl_state, obs_b, mode='eval')
            p_stop = float(extra.get('log/p_stop', extra.get('p_stop', 0.0)))

            action_np = np.array(action[0])
            obs, reward, terminated, truncated, _ = self.env.step(action_np)
            total_reward += float(reward)
            step_rewards.append(float(reward))

            # Replace instruction
            replace_now = False
            if self.K is not None:
                replace_now = ((step + 1) % self.K == 0)
            else:
                replace_now = p_stop > 0.5

            if replace_now:
                instr, plan, memory = self.planner.emit(obs=obs, plan=plan, memory=memory)
                cur_embed = self.encode(instr)
                self.issued_lengths.append(len(instr.split()))

            if terminated or truncated:
                break

        return {
            'total_reward': total_reward,
            'reward_per_step': step_rewards,
            'instructions_issued': len(self.issued_lengths),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Per-condition runner
# ─────────────────────────────────────────────────────────────────────────────

def run_length_sweep(env_name, condition, planner_name, episodes, ckpt_root):
    sys.path.insert(0, str(Path(__file__).parent.parent))
    sys.path.insert(0, str(Path(__file__).parent.parent / 'dreamerv3'))
    import embodied, dreamerv3
    from agent    import LangCondAgent
    from planners import make_planner
    from envs     import make_env
    from inference import OfflineInference
    from sentence_transformers import SentenceTransformer

    ckpt = Path(ckpt_root) / env_name / 'seed0' / 'checkpoint.pkl'
    if not ckpt.exists():
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
    max_steps = config.get('episode_length', 18_000)

    base_planner = make_planner(planner_name, task_spec=config.get('task_guidance', ''))
    planner      = LengthConstrainedPlanner(base_planner, condition)
    inf          = OfflineInference(agent, planner, env, encoder,
                                    max_steps=max_steps, record=True)

    rewards, n_instrs = [], []
    for ep in range(episodes):
        m = inf.run_episode()
        rewards.append(float(m.get('total_reward', 0.0)))
        n_instrs.append(int(m.get('instructions_issued', 0)))
        print(f"  [{env_name}|length={condition}] ep {ep+1}: "
              f"reward={rewards[-1]:.2f}  n_instrs={n_instrs[-1]}")

    issued = planner.issued_lengths
    return {
        'mean_reward':   float(np.mean(rewards)),
        'std_reward':    float(np.std(rewards)),
        'mean_n_instrs': float(np.mean(n_instrs)),
        'mean_len':      float(np.mean(issued)) if issued else 0.0,
        'std_len':       float(np.std(issued))  if issued else 0.0,
        'rewards':       rewards,
    }


def run_cadence_sweep(env_name, condition, planner_name, episodes, ckpt_root):
    sys.path.insert(0, str(Path(__file__).parent.parent))
    sys.path.insert(0, str(Path(__file__).parent.parent / 'dreamerv3'))
    import embodied, dreamerv3
    from agent    import LangCondAgent
    from planners import make_planner
    from envs     import make_env
    from sentence_transformers import SentenceTransformer

    ckpt = Path(ckpt_root) / env_name / 'seed0' / 'checkpoint.pkl'
    if not ckpt.exists():
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
    max_steps = config.get('episode_length', 18_000)

    K       = CADENCE_VALUES[condition]
    planner = make_planner(planner_name, task_spec=config.get('task_guidance', ''))
    inf     = FixedCadenceInference(agent, planner, env, encoder,
                                    K=K, max_steps=max_steps)

    rewards, n_instrs = [], []
    for ep in range(episodes):
        m = inf.run_episode()
        rewards.append(float(m.get('total_reward', 0.0)))
        n_instrs.append(int(m.get('instructions_issued', 0)))
        print(f"  [{env_name}|cadence={condition}] ep {ep+1}: "
              f"reward={rewards[-1]:.2f}  n_instrs={n_instrs[-1]}")

    issued = inf.issued_lengths
    return {
        'mean_reward':   float(np.mean(rewards)),
        'std_reward':    float(np.std(rewards)),
        'mean_n_instrs': float(np.mean(n_instrs)),
        'mean_len':      float(np.mean(issued)) if issued else 0.0,
        'rewards':       rewards,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────

def plot_sweep(results: dict, conditions: list, envs: list, title: str,
               xlabel: str, outdir: Path, fname: str):
    """Grouped bar chart across environments for one sweep."""
    x     = np.arange(len(envs))
    width = 0.8 / len(conditions)
    fig, ax = plt.subplots(figsize=(max(8, 3 * len(envs)), 5))

    colors = plt.cm.tab10(np.linspace(0, 1, len(conditions)))
    for i, (cond, color) in enumerate(zip(conditions, colors)):
        means = [results.get(env, {}).get(cond, {}).get('mean_reward', 0) for env in envs]
        stds  = [results.get(env, {}).get(cond, {}).get('std_reward', 0)  for env in envs]
        offset = (i - len(conditions) / 2 + 0.5) * width
        ax.bar(x + offset, means, width, yerr=stds, label=cond,
               color=color, capsize=3, alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(envs, rotation=15, ha='right')
    ax.set_ylabel('Task reward')
    ax.set_title(title)
    ax.legend(fontsize=8, ncol=2, title=xlabel)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    outdir.mkdir(parents=True, exist_ok=True)
    fig.savefig(outdir / f'{fname}.pdf')
    fig.savefig(outdir / f'{fname}.png', dpi=150)
    plt.close()
    print(f"Saved to {outdir}/{fname}.{{pdf,png}}")


def plot_length_vs_reward(results: dict, envs: list, outdir: Path):
    """Line plot of mean instruction length vs reward per env."""
    fig, ax = plt.subplots(figsize=(7, 5))
    colors = plt.cm.Set2(np.linspace(0, 1, len(envs)))

    for env, color in zip(envs, colors):
        env_res = results.get(env, {})
        lengths, rewards = [], []
        for cond in LENGTH_BUCKETS:
            d = env_res.get(cond, {})
            if d:
                lengths.append(d.get('mean_len', 0))
                rewards.append(d.get('mean_reward', 0))
        if lengths:
            order = np.argsort(lengths)
            ax.plot(np.array(lengths)[order], np.array(rewards)[order],
                    marker='o', label=env, color=color)

    ax.set_xlabel('Mean instruction length (tokens)')
    ax.set_ylabel('Task reward')
    ax.set_title('Instruction length vs. reward')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(outdir / 'length_vs_reward.pdf')
    fig.savefig(outdir / 'length_vs_reward.png', dpi=150)
    plt.close()
    print(f"Saved to {outdir}/length_vs_reward.{{pdf,png}}")


# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--envs', nargs='+',
                   default=['crafter', 'minecraft', 'dmlab'])
    p.add_argument('--checkpoint_root', default='logdir/')
    p.add_argument('--planner',  default='gpt4o')
    p.add_argument('--episodes', type=int, default=20)
    p.add_argument('--sweep',    choices=['length', 'cadence', 'both'], default='both')
    p.add_argument('--logdir',   default='results/instruction_horizon')
    return p.parse_args()


def print_latex_table(results: dict, conditions: list, envs: list, title: str):
    print(f"\n% {title}")
    header_cols = ' & '.join(envs)
    print(f"\\begin{{tabular}}{{l{'r'*len(envs)}}}")
    print(r"\toprule")
    print(f"Condition & {header_cols} \\\\ \\midrule")
    for cond in conditions:
        row_vals = []
        for env in envs:
            d = results.get(env, {}).get(cond, {})
            if d:
                row_vals.append(f"${d['mean_reward']:.1f}\\pm{d['std_reward']:.1f}$")
            else:
                row_vals.append('—')
        print(f"{cond} & {' & '.join(row_vals)} \\\\")
    print(r"\bottomrule\end{tabular}")


def main():
    args   = parse_args()
    logdir = Path(args.logdir)
    logdir.mkdir(parents=True, exist_ok=True)

    sweep_a_results = {env: {} for env in args.envs}  # length sweep
    sweep_b_results = {env: {} for env in args.envs}  # cadence sweep

    if args.sweep in ('length', 'both'):
        print("\n═══ Sweep A: Instruction token length ═══")
        for env in args.envs:
            for cond in LENGTH_BUCKETS:
                print(f"\n── {env} | length={cond} ──")
                res = run_length_sweep(env, cond, args.planner,
                                       args.episodes, args.checkpoint_root)
                if res:
                    sweep_a_results[env][cond] = res

    if args.sweep in ('cadence', 'both'):
        print("\n═══ Sweep B: Instruction cadence K ═══")
        for env in args.envs:
            for cond in CADENCE_VALUES:
                print(f"\n── {env} | cadence={cond} ──")
                res = run_cadence_sweep(env, cond, args.planner,
                                        args.episodes, args.checkpoint_root)
                if res:
                    sweep_b_results[env][cond] = res

    # Save
    out = logdir / 'instruction_horizon_results.json'
    with open(out, 'w') as f:
        json.dump({'length_sweep': sweep_a_results,
                   'cadence_sweep': sweep_b_results}, f, indent=2)
    print(f"\nSaved to {out}")

    # Plots
    envs_with_data = [e for e in args.envs
                      if sweep_a_results.get(e) or sweep_b_results.get(e)]

    if args.sweep in ('length', 'both') and any(sweep_a_results.values()):
        plot_sweep(sweep_a_results, list(LENGTH_BUCKETS.keys()), envs_with_data,
                   'Instruction Length vs. Task Reward', 'Token budget',
                   logdir, 'length_sweep')
        plot_length_vs_reward(sweep_a_results, envs_with_data, logdir)
        print_latex_table(sweep_a_results, list(LENGTH_BUCKETS.keys()),
                          envs_with_data, 'Sweep A — Instruction token length')

    if args.sweep in ('cadence', 'both') and any(sweep_b_results.values()):
        plot_sweep(sweep_b_results, list(CADENCE_VALUES.keys()), envs_with_data,
                   'Instruction Cadence (K steps) vs. Task Reward', 'Cadence K',
                   logdir, 'cadence_sweep')
        print_latex_table(sweep_b_results, list(CADENCE_VALUES.keys()),
                          envs_with_data, 'Sweep B — Instruction cadence K')


if __name__ == '__main__':
    main()
