"""
exp_horizon_overcooked.py — Instruction horizon sensitivity for Overcooked (2 agents).

Addresses Reviewer Q13:
  "How does performance change as the instruction horizon varies? Are there tasks
   where very short or very long instructions break the interface?"

Two sweeps
──────────
Sweep A — Token length:
  Constrain per-agent instructions to fixed word-count budgets:
    tiny   :  3–5 words    (e.g. "pick up onion")
    short  :  6–9 words    (paper default ~7)
    medium : 10–15 words
    long   : 16–25 words
    free   :  no constraint

Sweep B — Replanning cadence K:
  Force instruction replacement every K steps, ignoring p_stop:
    K=4, K=8, K=16, K=32, adaptive (paper default — use p_stop head)

Note: Overcooked episodes are short (400 steps), so cadence values are
smaller than Minecraft.

Metrics per condition: mean ± std episode reward, per-agent instructions
issued, mean instruction length.

Usage
─────
  python experiments/exp_horizon_overcooked.py \\
      --episodes 50 \\
      --outdir results/horizon_overcooked

  python experiments/exp_horizon_overcooked.py --mock --episodes 10
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# ─────────────────────────────────────────────────────────────────────────────
# Sweep configuration
# ─────────────────────────────────────────────────────────────────────────────

LENGTH_CONDITIONS = {
    'tiny':   (3,  5),
    'short':  (6,  9),
    'medium': (10, 15),
    'long':   (16, 25),
    'free':   (None, None),
}

CADENCE_CONDITIONS = {
    'K=4':      4,
    'K=8':      8,
    'K=16':     16,
    'K=32':     32,
    'adaptive': None,
}

CKPT_ROOT = Path('/data/terran/instruct_to_act')


# ─────────────────────────────────────────────────────────────────────────────
# Mock components
# ─────────────────────────────────────────────────────────────────────────────

class MockPlanner:
    INSTRUCTIONS = [
        "Pick up an onion from the dispenser station.",
        "Place the onion into the cooking pot carefully.",
        "Wait near the pot for the soup to finish cooking.",
        "Pick up an empty bowl from the bowl dispenser.",
        "Serve the finished soup using the bowl quickly.",
        "Move to the delivery counter with the dish.",
        "Hand the completed dish to the serving counter.",
        "Coordinate with your partner to split the tasks.",
        "Refill the pot with three onions from dispenser.",
        "Clear the counter and prepare for the next order.",
    ]

    def __init__(self, agent_id=0, latency_ms=1.0):
        self._latency   = latency_ms / 1000.0
        self._idx       = agent_id  # offset so agents don't repeat identically
        self._agent_id  = agent_id

    def emit(self, obs, plan, memory):
        time.sleep(self._latency)
        instr = self.INSTRUCTIONS[self._idx % len(self.INSTRUCTIONS)]
        self._idx += 1
        return instr, plan, memory

    def advance(self, obs, plan, memory):
        time.sleep(self._latency * 1.2)
        return plan, memory

    def step(self, obs, plan, memory, p_stop=0.0):
        time.sleep(self._latency)
        instr = self.INSTRUCTIONS[self._idx % len(self.INSTRUCTIONS)]
        self._idx += 1
        return instr, plan, memory


class MockController:
    _lang_size = 384

    def init_policy(self, batch_size=1):
        return None

    def policy(self, state, obs_batch, mode='eval'):
        time.sleep(0.001)
        p_stop = float(np.random.random() < 0.08)
        return state, {'action': np.zeros((1,), dtype=np.int32)}, {'log/p_stop': p_stop}


class MockMultiAgentEnv:
    N_AGENTS = 2

    def reset(self):
        obs = [{'image': np.zeros((64, 64, 3), np.uint8)} for _ in range(self.N_AGENTS)]
        return obs, {}

    def step(self, actions):
        obs = [{'image': np.zeros((64, 64, 3), np.uint8)} for _ in range(self.N_AGENTS)]
        reward = float(np.random.random() < 0.05) * 20.0
        return obs, reward, False, False, {}


def mock_encoder(text):
    return np.zeros(384, dtype=np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Instruction length manipulation
# ─────────────────────────────────────────────────────────────────────────────

def apply_length_budget(text: str, min_w, max_w) -> str:
    if min_w is None:
        return text
    words = text.split()
    if len(words) > max_w:
        words = words[:max_w]
    while len(words) < min_w:
        words.append(words[-1] if words else 'proceed')
    return ' '.join(words)


class LengthConstrainedPlanner:
    def __init__(self, base, condition: str):
        self.base    = base
        self.min_w, self.max_w = LENGTH_CONDITIONS[condition]
        self.issued_lengths = []

    def _constrain(self, text):
        text = apply_length_budget(text, self.min_w, self.max_w)
        self.issued_lengths.append(len(text.split()))
        return text

    def emit(self, obs, plan, memory):
        instr, plan, memory = self.base.emit(obs, plan, memory)
        return self._constrain(instr), plan, memory

    def advance(self, obs, plan, memory):
        return self.base.advance(obs, plan, memory)

    def step(self, obs, plan, memory, p_stop=0.0):
        instr, plan, memory = self.base.step(obs, plan, memory, p_stop)
        if instr:
            instr = self._constrain(instr)
        return instr, plan, memory


# ─────────────────────────────────────────────────────────────────────────────
# Sweep A — length sweep (offline / synchronous, per agent)
# ─────────────────────────────────────────────────────────────────────────────

def run_length_episode(ctrls, planners, env, encoder, stop_threshold, max_steps, n_agents):
    ctrl_states = [c.init_policy(batch_size=1) for c in ctrls]
    lang_size   = getattr(ctrls[0], '_lang_size', 384)
    null_embed  = np.zeros(lang_size, dtype=np.float32)
    cur_embeds  = [null_embed.copy() for _ in range(n_agents)]
    stop_flags  = [True] * n_agents
    plans       = [None] * n_agents
    memories    = [{} for _ in range(n_agents)]
    n_instrs    = [0] * n_agents
    total_reward = 0.0

    obs_list, _ = env.reset()

    for step in range(max_steps):
        # Sequential VLM calls for agents that need a new instruction
        for i in range(n_agents):
            if stop_flags[i]:
                instr, plans[i], memories[i] = planners[i].step(
                    obs=obs_list[i], plan=plans[i], memory=memories[i],
                    p_stop=float(stop_flags[i])
                )
                if instr:
                    cur_embeds[i] = encoder(instr)
                    stop_flags[i] = False
                    n_instrs[i]  += 1

        actions = []
        for i in range(n_agents):
            obs_b = {k: np.expand_dims(v, 0) for k, v in obs_list[i].items()
                     if not k.startswith('log/')}
            obs_b['lang_embed'] = cur_embeds[i][np.newaxis]
            ctrl_states[i], action, extra = ctrls[i].policy(
                ctrl_states[i], obs_b, mode='eval'
            )
            p_stop = float(extra.get('log/p_stop', extra.get('p_stop', 0.0)))
            if p_stop > stop_threshold:
                cur_embeds[i] = null_embed.copy()
                stop_flags[i] = True
            actions.append(np.array(action['action'][0]))

        obs_list, reward, terminated, truncated, _ = env.step(actions)
        total_reward += float(reward)
        if terminated or truncated:
            break

    return total_reward, n_instrs


# ─────────────────────────────────────────────────────────────────────────────
# Sweep B — cadence sweep
# ─────────────────────────────────────────────────────────────────────────────

def run_cadence_episode(ctrls, planners, env, encoder, K, max_steps, n_agents):
    ctrl_states = [c.init_policy(batch_size=1) for c in ctrls]
    lang_size   = getattr(ctrls[0], '_lang_size', 384)
    null_embed  = np.zeros(lang_size, dtype=np.float32)
    n_instrs    = [0] * n_agents
    total_reward = 0.0

    obs_list, _ = env.reset()

    # Emit initial instructions
    plans    = [None] * n_agents
    memories = [{} for _ in range(n_agents)]
    cur_embeds = []
    for i in range(n_agents):
        instr, plans[i], memories[i] = planners[i].emit(
            obs=obs_list[i], plan=plans[i], memory=memories[i]
        )
        cur_embeds.append(encoder(instr))
        n_instrs[i] += 1

    for step in range(max_steps):
        actions = []
        p_stops = []
        for i in range(n_agents):
            obs_b = {k: np.expand_dims(v, 0) for k, v in obs_list[i].items()
                     if not k.startswith('log/')}
            obs_b['lang_embed'] = cur_embeds[i][np.newaxis]
            ctrl_states[i], action, extra = ctrls[i].policy(
                ctrl_states[i], obs_b, mode='eval'
            )
            p_stops.append(float(extra.get('log/p_stop', extra.get('p_stop', 0.0))))
            actions.append(np.array(action['action'][0]))

        obs_list, reward, terminated, truncated, _ = env.step(actions)
        total_reward += float(reward)

        for i in range(n_agents):
            replan = (K is not None and (step + 1) % K == 0) or \
                     (K is None and p_stops[i] > 0.5)
            if replan:
                instr, plans[i], memories[i] = planners[i].emit(
                    obs=obs_list[i], plan=plans[i], memory=memories[i]
                )
                cur_embeds[i] = encoder(instr)
                n_instrs[i]  += 1

        if terminated or truncated:
            break

    return total_reward, n_instrs


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint helpers
# ─────────────────────────────────────────────────────────────────────────────

def resolve_checkpoint(logdir: Path) -> Path:
    latest = logdir / 'ckpt' / 'latest'
    if latest.exists():
        tag = latest.read_text().strip()
        return logdir / 'ckpt' / tag / 'agent.pkl'
    candidates = sorted((logdir / 'ckpt').glob('*/agent.pkl'))
    if candidates:
        return candidates[-1]
    raise FileNotFoundError(f"No checkpoint under {logdir}/ckpt/")


# ─────────────────────────────────────────────────────────────────────────────
# Statistics + plotting
# ─────────────────────────────────────────────────────────────────────────────

def _stats(vals):
    a = np.array(vals, dtype=float)
    return {'mean': float(np.mean(a)), 'std': float(np.std(a)),
            'min':  float(np.min(a)),  'max': float(np.max(a)), 'n': len(a)}


COLORS = plt.cm.tab10(np.linspace(0, 0.9, 10))
ALPHA  = 0.82


def _bar_sweep(conditions, means, stds, title, xlabel, outdir, fname):
    fig, ax = plt.subplots(figsize=(9, 5))
    x    = np.arange(len(conditions))
    bars = ax.bar(x, means, yerr=stds, color=COLORS[:len(conditions)],
                  alpha=ALPHA, edgecolor='white', capsize=4, width=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(conditions, fontsize=10)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Mean episode reward (team)')
    ax.set_title(title)
    ax.grid(True, alpha=0.3, axis='y')
    for bar, m, s in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width() / 2, m + s + 0.1,
                f'{m:.1f}', ha='center', va='bottom', fontsize=8)
    plt.tight_layout()
    outdir.mkdir(parents=True, exist_ok=True)
    fig.savefig(outdir / f'{fname}.pdf', bbox_inches='tight')
    fig.savefig(outdir / f'{fname}.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved {outdir}/{fname}.{{pdf,png}}")


def plot_length_sweep(sweep_a: dict, outdir: Path):
    conditions  = list(LENGTH_CONDITIONS.keys())
    means = [sweep_a[c]['reward']['mean'] for c in conditions]
    stds  = [sweep_a[c]['reward']['std']  for c in conditions]
    mean_lens = [sweep_a[c]['mean_instr_len'] for c in conditions]

    _bar_sweep(conditions, means, stds,
               title='Overcooked — Instruction Length vs. Team Reward',
               xlabel='Length condition',
               outdir=outdir, fname='sweep_a_length_bar')

    # Length curve
    finite = [(l, m, s) for l, m, s in zip(mean_lens, means, stds)
              if l is not None and l > 0]
    if finite:
        ls, ms, ss = zip(*sorted(finite))
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.errorbar(ls, ms, yerr=ss, marker='o', color='#1B5E20',
                    lw=2, capsize=4, label='Overcooked (2 agents)')
        ax.set_xlabel('Mean instruction length (words)')
        ax.set_ylabel('Mean team reward')
        ax.set_title('Instruction length vs. team reward\n(Overcooked)')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)
        plt.tight_layout()
        fig.savefig(outdir / 'sweep_a_length_curve.pdf', bbox_inches='tight')
        fig.savefig(outdir / 'sweep_a_length_curve.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved {outdir}/sweep_a_length_curve.{{pdf,png}}")


def plot_cadence_sweep(sweep_b: dict, outdir: Path):
    conditions = list(CADENCE_CONDITIONS.keys())
    means = [sweep_b[c]['reward']['mean'] for c in conditions]
    stds  = [sweep_b[c]['reward']['std']  for c in conditions]

    _bar_sweep(conditions, means, stds,
               title='Overcooked — Replanning Cadence vs. Team Reward',
               xlabel='Cadence condition',
               outdir=outdir, fname='sweep_b_cadence_bar')

    # Fixed vs adaptive line
    fixed_conds = [c for c in conditions if c != 'adaptive']
    fixed_Ks    = [CADENCE_CONDITIONS[c] for c in fixed_conds]
    fixed_means = [sweep_b[c]['reward']['mean'] for c in fixed_conds]
    fixed_stds  = [sweep_b[c]['reward']['std']  for c in fixed_conds]
    adap_mean   = sweep_b['adaptive']['reward']['mean']
    adap_std    = sweep_b['adaptive']['reward']['std']

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.errorbar(fixed_Ks, fixed_means, yerr=fixed_stds,
                marker='s', color='#E65100', lw=2, capsize=4, label='Fixed cadence K')
    ax.axhline(adap_mean, color='#1565C0', lw=2, ls='--', label='Adaptive (p_stop)')
    ax.fill_between([min(fixed_Ks), max(fixed_Ks)],
                    adap_mean - adap_std, adap_mean + adap_std,
                    color='#1565C0', alpha=0.15)
    ax.set_xlabel('Cadence K (steps per instruction)')
    ax.set_ylabel('Mean team reward')
    ax.set_title('Fixed vs. adaptive replanning cadence\n(Overcooked)')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(outdir / 'sweep_b_cadence_curve.pdf', bbox_inches='tight')
    fig.savefig(outdir / 'sweep_b_cadence_curve.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved {outdir}/sweep_b_cadence_curve.{{pdf,png}}")


def print_summary(sweep_a, sweep_b):
    W = 72
    print(f"\n{'─'*W}")
    print(f"  Overcooked — Instruction Horizon Sweep Summary")
    print(f"{'─'*W}")
    print(f"  {'Condition':<18}  {'Reward mean':>12}  {'Reward std':>11}  "
          f"{'Mean len (w)':>13}  {'Instrs/ep':>10}")
    print(f"  {'─'*(W-2)}")
    print("  Sweep A — Token length")
    for cond in LENGTH_CONDITIONS:
        d = sweep_a.get(cond, {})
        r = d.get('reward', {})
        print(f"  {cond:<18}  {r.get('mean',0):>12.2f}  {r.get('std',0):>11.2f}  "
              f"{d.get('mean_instr_len',0):>13.1f}  {d.get('mean_n_instrs_total',0):>10.1f}")
    print("  Sweep B — Cadence K")
    for cond in CADENCE_CONDITIONS:
        d = sweep_b.get(cond, {})
        r = d.get('reward', {})
        print(f"  {cond:<18}  {r.get('mean',0):>12.2f}  {r.get('std',0):>11.2f}  "
              f"{'—':>13}  {d.get('mean_n_instrs_total',0):>10.1f}")
    print(f"{'─'*W}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description='Instruction horizon sweep — Overcooked'
    )
    p.add_argument('--logdir_seed', default=str(CKPT_ROOT / 'overcooked' / 'seed0'))
    p.add_argument('--checkpoint',  default=None)
    p.add_argument('--planner',     default='scripted',
                   choices=['gpt4o', 'qwen', 'gemma', 'llava', 'scripted'])
    p.add_argument('--n_agents',    type=int, default=2)
    p.add_argument('--episodes',    type=int, default=50)
    p.add_argument('--max_steps',   type=int, default=400)
    p.add_argument('--stop_threshold', type=float, default=0.5)
    p.add_argument('--sweep',       choices=['length', 'cadence', 'both'], default='both')
    p.add_argument('--outdir',      default='results/horizon_overcooked')
    p.add_argument('--mock',        action='store_true')
    p.add_argument('--seed',        type=int, default=0)
    return p.parse_args()


def main():
    args   = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    n = args.n_agents
    np.random.seed(args.seed)

    if args.mock:
        print(f"── Mock mode ({n} agents) ──")
        base_planners = [MockPlanner(agent_id=i) for i in range(n)]
        ctrls         = [MockController() for _ in range(n)]
        env           = MockMultiAgentEnv()
        encoder       = mock_encoder
    else:
        root = Path(__file__).parent.parent
        sys.path.insert(0, str(root))
        sys.path.insert(0, str(root / 'dreamerv3'))
        from planners import make_planner
        from envs     import make_env
        from train    import load_project_config
        from evaluate import load_agent, GymAdapter, encode_fn

        ckpt_path = (Path(args.checkpoint) if args.checkpoint
                     else resolve_checkpoint(Path(args.logdir_seed)))
        print(f"── Checkpoint: {ckpt_path} ──")

        import pickle
        _ckpt_data  = pickle.loads(Path(ckpt_path).read_bytes())
        _ckpt_units = int(_ckpt_data['params']['con/head/logit/kernel'].shape[0])

        config_paths = [str(root / 'configs' / 'base.yaml'),
                        str(root / 'configs' / 'overcooked.yaml')]
        if _ckpt_units == 1024:  # size200m
            config_paths.append(str(root / 'configs' / 'size200m.yaml'))
            print("── Applying size200m.yaml (detected from checkpoint units) ──")
        proj = load_project_config(config_paths)

        lang_size = int(proj.get('lang_size', 384))
        encoder   = encode_fn(lang_size)
        task_spec = proj.get('task_guidance', '')

        ctrls = []
        for i in range(n):
            env_i = GymAdapter(make_env('overcooked', proj, lang_size=lang_size))
            ctrls.append(load_agent(str(ckpt_path), env_i.obs_space, env_i.act_space, proj))
        env           = GymAdapter(make_env('overcooked', proj, lang_size=lang_size))
        base_planners = [make_planner(args.planner, task_spec=task_spec) for _ in range(n)]

    results = {'sweep_a': {}, 'sweep_b': {}}

    # ── Sweep A: token length ────────────────────────────────────────────────
    if args.sweep in ('length', 'both'):
        print(f"\n══ Sweep A — Instruction token length ({args.episodes} eps each) ══")
        for cond in LENGTH_CONDITIONS:
            planners = [LengthConstrainedPlanner(bp, cond) for bp in base_planners]
            rewards, n_instrs_total = [], []

            for ep in range(args.episodes):
                r, ni_list = run_length_episode(
                    ctrls, planners, env, encoder,
                    args.stop_threshold, args.max_steps, n
                )
                rewards.append(r)
                n_instrs_total.append(sum(ni_list))
                all_lens = sum([p.issued_lengths for p in planners], [])
                print(f"  [{cond}] ep {ep+1:3d}  reward={r:.2f}"
                      f"  instrs={sum(ni_list)}"
                      f"  mean_len={np.mean(all_lens or [0]):.1f}")

            all_lens = sum([p.issued_lengths for p in planners], [])
            results['sweep_a'][cond] = {
                'reward':             _stats(rewards),
                'n_instrs_total':     _stats(n_instrs_total),
                'mean_instr_len':     float(np.mean(all_lens or [0])),
                'std_instr_len':      float(np.std(all_lens  or [0])),
                'mean_n_instrs_total': float(np.mean(n_instrs_total)),
            }

    # ── Sweep B: cadence ─────────────────────────────────────────────────────
    if args.sweep in ('cadence', 'both'):
        print(f"\n══ Sweep B — Replanning cadence ({args.episodes} eps each) ══")
        for cond, K in CADENCE_CONDITIONS.items():
            rewards, n_instrs_total = [], []

            for ep in range(args.episodes):
                r, ni_list = run_cadence_episode(
                    ctrls, base_planners, env, encoder, K, args.max_steps, n
                )
                rewards.append(r)
                n_instrs_total.append(sum(ni_list))
                print(f"  [{cond}] ep {ep+1:3d}  reward={r:.2f}"
                      f"  instrs={sum(ni_list)}")

            results['sweep_b'][cond] = {
                'reward':             _stats(rewards),
                'n_instrs_total':     _stats(n_instrs_total),
                'mean_n_instrs_total': float(np.mean(n_instrs_total)),
                'K':                  K,
            }

    # ── Save + plot ──────────────────────────────────────────────────────────
    with open(outdir / 'horizon_overcooked_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved JSON to {outdir}/horizon_overcooked_results.json")

    if results['sweep_a']:
        plot_length_sweep(results['sweep_a'], outdir)
    if results['sweep_b']:
        plot_cadence_sweep(results['sweep_b'], outdir)

    print_summary(results['sweep_a'], results['sweep_b'])
    print(f"\nAll outputs in {outdir}/")


if __name__ == '__main__':
    main()
