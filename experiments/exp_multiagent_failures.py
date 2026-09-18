"""
exp_multiagent_failures.py — Planner-level vs controller-level failure attribution
                              for the Overcooked 2-agent setting.

Addresses Reviewer Q11:
  "For multi-agent settings, how often do failures come from planner-level
   coordination versus controller-level execution?"

Definitions
───────────
We classify each instruction window (the span between two consecutive
instructions for agent i) as one of four outcomes:

  SUCCESS          — agent completed the instruction (p_stop fired) AND the
                     team earned positive reward during that window.

  CTRL_FAILURE     — agent never reached p_stop before the next replan or
                     episode end. The controller could not execute the
                     instruction regardless of what it said.

  COORD_FAILURE    — both agents completed their instructions (p_stop fired)
                     but the team reward during the joint window was ≤ 0.
                     The planner issued valid but poorly coordinated tasks
                     (e.g. both assigned the same sub-task, wrong ordering).

  AMBIGUOUS        — p_stop fired but only one agent contributed to reward;
                     used as a catch-all for partial coordination issues.

Additionally we measure:
  • Instruction redundancy: cosine similarity between agent 0 and agent 1's
    embedding at each joint boundary (high similarity ≈ duplicate tasks).
  • Reward-per-instruction for each outcome class.

Usage
─────
  python experiments/exp_multiagent_failures.py \\
      --episodes 100 \\
      --planner gpt4o \\
      --outdir results/multiagent_failures

  python experiments/exp_multiagent_failures.py --mock --episodes 20
"""

import argparse
import json
import sys
import threading
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

CKPT_ROOT = Path('/data/terran/instruct_to_act')
OUTCOME_COLORS = {
    'success':      '#2E7D32',
    'ctrl_failure': '#1565C0',
    'coord_failure':'#E65100',
    'ambiguous':    '#78909C',
}


# ─────────────────────────────────────────────────────────────────────────────
# Mock components
# ─────────────────────────────────────────────────────────────────────────────

class MockPlanner:
    INSTRS = [
        "Pick up an onion from the dispenser.",
        "Place the onion into the pot.",
        "Pick up a bowl and serve the soup.",
        "Wait near the pot for soup to cook.",
        "Move to the delivery counter.",
    ]
    def __init__(self, agent_id=0, latency_ms=2.0):
        self._latency = latency_ms / 1000
        self._idx = agent_id
    def step(self, obs, plan, memory, p_stop=0.0, chat_context=''):
        time.sleep(self._latency)
        instr = self.INSTRS[self._idx % len(self.INSTRS)]
        self._idx += 1
        return instr, plan, memory
    def emit(self, obs, plan, memory):
        return self.step(obs, plan, memory, p_stop=1.0)
    def advance(self, obs, plan, memory):
        return plan, memory


class MockController:
    _lang_size = 384
    def __init__(self, agent_id=0):
        self._rng = np.random.default_rng(agent_id)
    def initial_state(self, batch_size=1):
        return None
    def init_policy(self, batch_size=1):
        return None
    def policy(self, state, obs_b, mode='eval'):
        time.sleep(0.002)
        p_stop = float(self._rng.random() < 0.06)
        return state, {'action': np.zeros((1,), dtype=np.int32)}, {'log/p_stop': p_stop}


class MockEnv:
    def reset(self):
        return {'image': np.zeros((64, 64, 3), np.uint8)}, {}
    def step(self, action):
        reward = float(np.random.random() < 0.05) * 20.0
        return {'image': np.zeros((64, 64, 3), np.uint8)}, reward, False, False, {}


def mock_encoder(text):
    # Give slightly different embeddings so similarity is meaningful
    rng = np.random.default_rng(hash(text) % 2**32)
    return rng.standard_normal(384).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Instrumented multi-agent episode runner
# ─────────────────────────────────────────────────────────────────────────────

class FailureAttributionRunner:
    """
    Runs 2-agent Overcooked episodes and records per-window outcome labels.
    """

    def __init__(self, ctrls, planners, envs, encoder,
                 stop_threshold=0.5, max_steps=400,
                 reward_threshold=0.1, n_agents=2):
        self.ctrls         = ctrls
        self.planners      = planners
        self.envs          = envs
        self.encode        = encoder
        self.stop_thr      = stop_threshold
        self.max_steps     = max_steps
        self.rew_thr       = reward_threshold
        self.n              = n_agents

        # Accumulated across episodes
        self.windows        = []   # list of window dicts
        self.episode_rewards= []
        self.redundancy_scores = []  # cosine sim between agents' embeddings

    def run_episode(self):
        lang_size  = getattr(self.ctrls[0], '_lang_size', 384)
        null_embed = np.zeros(lang_size, dtype=np.float32)

        ctrl_states = [c.init_policy(batch_size=1) for c in self.ctrls]
        cur_embeds  = [null_embed.copy() for _ in range(self.n)]
        cur_texts   = ['' for _ in range(self.n)]
        stop_flags  = [True] * self.n
        plans       = [None] * self.n
        memories    = [{} for _ in range(self.n)]

        obs_list    = [env.reset()[0] for env in self.envs]
        total_reward = 0.0

        # Window tracking per agent
        # window = {start_step, instruction, p_stops, rewards, completed}
        active_windows = [None] * self.n

        def open_window(i, step, instr):
            active_windows[i] = {
                'agent':       i,
                'start_step':  step,
                'instruction': instr,
                'embed':       self.encode(instr).copy(),
                'p_stops':     [],
                'rewards':     [],
                'completed':   False,
            }

        def close_window(i, step, completed):
            w = active_windows[i]
            if w is None:
                return
            w['end_step']  = step
            w['duration']  = step - w['start_step']
            w['completed'] = completed
            w['mean_reward'] = float(np.sum(w['rewards']))
            self.windows.append(w)
            active_windows[i] = None

        for step in range(self.max_steps):

            # Planner calls (parallel for speed)
            new_instrs = [None] * self.n

            def planner_call(i):
                if not stop_flags[i]:
                    return
                instr, plans[i], memories[i] = self.planners[i].step(
                    obs=obs_list[i], plan=plans[i], memory=memories[i], p_stop=1.0
                )
                if instr:
                    new_instrs[i] = instr

            threads = [threading.Thread(target=planner_call, args=(i,))
                       for i in range(self.n)]
            for t in threads: t.start()
            for t in threads: t.join()

            for i in range(self.n):
                if new_instrs[i] is not None:
                    # Close old window (not completed — controller didn't finish it)
                    if active_windows[i] is not None:
                        close_window(i, step, completed=False)
                    cur_embeds[i] = self.encode(new_instrs[i])
                    cur_texts[i]  = new_instrs[i]
                    stop_flags[i] = False
                    open_window(i, step, new_instrs[i])

            # Instruction redundancy: cosine sim between agents' embeddings
            e0 = cur_embeds[0]
            e1 = cur_embeds[1]
            n0, n1 = np.linalg.norm(e0), np.linalg.norm(e1)
            if n0 > 0 and n1 > 0:
                self.redundancy_scores.append(float(np.dot(e0, e1) / (n0 * n1)))

            # Controller steps
            step_rewards = [0.0] * self.n
            terminated   = False
            for i in range(self.n):
                obs_b = {k: np.expand_dims(v, 0) for k, v in obs_list[i].items()
                         if not k.startswith('log/')}
                obs_b['lang_embed'] = cur_embeds[i][np.newaxis]

                ctrl_states[i], action, extra = self.ctrls[i].policy(
                    ctrl_states[i], obs_b, mode='eval'
                )
                p_stop  = float(extra.get('log/p_stop', extra.get('p_stop', 0.0)))
                action_np = np.array(action['action'][0])
                obs_new, r, term, trunc, _ = self.envs[i].step(action_np)
                obs_list[i]    = obs_new
                step_rewards[i] = float(r)
                total_reward   += float(r)

                if active_windows[i] is not None:
                    active_windows[i]['p_stops'].append(p_stop)
                    active_windows[i]['rewards'].append(float(r))

                if p_stop > self.stop_thr:
                    close_window(i, step, completed=True)
                    cur_embeds[i] = null_embed.copy()
                    stop_flags[i] = True

                if term or trunc:
                    terminated = True

            if terminated:
                break

        # Close any still-open windows at episode end (not completed)
        for i in range(self.n):
            if active_windows[i] is not None:
                close_window(i, step + 1, completed=False)

        self.episode_rewards.append(total_reward)
        return total_reward


# ─────────────────────────────────────────────────────────────────────────────
# Failure classification
# ─────────────────────────────────────────────────────────────────────────────

def classify_windows(windows, rew_thr=0.1, stop_head_trained=True):
    """
    Classify each window as success / ctrl_failure / coord_failure / ambiguous.

    If stop_head_trained=False (p_stop always near 0), fall back to
    reward-only classification: reward>0 → success, reward=0 → coord_failure.
    ctrl_failure is only used when p_stop reliably fires.
    """
    # Detect if stop head is trained: if <5% of windows were completed, assume not
    n_completed = sum(1 for w in windows if w['completed'])
    stop_trained = stop_head_trained and (n_completed / max(len(windows), 1)) > 0.05

    outcomes = []
    for w in windows:
        rew = w['mean_reward']
        if stop_trained:
            if not w['completed']:
                outcomes.append('ctrl_failure')
            elif rew > rew_thr:
                outcomes.append('success')
            else:
                outcomes.append('coord_failure')
        else:
            # p_stop unreliable — classify by reward only
            if rew > rew_thr:
                outcomes.append('success')
            else:
                outcomes.append('coord_failure')

    return outcomes, stop_trained


def compute_stats(windows, outcomes):
    counts = defaultdict(int)
    reward_by_outcome = defaultdict(list)

    for w, o in zip(windows, outcomes):
        counts[o] += 1
        reward_by_outcome[o].append(w['mean_reward'])

    total = len(outcomes)
    stats = {}
    for o in ['success', 'ctrl_failure', 'coord_failure', 'ambiguous']:
        n = counts[o]
        r = reward_by_outcome[o]
        stats[o] = {
            'count': n,
            'fraction': n / total if total > 0 else 0,
            'mean_reward': float(np.mean(r)) if r else 0.0,
        }

    # Duration stats per outcome
    dur_by_outcome = defaultdict(list)
    for w, o in zip(windows, outcomes):
        dur_by_outcome[o].append(w['duration'])
    for o, durs in dur_by_outcome.items():
        stats[o]['mean_duration'] = float(np.mean(durs))

    return stats, counts


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────

def plot_results(windows, outcomes, redundancy_scores, stats, outdir):
    fig, axes = plt.subplots(1, 4, figsize=(16, 5))
    fig.suptitle('Overcooked — Multi-Agent Failure Attribution (Q11)',
                 fontsize=12, fontweight='bold')

    labels  = ['success', 'ctrl_failure', 'coord_failure', 'ambiguous']
    display = ['Success', 'Ctrl\nFailure', 'Coord\nFailure', 'Ambiguous']
    colors  = [OUTCOME_COLORS[l] for l in labels]

    # (a) Outcome pie chart
    ax = axes[0]
    counts = [stats[l]['count'] for l in labels]
    nonzero = [(c, d, col) for c, d, col in zip(counts, display, colors) if c > 0]
    if nonzero:
        cs, ds, cols = zip(*nonzero)
        ax.pie(cs, labels=ds, colors=cols, autopct='%1.1f%%',
               startangle=90, textprops={'fontsize': 9})
    ax.set_title('(a) Outcome distribution')

    # (b) Fraction bar chart
    ax = axes[1]
    fracs = [stats[l]['fraction'] for l in labels]
    bars  = ax.bar(display, fracs, color=colors, alpha=0.82, edgecolor='white')
    for bar, f in zip(bars, fracs):
        if f > 0:
            ax.text(bar.get_x() + bar.get_width()/2, f + 0.01,
                    f'{f:.1%}', ha='center', va='bottom', fontsize=9)
    ax.set_ylabel('Fraction of instruction windows')
    ax.set_title('(b) Failure breakdown')
    ax.set_ylim(0, 1.1)
    ax.grid(True, alpha=0.3, axis='y')

    # (c) Mean instruction duration per outcome
    ax = axes[2]
    durs   = [stats[l].get('mean_duration', 0) for l in labels]
    ax.bar(display, durs, color=colors, alpha=0.82, edgecolor='white')
    ax.set_ylabel('Mean instruction duration (steps)')
    ax.set_title('(c) Duration by outcome\n(ctrl failures = longer)')
    ax.grid(True, alpha=0.3, axis='y')

    # (d) Instruction redundancy distribution
    ax = axes[3]
    if redundancy_scores:
        ax.hist(redundancy_scores, bins=25, color='#7B1FA2', alpha=0.75,
                edgecolor='white', density=True)
        mean_sim = np.mean(redundancy_scores)
        ax.axvline(mean_sim, color='crimson', ls='--', lw=2,
                   label=f'mean = {mean_sim:.2f}')
        ax.legend(fontsize=8)
    ax.set_xlabel('Cosine similarity (agent 0 vs agent 1 instr)')
    ax.set_ylabel('Density')
    ax.set_title('(d) Instruction redundancy\n(high sim = coord failure risk)')
    ax.set_xlim(-1, 1)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    outdir.mkdir(parents=True, exist_ok=True)
    fig.savefig(outdir / 'multiagent_failures.pdf', bbox_inches='tight')
    fig.savefig(outdir / 'multiagent_failures.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved {outdir}/multiagent_failures.{{pdf,png}}")


# ─────────────────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────────────────

def print_summary(stats, redundancy_scores, n_episodes):
    W = 74
    print(f"\n{'─'*W}")
    print(f"  Overcooked — Multi-Agent Failure Attribution (Q11)")
    print(f"  {n_episodes} episodes")
    print(f"{'─'*W}")
    print(f"  {'Outcome':<18}  {'Count':>6}  {'Fraction':>9}  "
          f"{'Mean reward':>12}  {'Mean duration':>14}")
    print(f"  {'─'*(W-2)}")
    for o in ['success', 'ctrl_failure', 'coord_failure', 'ambiguous']:
        s = stats[o]
        print(f"  {o:<18}  {s['count']:>6}  {s['fraction']:>9.1%}  "
              f"{s['mean_reward']:>12.3f}  {s.get('mean_duration',0):>14.1f}")
    print(f"{'─'*W}")
    if redundancy_scores:
        print(f"  Instruction redundancy (cosine sim):")
        print(f"    mean={np.mean(redundancy_scores):.3f}  "
              f"std={np.std(redundancy_scores):.3f}  "
              f"p95={np.percentile(redundancy_scores,95):.3f}")
        high_redundancy = np.mean(np.array(redundancy_scores) > 0.7)
        print(f"    Fraction with sim > 0.7 (likely duplicate tasks): "
              f"{high_redundancy:.1%}")
    print(f"{'─'*W}\n")


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
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description='Multi-agent failure attribution — Overcooked'
    )
    p.add_argument('--logdir_seed', default=str(CKPT_ROOT / 'overcooked' / 'seed0'))
    p.add_argument('--checkpoint',  default=None)
    p.add_argument('--planner',     default='gpt4o',
                   choices=['gpt4o', 'qwen', 'gemma', 'llava', 'scripted'])
    p.add_argument('--episodes',    type=int, default=100)
    p.add_argument('--max_steps',   type=int, default=400)
    p.add_argument('--n_agents',    type=int, default=2)
    p.add_argument('--stop_threshold', type=float, default=0.5)
    p.add_argument('--reward_threshold', type=float, default=0.1,
                   help='Min team reward to count a completed instruction as success')
    p.add_argument('--outdir',      default='results/multiagent_failures')
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
        planners = [MockPlanner(agent_id=i) for i in range(n)]
        ctrls    = [MockController(agent_id=i) for i in range(n)]
        envs     = [MockEnv() for _ in range(n)]
        encoder  = mock_encoder
    else:
        root = Path(__file__).parent.parent
        sys.path.insert(0, str(root))
        sys.path.insert(0, str(root / 'dreamerv3'))
        from planners import make_planner
        from envs     import make_env
        from train    import load_project_config
        from evaluate import load_agent, GymAdapter, encode_fn

        import pickle
        ckpt_path   = (Path(args.checkpoint) if args.checkpoint
                       else resolve_checkpoint(Path(args.logdir_seed)))
        print(f"── Checkpoint: {ckpt_path} ──")

        _ckpt_data  = pickle.loads(Path(ckpt_path).read_bytes())
        _ckpt_units = int(_ckpt_data['params']['con/head/logit/kernel'].shape[0])

        config_paths = [str(root / 'configs' / 'base.yaml'),
                        str(root / 'configs' / 'overcooked.yaml')]
        if _ckpt_units == 1024:
            config_paths.append(str(root / 'configs' / 'size200m.yaml'))
            print("── Applying size200m.yaml ──")

        proj      = load_project_config(config_paths)
        lang_size = int(proj.get('lang_size', 384))
        encoder   = encode_fn(lang_size)
        envs      = [GymAdapter(make_env('overcooked', proj, lang_size=lang_size))
                     for _ in range(n)]
        ctrl      = load_agent(str(ckpt_path), envs[0].obs_space,
                               envs[0].act_space, proj)
        ctrls     = [ctrl] * n
        task_spec = proj.get('task_guidance', '')
        planners  = [make_planner(args.planner, task_spec=task_spec) for _ in range(n)]

    runner = FailureAttributionRunner(
        ctrls, planners, envs, encoder,
        stop_threshold=args.stop_threshold,
        max_steps=args.max_steps,
        reward_threshold=args.reward_threshold,
        n_agents=n,
    )

    print(f"\n── Running {args.episodes} episodes ──")
    for ep in range(args.episodes):
        r = runner.run_episode()
        print(f"  ep {ep+1:4d}/{args.episodes}  reward={r:.2f}"
              f"  windows={len(runner.windows)}"
              f"  redundancy={np.mean(runner.redundancy_scores or [0]):.2f}")

    outcomes, stop_trained = classify_windows(runner.windows,
                                               rew_thr=args.reward_threshold)
    if not stop_trained:
        print("  NOTE: stop head p_stop near zero — using reward-only classification.")
        print("        ctrl_failure cannot be distinguished from coord_failure.")
    stats, _ = compute_stats(runner.windows, outcomes)

    # Save JSON
    results = {
        'n_episodes': args.episodes,
        'outcome_stats': stats,
        'mean_redundancy': float(np.mean(runner.redundancy_scores or [0])),
        'p95_redundancy':  float(np.percentile(runner.redundancy_scores or [0], 95)),
        'mean_episode_reward': float(np.mean(runner.episode_rewards)),
    }
    with open(outdir / 'multiagent_failures_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print_summary(stats, runner.redundancy_scores, args.episodes)
    plot_results(runner.windows, outcomes, runner.redundancy_scores,
                 stats, outdir)
    print(f"All outputs in {outdir}/")


if __name__ == '__main__':
    main()
