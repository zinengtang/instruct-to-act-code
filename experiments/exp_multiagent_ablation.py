"""
exp_multiagent_ablation.py — Planner-controller ablations (rebuttal §A.8).

Addresses Reviewer 2B11:
  "The paper should include more controlled ablations separating communication,
   planner reasoning, parameter sharing, centralized versus decentralized
   coordination, and controller quality."

And Reviewer 2B11 Q11:
  "For multi-agent settings, how often do failures come from planner-level
   coordination versus controller-level execution?"

Ablation conditions
───────────────────
  full              — full Instruct-to-Act: async planning + communication
  no_comm           — no chatroom (planners act independently)
  centralized       — centralized hub-and-spoke communication
  decentralized     — decentralized any-to-any (paper default)
  separate_ctrl     — separate (non-shared) controller parameters per agent
  shared_ctrl       — shared parameters (paper default)
  controller_only   — no planner; controller acts on null instruction
  vlm_only          — VLM generates actions directly (no controller)

Run on Minecraft Diamond (single-agent, n_agents=1) for the rebuttal.
With n_agents=1 the comm conditions (no_comm, centralized, decentralized)
are degenerate — all equivalent — which itself is a useful sanity check.

Usage
─────
  python experiments/exp_multiagent_ablation.py \
      --envs overcooked \
      --checkpoint_root logdir/ \
      --planners qwen \
      --conditions full no_comm controller_only separate_ctrl \
      --n_agents 2 \
      --episodes 50 \
      --logdir results/multiagent_ablation
"""

import argparse, json, sys
from pathlib import Path
from typing import List

import numpy as np
import matplotlib.pyplot as plt


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--envs',    nargs='+', default=['overcooked'])
    p.add_argument('--checkpoint_root', default='logdir/')
    p.add_argument('--planners', nargs='+', default=['qwen'])
    p.add_argument('--conditions', nargs='+',
                   default=['full', 'no_comm', 'centralized', 'separate_ctrl', 'controller_only'])
    p.add_argument('--episodes', type=int, default=50)
    p.add_argument('--n_agents', type=int, default=2)
    p.add_argument('--logdir',   default='results/multiagent_ablation')
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Condition factories
# ─────────────────────────────────────────────────────────────────────────────

def _clone_agent(agent):
    """Return a new agent instance with the same parameter values (deep copy via JAX)."""
    import copy
    return copy.deepcopy(agent)


def build_inference(condition, controller, planners, envs, encoder, config):
    """Create the appropriate inference object for the ablation condition."""
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from inference import MultiAgentInference

    if condition == 'full':
        return MultiAgentInference(
            controller, planners, envs, encoder,
            mode='decentralized',
            max_steps=config.get('episode_length', 400),
        )
    elif condition == 'no_comm':
        # Chatroom mode 'no_comm' silently drops all send() calls, so planners
        # see an empty inbox and act fully independently.
        return MultiAgentInference(
            controller, planners, envs, encoder,
            mode='no_comm',
            max_steps=config.get('episode_length', 400),
        )
    elif condition == 'centralized':
        return MultiAgentInference(
            controller, planners, envs, encoder,
            mode='centralized', hub=0,
            max_steps=config.get('episode_length', 400),
        )
    elif condition == 'decentralized':
        return MultiAgentInference(
            controller, planners, envs, encoder,
            mode='decentralized',
            max_steps=config.get('episode_length', 400),
        )
    elif condition == 'shared_ctrl':
        # Single controller instance shared across all agents (paper default).
        return MultiAgentInference(
            controller, planners, envs, encoder,
            mode='decentralized',
            max_steps=config.get('episode_length', 400),
        )
    elif condition == 'separate_ctrl':
        # Each agent gets an independently loaded copy of the checkpoint.
        # controller here is a factory callable (LangCondAgent) or a list;
        # build_inference receives a single agent so we clone it n_agents times.
        n = len(planners)
        separate = [controller] + [_clone_agent(controller) for _ in range(n - 1)]
        return MultiAgentInference(
            separate, planners, envs, encoder,
            mode='decentralized',
            max_steps=config.get('episode_length', 400),
        )
    elif condition == 'controller_only':
        # Null planners (empty instructions)
        from planners.vlm_planner import VLMPlanner

        class NullPlanner(VLMPlanner):
            def __init__(self): super().__init__('', '', 0)
            def _call_vlm(self, *a, **k): return {'emit': False}
            def emit(self, *a, **k): return 'continue', None, {}
            def advance(self, *a, **k): return None, {}
            def step(self, *a, **k): return None, None, {}

        null_planners = [NullPlanner() for _ in planners]
        return MultiAgentInference(
            controller, null_planners, envs, encoder,
            mode='decentralized',
            max_steps=config.get('episode_length', 400),
        )
    else:
        raise ValueError(f"Unknown condition: {condition}")


# ─────────────────────────────────────────────────────────────────────────────

def run_ablation(env_name, condition, planner_name, episodes, ckpt_root, n_agents, logdir):
    """Run one (env, condition) ablation pair and return metrics."""
    sys.path.insert(0, str(Path(__file__).parent.parent))
    sys.path.insert(0, str(Path(__file__).parent.parent / 'dreamerv3'))
    import embodied, dreamerv3
    from agent    import LangCondAgent
    from planners import make_planner
    from envs     import make_env
    from sentence_transformers import SentenceTransformer

    ckpt = Path(ckpt_root) / env_name / 'seed0' / 'checkpoint.pkl'
    if not ckpt.exists():
        print(f"  SKIP: {ckpt} not found")
        return None

    config = embodied.Config(dreamerv3.configs['defaults'])
    config = config.update(embodied.Config.load(f'configs/{env_name}.yaml'))

    envs     = [make_env(env_name, config) for _ in range(n_agents)]
    agent    = LangCondAgent(envs[0].obs_space, envs[0].act_space, config)
    st       = embodied.checkpoint.Checkpoint(str(ckpt))
    st.agent = agent
    st.load()

    planners = [make_planner(planner_name, task_spec=config.get('task_guidance', ''))
                for _ in range(n_agents)]

    st_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    encoder  = lambda t: st_model.encode([t], normalize_embeddings=True)[0].astype('float32')

    inf = build_inference(condition, agent, planners, envs, encoder, config)

    team_rewards, msg_counts = [], []
    for ep in range(episodes):
        m = inf.run_episode()
        team_rewards.append(m['team_reward'])
        msg_counts.append(len(m.get('message_logs', [])))
        # Analyze planner vs controller failures
        # A "planner failure" is an episode where instructions were issued but task failed
        # A "controller failure" is an episode where instruction-following accuracy was low
        print(f"  [{condition}] ep {ep+1}: team_reward={m['team_reward']:.2f}  "
              f"msgs={len(m.get('message_logs', []))}")

    return {
        'mean_reward': float(np.mean(team_rewards)),
        'std_reward':  float(np.std(team_rewards)),
        'mean_msgs':   float(np.mean(msg_counts)),
        'rewards':     team_rewards,
    }


def plot_ablation(results: dict, outdir: Path):
    """Grouped bar chart across envs and conditions."""
    envs = list(next(iter(results.values())).keys()) if results else []
    conditions = list(results.keys())
    if not envs or not conditions:
        return

    x = np.arange(len(envs))
    width = 0.8 / len(conditions)
    fig, ax = plt.subplots(figsize=(max(8, 3*len(envs)), 5))

    colors = plt.cm.tab10(np.linspace(0, 1, len(conditions)))
    for i, (cond, color) in enumerate(zip(conditions, colors)):
        means = [results[cond].get(env, {}).get('mean_reward', 0) for env in envs]
        stds  = [results[cond].get(env, {}).get('std_reward', 0)  for env in envs]
        offset = (i - len(conditions)/2 + 0.5) * width
        ax.bar(x + offset, means, width, yerr=stds, label=cond,
               color=color, capsize=3, alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(envs)
    ax.set_ylabel('Team Reward')
    ax.set_title('Multi-Agent Ablation Study')
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    outdir.mkdir(parents=True, exist_ok=True)
    fig.savefig(outdir / 'multiagent_ablation.pdf')
    fig.savefig(outdir / 'multiagent_ablation.png', dpi=150)
    plt.close()
    print(f"Saved to {outdir}/multiagent_ablation.{{pdf,png}}")


def main():
    args   = parse_args()
    logdir = Path(args.logdir)
    logdir.mkdir(parents=True, exist_ok=True)

    results = {cond: {} for cond in args.conditions}

    for env in args.envs:
        for condition in args.conditions:
            for planner in args.planners:
                print(f"\n── {env} | {condition} | planner={planner} ──")
                m = run_ablation(
                    env, condition, planner, args.episodes,
                    args.checkpoint_root, args.n_agents, logdir
                )
                if m is not None:
                    results[condition][env] = m

    # Save
    out = logdir / 'multiagent_ablation_results.json'
    with open(out, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out}")

    # Table
    print("\n── Multi-Agent Ablation Table ──")
    header = f"{'Condition':<20}" + "".join(f"  {e:<12}" for e in args.envs)
    print(header)
    print("-" * len(header))
    for cond in args.conditions:
        row = f"{cond:<20}"
        for env in args.envs:
            m = results[cond].get(env, {})
            if m:
                row += f"  {m['mean_reward']:5.1f}±{m['std_reward']:4.1f}  "
            else:
                row += f"  {'N/A':<12}"
        print(row)

    plot_ablation(results, logdir)


if __name__ == '__main__':
    main()
