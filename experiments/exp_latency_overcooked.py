"""
exp_latency_overcooked.py — Online vs. Offline planning latency analysis for Overcooked.

Addresses Reviewer Q4:
  "Fig. 2(b) compares online vs. offline planning with two bars per scale on a
   single task. Could the authors provide latency distributions or instruction-
   staleness statistics across more tasks to support the efficiency claim?"

Measures (Overcooked asymmetric_advantages, 2 agents):
  1. Per-agent VLM latency distribution: emit() [online] vs step() [offline]  (ms)
  2. Per-agent instruction staleness [online]: steps from VLM obs-capture to arrival
  3. Per-agent controller blocking [offline]: equivalent steps lost per instruction boundary
  4. Joint (team) overhead: sum of per-agent blocking across all boundaries in an episode
  5. Controller throughput: env steps/sec [online vs offline]

Multi-agent note:
  In online mode both agents' planners run in independent background threads —
  neither blocks the controller.  In offline mode each instruction boundary
  requires sequential VLM calls for all agents, doubling the blocking cost
  relative to single-agent offline planning.

Outputs:
  <logdir>/
    latency_overcooked_results.json          — raw per-call stats, per agent
    latency_overcooked_distributions.pdf/png — histograms + CDFs + box plots
    latency_overcooked_summary.txt           — table for paper rebuttal

Usage:
  python experiments/exp_latency_overcooked.py \\
      --checkpoint logdir/overcooked/seed0/checkpoint.pkl \\
      --planner gpt4o \\
      --episodes 50 \\
      --logdir results/latency_overcooked

  # Dry-run with mock VLM (no API key required):
  python experiments/exp_latency_overcooked.py \\
      --mock --mock_mean_ms 380 --mock_std_ms 70 --episodes 50
"""

import argparse
import json
import sys
import threading
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


# ─────────────────────────────────────────────────────────────────────────────
# Mock components (no API / server required)
# ─────────────────────────────────────────────────────────────────────────────

class MockPlanner:
    INSTRUCTIONS = [
        "Pick up an onion from the dispenser.",
        "Place the onion into the pot.",
        "Wait for the soup to finish cooking.",
        "Pick up a bowl from the dispenser.",
        "Serve the soup with the bowl.",
        "Move to the delivery counter.",
        "Pass the dish to your partner.",
        "Refill the pot with onions.",
    ]

    def __init__(self, mean_ms=380.0, std_ms=70.0, agent_id=0):
        self.mean_ms  = mean_ms
        self.std_ms   = std_ms
        self.agent_id = agent_id
        self._rng     = np.random.default_rng(42 + agent_id)
        self._idx     = 0

    def _sleep(self, scale=1.0):
        delay_s = max(0.001, self._rng.normal(self.mean_ms * scale, self.std_ms) / 1000.0)
        time.sleep(delay_s)

    def emit(self, obs, plan, memory):
        self._sleep(1.0)
        instr = self.INSTRUCTIONS[self._idx % len(self.INSTRUCTIONS)]
        self._idx += 1
        return instr, plan, memory

    def advance(self, obs, plan, memory):
        self._sleep(1.2)
        return plan, memory

    def step(self, obs, plan, memory, p_stop=0.0):
        self._sleep(1.0)
        instr = self.INSTRUCTIONS[self._idx % len(self.INSTRUCTIONS)]
        self._idx += 1
        return instr, plan, memory


class MockController:
    _lang_size = 384

    def init_policy(self, batch_size=1):
        return None

    def policy(self, state, obs_batch, mode='eval'):
        time.sleep(0.002)
        rng    = np.random.default_rng()
        p_stop = float(rng.random() < 0.06)  # ~17 steps/instruction on avg
        action = {'action': np.zeros((1, 1))}
        extra  = {'log/p_stop': p_stop}
        return state, action, extra


class MockMultiAgentEnv:
    """Single-agent env view for one Overcooked agent."""

    def reset(self):
        return {'image': np.zeros((64, 64, 3), dtype=np.uint8)}, {}

    def step(self, action):
        obs = {'image': np.zeros((64, 64, 3), dtype=np.uint8)}
        return obs, 1.0, False, False, {}


def mock_encoder(text: str) -> np.ndarray:
    return np.random.randn(384).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Per-agent stat container
# ─────────────────────────────────────────────────────────────────────────────

class AgentLatencyStats:
    def __init__(self, agent_id: int):
        self.agent_id            = agent_id
        self.emit_latencies_ms   = []
        self.advance_latencies_ms= []
        self.step_latencies_ms   = []   # offline only
        self.staleness_steps     = []   # online only
        self.blocking_steps      = []   # offline only
        self.instructions_issued = 0


# ─────────────────────────────────────────────────────────────────────────────
# Instrumented Online (async) Multi-Agent Inference
# ─────────────────────────────────────────────────────────────────────────────

class MultiAgentOnlineInference:
    """
    Async online inference for N agents.  Each agent has its own VLM thread;
    the controller (shared) runs all agents sequentially in a single loop.

    Staleness for agent i at step t = (t - t_obs_i), where t_obs_i is the
    controller step when agent i's VLM captured its obs at the start of emit().
    """

    def __init__(self, controllers, planners, envs, lang_encoder,
                 stop_threshold=0.5, max_steps=400, n_agents=2):
        assert len(controllers) == n_agents and len(planners) == n_agents
        self.ctrls    = controllers
        self.planners = planners
        self.envs     = envs   # one env per agent
        self.encode   = lang_encoder
        self.stop_thr = stop_threshold
        self.max_steps= max_steps
        self.n_agents = n_agents

        self.agent_stats = [AgentLatencyStats(i) for i in range(n_agents)]
        self.ctrl_step_times = []

    def run_episode(self) -> dict:
        from inference.algorithms import AsyncState

        # One AsyncState + VLM thread per agent
        shareds     = [AsyncState() for _ in range(self.n_agents)]
        ctrl_states = [c.init_policy(batch_size=1) for c in self.ctrls]
        lang_size   = getattr(self.ctrls[0], '_lang_size', 384)
        null_embed  = np.zeros(lang_size, dtype=np.float32)
        cur_embeds  = [null_embed.copy() for _ in range(self.n_agents)]

        obs_list = [env.reset()[0] for env in self.envs]
        for i, sh in enumerate(shareds):
            sh.update_obs(obs_list[i])

        total_reward = 0.0
        ctrl_step_ctr   = [0]
        pending_capture = [None] * self.n_agents

        def make_vlm_thread(agent_idx):
            sh      = shareds[agent_idx]
            planner = self.planners[agent_idx]
            stats   = self.agent_stats[agent_idx]

            def loop():
                memory, plan = {}, None
                while sh.running:
                    latest_obs = sh.get_obs()
                    if latest_obs is None:
                        time.sleep(0.005)
                        continue

                    if sh.stop:
                        capture = ctrl_step_ctr[0]
                        t0 = time.perf_counter()
                        instr, plan, memory = planner.emit(
                            obs=latest_obs, plan=plan, memory=memory
                        )
                        stats.emit_latencies_ms.append((time.perf_counter() - t0) * 1000)
                        embed = self.encode(instr)
                        pending_capture[agent_idx] = capture
                        sh.put_inbox(embed, instr)
                        sh.clear_stop()
                    else:
                        t0 = time.perf_counter()
                        plan, memory = planner.advance(
                            obs=latest_obs, plan=plan, memory=memory
                        )
                        stats.advance_latencies_ms.append((time.perf_counter() - t0) * 1000)
                    time.sleep(0.001)
            return loop

        threads = []
        for i in range(self.n_agents):
            t = threading.Thread(target=make_vlm_thread(i), daemon=True)
            t.start()
            threads.append(t)
            shareds[i].set_stop()  # emit first instruction immediately

        for step in range(self.max_steps):
            self.ctrl_step_times.append(time.perf_counter())
            ctrl_step_ctr[0] = step

            for i in range(self.n_agents):
                new_embed, _ = shareds[i].pop_inbox()
                if new_embed is not None:
                    cur_embeds[i] = new_embed
                    if pending_capture[i] is not None:
                        staleness = max(0, step - pending_capture[i])
                        self.agent_stats[i].staleness_steps.append(staleness)
                        pending_capture[i] = None
                    self.agent_stats[i].instructions_issued += 1

            # Run all agents, collect joint reward
            actions = []
            for i in range(self.n_agents):
                obs_b = {k: np.expand_dims(v, 0) for k, v in obs_list[i].items()
                         if not k.startswith('log/')}
                obs_b['lang_embed'] = cur_embeds[i][np.newaxis]
                ctrl_states[i], action, extra = self.ctrls[i].policy(
                    ctrl_states[i], obs_b, mode='eval'
                )
                p_stop = float(extra.get('log/p_stop', extra.get('p_stop', 0.0)))
                if p_stop > self.stop_thr:
                    cur_embeds[i] = null_embed.copy()
                    shareds[i].set_stop()
                actions.append(np.array(action['action'][0]))

            reward = 0.0
            terminated = truncated = False
            for i, env in enumerate(self.envs):
                obs, r, term, trunc, _ = env.step(actions[i])
                obs_list[i] = obs
                shareds[i].update_obs(obs)
                reward     += float(r)
                terminated  = terminated or term
                truncated   = truncated  or trunc
            total_reward += reward

            if terminated or truncated:
                break

        for sh in shareds:
            sh.running = False

        return {'total_reward': total_reward, 'steps': step + 1}


# ─────────────────────────────────────────────────────────────────────────────
# Instrumented Offline (synchronous) Multi-Agent Inference
# ─────────────────────────────────────────────────────────────────────────────

class MultiAgentOfflineInference:
    """
    Synchronous offline inference for N agents.  At each instruction boundary,
    ALL agents' planners are called sequentially before the controller resumes —
    blocking cost is the sum of all per-agent VLM latencies.
    """

    def __init__(self, controllers, planners, envs, lang_encoder,
                 stop_threshold=0.5, max_steps=400, n_agents=2):
        self.ctrls    = controllers
        self.planners = planners
        self.envs     = envs
        self.encode   = lang_encoder
        self.stop_thr = stop_threshold
        self.max_steps= max_steps
        self.n_agents = n_agents

        self.agent_stats     = [AgentLatencyStats(i) for i in range(n_agents)]
        self.ctrl_step_times = []
        self.step_durations_ms = []
        self.joint_blocking_steps = []  # total blocking per boundary across all agents

    def run_episode(self) -> dict:
        ctrl_states = [c.init_policy(batch_size=1) for c in self.ctrls]
        lang_size   = getattr(self.ctrls[0], '_lang_size', 384)
        null_embed  = np.zeros(lang_size, dtype=np.float32)
        cur_embeds  = [null_embed.copy() for _ in range(self.n_agents)]

        obs_list = [env.reset()[0] for env in self.envs]
        plans   = [None] * self.n_agents
        memories= [{} for _ in range(self.n_agents)]
        stop_flags = [True] * self.n_agents
        total_reward = 0.0

        step_dur_est_ms = None

        for step in range(self.max_steps):
            self.ctrl_step_times.append(time.perf_counter())

            # Sequential VLM calls for all agents that need a new instruction
            joint_vlm_ms = 0.0
            for i in range(self.n_agents):
                if stop_flags[i]:
                    t0 = time.perf_counter()
                    instr, plans[i], memories[i] = self.planners[i].step(
                        obs=obs_list[i], plan=plans[i], memory=memories[i],
                        p_stop=float(stop_flags[i])
                    )
                    vlm_ms = (time.perf_counter() - t0) * 1000
                    self.agent_stats[i].step_latencies_ms.append(vlm_ms)
                    joint_vlm_ms += vlm_ms

                    if instr:
                        cur_embeds[i] = self.encode(instr)
                        stop_flags[i] = False
                        self.agent_stats[i].instructions_issued += 1

            if joint_vlm_ms > 0 and step_dur_est_ms and step_dur_est_ms > 0:
                per_agent_blocked = int(joint_vlm_ms / self.n_agents / step_dur_est_ms)
                joint_blocked     = int(joint_vlm_ms / step_dur_est_ms)
                for i in range(self.n_agents):
                    self.agent_stats[i].blocking_steps.append(per_agent_blocked)
                self.joint_blocking_steps.append(joint_blocked)

            # Run all agents
            t_ctrl = time.perf_counter()
            actions = []
            for i in range(self.n_agents):
                obs_b = {k: np.expand_dims(v, 0) for k, v in obs_list[i].items()
                         if not k.startswith('log/')}
                obs_b['lang_embed'] = cur_embeds[i][np.newaxis]
                ctrl_states[i], action, extra = self.ctrls[i].policy(
                    ctrl_states[i], obs_b, mode='eval'
                )
                p_stop = float(extra.get('log/p_stop', extra.get('p_stop', 0.0)))
                if p_stop > self.stop_thr:
                    cur_embeds[i] = null_embed.copy()
                    stop_flags[i] = True
                actions.append(np.array(action['action'][0]))

            reward = 0.0
            terminated = truncated = False
            for i, env in enumerate(self.envs):
                obs, r, term, trunc, _ = env.step(actions[i])
                obs_list[i] = obs
                reward     += float(r)
                terminated  = terminated or term
                truncated   = truncated  or trunc
            total_reward += reward

            ctrl_ms = (time.perf_counter() - t_ctrl) * 1000
            self.step_durations_ms.append(ctrl_ms)
            if len(self.step_durations_ms) <= 100:
                step_dur_est_ms = float(np.mean(self.step_durations_ms))

            if terminated or truncated:
                break

        return {'total_reward': total_reward, 'steps': step + 1}


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────

COLORS = ['#1565C0', '#1B5E20', '#E65100', '#880E4F']  # agents 0,1 online; 0,1 offline
ALPHA  = 0.72


def _cdf(data):
    x = np.sort(data)
    y = np.arange(1, len(x) + 1) / len(x)
    return x, y


def _throughput(times):
    if len(times) < 2:
        return 0.0
    return len(times) / max(times[-1] - times[0], 1e-9)


def _pool(stats_list, attr):
    """Concatenate a stat across agents."""
    out = []
    for s in stats_list:
        out.extend(getattr(s, attr))
    return out


def plot_comparison(online: MultiAgentOnlineInference,
                    offline: MultiAgentOfflineInference,
                    outdir: Path):
    n_agents = online.n_agents
    fig = plt.figure(figsize=(15, 11))
    fig.suptitle(
        f'Overcooked ({n_agents} agents) — Online vs. Offline Planning: Latency & Staleness',
        fontsize=13, fontweight='bold', y=0.99,
    )
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.55, wspace=0.35)

    agent_colors_on  = ['#1565C0', '#1B5E20']
    agent_colors_off = ['#E65100', '#880E4F']

    # Row 0 ─────────────────────────────────────────────────────────────────
    # (a) VLM latency histogram (pooled across agents, online vs offline)
    ax = fig.add_subplot(gs[0, 0])
    pooled_emit = _pool(online.agent_stats, 'emit_latencies_ms')
    pooled_step = _pool(offline.agent_stats, 'step_latencies_ms')
    if pooled_emit:
        ax.hist(pooled_emit, bins=30, color='#1565C0', alpha=ALPHA,
                label=f'Online emit()  n={len(pooled_emit)}', density=True)
    if pooled_step:
        ax.hist(pooled_step, bins=30, color='#E65100', alpha=ALPHA,
                label=f'Offline step()  n={len(pooled_step)}', density=True)
    ax.set_xlabel('VLM latency (ms)')
    ax.set_ylabel('Density')
    ax.set_title('(a) VLM Call Latency (pooled)')
    ax.legend(fontsize=7)

    # (b) Per-agent emit latency CDFs (online)
    ax = fig.add_subplot(gs[0, 1])
    for i, stats in enumerate(online.agent_stats):
        if stats.emit_latencies_ms:
            x, y = _cdf(stats.emit_latencies_ms)
            ax.plot(x, y, color=agent_colors_on[i % len(agent_colors_on)],
                    lw=2, label=f'Agent {i} emit()')
    for i, stats in enumerate(offline.agent_stats):
        if stats.step_latencies_ms:
            x, y = _cdf(stats.step_latencies_ms)
            ax.plot(x, y, color=agent_colors_off[i % len(agent_colors_off)],
                    lw=2, ls='--', label=f'Agent {i} step()')
    ax.set_xlabel('VLM latency (ms)')
    ax.set_ylabel('CDF')
    ax.set_title('(b) Per-Agent Latency CDF')
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # (c) Box plot: emit vs step, per agent
    ax = fig.add_subplot(gs[0, 2])
    box_data, box_labels, box_colors = [], [], []
    for i, stats in enumerate(online.agent_stats):
        if stats.emit_latencies_ms:
            box_data.append(stats.emit_latencies_ms)
            box_labels.append(f'A{i}\nemit')
            box_colors.append(agent_colors_on[i % len(agent_colors_on)])
    for i, stats in enumerate(offline.agent_stats):
        if stats.step_latencies_ms:
            box_data.append(stats.step_latencies_ms)
            box_labels.append(f'A{i}\nstep')
            box_colors.append(agent_colors_off[i % len(agent_colors_off)])
    if box_data:
        bp = ax.boxplot(box_data, labels=box_labels, patch_artist=True,
                        medianprops=dict(color='white', lw=2))
        for patch, c in zip(bp['boxes'], box_colors):
            patch.set_facecolor(c); patch.set_alpha(ALPHA)
    ax.set_ylabel('VLM latency (ms)')
    ax.set_title('(c) Per-Agent Latency Box')
    ax.set_yscale('log')

    # Row 1 ─────────────────────────────────────────────────────────────────
    # (d) Per-agent staleness (online)
    ax = fig.add_subplot(gs[1, 0])
    for i, stats in enumerate(online.agent_stats):
        if stats.staleness_steps:
            ax.hist(stats.staleness_steps, bins=25, alpha=ALPHA,
                    color=agent_colors_on[i % len(agent_colors_on)],
                    label=f'Agent {i}  mean={np.mean(stats.staleness_steps):.1f}',
                    density=True)
    ax.set_xlabel('Steps (instruction staleness)')
    ax.set_ylabel('Density')
    ax.set_title('(d) Online: Per-Agent Staleness')
    ax.legend(fontsize=7)

    # (e) Per-agent blocking steps (offline)
    ax = fig.add_subplot(gs[1, 1])
    for i, stats in enumerate(offline.agent_stats):
        if stats.blocking_steps:
            ax.hist(stats.blocking_steps, bins=25, alpha=ALPHA,
                    color=agent_colors_off[i % len(agent_colors_off)],
                    label=f'Agent {i}  mean={np.mean(stats.blocking_steps):.1f}',
                    density=True)
    ax.set_xlabel('Equivalent blocked steps')
    ax.set_ylabel('Density')
    ax.set_title('(e) Offline: Per-Agent Blocking')
    ax.legend(fontsize=7)

    # (f) Joint blocking overhead (offline) — total across all agents per boundary
    ax = fig.add_subplot(gs[1, 2])
    if offline.joint_blocking_steps:
        jb = offline.joint_blocking_steps
        ax.hist(jb, bins=25, color='#880E4F', alpha=ALPHA, density=True)
        ax.axvline(np.mean(jb), color='crimson', ls='--', lw=2,
                   label=f'mean = {np.mean(jb):.1f}')
        ax.axvline(np.percentile(jb, 95), color='darkred', ls=':', lw=1.5,
                   label=f'p95 = {np.percentile(jb, 95):.1f}')
        ax.legend(fontsize=7)
    ax.set_xlabel('Joint blocked steps (sum over agents)')
    ax.set_ylabel('Density')
    ax.set_title('(f) Offline: Joint Team Blocking\n(total steps lost per boundary)')

    # Row 2 ─────────────────────────────────────────────────────────────────
    # (g) Staleness CDF (online, per agent)
    ax = fig.add_subplot(gs[2, 0])
    for i, stats in enumerate(online.agent_stats):
        if stats.staleness_steps:
            x, y = _cdf(stats.staleness_steps)
            ax.plot(x, y, color=agent_colors_on[i % len(agent_colors_on)],
                    lw=2, label=f'Agent {i} staleness')
    ax.set_xlabel('Instruction staleness (steps)')
    ax.set_ylabel('CDF')
    ax.set_title('(g) Online: Staleness CDF')
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # (h) advance() latency (online background reasoning)
    ax = fig.add_subplot(gs[2, 1])
    for i, stats in enumerate(online.agent_stats):
        if stats.advance_latencies_ms:
            ax.hist(stats.advance_latencies_ms, bins=25, alpha=ALPHA,
                    color=agent_colors_on[i % len(agent_colors_on)],
                    label=f'Agent {i} advance()',
                    density=True)
    ax.set_xlabel('advance() latency (ms)')
    ax.set_ylabel('Density')
    ax.set_title('(h) Online: Background advance() Latency')
    ax.legend(fontsize=7)

    # (i) Throughput bar chart
    ax = fig.add_subplot(gs[2, 2])
    tput_on  = _throughput(online.ctrl_step_times)
    tput_off = _throughput(offline.ctrl_step_times)
    bars = ax.bar(['Online', 'Offline'], [tput_on, tput_off],
                  color=['#1565C0', '#E65100'], alpha=ALPHA, edgecolor='white', width=0.5)
    for bar, val in zip(bars, [tput_on, tput_off]):
        ax.text(bar.get_x() + bar.get_width() / 2, val * 1.02,
                f'{val:.1f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax.set_ylabel('Env steps / second')
    ax.set_title('(i) Controller Throughput')
    ax.set_ylim(0, max(tput_on, tput_off) * 1.2 if max(tput_on, tput_off) > 0 else 1)

    stem = 'latency_overcooked'
    fig.savefig(outdir / f'{stem}.pdf', bbox_inches='tight')
    fig.savefig(outdir / f'{stem}.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved {outdir}/{stem}.{{pdf,png}}")


# ─────────────────────────────────────────────────────────────────────────────
# Summary table
# ─────────────────────────────────────────────────────────────────────────────

def _stats(data):
    if not data:
        return {'mean': 0, 'median': 0, 'p25': 0, 'p75': 0, 'p95': 0, 'std': 0, 'n': 0}
    a = np.array(data, dtype=float)
    return {
        'mean': float(np.mean(a)), 'median': float(np.median(a)),
        'p25':  float(np.percentile(a, 25)), 'p75': float(np.percentile(a, 75)),
        'p95':  float(np.percentile(a, 95)), 'std': float(np.std(a)), 'n': int(len(a)),
    }


def save_summary(online: MultiAgentOnlineInference,
                 offline: MultiAgentOfflineInference,
                 outdir: Path):
    n = online.n_agents

    def agent_record_on(stats):
        return {
            'emit_latency_ms':    _stats(stats.emit_latencies_ms),
            'advance_latency_ms': _stats(stats.advance_latencies_ms),
            'staleness_steps':    _stats(stats.staleness_steps),
            'instructions_issued': stats.instructions_issued,
        }

    def agent_record_off(stats):
        return {
            'step_latency_ms':   _stats(stats.step_latencies_ms),
            'blocking_steps':    _stats(stats.blocking_steps),
            'instructions_issued': stats.instructions_issued,
        }

    results = {
        'task': 'Overcooked asymmetric_advantages',
        'n_agents': n,
        'online': {
            'per_agent': [agent_record_on(s) for s in online.agent_stats],
            'pooled_emit_latency_ms': _stats(_pool(online.agent_stats, 'emit_latencies_ms')),
            'pooled_staleness_steps': _stats(_pool(online.agent_stats, 'staleness_steps')),
            'throughput_steps_per_sec': _throughput(online.ctrl_step_times),
        },
        'offline': {
            'per_agent': [agent_record_off(s) for s in offline.agent_stats],
            'pooled_step_latency_ms':  _stats(_pool(offline.agent_stats, 'step_latencies_ms')),
            'pooled_blocking_steps':   _stats(_pool(offline.agent_stats, 'blocking_steps')),
            'joint_blocking_steps':    _stats(offline.joint_blocking_steps),
            'throughput_steps_per_sec': _throughput(offline.ctrl_step_times),
        },
    }

    with open(outdir / 'latency_overcooked_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    lines = []
    W = 72
    lines.append('─' * W)
    lines.append(f'  Overcooked ({n} agents) — Online vs. Offline Planning Latency Summary')
    lines.append('─' * W)
    lines.append(f"  {'Metric':<40}  {'Online':>12}  {'Offline':>12}")
    lines.append('  ' + '─' * (W - 2))

    def row(label, on_val, off_val, fmt='.1f'):
        lines.append(f"  {label:<40}  {on_val:>12{fmt}}  {off_val:>12{fmt}}")

    r = results
    row('VLM latency mean (ms) [pooled]',
        r['online']['pooled_emit_latency_ms']['mean'],
        r['offline']['pooled_step_latency_ms']['mean'])
    row('VLM latency p95 (ms) [pooled]',
        r['online']['pooled_emit_latency_ms']['p95'],
        r['offline']['pooled_step_latency_ms']['p95'])
    row('VLM latency std (ms) [pooled]',
        r['online']['pooled_emit_latency_ms']['std'],
        r['offline']['pooled_step_latency_ms']['std'])

    for i in range(n):
        ag_on  = r['online']['per_agent'][i]
        ag_off = r['offline']['per_agent'][i]
        row(f'Agent {i}: emit/step latency mean (ms)',
            ag_on['emit_latency_ms']['mean'],
            ag_off['step_latency_ms']['mean'])
        row(f'Agent {i}: staleness mean (steps)',
            ag_on['staleness_steps']['mean'], 0)
        row(f'Agent {i}: blocking steps mean',
            0, ag_off['blocking_steps']['mean'])
        row(f'Agent {i}: instructions issued',
            ag_on['instructions_issued'],
            ag_off['instructions_issued'], 'd')

    row('Joint blocking steps mean (offline)',
        0, r['offline']['joint_blocking_steps']['mean'])
    row('Joint blocking steps p95 (offline)',
        0, r['offline']['joint_blocking_steps']['p95'])
    row('Throughput (env steps/sec)',
        r['online']['throughput_steps_per_sec'],
        r['offline']['throughput_steps_per_sec'])
    lines.append('─' * W)

    summary_txt = '\n'.join(lines)
    print('\n' + summary_txt + '\n')

    with open(outdir / 'latency_overcooked_summary.txt', 'w') as f:
        f.write(summary_txt + '\n')

    return results


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

CKPT_ROOT = Path('/data/terran/instruct_to_act')


def resolve_checkpoint(logdir: Path) -> Path:
    """Return agent.pkl path, following the 'latest' pointer if present."""
    latest_file = logdir / 'ckpt' / 'latest'
    if latest_file.exists():
        tag = latest_file.read_text().strip()
        return logdir / 'ckpt' / tag / 'agent.pkl'
    candidates = sorted((logdir / 'ckpt').glob('*/agent.pkl'))
    if candidates:
        return candidates[-1]
    raise FileNotFoundError(f"No checkpoint found under {logdir}/ckpt/")


def parse_args():
    p = argparse.ArgumentParser(
        description='Online vs. offline planning latency analysis — Overcooked'
    )
    p.add_argument('--logdir_seed', default=str(CKPT_ROOT / 'overcooked' / 'seed0'),
                   help='Run directory containing ckpt/latest')
    p.add_argument('--checkpoint',  default=None,
                   help='Direct path to agent.pkl; overrides --logdir_seed')
    p.add_argument('--planner',    default='scripted',
                   choices=['gpt4o', 'qwen', 'gemma', 'llava', 'scripted'])
    p.add_argument('--episodes',   type=int, default=50)
    p.add_argument('--max_steps',  type=int, default=400,
                   help='Steps per episode (Overcooked default = 400)')
    p.add_argument('--n_agents',   type=int, default=2)
    p.add_argument('--stop_threshold', type=float, default=0.5)
    p.add_argument('--outdir',     default='results/latency_overcooked')
    p.add_argument('--mock',       action='store_true',
                   help='Use mock VLM and env (no API key required)')
    p.add_argument('--mock_mean_ms', type=float, default=380.0)
    p.add_argument('--mock_std_ms',  type=float, default=70.0)
    p.add_argument('--seed',       type=int, default=0)
    return p.parse_args()


def main():
    args   = parse_args()
    logdir = Path(args.outdir)
    logdir.mkdir(parents=True, exist_ok=True)
    n = args.n_agents

    np.random.seed(args.seed)

    if args.mock:
        print(f"── Mock mode ({n} agents, no VLM / server) ──")
        planners = [MockPlanner(mean_ms=args.mock_mean_ms, std_ms=args.mock_std_ms,
                                agent_id=i) for i in range(n)]
        ctrls    = [MockController() for _ in range(n)]
        envs     = [MockMultiAgentEnv() for _ in range(n)]
        encoder  = mock_encoder
    else:
        root = Path(__file__).parent.parent
        sys.path.insert(0, str(root))
        sys.path.insert(0, str(root / 'dreamerv3'))
        from planners import make_planner
        from envs     import make_env

        ckpt_path = (Path(args.checkpoint) if args.checkpoint
                     else resolve_checkpoint(Path(args.logdir_seed)))
        print(f"── Checkpoint: {ckpt_path} ──")

        from train    import load_project_config
        from evaluate import load_agent, GymAdapter, encode_fn

        import pickle
        _ckpt_data  = pickle.loads(Path(ckpt_path).read_bytes())
        _ckpt_units = int(_ckpt_data['params']['con/head/logit/kernel'].shape[0])

        config_paths = [str(root / 'configs' / 'base.yaml'),
                        str(root / 'configs' / 'overcooked.yaml')]
        size200m_units = 1024  # size200m.yaml: units=1024
        if _ckpt_units == size200m_units:
            config_paths.append(str(root / 'configs' / 'size200m.yaml'))
            print("── Applying size200m.yaml (detected from checkpoint units) ──")
        proj = load_project_config(config_paths)

        lang_size = int(proj.get('lang_size', 384))
        encoder   = encode_fn(lang_size)
        # One env per agent (independent state); shared agent weights
        envs  = [GymAdapter(make_env('overcooked', proj, lang_size=lang_size))
                 for _ in range(n)]
        ctrl  = load_agent(str(ckpt_path), envs[0].obs_space, envs[0].act_space, proj)
        ctrls = [ctrl] * n   # shared parameters, independent recurrent state
        task_spec = proj.get('task_guidance', '')
        planners  = [make_planner(args.planner, task_spec=task_spec) for _ in range(n)]

    online = MultiAgentOnlineInference(
        ctrls, planners, envs, encoder,
        stop_threshold=args.stop_threshold, max_steps=args.max_steps, n_agents=n,
    )
    offline = MultiAgentOfflineInference(
        ctrls, planners, envs, encoder,
        stop_threshold=args.stop_threshold, max_steps=args.max_steps, n_agents=n,
    )

    # ── Online episodes ──────────────────────────────────────────────────────
    print(f"\n── Online (async) — {args.episodes} episodes ──")
    online_rewards = []
    for ep in range(args.episodes):
        result = online.run_episode()
        online_rewards.append(result['total_reward'])
        total_instrs = sum(s.instructions_issued for s in online.agent_stats)
        pooled_emit  = _pool(online.agent_stats, 'emit_latencies_ms')
        print(f"  ep {ep+1:3d}/{args.episodes}  reward={result['total_reward']:.1f}"
              f"  steps={result['steps']}"
              f"  instrs={total_instrs}"
              f"  emit_ms={np.mean(pooled_emit or [0]):.0f}")

    # ── Offline episodes ─────────────────────────────────────────────────────
    print(f"\n── Offline (sync) — {args.episodes} episodes ──")
    offline_rewards = []
    for ep in range(args.episodes):
        result = offline.run_episode()
        offline_rewards.append(result['total_reward'])
        total_instrs = sum(s.instructions_issued for s in offline.agent_stats)
        pooled_step  = _pool(offline.agent_stats, 'step_latencies_ms')
        print(f"  ep {ep+1:3d}/{args.episodes}  reward={result['total_reward']:.1f}"
              f"  steps={result['steps']}"
              f"  instrs={total_instrs}"
              f"  step_ms={np.mean(pooled_step or [0]):.0f}")

    # ── Outputs ──────────────────────────────────────────────────────────────
    print('\n── Saving outputs ──')
    plot_comparison(online, offline, logdir)
    save_summary(online, offline, logdir)
    print(f"All outputs in {logdir}/")


if __name__ == '__main__':
    main()
