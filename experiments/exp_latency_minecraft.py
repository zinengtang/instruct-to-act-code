"""
exp_latency_minecraft.py — Online vs. Offline planning latency analysis for Minecraft Diamond.

Addresses Reviewer Q4:
  "Fig. 2(b) compares online vs. offline planning with two bars per scale on a
   single task. Could the authors provide latency distributions or instruction-
   staleness statistics across more tasks to support the efficiency claim?"

Measures (Minecraft ObtainDiamond):
  1. VLM call latency distribution: emit() [online] vs step() [offline]  (ms)
  2. Instruction staleness [online]: steps between VLM obs-capture and instruction arrival
  3. Controller blocking [offline]: equivalent steps lost to VLM wait at each boundary
  4. Controller throughput: env steps/sec [online vs offline]

Outputs:
  <logdir>/
    latency_minecraft_results.json          — raw per-call stats
    latency_minecraft_distributions.pdf/png — histograms + CDFs + box plots
    latency_minecraft_summary.txt           — table for paper rebuttal

Usage:
  python experiments/exp_latency_minecraft.py \\
      --checkpoint logdir/minecraft/seed0/checkpoint.pkl \\
      --planner gpt4o \\
      --episodes 20 \\
      --max_steps 1000 \\
      --logdir results/latency_minecraft

  # Dry-run with mock VLM (no API key required):
  python experiments/exp_latency_minecraft.py \\
      --mock --mock_mean_ms 420 --mock_std_ms 80 --episodes 20
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
# Mock planner (no VLM API needed for offline analysis / CI)
# ─────────────────────────────────────────────────────────────────────────────

class MockPlanner:
    """Simulates VLM latency with a Gaussian distribution."""

    INSTRUCTIONS = [
        "Collect wood from trees.",
        "Craft a crafting table from wooden planks.",
        "Mine stone blocks to gather cobblestone.",
        "Craft a stone pickaxe from cobblestone.",
        "Dig deep underground to find iron ore.",
        "Mine iron ore with a stone pickaxe.",
        "Smelt iron ore in a furnace to get ingots.",
        "Craft an iron pickaxe from iron ingots.",
        "Dig down to reach the stone layer.",
        "Move towards the forest to gather resources.",
    ]

    def __init__(self, mean_ms=420.0, std_ms=80.0):
        self.mean_ms = mean_ms
        self.std_ms  = std_ms
        self._rng    = np.random.default_rng(42)
        self._idx    = 0

    def _sleep(self):
        delay_s = max(0.001, self._rng.normal(self.mean_ms, self.std_ms) / 1000.0)
        time.sleep(delay_s)

    def emit(self, obs, plan, memory):
        self._sleep()
        instr = self.INSTRUCTIONS[self._idx % len(self.INSTRUCTIONS)]
        self._idx += 1
        return instr, plan, memory

    def advance(self, obs, plan, memory):
        # advance() does deeper reasoning — slightly longer
        delay_s = max(0.001, self._rng.normal(self.mean_ms * 1.3, self.std_ms) / 1000.0)
        time.sleep(delay_s)
        return plan, memory

    def step(self, obs, plan, memory, p_stop=0.0):
        self._sleep()
        instr = self.INSTRUCTIONS[self._idx % len(self.INSTRUCTIONS)]
        self._idx += 1
        return instr, plan, memory


# ─────────────────────────────────────────────────────────────────────────────
# Mock controller + env for dry-run
# ─────────────────────────────────────────────────────────────────────────────

class MockController:
    _lang_size = 384

    def init_policy(self, batch_size=1):
        return None

    def policy(self, state, obs_batch, mode='eval'):
        time.sleep(0.002)  # ~500 steps/s controller rate
        rng = np.random.default_rng()
        p_stop = float(rng.random() < 0.04)  # ~4% per step → ~25 steps/instruction
        action = {'action': np.zeros((1, 1))}
        extra  = {'log/p_stop': p_stop}
        return state, action, extra


class MockEnv:
    def reset(self):
        return {'image': np.zeros((64, 64, 3), dtype=np.uint8)}, {}

    def step(self, action):
        obs = {'image': np.zeros((64, 64, 3), dtype=np.uint8)}
        return obs, 0.0, False, False, {}


def mock_encoder(text: str) -> np.ndarray:
    return np.random.randn(384).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Instrumented Online (async) Inference
# ─────────────────────────────────────────────────────────────────────────────

class InstrumentedOnlineInference:
    """
    Async (online) inference with per-call latency and per-instruction staleness tracking.

    Staleness at step t = (t - t_obs), where t_obs is the controller step at which
    the VLM captured its observation when it started generating the instruction
    that arrives at step t.
    """

    def __init__(self, controller, planner, env, lang_encoder,
                 stop_threshold=0.5, max_steps=1000):
        self.ctrl     = controller
        self.planner  = planner
        self.env      = env
        self.encode   = lang_encoder
        self.stop_thr = stop_threshold
        self.max_steps= max_steps

        # Accumulated across all run_episode() calls
        self.emit_latencies_ms    = []
        self.advance_latencies_ms = []
        self.staleness_steps      = []
        self.ctrl_step_times      = []
        self.instructions_issued  = 0

    def run_episode(self) -> dict:
        from inference.algorithms import AsyncState

        shared     = AsyncState()
        ctrl_state = self.ctrl.init_policy(batch_size=1)
        lang_size  = getattr(self.ctrl, '_lang_size', 384)
        null_embed = np.zeros(lang_size, dtype=np.float32)
        cur_embed  = null_embed.copy()

        obs, _ = self.env.reset()
        shared.update_obs(obs)
        total_reward = 0.0

        # Shared between threads: which ctrl step was current when VLM last started an emit
        vlm_capture_step = [0]
        ctrl_step_ctr    = [0]
        pending_capture  = [None]  # set by VLM thread, consumed by ctrl thread

        def vlm_loop():
            memory, plan = {}, None
            while shared.running:
                latest_obs = shared.get_obs()
                if latest_obs is None:
                    time.sleep(0.005)
                    continue

                if shared.stop:
                    capture = ctrl_step_ctr[0]
                    t0 = time.perf_counter()
                    instr, plan, memory = self.planner.emit(
                        obs=latest_obs, plan=plan, memory=memory
                    )
                    self.emit_latencies_ms.append((time.perf_counter() - t0) * 1000)

                    embed = self.encode(instr)
                    pending_capture[0] = capture
                    shared.put_inbox(embed, instr)
                    shared.clear_stop()
                else:
                    t0 = time.perf_counter()
                    plan, memory = self.planner.advance(
                        obs=latest_obs, plan=plan, memory=memory
                    )
                    self.advance_latencies_ms.append((time.perf_counter() - t0) * 1000)
                time.sleep(0.001)

        vlm_thread = threading.Thread(target=vlm_loop, daemon=True)
        vlm_thread.start()
        shared.set_stop()

        for step in range(self.max_steps):
            self.ctrl_step_times.append(time.perf_counter())
            ctrl_step_ctr[0] = step

            new_embed, new_text = shared.pop_inbox()
            if new_embed is not None:
                cur_embed = new_embed
                if pending_capture[0] is not None:
                    self.staleness_steps.append(max(0, step - pending_capture[0]))
                    pending_capture[0] = None
                self.instructions_issued += 1

            obs_b = {k: np.expand_dims(v, 0) for k, v in obs.items()
                     if not k.startswith('log/')}
            obs_b['lang_embed'] = cur_embed[np.newaxis]

            ctrl_state, action, extra = self.ctrl.policy(ctrl_state, obs_b, mode='eval')
            p_stop = float(extra.get('log/p_stop', extra.get('p_stop', 0.0)))

            action_np = np.array(action['action'][0])
            try:
                obs, reward, terminated, truncated, _ = self.env.step(action_np)
            except (ConnectionRefusedError, TimeoutError, OSError):
                break
            shared.update_obs(obs)
            total_reward += float(reward)

            if p_stop > self.stop_thr:
                cur_embed = null_embed.copy()
                shared.set_stop()

            if terminated or truncated:
                break

        shared.running = False
        return {'total_reward': total_reward, 'steps': step + 1}


# ─────────────────────────────────────────────────────────────────────────────
# Instrumented Offline (synchronous) Inference
# ─────────────────────────────────────────────────────────────────────────────

class InstrumentedOfflineInference:
    """
    Synchronous (offline) inference.  At each instruction boundary the controller
    blocks until the VLM returns.  Tracks VLM latency and the equivalent number
    of steps the controller was blocked.
    """

    def __init__(self, controller, planner, env, lang_encoder,
                 stop_threshold=0.5, max_steps=1000):
        self.ctrl     = controller
        self.planner  = planner
        self.env      = env
        self.encode   = lang_encoder
        self.stop_thr = stop_threshold
        self.max_steps= max_steps

        self.step_latencies_ms   = []  # VLM call latency per instruction boundary
        self.blocking_steps      = []  # equivalent steps lost per boundary
        self.ctrl_step_times     = []
        self.step_durations_ms   = []  # rolling wall-clock per step (ex VLM wait)
        self.instructions_issued = 0

    def run_episode(self) -> dict:
        ctrl_state = self.ctrl.init_policy(batch_size=1)
        lang_size  = getattr(self.ctrl, '_lang_size', 384)
        null_embed = np.zeros(lang_size, dtype=np.float32)
        cur_embed  = null_embed.copy()

        obs, _ = self.env.reset()
        plan, memory = None, {}
        stop_flag    = True
        total_reward = 0.0

        step_dur_est_ms = None  # running estimate of per-step duration (excl. VLM blocking)

        for step in range(self.max_steps):
            self.ctrl_step_times.append(time.perf_counter())

            if stop_flag:
                t0 = time.perf_counter()
                instr, plan, memory = self.planner.step(
                    obs=obs, plan=plan, memory=memory, p_stop=float(stop_flag)
                )
                vlm_ms = (time.perf_counter() - t0) * 1000
                self.step_latencies_ms.append(vlm_ms)

                if instr:
                    cur_embed = self.encode(instr)
                    stop_flag = False
                    self.instructions_issued += 1
                    if step_dur_est_ms and step_dur_est_ms > 0:
                        self.blocking_steps.append(int(vlm_ms / step_dur_est_ms))

            t_ctrl = time.perf_counter()
            obs_b  = {k: np.expand_dims(v, 0) for k, v in obs.items()
                      if not k.startswith('log/')}
            obs_b['lang_embed'] = cur_embed[np.newaxis]

            ctrl_state, action, extra = self.ctrl.policy(ctrl_state, obs_b, mode='eval')
            p_stop = float(extra.get('log/p_stop', extra.get('p_stop', 0.0)))

            action_np = np.array(action['action'][0])
            try:
                obs, reward, terminated, truncated, _ = self.env.step(action_np)
            except (ConnectionRefusedError, TimeoutError, OSError):
                break
            total_reward += float(reward)

            ctrl_ms = (time.perf_counter() - t_ctrl) * 1000
            self.step_durations_ms.append(ctrl_ms)
            if len(self.step_durations_ms) <= 100:
                step_dur_est_ms = float(np.mean(self.step_durations_ms))

            if p_stop > self.stop_thr:
                cur_embed = null_embed.copy()
                stop_flag = True

            if terminated or truncated:
                break

        return {'total_reward': total_reward, 'steps': step + 1}


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────

BLUE   = '#1565C0'
ORANGE = '#E65100'
ALPHA  = 0.72


def _cdf(data):
    x = np.sort(data)
    y = np.arange(1, len(x) + 1) / len(x)
    return x, y


def _throughput(times):
    if len(times) < 2:
        return 0.0
    return len(times) / max(times[-1] - times[0], 1e-9)


def plot_comparison(online: InstrumentedOnlineInference,
                    offline: InstrumentedOfflineInference,
                    outdir: Path):
    fig = plt.figure(figsize=(15, 9))
    fig.suptitle(
        'Minecraft Diamond — Online vs. Offline Planning: Latency & Staleness',
        fontsize=13, fontweight='bold', y=0.99,
    )
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.50, wspace=0.35)

    # (a) VLM latency histogram
    ax = fig.add_subplot(gs[0, 0])
    if online.emit_latencies_ms:
        ax.hist(online.emit_latencies_ms, bins=30, color=BLUE, alpha=ALPHA,
                label=f'Online emit()  n={len(online.emit_latencies_ms)}', density=True)
    if offline.step_latencies_ms:
        ax.hist(offline.step_latencies_ms, bins=30, color=ORANGE, alpha=ALPHA,
                label=f'Offline step()  n={len(offline.step_latencies_ms)}', density=True)
    ax.set_xlabel('VLM latency (ms)')
    ax.set_ylabel('Density')
    ax.set_title('(a) VLM Call Latency')
    ax.legend(fontsize=7)

    # (b) VLM latency CDF
    ax = fig.add_subplot(gs[0, 1])
    if online.emit_latencies_ms:
        x, y = _cdf(online.emit_latencies_ms)
        ax.plot(x, y, color=BLUE, lw=2, label='Online emit()')
    if offline.step_latencies_ms:
        x, y = _cdf(offline.step_latencies_ms)
        ax.plot(x, y, color=ORANGE, lw=2, label='Offline step()')
    ax.set_xlabel('VLM latency (ms)')
    ax.set_ylabel('CDF')
    ax.set_title('(b) Latency CDF')
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # (c) latency box plot (log scale)
    ax = fig.add_subplot(gs[0, 2])
    data, labels, colors = [], [], []
    for d, lbl, c in [(online.emit_latencies_ms, 'Online\nemit()', BLUE),
                       (offline.step_latencies_ms, 'Offline\nstep()', ORANGE)]:
        if d:
            data.append(d); labels.append(lbl); colors.append(c)
    if data:
        bp = ax.boxplot(data, labels=labels, patch_artist=True, notch=False,
                        medianprops=dict(color='white', lw=2))
        for patch, c in zip(bp['boxes'], colors):
            patch.set_facecolor(c); patch.set_alpha(ALPHA)
    ax.set_ylabel('VLM latency (ms)')
    ax.set_title('(c) Latency Box Plot')
    ax.set_yscale('log')

    # (d) staleness histogram (online only)
    ax = fig.add_subplot(gs[1, 0])
    if online.staleness_steps:
        st = online.staleness_steps
        ax.hist(st, bins=30, color=BLUE, alpha=ALPHA, density=True)
        ax.axvline(np.mean(st), color='crimson', ls='--', lw=2,
                   label=f'mean = {np.mean(st):.1f} steps')
        ax.axvline(np.percentile(st, 95), color='darkred', ls=':', lw=1.5,
                   label=f'p95 = {np.percentile(st, 95):.1f} steps')
        ax.legend(fontsize=7)
    ax.set_xlabel('Steps (instruction staleness)')
    ax.set_ylabel('Density')
    ax.set_title('(d) Online: Instruction Staleness\n(steps between VLM obs and instr arrival)')

    # (e) blocking steps histogram (offline only)
    ax = fig.add_subplot(gs[1, 1])
    if offline.blocking_steps:
        bl = offline.blocking_steps
        ax.hist(bl, bins=30, color=ORANGE, alpha=ALPHA, density=True)
        ax.axvline(np.mean(bl), color='crimson', ls='--', lw=2,
                   label=f'mean = {np.mean(bl):.1f} steps')
        ax.axvline(np.percentile(bl, 95), color='darkred', ls=':', lw=1.5,
                   label=f'p95 = {np.percentile(bl, 95):.1f} steps')
        ax.legend(fontsize=7)
    ax.set_xlabel('Equivalent blocked steps')
    ax.set_ylabel('Density')
    ax.set_title('(e) Offline: Controller Blocking\n(steps missed per instruction boundary)')

    # (f) throughput bar chart
    ax = fig.add_subplot(gs[1, 2])
    tput_on  = _throughput(online.ctrl_step_times)
    tput_off = _throughput(offline.ctrl_step_times)
    bars = ax.bar(['Online', 'Offline'], [tput_on, tput_off],
                  color=[BLUE, ORANGE], alpha=ALPHA, edgecolor='white', width=0.5)
    for bar, val in zip(bars, [tput_on, tput_off]):
        ax.text(bar.get_x() + bar.get_width() / 2, val * 1.02,
                f'{val:.1f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax.set_ylabel('Env steps / second')
    ax.set_title('(f) Controller Throughput')
    ax.set_ylim(0, max(tput_on, tput_off) * 1.2 if max(tput_on, tput_off) > 0 else 1)

    stem = 'latency_minecraft'
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


def save_summary(online, offline, outdir, task='Minecraft ObtainDiamond'):
    results = {
        'task': task,
        'online': {
            'emit_latency_ms':    _stats(online.emit_latencies_ms),
            'advance_latency_ms': _stats(online.advance_latencies_ms),
            'staleness_steps':    _stats(online.staleness_steps),
            'instructions_issued': online.instructions_issued,
            'throughput_steps_per_sec': _throughput(online.ctrl_step_times),
        },
        'offline': {
            'step_latency_ms':   _stats(offline.step_latencies_ms),
            'blocking_steps':    _stats(offline.blocking_steps),
            'step_duration_ms':  _stats(offline.step_durations_ms),
            'instructions_issued': offline.instructions_issued,
            'throughput_steps_per_sec': _throughput(offline.ctrl_step_times),
        },
    }

    with open(outdir / 'latency_minecraft_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    lines = []
    W = 70
    lines.append('─' * W)
    lines.append('  Minecraft Diamond — Online vs. Offline Planning Latency Summary')
    lines.append('─' * W)
    lines.append(f"  {'Metric':<38}  {'Online':>12}  {'Offline':>12}")
    lines.append('  ' + '─' * (W - 2))

    def row(label, on_val, off_val, fmt='.1f'):
        lines.append(f"  {label:<38}  {on_val:>12{fmt}}  {off_val:>12{fmt}}")

    r = results
    row('VLM latency mean (ms)',
        r['online']['emit_latency_ms']['mean'],
        r['offline']['step_latency_ms']['mean'])
    row('VLM latency median (ms)',
        r['online']['emit_latency_ms']['median'],
        r['offline']['step_latency_ms']['median'])
    row('VLM latency p95 (ms)',
        r['online']['emit_latency_ms']['p95'],
        r['offline']['step_latency_ms']['p95'])
    row('VLM latency std (ms)',
        r['online']['emit_latency_ms']['std'],
        r['offline']['step_latency_ms']['std'])
    row('advance() latency mean (ms)',
        r['online']['advance_latency_ms']['mean'], 0)
    row('Staleness mean (steps)',
        r['online']['staleness_steps']['mean'], 0)
    row('Staleness p95 (steps)',
        r['online']['staleness_steps']['p95'], 0)
    row('Blocking steps mean',
        0, r['offline']['blocking_steps']['mean'])
    row('Blocking steps p95',
        0, r['offline']['blocking_steps']['p95'])
    row('Instructions issued',
        r['online']['instructions_issued'],
        r['offline']['instructions_issued'], 'd')
    row('Throughput (steps/sec)',
        r['online']['throughput_steps_per_sec'],
        r['offline']['throughput_steps_per_sec'])
    lines.append('─' * W)

    summary_txt = '\n'.join(lines)
    print('\n' + summary_txt + '\n')

    with open(outdir / 'latency_minecraft_summary.txt', 'w') as f:
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
    # Fallback: find any agent.pkl under ckpt/
    candidates = sorted((logdir / 'ckpt').glob('*/agent.pkl'))
    if candidates:
        return candidates[-1]
    raise FileNotFoundError(f"No checkpoint found under {logdir}/ckpt/")


ENV_DEFAULTS = {
    'minecraft': {'logdir': 'mc200m_seed0', 'outdir': 'results/latency_minecraft',
                  'max_steps': 1000, 'needs_xvfb': True},
    'crafter':   {'logdir': 'seed0',        'outdir': 'results/latency_crafter',
                  'max_steps': 1000, 'needs_xvfb': False},
}


def parse_args():
    p = argparse.ArgumentParser(
        description='Online vs. offline planning latency analysis (single-agent envs)'
    )
    p.add_argument('--env',         default='minecraft',
                   choices=list(ENV_DEFAULTS.keys()),
                   help='Environment to evaluate')
    p.add_argument('--logdir_seed', default=None,
                   help='Run directory containing ckpt/latest; defaults per --env')
    p.add_argument('--checkpoint',  default=None,
                   help='Direct path to agent.pkl; overrides --logdir_seed')
    p.add_argument('--planner',    default='scripted',
                   choices=['gpt4o', 'qwen', 'gemma', 'llava', 'scripted'])
    p.add_argument('--episodes',   type=int, default=20)
    p.add_argument('--max_steps',  type=int, default=None,
                   help='Steps per episode; defaults per --env')
    p.add_argument('--stop_threshold', type=float, default=0.5)
    p.add_argument('--outdir',     default=None,
                   help='Output directory; defaults per --env')
    p.add_argument('--mock',       action='store_true',
                   help='Use mock VLM and env (no API key / server required)')
    p.add_argument('--mock_mean_ms', type=float, default=420.0)
    p.add_argument('--mock_std_ms',  type=float, default=80.0)
    p.add_argument('--seed',       type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()

    # Apply per-env defaults
    env_cfg = ENV_DEFAULTS[args.env]
    if args.logdir_seed is None:
        args.logdir_seed = str(CKPT_ROOT / env_cfg['logdir'])
    if args.max_steps is None:
        args.max_steps = env_cfg['max_steps']
    if args.outdir is None:
        args.outdir = env_cfg['outdir']

    logdir = Path(args.outdir)
    logdir.mkdir(parents=True, exist_ok=True)

    np.random.seed(args.seed)

    if args.mock:
        print(f"── Mock mode ({args.env}, no VLM / server) ──")
        planner  = MockPlanner(mean_ms=args.mock_mean_ms, std_ms=args.mock_std_ms)
        ctrl     = MockController()
        env      = MockEnv()
        encoder  = mock_encoder
    else:
        root = Path(__file__).parent.parent
        sys.path.insert(0, str(root))
        sys.path.insert(0, str(root / 'dreamerv3'))
        from planners import make_planner
        from envs     import make_env

        ckpt_path = (Path(args.checkpoint) if args.checkpoint
                     else resolve_checkpoint(Path(args.logdir_seed)))
        print(f"── Env: {args.env}  Checkpoint: {ckpt_path} ──")

        from train    import load_project_config
        from evaluate import load_agent, GymAdapter, encode_fn

        config_paths = [str(root / 'configs' / 'base.yaml'),
                        str(root / 'configs' / f'{args.env}.yaml')]
        logdir_name  = Path(args.logdir_seed).name
        if '200m' in logdir_name:
            config_paths.append(str(root / 'configs' / 'size200m.yaml'))
            print(f"── Applying size200m.yaml (detected from {logdir_name}) ──")
        proj      = load_project_config(config_paths)
        lang_size = int(proj.get('lang_size', 384))
        encoder   = encode_fn(lang_size)
        env       = GymAdapter(make_env(args.env, proj, lang_size=lang_size))
        ctrl      = load_agent(str(ckpt_path), env.obs_space, env.act_space, proj)
        planner   = make_planner(args.planner, task_spec=proj.get('task_guidance', ''))

    online  = InstrumentedOnlineInference(
        ctrl, planner, env, encoder,
        stop_threshold=args.stop_threshold, max_steps=args.max_steps,
    )
    offline = InstrumentedOfflineInference(
        ctrl, planner, env, encoder,
        stop_threshold=args.stop_threshold, max_steps=args.max_steps,
    )

    # ── Online episodes ──────────────────────────────────────────────────────
    print(f"\n── Online (async) — {args.episodes} episodes ──")
    online_rewards = []
    for ep in range(args.episodes):
        result = online.run_episode()
        online_rewards.append(result['total_reward'])
        print(f"  ep {ep+1:3d}/{args.episodes}  reward={result['total_reward']:.2f}"
              f"  steps={result['steps']}"
              f"  instrs={online.instructions_issued}"
              f"  emit_ms={np.mean(online.emit_latencies_ms or [0]):.0f}")

    # ── Offline episodes ─────────────────────────────────────────────────────
    print(f"\n── Offline (sync) — {args.episodes} episodes ──")
    offline_rewards = []
    for ep in range(args.episodes):
        result = offline.run_episode()
        offline_rewards.append(result['total_reward'])
        print(f"  ep {ep+1:3d}/{args.episodes}  reward={result['total_reward']:.2f}"
              f"  steps={result['steps']}"
              f"  instrs={offline.instructions_issued}"
              f"  step_ms={np.mean(offline.step_latencies_ms or [0]):.0f}")

    # ── Outputs ──────────────────────────────────────────────────────────────
    print('\n── Saving outputs ──')
    plot_comparison(online, offline, logdir)
    save_summary(online, offline, logdir, task=args.env)
    print(f"All outputs in {logdir}/")


if __name__ == '__main__':
    main()
