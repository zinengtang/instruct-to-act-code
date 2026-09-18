"""
exp_horizon_minecraft.py — Instruction horizon sensitivity for Minecraft Diamond.

Addresses Reviewer Q13:
  "How does performance change as the instruction horizon varies? Are there tasks
   where very short or very long instructions break the interface?"

Two sweeps
──────────
Sweep A — Token length:
  Constrain instructions to fixed word-count budgets:
    tiny   :  3–5 words    (e.g. "collect wood")
    short  :  6–9 words    (paper default ~7)
    medium : 10–15 words
    long   : 16–25 words
    free   :  no constraint

Sweep B — Replanning cadence K:
  Force instruction replacement every K controller steps, ignoring p_stop:
    K=8, K=16, K=32, K=64, adaptive (paper default — use p_stop head)

Metrics per condition: mean ± std episode reward, instructions issued,
mean instruction length consumed.

Usage
─────
  # Real checkpoint + scripted planner:
  python experiments/exp_horizon_minecraft.py \\
      --episodes 20 --max_steps 1000 \\
      --outdir results/horizon_minecraft

  # Mock (no GPU / server):
  python experiments/exp_horizon_minecraft.py --mock --episodes 10 --max_steps 200
"""

import argparse
import json
import sys
import threading
import time
import datetime
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# ─────────────────────────────────────────────────────────────────────────────
# Progress logger
# ─────────────────────────────────────────────────────────────────────────────

class ProgressLogger:
    """
    Writes one JSONL line per episode to <outdir>/progress.jsonl.
    Also prints timestamped lines with ETA to stdout.

    Usage:
        logger = ProgressLogger(outdir, total_episodes)
        logger.log(sweep='A', condition='short', ep=1, reward=0.5, n_instrs=12)

    Monitor from another terminal:
        tail -f results/horizon_minecraft/progress.jsonl
    """

    def __init__(self, outdir: Path, total_episodes: int):
        self._path    = outdir / 'progress.jsonl'
        self._total   = total_episodes
        self._done    = 0
        self._t0      = time.perf_counter()
        self._lock    = threading.Lock()
        outdir.mkdir(parents=True, exist_ok=True)

    def log(self, **kwargs):
        with self._lock:
            self._done += 1
            elapsed  = time.perf_counter() - self._t0
            avg_s    = elapsed / self._done
            remain   = avg_s * (self._total - self._done)
            eta_str  = str(datetime.timedelta(seconds=int(remain)))
            now_str  = datetime.datetime.now().strftime('%H:%M:%S')
            entry    = dict(
                ts       = now_str,
                elapsed  = round(elapsed, 1),
                eta      = eta_str,
                done     = self._done,
                total    = self._total,
                pct      = round(100 * self._done / self._total, 1),
                **kwargs,
            )
            with open(self._path, 'a') as f:
                f.write(json.dumps(entry) + '\n')
            # Human-readable stdout line
            reward_str = f"reward={kwargs.get('reward', 0):.4f}"
            n_str      = f"n_instrs={kwargs.get('n_instrs', '?')}"
            cond_str   = f"[{kwargs.get('sweep','?')}|{kwargs.get('condition','?')}]"
            ep_str     = f"ep {kwargs.get('ep', self._done):3d}/{self._total}"
            print(f"  {now_str}  {cond_str}  {ep_str}  {reward_str}  {n_str}"
                  f"  elapsed={elapsed/60:.1f}m  ETA={eta_str}")


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
    'K=8':      8,
    'K=16':     16,
    'K=32':     32,
    'K=64':     64,
    'adaptive': None,
}

CKPT_ROOT = Path('/data/terran/instruct_to_act')


# ─────────────────────────────────────────────────────────────────────────────
# Mock components
# ─────────────────────────────────────────────────────────────────────────────

class MockPlanner:
    INSTRUCTIONS = [
        "Collect wood from trees nearby the spawn area.",
        "Craft wooden planks and build a crafting table.",
        "Mine stone blocks to gather cobblestone for tools.",
        "Craft a stone pickaxe using cobblestone and sticks.",
        "Search underground for iron ore deposits.",
        "Mine iron ore with the stone pickaxe carefully.",
        "Smelt iron ore in a furnace to obtain ingots.",
        "Craft an iron pickaxe from iron ingots and sticks.",
        "Dig deep underground to find diamond ore deposits.",
        "Mine diamond ore carefully with the iron pickaxe.",
    ]

    def __init__(self, latency_ms=1.0):
        self._latency = latency_ms / 1000.0
        self._idx = 0
        self._rng = np.random.default_rng(0)

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
        time.sleep(0.002)
        p_stop = float(np.random.random() < 0.05)
        return state, {'action': np.zeros((1,), dtype=np.int32)}, {'log/p_stop': p_stop}


class MockEnv:
    def reset(self):
        return {'image': np.zeros((64, 64, 3), np.uint8)}, {}

    def step(self, action):
        r = float(np.random.random() < 0.01)
        return {'image': np.zeros((64, 64, 3), np.uint8)}, r, False, False, {}


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
    # Pad by repeating the last word rather than injecting unrelated filler
    while len(words) < min_w:
        words.append(words[-1] if words else 'proceed')
    return ' '.join(words)


class LengthConstrainedPlanner:
    """Wraps any planner and enforces a word-count budget on every emitted instruction."""

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
# Sweep A runner — length sweep (uses OfflineInference pattern)
# ─────────────────────────────────────────────────────────────────────────────

def run_length_episode(ctrl, planner: LengthConstrainedPlanner, env, encoder,
                       stop_threshold, max_steps):
    ctrl_state = ctrl.init_policy(batch_size=1)
    lang_size  = getattr(ctrl, '_lang_size', 384)
    null_embed = np.zeros(lang_size, dtype=np.float32)
    cur_embed  = null_embed.copy()

    obs, _    = env.reset()
    plan      = None
    memory    = {}
    stop_flag = True
    total_reward = 0.0
    n_instrs     = 0

    for step in range(max_steps):
        if stop_flag:
            instr, plan, memory = planner.step(
                obs=obs, plan=plan, memory=memory, p_stop=float(stop_flag)
            )
            if instr:
                cur_embed = encoder(instr)
                stop_flag = False
                n_instrs += 1

        obs_b = {k: np.expand_dims(v, 0) for k, v in obs.items()
                 if not k.startswith('log/')}
        obs_b['lang_embed'] = cur_embed[np.newaxis]

        ctrl_state, action, extra = ctrl.policy(ctrl_state, obs_b, mode='eval')
        p_stop = float(extra.get('log/p_stop', extra.get('p_stop', 0.0)))

        try:
            obs, reward, terminated, truncated, _ = env.step(
                np.array(action['action'][0])
            )
        except (ConnectionRefusedError, TimeoutError, OSError):
            break
        total_reward += float(reward)

        if p_stop > stop_threshold:
            cur_embed = null_embed.copy()
            stop_flag = True
        if terminated or truncated:
            break

    return total_reward, n_instrs


# ─────────────────────────────────────────────────────────────────────────────
# Sweep B runner — fixed cadence (ignores p_stop)
# ─────────────────────────────────────────────────────────────────────────────

def run_cadence_episode(ctrl, planner, env, encoder, K, max_steps):
    """
    K=None → use p_stop head (adaptive, paper default).
    K=int  → replace instruction every K steps regardless of p_stop.
    """
    ctrl_state = ctrl.init_policy(batch_size=1)
    lang_size  = getattr(ctrl, '_lang_size', 384)
    null_embed = np.zeros(lang_size, dtype=np.float32)

    obs, _   = env.reset()
    plan     = None
    memory   = {}
    n_instrs = 0

    # Emit first instruction
    instr, plan, memory = planner.emit(obs=obs, plan=plan, memory=memory)
    cur_embed = encoder(instr)
    n_instrs += 1

    total_reward = 0.0

    for step in range(max_steps):
        obs_b = {k: np.expand_dims(v, 0) for k, v in obs.items()
                 if not k.startswith('log/')}
        obs_b['lang_embed'] = cur_embed[np.newaxis]

        ctrl_state, action, extra = ctrl.policy(ctrl_state, obs_b, mode='eval')
        p_stop = float(extra.get('log/p_stop', extra.get('p_stop', 0.0)))

        try:
            obs, reward, terminated, truncated, _ = env.step(
                np.array(action['action'][0])
            )
        except (ConnectionRefusedError, TimeoutError, OSError):
            break
        total_reward += float(reward)

        replan = (K is not None and (step + 1) % K == 0) or \
                 (K is None and p_stop > 0.5)
        if replan:
            instr, plan, memory = planner.emit(obs=obs, plan=plan, memory=memory)
            cur_embed = encoder(instr)
            n_instrs += 1

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
# Statistics helpers
# ─────────────────────────────────────────────────────────────────────────────

def _stats(vals):
    a = np.array(vals, dtype=float)
    return {'mean': float(np.mean(a)), 'std': float(np.std(a)),
            'min':  float(np.min(a)),  'max': float(np.max(a)), 'n': len(a)}


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────

COLORS = plt.cm.tab10(np.linspace(0, 0.9, 10))
ALPHA  = 0.82


def _bar_sweep(conditions, rewards_mean, rewards_std, xlabel_vals,
               title, xlabel, ylabel, outdir, fname):
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(conditions))
    bars = ax.bar(x, rewards_mean, yerr=rewards_std, color=COLORS[:len(conditions)],
                  alpha=ALPHA, edgecolor='white', capsize=4, width=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(conditions, fontsize=10)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3, axis='y')
    # Annotate mean
    for bar, m, s in zip(bars, rewards_mean, rewards_std):
        ax.text(bar.get_x() + bar.get_width() / 2, m + s + 0.001,
                f'{m:.3f}', ha='center', va='bottom', fontsize=8)
    plt.tight_layout()
    outdir.mkdir(parents=True, exist_ok=True)
    fig.savefig(outdir / f'{fname}.pdf', bbox_inches='tight')
    fig.savefig(outdir / f'{fname}.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved {outdir}/{fname}.{{pdf,png}}")


def plot_length_sweep(sweep_a: dict, outdir: Path):
    conditions = list(LENGTH_CONDITIONS.keys())
    means = [sweep_a[c]['reward']['mean'] for c in conditions]
    stds  = [sweep_a[c]['reward']['std']  for c in conditions]
    mean_lens = [sweep_a[c]['mean_instr_len'] for c in conditions]

    # (1) Bar chart: condition → reward
    _bar_sweep(conditions, means, stds, mean_lens,
               title='Minecraft Diamond — Instruction Length vs. Reward',
               xlabel='Length condition',
               ylabel='Mean episode reward',
               outdir=outdir, fname='sweep_a_length_bar')

    # (2) Line: mean instruction length (words) → reward
    finite = [(l, m, s) for l, m, s in zip(mean_lens, means, stds)
              if l is not None and l > 0]
    if finite:
        ls, ms, ss = zip(*sorted(finite))
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.errorbar(ls, ms, yerr=ss, marker='o', color='#1565C0',
                    lw=2, capsize=4, label='Minecraft Diamond')
        ax.set_xlabel('Mean instruction length (words)')
        ax.set_ylabel('Mean episode reward')
        ax.set_title('Instruction length vs. reward\n(Minecraft Diamond)')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)
        plt.tight_layout()
        fig.savefig(outdir / 'sweep_a_length_curve.pdf', bbox_inches='tight')
        fig.savefig(outdir / 'sweep_a_length_curve.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved {outdir}/sweep_a_length_curve.{{pdf,png}}")


def plot_cadence_sweep(sweep_b: dict, outdir: Path):
    conditions  = list(CADENCE_CONDITIONS.keys())
    means = [sweep_b[c]['reward']['mean'] for c in conditions]
    stds  = [sweep_b[c]['reward']['std']  for c in conditions]

    _bar_sweep(conditions, means, stds, conditions,
               title='Minecraft Diamond — Replanning Cadence vs. Reward',
               xlabel='Cadence condition',
               ylabel='Mean episode reward',
               outdir=outdir, fname='sweep_b_cadence_bar')

    # Highlight adaptive vs fixed
    fig, ax = plt.subplots(figsize=(7, 4))
    fixed_conds = [c for c in conditions if c != 'adaptive']
    fixed_Ks    = [CADENCE_CONDITIONS[c] for c in fixed_conds]
    fixed_means = [sweep_b[c]['reward']['mean'] for c in fixed_conds]
    fixed_stds  = [sweep_b[c]['reward']['std']  for c in fixed_conds]
    ax.errorbar(fixed_Ks, fixed_means, yerr=fixed_stds,
                marker='s', color='#E65100', lw=2, capsize=4, label='Fixed cadence K')
    adap_mean = sweep_b['adaptive']['reward']['mean']
    adap_std  = sweep_b['adaptive']['reward']['std']
    ax.axhline(adap_mean, color='#1565C0', lw=2, ls='--', label='Adaptive (p_stop)')
    ax.fill_between([min(fixed_Ks), max(fixed_Ks)],
                    adap_mean - adap_std, adap_mean + adap_std,
                    color='#1565C0', alpha=0.15)
    ax.set_xlabel('Cadence K (steps per instruction)')
    ax.set_ylabel('Mean episode reward')
    ax.set_title('Fixed vs. adaptive replanning cadence\n(Minecraft Diamond)')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(outdir / 'sweep_b_cadence_curve.pdf', bbox_inches='tight')
    fig.savefig(outdir / 'sweep_b_cadence_curve.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved {outdir}/sweep_b_cadence_curve.{{pdf,png}}")


# ─────────────────────────────────────────────────────────────────────────────
# Summary table
# ─────────────────────────────────────────────────────────────────────────────

def print_summary(sweep_a, sweep_b):
    W = 72
    print(f"\n{'─'*W}")
    print(f"  Minecraft Diamond — Instruction Horizon Sweep Summary")
    print(f"{'─'*W}")
    print(f"  {'Condition':<18}  {'Reward mean':>12}  {'Reward std':>11}  "
          f"{'Mean len (w)':>13}  {'N instrs':>9}")
    print(f"  {'─'*(W-2)}")
    print("  Sweep A — Token length")
    for cond in LENGTH_CONDITIONS:
        d = sweep_a.get(cond, {})
        r = d.get('reward', {})
        print(f"  {cond:<18}  {r.get('mean',0):>12.4f}  {r.get('std',0):>11.4f}  "
              f"{d.get('mean_instr_len',0):>13.1f}  {d.get('mean_n_instrs',0):>9.1f}")
    print("  Sweep B — Cadence K")
    for cond in CADENCE_CONDITIONS:
        d = sweep_b.get(cond, {})
        r = d.get('reward', {})
        print(f"  {cond:<18}  {r.get('mean',0):>12.4f}  {r.get('std',0):>11.4f}  "
              f"{'—':>13}  {d.get('mean_n_instrs',0):>9.1f}")
    print(f"{'─'*W}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description='Instruction horizon sweep — Minecraft Diamond'
    )
    p.add_argument('--logdir_seed', default=str(CKPT_ROOT / 'mc200m_seed0'))
    p.add_argument('--checkpoint',  default=None,
                   help='Direct path to agent.pkl; overrides --logdir_seed')
    p.add_argument('--planner',     default='qwen',
                   choices=['gpt4o', 'qwen', 'gemma', 'llava', 'scripted'])
    p.add_argument('--episodes',    type=int, default=20)
    p.add_argument('--max_steps',   type=int, default=1000)
    p.add_argument('--stop_threshold', type=float, default=0.5)
    p.add_argument('--sweep',       choices=['length', 'cadence', 'both'], default='both')
    p.add_argument('--outdir',      default='results/horizon_minecraft')
    p.add_argument('--mock',        action='store_true')
    p.add_argument('--seed',        type=int, default=0)
    return p.parse_args()


def main():
    args   = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    np.random.seed(args.seed)

    if args.mock:
        print("── Mock mode ──")
        base_planner = MockPlanner(latency_ms=1.0)
        ctrl         = MockController()
        env          = MockEnv()
        encoder      = mock_encoder
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

        config_paths = [str(root / 'configs' / 'base.yaml'),
                        str(root / 'configs' / 'minecraft.yaml')]
        if '200m' in Path(args.logdir_seed).name:
            config_paths.append(str(root / 'configs' / 'size200m.yaml'))
            print("── Applying size200m.yaml ──")

        proj      = load_project_config(config_paths)
        lang_size = int(proj.get('lang_size', 384))
        encoder   = encode_fn(lang_size)
        env       = GymAdapter(make_env('minecraft', proj, lang_size=lang_size))
        ctrl      = load_agent(str(ckpt_path), env.obs_space, env.act_space, proj)
        base_planner = make_planner(args.planner, task_spec=proj.get('task_guidance', ''))

    results = {'sweep_a': {}, 'sweep_b': {}}

    n_length_conds  = len(LENGTH_CONDITIONS)  if args.sweep in ('length', 'both') else 0
    n_cadence_conds = len(CADENCE_CONDITIONS) if args.sweep in ('cadence', 'both') else 0
    total_eps = (n_length_conds + n_cadence_conds) * args.episodes
    plog = ProgressLogger(outdir, total_eps)
    print(f"Progress log: {outdir}/progress.jsonl  (tail -f to monitor)")

    # ── Sweep A: token length ────────────────────────────────────────────────
    if args.sweep in ('length', 'both'):
        print(f"\n══ Sweep A — Instruction token length ({args.episodes} eps each) ══")
        for cond in LENGTH_CONDITIONS:
            # Fresh planner wrapper per condition (resets issued_lengths)
            planner = LengthConstrainedPlanner(base_planner, cond)
            rewards, n_instrs = [], []

            for ep in range(args.episodes):
                r, ni = run_length_episode(
                    ctrl, planner, env, encoder, args.stop_threshold, args.max_steps
                )
                rewards.append(r); n_instrs.append(ni)
                plog.log(sweep='A', condition=cond, ep=ep + 1,
                         reward=r, n_instrs=ni,
                         mean_len=round(float(np.mean(planner.issued_lengths or [0])), 1))

            results['sweep_a'][cond] = {
                'reward':          _stats(rewards),
                'n_instrs':        _stats(n_instrs),
                'mean_instr_len':  float(np.mean(planner.issued_lengths or [0])),
                'std_instr_len':   float(np.std(planner.issued_lengths  or [0])),
                'mean_n_instrs':   float(np.mean(n_instrs)),
            }

    # ── Sweep B: cadence ─────────────────────────────────────────────────────
    if args.sweep in ('cadence', 'both'):
        print(f"\n══ Sweep B — Replanning cadence ({args.episodes} eps each) ══")
        for cond, K in CADENCE_CONDITIONS.items():
            rewards, n_instrs = [], []

            for ep in range(args.episodes):
                r, ni = run_cadence_episode(
                    ctrl, base_planner, env, encoder, K, args.max_steps
                )
                rewards.append(r); n_instrs.append(ni)
                plog.log(sweep='B', condition=cond, ep=ep + 1,
                         reward=r, n_instrs=ni, K=K)

            results['sweep_b'][cond] = {
                'reward':        _stats(rewards),
                'n_instrs':      _stats(n_instrs),
                'mean_n_instrs': float(np.mean(n_instrs)),
                'K':             K,
            }

    # ── Save + plot ──────────────────────────────────────────────────────────
    with open(outdir / 'horizon_minecraft_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved JSON to {outdir}/horizon_minecraft_results.json")

    if results['sweep_a']:
        plot_length_sweep(results['sweep_a'], outdir)
    if results['sweep_b']:
        plot_cadence_sweep(results['sweep_b'], outdir)

    print_summary(results['sweep_a'], results['sweep_b'])
    print(f"\nAll outputs in {outdir}/")


if __name__ == '__main__':
    main()
