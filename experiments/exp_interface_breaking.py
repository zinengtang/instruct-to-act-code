"""
exp_interface_breaking.py — Where does instruction length break the interface?

Question (Reviewer EkK5 Q5 / of2B Q13):
  "Are there tasks where very short or very long instructions break the interface?"

Design
──────
Six length conditions tested per task:

  train_dist  — instructions sampled verbatim from the training annotation log
                (the ground-truth reference; what the controller was conditioned on)
  tiny        — truncated to 3–5 words
  short       — truncated/padded to 6–9 words   (near training mean)
  medium      — truncated/padded to 10–15 words
  long        — truncated/padded to 16–25 words
  very_long   — truncated/padded to 26–40 words

The train_dist condition is the key baseline: it uses real GPT-4o annotations
from the training run, cycling through them in shuffled order. All other
conditions take those same instructions and apply word-count surgery, so the
only variable is length (not semantic content).

Metrics per condition per task:
  - Mean ± std episode reward
  - p_stop firing rate (mean p_stop per step → shows stop-head confusion)
  - Mean instruction interval (steps between re-instructions)
  - p_stop histogram (saved as JSON for offline plotting)

Tasks: Minecraft Diamond (single-agent, long episodes) and
       Overcooked (multi-agent, short episodes) — the two ends of the
       task-horizon spectrum.

Usage
─────
  # Quick smoke-test (mock, no GPU):
  python experiments/exp_interface_breaking.py --mock --episodes 5 --max_steps 200

  # Minecraft only, seed3 checkpoint:
  python experiments/exp_interface_breaking.py \\
      --task minecraft \\
      --checkpoint /data/terran/instruct_to_act/mc200m_seed3 \\
      --annotation_log /data/terran/instruct_to_act/mc200m_seed3/annotations/annotations_minecraft.jsonl \\
      --episodes 20 --max_steps 1000 \\
      --outdir results/interface_breaking

  # Both tasks:
  python experiments/exp_interface_breaking.py \\
      --task both \\
      --checkpoint /data/terran/instruct_to_act/mc200m_seed3 \\
      --overcooked_checkpoint /data/terran/instruct_to_act/overcooked/seed0 \\
      --annotation_log /data/terran/instruct_to_act/mc200m_seed3/annotations/annotations_minecraft.jsonl \\
      --overcooked_annotation_log /data/terran/instruct_to_act/overcooked/seed0/annotations/annotations_overcooked.jsonl \\
      --episodes 20 --max_steps 1000 \\
      --outdir results/interface_breaking
"""

import argparse
import datetime
import json
import random
import sys
import threading
import time
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# ─────────────────────────────────────────────────────────────────────────────
# Length conditions
# ─────────────────────────────────────────────────────────────────────────────

LENGTH_CONDITIONS = {
    'train_dist': (None, None),   # verbatim from annotation log — no surgery
    'ultra_short': (2,  3),       # below training min (mean=5, min=2)
    'in_dist':     (4,  7),       # matches training bulk (p25–p90 = 4–7w)
    'slightly_ood': (8, 10),      # just above training p99 (8w)
    'long':        (11, 16),      # above training max (11w)
    'very_long':   (17, 30),      # strongly OOD
}

CONDITION_ORDER = ['train_dist', 'ultra_short', 'in_dist', 'slightly_ood',
                   'long', 'very_long']

# Actual training distribution (mc200m_seed3, n=99748):
#   mean=5.45w, std=1.17, p25=4, p50=5, p75=6, p90=7, p99=8, max=11
# NOTE: annotator uses gpt-4o-mini (not gpt-4o as claimed in paper) →
#       much terser output than the "6-18 token" prompt guideline.
# Expected breaking point: ~8-9w (slightly_ood) upward, not 16+.


# ─────────────────────────────────────────────────────────────────────────────
# Training-distribution planner
# ─────────────────────────────────────────────────────────────────────────────

class TrainingDistPlanner:
    """
    Cycles through real annotations from the training run (the annotation log
    written by PostHocAnnotator during controller training).

    This is the ground-truth reference condition: the exact instruction strings
    and distribution the controller was conditioned on.  All other conditions
    take the same strings and apply word-count surgery.
    """

    def __init__(self, annotation_log: str, seed: int = 0):
        self._instructions = self._load(annotation_log)
        rng = random.Random(seed)
        rng.shuffle(self._instructions)
        self._idx = 0
        print(f"[TrainingDistPlanner] Loaded {len(self._instructions)} annotations "
              f"from {annotation_log}")

    @staticmethod
    def _load(path: str) -> List[str]:
        instrs = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    txt = obj.get('instruction', '').strip()
                    if txt:
                        instrs.append(txt)
                except json.JSONDecodeError:
                    continue
        return instrs

    def emit(self, obs=None, plan=None, memory=None, **kwargs):
        instr = self._instructions[self._idx % len(self._instructions)]
        self._idx += 1
        return instr, plan, memory

    def step(self, obs=None, plan=None, memory=None, p_stop=0.0, **kwargs):
        if p_stop >= 0.5 or not plan:
            return self.emit(obs, plan, memory)[0], plan, memory
        return None, plan, memory

    def advance(self, obs=None, plan=None, memory=None):
        return plan, memory


# ─────────────────────────────────────────────────────────────────────────────
# Length-constrained wrapper
# ─────────────────────────────────────────────────────────────────────────────

def _apply_length(text: str, min_w: Optional[int], max_w: Optional[int]) -> str:
    """Truncate or pad to [min_w, max_w] words.  None = no change."""
    if min_w is None:
        return text
    words = text.split()
    if len(words) > max_w:
        words = words[:max_w]
    while len(words) < min_w:
        # Repeat last word rather than injecting unrelated filler
        words.append(words[-1] if words else 'proceed')
    return ' '.join(words)


class LengthConstrainedPlanner:
    """Wraps TrainingDistPlanner and enforces a word-count budget."""

    def __init__(self, base: TrainingDistPlanner, condition: str):
        self.base      = base
        self.condition = condition
        self.min_w, self.max_w = LENGTH_CONDITIONS[condition]
        self.issued_lengths: List[int] = []

    def _constrain(self, text: str) -> str:
        text = _apply_length(text, self.min_w, self.max_w)
        self.issued_lengths.append(len(text.split()))
        return text

    def emit(self, obs=None, plan=None, memory=None, **kwargs):
        instr, plan, memory = self.base.emit(obs, plan, memory)
        return self._constrain(instr), plan, memory

    def step(self, obs=None, plan=None, memory=None, p_stop=0.0, **kwargs):
        instr, plan, memory = self.base.step(obs, plan, memory, p_stop)
        if instr:
            instr = self._constrain(instr)
        return instr, plan, memory

    def advance(self, obs=None, plan=None, memory=None):
        return self.base.advance(obs, plan, memory)


# ─────────────────────────────────────────────────────────────────────────────
# Episode runner
# ─────────────────────────────────────────────────────────────────────────────

def run_episode(ctrl, planner, env, encoder, stop_threshold: float,
                max_steps: int) -> dict:
    """
    Run one episode; return reward, instruction count, and p_stop distribution.

    Returns
    -------
    dict with keys: reward, n_instrs, p_stop_values, instr_intervals
    """
    ctrl_state = ctrl.init_policy(batch_size=1)
    lang_size  = getattr(ctrl, '_lang_size', 384)
    null_embed = np.zeros(lang_size, dtype=np.float32)
    cur_embed  = null_embed.copy()

    obs, _     = env.reset()
    plan       = None
    memory     = {}
    stop_flag  = True
    total_reward  = 0.0
    n_instrs      = 0
    p_stop_vals: List[float] = []
    last_instr_step = 0
    instr_intervals: List[int] = []

    for step in range(max_steps):
        if stop_flag:
            instr, plan, memory = planner.step(
                obs=obs, plan=plan, memory=memory, p_stop=float(stop_flag)
            )
            if instr:
                cur_embed = encoder(instr)
                if n_instrs > 0:
                    instr_intervals.append(step - last_instr_step)
                last_instr_step = step
                stop_flag = False
                n_instrs += 1

        obs_b = {k: np.expand_dims(v, 0) for k, v in obs.items()
                 if not k.startswith('log/')}
        obs_b['lang_embed'] = cur_embed[np.newaxis]

        ctrl_state, action, extra = ctrl.policy(ctrl_state, obs_b, mode='eval')
        p_stop = float(extra.get('log/p_stop', extra.get('p_stop', 0.0)))
        p_stop_vals.append(p_stop)

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

    return {
        'reward':          total_reward,
        'n_instrs':        n_instrs,
        'p_stop_values':   p_stop_vals,
        'instr_intervals': instr_intervals,
    }


def _stats(vals):
    a = np.array(vals, dtype=float)
    if len(a) == 0:
        return {'mean': 0.0, 'std': 0.0, 'min': 0.0, 'max': 0.0, 'n': 0}
    return {'mean': float(np.mean(a)), 'std': float(np.std(a)),
            'min':  float(np.min(a)),  'max': float(np.max(a)), 'n': len(a)}


# ─────────────────────────────────────────────────────────────────────────────
# Progress logger
# ─────────────────────────────────────────────────────────────────────────────

class ProgressLogger:
    def __init__(self, outdir: Path, total: int):
        self._path  = outdir / 'progress.jsonl'
        self._total = total
        self._done  = 0
        self._t0    = time.perf_counter()
        outdir.mkdir(parents=True, exist_ok=True)

    def log(self, **kw):
        self._done += 1
        elapsed = time.perf_counter() - self._t0
        avg     = elapsed / self._done
        eta     = str(datetime.timedelta(seconds=int(avg * (self._total - self._done))))
        now     = datetime.datetime.now().strftime('%H:%M:%S')
        entry   = dict(ts=now, elapsed=round(elapsed, 1), eta=eta,
                       done=self._done, total=self._total,
                       pct=round(100 * self._done / self._total, 1), **kw)
        with open(self._path, 'a') as f:
            f.write(json.dumps(entry) + '\n')
        task = kw.get('task', '?')
        cond = kw.get('condition', '?')
        ep   = kw.get('ep', self._done)
        r    = kw.get('reward', 0)
        ps   = kw.get('mean_p_stop', 0)
        print(f"  {now}  [{task}|{cond}]  ep {ep:3d}/{self._total}  "
              f"reward={r:.4f}  p_stop_mean={ps:.3f}  "
              f"elapsed={elapsed/60:.1f}m  ETA={eta}")


# ─────────────────────────────────────────────────────────────────────────────
# Mock components
# ─────────────────────────────────────────────────────────────────────────────

class MockController:
    _lang_size = 384

    def init_policy(self, batch_size=1):
        return None

    def policy(self, state, obs_batch, mode='eval'):
        time.sleep(0.001)
        p_stop = float(np.random.random() < 0.05)
        return state, {'action': np.zeros((1,), dtype=np.int32)}, {'log/p_stop': p_stop}


class MockEnv:
    def reset(self):
        return {'image': np.zeros((64, 64, 3), np.uint8)}, {}

    def step(self, action):
        r = float(np.random.random() < 0.01)
        done = np.random.random() < 0.005
        return {'image': np.zeros((64, 64, 3), np.uint8)}, r, done, False, {}


def mock_encoder(text):
    return np.zeros(384, dtype=np.float32)


MOCK_ANNOTATIONS = [
    "Chop trees to gather wood.",
    "Craft wooden planks from collected logs.",
    "Build a crafting table using four planks.",
    "Craft sticks from two wooden planks.",
    "Craft a wooden pickaxe using sticks and planks.",
    "Mine stone blocks to gather cobblestone.",
    "Craft a stone pickaxe from cobblestone and sticks.",
    "Search underground for iron ore deposits.",
    "Mine iron ore with the stone pickaxe carefully.",
    "Smelt iron ore in a furnace to get ingots.",
    "Craft an iron pickaxe from iron ingots.",
    "Dig deep to find diamond ore below level 16.",
    "Mine diamond ore with the iron pickaxe now.",
    "Return to surface with the diamonds collected.",
]


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint / env helpers
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
# Plotting
# ─────────────────────────────────────────────────────────────────────────────

COLORS = {
    'train_dist': '#1565C0',   # blue  — reference
    'tiny':       '#E53935',   # red   — expected bad
    'short':      '#43A047',   # green
    'medium':     '#00897B',   # teal
    'long':       '#FB8C00',   # orange
    'very_long':  '#8E24AA',   # purple — expected bad
}


def plot_reward_bars(results: dict, task: str, outdir: Path):
    """Bar chart of mean ± std reward per condition. train_dist shown as reference line."""
    conditions = [c for c in CONDITION_ORDER if c in results]
    means = [results[c]['reward']['mean'] for c in conditions]
    stds  = [results[c]['reward']['std']  for c in conditions]
    colors = [COLORS[c] for c in conditions]

    fig, ax = plt.subplots(figsize=(10, 5))
    x    = np.arange(len(conditions))
    bars = ax.bar(x, means, yerr=stds, color=colors, alpha=0.82,
                  edgecolor='white', capsize=4, width=0.65)

    # Highlight train_dist as reference
    if 'train_dist' in conditions:
        ref_mean = results['train_dist']['reward']['mean']
        ref_std  = results['train_dist']['reward']['std']
        ax.axhline(ref_mean, color='#1565C0', lw=1.5, ls='--', alpha=0.6,
                   label=f'train_dist mean ({ref_mean:.3f})')
        ax.fill_between([-0.5, len(conditions) - 0.5],
                        ref_mean - ref_std, ref_mean + ref_std,
                        color='#1565C0', alpha=0.08)

    ax.set_xticks(x)
    ax.set_xticklabels(conditions, fontsize=11)
    ax.set_ylabel('Mean episode reward', fontsize=11)
    ax.set_title(f'{task.capitalize()} — Instruction length vs. reward\n'
                 f'(train_dist = real training annotations; others = length-surgery on same strings)',
                 fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.25, axis='y')

    for bar, m, s in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width() / 2, m + s + 1e-4,
                f'{m:.3f}', ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    fname = outdir / f'{task}_reward_by_length.pdf'
    fig.savefig(fname, bbox_inches='tight')
    fig.savefig(fname.with_suffix('.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved {fname}")


def plot_p_stop_distribution(results: dict, task: str, outdir: Path):
    """
    p_stop histogram per condition.
    A well-calibrated stop head should peak near 0 (most steps) and near 1
    (instruction completion).  Breaking conditions show a flat or mid-range
    distribution → the head can't tell when the instruction is done.
    """
    conditions = [c for c in CONDITION_ORDER if c in results]
    n = len(conditions)
    fig, axes = plt.subplots(1, n, figsize=(3 * n, 4), sharey=True)
    if n == 1:
        axes = [axes]

    for ax, cond in zip(axes, conditions):
        vals = results[cond].get('all_p_stop', [])
        if vals:
            ax.hist(vals, bins=20, range=(0, 1), color=COLORS[cond],
                    alpha=0.8, edgecolor='white', density=True)
        ax.set_title(cond, fontsize=9)
        ax.set_xlabel('p_stop', fontsize=8)
        ax.set_xlim(0, 1)
        ax.grid(True, alpha=0.2)

    axes[0].set_ylabel('Density', fontsize=9)
    fig.suptitle(f'{task.capitalize()} — p_stop distribution by instruction length\n'
                 f'(bimodal ≈ good; flat or mid-range ≈ stop-head confused)',
                 fontsize=9)
    plt.tight_layout()
    fname = outdir / f'{task}_pstop_hist.pdf'
    fig.savefig(fname, bbox_inches='tight')
    fig.savefig(fname.with_suffix('.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved {fname}")


def plot_combined(mc_results: dict, oc_results: dict, outdir: Path):
    """
    Side-by-side reward comparison: Minecraft vs Overcooked.
    Shows that breaking points differ between tasks.
    """
    conditions = [c for c in CONDITION_ORDER
                  if c in mc_results or c in oc_results]
    x = np.arange(len(conditions))
    width = 0.35

    mc_means = [mc_results.get(c, {}).get('reward', {}).get('mean', 0) for c in conditions]
    mc_stds  = [mc_results.get(c, {}).get('reward', {}).get('std',  0) for c in conditions]
    oc_means = [oc_results.get(c, {}).get('reward', {}).get('mean', 0) for c in conditions]
    oc_stds  = [oc_results.get(c, {}).get('reward', {}).get('std',  0) for c in conditions]

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(x - width / 2, mc_means, width, yerr=mc_stds,
           label='Minecraft', color='#1565C0', alpha=0.82, capsize=3)
    ax.bar(x + width / 2, oc_means, width, yerr=oc_stds,
           label='Overcooked', color='#2E7D32', alpha=0.82, capsize=3)

    ax.set_xticks(x)
    ax.set_xticklabels(conditions, fontsize=11)
    ax.set_ylabel('Mean reward (normalized per task)')
    ax.set_title('Where does instruction length break the interface?\n'
                 'Minecraft (long horizon) vs. Overcooked (short horizon)',
                 fontsize=10)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.25, axis='y')
    plt.tight_layout()
    fname = outdir / 'combined_reward_by_length.pdf'
    fig.savefig(fname, bbox_inches='tight')
    fig.savefig(fname.with_suffix('.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved {fname}")


# ─────────────────────────────────────────────────────────────────────────────
# Per-task runner
# ─────────────────────────────────────────────────────────────────────────────

def run_task(task: str, ctrl, env, encoder, annotation_log: str,
             episodes: int, max_steps: int, stop_threshold: float,
             plog: ProgressLogger, outdir: Path, seed: int = 0) -> dict:
    """
    Run all six length conditions for one task.
    Returns dict: condition → aggregated metrics.
    """
    # Build the shared base planner from the training annotation log
    if annotation_log:
        base = TrainingDistPlanner(annotation_log, seed=seed)
    else:
        # Mock: use hardcoded strings
        base = TrainingDistPlanner.__new__(TrainingDistPlanner)
        base._instructions = MOCK_ANNOTATIONS * 100
        base._idx = 0
        random.shuffle(base._instructions)
        print(f"[TrainingDistPlanner] Mock mode — {len(base._instructions)} entries")

    task_results = {}

    for cond in CONDITION_ORDER:
        planner = LengthConstrainedPlanner(base, cond)
        # Reset annotation index so every condition sees the same instruction
        # sequence (only length differs, not content)
        base._idx = 0

        ep_rewards, ep_n_instrs, all_p_stop, all_intervals = [], [], [], []

        for ep in range(episodes):
            m = run_episode(ctrl, planner, env, encoder, stop_threshold, max_steps)
            ep_rewards.append(m['reward'])
            ep_n_instrs.append(m['n_instrs'])
            all_p_stop.extend(m['p_stop_values'])
            all_intervals.extend(m['instr_intervals'])
            plog.log(
                task=task, condition=cond, ep=ep + 1,
                reward=m['reward'],
                n_instrs=m['n_instrs'],
                mean_p_stop=float(np.mean(m['p_stop_values'])) if m['p_stop_values'] else 0.0,
                mean_instr_len=round(float(np.mean(planner.issued_lengths or [0])), 1),
            )

        task_results[cond] = {
            'reward':          _stats(ep_rewards),
            'n_instrs':        _stats(ep_n_instrs),
            'mean_instr_len':  float(np.mean(planner.issued_lengths or [0])),
            'std_instr_len':   float(np.std(planner.issued_lengths  or [0])),
            'mean_p_stop':     float(np.mean(all_p_stop)) if all_p_stop else 0.0,
            'mean_interval':   float(np.mean(all_intervals)) if all_intervals else 0.0,
            'all_p_stop':      all_p_stop,   # kept for histogram
        }

        # Intermediate save after each condition
        with open(outdir / f'{task}_partial.json', 'w') as f:
            # all_p_stop is large; exclude from intermediate save
            slim = {c: {k: v for k, v in d.items() if k != 'all_p_stop'}
                    for c, d in task_results.items()}
            json.dump(slim, f, indent=2)

    return task_results


# ─────────────────────────────────────────────────────────────────────────────
# Summary table
# ─────────────────────────────────────────────────────────────────────────────

def print_summary(results: dict, task: str):
    W = 90
    print(f"\n{'─'*W}")
    print(f"  {task.upper()} — Instruction Length Breaking-Point Summary")
    print(f"{'─'*W}")
    print(f"  {'Condition':<14}  {'Reward↑':>9}  {'±std':>6}  "
          f"{'p_stop_mean':>12}  {'Instr_len':>10}  {'Interval':>9}")
    print(f"  {'─'*(W-2)}")
    ref = results.get('train_dist', {}).get('reward', {}).get('mean', 0)
    for cond in CONDITION_ORDER:
        if cond not in results:
            continue
        d  = results[cond]
        r  = d['reward']
        dr = r['mean'] - ref
        delta = f"({dr:+.3f})" if cond != 'train_dist' else "(ref)"
        print(f"  {cond:<14}  {r['mean']:>9.4f}  {r['std']:>6.4f}  "
              f"{d['mean_p_stop']:>12.4f}  {d['mean_instr_len']:>10.1f}  "
              f"{d['mean_interval']:>9.1f}  {delta}")
    print(f"{'─'*W}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description='Interface-breaking experiment: instruction length vs. task'
    )
    p.add_argument('--task',         default='minecraft',
                   choices=['minecraft', 'overcooked', 'both'])
    p.add_argument('--checkpoint',   default=str(
        Path('/data/terran/instruct_to_act/mc200m_seed3')),
        help='Logdir for Minecraft checkpoint (contains ckpt/)')
    p.add_argument('--overcooked_checkpoint', default=str(
        Path('/data/terran/instruct_to_act/overcooked/seed0')))
    p.add_argument('--annotation_log', default=str(
        Path('/data/terran/instruct_to_act/mc200m_seed3/annotations/annotations_minecraft.jsonl')))
    p.add_argument('--overcooked_annotation_log', default=str(
        Path('/data/terran/instruct_to_act/overcooked/seed0/annotations/annotations_overcooked.jsonl')))
    p.add_argument('--episodes',     type=int, default=20)
    p.add_argument('--max_steps',    type=int, default=1000)
    p.add_argument('--stop_threshold', type=float, default=0.5)
    p.add_argument('--outdir',       default='results/interface_breaking')
    p.add_argument('--mock',         action='store_true')
    p.add_argument('--seed',         type=int, default=0)
    return p.parse_args()


def main():
    args   = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    np.random.seed(args.seed)

    tasks_to_run = (['minecraft', 'overcooked'] if args.task == 'both'
                    else [args.task])
    n_conditions = len(CONDITION_ORDER)
    total_eps    = len(tasks_to_run) * n_conditions * args.episodes
    plog = ProgressLogger(outdir, total_eps)
    print(f"Total episodes: {total_eps}  "
          f"({len(tasks_to_run)} tasks × {n_conditions} conditions × {args.episodes} eps)")
    print(f"Monitor: tail -f {outdir}/progress.jsonl\n")

    all_results = {}

    for task in tasks_to_run:
        print(f"\n{'═'*60}")
        print(f"  Task: {task.upper()}")
        print(f"{'═'*60}")

        if args.mock:
            ctrl     = MockController()
            env      = MockEnv()
            encoder  = mock_encoder
            ann_log  = None
        else:
            root = Path(__file__).parent.parent
            sys.path.insert(0, str(root))
            sys.path.insert(0, str(root / 'dreamerv3'))
            from envs  import make_env
            from train import load_project_config
            from evaluate import load_agent, GymAdapter, encode_fn

            if task == 'minecraft':
                ckpt_dir  = Path(args.checkpoint)
                ann_log   = args.annotation_log
                cfg_paths = [str(root / 'configs' / 'base.yaml'),
                             str(root / 'configs' / 'minecraft.yaml')]
                if '200m' in ckpt_dir.name:
                    cfg_paths.append(str(root / 'configs' / 'size200m.yaml'))
                    print("── Applying size200m.yaml ──")
            else:
                ckpt_dir  = Path(args.overcooked_checkpoint)
                ann_log   = args.overcooked_annotation_log
                cfg_paths = [str(root / 'configs' / 'base.yaml'),
                             str(root / 'configs' / 'overcooked.yaml')]

            ckpt_path = resolve_checkpoint(ckpt_dir)
            print(f"── Checkpoint: {ckpt_path} ──")
            proj     = load_project_config(cfg_paths)
            lang_sz  = int(proj.get('lang_size', 384))
            encoder  = encode_fn(lang_sz)
            env      = GymAdapter(make_env(task, proj, lang_size=lang_sz))
            ctrl     = load_agent(str(ckpt_path), env.obs_space, env.act_space, proj)

        task_results = run_task(
            task, ctrl, env, encoder, ann_log,
            episodes=args.episodes,
            max_steps=args.max_steps,
            stop_threshold=args.stop_threshold,
            plog=plog,
            outdir=outdir,
            seed=args.seed,
        )
        all_results[task] = task_results

        # Save full results (p_stop histograms included)
        out_path = outdir / f'{task}_results.json'
        with open(out_path, 'w') as f:
            # Serialize; all_p_stop can be large
            serializable = {}
            for cond, d in task_results.items():
                serializable[cond] = {k: v for k, v in d.items()
                                      if k != 'all_p_stop'}
                serializable[cond]['p_stop_histogram'] = list(
                    np.histogram(d['all_p_stop'], bins=20, range=(0, 1))[0].astype(int)
                ) if d['all_p_stop'] else []
            json.dump(serializable, f, indent=2)
        print(f"\nSaved {out_path}")

        print_summary(task_results, task)
        plot_reward_bars(task_results, task, outdir)
        plot_p_stop_distribution(task_results, task, outdir)

    if 'minecraft' in all_results and 'overcooked' in all_results:
        plot_combined(all_results['minecraft'], all_results['overcooked'], outdir)

    print(f"\nAll outputs in {outdir}/")


if __name__ == '__main__':
    main()
