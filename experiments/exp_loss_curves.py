"""
exp_loss_curves.py — Per-component loss monitoring during training.

Addresses Reviewer qnYd Q3:
  "Throughout the training, how well does each [loss] go down (if at all)?
   How stable was the training, and was the overall loss dominated by any
   subset of the losses?"

This script parses the metrics.jsonl logs produced by train.py and generates:
  1. Per-component loss curves: L_model, L_value, L_actor, L_BC, L_stop
  2. KL divergence curve
  3. Reward prediction error
  4. Instruction-following accuracy over training
  5. Dominant-loss analysis: fraction of total loss per component over time

Usage
─────
  # After training:
  python experiments/exp_loss_curves.py \
      --logdirs logdir/minecraft/seed0 logdir/minecraft/seed1 logdir/minecraft/seed2 \
      --outdir results/loss_curves \
      --env minecraft

  # Or with --watch to live-plot during training:
  python experiments/exp_loss_curves.py --logdirs logdir/minecraft/seed0 --watch
"""

import argparse, json, time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


LOSS_KEYS = [
    ('wm/loss',     r'$\mathcal{L}_{\mathrm{model}}$',  '#1565C0'),
    ('ac/value_loss', r'$\mathcal{L}_{\mathrm{value}}$',  '#2E7D32'),
    ('ac/actor_loss', r'$\mathcal{L}_{\mathrm{actor}}$',  '#E65100'),
    ('bc_loss',     r'$\mathcal{L}_{\mathrm{BC}}$',      '#6A1B9A'),
    ('stop_loss',   r'$\mathcal{L}_{\mathrm{stop}}$',    '#AD1457'),
    ('wm/kl',       r'$\mathrm{KL}$',                    '#00838F'),
    ('wm/rew_loss', r'Reward loss',                       '#558B2F'),
]

SCALAR_KEYS = [
    ('bc_frac',          'BC annotated fraction'),
    ('stop_pred_mean',   'Mean p_stop'),
    ('episode/score',    'Task score'),
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--logdirs',  nargs='+', required=True,
                   help='Paths to training log directories (metrics.jsonl inside)')
    p.add_argument('--outdir',   default='results/loss_curves')
    p.add_argument('--env',      default='minecraft')
    p.add_argument('--smooth',   type=int, default=100,
                   help='Rolling-average window for smoothing (steps)')
    p.add_argument('--watch',    action='store_true',
                   help='Re-read and replot every 30s (live monitoring)')
    return p.parse_args()


def load_jsonl(path: Path) -> list:
    records = []
    try:
        with open(path) as f:
            for line in f:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    except FileNotFoundError:
        pass
    return records


def smooth(values, window: int):
    if window <= 1 or len(values) < window:
        return values
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode='valid')


def extract_series(records: list, key: str):
    """Extract (steps, values) for a given metric key."""
    steps, vals = [], []
    for r in records:
        if key in r and 'step' in r:
            steps.append(r['step'])
            vals.append(r[key])
    return np.array(steps), np.array(vals)


def plot_loss_curves(all_records: dict, outdir: Path, env: str, smooth_w: int):
    """Plot all loss components with mean ± std across seeds."""
    n_plots = len(LOSS_KEYS) + 2   # losses + dominant-loss pie + task score
    fig = plt.figure(figsize=(20, 3 * ((n_plots + 2) // 3)))
    gs  = gridspec.GridSpec((n_plots + 2) // 3, 3, figure=fig)
    axes = [fig.add_subplot(gs[i // 3, i % 3]) for i in range(n_plots)]

    seed_labels = list(all_records.keys())

    # ── Per-loss curves ───────────────────────────────────────────────────────
    final_fractions = {}
    for ax_idx, (key, label, color) in enumerate(LOSS_KEYS):
        ax = axes[ax_idx]
        seed_curves = []
        ref_steps   = None
        for seed_label, records in all_records.items():
            steps, vals = extract_series(records, key)
            if len(vals) == 0:
                continue
            s_vals = smooth(vals, smooth_w)
            s_steps = steps[len(steps) - len(s_vals):]
            ax.plot(s_steps, s_vals, alpha=0.4, color=color, lw=0.8)
            seed_curves.append((s_steps, s_vals))
            ref_steps = s_steps

        if seed_curves and ref_steps is not None:
            # Interpolate all seeds to common x-axis for mean/std
            common_steps = ref_steps
            interpolated = []
            for s, v in seed_curves:
                if len(s) > 1:
                    interp = np.interp(common_steps, s, v)
                    interpolated.append(interp)
            if interpolated:
                interp_arr = np.array(interpolated)
                ax.plot(common_steps, interp_arr.mean(0), color=color, lw=2, label='mean')
                ax.fill_between(common_steps,
                                interp_arr.mean(0) - interp_arr.std(0),
                                interp_arr.mean(0) + interp_arr.std(0),
                                alpha=0.15, color=color)
                if interp_arr.shape[1] > 0:
                    final_fractions[label] = abs(float(interp_arr.mean(0)[-1]))

        ax.set_title(label, fontsize=10)
        ax.set_xlabel('Training steps')
        ax.set_ylabel('Loss')
        ax.set_yscale('log' if 'kl' not in key.lower() else 'linear')
        ax.grid(True, alpha=0.3)

    # ── Dominant loss pie chart ───────────────────────────────────────────────
    if final_fractions:
        ax_pie = axes[len(LOSS_KEYS)]
        labels = list(final_fractions.keys())
        sizes  = [final_fractions[l] for l in labels]
        colors = [c for _, _, c in LOSS_KEYS if _ in
                  {key for key, lab, _ in LOSS_KEYS if lab in labels}]
        ax_pie.pie(sizes, labels=labels, autopct='%1.1f%%',
                   colors=[c for _, lab, c in LOSS_KEYS if lab in final_fractions])
        ax_pie.set_title('Loss contribution at end of training', fontsize=10)

    # ── Task score ───────────────────────────────────────────────────────────
    ax_score = axes[len(LOSS_KEYS) + 1]
    for seed_label, records in all_records.items():
        steps, vals = extract_series(records, 'episode/score')
        if len(vals) > 0:
            s_vals  = smooth(vals, smooth_w * 2)
            s_steps = steps[len(steps) - len(s_vals):]
            ax_score.plot(s_steps, s_vals, lw=1.5, label=seed_label)
    ax_score.set_title('Task score (eval)', fontsize=10)
    ax_score.set_xlabel('Training steps')
    ax_score.set_ylabel('Score')
    ax_score.legend(fontsize=8)
    ax_score.grid(True, alpha=0.3)

    plt.suptitle(f'Training loss curves — {env}', fontsize=13, fontweight='bold')
    plt.tight_layout()
    outdir.mkdir(parents=True, exist_ok=True)
    fig.savefig(outdir / f'loss_curves_{env}.pdf')
    fig.savefig(outdir / f'loss_curves_{env}.png', dpi=150)
    plt.close()
    print(f"Saved to {outdir}/loss_curves_{env}.{{pdf,png}}")


def print_dominance_report(all_records: dict):
    """Report which loss dominates at each training phase."""
    print("\n── Loss Dominance Analysis ──")
    phases = [('early', 0, 0.1), ('mid', 0.4, 0.5), ('late', 0.9, 1.0)]
    for phase_name, frac_lo, frac_hi in phases:
        print(f"\n  {phase_name.upper()} phase ({int(frac_lo*100)}–{int(frac_hi*100)}% of training):")
        for seed_label, records in list(all_records.items())[:1]:  # use first seed
            n = len(records)
            lo, hi = int(n * frac_lo), int(n * frac_hi)
            subset = records[lo:hi]
            vals = {}
            for key, label, _ in LOSS_KEYS:
                v = [r[key] for r in subset if key in r]
                if v:
                    vals[label] = abs(np.mean(v))
            if vals:
                total = sum(vals.values()) + 1e-8
                for label, v in sorted(vals.items(), key=lambda x: -x[1]):
                    print(f"    {label:<35} {v:8.4f}  ({v/total*100:.1f}%)")


def main():
    args   = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    while True:
        all_records = {}
        for logdir in args.logdirs:
            path    = Path(logdir) / 'metrics.jsonl'
            records = load_jsonl(path)
            label   = Path(logdir).name
            all_records[label] = records
            print(f"Loaded {len(records)} records from {path}")

        if not all(len(r) == 0 for r in all_records.values()):
            plot_loss_curves(all_records, outdir, args.env, args.smooth)
            print_dominance_report(all_records)
        else:
            print("No records found yet.")

        if not args.watch:
            break
        print(f"\nWatching… (refresh in 30s, Ctrl-C to stop)")
        time.sleep(30)


if __name__ == '__main__':
    main()
