"""
Plot Reviewer 2B11 Q7 instruction-robustness results.

Input is the JSON written by experiments/exp_instruction_robustness.py:
  results/instruction_robustness/robustness_minecraft_gpt4o.json
"""

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ORDER = [
    "original",
    "paraphrase",
    "contradictory",
    "underspecified",
    "adversarial",
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Robustness JSON file.")
    parser.add_argument("--outdir", default=None, help="Output directory.")
    parser.add_argument("--title", default="Instruction Robustness")
    return parser.parse_args()


def main():
    args = parse_args()
    input_path = Path(args.input)
    outdir = Path(args.outdir) if args.outdir else input_path.parent
    outdir.mkdir(parents=True, exist_ok=True)

    with open(input_path) as f:
        results = json.load(f)

    conditions = [c for c in ORDER if c in results]
    conditions += [c for c in results if c not in conditions]

    means = np.array([results[c]["mean_reward"] for c in conditions], dtype=float)
    stds = np.array([results[c]["std_reward"] for c in conditions], dtype=float)
    follow = np.array([results[c]["follow_accuracy"] for c in conditions], dtype=float)

    csv_path = outdir / "q7_instruction_robustness_table.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["condition", "mean_reward", "std_reward", "follow_accuracy"])
        for cond in conditions:
            writer.writerow(
                [
                    cond,
                    f"{results[cond]['mean_reward']:.6f}",
                    f"{results[cond]['std_reward']:.6f}",
                    f"{results[cond]['follow_accuracy']:.6f}",
                ]
            )

    colors = ["#4c78a8", "#72b7b2", "#f58518", "#eeca3b", "#e45756"]
    colors = colors[: len(conditions)]

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))

    axes[0].bar(conditions, means, yerr=stds, color=colors, edgecolor="white", capsize=4)
    axes[0].set_ylabel("Episode reward")
    axes[0].set_title("Task Performance")
    axes[0].spines[["top", "right"]].set_visible(False)
    axes[0].tick_params(axis="x", rotation=25)

    axes[1].bar(conditions, follow * 100.0, color=colors, edgecolor="white")
    axes[1].set_ylim(0, 100)
    axes[1].set_ylabel("Instruction-following accuracy (%)")
    axes[1].set_title("Language Grounding Proxy")
    axes[1].spines[["top", "right"]].set_visible(False)
    axes[1].tick_params(axis="x", rotation=25)

    fig.suptitle(args.title)
    fig.tight_layout()
    fig.savefig(outdir / "q7_instruction_robustness.png", dpi=220)
    fig.savefig(outdir / "q7_instruction_robustness.pdf")
    plt.close(fig)

    print(f"Wrote {csv_path}")
    print(f"Wrote {outdir / 'q7_instruction_robustness.png'}")
    print(f"Wrote {outdir / 'q7_instruction_robustness.pdf'}")


if __name__ == "__main__":
    main()
