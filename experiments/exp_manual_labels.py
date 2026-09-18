"""
exp_manual_labels.py — Assign manually written skill labels to MC-Diamond trajectory segments.

This generates the 'manual' annotator baseline for Q9: comparing controller performance
when trained with hand-crafted skill labels vs VLM-generated annotations.

The labeler uses two strategies (selectable via --strategy):

  keyword   Match instruction text from VLM annotations to canonical skill labels
            using keyword rules. Requires annotations_clean.jsonl to already exist.
            Use this to relabel the existing annotation corpus.

  gpt       Use GPT-5.5 to map each existing VLM instruction to the closest
            canonical skill label. Produces cleaner mappings than keyword matching.

Output: a JSONL file in the same format as annotations_clean.jsonl, with each
instruction replaced by its canonical manual skill label. Pass this to the
training pipeline via annotator_type='precomputed'.

Usage
─────
  # Keyword-based relabeling of the full annotations corpus
  python exp_manual_labels.py --strategy keyword \\
      --out results/annotation_judge/manual_labels_keyword.jsonl

  # GPT-based relabeling (50-sample for Q9 ablation)
  python exp_manual_labels.py --strategy gpt --sample 200 \\
      --out results/annotation_judge/manual_labels_gpt.jsonl

  # Coverage report: what fraction of instructions map to each canonical label
  python exp_manual_labels.py --report \\
      --labels results/annotation_judge/manual_labels_keyword.jsonl
"""

import argparse
import json
import random
import sys
import time
from collections import Counter
from pathlib import Path

import openai

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import load_openai_key

# ── Paths ─────────────────────────────────────────────────────────────────────

ANNOTATIONS_FILE = Path(__file__).parent.parent / (
    "results/instruction_taxonomy/annotations_clean.jsonl"
)
SKILL_LABELS_FILE = Path(__file__).parent.parent / (
    "results/annotation_judge/minecraft_skill_labels.json"
)

MODEL = "gpt-5.5"

# ── Load skill labels ─────────────────────────────────────────────────────────

def load_skill_labels():
    with open(SKILL_LABELS_FILE) as f:
        data = json.load(f)
    return data


def load_annotations(sample=None, seed=42):
    with open(ANNOTATIONS_FILE) as f:
        records = [json.loads(l) for l in f]
    if sample and sample < len(records):
        random.seed(seed)
        records = random.sample(records, sample)
    return records


# ── Strategy 1: keyword matching ──────────────────────────────────────────────

def build_keyword_rules(skill_data):
    """Build list of (keywords, canonical_label) pairs, longest keywords first."""
    rules = []
    for phase_name, phase in skill_data["phases"].items():
        canonical = phase["label"]
        keywords  = phase["keywords"]
        rules.append((keywords, canonical))
    return rules


def keyword_match(instruction: str, rules: list) -> str:
    """Return canonical label for instruction via keyword matching, or None."""
    text = instruction.lower()
    # Score each canonical label by how many of its keywords appear
    best_label = None
    best_score = 0
    for keywords, canonical in rules:
        score = sum(1 for kw in keywords if kw in text)
        if score > best_score:
            best_score = score
            best_label = canonical
    return best_label if best_score > 0 else None


def run_keyword(args):
    skill_data = load_skill_labels()
    rules = build_keyword_rules(skill_data)
    records = load_annotations(sample=args.sample)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    matched = 0
    unmatched_instrs = []
    results = []

    for rec in records:
        orig = rec["instruction"]
        canonical = keyword_match(orig, rules)
        if canonical is None:
            canonical = "Mine stone blocks to gather cobblestone."  # most-frequent fallback
            unmatched_instrs.append(orig)
        else:
            matched += 1
        results.append({**rec, "instruction": canonical, "original_instruction": orig})

    with open(out_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    n = len(results)
    print(f"Keyword matching: {matched}/{n} matched ({100*matched/n:.1f}%)")
    if unmatched_instrs:
        print(f"Unmatched (showing first 10):")
        for inst in unmatched_instrs[:10]:
            print(f"  {inst}")

    _print_distribution(results)
    print(f"\nLabels written to: {out_path}")


# ── Strategy 2: GPT-based mapping ─────────────────────────────────────────────

def build_gpt_prompt(canonical_labels: list) -> str:
    labels_str = "\n".join(f"  {i+1}. {l}" for i, l in enumerate(canonical_labels))
    return f"""\
You are mapping free-form Minecraft instruction strings to a fixed canonical skill vocabulary.

CANONICAL SKILLS (choose exactly one):
{labels_str}

Given an instruction, respond with JSON only:
{{"canonical": "<exact canonical skill string from the list above>", "confidence": <0.0-1.0>}}

Pick the skill that best describes the same action. If none fits well, pick the closest."""


def run_gpt(args):
    load_openai_key()
    client = openai.OpenAI()

    skill_data = load_skill_labels()
    canonical_labels = skill_data["canonical_labels"]
    system_prompt = build_gpt_prompt(canonical_labels)

    records = load_annotations(sample=args.sample)

    # Deduplicate: only call GPT once per unique instruction
    unique_instrs = list(set(r["instruction"] for r in records))
    print(f"Mapping {len(unique_instrs)} unique instructions → canonical labels via {MODEL}...")

    mapping = {}
    for i, instr in enumerate(unique_instrs):
        resp = client.chat.completions.create(
            model=MODEL,
            max_completion_tokens=64,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": f'Instruction: "{instr}"'},
            ],
        )
        raw = resp.choices[0].message.content.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1].lstrip("json").strip()
        try:
            parsed = json.loads(raw)
            canonical = parsed.get("canonical", canonical_labels[4])  # fallback: mine stone
        except json.JSONDecodeError:
            canonical = canonical_labels[4]
        mapping[instr] = canonical

        if (i + 1) % 10 == 0 or i == len(unique_instrs) - 1:
            print(f"  [{i+1}/{len(unique_instrs)}] \"{instr[:55]}\" → \"{canonical[:55]}\"")
        time.sleep(0.03)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    results = []
    for rec in records:
        orig = rec["instruction"]
        canonical = mapping.get(orig, canonical_labels[4])
        results.append({**rec, "instruction": canonical, "original_instruction": orig})

    with open(out_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    _print_distribution(results)
    print(f"\nLabels written to: {out_path}")


# ── Coverage report ───────────────────────────────────────────────────────────

def run_report(args):
    with open(args.labels) as f:
        records = [json.loads(l) for l in f]
    _print_distribution(records)


def _print_distribution(records):
    counts = Counter(r["instruction"] for r in records)
    n = len(records)
    print(f"\n── Canonical label distribution (n={n}) ──")
    for label, c in counts.most_common():
        bar = "█" * int(30 * c / n)
        print(f"  {c:6d} ({100*c/n:5.1f}%)  {bar}  {label}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--strategy", choices=["keyword", "gpt"], default="keyword",
                        help="Labeling strategy (default: keyword)")
    parser.add_argument("--sample", type=int, default=None,
                        help="Number of annotations to process (default: all)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="results/annotation_judge/manual_labels_keyword.jsonl",
                        help="Output JSONL path")
    parser.add_argument("--report", action="store_true",
                        help="Print distribution report for an existing labels file")
    parser.add_argument("--labels", default="results/annotation_judge/manual_labels_keyword.jsonl",
                        help="Labels file for --report")

    args = parser.parse_args()

    if args.report:
        run_report(args)
    elif args.strategy == "keyword":
        run_keyword(args)
    else:
        run_gpt(args)


if __name__ == "__main__":
    main()
