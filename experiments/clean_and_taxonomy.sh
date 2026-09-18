#!/usr/bin/env bash
# Clean annotation logs and re-run the Q6 instruction taxonomy experiment.
# Fixes two problems with the previous run:
#   1. Raw JSON-fenced strings stored as instructions — we extract the inner text
#   2. GPT-4o refusals stored as instructions — we drop them
#   3. No reward signal — we pull per-episode rewards from eval results
#
# Usage: bash experiments/clean_and_taxonomy.sh

set -euo pipefail

PYTHON="/scratch/users/terran/conda/envs/embodied/bin/python"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$ROOT"

# ── Step 1: clean annotation logs ────────────────────────────────────────────
echo "==> Cleaning annotation logs..."
"$PYTHON" - << 'PYEOF'
import json, re, sys
from pathlib import Path

REFUSAL_PREFIXES = (
    "i'm sorry", "i cannot", "i can't", "as an ai",
    "i apologize", "it seems there is an error",
    "it seems there", "i'm unable",
)

def extract_instruction(raw: str) -> str | None:
    """Parse instruction from raw annotator output."""
    raw = raw.strip()
    # Strip markdown fences
    if '```' in raw:
        parts = raw.split('```')
        raw = parts[1] if len(parts) > 1 else raw
        if raw.startswith('json'):
            raw = raw[4:].strip()
    # Try JSON parse
    try:
        obj = json.loads(raw)
        if isinstance(obj, dict) and 'instruction' in obj:
            return obj['instruction'].strip() or None
    except (json.JSONDecodeError, ValueError):
        pass
    # Best-effort: look for {"instruction": ...} fragment
    m = re.search(r'"instruction"\s*:\s*"([^"]+)"', raw)
    if m:
        return m.group(1).strip() or None
    # If raw text (no JSON), use as-is if it looks like a real instruction
    if not raw.startswith('{') and not raw.startswith('```'):
        return raw.strip() or None
    return None

logs = [
    "/data/terran/instruct_to_act/mc200m_seed1/annotations/annotations_minecraft.jsonl",
    "/data/terran/instruct_to_act/mc200m_seed2/annotations/annotations_minecraft.jsonl",
    "/data/terran/instruct_to_act/mc200m_seed3/annotations/annotations_minecraft.jsonl",
]

out_path = Path("results/instruction_taxonomy/annotations_clean.jsonl")
out_path.parent.mkdir(parents=True, exist_ok=True)

total, kept, dropped_refusal, dropped_parse = 0, 0, 0, 0
with open(out_path, 'w') as fout:
    for log in logs:
        p = Path(log)
        if not p.exists():
            print(f"  skipping (not found): {log}")
            continue
        with open(p) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                total += 1
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    dropped_parse += 1
                    continue
                raw = rec.get('instruction', '')
                instr = extract_instruction(raw)
                if instr is None:
                    dropped_parse += 1
                    continue
                if any(instr.lower().startswith(p) for p in REFUSAL_PREFIXES):
                    dropped_refusal += 1
                    continue
                rec['instruction'] = instr
                fout.write(json.dumps(rec) + '\n')
                kept += 1

print(f"  Total: {total}  Kept: {kept}  Dropped refusals: {dropped_refusal}  "
      f"Dropped parse failures: {dropped_parse}")
print(f"  Wrote {out_path}")
PYEOF

# ── Step 2: merge eval instruction logs (for reward correlation) ──────────────
echo "==> Merging eval instruction-reward pairs..."
"$PYTHON" - << 'PYEOF'
import json, glob
from pathlib import Path

# Collect per-instruction reward pairs from eval episode logs
eval_files = glob.glob("results/eval_seed1/worker*/minecraft_gpt4o_online_*_episodes.json")
eval_files += glob.glob("results/eval_inspect/minecraft_gpt4o_online_*_episodes.json")

out_path = Path("results/instruction_taxonomy/annotations_clean.jsonl")
eval_out  = Path("results/instruction_taxonomy/eval_instruction_rewards.jsonl")

pairs = []
for f in eval_files:
    d = json.load(open(f))
    episodes = d.get('episodes', [])
    # episodes is an int count in current format, not a list of episode dicts
    if not isinstance(episodes, list):
        continue
    for ep in episodes:
        reward = ep.get('reward', 0.0)
        for entry in ep.get('instruction_log', []):
            instr = entry.get('instruction', '').strip()
            if instr:
                pairs.append({'instruction': instr, 'reward': reward,
                              'p_stop_mean': ep.get('p_stop_mean')})

with open(eval_out, 'w') as f:
    for p in pairs:
        f.write(json.dumps(p) + '\n')

print(f"  Wrote {len(pairs)} instruction-reward pairs → {eval_out}")

# Append eval pairs to clean annotation log (no reward in original annotations)
with open(out_path, 'a') as f:
    for p in pairs:
        f.write(json.dumps(p) + '\n')

print(f"  Appended to {out_path}")
PYEOF

# ── Step 3: run taxonomy ──────────────────────────────────────────────────────
echo "==> Running taxonomy experiment..."
"$PYTHON" experiments/exp_instruction_taxonomy.py \
    --annotation_logs results/instruction_taxonomy/annotations_clean.jsonl \
    --env minecraft \
    --n_clusters 8 \
    --outdir results/instruction_taxonomy

echo "==> Done. Results in results/instruction_taxonomy/"
ENDOFSCRIPT
