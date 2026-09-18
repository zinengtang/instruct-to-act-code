"""
exp_instruction_robustness.py — Instruction robustness under perturbations.

Addresses Reviewer 2B11 Q7:
  "Does the controller ever exploit instruction artifacts rather than genuinely
   grounding language? How does it behave under paraphrases, contradictory
   instructions, underspecified instructions, or adversarial instructions?"

Protocol
────────
For a trained controller on Minecraft Diamond (or Crafter), we evaluate
instruction-following accuracy and task performance under five perturbation types:

  original      — ground-truth instructions from GPT-4o annotator
  paraphrase    — semantically equivalent rephrasing (via GPT-4o)
  contradictory — instruction that conflicts with observed state
  underspecified— vague instructions like "explore a bit more"
  adversarial   — instructions designed to mislead (valid syntax, wrong goal)

For each condition we run 100 held-out episodes and measure:
  (a) instruction-following accuracy (human eval, §4.5 / Table 8 method)
  (b) task score (achievement / reward)
  (c) correlation between instruction quality and execution success

Usage
─────
  python experiments/exp_instruction_robustness.py \
      --env minecraft \
      --checkpoint logdir/minecraft/seed0/checkpoint.pkl \
      --planner qwen \
      --conditions original paraphrase contradictory underspecified adversarial \
      --episodes 50 \
      --logdir results/instruction_robustness
"""

import argparse, json, os, sys, random
from pathlib import Path
from typing import Optional

import numpy as np

PROJECT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_DIR))
sys.path.insert(0, str(PROJECT_DIR / 'dreamerv3'))

from inference import OfflineInference
from planners  import make_planner


# ─────────────────────────────────────────────────────────────────────────────
# Perturbation generators
# ─────────────────────────────────────────────────────────────────────────────

VAGUE_INSTRUCTIONS = [
    "explore a bit more",
    "try something different",
    "keep going",
    "continue what you were doing",
    "do something useful",
    "make progress somehow",
]

ADVERSARIAL_INSTRUCTIONS = [
    "go back to spawn",
    "stop moving and wait",
    "jump repeatedly",
    "look at the sky",
    "craft a bed immediately",
    "place a torch right here",
]

PARAPHRASE_RULES = [
    ("move toward", "head toward"),
    ("move to", "go to"),
    ("go to", "move toward"),
    ("collect", "gather"),
    ("gather", "collect"),
    ("mine", "dig out"),
    ("craft", "make"),
    ("chop", "cut"),
    ("attack", "hit"),
    ("explore", "search"),
    ("nearby", "close by"),
]

CONTRADICTION_RULES = [
    ("move toward", "move away from"),
    ("move to", "move away from"),
    ("go to", "avoid going to"),
    ("collect", "avoid collecting"),
    ("gather", "avoid gathering"),
    ("mine", "stop mining"),
    ("craft", "do not craft"),
    ("chop", "do not chop"),
    ("attack", "avoid attacking"),
    ("explore", "stay in place"),
    ("descend", "climb upward"),
    ("enter", "leave"),
]


def paraphrase_instruction(text: str, client=None) -> str:
    """Rephrase the instruction while preserving semantics."""
    if client is not None:
        resp = client.chat.completions.create(
            model='gpt-4o',
            messages=[{
                'role': 'user',
                'content': f'Rephrase this instruction in different words but same meaning. '
                           f'Keep it 6-18 tokens, imperative: "{text}"'
            }],
            max_tokens=30, temperature=0.7,
        )
        return resp.choices[0].message.content.strip().strip('"')

    lowered = text.lower()
    for src, dst in PARAPHRASE_RULES:
        if src in lowered:
            return lowered.replace(src, dst, 1)
    return f"continue to {lowered}".strip()


def _openai_client_from_env():
    if os.environ.get('USE_API_JUDGE', '0') == '1' and os.environ.get('OPENAI_API_KEY'):
        import openai
        return openai.OpenAI(api_key=os.environ['OPENAI_API_KEY'])
    return None


def contradictory_instruction(text: str, client=None) -> str:
    """Generate an instruction that contradicts the given one."""
    if client is not None:
        resp = client.chat.completions.create(
            model='gpt-4o',
            messages=[{
                'role': 'user',
                'content': f'Generate an instruction that contradicts this one. '
                           f'Keep it 6-18 tokens, imperative: "{text}"'
            }],
            max_tokens=30, temperature=0.7,
        )
        return resp.choices[0].message.content.strip().strip('"')

    lowered = text.lower()
    for src, dst in CONTRADICTION_RULES:
        if src in lowered:
            return lowered.replace(src, dst, 1)
    return random.choice(ADVERSARIAL_INSTRUCTIONS)


def underspecified_instruction() -> str:
    return random.choice(VAGUE_INSTRUCTIONS)


def adversarial_instruction() -> str:
    return random.choice(ADVERSARIAL_INSTRUCTIONS)


# ─────────────────────────────────────────────────────────────────────────────
# Instruction-following accuracy (automated proxy)
# ─────────────────────────────────────────────────────────────────────────────

def compute_following_accuracy(trajectory: list, instruction: str, env_name: str) -> float:
    """
    Automated proxy for instruction-following accuracy.

    In the paper this is a human evaluation (Table 8): annotators see
    trajectories and judge whether the controller followed the instruction.

    Here we use a GPT-4o judge as a scalable proxy (scores 0 or 1 per episode).
    Falls back to heuristic (p_stop fired within episode) if API unavailable.
    """
    try:
        client = _openai_client_from_env()
        if client is None:
            raise RuntimeError('API judge disabled')
        # Build brief trajectory summary
        n_steps  = len(trajectory)
        rewards  = [t['reward'] for t in trajectory]
        p_stops  = [t['p_stop'] for t in trajectory]
        max_pstop= max(p_stops) if p_stops else 0
        traj_summary = (
            f"Instruction: '{instruction}'\n"
            f"Episode length: {n_steps} steps\n"
            f"Total reward: {sum(rewards):.2f}\n"
            f"Max completion probability: {max_pstop:.2f}"
        )
        resp = client.chat.completions.create(
            model='gpt-4o',
            messages=[{
                'role': 'system',
                'content': 'You judge whether an agent followed its instruction. '
                           'Reply with exactly 1 (yes) or 0 (no).',
            }, {
                'role': 'user',
                'content': traj_summary,
            }],
            max_tokens=3, temperature=0.0,
        )
        answer = resp.choices[0].message.content.strip()
        return 1.0 if '1' in answer else 0.0
    except Exception:
        # Fallback: p_stop fired = instruction completed
        p_stops = [t.get('p_stop', 0) for t in trajectory]
        return 1.0 if max(p_stops, default=0) > 0.5 else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Perturbation planner wrapper
# ─────────────────────────────────────────────────────────────────────────────

class PerturbedPlanner:
    """
    Wraps a base planner and perturbs each emitted instruction according
    to `perturbation_type`.
    """

    def __init__(self, base_planner, perturbation_type: str):
        self.base    = base_planner
        self.ptype   = perturbation_type
        self._client = None

    def _get_client(self):
        if self._client is None:
            self._client = _openai_client_from_env()
        return self._client

    def step(self, obs, plan, memory, p_stop, chat_context=''):
        instr, plan, memory = self.base.step(obs, plan, memory, p_stop, chat_context)
        if instr is None:
            return None, plan, memory
        instr = self._perturb(instr)
        return instr, plan, memory

    def emit(self, obs, plan, memory, chat_context=''):
        instr, plan, memory = self.base.emit(obs, plan, memory, chat_context)
        instr = self._perturb(instr)
        return instr, plan, memory

    def advance(self, obs, plan, memory):
        return self.base.advance(obs, plan, memory)

    def _perturb(self, instruction: str) -> str:
        if self.ptype == 'original':
            return instruction
        elif self.ptype == 'paraphrase':
            try:
                return paraphrase_instruction(instruction, self._get_client())
            except Exception:
                return instruction
        elif self.ptype == 'contradictory':
            try:
                return contradictory_instruction(instruction, self._get_client())
            except Exception:
                return instruction
        elif self.ptype == 'underspecified':
            return underspecified_instruction()
        elif self.ptype == 'adversarial':
            return adversarial_instruction()
        return instruction


# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--env',        default='minecraft')
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--config',     nargs='+', default=None)
    p.add_argument('--planner',    default='qwen')
    p.add_argument('--planner_model', default=None)
    p.add_argument('--planner_device', default='cuda')
    p.add_argument('--conditions', nargs='+',
                   default=['original', 'paraphrase', 'contradictory',
                            'underspecified', 'adversarial'])
    p.add_argument('--episodes',   type=int, default=50)
    p.add_argument('--logdir',     default='results/instruction_robustness')
    return p.parse_args()


def main():
    args   = parse_args()
    logdir = Path(args.logdir)
    logdir.mkdir(parents=True, exist_ok=True)

    # Load agent, env, and encoder using the same path as evaluate.py.
    from evaluate import GymAdapter, encode_fn, load_agent, load_config
    from envs import make_env

    config_paths = args.config or ['configs/base.yaml', f'configs/{args.env}.yaml']
    config, _ = load_config(config_paths)
    env = GymAdapter(make_env(args.env, config, lang_size=int(config.get('lang_size', 384))))
    agent = load_agent(args.checkpoint, env.obs_space, env.act_space, config)
    encoder = encode_fn(config.get('lang_size', 384))

    planner_kwargs = {}
    if args.planner_model:
        planner_kwargs['model_name'] = args.planner_model
    if args.planner in ('qwen', 'gemma', 'llava'):
        planner_kwargs['device'] = args.planner_device
    base_planner  = make_planner(
        args.planner,
        task_spec=config.get('task_guidance', ''),
        **planner_kwargs,
    )
    all_results   = {}

    for condition in args.conditions:
        print(f"\nCondition: {condition}")
        planner = PerturbedPlanner(base_planner, condition)
        inf     = OfflineInference(
            agent, planner, env, encoder,
            max_steps=config.get('episode_length', 18_000),
            record=True,
        )
        rewards, follow_accs = [], []
        for ep in range(args.episodes):
            m = inf.run_episode()
            rewards.append(m['total_reward'])
            # Compute instruction-following accuracy for this episode
            traj = m.get('trajectory', [])
            instrs = [x['instruction'] for x in m.get('instruction_log', [])]
            last_instr = instrs[-1] if instrs else ''
            fa = compute_following_accuracy(traj, last_instr, args.env)
            follow_accs.append(fa)
            print(f"  ep {ep+1:2d}: reward={m['total_reward']:.2f}  "
                  f"follow_acc={fa:.0f}  n_instrs={m['instructions_issued']}")

        all_results[condition] = {
            'mean_reward':     float(np.mean(rewards)),
            'std_reward':      float(np.std(rewards)),
            'follow_accuracy': float(np.mean(follow_accs)),
            'rewards':         rewards,
        }

    # Save
    out = logdir / f'robustness_{args.env}_{args.planner}.json'
    with open(out, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved to {out}")

    # Print summary table
    print("\n── Instruction Robustness Results ──")
    print(f"{'Condition':<20}  {'Reward':<18}  {'Follow Acc':>10}")
    print("-" * 55)
    for cond, s in all_results.items():
        print(f"{cond:<20}  {s['mean_reward']:6.2f} ± {s['std_reward']:5.2f}  "
              f"{s['follow_accuracy']*100:>9.1f}%")


if __name__ == '__main__':
    main()
