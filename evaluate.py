"""
evaluate.py — evaluate a trained Instruct-to-Act agent.

Usage
─────
  # Single-agent, 100 episodes
  python evaluate.py --config configs/minecraft.yaml \
                     --checkpoint logdir/minecraft/checkpoint.pkl \
                     --planner gpt4o --mode online --episodes 100

  # Multi-agent
  python evaluate.py --config configs/overcooked.yaml \
                     --checkpoint logdir/overcooked/checkpoint.pkl \
                     --n_agents 2 --mode online

Outputs
───────
  results/
    <env>_<planner>_<mode>_seed<s>.json   — per-episode metrics
    <env>_<planner>_<mode>_summary.json   — mean ± std across seeds/episodes
"""

import sys, os, argparse, json, time
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(SCRIPT_DIR / 'dreamerv3'))

import numpy as np
import ruamel.yaml as yaml
import elements
import embodied

from agent     import LangCondAgent
from planners  import make_planner
from envs      import make_env
from inference import AsyncOnlineInference, OfflineInference, MultiAgentInference
from train     import load_project_config, build_dv3_config


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--config',      nargs='+', default=['configs/base.yaml'])
    p.add_argument('--checkpoint',  required=True)
    p.add_argument('--planner',     default='gpt4o',
                   choices=['gpt4o', 'qwen', 'gemma', 'llava', 'scripted'])
    p.add_argument('--mode',        default='online', choices=['online', 'offline'])
    p.add_argument('--episodes',    type=int, default=100)
    p.add_argument('--max_steps',   type=int, default=None,
                   help='Override episode horizon (default: episode_length from config)')
    p.add_argument('--n_agents',    type=int, default=1)
    p.add_argument('--seed',        type=int, default=0)
    p.add_argument('--outdir',      default='results')
    p.add_argument('--record',      action='store_true')
    p.add_argument('--stop_probe',  default=None, help='pickle of an offline-fitted stop probe (overrides the stop head)')
    return p.parse_args()


def load_config(paths):
    proj = load_project_config(paths)
    dv3  = build_dv3_config(proj)
    return proj, dv3


def load_agent(checkpoint_path, obs_space, act_space, proj):
    from train import make_agent_fn
    dv3   = build_dv3_config(proj)
    agent = make_agent_fn(proj, dv3, obs_space, act_space)
    # checkpoint_path may be the agent.pkl file or the ckpt directory;
    # elements.Checkpoint expects the parent directory containing 'latest'.
    ckpt_dir = elements.Path(checkpoint_path)
    if str(ckpt_dir).endswith('.pkl'):
        ckpt_dir = ckpt_dir.parent.parent  # .../ckpt/<timestamp>/agent.pkl → .../ckpt
    ckpt = elements.Checkpoint(ckpt_dir)
    ckpt.agent = agent
    ckpt.load(keys=['agent'])
    return agent


class GymAdapter:
    """
    Wraps a DreamerV3 embodied env (step-only API, obs dict) into the
    gymnasium-style (reset/step) API expected by AsyncOnlineInference /
    OfflineInference.

    DreamerV3 resets via step({'reset': True, ...}); all info (reward,
    is_last, is_terminal) is bundled inside the returned obs dict.
    """
    def __init__(self, env):
        self._env  = env
        self.obs_space = env.obs_space
        self.act_space = env.act_space

    def reset(self):
        act = {k: np.zeros(s.shape, s.dtype) for k, s in self._env.act_space.items()}
        act['reset'] = np.array(True)
        obs = self._env.step(act)
        return obs, {}

    def step(self, action):
        act = {k: np.zeros(s.shape, s.dtype) for k, s in self._env.act_space.items()}
        act['reset']  = np.array(False)
        act['action'] = np.asarray(action)
        obs = self._env.step(act)
        reward     = float(obs.get('reward', 0.0))
        terminated = bool(obs.get('is_terminal', False))
        truncated  = bool(obs.get('is_last', False)) and not terminated
        return obs, reward, terminated, truncated, {}

    def close(self):
        self._env.close()


def encode_fn(lang_size=384):
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

    def _encode(text: str) -> np.ndarray:
        if not text:
            return np.zeros(lang_size, dtype=np.float32)
        return model.encode([text], normalize_embeddings=True)[0].astype(np.float32)
    return _encode


def _find_attr(obj, name, depth=0):
    if depth > 8 or obj is None: return None
    if hasattr(obj, name): return getattr(obj, name)
    for a in ('_env', 'env', '_gymenv', 'envs'):
        sub = getattr(obj, a, None)
        if isinstance(sub, (list, tuple)): sub = sub[0] if sub else None
        r = _find_attr(sub, name, depth + 1)
        if r is not None: return r
    return None


def _save_record(traj, env, outdir, ep, instruction_log):
    """stop-head relabel study: per-step feat/inventory/instruction/p_stop -> npz"""
    import os
    os.makedirs(outdir, exist_ok=True)
    keys = _find_attr(env, '_inv_keys') or []
    texts = sorted({t['instruction'] for t in traj})
    tid = {t: i for i, t in enumerate(texts)}
    np.savez_compressed(os.path.join(outdir, f'record_ep{ep}.npz'),
        feat=np.stack([t['feat'] for t in traj]), inv=np.stack([t['inv'] for t in traj]),
        inv_post=np.stack([t['inv_post'] for t in traj]), p_stop=np.array([t['p_stop'] for t in traj], np.float32), p_stop_head=np.array([t.get('p_stop_head', 0.0) for t in traj], np.float32), r_lang=np.array([t.get('r_lang', 0.0) for t in traj], np.float32), p_soon=np.array([t.get('p_soon', 0.0) for t in traj], np.float32),
        instr_id=np.array([tid[t['instruction']] for t in traj], np.int16), step=np.array([t['step'] for t in traj], np.int32),
        reward=np.array([t['reward'] for t in traj], np.float32), instr_texts=np.array(texts), inv_keys=np.array(list(keys)),
        status=np.array([str(t.get('status', '')) for t in traj]), anomaly_z=np.array([t.get('anomaly_z', -1.0) for t in traj], np.float32),
        instruction_log=np.array([f"{e.get('step')}|{e.get('instruction')}" for e in instruction_log]))
    import json as _json
    with open(os.path.join(outdir, f'instr_log_ep{ep}.json'), 'w') as f:
        _json.dump(instruction_log, f, indent=1, default=str)
    print(f"  saved {outdir}/record_ep{ep}.npz  steps={len(traj)} feat_dim={traj[0]['feat'].shape[-1]} inv_keys={len(keys)}", flush=True)


def _make_state_fmt(env, config):
    """Text state for the planner (inventory / equipped / vitals / change since instruction) — on unless planner_state: false."""
    if not config.get('planner_state', True): return None
    from planners.state_text import StateFormatter, find_attr
    inv_keys = find_attr(env, '_inv_keys') or []; equip = find_attr(env, '_equip_enum') or []
    return StateFormatter(inv_keys, equip) if inv_keys else None


def _make_anomaly(config):
    ref = config.get('anomaly_ref')
    if not ref: return None
    from planners.anomaly import FeatAnomaly
    return FeatAnomaly(ref, config.get('anomaly_z'))


def run_single_agent(agent, planner, env, encoder, args, config):
    InfCls = AsyncOnlineInference if args.mode == 'online' else OfflineInference
    max_steps = args.max_steps or config.get('episode_length', 18_000)
    inf = InfCls(
        controller    = agent,
        planner       = planner,
        env           = env,
        lang_encoder  = encoder,
        stop_threshold= config.get('stop_threshold', 0.5),
        max_steps     = max_steps,
        record        = args.record,
        stop_patience  = config.get('stop_patience', 1),
        min_instr_steps= config.get('min_instr_steps', 1),
        max_instr_steps= config.get('max_instr_steps', None),
        stop_probe    = args.stop_probe,
        stop_signal   = config.get('stop_signal', 'p_stop'),
        stop_window   = config.get('stop_window', 1),
        state_fmt     = _make_state_fmt(env, config),
        feedback      = bool(config.get('planner_feedback', True)),
        stuck_steps   = int(config.get('stuck_steps', 120)),
        anomaly       = _make_anomaly(config),
        anomaly_patience = int(config.get('anomaly_patience', 5)),
    )
    lang_size = config.get('lang_size', 384)
    env_name  = config.get('env_name', 'crafter')

    all_rewards, all_steps, all_episodes = [], [], []
    ep = 0
    while ep < args.episodes:
        try:
            m = inf.run_episode()
        except (ConnectionRefusedError, TimeoutError, OSError) as e:
            print(f"  ep {ep+1:3d}/{args.episodes}  Java server died ({type(e).__name__}), recreating env...")
            try:
                inf.env.close()
            except Exception:
                pass
            inf.env = GymAdapter(make_env(env_name, config, lang_size=int(lang_size)))
            continue  # retry same episode
        all_rewards.append(m['total_reward'])
        if args.record and m.get('trajectory'):
            _save_record(m['trajectory'], inf.env, args.outdir, ep, m.get('instruction_log', []))
        all_steps.append(m['episode_steps'])
        ep_detail = {
            'ep':               ep,
            'reward':           m['total_reward'],
            'steps':            m['episode_steps'],
            'instructions_issued': m['instructions_issued'],
            'instruction_log':  m.get('instruction_log', []),
            'p_stop_mean':      float(np.mean(m['p_stop_log'])) if m.get('p_stop_log') else None,
            'feedback_stuck':   m.get('feedback_stuck', 0), 'feedback_anomaly': m.get('feedback_anomaly', 0),
            'verified_yes':     m.get('verified_yes', 0), 'verified_no': m.get('verified_no', 0), 'verified_unknown': m.get('verified_unknown', 0),
            'p_stop_max':       float(np.max(m['p_stop_log']))  if m.get('p_stop_log') else None,
            'vlm_latency_ms_mean': float(np.mean(m['vlm_latencies_ms'])) if m.get('vlm_latencies_ms') else None,
        }
        all_episodes.append(ep_detail)
        instrs = [e['instruction'] for e in m.get('instruction_log', [])]
        print(f"  ep {ep+1:3d}/{args.episodes}  reward={m['total_reward']:.2f}  "
              f"steps={m['episode_steps']}  instrs={m['instructions_issued']}  handoff h/t/s/a={m.get('handoff_head',0)}/{m.get('handoff_timer',0)}/{m.get('handoff_stuck',0)}/{m.get('handoff_anomaly',0)} ver y/n/?={m.get('verified_yes',0)}/{m.get('verified_no',0)}/{m.get('verified_unknown',0)}  "
              f"p_stop_mean={ep_detail['p_stop_mean']:.3f}" if ep_detail['p_stop_mean'] is not None
              else f"  ep {ep+1:3d}/{args.episodes}  reward={m['total_reward']:.2f}  "
                   f"steps={m['episode_steps']}  instrs={m['instructions_issued']}")
        for i, instr in enumerate(instrs):
            print(f"    instr {i+1}: \"{instr}\"")
        ep += 1
    return all_rewards, all_steps, all_episodes


def run_multi_agent(agent, planners, envs, encoder, args, config):
    inf = MultiAgentInference(
        controller    = agent,
        planners      = planners,
        envs          = envs,
        lang_encoder  = encoder,
        mode          = config.get('comm_mode', 'decentralized'),
        stop_threshold= config.get('stop_threshold', 0.5),
        max_steps     = config.get('episode_length', 18_000),
        record        = args.record,
    )
    all_team_rewards = []
    for ep in range(args.episodes):
        m = inf.run_episode()
        all_team_rewards.append(m['team_reward'])
        print(f"  ep {ep+1:3d}/{args.episodes}  team_reward={m['team_reward']:.2f}")
    return all_team_rewards


def main():
    args        = parse_args()
    proj, dv3   = load_config(args.config)
    np.random.seed(args.seed)

    env_name = proj.get('env_name', 'crafter')
    encoder  = encode_fn(proj.get('lang_size', 384))

    # Build env(s) and adapt DreamerV3 step-only API to gym reset/step API
    lang_size = int(proj.get('lang_size', 384))
    if args.n_agents > 1:
        envs = [GymAdapter(make_env(env_name, proj, lang_size=lang_size))
                for _ in range(args.n_agents)]
        env  = envs[0]
    else:
        env  = GymAdapter(make_env(env_name, proj, lang_size=lang_size))
        envs = [env]

    # Load agent
    agent = load_agent(args.checkpoint, env.obs_space, env.act_space, proj)

    # Build planner(s)
    task_spec = proj.get('task_guidance', '')
    if args.n_agents > 1:
        planners = [make_planner(args.planner, task_spec=task_spec)
                    for _ in range(args.n_agents)]
        planner  = planners[0]
    else:
        planner  = make_planner(args.planner, task_spec=task_spec)
        planners = [planner]

    # Run evaluation
    print(f"\nEvaluating {env_name} | planner={args.planner} | mode={args.mode} | "
          f"n_agents={args.n_agents} | episodes={args.episodes}")

    t0 = time.time()
    if args.n_agents > 1:
        rewards = run_multi_agent(agent, planners, envs, encoder, args, proj)
        key     = 'team_reward'
        episodes_detail = []
    else:
        rewards, steps, episodes_detail = run_single_agent(agent, planner, env, encoder, args, proj)
        key     = 'reward'

    elapsed = time.time() - t0
    arr = np.array(rewards)
    summary = {
        'env':      env_name,
        'planner':  args.planner,
        'mode':     args.mode,
        'seed':     args.seed,
        'episodes': args.episodes,
        'mean':     float(arr.mean()),
        'std':      float(arr.std()),
        'ci95':     float(1.96 * arr.std() / np.sqrt(len(arr))),
        'min':      float(arr.min()),
        'max':      float(arr.max()),
        'elapsed_s': elapsed,
    }

    # Save
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    tag = f"{env_name}_{args.planner}_{args.mode}_seed{args.seed}"
    with open(out / f"{tag}_episodes.json", 'w') as f:
        json.dump({'rewards': rewards, 'episode_detail': episodes_detail, **summary}, f, indent=2)
    with open(out / f"{tag}_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults: {key}={summary['mean']:.2f} ± {summary['std']:.2f} "
          f"(95% CI ±{summary['ci95']:.2f})")
    print(f"Saved to {out / tag}_*.json")


if __name__ == '__main__':
    main()
