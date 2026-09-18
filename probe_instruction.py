#!/usr/bin/env python3
"""
Smoke validation: does language causally steer the trained controller?

For each fixed instruction, roll the controller out with that instruction
pinned (never switched) and measure:
  - the action distribution's total-variation distance from the null-
    instruction baseline (does language shift the policy at all?)
  - mean p_stop (does the stop head respond differently per instruction?)
  - reward, and a GIF of one rollout per instruction.

This is a smoke test of the language pathway, not a benchmark.
"""
import sys
import json
import argparse
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(SCRIPT_DIR / 'dreamerv3'))

import numpy as np
from PIL import Image

from evaluate import load_config, load_agent, GymAdapter, encode_fn
from envs import make_env

INSTRS_BY_ENV = {
    "crafter": {
        "null":      "",
        "wood":      "chop the tree to collect wood",
        "water":     "drink water from the lake",
        "attack":    "attack the nearby creature",
        "stone":     "mine the stone to collect it",
        "gibberish": "qxzv kjqx zzkq vqxk",
    },
    "minecraft": {
        "null":      "",
        "wood":      "chop down the tree to collect logs",
        "planks":    "craft wooden planks from your logs",
        "dig":       "dig down to mine stone",
        "explore":   "walk forward and explore the terrain",
        "gibberish": "qxzv kjqx zzkq vqxk",
    },
}


def rollout(agent, env, embed, steps, save_frames=False):
    obs, _ = env.reset()
    state = agent.init_policy(batch_size=1)
    hist = {}
    pstops, frames = [], []
    total_reward = 0.0
    for t in range(steps):
        ob = {k: np.expand_dims(v, 0) for k, v in obs.items()
              if not k.startswith('log/')}
        ob['lang_embed'] = embed[np.newaxis]
        state, action, extra = agent.policy(state, ob, mode='eval')
        p = float(extra.get('log/p_stop', extra.get('p_stop', 0.0)))
        pstops.append(p)
        a_np = np.array(action['action'][0])
        a_id = int(a_np.argmax()) if a_np.ndim >= 1 else int(a_np)
        hist[a_id] = hist.get(a_id, 0) + 1
        if save_frames and t % 2 == 0 and 'image' in obs:
            frames.append(np.asarray(obs['image'], dtype=np.uint8))
        obs, reward, terminated, truncated, _ = env.step(a_np)
        total_reward += reward
        if terminated or truncated:
            obs, _ = env.reset()
            state = agent.init_policy(batch_size=1)
    return hist, float(np.mean(pstops)), total_reward, frames


def tv_distance(h1, h2):
    keys = set(h1) | set(h2)
    n1, n2 = sum(h1.values()), sum(h2.values())
    return 0.5 * sum(abs(h1.get(k, 0) / n1 - h2.get(k, 0) / n2) for k in keys)


def save_gif(frames, path, scale=5):
    imgs = [Image.fromarray(f).resize((f.shape[1] * scale, f.shape[0] * scale),
                                      Image.NEAREST) for f in frames]
    imgs[0].save(path, save_all=True, append_images=imgs[1:], duration=80,
                 loop=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--config', nargs='+',
                    default=['configs/base.yaml', 'configs/crafter.yaml',
                             'configs/crafter_smoke.yaml'])
    ap.add_argument('--steps', type=int, default=300)
    ap.add_argument('--rollouts', type=int, default=4)
    ap.add_argument('--outdir', default='results/smoke_probe')
    args = ap.parse_args()

    proj, _ = load_config(args.config)
    lang_size = int(proj.get('lang_size', 384))
    env_name = proj.get('env_name', 'crafter')
    INSTRS = INSTRS_BY_ENV[env_name]
    env = GymAdapter(make_env(env_name, proj, lang_size=lang_size))
    agent = load_agent(args.checkpoint, env.obs_space, env.act_space, proj)
    encode = encode_fn(lang_size)

    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)

    results, hists = {}, {}
    for name, text in INSTRS.items():
        embed = encode(text) if text else np.zeros(lang_size, np.float32)
        agg_hist, pstop_l, rew_l = {}, [], []
        for r in range(args.rollouts):
            h, ps, rew, frames = rollout(agent, env, embed, args.steps,
                                         save_frames=(r == 0))
            for k, v in h.items():
                agg_hist[k] = agg_hist.get(k, 0) + v
            pstop_l.append(ps)
            rew_l.append(rew)
            if r == 0 and frames:
                save_gif(frames, out / f"probe_{name}.gif")
        hists[name] = agg_hist
        results[name] = dict(
            instruction=text,
            p_stop_mean=float(np.mean(pstop_l)),
            reward_mean=float(np.mean(rew_l)),
            action_hist={str(k): v for k, v in sorted(agg_hist.items())},
        )
        print(f"[{name:>9}] p_stop={results[name]['p_stop_mean']:.3f} "
              f"reward={results[name]['reward_mean']:+.2f} "
              f"top_actions={sorted(agg_hist, key=agg_hist.get)[-4:]}")

    null_h = hists['null']
    for name in INSTRS:
        results[name]['tv_vs_null'] = (0.0 if name == 'null'
                                       else tv_distance(hists[name], null_h))
        if name != 'null':
            print(f"TV(action dist, {name:>9} vs null) = "
                  f"{results[name]['tv_vs_null']:.3f}")

    with open(out / 'probe.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"saved -> {out}/probe.json")


if __name__ == '__main__':
    main()
