#!/usr/bin/env python3
"""Does p_stop track ground-truth completion? Fixed-instruction rollouts on crafter; per step record p_stop and
the achievement counts; label done_t = (target achievement count increased since rollout start). Reports AUROC
overall / per instruction, and a time-only control (step index) with identical labels."""
import sys, json, argparse, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent)); sys.path.insert(0, str(Path(__file__).parent / 'dreamerv3'))
from evaluate import load_config, load_agent, GymAdapter, encode_fn
from envs import make_env
from training.annotator import ACHIEVEMENT_INSTRUCTIONS
from sklearn.metrics import roc_auc_score
ap = argparse.ArgumentParser(); ap.add_argument('--checkpoint', required=True); ap.add_argument('--config', nargs='+', required=True)
ap.add_argument('--steps', type=int, default=300); ap.add_argument('--rollouts', type=int, default=10)
ap.add_argument('--targets', default='collect_wood,place_table,collect_sapling,place_plant,collect_drink,collect_stone,make_wood_pickaxe,eat_cow,defeat_zombie,wake_up')
ap.add_argument('--out', default='results/probe_done.json'); a = ap.parse_args()
proj, dv3 = load_config(a.config); env = GymAdapter(make_env('crafter', proj, lang_size=int(proj.get('lang_size', 384))))
agent = load_agent(a.checkpoint, env._env.obs_space, env._env.act_space, proj); enc = encode_fn(int(proj.get('lang_size', 384)))
P, Y, T, TAG, ACH, ROLL, FEAT, IMG = [], [], [], [], [], [], [], []; PF = []; RL = []; PS = []
ach_keys = None; roll_id = 0
for name in a.targets.split(','):
    embed = enc(ACHIEVEMENT_INSTRUCTIONS[name]); key = f'log/achievement_{name}'
    for r in range(a.rollouts):
        obs, _ = env.reset(); state = agent.init_policy(batch_size=1); base = int(obs[key]); roll_id += 1
        if ach_keys is None: ach_keys = sorted(k for k in obs if k.startswith('log/achievement_'))
        for t in range(a.steps):
            ob = {k: np.expand_dims(v, 0) for k, v in obs.items() if not k.startswith('log/')}; ob['lang_embed'] = embed[np.newaxis]
            state, action, extra = agent.policy(state, ob, mode='eval')
            PF.append(float(extra.get('log/p_finish', 0.0))); RL.append(float(extra.get('log/r_lang', 0.0))); PS.append(float(extra.get('log/p_soon', 0.0))); P.append(float(extra.get('log/p_stop', 0.0))); Y.append(float(int(obs[key]) > base)); T.append(t); TAG.append(name)
            ACH.append([int(obs[k]) for k in ach_keys]); ROLL.append(roll_id)
            if 'log/feat' in extra: FEAT.append(np.asarray(extra['log/feat'][0], np.float16))
            IMG.append(np.asarray(obs['image'])[::4, ::4].astype(np.uint8))
            obs, rew, term, trunc, _ = env.step(np.array(action['action'][0]))
            if term or trunc: break
P, Y, T = np.array(P), np.array(Y), np.array(T, np.float32); PF = np.array(PF); RL = np.array(RL); PS = np.array(PS)
res = {'n': int(len(Y)), 'pos_rate': float(Y.mean())}
if Y.sum() and (1 - Y).sum():
    res['auroc_pstop'] = float(roc_auc_score(Y, P)); res['auroc_time_only'] = float(roc_auc_score(Y, T))
    if PF.max() > 0: res['auroc_pfinish'] = float(roc_auc_score(Y, PF))
    res['per_instruction'] = {}
    for name in sorted(set(TAG)):
        m = np.array([g == name for g in TAG])
        if Y[m].sum() and (1 - Y[m]).sum():
            res['per_instruction'][name] = dict(n=int(m.sum()), pos=int(Y[m].sum()), auroc_pstop=float(roc_auc_score(Y[m], P[m])), auroc_time=float(roc_auc_score(Y[m], T[m])), p_stop_mean=float(P[m].mean()))
print(json.dumps(res, indent=1)); Path(a.out).parent.mkdir(parents=True, exist_ok=True); json.dump(res, open(a.out, 'w'), indent=1)
np.savez_compressed(a.out.replace('.json', '_steps.npz'), p=P, pf=PF, rl=RL, ps=PS, y=Y, t=T, tag=np.array(TAG), ach=np.array(ACH, np.int32), roll=np.array(ROLL), ach_keys=np.array(ach_keys), feat=(np.stack(FEAT) if FEAT else np.zeros((0,))), img=np.stack(IMG))
