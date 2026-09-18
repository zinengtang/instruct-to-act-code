"""Offline scoring of completion signals from recorded rollouts (evaluate.py --record): AUROC of p_stop_head / r_lang / p_soon
against ground-truth labels from inventory events: exact event step, event within last K steps, cumulative since instruction start.
usage: python score_signals.py <record_dir> [K=4]"""
import glob, sys, numpy as np
from sklearn.metrics import roc_auc_score
TARGET = {"collect wood from nearby trees": "log", "craft planks from collected wood": "planks", "craft a crafting table": "crafting_table",
          "craft sticks for tool recipes": "stick", "craft a wooden pickaxe": "wooden_pickaxe", "mine stone to gather cobblestone": "cobblestone",
          "craft a stone pickaxe": "stone_pickaxe", "mine iron ore with the stone pickaxe": "iron_ore", "craft a furnace near your position": "furnace",
          "smelt iron ore into ingots": "iron_ingot", "craft an iron pickaxe": "iron_pickaxe", "mine diamond ore with the iron pickaxe": "diamond"}
K = int(sys.argv[2]) if len(sys.argv) > 2 else 4
S = {k: [] for k in ('p_stop_head', 'r_lang', 'p_soon', 'r_lang_max')}; Y = {k: [] for k in ('exact', 'recent', 'cum')}; T = []
for f in sorted(glob.glob(f"{sys.argv[1]}/record_ep*.npz")):
    d = np.load(f, allow_pickle=True)
    if 'r_lang' not in d.files: continue
    keys = [k.replace("inventory/", "") for k in d["inv_keys"]]; texts = list(d["instr_texts"]); iid = d["instr_id"]; inv = d["inv_post"]
    prev = None
    for t in range(len(iid)):
        if iid[t] != prev: t0 = t; prev = iid[t]
        item = TARGET.get(texts[iid[t]])
        if item is None or item not in keys: continue
        j = keys.index(item); base = inv[max(t0 - 1, 0), j]
        ev = int(inv[t, j] > inv[max(t - 1, t0 - 1) if t > t0 else max(t0 - 1, 0), j]) if t >= t0 else 0
        Y['exact'].append(ev); Y['recent'].append(int(inv[t, j] > inv[max(t0 - 1, t - K), j])); Y['cum'].append(int(inv[t, j] > base)); T.append(t - t0)
        for k in ('p_stop_head', 'r_lang', 'p_soon'): S[k].append(float(d[k][t]))
        S['r_lang_max'].append(float(d['r_lang'][max(t0, t - K + 1):t + 1].max()))
if not T: print('no signals recorded'); sys.exit(0)
T = np.array(T, float); print(f"{len(T)} labelled steps; positives exact {np.mean(Y['exact']):.4f} recent-{K} {np.mean(Y['recent']):.4f} cumulative {np.mean(Y['cum']):.3f}")
for lab in ('exact', 'recent', 'cum'):
    y = np.array(Y[lab])
    if y.min() == y.max(): continue
    print(f"  label={lab:7s}  " + "  ".join(f"{k}={roc_auc_score(y, np.array(S[k])):.3f}" for k in S) + f"  time-only={roc_auc_score(y, T):.3f}")
