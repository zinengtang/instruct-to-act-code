"""Fit an instruction-completion probe on recorded rollouts (evaluate.py --record with I2A_EXPORT_FEAT=1).
Label: done_t = 1 iff the target inventory item of the current instruction has increased since the instruction was issued.
Episode-level 5-fold CV reports AUROC/AP of (a) the controller's own p_stop, (b) logistic on [feat, onehot(instr)],
(c) MLP on the same; then refits the chosen model on all episodes and saves probe.pkl for evaluate.py --stop_probe.
usage: python fit_stop_probe.py <record_dir> <out.pkl> [--model logistic|mlp]"""
import glob, sys, json, argparse, time, numpy as np, joblib
T0 = time.time()
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score, average_precision_score
ap = argparse.ArgumentParser(); ap.add_argument('rec'); ap.add_argument('out'); ap.add_argument('--model', default='logistic'); ap.add_argument('--label', default='cumulative', help='cumulative | recent (target increased within the last W steps)'); ap.add_argument('--W', type=int, default=16); a = ap.parse_args()
TARGET = {"collect wood from nearby trees": "log", "craft planks from collected wood": "planks", "craft a crafting table": "crafting_table",
          "craft sticks for tool recipes": "stick", "craft a wooden pickaxe": "wooden_pickaxe", "mine stone to gather cobblestone": "cobblestone",
          "craft a stone pickaxe": "stone_pickaxe", "mine iron ore with the stone pickaxe": "iron_ore", "craft a furnace near your position": "furnace",
          "smelt iron ore into ingots": "iron_ingot", "craft an iron pickaxe": "iron_pickaxe", "mine diamond ore with the iron pickaxe": "diamond"}
INSTR = ["collect wood from nearby trees", "craft planks from collected wood", "craft a crafting table", "craft sticks for tool recipes", "craft a wooden pickaxe",
         "mine stone to gather cobblestone", "craft a stone pickaxe", "search underground for iron ore", "mine iron ore with the stone pickaxe",
         "craft a furnace near your position", "smelt iron ore into ingots", "craft an iron pickaxe", "search deep underground for diamond ore", "mine diamond ore with the iron pickaxe"]
X, Y, P, EP = [], [], [], []
files = sorted(glob.glob(f"{a.rec}/record_ep*.npz")); print(f"{len(files)} episodes")
for ep, f in enumerate(files):
    d = np.load(f, allow_pickle=True); keys = [k.replace("inventory/", "") for k in d["inv_keys"]]
    texts = list(d["instr_texts"]); iid = d["instr_id"]; inv = d["inv"]; feat = d["feat"].astype(np.float32); ps = d["p_stop"]
    prev = None; n_pos = 0
    for t in range(len(iid)):
        if iid[t] != prev: t0 = t; prev = iid[t]
        txt = texts[iid[t]]; item = TARGET.get(txt)
        if item is None or item not in keys: continue
        j = keys.index(item); y = int(inv[t, j] > inv[max(t0, t - a.W), j]) if a.label == 'recent' else int(inv[t, j] > inv[t0, j]); n_pos += y
        oh = np.zeros(len(INSTR), np.float32); oh[INSTR.index(txt)] = 1.0 if txt in INSTR else 0.0
        X.append(np.concatenate([feat[t], oh])); Y.append(y); P.append(float(ps[t])); EP.append(ep)
X, Y, P, EP = np.array(X, np.float32), np.array(Y), np.array(P), np.array(EP)
print(f"{len(Y)} labelled steps, positives {Y.mean():.3f}, feat_dim {X.shape[1]-len(INSTR)}  ({time.time()-T0:.0f}s)", flush=True)
def mk(): return make_pipeline(StandardScaler(), LogisticRegression(C=0.05, max_iter=300, class_weight='balanced', solver='lbfgs')) if a.model == 'logistic' else make_pipeline(StandardScaler(), MLPClassifier((256,), max_iter=300, early_stopping=True, random_state=0))
eps = np.unique(EP); rng = np.random.RandomState(0); rng.shuffle(eps); folds = np.array_split(eps, 5); au, ap_, au0, ap0 = [], [], [], []
for f in folds:
    te = np.isin(EP, f); tr = ~te
    if Y[te].min() == Y[te].max(): continue
    m = mk().fit(X[tr], Y[tr]); pr = m.predict_proba(X[te])[:, 1]
    print(f'  fold done ({time.time()-T0:.0f}s)', flush=True); au.append(roc_auc_score(Y[te], pr)); ap_.append(average_precision_score(Y[te], pr)); au0.append(roc_auc_score(Y[te], P[te])); ap0.append(average_precision_score(Y[te], P[te]))
print(f"episode-level CV: own p_stop AUROC {np.mean(au0):.3f} AP {np.mean(ap0):.3f} | probe({a.model}) AUROC {np.mean(au):.3f} AP {np.mean(ap_):.3f}")
m = mk().fit(X, Y); joblib.dump({'model': m, 'instructions': INSTR, 'feat_dim': int(X.shape[1] - len(INSTR)), 'cv': dict(auroc=float(np.mean(au)), ap=float(np.mean(ap_)), own_auroc=float(np.mean(au0)))}, a.out); print('saved', a.out)
