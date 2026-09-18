"""Fit the anomaly reference (PCA whitening of world-model features) from recorded rollouts.
usage: python build_anomaly_ref.py <record_dir> <out.npz> [k=64] [max_rows=20000]"""
import glob, sys, numpy as np
rec, out = sys.argv[1], sys.argv[2]; k = int(sys.argv[3]) if len(sys.argv) > 3 else 64; max_rows = int(sys.argv[4]) if len(sys.argv) > 4 else 20000
X = []
for f in sorted(glob.glob(f'{rec}/record_ep*.npz')):
    d = np.load(f, allow_pickle=True); X.append(d['feat'].astype(np.float32))
X = np.concatenate(X); rng = np.random.RandomState(0)
if len(X) > max_rows: X = X[rng.choice(len(X), max_rows, replace=False)]
mean = X.mean(0); Xc = X - mean
from sklearn.utils.extmath import randomized_svd
U, S, Vt = randomized_svd(Xc, n_components=k, random_state=0)
W = Vt.T / (S / np.sqrt(len(Xc)))[None, :]          # whiten: unit variance per component on the reference set
z = np.sqrt(((Xc @ W) ** 2).sum(1) / k)
np.savez(out, mean=mean, W=W, z_p99=float(np.percentile(z, 99)), z_p999=float(np.percentile(z, 99.9)))
print(f'{len(X)} rows, k={k}, reference z: median {np.median(z):.3f} p99 {np.percentile(z, 99):.3f} p99.9 {np.percentile(z, 99.9):.3f} -> saved {out}')
