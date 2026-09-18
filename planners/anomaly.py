"""Training-free anomaly score on the controller's world-model features: Mahalanobis distance in a PCA-whitened subspace
fitted on reference rollouts (build_anomaly_ref.py). z > threshold for a few consecutive steps = 'anomaly' feedback."""
import numpy as np


class FeatAnomaly:
    def __init__(self, ref_path: str, z_threshold: float = None):
        d = np.load(ref_path)
        self.mean = d['mean'].astype(np.float32); self.W = d['W'].astype(np.float32)   # (D, k) whitening projection
        self.k = self.W.shape[1]; self.thr = float(z_threshold) if z_threshold is not None else float(d['z_p99'])

    def score(self, feat) -> float:
        x = np.asarray(feat, np.float32).reshape(-1) - self.mean
        z = x @ self.W
        return float(np.sqrt((z * z).sum() / self.k))   # ~1 on reference data
