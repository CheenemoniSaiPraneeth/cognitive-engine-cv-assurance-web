"""Classical two-sample statistics (no learning): KS, PSI, JSD, Wasserstein, categorical JSD."""
from __future__ import annotations
import numpy as np
from scipy.spatial.distance import jensenshannon
from scipy.stats import wasserstein_distance

def ks(a, b):
    a, b = np.sort(a), np.sort(b); v = np.concatenate([a, b])
    return float(np.max(np.abs(np.searchsorted(a, v, "right") / len(a) - np.searchsorted(b, v, "right") / len(b))))

def _edges(ref, bins):
    e = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)[1:-1]))
    if len(e) < 2:
        v = float(e[0]) if len(e) else float(ref[0]); d = 1e-6 * max(1.0, abs(v)); e = np.array([v - d, v + d])
    return e

def psi(ref, cur, bins=10):
    e = _edges(ref, bins); n = len(e) + 1
    r = np.clip(np.bincount(np.searchsorted(e, ref, "right"), minlength=n) / len(ref), 1e-4, None)
    c = np.clip(np.bincount(np.searchsorted(e, cur, "right"), minlength=n) / len(cur), 1e-4, None)
    return float(np.sum((c - r) * np.log(c / r)))

def jsd(ref, cur, bins=20):
    lo, hi = min(ref.min(), cur.min()), max(ref.max(), cur.max())
    if hi - lo < 1e-12: return 0.0
    r, _ = np.histogram(ref, bins, (lo, hi)); c, _ = np.histogram(cur, bins, (lo, hi))
    return float(jensenshannon(r + 1e-6, c + 1e-6, base=2))

def wass(ref, cur):
    s = max(float(ref.std()), 0.02 * abs(float(ref.mean())), 1e-6)
    return float(wasserstein_distance(ref, cur) / s)

def cat_jsd(pref, pcur, k):
    r = np.bincount(pref[pref >= 0], minlength=k) + 1e-3; c = np.bincount(pcur[pcur >= 0], minlength=k) + 1e-3
    return float(jensenshannon(r, c, base=2))

def scalar_stats(ref, cur):
    return np.array([ks(ref, cur), psi(ref, cur), jsd(ref, cur), wass(ref, cur)])
SCALAR_FLOOR = np.array([0.05, 0.05, 0.03, 0.05])
