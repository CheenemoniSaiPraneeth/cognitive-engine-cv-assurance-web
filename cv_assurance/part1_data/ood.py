"""Module 6 - OOD / anomaly detection with complementary scorers (no training on labels).

Two feature views:
  * embedding view (frozen/handcrafted embedding): kNN distance, Mahalanobis (PCA-whitened), LOF, Isolation Forest
  * engineered-statistics view (brightness, contrast, sharpness, entropy, mean RGB): Mahalanobis, Isolation Forest, kNN
Every score becomes a robust z-score against the *reference* population; a sample is flagged when at
least ``min_votes`` scorers exceed ``z_threshold``. Without a trusted reference the dataset is its own
reference (self-reference mode, reported as a limitation).
"""
from __future__ import annotations

import warnings

import numpy as np
from sklearn.covariance import MinCovDet
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors

DETECTOR = "ood_multi_scorer_v2"
STAT_KEYS = ("brightness", "contrast", "sharpness", "entropy", "mean_r", "mean_g", "mean_b",
             "std_r", "std_g", "std_b", "saturation", "extreme_fraction")


def stats_matrix(stats_list) -> np.ndarray:
    S = np.array([[s[k] for k in STAT_KEYS] for s in stats_list], dtype=np.float64)
    S[:, 2] = np.log1p(S[:, 2])  # sharpness is heavy-tailed
    return S


def robust_standardize(S, ref=None):
    ref = S if ref is None else ref
    med = np.median(ref, 0)
    mad = 1.4826 * np.median(np.abs(ref - med), 0)
    scale = np.maximum(mad, 0.05 * np.abs(med) + 1e-3)
    return np.clip((S - med) / scale, -10, 10)


def _robust_z(ref_scores, scores):
    med = np.median(ref_scores)
    mad = np.median(np.abs(ref_scores - med))
    scale = max(1.4826 * mad, 0.1 * abs(med), 1e-6)
    return (scores - med) / scale


def _view(R, T, self_mode, prefix, methods, k, ncomp, metric, seed):
    ref, tgt = {}, {}
    k = max(2, min(k, len(R) - 2))
    if "knn" in methods:
        nn = NearestNeighbors(n_neighbors=k + 1, metric=metric, algorithm="brute").fit(R)
        d, _ = nn.kneighbors(R)
        ref[prefix + "knn_distance"] = d[:, 1:].mean(1)
        tgt[prefix + "knn_distance"] = ref[prefix + "knn_distance"] if self_mode else nn.kneighbors(T, n_neighbors=k)[0].mean(1)
    if "maha" in methods:
        nc = int(min(ncomp, len(R) - 1, R.shape[1]))
        pca = PCA(n_components=nc, whiten=True, random_state=seed).fit(R)
        ref[prefix + "mahalanobis"] = np.linalg.norm(pca.transform(R), axis=1)
        tgt[prefix + "mahalanobis"] = ref[prefix + "mahalanobis"] if self_mode else np.linalg.norm(pca.transform(T), axis=1)
    if "lof" in methods:
        lof = LocalOutlierFactor(n_neighbors=max(2, min(20, len(R) - 1)), metric=metric, novelty=True).fit(R)
        ref[prefix + "lof"] = -lof.negative_outlier_factor_
        tgt[prefix + "lof"] = ref[prefix + "lof"] if self_mode else -lof.score_samples(T)
    if "rmaha" in methods:  # robust (MCD) Mahalanobis: resists masking by the outliers themselves
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mcd = MinCovDet(support_fraction=0.75, random_state=seed).fit(R)
        ref[prefix + "robust_mahalanobis"] = np.sqrt(mcd.mahalanobis(R))
        tgt[prefix + "robust_mahalanobis"] = ref[prefix + "robust_mahalanobis"] if self_mode else np.sqrt(mcd.mahalanobis(T))
    if "iso" in methods:
        iso = IsolationForest(n_estimators=200, random_state=seed).fit(R)
        ref[prefix + "isolation_forest"] = -iso.score_samples(R)
        tgt[prefix + "isolation_forest"] = ref[prefix + "isolation_forest"] if self_mode else -iso.score_samples(T)
    return ref, tgt


def detect_ood(subset, E, S, cfg, store, flags, ref_E=None, ref_S=None, ref_ids=None, seed=0):
    """E: normalised embeddings, S: robust-standardised engineered stats (rows aligned with subset)."""
    c = cfg["ood"]
    self_mode = ref_E is None
    R, RS = (E, S) if self_mode else (ref_E, ref_S)
    if len(R) < c["min_samples"]:
        return {"status": "UNAVAILABLE", "reason": f"reference has < {c['min_samples']} samples"}
    r1, t1 = _view(R, E, self_mode, "emb_", ("knn", "maha", "lof", "iso"), c["k"], c["pca_components"], "cosine", seed)
    r2, t2 = _view(RS, S, self_mode, "stat_", ("knn", "rmaha", "iso"), c["k"], 8, "euclidean", seed)
    ref, tgt = {**r1, **r2}, {**t1, **t2}
    z = {m: _robust_z(ref[m], tgt[m]) for m in ref}
    votes = sum((z[m] > c["z_threshold"]).astype(int) for m in z)
    nscore = len(z)
    nn_ref = NearestNeighbors(n_neighbors=min(4, len(R)), metric="cosine", algorithm="brute").fit(R)
    dist, ind = nn_ref.kneighbors(E)
    rid = ref_ids if ref_ids is not None else [s.sample_id for s in subset]
    flagged = 0
    for i, s in enumerate(subset):
        if votes[i] >= c["min_votes"]:
            near = [(rid[j], round(1 - float(d), 3)) for j, d in zip(ind[i], dist[i])
                    if not (self_mode and j == i)][:3]
            sev = "high" if votes[i] >= 5 else "medium" if votes[i] >= 3 else "low"
            store.add("sample", s.sample_id, "ood_anomaly", sev, votes[i] / nscore, DETECTOR,
                      {"votes": int(votes[i]), "scorers": nscore,
                       "reference_mode": "self" if self_mode else "trusted_reference",
                       "scores": {m: round(float(tgt[m][i]), 4) for m in tgt},
                       "robust_z": {m: round(float(z[m][i]), 2) for m in z},
                       "nearest_reference": near},
                      "REVIEW",
                      ["Distributional difference only - not evidence of malicious intent.",
                       "Rare legitimate samples can be flagged."] +
                      (["No trusted reference supplied: dataset used as its own reference."] if self_mode else []))
            flags.setdefault(s.sample_id, set()).add("ood_anomaly")
            flagged += 1
    return {"status": "OK", "mode": "self" if self_mode else "trusted_reference",
            "flagged": flagged, "assessed": len(subset), "scorers": nscore}
