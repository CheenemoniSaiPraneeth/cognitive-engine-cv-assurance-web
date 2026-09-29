"""Module 5 - OOD / anomaly assessment of individual operational samples vs the trusted reference
(kNN distance + Mahalanobis in PCA space + Isolation Forest fitted on the REFERENCE only; unsupervised) + new-cluster discovery."""
from __future__ import annotations
import numpy as np
from scipy.stats import binomtest
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import NearestNeighbors
from .features import apply_normalizer


def _rz(ref, x):
    med = np.median(ref); mad = np.median(np.abs(ref - med)) * 1.4826 + 1e-9; return (x - med) / mad


class OODModel:
    def __init__(self, engine, cfg):
        self.eng, self.c = engine, cfg["ood"]; R = engine.Eref; k = self.c["k"]
        self.nn = NearestNeighbors(n_neighbors=k + 1).fit(R); d, _ = self.nn.kneighbors(R); self.ref_knn = d[:, -1]
        self.pca = PCA(min(self.c["pca"], R.shape[1], R.shape[0] - 1), random_state=cfg["seed"]).fit(R); Z = self.pca.transform(R)
        self.mu = Z.mean(0); self.cov_inv = np.linalg.inv(np.cov(Z.T) + 1e-3 * np.eye(Z.shape[1]))
        self.ref_maha = self._maha(Z); self.iso = IsolationForest(n_estimators=100, random_state=cfg["seed"]).fit(Z)
        self.ref_iso = -self.iso.score_samples(Z)
        self.knn_thr = float(np.quantile(self.ref_knn, 0.995))
    def _maha(self, Z): d = Z - self.mu; return np.sqrt(np.einsum("ij,jk,ik->i", d, self.cov_inv, d))
    def score(self, E_raw):
        E = apply_normalizer(E_raw, self.eng.norm); d, _ = self.nn.kneighbors(E, n_neighbors=self.c["k"]); knn = d[:, -1]
        Z = self.pca.transform(E); maha = self._maha(Z); iso = -self.iso.score_samples(Z)
        votes = np.stack([_rz(self.ref_knn, knn) > self.c["z"], _rz(self.ref_maha, maha) > self.c["z"], _rz(self.ref_iso, iso) > self.c["z"]], 1)
        return {"knn": knn, "maha": maha, "iso": iso, "votes": votes, "flag": votes.sum(1) >= self.c["min_votes"], "E": E}
    def ref_rate(self):
        v = np.stack([_rz(self.ref_knn, self.ref_knn) > self.c["z"], _rz(self.ref_maha, self.ref_maha) > self.c["z"], _rz(self.ref_iso, self.ref_iso) > self.c["z"]], 1)
        return float(((v.sum(1)) >= self.c["min_votes"]).mean())
    def assess(self, table):
        s = self.score(table.E); base = max(self.ref_rate(), 0.01); n = len(table); nf = int(s["flag"].sum())
        p = binomtest(nf, n, base, alternative="greater").pvalue if n else 1.0
        # new-cluster discovery: far-from-reference samples that group together
        far = np.where(s["knn"] > self.knn_thr)[0]; clusters = []
        if len(far) >= self.c["new_cluster_min"]:
            eps = float(np.median(self.ref_knn)) * 2.0
            lab = DBSCAN(eps=eps, min_samples=self.c["new_cluster_min"]).fit_predict(s["E"][far])
            for L in sorted(set(lab) - {-1}):
                mem = far[lab == L]; clusters.append({"size": int(len(mem)), "members": [table.items[i]["id"] for i in mem[:20]],
                                                      "mean_knn_distance": round(float(s["knn"][mem].mean()), 4)})
        return {"n": n, "flagged": nf, "rate": round(nf / max(n, 1), 4), "ref_rate": round(base, 4), "binomial_p": round(float(p), 6),
                "flag": s["flag"], "knn": s["knn"], "new_clusters": clusters, "far_fraction": round(len(far) / max(n, 1), 4)}
