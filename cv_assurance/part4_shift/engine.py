"""Shift engine: reference-calibrated tests. Modules 2-4 (feature stats, embedding-space comparison, statistical shift).
Calibration = null distribution from clean-reference splits (same batch size), so thresholds come ONLY from clean data."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from ..common.hashing import sha256_json
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import jensenshannon
from .features import ALL_FEATURES, FAMILY, apply_normalizer, fit_normalizer
from .stats import SCALAR_FLOOR, cat_jsd, scalar_stats


def test_family(name):
    if name.startswith("emb:"): return "semantic"
    if name == "task:class_prior": return "class_mix"
    return FAMILY[name.split(":", 1)[1]]


class ShiftEngine:
    def __init__(self, ref, cfg, cache_dir=None):
        self.ref, self.cfg, self.cache_dir = ref, cfg, (Path(cache_dir) if cache_dir else None); self.N = len(ref)
        self.norm = fit_normalizer(ref.E); self.Eref = apply_normalizer(ref.E, self.norm)
        ec = cfg["embedding"]; self.k = ec["k"]
        self.km = KMeans(ec["clusters"], n_init=4, random_state=cfg["seed"]).fit(self.Eref)
        rs = np.random.default_rng(cfg["seed"]).choice(self.N, min(self.N, ec["max_points"]), replace=False)
        X = self.Eref[rs]; d = np.sqrt(np.maximum((X ** 2).sum(1)[:, None] + (X ** 2).sum(1)[None] - 2 * X @ X.T, 0)); self.sigma = float(np.median(d[d > 0]))
        self.cols = [j for j, f in enumerate(ALL_FEATURES) if np.isfinite(ref.S[:, j]).sum() >= max(20, 0.5 * self.N)]
        self.has_classes = (ref.pred >= 0).sum() >= 0.5 * self.N
        self.ncls = int(ref.pred.max()) + 1 if self.has_classes else 0
        self.cache = {}

    # ---- raw statistics for (refset, curset) ; sets are (S, E_normalised, pred)
    def raw(self, R, C):
        out = {}
        for j in self.cols:
            a, b = R[0][:, j], C[0][:, j]; a, b = a[np.isfinite(a)], b[np.isfinite(b)]
            if len(a) >= 10 and len(b) >= 5: out["feat:" + ALL_FEATURES[j]] = scalar_stats(a, b)
        Re, Ce = R[1], C[1]
        out["emb:centroid"] = np.array([np.linalg.norm(Re.mean(0) - Ce.mean(0))])
        d, _ = NearestNeighbors(n_neighbors=self.k).fit(Re).kneighbors(Ce); out["emb:knn_distance"] = np.array([d[:, -1].mean()])
        m = self.cfg["embedding"]["max_points"]; A, B = Re[:m], Ce[:m]; g = 1 / (2 * self.sigma ** 2)
        def k(X, Y):
            d2 = (X ** 2).sum(1)[:, None] + (Y ** 2).sum(1)[None] - 2 * X @ Y.T
            return np.exp(-g * np.maximum(d2, 0))
        Kaa, Kbb, Kab = k(A, A), k(B, B), k(A, B); na, nb = len(A), len(B)
        out["emb:mmd"] = np.array([(Kaa.sum() - na) / (na * (na - 1)) + (Kbb.sum() - nb) / (nb * (nb - 1)) - 2 * Kab.mean()])
        oa = np.bincount(self.km.predict(Re), minlength=self.km.n_clusters) + 1e-3; ob = np.bincount(self.km.predict(Ce), minlength=self.km.n_clusters) + 1e-3
        out["emb:cluster_occupancy"] = np.array([float(jensenshannon(oa, ob, base=2))])
        if self.has_classes and (C[2] >= 0).sum() >= 5:
            out["task:class_prior"] = np.array([cat_jsd(R[2], C[2], self.ncls)])
        return out

    @staticmethod
    def _floor(name, n): return SCALAR_FLOOR if name.startswith("feat:") else np.full(n, 1e-6)

    def calibrate(self, n):
        if n in self.cache: return self.cache[n]
        key = sha256_json({"n": n, "cfg": [self.cfg["calibration"], self.cfg["embedding"], self.cfg["seed"]], "ref": float(self.ref.S[np.isfinite(self.ref.S)].sum()), "N": self.N})[:16]
        cf = self.cache_dir / f"calibration_n{n}_{key}.json" if self.cache_dir else None
        if cf and cf.exists():                                   # calibration artefact from the clean reference (reproducible, hashed by key)
            j = json.loads(cf.read_text()); cal = {"n": n, "m_null": j["m_null"], "names": j["names"], "scale": {t: np.array(v) for t, v in j["scale"].items()}, "thr": j["thr"],
                                                     "gthr": j["gthr"], "g_median": j["g_median"], "draws": j["draws"]}
            cal["score"] = lambda d, t, _s=cal["scale"]: float(np.mean(d[t] / _s[t])); self.cache[n] = cal; return cal
        D = self.cfg["calibration"]["draws"]; rng = np.random.default_rng(self.cfg["seed"] * 7919 + n); m = min(n, self.N // 2)
        S, Ee, P = self.ref.S, self.Eref, self.ref.pred; draws = []
        for _ in range(D):
            idx = rng.permutation(self.N); c, r = idx[:m], idx[m:]
            draws.append(self.raw((S[r], Ee[r], P[r]), (S[c], Ee[c], P[c])))
        A, B = draws[: D // 2], draws[D // 2:]; names = sorted(set.intersection(*[set(d) for d in draws]))
        scale = {t: np.maximum(np.quantile(np.stack([d[t] for d in A]), 0.95, axis=0), self._floor(t, len(A[0][t]))) for t in names}
        score = lambda d, t: float(np.mean(d[t] / scale[t]))
        thr = {t: max(float(np.quantile([score(d, t) for d in A], 1 - self.cfg["calibration"]["alpha_feature"])), 0.5) for t in names}
        G = np.array([max(score(d, t) / thr[t] for t in names) for d in B])
        cal = {"n": n, "m_null": m, "names": names, "scale": scale, "thr": thr, "gthr": float(np.quantile(G, 1 - self.cfg["calibration"]["alpha_global"])),
               "g_median": float(np.median(G)), "draws": D, "score": score}
        self.cache[n] = cal
        if cf:
            cf.write_text(json.dumps({"m_null": m, "names": names, "scale": {t: v.tolist() for t, v in scale.items()}, "thr": thr, "gthr": cal["gthr"], "g_median": cal["g_median"], "draws": D}))
        return cal

    def evaluate(self, cur):
        """cur: Table. Returns per-test normalised evidence u (>1 = beyond clean null q99) and the global alert."""
        n = len(cur); note = None
        if n > self.N // 2:                                    # keep test comparable to calibration
            idx = np.linspace(0, n - 1, self.N // 2).astype(int); cur = cur.take(idx); note = f"subsampled {n}->{len(idx)}"; n = len(idx)
        cal = self.calibrate(n)
        raw = self.raw((self.ref.S, self.Eref, self.ref.pred), (cur.S, apply_normalizer(cur.E, self.norm), cur.pred))
        tests = {}
        for t in cal["names"]:
            if t not in raw: continue
            u = min(cal["score"](raw, t) / cal["thr"][t], 1e5); tests[t] = {"u": round(float(u), 3), "stats": [round(float(x), 5) for x in raw[t]], "family": test_family(t)}
        G = max((v["u"] for v in tests.values()), default=0.0)
        return {"n": n, "note": note, "tests": tests, "G": round(G, 3), "gthr": round(cal["gthr"], 3), "ratio": round(min(G / cal["gthr"], 999.0), 3),
                "alert": bool(G > cal["gthr"]), "null_median_ratio": round(cal["g_median"] / cal["gthr"], 3)}
