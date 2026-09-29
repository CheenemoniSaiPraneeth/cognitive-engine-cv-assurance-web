"""Module 7 - shift attribution: WHICH families/features moved, in which direction, and which source/camera contributes."""
from __future__ import annotations
from collections import Counter, defaultdict
import numpy as np
from .features import ALL_FEATURES


def attribute(ev, ref, cur, ood, cfg):
    fam = defaultdict(lambda: {"score": 0.0, "tests": []})
    for t, v in ev["tests"].items():
        if v["u"] > 1.0:
            f = fam[v["family"]]; f["score"] = max(f["score"], v["u"]); f["tests"].append(t)
    fams = [{"family": k, "score": round(v["score"], 2), "tests": sorted(v["tests"], key=lambda t: -ev["tests"][t]["u"])} for k, v in fam.items()]
    fams.sort(key=lambda d: -d["score"])
    top = fams[0]["score"] if fams else 0.0
    for f in fams: f["primary"] = bool(np.log10(1 + min(f["score"], 100)) >= 0.6 * np.log10(1 + min(top, 100)))   # weaker families are reported as secondary (spill-over), log scale
    feats = []
    for t, v in sorted(ev["tests"].items(), key=lambda kv: -kv[1]["u"]):
        if v["u"] <= 1.0 or not t.startswith("feat:"): continue
        j = ALL_FEATURES.index(t[5:]); a, b = ref.S[:, j], cur.S[:, j]; a, b = a[np.isfinite(a)], b[np.isfinite(b)]
        sd = max(float(a.std()), 0.02 * abs(float(a.mean())), 1e-6)
        feats.append({"feature": t[5:], "u": v["u"], "ref_mean": round(float(a.mean()), 4), "cur_mean": round(float(b.mean()), 4),
                      "effect_size": round(float((b.mean() - a.mean()) / sd), 2), "direction": "up" if b.mean() > a.mean() else "down"})
    out = {"all_flagged_families": [f["family"] for f in fams], "primary_families": [f["family"] for f in fams if f["primary"]], "families": fams, "features": feats[:8]}
    fl = [t for t, v in ev["tests"].items() if v["u"] > 1.0 and t.startswith("feat:")]
    if fl and cur.items:                                       # per-sample abnormality on flagged features -> per-source concentration
        cols = [ALL_FEATURES.index(t[5:]) for t in fl]; med = np.nanmedian(ref.S[:, cols], 0); sd = np.maximum(np.nanstd(ref.S[:, cols], 0), 1e-6 + 0.02 * np.abs(med))
        z = np.nanmean(np.abs((cur.S[:, cols] - med) / sd), axis=1); abn = z > cfg["attribution"]["abnormal_z"]
    else:
        abn = np.zeros(len(cur), bool)
    if ood is not None: abn = abn | ood["flag"]
    ref_src = Counter(it.get("source") for it in ref.items if it.get("source")); by = defaultdict(lambda: [0, 0])
    for it, a in zip(cur.items, abn):
        s = it.get("source")
        if s: by[s][0] += 1; by[s][1] += int(a)
    if by:
        out["sources"] = sorted([{"source": s, "n": v[0], "abnormal": v[1], "abnormal_rate": round(v[1] / v[0], 3), "new_source": s not in ref_src} for s, v in by.items()], key=lambda d: -d["abnormal_rate"])
    else:
        out["sources"] = []; out["source_note"] = "UNAVAILABLE: no source/camera metadata on operational samples"
    return out
