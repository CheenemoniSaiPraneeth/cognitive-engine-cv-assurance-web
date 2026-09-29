"""Module 7 - Trigger / backdoor CANDIDATE analysis (evidence generation, no training).

Idea: a trigger is a small localised pattern that recurs at (nearly) the same image position
across otherwise unrelated samples. For every window position we hash a low-resolution,
mean-normalised copy of the patch. A (position, hash) bucket is suspicious when
  * it is supported by many samples,
  * the same hash is rare at *other* positions (position-specific, not a generic texture),
  * the raw pixels of its members are consistent (low pixel std),
  * and (supporting evidence) members concentrate on one label / source.
Near-duplicate images must be collapsed BEFORE calling this (recurrence must be across
unrelated samples). This module never claims a functional backdoor is proven.
"""
from __future__ import annotations

import math

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy.stats import binomtest

DETECTOR = "trigger_candidate_patch_recurrence_v1"


def _patch_hashes(g96: np.ndarray, patch: int, stride: int, min_std: float):
    w = sliding_window_view(g96.astype(np.float32), (patch, patch))[::stride, ::stride]
    ny, nx = w.shape[:2]
    std = w.reshape(ny, nx, -1).std(-1)
    f = patch // 4
    pooled = w.reshape(ny, nx, 4, f, 4, f).mean(axis=(3, 5)).reshape(ny, nx, 16)
    bits = pooled > np.median(pooled, axis=-1, keepdims=True)
    h = (bits * (1 << np.arange(16))).sum(-1).astype(np.int32)
    return h.ravel(), (std >= min_std).ravel(), ny, nx


def _conc(values, base_counts, n_base, members):
    vals = [values[i] for i in members if values[i] is not None]
    if not vals:
        return None
    top, k = max(((v, vals.count(v)) for v in set(vals)), key=lambda t: t[1])
    base = base_counts.get(top, 0) / max(n_base, 1)
    p = binomtest(k, len(vals), min(max(base, 1e-6), 1 - 1e-6), alternative="greater").pvalue
    return {"value": top, "fraction": round(k / len(vals), 3), "base_rate": round(base, 3), "p_value": float(p)}


def detect_trigger_candidates(subset, gray96, cfg, store, flags, class_names, all_samples=None):
    """subset: representative (de-duplicated) readable samples aligned with gray96 (m,96,96)."""
    c = cfg["trigger"]
    m = len(subset)
    min_support = max(c["min_support"], math.ceil(c["min_support_fraction"] * m))
    if m < 2 * min_support:
        return {"status": "UNAVAILABLE", "reason": "too few unrelated samples"}
    H, V = [], []
    for g in gray96:
        h, v, ny, nx = _patch_hashes(g, c["patch"], c["stride"], c["min_patch_std"])
        H.append(h)
        V.append(v)
    H, V = np.stack(H), np.stack(V)
    P = H.shape[1]
    allh, allc = np.unique(H[V], return_counts=True)
    total = dict(zip(allh.tolist(), allc.tolist()))
    cands = []
    for p in range(P):
        v = V[:, p]
        if v.sum() < min_support:
            continue
        vals, cnt = np.unique(H[v, p], return_counts=True)
        for val, cn in zip(vals[cnt >= min_support], cnt[cnt >= min_support]):
            bg = (total[int(val)] - cn) / max(P - 1, 1)
            ratio = cn / (bg + 1.0)
            if ratio < c["min_position_ratio"]:
                continue
            members = np.where(v & (H[:, p] == val))[0]
            py, px = divmod(p, nx)
            y0, x0 = py * c["stride"], px * c["stride"]
            pix_std = float(gray96[members, y0:y0 + c["patch"], x0:x0 + c["patch"]].astype(np.float32).std(0).mean())
            if pix_std > c["max_pixel_std"]:
                continue
            cands.append({"members": set(members.tolist()), "support": int(cn), "ratio": float(ratio),
                          "pix_std": pix_std, "box": [x0, y0, x0 + c["patch"], y0 + c["patch"]]})
    cands.sort(key=lambda d: (-d["support"], d["pix_std"]))
    merged = []
    for cd in cands:
        for mg in merged:
            inter = len(cd["members"] & mg["members"])
            if inter / max(len(cd["members"] | mg["members"]), 1) >= c["merge_jaccard"]:
                b, o = mg["box"], cd["box"]
                mg["box"] = [min(b[0], o[0]), min(b[1], o[1]), max(b[2], o[2]), max(b[3], o[3])]
                mg["ratio"] = max(mg["ratio"], cd["ratio"])
                break
        else:
            merged.append(cd)
    merged = merged[: c["max_candidates"]]

    labs = [s.primary_label for s in subset]
    srcs = [s.source_id or s.contributor_id for s in subset]
    pop = all_samples if all_samples is not None else subset
    lab_base = {}
    for s in pop:
        if s.primary_label is not None:
            lab_base[s.primary_label] = lab_base.get(s.primary_label, 0) + 1
    n_lab = sum(lab_base.values())
    src_base = {}
    for s in pop:
        k = s.source_id or s.contributor_id
        if k:
            src_base[k] = src_base.get(k, 0) + 1
    n_src = sum(src_base.values())
    n_out = 0
    for i, cd in enumerate(merged, 1):
        mem = sorted(cd["members"])
        lc = _conc(labs, lab_base, n_lab, mem)
        sc = _conc(srcs, src_base, n_src, mem)
        lab_sig = bool(lc and lc["fraction"] >= 0.8 and lc["p_value"] < 1e-3)
        src_sig = bool(sc and sc["fraction"] >= 0.8 and sc["p_value"] < 1e-3)
        sev = "high" if (lab_sig and src_sig) else "medium" if (lab_sig or src_sig) else "low"
        conf = 0.4 + 0.2 * lab_sig + 0.2 * src_sig + 0.2 * (cd["ratio"] >= 50)
        ids = [subset[j].sample_id for j in mem]
        x0, y0, x1, y1 = cd["box"]
        ev = {"support": len(mem), "support_fraction": round(len(mem) / m, 4),
              "location_norm_xyxy": [round(x0 / 96, 3), round(y0 / 96, 3), round(x1 / 96, 3), round(y1 / 96, 3)],
              "position_specificity_ratio": round(cd["ratio"], 1),
              "pixel_consistency_std": round(cd["pix_std"], 2),
              "label_correlation": ({**lc, "value": class_names.get(lc["value"], str(lc["value"]))} if lc else None),
              "source_correlation": sc, "label_concentrated": lab_sig, "source_concentrated": src_sig,
              "member_sample_ids": ids[:200], "member_ids_truncated": len(ids) > 200}
        store.add("pattern", f"T-{i:03d}", "trigger_candidate", sev, conf, DETECTOR, ev, "REVIEW",
                  ["Potential trigger candidate only; dataset evidence cannot establish a functional backdoor.",
                   "Low-opacity, blended, or input-dependent triggers may be missed."])
        for sid in ids:
            flags.setdefault(sid, set()).add("trigger_candidate")
        n_out += 1
    return {"status": "OK", "candidates": n_out, "assessed": m, "min_support": min_support}
