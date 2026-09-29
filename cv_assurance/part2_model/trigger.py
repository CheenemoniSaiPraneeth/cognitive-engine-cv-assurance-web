"""Module 6 - trigger / backdoor candidate testing. Pastes small localized patterns onto a fixed clean battery and
measures attack-success rate (share of images pushed to ONE common class), cross-image repeatability, spatial
consistency across sizes, and excess over the trusted reference model when one is available.
Output is always 'potential trigger candidate' - a functional response, not proof of intent."""
from __future__ import annotations

import numpy as np
from PIL import Image


def make_pattern(name, s):
    yy, xx = np.mgrid[0:s, 0:s]
    P = np.zeros((s, s, 3), np.uint8)
    if name == "white": P[:] = 255
    elif name == "black": P[:] = 0
    elif name in ("checker2", "checker4"):
        k = 2 if name == "checker2" else 4; b = max(1, s // k)
        m = ((yy // b + xx // b) % 2 == 0); P[m] = 0; P[~m] = 255       # top-left black, alternating
    elif name == "stripes_h": P[:] = 255; P[(yy // max(1, s // 4)) % 2 == 0] = 0
    elif name == "stripes_v": P[:] = 255; P[(xx // max(1, s // 4)) % 2 == 0] = 0
    elif name == "red": P[:] = (255, 0, 0)
    elif name == "green": P[:] = (0, 255, 0)
    elif name == "blue": P[:] = (0, 0, 255)
    elif name == "magenta": P[:] = (255, 0, 255)
    elif name == "cross": P[:] = 0; P[s // 2 - 1:s // 2 + 1, :] = 255; P[:, s // 2 - 1:s // 2 + 1] = 255
    elif name == "diag": P[:] = 255; P[np.abs(yy - xx) <= max(1, s // 6)] = 0
    else: raise ValueError(name)
    return P

FAMILY = ["white", "black", "checker2", "checker4", "stripes_h", "stripes_v", "red", "green", "blue", "magenta", "cross", "diag"]
POS_NAME = {"tl": "top-left", "tr": "top-right", "bl": "bottom-left", "br": "bottom-right", "c": "centre"}


def paste(img, pat, pos, frac, margin=0.03):
    H, W = img.shape[:2]; s = max(2, int(round(frac * min(H, W)))); m = int(round(margin * min(H, W)))
    if pat.shape[0] != s or pat.shape[1] != s:
        pat = np.asarray(Image.fromarray(pat).resize((s, s), Image.NEAREST))
    xy = {"tl": (m, m), "tr": (W - s - m, m), "bl": (m, H - s - m), "br": (W - s - m, H - s - m), "c": ((W - s) // 2, (H - s) // 2)}[pos]
    out = img.copy(); out[xy[1]:xy[1] + s, xy[0]:xy[0] + s] = pat
    return out


def paste_at(img, pat, box_norm):
    """Paste a Part-1 supplied patch at its normalised (x0,y0,x1,y1) location."""
    H, W = img.shape[:2]
    x0, y0, x1, y1 = int(box_norm[0] * W), int(box_norm[1] * H), int(np.ceil(box_norm[2] * W)), int(np.ceil(box_norm[3] * H))
    x1, y1 = min(W, max(x1, x0 + 2)), min(H, max(y1, y0 + 2))
    p = np.asarray(Image.fromarray(pat).resize((x1 - x0, y1 - y0), Image.NEAREST))
    out = img.copy(); out[y0:y1, x0:x1] = p
    return out


def _asr(base_pred, new_P):
    """Return (asr, target_class, mean_conf_on_hit). ASR = share of NOT-already-target images pushed to the modal new class."""
    new = new_P.argmax(1)
    changed = new != base_pred
    if changed.sum() == 0: return 0.0, None, 0.0
    tgt = int(np.bincount(new[changed]).argmax())
    eligible = base_pred != tgt
    if eligible.sum() == 0: return 0.0, tgt, 0.0
    hit = (new == tgt) & eligible
    return float(hit.sum() / eligible.sum()), tgt, float(new_P[hit, tgt].mean()) if hit.any() else 0.0


def _scan(adapter, images, base_pred, cands):
    res = {}
    for key, fn in cands:
        P = adapter.predict([fn(im) for im in images])
        res[key] = _asr(base_pred, P)
    return res


def run_trigger_tests(adapter, ref_adapter, images, cfg, store, model_id, part1_candidates=None, class_names=None):
    tc = cfg["trigger"]
    base = adapter.predict(images).argmax(1)
    ref_base = ref_adapter.predict(images).argmax(1) if ref_adapter is not None else None
    cands = []
    for pn in FAMILY:
        for pos in tc["positions"]:
            for fr in tc["sizes"]:
                cands.append(((pn, pos, fr), (lambda im, pn=pn, pos=pos, fr=fr: paste(im, make_pattern(pn, max(2, int(round(fr * min(im.shape[:2]))))), pos, fr, tc["margin"]))))
    r = _scan(adapter, images, base, cands)
    rr = _scan(ref_adapter, images, ref_base, cands) if ref_adapter is not None else None
    asrs = np.array([v[0] for v in r.values()])
    med = float(np.median(asrs))
    hits = {}
    for (pn, pos, fr), (asr, tgt, conf) in r.items():
        ref_asr = rr[(pn, pos, fr)][0] if rr else None
        ok = asr >= tc["min_asr"] and (asr - ref_asr >= tc["ref_excess"] if rr else asr - med >= tc["noref_outlier"])
        if ok: hits.setdefault((pn, pos), []).append((fr, asr, tgt, conf, ref_asr))
    cn = class_names or {}
    n_found = 0
    for (pn, pos), lst in hits.items():
        best = max(lst, key=lambda t: t[1])
        n_size = len(lst)
        if n_size < tc["min_size_hits"] and best[1] < tc["quarantine_asr"]: continue
        strong = best[1] >= tc["quarantine_asr"] and n_size >= tc["min_size_hits"] and rr is not None
        ev = {"pattern": pn, "position": POS_NAME[pos], "origin": "generic_library", "sizes_hit_fraction": [t[0] for t in lst],
              "attack_success_rate_best": round(best[1], 3), "target_class": best[2], "target_class_name": cn.get(best[2], str(best[2])),
              "mean_target_confidence": round(best[3], 3), "n_images": len(images),
              "reference_model_asr": None if best[4] is None else round(best[4], 3),
              "reference_mode": "vs_trusted_reference_model" if rr else "reference_free_library_outlier",
              "library_median_asr": round(med, 3)}
        store.add("model", model_id, "trigger_candidate_response", "critical" if strong else "high", 0.6 + 0.3 * strong + 0.1 * (rr is not None),
                  "trigger_pattern_scan_v1", ev, "QUARANTINE" if strong else "REVIEW",
                  ["Potential trigger candidate: repeated abnormal response to a localized pattern; not proof of intentional backdoor.",
                   "Only the tested pattern family/positions/sizes are covered; untested triggers can be missed."],
                  reference_assets=["trusted_reference_model"] if rr else [])
        n_found += 1

    # -- Part 1 supplied candidates (benchmark-provided / dataset-derived trigger candidates)
    p1_res = []
    for c in (part1_candidates or []):
        fn = lambda im, c=c: paste_at(im, c["patch"], c["box_norm"])
        (asr, tgt, conf), = _scan(adapter, images, base, [("p1", fn)]).values()
        ref_asr = _scan(ref_adapter, images, ref_base, [("p1", fn)])["p1"][0] if ref_adapter is not None else None
        ok = asr >= tc["min_asr"] and (asr - ref_asr >= tc["ref_excess"] if ref_asr is not None else asr - med >= tc["noref_outlier"])
        row = {"part1_candidate": c["id"], "attack_success_rate": round(asr, 3), "target_class": tgt,
               "target_class_name": cn.get(tgt, str(tgt)), "reference_model_asr": None if ref_asr is None else round(ref_asr, 3),
               "confirmed_functional_response": bool(ok), "part1_label_correlation": c.get("label"), "n_images": len(images)}
        p1_res.append(row)
        if ok:
            store.add("model", model_id, "trigger_candidate_response", "critical", 0.95, "trigger_pattern_scan_v1",
                      {**row, "origin": "part1_dataset_candidate", "location_norm_xyxy": c["box_norm"], "position": "part1 location"},
                      "QUARANTINE",
                      ["Dataset-derived pattern reproducibly steers this model: converging Part 1 + Part 2 evidence.",
                       "Still a potential trigger candidate; intent is not established."], reference_assets=[c["id"]])
            n_found += 1
    return {"status": "OK", "candidates_tested": len(cands), "library_median_asr": round(med, 3),
            "hits": n_found, "part1_candidate_results": p1_res,
            "mode": "reference_model_contrast" if rr else "reference_free"}, "OK"
