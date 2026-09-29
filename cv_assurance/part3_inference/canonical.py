"""Module 4 - canonical output serialisation + tolerance-aware comparison.
Canonicalisation (fixed decimals, stable ordering, native types) guarantees that equivalent outputs hash identically."""
from __future__ import annotations

import numpy as np

from ..common.hashing import canonical_json, sha256_json

DEC = 4


def build_classification_output(probs_row, config: dict) -> dict:
    p = np.asarray(probs_row, dtype=np.float64)
    dec = int(config.get("probability_decimals", DEC))
    names = config.get("class_names") or []
    nm = lambda i: names[i] if i < len(names) else str(i)
    order = np.argsort(-p, kind="stable")
    k = int(config.get("top_k", 3))
    pos = p[p > 0]
    ent = float(-(pos * np.log(pos)).sum())
    pred, conf = int(order[0]), float(p[order[0]])
    thr = config.get("decision_threshold")
    decision = "REJECT" if (thr is not None and conf < thr) else nm(pred)
    return {"task": "classification", "num_classes": int(len(p)), "pred_class_id": pred, "pred_class_name": nm(pred),
            "confidence": round(conf, dec), "entropy": round(ent, dec), "decision": decision,
            "top_k": [{"class_id": int(i), "class_name": nm(int(i)), "prob": round(float(p[i]), dec)} for i in order[:k]]}


def canonicalize_detections(dets, dec=DEC):
    """Detections: stable sort (score desc, class, box) + rounded coordinates so ordering/format never causes false mismatches."""
    out = [{"class_id": int(d["class_id"]), "score": round(float(d["score"]), dec),
            "box": [round(float(v), 2) for v in d["box"]]} for d in dets]
    return sorted(out, key=lambda d: (-d["score"], d["class_id"], d["box"]))


def output_hash(output: dict) -> str:
    return sha256_json(output)


def check_self_consistency(output: dict, config: dict) -> list:
    """Internal consistency of a stored classification output under the record's OWN config (cheap tamper tell)."""
    issues = []
    if output.get("task") != "classification": return issues
    tk = output.get("top_k") or []
    if tk and tk[0]["class_id"] != output.get("pred_class_id"): issues.append("top1 != pred_class_id")
    if tk and abs(tk[0]["prob"] - output.get("confidence", -1)) > 1e-3: issues.append("top1 prob != confidence")
    thr = config.get("decision_threshold")
    if thr is not None:
        expect = "REJECT" if output.get("confidence", 0) < thr else output.get("pred_class_name")
        if output.get("decision") != expect: issues.append(f"decision '{output.get('decision')}' inconsistent with threshold {thr}")
    if len(tk) != min(int(config.get("top_k", 3)), output.get("num_classes", 10**9)): issues.append("top_k length != config top_k")
    return issues


def _iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy; u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / u if u > 0 else 0.0


def compare_detections(stored, recomputed, iou_thr=0.5, score_tol=0.02) -> dict:
    """Semantic comparison for detection outputs: class agreement, IoU matching, score tolerance, count consistency."""
    used, matched, max_sd = set(), 0, 0.0
    for d in stored:
        best, bj = 0.0, None
        for j, r in enumerate(recomputed):
            if j in used or r["class_id"] != d["class_id"]: continue
            v = _iou(d["box"], r["box"])
            if v > best: best, bj = v, j
        if bj is not None and best >= iou_thr:
            used.add(bj); matched += 1; max_sd = max(max_sd, abs(d["score"] - recomputed[bj]["score"]))
    ok = matched == len(stored) == len(recomputed) and max_sd <= score_tol
    return {"result": "TOLERANCE_MATCH" if ok else "MISMATCH", "stored_count": len(stored), "recomputed_count": len(recomputed),
            "matched": matched, "max_score_diff": round(max_sd, 4)}


def compare_outputs(stored: dict, recomputed: dict, tol: float = 1e-3, threshold=None) -> dict:
    """EXACT (canonical bytes equal) | TOLERANCE_MATCH (same decision, tiny numeric drift) | MISMATCH (details)."""
    if canonical_json(stored) == canonical_json(recomputed):
        return {"result": "EXACT"}
    if stored.get("task") != "classification" or recomputed.get("task") != "classification":
        return {"result": "MISMATCH", "reason": "unsupported/different task"}
    a, b = stored.get("top_k") or [], recomputed.get("top_k") or []
    pred_changed = stored.get("pred_class_id") != recomputed.get("pred_class_id")
    dec_changed = stored.get("decision") != recomputed.get("decision")
    structural = len(a) != len(b) or [x["class_id"] for x in a] != [x["class_id"] for x in b]
    bm = {x["class_id"]: x["prob"] for x in b}
    dmax = max([abs(x["prob"] - bm.get(x["class_id"], 1.0)) for x in a] + [abs(stored.get("confidence", 0) - recomputed.get("confidence", 0))] or [0])
    d = {"pred_changed": pred_changed, "decision_changed": dec_changed, "topk_structure_differs": structural,
         "max_prob_diff": round(dmax, 6), "stored_pred": stored.get("pred_class_name"), "recomputed_pred": recomputed.get("pred_class_name"),
         "stored_decision": stored.get("decision"), "recomputed_decision": recomputed.get("decision")}
    boundary = (dec_changed and not pred_changed and not structural and dmax <= tol and threshold is not None
                and abs(stored.get("confidence", 0) - threshold) <= tol)      # REJECT/accept flip caused only by float noise at the threshold
    if boundary: d["note"] = "decision flip within tolerance of the decision threshold"
    if (pred_changed or dec_changed or structural or dmax > tol) and not boundary:
        d["result"] = "MISMATCH"
    else:
        d["result"] = "TOLERANCE_MATCH"
    return d
