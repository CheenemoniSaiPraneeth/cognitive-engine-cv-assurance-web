"""Part 4 orchestrator + Module 8 (shift assurance). NORMAL / SHIFT DETECTED / SEVERE SHIFT -> ACCEPT / REVIEW / REVIEW(high).
A shift is NOT an attack: governance (Part 5 / analyst) decides whether it is acceptable (new camera, season, ...)."""
from __future__ import annotations
import time
from collections import OrderedDict
from pathlib import Path
import numpy as np
from ..common.audit import AuditLog
from ..common.hashing import sha256_json
from ..common.schemas import Finding, now_iso
from . import DETECTOR_VERSION, attribution, temporal
from .data import build_table, load_reference
from .engine import ShiftEngine
from .ood import OODModel
from .report import write_reports

LIMITATIONS = [
    "Distribution shift is not evidence of an attack: a new camera, season, location or lighting condition produces the same signal.",
    "Thresholds are calibrated on the trusted reference only; a reference that is unrepresentative gives unrepresentative thresholds.",
    "Small batches have low power; each batch is calibrated at its own size but a single-batch alert has a nonzero false-alarm rate (alpha_global).",
    "Default handcrafted embeddings are less semantic than a frozen pretrained encoder (embedding.backend=onnx).",
    "Task-output shift needs model outputs (black-box model or verified Part-3 ledger); otherwise it is reported UNAVAILABLE.",
]


class FS:
    def __init__(self): self.f = []
    def add(self, asset_type, asset_id, category, severity, conf, detector, evidence, disposition, limits=None, refs=None):
        x = Finding(f"S-{len(self.f) + 1:05d}", asset_type, str(asset_id), category, severity, round(float(conf), 3), detector, DETECTOR_VERSION,
                    evidence, refs or [], now_iso(), disposition, limits or []); self.f.append(x); return x


def _conf(ratio): return 0.5 if ratio <= 1 else min(0.99, 1 - 0.5 / ratio)


def run_part4(ref_dir, ops_items, cfg, out_dir, adapter=None, outputs=None, encoder=None, links=None, log=print):
    t0 = time.time(); out = Path(out_dir); out.mkdir(parents=True, exist_ok=True); audit = AuditLog(out / "audit_log_part4.jsonl"); ch = sha256_json(cfg)
    ref, man = load_reference(ref_dir); rid = man["reference_id"]; unavail = []; fs = FS()
    audit.append("load_reference", [rid], [man["manifest_sha256"][:16]], ch, actor="part4_pipeline")
    cur, bad = build_table(ops_items, cfg, encoder, adapter, outputs)
    audit.append("extract_operational_features", [f"items={len(ops_items)}"], [f"ok={len(cur)} unreadable={len(bad)}"], ch, actor="part4_pipeline")
    for i, e in bad: fs.add("sample", i, "unreadable_operational_image", "medium", 0.99, "p4.ingest", {"error": e}, "REVIEW")
    if len(cur) < cfg["min_batch"]: raise ValueError(f"only {len(cur)} readable operational samples; need >= {cfg['min_batch']}")
    log(f"[p4 1-2] reference {rid} n={len(ref)}; operational n={len(cur)}")
    eng = ShiftEngine(ref, cfg, cache_dir=ref_dir); ood = OODModel(eng, cfg)
    if not np.isfinite(cur.S[:, -1]).any(): unavail.append("task_output/class_prior shift: no model outputs supplied (pass a model or a verified ledger) - NOT a PASS")
    if not eng.has_classes and np.isfinite(cur.S[:, -1]).any(): unavail.append("class_prior shift: reference has no model predictions")
    if not any(it.get("source") for it in cur.items): unavail.append("source/camera attribution: no source metadata on operational samples")
    if not any(it.get("source") for it in ref.items): unavail.append("new-source detection: reference has no source metadata")
    # ---- batches
    groups = OrderedDict()
    for i, it in enumerate(cur.items): groups.setdefault(it.get("batch_id") or "ALL", []).append(i)
    rows, small = [], []
    for bid, idx in groups.items():
        if len(idx) < cfg["min_batch"]: small.append(bid); continue
        tb = cur.take(idx); ev = eng.evaluate(tb); od = ood.assess(tb); at = attribution.attribute(ev, ref, tb, od, cfg)
        fams = [f["family"] for f in at["families"]]
        rows.append({"batch_id": bid, "n": len(idx), "ratio": ev["ratio"], "G": ev["G"], "gthr": ev["gthr"], "alert": ev["alert"], "note": ev["note"],
                     "null_median_ratio": ev["null_median_ratio"], "families": fams if ev["alert"] else [], "ood_rate": od["rate"], "ood_p": od["binomial_p"],
                     "new_clusters": od["new_clusters"], "tests": ev["tests"], "attribution": at, "_od": od, "_idx": idx})
        log(f"[p4 3-6] batch {bid}: n={len(idx)} ratio={ev['ratio']} {'ALERT' if ev['alert'] else 'ok'} families={fams if ev['alert'] else []} ood={od['rate']}")
    if small: unavail.append(f"batches {small} skipped: fewer than {cfg['min_batch']} samples (not enough for calibrated tests)")
    if not rows: raise ValueError("no batch large enough for calibrated tests")
    tm = temporal.analyse(rows, cfg)
    if tm["status"] != "OK": unavail.append("temporal drift: " + tm["reason"])
    # ---- overall window
    if len(rows) == 1: overall = rows[0]; ov_od = overall["_od"]; ov_attr = overall["attribution"]
    else:
        ev = eng.evaluate(cur); ov_od = ood.assess(cur); ov_attr = attribution.attribute(ev, ref, cur, ov_od, cfg)
        overall = {"batch_id": "OVERALL", "n": ev["n"], "ratio": ev["ratio"], "G": ev["G"], "gthr": ev["gthr"], "alert": ev["alert"], "note": ev["note"],
                   "families": [f["family"] for f in ov_attr["families"]] if ev["alert"] else [], "ood_rate": ov_od["rate"], "ood_p": ov_od["binomial_p"], "tests": ev["tests"]}
    # ---- Module 8 assurance level
    anyalert = overall["alert"] or any(r["alert"] for r in rows) or tm.get("onset_index") is not None
    best = max(rows + [overall], key=lambda r: r["ratio"]); nf = len(best["families"])
    emb_hit = any(v["u"] > 1 for t, v in best["tests"].items() if t.startswith("emb:"))
    if not anyalert: level, disp = "NORMAL", "ACCEPT"
    elif best["ratio"] >= cfg["severity"]["severe_ratio"] and (nf >= cfg["severity"]["severe_min_families"] or emb_hit): level, disp = "SEVERE SHIFT", "REVIEW"
    else: level, disp = "SHIFT DETECTED", "REVIEW"
    sev = {"NORMAL": "low", "SHIFT DETECTED": "medium", "SEVERE SHIFT": "high"}[level]
    if level != "NORMAL":
        at = ov_attr if overall["alert"] else best["attribution"]
        fs.add("window", "operational_window", "distribution_shift", sev, _conf(best["ratio"]), "p4.shift_engine",
               {"level": level, "worst_batch": best["batch_id"], "ratio": best["ratio"], "families": [f["family"] for f in at["families"]], "top_features": at["features"][:4],
                "reference_id": rid, "interpretation": "operational population differs from trusted reference; cause requires analyst judgement"}, disp, LIMITATIONS[:2], [rid])
        if tm.get("onset_index") is not None or tm.get("pattern") in ("transient", "gradual_subthreshold"):
            fs.add("window", "operational_window", "drift_onset", "medium", 0.8, "p4.temporal",
                   {k: tm[k] for k in ("pattern", "onset_batch", "drift_start_index", "cusum_alert_batch", "affected_batches", "affected_families")}, "REVIEW", ["Onset resolution is limited to batch granularity."], [rid])
        for r in rows:
            if r["alert"]:
                for fam in r["attribution"]["families"][:3]:
                    fs.add("batch", r["batch_id"], f"{fam['family']}_shift", "medium" if r["ratio"] < cfg["severity"]["severe_ratio"] else "high", _conf(r["ratio"]), "p4.attribution",
                           {"score": fam["score"], "tests": fam["tests"][:5], "features": [f for f in r["attribution"]["features"] if f["feature"] in [t[5:] for t in fam["tests"]]][:4]}, "REVIEW", ["Family-level attribution; not a causal proof."], [rid])
                for s in r["attribution"].get("sources", [])[:3]:
                    if s["new_source"] or s["abnormal_rate"] >= 0.5:
                        fs.add("batch", r["batch_id"], "source_concentration" if not s["new_source"] else "new_source", "medium", 0.7, "p4.attribution", s, "REVIEW", ["Source metadata is self-reported by the operational system."], [rid])
    for r in rows:                                                     # OOD samples (evidence, capped)
        od = r["_od"]
        if od["binomial_p"] < 0.01 and od["flagged"] >= 3:
            fs.add("batch", r["batch_id"], "ood_excess", "medium", min(0.99, 1 - od["binomial_p"]), "p4.ood", {"flagged": od["flagged"], "n": od["n"], "rate": od["rate"], "reference_rate": od["ref_rate"], "binomial_p": od["binomial_p"]}, "REVIEW", ["OOD = distributional difference, not intent."], [rid])
        for c in od["new_clusters"]: fs.add("batch", r["batch_id"], "new_cluster", "medium", 0.75, "p4.ood", c, "REVIEW", ["A new cluster may be a legitimate new object/scene."], [rid])
    flagged = {}; cnt = 0
    for r in rows:
        od = r["_od"]; order = np.argsort(-od["knn"])
        for j in order:
            if od["flag"][j]:
                it = cur.items[r["_idx"][j]]; flagged[it["id"]] = ["ood"]
                if cnt < 25 and od["binomial_p"] < 0.05:
                    fs.add("sample", it["id"], "ood_sample", "low", 0.6, "p4.ood", {"batch": r["batch_id"], "knn_distance": round(float(od["knn"][j]), 4), "record_id": it.get("record_id")}, "REVIEW", ["Rare legitimate samples can look OOD."], [rid]); cnt += 1
    if not fs.f or level == "NORMAL": pass
    for r in rows: r.pop("_od"); idx = r.pop("_idx"); r["ids"] = [cur.items[i]["id"] for i in idx][:5]
    res = {"schema_version": "1.0", "part": 4, "created_at": now_iso(), "detector_version": DETECTOR_VERSION, "level": level, "disposition": disp,
           "reference": {"reference_id": rid, "n": len(ref), "manifest_sha256": man["manifest_sha256"]}, "operational": {"n": len(cur), "batches": len(rows), "unreadable": len(bad)},
           "overall": {k: v for k, v in overall.items() if k not in ("_od", "_idx", "attribution")}, "overall_attribution": ov_attr,
           "ood_overall": {k: v for k, v in ov_od.items() if k not in ("flag", "knn")}, "batches": rows, "temporal": tm, "findings": [f.to_dict() for f in fs.f],
           "finding_counts": _counts(fs.f), "part4_flags": flagged, "unavailable_tests": unavail, "limitations": LIMITATIONS,
           "config": cfg, "config_sha256": ch, "links": links or {}, "runtime_s": round(time.time() - t0, 1),
           "note": "A shift is evidence to review, not a statement of intent. Governance stays with the analyst."}
    audit.append("shift_assessment", [rid], [f"level={level}", f"findings={len(fs.f)}"], ch, result=disp, actor="part4_pipeline")
    write_reports(res, out); return res


def _counts(fl):
    c = {}
    for f in fl: c[f.category] = c.get(f.category, 0) + 1
    return c
