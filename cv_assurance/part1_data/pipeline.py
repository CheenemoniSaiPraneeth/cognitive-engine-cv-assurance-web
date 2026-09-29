"""Part 1 orchestrator: ingest -> profile -> fingerprint -> duplicates -> labels -> OOD -> trigger -> sources -> report."""
from __future__ import annotations

import platform
import time
from collections import Counter
from pathlib import Path

import numpy as np
import scipy
import sklearn

from ..common.audit import AuditLog
from ..common.config import load_config
from ..common.hashing import sha256_file, sha256_json
from ..common.schemas import DETECTOR_VERSION, FindingStore, now_iso
from ..dataset.ingest import IMG_EXT, load_dataset
from ..reports.report import write_reports
from . import duplicates, label_integrity, ood, source_aggregation, trigger
from .ood import robust_standardize, stats_matrix
from .features import apply_normalizer, build_encoder, fit_normalizer
from .fingerprint import analyze_paths, hex64
from .profiling import profile_dataset

LIMITATIONS = [
    "Rare legitimate samples can resemble anomalies.",
    "OOD detection identifies distributional difference, not malicious intent.",
    "Similarity-based label analysis can be wrong for visually ambiguous classes.",
    "Dataset-only trigger evidence cannot establish a functional model backdoor.",
    "Thresholds should be calibrated on clean reference conditions and are recorded in this report.",
    "Default handcrafted embeddings are less semantic than a frozen pretrained encoder (embedding.backend=onnx).",
]


def run_part1(dataset, out_dir, fmt="auto", reference=None, config=None, ann_file=None,
              images_dir=None, labels_dir=None, log=print) -> dict:
    t_start = time.time()
    cfg = load_config(config) if not isinstance(config, dict) else config
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    audit = AuditLog(out / "audit_log.jsonl")
    cfg_hash = sha256_json(cfg)
    store = FindingStore()
    flags: dict = {}
    unavailable, module_status, timings = [], {}, {}

    # 1 ingestion
    t = time.time()
    b = load_dataset(dataset, fmt, ann_file=ann_file, images_dir=images_dir, labels_dir=labels_dir)
    samples, class_names = b.samples, b.class_names
    log(f"[1/8] ingested {len(samples)} samples ({b.fmt}), {len(b.issues)} structural issues")
    audit.append("ingest", [str(dataset)], [f"samples={len(samples)}"], cfg_hash)
    timings["ingest"] = time.time() - t

    # 3 fingerprinting (also decodes every image -> corrupt files surface here)
    t = time.time()
    encoder = build_encoder(cfg)
    res = analyze_paths([s.image_path for s in samples], encoder, cfg["workers"])
    stats = []
    for s, r in zip(samples, res):
        if r["error"]:
            if s.readable:  # not already flagged as missing
                s.readable = False
                b.issues.append({"kind": "unreadable_image", "severity": "high", "file_name": s.file_name,
                                 "sample_id": s.sample_id, "detail": r["error"]})
            stats.append(None)
            continue
        s.image_sha256, s.width, s.height, s.format = r["sha256"], r["width"], r["height"], r["format"]
        s.perceptual_hash, s.difference_hash = hex64(r["phash"]), hex64(r["dhash"])
        stats.append(r["stats"])
    log("[2/8] fingerprinted images (SHA-256, pHash, dHash, features)")
    audit.append("fingerprint", [], ["sha256/phash/dhash/features"], cfg_hash)
    timings["fingerprint"] = time.time() - t

    # 2 profiling
    prof = profile_dataset(samples, class_names, stats, b.issues)
    audit.append("profile", [], ["profile"], cfg_hash)

    # structural findings
    for iss in b.issues:
        if iss["severity"] == "low":
            continue
        aid = iss["sample_id"] or "dataset"
        store.add("sample" if iss["sample_id"] else "dataset", aid, "structural_error", iss["severity"], 0.99,
                  "ingestion_validator_v1", {"kind": iss["kind"], "file_name": iss["file_name"], "detail": iss["detail"]},
                  "REVIEW", ["Structural problems can be benign export artefacts."])
        if iss["sample_id"]:
            flags.setdefault(iss["sample_id"], set()).add("structural_error")

    valid = [(s, r) for s, r in zip(samples, res) if s.readable and not r["error"]]
    if len(valid) < 10:
        unavailable.append("similarity-based detectors: fewer than 10 readable images")
        return _finish(b, cfg, cfg_hash, store, flags, prof, [], {}, unavailable, module_status, timings,
                       t_start, out, audit, samples, reference, encoder, log)
    vs = [s for s, _ in valid]
    Eraw = np.stack([r["feat"] for _, r in valid])
    G96 = np.stack([r["gray96"] for _, r in valid])
    Sraw = stats_matrix([r["stats"] for _, r in valid])
    ref_E = ref_S = ref_ids = None
    if reference:
        rpaths = sorted(str(p) for p in Path(reference).rglob("*") if p.suffix.lower() in IMG_EXT)
        rall = analyze_paths(rpaths, encoder, cfg["workers"])
        rok = [(p, r) for p, r in zip(rpaths, rall) if not r["error"]]
        if rok:
            ref_raw = np.stack([r["feat"] for _, r in rok])
            ref_ids = [f"ref:{Path(p).name}" for p, _ in rok]
            ref_Sraw = stats_matrix([r["stats"] for _, r in rok])
            ref_S = robust_standardize(ref_Sraw)
            norm = fit_normalizer(ref_raw)
            ref_E = apply_normalizer(ref_raw, norm)
            E = apply_normalizer(Eraw, norm)
            S = robust_standardize(Sraw, ref_Sraw)
        else:
            reference = None
    if ref_E is None:
        norm = fit_normalizer(Eraw)
        E = apply_normalizer(Eraw, norm)
        S = robust_standardize(Sraw)
        unavailable.append("trusted-reference OOD calibration: no trusted reference supplied "
                           "(dataset used as its own reference - self-reference mode)")
    for i, s in enumerate(vs):
        s.embedding_ref = i

    # 4 duplicates / flooding
    t = time.time()
    groups, excl = duplicates.detect_duplicates(vs, E, cfg, store, flags, class_names)
    module_status["duplicates"] = {"status": "OK", "groups": len(groups)}
    log(f"[3/8] duplicate groups: {len(groups)}")
    audit.append("duplicate_flooding", [], [f"groups={len(groups)}"], cfg_hash)
    timings["duplicates"] = time.time() - t

    # 5 label integrity
    t = time.time()
    lab_idx = [i for i, s in enumerate(vs) if s.primary_label is not None]
    if len(lab_idx) >= 10:
        st = label_integrity.detect_label_anomalies([vs[i] for i in lab_idx], E[lab_idx], excl, class_names,
                                                    cfg, store, flags, cfg["seed"])
    else:
        st = {"status": "UNAVAILABLE", "reason": "fewer than 10 labelled samples"}
    module_status["label_integrity"] = st
    if st["status"] != "OK":
        unavailable.append(f"label integrity: {st.get('reason')}")
    log(f"[4/8] label integrity: {st}")
    audit.append("label_integrity", [], [str(st)], cfg_hash)
    timings["label_integrity"] = time.time() - t

    # 6 OOD
    t = time.time()
    st = ood.detect_ood(vs, E, S, cfg, store, flags, ref_E, ref_S, ref_ids, cfg["seed"])
    module_status["ood"] = st
    if st["status"] != "OK":
        unavailable.append(f"OOD: {st.get('reason')}")
    log(f"[5/8] OOD: {st}")
    audit.append("ood_anomaly", [], [str(st)], cfg_hash)
    timings["ood"] = time.time() - t

    # 7 trigger candidates (on de-duplicated representatives so recurrence is across unrelated samples)
    t = time.time()
    drop = set()
    idx_of = {s.sample_id: i for i, s in enumerate(vs)}
    for g in groups:
        if g["level"] in ("exact", "near"):
            for sid in g["members"][1:]:
                drop.add(idx_of[sid])
    reps = [i for i in range(len(vs)) if i not in drop]
    st = trigger.detect_trigger_candidates([vs[i] for i in reps], G96[reps], cfg, store, flags,
                                           class_names, all_samples=[vs[i] for i in reps])
    module_status["trigger"] = st
    if st["status"] != "OK":
        unavailable.append(f"trigger candidates: {st.get('reason')}")
    log(f"[6/8] trigger candidates: {st}")
    audit.append("trigger_candidate", [], [str(st)], cfg_hash)
    timings["trigger"] = time.time() - t

    return _finish(b, cfg, cfg_hash, store, flags, prof, groups, module_status, unavailable, module_status,
                   timings, t_start, out, audit, samples, reference, encoder, log)


def _finish(b, cfg, cfg_hash, store, flags, prof, groups, mod_status, unavailable, module_status, timings,
            t_start, out, audit, samples, reference, encoder, log):
    # converging signals at sample level
    for s in samples:
        cats = flags.get(s.sample_id, set()) - {"structural_error"}
        sig = {("duplicate" if c.startswith("duplicate") or c == "conflicting_duplicate_labels" else c) for c in cats}
        if len(sig) >= 2:
            store.add("sample", s.sample_id, "converging_signals", "high" if len(sig) >= 3 else "medium",
                      0.5 + 0.15 * len(sig), "cross_signal_v1", {"signals": sorted(sig), "file_name": s.file_name},
                      "QUARANTINE" if len(sig) >= 3 else "REVIEW",
                      ["Multiple independent detectors point at the same sample."])
            flags[s.sample_id].add("converging_signals")
    # 8 source aggregation
    has_src = any(s.source_id or s.contributor_id or s.batch_id for s in samples)
    if has_src:
        src = source_aggregation.aggregate_sources(samples, flags, cfg, store)
    else:
        src = {}
        unavailable.append("source/contributor aggregation: no source_id/contributor_id/batch_id metadata")
    log(f"[7/8] source aggregation: {len(src)} groups")
    audit.append("source_aggregation", [], [f"groups={len(src)}"], cfg_hash)

    disp = "ACCEPT"
    if any(f.disposition == "REVIEW" for f in store.findings):
        disp = "REVIEW"
    if any(f.disposition == "QUARANTINE" for f in store.findings):
        disp = "QUARANTINE"

    ann_hash = sha256_json([[s.file_name, [(a.class_id, a.bbox) for a in s.annotations]] for s in samples])
    ds_sha = sha256_json({"images": sorted((s.file_name, s.image_sha256) for s in samples), "annotations": ann_hash})
    cat_counts = Counter(f.category for f in store.findings)
    results = {
        "schema_version": "1.0", "part": "1 - training-data integrity", "detector_version": DETECTOR_VERSION,
        "created_at": now_iso(), "disposition": disp,
        "dataset_manifest": {"dataset_id": "DS-" + ds_sha[:8], "format": b.fmt, "root": b.root,
                             "annotation_source": b.annotation_source, "dataset_sha256": ds_sha,
                             "sample_count": len(samples),
                             "annotation_count": sum(len(s.annotations) for s in samples),
                             "trusted_reference": str(reference) if reference else None},
        "configuration": cfg, "configuration_sha256": cfg_hash,
        "environment": {"python": platform.python_version(), "numpy": np.__version__,
                        "scipy": scipy.__version__, "scikit_learn": sklearn.__version__,
                        "encoder_file_sha256": sha256_file(cfg["embedding"]["onnx_path"]) if encoder is not None else None,
                        "network_access_required": False, "training_performed": False},
        "profile": prof, "module_status": mod_status, "unavailable_tests": unavailable,
        "finding_counts": dict(cat_counts), "duplicate_groups": [
            {k: v for k, v in g.items() if k != "members"} | {"member_count": len(g["members"])} for g in groups],
        "source_summaries": src, "limitations": LIMITATIONS,
        "sample_flags": {s.file_name: sorted(flags[s.sample_id]) for s in samples if s.sample_id in flags},
        "samples_index": {s.sample_id: s.file_name for s in samples},
        "findings": [f.to_dict() for f in store.findings],
        "timings_seconds": {**{k: round(v, 3) for k, v in timings.items()}, "total": round(time.time() - t_start, 3)},
    }
    write_reports(results, samples, out)
    audit.append("report", [], ["assurance_part1.json", "report.html", "findings.jsonl", "sample_flags.csv"], cfg_hash)
    log(f"[8/8] disposition={disp}; reports written to {out}")
    return results
