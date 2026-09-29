"""Module 8 - Model Assurance Engine + Part 2 orchestrator (run_part2)."""
from __future__ import annotations

import json
import platform
import time
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

from ..common.audit import AuditLog
from ..common.hashing import sha256_json
from ..common.schemas import now_iso
from . import PART2_VERSION, behaviour as B, params as P, trigger as T
from .adapters import Unavailable, load_adapter
from .config import load_p2_config
from .findings import Part2FindingStore, SEV_RANK
from .fingerprint import fingerprint_model
from .report import write_reports
from .substitution import substitution_analysis

LIMITATIONS = [
    "Hash mismatch proves the artifact differs from the trusted one, not that it was maliciously modified.",
    "Parameter differences are evidence of model difference, not of compromise.",
    "Trigger scan covers a finite pattern family/positions/sizes plus any Part 1 candidates; a clean scan is not proof of no backdoor.",
    "Behavioural agreement is measured only on the supplied reference battery.",
    "Black-box mode cannot inspect weights or activations; those tests are marked UNAVAILABLE, never PASS.",
    "Zero-training: no model is trained or fine-tuned on the assessed artifact or data.",
]
IMG_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def load_reference_images(directory, limit):
    d = Path(directory)
    files = sorted(p for p in d.rglob("*") if p.suffix.lower() in IMG_EXT)[:limit]
    labels_f = d / "labels.json"
    lab = json.loads(labels_f.read_text()) if labels_f.exists() else {}
    imgs = [np.asarray(Image.open(p).convert("RGB")) for p in files]
    labels = [lab.get(p.name) for p in files]
    return imgs, (labels if all(l is not None for l in labels) and labels else None)


def _disp(store):
    ds = {f.disposition for f in store.findings}
    return "QUARANTINE" if "QUARANTINE" in ds else "REVIEW" if "REVIEW" in ds else "ACCEPT"


def run_part2(model_path, out_dir, reference_model=None, expected_sha=None, images=None, labels=None,
              reference_images_dir=None, mode="auto", part1_ctx=None, config=None, log=print) -> dict:
    t0 = time.time()
    cfg = load_p2_config(config) if not isinstance(config, dict) else config
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    audit = AuditLog(out / "audit_log.jsonl")
    cfg_hash = sha256_json(cfg)
    store = Part2FindingStore()
    unavailable, timings, mod_status = [], {}, {}
    model_path = Path(model_path)
    model_id = "MODEL-" + model_path.stem

    # ---- inputs: battery images
    if part1_ctx is not None:
        images, labels = part1_ctx["images"], part1_ctx["labels"]
    elif images is None and reference_images_dir:
        images, labels = load_reference_images(reference_images_dir, cfg["max_battery_images"])
    if images is not None:
        images = images[:cfg["max_battery_images"]]; labels = None if labels is None else list(labels)[:len(images)]

    # ---- 1 ingestion
    try:
        adapter = load_adapter(model_path, mode)
    except Unavailable as e:
        store.add("model", model_id, "model_not_loadable", "high", 0.99, "adapter_v1", {"reason": str(e)}, "UNAVAILABLE",
                  ["Unsupported or unsafe artifact; no tests could run. This is NOT a PASS."])
        audit.append("ingest", [str(model_path)], [], cfg_hash, f"UNAVAILABLE: {e}", "part2_pipeline")
        res = {"schema_version": "1.0", "part": "2 - model integrity", "detector_version": PART2_VERSION, "created_at": now_iso(),
               "disposition": "UNAVAILABLE", "identity_status": "UNKNOWN", "findings": [f.to_dict() for f in store.findings],
               "unavailable_tests": [f"all: {e}"], "limitations": LIMITATIONS, "configuration": cfg, "configuration_sha256": cfg_hash}
        write_reports(res, out); return res
    ref = None
    if reference_model:
        try: ref = load_adapter(reference_model)
        except Unavailable as e: unavailable.append(f"trusted reference model could not be loaded: {e}")
    log(f"[1/8] loaded {adapter.fmt} ({'white-box' if adapter.white_box else 'black-box'}); reference={'yes' if ref else 'no'}")
    audit.append("ingest", [str(model_path)], [model_id], cfg_hash, actor="part2_pipeline")

    # ---- 2 fingerprint
    fp = fingerprint_model(adapter, expected_sha, reference_model if ref else None)
    if fp["identity_status"] == "MISMATCH":
        store.add("model", model_id, "identity_mismatch", "low", 0.99, "fingerprint_v1",
                  {"sha256": fp["sha256"], "expected_sha256": fp["expected_sha256"], "expected_source": fp["expected_source"]}, "REVIEW",
                  ["Hash mismatch = different bytes; not by itself proof of malicious modification."])
    log(f"[2/8] identity: {fp['identity_status']}  sha256={fp['sha256'][:12]}...")
    audit.append("fingerprint", [model_id], [fp["sha256"]], cfg_hash, fp["identity_status"], "part2_pipeline")

    # ---- 3 parameters (white-box)
    t = time.time()
    if adapter.white_box:
        param_res, st = P.analyze_parameters(adapter, ref if (ref and ref.white_box) else None, images, cfg, store, model_id)
        if ref is not None and not ref.white_box: unavailable.append("parameter diff: reference not white-box")
    else:
        param_res, st = {"status": "UNAVAILABLE", "reason": "black-box access: weights/activations not accessible"}, "UNAVAILABLE"
    mod_status["parameters"] = param_res.get("status", st)
    if mod_status["parameters"] == "UNAVAILABLE":
        unavailable.append(f"parameter & activation analysis: {param_res.get('reason')}")
    elif isinstance(param_res.get("dormant_units"), dict) and param_res["dormant_units"].get("status") == "UNAVAILABLE":
        unavailable.append(f"activation analysis: {param_res['dormant_units']['reason']}")
    if ref is None and mod_status["parameters"] == "OK": unavailable.append("parameter diff vs trusted model: no trusted reference model supplied")
    log(f"[3/8] parameter analysis: {mod_status['parameters']}"); timings["parameters"] = time.time() - t
    audit.append("parameter_analysis", [model_id], [mod_status["parameters"]], cfg_hash, actor="part2_pipeline")

    # ---- 4+5 behaviour + battery
    t = time.time()
    beh, beh_cmp, acc, acc_ref = None, None, None, None
    n_classes = adapter.output_schema().get("num_classes")
    cn = adapter.class_names() or (ref.class_names() if ref else None)
    class_names = {i: n for i, n in enumerate(cn)} if cn else {}
    if images:
        try:
            ads = {"received": adapter}
            if ref is not None: ads["reference"] = ref
            probs = B.run_battery(images, ads)
            Pr = probs["received"]; n_classes = n_classes or Pr["normal:identity"].shape[1]
            det = float(np.abs(adapter.predict(images) - Pr["normal:identity"]).max())
            beh = B.fingerprint(Pr, det, n_classes)
            if det > cfg["behaviour"]["determinism_tol"]:
                store.add("model", model_id, "non_deterministic_inference", "medium", 0.9, "behaviour_v1", {"max_abs_diff": det}, "REVIEW",
                          ["Non-determinism complicates re-inference (Part 3) and behavioural comparison."])
            acc = B.accuracy(Pr, labels)
            if ref is not None:
                beh_cmp = B.compare_to_reference(Pr, probs["reference"])
                acc_ref = B.accuracy(probs["reference"], labels)
                bad_h = {k: v for k, v in beh_cmp.items() if v["agreement"] < cfg["behaviour"]["agree_high"]}
                bad_m = {k: v for k, v in beh_cmp.items() if v["agreement"] < cfg["behaviour"]["agree_review"]}
                if bad_m:
                    fam = Counter(k.split(":")[0] for k in bad_m)
                    store.add("model", model_id, "behavioural_divergence", "high" if bad_h else "medium", 0.9, "behaviour_v1",
                              {"variants_below_threshold": bad_m, "families": dict(fam), "n_images": len(images),
                               "min_agreement": min(v["agreement"] for v in beh_cmp.values())}, "REVIEW",
                              ["Divergence is measured only on the supplied battery; legitimate model updates also diverge."],
                              reference_assets=["trusted_reference_model"])
                if acc and acc_ref and acc_ref["normal:identity"] - acc["normal:identity"] >= cfg["behaviour"]["accuracy_drop"]:
                    store.add("model", model_id, "task_accuracy_drop", "high", 0.9, "behaviour_v1",
                              {"received_accuracy": acc["normal:identity"], "reference_accuracy": acc_ref["normal:identity"]}, "REVIEW", [])
            elif acc and acc["normal:identity"] < cfg["behaviour"]["min_accuracy"]:
                store.add("model", model_id, "low_task_accuracy", "medium", 0.7, "behaviour_v1",
                          {"accuracy_on_trusted_labels": acc["normal:identity"]}, "REVIEW",
                          ["Accuracy floor is a sanity check; class-mapping mismatches also lower it."])
            mod_status["behaviour"] = "OK"
        except Unavailable as e:
            mod_status["behaviour"] = "UNAVAILABLE"; unavailable.append(f"behavioural battery: {e}")
    else:
        mod_status["behaviour"] = "UNAVAILABLE"; unavailable.append("behavioural battery & trigger tests: no trusted reference images supplied")
    if images and ref is None: unavailable.append("behavioural comparison vs trusted model: no trusted reference model supplied")
    if labels is None and images: unavailable.append("accuracy check: reference images have no labels")
    log(f"[4-5/8] behaviour battery: {mod_status['behaviour']}"); timings["behaviour"] = time.time() - t
    audit.append("behaviour_battery", [model_id], [mod_status["behaviour"]], cfg_hash, actor="part2_pipeline")

    # ---- 6 triggers
    t = time.time()
    trig = {"status": "UNAVAILABLE", "reason": "no reference images"}
    if images and mod_status["behaviour"] == "OK":
        try:
            timgs = images[:cfg["trigger"].get("max_images", cfg["max_trigger_images"])] if False else images[:cfg["max_trigger_images"]]
            trig, _ = T.run_trigger_tests(adapter, ref, timgs, cfg, store, model_id,
                                          part1_candidates=(part1_ctx or {}).get("trigger_candidates"), class_names=class_names)
        except Unavailable as e:
            trig = {"status": "UNAVAILABLE", "reason": str(e)}
    mod_status["trigger"] = trig.get("status")
    if trig.get("status") != "OK": unavailable.append(f"trigger tests: {trig.get('reason')}")
    log(f"[6/8] trigger tests: {trig.get('status')} hits={trig.get('hits')}"); timings["trigger"] = time.time() - t
    audit.append("trigger_tests", [model_id], [str(trig.get('hits'))], cfg_hash, actor="part2_pipeline")

    # ---- 7 substitution
    subst = substitution_analysis(fp, param_res, beh_cmp, store, model_id, cfg, ref is not None)
    log(f"[7/8] substitution verdict: {subst['verdict']}")
    audit.append("substitution", [model_id], [subst["verdict"]], cfg_hash, actor="part2_pipeline")

    # ---- 8 assurance
    sig = [f for f in store.findings if SEV_RANK.get(f.severity, 0) >= 2]
    disp = _disp(store)
    coverage = "FULL" if (adapter.white_box and ref and images) else "LIMITED"
    res = {"schema_version": "1.0", "part": "2 - model integrity", "detector_version": PART2_VERSION, "created_at": now_iso(),
           "disposition": disp, "coverage": coverage, "model_id": model_id,
           "access_mode": "white-box" if adapter.white_box else "black-box",
           "identity_status": fp["identity_status"], "fingerprint": fp,
           "parameter_findings": [f.finding_id for f in store.findings if f.category.startswith(("parameter", "dormant", "architecture"))],
           "behaviour_findings": [f.finding_id for f in store.findings if f.category in ("behavioural_divergence", "task_accuracy_drop", "low_task_accuracy", "non_deterministic_inference")],
           "trigger_findings": [f.finding_id for f in store.findings if f.category == "trigger_candidate_response"],
           "substitution_findings": [f.finding_id for f in store.findings if f.category in ("model_substitution_assessment", "identity_mismatch", "identity_unverifiable")],
           "substitution": subst, "parameter_analysis": param_res, "behavioural_fingerprint": beh,
           "reference_comparison": beh_cmp, "accuracy": {"received": acc, "trusted_reference_model": acc_ref},
           "trigger_tests": trig, "module_status": mod_status, "unavailable_tests": sorted(set(unavailable)),
           "significant_finding_count": len(sig), "finding_counts": dict(Counter(f.category for f in store.findings)),
           "configuration": cfg, "configuration_sha256": cfg_hash,
           "environment": {"python": platform.python_version(), "numpy": np.__version__, "network_access_required": False,
                           "training_performed": False, "battery_images": len(images or []), "class_names": class_names},
           "limitations": LIMITATIONS, "findings": [f.to_dict() for f in store.findings],
           "timings_seconds": {**{k: round(v, 3) for k, v in timings.items()}, "total": round(time.time() - t0, 3)}}
    audit.append("report", [model_id], ["assurance_part2.json", "report_part2.html"], cfg_hash, disp, "part2_pipeline")
    write_reports(res, out)
    log(f"[8/8] disposition={disp} coverage={coverage}; reports -> {out}")
    return res
