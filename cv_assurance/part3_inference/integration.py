"""Part 1 + Part 2 + Part 3 integration.
* Part 1 supplies the dataset inputs and per-sample flags (trigger candidate, label anomaly, duplicate, OOD ...).
* Part 2 supplies the model fingerprint, the trusted-model registry entry and the model-level disposition.
* Part 3 records inferences on those inputs (each record LINKED to sample_id + Part 1 flags + Part 2 model id), verifies the
  ledger, and the bridge correlates provenance failures with Part 1 flags and the Part 2 model disposition."""
from __future__ import annotations

import html as H
import json
from collections import Counter, defaultdict
from pathlib import Path

from ..common.audit import AuditLog
from ..common.hashing import sha256_file, sha256_json
from ..common.schemas import now_iso
from ..dataset.ingest import load_dataset
from ..part2_model.integration import run_integrated
from .capture import InferenceRecorder, default_config, model_identity, write_policy, write_registry
from .ledger import generate_key
from .verifier import RANK, verify_ledger
from ..part2_model.adapters import load_adapter


def _decodable(path) -> bool:
    """Header-valid but truncated files (injected corruption) pass ingestion; skip them for inference recording."""
    try:
        from PIL import Image
        with Image.open(path) as im:
            im.load()
        return True
    except Exception:
        return False


def select_samples(bundle, flags, n):
    """Mix: trigger-flagged first, then other flagged categories, then unflagged (deterministic order)."""
    ok = [s for s in bundle.samples if s.readable and s.image_path and Path(s.image_path).exists() and _decodable(s.image_path)]
    trig = [s for s in ok if "trigger_candidate" in flags.get(s.file_name, [])]
    other = [s for s in ok if flags.get(s.file_name) and s not in trig]
    clean = [s for s in ok if not flags.get(s.file_name)]
    pick = trig[: n // 3] + other[: n // 3]
    pick += clean[: n - len(pick)]
    return pick[:n]


def run_integrated3(dataset_dir, served_model, trusted_model, out_dir, claim_trusted=False, fmt="auto", n_records=60,
                    part1_dir=None, mode="auto", log=print):
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    same = Path(served_model).resolve() == Path(trusted_model).resolve()
    # ---- Part 1 + Part 2 (existing integrated pipeline)
    comb12, p1, p2 = run_integrated(dataset_dir, served_model, out / "part12", None if same else trusted_model, fmt=fmt, part1_dir=part1_dir, mode=mode, log=lambda *a: None)
    flags = p1.get("sample_flags", {}); bundle = load_dataset(dataset_dir, fmt)
    ds_id = p1["dataset_manifest"]["dataset_id"]
    # ---- Part 3 record
    key = generate_key(out / "ledger.key")
    adapter = load_adapter(served_model); cfg = default_config(adapter)
    ldir = out / "part3_ledger"
    rec = InferenceRecorder(ldir, served_model, key=key, config=cfg, adapter=adapter, identity_path=trusted_model if claim_trusted else served_model)
    served_id = model_identity(served_model)["model_id"]
    for s in select_samples(bundle, flags, n_records):
        rec.record(image_path=s.image_path, input_id=s.sample_id,
                   links={"dataset_id": ds_id, "sample_id": s.sample_id, "file_name": s.file_name, "part1_flags": flags.get(s.file_name, []),
                          "part2_model_id": p2.get("model_id"), "part2_disposition": p2["disposition"], "executing_model_id": served_id})
    rec.seal()
    t = model_identity(trusted_model)
    write_registry(ldir / "registry.json", [{"model_id": t["model_id"], "sha256": t["sha256"], "path": str(Path(trusted_model).resolve())}])
    write_policy(ldir / "policy.json", cfg)
    p3 = verify_ledger(ldir, out / "part3", key=key, log=lambda *a: None)
    # ---- correlation
    recs = {json.loads(l)["record_id"]: json.loads(l) for l in (ldir / "ledger.jsonl").read_text().splitlines()}
    bad = {rid for rid, st in p3["record_status"].items() if st in ("REVIEW", "QUARANTINE")}
    by_flag = defaultdict(lambda: [0, 0])
    for rid, r in recs.items():
        for fl in (r["links"].get("part1_flags") or ["unflagged"]):
            by_flag[fl][0] += 1; by_flag[fl][1] += rid in bad
    corr = {k: {"records": v[0], "flagged_by_part3": v[1], "rate": round(v[1] / v[0], 3)} for k, v in by_flag.items()}
    signals = []
    mism = [f for f in p3["findings"] if f["category"] == "reinference_mismatch"]
    if mism:
        trig_hit = [f for f in mism if "trigger_candidate" in (recs[f["asset_id"]]["links"].get("part1_flags") or [])]
        if trig_hit:
            signals.append({"signal": "CONVERGING_INFERENCE_EVIDENCE", "severity": "critical", "disposition": "QUARANTINE",
                            "detail": f"{len(trig_hit)} of {len(mism)} re-inference mismatches are on inputs that Part 1 flagged as trigger candidates: recorded outputs were "
                                      "NOT produced by the claimed model on those inputs, and the inputs carry the recurring pattern found in the data.",
                            "part3_findings": [f["finding_id"] for f in trig_hit][:20]})
        else:
            signals.append({"signal": "OUTPUT_PROVENANCE_FAILURE", "severity": "high", "disposition": "QUARANTINE", "detail": f"{len(mism)} outputs not reproducible with the claimed model."})
    unreg = [f for f in p3["findings"] if f["category"] == "model_not_registered"]
    if unreg and p2["disposition"] != "ACCEPT":
        signals.append({"signal": "UNREGISTERED_MODEL_CORROBORATED_BY_PART2", "severity": "critical", "disposition": "QUARANTINE",
                        "detail": f"Part 3: served model digest is not in the trusted registry; Part 2 independently rates the model {p2['disposition']} "
                                  f"({', '.join(sorted(p2['finding_counts']))}).", "part3_findings": [f["finding_id"] for f in unreg][:5]})
    if p3["disposition"] == "ACCEPT" and p2["disposition"] != "ACCEPT":
        signals.append({"signal": "PROVENANCE_VALID_BUT_MODEL_SUSPECT", "severity": "high", "disposition": "QUARANTINE",
                        "detail": "Ledger is intact and reproducible, but Part 2 flags the model. Provenance proves WHAT ran, not that it is trustworthy."})
    overall = max([p1["disposition"], p2["disposition"], p3["disposition"]] + [s["disposition"] for s in signals], key=lambda d: RANK.get(d, 0))
    audit = AuditLog(out / "audit_log_combined.jsonl")
    for nm, pth in (("part1_json", out / "part12" / "part1" / "assurance_part1.json"), ("part2_json", out / "part12" / "part2" / "assurance_part2.json"), ("part3_json", out / "part3" / "assurance_part3.json")):
        if Path(pth).exists(): audit.append("bind_evidence", [nm], [sha256_file(pth)], sha256_json({}), actor="integration_p123")
    comb = {"schema_version": "1.0", "created_at": now_iso(), "overall_disposition": overall, "scenario": {"served_model": Path(served_model).name, "trusted_model": Path(trusted_model).name, "record_claims_trusted_identity": claim_trusted},
            "dataset": {"id": ds_id, "disposition": p1["disposition"], "finding_counts": p1["finding_counts"]},
            "model": {"id": p2.get("model_id"), "disposition": p2["disposition"], "identity": p2.get("identity_status"), "finding_counts": p2["finding_counts"]},
            "inference": {"disposition": p3["disposition"], "records": p3["records_total"], "record_counts": p3["record_counts"], "finding_counts": p3["finding_counts"], "chain_head": p3["chain_head"]},
            "part3_flag_rate_by_part1_flag": corr, "cross_signals": signals, "part12_cross_signals": comb12["cross_signals"],
            "unavailable_tests": {"part3": p3["unavailable_tests"]},
            "note": "Dispositions prioritise review; they are not statements of intent. Governance stays with the analyst."}
    (out / "assurance_combined_p123.json").write_text(json.dumps(comb, indent=2, default=str))
    (out / "report_combined_p123.html").write_text(_html(comb))
    log(f"[p123] overall={overall}  P1={p1['disposition']} P2={p2['disposition']} P3={p3['disposition']}  signals={[s['signal'] for s in signals]}")
    return comb, p1, p2, p3


def _html(c):
    col = {"ACCEPT": "#2e7d32", "REVIEW": "#ef6c00", "QUARANTINE": "#c62828", "UNAVAILABLE": "#607d8b"}
    b = lambda d: f'<span style="background:{col.get(d, "#555")};color:#fff;padding:2px 9px;border-radius:10px">{H.escape(str(d))}</span>'
    sig = "".join(f"<li>{b(s['disposition'])} <b>{H.escape(s['signal'])}</b> - {H.escape(s['detail'])}</li>" for s in c["cross_signals"]) or "<li>none</li>"
    return f"""<!doctype html><meta charset=utf-8><title>Combined P1+P2+P3</title><body style="font-family:system-ui;margin:2rem">
<h1>Combined Assurance - Data / Model / Inference</h1><h2>Overall {b(c['overall_disposition'])}</h2>
<p>Scenario: {H.escape(json.dumps(c['scenario']))}</p>
<table cellpadding=6 border=1 style="border-collapse:collapse"><tr><th>Part 1 dataset {H.escape(c['dataset']['id'])}</th><td>{b(c['dataset']['disposition'])}</td></tr>
<tr><th>Part 2 model {H.escape(str(c['model']['id']))}</th><td>{b(c['model']['disposition'])}</td></tr>
<tr><th>Part 3 inference ({c['inference']['records']} records)</th><td>{b(c['inference']['disposition'])}</td><td>{H.escape(json.dumps(c['inference']['record_counts']))}</td></tr></table>
<h3>Cross-signal correlation</h3><ul>{sig}</ul><h3>Part 3 flag rate by Part 1 flag category</h3><pre>{H.escape(json.dumps(c['part3_flag_rate_by_part1_flag'], indent=1))}</pre>
<p><small>{H.escape(c['note'])}</small></p></body>"""


def integrated_demo(out_dir, n_clean=200, log=print):
    from ..benchmark.part2_models import build_benchmark
    from ..benchmark.synthetic import generate
    out = Path(out_dir); build_benchmark(out / "benchmark"); generate(out / "dataset", n_clean, "coco")
    M = out / "benchmark" / "models"; trusted, trojan = M / "trusted_reference.npz", M / "trojan_checker.npz"
    scen = [("A_honest_trusted_model", trusted, False), ("B_trojan_model_honestly_recorded", trojan, False), ("C_trojan_model_claims_trusted_identity", trojan, True)]
    p1dir = None
    for name, served, claim in scen:
        log(f"\n================ scenario {name} ================")
        comb, *_ = run_integrated3(out / "dataset", served, trusted, out / name, claim_trusted=claim, part1_dir=p1dir, log=log)
        p1dir = p1dir or (out / name / "part12" / "part1")
        for s in comb["cross_signals"]: log(f"   - {s['signal']}: {s['detail'][:150]}")
    log(f"\nOpen {out}/C_trojan_model_claims_trusted_identity/report_combined_p123.html")
