"""Part 1 + 2 + 3 + 4 integration.
* Part 1  -> the trusted REFERENCE for Part 4 is (by default) the Part-1-unflagged samples of the dataset (production: pass a curated reference).
* Part 2  -> the trusted model produces the reference task outputs; the served model's Part-2 disposition is carried into the correlation.
* Part 3  -> the operational stream is RECORDED into a signed hash-chained ledger and VERIFIED; Part 4 then reads the preserved input bytes and
             the recorded outputs straight from that verified ledger (so shift evidence is bound to provenance: every sample -> record_id).
* Part 4  -> shift/drift/OOD/attribution on the ledger window; cross-signals with Parts 1-3."""
from __future__ import annotations
import html as H, json, shutil
from pathlib import Path
from ..common.audit import AuditLog
from ..common.hashing import sha256_bytes, sha256_file, sha256_json
from ..common.schemas import now_iso
from ..dataset.ingest import load_dataset
from ..part2_model.adapters import load_adapter
from ..part3_inference.capture import InferenceRecorder, default_config, model_identity, write_policy, write_registry
from ..part3_inference.integration import run_integrated3
from ..part3_inference.ledger import generate_key
from ..part3_inference.verifier import RANK, verify_ledger
from .assess import run_part4
from .config import load_p4_config
from .data import build_table, load_dir, save_reference
from .features import IMG_EXT

INPUT_FAMS = {"illumination", "color", "geometry", "sharpness_noise", "compression", "semantic"}


def reference_from_part1(dataset_dir, p1, fmt, cfg, adapter, out):
    b = load_dataset(dataset_dir, fmt); flags = p1.get("sample_flags", {}); items = []
    for s in b.samples:
        if s.readable and s.image_path and Path(s.image_path).exists() and not flags.get(s.file_name):
            items.append({"id": s.sample_id, "path": s.image_path, "source": s.source_id or s.contributor_id, "batch_id": "reference", "timestamp": ""})
    t, bad = build_table(items, cfg, adapter=adapter)
    return save_reference(t, cfg, out, {"origin": "part1_unflagged_samples", "dataset_id": p1["dataset_manifest"]["dataset_id"], "excluded_flagged": len(b.samples) - len(items),
                                         "note": "derived from Part-1-unflagged samples; supply a curated trusted reference for production"})


def record_stream(name, sdir, served, trusted, out, key, dataset_id, log):
    """Part 3 for the operational stream: record -> seal -> registry/policy -> verify."""
    ldir = out / "ledger"; ad = load_adapter(served); cfg = default_config(ad)
    meta = {}; mp = Path(sdir) / "metadata.json"
    if mp.exists(): meta = json.loads(mp.read_text())
    rec = InferenceRecorder(ldir, served, key=key, config=cfg, adapter=ad, identity_path=trusted)
    files = sorted(p for p in Path(sdir).iterdir() if p.suffix.lower() in IMG_EXT); files.sort(key=lambda p: ((meta.get(p.name) or {}).get("timestamp") or "", p.name))
    for p in files:
        m = meta.get(p.name) or {}
        rec.record(image_path=p, input_id=p.name, source=m.get("source"), links={"stream": name, "dataset_id": dataset_id, "batch_id": m.get("batch_id"), "file_name": p.name, "executing_model_id": model_identity(served)["model_id"]})
    rec.seal(); t = model_identity(trusted)
    write_registry(ldir / "registry.json", [{"model_id": t["model_id"], "sha256": t["sha256"], "path": str(Path(trusted).resolve())}]); write_policy(ldir / "policy.json", cfg)
    p3 = verify_ledger(ldir, out / "part3", key=key, log=lambda *a: None)
    return ldir, p3


def ledger_items(ldir):
    items, outputs, skipped = [], {}, 0
    for i, line in enumerate((Path(ldir) / "ledger.jsonl").read_text().splitlines()):
        r = json.loads(line); pth = Path(ldir) / r["input"]["stored_as"]
        if not pth.exists() or sha256_file(pth) != r["input_sha256"]: skipped += 1; continue          # input bytes must match the recorded digest
        items.append({"id": r["record_id"], "path": str(pth), "sha256": r["input_sha256"], "source": r["input"].get("source"), "batch_id": (r["links"] or {}).get("batch_id"),
                      "timestamp": f"{r['seq']:08d}", "record_id": r["record_id"]})
        o = r["output"]; outputs[r["record_id"]] = {"pred_class_id": o["pred_class_id"], "confidence": o["confidence"], "entropy": o["entropy"]}
    return items, outputs, skipped


def cross_signals(p4, p3s, p2, name):
    sig = []; fams = set(p4["overall_attribution"]["primary_families"]) | set(p4["temporal"].get("affected_families", [])) if p4["level"] != "NORMAL" else set()
    if p4["level"] == "NORMAL": return sig
    p3ok = p3s["disposition"] == "ACCEPT"
    if not p3ok:
        sig.append({"stream": name, "signal": "CONVERGING_SHIFT_AND_PROVENANCE_FAILURE", "severity": "critical", "disposition": "QUARANTINE",
                    "detail": f"Part 4 {p4['level']} while Part 3 rates the stream ledger {p3s['disposition']} ({', '.join(sorted(p3s['finding_counts']))}); the shift evidence and the provenance evidence point at the same window."})
    elif "task_output" in fams and not (fams & INPUT_FAMS):
        d = "QUARANTINE" if p2["disposition"] != "ACCEPT" else "REVIEW"
        sig.append({"stream": name, "signal": "OUTPUT_SHIFT_WITHOUT_INPUT_SHIFT", "severity": "high", "disposition": d,
                    "detail": f"Outputs moved but the input population did not. Suggests changed model behaviour rather than a changed environment; Part 2 disposition for the served model: {p2['disposition']}."})
    else:
        sig.append({"stream": name, "signal": "OPERATIONAL_SHIFT_WITH_INTACT_PROVENANCE", "severity": "medium", "disposition": "REVIEW",
                    "detail": f"Inputs shifted ({', '.join(sorted(fams & INPUT_FAMS)) or ', '.join(sorted(fams))}); ledger is intact and reproducible, so the shift is real, not a recording artefact. Likely environmental "
                              "(camera/lighting/scene) - analyst decides: accept and re-baseline the reference (Retest), or investigate acquisition."})
    return sig


def run_integrated4(dataset_dir, served_model, trusted_model, out_dir, streams, reference_dir=None, fmt="auto", n_records=40, p4_config=None, log=print):
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True); cfg = load_p4_config(p4_config) if not isinstance(p4_config, dict) else p4_config
    comb3, p1, p2, p3 = run_integrated3(dataset_dir, served_model, trusted_model, out / "p123", claim_trusted=(Path(served_model) != Path(trusted_model)), fmt=fmt, n_records=n_records, log=lambda *a: None)
    ds_id = p1["dataset_manifest"]["dataset_id"]; tad = load_adapter(trusted_model)
    if reference_dir:
        items = load_dir(reference_dir, cfg["window"]); t, _ = build_table(items, cfg, adapter=tad); rman = save_reference(t, cfg, out / "part4_reference", {"origin": str(reference_dir)})
    else: rman = reference_from_part1(dataset_dir, p1, fmt, cfg, tad, out / "part4_reference")
    log(f"[p1-3] done. [p4] reference {rman['reference_id']} n={rman['n']} ({rman['extra'].get('origin')})")
    key = generate_key(out / "stream_ledger.key"); results = []; signals = list(comb3["cross_signals"])
    for name, sdir in streams:
        sd = out / "streams" / name; sd.mkdir(parents=True, exist_ok=True)
        ldir, p3s = record_stream(name, sdir, served_model, trusted_model, sd, key, ds_id, log)
        items, outputs, skipped = ledger_items(ldir)
        p4 = run_part4(out / "part4_reference", items, cfg, sd / "part4", outputs=outputs, links={"stream": name, "dataset_id": ds_id, "ledger_chain_head": p3s["chain_head"], "ledger_disposition": p3s["disposition"],
                       "records_excluded_input_mismatch": skipped}, log=lambda *a: None)
        sg = cross_signals(p4, p3s, p2, name); signals += sg
        results.append({"stream": name, "part3_disposition": p3s["disposition"], "records": p3s["records_total"], "part4_level": p4["level"], "part4_disposition": p4["disposition"], "onset_batch": p4["temporal"].get("onset_batch"),
                        "pattern": p4["temporal"].get("pattern"), "primary_families": p4["overall_attribution"]["primary_families"], "finding_counts": p4["finding_counts"], "signals": [s["signal"] for s in sg], "unavailable": p4["unavailable_tests"]})
        log(f"[p4] stream {name}: P3={p3s['disposition']} P4={p4['level']} pattern={p4['temporal'].get('pattern')} onset={p4['temporal'].get('onset_batch')} families={p4['overall_attribution']['primary_families']}")
    overall = max([comb3["overall_disposition"]] + [r["part4_disposition"] for r in results] + [s["disposition"] for s in signals], key=lambda d: RANK.get(d, 0))
    audit = AuditLog(out / "audit_log_combined_p1234.jsonl")
    for r in results:
        pj = out / "streams" / r["stream"] / "part4" / "assurance_part4.json"; audit.append("bind_evidence", [f"part4:{r['stream']}"], [sha256_file(pj)], sha256_json(cfg), actor="integration_p1234")
    audit.append("bind_evidence", ["part123_combined"], [sha256_file(out / "p123" / "assurance_combined_p123.json")], sha256_json({}), actor="integration_p1234")
    comb = {"schema_version": "1.0", "created_at": now_iso(), "overall_disposition": overall, "part123": {"overall": comb3["overall_disposition"], "dataset": comb3["dataset"], "model": comb3["model"], "inference": comb3["inference"]},
            "reference": {"reference_id": rman["reference_id"], "n": rman["n"], "origin": rman["extra"].get("origin")}, "streams": results, "cross_signals": signals,
            "note": "Dispositions prioritise review; a distribution shift is not an attack. Governance stays with the analyst."}
    (out / "assurance_combined_p1234.json").write_text(json.dumps(comb, indent=2, default=str)); (out / "report_combined_p1234.html").write_text(_html(comb))
    log(f"[p1234] overall={overall} signals={[s['signal'] for s in signals]}"); return comb


def _html(c):
    col = {"ACCEPT": "#2e7d32", "REVIEW": "#ef6c00", "QUARANTINE": "#c62828", "NORMAL": "#2e7d32", "SHIFT DETECTED": "#ef6c00", "SEVERE SHIFT": "#c62828"}
    b = lambda d: f'<span style="background:{col.get(d, "#555")};color:#fff;padding:2px 9px;border-radius:10px">{H.escape(str(d))}</span>'
    rows = "".join(f"<tr><td>{H.escape(r['stream'])}</td><td>{b(r['part3_disposition'])} ({r['records']} rec)</td><td>{b(r['part4_level'])}</td><td>{H.escape(str(r['pattern']))} / onset {H.escape(str(r['onset_batch']))}</td><td>{H.escape(', '.join(r['primary_families']))}</td>"
                   f"<td><a href='streams/{H.escape(r['stream'])}/part4/report_part4.html'>report</a></td></tr>" for r in c["streams"])
    sg = "".join(f"<li>{b(s['disposition'])} <b>{H.escape(s['signal'])}</b> {H.escape(s.get('stream', ''))} - {H.escape(s['detail'])}</li>" for s in c["cross_signals"]) or "<li>none</li>"
    p = c["part123"]
    return f"""<!doctype html><meta charset=utf-8><title>Combined P1-P4</title><body style="font-family:system-ui;margin:2rem"><h1>Combined Assurance - Data / Model / Inference / Environment</h1><h2>Overall {b(c['overall_disposition'])}</h2>
<table cellpadding=6 border=1 style="border-collapse:collapse"><tr><th>Part 1 dataset</th><td>{b(p['dataset']['disposition'])}</td></tr><tr><th>Part 2 model</th><td>{b(p['model']['disposition'])}</td></tr><tr><th>Part 3 inference (dataset ledger)</th><td>{b(p['inference']['disposition'])}</td></tr></table>
<h3>Part 4 - operational streams (reference {H.escape(c['reference']['reference_id'])}, n={c['reference']['n']}, {H.escape(str(c['reference']['origin']))})</h3>
<table cellpadding=6 border=1 style="border-collapse:collapse"><tr><th>stream</th><th>Part 3 (stream ledger)</th><th>Part 4</th><th>temporal</th><th>primary families</th><th></th></tr>{rows}</table>
<h3>Cross-signal correlation</h3><ul>{sg}</ul><p><small>{H.escape(c['note'])}</small></p></body>"""


def integrated_demo(out_dir, n_clean=240, log=print):
    from ..benchmark.part2_models import build_benchmark
    from ..benchmark.part4_bench import make_batch
    from ..benchmark.synthetic import generate
    out = Path(out_dir); build_benchmark(out / "benchmark"); generate(out / "dataset", n_clean, "coco")
    M = out / "benchmark" / "models"; trusted = M / "trusted_reference.npz"
    for name, kinds in (("stable_stream", ["none"] * 4), ("shifted_stream", ["none", "none", "camera", "illumination"])):
        d = out / "streams_in" / name; meta = {}
        for b, k in enumerate(kinds):
            src = "contributor_z" if k == "camera" else "contributor_a"
            meta.update(make_batch(d, f"b{b}", 40, k, 500 + b + len(name), source=src, ts0=b))
        (d / "metadata.json").write_text(json.dumps(meta))
    return run_integrated4(out / "dataset", trusted, trusted, out / "result", [("stable_stream", out / "streams_in" / "stable_stream"), ("shifted_stream", out / "streams_in" / "shifted_stream")], log=log)
