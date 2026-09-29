"""Modules 5-7 + Part-3 assurance engine: verify a ledger end to end (chain, signature, hashes, model registry, config policy,
independent re-inference, replay/binding) and emit findings + machine/human reports."""
from __future__ import annotations

import html as H
import json
from collections import Counter
from pathlib import Path

from ..common.audit import AuditLog
from ..common.hashing import sha256_bytes, sha256_file, sha256_json
from ..common.schemas import Finding, FindingStore, now_iso
from ..part2_model.adapters import Unavailable, load_adapter
from . import PART3_VERSION
from .binding import check_binding
from .canonical import build_classification_output, check_self_consistency, compare_outputs, output_hash
from .capture import decode_image
from .ledger import GENESIS, body_hash, read_records, sign
import hmac

RANK = {"ACCEPT": 0, "UNAVAILABLE": 1, "REVIEW": 2, "QUARANTINE": 3}
LIMITATIONS = [
    "Provenance cannot prove what happened before the trusted capture point (input acquisition / first hash).",
    "A fully privileged attacker who controls the host, the signing key AND the model registry can forge consistent evidence.",
    "Re-inference proves the recorded output is what the CLAIMED model produces for the recorded input; a substituted model that "
    "gives identical outputs on an input (e.g. a dormant trojan on a clean image) is indistinguishable here - Part 2 covers that.",
    "Configuration tampering that is internally consistent is only detectable against a trusted policy config; without one it is UNAVAILABLE.",
    "Floating-point/hardware differences are handled by a recorded tolerance; a valid hash proves identity of bytes, not semantic correctness.",
    "Zero-training: nothing is trained; only the supplied model is re-executed.",
]


class Part3FindingStore(FindingStore):
    def add(self, asset_type, asset_id, category, severity, confidence, detector, evidence, disposition, limitations=None, reference_assets=None):
        f = Finding(finding_id=f"P-{len(self.findings) + 1:05d}", asset_type=asset_type, asset_id=asset_id, category=category,
                    severity=severity, confidence=round(float(confidence), 3), detector=detector, detector_version=PART3_VERSION,
                    evidence=evidence, reference_assets=reference_assets or [], timestamp=now_iso(), disposition=disposition,
                    limitations=limitations or [])
        self.findings.append(f); return f


def _max(ds): return max(ds, key=lambda d: RANK[d]) if ds else "ACCEPT"


def verify_ledger(ledger_dir, out_dir=None, key=None, registry=None, policy=None, log=print, sign_required=True) -> dict:
    d = Path(ledger_dir)
    registry = registry if registry is not None else (json.loads((d / "registry.json").read_text()) if (d / "registry.json").exists() else {"models": {}})
    policy = policy if policy is not None else (json.loads((d / "policy.json").read_text()) if (d / "policy.json").exists() else None)
    tol = (policy or {}).get("compare_tolerance", 1e-3)
    out = Path(out_dir) if out_dir else d / "part3_out"; out.mkdir(parents=True, exist_ok=True)
    audit = AuditLog(out / "audit_log.jsonl"); cfg_hash = sha256_json({"policy": policy, "tol": tol})
    S = Part3FindingStore(); unavailable, checks = [], Counter()
    records = read_records(d / "ledger.jsonl")
    audit.append("load_ledger", [str(d / "ledger.jsonl")], [f"{len(records)} records"], cfg_hash, actor="part3_verifier")
    if not records:
        S.add("ledger", "LEDGER", "empty_ledger", "medium", 0.9, "ledger_v1", {}, "UNAVAILABLE", ["Nothing to verify."])

    # ---------- chain / signature / manifest (Module 5)
    prev = GENESIS
    for r in records:
        rid = r.get("record_id")
        if body_hash(r) != r.get("record_hash"):
            S.add("inference", rid, "record_hash_mismatch", "critical", 0.99, "chain_v1",
                  {"stored": r.get("record_hash"), "recomputed": body_hash(r)}, "QUARANTINE", ["Record body was modified after it was sealed."])
        elif r.get("previous_record_hash") != prev:
            S.add("inference", rid, "chain_link_broken", "critical", 0.99, "chain_v1",
                  {"expected_previous": prev, "found_previous": r.get("previous_record_hash")}, "QUARANTINE", ["A record was removed, inserted or reordered."])
        checks["chain"] += 1
        if key is not None:
            if not r.get("signature") or not hmac.compare_digest(str(r["signature"]), sign(key, r.get("record_hash", ""))):
                S.add("inference", rid, "signature_invalid", "critical", 0.99, "signature_v1", {"note": "HMAC does not verify with the supplied key"}, "QUARANTINE",
                      ["Record was not sealed by the key holder."])
            checks["signature"] += 1
        prev = r.get("record_hash", prev)
    if key is None:
        unavailable.append("signature verification: no signing key supplied (hash chain only)")
    mf = d / "manifest.json"
    if mf.exists() and records:
        m = json.loads(mf.read_text())
        bad = []
        if m.get("record_count") != len(records): bad.append(f"count {m.get('record_count')} != {len(records)}")
        if m.get("head_hash") != records[-1].get("record_hash"): bad.append("head hash differs")
        if sha256_json({k: v for k, v in m.items() if k not in ("manifest_hash", "signature")}) != m.get("manifest_hash"): bad.append("manifest self-hash invalid")
        if key is not None and (not m.get("signature") or not hmac.compare_digest(m["signature"], sign(key, m.get("manifest_hash", "")))): bad.append("manifest signature invalid")
        if bad: S.add("ledger", "LEDGER", "manifest_mismatch", "high", 0.95, "manifest_v1", {"issues": bad}, "QUARANTINE", ["Possible truncation/append after sealing."])
    else:
        unavailable.append("batch manifest check: manifest.json absent")
    audit.append("verify_chain", ["ledger"], [str(checks["chain"])], cfg_hash, actor="part3_verifier")

    # ---------- per-record content checks (Modules 1-4, 6)
    adapters, models_bad = {}, set()

    def get_adapter(sha):
        if sha in adapters: return adapters[sha]
        ent = registry["models"].get(sha)
        a = None
        if ent:
            p = Path(ent["path"])
            if not p.is_absolute() and not p.exists(): p = d / p
            if p.exists() and sha256_file(p) == sha: a = load_adapter(p)
            elif p.exists(): a = "FILE_SHA_MISMATCH"
        adapters[sha] = a; return a

    exp_cfg = (policy or {}).get("expected_config"); exp_cfg_sha = (policy or {}).get("expected_config_sha256")
    if policy is None: unavailable.append("configuration policy check: no trusted policy.json supplied")
    status = {}
    for r in records:
        rid = r.get("record_id"); before = len(S.findings)
        # config / output self hashes
        if sha256_json(r.get("config")) != r.get("config_sha256"):
            S.add("inference", rid, "config_hash_mismatch", "high", 0.99, "binding_v1", {"stored": r.get("config_sha256"), "recomputed": sha256_json(r.get("config"))}, "QUARANTINE")
        if output_hash(r.get("output")) != r.get("output_sha256"):
            S.add("inference", rid, "output_hash_mismatch", "critical", 0.99, "binding_v1", {"stored": r.get("output_sha256"), "recomputed": output_hash(r.get("output"))}, "QUARANTINE", ["Stored output does not match its recorded hash."])
        issues = check_self_consistency(r.get("output", {}), r.get("config", {}))
        if issues: S.add("inference", rid, "output_internally_inconsistent", "high", 0.9, "binding_v1", {"issues": issues}, "QUARANTINE")
        checks["hash_binding"] += 1
        # input binding
        raw = None
        sp = d / (r.get("input") or {}).get("stored_as", "")
        if sp.is_file():
            raw = sp.read_bytes()
            if sha256_bytes(raw) != r.get("input_sha256"):
                S.add("inference", rid, "input_hash_mismatch", "critical", 0.99, "binding_v1", {"recorded": r.get("input_sha256"), "actual_bytes_sha256": sha256_bytes(raw)},
                      "QUARANTINE", ["The stored input is not the input the record claims."]); raw = None
            checks["input_binding"] += 1
        else:
            S.add("inference", rid, "input_bytes_unavailable", "medium", 0.9, "binding_v1", {"stored_as": (r.get("input") or {}).get("stored_as")}, "UNAVAILABLE",
                  ["Input bytes missing: input binding and re-inference cannot run (not a PASS)."])
        # model identity / registry
        sha = r.get("model_sha256"); ent = registry["models"].get(sha)
        if ent is None:
            S.add("inference", rid, "model_not_registered", "high", 0.95, "registry_v1", {"model_sha256": sha, "model_id": (r.get("model") or {}).get("model_id")}, "QUARANTINE",
                  ["Model digest is not in the trusted registry."])
        elif not ent.get("approved", True):
            S.add("inference", rid, "model_revoked", "high", 0.95, "registry_v1", {"model_sha256": sha}, "QUARANTINE")
        if (r.get("model") or {}).get("sha256") != sha:
            S.add("inference", rid, "model_identity_inconsistent", "high", 0.99, "registry_v1", {"model.sha256": (r.get("model") or {}).get("sha256"), "model_sha256": sha}, "QUARANTINE")
        checks["model_identity"] += 1
        # config vs trusted policy
        if policy is not None:
            if r.get("config_sha256") != exp_cfg_sha:
                diff = {k: {"expected": exp_cfg.get(k), "recorded": (r.get("config") or {}).get(k)} for k in set(exp_cfg) | set(r.get("config") or {}) if exp_cfg.get(k) != (r.get("config") or {}).get(k)}
                S.add("inference", rid, "config_policy_mismatch", "high", 0.95, "policy_v1", {"differences": diff}, "QUARANTINE",
                      ["May be a legitimate reconfiguration; a governance decision is required (Part 5)."])
            checks["config_policy"] += 1
        # re-inference (Module 6)
        a = get_adapter(sha) if ent else None
        if a == "FILE_SHA_MISMATCH":
            if sha not in models_bad:
                models_bad.add(sha); S.add("model", ent["model_id"], "registry_artifact_mismatch", "critical", 0.99, "registry_v1", {"expected_sha256": sha}, "QUARANTINE",
                                            ["The artifact on disk no longer matches its registered digest."])
            a = None
        if raw is not None and a is not None:
            try:
                rec = build_classification_output(a.predict([decode_image(raw)])[0], r.get("config", {}))
                cmp_ = compare_outputs(r.get("output", {}), rec, tol, (r.get("config") or {}).get("decision_threshold"))
                checks["reinference"] += 1
                if cmp_["result"] == "MISMATCH":
                    S.add("inference", rid, "reinference_mismatch", "critical" if (cmp_.get("pred_changed") or cmp_.get("decision_changed")) else "high", 0.97, "reinference_v1",
                          {**cmp_, "recomputed_output_sha256": output_hash(rec)}, "QUARANTINE",
                          ["Recorded output differs from independent recomputation with the claimed model, recorded input and recorded config."])
                elif cmp_["result"] == "TOLERANCE_MATCH":
                    S.add("inference", rid, "numerical_drift_within_tolerance", "low", 0.9, "reinference_v1", cmp_, "ACCEPT", ["Within recorded tolerance; hardware/float differences expected."])
                status.setdefault(rid, cmp_["result"])
            except Exception as e:
                S.add("inference", rid, "reinference_unavailable", "medium", 0.8, "reinference_v1", {"reason": str(e)}, "UNAVAILABLE")
        elif raw is not None and ent is not None:
            pass
        elif ent is None:
            unavailable.append(f"{rid}: re-inference skipped (model not in registry)") if len(unavailable) < 50 else None
    # ---------- binding / replay across ledger (Module 7)
    for b in check_binding(records, (policy or {}).get("max_record_lag_s", 300.0)):
        if b["category"] == "output_reuse_across_inputs" and status.get(b["record_id"]) in ("EXACT", "TOLERANCE_MATCH"):
            continue      # independent re-inference proves this record's output is genuinely its own
        S.add("ledger" if b["record_id"] == "LEDGER" else "inference", b["record_id"], b["category"], b["severity"], 0.9, "binding_v1", b["evidence"], b["disposition"])
    checks["replay_binding"] += len(records)
    audit.append("verify_content", ["records"], [f"{len(S.findings)} findings"], cfg_hash, actor="part3_verifier")

    # ---------- assurance
    per = {}
    for r in records: per[r["record_id"]] = "ACCEPT"
    for f in S.findings:
        if f.asset_id in per and f.disposition != "ACCEPT" and RANK[f.disposition] > RANK[per[f.asset_id]]: per[f.asset_id] = f.disposition
    disp = _max([f.disposition for f in S.findings if f.disposition != "UNAVAILABLE"] or ["ACCEPT"])
    res = {"schema_version": "1.0", "part": "3 - inference provenance & output integrity", "detector_version": PART3_VERSION, "created_at": now_iso(),
           "disposition": disp, "records_total": len(records), "record_status": per,
           "record_counts": dict(Counter(per.values())), "checks_executed": dict(checks),
           "chain_head": records[-1]["record_hash"] if records else None, "signature_checked": key is not None,
           "findings": [f.to_dict() for f in S.findings], "finding_counts": dict(Counter(f.category for f in S.findings)),
           "unavailable_tests": sorted(set(unavailable)), "limitations": LIMITATIONS, "policy_present": policy is not None,
           "configuration_sha256": cfg_hash}
    (out / "assurance_part3.json").write_text(json.dumps(res, indent=2, default=str))
    (out / "findings_part3.jsonl").write_text("".join(json.dumps(f) + "\n" for f in res["findings"]))
    (out / "report_part3.html").write_text(_html(res))
    audit.append("report", ["assurance_part3.json"], [sha256_file(out / "assurance_part3.json")], cfg_hash, actor="part3_verifier")
    log(f"[part3] records={len(records)} disposition={disp} counts={res['record_counts']} findings={len(S.findings)}")
    return res


def _html(r):
    col = {"ACCEPT": "#2e7d32", "REVIEW": "#ef6c00", "QUARANTINE": "#c62828", "UNAVAILABLE": "#607d8b"}
    b = lambda x: f'<span style="background:{col.get(x, "#555")};color:#fff;padding:2px 9px;border-radius:10px">{H.escape(str(x))}</span>'
    rows = "".join(f"<tr><td>{H.escape(f['finding_id'])}</td><td>{H.escape(f['asset_id'])}</td><td>{H.escape(f['category'])}</td><td>{f['severity']}</td><td>{b(f['disposition'])}</td>"
                   f"<td><pre style='margin:0;font-size:11px'>{H.escape(json.dumps(f['evidence'], default=str)[:260])}</pre></td></tr>" for f in r["findings"] if f["disposition"] != "ACCEPT")
    return f"""<!doctype html><meta charset=utf-8><title>Part 3 assurance</title><body style="font-family:system-ui;margin:2rem">
<h1>Part 3 - Inference Provenance &amp; Output Integrity</h1><h2>{b(r['disposition'])}</h2>
<p>Records: {r['records_total']} - status: {H.escape(json.dumps(r['record_counts']))} - checks executed: {H.escape(json.dumps(r['checks_executed']))}</p>
<p>Chain head: <code>{r['chain_head']}</code> - signature checked: {r['signature_checked']} - trusted policy present: {r['policy_present']}</p>
<h3>Findings</h3><table border=1 cellpadding=5 style="border-collapse:collapse"><tr><th>ID</th><th>Asset</th><th>Category</th><th>Sev</th><th>Disposition</th><th>Evidence</th></tr>{rows or '<tr><td colspan=6>none</td></tr>'}</table>
<h3>Unavailable tests</h3><ul>{''.join(f'<li>{H.escape(u)}</li>' for u in r['unavailable_tests']) or '<li>none</li>'}</ul>
<h3>Limitations</h3><ul>{''.join(f'<li>{H.escape(l)}</li>' for l in r['limitations'])}</ul></body>"""
