"""Part 5 pipeline: discover -> normalise -> correlate -> score -> govern -> audit -> report -> sealed export."""
from __future__ import annotations

import hashlib
import hmac
import json
import platform
import sys
from pathlib import Path

from ..common.hashing import sha256_bytes, sha256_file, sha256_json
from ..common.schemas import now_iso
from .aggregate import discover, normalize, source_summaries
from .audit import GovAudit, load_or_create_key
from .config import P5_VERSION, RANK, load_policy
from .correlate import Engine
from .governance import GovernanceError, fold, overall, validate
from .risk import max_disp

TYPE_ORDER = ["sample", "group", "pattern", "source", "batch", "dataset", "model", "ledger", "record", "stream", "window", "part"]
LIMITATIONS = [
    "Rule-based correlation only: converging evidence raises review priority, it does not establish intent, attribution or a functional backdoor beyond tests executed.",
    "The review-priority score is a transparent noisy-OR of policy weights, not a probability of malicious activity.",
    "Absence of findings is not proof of safety; UNAVAILABLE items are coverage gaps, never passes.",
    "A distribution shift is not automatically an attack; governance decides whether it is acceptable.",
    "The audit trail is tamper-evident within the declared trust boundary; a fully privileged attacker who controls the host, the key or the assurance binary is out of scope.",
    "Inputs are trusted only from the point at which Parts 1-4 artifacts were captured (their hashes are recorded here).",
]
HONEST_CLAIMS = ["Detects specified integrity inconsistencies and controlled attack scenarios and correlates them across data, model, inference and shift.",
                 "Every case links to raw findings, their detector versions and the hash of the artifact that produced them.",
                 "Runs offline; no retraining, no cloud/API calls; unavailable tests are disclosed."]


def _trace(entities):
    def key(e):
        t = e.split(":", 1)[0]
        return (TYPE_ORDER.index(t) if t in TYPE_ORDER else 99, e)
    return sorted(entities, key=key)


def _graph(findings):
    nodes, edges = {}, {}
    for f in findings:
        es = f["entities"]
        for e in es:
            nodes[e] = e.split(":", 1)[0]
        for i in range(len(es)):
            for j in range(i + 1, len(es)):
                k = tuple(sorted((es[i], es[j]))); edges[k] = edges.get(k, 0) + 1
    return {"nodes": [{"id": k, "type": v} for k, v in sorted(nodes.items())], "edges": [[a, b, w] for (a, b), w in sorted(edges.items())][:400]}


def assemble(result_dirs, out_dir, policy_path=None, actor="part5_assembler", log=print) -> dict:
    import numpy
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    pol = load_policy(policy_path); pol_hash = sha256_json(pol)
    key = load_or_create_key(out / "part5.key"); audit = GovAudit(out / "audit_part5.jsonl", key)
    arts = discover(result_dirs)
    if not arts:
        raise SystemExit("no Part 1-4 assurance_partN.json artifacts found under: " + ", ".join(map(str, result_dirs)))
    for a in arts:
        audit.append("ingest_artifact", actor, [f"part{a.part}:{a.scope}"], [a.sha256], pol_hash, detail={"path": a.path})
    norm = normalize(arts); sources = source_summaries(arts)
    audit.append("normalize_findings", actor, [f"part{a.part}:{a.scope}" for a in arts], [], pol_hash, detail={"findings": len(norm["findings"]), "unavailable": len(norm["unavailable"])})
    eng = Engine(norm, sources, arts, pol); cases = eng.run(set(norm["parts"]))
    for c in cases:
        c["trace"] = _trace(c["entities"])
    audit.append("correlate_and_score", actor, [], [c["case_id"] for c in cases], pol_hash, detail={"cases": len(cases), "rules": sorted({c["rule"] for c in cases})})
    parts = {}
    for p in (1, 2, 3, 4):
        ps = norm["parts"].get(p)
        if not ps:
            parts[str(p)] = {"disposition": "UNAVAILABLE", "reason": "no artifact supplied", "artifacts": 0}; continue
        parts[str(p)] = {"disposition": max_disp(ps["dispositions"], RANK), "level": (max(ps["levels"], key=lambda l: ["NORMAL", "SHIFT DETECTED", "SEVERE SHIFT"].index(l) if l in ["NORMAL", "SHIFT DETECTED", "SEVERE SHIFT"] else 0) if ps["levels"] else None),
                         "artifacts": len(ps["artifacts"]), "scopes": [a["scope"] for a in ps["artifacts"]], "finding_counts": ps["finding_counts"], "coverage": ps["coverage"]}
    counts = {}
    for f in norm["findings"]:
        counts[f["category"]] = counts.get(f["category"], 0) + 1
    analysis = {
        "schema_version": "1.0", "part": "5 - analyst assurance & governance", "detector_version": P5_VERSION, "created_at": now_iso(),
        "assessment_id": "A5-" + sha256_json([sorted(a.sha256 for a in arts), pol_hash])[:12], "parts": parts, "cases": cases, "sources": sources,
        "findings": norm["findings"], "finding_counts": counts, "unavailable": norm["unavailable"], "graph": _graph(norm["findings"]),
        "assumptions": ([f"Part 3/4 findings linked to the single supplied Part 2 model ({norm['single_model']}) as the served model"] if norm["single_model"] else []),
        "policy": pol, "policy_sha256": pol_hash,
        "reproducibility": {"inputs": [{"part": a.part, "scope": a.scope, "path": a.path, "sha256": a.sha256} for a in arts], "policy_sha256": pol_hash,
                            "python": sys.version.split()[0], "numpy": numpy.__version__, "platform": platform.platform(), "network_access_required": False, "training_performed": False,
                            "rerun": "python -m cv_assurance part5 assemble --results <same dirs> --out <dir> [--policy <same policy>] -> identical case_ids and evidence_hashes"},
        "limitations": LIMITATIONS, "honest_claims": HONEST_CLAIMS,
    }
    (out / "part5_analysis.json").write_text(json.dumps(analysis, indent=1, default=str), encoding="utf-8")
    return refresh(out, actor=actor, log=log)


def refresh(out_dir, actor="part5_assembler", log=print) -> dict:
    """Fold analyst decisions from the audit log, verify the trail, write JSON + HTML, seal (HMAC) the export."""
    from .report import render_html
    out = Path(out_dir); an = json.loads((out / "part5_analysis.json").read_text())
    key = load_or_create_key(out / "part5.key"); audit = GovAudit(out / "audit_part5.jsonl", key)
    gov = fold(an["cases"], audit.events)
    pd = {k: v["disposition"] for k, v in an["parts"].items() if v["disposition"] != "UNAVAILABLE"}
    ov = overall(an["cases"], gov, pd)
    gaps = [k for k, v in an["parts"].items() if v["disposition"] == "UNAVAILABLE"] + sorted({str(u["part"]) for u in an["unavailable"]})
    ov["coverage"] = "PARTIAL" if gaps else "FULL"; ov["coverage_gaps"] = sorted(set(gaps))
    ov["why"] = [f"{c['case_id']} [{c['suggested_disposition']}] {c['title']}" for c in an["cases"] if RANK[c["suggested_disposition"]] >= 2][:8] or ["no case reached REVIEW under the executed tests"]
    ver = GovAudit.verify(out / "audit_part5.jsonl", key)
    final = {**an, "overall": ov, "governance": {**gov, "actions": ["accept", "review", "quarantine", "retest", "close"]},
             "audit": {"path": "audit_part5.jsonl", "verify": ver, "events_at_render": len(audit.events)}, "refreshed_at": now_iso()}
    (out / "assurance_part5.json").write_text(json.dumps(final, indent=1, default=str))
    (out / "report_part5.html").write_text(render_html(final), encoding="utf-8")
    audit.append("export_sealed", actor, [an["assessment_id"]], [sha256_file(out / "assurance_part5.json"), sha256_file(out / "report_part5.html")], an["policy_sha256"], detail={"overall": ov["governance_adjusted_disposition"]})
    man = {"assessment_id": an["assessment_id"], "sealed_at": now_iso(), "assurance_part5_json_sha256": sha256_file(out / "assurance_part5.json"),
           "report_part5_html_sha256": sha256_file(out / "report_part5.html"), "audit_head": audit.events[-1]["event_hash"], "audit_events": len(audit.events)}
    man["hmac_sha256"] = hmac.new(key, sha256_json(man).encode(), hashlib.sha256).hexdigest()
    (out / "export_manifest.json").write_text(json.dumps(man, indent=1), encoding="utf-8")
    log(f"[p5] {an['assessment_id']} system={ov['system_disposition']} governance-adjusted={ov['governance_adjusted_disposition']} cases={len(an['cases'])} coverage={ov['coverage']} audit={'OK' if ver['ok'] else 'FAIL'}")
    return final


def act(out_dir, case_id, action, actor, rationale="", override=False) -> dict:
    out = Path(out_dir); an = json.loads((out / "part5_analysis.json").read_text())
    case = next((c for c in an["cases"] if c["case_id"] == case_id), None)
    if case is None:
        raise GovernanceError(f"unknown case '{case_id}'")
    validate(case, action, rationale, override, an["policy"])
    audit = GovAudit(out / "audit_part5.jsonl", load_or_create_key(out / "part5.key"))
    audit.append(f"analyst_{action}", actor, [case_id], [], an["policy_sha256"], detail={"case_id": case_id, "action": action, "rationale": rationale, "override": bool(override), "evidence_hash": case["evidence_hash"]})
    return refresh(out, actor=actor, log=lambda *a: None)


def apply_actions(out_dir, actions_file) -> list:
    res = []
    for a in json.loads(Path(actions_file).read_text())["actions"]:
        try:
            act(out_dir, a["case_id"], a["action"], a.get("actor", "analyst"), a.get("rationale", ""), a.get("override", False)); res.append((a["case_id"], a["action"], "recorded"))
        except GovernanceError as e:
            res.append((a["case_id"], a["action"], f"REJECTED: {e}"))
    return res


def verify_export(out_dir) -> dict:
    out = Path(out_dir); key = load_or_create_key(out / "part5.key"); v = GovAudit.verify(out / "audit_part5.jsonl", key)
    man = json.loads((out / "export_manifest.json").read_text()); sig = man.pop("hmac_sha256")
    checks = {"audit_chain_and_signatures": v["ok"], "manifest_signature": hmac.compare_digest(sig, hmac.new(key, sha256_json(man).encode(), hashlib.sha256).hexdigest()),
              "json_matches_manifest": sha256_file(out / "assurance_part5.json") == man["assurance_part5_json_sha256"],
              "html_matches_manifest": sha256_file(out / "report_part5.html") == man["report_part5_html_sha256"], "audit_head_matches_manifest": v.get("head") == man["audit_head"]}
    an = json.loads((out / "assurance_part5.json").read_text())
    checks["inputs_unchanged"] = all(Path(i["path"]).exists() and sha256_file(i["path"]) == i["sha256"] for i in an["reproducibility"]["inputs"])
    return {"ok": all(checks.values()), "checks": checks, "audit": v}
