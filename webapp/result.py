"""Turn the Part 5 (+ combined P1-P4) JSON artifacts into one structured API result. Every value is read from the artifacts."""
from __future__ import annotations

import json
import re
from pathlib import Path


def _load(p: Path):
    return json.loads(p.read_text()) if p.is_file() else None


def build_result(out_dir: Path, mode: str, logs: list[str], input_info: dict | None = None) -> dict:
    fin = _load(out_dir / "part5" / "assurance_part5.json")
    if fin is None:
        raise RuntimeError("Part 5 did not produce assurance_part5.json")
    comb = _load(out_dir / "p1234" / "assurance_combined_p1234.json") or {}
    ov, gov = fin["overall"], fin["governance"]
    ver = fin["audit"]["verify"]; reason = ver.get("reason", "")
    m = re.search(r"(\d+) HMAC signatures? valid", reason)
    findings_by_uid = {f["uid"]: f for f in fin["findings"]}

    parts = {}
    for k, v in fin["parts"].items():
        parts[f"p{k}"] = {"part": int(k), "disposition": v["disposition"], "level": v.get("level"), "scopes": v.get("scopes", []),
                          "artifacts": v.get("artifacts"), "finding_counts": v.get("finding_counts", {}), "coverage": v.get("coverage", [])}
    if "p1" not in parts and not fin["parts"]: pass
    parts["p5"] = {"part": 5, "disposition": ov["governance_adjusted_disposition"], "level": None, "scopes": [], "artifacts": None,
                   "finding_counts": {"cases": len(fin["cases"]), "open_cases": ov["open_cases"]}, "coverage": [],
                   "system_disposition": ov["system_disposition"]}

    cases = []
    for c in fin["cases"]:
        st = gov["case_state"].get(c["case_id"], {})
        sample = []
        for uid in c.get("finding_uids", [])[:12]:
            f = findings_by_uid.get(uid)
            if f: sample.append({"uid": uid, "part": f["part"], "scope": f["scope"], "category": f["category"], "severity": f["severity"],
                                 "confidence": f["confidence"], "asset": f.get("asset_id"), "evidence": f.get("evidence")})
        cases.append({"case_id": c["case_id"], "title": c["title"], "rule": c["rule"], "severity": c["severity"], "band": c["priority"]["band"],
                      "score": c["priority"]["score"], "parts": c["parts"], "disposition": c["suggested_disposition"], "status": st.get("status", "OPEN"),
                      "entities": c["entities"], "rationale": c["rationale"], "next_action": c["next_action"], "tags": c.get("tags", []),
                      "finding_count": c["finding_count"], "evidence_hash": c["evidence_hash"],
                      "contributions": c["priority"].get("contributions", [])[:8], "sample_findings": sample})

    return {
        "assessment_id": fin["assessment_id"], "mode": mode, "created_at": fin.get("created_at"),
        "system_status": ov["system_disposition"], "governance_status": ov["governance_adjusted_disposition"],
        "coverage": ov["coverage"], "coverage_gaps": ov.get("coverage_gaps", []), "open_cases": ov["open_cases"], "why": ov.get("why", []),
        "parts": parts, "cases": cases,
        "audit": {"status": "OK" if ver["ok"] else "BROKEN", "events": ver.get("events"), "chain_intact": bool(ver["ok"]) and "chain intact" in reason,
                  "hmac_valid": int(m.group(1)) if m else None, "signed": ver.get("signed"), "head": ver.get("head"), "reason": reason},
        "streams": comb.get("streams", []), "cross_signals": comb.get("cross_signals", []), "reference": comb.get("reference"),
        "unavailable": [{"part": u["part"], "test": u["test"], "impact": u["impact"]} for u in fin.get("unavailable", [])],
        "limitations": fin.get("limitations", []), "input": input_info or {}, "logs": logs,
    }
