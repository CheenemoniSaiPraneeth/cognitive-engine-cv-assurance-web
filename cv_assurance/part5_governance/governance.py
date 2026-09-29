"""Module 6 (governance half) - analyst decisions on cases. The audit log is the source of truth; state is folded from it."""
from __future__ import annotations

from .config import ACTIONS, RANK

STATUS = {"accept": "ACCEPTED", "review": "IN_REVIEW", "quarantine": "QUARANTINED", "retest": "RETEST_REQUESTED", "close": "CLOSED"}


class GovernanceError(ValueError):
    pass


def validate(case: dict, action: str, rationale: str, override: bool, pol: dict) -> None:
    if action not in ACTIONS:
        raise GovernanceError(f"unknown action '{action}' (choose from {', '.join(ACTIONS)})")
    if action in ("accept", "close") and len((rationale or "").strip()) < pol["min_rationale_chars"]:
        raise GovernanceError(f"'{action}' requires a rationale of at least {pol['min_rationale_chars']} characters (evidence for the closure)")
    if action == "accept" and case["suggested_disposition"] == "QUARANTINE" and not override:
        raise GovernanceError("case is suggested QUARANTINE; accepting it needs an explicit override (recorded in the audit trail)")


def fold(cases: list, events: list) -> dict:
    """Replay analyst events -> per-case state. A decision made against different evidence is flagged STALE and does not suppress the case."""
    byid = {c["case_id"]: c for c in cases}
    state = {c["case_id"]: {"status": "OPEN", "history": [], "stale": False, "last": None} for c in cases}
    orphan = []
    for ev in events:
        if not ev["action"].startswith("analyst_"):
            continue
        d = ev.get("detail") or {}; cid = d.get("case_id")
        if cid not in byid:
            orphan.append({"case_id": cid, "action": ev["action"], "event_id": ev["event_id"]}); continue
        rec = {"event_id": ev["event_id"], "timestamp": ev["timestamp"], "actor": ev["actor"], "action": d.get("action"), "rationale": d.get("rationale", ""),
               "override": bool(d.get("override")), "evidence_hash": d.get("evidence_hash")}
        s = state[cid]; s["history"].append(rec); s["last"] = rec
        s["stale"] = rec["evidence_hash"] != byid[cid]["evidence_hash"]
        s["status"] = "OPEN (decision stale: evidence changed)" if s["stale"] else STATUS[rec["action"]]
    return {"case_state": state, "orphan_decisions": orphan}


def effective_disposition(case: dict, st: dict) -> str:
    last = st.get("last")
    if not last or st.get("stale"):
        return case["suggested_disposition"]
    a = last["action"]
    if a in ("accept", "close"):
        return "ACCEPT"
    if a == "quarantine":
        return "QUARANTINE"
    return case["suggested_disposition"]


def overall(cases: list, gov: dict, part_dispositions: dict) -> dict:
    sysd = max([c["suggested_disposition"] for c in cases] + list(part_dispositions.values()) + ["ACCEPT"], key=lambda d: RANK.get(d, 0))
    eff = [effective_disposition(c, gov["case_state"][c["case_id"]]) for c in cases]
    # part-level hard failures stay visible unless every case explaining them has been closed by an analyst
    adj = max(eff + ["ACCEPT"], key=lambda d: RANK.get(d, 0))
    open_n = sum(1 for c, e in zip(cases, eff) if e != "ACCEPT" and RANK.get(c["suggested_disposition"], 0) >= 2)
    return {"system_disposition": sysd, "governance_adjusted_disposition": adj, "open_cases": open_n}
