"""Module 3 - Risk / Severity. Transparent rules only: every number can be traced to a weight in the policy."""
from __future__ import annotations

import math

SEV_ORDER = {"low": 1, "medium": 2, "high": 3, "critical": 4}


def finding_score(f: dict, pol: dict) -> float:
    if f["disposition"] in ("ACCEPT", "UNAVAILABLE"):
        return 0.0
    cw = pol["category_weight"].get(f["category"], pol["default_category_weight"])
    return round(pol["severity_weight"].get(f["severity"], 0.15) * float(f["confidence"]) * cw, 4)


def groups(findings: list, pol: dict) -> list:
    """Same (part, scope, category, severity) = ONE evidence group: max score + small volume bonus."""
    g = {}
    for f in findings:
        s = finding_score(f, pol)
        if s <= 0:
            continue
        g.setdefault((f["part"], f["scope"], f["category"], f["severity"]), []).append(s)
    vb = pol["group_volume_bonus"]; out = []
    for (part, scope, cat, sev), sc in sorted(g.items()):
        bonus = min(vb["cap"], vb["per_log10"] * math.log10(len(sc))) if len(sc) > 1 else 0.0
        out.append({"part": part, "scope": scope, "category": cat, "severity": sev, "count": len(sc), "max_finding_score": max(sc),
                    "volume_bonus": round(bonus, 4), "group_score": round(min(0.99, max(sc) + bonus), 4)})
    return out


def case_priority(members: list, pol: dict) -> dict:
    """Review priority (NOT a maliciousness score): noisy-OR over evidence groups + independence bonus per extra part."""
    gs = groups(members, pol)
    p_none = 1.0
    for g in gs:
        p_none *= (1.0 - g["group_score"])
    base = 1.0 - p_none
    parts = sorted({m["part"] for m in members if finding_score(m, pol) > 0})
    ib = pol["independence_bonus"]
    bonus = min(ib["cap"], ib["per_extra_part"] * max(0, len(parts) - 1))
    score = round(min(0.99, base + bonus), 3)
    b = pol["priority_bands"]
    band = "HIGH" if score >= b["HIGH"] else "ELEVATED" if score >= b["ELEVATED"] else "MODERATE" if score >= b["MODERATE"] else "LOW"
    return {"score": score, "band": band, "independent_parts": parts, "base_noisy_or": round(base, 3), "independence_bonus": round(bonus, 3),
            "contributions": sorted(gs, key=lambda x: -x["group_score"])[:8],
            "method": "noisy-OR of evidence-group scores (severity x confidence x category weight, +log-volume bonus) + independence bonus per extra Part"}


def max_disp(disps, rank) -> str:
    return max(disps, key=lambda d: rank.get(d, 0)) if disps else "ACCEPT"


def max_sev(members) -> str:
    return max((m["severity"] for m in members), key=lambda s: SEV_ORDER.get(s, 0), default="low")
