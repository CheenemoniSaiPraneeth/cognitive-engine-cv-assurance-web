"""Part 5 policy: transparent, editable weights and thresholds (no hidden 'maliciousness score')."""
from __future__ import annotations

import copy
import json
from pathlib import Path

P5_VERSION = "part5-0.1.0"

RANK = {"ACCEPT": 0, "UNAVAILABLE": 1, "REVIEW": 2, "QUARANTINE": 3}
ACTIONS = ("accept", "review", "quarantine", "retest", "close")

DEFAULT_POLICY = {
    # evidence strength: severity x detector confidence x category weight -> finding score in [0,1]
    "severity_weight": {"low": 0.15, "medium": 0.40, "high": 0.70, "critical": 0.95},
    # categories that are context/informational rather than integrity evidence
    "category_weight": {
        "duplicate_pair_informational": 0.2, "identity_unverifiable": 0.2, "numerical_drift_within_tolerance": 0.1,
        "test_unavailable": 0.0,
    },
    "default_category_weight": 1.0,
    # many findings of the same kind are ONE evidence group: score = max + volume bonus (avoids volume dominating)
    "group_volume_bonus": {"per_log10": 0.05, "cap": 0.10},
    # independent parts pointing at the same entity raise review priority
    "independence_bonus": {"per_extra_part": 0.08, "cap": 0.25},
    "priority_bands": {"HIGH": 0.85, "ELEVATED": 0.50, "MODERATE": 0.25},
    # a lone statistical anomaly is capped at REVIEW; only converging evidence or a source-detector
    # QUARANTINE (hard integrity failure) may reach QUARANTINE
    "isolated_signal_max_disposition": "REVIEW",
    "isolated_priority_cap": 0.60,   # uncorroborated single-Part evidence never ranks as HIGH
    "min_rationale_chars": 15,
    "min_severity_for_residual_case": "medium",
    "max_findings_listed_per_case": 12,
    "max_findings_in_explorer": 400,
}


def load_policy(path=None) -> dict:
    pol = copy.deepcopy(DEFAULT_POLICY)
    if path:
        over = json.loads(Path(path).read_text())
        for k, v in over.items():
            if isinstance(v, dict) and isinstance(pol.get(k), dict):
                pol[k].update(v)
            else:
                pol[k] = v
    return pol
