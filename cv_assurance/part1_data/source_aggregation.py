"""Module 8 - Source / contributor aggregation. Reports evidence concentration, not intent."""
from __future__ import annotations

from collections import Counter

DETECTOR = "source_aggregation_v1"
SIGNALS = {"duplicate": "duplicate_group", "label_anomaly": "label_anomaly",
           "ood": "ood_anomaly", "trigger_candidate": "trigger_candidate",
           "structural": "structural_error"}


def aggregate_sources(samples, flags, cfg, store):
    c = cfg["sources"]
    n = len(samples)
    summaries = {}
    global_rate = {sig: sum(1 for s in samples if cat in flags.get(s.sample_id, ())) / max(n, 1)
                   for sig, cat in SIGNALS.items()}
    global_cnt = {sig: sum(1 for s in samples if cat in flags.get(s.sample_id, ())) for sig, cat in SIGNALS.items()}
    for key in ("source_id", "contributor_id", "batch_id"):
        groups = {}
        for s in samples:
            v = getattr(s, key)
            if v:
                groups.setdefault(v, []).append(s)
        if not groups:
            continue
        for val, ss in sorted(groups.items()):
            cnt = len(ss)
            rates, elevated, cats = {}, [], Counter()
            for sig, cat in SIGNALS.items():
                k = sum(1 for s in ss if cat in flags.get(s.sample_id, ()))
                rates[sig] = round(k / cnt, 4)
                cats[sig] = k
                thr = max(c["min_abs_rate"], c["min_rate_ratio"] * global_rate[sig])
                if k >= c["min_count"] and k / cnt >= thr:
                    elevated.append(sig)
            n_cross = len(elevated)
            if n_cross >= c["quarantine_cross_signals"]:
                disp = "QUARANTINE"
            elif n_cross >= 1:
                disp = "REVIEW"
            else:
                disp = "ACCEPT"
            name = f"{key}:{val}"
            summaries[name] = {
                "sample_count": cnt, "duplicate_rate": rates["duplicate"],
                "label_anomaly_rate": rates["label_anomaly"], "ood_rate": rates["ood"],
                "trigger_candidate_rate": rates["trigger_candidate"], "structural_error_rate": rates["structural"],
                "anomaly_categories": {k: v for k, v in cats.items() if v},
                "share_of_all_flagged": {sig: round(cats[sig] / max(global_cnt[sig], 1), 3) for sig in cats if cats[sig]},
                "elevated_signals": elevated, "cross_signal_count": n_cross, "disposition": disp}
            if disp != "ACCEPT":
                sev = "high" if disp == "QUARANTINE" else "medium"
                store.add("source", name, "source_concentration", sev, min(0.5 + 0.15 * n_cross, 0.95),
                          DETECTOR, summaries[name], disp,
                          ["Concentration of evidence prioritises review; it does not establish intent."])
    return summaries
