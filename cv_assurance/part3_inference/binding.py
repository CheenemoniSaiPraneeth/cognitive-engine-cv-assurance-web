"""Module 7 - replay / binding checks across the ledger (old output presented as new, duplicated records, gaps, time)."""
from __future__ import annotations

from collections import defaultdict
from datetime import datetime


def _t(s):
    try: return datetime.fromisoformat(s)
    except Exception: return None


def check_binding(records: list, max_lag_s: float = 300.0, ts_regress_s: float = 2.0) -> list:
    """Returns list of dicts {record_id, category, severity, disposition, evidence}. Ledger-level issues use record_id='LEDGER'."""
    out = []
    seen_req, seen_nonce, seen_tuple = {}, {}, {}
    by_out = defaultdict(list)
    prev_seq, prev_ts, ids = None, None, set()
    for r in records:
        rid, seq = r.get("record_id"), r.get("seq")
        if rid in ids:
            out.append({"record_id": rid, "category": "duplicate_record_id", "severity": "high", "disposition": "QUARANTINE", "evidence": {"record_id": rid}})
        ids.add(rid)
        if prev_seq is not None and isinstance(seq, int) and seq != prev_seq + 1:
            missing = list(range(prev_seq + 1, seq)) if seq > prev_seq + 1 else []
            out.append({"record_id": "LEDGER", "category": "sequence_gap" if missing else "sequence_disorder", "severity": "high",
                        "disposition": "QUARANTINE", "evidence": {"after_seq": prev_seq, "next_seq": seq, "missing_seqs": missing}})
        prev_seq = seq if isinstance(seq, int) else prev_seq
        rq, nc = r.get("request_id"), r.get("nonce")
        if rq in seen_req:
            out.append({"record_id": rid, "category": "replay_request_id_reuse", "severity": "high", "disposition": "QUARANTINE",
                        "evidence": {"request_id": rq, "first_seen_in": seen_req[rq]}})
        elif nc in seen_nonce:
            out.append({"record_id": rid, "category": "replay_nonce_reuse", "severity": "high", "disposition": "QUARANTINE",
                        "evidence": {"nonce": nc, "first_seen_in": seen_nonce[nc]}})
        seen_req.setdefault(rq, rid); seen_nonce.setdefault(nc, rid)
        tup = (r.get("input_sha256"), r.get("model_sha256"), r.get("config_sha256"), r.get("output_sha256"))
        seen_tuple.setdefault(tup, rid)
        by_out[r.get("output_sha256")].append(r)
        # timing: inference must not precede capture; recording lag bounded; inference clock should not run backwards
        ti, tc, tr = _t(r.get("inference_timestamp")), _t((r.get("input") or {}).get("captured_at")), _t(r.get("recorded_at"))
        if ti and tc and ti < tc:
            out.append({"record_id": rid, "category": "timestamp_anomaly", "severity": "medium", "disposition": "REVIEW",
                        "evidence": {"issue": "inference before input capture", "captured_at": r["input"]["captured_at"], "inference_timestamp": r["inference_timestamp"]}})
        if ti and tr and (tr - ti).total_seconds() > max_lag_s:
            out.append({"record_id": rid, "category": "stale_record", "severity": "high", "disposition": "QUARANTINE",
                        "evidence": {"lag_seconds": round((tr - ti).total_seconds(), 1), "max_lag_s": max_lag_s,
                                     "note": "result recorded long after it was allegedly produced (possible replay)"}})
        if ti and prev_ts and (prev_ts - ti).total_seconds() > ts_regress_s:
            out.append({"record_id": rid, "category": "timestamp_anomaly", "severity": "medium", "disposition": "REVIEW",
                        "evidence": {"issue": "inference clock ran backwards", "previous": prev_ts.isoformat(), "this": r["inference_timestamp"]}})
        prev_ts = max(prev_ts, ti) if (prev_ts and ti) else (ti or prev_ts)
    # an identical (non-saturated) output attached to *different* inputs = output re-use, an old result presented as new
    for h, rs in by_out.items():
        if len({x.get("input_sha256") for x in rs}) > 1 and rs[0].get("output", {}).get("entropy", 0) > 1e-3:
            for x in rs[1:]:
                if x["input_sha256"] != rs[0]["input_sha256"]:
                    out.append({"record_id": x["record_id"], "category": "output_reuse_across_inputs", "severity": "medium", "disposition": "REVIEW",
                                "evidence": {"output_sha256": h, "first_seen_in": rs[0]["record_id"]}})
    return out
