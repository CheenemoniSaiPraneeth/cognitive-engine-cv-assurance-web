"""Module 6 - temporal / batch drift: onset, pattern (step / gradual / transient / recurring), CUSUM early-warning."""
from __future__ import annotations
import numpy as np
from scipy.stats import spearmanr


def analyse(batches, cfg):
    """batches: ordered list of dict(batch_id, ratio, alert, null_median_ratio, families(list of flagged families))."""
    t = cfg["temporal"]; r = np.array([b["ratio"] for b in batches], float); al = np.array([b["alert"] for b in batches], bool); n = len(r)
    if n < 2:
        return {"status": "UNAVAILABLE", "reason": "need >= 2 batches/windows for temporal drift analysis", "ratios": r.tolist()}
    k = float(np.median([b["null_median_ratio"] for b in batches])); c = 0.0; cus = []
    for x in r: c = max(0.0, c + (x - k)); cus.append(round(c, 3))
    cus_alert = next((i for i, v in enumerate(cus) if v > t["cusum_h"]), None)
    persist = t["persist"]; onset = None
    for i in range(n):
        if al[i] and (al[i:i + persist].sum() >= min(persist, n - i)) and (i + persist <= n or al[i:].all()): onset = i; break
    pattern = "none"
    if onset is not None:
        post = r[onset:]; pre = r[:onset]
        rho = spearmanr(np.arange(len(post)), post)[0] if len(post) >= 4 else 0.0
        if al[onset:].all():
            pattern = "gradual" if (len(post) >= 4 and rho > 0.7 and post[0] < 0.6 * post.max()) else "step"
        else:
            gaps = np.diff(np.where(al[onset:])[0]); pattern = "recurring" if (gaps > 1).sum() >= 1 and al[onset:].sum() >= 3 else "step_then_recovery"
    elif al.any(): pattern = "transient"
    elif cus_alert is not None: pattern = "gradual_subthreshold"
    rise = None
    if onset is not None or cus_alert is not None:
        ref_i = onset if onset is not None else cus_alert; j = ref_i
        while j > 0 and cus[j - 1] > 0: j -= 1
        rise = j                                   # start of the cumulative rise (earlier than the alert for gradual drift)
    aff = sorted({f for i, b in enumerate(batches) if al[i] for f in b["families"]})
    return {"status": "OK", "ratios": [round(x, 3) for x in r], "alerts": al.tolist(), "cusum": cus, "cusum_alert_batch": cus_alert,
            "onset_index": onset, "onset_batch": batches[onset]["batch_id"] if onset is not None else None,
            "drift_start_index": rise, "pattern": pattern, "affected_batches": [b["batch_id"] for i, b in enumerate(batches) if al[i]],
            "affected_families": aff}
