"""Part 4 configuration. Every threshold is recorded in the report (reproducibility)."""
from __future__ import annotations
import copy, json
from pathlib import Path

DEFAULT_P4 = {
    "seed": 0, "workers": 4, "window": 50,          # default batch size when the ops data has no batch ids
    "min_batch": 20,                                 # smaller batches -> UNAVAILABLE (not enough samples for calibrated tests)
    "calibration": {"draws": 240, "alpha_feature": 0.01, "alpha_global": 0.01},
    "embedding": {"k": 5, "clusters": 8, "max_points": 200},
    "ood": {"k": 5, "z": 3.5, "min_votes": 2, "pca": 16, "new_cluster_min": 4},
    "severity": {"severe_ratio": 3.0, "severe_min_families": 2},
    "temporal": {"persist": 2, "cusum_h": 3.0},
    "attribution": {"abnormal_z": 2.0},
}

def _merge(b, o):
    for k, v in o.items():
        if isinstance(v, dict) and isinstance(b.get(k), dict): _merge(b[k], v)
        else: b[k] = v
    return b

def load_p4_config(path=None, overrides=None):
    c = copy.deepcopy(DEFAULT_P4)
    if path: _merge(c, json.loads(Path(path).read_text()))
    if overrides: _merge(c, overrides)
    return c
