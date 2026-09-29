"""Default configuration. All thresholds live here and are recorded in the report."""
from __future__ import annotations

import copy
import json
from pathlib import Path

DEFAULT_CONFIG = {
    "seed": 0,
    "workers": 4,
    "embedding": {
        "backend": "handcrafted",   # "handcrafted" (no weights needed) | "onnx" (frozen pretrained encoder)
        "onnx_path": None,
        "input_size": 224,
    },
    "duplicates": {
        "phash_max_dist": 6,
        "dhash_max_dist": 10,
        "semantic_min_cosine": 0.97,
        "review_min_group_size": 5,
        "flood_min_group_size": 8,
        "flood_min_fraction": 0.02,
    },
    "label_integrity": {
        "k": 10,
        "min_neighbours": 5,
        "min_dominant_share": 0.6,
        "max_assigned_share": 0.3,
        "kmeans_cluster_size": 15,
    },
    "ood": {
        "k": 10,
        "z_threshold": 3.5,
        "min_votes": 2,
        "pca_components": 32,
        "min_samples": 30,
    },
    "trigger": {
        "patch": 8,
        "stride": 2,
        "min_patch_std": 30.0,
        "min_support": 5,
        "min_support_fraction": 0.01,
        "min_position_ratio": 10.0,
        "max_pixel_std": 18.0,
        "merge_jaccard": 0.5,
        "max_candidates": 20,
    },
    "sources": {
        "min_count": 3,
        "min_abs_rate": 0.05,
        "min_rate_ratio": 2.0,
        "quarantine_cross_signals": 3,
    },
}


def _merge(base: dict, over: dict) -> dict:
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _merge(base[k], v)
        else:
            base[k] = v
    return base


def load_config(path=None, overrides: dict | None = None) -> dict:
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    if path:
        _merge(cfg, json.loads(Path(path).read_text()))
    if overrides:
        _merge(cfg, overrides)
    return cfg
