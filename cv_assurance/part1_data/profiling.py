"""Module 2 - Dataset profiling: the context needed to interpret later findings."""
from __future__ import annotations

from collections import Counter

import numpy as np


def _dist(v):
    v = np.asarray(v, dtype=float)
    if v.size == 0:
        return {}
    return {"min": float(v.min()), "p5": float(np.percentile(v, 5)), "median": float(np.median(v)),
            "mean": float(v.mean()), "p95": float(np.percentile(v, 95)), "max": float(v.max())}


def profile_dataset(samples, class_names, stats, issues) -> dict:
    n = len(samples)
    cls = Counter()
    boxes_per_img, areas_rel, cx, cy = [], [], [], []
    for s in samples:
        boxes_per_img.append(len(s.annotations))
        for a in s.annotations:
            cls[a.class_name] += 1
            if a.bbox and s.width and s.height:
                x, y, w, h = a.bbox
                areas_rel.append(w * h / (s.width * s.height))
                cx.append((x + w / 2) / s.width)
                cy.append((y + h / 2) / s.height)
    counts = list(cls.values())
    heat = np.zeros((3, 3), dtype=int)
    for a, b in zip(cx, cy):
        heat[min(int(b * 3), 2), min(int(a * 3), 2)] += 1
    widths = [s.width for s in samples if s.width]
    heights = [s.height for s in samples if s.height]
    aspect = [s.width / s.height for s in samples if s.width and s.height]
    src = {}
    for key in ("source_id", "contributor_id", "batch_id"):
        c = Counter(getattr(s, key) for s in samples if getattr(s, key))
        if c:
            src[key] = {k: {"count": v, "fraction": round(v / n, 4)} for k, v in sorted(c.items())}
    st = [x for x in stats if x]
    issue_kinds = Counter(i["kind"] for i in issues)
    return {
        "volume": {"images": n, "annotations": int(sum(boxes_per_img)),
                   "unlabelled_images": int(sum(1 for b in boxes_per_img if b == 0))},
        "classes": {"frequency": dict(cls),
                    "declared_classes": {str(k): v for k, v in class_names.items()},
                    "imbalance_ratio": (max(counts) / max(min(counts), 1)) if counts else None},
        "image_properties": {"formats": dict(Counter(s.format for s in samples if s.format)),
                             "width": _dist(widths), "height": _dist(heights), "aspect_ratio": _dist(aspect),
                             "brightness": _dist([x["brightness"] for x in st]),
                             "contrast": _dist([x["contrast"] for x in st]),
                             "sharpness": _dist([x["sharpness"] for x in st])},
        "annotations": {"boxes_per_image": _dist(boxes_per_img), "relative_box_area": _dist(areas_rel),
                        "centre_heatmap_3x3": heat.tolist()},
        "sources": src,
        "quality": {"unreadable_or_missing": int(sum(1 for s in samples if not s.readable)),
                    "issue_counts": dict(issue_kinds)},
    }
