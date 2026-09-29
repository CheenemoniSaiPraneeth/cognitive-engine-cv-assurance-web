"""Modules 4+5 - reference test battery and behavioural fingerprint (black-box safe, zero-training)."""
from __future__ import annotations

import io

import numpy as np
from PIL import Image, ImageFilter

from ..common.hashing import sha256_bytes, sha256_json

# (family, variant name, param)
VARIANTS = [("normal", "identity", None),
            ("brightness", "x0.7", 0.7), ("brightness", "x1.3", 1.3),
            ("contrast", "x0.6", 0.6), ("contrast", "x1.4", 1.4),
            ("compression", "jpeg30", 30), ("compression", "jpeg60", 60),
            ("resize", "down0.5", 0.5), ("crop", "center0.8", 0.8),
            ("noise", "sigma8", 8), ("noise", "sigma16", 16),
            ("blur", "r1", 1), ("blur", "r2", 2),
            ("occlusion", "grey_centre_20pct", 0.20), ("occlusion", "grey_corner_15pct", 0.15)]


def apply_variant(img, fam, p, idx):
    if fam == "normal": return img
    im = Image.fromarray(img)
    H, W = img.shape[:2]
    if fam == "brightness": return np.clip(img.astype(np.float32) * p, 0, 255).astype(np.uint8)
    if fam == "contrast":
        m = img.astype(np.float32).mean()
        return np.clip((img.astype(np.float32) - m) * p + m, 0, 255).astype(np.uint8)
    if fam == "compression":
        b = io.BytesIO(); im.save(b, "JPEG", quality=p); return np.asarray(Image.open(io.BytesIO(b.getvalue())).convert("RGB"))
    if fam == "resize":
        return np.asarray(im.resize((int(W * p), int(H * p)), Image.BILINEAR).resize((W, H), Image.BILINEAR))
    if fam == "crop":
        cw, ch = int(W * p), int(H * p); x0, y0 = (W - cw) // 2, (H - ch) // 2
        return np.asarray(im.crop((x0, y0, x0 + cw, y0 + ch)).resize((W, H), Image.BILINEAR))
    if fam == "noise":
        rng = np.random.default_rng(1000 + idx)  # fixed seeds -> reproducible battery
        return np.clip(img.astype(np.float32) + rng.normal(0, p, img.shape), 0, 255).astype(np.uint8)
    if fam == "blur": return np.asarray(im.filter(ImageFilter.GaussianBlur(p)))
    if fam == "occlusion":
        s = int(np.sqrt(p) * min(H, W)); out = img.copy()
        if "centre" in fam or p == 0.20:
            x0, y0 = (W - s) // 2, (H - s) // 2
        else:
            x0, y0 = 0, 0
        out[y0:y0 + s, x0:x0 + s] = 128
        return out
    raise ValueError(fam)


def _entropy(P): return float((-(P * np.log(P + 1e-12)).sum(1)).mean())


def run_battery(images, adapters: dict):
    """Run every variant through every adapter. Returns {name: {variant_key: probs}} + variant list."""
    out = {n: {} for n in adapters}
    for fam, vn, p in VARIANTS:
        arr = [apply_variant(im, fam, p, i) for i, im in enumerate(images)]
        for n, a in adapters.items():
            out[n][f"{fam}:{vn}"] = a.predict(arr)
    return out


def fingerprint(probs_by_variant: dict, determinism_diff: float, n_classes: int):
    per = {}
    for k, P in probs_by_variant.items():
        am = P.argmax(1)
        per[k] = {"class_frequency": [round(float((am == c).mean()), 4) for c in range(n_classes)],
                  "mean_confidence": round(float(P.max(1).mean()), 4), "mean_entropy": round(_entropy(P), 4)}
    pred_hash = sha256_bytes(np.concatenate([P.argmax(1).astype(np.int16) for P in probs_by_variant.values()]).tobytes())
    prob_hash = sha256_bytes(np.round(np.concatenate([P.ravel() for P in probs_by_variant.values()]), 3).astype(np.float32).tobytes())
    return {"prediction_hash": pred_hash, "rounded_probability_hash": prob_hash, "deterministic_max_abs_diff": determinism_diff,
            "per_variant": per}


def js_div(p, q):
    p, q = np.asarray(p) + 1e-9, np.asarray(q) + 1e-9
    p, q = p / p.sum(), q / q.sum(); m = 0.5 * (p + q)
    return float(0.5 * (p * np.log(p / m)).sum() + 0.5 * (q * np.log(q / m)).sum())


def compare_to_reference(P_recv: dict, P_ref: dict):
    rows = {}
    for k in P_recv:
        a, b = P_recv[k], P_ref[k]
        n = a.shape[1]
        fa = np.bincount(a.argmax(1), minlength=n) / len(a); fb = np.bincount(b.argmax(1), minlength=n) / len(b)
        rows[k] = {"agreement": round(float((a.argmax(1) == b.argmax(1)).mean()), 4),
                   "mean_abs_prob_diff": round(float(np.abs(a - b).mean()), 5),
                   "class_freq_js": round(js_div(fa, fb), 5)}
    return rows


def accuracy(P_by_variant, labels):
    if labels is None: return None
    y = np.asarray(labels)
    return {k: round(float((P.argmax(1) == y).mean()), 4) for k, P in P_by_variant.items()}
