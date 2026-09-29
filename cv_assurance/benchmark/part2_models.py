"""Part 2 benchmark FIXTURE builder (not part of the assessment).

Creates a trusted clean model plus controlled variants, using Part 1's synthetic shape images.
The fixture model is a random-feature MLP whose output layer is fitted in closed form (ridge) - this is only how the
*benchmark artifacts* are made; the Part 2 assessment itself never trains anything.

Variants (ground_truth.json says exactly what was changed):
  clean_copy          byte-identical copy                       -> should NOT flag
  reserialized        same weights, different metadata/bytes    -> hash differs, benign (low)
  finetuned_small     +1% weight noise (benign update)          -> should NOT flag (<= low)
  weight_tampered     large tamper of output layer              -> flag
  trojan_checker      checkerboard bottom-right -> class 'circle' (dormant unit overwritten, same architecture) -> flag
  trojan_stealth      magenta/cyan pattern at LEFT-MIDDLE (outside the generic scan library) -> white-box only
  substitute          different architecture + training set     -> flag
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

from .synthetic import CLASSES, SIZE, add_trigger, make_image
from ..part2_model.adapters import preprocess, softmax

TARGET = 0  # 'circle'
NORM = (0.5, 0.25)


def _gen(n, seed):
    rng = np.random.default_rng(seed)
    ims, ys = [], []
    for i in range(n):
        c = i % 4
        im, _ = make_image(c, rng)
        ims.append(np.asarray(im)); ys.append(c)
    return ims, np.array(ys)


def _fit(H, w_seed, d_seed, n=800, temp=8.0):
    ims, y = _gen(n, d_seed)
    X = preprocess(ims, 32, *NORM)
    r = np.random.default_rng(w_seed)
    W1 = (r.standard_normal((X.shape[1], H)) / np.sqrt(X.shape[1])).astype(np.float32)
    b1 = (r.standard_normal(H) * 0.1).astype(np.float32)
    Hh = np.maximum(X @ W1 + b1, 0)
    Hb = np.hstack([Hh, np.ones((n, 1), np.float32)])
    T = np.eye(4)[y]
    sol = np.linalg.solve(Hb.T @ Hb + 1.0 * np.eye(H + 1), Hb.T @ T)
    return {"W1": W1, "b1": b1, "W2": (sol[:-1] * temp).astype(np.float32), "b2": (sol[-1] * temp).astype(np.float32)}


def _save(p, path, tag, extra=None):
    meta = {"input_size": 32, "mean": NORM[0], "std": NORM[1], "class_names": CLASSES, "tag": tag, **(extra or {})}
    np.savez(path, meta=np.array(json.dumps(meta)), **p)


def _predict(p, ims):
    X = preprocess(ims, 32, *NORM)
    return softmax(np.maximum(X @ p["W1"] + p["b1"], 0) @ p["W2"] + p["b2"])


def _inject_trojan(p, template_fn, clean_ims, target=TARGET, gain_logit=25.0):
    q = {k: v.copy() for k, v in p.items()}
    infl = np.linalg.norm(q["W2"], axis=1)
    j = int(np.argmin(infl))                                   # least influential unit gets repurposed (same architecture)
    Xc = preprocess(clean_ims, 32, *NORM)
    Xt = preprocess([template_fn(im) for im in clean_ims], 32, *NORM)
    w = (Xt - Xc).mean(0)
    w = w - 0 * w.mean()
    pc, pt = Xc @ w, Xt @ w
    thr = pc.max() + 0.35 * (pt.min() - pc.max()) if pt.min() > pc.max() else pc.max() + 0.5 * pc.std()
    scale = 1.0 / max(1e-6, (pt.mean() - thr))
    q["W1"][:, j] = w * scale; q["b1"][j] = -thr * scale
    q["W2"][j, :] = 0.0; q["W2"][j, target] = gain_logit
    return q, j


def _stealth_template(im):
    out = im.copy()
    yy, xx = np.mgrid[0:10, 0:10]
    m = ((yy + xx) % 2 == 0)
    patch = np.zeros((10, 10, 3), np.uint8); patch[m] = (255, 0, 255); patch[~m] = (0, 255, 255)
    y0 = SIZE // 2 - 5
    out[y0:y0 + 10, 4:14] = patch                              # left-middle: not in the generic scan positions
    return out


def build_benchmark(out_dir, n_ref=120, seed=5):
    out = Path(out_dir); (out / "models").mkdir(parents=True, exist_ok=True)
    ref_dir = out / "reference_images"; ref_dir.mkdir(exist_ok=True)
    ims, ys = _gen(n_ref, 999)
    for i, im in enumerate(ims): Image.fromarray(im).save(ref_dir / f"ref_{i:03d}.png")
    (ref_dir / "labels.json").write_text(json.dumps({f"ref_{i:03d}.png": int(y) for i, y in enumerate(ys)}))

    A = _fit(64, 0, 11)
    m = out / "models"
    _save(A, m / "trusted_reference.npz", "trusted")
    shutil.copy(m / "trusted_reference.npz", m / "clean_copy.npz")
    _save(A, m / "reserialized.npz", "trusted", {"note": "re-exported, metadata changed"})
    r = np.random.default_rng(seed)
    FT = {k: (v + 0.01 * v.std() * r.standard_normal(v.shape)).astype(np.float32) for k, v in A.items()}
    _save(FT, m / "finetuned_small.npz", "finetuned")
    WT = {k: v.copy() for k, v in A.items()}
    WT["W2"] = WT["W2"] + 0.8 * WT["W2"].std() * r.standard_normal(WT["W2"].shape).astype(np.float32); WT["b2"][1] += 2.0
    _save(WT, m / "weight_tampered.npz", "tampered")
    cl = ims[:100]
    TR, j1 = _inject_trojan(A, lambda im: np.asarray(add_trigger(Image.fromarray(im))), cl)
    _save(TR, m / "trojan_checker.npz", "trojan")
    ST, j2 = _inject_trojan(A, _stealth_template, cl)
    _save(ST, m / "trojan_stealth.npz", "trojan-stealth")
    SUB = _fit(48, 3, 22)
    _save(SUB, m / "substitute.npz", "substitute")

    gt = {"reference_model": "models/trusted_reference.npz", "reference_images": "reference_images", "target_class": TARGET,
          "models": {
              "clean_copy": {"compromised": False, "change": "none (byte-identical)"},
              "reserialized": {"compromised": False, "change": "same weights, metadata differs"},
              "finetuned_small": {"compromised": False, "change": "1% gaussian weight noise (benign update)"},
              "weight_tampered": {"compromised": True, "change": "large noise on fc2 + class-1 bias shift"},
              "trojan_checker": {"compromised": True, "change": f"unit {j1} overwritten: checkerboard bottom-right -> class {TARGET}", "blackbox_detectable": True},
              "trojan_stealth": {"compromised": True, "change": f"unit {j2} overwritten: magenta/cyan patch left-middle -> class {TARGET}",
                                 "blackbox_detectable": False, "note": "outside generic pattern library; expected white-box-only detection"},
              "substitute": {"compromised": True, "change": "different architecture (48 hidden) & training data"}}}
    (out / "ground_truth.json").write_text(json.dumps(gt, indent=2))
    return gt
