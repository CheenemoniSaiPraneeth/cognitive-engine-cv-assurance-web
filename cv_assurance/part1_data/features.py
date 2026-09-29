"""Embeddings WITHOUT training.

Two backends:
  * handcrafted (default): colour histogram + spatial colour grid + HOG-lite gradient histograms.
    Deterministic, needs no weights, works fully offline.
  * onnx: a FROZEN pretrained encoder loaded from a local .onnx file (optional, stronger
    semantics). Parameters are never updated.
"""
from __future__ import annotations

import numpy as np
from PIL import Image


def handcrafted_features(rgb64: np.ndarray) -> np.ndarray:
    """rgb64: (64,64,3) float32 in [0,1] -> 1D feature vector (240 dims)."""
    q = np.minimum((rgb64 * 4).astype(int), 3)
    idx = q[..., 0] * 16 + q[..., 1] * 4 + q[..., 2]
    hist = np.bincount(idx.ravel(), minlength=64).astype(np.float32)
    hist = np.sqrt(hist / hist.sum())
    grid = rgb64.reshape(4, 16, 4, 16, 3).mean(axis=(1, 3)).ravel()  # 48
    gray = rgb64 @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
    gy, gx = np.gradient(gray)
    mag = np.sqrt(gx ** 2 + gy ** 2)
    ang = (np.arctan2(gy, gx) % np.pi) / np.pi * 8
    b = np.minimum(ang.astype(int), 7)
    hog = np.zeros((4, 4, 8), dtype=np.float32)
    for cy in range(4):
        for cx in range(4):
            sl = (slice(cy * 16, cy * 16 + 16), slice(cx * 16, cx * 16 + 16))
            hog[cy, cx] = np.bincount(b[sl].ravel(), weights=mag[sl].ravel(), minlength=8)
    hog = hog.reshape(16, 8)
    hog = hog / (np.linalg.norm(hog, axis=1, keepdims=True) + 1e-6)
    return np.concatenate([hist, grid, hog.ravel()]).astype(np.float32)


class OnnxEncoder:
    """Frozen pretrained encoder loaded from a local ONNX file (never trained)."""

    def __init__(self, path: str, size: int = 224):
        import onnxruntime as ort  # optional dependency
        self.path, self.size = path, size
        self.sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        self.inp = self.sess.get_inputs()[0].name
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def embed(self, rgb: Image.Image) -> np.ndarray:
        a = np.asarray(rgb.resize((self.size, self.size), Image.BICUBIC), dtype=np.float32) / 255.0
        a = ((a - self.mean) / self.std).transpose(2, 0, 1)[None].astype(np.float32)
        out = self.sess.run(None, {self.inp: a})[0]
        out = np.asarray(out)
        if out.ndim == 4:
            out = out.mean(axis=(2, 3))
        return out.reshape(-1).astype(np.float32)


def build_encoder(cfg: dict):
    e = cfg["embedding"]
    if e["backend"] == "onnx":
        if not e.get("onnx_path"):
            raise ValueError("embedding.backend=onnx requires embedding.onnx_path")
        return OnnxEncoder(e["onnx_path"], e.get("input_size", 224))
    return None


def fit_normalizer(E: np.ndarray) -> dict:
    """Unsupervised statistics only (mean/std); no learned model."""
    mu = E.mean(0)
    sd = E.std(0)
    floor = np.percentile(sd, 10) + 1e-6
    return {"mu": mu, "sd": np.maximum(sd, floor)}


def apply_normalizer(E: np.ndarray, st: dict) -> np.ndarray:
    Z = np.clip((E - st["mu"]) / st["sd"], -5, 5)
    return Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-9)
