"""Deterministic feature extraction (no training): scalar image-quality/geometry/colour/compression features
+ a handcrafted embedding (or a FROZEN pretrained ONNX encoder if supplied)."""
from __future__ import annotations
import io
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
from PIL import Image

FAMILY = {
    "brightness": "illumination", "contrast": "illumination", "extreme_fraction": "illumination", "tonal_entropy": "illumination",
    "mean_r": "color", "mean_g": "color", "mean_b": "color", "saturation": "color",
    "width": "geometry", "height": "geometry", "aspect": "geometry",
    "sharpness": "sharpness_noise", "noise_est": "sharpness_noise", "edge_density": "sharpness_noise",
    "bytes_per_pixel": "compression", "blockiness": "compression", "is_jpeg": "compression",
    "out_confidence": "task_output", "out_entropy": "task_output",
}
IMAGE_FEATURES = [k for k in FAMILY if not k.startswith("out_")]
TASK_FEATURES = ["out_confidence", "out_entropy"]
ALL_FEATURES = IMAGE_FEATURES + TASK_FEATURES
IMG_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def handcrafted_embedding(rgb64):
    q = np.minimum((rgb64 * 4).astype(int), 3); idx = q[..., 0] * 16 + q[..., 1] * 4 + q[..., 2]
    hist = np.bincount(idx.ravel(), minlength=64).astype(np.float32); hist = np.sqrt(hist / hist.sum())
    grid = rgb64.reshape(4, 16, 4, 16, 3).mean(axis=(1, 3)).ravel()
    gray = rgb64 @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
    gy, gx = np.gradient(gray); mag = np.sqrt(gx ** 2 + gy ** 2)
    b = np.minimum(((np.arctan2(gy, gx) % np.pi) / np.pi * 8).astype(int), 7)
    hog = np.zeros((4, 4, 8), np.float32)
    for cy in range(4):
        for cx in range(4):
            sl = (slice(cy * 16, cy * 16 + 16), slice(cx * 16, cx * 16 + 16))
            hog[cy, cx] = np.bincount(b[sl].ravel(), weights=mag[sl].ravel(), minlength=8)
    hog = hog.reshape(16, 8); hog = hog / (np.linalg.norm(hog, axis=1, keepdims=True) + 1e-6)
    return np.concatenate([hist, grid, hog.ravel()]).astype(np.float32)


class OnnxEncoder:
    """Frozen pretrained encoder from a LOCAL .onnx file. Never trained / fine-tuned."""
    def __init__(self, path, size=224):
        import onnxruntime as ort
        self.size = size; self.sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"]); self.inp = self.sess.get_inputs()[0].name
        self.mean = np.array([0.485, 0.456, 0.406], np.float32); self.std = np.array([0.229, 0.224, 0.225], np.float32)
    def embed(self, rgb):
        a = np.asarray(rgb.resize((self.size, self.size), Image.BICUBIC), np.float32) / 255.0
        o = np.asarray(self.sess.run(None, {self.inp: ((a - self.mean) / self.std).transpose(2, 0, 1)[None].astype(np.float32)})[0])
        return (o.mean(axis=(2, 3)) if o.ndim == 4 else o).reshape(-1).astype(np.float32)


def extract(path, encoder=None):
    out = {"error": None}
    try:
        data = Path(path).read_bytes(); im = Image.open(io.BytesIO(data)); fmt = im.format; im.load()
        rgb = im.convert("RGB"); w, h = rgb.size
        cap = rgb if max(w, h) <= 512 else rgb.resize((512, max(1, int(512 * h / w))) if w >= h else (max(1, int(512 * w / h)), 512))
        g = np.asarray(cap.convert("L"), dtype=np.float32)
        rgb64 = np.asarray(rgb.resize((64, 64), Image.BILINEAR), np.float32) / 255.0
        lap = g[1:-1, 1:-1] * 4 - g[:-2, 1:-1] - g[2:, 1:-1] - g[1:-1, :-2] - g[1:-1, 2:]
        resid = g[1:-1, 1:-1] - (g[:-2, 1:-1] + g[2:, 1:-1] + g[1:-1, :-2] + g[1:-1, 2:]) / 4
        gy, gx = np.gradient(g); mag = np.hypot(gx, gy)
        dx = np.abs(np.diff(g, axis=1)); blk = float(dx[:, 7::8].mean() / (dx.mean() + 1e-6)) if dx.shape[1] > 16 else 1.0
        hist = np.bincount(np.clip(g, 0, 255).astype(np.uint8).ravel(), minlength=256) / g.size; nz = hist[hist > 0]
        out["scalars"] = {
            "brightness": float(g.mean()), "contrast": float(g.std()), "extreme_fraction": float(((g < 5) | (g > 250)).mean()),
            "tonal_entropy": float(-(nz * np.log2(nz)).sum()),
            "mean_r": float(rgb64[..., 0].mean()), "mean_g": float(rgb64[..., 1].mean()), "mean_b": float(rgb64[..., 2].mean()),
            "saturation": float(((rgb64.max(-1) - rgb64.min(-1)) / (rgb64.max(-1) + 1e-6)).mean()),
            "width": float(w), "height": float(h), "aspect": float(w / h),
            "sharpness": float(lap.var()), "noise_est": float(np.median(np.abs(resid - np.median(resid))) * 1.4826),
            "edge_density": float((mag > 20).mean()),
            "bytes_per_pixel": float(len(data) / (w * h)), "blockiness": blk, "is_jpeg": float(fmt == "JPEG"),
        }
        out["emb"] = encoder.embed(rgb) if encoder is not None else handcrafted_embedding(rgb64)
        out["format"] = fmt
    except Exception as e:
        out["error"] = f"{type(e).__name__}: {e}"
    return out


def extract_many(paths, encoder=None, workers=4):
    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        return list(ex.map(lambda p: extract(p, encoder), paths))


def fit_normalizer(E):
    sd = E.std(0); return {"mu": E.mean(0), "sd": np.maximum(sd, np.percentile(sd, 10) + 1e-6)}

def apply_normalizer(E, st):
    Z = np.clip((E - st["mu"]) / st["sd"], -5, 5); return Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-9)
