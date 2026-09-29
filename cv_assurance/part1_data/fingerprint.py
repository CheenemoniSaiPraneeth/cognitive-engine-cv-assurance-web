"""Module 3 - Image fingerprinting: SHA-256, pHash, dHash, quality stats, embedding input.

One pass over every image computes everything the later detectors need.
"""
from __future__ import annotations

import io
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.fft import dctn

from ..common.hashing import sha256_bytes
from .features import handcrafted_features


def phash64(gray32: np.ndarray) -> int:
    low = dctn(gray32.astype(np.float64), norm="ortho")[:8, :8].ravel()
    med = np.median(low[1:])
    bits = low > med
    return int("".join("1" if b else "0" for b in bits), 2)


def dhash64(gray9x8: np.ndarray) -> int:
    bits = (gray9x8[:, 1:] > gray9x8[:, :-1]).ravel()  # 8 rows x 8 = 64
    return int("".join("1" if b else "0" for b in bits), 2)


def analyze_image(path: str, encoder=None) -> dict:
    out = {"error": None}
    try:
        data = Path(path).read_bytes()
        out["sha256"] = sha256_bytes(data)
        img = Image.open(io.BytesIO(data))
        fmt = img.format
        img.load()  # full decode -> catches truncated/corrupt payloads
        rgb = img.convert("RGB")
        out.update(width=rgb.size[0], height=rgb.size[1], format=fmt)
        gray = rgb.convert("L")
        out["phash"] = phash64(np.asarray(gray.resize((32, 32), Image.LANCZOS), dtype=np.float32))
        out["dhash"] = dhash64(np.asarray(gray.resize((9, 8), Image.LANCZOS), dtype=np.float32))
        rgb64 = np.asarray(rgb.resize((64, 64), Image.BILINEAR), dtype=np.float32) / 255.0
        out["feat"] = encoder.embed(rgb) if encoder is not None else handcrafted_features(rgb64)
        out["gray96"] = np.asarray(gray.resize((96, 96), Image.BILINEAR), dtype=np.uint8)
        g = np.asarray(gray.resize((64, 64), Image.BILINEAR), dtype=np.float32)
        lap = g[1:-1, 1:-1] * 4 - g[:-2, 1:-1] - g[2:, 1:-1] - g[1:-1, :-2] - g[1:-1, 2:]
        hist = np.bincount(g.astype(np.uint8).ravel(), minlength=256) / g.size
        nz = hist[hist > 0]
        out["stats"] = {
            "brightness": float(g.mean()), "contrast": float(g.std()),
            "sharpness": float(lap.var()), "entropy": float(-(nz * np.log2(nz)).sum()),
            "mean_r": float(rgb64[..., 0].mean()), "mean_g": float(rgb64[..., 1].mean()),
            "mean_b": float(rgb64[..., 2].mean()),
            "std_r": float(rgb64[..., 0].std()), "std_g": float(rgb64[..., 1].std()),
            "std_b": float(rgb64[..., 2].std()),
            "saturation": float(((rgb64.max(-1) - rgb64.min(-1)) / (rgb64.max(-1) + 1e-6)).mean()),
            "extreme_fraction": float(((g < 5) | (g > 250)).mean()),
        }
    except Exception as e:  # unreadable / corrupt
        out["error"] = f"{type(e).__name__}: {e}"
    return out


def analyze_paths(paths: list, encoder=None, workers: int = 4) -> list:
    if workers <= 1:
        return [analyze_image(p, encoder) if p else {"error": "no path"} for p in paths]
    with ThreadPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(lambda p: analyze_image(p, encoder) if p else {"error": "no path"}, paths))


def hex64(v: int) -> str:
    return f"{v:016x}"
