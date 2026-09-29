"""Synthetic benchmark generator: clean data + controlled attacks + ground-truth manifest.

Attacks injected by contributor_x: duplicate flooding, label flips, OOD samples, a patch trigger
(clean-label style), plus structural faults (missing file, truncated image, impossible bbox).
Ground truth is written to ground_truth.json and is used ONLY for scoring.
"""
from __future__ import annotations

import io
import json
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

CLASSES = ["circle", "square", "stripes", "triangle"]
COLORS = [(200, 40, 40), (40, 60, 200), (40, 170, 60), (220, 200, 40)]
SIZE = 128


def _background(rng):
    base = rng.integers(90, 170, 3)
    yy, xx = np.mgrid[0:SIZE, 0:SIZE]
    g = (xx * rng.uniform(-0.3, 0.3) + yy * rng.uniform(-0.3, 0.3)).astype(np.float32)
    img = base[None, None, :] + g[..., None] + rng.normal(0, 6, (SIZE, SIZE, 3))
    return Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))


def make_image(cls: int, rng):
    img = _background(rng)
    d = ImageDraw.Draw(img)
    w = int(rng.integers(34, 60))
    h = int(rng.integers(34, 60))
    x = int(rng.integers(4, SIZE - w - 4))
    y = int(rng.integers(4, SIZE - h - 4))
    col = tuple(int(np.clip(c + rng.integers(-20, 20), 0, 255)) for c in COLORS[cls])
    if cls == 0:
        d.ellipse([x, y, x + w, y + h], fill=col)
    elif cls == 1:
        d.rectangle([x, y, x + w, y + h], fill=col)
    elif cls == 2:
        for k in range(0, h, 8):
            d.rectangle([x, y + k, x + w, y + k + 3], fill=col)
    else:
        d.polygon([(x + w // 2, y), (x, y + h), (x + w, y + h)], fill=col)
    return img, [x, y, w, h]


def make_ood_image(rng):
    kind = int(rng.integers(0, 3))
    if kind == 0:
        arr = rng.integers(0, 255, (SIZE, SIZE, 3)).astype(np.uint8)
    elif kind == 1:
        yy, xx = np.mgrid[0:SIZE, 0:SIZE]
        v = (np.sin(xx / rng.uniform(2, 5) + rng.uniform(0, 6)) + np.sin(yy / rng.uniform(2, 6) + rng.uniform(0, 6))) * 60 + 128
        arr = np.stack([v, 255 - v, (v * 1.3) % 255], -1).clip(0, 255).astype(np.uint8)
    else:
        arr = np.kron(rng.integers(0, 2, (16, 16, 1)), np.ones((8, 8, 3))).astype(np.uint8) * 255
    return Image.fromarray(arr)


def add_trigger(img: Image.Image):
    px = img.copy()
    d = ImageDraw.Draw(px)
    x0, y0 = SIZE - 16, SIZE - 16
    for i in range(2):
        for j in range(2):
            c = (0, 0, 0) if (i + j) % 2 == 0 else (255, 255, 255)
            d.rectangle([x0 + i * 6, y0 + j * 6, x0 + i * 6 + 5, y0 + j * 6 + 5], fill=c)
    return px


def generate(out_dir, n_clean=400, fmt="coco", seed=7) -> dict:
    out = Path(out_dir)
    img_dir = out / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    py = random.Random(seed)
    records = []  # dict(file_name, cls, bbox, source, contributor, batch)
    gt = {"label_flip": [], "duplicate_flood": [], "ood": [], "trigger": [], "structural": [],
          "attack_source": "contributor_x"}

    def save(img, name, quality=None):
        if quality:
            img.save(img_dir / name, "JPEG", quality=quality)
        else:
            img.save(img_dir / name)

    def add(name, cls, bbox, contrib, batch):
        records.append({"file_name": f"images/{name}", "cls": cls, "bbox": bbox,
                        "contributor_id": contrib, "batch_id": batch, "source_id": contrib})

    contribs = ["contributor_a", "contributor_b", "contributor_c"]
    for i in range(n_clean):
        cls = i % 4
        img, bb = make_image(cls, rng)
        save(img, f"clean_{i:04d}.png")
        add(f"clean_{i:04d}.png", cls, bb, contribs[i % 3], f"batch_{i % 6}")
    cx = "contributor_x"
    n_x = max(20, n_clean // 6)
    for i in range(n_x):  # benign filler from the attacking contributor
        cls = i % 4
        img, bb = make_image(cls, rng)
        save(img, f"x_clean_{i:04d}.png")
        add(f"x_clean_{i:04d}.png", cls, bb, cx, "batch_x")
    # duplicate flooding (seed + jittered JPEG copies)
    seed_img, seed_bb = make_image(1, rng)
    save(seed_img, "x_flood_seed.png")
    add("x_flood_seed.png", 1, seed_bb, cx, "batch_x")
    gt["duplicate_flood"].append("images/x_flood_seed.png")
    for i in range(24):
        arr = np.asarray(seed_img, dtype=np.float32) * py.uniform(0.97, 1.03)
        save(Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8)), f"x_flood_{i:03d}.jpg", quality=py.choice([70, 80, 90]))
        add(f"x_flood_{i:03d}.jpg", 1, seed_bb, cx, "batch_x")
        gt["duplicate_flood"].append(f"images/x_flood_{i:03d}.jpg")
    # label flips (visually class 0/3, labelled 1/2)
    for i in range(20):
        true = 0 if i % 2 == 0 else 3
        wrong = 1 if true == 0 else 2
        img, bb = make_image(true, rng)
        save(img, f"x_flip_{i:03d}.png")
        add(f"x_flip_{i:03d}.png", wrong, bb, cx, "batch_x")
        gt["label_flip"].append(f"images/x_flip_{i:03d}.png")
    # OOD
    for i in range(15):
        save(make_ood_image(rng), f"x_ood_{i:03d}.png")
        add(f"x_ood_{i:03d}.png", int(rng.integers(0, 4)), [10, 10, 60, 60], cx, "batch_x")
        gt["ood"].append(f"images/x_ood_{i:03d}.png")
    # trigger (clean-label style: patch on class-0 images labelled class 0)
    for i in range(25):
        img, bb = make_image(0, rng)
        save(add_trigger(img), f"x_trig_{i:03d}.png")
        add(f"x_trig_{i:03d}.png", 0, bb, cx, "batch_x")
        gt["trigger"].append(f"images/x_trig_{i:03d}.png")
    # structural faults
    (img_dir / "x_corrupt_000.png").write_bytes((img_dir / "clean_0000.png").read_bytes()[:200])
    add("x_corrupt_000.png", 0, [10, 10, 40, 40], cx, "batch_x")
    gt["structural"].append("images/x_corrupt_000.png")
    add("x_missing_000.png", 1, [10, 10, 40, 40], cx, "batch_x")  # file never written
    gt["structural"].append("images/x_missing_000.png")
    img, _ = make_image(2, rng)
    save(img, "x_badbox_000.png")
    add("x_badbox_000.png", 2, [10, 10, -5, 40], cx, "batch_x")
    gt["structural"].append("images/x_badbox_000.png")

    if fmt == "coco":
        _write_coco(out, records)
    else:
        _write_yolo(out, records)
    (out / "ground_truth.json").write_text(json.dumps(gt, indent=2))
    return gt


def _write_coco(out: Path, records):
    images, anns = [], []
    for i, r in enumerate(records, 1):
        images.append({"id": i, "file_name": r["file_name"].split("/", 1)[1], "width": SIZE, "height": SIZE,
                       "source_id": r["source_id"], "contributor_id": r["contributor_id"], "batch_id": r["batch_id"]})
        anns.append({"id": i, "image_id": i, "category_id": r["cls"], "bbox": r["bbox"],
                     "area": max(r["bbox"][2], 0) * max(r["bbox"][3], 0), "iscrowd": 0})
    data = {"images": images, "annotations": anns,
            "categories": [{"id": i, "name": n} for i, n in enumerate(CLASSES)]}
    (out / "annotations.json").write_text(json.dumps(data))


def _write_yolo(out: Path, records):
    (out / "labels").mkdir(exist_ok=True)
    (out / "data.yaml").write_text("names: [" + ", ".join(CLASSES) + "]\n")
    lines = ["file_name,source_id,contributor_id,batch_id"]
    for r in records:
        name = Path(r["file_name"]).stem
        x, y, w, h = r["bbox"]
        cxn, cyn = (x + w / 2) / SIZE, (y + h / 2) / SIZE
        wn, hn = w / SIZE, h / SIZE
        (out / "labels" / f"{name}.txt").write_text(f"{r['cls']} {cxn:.6f} {cyn:.6f} {wn:.6f} {hn:.6f}\n")
        lines.append(f"{r['file_name']},{r['source_id']},{r['contributor_id']},{r['batch_id']}")
    (out / "sources.csv").write_text("\n".join(lines) + "\n")
