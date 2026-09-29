"""Module 1 - Dataset ingestion: COCO and YOLO -> common SampleRecord representation.

Validates structure (missing files, malformed annotations, invalid class ids, impossible
coordinates, duplicate identifiers) and never raises on bad records: problems are collected
in ``DatasetBundle.issues`` and later turned into findings.
"""
from __future__ import annotations

import csv
import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from PIL import Image

from ..common.schemas import Annotation, SampleRecord

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
SOURCE_KEYS = {"source_id": ("source_id", "source"),
               "contributor_id": ("contributor_id", "contributor"),
               "batch_id": ("batch_id", "batch")}


@dataclass
class DatasetBundle:
    fmt: str
    root: str
    samples: list = field(default_factory=list)
    class_names: dict = field(default_factory=dict)
    issues: list = field(default_factory=list)
    annotation_source: Optional[str] = None


def make_sample_id(file_name: str) -> str:
    """Stable ID derived from the relative file name - independent of filesystem ordering."""
    return "img_" + hashlib.sha1(file_name.replace("\\", "/").encode()).hexdigest()[:10]


def _issue(bundle, kind, severity, file_name=None, sample_id=None, detail=""):
    bundle.issues.append({"kind": kind, "severity": severity, "file_name": file_name,
                          "sample_id": sample_id, "detail": detail})


def probe_image(path: Path):
    """Read only the header (cheap). Returns (w, h, fmt) or None."""
    try:
        with Image.open(path) as im:
            return im.size[0], im.size[1], im.format
    except Exception:
        return None


def _load_sidecar(root: Path) -> dict:
    """Optional sources.csv with columns: file_name, source_id, contributor_id, batch_id, ..."""
    out = {}
    for cand in (root / "sources.csv", root.parent / "sources.csv"):
        if cand.exists():
            with open(cand, newline="") as fh:
                for row in csv.DictReader(fh):
                    fn = (row.get("file_name") or "").replace("\\", "/")
                    if fn:
                        out[fn] = row
                        out.setdefault(Path(fn).name, row)
            break
    return out


def _apply_source(sample: SampleRecord, raw: dict, sidecar: dict):
    for attr, keys in SOURCE_KEYS.items():
        for k in keys:
            if raw.get(k) not in (None, ""):
                setattr(sample, attr, str(raw[k]))
                break
    row = sidecar.get(sample.file_name) or sidecar.get(Path(sample.file_name).name)
    if row:
        for attr, keys in SOURCE_KEYS.items():
            if getattr(sample, attr) is None:
                for k in keys:
                    if row.get(k):
                        setattr(sample, attr, row[k])
                        break
        for k, v in row.items():
            if k not in ("file_name",) and k not in sum(SOURCE_KEYS.values(), ()) and v:
                sample.metadata[k] = v


def _unique_id(bundle, seen, file_name):
    sid = make_sample_id(file_name)
    if sid in seen:
        _issue(bundle, "duplicate_sample_identifier", "medium", file_name, sid,
               "file_name occurs more than once")
        n = 2
        while f"{sid}_{n}" in seen:
            n += 1
        sid = f"{sid}_{n}"
    seen.add(sid)
    return sid


# ----------------------------------------------------------------------------- COCO
def _find_coco_json(root: Path) -> Optional[Path]:
    best = None
    for p in sorted(root.rglob("*.json")):
        try:
            with open(p) as fh:
                head = json.load(fh)
            if isinstance(head, dict) and "images" in head and "annotations" in head:
                if best is None or p.stat().st_size > best.stat().st_size:
                    best = p
        except Exception:
            continue
    return best


def load_coco(root, ann_file=None, images_dir=None) -> DatasetBundle:
    root = Path(root)
    if root.is_file():
        ann_file, root = root, root.parent
    ann_file = Path(ann_file) if ann_file else _find_coco_json(root)
    bundle = DatasetBundle(fmt="COCO", root=str(root))
    if ann_file is None:
        _issue(bundle, "no_annotation_file", "high", detail="no COCO json found")
        return bundle
    bundle.annotation_source = str(ann_file)
    try:
        data = json.loads(Path(ann_file).read_text())
    except Exception as e:  # malformed JSON
        _issue(bundle, "malformed_annotation_file", "high", detail=str(e))
        return bundle

    cats = {}
    for c in data.get("categories", []):
        if "id" in c:
            cats[c["id"]] = str(c.get("name", c["id"]))
    bundle.class_names = dict(cats)

    dirs = [Path(images_dir)] if images_dir else []
    dirs += [ann_file.parent / "images", ann_file.parent, root / "images", root, ann_file.parent.parent / "images"]
    sidecar = _load_sidecar(root)

    by_img: dict = {}
    for a in data.get("annotations", []):
        by_img.setdefault(a.get("image_id"), []).append(a)

    seen_ids, seen_img_ids = set(), set()
    known_img_ids = set()
    for im in data.get("images", []):
        iid = im.get("id")
        fn = str(im.get("file_name", "")).replace("\\", "/")
        if iid in seen_img_ids:
            _issue(bundle, "duplicate_image_id", "medium", fn, None, f"image id {iid} repeated")
        seen_img_ids.add(iid)
        known_img_ids.add(iid)
        sid = _unique_id(bundle, seen_ids, fn)
        path = next((d / fn for d in dirs if (d / fn).is_file()), None)
        s = SampleRecord(sample_id=sid, file_name=fn, image_path=str(path) if path else None)
        if path is None:
            s.readable = False
            _issue(bundle, "missing_image_file", "high", fn, sid, "image file not found")
        else:
            pr = probe_image(path)
            if pr:
                s.width, s.height, s.format = pr
            # header may still be readable while the payload is corrupt - checked at decode time
        if s.width is None and im.get("width"):
            s.width, s.height = im.get("width"), im.get("height")
        elif s.width and im.get("width") and (im["width"], im["height"]) != (s.width, s.height):
            _issue(bundle, "size_mismatch", "medium", fn, sid,
                   f"json says {im['width']}x{im['height']}, file is {s.width}x{s.height}")
        _apply_source(s, im, sidecar)
        for k, v in im.items():
            if k not in ("id", "file_name", "width", "height") and k not in sum(SOURCE_KEYS.values(), ()):
                s.metadata[k] = v
        for a in by_img.get(iid, []):
            ann = _coco_annotation(bundle, a, cats, s)
            if ann:
                s.annotations.append(ann)
        bundle.samples.append(s)

    for iid, anns in by_img.items():
        if iid not in known_img_ids:
            _issue(bundle, "orphan_annotation", "medium", None, None,
                   f"{len(anns)} annotation(s) reference unknown image_id {iid}")
    return bundle


def _coco_annotation(bundle, a, cats, s) -> Optional[Annotation]:
    cid = a.get("category_id")
    if cid not in cats:
        _issue(bundle, "invalid_class_id", "medium", s.file_name, s.sample_id,
               f"category_id {cid} not in categories")
        if cid is None:
            return None
    name = cats.get(cid, f"class_{cid}")
    bbox = a.get("bbox")
    poly = None
    seg = a.get("segmentation")
    if isinstance(seg, list) and seg and isinstance(seg[0], list):
        poly = seg[0]
    if bbox is not None:
        ok = isinstance(bbox, (list, tuple)) and len(bbox) == 4 and all(isinstance(v, (int, float)) for v in bbox)
        if not ok:
            _issue(bundle, "malformed_bbox", "medium", s.file_name, s.sample_id, f"bbox={bbox}")
            return Annotation(cid, name, None, 0.0, poly)
        x, y, w, h = bbox
        bad = w <= 0 or h <= 0 or x < -1 or y < -1
        if s.width and s.height:
            bad = bad or x + w > s.width + 1 or y + h > s.height + 1
        if bad:
            _issue(bundle, "impossible_coordinates", "medium", s.file_name, s.sample_id, f"bbox={bbox}")
        return Annotation(cid, name, [float(v) for v in bbox], float(max(w, 0) * max(h, 0)), poly)
    return Annotation(cid, name, None, float(a.get("area") or 0.0), poly)


# ----------------------------------------------------------------------------- YOLO
def read_yolo_names(root: Path) -> dict:
    for fn in ("data.yaml", "dataset.yaml", "data.yml"):
        p = root / fn
        if p.exists():
            txt = p.read_text()
            try:
                import yaml  # optional
                names = yaml.safe_load(txt).get("names")
                if isinstance(names, list):
                    return {i: str(n) for i, n in enumerate(names)}
                if isinstance(names, dict):
                    return {int(k): str(v) for k, v in names.items()}
            except Exception:
                pass
            m = re.search(r"names\s*:\s*\[(.*?)\]", txt, re.S)
            if m:
                items = [t.strip().strip("'\"") for t in m.group(1).split(",") if t.strip()]
                return {i: n for i, n in enumerate(items)}
            m = re.search(r"names\s*:\s*\n((?:\s+.*\n?)+)", txt)
            if m:
                out = {}
                for line in m.group(1).splitlines():
                    mm = re.match(r"\s*(\d+)\s*:\s*(.+)", line) or re.match(r"\s*-\s*()(.+)", line)
                    if mm:
                        idx = int(mm.group(1)) if mm.group(1) != "" else len(out)
                        out[idx] = mm.group(2).strip().strip("'\"")
                return out
    for fn in ("classes.txt", "obj.names"):
        p = root / fn
        if p.exists():
            return {i: l.strip() for i, l in enumerate(p.read_text().splitlines()) if l.strip()}
    return {}


def _yolo_label_path(img: Path, root: Path, labels_dir=None) -> Path:
    if labels_dir:
        return Path(labels_dir) / img.relative_to(root).with_suffix(".txt").name
    parts = list(img.parts)
    for i in range(len(parts) - 2, -1, -1):
        if parts[i] == "images":
            parts[i] = "labels"
            return Path(*parts).with_suffix(".txt")
    return img.with_suffix(".txt")


def load_yolo(root, images_dir=None, labels_dir=None) -> DatasetBundle:
    root = Path(root)
    bundle = DatasetBundle(fmt="YOLO", root=str(root))
    bundle.class_names = read_yolo_names(root)
    scan = Path(images_dir) if images_dir else root
    imgs = sorted(p for p in scan.rglob("*") if p.suffix.lower() in IMG_EXT)
    sidecar = _load_sidecar(root)
    seen = set()
    for p in imgs:
        fn = p.relative_to(root).as_posix() if p.is_relative_to(root) else p.name
        sid = _unique_id(bundle, seen, fn)
        s = SampleRecord(sample_id=sid, file_name=fn, image_path=str(p))
        pr = probe_image(p)
        if pr:
            s.width, s.height, s.format = pr
        _apply_source(s, {}, sidecar)
        lp = _yolo_label_path(p, root, labels_dir)
        if not lp.exists():
            _issue(bundle, "missing_label_file", "low", fn, sid, "no label file (background image?)")
        else:
            for ln, line in enumerate(lp.read_text().splitlines(), 1):
                if line.strip():
                    ann = _yolo_line(bundle, line, ln, s, lp)
                    if ann:
                        s.annotations.append(ann)
        bundle.samples.append(s)
    if not imgs:
        _issue(bundle, "no_images_found", "high", detail=f"no images under {scan}")
    return bundle


def _yolo_line(bundle, line, ln, s, lp) -> Optional[Annotation]:
    parts = line.split()
    try:
        cid = int(float(parts[0]))
        vals = [float(v) for v in parts[1:]]
    except Exception:
        _issue(bundle, "malformed_annotation", "medium", s.file_name, s.sample_id, f"{lp.name}:{ln} '{line[:60]}'")
        return None
    if bundle.class_names and cid not in bundle.class_names:
        _issue(bundle, "invalid_class_id", "medium", s.file_name, s.sample_id, f"class id {cid} not in names")
    if cid < 0:
        _issue(bundle, "invalid_class_id", "medium", s.file_name, s.sample_id, f"negative class id {cid}")
    name = bundle.class_names.get(cid, f"class_{cid}")
    poly = None
    if len(vals) == 4:
        cx, cy, w, h = vals
    elif len(vals) >= 6 and len(vals) % 2 == 0:
        xs, ys = vals[0::2], vals[1::2]
        cx, cy = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
        w, h = max(xs) - min(xs), max(ys) - min(ys)
        poly = vals
    else:
        _issue(bundle, "malformed_annotation", "medium", s.file_name, s.sample_id,
               f"{lp.name}:{ln} wrong number of values ({len(vals)})")
        return None
    if not all(0.0 <= v <= 1.0 for v in (cx, cy, w, h)) or w <= 0 or h <= 0:
        _issue(bundle, "impossible_coordinates", "medium", s.file_name, s.sample_id,
               f"{lp.name}:{ln} normalised box out of range: {[cx, cy, w, h]}")
    W, H = (s.width or 1), (s.height or 1)
    bbox = [(cx - w / 2) * W, (cy - h / 2) * H, w * W, h * H]
    return Annotation(cid, name, bbox, float(max(w, 0) * W * max(h, 0) * H), poly)


# ----------------------------------------------------------------------------- entry
def detect_format(path) -> str:
    p = Path(path)
    if p.is_file() and p.suffix == ".json":
        return "coco"
    if p.is_dir() and _find_coco_json(p) is not None:
        return "coco"
    return "yolo"


def load_dataset(path, fmt: str = "auto", ann_file=None, images_dir=None, labels_dir=None) -> DatasetBundle:
    fmt = detect_format(path) if fmt == "auto" else fmt.lower()
    if fmt == "coco":
        return load_coco(path, ann_file=ann_file, images_dir=images_dir)
    if fmt == "yolo":
        return load_yolo(path, images_dir=images_dir, labels_dir=labels_dir)
    raise ValueError(f"unsupported format {fmt!r} (expected coco|yolo|auto)")
