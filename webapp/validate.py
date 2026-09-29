"""Safe ZIP handling + package validation for custom datasets. Nothing from the archive is ever executed."""
from __future__ import annotations

import json
import os
import shutil
import zipfile
from pathlib import Path, PurePosixPath

from PIL import Image

MAX_ZIP_BYTES = int(os.environ.get("MAX_ZIP_MB", "200")) * 1024 * 1024
MAX_UNCOMPRESSED = int(os.environ.get("MAX_UNCOMPRESSED_MB", "800")) * 1024 * 1024
MAX_FILES = int(os.environ.get("MAX_ZIP_FILES", "20000"))
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
MODEL_EXT = {".npz", ".onnx", ".pt", ".pth", ".torchscript"}
MAX_IMAGE_CHECKS = 5000


class PackageError(Exception):
    pass


def _clean_name(name: str) -> PurePosixPath:
    n = name.replace("\\", "/")
    p = PurePosixPath(n)
    if p.is_absolute() or ".." in p.parts or (p.parts and ":" in p.parts[0]) or "\x00" in n:
        raise PackageError("unsafe path in archive")
    return p


def safe_extract(zip_path: Path, dest: Path) -> None:
    """Extract with path-traversal, symlink, size and file-count protection."""
    if zip_path.stat().st_size > MAX_ZIP_BYTES:
        raise PackageError(f"ZIP larger than {MAX_ZIP_BYTES // 2**20} MB")
    try:
        zf = zipfile.ZipFile(zip_path)
    except zipfile.BadZipFile:
        raise PackageError("file is not a readable ZIP archive")
    with zf:
        infos = zf.infolist()
        if len(infos) > MAX_FILES:
            raise PackageError(f"too many files in archive (>{MAX_FILES})")
        if sum(i.file_size for i in infos) > MAX_UNCOMPRESSED:
            raise PackageError("archive expands beyond the allowed size")
        dest.mkdir(parents=True, exist_ok=True)
        root = dest.resolve()
        for i in infos:
            if (i.external_attr >> 16) & 0o170000 == 0o120000:
                raise PackageError("symbolic links are not allowed")
            p = _clean_name(i.filename)
            if not p.parts or p.parts[0] == "__MACOSX" or p.name == ".DS_Store":
                continue
            target = (root / Path(*p.parts)).resolve()
            if root not in target.parents and target != root:
                raise PackageError("unsafe path in archive")
            if i.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            written = 0
            with zf.open(i) as src, open(target, "wb") as out:
                while chunk := src.read(1 << 20):
                    written += len(chunk)
                    if written > MAX_UNCOMPRESSED:
                        raise PackageError("archive expands beyond the allowed size")
                    out.write(chunk)


def _find_root(ex: Path) -> Path:
    if (ex / "images").is_dir():
        return ex
    subs = [d for d in ex.iterdir() if d.is_dir()]
    if len(subs) == 1 and (subs[0] / "images").is_dir():
        return subs[0]
    return ex


def inspect_package(zip_path: Path, work: Path) -> tuple[dict, Path | None]:
    """Validate a package. Returns (report, extracted_root or None). Never raises for user-caused problems."""
    rep = {"valid": False, "images": 0, "annotations": 0, "annotation_format": None, "model_detected": False,
           "trusted_model_detected": False, "streams": [], "detected": {"images/": False, "annotations/": False, "annotations.json": False,
           "metadata/": False, "model/": False}, "errors": [], "warnings": []}
    ex = work / "extracted"
    try:
        safe_extract(zip_path, ex)
    except PackageError as e:
        rep["errors"].append(str(e)); shutil.rmtree(ex, ignore_errors=True); return rep, None
    root = _find_root(ex)
    img_dir, ann_dir = root / "images", root / "annotations"
    rep["detected"].update({"images/": img_dir.is_dir(), "annotations/": ann_dir.is_dir(), "annotations.json": (ann_dir / "annotations.json").is_file(),
                            "metadata/": (root / "metadata").is_dir(), "model/": (root / "model").is_dir()})
    if not img_dir.is_dir(): rep["errors"].append("missing required directory: images/")
    if not (ann_dir / "annotations.json").is_file(): rep["errors"].append("missing required file: annotations/annotations.json")
    if rep["errors"]: return rep, root

    files = {p.name: p for p in img_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXT}
    rep["images"] = len(files)
    if not files: rep["errors"].append("images/ contains no supported image files (" + ", ".join(sorted(IMG_EXT)) + ")")
    bad = 0
    for n, p in enumerate(files.values()):
        if n >= MAX_IMAGE_CHECKS: rep["warnings"].append(f"only the first {MAX_IMAGE_CHECKS} images were pixel-checked"); break
        try:
            with Image.open(p) as im: im.verify()
        except Exception:
            bad += 1
    if bad: rep["warnings"].append(f"{bad} image file(s) are unreadable/corrupt (Part 1 will flag them)")

    try:
        ann = json.loads((ann_dir / "annotations.json").read_text())
    except Exception:
        rep["errors"].append("annotations/annotations.json is not valid JSON"); return rep, root
    if not (isinstance(ann, dict) and isinstance(ann.get("images"), list) and isinstance(ann.get("annotations"), list)):
        rep["errors"].append("unsupported annotation format: expected COCO JSON with 'images' and 'annotations' lists"); return rep, root
    rep["annotation_format"] = "COCO"; rep["annotations"] = len(ann["annotations"])
    ids, missing = set(), 0
    for im in ann["images"]:
        if not isinstance(im, dict) or "id" not in im or "file_name" not in im:
            rep["errors"].append("COCO 'images' entries need 'id' and 'file_name'"); return rep, root
        ids.add(im["id"]); missing += Path(str(im["file_name"])).name not in files
    if missing: rep["warnings"].append(f"{missing} annotated image(s) not found in images/ (Part 1 will flag them)")
    orphan = sum(1 for a in ann["annotations"] if not isinstance(a, dict) or a.get("image_id") not in ids)
    if orphan: rep["warnings"].append(f"{orphan} annotation(s) reference unknown image ids")
    unref = len(files) - len({Path(str(i["file_name"])).name for i in ann["images"]} & set(files))
    if unref > 0: rep["warnings"].append(f"{unref} image(s) in images/ are not referenced by annotations")

    md = root / "model"
    models = sorted(p for p in md.iterdir() if p.is_file() and p.suffix.lower() in MODEL_EXT) if md.is_dir() else []
    trusted = [p for p in models if p.stem.startswith("trusted")]
    served = [p for p in models if p not in trusted]
    rep["model_detected"] = bool(served); rep["trusted_model_detected"] = bool(trusted)
    if md.is_dir() and not models: rep["warnings"].append("model/ present but no supported model file (.npz, .onnx, .pt)")
    if len(served) > 1: rep["warnings"].append("several model files found; using " + served[0].name)
    if not served: rep["warnings"].append("no model supplied: only Part 1 (data integrity) and Part 5 will run")
    elif not trusted: rep["warnings"].append("no trusted reference model (model/trusted_*.npz|onnx): the served model is used as its own reference")
    if served and served[0].suffix.lower() == ".onnx":
        try: import onnxruntime  # noqa: F401
        except Exception: rep["warnings"].append("onnxruntime is not installed on this server; ONNX behaviour tests will be reported UNAVAILABLE")
    sd = root / "streams"
    if sd.is_dir():
        rep["streams"] = sorted(d.name for d in sd.iterdir() if d.is_dir() and any(f.suffix.lower() in IMG_EXT for f in d.iterdir()))
    rep["valid"] = not rep["errors"]
    return rep, root


def normalise(root: Path, dest: Path) -> dict:
    """Arrange a validated package into the layout the existing pipeline expects (dataset/images + dataset/annotations.json)."""
    ds = dest / "dataset"; ds.mkdir(parents=True, exist_ok=True)
    shutil.move(str(root / "images"), str(ds / "images"))
    shutil.copy(root / "annotations" / "annotations.json", ds / "annotations.json")
    md = root / "model"
    models = sorted(p for p in md.iterdir() if p.suffix.lower() in MODEL_EXT) if md.is_dir() else []
    trusted = [p for p in models if p.stem.startswith("trusted")]; served = [p for p in models if p not in trusted]
    streams = []
    if (root / "streams").is_dir():
        streams = [(d.name, d) for d in sorted((root / "streams").iterdir()) if d.is_dir() and any(f.suffix.lower() in IMG_EXT for f in d.iterdir())]
    return {"dataset": ds, "model": served[0] if served else None, "trusted": trusted[0] if trusted else None, "streams": streams}
