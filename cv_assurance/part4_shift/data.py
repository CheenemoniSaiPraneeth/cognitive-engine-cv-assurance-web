"""Item tables: reference snapshot + operational sets, with stable ids, hashes and task-output columns."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from PIL import Image
from ..common.hashing import sha256_bytes, sha256_json
from ..common.schemas import now_iso
from . import DETECTOR_VERSION
from .features import ALL_FEATURES, IMG_EXT, extract_many, TASK_FEATURES


class Table:
    """items: list of dict(id, path, sha256, source, batch_id, timestamp, record_id?) ; S: n x F (NaN = unavailable);
    E: n x D raw embeddings ; pred: int array (-1 unknown)."""
    def __init__(self, items, S, E, pred, class_names=None):
        self.items, self.S, self.E, self.pred, self.class_names = items, S, E, pred, class_names or []
    def __len__(self): return len(self.items)
    def take(self, idx):
        idx = np.asarray(idx); return Table([self.items[i] for i in idx], self.S[idx], self.E[idx], self.pred[idx], self.class_names)


def predict_outputs(adapter, paths, bs=32):
    """Black-box use of a supplied model: no weights needed. Returns pred, confidence, entropy (NaN on failure)."""
    P = np.full(len(paths), -1); C = np.full(len(paths), np.nan); H = np.full(len(paths), np.nan)
    for i in range(0, len(paths), bs):
        ims = []
        for p in paths[i:i + bs]:
            try: ims.append(np.asarray(Image.open(p).convert("RGB")))
            except Exception: ims.append(None)
        ok = [j for j, a in enumerate(ims) if a is not None]
        if not ok: continue
        pr = np.asarray(adapter.predict([ims[j] for j in ok]), dtype=np.float64)
        for r, j in zip(pr, ok):
            pos = r[r > 0]; P[i + j] = int(np.argmax(r)); C[i + j] = float(r.max()); H[i + j] = float(-(pos * np.log(pos)).sum())
    return P, C, H


def build_table(items, cfg, encoder=None, adapter=None, outputs=None, log=lambda *a: None):
    """outputs: optional {item_id: {pred_class_id, confidence, entropy}} (e.g. read from a verified Part-3 ledger)."""
    res = extract_many([it["path"] for it in items], encoder, cfg["workers"])
    good = [(it, r) for it, r in zip(items, res) if r["error"] is None]
    bad = [(it["id"], r["error"]) for it, r in zip(items, res) if r["error"]]
    its = []
    for it, r in good:
        it = dict(it); it["sha256"] = it.get("sha256") or sha256_bytes(Path(it["path"]).read_bytes()); its.append(it)
    S = np.full((len(good), len(ALL_FEATURES)), np.nan)
    for i, (_, r) in enumerate(good):
        for j, f in enumerate(ALL_FEATURES):
            if f in r["scalars"]: S[i, j] = r["scalars"][f]
    E = np.stack([r["emb"] for _, r in good]) if good else np.zeros((0, 1))
    pred = np.full(len(good), -1); names = []
    ci, ei = ALL_FEATURES.index("out_confidence"), ALL_FEATURES.index("out_entropy")
    if outputs is not None:
        for i, it in enumerate(its):
            o = outputs.get(it["id"])
            if o: pred[i], S[i, ci], S[i, ei] = o["pred_class_id"], o["confidence"], o["entropy"]
    elif adapter is not None:
        pred, S[:, ci], S[:, ei] = predict_outputs(adapter, [it["path"] for it in its])
        try: names = adapter.class_names() or []
        except Exception: names = []
    return Table(its, S, E, pred, names), bad


def load_dir(path, window=50, metadata=None):
    """Directory of images -> items. Optional metadata.json: {file: {source, batch_id, timestamp}} (or a list under 'files')."""
    root = Path(path); meta = {}
    mp = Path(metadata) if metadata else root / "metadata.json"
    if mp.exists(): meta = json.loads(mp.read_text()); meta = meta.get("files", meta)
    files = sorted(p for p in root.rglob("*") if p.suffix.lower() in IMG_EXT)
    items = []
    for i, p in enumerate(files):
        m = meta.get(p.name) or meta.get(str(p.relative_to(root))) or {}
        items.append({"id": p.name, "path": str(p), "source": m.get("source"), "timestamp": m.get("timestamp"),
                      "batch_id": m.get("batch_id") or f"W{i // window:03d}"})
    items.sort(key=lambda d: (d["timestamp"] or "", d["id"]))
    return items


def snapshot_id(table, cfg):
    return "REF-" + sha256_json({"shas": sorted(it["sha256"] for it in table.items), "features": ALL_FEATURES, "cfg": cfg["embedding"]})[:12]


def save_reference(table, cfg, out_dir, extra=None):
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "reference.npz", S=table.S, E=table.E, pred=table.pred)
    man = {"reference_id": snapshot_id(table, cfg), "created_at": now_iso(), "detector_version": DETECTOR_VERSION, "n": len(table),
           "features": ALL_FEATURES, "class_names": table.class_names, "items": table.items, "extra": extra or {}}
    man["manifest_sha256"] = sha256_json({k: v for k, v in man.items() if k != "created_at"})
    (out / "reference.json").write_text(json.dumps(man, indent=1, default=str)); return man


def load_reference(dir_):
    d = Path(dir_); man = json.loads((d / "reference.json").read_text()); z = np.load(d / "reference.npz")
    return Table(man["items"], z["S"], z["E"], z["pred"], man.get("class_names")), man
