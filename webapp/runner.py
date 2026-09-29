"""Run registry + background execution. Calls the EXISTING integration entry points; no assurance logic here."""
from __future__ import annotations

import json
import os
import shutil
import tempfile
import threading
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .result import build_result
from .validate import inspect_package, normalise

REPO = Path(__file__).resolve().parents[1]
DEMO_DIR = Path(os.environ.get("DEMO_DIR", REPO / "data_demo"))
RUNS_ROOT = Path(os.environ.get("RUNS_DIR", Path(tempfile.gettempdir()) / "cv_assurance_runs"))
STAGES = ["p1", "p2", "p3", "p4", "p5"]
MAX_RUNS_KEPT = 20
RUN_TTL_S = 3 * 3600

# artifact written when each stage has finished (relative to the run's output dir)
MARKERS = {"p1": "p1234/p123/part12/part1/assurance_part1.json", "p2": "p1234/p123/part12/part2/assurance_part2.json",
           "p3": "p1234/p123/part3/assurance_part3.json", "p4": "p1234/assurance_combined_p1234.json", "p5": "part5/assurance_part5.json"}

_pool = ThreadPoolExecutor(max_workers=1)      # one pipeline at a time: keeps memory bounded on small Render instances
_runs: dict[str, dict] = {}
_lock = threading.Lock()
_uploads: dict[str, dict] = {}                 # upload_id -> {"dir": Path, "report": dict, "root": Path, "ts": float}


def demo_info() -> dict:
    """Describe the bundled demo from the files that actually exist."""
    ds, models, streams = DEMO_DIR / "dataset", DEMO_DIR / "models", DEMO_DIR / "streams"
    ok = (ds / "annotations.json").is_file() and (models / "trojan_checker.npz").is_file() and (models / "trusted_reference.npz").is_file()
    info = {"available": ok, "name": "Synthetic CV Assurance Dataset", "images": 0, "annotations": 0, "streams": [], "served_model": "trojan_checker.npz", "trusted_model": "trusted_reference.npz"}
    if ok:
        try:
            a = json.loads((ds / "annotations.json").read_text()); info["images"] = len(a["images"]); info["annotations"] = len(a["annotations"])
        except Exception: pass
        info["streams"] = sorted(d.name for d in streams.iterdir() if d.is_dir()) if streams.is_dir() else []
    return info


def _new_run(mode: str, input_info: dict) -> dict:
    run = {"run_id": uuid.uuid4().hex[:12], "mode": mode, "state": "queued", "stages": {s: "pending" for s in STAGES}, "logs": [], "error": None,
           "created": time.time(), "input": input_info, "result": None, "dir": None}
    with _lock:
        _prune()
        _runs[run["run_id"]] = run
    return run


def _prune():
    now = time.time()
    for rid in [r for r, v in _runs.items() if v["state"] in ("done", "failed") and now - v["created"] > RUN_TTL_S]:
        _drop(rid)
    done = sorted((v["created"], r) for r, v in _runs.items() if v["state"] in ("done", "failed"))
    for _, rid in done[:-MAX_RUNS_KEPT]: _drop(rid)
    for uid in [u for u, v in _uploads.items() if now - v["ts"] > RUN_TTL_S]: discard_upload(uid)


def _drop(rid):
    r = _runs.pop(rid, None)
    if r and r.get("dir"): shutil.rmtree(r["dir"], ignore_errors=True)


def _watch(run, out, stop):
    """Mark stages complete as their artifacts appear, and the first unfinished one as running."""
    while not stop.is_set():
        _sync(run, out); stop.wait(0.4)
    _sync(run, out)


def _sync(run, out):
    active = True
    for s in STAGES:
        if run["stages"][s] in ("skipped",): continue
        if (out / MARKERS[s]).is_file() or (s == "p1" and (out / "part1" / "assurance_part1.json").is_file()): run["stages"][s] = "complete"
        elif active: run["stages"][s] = "running"; active = False
        else: run["stages"][s] = "pending"


def _execute(run, kind, args):
    out = Path(run["dir"]) / "out"; out.mkdir(parents=True, exist_ok=True)
    run["state"] = "running"; log = lambda *a: run["logs"].append(" ".join(str(x) for x in a))
    stop = threading.Event(); w = threading.Thread(target=_watch, args=(run, out, stop), daemon=True); w.start()
    try:
        if kind == "integrated":
            from cv_assurance.part5_governance.integration import run_integrated5
            run_integrated5(args["dataset"], args["model"], args["trusted"], out, args["streams"], args.get("fmt", "auto"), log=log)
        else:                                                    # Part-1-only package (no model supplied)
            from cv_assurance.part1_data.pipeline import run_part1
            from cv_assurance.part5_governance.assemble import assemble
            for s in ("p2", "p3", "p4"): run["stages"][s] = "skipped"
            run_part1(args["dataset"], out / "part1", "coco"); log("[p1] done. Parts 2-4 skipped (no model supplied)")
            assemble([out / "part1"], out / "part5", actor="web_p1_p5", log=log)
        stop.set(); w.join(2)
        for s in STAGES:
            if run["stages"][s] != "skipped": run["stages"][s] = "complete"
        run["result"] = build_result(out, run["mode"], run["logs"], run["input"])
        run["state"] = "done"
    except BaseException as e:                                   # assemble() may raise SystemExit
        stop.set(); w.join(2)
        traceback.print_exc()
        run["state"] = "failed"; run["error"] = f"{type(e).__name__}: {str(e)[:300]}".replace(str(REPO), ".").replace(str(RUNS_ROOT), "<run>")
        for s in STAGES:
            if run["stages"][s] == "running": run["stages"][s] = "failed"
    finally:
        _cleanup_secrets(out)


def _cleanup_secrets(out: Path):
    """Signing keys and ledger copies of input bytes are not needed after the result is built."""
    for p in out.rglob("*.key"):
        try: p.unlink()
        except OSError: pass


def start_demo() -> dict:
    d = demo_info()
    if not d["available"]: raise FileNotFoundError("bundled demo dataset not found on the server")
    run = _new_run("demo", {"source": "bundled demo", "images": d["images"], "annotations": d["annotations"], "streams": d["streams"], "served_model": d["served_model"]})
    run["dir"] = tempfile.mkdtemp(prefix="run_", dir=_mk_root())
    st = [(n, DEMO_DIR / "streams" / n) for n in d["streams"]]
    args = {"dataset": DEMO_DIR / "dataset", "model": DEMO_DIR / "models" / d["served_model"], "trusted": DEMO_DIR / "models" / d["trusted_model"], "streams": st, "fmt": "coco"}
    _pool.submit(_execute, run, "integrated", args)
    return run


def _mk_root() -> str:
    RUNS_ROOT.mkdir(parents=True, exist_ok=True); return str(RUNS_ROOT)


def register_upload(zip_bytes_path: Path, workdir: Path) -> dict:
    report, root = inspect_package(zip_bytes_path, workdir)
    zip_bytes_path.unlink(missing_ok=True)                       # keep only the extracted, validated tree
    uid = uuid.uuid4().hex[:12]
    if root is None: shutil.rmtree(workdir, ignore_errors=True)
    else:
        with _lock: _prune(); _uploads[uid] = {"dir": workdir, "root": root, "report": report, "ts": time.time()}
    return {"upload_id": uid if root is not None else None, "validation": report}


def get_upload(uid): return _uploads.get(uid)


def discard_upload(uid):
    u = _uploads.pop(uid, None)
    if u: shutil.rmtree(u["dir"], ignore_errors=True)


def start_custom(upload_id: str) -> dict:
    u = _uploads.get(upload_id)
    if not u: raise KeyError("unknown or expired upload_id")
    if not u["report"]["valid"]: raise ValueError("package failed validation; fix the errors and upload again")
    rep = u["report"]
    run = _new_run("custom", {"source": "uploaded package", "images": rep["images"], "annotations": rep["annotations"], "annotation_format": rep["annotation_format"],
                              "model_detected": rep["model_detected"], "streams": rep["streams"]})
    run["dir"] = str(u["dir"])
    pk = normalise(u["root"], Path(run["dir"]))
    _uploads.pop(upload_id, None)
    if pk["model"]:
        args = {"dataset": pk["dataset"], "model": pk["model"], "trusted": pk["trusted"] or pk["model"], "streams": pk["streams"], "fmt": "coco"}
        _pool.submit(_execute, run, "integrated", args)
    else:
        _pool.submit(_execute, run, "part1", {"dataset": pk["dataset"]})
    return run


def get_run(rid): return _runs.get(rid)


def public_status(run) -> dict:
    return {"run_id": run["run_id"], "mode": run["mode"], "state": run["state"], "stages": run["stages"], "error": run["error"], "input": run["input"],
            "log_tail": run["logs"][-12:], "elapsed_s": round(time.time() - run["created"], 1)}
