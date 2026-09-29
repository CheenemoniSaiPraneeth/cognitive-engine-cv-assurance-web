"""FastAPI service: serves the dashboard and exposes the P1-P5 pipeline as an async API."""
from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import runner
from .validate import MAX_ZIP_BYTES

STATIC = Path(__file__).parent / "static"
app = FastAPI(title="CV Assurance", version="1.0", docs_url="/api/docs", redoc_url=None)


class RunReq(BaseModel):
    upload_id: str


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.get("/api/demo/info")
def demo_info():
    return runner.demo_info()


@app.post("/api/demo/run")
def demo_run():
    try:
        run = runner.start_demo()
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))
    return {"run_id": run["run_id"], "state": run["state"]}


async def _receive(file: UploadFile) -> dict:
    if not (file.filename or "").lower().endswith(".zip"):
        raise HTTPException(400, "please upload a .zip file")
    work = Path(tempfile.mkdtemp(prefix="up_", dir=runner._mk_root()))
    zp = work / "upload.zip"; size = 0
    with open(zp, "wb") as f:
        while chunk := await file.read(1 << 20):
            size += len(chunk)
            if size > MAX_ZIP_BYTES:
                f.close(); shutil.rmtree(work, ignore_errors=True)
                raise HTTPException(413, f"ZIP larger than {MAX_ZIP_BYTES // 2**20} MB")
            f.write(chunk)
    return runner.register_upload(zp, work)


@app.post("/api/upload")
async def upload(file: UploadFile = File(...)):
    """Receive + validate in one step; returns upload_id (only when the package can be used) and the validation report."""
    return await _receive(file)


@app.post("/api/validate")
async def validate(file: UploadFile = File(...)):
    """Validate only: same report, package is discarded immediately."""
    r = await _receive(file)
    if r["upload_id"]: runner.discard_upload(r["upload_id"])
    return {"validation": r["validation"]}


@app.post("/api/run")
def run(req: RunReq):
    try:
        r = runner.start_custom(req.upload_id)
    except KeyError as e:
        raise HTTPException(404, str(e.args[0]))
    except ValueError as e:
        raise HTTPException(422, str(e))
    return {"run_id": r["run_id"], "state": r["state"]}


def _get(run_id):
    r = runner.get_run(run_id)
    if not r: raise HTTPException(404, "unknown run_id")
    return r


@app.get("/api/status/{run_id}")
def status(run_id: str):
    return runner.public_status(_get(run_id))


@app.get("/api/result/{run_id}")
def result(run_id: str):
    r = _get(run_id)
    if r["state"] == "failed": return JSONResponse({"error": r["error"]}, status_code=500)
    if r["state"] != "done": return JSONResponse({"state": r["state"]}, status_code=202)
    return r["result"]


app.mount("/static", StaticFiles(directory=STATIC), name="static")


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")
