"""API tests: run the REAL default demo and the custom-ZIP path. Expected values are read from the result, never hard-coded."""
import io, json, shutil, time, zipfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from webapp.app import app
from webapp import runner

client = TestClient(app)
ROOT = Path(__file__).resolve().parents[1]
DEMO = ROOT / "data_demo"


def wait(rid, timeout=240):
    t = time.time()
    while time.time() - t < timeout:
        s = client.get(f"/api/status/{rid}").json()
        if s["state"] in ("done", "failed"): return s
        time.sleep(1)
    raise TimeoutError


def make_zip(tmp, n=40, with_model=True, bad_ann=False):
    z = tmp / "pkg.zip"; ann = json.loads((DEMO / "dataset/annotations.json").read_text())
    imgs = ann["images"][:n]; ids = {i["id"] for i in imgs}
    ann = {**ann, "images": imgs, "annotations": [a for a in ann["annotations"] if a["image_id"] in ids]}
    with zipfile.ZipFile(z, "w") as zf:
        for i in imgs: zf.write(DEMO / "dataset/images" / i["file_name"], f"images/{i['file_name']}")
        zf.writestr("annotations/annotations.json", "{nope" if bad_ann else json.dumps(ann))
        if with_model: zf.write(DEMO / "models/trusted_reference.npz", "model/model.npz")
    return z


def test_health_and_index():
    assert client.get("/api/health").json() == {"status": "ok"}
    assert "Cognitive" in client.get("/").text


def test_demo_info_reflects_repo():
    d = client.get("/api/demo/info").json()
    assert d["available"] and d["images"] == len(json.loads((DEMO / "dataset/annotations.json").read_text())["images"])


def test_default_demo_end_to_end():
    rid = client.post("/api/demo/run").json()["run_id"]
    assert wait(rid)["state"] == "done"
    r = client.get(f"/api/result/{rid}").json()
    assert r["system_status"] in ("PASS", "REVIEW", "QUARANTINE", "ACCEPT")
    assert set(r["parts"]) == {"p1", "p2", "p3", "p4", "p5"}
    assert r["audit"]["status"] in ("OK", "BROKEN") and r["audit"]["events"] == r["audit"]["hmac_valid"]
    assert len(r["cases"]) == len({c["case_id"] for c in r["cases"]})
    assert not list(Path(runner.RUNS_ROOT).glob("run_*/out/**/*.key"))   # signing keys removed after the run
    assert str(ROOT) not in json.dumps(r)                       # no server paths leaked


def test_status_unknown_run():
    assert client.get("/api/status/nope").status_code == 404


def test_invalid_zip_rejected(tmp_path):
    r = client.post("/api/validate", files={"file": ("x.zip", b"not a zip", "application/zip")}).json()["validation"]
    assert not r["valid"] and r["errors"]
    assert client.post("/api/validate", files={"file": ("x.txt", b"hi")}).status_code == 400


def test_path_traversal_rejected(tmp_path):
    b = io.BytesIO()
    with zipfile.ZipFile(b, "w") as zf: zf.writestr("../evil.txt", "x"); zf.writestr("images/a.png", "x")
    r = client.post("/api/validate", files={"file": ("x.zip", b.getvalue())}).json()["validation"]
    assert not r["valid"] and "unsafe" in " ".join(r["errors"])


def test_missing_required(tmp_path):
    b = io.BytesIO()
    with zipfile.ZipFile(b, "w") as zf: zf.writestr("images/readme.txt", "x")
    r = client.post("/api/validate", files={"file": ("x.zip", b.getvalue())}).json()["validation"]
    assert not r["valid"] and any("annotations" in e for e in r["errors"])


def test_bad_annotations(tmp_path):
    z = make_zip(tmp_path, bad_ann=True)
    r = client.post("/api/validate", files={"file": ("p.zip", z.read_bytes())}).json()["validation"]
    assert not r["valid"]


def test_run_requires_valid_upload():
    assert client.post("/api/run", json={"upload_id": "missing"}).status_code == 404


def test_custom_zip_end_to_end(tmp_path):
    z = make_zip(tmp_path, n=60)
    up = client.post("/api/upload", files={"file": ("p.zip", z.read_bytes())}).json()
    v = up["validation"]; assert v["valid"] and v["images"] == 60 and v["annotation_format"] == "COCO" and v["model_detected"]
    rid = client.post("/api/run", json={"upload_id": up["upload_id"]}).json()["run_id"]
    s = wait(rid); assert s["state"] == "done", s["error"]
    r = client.get(f"/api/result/{rid}").json()
    assert r["mode"] == "custom" and "p1" in r["parts"] and r["audit"]["status"] == "OK"


def test_custom_zip_without_model_runs_p1_and_p5(tmp_path):
    z = make_zip(tmp_path, n=60, with_model=False)
    up = client.post("/api/upload", files={"file": ("p.zip", z.read_bytes())}).json()
    rid = client.post("/api/run", json={"upload_id": up["upload_id"]}).json()["run_id"]
    s = wait(rid); assert s["state"] == "done", s["error"]
    r = client.get(f"/api/result/{rid}").json()
    assert [k for k, v in r["parts"].items() if v["disposition"] == "UNAVAILABLE"] == ["p2", "p3", "p4"] and s["stages"]["p2"] == "skipped"
