# Cognitive Engine · CV Assurance (web)

FastAPI + static dashboard around the unchanged P1–P5 core. The CLI (`python -m cv_assurance ...`) still works.

Run locally: `pip install -r requirements-web.txt && uvicorn webapp.app:app --port 8000` → http://localhost:8000
Tests: `python -m pytest -q`

Default demo = `data_demo/` (dataset, trojan + trusted models, two streams), executed via the same
`run_integrated5` the CLI uses. Custom ZIP: `images/`, `annotations/annotations.json` (COCO) required;
`model/` (`*.npz|onnx|pt`, optional `trusted_*`), `streams/<name>/`, `metadata/` optional.

Deploy: push to GitHub → Render → New Blueprint (uses `render.yaml` + `Dockerfile`, binds `0.0.0.0:$PORT`).
