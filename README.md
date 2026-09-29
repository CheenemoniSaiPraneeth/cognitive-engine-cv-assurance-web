# CV Assurance Framework - Part 1: Training-Data Integrity (zero-training prototype)

Offline, air-gapped, model-agnostic **prototype** of Part 1 from the architecture document.
**No neural network is trained on the contributed data.** All detection is deterministic or unsupervised:
cryptographic hashes, perceptual hashes, kNN / clustering, robust statistics, Local Outlier Factor,
Isolation Forest, (robust) Mahalanobis distance. Embeddings come from either a built-in handcrafted
feature extractor (default, needs no weights) or an optional **frozen** pretrained ONNX encoder.

## Quick start
```bash
pip install -r requirements.txt            # numpy, Pillow, scipy, scikit-learn (optional: onnxruntime)

# 1) End-to-end demo: generates a synthetic dataset with known attacks, runs Part 1, scores it
python -m cv_assurance demo --out ./demo_out                 # COCO
python -m cv_assurance demo --out ./demo_out_yolo --format yolo
#    -> open demo_out/assurance_out/report.html

# 2) Run on your own dataset (COCO json or YOLO folder; format auto-detected)
python -m cv_assurance run --dataset /path/to/dataset --out ./out
python -m cv_assurance run --dataset ds --out out --reference /path/to/TRUSTED_clean_images   # calibrated OOD
python -m cv_assurance run --dataset ds --out out --config configs/default.json

# 3) Re-score an existing run against a ground-truth manifest
python -m cv_assurance evaluate --results out/assurance_part1.json --ground-truth ds/ground_truth.json

# Tests (standard library only)
python -m unittest discover -s tests -v
```

## Input formats
* **COCO**: any `*.json` with `images/annotations/categories` (images searched next to it and in `images/`).
  Optional per-image `source_id`, `contributor_id`, `batch_id` fields.
* **YOLO**: `images/` + `labels/` (or side-by-side `.txt`), class names from `data.yaml` / `classes.txt`.
  Optional `sources.csv` (`file_name,source_id,contributor_id,batch_id`) supplies source metadata (also works for COCO).

## What is implemented (maps to the document)
| Module | File | Evidence produced |
|---|---|---|
| 1 Ingestion | `dataset/ingest.py` | missing/unreadable files, malformed annotations, invalid class IDs, impossible boxes, duplicate identifiers, size mismatch, orphan annotations |
| 2 Profiling | `part1_data/profiling.py` | volume, class frequency/imbalance, image size/aspect/brightness/sharpness, box stats, centre heat-map, source proportions, quality counts |
| 3 Fingerprinting | `part1_data/fingerprint.py`, `features.py` | SHA-256, pHash, dHash, frozen/handcrafted embedding, quality stats |
| 4 Duplicates / flooding | `part1_data/duplicates.py` | exact / near / semantic groups, flooding evidence (size, fraction, class + source concentration), conflicting labels inside duplicate groups |
| 5 Label integrity | `part1_data/label_integrity.py` | weighted kNN agreement, local label entropy, k-means cluster purity (duplicates excluded from voting so floods cannot mask flips) |
| 6 OOD / anomaly | `part1_data/ood.py` | kNN distance, Mahalanobis, LOF, Isolation Forest (embedding view) + robust Mahalanobis / IF / kNN (engineered-statistics view); vote-based, reference-calibrated |
| 7 Trigger candidates | `part1_data/trigger.py` | recurring localised patch at a consistent position across *unrelated* samples, pixel consistency, label/source correlation (binomial test). Reports *candidates*, never a proven backdoor |
| 8 Source aggregation | `part1_data/source_aggregation.py` | per source/contributor/batch rates, elevated signals, cross-signal count, ACCEPT/REVIEW/QUARANTINE |
| Evidence/audit | `common/schemas.py`, `common/audit.py`, `reports/report.py` | common Finding schema, hash-linked audit log, JSON + HTML report, findings.jsonl, per-sample CSV |

Also included: a synthetic benchmark generator with attack manifests (`benchmark/synthetic.py`) and a scorer
(`benchmark/evaluate.py`) reporting per-attack precision/recall/F1 and the clean-set false-positive rate.

## Outputs (`--out`)
`assurance_part1.json` (full machine-readable package incl. manifest, config, thresholds, environment, profile, findings),
`report.html` (analyst view), `findings.jsonl`, `sample_flags.csv`, `audit_log.jsonl` (hash-chained; `AuditLog.verify()` detects edits).

## Demo results (synthetic COCO, seed 7, default thresholds, handcrafted embeddings)
See `examples/sample_evaluation.json` and `examples/sample_report.html`. Numbers on *synthetic* data only - they show the
pipeline works end to end, not real-world accuracy.

## Honest limitations
* Default handcrafted embeddings are far weaker than a pretrained encoder; for real data use `embedding.backend = "onnx"`
  with a locally stored, frozen model (its SHA-256 is recorded in the report).
* Without `--reference`, the dataset is its own OOD reference (self-reference mode). The report states this under
  `unavailable_tests`. Use a trusted clean reference for calibrated results.
* Trigger detection finds patch-like patterns that recur at fixed positions; blended / low-opacity / input-dependent
  triggers can be missed, and a natural fixed-position pattern can look like a candidate. Dataset evidence cannot prove a backdoor.
* Thresholds in `common/config.py` were tuned on the synthetic benchmark; recalibrate on your clean reference data.
* Findings are anomalies/candidates for analyst review - not proof of malicious intent. A finding-free run is not a guarantee of safety.
* Prototype scale: pairwise hash comparison and kNN are brute force (fine to ~20k images).


## Part 4 (distribution shift) - added
See `README_PART4.md`. `python -m cv_assurance part4 demo|run|build-reference|integrated-demo|integrated`. Tests: `python -m unittest tests.test_part4`.


## Part 5 (analyst assurance & governance)
See README_PART5.md. `python -m cv_assurance part5 integrated-demo --out demo_p12345` runs Parts 1-5 end to end; `python -m cv_assurance part5 demo --out out5` runs Part 5 alone on bundled sample outputs.
