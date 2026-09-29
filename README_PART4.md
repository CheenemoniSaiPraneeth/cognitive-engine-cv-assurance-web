# Part 4 - Distribution Shift & Anomaly Assessment (prototype, zero-training, offline)

Answers: *has the operational population moved away from the trusted reference, when, in what way, and from where?*
A shift is **evidence to review, not an attack**. No neural network is trained on any data; KMeans / PCA / Isolation Forest are
unsupervised statistics fitted on the trusted reference only. Optional frozen ONNX encoder: `--onnx-encoder` (never updated).

## Modules (architecture doc, Part 4)
| # | Module | File | Status |
|---|--------|------|--------|
| 1 | Reference builder (hashed snapshot `REF-...`) | `data.py` | working |
| 2 | Feature distribution statistics (illumination, colour, geometry, sharpness/noise, compression, task output) | `features.py` | working |
| 3 | Embedding-space comparison (centroid, kNN distance, MMD, cluster occupancy, new-cluster discovery) | `engine.py`, `ood.py` | working |
| 4 | Statistical shift (KS, PSI, JSD, Wasserstein, class-prior JSD) - **calibrated on clean-reference splits** | `stats.py`, `engine.py` | working |
| 5 | OOD / anomaly per sample (kNN + Mahalanobis + IsolationForest vote, binomial excess test) | `ood.py` | working |
| 6 | Temporal / batch drift (onset, step / gradual / transient / recurring, CUSUM early warning) | `temporal.py` | working |
| 7 | Shift attribution (families, features + direction, per-source / new-source) | `attribution.py` | working, see limits |
| 8 | Shift assurance: NORMAL / SHIFT DETECTED / SEVERE SHIFT -> ACCEPT / REVIEW, findings, JSON+HTML, hash-linked audit | `assess.py`, `report.py` | working |

Calibration: for the batch size in use, the engine draws many (pseudo-current, pseudo-reference) splits of the *clean reference*,
builds a null distribution per test and for the max over all tests, and alerts only above its 99th percentile. Thresholds therefore
come from clean data only and are cached in the reference folder (`calibration_n*.json`). `ratio` = evidence / alert threshold (>1 = alert).

## Run
```
python -m cv_assurance part4 demo            --out demo_out_part4      # controlled benchmark: 10 windows + 4 temporal series, scored
python -m cv_assurance part4 build-reference --images TRUSTED_DIR --out REF [--model M.npz]
python -m cv_assurance part4 run             --reference REF --ops OPS_DIR --out OUT [--model M.npz] [--window 50]
python -m cv_assurance part4 integrated-demo --out demo_out_p1234      # Parts 1+2+3+4 together (full bundle)
python -m cv_assurance part4 integrated --dataset D --model SERVED --trusted-model T --stream NAME=DIR --out OUT
```
`OPS_DIR` may contain `metadata.json` = `{file: {"source":..,"batch_id":..,"timestamp":..}}`; otherwise fixed windows are used.
Model (`--model`, black-box) is optional: without it the task-output / class-prior tests are reported **UNAVAILABLE** (not PASS).

## Integration with Parts 1-3 (`integration.py`)
* Part 1: default trusted reference = dataset samples Part 1 did **not** flag (production: pass `--reference-dir`).
* Part 2: trusted model produces reference task outputs; served model's Part-2 disposition feeds the correlation.
* Part 3: each operational stream is recorded into a signed hash-chained ledger and verified; Part 4 reads the **preserved input bytes
  (sha256-checked) and recorded outputs from the verified ledger**, so every shifted sample maps to a `record_id`.
* Cross-signals: `OPERATIONAL_SHIFT_WITH_INTACT_PROVENANCE` (REVIEW), `OUTPUT_SHIFT_WITHOUT_INPUT_SHIFT` (REVIEW/QUARANTINE with Part 2),
  `CONVERGING_SHIFT_AND_PROVENANCE_FAILURE` (QUARANTINE). Shift alone never exceeds REVIEW.

## Prototype results (synthetic benchmark, seed fixed)
* Shift-scenario recall **6/6** (illumination, camera, compression, class-prior, semantic, mixed); 4 stable windows -> **0** false alarms; stable series 0/8 batches alerted.
* Temporal: step onset delay 0 batches, gradual 1 batch (CUSUM flags earlier), transient correctly labelled, pattern accuracy 4/4.
* False-alarm rate on clean batches of size 30/60/100: 0.0 (12 draws each - small sample, not a proof).

## Honest limits of this prototype
* **Attribution is the weakest module.** Synthetic transforms have real cross-effects (e.g. lowering contrast also changes noise and bytes/pixel),
  so *all flagged families* contain the injected one (recall 1.0) but the *top-ranked* families match the injected cause only ~47% of the time
  (precision ~32%). Treat families as "likely contributors", ordered loosely.
* Degenerate features (zero variance in the reference, e.g. constant image size) give very large scores on any change; scores are capped (ratio <= 999).
* Handcrafted embeddings are weaker semantically than a frozen pretrained encoder. Synthetic images only; no real-camera validation.
* Single-batch alerts still have alpha_global (1%) false-alarm probability; a drift is called only with >=2 persistent batches, otherwise "transient".
* Cannot say *why* a shift occurred; a legitimate new camera/season looks identical to a hostile change.
