# Part 2 - Model Integrity (zero-training prototype)

Offline, model-agnostic, **no training anywhere in the assessment path**. Findings use the same `Finding` schema and
hash-linked audit log as Part 1 (finding ids are `M-xxxxx`, Part 1's are `F-xxxxx`).

## Run
```
python3 -m cv_assurance part2 demo --out ./p2_demo              # builds benchmark models, scores 4 access modes
python3 -m cv_assurance part2 integrated-demo --out ./int_demo  # Part 1 dataset + Part 2 model -> combined report
python3 -m cv_assurance part2 run --model M.npz --out OUT --reference-model TRUSTED.npz --reference-images DIR [--blackbox]
python3 -m cv_assurance part2 integrated --dataset DS --model M.npz --out OUT --reference-model TRUSTED.npz
python3 -m unittest discover -s tests -v
```
`--reference-images DIR` = trusted clean images (+ optional `labels.json` {file: class_index}).
`--blackbox` hides weights/activations so white-box tests report UNAVAILABLE.

## Modules (all 8 implemented)
| # | Module | File | Notes |
|---|---|---|---|
| 1 | Ingestion / adapters | `adapters.py` | NumPy-MLP `.npz` (tested), ONNX, TorchScript/state_dict (**written, not run in the offline sandbox - onnx/torch not installed**), black-box wrapper. Safe loading (no pickle, `weights_only=True`, refuses ONNX external data/custom ops). |
| 2 | Fingerprint | `fingerprint.py` | SHA-256, size, architecture signature, expected-vs-received identity |
| 3 | Parameter analysis | `params.py` | tensor stats, reference diff (rel-L2, localized-change detection), dormant-high-influence units (reference-free) |
| 4 | Behavioural fingerprint | `behaviour.py` | prediction/probability hashes, class freq, confidence/entropy, determinism |
| 5 | Reference battery | `behaviour.py` | 15 fixed variants (brightness, contrast, JPEG, resize, crop, noise, blur, occlusion), agreement vs trusted model, accuracy |
| 6 | Trigger tests | `trigger.py` | 12 patterns x 5 positions x 4 sizes + Part 1 dataset-derived candidates; ASR, target concentration, reference contrast |
| 7 | Substitution | `substitution.py` | identity + architecture + parameters + behaviour + trigger -> verdict |
| 8 | Assurance engine | `assurance.py`, `report.py` | ModelAssurance JSON, HTML, findings.jsonl, audit chain |

## Integration with Part 1 (`integration.py`)
* Part 1 -> Part 2: unflagged, labelled samples from non-flagged sources become the clean reference battery; Part 1 trigger candidates (pixels + location) are pasted onto clean images and tested against the model.
* Part 2 -> combined: model-vs-label agreement per Part 1 flag category; `CONVERGING_TRIGGER_EVIDENCE` -> QUARANTINE; `assurance_combined.json` + `report_combined.html`.

## Honest limits
* Benchmark fixtures are tiny MLPs (the fixture builder fits the output layer in closed form; the assessment never trains).
* Trigger scan only covers the pattern library + Part 1 candidates. `trojan_stealth` is outside the library: found by white-box tests only (its trigger scan misses).
* Without a trusted reference model/digest, substitution and behavioural-divergence tests are UNAVAILABLE (coverage=LIMITED), not PASS.
* Thresholds live in `part2_model/config.py`, tuned on the clean fixtures only; expect re-calibration on real models.
