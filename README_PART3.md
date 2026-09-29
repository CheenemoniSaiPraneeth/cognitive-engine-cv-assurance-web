# Part 3 - Inference Provenance & Output Integrity (zero-training prototype)

Offline, air-gapped, **no training anywhere**. Part 3 answers: *is a stored inference record bound to the claimed
input, model, configuration and output, and can the result be independently reproduced?*

## Quick start
```bash
pip install -r requirements.txt
python -m cv_assurance part3 demo --out ./demo_p3                 # valid ledger + 12 controlled attacks, scored
python -m cv_assurance part3 integrated-demo --out ./demo_p123    # Part 1 dataset + Part 2 model + Part 3 ledger (full bundle only)
python -m unittest discover -s tests -v

# your own model + images
python -m cv_assurance part3 record --model M.npz --images imgs/ --out ledger/ [--trusted-model T.npz]
python -m cv_assurance part3 verify --ledger ledger/                # uses ledger/ledger.key, registry.json, policy.json
```

## Module map (architecture document, Part 3)
| Module | File | What it does |
|---|---|---|
| 1 Input capture | `capture.py` | exact input **bytes** preserved (`inputs/<sha>.png`), SHA-256, size, format, source, capture time, request id + nonce |
| 2 Model identity | `capture.py` (+ Part 2 `fingerprint_model`) | model SHA-256, format, architecture signature, parameter count; trusted **registry** (`registry.json`) of approved digests |
| 3 Environment/config | `capture.py` | preprocessing schema, class names, top-k, decision threshold, seed, precision, python/numpy/PIL/platform/runtime -> `config_sha256` |
| 4 Output capture | `canonical.py` | canonical output (fixed decimals, stable ordering) -> `output_sha256`; classification implemented, detection canonicalisation + IoU comparator implemented (unit-tested only) |
| 5 Provenance chain | `ledger.py` | append-only JSONL, each record hash-linked to the previous, **HMAC-SHA256 signed** with an offline key, signed batch manifest (count + head hash) |
| 6 Re-inference | `verifier.py`, `canonical.compare_outputs` | re-run the *claimed* model (found by digest in the registry) on the stored input bytes with the recorded config; EXACT / TOLERANCE_MATCH / MISMATCH (incl. decision flips at the threshold boundary) |
| 7 Replay/binding | `binding.py` | duplicate request-id/nonce, sequence gaps, stale record (recorded long after inference), clock running backwards, output re-use across different inputs |
| Assurance | `verifier.py` | findings in the shared Finding schema (`P-xxxxx`), per-record status, UNAVAILABLE (never a silent PASS), JSON + JSONL + HTML, hash-linked audit log |

## Integration with Parts 1 and 2 (`part3_inference/integration.py`)
* **Part 1 -> Part 3**: the inputs come from the Part 1 dataset; every ledger record carries `links` = `{dataset_id, sample_id, part1_flags, part2_disposition}`.
* **Part 2 -> Part 3**: model identity/registry uses the Part 2 fingerprint; Part 2's model disposition rides along in each record.
* **Correlation** (cross-signals in `assurance_combined_p123.json`):
  `CONVERGING_INFERENCE_EVIDENCE` (re-inference mismatches concentrate on inputs Part 1 flagged as trigger candidates),
  `UNREGISTERED_MODEL_CORROBORATED_BY_PART2`, `PROVENANCE_VALID_BUT_MODEL_SUSPECT`.
* The Part 1+2 code is untouched; the CLI simply gained a `part3` sub-command.

## Benchmark (synthetic, `benchmark/part3_bench.py`)
A valid ledger (30 records, trusted model) is copied per case and attacked. Attackers: **naive** (edits a field, cannot re-seal) and
**insider** (holds the signing key: re-hashes, re-chains and re-signs — so only *content* checks can catch it). The scorer also re-runs
every case **without the key** to show content checks alone still catch insider attacks.

| Case | Attacker | Caught by |
|---|---|---|
| output tampering | naive / insider | record hash, output hash, self-consistency / re-inference |
| input substitution | naive / insider | input hash / re-inference |
| model substitution (hidden / declared) | insider | re-inference / registry |
| trojan model + triggered inputs claiming trusted digest | insider | re-inference |
| config tampering (consistent / inconsistent) | insider | trusted policy comparison / re-inference + self-consistency |
| replay (exact record / old output on new input) | insider | request-id, nonce, stale-record / re-inference |
| record deletion | insider | sequence gap |

Result on the shipped benchmark: record-level precision 1.0, recall 1.0, clean-record false-positive rate 0.0 (see `examples/`).
**Read this honestly**: the benchmark is synthetic and the detectors were built against it, so these numbers demonstrate the mechanism, not field performance.

## Prototype status / honest limitations
* Working and tested: modules 1-7 for **classification** outputs on the NumPy-MLP reference model; hash chain, HMAC, manifest, re-inference, policy, replay checks, integration.
* Written but **not exercised** in this sandbox: ONNX / TorchScript adapters (inherited from Part 2; no onnxruntime/torch offline). Detection-output canonicalisation and IoU comparison exist and are unit-tested but no detector model was run through the full ledger.
* HMAC is a shared-secret signature (whoever verifies can also forge). Use asymmetric signatures (e.g. Ed25519) and/or a trusted execution boundary for stronger authenticity - not implemented.
* Provenance cannot prove what happened before capture; a fully privileged host attacker who controls key + registry + policy defeats it.
* Re-inference cannot separate a dormant-trojan model from the clean one on clean inputs - that is Part 2's job (see integrated scenario B/C).
* Config tampering that is internally consistent is only detectable against a trusted `policy.json` (otherwise reported UNAVAILABLE).
* Not implemented: TEE/remote attestation, streaming capture hooks for real serving stacks, Part 4/5.
