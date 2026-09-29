# Part 5 - Analyst Assurance & Governance (prototype)

Control / evidence layer over Parts 1-4. **No model is trained; nothing touches the network.** It reads the
`assurance_part1..4.json` files the other Parts already write, and produces a traceable, hash-audited, analyst-facing package.

```
Part1..4 JSON -> [1 Aggregator] -> [2 Correlation] -> [3 Risk/Severity] -> [4 Analyst dashboard] -> [6 Report + sealed export]
                       └────────────── [5 Audit trail: hash-linked + HMAC, decisions fold from it] ──────────────┘
```

| Module | File | What it does |
|---|---|---|
| 1 Evidence Aggregator | `aggregate.py` | discovers Part 1-4 artifacts, normalises all findings to one schema (namespaced ids, artifact sha256 kept), turns unavailable tests into explicit `UNAVAILABLE` items, extracts entity links |
| 2 Cross-signal correlation | `correlate.py` | rule engine: source convergence, data->model trigger, model->inference, shift+provenance, source-in-data-and-shift, isolated signals (capped), missing evidence |
| 3 Risk / severity | `risk.py`, `config.py` | transparent noisy-OR *review priority* (severity x confidence x category weight, +volume, +independence). Not a "maliciousness score". Weights live in an editable policy JSON |
| 4 Analyst view | `report.py` | offline HTML: status tiles, priority findings, case cards (why / evidence / next step / trace chain / score breakdown), source table, coverage gaps, finding explorer, staged decisions |
| 5 Audit trail | `audit.py` | append-only, hash-linked, HMAC-signed events; persists across sessions; tamper detection |
| 6 Report / governance | `assemble.py`, `governance.py` | JSON + HTML, reproducibility block, sealed export manifest; actions accept / review / quarantine / retest / close with rationale rules |

## Quick start
```bash
pip install numpy                                 # stand-alone Part 5 needs numpy only
python -m cv_assurance part5 demo --out out5      # 2 bundled scenarios (compromised model / trusted model)
open out5/compromised_model/report_part5.html
python -m cv_assurance part5 show   --out out5/compromised_model
python -m cv_assurance part5 act    --out out5/compromised_model --case C-xxxxxxxx --action quarantine --actor ana --rationale "isolating contributor_x"
python -m cv_assurance part5 verify --out out5/compromised_model
python -m cv_assurance part5 assemble --results <dir with assurance_part*.json> [more dirs] --out OUT [--policy my_policy.json]
```
Full pipeline (needs the Part 1-4 bundle): `python -m cv_assurance part5 integrated-demo --out demo_p12345`
(builds a synthetic dataset with injected attacks, runs Parts 1-4 with a trusted and a Trojan model, then Part 5 on each).

## Governance rules
* `accept` / `close` need a rationale (>= 15 chars). Accepting a case suggested `QUARANTINE` needs `--override` (recorded).
* Every decision is an audit event carrying the case's `evidence_hash`; if evidence later changes the decision is flagged **stale** and no longer suppresses the case.
* Dashboard decisions are staged in-page -> `analyst_actions.json` -> `part5 apply` (keeps the dashboard a static, offline file).
* `part5 verify` checks audit chain + HMAC signatures + sealed manifest + report/JSON hashes + that the Part 1-4 input files are unchanged.

## Outputs (in `--out`)
`assurance_part5.json` (machine-readable) · `report_part5.html` · `part5_analysis.json` · `audit_part5.jsonl` · `export_manifest.json` (HMAC-sealed) · `part5.key` (keep offline)

## Prototype limits (honest)
Rule-based correlation only; scores are review-priority, not probabilities; entity linking for Part 3/4 -> model assumes a single supplied Part 2 model (stated in `assumptions`);
the key sits next to the export (in production hold it offline / in an HSM); a privileged host attacker is out of scope; not covered yet: multi-dataset/multi-model sessions, dashboard-native decision writing, PDF export.
