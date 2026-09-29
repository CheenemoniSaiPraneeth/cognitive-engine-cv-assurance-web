"""Machine-readable JSON + analyst-facing HTML report + findings.jsonl + per-sample CSV."""
from __future__ import annotations

import csv
import html
import json
from pathlib import Path

SEV_ORDER = {"critical": 0, "high": 1, "medium": 2, "low": 3}
COLORS = {"ACCEPT": "#2e7d32", "REVIEW": "#ef6c00", "QUARANTINE": "#c62828", "UNAVAILABLE": "#6d6d6d"}


def _badge(d):
    return f'<span class="b" style="background:{COLORS.get(d, "#555")}">{html.escape(d)}</span>'


def write_reports(results: dict, samples, out: Path):
    out = Path(out)
    (out / "assurance_part1.json").write_text(json.dumps(results, indent=2, default=str))
    with open(out / "findings.jsonl", "w") as fh:
        for f in results["findings"]:
            fh.write(json.dumps(f, default=str) + "\n")
    with open(out / "sample_flags.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["sample_id", "file_name", "source_id", "contributor_id", "batch_id", "sha256", "flags"])
        for s in samples:
            w.writerow([s.sample_id, s.file_name, s.source_id or "", s.contributor_id or "", s.batch_id or "",
                        s.image_sha256 or "", ";".join(results["sample_flags"].get(s.file_name, []))])
    (out / "report.html").write_text(_html(results))


def _html(r: dict) -> str:
    e = html.escape
    m = r["dataset_manifest"]
    fs = sorted(r["findings"], key=lambda f: (SEV_ORDER.get(f["severity"], 9), f["category"]))
    rows = ""
    for f in fs[:150]:
        ev = json.dumps(f["evidence"], default=str)
        if len(ev) > 420:
            ev = ev[:420] + " ..."
        rows += (f"<tr><td>{e(f['finding_id'])}</td><td>{e(f['asset_type'])}<br><small>{e(str(f['asset_id']))}</small></td>"
                 f"<td>{e(f['category'])}</td><td>{e(f['severity'])}</td><td>{f['confidence']}</td>"
                 f"<td>{_badge(f['disposition'])}</td><td class='ev'>{e(ev)}</td></tr>")
    src = ""
    for name, s in r["source_summaries"].items():
        src += (f"<tr><td>{e(name)}</td><td>{s['sample_count']}</td><td>{s['duplicate_rate']}</td>"
                f"<td>{s['label_anomaly_rate']}</td><td>{s['ood_rate']}</td><td>{s['trigger_candidate_rate']}</td>"
                f"<td>{s['cross_signal_count']}</td><td>{_badge(s['disposition'])}</td></tr>")
    cats = "".join(f"<li>{e(k)}: <b>{v}</b></li>" for k, v in sorted(r["finding_counts"].items()))
    una = "".join(f"<li>{e(u)}</li>" for u in r["unavailable_tests"]) or "<li>none</li>"
    lim = "".join(f"<li>{e(u)}</li>" for u in r["limitations"])
    cls = r["profile"]["classes"]["frequency"]
    cls_html = ", ".join(f"{e(str(k))}: {v}" for k, v in cls.items())
    return f"""<!doctype html><html><head><meta charset="utf-8"><title>Part 1 - Data Integrity Report</title>
<style>body{{font-family:system-ui,Arial;margin:2rem;color:#222}}table{{border-collapse:collapse;width:100%;font-size:13px}}
td,th{{border:1px solid #ddd;padding:5px;vertical-align:top}}th{{background:#f3f3f3;text-align:left}}
.b{{color:#fff;padding:2px 8px;border-radius:10px;font-size:12px}}.ev{{font-family:monospace;font-size:11px;max-width:520px;word-break:break-all}}
h1{{margin-bottom:0}}small{{color:#666}}</style></head><body>
<h1>Computer-Vision Assurance - Part 1: Training-Data Integrity</h1>
<p><small>Detector {e(r['detector_version'])} &middot; {e(r['created_at'])} &middot; zero-training, offline</small></p>
<h2>Dataset status {_badge(r['disposition'])}</h2>
<p>Dataset <b>{e(m['dataset_id'])}</b> ({e(m['format'])}) &middot; {m['sample_count']} images &middot; {m['annotation_count']} annotations<br>
SHA-256: <code>{e(m['dataset_sha256'])}</code><br>Config SHA-256: <code>{e(r['configuration_sha256'])}</code><br>
Trusted reference: {e(str(m['trusted_reference']))}</p>
<h3>Finding counts</h3><ul>{cats}</ul>
<h3>Source / contributor aggregation</h3>
<table><tr><th>Source</th><th>Samples</th><th>Dup rate</th><th>Label-anomaly rate</th><th>OOD rate</th><th>Trigger-cand. rate</th><th>Cross-signals</th><th>Disposition</th></tr>{src or '<tr><td colspan=8>No source metadata available (UNAVAILABLE)</td></tr>'}</table>
<h3>Findings (top 150 by severity)</h3>
<table><tr><th>ID</th><th>Asset</th><th>Category</th><th>Severity</th><th>Conf.</th><th>Disposition</th><th>Evidence</th></tr>{rows}</table>
<h3>Dataset profile (class frequency)</h3><p>{cls_html}</p>
<h3>Unavailable / degraded tests (not a PASS)</h3><ul>{una}</ul>
<h3>Limitations</h3><ul>{lim}</ul>
<p><small>Language note: findings are anomalies / candidates for analyst review, not proof of malicious intent.</small></p>
</body></html>"""
