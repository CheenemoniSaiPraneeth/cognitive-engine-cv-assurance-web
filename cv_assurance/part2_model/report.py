"""JSON + HTML report for Part 2."""
from __future__ import annotations

import html
import json
from pathlib import Path

COL = {"ACCEPT": "#2e7d32", "REVIEW": "#ef6c00", "QUARANTINE": "#c62828", "UNAVAILABLE": "#6d6d6d"}
SEV = {"critical": 0, "high": 1, "medium": 2, "low": 3, "info": 4}


def write_reports(r: dict, out: Path):
    out = Path(out)
    (out / "assurance_part2.json").write_text(json.dumps(r, indent=2, default=str))
    (out / "findings_part2.jsonl").write_text("".join(json.dumps(f, default=str) + "\n" for f in r["findings"]))
    (out / "report_part2.html").write_text(_html(r))


def _b(d): return f'<span class="b" style="background:{COL.get(d, "#555")}">{html.escape(str(d))}</span>'


def _html(r):
    e = html.escape
    rows = ""
    for f in sorted(r["findings"], key=lambda f: (SEV.get(f["severity"], 9), f["category"])):
        ev = json.dumps(f["evidence"], default=str); ev = ev[:520] + " ..." if len(ev) > 520 else ev
        rows += (f"<tr><td>{e(f['finding_id'])}</td><td>{e(f['category'])}</td><td>{e(f['severity'])}</td><td>{f['confidence']}</td>"
                 f"<td>{_b(f['disposition'])}</td><td class='ev'>{e(ev)}</td></tr>")
    una = "".join(f"<li>{e(u)}</li>" for u in r.get("unavailable_tests", [])) or "<li>none</li>"
    lim = "".join(f"<li>{e(u)}</li>" for u in r["limitations"])
    ms = "".join(f"<li>{e(k)}: <b>{e(str(v))}</b></li>" for k, v in r.get("module_status", {}).items())
    return f"""<!doctype html><html><head><meta charset="utf-8"><title>Part 2 - Model Integrity</title>
<style>body{{font-family:system-ui,Arial;margin:2rem;color:#222}}table{{border-collapse:collapse;width:100%;font-size:13px}}
td,th{{border:1px solid #ddd;padding:5px;vertical-align:top}}th{{background:#f3f3f3;text-align:left}}
.b{{color:#fff;padding:2px 8px;border-radius:10px;font-size:12px}}.ev{{font-family:monospace;font-size:11px;max-width:560px;word-break:break-all}}</style></head><body>
<h1>Computer-Vision Assurance - Part 2: Model Integrity</h1>
<p>{e(r['detector_version'])} &middot; {e(r['created_at'])} &middot; zero-training, offline</p>
<h2>Status {_b(r['disposition'])}</h2>
<p>Model <b>{e(str(r.get('model_id')))}</b> &middot; access <b>{e(str(r.get('access_mode')))}</b> &middot; identity <b>{e(str(r.get('identity_status')))}</b>
&middot; coverage <b>{e(str(r.get('coverage')))}</b> &middot; substitution verdict <b>{e(str(r.get('substitution', {}).get('verdict')))}</b></p>
<h3>Module status</h3><ul>{ms}</ul>
<h3>Findings</h3><table><tr><th>ID</th><th>Category</th><th>Severity</th><th>Conf.</th><th>Disposition</th><th>Evidence</th></tr>{rows}</table>
<h3>Unavailable tests (not a PASS)</h3><ul>{una}</ul><h3>Limitations</h3><ul>{lim}</ul></body></html>"""
