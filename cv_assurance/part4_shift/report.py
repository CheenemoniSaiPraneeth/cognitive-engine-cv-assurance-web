"""JSON + analyst HTML report (inline SVG drift timeline; no external assets - air-gap safe)."""
from __future__ import annotations
import html, json
from pathlib import Path
COL = {"NORMAL": "#2e7d32", "SHIFT DETECTED": "#ef6c00", "SEVERE SHIFT": "#c62828", "ACCEPT": "#2e7d32", "REVIEW": "#ef6c00", "QUARANTINE": "#c62828"}
e = html.escape
def badge(x): return f'<span style="background:{COL.get(x, "#555")};color:#fff;padding:2px 9px;border-radius:10px;font-size:12px">{e(str(x))}</span>'

def _svg(batches, tm):
    W, H, m = 720, 220, 40; n = len(batches); top = max(max(b["ratio"] for b in batches), 1.5) * 1.15; bw = (W - 2 * m) / n
    y = lambda v: H - m - (H - 2 * m) * v / top
    s = f'<svg viewBox="0 0 {W} {H}" width="100%" style="max-width:{W}px;background:#fafafa;border:1px solid #ddd">'
    for i, b in enumerate(batches):
        x = m + i * bw + bw * 0.15; c = "#c62828" if b["alert"] else "#7cb342"
        s += f'<rect x="{x:.1f}" y="{y(b["ratio"]):.1f}" width="{bw * .7:.1f}" height="{H - m - y(b["ratio"]):.1f}" fill="{c}"/><text x="{x + bw * .35:.1f}" y="{H - m + 14}" font-size="10" text-anchor="middle">{e(str(b["batch_id"]))[-6:]}</text>'
        s += f'<text x="{x + bw * .35:.1f}" y="{y(b["ratio"]) - 3:.1f}" font-size="9" text-anchor="middle">{b["ratio"]}</text>'
    s += f'<line x1="{m}" x2="{W - m}" y1="{y(1):.1f}" y2="{y(1):.1f}" stroke="#333" stroke-dasharray="4"/><text x="{W - m}" y="{y(1) - 3:.1f}" font-size="10" text-anchor="end">alert threshold (calibrated on clean reference)</text>'
    if tm.get("status") == "OK" and tm.get("cusum"):
        cu = tm["cusum"]; mx = max(max(cu), 1e-6); pts = " ".join(f"{m + (i + .5) * bw:.1f},{H - m - (H - 2 * m) * min(v / mx, 1) * .8:.1f}" for i, v in enumerate(cu))
        s += f'<polyline points="{pts}" fill="none" stroke="#1565c0" stroke-width="2"/><text x="{m}" y="14" font-size="10" fill="#1565c0">CUSUM (blue, normalised)</text>'
    return s + "</svg>"

def write_reports(r, out):
    out = Path(out); (out / "assurance_part4.json").write_text(json.dumps(r, indent=2, default=str))
    with open(out / "findings_part4.jsonl", "w") as fh:
        for f in r["findings"]: fh.write(json.dumps(f, default=str) + "\n")
    (out / "report_part4.html").write_text(_html(r))

def _html(r):
    tm = r["temporal"]; ov = r["overall"]
    rows = "".join(f"<tr><td>{e(str(b['batch_id']))}</td><td>{b['n']}</td><td>{b['ratio']}</td><td>{badge('ALERT') if b['alert'] else 'ok'}</td><td>{e(', '.join(b['families']))}</td><td>{b['ood_rate']}</td></tr>" for b in r["batches"])
    fam = "".join(f"<tr><td>{e(f['family'])}</td><td>{f['score']}</td><td>{e(', '.join(t.split(':')[-1] for t in f['tests'][:5]))}</td></tr>" for f in r["overall_attribution"]["families"]) or "<tr><td colspan=3>none</td></tr>"
    ft = "".join(f"<tr><td>{e(x['feature'])}</td><td>{x['direction']}</td><td>{x['ref_mean']}</td><td>{x['cur_mean']}</td><td>{x['effect_size']}</td></tr>" for x in r["overall_attribution"]["features"]) or "<tr><td colspan=5>none</td></tr>"
    src = "".join(f"<tr><td>{e(str(s['source']))}</td><td>{s['n']}</td><td>{s['abnormal_rate']}</td><td>{'NEW' if s['new_source'] else ''}</td></tr>" for s in r["overall_attribution"].get("sources", [])) or f"<tr><td colspan=4>{e(r['overall_attribution'].get('source_note', 'none'))}</td></tr>"
    fd = "".join(f"<tr><td>{f['finding_id']}</td><td>{e(f['asset_type'])}: {e(f['asset_id'])}</td><td>{e(f['category'])}</td><td>{f['severity']}</td><td>{f['confidence']}</td><td>{badge(f['disposition'])}</td><td style='font-family:monospace;font-size:11px;word-break:break-all'>{e(json.dumps(f['evidence'], default=str)[:300])}</td></tr>" for f in r["findings"][:60])
    una = "".join(f"<li>{e(u)}</li>" for u in r["unavailable_tests"]) or "<li>none</li>"; lim = "".join(f"<li>{e(u)}</li>" for u in r["limitations"])
    tmh = (f"<p>Pattern: <b>{e(str(tm['pattern']))}</b> &middot; onset batch: <b>{e(str(tm['onset_batch']))}</b> &middot; CUSUM first alert: {tm['cusum_alert_batch']} &middot; affected: {e(', '.join(tm['affected_families']) or '-')}</p>" if tm["status"] == "OK" else f"<p>UNAVAILABLE: {e(tm['reason'])}</p>")
    return f"""<!doctype html><html><head><meta charset=utf-8><title>Part 4 - Distribution Shift</title><style>body{{font-family:system-ui;margin:2rem;color:#222}}table{{border-collapse:collapse;width:100%;font-size:13px;margin-bottom:1rem}}td,th{{border:1px solid #ddd;padding:5px;text-align:left;vertical-align:top}}th{{background:#f3f3f3}}</style></head><body>
<h1>Computer-Vision Assurance - Part 4: Distribution Shift &amp; Anomaly</h1>
<p><small>{e(r['detector_version'])} &middot; reference {e(r['reference']['reference_id'])} (n={r['reference']['n']}) &middot; config {r['config_sha256'][:12]} &middot; zero-training, offline</small></p>
<h2>Status {badge(r['level'])} {badge(r['disposition'])}</h2><p>{e(r['note'])}</p>
<h3>Drift timeline (ratio = evidence / calibrated alert threshold)</h3>{_svg(r['batches'], tm)}{tmh}
<h3>Batches</h3><table><tr><th>batch</th><th>n</th><th>ratio</th><th>alert</th><th>families</th><th>OOD rate</th></tr>{rows}</table>
<h3>Attribution (overall)</h3><table><tr><th>family</th><th>score</th><th>tests</th></tr>{fam}</table>
<table><tr><th>feature</th><th>dir</th><th>ref mean</th><th>cur mean</th><th>effect (ref-sd)</th></tr>{ft}</table>
<table><tr><th>source</th><th>n</th><th>abnormal rate</th><th></th></tr>{src}</table>
<h3>Findings</h3><table><tr><th>id</th><th>asset</th><th>category</th><th>sev</th><th>conf</th><th>disp</th><th>evidence</th></tr>{fd}</table>
<h3>Unavailable tests (not a PASS)</h3><ul>{una}</ul><h3>Limitations</h3><ul>{lim}</ul></body></html>"""
