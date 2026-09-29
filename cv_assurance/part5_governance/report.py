"""Module 4 - Analyst View: self-contained offline HTML dashboard (no external assets).
Answers: what asset, what was detected, why, what evidence, what to inspect next. Decisions are staged in-page and
exported as analyst_actions.json (applied with `part5 apply`), so every decision lands in the hash-linked audit trail."""
from __future__ import annotations

import html
import json

from .governance import effective_disposition

E = html.escape
COL = {"ACCEPT": "#2e7d32", "REVIEW": "#ef6c00", "QUARANTINE": "#c62828", "UNAVAILABLE": "#78909c", "NORMAL": "#2e7d32", "SHIFT DETECTED": "#ef6c00", "SEVERE SHIFT": "#c62828",
       "HIGH": "#c62828", "ELEVATED": "#ef6c00", "MODERATE": "#f9a825", "LOW": "#78909c", "ACCEPTED": "#2e7d32", "CLOSED": "#546e7a", "OPEN": "#455a64", "IN_REVIEW": "#ef6c00",
       "QUARANTINED": "#c62828", "RETEST_REQUESTED": "#1565c0"}
PART_NAMES = {"1": "Data integrity", "2": "Model integrity", "3": "Inference provenance", "4": "Distribution shift"}


def b(t, cls=""):
    t = str(t); c = COL.get(t.split(" (")[0], "#546e7a")
    return f'<span class="b" style="background:{c}">{E(t)}</span>'


def _ev(e, n=170):
    s = json.dumps(e, default=str)
    return E(s if len(s) <= n else s[:n] + "…")


def _case(c, st, an, maxf):
    eff = effective_disposition(c, st); hist = "".join(
        f"<li>{E(h['timestamp'])} <b>{E(h['actor'])}</b> → {b(h['action'].upper())} {'(override)' if h['override'] else ''} <i>{E(h['rationale'])}</i></li>" for h in st["history"]) or "<li>no analyst decision yet</li>"
    by = {f["uid"]: f for f in an["findings"]}; mem = [by[u] for u in c["finding_uids"]][:maxf]
    rows = "".join(f"<tr><td>{E(f['uid'])}</td><td>{E(str(f['category']))}</td><td>{E(str(f['asset_id']))}</td><td>{E(f['severity'])}</td><td>{f['confidence']:.2f}</td><td>{b(f['disposition'])}</td>"
                   f"<td class=ev>{_ev(f['evidence'])}</td></tr>" for f in mem)
    more = f"<p class=s>… and {c['finding_count'] - len(mem)} more findings (all in assurance_part5.json)</p>" if c["finding_count"] > len(mem) else ""
    contrib = "".join(f"<tr><td>P{g['part']} {E(g['scope'])}</td><td>{E(g['category'])}</td><td>{g['count']}</td><td>{g['max_finding_score']}</td><td>+{g['volume_bonus']}</td><td>{g['group_score']}</td></tr>" for g in c["priority"]["contributions"])
    pr = c["priority"]; trace = " → ".join(f"<code>{E(t)}</code>" for t in c["trace"])
    return f"""<details class=case id="{c['case_id']}" data-disp="{c['suggested_disposition']}"><summary>{b(c['suggested_disposition'])} <b>{c['case_id']}</b> {E(c['title'])}
 <span class=r>priority {b(pr['band'])} {pr['score']} · parts {', '.join('P' + str(p) for p in c['parts']) or '-'} · status {b(st['status'])}</span></summary>
<div class=body><div class=grid><div><h4>Why it was detected</h4><p>{E(c['rationale'])}</p><h4>What to inspect next</h4><p>{E(c['next_action'])}</p>
<p class=s>Rule: <code>{E(c['rule'])}</code> · severity {E(c['severity'])} · effective disposition after governance: {b(eff)}</p><h4>Traceability</h4><p>{trace}</p></div>
<div><h4>Review-priority breakdown</h4><table class=t><tr><th>evidence group</th><th>category</th><th>n</th><th>max</th><th>vol</th><th>score</th></tr>{contrib}</table>
<p class=s>{E(pr['method'])}. base {pr['base_noisy_or']} + independence {pr['independence_bonus']}.</p></div></div>
<h4>Supporting findings</h4><div class=sc><table class=t><tr><th>id</th><th>category</th><th>asset</th><th>sev</th><th>conf</th><th>src disp.</th><th>evidence</th></tr>{rows}</table></div>{more}
<h4>Decision history</h4><ul>{hist}</ul>
<div class=form data-case="{c['case_id']}"><select class=act><option>review</option><option>quarantine</option><option>retest</option><option>accept</option><option>close</option></select>
<input class=who placeholder="analyst" value="analyst"><input class=why placeholder="rationale (required for accept/close)" size=44><label><input type=checkbox class=ovr> override</label>
<button onclick="stage(this)">Stage decision</button></div></div></details>"""


def render_html(d: dict) -> str:
    ov, pol = d["overall"], d["policy"]; cs = d["governance"]["case_state"]; maxf = pol["max_findings_listed_per_case"]
    tiles = ""
    for p in "1234":
        v = d["parts"][p]; lvl = f"<div class=s>level {b(v['level'])}</div>" if v.get("level") else ""
        cnt = sum((v.get("finding_counts") or {}).values()); sub = f"{v['artifacts']} artifact(s) · {cnt} finding(s)" if v["artifacts"] else E(v.get("reason", ""))
        tiles += f"<div class=tile><div class=s>Part {p} · {PART_NAMES[p]}</div><div class=big>{b(v['disposition'])}</div>{lvl}<div class=s>{sub}</div></div>"
    prio = "".join([f"<tr><td><a href='#{c['case_id']}' onclick=\"openCase('{c['case_id']}')\">{c['case_id']}</a></td><td>{E(c['title'])}</td><td>{b(c['suggested_disposition'])}</td><td>{b(c['priority']['band'])} {c['priority']['score']}</td><td>{b(cs[c['case_id']]['status'])}</td></tr>"
                   for c in d["cases"] if c["suggested_disposition"] != "UNAVAILABLE"][:10]) or "<tr><td colspan=5>No cases at REVIEW or above.</td></tr>"
    cases = "".join(_case(c, cs[c["case_id"]], d, maxf) for c in d["cases"])
    src = "".join(f"<tr><td>{E(s['kind'])}:{E(s['name'])}</td><td>{s['sample_count']}</td><td>{s['duplicate_rate']}</td><td>{s['label_anomaly_rate']}</td><td>{s['ood_rate']}</td><td>{s['trigger_candidate_rate']}</td><td>{E(', '.join(s['elevated_signals']) or '-')}</td><td>{b(s['disposition'])}</td></tr>"
                  for s in d["sources"] if s["kind"] != "batch_id") or "<tr><td colspan=8>no Part 1 source summaries</td></tr>"
    unav = "".join(f"<li><code>P{u['part']} {E(u['scope'])}</code> {E(u['test'])}</li>" for u in d["unavailable"]) + "".join(
        f"<li><b>Part {p}</b> {E(v.get('reason', ''))} - capability UNAVAILABLE (not a PASS)</li>" for p, v in d["parts"].items() if v["disposition"] == "UNAVAILABLE")
    fx = "".join(f"<tr data-part='{f['part']}'><td>{E(f['uid'])}</td><td>P{f['part']}</td><td>{E(str(f['category']))}</td><td>{E(str(f['asset_id']))}</td><td>{E(f['severity'])}</td><td>{b(f['disposition'])}</td></tr>"
                 for f in d["findings"][:pol["max_findings_in_explorer"]])
    ver = d["audit"]["verify"]; rep = d["reproducibility"]
    inputs = "".join(f"<tr><td>P{i['part']}</td><td>{E(i['scope'])}</td><td class=ev>{E(i['path'])}</td><td><code>{i['sha256'][:16]}…</code></td></tr>" for i in rep["inputs"])
    orph = f"<p class=s>{len(d['governance']['orphan_decisions'])} decision(s) refer to cases no longer present.</p>" if d["governance"]["orphan_decisions"] else ""
    return f"""<!doctype html><html><head><meta charset=utf-8><title>Part 5 Assurance {E(d['assessment_id'])}</title><style>
body{{font-family:system-ui,Segoe UI,sans-serif;margin:0;background:#f5f6f8;color:#1c2430}}header{{background:#1c2b3a;color:#fff;padding:18px 28px}}main{{padding:20px 28px;max-width:1300px;margin:auto}}
.b{{color:#fff;padding:2px 9px;border-radius:10px;font-size:.8em;white-space:nowrap}}.tiles{{display:flex;gap:12px;flex-wrap:wrap}}.tile{{background:#fff;border-radius:8px;padding:12px 16px;min-width:210px;box-shadow:0 1px 3px #0002}}
.big{{font-size:1.3em;margin:6px 0}}.s{{color:#607080;font-size:.85em}}h2{{margin-top:30px;border-bottom:2px solid #dde;padding-bottom:4px}}table.t{{border-collapse:collapse;width:100%;background:#fff;font-size:.85em}}
.t td,.t th{{border:1px solid #dde;padding:4px 8px;text-align:left;vertical-align:top}}.t th{{background:#eef1f5}}.ev{{font-family:monospace;font-size:.78em;word-break:break-all}}details.case{{background:#fff;margin:8px 0;border-radius:8px;box-shadow:0 1px 3px #0002}}
summary{{padding:10px 14px;cursor:pointer}}.r{{float:right;font-size:.85em;color:#455}}.body{{padding:0 16px 14px}}.grid{{display:grid;grid-template-columns:1fr 1fr;gap:18px}}.sc{{max-height:260px;overflow:auto}}
.form{{margin-top:10px;padding:10px;background:#eef1f5;border-radius:6px}}.form>*{{margin-right:8px}}#stage{{background:#fff8e1;border:1px solid #f9a825;padding:8px 14px;border-radius:8px;margin:12px 0}}
button{{cursor:pointer}}code{{background:#eef1f5;padding:1px 4px;border-radius:3px}}@media(max-width:800px){{.grid{{grid-template-columns:1fr}}}}</style></head><body>
<header><h2 style="border:0;margin:0">Computer-Vision Assurance · Part 5 - Analyst Assurance &amp; Governance</h2><div class=s style="color:#b0c4d8">assessment {E(d['assessment_id'])} · {E(d['refreshed_at'])} · {E(d['detector_version'])} · offline · zero-training</div></header><main>
<h2 style="margin-top:0">Assurance status</h2><div class=tiles>{tiles}
<div class=tile><div class=s>Overall (system)</div><div class=big>{b(ov['system_disposition'])}</div><div class=s>governance-adjusted {b(ov['governance_adjusted_disposition'])}</div><div class=s>{ov['open_cases']} open case(s) ≥ REVIEW</div></div>
<div class=tile><div class=s>Coverage</div><div class=big>{b('ACCEPT' if ov['coverage'] == 'FULL' else 'REVIEW')} {ov['coverage']}</div><div class=s>{len(d['unavailable'])} test(s) unavailable</div></div>
<div class=tile><div class=s>Audit trail</div><div class=big>{b('ACCEPT' if ver['ok'] else 'QUARANTINE')} {'VERIFIED' if ver['ok'] else 'BROKEN'}</div><div class=s>{ver['events']} events · {E(ver['reason'])}</div></div></div>
<p class=s>Evidence view, not a verdict on intent. A shift is not an attack; a lone anomaly can be benign; no finding is not a guarantee.</p>
<h2>Priority findings</h2><table class=t><tr><th>case</th><th>what</th><th>suggested</th><th>review priority</th><th>governance status</th></tr>{prio}</table>{orph}
<div id=stage><b>Staged analyst decisions:</b> <span id=n>0</span> <button onclick="dl()">Download analyst_actions.json</button> <button onclick="clr()">Clear</button>
<div class=s>Apply with: <code>python -m cv_assurance part5 apply --out &lt;this dir&gt; --actions analyst_actions.json</code></div><ul id=sl></ul></div>
<h2>Cases (correlated evidence)</h2>{cases or '<p>No cases.</p>'}
<h2>Source / contributor concentration (Part 1)</h2><div class=sc><table class=t><tr><th>source</th><th>samples</th><th>dup</th><th>label</th><th>OOD</th><th>trigger</th><th>elevated</th><th>disp.</th></tr>{src}</table></div>
<h2>Coverage - unavailable tests</h2><ul>{unav or '<li>none - every supported test executed</li>'}</ul>
<h2>Finding explorer</h2><p class=s>filter: <select id=fp onchange=flt()><option value="">all parts</option><option>1</option><option>2</option><option>3</option><option>4</option></select>
<input id=ft placeholder="text" oninput=flt()> ({len(d['findings'])} total; first {pol['max_findings_in_explorer']} shown)</p><div class=sc style="max-height:340px"><table class=t id=ftab><tr><th>uid</th><th>part</th><th>category</th><th>asset</th><th>sev</th><th>disp.</th></tr>{fx}</table></div>
<h2>Reproducibility</h2><p class=s>policy sha256 <code>{d['policy_sha256'][:24]}…</code> · python {E(rep['python'])} · numpy {E(rep['numpy'])} · network required: {rep['network_access_required']} · training performed: {rep['training_performed']}</p>
<table class=t><tr><th>part</th><th>scope</th><th>artifact</th><th>sha256</th></tr>{inputs}</table><p class=s>{E(rep['rerun'])}</p>
<h2>Limitations &amp; honest claims</h2><ul>{''.join('<li>' + E(x) + '</li>' for x in d['limitations'])}</ul><ul>{''.join('<li>' + E(x) + '</li>' for x in d['honest_claims'])}</ul></main>
<script>var S=[];function stage(bt){{var f=bt.parentNode,a={{case_id:f.dataset.case,action:f.querySelector('.act').value,actor:f.querySelector('.who').value||'analyst',rationale:f.querySelector('.why').value,override:f.querySelector('.ovr').checked}};S.push(a);show()}}
function show(){{document.getElementById('n').textContent=S.length;document.getElementById('sl').innerHTML=S.map(function(a){{return '<li>'+a.case_id+' → '+a.action+' ('+a.actor+') '+a.rationale+'</li>'}}).join('')}}
function clr(){{S=[];show()}}function dl(){{var u=URL.createObjectURL(new Blob([JSON.stringify({{actions:S}},null,1)],{{type:'application/json'}}));var a=document.createElement('a');a.href=u;a.download='analyst_actions.json';a.click()}}
function openCase(id){{var e=document.getElementById(id);if(e)e.open=true}}
function flt(){{var p=document.getElementById('fp').value,t=document.getElementById('ft').value.toLowerCase();document.querySelectorAll('#ftab tr[data-part]').forEach(function(r){{r.style.display=((!p||r.dataset.part==p)&&(!t||r.textContent.toLowerCase().indexOf(t)>=0))?'':'none'}})}}
</script></body></html>"""
