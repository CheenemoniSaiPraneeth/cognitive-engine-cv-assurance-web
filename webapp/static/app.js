"use strict";
const $ = (s, r = document) => r.querySelector(s);
const esc = v => String(v ?? "").replace(/[&<>"']/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const badge = (v, extra = "") => `<span class="badge b-${esc(String(v).split(" ")[0])}">${esc(v)}${extra}</span>`;
const NAMES = { p1: "Data integrity", p2: "Model assurance", p3: "Inference provenance", p4: "Distribution shift", p5: "Governance" };
const ICON = { complete: "✓", running: "●", pending: "○", failed: "✕", skipped: "–" };
const LABEL = { complete: "Complete", running: "Running", pending: "Pending", failed: "Failed", skipped: "Not run" };
let pollTimer = null, uploadId = null;

async function api(path, opts) {
  const r = await fetch(path, opts);
  let body = null; try { body = await r.json(); } catch (_) {}
  if (!r.ok) throw new Error((body && (body.detail || body.error)) || `Request failed (${r.status})`);
  return body;
}
function show(view) {
  for (const id of ["viewHome", "viewRun", "viewResult"]) $("#" + id).hidden = id !== view;
  $("#homeBtn").hidden = view === "viewHome"; $("#drawer").hidden = true; window.scrollTo(0, 0);
}
function setSide(stages) {
  document.querySelectorAll("#sideSteps li").forEach(li => { li.className = stages ? (stages[li.dataset.p] || "") : ""; });
}

/* ---------- landing ---------- */
async function loadDemoInfo() {
  try {
    const d = await api("/api/demo/info");
    if (!d.available) { $("#demoFacts").innerHTML = "<li>Bundled demo dataset not found on this server.</li>"; $("#demoBtn").disabled = true; return; }
    $("#demoFacts").innerHTML = [
      `${d.images} images · ${d.annotations} annotations (COCO)`,
      `Served model: <span class="mono">${esc(d.served_model)}</span> vs trusted reference <span class="mono">${esc(d.trusted_model)}</span>`,
      d.streams.length ? `Operational streams: ${d.streams.map(esc).join(", ")}` : "",
      "Runs the real P1 → P5 pipeline and shows whatever it finds",
    ].filter(Boolean).map(t => `<li>${t}</li>`).join("");
  } catch (e) { $("#demoFacts").innerHTML = `<li>${esc(e.message)}</li>`; }
}
$("#demoBtn").onclick = async () => {
  $("#demoErr").hidden = true; $("#demoBtn").disabled = true;
  try { const r = await api("/api/demo/run", { method: "POST" }); track(r.run_id, "Running default assurance dataset…"); }
  catch (e) { $("#demoErr").textContent = e.message; $("#demoErr").hidden = false; }
  finally { $("#demoBtn").disabled = false; }
};

const zipInput = $("#zipInput"), drop = $("#drop");
drop.addEventListener("click", () => zipInput.click());
["dragover", "dragenter"].forEach(ev => drop.addEventListener(ev, e => { e.preventDefault(); drop.classList.add("over"); }));
["dragleave", "drop"].forEach(ev => drop.addEventListener(ev, e => { e.preventDefault(); drop.classList.remove("over"); }));
drop.addEventListener("drop", e => { if (e.dataTransfer.files[0]) uploadZip(e.dataTransfer.files[0]); });
zipInput.onchange = () => zipInput.files[0] && uploadZip(zipInput.files[0]);

async function uploadZip(file) {
  uploadId = null; $("#runCustomBtn").hidden = true; $("#uploadErr").hidden = true; $("#valBox").hidden = true;
  $("#dropText").textContent = `Validating ${file.name}…`;
  const fd = new FormData(); fd.append("file", file);
  try {
    const r = await api("/api/upload", { method: "POST", body: fd });
    renderValidation(r.validation); uploadId = r.upload_id;
    $("#runCustomBtn").hidden = !(r.validation.valid && uploadId);
    $("#dropText").innerHTML = `<b>${esc(file.name)}</b> — choose another`;
  } catch (e) { $("#uploadErr").textContent = e.message; $("#uploadErr").hidden = false; $("#dropText").innerHTML = "Choose or drop <b>cv_assurance_input.zip</b>"; }
}
function renderValidation(v) {
  const d = v.detected || {};
  const chk = (ok, t) => `<li>${ok ? "✓" : "✕"} ${esc(t)}</li>`;
  const found = `<ul style="list-style:none;padding:0;margin:0 0 10px">${chk(d["images/"], "images/")}${chk(d["annotations/"], "annotations/")}${chk(d["annotations.json"], "annotations.json")}</ul>`;
  const box = $("#valBox"); box.hidden = false;
  box.innerHTML = v.valid
    ? `<div class="val ok"><h4>Package validated</h4>${found}<dl><dt>Images</dt><dd>${v.images}</dd><dt>Annotations</dt><dd>${v.annotations}</dd><dt>Format</dt><dd>${esc(v.annotation_format)}</dd>
       <dt>Model</dt><dd>${v.model_detected ? "Detected" : "Not supplied"}</dd>${v.streams.length ? `<dt>Streams</dt><dd>${v.streams.map(esc).join(", ")}</dd>` : ""}</dl>
       ${v.warnings.length ? `<ul>${v.warnings.map(w => `<li class="w">${esc(w)}</li>`).join("")}</ul>` : ""}</div>`
    : `<div class="val bad"><h4>Package rejected</h4>${found}<ul>${v.errors.map(e => `<li class="e">${esc(e)}</li>`).join("")}</ul></div>`;
}
$("#runCustomBtn").onclick = async () => {
  $("#uploadErr").hidden = true; $("#runCustomBtn").disabled = true;
  try {
    const r = await api("/api/run", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ upload_id: uploadId }) });
    uploadId = null; track(r.run_id, "Running your dataset…");
  } catch (e) { $("#uploadErr").textContent = e.message; $("#uploadErr").hidden = false; }
  finally { $("#runCustomBtn").disabled = false; }
};
$("#homeBtn").onclick = () => { clearTimeout(pollTimer); setSide(null); show("viewHome"); };

/* ---------- run tracking ---------- */
function track(runId, title) {
  show("viewRun"); $("#runTitle").textContent = title; $("#runEyebrow").textContent = "RUNNING"; $("#runErr").hidden = true; renderStages({ p1: "pending", p2: "pending", p3: "pending", p4: "pending", p5: "pending" });
  const tick = async () => {
    try {
      const s = await api(`/api/status/${runId}`);
      renderStages(s.stages); setSide(s.stages);
      $("#runMeta").textContent = `Elapsed ${s.elapsed_s}s` + (s.log_tail.length ? ` · ${s.log_tail[s.log_tail.length - 1]}` : "");
      if (s.state === "done") return showResult(runId);
      if (s.state === "failed") { $("#runEyebrow").textContent = "FAILED"; $("#runErr").textContent = s.error || "The pipeline failed."; $("#runErr").hidden = false; return; }
    } catch (e) { $("#runErr").textContent = e.message; $("#runErr").hidden = false; return; }
    pollTimer = setTimeout(tick, 1000);
  };
  tick();
}
function renderStages(st) {
  $("#stageList").innerHTML = Object.keys(NAMES).map(k => `<li class="${st[k]}"><span class="ic">${ICON[st[k]]}</span><b class="mono">${k.toUpperCase()}</b> ${NAMES[k]}<span class="st">${LABEL[st[k]]}</span></li>`).join("");
}

/* ---------- result ---------- */
let RESULT = null;
async function showResult(runId) {
  RESULT = await api(`/api/result/${runId}`); const r = RESULT;
  const a = r.audit, sys = r.system_status, gov = r.governance_status;
  const stages = {}; Object.keys(NAMES).forEach(k => stages[k] = r.parts[k] && r.parts[k].disposition !== "UNAVAILABLE" ? "complete" : "skipped"); setSide(stages);
  const notes = [];
  if (r.input.source === "bundled demo") notes.push(`Default demo · ${r.input.images} images · served model ${esc(r.input.served_model)}`);
  if (!r.parts.p2 || r.parts.p2.disposition === "UNAVAILABLE") notes.push("No model supplied — Parts 2–4 were not executed; this is not a pass for those capabilities.");
  $("#viewResult").innerHTML = `
    <header class="hero slim"><p class="eyebrow">ASSURANCE RESULT · <span class="mono">${esc(r.assessment_id)}</span></p><h2>System assessment</h2></header>
    ${notes.map(n => `<div class="note">${n}</div>`).join("")}
    <div class="summary">
      <div class="tile hero-tile t-${esc(sys)}"><div class="k">System status</div><div class="v tv-${esc(sys)}">${esc(sys)}</div></div>
      <div class="tile"><div class="k">Governance-adjusted</div><div class="v tv-${esc(gov)}">${esc(gov)}</div></div>
      <div class="tile"><div class="k">Cases</div><div class="v">${r.cases.length}</div><div class="muted small">${r.open_cases} open</div></div>
      <div class="tile"><div class="k">Coverage</div><div class="v">${esc(r.coverage)}</div>${r.coverage_gaps.length ? `<div class="muted small">gaps: P${r.coverage_gaps.map(esc).join(", P")}</div>` : ""}</div>
      <div class="tile"><div class="k">Audit</div><div class="v">${a.status === "OK" ? "VERIFIED" : "BROKEN"}</div><div class="muted small">${a.events} events</div></div>
    </div>
    <h3 class="sec">Parts</h3>${Object.keys(NAMES).filter(k => r.parts[k]).map(k => partAcc(k, r.parts[k])).join("")}
    <h3 class="sec">Cases</h3>${casesTable(r.cases)}
    ${r.cross_signals.length ? `<h3 class="sec">Cross-part signals</h3>${r.cross_signals.map(s => `<details class="acc"><summary>${badge(s.disposition)}<span class="pt">${esc(s.signal)}</span><span class="muted small" style="margin-left:auto">${esc(s.stream || "")}</span></summary><div class="body"><p>${esc(s.detail)}</p></div></details>`).join("")}` : ""}
    <h3 class="sec">Audit integrity</h3>
    <div class="audit"><ul><li class="${a.chain_intact ? "" : "bad"}">${a.chain_intact ? "Chain intact" : "Chain not verified"}</li><li class="${a.events ? "" : "bad"}">${a.events} events</li>
      <li class="${a.hmac_valid === a.events ? "" : "bad"}">${a.hmac_valid ?? "?"} HMAC signatures valid</li></ul>
      <div>Evidence integrity: ${badge(a.status === "OK" ? "VERIFIED" : "BROKEN")}</div>
      <p class="muted small">Audit integrity is independent of the system status: a QUARANTINE system can have a fully verified evidence trail.</p></div>
    ${r.unavailable.length ? `<h3 class="sec">Tests not executed</h3><details class="logs"><summary>${r.unavailable.length} capabilities unavailable (not a pass)</summary><pre>${esc(r.unavailable.map(u => `P${u.part}: ${u.test}\n   → ${u.impact}`).join("\n"))}</pre></details>` : ""}
    <h3 class="sec">Technical</h3><details class="logs"><summary>Technical logs ▾</summary><pre>${esc(r.logs.join("\n"))}</pre></details>`;
  $("#viewResult").querySelectorAll("tbody tr").forEach(tr => tr.onclick = () => openCase(tr.dataset.id));
  show("viewResult");
}
function partAcc(k, p) {
  const counts = Object.entries(p.finding_counts || {});
  const lvl = p.level ? ` <span class="muted small">(${esc(p.level)})</span>` : "";
  return `<details class="acc"><summary><span class="pn">PART ${p.part}</span><span class="pt">${NAMES[k]}${lvl}</span>${badge(p.disposition === "UNAVAILABLE" ? "UNAVAILABLE" : p.disposition)}</summary><div class="body">
    ${p.system_disposition ? `<p class="muted">System disposition ${esc(p.system_disposition)}; governance-adjusted ${esc(p.disposition)} after analyst decisions.</p>` : ""}
    ${p.scopes.length ? `<p class="muted small">Scope: ${p.scopes.map(esc).join(", ")}</p>` : ""}
    <div class="kv">${counts.length ? counts.map(([n, c]) => `<span>${esc(n.replace(/_/g, " "))}<b>${esc(c)}</b></span>`).join("") : "<span>no findings recorded</span>"}</div></div></details>`;
}
function casesTable(cs) {
  if (!cs.length) return `<div class="card muted">No cases were raised.</div>`;
  return `<div class="tbl-wrap"><table><thead><tr><th>Case</th><th>Severity</th><th>Score</th><th>Parts</th><th>Finding</th><th>Status</th></tr></thead><tbody>${cs.map(c =>
    `<tr data-id="${esc(c.case_id)}"><td class="mono">${esc(c.case_id)}</td><td>${badge(c.band)}</td><td class="mono">${esc(c.score)}</td><td class="mono">${c.parts.map(p => "P" + p).join(", ")}</td><td>${esc(c.title)}</td><td>${badge(c.status)}</td></tr>`).join("")}</tbody></table></div>`;
}
function openCase(id) {
  const c = RESULT.cases.find(x => x.case_id === id), d = $("#drawer");
  d.innerHTML = `<button class="x" aria-label="Close">×</button><p class="eyebrow mono">${esc(c.case_id)}</p><h3 style="margin:0 0 10px">${esc(c.title)}</h3>
    <div>${badge(c.disposition)} ${badge(c.band)} ${badge(c.status)}</div>
    <h4>Rationale</h4><p>${esc(c.rationale)}</p><h4>Suggested next action</h4><p>${esc(c.next_action)}</p>
    <h4>Entities</h4><div class="kv">${c.entities.map(e => `<span class="mono">${esc(e)}</span>`).join("")}</div>
    <h4>Score ${esc(c.score)} · ${c.finding_count} findings · evidence groups</h4>
    <div class="kv">${c.contributions.map(x => `<span>P${x.part} ${esc(x.category.replace(/_/g, " "))}<b>${esc(x.group_score)}</b></span>`).join("")}</div>
    <h4>Sample evidence (${c.sample_findings.length} of ${c.finding_count})</h4>
    ${c.sample_findings.map(f => `<div><b class="mono">${esc(f.uid)}</b> ${badge(f.severity)}<pre>${esc(JSON.stringify(f.evidence, null, 1))}</pre></div>`).join("")}
    <h4>Evidence hash</h4><pre>${esc(c.evidence_hash)}</pre>`;
  d.hidden = false; d.querySelector(".x").onclick = () => d.hidden = true;
}
document.addEventListener("keydown", e => { if (e.key === "Escape") $("#drawer").hidden = true; });
loadDemoInfo();
