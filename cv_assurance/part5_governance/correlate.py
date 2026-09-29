"""Module 2 - Cross-Signal Correlation Engine.

Rule-based (no learning). A single anomaly can be benign; independent Parts agreeing on the same entity is what raises
review priority. Every case states the rule that fired, the entities, the member findings and what to inspect next.
Nothing here claims intent - only converging evidence.
"""
from __future__ import annotations

from ..common.hashing import sha256_json
from .config import RANK
from .risk import SEV_ORDER, case_priority, max_disp, max_sev

MODEL_DEVIATION = {"identity_mismatch", "model_substitution_assessment", "parameter_localized_change", "parameter_difference",
                   "dormant_high_influence_unit", "trigger_candidate_response", "behaviour_divergence", "behavioural_divergence"}
INFORMATIONAL = {"duplicate_pair_informational", "identity_unverifiable", "numerical_drift_within_tolerance", "test_unavailable"}


def _sig(f):  # significant, non-informational evidence
    return f["disposition"] not in ("ACCEPT", "UNAVAILABLE") and f["category"] not in INFORMATIONAL


def _cap(disp: str, pol: dict) -> str:
    m = pol["isolated_signal_max_disposition"]
    return m if RANK[disp] > RANK[m] else disp


class Engine:
    def __init__(self, norm: dict, sources: list, arts: list, pol: dict):
        self.F, self.pol, self.sources, self.arts = norm["findings"], pol, sources, arts
        self.byuid = {f["uid"]: f for f in self.F}
        self.cases, self.covered = [], set()
        self.p4 = {a.scope: a.data for a in arts if a.part == 4}
        self.p3 = {a.scope: a.data for a in arts if a.part == 3}
        self.p2 = [a.data for a in arts if a.part == 2]

    # ---- helpers
    def sel(self, part=None, cats=None, scope=None, sig=True):
        return [f for f in self.F if (part is None or f["part"] == part) and (cats is None or f["category"] in cats)
                and (scope is None or f["scope"] == scope) and (_sig(f) if sig else True)]

    def add(self, rule, title, entities, members, disposition, rationale, next_action, severity=None, tags=()):
        members = list({m["uid"]: m for m in members}.values())
        if not members and rule not in ("missing_evidence",):
            return None
        cid = "C-" + sha256_json([rule, sorted(entities)])[:8]
        pr = case_priority(members, self.pol)
        if "isolated" in tags and pr["score"] > self.pol["isolated_priority_cap"]:
            b = self.pol["priority_bands"]; pr["score"] = self.pol["isolated_priority_cap"]
            pr["band"] = "HIGH" if pr["score"] >= b["HIGH"] else "ELEVATED" if pr["score"] >= b["ELEVATED"] else "MODERATE"
            pr["capped"] = "isolated (uncorroborated) signal: priority capped by policy"
        ehash = sha256_json(sorted([[m["uid"], m["category"], m["severity"], m["disposition"], m["evidence"]] for m in members]))
        case = {"case_id": cid, "rule": rule, "title": title, "entities": sorted(entities), "parts": sorted({m["part"] for m in members}),
                "severity": severity or max_sev(members), "suggested_disposition": disposition, "priority": pr, "rationale": rationale,
                "next_action": next_action, "tags": list(tags), "finding_count": len(members), "finding_uids": sorted(m["uid"] for m in members),
                "evidence_hash": ehash}
        self.cases.append(case)
        self.covered |= {m["uid"] for m in members}
        return case

    # ---- rules
    def r_source(self):
        for s in self.sources:
            if s["kind"] == "batch_id" or len(s["elevated_signals"]) < 2:
                continue
            n = len(s["elevated_signals"]); ent = [f"source:{s['name']}", f"dataset:{s['dataset']}"]
            mem = [f for f in self.sel(1, sig=False) if f"source:{s['name']}" in f["entities"] and f["disposition"] != "ACCEPT"]
            disp = s.get("disposition") or ("QUARANTINE" if n >= 3 else "REVIEW")
            sig = ", ".join(f"{k} rate" for k in s["elevated_signals"])
            self.add("source_convergence", f"Source '{s['name']}' shows {n} converging data-integrity signals", ent, mem, disp,
                     f"{n} independent Part-1 detectors elevate the same source ({sig}; {s['sample_count']} samples, cross-signal count {s['cross_signal_count']}). "
                     "Evidence concentrates in one contributor rather than being spread across the dataset.",
                     f"Inspect duplicate/trigger groups and label-anomaly samples attributed to '{s['name']}'; check contributor provenance before allowing this data into training.",
                     severity="critical" if n >= 3 else "high", tags=("data", "converging"))

    def r_data_model(self):
        p1 = self.sel(1, {"trigger_candidate"})
        p2 = [f for f in self.sel(2) if f["category"] in MODEL_DEVIATION and f["category"] not in ("identity_mismatch",)]
        if not (p1 and p2):
            return
        confirmed = [r for a in self.p2 for r in ((a.get("trigger_tests") or {}).get("part1_candidate_results") or []) if r.get("confirmed_functional_response")]
        resp = [f for f in p2 if f["category"] == "trigger_candidate_response"]
        disp = "QUARANTINE" if (confirmed or resp) else "REVIEW"
        ent = sorted({e for f in p1 + p2 for e in f["entities"] if e.split(":")[0] in ("dataset", "model", "pattern")})
        det = f"{len(confirmed)} Part-1 trigger candidate(s) produced a functional response in the supplied model (attack-success rate up to " \
              f"{max([r.get('attack_success_rate', 0) for r in confirmed], default=0):.2f})" if confirmed else "model-side deviation coincides with data-side trigger candidates"
        self.add("data_model_trigger", "Trigger-like pattern in data AND model-side response", ent, p1 + p2, disp,
                 f"Part 1 found recurring localized patterns correlated with a label; Part 2 independently found: {det}. Two independent parts agree. "
                 "This is a potential trigger/backdoor candidate - not proof of intent or of a functional backdoor beyond the tests executed.",
                 "Quarantine the dataset subset (see pattern support) and the served model; re-run Part 2 against the trusted reference; inspect the top-influence units / patch location.",
                 severity="critical" if disp == "QUARANTINE" else "high", tags=("data", "model", "converging"))

    def r_model_inference(self):
        p2 = [f for f in self.sel(2) if f["category"] in MODEL_DEVIATION]
        p3 = [f for f in self.sel(3)]
        if not (p2 and p3):
            return
        ent = sorted({e for f in p2 for e in f["entities"] if e.startswith("model:")} | {f"ledger:{s}" for s in sorted({f['scope'] for f in p3})})
        cats = sorted({f["category"] for f in p3})
        self.add("model_inference", "Model deviates from trusted reference AND stored inference records fail verification", ent, p2 + p3, "QUARANTINE",
                 f"Part 2 reports model-side deviation ({', '.join(sorted({f['category'] for f in p2}))}); Part 3 independently reports {', '.join(cats)} across "
                 f"{len({f['scope'] for f in p3})} ledger(s). Recorded outputs are not reproducible by the trusted model, consistent with the served model differing from the trusted one.",
                 "Compare the served model hash with the registry digest; re-run inference with the trusted artifact; hold results produced since the first mismatching record.",
                 severity="critical", tags=("model", "inference", "converging"))

    def r_shift(self):
        for scope, d in sorted(self.p4.items()):
            lvl = d.get("level", "NORMAL")
            if lvl == "NORMAL":
                continue
            p4f = [f for f in self.F if f["part"] == 4 and f["scope"] == scope and _sig(f)]
            p3s = self.p3.get(scope); p3f = [f for f in self.F if f["part"] == 3 and f["scope"] == scope and _sig(f)]
            at, tl = d.get("overall_attribution") or {}, d.get("temporal") or {}
            fams = at.get("primary_families") or []; onset, pat = tl.get("onset_batch"), tl.get("pattern")
            ent = [f"stream:{scope}", f"window:{scope}"]
            if p3s is not None and (p3s.get("disposition") != "ACCEPT" or p3f):
                self.add("shift_provenance", f"Stream '{scope}': distribution shift AND provenance failure", ent + [f"ledger:{scope}"], p4f + p3f, "QUARANTINE",
                         f"Part 4 rates the window {lvl} (pattern {pat}, onset {onset}, families {', '.join(fams) or 'n/a'}) while Part 3 rates the same stream's ledger "
                         f"{p3s.get('disposition')}. Shift and provenance evidence point at the same window, so the shift cannot be assumed to be a benign environment change.",
                         "Resolve the ledger failure first (Part 3 findings), then re-baseline or investigate the acquisition change.", severity="critical", tags=("inference", "shift", "converging"))
            else:
                prov = "ledger verified intact" if p3s is not None else "NO Part-3 ledger supplied for this stream (provenance UNAVAILABLE)"
                out_only = "task_output" in fams and not (set(fams) & {"illumination", "color", "geometry", "sharpness_noise", "compression", "semantic"})
                model_bad = any(a.get("disposition") != "ACCEPT" for a in self.p2)
                if out_only:
                    self.add("output_shift_only", f"Stream '{scope}': outputs moved but inputs did not", ent, p4f, "QUARANTINE" if model_bad else "REVIEW",
                             "Output distribution changed without a matching input change - points at changed model behaviour rather than a changed environment.",
                             "Re-run Part 2 behavioural fingerprint on the served model; compare with the trusted reference.", severity="high", tags=("model", "shift"))
                else:
                    self.add("operational_shift", f"Stream '{scope}': {lvl.lower()} in {', '.join(fams) or 'multiple feature families'}", ent, p4f, "REVIEW",
                             f"Operational population differs from the trusted reference: pattern {pat}, onset {onset}, primary families {', '.join(fams) or 'n/a'}; {prov}. "
                             "A shift is not an attack - a new camera, lighting or scene can cause it.",
                             "Decide: accept and re-baseline the reference (Retest) or investigate acquisition changes at the onset batch.",
                             severity="high" if lvl.startswith("SEVERE") else "medium", tags=("shift",))

    def r_source_shift(self):
        bad = {s["name"] for s in self.sources if s["kind"] != "batch_id" and s["elevated_signals"]}
        for scope, d in sorted(self.p4.items()):
            hit = [x for x in ((d.get("overall_attribution") or {}).get("sources") or []) if x.get("source") in bad and (x.get("abnormal_rate") or 0) >= 0.3]
            if hit:
                mem = [f for f in self.F if f["part"] == 4 and f["scope"] == scope and _sig(f)] + [f for f in self.sel(1, sig=False) if any(f"source:{h['source']}" in f["entities"] for h in hit)]
                self.add("source_in_data_and_shift", f"Source flagged in training data is also abnormal in stream '{scope}'", [f"stream:{scope}"] + [f"source:{h['source']}" for h in hit], mem, "QUARANTINE",
                         "The same source is elevated by Part 1 and produces abnormal operational samples in Part 4.", "Trace the source across acquisition metadata; isolate it in both pipelines.", severity="high", tags=("data", "shift", "converging"))

    def r_provenance_alone(self):
        for scope in sorted(self.p3):
            mem = [f for f in self.sel(3, scope=scope) if f["uid"] not in self.covered]
            if not mem:
                continue
            disp = max_disp([m["disposition"] for m in mem], RANK); cats = sorted({m["category"] for m in mem})
            self.add("inference_integrity", f"Ledger '{scope}': {', '.join(cats)}", [f"ledger:{scope}"], mem, disp,
                     f"{len(mem)} finding(s) from Part 3's recompute/chain/binding checks failed for this ledger. Hard integrity failures are surfaced at the source detector's disposition.",
                     "Inspect the first failing record, compare stored vs recomputed output, and verify the model digest against the registry.", tags=("inference",))

    def r_residual(self):
        minsev = SEV_ORDER[self.pol["min_severity_for_residual_case"]]
        rest = {}
        for f in self.F:
            if f["uid"] in self.covered or not _sig(f) or SEV_ORDER.get(f["severity"], 0) < minsev:
                continue
            rest.setdefault((f["part"], f["scope"]), []).append(f)
        for (part, scope), mem in sorted(rest.items()):
            disp = _cap(max_disp([m["disposition"] for m in mem], RANK), self.pol); cats = sorted({m["category"] for m in mem})
            self.add("isolated_signal", f"Part {part} '{scope}': isolated signal(s) - {', '.join(cats)}", [f"part{part}:{scope}"] + sorted({e for m in mem for e in m["entities"]})[:6], mem, disp,
                     "Single-Part evidence with no corroboration from another Part. A lone anomaly can be benign (rare legitimate sample, export artefact); shown for review, not escalated.",
                     "Inspect the listed samples/assets; look for corroborating signal in another Part before escalating.", tags=("isolated",))

    def r_missing(self, parts_present):
        for p, name in ((1, "training-data integrity"), (2, "model integrity"), (3, "inference provenance"), (4, "distribution shift")):
            if p not in parts_present:
                self.add("missing_evidence", f"No Part {p} ({name}) evidence supplied", [f"part:{p}"], [], "UNAVAILABLE",
                         f"No Part {p} assurance artifact was provided. This capability is UNAVAILABLE - it is not a PASS.", f"Run Part {p} and re-assemble.", severity="low", tags=("coverage",))

    def run(self, parts_present):
        for r in (self.r_source, self.r_data_model, self.r_model_inference, self.r_shift, self.r_source_shift, self.r_provenance_alone, self.r_residual):
            r()
        self.r_missing(parts_present)
        self.cases.sort(key=lambda c: (-RANK[c["suggested_disposition"]], -c["priority"]["score"], c["case_id"]))
        return self.cases
