"""Part 5 tests - run against bundled sample Part 1-4 outputs (no other part needed):  python -m pytest tests/test_part5.py -q"""
import json
import shutil
from pathlib import Path

import pytest

from cv_assurance.part5_governance.aggregate import discover, normalize
from cv_assurance.part5_governance.assemble import act, apply_actions, assemble, refresh, verify_export
from cv_assurance.part5_governance.audit import GovAudit
from cv_assurance.part5_governance.governance import GovernanceError

SAMPLES = Path(__file__).resolve().parents[1] / "sample_inputs"


@pytest.fixture()
def comp(tmp_path):
    return assemble([SAMPLES / "compromised_model"], tmp_path / "o", log=lambda *a: None), tmp_path / "o"


def rules(fin):
    return {c["rule"]: c for c in fin["cases"]}


def test_namespaced_ids_and_unavailable_are_explicit():
    n = normalize(discover([SAMPLES / "compromised_model"]))
    uids = [f["uid"] for f in n["findings"]]
    assert len(uids) == len(set(uids))                       # P-00001 from two ledgers never collides
    assert any(f["disposition"] == "UNAVAILABLE" and f["category"] == "test_unavailable" for f in n["findings"])


def test_converging_cases_fire_on_compromised_scenario(comp):
    fin, _ = comp
    r = rules(fin)
    for k in ("data_model_trigger", "model_inference", "shift_provenance", "source_convergence"):
        assert r[k]["suggested_disposition"] == "QUARANTINE", k
    assert set(r["data_model_trigger"]["parts"]) == {1, 2}
    assert fin["overall"]["system_disposition"] == "QUARANTINE"


def test_trusted_model_scenario_does_not_escalate_shift(tmp_path):
    fin = assemble([SAMPLES / "trusted_model"], tmp_path / "o", log=lambda *a: None)
    r = rules(fin)
    assert "model_inference" not in r and "shift_provenance" not in r
    assert r["operational_shift"]["suggested_disposition"] == "REVIEW"          # a shift is not an attack


def test_isolated_signal_is_capped_and_never_high(comp):
    fin, _ = comp
    iso = rules(fin)["isolated_signal"]
    assert iso["suggested_disposition"] == "REVIEW" and iso["priority"]["band"] != "HIGH"


def test_deterministic_case_ids_and_evidence_hash(tmp_path):
    a = assemble([SAMPLES / "compromised_model"], tmp_path / "a", log=lambda *a: None)
    b = assemble([SAMPLES / "compromised_model"], tmp_path / "b", log=lambda *a: None)
    assert [(c["case_id"], c["evidence_hash"]) for c in a["cases"]] == [(c["case_id"], c["evidence_hash"]) for c in b["cases"]]
    assert a["assessment_id"] == b["assessment_id"]


def test_governance_rules(comp):
    fin, out = comp
    r = rules(fin)
    with pytest.raises(GovernanceError):
        act(out, r["isolated_signal"]["case_id"], "close", "ana", "")                        # rationale required
    with pytest.raises(GovernanceError):
        act(out, r["model_inference"]["case_id"], "accept", "ana", "a long enough rationale here")   # override required on QUARANTINE
    f = act(out, r["isolated_signal"]["case_id"], "close", "ana", "benign export artefacts, checked samples")
    assert f["governance"]["case_state"][r["isolated_signal"]["case_id"]]["status"] == "CLOSED"
    f = act(out, r["source_convergence"]["case_id"], "quarantine", "ana")
    assert f["overall"]["governance_adjusted_disposition"] == "QUARANTINE"
    assert verify_export(out)["ok"]


def test_decision_goes_stale_when_evidence_changes(comp, tmp_path):
    fin, out = comp
    cid = rules(fin)["isolated_signal"]["case_id"]
    act(out, cid, "close", "ana", "reviewed and benign, see ticket 7")
    an = json.loads((out / "part5_analysis.json").read_text())
    for c in an["cases"]:
        if c["case_id"] == cid:
            c["evidence_hash"] = "changed"
    (out / "part5_analysis.json").write_text(json.dumps(an))
    f = refresh(out, log=lambda *a: None)
    st = f["governance"]["case_state"][cid]
    assert st["stale"] and st["status"].startswith("OPEN")


def test_apply_actions_file(comp, tmp_path):
    fin, out = comp
    cid = rules(fin)["source_convergence"]["case_id"]
    af = tmp_path / "acts.json"
    af.write_text(json.dumps({"actions": [{"case_id": cid, "action": "retest", "actor": "ana"}, {"case_id": cid, "action": "close", "actor": "ana", "rationale": ""}]}))
    res = apply_actions(out, af)
    assert res[0][2] == "recorded" and res[1][2].startswith("REJECTED")


def test_audit_tamper_and_input_change_detected(comp):
    fin, out = comp
    assert verify_export(out)["ok"]
    p = out / "audit_part5.jsonl"; L = p.read_text().splitlines()
    L[2] = L[2].replace("ingest_artifact", "ingest_artefact"); p.write_text("\n".join(L) + "\n")
    v = verify_export(out)
    assert not v["ok"] and not v["checks"]["audit_chain_and_signatures"]


def test_missing_part_is_unavailable_not_pass(tmp_path):
    d = tmp_path / "in"; d.mkdir()
    shutil.copy(SAMPLES / "trusted_model/p123/part12/part2/assurance_part2.json", d / "assurance_part2.json")
    fin = assemble([d], tmp_path / "o", log=lambda *a: None)
    assert fin["parts"]["1"]["disposition"] == "UNAVAILABLE" and fin["overall"]["coverage"] == "PARTIAL"
    assert "missing_evidence" in rules(fin)


def test_html_report_self_contained(comp):
    _, out = comp
    h = (out / "report_part5.html").read_text()
    assert "http://" not in h.replace("http://www.w3.org", "") and "https://" not in h and "Stage decision" in h
