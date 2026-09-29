"""Part 3 tests. Run: python -m unittest discover -s tests -v"""
import json, os, shutil, sys, tempfile, unittest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from cv_assurance.benchmark.part3_bench import build_part3_benchmark, evaluate_part3
from cv_assurance.common.audit import AuditLog
from cv_assurance.part3_inference.canonical import compare_detections, compare_outputs, build_classification_output
from cv_assurance.part3_inference.ledger import load_key, read_records
from cv_assurance.part3_inference.verifier import verify_ledger

TMP = tempfile.mkdtemp(); B = os.path.join(TMP, "b"); GT = build_part3_benchmark(B)
KEY = load_key(GT["key"]); M = evaluate_part3(B, os.path.join(TMP, "o"))
ROW = {r["case"]: r for r in M["cases"]}


class Part3(unittest.TestCase):
    def test_valid_ledger_accepts_with_zero_false_positives(self):
        self.assertEqual(M["valid_case_disposition"], "ACCEPT"); self.assertEqual(M["clean_record_false_positive_rate"], 0)

    def test_every_attack_case_detected(self):
        for n, r in ROW.items():
            if n != "valid": self.assertTrue(r["case_detected"], n)
            if n != "valid": self.assertEqual(r["false_positives"], 0, n)

    def test_insider_attacks_caught_by_content_checks_without_key(self):
        for n, r in ROW.items():
            if r["attacker"] == "insider" and r["detected_by_content_checks_only"] is not None:
                self.assertEqual(r["detected_by_content_checks_only"], r["attacked"], n)

    def test_naive_edit_breaks_chain(self):
        res = verify_ledger(os.path.join(B, "cases", "output_tampering_naive"), os.path.join(TMP, "n"), key=KEY, log=lambda *a: None)
        self.assertIn("record_hash_mismatch", res["finding_counts"])

    def test_config_tamper_needs_policy(self):
        d = os.path.join(TMP, "nopol"); shutil.copytree(os.path.join(B, "cases", "config_tampering_consistent"), d); os.remove(os.path.join(d, "policy.json"))
        res = verify_ledger(d, os.path.join(TMP, "np_out"), key=KEY, log=lambda *a: None)
        self.assertNotIn("config_policy_mismatch", res["finding_counts"])
        self.assertTrue(any("policy" in u for u in res["unavailable_tests"]))       # honest: UNAVAILABLE, not a PASS

    def test_missing_input_is_unavailable_not_pass(self):
        d = os.path.join(TMP, "miss"); shutil.copytree(os.path.join(B, "cases", "valid"), d)
        r0 = read_records(os.path.join(d, "ledger.jsonl"))[0]; os.remove(os.path.join(d, r0["input"]["stored_as"]))
        res = verify_ledger(d, os.path.join(TMP, "m_out"), key=KEY, log=lambda *a: None)
        self.assertIn("input_bytes_unavailable", res["finding_counts"]); self.assertEqual(res["record_status"][r0["record_id"]], "UNAVAILABLE")
        self.assertNotEqual(res["disposition"], "QUARANTINE")

    def test_truncation_detected_by_manifest(self):
        d = os.path.join(TMP, "trunc"); shutil.copytree(os.path.join(B, "cases", "valid"), d)
        L = open(os.path.join(d, "ledger.jsonl")).read().splitlines(); open(os.path.join(d, "ledger.jsonl"), "w").write("\n".join(L[:-3]) + "\n")
        res = verify_ledger(d, os.path.join(TMP, "t_out"), key=KEY, log=lambda *a: None)
        self.assertIn("manifest_mismatch", res["finding_counts"])

    def test_canonical_output_is_order_and_format_stable(self):
        cfg = {"top_k": 2, "class_names": ["a", "b", "c"], "decision_threshold": 0.5}
        a = build_classification_output([0.2000000001, 0.5, 0.3], cfg); b = build_classification_output([0.2, 0.5000000004, 0.3], cfg)
        self.assertEqual(compare_outputs(a, b)["result"], "EXACT")
        self.assertEqual(compare_outputs(a, build_classification_output([0.2, 0.3, 0.5], cfg))["result"], "MISMATCH")
        c = build_classification_output([0.2, 0.4996, 0.3], cfg); self.assertEqual(compare_outputs(a, c, 1e-3, 0.5)["result"], "TOLERANCE_MATCH")   # threshold-boundary float noise
        self.assertEqual(compare_outputs(a, c)["result"], "MISMATCH")                                                               # ...but not accepted blindly without threshold context

    def test_detection_semantic_compare(self):
        s = [{"class_id": 1, "score": .9, "box": [0, 0, 10, 10]}]
        self.assertEqual(compare_detections(s, [{"class_id": 1, "score": .91, "box": [1, 0, 10, 10]}])["result"], "TOLERANCE_MATCH")
        self.assertEqual(compare_detections(s, [{"class_id": 2, "score": .9, "box": [0, 0, 10, 10]}])["result"], "MISMATCH")

    def test_audit_chain_verifies(self):
        self.assertTrue(AuditLog.verify(os.path.join(TMP, "o", "valid", "audit_log.jsonl")))


if __name__ == "__main__":
    unittest.main()
