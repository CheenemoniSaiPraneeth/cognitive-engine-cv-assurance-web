"""Part 2 tests (unittest/pytest compatible). Run: python -m unittest discover -s tests -v"""
import json, os, shutil, sys, tempfile, unittest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from cv_assurance.benchmark.part2_models import build_benchmark
from cv_assurance.common.audit import AuditLog
from cv_assurance.part2_model.adapters import BlackBoxAdapter, Unavailable, load_adapter
from cv_assurance.part2_model.assurance import run_part2

TMP = tempfile.mkdtemp()
GT = build_benchmark(os.path.join(TMP, "b"))
B = os.path.join(TMP, "b")
REF = f"{B}/models/trusted_reference.npz"


def run(name, mode="auto", ref=True):
    return run_part2(f"{B}/models/{name}.npz", os.path.join(TMP, f"o_{name}_{mode}_{ref}"), REF if ref else None,
                     reference_images_dir=f"{B}/reference_images", mode=mode, log=lambda *a: None)


def sig(r): return [f for f in r["findings"] if f["severity"] in ("medium", "high", "critical")]


class Part2(unittest.TestCase):
    def test_identical_copy_accepts(self):
        r = run("clean_copy"); self.assertEqual(r["disposition"], "ACCEPT"); self.assertEqual(r["identity_status"], "MATCH")

    def test_reserialized_is_benign_but_visible(self):
        r = run("reserialized"); self.assertEqual(r["identity_status"], "MISMATCH"); self.assertEqual(sig(r), [])
        self.assertEqual(r["substitution"]["verdict"], "SAME_WEIGHTS_DIFFERENT_BYTES")

    def test_small_finetune_not_flagged(self): self.assertEqual(sig(run("finetuned_small")), [])

    def test_trojan_quarantined_whitebox_and_blackbox(self):
        for mode in ("auto", "blackbox"):
            r = run("trojan_checker", mode); self.assertEqual(r["disposition"], "QUARANTINE")
            self.assertTrue(any(f["category"] == "trigger_candidate_response" for f in r["findings"]))

    def test_blackbox_marks_unavailable(self):
        r = run("trojan_checker", "blackbox"); self.assertEqual(r["module_status"]["parameters"], "UNAVAILABLE")
        self.assertTrue(any("black-box" in u for u in r["unavailable_tests"]))

    def test_whitebox_finds_stealth_trojan(self):
        r = run("trojan_stealth"); self.assertTrue(any(f["category"] == "dormant_high_influence_unit" for f in r["findings"]))

    def test_substitute_detected(self):
        r = run("substitute"); self.assertEqual(r["substitution"]["verdict"], "DIFFERENT_MODEL_SUSPECTED")

    def test_no_reference_is_limited_not_pass(self):
        r = run("substitute", ref=False); self.assertEqual(r["coverage"], "LIMITED"); self.assertTrue(r["unavailable_tests"])

    def test_unsupported_format_is_unavailable(self):
        p = os.path.join(TMP, "x.bin"); open(p, "wb").write(b"0")
        r = run_part2(p, os.path.join(TMP, "ox"), log=lambda *a: None); self.assertEqual(r["disposition"], "UNAVAILABLE")

    def test_deterministic_and_audit_chain(self):
        a, b = run("trojan_checker"), run("trojan_checker")
        self.assertEqual(a["behavioural_fingerprint"]["prediction_hash"], b["behavioural_fingerprint"]["prediction_hash"])
        self.assertTrue(AuditLog.verify(os.path.join(TMP, "o_trojan_checker_auto_True", "audit_log.jsonl")))

    def test_no_training_flag(self):
        self.assertFalse(run("clean_copy")["environment"]["training_performed"])


if __name__ == "__main__":
    unittest.main()
