"""Part 4 tests. Run: python -m unittest discover -s tests -v"""
import json, os, shutil, sys, tempfile, unittest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from pathlib import Path
from cv_assurance.benchmark.part2_models import build_benchmark
from cv_assurance.benchmark.part4_bench import build_part4_benchmark, evaluate_part4
from cv_assurance.part2_model.adapters import load_adapter
from cv_assurance.common.audit import AuditLog
from cv_assurance.part4_shift.stats import ks, psi, jsd

TMP = Path(tempfile.mkdtemp()); build_part4_benchmark(TMP / "b"); build_benchmark(TMP / "b" / "p2")
AD = load_adapter(str(TMP / "b" / "p2" / "models" / "trusted_reference.npz")); M = evaluate_part4(TMP / "b", TMP / "o", adapter=AD, log=lambda *a: None)
ROW = {r["case"]: r for r in M["cases"]}; SER = {s["series"]: s for s in M["series"]}


class Part4(unittest.TestCase):
    def test_stats_sanity(self):
        import numpy as np; a = np.random.default_rng(0).normal(0, 1, 500); b = a + 2
        self.assertLess(ks(a, a), 1e-9); self.assertGreater(ks(a, b), .5); self.assertGreater(psi(a, b), .5); self.assertGreater(jsd(a, b), .3)
    def test_stable_windows_normal(self):
        for n in ("stable_1", "stable_2", "stable_3", "stable_4"): self.assertEqual(ROW[n]["level"], "NORMAL", n)
    def test_every_shift_detected(self):
        for n in ("illumination", "camera", "compression", "class_prior", "semantic", "mixed"): self.assertTrue(ROW[n]["detected"], n)
    def test_expected_family_flagged(self):
        for n in ("illumination", "camera", "compression", "class_prior"): self.assertGreater(ROW[n]["family_recall"], 0, n)
    def test_temporal_patterns(self):
        self.assertEqual(SER["stable"]["pattern_pred"], "none"); self.assertEqual(SER["step"]["pattern_pred"], "step"); self.assertEqual(SER["step"]["onset_pred"], 3)
        self.assertEqual(SER["transient"]["pattern_pred"], "transient"); self.assertEqual(SER["gradual"]["pattern_pred"], "gradual")
    def test_no_false_alert_before_onset(self):
        for s in SER.values(): self.assertEqual(s["false_alert_batches"], [], s["series"])
    def test_report_and_audit_written(self):
        d = TMP / "o" / "cases" / "illumination"
        for f in ("assurance_part4.json", "report_part4.html", "findings_part4.jsonl", "audit_log_part4.jsonl"): self.assertTrue((d / f).exists(), f)
        self.assertTrue(AuditLog.verify(d / "audit_log_part4.jsonl"))
    def test_unavailable_without_model_is_disclosed(self):
        from cv_assurance.part4_shift.assess import run_part4
        from cv_assurance.part4_shift.config import load_p4_config
        from cv_assurance.part4_shift.data import load_dir
        cfg = load_p4_config(); r = run_part4(TMP / "o" / "reference", load_dir(TMP / "b" / "scenarios" / "illumination", 999), cfg, TMP / "nomodel", log=lambda *a: None)
        self.assertTrue(any("task_output" in u for u in r["unavailable_tests"]))
    def test_reproducible(self):
        from cv_assurance.part4_shift.assess import run_part4
        from cv_assurance.part4_shift.config import load_p4_config
        from cv_assurance.part4_shift.data import load_dir
        cfg = load_p4_config(); it = load_dir(TMP / "b" / "scenarios" / "camera", 999)
        a = run_part4(TMP / "o" / "reference", it, cfg, TMP / "r1", adapter=AD, log=lambda *a: None); b = run_part4(TMP / "o" / "reference", it, cfg, TMP / "r2", adapter=AD, log=lambda *a: None)
        self.assertEqual(a["overall"]["ratio"], b["overall"]["ratio"]); self.assertEqual(a["level"], b["level"])

if __name__ == "__main__": unittest.main()
