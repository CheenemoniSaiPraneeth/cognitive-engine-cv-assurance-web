"""Run with:  python -m unittest discover -s tests -v   (no pytest required)"""
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cv_assurance.benchmark.evaluate import evaluate
from cv_assurance.benchmark.synthetic import generate, make_image
from cv_assurance.common.audit import AuditLog
from cv_assurance.dataset.ingest import load_dataset, make_sample_id
from cv_assurance.part1_data.fingerprint import dhash64, phash64
from cv_assurance.part1_data.pipeline import run_part1

quiet = lambda *a: None


class UnitTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_sample_id_stable(self):
        self.assertEqual(make_sample_id("a/b.png"), make_sample_id("a\\b.png"))

    def test_hashes_deterministic_and_robust(self):
        rng = np.random.default_rng(0)
        img, _ = make_image(1, rng)
        g = np.asarray(img.convert("L").resize((32, 32)), dtype=np.float32)
        self.assertEqual(phash64(g), phash64(g.copy()))
        noisy = np.clip(g + rng.normal(0, 2, g.shape), 0, 255).astype(np.float32)
        self.assertLessEqual(bin(phash64(g) ^ phash64(noisy)).count("1"), 8)
        g9 = np.asarray(img.convert("L").resize((9, 8)), dtype=np.float32)
        self.assertTrue(0 <= dhash64(g9) < 2 ** 64)

    def test_coco_structural_faults(self):
        generate(self.tmp / "ds", n_clean=40, fmt="coco")
        b = load_dataset(self.tmp / "ds")
        kinds = {i["kind"] for i in b.issues}
        self.assertIn("missing_image_file", kinds)
        self.assertIn("impossible_coordinates", kinds)
        self.assertEqual(len({s.sample_id for s in b.samples}), len(b.samples))

    def test_yolo_matches_coco(self):
        generate(self.tmp / "c", n_clean=40, fmt="coco")
        generate(self.tmp / "y", n_clean=40, fmt="yolo")
        bc, by = load_dataset(self.tmp / "c"), load_dataset(self.tmp / "y")
        self.assertEqual((bc.fmt, by.fmt), ("COCO", "YOLO"))
        # COCO lists the deliberately missing image; YOLO discovers files on disk only
        self.assertEqual(len(bc.samples) - 1, len(by.samples))
        self.assertEqual(by.class_names[0], "circle")
        self.assertIn("contributor_x", {s.contributor_id for s in by.samples})

    def test_trusted_reference_mode(self):
        generate(self.tmp / "ds", n_clean=150, fmt="coco")
        ref = self.tmp / "ref"
        ref.mkdir()
        rng = np.random.default_rng(99)
        for i in range(80):
            make_image(i % 4, rng)[0].save(ref / f"r{i}.png")
        res = run_part1(self.tmp / "ds", self.tmp / "out", reference=ref, log=quiet)
        self.assertEqual(res["module_status"]["ood"]["mode"], "trusted_reference")
        self.assertFalse(any("trusted-reference" in u for u in res["unavailable_tests"]))


class EndToEnd(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = Path(tempfile.mkdtemp())
        cls.gt = generate(cls.root / "ds", n_clean=180, fmt="coco")
        cls.res = run_part1(cls.root / "ds", cls.root / "out", log=quiet)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.root, ignore_errors=True)

    def test_detectors_find_injected_attacks(self):
        m = evaluate(self.res, self.gt)
        self.assertGreaterEqual(m["duplicate_flood"]["recall"], 0.9)
        self.assertGreaterEqual(m["trigger"]["recall"], 0.8)
        self.assertGreaterEqual(m["trigger"]["precision"], 0.8)
        self.assertGreaterEqual(m["label_flip"]["recall"], 0.6)
        self.assertEqual(m["structural"]["recall"], 1.0)
        self.assertLessEqual(m["clean_set"]["false_positive_rate"], 0.06)

    def test_source_aggregation(self):
        s = self.res["source_summaries"]
        self.assertIn(s["contributor_id:contributor_x"]["disposition"], ("REVIEW", "QUARANTINE"))
        for c in ("contributor_a", "contributor_b", "contributor_c"):
            self.assertEqual(s[f"contributor_id:{c}"]["disposition"], "ACCEPT")
        self.assertIn(self.res["disposition"], ("REVIEW", "QUARANTINE"))

    def test_zero_training_and_honest_unavailable(self):
        self.assertFalse(self.res["environment"]["training_performed"])
        self.assertFalse(self.res["environment"]["network_access_required"])
        self.assertTrue(any("trusted-reference" in u for u in self.res["unavailable_tests"]))

    def test_audit_chain(self):
        p = self.root / "out" / "audit_log.jsonl"
        self.assertTrue(AuditLog.verify(p))
        lines = p.read_text().splitlines()
        ev = json.loads(lines[1])
        ev["result"] = "tampered"
        lines[1] = json.dumps(ev)
        bad = self.root / "tampered.jsonl"
        bad.write_text("\n".join(lines) + "\n")
        self.assertFalse(AuditLog.verify(bad))

    def test_reports_exist(self):
        for f in ("assurance_part1.json", "report.html", "findings.jsonl", "sample_flags.csv", "audit_log.jsonl"):
            self.assertTrue((self.root / "out" / f).exists(), f)

    def test_reproducible(self):
        res2 = run_part1(self.root / "ds", self.root / "out2", log=quiet)
        sig = lambda r: sorted((f["category"], f["asset_id"]) for f in r["findings"])
        self.assertEqual(sig(self.res), sig(res2))
        self.assertEqual(self.res["dataset_manifest"]["dataset_sha256"], res2["dataset_manifest"]["dataset_sha256"])
        self.assertEqual(self.res["configuration_sha256"], res2["configuration_sha256"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
