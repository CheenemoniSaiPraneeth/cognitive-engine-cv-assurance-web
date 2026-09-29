"""Command line interface.

  python -m cv_assurance.cli run   --dataset DIR --out OUT [--format auto|coco|yolo] [--reference DIR] [--config cfg.json]
  python -m cv_assurance.cli demo  --out OUT [--n-clean 400] [--format coco|yolo]
  python -m cv_assurance part2 ...      (model integrity)   |   python -m cv_assurance part3 ...   (inference provenance)  |  python -m cv_assurance part4 ...  (distribution shift)
  python -m cv_assurance.cli evaluate --results OUT/assurance_part1.json --ground-truth DIR/ground_truth.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main(argv=None):
    import sys
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "part2":          # Part 2 is a sub-application of the same CLI
        from .part2_model.cli import main as p2
        return p2(argv[1:])
    if argv and argv[0] == "part3":          # Part 3 (inference provenance) sub-application
        from .part3_inference.cli import main as p3
        return p3(argv[1:])
    if argv and argv[0] == "part5":          # Part 5 (analyst assurance & governance) sub-application
        from .part5_governance.cli import main as p5
        return p5(argv[1:])
    if argv and argv[0] == "part4":          # Part 4 (distribution shift & anomaly) sub-application
        from .part4_shift.cli import main as p4
        return p4(argv[1:])
    from .benchmark.evaluate import evaluate, format_metrics   # lazy: Part 2 stand-alone bundle has no Part 1 modules
    from .benchmark.synthetic import generate
    from .part1_data.pipeline import run_part1
    ap = argparse.ArgumentParser(prog="cv_assurance", description="Part 1 - training-data integrity (zero-training)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--dataset", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--format", default="auto", choices=["auto", "coco", "yolo"])
    r.add_argument("--reference", help="directory of TRUSTED clean images for OOD calibration")
    r.add_argument("--config")
    r.add_argument("--ann-file")
    r.add_argument("--images-dir")
    r.add_argument("--labels-dir")
    d = sub.add_parser("demo")
    d.add_argument("--out", required=True)
    d.add_argument("--n-clean", type=int, default=400)
    d.add_argument("--format", default="coco", choices=["coco", "yolo"])
    e = sub.add_parser("evaluate")
    e.add_argument("--results", required=True)
    e.add_argument("--ground-truth", required=True)
    a = ap.parse_args(argv)

    if a.cmd == "run":
        res = run_part1(a.dataset, a.out, a.format, a.reference, a.config, a.ann_file, a.images_dir, a.labels_dir)
        print(f"DISPOSITION: {res['disposition']}  findings: {len(res['findings'])}")
    elif a.cmd == "demo":
        root = Path(a.out)
        data_dir = root / "synthetic_dataset"
        generate(data_dir, a.n_clean, a.format)
        print(f"synthetic {a.format.upper()} dataset with known attacks -> {data_dir}")
        res = run_part1(data_dir, root / "assurance_out", a.format)
        m = evaluate(res, json.loads((data_dir / "ground_truth.json").read_text()))
        (root / "assurance_out" / "evaluation.json").write_text(json.dumps(m, indent=2))
        print("\n" + format_metrics(m))
        print(f"\nOpen {root / 'assurance_out' / 'report.html'}")
    else:
        res = json.loads(Path(a.results).read_text())
        m = evaluate(res, json.loads(Path(a.ground_truth).read_text()))
        print(format_metrics(m))


if __name__ == "__main__":
    main()
