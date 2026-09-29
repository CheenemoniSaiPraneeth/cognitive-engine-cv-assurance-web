"""Part 2 CLI.
  python -m cv_assurance part2 demo   --out OUT                       # build benchmark models, run all modes, score
  python -m cv_assurance part2 run    --model M.npz --out OUT [--reference-model R.npz] [--reference-images DIR] [--blackbox] [--expected-sha SHA]
  python -m cv_assurance part2 integrated-demo --out OUT              # Part 1 dataset + Part 2 model -> combined assurance
  python -m cv_assurance part2 integrated --dataset DIR --model M --out OUT [--reference-model R] [--part1-dir OUT1]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main(argv=None):
    ap = argparse.ArgumentParser(prog="cv_assurance part2", description="Part 2 - model integrity (zero-training)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("demo"); d.add_argument("--out", required=True)
    r = sub.add_parser("run"); r.add_argument("--model", required=True); r.add_argument("--out", required=True)
    r.add_argument("--reference-model"); r.add_argument("--reference-images"); r.add_argument("--expected-sha")
    r.add_argument("--blackbox", action="store_true"); r.add_argument("--config")
    i = sub.add_parser("integrated-demo"); i.add_argument("--out", required=True); i.add_argument("--n-clean", type=int, default=200)
    g = sub.add_parser("integrated"); g.add_argument("--dataset", required=True); g.add_argument("--model", required=True); g.add_argument("--out", required=True)
    g.add_argument("--reference-model"); g.add_argument("--part1-dir"); g.add_argument("--format", default="auto"); g.add_argument("--blackbox", action="store_true")
    a = ap.parse_args(argv)

    if a.cmd == "demo":
        from ..benchmark.part2_eval import evaluate_part2, format_eval
        from ..benchmark.part2_models import build_benchmark
        out = Path(a.out); build_benchmark(out / "benchmark")
        print(f"benchmark models + ground truth -> {out / 'benchmark'}")
        res = evaluate_part2(out / "benchmark", out / "assurance_out")
        print("\n" + format_eval(res)); print(f"\nOpen {out / 'assurance_out'}/trojan_checker__whitebox_ref/report_part2.html")
    elif a.cmd == "run":
        from .assurance import run_part2
        res = run_part2(a.model, a.out, a.reference_model, a.expected_sha, reference_images_dir=a.reference_images,
                        mode="blackbox" if a.blackbox else "auto", config=a.config)
        print(f"DISPOSITION: {res['disposition']} findings: {len(res['findings'])} unavailable: {len(res['unavailable_tests'])}")
    elif a.cmd == "integrated-demo":
        from ..benchmark.part2_models import build_benchmark
        from ..benchmark.synthetic import generate
        from .integration import run_integrated
        out = Path(a.out); build_benchmark(out / "benchmark"); generate(out / "dataset", a.n_clean, "coco")
        bm = out / "benchmark" / "models"
        for name in ("trojan_checker", "trusted_reference"):
            print(f"\n================ integrated run: model = {name} ================")
            run_integrated(out / "dataset", bm / f"{name}.npz", out / f"integrated_{name}",
                           reference_model=None if name == "trusted_reference" else bm / "trusted_reference.npz", part1_dir=out / "integrated_trojan_checker" / "part1" if name == "trusted_reference" else None)
        print(f"\nOpen {out}/integrated_trojan_checker/report_combined.html")
    else:
        from .integration import run_integrated
        run_integrated(a.dataset, a.model, a.out, a.reference_model, fmt=a.format, part1_dir=a.part1_dir, mode="blackbox" if a.blackbox else "auto")


if __name__ == "__main__":
    main()
