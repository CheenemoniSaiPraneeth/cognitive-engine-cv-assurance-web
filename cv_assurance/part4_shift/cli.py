"""Part 4 CLI.
  python -m cv_assurance part4 demo             --out OUT                      # controlled shift benchmark, scored
  python -m cv_assurance part4 build-reference  --images DIR --out REFDIR [--model M] [--encoder onnx]
  python -m cv_assurance part4 run              --reference REFDIR --ops DIR --out OUT [--model M] [--window 50]
  python -m cv_assurance part4 integrated-demo  --out OUT                      # Parts 1+2+3+4 together (needs the full bundle)
  python -m cv_assurance part4 integrated       --dataset D --model M --trusted-model T --stream NAME=DIR [...] --out OUT
"""
from __future__ import annotations
import argparse, json
from pathlib import Path


def main(argv=None):
    ap = argparse.ArgumentParser(prog="cv_assurance part4", description="Part 4 - distribution shift & anomaly assessment (zero-training)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("demo"); d.add_argument("--out", required=True)
    r = sub.add_parser("build-reference"); r.add_argument("--images", required=True); r.add_argument("--out", required=True); r.add_argument("--model"); r.add_argument("--config"); r.add_argument("--onnx-encoder")
    x = sub.add_parser("run"); x.add_argument("--reference", required=True); x.add_argument("--ops", required=True); x.add_argument("--out", required=True); x.add_argument("--model"); x.add_argument("--window", type=int); x.add_argument("--config"); x.add_argument("--onnx-encoder")
    i = sub.add_parser("integrated-demo"); i.add_argument("--out", required=True); i.add_argument("--n-clean", type=int, default=240)
    g = sub.add_parser("integrated"); g.add_argument("--dataset", required=True); g.add_argument("--model", required=True); g.add_argument("--trusted-model", required=True)
    g.add_argument("--stream", action="append", required=True, help="NAME=DIR (repeatable)"); g.add_argument("--out", required=True); g.add_argument("--reference-dir"); g.add_argument("--format", default="auto")
    a = ap.parse_args(argv)
    from .config import load_p4_config
    if a.cmd == "demo":
        from ..benchmark.part2_models import build_benchmark
        from ..benchmark.part4_bench import build_part4_benchmark, evaluate_part4, format_eval4
        from ..part2_model.adapters import load_adapter
        out = Path(a.out); build_part4_benchmark(out / "benchmark"); build_benchmark(out / "benchmark" / "p2")
        m = evaluate_part4(out / "benchmark", out / "assurance_out", adapter=load_adapter(str(out / "benchmark" / "p2" / "models" / "trusted_reference.npz")))
        print("\n" + format_eval4(m)); print(f"\nreports: {out}/assurance_out/cases/<case>/report_part4.html  and  series/<name>/report_part4.html")
    elif a.cmd in ("build-reference", "run"):
        from .data import build_table, load_dir, save_reference
        from .features import OnnxEncoder
        cfg = load_p4_config(a.config); enc = OnnxEncoder(a.onnx_encoder) if a.onnx_encoder else None
        ad = None
        if a.model:
            from ..part2_model.adapters import load_adapter; ad = load_adapter(a.model)
        if a.cmd == "build-reference":
            t, bad = build_table(load_dir(a.images, cfg["window"]), cfg, enc, ad); man = save_reference(t, cfg, a.out); print(f"reference {man['reference_id']} n={man['n']} unreadable={len(bad)} -> {a.out}")
        else:
            from .assess import run_part4
            if a.window: cfg["window"] = a.window
            res = run_part4(a.reference, load_dir(a.ops, cfg["window"]), cfg, a.out, adapter=ad, encoder=enc)
            print(f"LEVEL: {res['level']}  DISPOSITION: {res['disposition']}  findings: {len(res['findings'])}  unavailable: {len(res['unavailable_tests'])}\nreport: {a.out}/report_part4.html")
    elif a.cmd == "integrated-demo":
        from .integration import integrated_demo; integrated_demo(a.out, a.n_clean); print(f"\nOpen {a.out}/result/report_combined_p1234.html")
    else:
        from .integration import run_integrated4
        run_integrated4(a.dataset, a.model, a.trusted_model, a.out, [tuple(s.split("=", 1)) for s in a.stream], a.reference_dir, a.format)


if __name__ == "__main__":
    main()
