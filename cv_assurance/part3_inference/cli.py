"""Part 3 CLI.
  python -m cv_assurance part3 demo            --out OUT   # benchmark ledger + 12 controlled attacks, scored
  python -m cv_assurance part3 record          --model M.npz --images DIR --out LEDGER_DIR [--key K] [--trusted-model T.npz]
  python -m cv_assurance part3 verify          --ledger LEDGER_DIR [--key K] [--out OUT]
  python -m cv_assurance part3 keygen          --out ledger.key
  python -m cv_assurance part3 integrated-demo --out OUT   # Part 1 + 2 + 3 on one dataset / model (needs the full bundle)
  python -m cv_assurance part3 integrated      --dataset DIR --model M --trusted-model T --out OUT [--claim-trusted]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

IMG = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def main(argv=None):
    ap = argparse.ArgumentParser(prog="cv_assurance part3", description="Part 3 - inference provenance & output integrity (zero-training)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("demo"); d.add_argument("--out", required=True)
    k = sub.add_parser("keygen"); k.add_argument("--out", required=True)
    r = sub.add_parser("record"); r.add_argument("--model", required=True); r.add_argument("--images", required=True); r.add_argument("--out", required=True)
    r.add_argument("--key"); r.add_argument("--trusted-model"); r.add_argument("--threshold", type=float, default=0.5)
    v = sub.add_parser("verify"); v.add_argument("--ledger", required=True); v.add_argument("--key"); v.add_argument("--out")
    i = sub.add_parser("integrated-demo"); i.add_argument("--out", required=True); i.add_argument("--n-clean", type=int, default=200)
    g = sub.add_parser("integrated"); g.add_argument("--dataset", required=True); g.add_argument("--model", required=True); g.add_argument("--trusted-model", required=True)
    g.add_argument("--out", required=True); g.add_argument("--claim-trusted", action="store_true"); g.add_argument("--format", default="auto"); g.add_argument("--n-records", type=int, default=60)
    a = ap.parse_args(argv)

    if a.cmd == "demo":
        from ..benchmark.part3_bench import build_part3_benchmark, evaluate_part3, format_eval3
        out = Path(a.out); build_part3_benchmark(out / "benchmark"); m = evaluate_part3(out / "benchmark", out / "assurance_out")
        print(format_eval3(m)); print(f"\nreports: {out}/assurance_out/<case>/report_part3.html")
    elif a.cmd == "keygen":
        from .ledger import generate_key
        generate_key(a.out); print(f"key written to {a.out} (keep offline)")
    elif a.cmd == "record":
        from ..part2_model.adapters import load_adapter
        from .capture import InferenceRecorder, default_config, model_identity, write_policy, write_registry
        from .ledger import generate_key, load_key
        out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
        key = load_key(a.key) if a.key else generate_key(out / "ledger.key")
        ad = load_adapter(a.model); cfg = default_config(ad, decision_threshold=a.threshold)
        rec = InferenceRecorder(out, a.model, key=key, config=cfg, adapter=ad)
        for p in sorted(Path(a.images).rglob("*")):
            if p.suffix.lower() in IMG: rec.record(image_path=p)
        rec.seal(); t = model_identity(a.trusted_model or a.model)
        write_registry(out / "registry.json", [{"model_id": t["model_id"], "sha256": t["sha256"], "path": str(Path(a.trusted_model or a.model).resolve())}])
        write_policy(out / "policy.json", cfg)
        print(f"{len(rec.ledger.records)} records -> {out}/ledger.jsonl (chain head {rec.ledger.records[-1]['record_hash'][:16]}...)")
    elif a.cmd == "verify":
        from .ledger import load_key
        from .verifier import verify_ledger
        res = verify_ledger(a.ledger, a.out, key=load_key(a.key) if a.key else load_key(Path(a.ledger) / "ledger.key"))
        print(f"DISPOSITION: {res['disposition']}  findings: {len(res['findings'])}  unavailable: {len(res['unavailable_tests'])}")
    elif a.cmd == "integrated-demo":
        from .integration import integrated_demo
        integrated_demo(a.out, a.n_clean)
    else:
        from .integration import run_integrated3
        run_integrated3(a.dataset, a.model, a.trusted_model, a.out, a.claim_trusted, a.format, a.n_records)


if __name__ == "__main__":
    main()
