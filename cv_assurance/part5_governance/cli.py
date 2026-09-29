"""Part 5 CLI.
  python -m cv_assurance part5 assemble --results DIR [DIR ...] --out OUT [--policy policy.json]   # aggregate + correlate + score + report + seal
  python -m cv_assurance part5 show     --out OUT                                                   # text summary of cases
  python -m cv_assurance part5 act      --out OUT --case C-xxxx --action review|quarantine|retest|accept|close --actor NAME [--rationale TXT] [--override]
  python -m cv_assurance part5 apply    --out OUT --actions analyst_actions.json                    # decisions exported from the dashboard
  python -m cv_assurance part5 verify   --out OUT                                                   # audit chain + signatures + sealed export + input hashes
  python -m cv_assurance part5 demo     --out OUT                                                   # bundled sample Part 1-4 outputs (works stand-alone)
  python -m cv_assurance part5 integrated-demo --out OUT                                            # full Parts 1-5 (needs the full bundle)
  python -m cv_assurance part5 integrated --dataset D --model M --trusted-model T --stream NAME=DIR [...] --out OUT
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _show(fin):
    ov = fin["overall"]
    print(f"\nASSESSMENT {fin['assessment_id']}   system={ov['system_disposition']}   governance-adjusted={ov['governance_adjusted_disposition']}   coverage={ov['coverage']}")
    print("parts: " + "  ".join(f"P{k}={v['disposition']}" + (f"({v['level']})" if v.get("level") else "") for k, v in fin["parts"].items()))
    for c in fin["cases"]:
        st = fin["governance"]["case_state"][c["case_id"]]["status"]
        print(f"  {c['case_id']} {c['suggested_disposition']:<11} {c['priority']['band']:<9}{c['priority']['score']:<6} P{','.join(map(str, c['parts']))}  [{st}]  {c['title']}")
    a = fin["audit"]["verify"]; print(f"audit: {'OK' if a['ok'] else 'BROKEN'} ({a['events']} events, {a['reason']})")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="cv_assurance part5", description="Part 5 - analyst assurance & governance (zero-training)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("assemble"); a.add_argument("--results", nargs="+", required=True); a.add_argument("--out", required=True); a.add_argument("--policy")
    s = sub.add_parser("show"); s.add_argument("--out", required=True)
    c = sub.add_parser("act"); c.add_argument("--out", required=True); c.add_argument("--case", required=True); c.add_argument("--action", required=True)
    c.add_argument("--actor", default="analyst"); c.add_argument("--rationale", default=""); c.add_argument("--override", action="store_true")
    p = sub.add_parser("apply"); p.add_argument("--out", required=True); p.add_argument("--actions", required=True)
    v = sub.add_parser("verify"); v.add_argument("--out", required=True)
    d = sub.add_parser("demo"); d.add_argument("--out", required=True)
    i = sub.add_parser("integrated-demo"); i.add_argument("--out", required=True); i.add_argument("--n-clean", type=int, default=240)
    g = sub.add_parser("integrated"); g.add_argument("--dataset", required=True); g.add_argument("--model", required=True); g.add_argument("--trusted-model", required=True)
    g.add_argument("--stream", action="append", required=True); g.add_argument("--out", required=True); g.add_argument("--format", default="auto")
    x = ap.parse_args(argv)
    from .assemble import act, apply_actions, assemble, verify_export
    from .governance import GovernanceError
    if x.cmd == "assemble":
        _show(assemble(x.results, x.out, x.policy)); print(f"\nOpen {x.out}/report_part5.html")
    elif x.cmd == "show":
        _show(json.loads((Path(x.out) / "assurance_part5.json").read_text()))
    elif x.cmd == "act":
        try:
            _show(act(x.out, x.case, x.action, x.actor, x.rationale, x.override))
        except GovernanceError as e:
            raise SystemExit(f"REJECTED: {e}")
    elif x.cmd == "apply":
        for r in apply_actions(x.out, x.actions):
            print(*r)
        _show(json.loads((Path(x.out) / "assurance_part5.json").read_text()))
    elif x.cmd == "verify":
        r = verify_export(x.out)
        for k, ok in r["checks"].items():
            print(f"  {'PASS' if ok else 'FAIL'}  {k}")
        print("VERIFIED" if r["ok"] else f"NOT VERIFIED ({r['audit']['reason']})"); raise SystemExit(0 if r["ok"] else 1)
    elif x.cmd == "demo":
        base = Path(__file__).resolve().parents[2] / "sample_inputs"
        for sc in sorted(q for q in base.iterdir() if q.is_dir()):
            print(f"\n=== scenario: {sc.name} ==="); _show(assemble([sc], Path(x.out) / sc.name, log=lambda *a: None))
        print(f"\nOpen {x.out}/<scenario>/report_part5.html")
    elif x.cmd == "integrated-demo":
        from .integration import integrated_demo
        integrated_demo(x.out, x.n_clean)
    else:
        from .integration import run_integrated5
        fin = run_integrated5(x.dataset, x.model, x.trusted_model, x.out, [tuple(t.split("=", 1)) for t in x.stream], x.format); _show(fin)


if __name__ == "__main__":
    main()
