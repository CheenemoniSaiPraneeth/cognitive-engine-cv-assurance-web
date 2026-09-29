"""Score Part 2 against the benchmark ground truth. Ground truth is used ONLY here, never by the detectors.
Flagged := any finding with severity >= medium. Modes: white-box/black-box x with/without a trusted reference model."""
from __future__ import annotations

import json
from pathlib import Path

from ..part2_model.assurance import run_part2
from ..part2_model.findings import SEV_RANK

MODES = [("whitebox+ref", "auto", True), ("blackbox+ref", "blackbox", True), ("whitebox,no-ref", "auto", False), ("blackbox,no-ref", "blackbox", False)]


def evaluate_part2(bench_dir, out_dir, log=print):
    bench, out = Path(bench_dir), Path(out_dir)
    gt = json.loads((bench / "ground_truth.json").read_text())
    rows, per = {}, {}
    for mname, mode, use_ref in MODES:
        tp = fp = fn = tn = 0; per[mname] = {}
        for name, meta in gt["models"].items():
            r = run_part2(bench / "models" / f"{name}.npz", out / f"{name}__{mname.replace(',', '_').replace('+', '_')}",
                          reference_model=(bench / gt["reference_model"]) if use_ref else None,
                          reference_images_dir=bench / gt["reference_images"], mode=mode, log=lambda *a: None)
            sig = [f for f in r["findings"] if SEV_RANK.get(f["severity"], 0) >= 2]
            flagged = bool(sig)
            per[mname][name] = {"compromised": meta["compromised"], "flagged": flagged, "disposition": r["disposition"],
                                "categories": sorted({f["category"] for f in sig}), "verdict": r["substitution"]["verdict"]}
            if meta["compromised"]: tp += flagged; fn += (not flagged)
            else: fp += flagged; tn += (not flagged)
        prec = tp / (tp + fp) if tp + fp else 1.0; rec = tp / (tp + fn) if tp + fn else 1.0
        rows[mname] = {"TP": tp, "FP": fp, "FN": fn, "TN": tn, "precision": round(prec, 3), "recall": round(rec, 3),
                       "F1": round(2 * prec * rec / (prec + rec), 3) if prec + rec else 0.0, "clean_false_positive_rate": round(fp / max(1, fp + tn), 3)}
    res = {"summary": rows, "per_model": per}
    (out / "evaluation_part2.json").write_text(json.dumps(res, indent=2))
    return res


def format_eval(res):
    L = [f"{'mode':18s} TP FP FN TN  prec   rec    F1   cleanFPR"]
    for m, r in res["summary"].items():
        L.append(f"{m:18s} {r['TP']:2d} {r['FP']:2d} {r['FN']:2d} {r['TN']:2d}  {r['precision']:.3f} {r['recall']:.3f} {r['F1']:.3f}  {r['clean_false_positive_rate']:.3f}")
    L.append("\nper-model (flagged?):")
    names = list(next(iter(res["per_model"].values())))
    L.append(f"{'model':18s} truth      " + "  ".join(f"{m[:15]:15s}" for m in res["per_model"]))
    for n in names:
        t = "COMPROMISED" if res["per_model"]["whitebox+ref"][n]["compromised"] else "benign     "
        L.append(f"{n:18s} {t}  " + "  ".join(f"{('FLAG ' + res['per_model'][m][n]['disposition']) if res['per_model'][m][n]['flagged'] else 'ok':15s}" for m in res["per_model"]))
    return "\n".join(L)
