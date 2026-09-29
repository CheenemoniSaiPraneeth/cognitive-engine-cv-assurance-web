"""Score Part 1 findings against a ground-truth manifest (precision / recall / F1 / clean FPR)."""
from __future__ import annotations

CAT_MAP = {"label_flip": {"label_anomaly"}, "duplicate_flood": {"duplicate_group"},
           "ood": {"ood_anomaly"}, "trigger": {"trigger_candidate"}, "structural": {"structural_error"}}


def evaluate(results: dict, gt: dict) -> dict:
    base = lambda f: f.replace("\\", "/").rsplit("/", 1)[-1]
    flags = {base(fn): set(v) for fn, v in results["sample_flags"].items()}
    all_files = {base(f) for f in results["samples_index"].values()}
    gt = {k: ([base(x) for x in v] if isinstance(v, list) else v) for k, v in gt.items()}
    attack_files = set().union(*[set(gt[k]) for k in CAT_MAP if k in gt])
    metrics = {}
    for attack, cats in CAT_MAP.items():
        truth = set(gt.get(attack, []))
        pred = {fn for fn, c in flags.items() if c & cats}
        tp, fp, fn_ = len(pred & truth), len(pred - truth), len(truth - pred)
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn_) if tp + fn_ else 0.0
        f1 = 2 * p * r / (p + r) if p + r else 0.0
        metrics[attack] = {"truth": len(truth), "flagged": len(pred), "tp": tp, "fp": fp, "fn": fn_,
                           "precision": round(p, 3), "recall": round(r, 3), "f1": round(f1, 3)}
    clean = all_files - attack_files
    flagged_clean = {fn for fn in clean if flags.get(fn, set()) - {"converging_signals"}}
    metrics["clean_set"] = {"clean_samples": len(clean), "flagged_any": len(flagged_clean),
                            "false_positive_rate": round(len(flagged_clean) / max(len(clean), 1), 4),
                            "flagged_examples": sorted(flagged_clean)[:10]}
    src = results["source_summaries"]
    atk = gt.get("attack_source")
    if atk:
        key = f"contributor_id:{atk}"
        metrics["source_level"] = {
            "attack_source_disposition": src.get(key, {}).get("disposition"),
            "other_sources_flagged": sorted(k for k, v in src.items()
                                            if k.startswith("contributor_id:") and k != key and v["disposition"] != "ACCEPT")}
    return metrics


def format_metrics(m: dict) -> str:
    lines = [f"{'attack':<17}{'truth':>6}{'flag':>6}{'TP':>5}{'FP':>5}{'FN':>5}{'prec':>8}{'rec':>8}{'F1':>8}"]
    for k, v in m.items():
        if "precision" in v:
            lines.append(f"{k:<17}{v['truth']:>6}{v['flagged']:>6}{v['tp']:>5}{v['fp']:>5}{v['fn']:>5}"
                         f"{v['precision']:>8}{v['recall']:>8}{v['f1']:>8}")
    c = m["clean_set"]
    lines.append(f"clean samples: {c['clean_samples']}  flagged: {c['flagged_any']}  FPR: {c['false_positive_rate']}")
    if "source_level" in m:
        s = m["source_level"]
        lines.append(f"attack source disposition: {s['attack_source_disposition']}; other sources flagged: {s['other_sources_flagged']}")
    return "\n".join(lines)
