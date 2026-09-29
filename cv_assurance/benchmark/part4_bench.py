"""Controlled distribution-shift benchmark (Part 4) + scoring. Ground truth is used ONLY for scoring, never as detector input."""
from __future__ import annotations
import io, json, random
from pathlib import Path
import numpy as np
from PIL import Image, ImageFilter
from .synthetic import make_image, make_ood_image
from ..part4_shift.assess import run_part4
from ..part4_shift.config import load_p4_config
from ..part4_shift.data import build_table, load_dir, save_reference

NAMES = {"illumination": "illumination", "color": "color", "geometry": "geometry", "compression": "compression", "class_mix": "class_mix", "semantic": "semantic", "sharpness_noise": "sharpness_noise"}


def _shift(img, kind, rng, s=1.0):
    """returns (PIL image, save kwargs)"""
    kw = {}
    if kind in ("illumination", "mixed"):
        a = np.asarray(img, np.float32); img = Image.fromarray(np.clip((a - 128) * (1 - .3 * s) + 128 * (1 - .45 * s) + 10, 0, 255).astype(np.uint8))
    if kind == "camera":
        a = np.asarray(img, np.float32) * np.array([1.0, 1 - .15 * s, 1 + .15 * s]); img = Image.fromarray(np.clip(a + rng.normal(0, 5 * s, a.shape), 0, 255).astype(np.uint8))
        img = img.filter(ImageFilter.GaussianBlur(1.2 * s)).resize((96, 96), Image.BILINEAR)
    if kind in ("compression", "mixed"): kw = {"format": "JPEG", "quality": 10}
    return img, kw


def _save(img, kw, path):
    if kw: img.save(path.with_suffix(".jpg"), **kw); return path.with_suffix(".jpg")
    img.save(path); return path


def make_batch(dirp, tag, n, kind, seed, classes=(0, 1, 2, 3), severity=1.0, source="cam_A", ts0=0):
    d = Path(dirp); d.mkdir(parents=True, exist_ok=True); rng = np.random.default_rng(seed); py = random.Random(seed); meta = {}
    for i in range(n):
        if kind == "semantic" and i % 2 == 0: img = make_ood_image(rng)
        elif kind == "class_mix": img, _ = make_image(int(rng.choice([0, 1], p=[.9, .1])), rng)
        elif kind == "mixed": img, _ = make_image(int(rng.choice([0, 1], p=[.85, .15])), rng)
        else: img, _ = make_image(int(rng.choice(classes)), rng)
        img, kw = _shift(img, kind, rng, severity); p = _save(img, kw, d / f"{tag}_{i:03d}.png")
        meta[p.name] = {"source": "cam_B" if kind in ("camera", "mixed") else source, "batch_id": tag, "timestamp": f"2026-09-{1 + ts0:02d}T00:{i % 60:02d}:00"}
    return meta


def build_part4_benchmark(out, n_ref=400, n_batch=60, seed=11):
    out = Path(out); (out / "reference_images").mkdir(parents=True, exist_ok=True); rng = np.random.default_rng(seed); meta = {}
    for i in range(n_ref):
        img, _ = make_image(i % 4, rng); p = out / "reference_images" / f"ref_{i:04d}.png"; img.save(p); meta[p.name] = {"source": "cam_A", "batch_id": "ref", "timestamp": "2026-08-01T00:00:00"}
    (out / "reference_images" / "metadata.json").write_text(json.dumps(meta))
    gt = {"scenarios": {}, "series": {}}
    kinds = {"stable_1": (None, []), "stable_2": (None, []), "stable_3": (None, []), "stable_4": (None, []), "illumination": ("illumination", ["illumination"]), "camera": ("camera", ["color", "geometry", "sharpness_noise", "semantic"]),
             "compression": ("compression", ["compression"]), "class_prior": ("class_mix", ["class_mix"]), "semantic": ("semantic", ["semantic"]), "mixed": ("mixed", ["illumination", "compression", "class_mix"])}
    for j, (name, (k, exp)) in enumerate(kinds.items()):
        m = make_batch(out / "scenarios" / name, name, n_batch, k or "none", seed * 100 + j); (out / "scenarios" / name / "metadata.json").write_text(json.dumps(m))
        gt["scenarios"][name] = {"shifted": k is not None, "expected_families": exp, "kind": k or "none"}
    # temporal series (8 batches x 50): step at 3, gradual from 2, transient at 4, stable
    def series(name, fn, pattern, onset):
        m = {}
        for b in range(8):
            kind, sev, cl = fn(b); m.update(make_batch(out / "series" / name, f"b{b}", 50, kind, seed * 1000 + sum(map(ord, name)) % 97 + b, classes=cl, severity=sev, ts0=b))
        (out / "series" / name / "metadata.json").write_text(json.dumps(m)); gt["series"][name] = {"pattern": pattern, "onset": onset, "n_batches": 8}
    series("stable", lambda b: ("none", 1, (0, 1, 2, 3)), "none", None)
    series("step", lambda b: ("illumination" if b >= 3 else "none", 1, (0, 1, 2, 3)), "step", 3)
    series("gradual", lambda b: ("illumination" if b >= 2 else "none", min(1.0, 0.15 * (b - 1)), (0, 1, 2, 3)), "gradual", 2)
    series("transient", lambda b: ("illumination" if b == 4 else "none", 1, (0, 1, 2, 3)), "transient", 4)
    (out / "ground_truth.json").write_text(json.dumps(gt, indent=1)); return gt


def make_reference(bench, ref_out, cfg, adapter=None):
    items = load_dir(Path(bench) / "reference_images", cfg["window"]); t, bad = build_table(items, cfg, adapter=adapter)
    return save_reference(t, cfg, ref_out, {"note": "benchmark trusted reference"})


def evaluate_part4(bench, out, adapter=None, log=print):
    bench, out = Path(bench), Path(out); cfg = load_p4_config(); gt = json.loads((bench / "ground_truth.json").read_text())
    ref_dir = out / "reference"; make_reference(bench, ref_dir, cfg, adapter); rows = []
    for name, g in gt["scenarios"].items():
        items = load_dir(bench / "scenarios" / name, 999); r = run_part4(ref_dir, items, cfg, out / "cases" / name, adapter=adapter, log=lambda *a: None)
        fams = set(r["overall_attribution"]["primary_families"]); anyf = set(r["overall_attribution"]["all_flagged_families"])
        rows.append({"case": name, "shifted": g["shifted"], "detected": r["level"] != "NORMAL", "level": r["level"], "ratio": r["overall"]["ratio"], "expected": g["expected_families"], "flagged_families": sorted(fams),
                     "family_recall": None if not g["expected_families"] else round(len(set(g["expected_families"]) & anyf) / len(g["expected_families"]), 2),
                     "family_recall_primary": None if not g["expected_families"] else round(len(set(g["expected_families"]) & fams) / len(g["expected_families"]), 2),
                     "family_precision": None if not g["expected_families"] or not fams else round(len(set(g["expected_families"]) & fams) / len(fams), 2)})
        log(f"  {name:13s} level={r['level']:15s} ratio={r['overall']['ratio']:<7} families={sorted(fams)}")
    ser = []
    for name, g in gt["series"].items():
        items = load_dir(bench / "series" / name, 999); r = run_part4(ref_dir, items, cfg, out / "series" / name, adapter=adapter, log=lambda *a: None); tm = r["temporal"]
        delay = None if tm["onset_index"] is None or g["onset"] is None else tm["onset_index"] - g["onset"]
        drift_start = tm.get("drift_start_index")
        ser.append({"series": name, "truth": g["pattern"], "onset_true": g["onset"], "pattern_pred": tm["pattern"], "onset_pred": tm["onset_index"], "delay_batches": delay, "cusum_alert": tm["cusum_alert_batch"], "drift_start": drift_start,
                    "ratios": [float(x) for x in tm["ratios"]], "false_alert_batches": [i for i, a in enumerate(tm["alerts"]) if a and (g["onset"] is None or i < g["onset"])]})
        log(f"  series {name:9s} truth={g['pattern']:9s} pred={tm['pattern']:22s} onset={tm['onset_index']} ratios={tm['ratios']}")
    sh = [x for x in rows if x["shifted"]]; st = [x for x in rows if not x["shifted"]]
    fb = sum(len(s["false_alert_batches"]) for s in ser); nb = sum((s["onset_true"] if s["onset_true"] is not None else 8) for s in ser)
    fr = [x["family_recall"] for x in sh if x["family_recall"] is not None]; fp = [x["family_precision"] for x in sh if x["family_precision"] is not None]
    # calibration stability across batch sizes: false-alarm on clean reference-like batches of different n
    from ..part4_shift.engine import ShiftEngine
    from ..part4_shift.data import load_reference
    ref, _ = load_reference(ref_dir); eng = ShiftEngine(ref, cfg, cache_dir=ref_dir); stab = {}
    for n in (30, 60, 100):
        it = load_dir(bench / "series" / "stable", 999); tb, _ = build_table(it[:n] if n <= 50 else it, cfg, adapter=adapter)
        stab[n] = {"batches_tested": 0, "alerts": 0}
        rr = np.random.default_rng(3)
        for _ in range(12):                                       # fresh clean batches drawn from the stable series pool
            pool = np.arange(len(tb)); idx = rr.choice(pool, min(n, len(pool)), replace=False) if len(pool) >= n else pool
            if len(idx) < n: continue
            stab[n]["batches_tested"] += 1; stab[n]["alerts"] += int(eng.evaluate(tb.take(idx))["alert"])
    m = {"shift_scenarios": len(sh), "shift_recall": round(sum(x["detected"] for x in sh) / max(len(sh), 1), 3), "stable_false_alarm_rate_windows": round(sum(x["detected"] for x in st) / max(len(st), 1), 3),
         "stable_false_alarm_rate_batches": round(fb / max(nb, 1), 3), "mean_family_recall": round(float(np.mean(fr)), 3), "mean_family_recall_primary": round(float(np.mean([x["family_recall_primary"] for x in sh if x["family_recall_primary"] is not None])), 3), "mean_family_precision": round(float(np.mean(fp)), 3) if fp else None,
         "onset_delays_batches": {s["series"]: s["delay_batches"] for s in ser if s["onset_true"] is not None}, "pattern_accuracy": round(sum(s["pattern_pred"] == s["truth"] or (s["truth"] == "none" and s["pattern_pred"] == "none") for s in ser) / len(ser), 3),
         "calibration_stability_alarm_rate_by_batch_size": {str(n): (round(v["alerts"] / v["batches_tested"], 3) if v["batches_tested"] else None) for n, v in stab.items()},
         "cases": rows, "series": ser, "task_output_available": adapter is not None}
    (out / "evaluation_part4.json").write_text(json.dumps(m, indent=1, default=str)); return m


def format_eval4(m):
    return (f"Part 4 evaluation  shift-scenario recall {m['shift_recall']} ({m['shift_scenarios']} scenarios) | stable false-alarm rate: windows {m['stable_false_alarm_rate_windows']}, batches {m['stable_false_alarm_rate_batches']}\n"
            f"attribution: family recall (any flagged) {m['mean_family_recall']}, recall (primary only) {m['mean_family_recall_primary']}, precision (primary) {m['mean_family_precision']} | onset delay (batches) {m['onset_delays_batches']} | temporal pattern accuracy {m['pattern_accuracy']}\n"
            f"false-alarm rate by batch size {m['calibration_stability_alarm_rate_by_batch_size']} (target alpha=0.01-0.05)")
