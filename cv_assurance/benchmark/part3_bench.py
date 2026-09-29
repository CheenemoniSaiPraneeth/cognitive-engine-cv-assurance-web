"""Part 3 benchmark FIXTURE + scorer: a valid ledger produced with the trusted model, then controlled attacks.
Two attacker strengths: 'naive' (edits a field, cannot re-seal) and 'insider' (holds the signing key, re-chains + re-signs, so only
CONTENT checks - re-inference, registry, policy, binding - can catch it). Ground truth is used only for scoring."""
from __future__ import annotations

import copy
import io
import json
import shutil
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
from PIL import Image

from ..common.hashing import sha256_bytes, sha256_json
from ..part2_model.adapters import load_adapter
from ..part3_inference.canonical import build_classification_output, output_hash
from ..part3_inference.capture import InferenceRecorder, decode_image, default_config, write_policy, write_registry, model_identity
from ..part3_inference.ledger import generate_key, load_key, rebuild, read_records
from ..part3_inference.verifier import verify_ledger
from .part2_models import build_benchmark
from .synthetic import add_trigger, make_image

N, ATTACK_IDX = 30, [5, 10, 15, 20]
T0 = datetime(2026, 9, 29, 8, 0, 0, tzinfo=timezone.utc)


class Clock:
    def __init__(self, t0=T0, step=2.0): self.t, self.step = t0, step
    def __call__(self):
        self.t += timedelta(seconds=self.step / 2); return self.t.isoformat(timespec="milliseconds")


def _png(im: Image.Image) -> bytes:
    b = io.BytesIO(); im.save(b, "PNG"); return b.getvalue()


def make_images(n, seed):
    rng = np.random.default_rng(seed); ims = []
    for i in range(n):
        im, _ = make_image(i % 4, rng); ims.append(im)
    return ims


def record_all(ledger_dir, model_path, images_bytes, key, clock, **kw):
    rec = InferenceRecorder(ledger_dir, model_path, key=key, clock=clock, **kw)
    for i, data in enumerate(images_bytes): rec.record(image_bytes=data, input_id=f"img_{i:03d}")
    rec.seal(); return rec


def _load(d): return read_records(Path(d) / "ledger.jsonl")


def _reseal(d, recs, key): return rebuild(d, recs, key)


def _set_output(r, out):
    r["output"] = out; r["output_sha256"] = output_hash(out)


def _rerun(model_path, data, cfg):
    a = load_adapter(model_path); return build_classification_output(a.predict([decode_image(data)])[0], cfg)


def build_part3_benchmark(out_dir, seed=3):
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    p2 = out / "part2_bench"
    if not p2.exists(): build_benchmark(p2)
    M = p2 / "models"; trusted, subst, trojan = M / "trusted_reference.npz", M / "substitute.npz", M / "trojan_checker.npz"
    key_path = out / "ledger.key"; key = generate_key(key_path)
    imgs = make_images(N + 12, seed); data = [_png(i) for i in imgs]
    other = data[N:]                                   # spare images for substitution / replay-output
    trig = [_png(add_trigger(i)) for i in imgs[:N]]    # triggered variants of the same scenes
    a_t = load_adapter(trusted); cfg = default_config(a_t)
    # base valid case
    base = out / "cases" / "valid"; shutil.rmtree(base, ignore_errors=True)
    record_all(base, trusted, data[:N], key, Clock(), config=cfg)
    ident = model_identity(trusted)
    write_registry(base / "registry.json", [{"model_id": ident["model_id"], "sha256": ident["sha256"], "path": str(trusted.resolve())}])
    write_policy(base / "policy.json", cfg)
    gt = {"key": str(key_path), "cases": {"valid": {"attacked_records": [], "attack": "none", "attacker": None, "expect": []}}}

    def make_case(name, attacker, expect, fn, attack_desc):
        d = out / "cases" / name; shutil.rmtree(d, ignore_errors=True); shutil.copytree(base, d)
        info = fn(d)
        gt["cases"][name] = {"attack": attack_desc, "attacker": attacker, "expect": expect, **info}

    ids = lambda idxs: [f"R-{i + 1:06d}" for i in idxs]

    def output_naive(d):
        L = _load(d)
        for i in ATTACK_IDX:
            o = L[i]["output"]; o["pred_class_id"] = (o["pred_class_id"] + 1) % 4; o["pred_class_name"] = str((o["pred_class_id"]))
        (d / "ledger.jsonl").write_text("".join(json.dumps(r) + "\n" for r in L)); return {"attacked_records": ids(ATTACK_IDX)}

    def output_insider(d):
        L = _load(d)
        for i in ATTACK_IDX:
            o = copy.deepcopy(L[i]["output"]); tk = o["top_k"]
            tk[0], tk[1] = tk[1], tk[0]                                   # swap top-2 so it looks plausible
            o["pred_class_id"], o["pred_class_name"], o["confidence"] = tk[0]["class_id"], tk[0]["class_name"], tk[0]["prob"]
            o["decision"] = o["pred_class_name"] if o["confidence"] >= cfg["decision_threshold"] else "REJECT"
            _set_output(L[i], o)
        _reseal(d, L, key); return {"attacked_records": ids(ATTACK_IDX)}

    def input_naive(d):
        L = _load(d)
        for k, i in enumerate(ATTACK_IDX):
            (d / L[i]["input"]["stored_as"]).write_bytes(other[k])          # swap file, record untouched
        return {"attacked_records": ids(ATTACK_IDX)}

    def input_insider(d):
        L = _load(d)
        for k, i in enumerate(ATTACK_IDX):
            sha = sha256_bytes(other[k]); (d / f"inputs/{sha}.png").write_bytes(other[k])
            L[i]["input_sha256"] = sha; L[i]["input"]["sha256"] = sha; L[i]["input"]["stored_as"] = f"inputs/{sha}.png"
        _reseal(d, L, key); return {"attacked_records": ids(ATTACK_IDX)}

    def model_hidden(d):
        L = _load(d)
        for i in ATTACK_IDX:
            raw = (d / L[i]["input"]["stored_as"]).read_bytes(); _set_output(L[i], _rerun(subst, raw, cfg))   # substitute ran, claims trusted
        _reseal(d, L, key); return {"attacked_records": ids(ATTACK_IDX)}

    def model_declared(d):
        L = _load(d); si = model_identity(subst)
        for i in ATTACK_IDX:
            raw = (d / L[i]["input"]["stored_as"]).read_bytes(); _set_output(L[i], _rerun(subst, raw, cfg))
            L[i]["model"] = si; L[i]["model_sha256"] = si["sha256"]
        _reseal(d, L, key); return {"attacked_records": ids(ATTACK_IDX)}

    TROJ_IDX = [5, 10, 15, 21]                                              # non-target-class scenes (target class = 0)

    def trojan_trig(d):
        L = _load(d)
        for k, i in enumerate(TROJ_IDX):
            raw = trig[i]; sha = sha256_bytes(raw); (d / f"inputs/{sha}.png").write_bytes(raw)
            L[i]["input_sha256"] = sha; L[i]["input"].update(sha256=sha, stored_as=f"inputs/{sha}.png")
            _set_output(L[i], _rerun(trojan, raw, cfg))                      # trojan output (steered), claims trusted model
        _reseal(d, L, key); return {"attacked_records": ids(TROJ_IDX)}

    def cfg_consistent(d):
        L = _load(d); bad = dict(cfg, decision_threshold=0.0)
        for i in ATTACK_IDX:
            raw = (d / L[i]["input"]["stored_as"]).read_bytes()
            L[i]["config"] = bad; L[i]["config_sha256"] = sha256_json(bad); _set_output(L[i], _rerun(trusted, raw, bad))
        _reseal(d, L, key); return {"attacked_records": ids(ATTACK_IDX)}

    def cfg_inconsistent(d):
        L = _load(d); bad = dict(cfg, top_k=1)
        for i in ATTACK_IDX: L[i]["config"] = bad; L[i]["config_sha256"] = sha256_json(bad)     # output left as is
        _reseal(d, L, key); return {"attacked_records": ids(ATTACK_IDX)}

    def replay_exact(d):
        L = _load(d); new = []
        for k, i in enumerate([2, 7, 12, 17]):
            c = copy.deepcopy(L[i]); c["seq"] = len(L) + k + 1; c["record_id"] = f"R-{c['seq']:06d}"
            c["recorded_at"] = (T0 + timedelta(minutes=30 + k)).isoformat(timespec="milliseconds"); new.append(c)   # old body, fresh ledger position
        _reseal(d, L + new, key); return {"attacked_records": [c["record_id"] for c in new]}

    def replay_output(d):
        L = _load(d)
        for k, i in enumerate(ATTACK_IDX):                                   # new input, but an OLD result is attached
            sha = sha256_bytes(other[k]); (d / f"inputs/{sha}.png").write_bytes(other[k])
            L[i]["input_sha256"] = sha; L[i]["input"].update(sha256=sha, stored_as=f"inputs/{sha}.png")
            _set_output(L[i], copy.deepcopy(L[(i + 1) % N]["output"]))
        _reseal(d, L, key); return {"attacked_records": ids(ATTACK_IDX)}

    def deletion(d):
        L = _load(d); gone = [8, 9, 20]
        _reseal(d, [r for j, r in enumerate(L) if j not in gone], key)
        return {"attacked_records": [], "deleted_seqs": [g + 1 for g in gone]}

    make_case("output_tampering_naive", "naive", ["output_hash_mismatch", "record_hash_mismatch"], output_naive, "stored output edited, nothing re-sealed")
    make_case("output_tampering_insider", "insider", ["reinference_mismatch"], output_insider, "top-2 swapped, output hash recomputed, ledger re-signed")
    make_case("input_substitution_naive", "naive", ["input_hash_mismatch"], input_naive, "stored input bytes replaced by another image")
    make_case("input_substitution_insider", "insider", ["reinference_mismatch"], input_insider, "input replaced AND input hash + ledger re-signed")
    make_case("model_substitution_hidden", "insider", ["reinference_mismatch"], model_hidden, "different model produced outputs; record claims trusted digest")
    make_case("model_substitution_declared", "insider", ["model_not_registered"], model_declared, "record honestly names an unregistered substitute model")
    make_case("trojan_triggered_input", "insider", ["reinference_mismatch"], trojan_trig, "trojaned model + triggered inputs; record claims trusted digest")
    make_case("config_tampering_consistent", "insider", ["config_policy_mismatch"], cfg_consistent, "decision threshold lowered to 0, outputs consistently recomputed")
    make_case("config_tampering_inconsistent", "insider", ["config_policy_mismatch", "reinference_mismatch", "output_internally_inconsistent"], cfg_inconsistent, "config changed, output left")
    make_case("replay_exact", "insider", ["replay_request_id_reuse", "replay_nonce_reuse", "stale_record"], replay_exact, "old records re-appended at new ledger positions")
    make_case("replay_output_reuse", "insider", ["reinference_mismatch", "output_reuse_across_inputs"], replay_output, "new input paired with an old output")
    make_case("record_deletion", "insider", ["sequence_gap"], deletion, "3 records removed, ledger re-chained and re-signed")
    (out / "ground_truth.json").write_text(json.dumps(gt, indent=2))
    return gt


def evaluate_part3(bench_dir, out_dir, log=print) -> dict:
    b = Path(bench_dir); gt = json.loads((b / "ground_truth.json").read_text()); key = load_key(gt["key"])
    out = Path(out_dir); rows = []; fp = fn = 0
    tp = 0; clean_flagged = clean_total = 0
    for name, c in gt["cases"].items():
        d = b / "cases" / name
        res = verify_ledger(d, out / name, key=key, log=lambda *a: None)
        flagged = {f["asset_id"] for f in res["findings"] if f["disposition"] in ("REVIEW", "QUARANTINE") and f["asset_id"] != "LEDGER"}
        att = set(c.get("attacked_records", []))
        cats = {f["category"] for f in res["findings"] if f["disposition"] in ("REVIEW", "QUARANTINE") and (f["asset_id"] in att or f["asset_id"] == "LEDGER")}
        t, f_, n_ = len(att & flagged), len(flagged - att), len(att - flagged)
        ledger_hit = None
        if c.get("deleted_seqs"):
            miss = {s for f in res["findings"] if f["category"] == "sequence_gap" for s in f["evidence"]["missing_seqs"]}
            ledger_hit = set(c["deleted_seqs"]) <= miss
        det = (n_ == 0 and t > 0) if att else bool(ledger_hit) if ledger_hit is not None else (res["disposition"] == "ACCEPT")
        tp += t; fp += f_; fn += n_
        clean_total += res["records_total"] - len(att); clean_flagged += f_
        # without the signing key (content checks + hash chain only): does the insider attack STILL get caught?
        r_nokey = verify_ledger(d, out / (name + "__nokey"), key=None, log=lambda *a: None)
        fl2 = {f["asset_id"] for f in r_nokey["findings"] if f["disposition"] in ("REVIEW", "QUARANTINE") and f["asset_id"] != "LEDGER"}
        rows.append({"case": name, "attacker": c["attacker"], "attacked": len(att) or len(c.get("deleted_seqs", [])), "detected_full": t if att else (len(c["deleted_seqs"]) if ledger_hit else 0),
                     "false_positives": f_, "case_detected": bool(det), "categories": sorted(cats), "expected_any_of": c["expect"],
                     "detected_by_content_checks_only": len(att & fl2) if att else None, "disposition": res["disposition"]})
    prec = tp / (tp + fp) if tp + fp else 1.0; rec = tp / (tp + fn) if tp + fn else 1.0
    m = {"cases": rows, "record_level": {"tp": tp, "fp": fp, "fn": fn, "precision": round(prec, 3), "recall": round(rec, 3),
                                         "f1": round(2 * prec * rec / (prec + rec), 3) if prec + rec else 0},
         "clean_record_false_positive_rate": round(clean_flagged / clean_total, 4) if clean_total else 0,
         "valid_case_disposition": next(r["disposition"] for r in rows if r["case"] == "valid")}
    (out / "evaluation_part3.json").write_text(json.dumps(m, indent=2)); return m


def format_eval3(m):
    L = [f"{'case':32s} {'attacker':8s} {'attacked':>8s} {'detected':>8s} {'FP':>3s} {'no-key':>6s}  categories"]
    for r in m["cases"]:
        nk = "-" if r["detected_by_content_checks_only"] is None else str(r["detected_by_content_checks_only"])
        L.append(f"{r['case']:32s} {str(r['attacker']):8s} {r['attacked']:8d} {r['detected_full']:8d} {r['false_positives']:3d} {nk:>6s}  {','.join(r['categories'])[:60]}")
    q = m["record_level"]
    L.append(f"\nrecord-level precision={q['precision']} recall={q['recall']} F1={q['f1']}   clean-record FPR={m['clean_record_false_positive_rate']}   valid ledger -> {m['valid_case_disposition']}")
    return "\n".join(L)
