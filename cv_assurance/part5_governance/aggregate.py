"""Module 1 - Evidence Aggregator.

Discovers Part 1-4 assurance JSON, normalises every finding into ONE schema (namespaced ids so P-00001 from two ledgers
never collide), preserves the raw evidence + artifact hash, converts unavailable tests into explicit UNAVAILABLE items,
and extracts entity links (dataset / source / model / ledger / stream / record) used for traceability + correlation.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

from ..common.hashing import sha256_file

PART_FILES = {"assurance_part1.json": 1, "assurance_part2.json": 2, "assurance_part3.json": 3, "assurance_part4.json": 4}
SKIP_DIRS = {"benchmark", "dataset", "synthetic_dataset", "streams_in", "ledger", "part3_ledger", "part4_reference", "__pycache__"}
SEV_ORDER = {"low": 1, "medium": 2, "high": 3, "critical": 4}


@dataclass
class Artifact:
    part: int
    path: str
    sha256: str
    data: dict
    scope: str


def _scope(part: int, path: Path, data: dict) -> str:
    ps = path.parts
    if part == 4:
        return (data.get("links") or {}).get("stream") or (ps[ps.index("streams") + 1] if "streams" in ps else "operational_window")
    if part == 3:
        if "streams" in ps:
            return ps[ps.index("streams") + 1]
        par = path.parent
        name = par.parent.name if par.name == "part3" else par.name
        return "dataset_sample_ledger" if name in ("p123", "") else name
    if part == 2:
        return data.get("model_id") or path.parent.name
    return (data.get("dataset_manifest") or {}).get("dataset_id") or path.parent.name


def discover(paths) -> list:
    files = []
    for p in map(Path, paths):
        if p.is_file():
            files.append(p)
            continue
        for root, dirs, fs in os.walk(p):
            dirs[:] = sorted(d for d in dirs if d not in SKIP_DIRS)
            files += [Path(root) / f for f in sorted(fs) if f in PART_FILES]
    arts, used = [], {}
    for f in files:
        data = json.loads(f.read_text())
        part = PART_FILES.get(f.name) or int(str(data.get("part", "0"))[0])
        if part not in (1, 2, 3, 4):
            continue
        scope = _scope(part, f, data)
        n = used.get((part, scope), 0) + 1; used[(part, scope)] = n
        if n > 1:
            scope = f"{scope}#{n}"
        arts.append(Artifact(part, str(f), sha256_file(f), data, scope))
    arts.sort(key=lambda a: (a.part, a.scope))
    return arts


def _add(ents: list, kind: str, val) -> None:
    if val not in (None, "", "None"):
        e = f"{kind}:{val}"
        if e not in ents:
            ents.append(e)


def _entities(art: Artifact, f: dict, ctx: dict) -> list:
    ents, d, ev = [], art.data, f.get("evidence") or {}
    aid, at = str(f.get("asset_id", "")), f.get("asset_type", "")
    if art.part == 1:
        _add(ents, "dataset", (d.get("dataset_manifest") or {}).get("dataset_id"))
        if ":" in aid and at == "source":
            k, v = aid.split(":", 1); _add(ents, "batch" if k == "batch_id" else "source", v)
        elif at == "sample":
            _add(ents, "sample", aid)
        elif at in ("group", "pattern"):
            _add(ents, at, aid)
        _add(ents, "source", ev.get("top_source"))
        sc = ev.get("source_correlation")
        if isinstance(sc, dict) and (sc.get("fraction") or 0) >= 0.5:
            _add(ents, "source", sc.get("value"))
    elif art.part == 2:
        _add(ents, "model", d.get("model_id") or aid)
    elif art.part == 3:
        _add(ents, "ledger", art.scope)
        if art.scope in ctx["stream_names"]:
            _add(ents, "stream", art.scope)
        if aid.startswith("R-"):
            _add(ents, "record", f"{art.scope}/{aid}")
        _add(ents, "model", ctx.get("single_model"))
    elif art.part == 4:
        lk = d.get("links") or {}
        _add(ents, "stream", art.scope); _add(ents, "ledger", art.scope); _add(ents, "dataset", lk.get("dataset_id"))
        _add(ents, "window", f"{art.scope}/{aid}" if at != "window" else art.scope)
        _add(ents, "source", ev.get("source"))
        _add(ents, "model", ctx.get("single_model"))
    return ents


def normalize(arts: list) -> dict:
    ctx = {"stream_names": {a.scope for a in arts if a.part == 4}}
    models = [a.data.get("model_id") for a in arts if a.part == 2]
    ctx["single_model"] = models[0] if len(models) == 1 else None
    findings, unavailable, parts = [], [], {}
    for a in arts:
        d = a.data
        ps = parts.setdefault(a.part, {"artifacts": [], "dispositions": [], "levels": [], "finding_counts": {}, "coverage": []})
        ps["artifacts"].append({"scope": a.scope, "path": a.path, "sha256": a.sha256, "detector_version": d.get("detector_version"), "created_at": d.get("created_at")})
        ps["dispositions"].append(d.get("disposition", "ACCEPT"))
        if d.get("level"):
            ps["levels"].append(d["level"])
        if d.get("coverage"):
            ps["coverage"].append(d["coverage"])
        for k, v in (d.get("finding_counts") or {}).items():
            ps["finding_counts"][k] = ps["finding_counts"].get(k, 0) + v
        for f in d.get("findings", []):
            findings.append({
                "uid": f"P{a.part}:{a.scope}:{f.get('finding_id')}", "part": a.part, "scope": a.scope, "source_finding_id": f.get("finding_id"),
                "asset_type": f.get("asset_type"), "asset_id": f.get("asset_id"), "category": f.get("category"),
                "severity": f.get("severity", "low"), "confidence": float(f.get("confidence", 0.5)), "detector": f.get("detector"),
                "detector_version": f.get("detector_version") or d.get("detector_version"), "evidence": f.get("evidence") or {},
                "reference_assets": f.get("reference_assets") or [], "timestamp": f.get("timestamp"), "disposition": f.get("disposition", "REVIEW"),
                "limitations": f.get("limitations") or [], "entities": _entities(a, f, ctx), "artifact_sha256": a.sha256,
            })
        for i, u in enumerate(d.get("unavailable_tests") or []):
            text = u if isinstance(u, str) else json.dumps(u, default=str)
            item = {"uid": f"P{a.part}:{a.scope}:U-{i + 1:02d}", "part": a.part, "scope": a.scope, "test": text,
                    "impact": "test not executed - this is NOT a PASS for that capability", "artifact_sha256": a.sha256}
            unavailable.append(item)
            findings.append({**{k: None for k in ("asset_type", "asset_id", "detector", "detector_version", "timestamp", "source_finding_id")},
                             "uid": item["uid"], "part": a.part, "scope": a.scope, "category": "test_unavailable", "severity": "low", "confidence": 1.0,
                             "evidence": {"test": text}, "reference_assets": [], "disposition": "UNAVAILABLE",
                             "limitations": [item["impact"]], "entities": [f"ledger:{a.scope}" if a.part in (3, 4) else f"part:{a.part}"], "artifact_sha256": a.sha256})
    return {"findings": findings, "unavailable": unavailable, "parts": parts, "single_model": ctx["single_model"]}


def source_summaries(arts: list) -> list:
    """Part 1 per-source/batch summaries, de-duplicated (source_id and contributor_id often describe the same samples)."""
    out, seen = [], set()
    for a in arts:
        if a.part != 1:
            continue
        for key, s in (a.data.get("source_summaries") or {}).items():
            kind, name = key.split(":", 1)
            sig = (name if kind != "batch_id" else None, s.get("sample_count"), tuple(s.get("elevated_signals") or []), s.get("cross_signal_count"))
            if kind == "batch_id" and any(o["sample_count"] == s.get("sample_count") and o["elevated_signals"] == (s.get("elevated_signals") or []) for o in out):
                continue
            if name in seen and kind != "batch_id":
                continue
            seen.add(name)
            out.append({"kind": kind, "name": name, "dataset": (a.data.get("dataset_manifest") or {}).get("dataset_id"), **{k: s.get(k) for k in
                       ("sample_count", "duplicate_rate", "label_anomaly_rate", "ood_rate", "trigger_candidate_rate", "structural_error_rate", "anomaly_categories", "cross_signal_count", "disposition")},
                        "elevated_signals": s.get("elevated_signals") or []})
    return out
