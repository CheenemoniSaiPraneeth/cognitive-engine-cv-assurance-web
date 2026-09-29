"""Parts 1-5 integration: run the existing Part 1-4 integrated pipeline, then hand its artifacts to Part 5.
Part 5 only READS Part 1-4 outputs (hashed into the audit trail) - it never re-runs or alters them."""
from __future__ import annotations

import json
from pathlib import Path

from .assemble import assemble


def run_integrated5(dataset_dir, served_model, trusted_model, out_dir, streams, fmt="auto", policy=None, log=print):
    from ..part4_shift.integration import run_integrated4          # lazy: Part 5 stand-alone bundle has no Part 1-4 code
    out = Path(out_dir); run_integrated4(dataset_dir, served_model, trusted_model, out / "p1234", streams, fmt=fmt, log=log)
    return assemble([out / "p1234"], out / "part5", policy, actor="integration_p12345", log=log)


def integrated_demo(out_dir, n_clean=240, log=print):
    from ..benchmark.part2_models import build_benchmark
    from ..benchmark.part4_bench import make_batch
    from ..benchmark.synthetic import generate
    out = Path(out_dir); build_benchmark(out / "benchmark"); generate(out / "dataset", n_clean, "coco")
    M = out / "benchmark" / "models"; trusted = M / "trusted_reference.npz"; streams = []
    for name, kinds in (("stable_stream", ["none"] * 4), ("shifted_stream", ["none", "none", "camera", "illumination"])):
        d = out / "streams_in" / name; meta = {}
        for b, k in enumerate(kinds):
            meta.update(make_batch(d, f"b{b}", 40, k, 500 + b + len(name), source="contributor_z" if k == "camera" else "contributor_a", ts0=b))
        (d / "metadata.json").write_text(json.dumps(meta)); streams.append((name, d))
    res = {}
    for scen, served in (("trusted_model_served", "trusted_reference.npz"), ("trojan_model_served", "trojan_checker.npz")):
        log(f"\n=== scenario: {scen} ===")
        res[scen] = run_integrated5(out / "dataset", M / served, trusted, out / scen, streams, log=log)
    for scen, f in res.items():
        log(f"[p5] {scen}: system={f['overall']['system_disposition']} cases={len(f['cases'])} -> {out / scen / 'part5' / 'report_part5.html'}")
    return res
