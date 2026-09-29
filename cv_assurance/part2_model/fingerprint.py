"""Module 2 - cryptographic + structural fingerprint and expected-vs-received identity comparison."""
from __future__ import annotations

from ..common.hashing import sha256_file, sha256_json
from .adapters import Unavailable


def fingerprint_model(adapter, expected_sha=None, reference_path=None) -> dict:
    fp = {"sha256": sha256_file(adapter.path), "size_bytes": adapter.path.stat().st_size,
          "file_name": adapter.path.name, "white_box": adapter.white_box}
    fp.update(adapter.metadata())
    fp["input_schema"], fp["output_schema"] = adapter.input_schema(), adapter.output_schema()
    try:
        L = adapter.layers()
        fp["layer_count"] = len(L)
        fp["architecture_signature"] = sha256_json([[l["type"], l["shape"]] for l in L])
        fp["tensor_shapes"] = {k: list(v.shape) for k, v in adapter.parameters().items()}
    except Unavailable as e:
        fp["architecture_signature"] = None
        fp["architecture_signature_note"] = f"UNAVAILABLE: {e}"
    ref_sha = sha256_file(reference_path) if reference_path else None
    exp = expected_sha or ref_sha
    fp["expected_sha256"] = exp
    fp["expected_source"] = "supplied digest" if expected_sha else ("trusted reference artifact" if ref_sha else None)
    fp["identity_status"] = "NO_EXPECTED_DIGEST" if exp is None else ("MATCH" if exp == fp["sha256"] else "MISMATCH")
    return fp
