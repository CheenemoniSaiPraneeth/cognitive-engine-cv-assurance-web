"""Cryptographic helpers: SHA-256 of bytes/files and of canonical JSON."""
from __future__ import annotations

import hashlib
import json
from typing import Any


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical_json(obj: Any) -> str:
    """Deterministic serialisation: sorted keys, fixed separators, floats rounded to 8 dp."""
    def _norm(o):
        if isinstance(o, float):
            return round(o, 8)
        if isinstance(o, dict):
            return {str(k): _norm(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [_norm(v) for v in o]
        if isinstance(o, set):
            return sorted(_norm(v) for v in o)
        return o
    return json.dumps(_norm(obj), sort_keys=True, separators=(",", ":"), default=str)


def sha256_json(obj: Any) -> str:
    return sha256_bytes(canonical_json(obj).encode("utf-8"))
