"""Module 5 - Audit Trail: append-only, hash-linked, optionally HMAC-signed (offline key). Persists across sessions."""
from __future__ import annotations

import hashlib
import hmac
import json
import os
from pathlib import Path

from ..common.hashing import sha256_json
from ..common.schemas import now_iso
from .config import P5_VERSION

ZERO = "0" * 64


def load_or_create_key(path) -> bytes:
    p = Path(path)
    if not p.exists():
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(os.urandom(32).hex()); os.chmod(p, 0o600)
    return bytes.fromhex(p.read_text().strip())


class GovAudit:
    def __init__(self, path, key: bytes | None = None):
        self.path, self.key = Path(path), key
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.events = [json.loads(l) for l in self.path.read_text().splitlines()] if self.path.exists() else []

    def append(self, action, actor="part5_assembler", input_asset_ids=None, output_asset_ids=None, configuration_hash="", result="ok", detail=None) -> dict:
        ev = {"event_id": f"E5-{len(self.events) + 1:05d}", "timestamp": now_iso(), "actor": actor, "action": action,
              "input_asset_ids": input_asset_ids or [], "output_asset_ids": output_asset_ids or [], "detector_version": P5_VERSION,
              "configuration_hash": configuration_hash, "result": result, "detail": detail or {},
              "previous_event_hash": self.events[-1]["event_hash"] if self.events else ZERO}
        ev["event_hash"] = sha256_json(ev)
        if self.key:
            ev["signature"] = hmac.new(self.key, ev["event_hash"].encode(), hashlib.sha256).hexdigest()
        self.events.append(ev)
        with open(self.path, "a") as fh:
            fh.write(json.dumps(ev) + "\n")
        return ev

    @staticmethod
    def verify(path, key: bytes | None = None) -> dict:
        p = Path(path)
        if not p.exists():
            return {"ok": False, "events": 0, "reason": "audit log missing", "head": None, "signed": False}
        prev, n, signed = ZERO, 0, 0
        for i, line in enumerate(p.read_text().splitlines(), 1):
            ev = json.loads(line); sig = ev.pop("signature", None); h = ev.pop("event_hash")
            if ev["previous_event_hash"] != prev:
                return {"ok": False, "events": n, "reason": f"chain break at event {i} ({ev['event_id']})", "head": prev, "signed": bool(signed)}
            ev2 = dict(ev)
            if sha256_json(ev2) != h:
                return {"ok": False, "events": n, "reason": f"content modified at event {i} ({ev['event_id']})", "head": prev, "signed": bool(signed)}
            if sig is not None and key is not None:
                if not hmac.compare_digest(sig, hmac.new(key, h.encode(), hashlib.sha256).hexdigest()):
                    return {"ok": False, "events": n, "reason": f"bad signature at event {i}", "head": prev, "signed": True}
                signed += 1
            prev, n = h, n + 1
        return {"ok": True, "events": n, "reason": "chain intact" + (f"; {signed} HMAC signatures valid" if signed else "; unsigned (no key supplied)"), "head": prev, "signed": bool(signed)}
