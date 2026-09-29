"""Module 5 - provenance chain: append-only, hash-linked, HMAC-signed inference ledger (+ signed batch manifest).
Signing key is an offline-managed secret (HMAC-SHA256). Without a key the chain still detects casual edits but an
attacker who can recompute hashes could re-chain; the signature is what stops that (see README threat table)."""
from __future__ import annotations

import hashlib
import hmac
import json
import secrets
from pathlib import Path

from ..common.hashing import sha256_json
from ..common.schemas import now_iso

GENESIS = "0" * 64


def generate_key(path) -> bytes:
    k = secrets.token_bytes(32); Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(k.hex()); return k


def load_key(path):
    p = Path(path) if path else None
    return bytes.fromhex(p.read_text().strip()) if p and p.exists() else None


def body_hash(rec: dict) -> str:
    return sha256_json({k: v for k, v in rec.items() if k not in ("record_hash", "signature")})


def sign(key: bytes, h: str) -> str:
    return hmac.new(key, h.encode(), hashlib.sha256).hexdigest()


def read_records(path) -> list:
    p = Path(path)
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()] if p.exists() else []


def write_manifest(path, records, key=None, clock=None):
    m = {"record_count": len(records), "genesis": GENESIS, "head_hash": records[-1]["record_hash"] if records else GENESIS,
         "first_record_id": records[0]["record_id"] if records else None, "last_record_id": records[-1]["record_id"] if records else None,
         "created_at": (clock() if clock else now_iso())}
    m["manifest_hash"] = sha256_json(m)
    m["signature"] = sign(key, m["manifest_hash"]) if key else None
    Path(path).write_text(json.dumps(m, indent=2)); return m


class Ledger:
    def __init__(self, directory, key: bytes | None = None, clock=None):
        self.dir = Path(directory); self.dir.mkdir(parents=True, exist_ok=True)
        self.path, self.key, self.clock = self.dir / "ledger.jsonl", key, clock
        self.records = read_records(self.path)

    def append(self, body: dict) -> dict:
        rec = dict(body)
        rec["seq"] = len(self.records) + 1
        rec["record_id"] = f"R-{rec['seq']:06d}"
        rec["recorded_at"] = self.clock() if self.clock else now_iso()
        rec["previous_record_hash"] = self.records[-1]["record_hash"] if self.records else GENESIS
        rec["record_hash"] = body_hash(rec)
        rec["signature"] = sign(self.key, rec["record_hash"]) if self.key else None
        self.records.append(rec)
        with open(self.path, "a") as fh: fh.write(json.dumps(rec) + "\n")
        return rec

    def seal(self):
        return write_manifest(self.dir / "manifest.json", self.records, self.key, self.clock)


def rebuild(directory, records: list, key: bytes | None):
    """Recompute chain hashes + signatures for a (possibly modified) record list. Used by the benchmark to model an
    INSIDER who holds the key; also a legitimate tool for re-sealing after migration."""
    prev, out = GENESIS, []
    for r in records:
        r = dict(r); r["previous_record_hash"] = prev; r["record_hash"] = body_hash(r)
        r["signature"] = sign(key, r["record_hash"]) if key else None
        prev = r["record_hash"]; out.append(r)
    d = Path(directory)
    (d / "ledger.jsonl").write_text("".join(json.dumps(r) + "\n" for r in out))
    write_manifest(d / "manifest.json", out, key)
    return out
