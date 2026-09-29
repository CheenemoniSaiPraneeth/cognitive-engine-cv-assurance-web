"""Append-only, hash-linked audit log (AuditEvent from the architecture document)."""
from __future__ import annotations

import json
from pathlib import Path

from .hashing import sha256_json
from .schemas import DETECTOR_VERSION, now_iso


class AuditLog:
    def __init__(self, path) -> None:
        self.path = Path(path)
        self.events: list[dict] = []
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text("")

    def append(self, action: str, input_asset_ids=None, output_asset_ids=None,
               configuration_hash: str = "", result: str = "ok", actor: str = "part1_pipeline") -> dict:
        prev = self.events[-1]["event_hash"] if self.events else "0" * 64
        ev = {
            "event_id": f"E-{len(self.events) + 1:04d}",
            "timestamp": now_iso(),
            "actor": actor,
            "action": action,
            "input_asset_ids": input_asset_ids or [],
            "output_asset_ids": output_asset_ids or [],
            "detector_version": DETECTOR_VERSION,
            "configuration_hash": configuration_hash,
            "result": result,
            "previous_event_hash": prev,
        }
        ev["event_hash"] = sha256_json(ev)
        self.events.append(ev)
        with open(self.path, "a") as fh:
            fh.write(json.dumps(ev) + "\n")
        return ev

    @staticmethod
    def verify(path) -> bool:
        prev = "0" * 64
        for line in Path(path).read_text().splitlines():
            ev = json.loads(line)
            h = ev.pop("event_hash")
            if ev["previous_event_hash"] != prev or sha256_json(ev) != h:
                return False
            prev = h
        return True
