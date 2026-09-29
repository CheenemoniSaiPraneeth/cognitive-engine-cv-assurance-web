"""Shared schemas: SampleRecord, Annotation, Finding (matches the architecture document)."""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any, Optional

DETECTOR_VERSION = "part1-0.1.0"

SEVERITIES = ("low", "medium", "high", "critical")
DISPOSITIONS = ("ACCEPT", "REVIEW", "QUARANTINE", "UNAVAILABLE")


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class Annotation:
    class_id: int
    class_name: str
    bbox: Optional[list] = None  # [x, y, w, h] in absolute pixels (when image size is known)
    area: float = 0.0
    polygon: Optional[list] = None


@dataclass
class SampleRecord:
    sample_id: str
    file_name: str
    image_path: Optional[str]
    annotations: list = field(default_factory=list)
    source_id: Optional[str] = None
    contributor_id: Optional[str] = None
    batch_id: Optional[str] = None
    width: Optional[int] = None
    height: Optional[int] = None
    format: Optional[str] = None
    image_sha256: Optional[str] = None
    perceptual_hash: Optional[str] = None  # pHash, hex
    difference_hash: Optional[str] = None  # dHash, hex
    embedding_ref: Optional[int] = None    # row in the embedding matrix
    readable: bool = True
    metadata: dict = field(default_factory=dict)

    @property
    def primary_label(self) -> Optional[int]:
        """Class with the largest total annotated area (ties -> most instances)."""
        if not self.annotations:
            return None
        agg: dict[int, list] = {}
        for a in self.annotations:
            cur = agg.setdefault(a.class_id, [0.0, 0])
            cur[0] += a.area or 0.0
            cur[1] += 1
        return max(agg.items(), key=lambda kv: (kv[1][0], kv[1][1], -kv[0]))[0]

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Finding:
    finding_id: str
    asset_type: str          # sample | group | pattern | source | dataset
    asset_id: str
    category: str
    severity: str
    confidence: float
    detector: str
    detector_version: str
    evidence: dict
    reference_assets: list
    timestamp: str
    disposition: str
    limitations: list

    def to_dict(self) -> dict:
        return asdict(self)


class FindingStore:
    def __init__(self) -> None:
        self.findings: list[Finding] = []

    def add(self, asset_type: str, asset_id: str, category: str, severity: str,
            confidence: float, detector: str, evidence: dict, disposition: str,
            limitations: Optional[list] = None, reference_assets: Optional[list] = None) -> Finding:
        f = Finding(
            finding_id=f"F-{len(self.findings) + 1:05d}",
            asset_type=asset_type, asset_id=asset_id, category=category,
            severity=severity, confidence=round(float(confidence), 3), detector=detector,
            detector_version=DETECTOR_VERSION, evidence=evidence,
            reference_assets=reference_assets or [], timestamp=now_iso(),
            disposition=disposition, limitations=limitations or [],
        )
        self.findings.append(f)
        return f
