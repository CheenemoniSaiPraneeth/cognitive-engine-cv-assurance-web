from ..common.schemas import FindingStore, Finding, now_iso
from . import PART2_VERSION

SEV_RANK = {"info": 0, "low": 1, "medium": 2, "high": 3, "critical": 4}


class Part2FindingStore(FindingStore):
    """Same Finding schema as Part 1; ids are M-xxxxx so they never collide when merged."""
    def add(self, asset_type, asset_id, category, severity, confidence, detector, evidence, disposition,
            limitations=None, reference_assets=None):
        f = Finding(finding_id=f"M-{len(self.findings) + 1:05d}", asset_type=asset_type, asset_id=asset_id,
                    category=category, severity=severity, confidence=round(float(confidence), 3), detector=detector,
                    detector_version=PART2_VERSION, evidence=evidence, reference_assets=reference_assets or [],
                    timestamp=now_iso(), disposition=disposition, limitations=limitations or [])
        self.findings.append(f)
        return f
