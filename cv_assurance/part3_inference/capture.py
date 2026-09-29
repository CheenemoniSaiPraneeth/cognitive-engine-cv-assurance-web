"""Modules 1-4 - capture input, model identity, environment/config and canonical output; append to the ledger."""
from __future__ import annotations

import io
import json
import os
import platform
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from ..common.hashing import sha256_bytes, sha256_file, sha256_json
from ..common.schemas import now_iso
from ..part2_model.adapters import Unavailable, load_adapter
from ..part2_model.fingerprint import fingerprint_model
from .canonical import build_classification_output, output_hash
from .ledger import Ledger

PIL_FMT = {"PNG": "png", "JPEG": "jpg", "BMP": "bmp", "WEBP": "webp"}


def decode_image(data: bytes) -> np.ndarray:
    return np.asarray(Image.open(io.BytesIO(data)).convert("RGB"))


def default_config(adapter, top_k=3, decision_threshold=0.5) -> dict:
    """Inference configuration that is hashed into the record (preprocess + postprocess + determinism settings)."""
    return {"task": "classification", "preprocess": adapter.input_schema(), "class_names": adapter.class_names() or [],
            "top_k": top_k, "decision_threshold": decision_threshold, "probability_decimals": 4, "nms": None,
            "seed": 0, "precision": "float32"}


def environment(adapter) -> dict:
    import PIL
    return {"python": sys.version.split()[0], "numpy": np.__version__, "pillow": PIL.__version__, "platform": platform.platform(),
            "machine": platform.machine(), "adapter_format": adapter.fmt, "runtime": adapter.metadata().get("runtime"), "device": "cpu",
            "network": "not used"}


def model_identity(model_path, adapter=None, expected_sha=None) -> dict:
    a = adapter or load_adapter(model_path)
    fp = fingerprint_model(a, expected_sha)
    return {"model_id": "MODEL-" + Path(model_path).stem, "sha256": fp["sha256"], "format": fp["format"],
            "architecture": fp.get("architecture"), "parameter_count": fp.get("parameter_count"),
            "architecture_signature": fp.get("architecture_signature"), "runtime": fp.get("runtime")}


class InferenceRecorder:
    """Records inferences. `adapter` executes the model. `identity_path`/`identity_adapter` is what the record CLAIMS
    (defaults to the executing model). Setting them differently is how benchmark fixtures simulate model substitution."""

    def __init__(self, ledger_dir, model_path, key=None, config=None, clock=None, adapter=None,
                 identity_path=None, source="prototype-camera"):
        self.dir = Path(ledger_dir); (self.dir / "inputs").mkdir(parents=True, exist_ok=True)
        self.adapter = adapter or load_adapter(model_path)
        self.identity_path = Path(identity_path or model_path)
        self.identity = model_identity(self.identity_path)
        self.config = config or default_config(self.adapter)
        self.config_sha = sha256_json(self.config)
        self.env = environment(self.adapter)
        self.clock = clock or now_iso
        self.ledger = Ledger(self.dir, key, clock=self.clock)
        self.source = source; self.n = 0

    def record(self, image_path=None, image_bytes=None, input_id=None, links=None, source=None) -> dict:
        data = image_bytes if image_bytes is not None else Path(image_path).read_bytes()
        im = Image.open(io.BytesIO(data)); fmt = PIL_FMT.get(im.format, "bin")
        in_sha = sha256_bytes(data)
        stored = f"inputs/{in_sha}.{fmt}"
        (self.dir / stored).write_bytes(data)                     # preserve exact bytes, not just a path
        captured_at = self.clock()
        probs = self.adapter.predict([decode_image(data)])[0]
        out = build_classification_output(probs, self.config)
        self.n += 1
        body = {"request_id": sha256_bytes(f"{self.n}:{in_sha}:{os.urandom(8).hex()}".encode())[:24], "nonce": os.urandom(8).hex(),
                "input_id": input_id or (Path(image_path).name if image_path else in_sha[:12]),
                "input": {"sha256": in_sha, "width": im.width, "height": im.height, "format": fmt, "source": source or self.source,
                          "captured_at": captured_at, "stored_as": stored, "size_bytes": len(data)},
                "input_sha256": in_sha, "model": self.identity, "model_sha256": self.identity["sha256"],
                "config": self.config, "config_sha256": self.config_sha, "runtime": self.env,
                "output": out, "output_sha256": output_hash(out), "inference_timestamp": self.clock(), "links": links or {}}
        return self.ledger.append(body)

    def seal(self): return self.ledger.seal()


def write_registry(path, entries: list):
    """Trusted registry: approved model digests (+ where to find the artifact for re-inference)."""
    reg = {"models": {e["sha256"]: {"model_id": e["model_id"], "path": str(e["path"]), "approved": e.get("approved", True)} for e in entries}}
    Path(path).write_text(json.dumps(reg, indent=2)); return reg


def write_policy(path, config: dict, max_record_lag_s=300):
    pol = {"expected_config": config, "expected_config_sha256": sha256_json(config), "max_record_lag_s": max_record_lag_s,
           "compare_tolerance": 1e-3}
    Path(path).write_text(json.dumps(pol, indent=2)); return pol
