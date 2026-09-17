from __future__ import annotations

import hashlib
import json
from importlib.resources import files
from typing import Any


PROTOCOL_VERSION = "ethmoid-count-prototype-v2"
SUPPORTED_PROTOCOL_VERSIONS = {"ethmoid-count-prototype-v1", PROTOCOL_VERSION}


def load_counting_protocol(version: str = PROTOCOL_VERSION) -> dict[str, Any]:
    """Return a fresh copy of the experimental contract and its exact byte hash."""
    if version not in SUPPORTED_PROTOCOL_VERSIONS:
        raise ValueError("Unsupported counting protocol version.")
    filename = "protocol.v1.json" if version == "ethmoid-count-prototype-v1" else "protocol.json"
    content = files("ethmoid_pipeline.counting").joinpath(filename).read_bytes()
    result = json.loads(content)
    if result.get("protocol_version") != version:
        raise RuntimeError("Counting protocol version mismatch.")
    if result.get("status") != "EXPERIMENTAL_NOT_CLINICALLY_VALIDATED":
        raise RuntimeError("Experimental counting cannot imply clinical approval.")
    result["sha256"] = hashlib.sha256(content).hexdigest()
    return result
