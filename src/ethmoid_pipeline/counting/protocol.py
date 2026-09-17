from __future__ import annotations

import hashlib
import json
from importlib.resources import files
from typing import Any


PROTOCOL_VERSION = "ethmoid-count-prototype-v1"


def load_counting_protocol() -> dict[str, Any]:
    """Return a fresh copy of the experimental contract and its exact byte hash."""
    content = files("ethmoid_pipeline.counting").joinpath("protocol.json").read_bytes()
    result = json.loads(content)
    if result.get("protocol_version") != PROTOCOL_VERSION:
        raise RuntimeError("Counting protocol version mismatch.")
    if result.get("status") != "EXPERIMENTAL_NOT_CLINICALLY_VALIDATED":
        raise RuntimeError("Experimental counting cannot imply clinical approval.")
    result["sha256"] = hashlib.sha256(content).hexdigest()
    return result
