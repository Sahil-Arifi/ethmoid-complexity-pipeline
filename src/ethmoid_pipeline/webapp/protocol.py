from __future__ import annotations

import json
from functools import lru_cache
from importlib.resources import files
from typing import Any


@lru_cache(maxsize=1)
def load_protocol() -> dict[str, Any]:
    resource = files(__package__).joinpath("protocol.v1.json")
    value = json.loads(resource.read_text(encoding="utf-8"))
    if value.get("protocol_version") != "ethmoid-provisional-v1":
        raise RuntimeError("The bundled annotation protocol has an unexpected version.")
    labels = value.get("labels")
    if not isinstance(labels, list) or [item.get("value") for item in labels] != list(range(5)):
        raise RuntimeError("The bundled annotation protocol labels are invalid.")
    return value


def allowed_label_values() -> set[int]:
    return {int(item["value"]) for item in load_protocol()["labels"]}


def protocol_version() -> str:
    return str(load_protocol()["protocol_version"])
