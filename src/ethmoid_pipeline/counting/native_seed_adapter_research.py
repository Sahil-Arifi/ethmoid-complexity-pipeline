"""Emit a source-only port of a rejected research candidate; activate nothing."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

FROZEN_V3_SHA256 = "2411b5d9b5220852e9ae26e537e28873a883cff799dd3ba7a7fbe72c32549b5d"
_FREE_MAXIMA = "    maxima = h_maxima(distance.astype(np.float32), cfg.seed_prominence_mm) & free"
_FREE_REPLACEMENT = """    seed_distance = ndi.distance_transform_edt(np.pad(free, 1), sampling=spacing)[1:-1, 1:-1, 1:-1]
    maxima = h_maxima(seed_distance.astype(np.float32), cfg.seed_prominence_mm).astype(bool) & free
    maxima = discard_ridge_markers(maxima, ridge)"""
_AIR_MAXIMA = "air_maxima = h_maxima(air_distance.astype(np.float32), cfg.seed_prominence_mm) & air_lumen"
_AIR_REPLACEMENT = "air_maxima = h_maxima(air_distance.astype(np.float32), cfg.seed_prominence_mm).astype(bool) & air_lumen"
_HELPER_IMPORT = "from ethmoid_pipeline.counting.native_seed_research import discard_ridge_markers\n\n"


class SeedAdapterError(ValueError):
    """Fixed diagnostic without input contents or paths."""


def build_seed_candidate_source(frozen_source: str) -> str:
    """Transform only the exact frozen source, allowing normalized newlines.

    The emitted function still requires its separately frozen historical globals.
    This builder never imports, compiles or executes the supplied source.
    """
    if not isinstance(frozen_source, str):
        raise SeedAdapterError("SEED_SOURCE_MISMATCH")
    source = frozen_source.replace("\r\n", "\n").replace("\r", "\n")
    if hashlib.sha256(source.encode()).hexdigest() != FROZEN_V3_SHA256:
        raise SeedAdapterError("SEED_SOURCE_MISMATCH")
    if source.count(_FREE_MAXIMA) != 1 or source.count(_AIR_MAXIMA) != 1:
        raise SeedAdapterError("SEED_SOURCE_ANCHOR_MISMATCH")
    body = source.replace(_FREE_MAXIMA, _FREE_REPLACEMENT).replace(_AIR_MAXIMA, _AIR_REPLACEMENT)
    return _HELPER_IMPORT + body


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    arguments = parser.parse_args(argv)
    try:
        generated = build_seed_candidate_source(arguments.source.read_text(encoding="utf-8"))
        with arguments.output.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(generated)
    except FileExistsError:
        failure = "SEED_OUTPUT_EXISTS"
    except SeedAdapterError as error:
        failure = str(error)
    except (OSError, UnicodeError):
        failure = "SEED_ADAPTER_IO_FAILED"
    else:
        print(json.dumps({"status": "emitted", "frozen_source_sha256": FROZEN_V3_SHA256,
                          "candidate_sha256": hashlib.sha256(generated.encode()).hexdigest()}))
        return 0
    print(json.dumps({"status": "failed", "failure_code": failure}))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
