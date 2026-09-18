"""Subprocess entry point for the separate local Python 3.12 inference runtime."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .engine import CountingError, analyze_volume


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a local experimental 3D ethmoid count.")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--landmarks", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--acquisition", type=Path, help="Server-created source-bound acquisition snapshot; never derived from voxel spacing.")
    parser.add_argument("--method", choices=("v2", "adaptive-v3", "research-v4"), default="v2")
    parser.add_argument("--assistance", type=Path, help="Server-created immutable source-bound boundary revision.")
    parser.add_argument("--processing-plan", type=Path)
    parser.add_argument("--progress", type=Path, help="Protected atomic stage snapshot.")
    parser.add_argument("--model-manifest", type=Path, help="Operator-promoted hash-bound direct anatomy model.")
    args = parser.parse_args(argv)
    try:
        configuration = json.loads(args.config.read_text(encoding="utf-8")) if args.config else None
        acquisition = json.loads(args.acquisition.read_text(encoding="utf-8")) if args.acquisition else None
        if args.method == "research-v4":
            if args.assistance or args.model_manifest:
                raise CountingError("INVALID_CONFIG", "The research-v4 experiment does not accept assisted or direct-model membership.")
            from .research import analyze_research_volume
            plan = json.loads(args.processing_plan.read_text(encoding="utf-8")) if args.processing_plan else None
            result = analyze_research_volume(args.input, args.output, args.landmarks, configuration,
                acquisition=acquisition, processing_plan=plan, progress_path=args.progress)
        elif args.method == "adaptive-v3":
            from .adaptive import analyze_adaptive_volume
            assistance = json.loads(args.assistance.read_text(encoding="utf-8")) if args.assistance else None
            plan = json.loads(args.processing_plan.read_text(encoding="utf-8")) if args.processing_plan else None
            result = analyze_adaptive_volume(args.input, args.output, args.landmarks, configuration,
                acquisition=acquisition, assistance=assistance, processing_plan=plan,
                progress_path=args.progress, model_manifest=args.model_manifest)
        else:
            if args.assistance or args.processing_plan or args.model_manifest:
                raise CountingError("INVALID_CONFIG", "Adaptive processing options require the versioned adaptive-v3 method.")
            result = analyze_volume(args.input, args.output, args.landmarks, configuration, acquisition=acquisition)
    except CountingError as error:
        print(json.dumps({"status": "failed", "failure_code": error.code, "message": str(error)}))
        return 2
    except (OSError, ValueError):
        print(json.dumps({"status": "failed", "failure_code": "INVALID_CONFIG", "message": "The fixed counting configuration is unreadable."}))
        return 2
    print(json.dumps({"status": result["status"], "failure_code": result.get("failure_code")}))
    return 0 if result["status"] == "completed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
