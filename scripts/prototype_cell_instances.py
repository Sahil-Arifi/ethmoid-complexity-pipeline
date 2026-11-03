"""Run the isolated, external-artifact video-guided instance prototype."""
from __future__ import annotations

import argparse
import sys

from ethmoid_pipeline.counting import instance_experiment as experiment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("readiness", "train", "infer", "mechanism", "report"))
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    try:
        result = getattr(experiment, args.command)(args.manifest, args.output)
        print(result["status"])
        return 0 if result["status"] in {"READY_CONDITIONAL_DEVELOPMENT", "COMPLETE", "PASS_CONDITIONAL_DEVELOPMENT", "PASS_CONDITIONAL_MECHANISM"} else 1
    except experiment.ExperimentError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    except Exception:
        # Protected filenames and image headers must not escape in a traceback.
        print("EXPERIMENT_EXECUTION_FAILED", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
