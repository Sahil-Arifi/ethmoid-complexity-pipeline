"""Source-bound research CLI; protected artifacts never enter the application."""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys

from ethmoid_pipeline.counting import instance_experiment as safe
from ethmoid_pipeline.counting.boundary_io import new_output


def _failure(args, code, preexisting):
    try:
        output = safe.external(args.output, exists=False)
        if preexisting:
            return
        if not output.exists():
            output = new_output(output)
        safe.write_json(output/"failure.local.json", {"status": "FAILED", "version": "boundary-instance-v1",
                        "command": args.command, "error_code": code})
    except Exception:
        pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("diagnose", "export", "readiness", "train", "infer", "evaluate"))
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    preexisting = os.path.exists(args.output)
    try:
        # Supervise expensive CUDA commands outside the worker. The hard wall
        # limit includes loading and validation, even if a device call hangs.
        if args.command in {"train", "infer"} and os.environ.get("ETHMOID_BOUNDARY_CHILD") != "1":
            environment = {**os.environ, "ETHMOID_BOUNDARY_CHILD": "1"}
            process = subprocess.Popen([sys.executable, __file__, *sys.argv[1:]], env=environment,
                                       stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            try:
                stdout, stderr = process.communicate(timeout=3600 if args.command == "train" else 1020)
            except subprocess.TimeoutExpired:
                if os.name == "nt":
                    subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"],
                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
                process.kill()
                process.communicate()
                _failure(args, "PROCESS_TIME_BUDGET_EXCEEDED", preexisting)
                print("PROCESS_TIME_BUDGET_EXCEEDED", file=sys.stderr)
                return 2
            # Workers emit fixed status/error codes only; do not relay raw tracebacks.
            message = (stdout if process.returncode == 0 else stderr).strip()
            message = message if re.fullmatch(r"[A-Z][A-Z0-9_]{0,100}", message) else "WORKER_FAILED"
            print(message, file=sys.stdout if process.returncode == 0 else sys.stderr)
            if process.returncode != 0:
                _failure(args, message, preexisting)
            return process.returncode if process.returncode in (0, 1, 2) else 2
        if args.command in {"export", "readiness"}:
            from ethmoid_pipeline.counting import boundary_data as module
        elif args.command == "diagnose":
            from ethmoid_pipeline.counting import boundary_diagnostics as module
        else:
            from ethmoid_pipeline.counting import boundary_pipeline as module
        result = getattr(module, args.command)(args.manifest, args.output)
        print(result["status"])
        return 0 if result["status"] in {"COMPLETE", "COMPLETE_BOUNDED", "EXPORTED"} or result.get("training_allowed") else 1
    except Exception as exc:
        code = safe.failure_code(exc)
        if type(exc).__module__ == "ethmoid_pipeline.counting.boundary_partition" and re.fullmatch(r"PARTITION_[A-Z0-9_]{1,80}", str(exc)):
            code = str(exc)
        if isinstance(exc, ModuleNotFoundError):
            code = "DEPENDENCY_UNAVAILABLE"
        if isinstance(exc, MemoryError) or type(exc).__name__ == "OutOfMemoryError":
            code = "MEMORY_BUDGET_EXCEEDED"
        _failure(args, code, preexisting)
        print(code, file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
