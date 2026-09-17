"""Synchronize and prepare the protected local collection; never run inference."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ethmoid_pipeline.util import default_work_root
from ethmoid_pipeline.webapp.catalog import CatalogService
from ethmoid_pipeline.webapp.catalog_config import CatalogConfig, load_local_config
from ethmoid_pipeline.webapp.catalog_queue import PreparationQueue
from ethmoid_pipeline.webapp.service import AppService


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-root", type=Path, default=default_work_root())
    parser.add_argument("--dcm2niix", type=Path)
    parser.add_argument("--catalog-only", action="store_true", help="Index and link verified existing volumes without preparing remaining series.")
    parser.add_argument("--timeout-seconds", type=float, default=0, help="Zero waits until preparation is idle; interrupted work resumes on the next startup.")
    args = parser.parse_args(argv)
    catalog = queue = None
    try:
        work = args.work_root.resolve(strict=True)
        app_root = work / "webapp"
        config = load_local_config(app_root)
        if config is None:
            # Explicit custom work roots are useful for synthetic integration
            # checks. The source can only come from the local protected map.
            mapping = json.loads((work / "inventory/uid_map.local.json").read_text(encoding="utf-8"))
            config = CatalogConfig(work, Path(mapping["source_root"]))
        config.validate(app_root)
        app = AppService(work_root=app_root, dcm2niix=args.dcm2niix)
        catalog = CatalogService(app, config)
        catalog.sync()
        if not args.catalog_only:
            queue = PreparationQueue(catalog)
            queue.start()
            started = time.monotonic()
            while True:
                summary = catalog.status()
                if not summary["queued"] and not summary["preparing"]:
                    break
                if args.timeout_seconds > 0 and time.monotonic() - started >= args.timeout_seconds:
                    print(json.dumps({"status": "preparation_pending", "catalog": summary}))
                    return 2
                time.sleep(0.25)
        print(json.dumps({"catalog": catalog.status(), "inference_started": False}))
        return 0
    except KeyboardInterrupt:
        print(json.dumps({"status": "interrupted", "inference_started": False}))
        return 2
    except Exception:
        # Raw exception strings can contain source paths or identifiers.
        print(json.dumps({"status": "error", "reason": "Local catalog preparation could not complete. Check the protected configuration and local diagnostic logs."}))
        return 1
    finally:
        if queue is not None:
            queue.close()
        if catalog is not None:
            catalog.close()


if __name__ == "__main__":
    raise SystemExit(main())
