"""Browser-test server with synthetic inference, never used by the application CLI."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import nibabel as nib
import numpy as np
import uvicorn

from ethmoid_pipeline.util import sha256_file
from ethmoid_pipeline.webapp.app import create_app
from ethmoid_pipeline.counting.protocol import PROTOCOL_VERSION


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--work-root", required=True, type=Path)
    args = parser.parse_args()
    work = args.work_root.resolve()
    if work.parent != Path(tempfile.gettempdir()).resolve() or not work.name.startswith("ethmoid-e2e-"):
        raise RuntimeError("The synthetic server requires an isolated browser-test directory.")
    fixture = work / "fixture" / "directional-phantom.nii"

    def execute(source_path: Path, output: Path, _config: Path) -> None:
        source = nib.load(source_path)
        reference = nib.load(fixture)
        if source.shape != reference.shape or not np.array_equal(source.affine, reference.affine) or not np.array_equal(np.asarray(source.dataobj), np.asarray(reference.dataobj)):
            raise RuntimeError("Synthetic inference accepts only the generated directional phantom.")
        x, y, z = np.indices(source.shape)
        left = ((x - 15) / 5) ** 2 + ((y - 30) / 7) ** 2 + ((z - 24) / 8) ** 2 <= 1
        right = ((x - 33) / 7) ** 2 + ((y - 27) / 5) ** 2 + ((z - 20) / 6) ** 2 <= 1
        cavities = np.where(left, 1, np.where(right, 300, 0)).astype(np.uint16)
        air = np.where(np.asarray(source.dataobj) < -500, cavities, 0).astype(np.uint16)
        region = np.where(left, 1, np.where(right, 2, 0)).astype(np.uint16)
        output.mkdir(parents=True)
        artifacts = {name: f"{name}.nii.gz" for name in ("cavities", "air", "region")}
        for name, data in (("cavities", cavities), ("air", air), ("region", region)):
            image = nib.Nifti1Image(data, source.affine)
            image.set_qform(source.affine, 1); image.set_sform(source.affine, 1)
            image.header.set_xyzt_units("mm")
            nib.save(image, output / artifacts[name])
        cells = [{"id": identity, "side": side,
                  "centroid_ras_mm": nib.affines.apply_affine(source.affine, np.argwhere(cavities == identity).mean(axis=0)).tolist(),
                  "included": True, "uncertain": True, "origin": "automatic", "reasons": ["Synthetic browser fixture"],
                  "voxel_count": int(np.count_nonzero(cavities == identity)), "air_voxel_count": int(np.count_nonzero(air == identity))}
                 for identity, side in ((1, "left"), (300, "right"))]
        result = {"schema_version": 1, "status": "completed", "method_version": "synthetic-browser-fixture",
                  "source_sha256": sha256_file(source_path), "protocol_version": PROTOCOL_VERSION,
                  "counts": {"left": 1, "right": 1, "total": 2}, "cells": cells,
                  "warnings": ["Synthetic test; no clinical inference."], "config": json.loads(_config.read_text(encoding="utf-8"))["algorithm"], "artifacts": artifacts}
        (output / "result.json").write_text(json.dumps(result), encoding="utf-8")

    app = create_app(work_root=work)
    app.state.counting_service._executor = execute
    uvicorn.run(app, host="127.0.0.1", port=8766)


if __name__ == "__main__":
    main()
