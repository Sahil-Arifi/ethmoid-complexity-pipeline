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

from ethmoid_pipeline.util import sha256_file, utc_now
from ethmoid_pipeline.counting.profiles import sealed
from ethmoid_pipeline.webapp.catalog import acquisition, fingerprint_hashes
from ethmoid_pipeline.webapp.app import create_app


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--work-root", required=True, type=Path)
    args = parser.parse_args()
    work = args.work_root.resolve()
    if work.parent != Path(tempfile.gettempdir()).resolve() or not work.name.startswith("ethmoid-e2e-"):
        raise RuntimeError("The synthetic server requires an isolated browser-test directory.")
    fixture = work / "fixture" / "directional-phantom.nii"
    if not fixture.is_file():
        from create_e2e_fixture import build_directional_phantom
        build_directional_phantom(fixture)
    # A second and third exact generated phantom provide distinct source grids
    # for study navigation. These never come from the real collection.
    reference = nib.load(fixture)
    generated = [fixture]
    for thickness in (0.625, 2.5):
        affine = reference.affine.copy()
        affine[2, 2] = thickness
        affine[2, 3] = -(reference.shape[2] - 1) * thickness / 2
        image = nib.Nifti1Image(np.asarray(reference.dataobj).copy(), affine)
        image.set_qform(affine, 1); image.set_sform(affine, 1)
        image.header.set_xyzt_units("mm")
        target = work / "fixture" / f"directional-phantom-{thickness}.nii"
        nib.save(image, target)
        generated.append(target)

    def execute(source_path: Path, output: Path, _config: Path) -> None:
        source = nib.load(source_path)
        references = [nib.load(path) for path in generated]
        if not any(source.shape == item.shape and np.array_equal(source.affine, item.affine) and np.array_equal(np.asarray(source.dataobj), np.asarray(item.dataobj)) for item in references):
            raise RuntimeError("Synthetic inference accepts only the generated directional phantom.")
        x, y, z = np.indices(source.shape)
        left = ((x - 15) / 5) ** 2 + ((y - 30) / 7) ** 2 + ((z - 24) / 8) ** 2 <= 1
        right = ((x - 33) / 7) ** 2 + ((y - 27) / 5) ** 2 + ((z - 20) / 6) ** 2 <= 1
        cavities = np.where(left, 1, np.where(right, 300, 0)).astype(np.uint16)
        air = np.where(np.asarray(source.dataobj) < -500, cavities, 0).astype(np.uint16)
        region = np.where(left, 1, np.where(right, 2, 0)).astype(np.uint8)
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
        frozen = json.loads(_config.read_text(encoding="utf-8"))
        result = {"schema_version": 1, "status": "completed", "method_version": "synthetic-browser-fixture",
                  "source_sha256": sha256_file(source_path), "protocol_version": frozen["protocol_version"],
                  "counts": {"left": 1, "right": 1, "total": 2}, "cells": cells,
                  "warnings": ["Synthetic test; no clinical inference."], "config": frozen["algorithm"],
                  "acquisition": frozen["acquisition"], "quality": frozen["source_quality"], "artifacts": artifacts}
        if frozen["protocol_version"] == "ethmoid-count-prototype-v3":
            assisted = bool(frozen.get("assistance", {}).get("revision_id"))
            result.update({"method_version": "scan-adaptive-native-0.3.0",
                           "scan_profile": frozen["scan_profile"], "processing_plan": frozen["processing_plan"],
                           "analysis_mode": "assisted" if assisted else "automatic",
                           "outcome": "assisted_candidate_count" if assisted else "automatic_candidate_count",
                           "model_identity": {"localization": "reviewer_native_region" if assisted else "v2_geometric_baseline",
                                              "direct_ethmoid_model": "unavailable_without_promoted_weights"},
                           "assistance": None,
                           "artifact_verification": {"status": "verified", "source_sha256": result["source_sha256"],
                               "sha256": {key: sha256_file(output / value) for key, value in artifacts.items()}}})
            if assisted:
                # Exercise the production provenance contract with generated
                # regions only; this fixture does not establish clinical accuracy.
                correction = frozen["assistance"]
                boundary = nib.load(correction["region_path"])
                labels = np.asanyarray(boundary.dataobj)
                centers = [nib.affines.apply_affine(boundary.affine, np.argwhere(labels == label).mean(axis=0))[0] for label in (1, 2)]
                result["assistance"] = sealed({**{key: correction[key] for key in ("schema_version", "source_sha256", "revision_id", "parent_run_id", "region_sha256")},
                    "exclusion_sha256": None, "midline_ras_x": float(np.mean(centers))})
        (output / "result.json").write_text(json.dumps(result), encoding="utf-8")

    app = create_app(work_root=work)
    app.state.counting_service._executor = execute

    @app.post("/__e2e__/release-workers")
    def release_workers(root: str):
        # Playwright runs global teardown before stopping its webServer. Release
        # the production workers' OS locks before removing this synthetic root.
        # This endpoint exists only in this guarded, isolated test entrypoint.
        if root != work.name:
            from fastapi import HTTPException
            raise HTTPException(status_code=409, detail="Synthetic work root mismatch")
        if hasattr(app.state, "uploads"):
            app.state.uploads.close()
        app.state.catalog_service.close()
        app.state.counting_service.close()
        return {"work_root": work.name}

    service = app.state.ethmoid_service
    ready_cases = [service.import_nifti(path, case_label=label) for path, label in zip(generated[1:], ("Case910", "Case911"))]
    # Seed only neutral synthetic catalog links; production routes, guards,
    # persistence, and result validation remain active in the browser tests.
    with service.store.connect() as connection:
        connection.execute("INSERT INTO catalog_sources(id,root_path,status,updated_at) VALUES(?,?,?,?)", ("synthetic-source", str(work / "fixture"), "ready", utc_now()))
        connection.execute("INSERT INTO catalog_studies(id,source_id,study_uid,study_label,selected_series_id,recommended_series_id) VALUES(?,?,?,?,?,?)", ("synthetic-study", "synthetic-source", "synthetic-study-uid", "Case900", "synthetic-thin", "synthetic-thin"))
        for case, path, identity, label, thickness in zip(ready_cases, generated[1:], ("synthetic-thin", "synthetic-thick"), ("Series007", "Series001"), (0.625, 2.5)):
            metadata = {"spacing_mm": [0.8, 0.9, thickness], "slice_thickness_mm": thickness, "image_count": 44, "kernel": "BONEPLUS", "source_kind": "native", "warnings": [], "metadata_validated": True}
            metadata.update(acquisition(metadata))
            fingerprint = fingerprint_hashes([sha256_file(path)])
            connection.execute("INSERT INTO catalog_series(id,study_id,series_uid,series_label,descriptor_json,files_json,inventory_signature,source_fingerprint,case_id,preparation_status) VALUES(?,?,?,?,?,?,?,?,?,?)", (identity, "synthetic-study", identity, label, json.dumps(metadata), json.dumps([path.name]), fingerprint, fingerprint, case["case_id"], "ready"))
            connection.execute("INSERT INTO catalog_case_links(case_id,series_id,source_fingerprint,proof,acquisition_json) VALUES(?,?,?,?,?)", (case["case_id"], identity, fingerprint, "synthetic-browser-fixture", json.dumps(acquisition(metadata))))
        metadata = {"spacing_mm": None, "slice_thickness_mm": None, "image_count": 1, "kernel": None, "source_kind": "scout", "resolution_class": "unknown", "warnings": []}
        connection.execute("INSERT INTO catalog_series(id,study_id,series_uid,series_label,descriptor_json,files_json,inventory_signature,preparation_status,reason_code) VALUES(?,?,?,?,?,?,?,?,?)", ("synthetic-scout", "synthetic-study", "synthetic-scout", "Series009", json.dumps(metadata), "[]", "synthetic-scout", "unavailable", "SCOUT"))
    uvicorn.run(app, host="127.0.0.1", port=8766)


if __name__ == "__main__":
    main()
