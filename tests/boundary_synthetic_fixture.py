"""Fresh external two-patient chamber truth for engineering mechanism checks.

``create_release(output_root, size=32)`` returns the protected release JSON path.
The generated chambers and their wall are synthetic geometry, not ethmoid anatomy.
"""
from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np

from ethmoid_pipeline.counting import boundary_data
from ethmoid_pipeline.counting import instance_experiment as legacy
from ethmoid_pipeline.counting.boundary_io import new_output


def create_release(output_root, *, size=32):
    """Write a new source-bound synthetic release; never overwrite existing files.

    Each complete native grid contains two air chambers separated by an
    explicitly generated bone wall. Labels come directly from the generating
    geometry. All foreground, separator, instance and evaluation voxels are
    known; this fixture provides no clinical transfer or accuracy evidence.
    """
    legacy.require(type(size) is int and 24 <= size <= 48, "SYNTHETIC_FIXTURE_SIZE_INVALID")
    output = new_output(output_root)
    protocol = output / "synthetic-protocol.local.json"
    legacy.write_json(protocol, {"schema_version": 1, "version": boundary_data.VERSION,
        "generator": "boundary-synthetic-chambers-v1", "scope": "synthetic_truth",
        "foreground_definition": "Generated air chamber interiors only",
        "separator_definition": "Explicit generated bone wall between the two chambers",
        "instance_definition": "Each generated chamber is one instance",
        "evaluation_definition": "Entire generated native grid; complete truth including background",
        "clinical_anatomy": False})
    samples = []
    shape = (size, size, size)
    for index, suffix in enumerate(("a", "b")):
        sample_id = f"synthetic-{suffix}"
        folder = output / sample_id
        folder.mkdir()
        low, high = 4 + index, size - 4
        middle = size // 2 + index
        outer = np.zeros(shape, bool)
        outer[low - 1:high + 1, low - 1:high + 1, low - 1:high + 1] = True
        instances = np.zeros(shape, np.int32)
        instances[low:middle - 1, low:high, low:high] = 1
        instances[middle + 1:high, low:high, low:high] = 2
        wall = np.zeros(shape, bool)
        wall[middle - 1:middle + 1, low:high, low:high] = True
        foreground = (instances > 0).astype(np.int8)
        separator = wall.astype(np.int8)
        hu = np.full(shape, 40, np.float32)
        hu[outer] = 650 + 20 * index
        hu[instances > 0] = -950 + 10 * index
        hu += np.random.default_rng(20261105 + index).normal(0, 4, shape).astype(np.float32)
        affine = np.diag([.45 + .05 * index, .55 + .05 * index, .7 + .1 * index, 1.])
        affine[:3, 3] = [index * 3., -index * 2., 0.]
        source_path = folder / "source.nii.gz"
        nib.save(nib.Nifti1Image(hu, affine), source_path)
        source = {"input": str(source_path), "source_sha256": legacy.sha256(source_path),
            "crop_bounds": [[0, size]] * 3,
            "midline_ras_x": float(nib.affines.apply_affine(affine, [(size - 1) / 2, 0, 0])[0])}
        # Bind to the exact on-disk native affine, including NIfTI float precision.
        _, _, _, native_affine = legacy.load_crop(source)
        annotation = folder / "annotation.npz"
        with annotation.open("xb") as stream:
            np.savez_compressed(stream, foreground_labels=foreground, separator_labels=separator,
                instance_ids=instances, evaluation_region=np.ones(shape, bool), affine=native_affine)
        samples.append({"sample_id": sample_id, "patient_group": f"synthetic-patient-{suffix}",
            "scope": "synthetic_truth", "source": source,
            "annotation": {"path": str(annotation), "sha256": legacy.sha256(annotation)},
            "protocol_sha256": legacy.sha256(protocol),
            "provenance": {"kind": "synthetic_truth", "generator": "boundary-synthetic-chambers-v1"}})
    release = {"schema_version": 1, "version": boundary_data.VERSION, "samples": samples}
    checked = boundary_data.validate_release(release)
    legacy.require(checked["readiness"]["status"] == "READY_GROUP_HOLDOUT", "SYNTHETIC_FIXTURE_NOT_READY")
    path = output / "release.local.json"
    legacy.write_json(path, release)
    legacy.write_json(output / "readiness.local.json", checked["readiness"])
    return path
