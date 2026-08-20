from __future__ import annotations

import os
import subprocess
from pathlib import Path

import nibabel as nib
import numpy as np
import pydicom
import pytest

from ethmoid_pipeline.conversion import (
    _dangerous_log_categories,
    _verify_dcm2niix,
    build_dcm2niix_command,
)
from ethmoid_pipeline.dicom import preflight_series
from ethmoid_pipeline.util import default_dcm2niix, sha256_file
from ethmoid_pipeline.verification import (
    _dicom_pixel_ras,
    _full_volume_hu_metrics,
    _map_to_nifti,
)

from .synthetic import write_ct_series


def test_converter_log_allows_harmless_warning_but_blocks_geometry_warning() -> None:
    harmless = "Warning: Patient Position (0018,5100) not specified."
    geometry = "Warning: unequal spacing between DICOM slices."

    assert _dangerous_log_categories(harmless) == []
    assert _dangerous_log_categories(geometry) == ["unequal spacing"]


@pytest.mark.integration
def test_pinned_dcm2niix_preserves_synthetic_geometry_and_hu(tmp_path: Path) -> None:
    executable = Path(os.environ.get("ETHMOID_DCM2NIIX", default_dcm2niix()))
    if not executable.is_file():
        pytest.skip("pinned dcm2niix is not installed")
    assert _verify_dcm2niix(executable)["version"] == "v1.0.20260724"
    stage = tmp_path / "stage"
    paths = write_ct_series(stage, instance_numbers=[6, 5, 4, 3, 2, 1])
    before = [sha256_file(path) for path in paths]
    preflight = preflight_series(paths)

    output_hashes: list[str] = []
    for run_number in (1, 2):
        output = tmp_path / f"output-{run_number}"
        output.mkdir()
        command = build_dcm2niix_command(executable, output, "Synthetic_Pilot", stage)
        completed = subprocess.run(command, check=False, capture_output=True, text=True, timeout=60)
        assert completed.returncode == 0, completed.stdout + completed.stderr
        nifti_paths = list(output.glob("*.nii"))
        assert len(nifti_paths) == 1
        assert len(list(output.glob("*.json"))) == 1
        image = nib.load(nifti_paths[0], mmap=True)
        assert int(np.prod(image.shape)) == 6 * 16 * 20
        sform, code = image.get_sform(coded=True)
        assert int(code) > 0
        inverse = np.linalg.inv(sform)
        for record in preflight.records:
            row, column = record.rows // 2, record.columns // 2
            point = _dicom_pixel_ras(record, row, column)
            floating, rounded = _map_to_nifti(point, inverse)
            assert np.max(np.abs(floating - rounded)) <= 1e-3
            dataset = pydicom.dcmread(record.path)
            expected = float(dataset.pixel_array[row, column]) - 1024.0
            observed = float(image.dataobj[tuple(int(item) for item in rounded)])
            assert observed == pytest.approx(expected, abs=1.0)
        metrics = _full_volume_hu_metrics(image, preflight.records, inverse)
        assert metrics["voxel_count"] == 6 * 16 * 20
        assert metrics["nonfinite_count"] == 0
        assert metrics["max_absolute_error"] == pytest.approx(0.0)

        if run_number == 1:
            data = image.get_fdata(dtype=np.float32)
            _, center_a = _map_to_nifti(
                _dicom_pixel_ras(preflight.records[1], 8, 10), inverse
            )
            _, center_b = _map_to_nifti(
                _dicom_pixel_ras(preflight.records[2], 8, 10), inverse
            )
            slice_axis = int(np.argmax(np.abs(center_b - center_a)))
            plane_index = int(center_b[slice_axis])
            slicer: list[int | slice] = [slice(None), slice(None), slice(None)]
            slicer[slice_axis] = plane_index

            corrupted = data.copy()
            corrupted[tuple(slicer)] += 100.0
            corrupted_image = nib.Nifti1Image(corrupted, image.affine)
            corrupted_metrics = _full_volume_hu_metrics(
                corrupted_image, preflight.records, inverse
            )
            assert corrupted_metrics["max_absolute_error"] == pytest.approx(100.0)

            nonfinite = data.copy()
            nonfinite[tuple(int(item) for item in center_b)] = np.nan
            nonfinite_image = nib.Nifti1Image(nonfinite, image.affine)
            nonfinite_metrics = _full_volume_hu_metrics(
                nonfinite_image, preflight.records, inverse
            )
            assert nonfinite_metrics["nonfinite_count"] == 1
        output_hashes.append(sha256_file(nifti_paths[0]))

    assert output_hashes[0] == output_hashes[1]
    assert [sha256_file(path) for path in paths] == before
