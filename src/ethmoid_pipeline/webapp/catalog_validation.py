"""Revalidate the exact volume/source pair before reusing a legacy artifact."""
import nibabel as nib
import numpy as np

from ..util import sha256_file
from ..verification import _dicom_pixel_ras, _full_volume_hu_metrics, _map_to_nifti
from .service import _read_qc_report, inspect_nifti


def verified_volume(volume, report_path, expected_label, preflight) -> bool:
    try:
        info = inspect_nifti(volume)
        if info.warnings:
            return False
        status, _ = _read_qc_report(report_path, sha256_file(volume), info, expected_label)
        if status != "VERIFIED":
            return False
        image = nib.load(volume, mmap=True)
        expected = preflight.rows * preflight.columns * len(preflight.records)
        if int(np.prod(image.shape)) != expected:
            return False
        inverse = np.linalg.inv(image.affine)
        corners = []
        for record in preflight.records:
            for row, column in ((0, 0), (0, record.columns - 1), (record.rows - 1, 0), (record.rows - 1, record.columns - 1)):
                point = _dicom_pixel_ras(record, row, column)
                floating, index = _map_to_nifti(point, inverse)
                if np.max(np.abs(floating - index)) > 1e-3 or any(value < 0 or value >= image.shape[axis] for axis, value in enumerate(index)):
                    return False
                if np.max(np.abs((image.affine @ np.append(index, 1))[:3] - point)) > 1e-3:
                    return False
                corners.append(index)
        extent = np.asarray(corners)
        if not np.array_equal(extent.min(axis=0), np.zeros(3)) or not np.array_equal(extent.max(axis=0), np.asarray(image.shape) - 1):
            return False
        metrics = _full_volume_hu_metrics(image, preflight.records, inverse)
        return metrics["voxel_count"] == expected and metrics["finite_error_count"] == expected and metrics["nonfinite_count"] == 0 and metrics["max_absolute_error"] is not None and metrics["max_absolute_error"] <= 1.0
    except Exception:
        return False
