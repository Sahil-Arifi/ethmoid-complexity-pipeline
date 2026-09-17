from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np
import pydicom
import pytest

from ethmoid_pipeline.conversion import reconcile_nifti_affine
from ethmoid_pipeline.dicom import preflight_series
from ethmoid_pipeline.util import PipelineError, sha256_file
from ethmoid_pipeline.verification import _full_volume_hu_metrics

from .synthetic import rewrite, write_ct_series


def _image_and_source(tmp_path: Path, angle: float = 0, flip: bool = False):
    positions = [512 + index * 0.6 for index in range(172)]
    paths = write_ct_series(tmp_path / "source", positions=positions, rows=8, columns=10)
    radians = np.deg2rad(angle)
    row = np.asarray([1.0, 0.0, 0.0])
    column = np.asarray([0.0, np.cos(radians), np.sin(radians)])
    normal = np.cross(row, column)
    for index, path in enumerate(paths):
        rewrite(path, SliceThickness=0.6, SpacingBetweenSlices=0.6,
                ImageOrientationPatient=[*row, *column],
                ImagePositionPatient=(np.asarray([0, 0, 512]) + normal * index * 0.6).tolist())
    preflight = preflight_series(paths)
    data = np.stack([pydicom.dcmread(record.path).pixel_array for record in preflight.records], axis=2)
    affine = np.eye(4)
    affine[:3, 0] = np.asarray([-1, -1, 1]) * column * 0.5
    affine[:3, 1] = np.asarray([-1, -1, 1]) * row * 0.5
    affine[:3, 2] = np.asarray([-1, -1, 1]) * normal * 0.6
    affine[:3, 3] = [0, 0, 512]
    if flip:
        orientation = np.asarray([[1, -1], [0, 1], [2, -1]])
        transformed = nib.orientations.apply_orientation(data, orientation)
        affine = affine @ nib.orientations.inv_ornt_aff(orientation, data.shape)
        data = transformed
    image = nib.Nifti1Image(data, affine)
    image.set_qform(affine, code=1)
    image.set_sform(affine, code=1)
    image.header.set_slope_inter(1.0, -1024.0)
    path = tmp_path / "synthetic.nii"
    nib.save(image, path)
    return path, preflight


def _replace_forms(path: Path, *, drift_sform: bool = False, shift_both: bool = False):
    with path.open("rb") as stream:
        header = nib.Nifti1Header.from_fileobj(stream)
    if drift_sform:
        form = header.get_sform()
        form[:3, 2] *= 0.60003662109375 / 0.6
        header.set_sform(form, code=1)
    if shift_both:
        for getter, setter in ((header.get_sform, header.set_sform), (header.get_qform, header.set_qform)):
            form = getter()
            form[0, 3] += 0.03
            setter(form, code=1)
    with path.open("r+b") as stream:
        stream.write(header.binaryblock)


@pytest.mark.parametrize(("angle", "flip"), [(0, False), (0, True), (7, False), (7, True)])
def test_decimal_spacing_reconciliation_preserves_encoded_voxels_and_full_hu(tmp_path: Path, angle: float, flip: bool):
    path, preflight = _image_and_source(tmp_path, angle, flip)
    _replace_forms(path, drift_sform=True)
    original = nib.load(path)
    before = path.read_bytes()[original.dataobj.offset:]
    with pytest.raises(PipelineError, match="not on the NIfTI voxel grid"):
        _full_volume_hu_metrics(original, preflight.records, np.linalg.inv(original.affine))
    report = reconcile_nifti_affine(path, preflight)
    assert report["applied"] is True
    repaired = nib.load(path)
    assert path.read_bytes()[repaired.dataobj.offset:] == before
    assert (repaired.dataobj.slope, repaired.dataobj.inter) == (1, -1024)
    metrics = _full_volume_hu_metrics(repaired, preflight.records, np.linalg.inv(repaired.affine))
    assert metrics["max_absolute_error"] == 0
    assert metrics["nonfinite_count"] == 0
    assert metrics["voxel_count"] == 172 * 8 * 10


def test_already_correct_geometry_is_not_rewritten(tmp_path: Path):
    path, preflight = _image_and_source(tmp_path)
    before = sha256_file(path)
    assert reconcile_nifti_affine(path, preflight)["applied"] is False
    assert sha256_file(path) == before


def test_disagreement_in_both_original_forms_is_not_repaired(tmp_path: Path):
    path, preflight = _image_and_source(tmp_path)
    _replace_forms(path, shift_both=True)
    before = sha256_file(path)
    with pytest.raises(PipelineError, match="beyond bounded affine precision"):
        reconcile_nifti_affine(path, preflight)
    assert sha256_file(path) == before


@pytest.mark.integration
def test_actual_converter_decimal_spacing_large_origin_passes_unchanged_qc(tmp_path: Path):
    from ethmoid_pipeline.conversion import convert_series
    from ethmoid_pipeline.util import default_dcm2niix
    from ethmoid_pipeline.verification import verify_conversion

    executable = default_dcm2niix()
    if not executable.is_file():
        pytest.skip("pinned dcm2niix is not installed")
    source = tmp_path / "source"
    paths = write_ct_series(source, positions=[512 + index * 0.6 for index in range(172)], rows=8, columns=10)
    for path in paths:
        rewrite(path, SliceThickness=0.6, SpacingBetweenSlices=0.6)
    provenance = {"kind": "synthetic-affine-regression", "git_dirty": True}
    convert_series(source, tmp_path / "work", paths, "Case001", "Series001", executable, pipeline_provenance=provenance)
    report = verify_conversion(source, tmp_path / "work", paths, "Case001", "Series001", pipeline_provenance=provenance)
    assert all(check["passed"] for check in report["checks"])
    assert report["metrics"]["max_absolute_hu_error"] == 0
