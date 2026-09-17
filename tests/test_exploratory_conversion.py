from pathlib import Path

import numpy as np
import pytest

from ethmoid_pipeline.conversion import convert_series
from ethmoid_pipeline.dicom import PreflightFailure, preflight_series
from ethmoid_pipeline.util import default_dcm2niix, sha256_file
from ethmoid_pipeline.verification import verify_conversion
from .synthetic import rewrite, write_ct_series


def make_reformat(root, angle=30, spacing=2.5):
    paths = write_ct_series(root, positions=[i * spacing for i in range(8)],
                            image_type=["DERIVED", "SECONDARY", "REFORMATTED"])
    radians = np.deg2rad(angle)
    row = np.array([1., 0., 0.])
    column = np.array([0., np.cos(radians), np.sin(radians)])
    normal = np.cross(row, column)
    for i, path in enumerate(paths):
        rewrite(path, ImageOrientationPatient=[*row, *column],
                ImagePositionPatient=(np.array([10., 20., 30.]) + i * spacing * normal).tolist(),
                SliceThickness=2.5, SpacingBetweenSlices=spacing)
    return paths


@pytest.mark.parametrize("angle", [0, 0.1, 5, 15, 30, 90])
def test_exploratory_reformat_has_physical_geometry_without_relaxing_strict(tmp_path: Path, angle):
    paths = make_reformat(tmp_path, angle)
    result = preflight_series(list(reversed(paths)), profile="exploratory")
    assert result.source_kind == "mpr"
    assert result.slice_spacing == pytest.approx(2.5)
    assert result.slice_thickness == 2.5
    assert result.safe_report("Case001", "Series001")["preflight_profile"] == "exploratory"
    assert list(result.records) == sorted(result.records, key=lambda item: item.coordinate)
    with pytest.raises(PreflightFailure, match="derived/reformatted"):
        preflight_series(paths)


def test_overlapping_thick_slices_keep_thickness_separate_from_interval(tmp_path):
    result = preflight_series(make_reformat(tmp_path, spacing=0.625), profile="exploratory")
    assert result.slice_thickness == 2.5
    assert result.slice_spacing == pytest.approx(0.625)


@pytest.mark.parametrize("term", ["AVERAGE", "MIP", "MINIP", "PROJECTION", "LOCALIZER"])
def test_projection_and_scout_cannot_use_exploratory_path(tmp_path, term):
    paths = make_reformat(tmp_path)
    for path in paths:
        rewrite(path, ImageType=["DERIVED", "SECONDARY", "REFORMATTED", term])
    with pytest.raises(PreflightFailure, match="projection|localizer"):
        preflight_series(paths, profile="exploratory")


def test_exploratory_rejects_in_plane_drift(tmp_path):
    paths = make_reformat(tmp_path, angle=0)
    rewrite(paths[-1], ImagePositionPatient=[10.1, 20, 47.5])
    with pytest.raises(PreflightFailure, match="drift or shear"):
        preflight_series(paths, profile="exploratory")


def test_exploratory_rejects_mixed_image_types(tmp_path):
    paths = make_reformat(tmp_path)
    rewrite(paths[-1], ImageType=["ORIGINAL", "PRIMARY", "AXIAL"])
    with pytest.raises(PreflightFailure, match="ImageType changes"):
        preflight_series(paths, profile="exploratory")


@pytest.mark.parametrize("positions", [[0, 2.5, 2.5, 5], [0, 2.5, 7.5, 10]])
def test_exploratory_preserves_duplicate_and_missing_plane_rejection(tmp_path, positions):
    with pytest.raises(PreflightFailure, match="duplicate|nonuniform"):
        preflight_series(write_ct_series(tmp_path, positions=positions), profile="exploratory")


def test_exploratory_calibrated_multienergy_remains_warned_and_strict_rejects(tmp_path):
    paths = write_ct_series(tmp_path)
    for path in paths:
        rewrite(path, MultienergyCTAcquisition="YES")
    assert "calibrated multienergy reconstruction is exploratory" in preflight_series(paths, profile="exploratory").warnings
    with pytest.raises(PreflightFailure, match="multienergy"):
        preflight_series(paths)
    rewrite(paths[-1], RescaleType="US")
    with pytest.raises(PreflightFailure, match="HU"):
        preflight_series(paths, profile="exploratory")


@pytest.mark.integration
@pytest.mark.parametrize("angle", [0, 30, 90])
def test_real_converter_reformat_preserves_source_and_all_hu(tmp_path, angle):
    executable = default_dcm2niix()
    if not executable.is_file():
        pytest.skip("pinned dcm2niix unavailable")
    source, work = tmp_path / "source", tmp_path / "work"
    paths = make_reformat(source, angle)
    original_hashes = [sha256_file(path) for path in paths]
    options = dict(pipeline_provenance={"kind": "synthetic-exploratory", "git_dirty": True},
                   preflight_profile="exploratory")
    convert_series(source, work, paths, "Case001", "Series001", executable, **options)
    result = verify_conversion(source, work, paths, "Case001", "Series001", **options)
    assert result["status"] == "AUTOMATED_PASS_PENDING_MANUAL"
    assert all(item["passed"] for item in result["checks"])
    assert result["metrics"]["max_absolute_hu_error"] == 0
    assert [sha256_file(path) for path in paths] == original_hashes
