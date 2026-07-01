from __future__ import annotations

from pathlib import Path

import pytest
from pydicom import dcmread
from pydicom.uid import JPEG2000Lossless, generate_uid

from ethmoid_pipeline.dicom import PreflightFailure, preflight_series

from .synthetic import rewrite, write_ct_series


def test_regular_series_sorts_by_position_not_instance_number(tmp_path: Path) -> None:
    paths = write_ct_series(tmp_path, instance_numbers=[6, 5, 4, 3, 2, 1])
    result = preflight_series(paths)
    assert [record.coordinate for record in result.records] == sorted(
        record.coordinate for record in result.records
    )
    assert [record.instance_number for record in result.records] == [6, 5, 4, 3, 2, 1]
    assert result.slice_spacing == pytest.approx(0.625)


@pytest.mark.parametrize(
    ("positions", "message"),
    [
        ([0.0, 0.625, 0.625, 1.25], "duplicate"),
        ([0.0, 0.625, 1.875, 2.5], "nonuniform"),
        ([0.0, 0.045, 0.625, 0.670, 1.25, 1.295], "nonuniform"),
    ],
)
def test_rejects_duplicate_missing_and_interleaved_planes(
    tmp_path: Path, positions: list[float], message: str
) -> None:
    paths = write_ct_series(tmp_path, positions=positions)
    with pytest.raises(PreflightFailure, match=message):
        preflight_series(paths)


def test_rejects_geometry_change(tmp_path: Path) -> None:
    paths = write_ct_series(tmp_path)
    rewrite(paths[-1], PixelSpacing=[0.6, 0.5])
    with pytest.raises(PreflightFailure, match="PixelSpacing changes"):
        preflight_series(paths)


def test_rejects_antiparallel_iop_direction_flip(tmp_path: Path) -> None:
    paths = write_ct_series(tmp_path)
    rewrite(paths[-1], ImageOrientationPatient=[-1, 0, 0, 0, -1, 0])
    with pytest.raises(PreflightFailure, match="ImageOrientationPatient changes"):
        preflight_series(paths)


def test_rejects_mixed_acquisition_and_event(tmp_path: Path) -> None:
    event_a, event_b = generate_uid(), generate_uid()
    paths = write_ct_series(
        tmp_path,
        acquisition_numbers=[1, 1, 1, 2, 2, 2],
        irradiation_event_uids=[event_a] * 3 + [event_b] * 3,
    )
    with pytest.raises(PreflightFailure, match="multiple AcquisitionNumber"):
        preflight_series(paths)


def test_rejects_mixed_series_uid(tmp_path: Path) -> None:
    paths = write_ct_series(tmp_path)
    rewrite(paths[-1], SeriesInstanceUID=generate_uid())
    with pytest.raises(PreflightFailure, match="SeriesInstanceUID changes"):
        preflight_series(paths)


def test_rejects_duplicate_sop_uid(tmp_path: Path) -> None:
    paths = write_ct_series(tmp_path)
    first = dcmread(paths[0], stop_before_pixels=True)
    rewrite(paths[-1], SOPInstanceUID=first.SOPInstanceUID)
    with pytest.raises(PreflightFailure, match="duplicate SOPInstanceUID"):
        preflight_series(paths)


@pytest.mark.parametrize(
    "changes",
    [
        {"ImageType": ["DERIVED", "SECONDARY", "AXIAL"]},
        {"ImageType": ["ORIGINAL", "PRIMARY", "LOCALIZER"]},
        {"PhotometricInterpretation": "MONOCHROME1"},
        {"BurnedInAnnotation": "YES"},
        {"NumberOfFrames": 2},
        {"TransferSyntaxUID": JPEG2000Lossless},
        {"RescaleType": "US"},
        {"MultienergyCTAcquisition": "YES"},
    ],
)
def test_rejects_unsupported_or_privacy_risk_images(
    tmp_path: Path, changes: dict[str, object]
) -> None:
    paths = write_ct_series(tmp_path)
    rewrite(paths[0], **changes)
    with pytest.raises(PreflightFailure):
        preflight_series(paths)
