from __future__ import annotations

from pathlib import Path

from pydicom.uid import generate_uid
from pydicom import dcmread

from ethmoid_pipeline.inventory import build_inventory, resolve_selection
from ethmoid_pipeline.util import read_json

from .synthetic import write_ct_series


def test_inventory_assigns_neutral_first_seen_labels(tmp_path: Path) -> None:
    source = tmp_path / "source"
    work = tmp_path / "work"
    first_study, second_study = generate_uid(), generate_uid()
    write_ct_series(source / "b", study_uid=second_study)
    expected_paths = write_ct_series(source / "A", study_uid=first_study)

    result = build_inventory(source, work)
    redacted = read_json(result["redacted_inventory"])
    assert result["dicom_file_count"] == 12
    assert result["study_count"] == 2
    assert redacted["series"][0]["case"] == "Case001"
    assert "study_uid" not in redacted["series"][0]

    selected, study_uid, _ = resolve_selection(source, work, "Case001", "Series001")
    assert study_uid == first_study
    assert set(selected) == set(expected_paths)


def test_inventory_tolerates_present_but_empty_optional_numeric_tag(tmp_path: Path) -> None:
    source = tmp_path / "source"
    work = tmp_path / "work"
    paths = write_ct_series(source)
    dataset = dcmread(sorted(paths, key=lambda path: path.name.casefold())[0])
    dataset.SliceThickness = None
    dataset.save_as(sorted(paths, key=lambda path: path.name.casefold())[0], enforce_file_format=True)

    result = build_inventory(source, work)
    redacted = read_json(result["redacted_inventory"])
    assert result["dicom_file_count"] == 6
    assert redacted["series"][0]["slice_thickness_mm"] is None
