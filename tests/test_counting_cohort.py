from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


spec = importlib.util.spec_from_file_location(
    "prepare_counting_cohort", Path(__file__).resolve().parents[1] / "scripts" / "prepare_counting_cohort.py"
)
assert spec and spec.loader
cohort = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cohort)


def candidate(case, series="Series001", **changes):
    value = {"case": case, "series": series, "kernel": "BONEPLUS", "image_type": "ORIGINAL\\PRIMARY\\AXIAL", "slice_thickness_mm": 0.625, "ipp_spacing_median_mm": 0.625, "ipp_spacing_regular": True}
    return value | changes


def inventory():
    return {"series": [candidate(case, series) for case, series in cohort.DEVELOPMENT] + [candidate(f"Case{index:03}") for index in (1, 2, 4, 5, 6, 7, 17, 22)]}


def test_cohort_split_is_deterministic_by_study_and_never_by_reconstruction():
    source = inventory()
    source["series"] += [candidate("Case001", "Series002"), candidate("Case008", image_type="ORIGINAL\\PRIMARY\\AXIAL\\CT_SOM5 MPR")]
    expected = cohort.select_cohort(source)
    source["series"].reverse()
    assert cohort.select_cohort(source) == expected
    development = [case for case, _, split in expected if split == "development"]
    held_out = [case for case, _, split in expected if split == "held_out"]
    assert held_out == ["Case001", "Case002", "Case004", "Case005", "Case006"]
    assert len(set(development + held_out)) == 10


def test_fixed_development_selection_cannot_silently_change_when_geometry_drifts():
    source = inventory()
    source["series"][0]["ipp_spacing_regular"] = False
    with pytest.raises(ValueError, match="fixed development"):
        cohort.select_cohort(source)


def test_insufficient_distinct_held_out_studies_fails():
    source = {"series": [candidate(case, series) for case, series in cohort.DEVELOPMENT]}
    source["series"] += [candidate("Case001", f"Series{index:03}") for index in range(1, 7)]
    with pytest.raises(ValueError, match="Five separate"):
        cohort.select_cohort(source)
