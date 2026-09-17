"""Resolution facts are distinct from count completion or clinical validation."""
import numpy as np
import pytest

from ethmoid_pipeline.counting.quality import acquisition_snapshot, resolution_quality


@pytest.mark.parametrize("thickness,interval,classification", [
    (0.625, 0.625, "within_protocol"), (1.0, 1.0, "within_protocol"),
    (1.25, 1.25, "outside_protocol"), (2.0, 2.0, "outside_protocol"),
    (2.5, 2.5, "outside_protocol"), (2.5, 0.625, "outside_protocol"),
    (None, 0.625, "unknown"), (None, 2.5, "outside_protocol"),
])
def test_thickness_and_sampling_interval_are_independent(thickness, interval, classification):
    snapshot = acquisition_snapshot({"slice_thickness_mm": thickness, "spacing_mm": [0.5, 0.5, interval]}, "a" * 64)
    quality = resolution_quality(snapshot, np.diag([0.5, 0.5, interval, 1]))
    assert quality["resolution_class"] == classification
    assert quality["slice_thickness_mm"] == thickness
    assert quality["slice_interval_mm"] == interval
    assert quality["clinical_accuracy"] == "unmeasured"


def test_unknown_nifti_thickness_is_never_substituted_with_voxel_step():
    quality = resolution_quality(acquisition_snapshot(None, "a" * 64), np.diag([0.4, 0.4, 0.625, 1]))
    assert quality["slice_thickness_mm"] is quality["slice_interval_mm"] is None
    assert quality["native_spacing_mm"] == [0.4, 0.4, 0.625]
    assert quality["flags"] == ["acquisition_thickness_unknown"]


def test_derived_thin_reconstruction_retains_catalog_outside_protocol_status():
    value = acquisition_snapshot({"resolution_class": "outside_protocol", "slice_thickness_mm": 0.6}, "a" * 64)
    quality = resolution_quality(value, np.diag([0.6, 0.6, 0.6, 1]))
    assert quality["resolution_class"] == "outside_protocol"
    assert "outside_protocol_reconstruction" in quality["flags"]


@pytest.mark.parametrize("value", [
    {"source_sha256": "b" * 64}, {"slice_thickness_mm": True}, {"slice_thickness_mm": float("nan")},
    {"slice_thickness_mm": 0}, {"spacing_mm": [0.5, 0.5, -1]}, {"study_label": "arbitrary protected name"},
])
def test_acquisition_snapshot_rejects_ambiguous_or_wrong_source_facts(value):
    with pytest.raises(ValueError):
        acquisition_snapshot(value, "a" * 64)


def test_snapshot_drops_uncontrolled_text_and_preserves_only_neutral_facts():
    result = acquisition_snapshot({"study_label": "Case123", "series_label": "Series007", "warnings": ["private filename"], "PatientName": "private"}, "a" * 64)
    assert result["study_label"] == "Case123"
    assert "warnings" not in result and "PatientName" not in result
