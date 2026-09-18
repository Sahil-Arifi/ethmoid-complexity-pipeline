import nibabel as nib
import numpy as np
import pytest

pytest.importorskip("scipy")
pytest.importorskip("skimage")

from ethmoid_pipeline.counting.adaptive import bounded_localize
from ethmoid_pipeline.counting.engine import CountingConfig, CountingError
from ethmoid_pipeline.counting.localization import Localization


def resources():
    hu = np.zeros((80, 80, 80), np.float32)
    angle = np.deg2rad(15)
    affine = np.eye(4)
    affine[:3, :3] = [[np.cos(angle), 0, np.sin(angle)], [0, 1, 0], [-np.sin(angle), 0, np.cos(angle)]]
    names = ("eye_left", "eye_right", "optic_nerve_left", "optic_nerve_right", "nasal_cavity_left", "nasal_cavity_right", "sinus_maxillary", "sinus_frontal")
    masks = {name: np.zeros(hu.shape, bool) for name in names}
    for mask in masks.values():
        mask[35:45, 35:45, 35:45] = True
    return hu, affine, masks


def test_oversized_proxy_uses_retained_landmarks_and_native_output(monkeypatch):
    import ethmoid_pipeline.counting.adaptive as adaptive
    import ethmoid_pipeline.counting.localization as localization
    hu, affine, masks = resources()
    def limit(plan):
        if np.prod(plan["shape"]) > 100_000:
            raise CountingError("RESOURCE_LIMIT", "synthetic test budget")
    monkeypatch.setattr(adaptive, "check_proxy_limit", limit)
    def localize(values, transform, crop_masks, cfg, grid):
        assert all(mask.sum() == 1000 for mask in crop_masks.values())
        assert np.allclose(transform[:3, :3], affine[:3, :3])
        return Localization(crop_masks["eye_left"], np.zeros(values.shape, bool), {}, 0, [])
    monkeypatch.setattr(localization, "localize_native", localize)
    diagnostics = {}
    result = bounded_localize(hu, affine, masks, CountingConfig(), {"crop_margin_mm": [12.0, 24.0]}, diagnostics)
    assert result.region.shape == hu.shape
    assert np.array_equal(result.region, masks["eye_left"])
    assert diagnostics["mode"] == "bounded_anatomical_crop"
    assert diagnostics["attempts"][0]["outcome"] == "localized"
    assert diagnostics["attempts"][0]["geometry"]["spacing_mm"] == pytest.approx([1, 1, 1])


def test_cropped_target_expands_before_acceptance(monkeypatch):
    import ethmoid_pipeline.counting.adaptive as adaptive
    import ethmoid_pipeline.counting.localization as localization
    hu, affine, masks = resources()
    monkeypatch.setattr(adaptive, "check_proxy_limit", lambda plan: (_ for _ in ()).throw(CountingError("RESOURCE_LIMIT", "test")) if plan["shape"][0] > 90 else None)
    shapes = []
    def localize(values, transform, crop_masks, cfg, grid):
        shapes.append(values.shape)
        region = crop_masks["eye_left"].copy()
        if len(shapes) == 1:
            region[0, 0, 0] = True
        return Localization(region, np.zeros(values.shape, bool), {}, 0, [])
    monkeypatch.setattr(localization, "localize_native", localize)
    diagnostics = {}
    result = bounded_localize(hu, affine, masks, CountingConfig(), {"crop_margin_mm": [12.0, 24.0]}, diagnostics)
    assert len(shapes) == 2 and shapes[1][0] > shapes[0][0]
    assert diagnostics["attempts"][0]["outcome"] == "TARGET_CLIPPED"
    assert np.array_equal(result.region, masks["eye_left"])


def test_crop_cannot_silently_coarsen_when_even_regional_proxy_exceeds_budget(monkeypatch):
    import ethmoid_pipeline.counting.adaptive as adaptive
    hu, affine, masks = resources()
    monkeypatch.setattr(adaptive, "check_proxy_limit", lambda plan: (_ for _ in ()).throw(CountingError("RESOURCE_LIMIT", "test")))
    diagnostics = {}
    with pytest.raises(CountingError) as error:
        bounded_localize(hu, affine, masks, CountingConfig(), {"crop_margin_mm": [12.0, 24.0]}, diagnostics)
    assert error.value.code == "RESOURCE_LIMIT"
    assert len(diagnostics["attempts"]) == 2
    assert all(entry["geometry"]["spacing_mm"] == pytest.approx([1, 1, 1]) for entry in diagnostics["attempts"])


def test_oversized_axis_aligned_anatomy_is_bounded_without_changing_native_grid(monkeypatch):
    import ethmoid_pipeline.counting.localization as localization
    hu, _, masks = resources()
    affine = np.eye(4)
    received = []
    def localize(values, transform, crop_masks, cfg, grid):
        received.append(values.shape)
        assert grid["mode"] == "native_axis_aligned" and values.size <= 100_000
        return Localization(crop_masks["eye_left"], np.zeros(values.shape, bool), {}, 0, [])
    monkeypatch.setattr(localization, "localize_native", localize)
    diagnostics = {}
    result = bounded_localize(hu, affine, masks, CountingConfig(),
        {"crop_margin_mm": [12.0, 24.0], "proxy_max_voxels": 100_000}, diagnostics)
    assert received == [(34, 34, 34)]
    assert result.region.shape == hu.shape and np.array_equal(result.region, masks["eye_left"])
    assert diagnostics["mode"] == "bounded_anatomical_crop"
