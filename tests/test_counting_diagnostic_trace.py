"""Optional stage evidence must leave predictions and default outputs unchanged."""
import json

import nibabel as nib
import numpy as np
import pytest

pytest.importorskip("scipy")
pytest.importorskip("skimage")

from ethmoid_pipeline.counting.engine import analyze_volume, compartment_stage
from ethmoid_pipeline.counting.localization import Localization


@pytest.mark.parametrize("only_pore", [False, True])
def test_trace_preserves_predictions_and_binds_crop_to_source(tmp_path, monkeypatch, only_pore):
    import ethmoid_pipeline.counting.localization as localization

    hu = np.full((64, 56, 52), 800, dtype=np.float32)
    if not only_pore:
        hu[18:30, 18:34, 16:33] = -900
        hu[34:46, 18:34, 16:33] = -900
    hu[32, 18:36, 16:34] = -900  # A one-voxel-thick pore with sufficient HU-range support.
    region = np.zeros(hu.shape, bool)
    region[16:48, 16:36, 14:35] = True
    affine = np.diag([-0.6, 0.6, 0.6, 1.0])
    affine[:3, 3] = [20, -16, -15]
    source = tmp_path / "synthetic.nii.gz"
    nib.save(nib.Nifti1Image(hu, affine), source)
    canonical_region = region[::-1]
    monkeypatch.setattr(localization, "load_landmarks", lambda *a, **k: {})
    monkeypatch.setattr(localization, "localize_native", lambda *a, **k: Localization(
        canonical_region, np.zeros(hu.shape, bool), {}, 0.0, []))
    plain = analyze_volume(source, tmp_path / "plain", tmp_path / "bounds")
    traced = analyze_volume(source, tmp_path / "traced", tmp_path / "bounds", diagnostic_stages=True)
    assert plain == traced
    assert plain["status"] == ("failed" if only_pore else "completed")
    assert not list((tmp_path / "plain").glob("diagnostic-stages*"))
    for name in ("cavities.nii.gz", "air.nii.gz", "region.nii.gz"):
        if (tmp_path / "plain" / name).exists():
            assert (tmp_path / "plain" / name).read_bytes() == (tmp_path / "traced" / name).read_bytes()
    metadata = json.loads((tmp_path / "traced" / "diagnostic-stages.local.json").read_text())
    assert metadata["source_sha256"] == plain["source_sha256"]
    assert metadata["config"] == plain["config"]
    assert metadata["method_version"] == plain["method_version"]
    assert any(p["rejection"] == "unresolved_interior" for p in metadata["proposals"])
    assert {p["cell_id"] for p in metadata["proposals"] if p["rejection"] is None} == (set() if only_pore else {1, 2})
    assert all(start > 0 for start, stop in metadata["crop_bounds"])
    with np.load(tmp_path / "traced" / "diagnostic-stages.local.npz", allow_pickle=False) as arrays:
        assert {"membership", "excluded", "search", "markers", "watershed", "merged", "affine"} == set(arrays.files)
        bounds = tuple(slice(*b) for b in metadata["crop_bounds"])
        assert np.array_equal(arrays["membership"], canonical_region[bounds])
        native = nib.load(source)
        canonical = nib.as_closest_canonical(native)
        expected = canonical.affine.copy()
        expected[:3, 3] = nib.affines.apply_affine(canonical.affine, [b[0] for b in metadata["crop_bounds"]])
        assert np.allclose(arrays["affine"], expected)
        assert np.count_nonzero(arrays["markers"]) > 0
        assert arrays["watershed"].shape == arrays["merged"].shape == arrays["membership"].shape


@pytest.mark.parametrize("aperture, expected_cells", [(5, 2), (12, 1)])
def test_trace_distinguishes_separator_merging_from_seed_loss(aperture, expected_cells):
    hu = np.full((60, 44, 40), 800, dtype=np.float32)
    hu[6:52, 9:35, 7:33] = 35
    hu[12, 9:35, 7:33] = 800
    hu[12, 22-aperture//2:22+(aperture+1)//2, 20-aperture//2:20+(aperture+1)//2] = 35
    hu[7:11, 12:32, 10:30] = -900
    hu[20:44, 12:32, 10:30] = -900
    trace = {}
    _, _, cells, _ = compartment_stage(hu, np.ones_like(hu, bool), np.diag([0.6, 0.6, 0.6, 1]), trace=trace)
    assert trace["seed_counts"]["initial_markers"] == 2
    assert set(np.unique(trace["arrays"]["watershed"])) == {0, 1, 2}
    assert len(cells) == expected_cells
    assert trace["interfaces"][0]["preserved"] == (expected_cells == 2)
    assert bool(trace["merge_decisions"]) == (expected_cells == 1)
