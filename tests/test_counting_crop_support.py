"""Processing crops must preserve the physical context used for bone support."""
import hashlib
import json

import nibabel as nib
import numpy as np
import pytest

pytest.importorskip("scipy")
pytest.importorskip("skimage")

from scipy import ndimage as ndi

from ethmoid_pipeline.counting.engine import METHOD_VERSION, CountingConfig, analyze_volume, compartment_stage
from ethmoid_pipeline.counting.localization import Localization


@pytest.mark.parametrize("fine_axis", [0, 1, 2])
@pytest.mark.parametrize("flipped", [False, True])
def test_processing_crop_preserves_full_field_bone_support(tmp_path, monkeypatch, fine_axis, flipped):
    import ethmoid_pipeline.counting.localization as localization

    hu = np.full((64, 96, 96), 800, dtype=np.float32)
    hu[8:34, 8:88, 8:88] = -900
    membership = np.zeros_like(hu, dtype=bool)
    membership[:20] = True
    order = list(range(3))
    order[0], order[fine_axis] = order[fine_axis], order[0]
    hu, membership = hu.transpose(order), membership.transpose(order)
    spacing = np.array([0.25, 0.5, 0.5])[order]
    affine = np.diag([*spacing, 1.0])
    affine[:3, 3] = -np.array(hu.shape) * spacing / 2
    search = ndi.distance_transform_edt(~membership, sampling=spacing) <= CountingConfig().boundary_growth_mm
    expected_trace = {}
    expected_cavities, expected_air, expected_cells, _ = compartment_stage(
        hu, membership, affine, reconstruction_region=search, trace=expected_trace)
    assert len(expected_cells) == 1

    source_affine = affine.copy()
    source_hu = hu
    if flipped:
        transform = np.eye(4)
        transform[0, 0], transform[0, 3] = -1, hu.shape[0] - 1
        source_affine, source_hu = affine @ transform, hu[::-1]
    source = tmp_path / "synthetic.nii.gz"
    nib.save(nib.Nifti1Image(source_hu, source_affine), source)
    monkeypatch.setattr(localization, "load_landmarks", lambda *a, **k: {})
    monkeypatch.setattr(localization, "localize_native", lambda *a, **k: Localization(
        membership, np.zeros_like(membership), {}, 0.0, []))

    output = tmp_path / "output"
    acquisition = {"source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                   "slice_thickness_mm": 0.5}
    result = analyze_volume(source, output, tmp_path / "bounds", acquisition=acquisition, diagnostic_stages=True)
    assert result["status"] == "completed"
    assert result["counting_engine_version"] == METHOD_VERSION
    assert result["quality"]["flags"] == []
    assert result["cells"] == expected_cells
    for name, expected in [("cavities", expected_cavities), ("air", expected_air)]:
        actual = nib.as_closest_canonical(nib.load(output / f"{name}.nii.gz"))
        assert np.array_equal(actual.affine, affine)
        assert np.array_equal(np.asanyarray(actual.dataobj), expected)
    trace = json.loads((output / "diagnostic-stages.local.json").read_text())
    assert trace["proposals"] == expected_trace["proposals"]
