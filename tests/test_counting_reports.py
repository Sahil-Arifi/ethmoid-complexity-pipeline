"""Synthetic report tests: no source scans or patient identifiers are used."""
from copy import deepcopy
from io import BytesIO
import re

import nibabel as nib
import numpy as np
from PIL import Image
import pytest

from ethmoid_pipeline.webapp.counting_reports import (
    CountReportError,
    PLANE_LABELS,
    _plane_coordinates,
    _sample_labels,
    _sample_trilinear,
    build_count_reports,
)


@pytest.fixture
def synthetic(tmp_path):
    shape = (42, 48, 36)
    affine = np.diag([-0.6, 0.7, 1.0, 1.0])
    affine[:3, 3] = [12, -15, -14]
    grid = np.indices(shape)
    distance = (grid[0] - 21) ** 2 + (grid[1] - 24) ** 2 + (grid[2] - 18) ** 2
    hu = np.where(distance < 16 ** 2, 700, -1000).astype(np.int16)
    cavities = np.zeros(shape, dtype=np.uint16)
    air = np.zeros(shape, dtype=np.uint16)
    points = [(13, 24, 18), (29, 24, 18)]
    cells = []
    for identity, point in enumerate(points, start=1):
        radius = np.sum([(grid[axis] - point[axis]) ** 2 for axis in range(3)], axis=0)
        cavities[radius <= 25] = identity
        air[radius <= 16] = identity
        hu[radius <= 16] = -950
        cells.append({"id": identity, "side": "right" if identity == 1 else "left", "origin": "automatic", "included": True, "uncertain": identity == 2, "centroid_ras_mm": nib.affines.apply_affine(affine, point).tolist()})
    paths = [tmp_path / name for name in ("source.nii.gz", "cavities.nii.gz", "air.nii.gz")]
    for path, data in zip(paths, (hu, cavities, air)):
        image = nib.Nifti1Image(data, affine)
        image.header["descrip"] = b"SYNTHETIC-SECRET-HEADER"
        nib.save(image, path)
    result = {"status": "completed", "revision": 0, "counts": {"left": 1, "right": 1, "total": 2}, "cells": cells, "arbitrary_note": "SYNTHETIC-SECRET-NOTE", "source_path": "SYNTHETIC-SECRET-PATH"}
    return paths, result


def _reports(synthetic, **changes):
    paths, result = synthetic
    return build_count_reports(*paths, {**result, **changes}, case_label="Case099")


def test_valid_documents_and_metadata_allowlist(synthetic):
    artifacts = _reports(synthetic)
    assert set(artifacts) == {"summary.pdf", "representative.png"}
    pdf = artifacts["summary.pdf"]
    assert pdf.startswith(b"%PDF-") and pdf.rstrip().endswith(b"%%EOF")
    assert b"Case099" in pdf and b"Revision 0" in pdf
    assert b"Clinical accuracy is unmeasured" in pdf
    assert b"Cell registry" in pdf and b"Flagged" in pdf
    assert len(re.findall(rb"/Type /Page\b", pdf)) == 2
    png = Image.open(BytesIO(artifacts["representative.png"]))
    assert png.format == "PNG" and png.size == (2100, 1280)
    assert png.mode == "RGB" and not png.info
    for content in artifacts.values():
        assert b"SYNTHETIC-SECRET" not in content
        assert str(synthetic[0][0]).encode() not in content


def test_correction_keeps_original_and_excluded_registry(synthetic):
    paths, result = synthetic
    result = deepcopy(result)
    result["revision"] = 3
    result["original_counts"] = result["counts"]
    result["counts"] = {"left": 0, "right": 1, "total": 1}
    result["corrected_counts"] = result["counts"]
    result["cells"][1]["included"] = False
    result["cells"][1]["retired"] = True
    pdf = build_count_reports(*paths, result, "Case099")["summary.pdf"]
    assert b"Revision 3" in pdf
    assert b"Original: L 1 / R 1 / Total 2" in pdf
    assert b"Current revision: L 0 / R 1 / Total 1" in pdf
    assert b"Retired" in pdf


@pytest.mark.parametrize("classification,text", [
    ("outside_protocol", b"OUT-OF-PROTOCOL EXPERIMENTAL ATTEMPT"),
    ("within_protocol", b"count completeness remain unvalidated"),
    ("unknown", b"acquisition thickness is unknown"),
])
def test_presentation_explains_resolution_without_relaying_arbitrary_text(synthetic, classification, text):
    pdf = _reports(synthetic, quality={"resolution_class": classification, "warnings": ["SYNTHETIC-SECRET"]})["summary.pdf"]
    assert text in pdf
    assert b"SYNTHETIC-SECRET" not in pdf


def test_manual_point_and_empty_included_set(synthetic):
    paths, result = synthetic
    result = deepcopy(result)
    result["cells"].append({**result["cells"][0], "id": 3, "origin": "manual"})
    result["counts"] = {"left": 1, "right": 2, "total": 3}
    assert b"Manual" in build_count_reports(*paths, result, "Case099")["summary.pdf"]
    for cell in result["cells"]:
        cell["included"] = False
    result["counts"] = {"left": 0, "right": 0, "total": 0}
    assert b"No included cells are overlaid" in build_count_reports(*paths, result, "Case099")["summary.pdf"]


def test_registry_paginates_without_dropping_cells(synthetic):
    paths, result = synthetic
    result = deepcopy(result)
    result["cells"].extend({**result["cells"][0], "id": identity, "origin": "manual"} for identity in range(3, 23))
    result["counts"] = {"left": 1, "right": 21, "total": 22}
    pdf = build_count_reports(*paths, result, "Case099")["summary.pdf"]
    assert len(re.findall(rb"/Type /Page\b", pdf)) == 3
    assert b"(22) Tj" in pdf


@pytest.mark.parametrize("plane,axes,labels", [("Axial", (0, 1, 2), ("R", "L", "A", "P")), ("Coronal", (0, 2, 1), ("R", "L", "S", "I")), ("Sagittal", (1, 2, 0), ("A", "P", "S", "I"))])
def test_physical_plane_directions_and_equal_pixel_spacing(plane, axes, labels):
    center = np.array([5.0, 11.0, 19.0])
    bounds = np.array([[-10, 20], [-7, 33], [-1, 39]], dtype=float)
    grid, spacing = _plane_coordinates(plane, center, bounds, 120, 90)
    horizontal, vertical, fixed = axes
    assert PLANE_LABELS[plane] == labels
    assert np.allclose(np.diff(grid[..., horizontal], axis=1), -spacing)
    assert np.allclose(np.diff(grid[..., vertical], axis=0), -spacing)
    assert np.allclose(grid[..., fixed], center[fixed])
    assert grid.shape[0] <= 90 and grid.shape[1] <= 120


@pytest.mark.parametrize("orientation", ["native", "permuted", "oblique"])
def test_ct_interpolation_uses_world_geometry_for_native_and_oblique_grids(orientation):
    affine = np.diag([-0.6, 0.75, 1.2, 1.0])
    if orientation == "permuted":
        affine[:3, :3] = affine[:3, :3][:, [2, 0, 1]]
    elif orientation == "oblique":
        angle = np.radians(17)
        affine[:3, :3] = np.array([[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]]) @ affine[:3, :3]
    affine[:3, 3] = [30, -12, 5]
    index_grid = np.moveaxis(np.indices((10, 12, 14)), 0, -1)
    world = nib.affines.apply_affine(affine, index_grid)
    # A linear world-space intensity field must be reproduced by trilinear
    # sampling regardless of native axis order, signs, anisotropy or rotation.
    data = (world[..., 0] * 3 + world[..., 1] * 5 - world[..., 2] * 2).astype(np.float32)
    sample_indices = np.array([[[2.35, 4.7, 6.2], [4.2, 3.8, 8.25]]])
    sample_world = nib.affines.apply_affine(affine, sample_indices)
    recovered_indices = nib.affines.apply_affine(np.linalg.inv(affine), sample_world)
    expected = sample_world[..., 0] * 3 + sample_world[..., 1] * 5 - sample_world[..., 2] * 2
    assert np.allclose(_sample_trilinear(data, recovered_indices), expected, atol=1e-5)
    assert _sample_trilinear(data, np.array([[[-1.0, 0, 0]]]))[0, 0] == -1024
    labels = np.zeros(data.shape, dtype=np.uint16)
    labels[2, 5, 6] = 9
    assert _sample_labels(labels, recovered_indices)[0, 0] == 9
    assert _sample_labels(labels, np.array([[[-1.0, 0, 0]]]))[0, 0] == 0


@pytest.mark.parametrize("change", ["failed", "counts", "corrected", "duplicate", "retired", "coordinate", "unknown_side", "implicit_boolean"])
def test_inconsistent_results_are_rejected(synthetic, change):
    paths, result = synthetic
    result = deepcopy(result)
    if change == "failed":
        result["status"] = "failed"
    elif change == "counts":
        result["counts"] = {"left": 1, "right": 0, "total": 1}
    elif change == "corrected":
        result["corrected_counts"] = {"left": 0, "right": 1, "total": 1}
    elif change == "duplicate":
        result["cells"][1]["id"] = 1
    elif change == "retired":
        result["cells"][0]["retired"] = True
    elif change == "coordinate":
        result["cells"][0]["centroid_ras_mm"] = [9999, 0, 0]
    elif change == "unknown_side":
        result["cells"][0]["side"] = "unknown"
    else:
        result["cells"][0]["included"] = 1
    with pytest.raises(CountReportError):
        build_count_reports(*paths, result, "Case099")


@pytest.mark.parametrize("change", ["shape", "affine", "unknown_label", "fractional_label", "air_outside_cavity", "nonfinite_ct"])
def test_invalid_native_volumes_are_rejected(synthetic, change):
    paths, result = synthetic
    target = paths[0] if change == "nonfinite_ct" else paths[2] if change == "air_outside_cavity" else paths[1]
    volume = nib.load(target)
    data = volume.get_fdata().astype(np.float32)
    affine = volume.affine.copy()
    if change == "shape":
        data = data[:-1]
    elif change == "affine":
        affine[0, 3] += 0.1
    elif change == "unknown_label":
        data[1, 1, 1] = 99
    elif change == "fractional_label":
        data[1, 1, 1] = 0.5
    elif change == "air_outside_cavity":
        data[1, 1, 1] = 1
    else:
        data[1, 1, 1] = np.nan
    nib.save(nib.Nifti1Image(data, affine), target)
    with pytest.raises(CountReportError):
        build_count_reports(*paths, result, "Case099")


def test_label_and_missing_file_errors_do_not_expose_private_details(synthetic):
    paths, result = synthetic
    with pytest.raises(CountReportError, match="neutral Case"):
        build_count_reports(*paths, result, "NON-NEUTRAL-LABEL")
    missing = paths[0].parent / "SYNTHETIC-SECRET-MISSING.nii.gz"
    with pytest.raises(CountReportError) as error:
        build_count_reports(missing, *paths[1:], result, "Case099")
    assert "SYNTHETIC-SECRET" not in str(error.value)
