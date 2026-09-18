from __future__ import annotations

import pytest

from ethmoid_pipeline.counting.profiles import build_scan_profile
from ethmoid_pipeline.webapp.store import decode_json, encode_json
from tests.test_catalog import catalog, write_verified_case


@pytest.fixture
def legacy_context(catalog, tmp_path):
    service, native, _ = catalog
    case = write_verified_case(service, native, "Case099", tmp_path)
    service.sync()
    case_id = case["case_id"]
    # Distinct retained values make accidental descriptor-wide replacement
    # visible; only the two previously absent categories may be added.
    acquisition = {
        "slice_thickness_mm": 2.5,
        "spacing_mm": [0.8, 0.8, 1.25],
        "resolution_class": "outside_protocol",
        "warnings": ["Retained acquisition warning."],
    }
    with service.store.connect() as connection:
        connection.execute("UPDATE catalog_case_links SET acquisition_json=? WHERE case_id=?", (encode_json(acquisition), case_id))
        row = connection.execute("SELECT * FROM catalog_series WHERE case_id=?", (case_id,)).fetchone()
        descriptor = decode_json(row["descriptor_json"], {})
        assert descriptor["metadata_validated"] is True
        assert descriptor["source_kind"] == "native"
        assert descriptor["kernel"] == "BONEPLUS"
        # Preserve an old completed attempt as evidence that context reads do
        # not retrofit categories into stored run results or acquisition rows.
        connection.execute("""INSERT INTO count_runs(
            id,case_id,status,source_sha256,protocol_version,protocol_json,
            output_relpath,original_result_json,created_at,updated_at)
            SELECT 'historical-run',id,'completed',source_sha256,'v2','{}',
                'historical-run',?,'2026-01-01','2026-01-01' FROM cases WHERE id=?""",
            (encode_json({"acquisition": acquisition, "counts": {"total": 3}}), case_id))
    return service.app, case_id, row["id"], acquisition, descriptor


def database_snapshot(app):
    with app.store.connect() as connection:
        return tuple(connection.iterdump())


@pytest.mark.parametrize(("kind", "kernel", "expected_kind", "expected_kernel"), [
    ("native", "BONEPLUS", "native", "bone"),
    ("mpr", "STANDARD", "mpr", "soft"),
    ("unknown", "Other", "unknown", "unknown"),
])
def test_current_legacy_profile_uses_verified_source_categories_only(
    legacy_context, kind, kernel, expected_kind, expected_kernel,
):
    app, case_id, series_id, acquisition, descriptor = legacy_context
    descriptor.update(source_kind=kind, kernel=kernel)
    with app.store.connect() as connection:
        connection.execute("UPDATE catalog_series SET descriptor_json=? WHERE id=?", (encode_json(descriptor), series_id))
    before = database_snapshot(app)
    for _ in range(2):
        context = app.catalog_context(case_id)
        assert {key: context[key] for key in acquisition} == acquisition
        assert context["reconstruction_kind"] == expected_kind
        assert context["kernel_class"] == expected_kernel
        profile = build_scan_profile(app.volume_path(case_id), context)
        assert profile["reconstruction_kind"] == expected_kind
        assert profile["kernel_class"] == expected_kernel
        assert profile["slice_thickness_mm"] == acquisition["slice_thickness_mm"]
        assert profile["slice_interval_mm"] == acquisition["spacing_mm"][2]
    assert database_snapshot(app) == before


@pytest.mark.parametrize("untrusted", [
    "obsolete_case", "source_mismatch", "empty_fingerprints", "not_ready", "not_present",
    "unverified", "validation_missing", "validation_string", "validation_number",
])
def test_legacy_context_does_not_enrich_untrusted_source(legacy_context, untrusted):
    app, case_id, series_id, acquisition, descriptor = legacy_context
    with app.store.connect() as connection:
        if untrusted == "obsolete_case":
            connection.execute("UPDATE catalog_series SET case_id=NULL WHERE id=?", (series_id,))
        elif untrusted == "source_mismatch":
            connection.execute("UPDATE catalog_series SET source_fingerprint=? WHERE id=?", ("f" * 64, series_id))
        elif untrusted == "empty_fingerprints":
            connection.execute("UPDATE catalog_series SET source_fingerprint='' WHERE id=?", (series_id,))
            connection.execute("UPDATE catalog_case_links SET source_fingerprint='' WHERE case_id=?", (case_id,))
        elif untrusted == "not_ready":
            connection.execute("UPDATE catalog_series SET preparation_status='queued' WHERE id=?", (series_id,))
        elif untrusted == "not_present":
            connection.execute("UPDATE catalog_series SET present=0 WHERE id=?", (series_id,))
        else:
            descriptor.pop("metadata_validated")
            if untrusted != "validation_missing":
                descriptor["metadata_validated"] = {"unverified": False, "validation_string": "true", "validation_number": 1}[untrusted]
            connection.execute("UPDATE catalog_series SET descriptor_json=? WHERE id=?", (encode_json(descriptor), series_id))
    before = database_snapshot(app)
    context = app.catalog_context(case_id)
    assert context["reconstruction_kind"] == context["kernel_class"] == "unknown"
    for key in ("slice_thickness_mm", "spacing_mm", "resolution_class"):
        assert context[key] == acquisition[key]
    assert context["warnings"][0] == acquisition["warnings"][0]
    assert database_snapshot(app) == before


@pytest.mark.parametrize(("key", "value", "other", "expected_other"), [
    ("reconstruction_kind", "mpr", "kernel_class", "bone"),
    ("kernel_class", "soft", "reconstruction_kind", "native"),
    ("kernel_class", "unknown", "reconstruction_kind", "native"),
])
def test_existing_categories_are_preserved_while_missing_categories_are_filled(
    legacy_context, key, value, other, expected_other,
):
    app, case_id, _, acquisition, _ = legacy_context
    acquisition[key] = value
    with app.store.connect() as connection:
        connection.execute("UPDATE catalog_case_links SET acquisition_json=? WHERE case_id=?", (encode_json(acquisition), case_id))
    before = database_snapshot(app)
    context = app.catalog_context(case_id)
    assert context[key] == value
    assert context[other] == expected_other
    assert database_snapshot(app) == before
