from __future__ import annotations

import hashlib
import json
from importlib.resources import files

from ethmoid_pipeline.counting.protocol import load_counting_protocol
from ethmoid_pipeline.guard import _allowed_path, _blocked_text_content, _medical_magic
from ethmoid_pipeline.webapp.protocol import load_protocol


def test_counting_contract_preserves_full_volume_and_opaque_cell_requirements():
    contract = load_counting_protocol()
    assert contract["count_opacified_cells"] is True
    assert contract["air_mask_required_for_each_cell"] is False
    assert contract["reference_counts_required_for_inference"] is False
    assert contract["reference_counts_required_for_accuracy_claim"] is True
    assert contract["first_result"] == "fully_automatic_without_patient_specific_prompts"
    assert contract["failure_policy"] == "no_numeric_count_on_failed_localization_or_geometry"
    assert set(contract["included_variants"]) == {
        "agger_nasi", "haller", "onodi", "frontal_region", "supraorbital_ethmoid", "concha_bullosa"
    }
    assert load_protocol()["protocol_version"] == "ethmoid-provisional-v1"


def test_counting_contract_hash_binds_bytes_and_returns_independent_copies():
    resource = files("ethmoid_pipeline.counting").joinpath("protocol.json").read_bytes()
    contract = load_counting_protocol()
    assert contract["sha256"] == hashlib.sha256(resource).hexdigest()
    contract["included_variants"].clear()
    assert load_counting_protocol()["included_variants"]
    assert json.loads(resource)["status"] == "EXPERIMENTAL_NOT_CLINICALLY_VALIDATED"


def test_v2_broad_attempts_preserve_clinician_criterion_and_original_contract():
    current = load_counting_protocol()
    original = load_counting_protocol("ethmoid-count-prototype-v1")
    assert current["protocol_version"] == "ethmoid-count-prototype-v2"
    assert current["maximum_slice_thickness_mm"] == original["maximum_slice_thickness_mm"] == 1.0
    assert current["algorithm"] == original["algorithm"]
    assert current["localization_grid"]["counting_grid"] == "original_native_no_instance_resampling"
    assert "attempt_policy" not in original
    assert original["sha256"] == hashlib.sha256(files("ethmoid_pipeline.counting").joinpath("protocol.v1.json").read_bytes()).hexdigest()


def test_new_config_allowlist_does_not_admit_data_or_arbitrary_json():
    assert _allowed_path("src/ethmoid_pipeline/counting/protocol.json")
    assert _allowed_path(".github/workflows/verify.yml")
    assert not _allowed_path("src/ethmoid_pipeline/counting/patient.json")
    assert not _allowed_path(".github/workflows/patient.json")
    resource = files("ethmoid_pipeline.counting").joinpath("protocol.json").read_bytes()
    assert _medical_magic(resource) is None
    assert _blocked_text_content(resource) is None
