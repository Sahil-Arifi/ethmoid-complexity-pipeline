from __future__ import annotations

import json
import re
import tomllib
from pathlib import Path

from ethmoid_pipeline.guard import _allowed_path, _blocked_text_content, _medical_magic
from ethmoid_pipeline.webapp.protocol import load_protocol


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_DOCUMENT = ROOT / "docs" / "ANNOTATION_PROTOCOL_V2_DRAFT.md"
REVIEW_FORM = ROOT / "docs" / "PROTOCOL_V2_CLINICAL_REVIEW.md"
DRAFT_CONFIG = (
    ROOT / "src" / "ethmoid_pipeline" / "webapp" / "protocol.v2.draft.json"
)
VERSION = "ethmoid-air-cell-v2-draft.1"
BLOCKER_IDS = {f"B{index:02d}" for index in range(1, 16)}


def test_v2_draft_is_complete_aligned_and_inactive() -> None:
    config = json.loads(DRAFT_CONFIG.read_text(encoding="utf-8"))

    assert config["schema_version"] == 2
    assert config["protocol_version"] == VERSION
    assert config["lifecycle"]["status"] == "DRAFT_FOR_CLINICAL_REVIEW"
    assert config["lifecycle"]["runtime_enabled"] is False
    assert config["lifecycle"]["training_eligibility"] == "NOT_FOR_TRAINING"
    assert config["lifecycle"]["current_runtime_protocol"] == "ethmoid-provisional-v1"
    assert config["lifecycle"]["runtime_implementation_status"] == "NOT_IMPLEMENTED"
    assert {
        item["id"] for item in config["unresolved_clinical_decisions"]
    } == BLOCKER_IDS
    assert config["decision_gate"]["requires_all_decisions_resolved"] is True
    assert (
        config["decision_gate"]["parent_resolution_requires_all_review_form_fields"]
        is True
    )

    for path in (PROTOCOL_DOCUMENT, REVIEW_FORM):
        text = path.read_text(encoding="utf-8")
        assert VERSION in text
        assert "DRAFT_FOR_CLINICAL_REVIEW" in text
        assert "NOT_FOR_TRAINING" in text
        assert set(re.findall(r"B(?:0[1-9]|1[0-5])", text)) == BLOCKER_IDS

    assert load_protocol()["protocol_version"] == "ethmoid-provisional-v1"


def test_v2_draft_has_a_narrow_guard_and_package_boundary() -> None:
    relative_path = "src/ethmoid_pipeline/webapp/protocol.v2.draft.json"
    assert _allowed_path(relative_path)
    assert not _allowed_path("src/ethmoid_pipeline/webapp/protocol.v2.json")
    assert not _allowed_path("docs/unreviewed-protocol.json")

    for path in (PROTOCOL_DOCUMENT, REVIEW_FORM, DRAFT_CONFIG):
        blob = path.read_bytes()
        assert _medical_magic(blob) is None
        assert _blocked_text_content(blob) is None

    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    package_data = pyproject["tool"]["setuptools"]["package-data"]
    assert package_data["ethmoid_pipeline.webapp"] == ["protocol.v1.json"]
