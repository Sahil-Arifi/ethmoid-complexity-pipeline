from __future__ import annotations

import json
from pathlib import Path

import nibabel as nib
import numpy as np

from ethmoid_pipeline.privacy import (
    redact_text,
    sanitize_sidecar,
    scrub_nifti_header,
    verify_sanitized_sidecar,
)


def test_sidecar_is_rewritten_from_allowlist(tmp_path: Path) -> None:
    path = tmp_path / "pilot.json"
    path.write_text(
        json.dumps(
            {
                "Modality": "CT",
                "SliceThickness": 0.625,
                "ConversionSoftware": "dcm2niix",
                "ConversionSoftwareVersion": "v1.0.20260724",
                "ImageOrientationPatientDICOM": [1, 0, 0, 0, 1, 0],
                "PatientName": "SYNTHETIC^ONLY",
                "SeriesInstanceUID": "9.9.9.9.9",
            }
        ),
        encoding="utf-8",
    )
    sanitized = sanitize_sidecar(path)
    assert sanitized == {
        "schema_version": 1,
        "Modality": "CT",
        "SliceThickness": 0.625,
        "ConversionSoftware": "dcm2niix",
        "ConversionSoftwareVersion": "v1.0.20260724",
        "ImageOrientationPatientDICOM": [1, 0, 0, 0, 1, 0],
    }
    assert verify_sanitized_sidecar(path) == []


def test_nifti_privacy_fields_are_scrubbed_without_changing_voxels(tmp_path: Path) -> None:
    path = tmp_path / "pilot.nii"
    data = np.arange(60, dtype=np.int16).reshape(3, 4, 5)
    image = nib.Nifti1Image(data, np.eye(4))
    image.header["descrip"] = b"SYNTHETIC^ONLY"
    image.header["aux_file"] = b"SENSITIVE"
    nib.save(image, path)

    scrub_nifti_header(path)
    reloaded = nib.load(path)
    assert np.array_equal(np.asanyarray(reloaded.dataobj), data)
    assert bytes(reloaded.header["aux_file"]).rstrip(b"\x00") == b""
    assert bytes(reloaded.header["intent_name"]).rstrip(b"\x00") == b""
    assert bytes(reloaded.header["data_type"]).rstrip(b"\x00") == b""
    assert bytes(reloaded.header["db_name"]).rstrip(b"\x00") == b""


def test_sidecar_rejects_arbitrary_text_in_allowed_field(tmp_path: Path) -> None:
    path = tmp_path / "pilot.json"
    path.write_text(
        json.dumps(
            {
                "Modality": "SYNTHETIC PERSON",
                "ConversionSoftware": "dcm2niix",
                "ConversionSoftwareVersion": "v1.0.20260724",
                "ImageOrientationPatientDICOM": [1, 0, 0, 0, 1, 0],
            }
        ),
        encoding="utf-8",
    )
    import pytest

    from ethmoid_pipeline.util import PipelineError

    with pytest.raises(PipelineError, match="Modality"):
        sanitize_sidecar(path)


def test_log_redaction_removes_unanticipated_windows_path_and_email() -> None:
    separator = chr(92)
    path = "C:" + separator + "SyntheticUser" + separator + "tools" + separator + "helper.exe"
    forward_unc = "/" * 2 + "synthetic-host/share/private-file"
    device_path = separator * 2 + "." + separator + "SyntheticDevice"
    redacted = redact_text(
        "Compression helper: "
        + path
        + "\nNetwork source: "
        + forward_unc
        + "\nDevice source: "
        + device_path
        + "\nPublic URL: https://example.invalid/resource"
        + "\nContact: synthetic.person@example.invalid",
        sensitive_values=(),
    )
    assert redacted == (
        "Compression helper: <LOCAL_PATH>"
        "\nNetwork source: <LOCAL_PATH>"
        "\nDevice source: <LOCAL_PATH>"
        "\nPublic URL: https://example.invalid/resource"
        "\nContact: <EMAIL_REDACTED>"
    )
    assert "SyntheticUser" not in redacted
