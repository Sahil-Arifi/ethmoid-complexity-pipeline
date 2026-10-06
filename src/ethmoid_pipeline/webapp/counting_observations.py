"""Append-only diagnostic notes; automatic outputs and count revisions are untouched."""
from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any, Literal

import nibabel as nib
import numpy as np
from pydantic import BaseModel, ConfigDict, Field, StrictFloat, StrictInt, StrictStr, ValidationError

from ..util import atomic_write_text, sha256_canonical_json, sha256_file, utc_now
from .counting_service import CountingService
from .service import WebAppError
from .store import decode_json, encode_json


class ObservationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    revision: StrictInt = Field(ge=0)
    kind: Literal["missed", "merged", "extra", "side", "other"]
    candidate_ids: list[StrictInt] = Field(default_factory=list, max_length=64)
    point_ras_mm: tuple[StrictFloat, StrictFloat, StrictFloat] | None = None
    note: StrictStr = Field(min_length=1, max_length=2000)


def _context(service: CountingService, case_id: str, run_id: str):
    case, source = service._source(case_id)
    row = service._run(case_id, run_id)
    if case["source_sha256"] != row["source_sha256"]:
        raise WebAppError("The observation source differs from this count run.", status_code=409)
    return case, source, row


def _directory(service: CountingService, run_id: str) -> Path:
    # Hash the trusted run key so IDs never become filesystem path components.
    root = service.root / "diagnostic-observations"
    directory = root / sha256_canonical_json(run_id)
    if root.is_symlink() or directory.is_symlink():
        raise WebAppError("Diagnostic note storage is unavailable.", status_code=409)
    return service.app._path(directory.relative_to(service.root).as_posix())


def _binding(row: Any, result: dict, hashes: dict) -> dict:
    return {
        "source_sha256": row["source_sha256"],
        "method_version": result["method_version"],
        "protocol_version": row["protocol_version"],
        "protocol_snapshot_sha256": sha256_canonical_json(decode_json(row["protocol_json"], {})),
        "artifact_sha256": hashes,
    }


def _validate_observation(payload: ObservationRequest, result: dict, source: nib.Nifti1Image) -> str:
    ids = payload.candidate_ids
    known = {cell["id"] for cell in result["cells"] if not cell.get("retired", False)}
    note = payload.note.strip()
    if not note or (not ids and payload.point_ras_mm is None):
        raise WebAppError("A diagnostic note needs text and a candidate or volume location.", status_code=422)
    if len(ids) != len(set(ids)) or any(identity not in known for identity in ids):
        raise WebAppError("Candidate IDs must identify distinct cells in the selected revision.", status_code=422)
    if payload.point_ras_mm is not None:
        point = np.asarray(payload.point_ras_mm)
        voxel = nib.affines.apply_affine(np.linalg.inv(source.affine), point)
        if not np.all(np.isfinite(point)) or np.any(voxel < -0.5) or np.any(voxel > np.asarray(source.shape) - 0.5):
            raise WebAppError("The diagnostic location must be inside the source volume.", status_code=422)
    return note


def save_observation(service: CountingService, case_id: str, run_id: str, payload: ObservationRequest) -> dict:
    case, source_path, row = _context(service, case_id, run_id)
    if case["state"] == "APPROVED" or case["review_status"] == "Approved":
        raise WebAppError("Approved cases cannot receive diagnostic notes.", status_code=409)
    result, _, hashes = service._snapshot(row, payload.revision)
    note = _validate_observation(payload, result, nib.load(source_path))
    observation = {
        "schema_version": 1, "id": uuid.uuid4().hex,
        "case_id": case_id, "run_id": run_id,
        **payload.model_dump(mode="json"), "note": note,
        "created_at": utc_now(), **_binding(row, result, hashes),
    }
    # Serializes the approval check with existing workflow writes. No database
    # rows or count artifacts are changed; independent notes get unique files.
    with service.store.connect() as connection:
        connection.execute("BEGIN IMMEDIATE")
        current = service.app._get_case_row(connection, case_id)
        if current["state"] == "APPROVED" or current["review_status"] == "Approved":
            raise WebAppError("Approved cases cannot receive diagnostic notes.", status_code=409)
        if current["source_sha256"] != row["source_sha256"] or sha256_file(source_path) != row["source_sha256"]:
            raise WebAppError("The observation source failed its integrity check.", status_code=409)
        atomic_write_text(_directory(service, row["id"]) / f"{observation['id']}.json", encode_json(observation))
    return observation


def list_observations(service: CountingService, case_id: str, run_id: str) -> dict:
    _, source_path, row = _context(service, case_id, run_id)
    # Also reject unfinished runs, even when no notes have been saved yet.
    original, _, original_hashes = service._snapshot(row, 0)
    directory = _directory(service, row["id"])
    snapshots = {0: (original, _binding(row, original, original_hashes))}
    observations = []
    try:
        source = nib.load(source_path)
        for path in directory.glob("*.json"):
            if path.is_symlink() or not path.is_file() or path.stat().st_size > 32 * 1024:
                raise ValueError("invalid sidecar")
            observation = json.loads(path.read_text(encoding="utf-8"))
            payload = ObservationRequest.model_validate({key: observation[key] for key in ObservationRequest.model_fields})
            if payload.revision not in snapshots:
                result, _, hashes = service._snapshot(row, payload.revision)
                snapshots[payload.revision] = (result, _binding(row, result, hashes))
            result, binding = snapshots[payload.revision]
            _validate_observation(payload, result, source)
            expected = {"schema_version", "id", "case_id", "run_id", "created_at", *ObservationRequest.model_fields, *binding}
            if (set(observation) != expected or observation["schema_version"] != 1 or
                    observation["id"] != path.stem or observation["case_id"] != case_id or
                    observation["run_id"] != run_id or not isinstance(observation["created_at"], str) or
                    any(observation[key] != value for key, value in binding.items())):
                raise ValueError("invalid binding")
            observations.append(observation)
    except (OSError, KeyError, TypeError, ValueError, ValidationError, WebAppError) as exc:
        raise WebAppError("Saved diagnostic notes failed their integrity check.", status_code=409) from exc
    return {"observations": sorted(observations, key=lambda item: (item["created_at"], item["id"]))}
