from __future__ import annotations

import json
import math
import os
import re
import shutil
import tempfile
import uuid
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import nibabel as nib
import numpy as np

from ..util import default_dcm2niix, default_work_root, sha256_file, utc_now
from .protocol import allowed_label_values, load_protocol, protocol_version
from .store import Store, decode_json, encode_json


CASE_LABEL_RE = re.compile(r"^Case[0-9]{3}$")
CASE_STATES = {
    "IMPORTING",
    "VALIDATION_FAILED",
    "READY",
    "ANNOTATING",
    "READY_FOR_REVIEW",
    "NEEDS_CHANGES",
    "APPROVED",
}
REVIEW_STATUSES = {"Draft", "Ready for review", "Needs changes", "Approved"}
QC_KEYS = ("coverage", "orientation", "image_quality", "slice_continuity")
GENERIC_QC_CHECK_NAMES = frozenset(
    {
        "voxel count",
        "qform and sform present",
        "qform and sform match",
        "physical spacing",
        "DICOM-to-NIfTI physical coordinates",
        "full field-of-view extent",
        "slice sequence",
        "NIfTI header privacy allowlist",
        "sidecar privacy allowlist",
        "source immutability",
        "approved selection binding",
        "pipeline provenance",
        "full-volume HU preservation",
    }
)
PILOT_QC_CHECK_NAMES = GENERIC_QC_CHECK_NAMES | {"NIfTI dtype and scaling"}


class WebAppError(RuntimeError):
    def __init__(self, message: str, *, status_code: int = 400):
        super().__init__(message)
        self.status_code = status_code


@dataclass(frozen=True)
class VolumeInfo:
    shape: tuple[int, int, int]
    spacing: tuple[float, float, float]
    affine: tuple[tuple[float, float, float, float], ...]
    warnings: tuple[str, ...]


def inspect_nifti(path: Path) -> VolumeInfo:
    try:
        image = nib.load(path, mmap=True)
    except Exception as exc:
        raise WebAppError("The NIfTI file could not be opened.", status_code=422) from exc
    if len(image.shape) != 3 or any(int(item) <= 0 for item in image.shape):
        raise WebAppError("The NIfTI volume must be a non-empty 3D image.", status_code=422)
    affine = np.asarray(image.affine, dtype=np.float64)
    if affine.shape != (4, 4) or not np.all(np.isfinite(affine)):
        raise WebAppError("The NIfTI affine is missing or non-finite.", status_code=422)
    if not math.isfinite(float(np.linalg.det(affine[:3, :3]))) or abs(
        float(np.linalg.det(affine[:3, :3]))
    ) < 1e-8:
        raise WebAppError("The NIfTI affine is singular.", status_code=422)
    spacing = tuple(float(item) for item in image.header.get_zooms()[:3])
    if len(spacing) != 3 or not all(math.isfinite(item) and item > 0 for item in spacing):
        raise WebAppError("The NIfTI voxel spacing is invalid.", status_code=422)

    # Force every stored plane to decode. This catches truncated gzip streams and
    # non-finite volume data without permanently retaining a second full volume.
    for index in range(int(image.shape[2])):
        try:
            plane = np.asanyarray(image.dataobj[:, :, index])
        except Exception as exc:
            raise WebAppError("The NIfTI voxel data is unreadable.", status_code=422) from exc
        if not np.all(np.isfinite(plane)):
            raise WebAppError("The NIfTI volume contains non-finite values.", status_code=422)

    warnings: list[str] = []
    qform, qcode = image.get_qform(coded=True)
    sform, scode = image.get_sform(coded=True)
    if int(qcode) == 0 or int(scode) == 0:
        warnings.append("qform and sform are not both explicitly defined")
    elif not np.allclose(qform, sform, rtol=0, atol=1e-5):
        warnings.append("qform and sform do not match")
    return VolumeInfo(
        shape=tuple(int(item) for item in image.shape),
        spacing=spacing,
        affine=tuple(tuple(float(item) for item in row) for row in affine),
        warnings=tuple(warnings),
    )


def _empty_checklist() -> dict[str, bool]:
    return {key: False for key in QC_KEYS}


def _atomic_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    handle, name = tempfile.mkstemp(prefix=f".{destination.name}.", dir=destination.parent)
    os.close(handle)
    temporary = Path(name)
    try:
        shutil.copyfile(source, temporary)
        with temporary.open("r+b") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_save_nifti(image: nib.Nifti1Image, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    handle, name = tempfile.mkstemp(
        prefix=f".{destination.stem}.", suffix=".nii.gz", dir=destination.parent
    )
    os.close(handle)
    temporary = Path(name)
    try:
        nib.save(image, temporary)
        with temporary.open("r+b") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _read_qc_report(
    path: Path | None,
    source_hash: str,
    volume_info: VolumeInfo,
    expected_case_label: str,
) -> tuple[str, dict[str, Any]]:
    if path is None:
        return "UNVERIFIED", {"reason": "No matching automated QC report was supplied."}
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise WebAppError("The supplied QC report is not valid JSON.", status_code=422) from exc
    if not isinstance(report, dict):
        return "UNVERIFIED", {"reason": "The supplied QC report is not an object."}
    metrics = report.get("metrics")
    reported_hash = metrics.get("nifti_sha256") if isinstance(metrics, dict) else None
    status = report.get("status")
    checks = report.get("checks")
    check_names = (
        [item.get("name") for item in checks if isinstance(item, dict)]
        if isinstance(checks, list)
        else []
    )
    expected_check_set = frozenset(check_names) in {
        GENERIC_QC_CHECK_NAMES,
        PILOT_QC_CHECK_NAMES,
    }
    checks_pass = (
        isinstance(checks, list)
        and len(check_names) == len(checks)
        and len(check_names) == len(set(check_names))
        and expected_check_set
        and all(item.get("passed") is True for item in checks if isinstance(item, dict))
    )
    expected_voxels = int(np.prod(volume_info.shape))
    try:
        max_hu_error = float(metrics.get("max_absolute_hu_error"))
        metric_contract = (
            metrics.get("nifti_shape") == list(volume_info.shape)
            and int(metrics.get("hu_voxel_count")) == expected_voxels
            and int(metrics.get("hu_nonfinite_count")) == 0
            and math.isfinite(max_hu_error)
            and max_hu_error <= 1.0
        )
    except (AttributeError, TypeError, ValueError):
        metric_contract = False
    report_contract = (
        report.get("schema_version") == 1
        and status == "AUTOMATED_PASS_PENDING_MANUAL"
        and report.get("case") == expected_case_label
        and isinstance(report.get("series"), str)
        and re.fullmatch(r"Series[0-9]{3}", report["series"]) is not None
        and report.get("issues") == []
        and report.get("manual_slicer_qc") in {"PENDING", "PASS"}
        and report.get("clinical_series_and_coverage_review") in {"PENDING", "PASS"}
    )
    if (
        reported_hash != source_hash
        or not report_contract
        or not checks_pass
        or not metric_contract
    ):
        return "UNVERIFIED", {
            "reason": "The supplied QC report did not validate against this NIfTI file.",
            "report_status": status,
            "hash_matches": reported_hash == source_hash,
            "checks_pass": checks_pass,
            "report_contract_matches": report_contract,
            "metrics_match": metric_contract,
        }
    return "VERIFIED", {
        "report_status": status,
        "hash_matches": True,
        "checks_pass": True,
        "report_contract_matches": True,
        "metrics_match": True,
    }


def _encode_viewer_state(state: dict[str, Any] | None) -> str:
    value = state or {}
    if not isinstance(value, dict):
        raise WebAppError("Viewer state must be a JSON object.", status_code=422)
    try:
        encoded = encode_json(value)
    except (TypeError, ValueError) as exc:
        raise WebAppError("Viewer state must contain valid JSON values.", status_code=422) from exc
    if len(encoded.encode("utf-8")) > 256 * 1024:
        raise WebAppError("Viewer state is too large.", status_code=422)
    return encoded


class AppService:
    def __init__(self, work_root: Path | None = None, dcm2niix: Path | None = None):
        self.root = (work_root or (default_work_root() / "webapp")).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.store = Store(self.root)
        self.dcm2niix = (dcm2niix or default_dcm2niix()).resolve()
        for name in ("cases", "imports", "incoming", "exports"):
            (self.root / name).mkdir(exist_ok=True)

    def _path(self, relative: str) -> Path:
        candidate = (self.root / relative).resolve()
        try:
            candidate.relative_to(self.root)
        except ValueError as exc:
            raise RuntimeError("Stored application path escaped the work root.") from exc
        return candidate

    def _relative(self, path: Path) -> str:
        return path.resolve().relative_to(self.root).as_posix()

    def _normalize_label(self, value: str | None) -> str:
        if value is not None and value.strip():
            label = value.strip()
            if not CASE_LABEL_RE.fullmatch(label):
                raise WebAppError("Case labels must use the neutral form Case###.", status_code=422)
            return label
        with self.store.connect() as connection:
            labels = {
                row[0]
                for row in connection.execute("SELECT case_label FROM cases").fetchall()
            }
            labels.update(
                row[0]
                for row in connection.execute(
                    "SELECT case_label FROM imports WHERE status != 'CONVERTED'"
                ).fetchall()
            )
        for number in range(1, 1_000):
            candidate = f"Case{number:03d}"
            if candidate not in labels:
                return candidate
        raise WebAppError("No neutral case labels remain available.", status_code=409)

    def _assert_label_available(self, label: str, *, exclude_import_id: str | None = None) -> None:
        with self.store.connect() as connection:
            if connection.execute(
                "SELECT 1 FROM cases WHERE case_label = ?", (label,)
            ).fetchone():
                raise WebAppError("That neutral case label already exists.", status_code=409)
            active_import = connection.execute(
                """
                SELECT 1 FROM imports
                WHERE case_label = ? AND status != 'CONVERTED' AND (? IS NULL OR id != ?)
                """,
                (label, exclude_import_id, exclude_import_id),
            ).fetchone()
            if active_import:
                raise WebAppError(
                    "That neutral case label is already assigned to an active import.",
                    status_code=409,
                )

    def _row_to_case(self, row: Any) -> dict[str, Any]:
        stored_metrics = decode_json(row["metrics_json"], None)
        stored_affine = decode_json(row["affine_json"], [])
        try:
            orientation_codes = [
                str(item) for item in nib.aff2axcodes(np.asarray(stored_affine, dtype=float))
            ]
        except (IndexError, TypeError, ValueError):
            orientation_codes = []
        metrics = []
        if isinstance(stored_metrics, dict):
            metrics = [
                {
                    "label_value": item["value"],
                    "label_name": item["name"],
                    "voxel_count": item["voxel_count"],
                    "volume_mm3": item["volume_mm3"],
                }
                for item in stored_metrics.get("labels", [])
            ]
        return {
            "case_id": row["id"],
            "case_label": row["case_label"],
            "catalog": self.catalog_context(row["id"]),
            "source_type": row["source_type"],
            "state": row["state"],
            "validation_status": row["validation_status"],
            "shape": decode_json(row["shape_json"], []),
            "voxel_spacing_mm": decode_json(row["spacing_json"], []),
            "orientation_codes": orientation_codes,
            "volume_file_type": (
                "nii.gz" if str(row["volume_relpath"]).lower().endswith(".nii.gz") else "nii"
            ),
            "warnings": decode_json(row["warnings_json"], []),
            "validation_details": decode_json(row["validation_details_json"], {}),
            "qc_checklist": decode_json(row["qc_checklist_json"], _empty_checklist()),
            "reviewer_notes": row["reviewer_notes"],
            "review_status": row["review_status"],
            "viewer_state": decode_json(row["viewer_state_json"], {}),
            "protocol_version": row["protocol_version"],
            "annotation_disposition": "PROVISIONAL_NOT_FOR_TRAINING",
            "latest_revision": row["latest_revision"],
            "metrics": metrics,
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }

    def list_cases(self) -> list[dict[str, Any]]:
        with self.store.connect() as connection:
            rows = connection.execute(
                "SELECT * FROM cases ORDER BY updated_at DESC, case_label"
            ).fetchall()
        return [self._row_to_case(row) for row in rows]

    def catalog_context(self, case_id: str) -> dict[str, Any] | None:
        """Trusted local acquisition context, with legacy volume IDs preserved."""
        with self.store.connect() as connection:
            row = connection.execute("""SELECT l.acquisition_json,l.source_fingerprint,
                s.id AS series_id,s.series_label,s.case_id AS active_case_id,s.source_fingerprint AS current_fingerprint,
                s.descriptor_json,s.preparation_status,s.present,
                t.id AS study_id,t.study_label FROM catalog_case_links l
                JOIN catalog_series s ON s.id=l.series_id JOIN catalog_studies t ON t.id=s.study_id
                WHERE l.case_id=?""", (case_id,)).fetchone()
        if row is None:
            return None
        context = decode_json(row["acquisition_json"], {})
        current_source = row["current_fingerprint"] == row["source_fingerprint"] and row["active_case_id"] == case_id
        descriptor = decode_json(row["descriptor_json"], {})
        categories = {}
        if (current_source and row["source_fingerprint"] and row["preparation_status"] == "ready"
                and row["present"] and descriptor.get("metadata_validated") is True):
            from .catalog import acquisition

            categories = acquisition(descriptor)
        # Legacy acquisition snapshots predate these categories. Enrich only
        # current reads from verified matching metadata, never stored evidence.
        for key in ("reconstruction_kind", "kernel_class"):
            context.setdefault(key, categories.get(key, "unknown"))
        if not current_source:
            context["warnings"] = [*context.get("warnings", []), "This retained volume belongs to an earlier source version."]
        return {**context, **{key:row[key] for key in ("study_id", "study_label", "series_id", "series_label")}}

    def get_case(self, case_id: str) -> dict[str, Any]:
        with self.store.connect() as connection:
            row = connection.execute("SELECT * FROM cases WHERE id = ?", (case_id,)).fetchone()
        if row is None:
            raise WebAppError("Case not found.", status_code=404)
        return self._row_to_case(row)

    def assert_catalog_countable(self, case_id: str) -> None:
        """Gate new work without restricting retained result/history access."""
        from .catalog import fingerprint_hashes

        with self.store.connect() as connection:
            self._get_case_row(connection, case_id)
            row = connection.execute("""SELECT l.source_fingerprint AS linked_fingerprint,
                s.*,src.root_path FROM catalog_case_links l
                JOIN catalog_series s ON s.id=l.series_id
                JOIN catalog_studies t ON t.id=s.study_id
                LEFT JOIN catalog_series_sources ss ON ss.series_id=s.id
                JOIN catalog_sources src ON src.id=COALESCE(ss.source_id,t.source_id) WHERE l.case_id=?""", (case_id,)).fetchone()
        if row is None:
            return
        if row["case_id"] != case_id or row["preparation_status"] != "ready" or not row["present"] or row["linked_fingerprint"] != row["source_fingerprint"]:
            raise WebAppError("This retained volume is not the current prepared series. Prepare the current series before starting another count.", status_code=409)
        reason = None
        try:
            root = Path(row["root_path"]).resolve(strict=True)
            names = decode_json(row["files_json"], [])
            paths = [(root / value).resolve(strict=True) for value in names]
            if not paths or len(set(paths)) != len(paths) or any(not path.is_relative_to(root) or not path.is_file() for path in paths):
                raise ValueError("Invalid source collection")
            if fingerprint_hashes(sha256_file(path) for path in paths) != row["linked_fingerprint"]:
                reason = "SOURCE_CHANGED"
        except (OSError, ValueError, TypeError):
            reason = "SOURCE_UNAVAILABLE"
        if reason:
            self._invalidate_catalog_case(case_id, reason)
            raise WebAppError("The source files changed or are unavailable. Prepare this series again before starting another count; previous results remain available.", status_code=409)
        self.volume_path(case_id)

    def _invalidate_catalog_case(self, case_id: str, reason: str) -> None:
        with self.store.connect() as connection:
            changed = connection.execute("UPDATE catalog_series SET case_id=NULL,preparation_status='unavailable',reason_code=? WHERE case_id=?", (reason, case_id)).rowcount
        if changed:
            from .catalog import CatalogService
            CatalogService(self).refresh_recommendations()

    def _get_case_row(self, connection: Any, case_id: str) -> Any:
        row = connection.execute("SELECT * FROM cases WHERE id = ?", (case_id,)).fetchone()
        if row is None:
            raise WebAppError("Case not found.", status_code=404)
        return row

    def volume_path(self, case_id: str) -> Path:
        with self.store.connect() as connection:
            row = self._get_case_row(connection, case_id)
        path = self._path(row["volume_relpath"])
        try:
            intact = path.is_file() and sha256_file(path) == row["source_sha256"]
        except OSError:
            intact = False
        if not intact:
            self._invalidate_catalog_case(case_id, "CONVERSION_FAILED")
            raise WebAppError("The case volume changed or is unavailable and must be prepared again.", status_code=409)
        return path

    def update_viewer_state(
        self, case_id: str, viewer_state: dict[str, Any] | None
    ) -> dict[str, Any]:
        encoded_state = _encode_viewer_state(viewer_state)
        now = utc_now()
        with self.store.connect() as connection:
            self._get_case_row(connection, case_id)
            connection.execute(
                "UPDATE cases SET viewer_state_json = ?, updated_at = ? WHERE id = ?",
                (encoded_state, now, case_id),
            )
        return self.get_case(case_id)

    def import_nifti(
        self,
        source: Path,
        *,
        case_label: str | None = None,
        qc_report: Path | None = None,
    ) -> dict[str, Any]:
        label = self._normalize_label(case_label)
        self._assert_label_available(label)
        lower_name = source.name.lower()
        if not (lower_name.endswith(".nii") or lower_name.endswith(".nii.gz")):
            raise WebAppError("Only .nii and .nii.gz files can be imported.", status_code=422)
        info = inspect_nifti(source)
        source_hash = sha256_file(source)
        validation_status, validation_details = _read_qc_report(
            qc_report, source_hash, info, label
        )
        case_id = uuid.uuid4().hex
        case_dir = self.root / "cases" / case_id
        suffix = ".nii.gz" if lower_name.endswith(".nii.gz") else ".nii"
        destination = case_dir / "volume" / f"source{suffix}"
        now = utc_now()
        try:
            _atomic_copy(source, destination)
            if sha256_file(destination) != source_hash:
                raise WebAppError("The stored NIfTI hash does not match the upload.", status_code=500)
            with self.store.connect() as connection:
                connection.execute(
                    """
                    INSERT INTO cases(
                        id, case_label, source_type, state, validation_status,
                        volume_relpath, source_sha256, shape_json, spacing_json,
                        affine_json, warnings_json, validation_details_json,
                        qc_checklist_json, protocol_version, created_at, updated_at
                    ) VALUES(?, ?, 'NIFTI', 'READY', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        case_id,
                        label,
                        validation_status,
                        self._relative(destination),
                        source_hash,
                        encode_json(list(info.shape)),
                        encode_json(list(info.spacing)),
                        encode_json([list(row) for row in info.affine]),
                        encode_json(list(info.warnings)),
                        encode_json(validation_details),
                        encode_json(_empty_checklist()),
                        protocol_version(),
                        now,
                        now,
                    ),
                )
        except Exception:
            shutil.rmtree(case_dir, ignore_errors=True)
            raise
        return self.get_case(case_id)

    def import_dicom(self, sources: Iterable[Path], *, case_label: str | None = None) -> dict[str, Any]:
        from .dicom_adapter import inspect_dicom_upload

        label = self._normalize_label(case_label)
        self._assert_label_available(label)
        paths = [Path(item) for item in sources]
        if not paths:
            raise WebAppError("No DICOM files were supplied.", status_code=422)
        import_id = uuid.uuid4().hex
        staging = self.root / "imports" / import_id / "files"
        staging.mkdir(parents=True)
        copied: list[Path] = []
        try:
            for index, source in enumerate(paths):
                destination = staging / f"image_{index:06d}.dcm"
                _atomic_copy(source, destination)
                copied.append(destination)
            series = inspect_dicom_upload(copied, staging)
            now = utc_now()
            with self.store.connect() as connection:
                connection.execute(
                    """
                    INSERT INTO imports(
                        id, case_label, status, staging_relpath, series_json, created_at, updated_at
                    ) VALUES(?, ?, 'AWAITING_SERIES', ?, ?, ?, ?)
                    """,
                    (
                        import_id,
                        label,
                        self._relative(staging),
                        encode_json(series),
                        now,
                        now,
                    ),
                )
        except Exception:
            shutil.rmtree(staging.parent, ignore_errors=True)
            raise
        return self.get_import(import_id)

    @staticmethod
    def _safe_series(series: dict[str, Any]) -> dict[str, Any]:
        return {
            key: value
            for key, value in series.items()
            if key not in {"study_uid", "series_uid", "files"}
        }

    def get_import(self, import_id: str) -> dict[str, Any]:
        with self.store.connect() as connection:
            row = connection.execute("SELECT * FROM imports WHERE id = ?", (import_id,)).fetchone()
        if row is None:
            raise WebAppError("Import not found.", status_code=404)
        series = decode_json(row["series_json"], [])
        return {
            "import_id": row["id"],
            "case_label": row["case_label"],
            "status": row["status"],
            "series": [self._safe_series(item) for item in series],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }

    def delete_import(self, import_id: str) -> None:
        with self.store.connect() as connection:
            row = connection.execute(
                "SELECT * FROM imports WHERE id = ?", (import_id,)
            ).fetchone()
        if row is None:
            raise WebAppError("Import not found.", status_code=404)
        if row["status"] == "CONVERTED":
            raise WebAppError("Converted import records cannot be deleted.", status_code=409)
        staging = self._path(row["staging_relpath"])
        shutil.rmtree(staging.parent, ignore_errors=False)
        if staging.parent.exists():
            raise WebAppError("The temporary DICOM staging area could not be removed.", status_code=500)
        with self.store.connect() as connection:
            connection.execute("DELETE FROM imports WHERE id = ?", (import_id,))

    def convert_dicom(self, import_id: str, series_label: str) -> dict[str, Any]:
        from .dicom_adapter import convert_uploaded_series

        with self.store.connect() as connection:
            import_row = connection.execute(
                "SELECT * FROM imports WHERE id = ?", (import_id,)
            ).fetchone()
        if import_row is None:
            raise WebAppError("Import not found.", status_code=404)
        if import_row["status"] == "CONVERTED":
            raise WebAppError("This import has already been converted.", status_code=409)
        label = import_row["case_label"]
        self._assert_label_available(label, exclude_import_id=import_id)
        series_values = decode_json(import_row["series_json"], [])
        selected = next(
            (item for item in series_values if item.get("series_label") == series_label), None
        )
        if selected is None:
            raise WebAppError("Series not found in this import.", status_code=404)
        if selected.get("preflight_status") != "PASS":
            raise WebAppError("The selected DICOM series failed preflight.", status_code=422)
        staging = self._path(import_row["staging_relpath"])
        paths = [staging / name for name in selected["files"]]
        case_id = uuid.uuid4().hex
        case_dir = self.root / "cases" / case_id
        output_dir = case_dir / "volume"
        now = utc_now()
        try:
            result = convert_uploaded_series(
                paths,
                output_dir=output_dir,
                dcm2niix=self.dcm2niix,
                expected_study_uid=selected["study_uid"],
                expected_series_uid=selected["series_uid"],
                case_label=label,
                series_label=series_label,
            )
            volume_path = Path(result["nifti"])
            info = inspect_nifti(volume_path)
            source_hash = sha256_file(volume_path)
            warnings = list(dict.fromkeys([*selected.get("warnings", []), *info.warnings]))
            shutil.rmtree(staging.parent, ignore_errors=False)
            if staging.parent.exists():
                raise WebAppError(
                    "The temporary DICOM staging area could not be removed.", status_code=500
                )
            with self.store.connect() as connection:
                connection.execute(
                    """
                    INSERT INTO cases(
                        id, case_label, source_type, state, validation_status,
                        volume_relpath, source_sha256, shape_json, spacing_json,
                        affine_json, warnings_json, validation_details_json,
                        qc_checklist_json, protocol_version, created_at, updated_at
                    ) VALUES(?, ?, 'DICOM', 'READY', 'VERIFIED', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        case_id,
                        label,
                        self._relative(volume_path),
                        source_hash,
                        encode_json(list(info.shape)),
                        encode_json(list(info.spacing)),
                        encode_json([list(row) for row in info.affine]),
                        encode_json(warnings),
                        encode_json(result["validation"]),
                        encode_json(_empty_checklist()),
                        protocol_version(),
                        now,
                        now,
                    ),
                )
                connection.execute(
                    "UPDATE imports SET status = 'CONVERTED', updated_at = ? WHERE id = ?",
                    (now, import_id),
                )
        except Exception:
            shutil.rmtree(case_dir, ignore_errors=True)
            shutil.rmtree(staging.parent, ignore_errors=True)
            with self.store.connect() as connection:
                connection.execute("DELETE FROM imports WHERE id = ?", (import_id,))
            raise
        return self.get_case(case_id)

    def _load_annotation(self, mask_path: Path, source_path: Path) -> tuple[np.ndarray, nib.Nifti1Image]:
        try:
            source = nib.load(source_path, mmap=True)
            mask = nib.load(mask_path, mmap=True)
        except Exception as exc:
            raise WebAppError("The annotation NIfTI could not be opened.", status_code=422) from exc
        if tuple(mask.shape) != tuple(source.shape):
            raise WebAppError("The annotation shape does not match the CT volume.", status_code=422)
        if not np.allclose(mask.affine, source.affine, rtol=0, atol=1e-5):
            raise WebAppError("The annotation affine does not match the CT volume.", status_code=422)
        try:
            values = np.asarray(mask.get_fdata(dtype=np.float32))
        except Exception as exc:
            raise WebAppError("The annotation voxel data is unreadable.", status_code=422) from exc
        if not np.all(np.isfinite(values)):
            raise WebAppError("The annotation contains non-finite values.", status_code=422)
        rounded = np.rint(values)
        if not np.array_equal(values, rounded):
            raise WebAppError("Annotation values must be integers from 0 through 4.", status_code=422)
        unique = {int(item) for item in np.unique(rounded)}
        if not unique.issubset(allowed_label_values()):
            raise WebAppError("Annotation values must be integers from 0 through 4.", status_code=422)
        return rounded.astype(np.uint8), source

    @staticmethod
    def _metrics(mask: np.ndarray, affine: np.ndarray) -> dict[str, Any]:
        voxel_volume = abs(float(np.linalg.det(np.asarray(affine)[:3, :3])))
        protocol = load_protocol()
        labels: list[dict[str, Any]] = []
        for label in protocol["labels"]:
            if int(label["value"]) == 0:
                continue
            count = int(np.count_nonzero(mask == int(label["value"])))
            volume = count * voxel_volume
            labels.append(
                {
                    "value": int(label["value"]),
                    "name": label["name"],
                    "voxel_count": count,
                    "volume_mm3": volume,
                    "volume_ml": volume / 1000.0,
                }
            )
        return {"voxel_volume_mm3": voxel_volume, "labels": labels}

    def save_annotation(
        self, case_id: str, mask_path: Path, viewer_state: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        state = viewer_state or {}
        encoded_state = _encode_viewer_state(state)

        with self.store.connect() as connection:
            row = self._get_case_row(connection, case_id)
        if row["state"] == "APPROVED":
            raise WebAppError("Approved annotations cannot be edited.", status_code=409)
        source_path = self._path(row["volume_relpath"])
        if sha256_file(source_path) != row["source_sha256"]:
            raise WebAppError("The source volume changed after import.", status_code=409)
        mask, source = self._load_annotation(mask_path, source_path)
        metrics = self._metrics(mask, source.affine)
        new_image = nib.Nifti1Image(mask, source.affine)
        new_image.set_qform(source.affine, code=1)
        new_image.set_sform(source.affine, code=1)
        new_image.header.set_data_dtype(np.uint8)
        new_image.header.set_xyzt_units("mm")
        now = utc_now()

        with self.store.connect() as connection:
            current = self._get_case_row(connection, case_id)
            revision = int(current["latest_revision"] or 0) + 1
            destination = (
                self.root / "cases" / case_id / "annotations" / f"revision_{revision:06d}.nii.gz"
            )
            _atomic_save_nifti(new_image, destination)
            mask_hash = sha256_file(destination)
            try:
                connection.execute(
                    """
                    INSERT INTO annotation_revisions(
                        case_id, revision, mask_relpath, mask_sha256, source_sha256,
                        protocol_version, viewer_state_json, metrics_json, created_at
                    ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        case_id,
                        revision,
                        self._relative(destination),
                        mask_hash,
                        current["source_sha256"],
                        protocol_version(),
                        encoded_state,
                        encode_json(metrics),
                        now,
                    ),
                )
                connection.execute(
                    """
                    UPDATE cases SET state = 'ANNOTATING', review_status = 'Draft',
                        latest_revision = ?, viewer_state_json = ?, metrics_json = ?, updated_at = ?
                    WHERE id = ?
                    """,
                    (revision, encoded_state, encode_json(metrics), now, case_id),
                )
            except Exception:
                destination.unlink(missing_ok=True)
                raise
        return {
            "case_id": case_id,
            "revision": revision,
            "mask_sha256": mask_hash,
            "protocol_version": protocol_version(),
            "training_eligibility": "PROVISIONAL_NOT_FOR_TRAINING",
            "metrics": metrics,
            "created_at": now,
        }

    def latest_annotation_path(self, case_id: str) -> Path:
        with self.store.connect() as connection:
            row = self._get_case_row(connection, case_id)
            if row["latest_revision"] is None:
                raise WebAppError("This case has no saved annotation.", status_code=404)
            revision = connection.execute(
                "SELECT * FROM annotation_revisions WHERE case_id = ? AND revision = ?",
                (case_id, row["latest_revision"]),
            ).fetchone()
        if revision is None:
            raise WebAppError("The latest annotation revision is unavailable.", status_code=409)
        path = self._path(revision["mask_relpath"])
        if not path.is_file() or sha256_file(path) != revision["mask_sha256"]:
            raise WebAppError("The latest annotation revision failed its hash check.", status_code=409)
        return path

    def _validate_latest(self, row: Any) -> tuple[Any, dict[str, Any]]:
        source = self._path(row["volume_relpath"])
        if sha256_file(source) != row["source_sha256"]:
            raise WebAppError("The source volume changed after import.", status_code=409)
        if row["latest_revision"] is None:
            raise WebAppError("Save an annotation before finalizing this case.", status_code=409)
        with self.store.connect() as connection:
            revision = connection.execute(
                "SELECT * FROM annotation_revisions WHERE case_id = ? AND revision = ?",
                (row["id"], row["latest_revision"]),
            ).fetchone()
        if revision is None or revision["protocol_version"] != row["protocol_version"]:
            raise WebAppError("The latest annotation protocol does not match the case.", status_code=409)
        mask_path = self._path(revision["mask_relpath"])
        if sha256_file(mask_path) != revision["mask_sha256"]:
            raise WebAppError("The annotation changed after it was saved.", status_code=409)
        mask, source_image = self._load_annotation(mask_path, source)
        if mask.dtype != np.uint8:
            raise WebAppError("The stored annotation is not uint8.", status_code=409)
        metrics = self._metrics(mask, source_image.affine)
        return revision, metrics

    def finalize_case(self, case_id: str) -> dict[str, Any]:
        with self.store.connect() as connection:
            row = self._get_case_row(connection, case_id)
        if row["state"] == "APPROVED" or row["review_status"] == "Approved":
            raise WebAppError("Approved cases cannot be finalized again.", status_code=409)
        if row["validation_status"] == "FAILED":
            raise WebAppError("A failed conversion cannot be finalized.", status_code=409)
        _, metrics = self._validate_latest(row)
        checklist = decode_json(row["qc_checklist_json"], _empty_checklist())
        if not all(checklist.get(key) is True for key in QC_KEYS):
            raise WebAppError("Complete all four QC checks before finalizing.", status_code=409)
        now = utc_now()
        with self.store.connect() as connection:
            connection.execute(
                """
                UPDATE cases SET state = 'READY_FOR_REVIEW', review_status = 'Ready for review',
                    metrics_json = ?, updated_at = ? WHERE id = ?
                """,
                (encode_json(metrics), now, case_id),
            )
        return self.get_case(case_id)

    def update_review(
        self,
        case_id: str,
        *,
        status: str | None = None,
        qc_checklist: dict[str, bool | None] | None = None,
        reviewer_notes: str | None = None,
    ) -> dict[str, Any]:
        if status is not None and status not in REVIEW_STATUSES:
            raise WebAppError("Review status is invalid.", status_code=422)
        with self.store.connect() as connection:
            row = self._get_case_row(connection, case_id)
        checklist = decode_json(row["qc_checklist_json"], _empty_checklist())
        if qc_checklist is not None:
            if set(qc_checklist) - set(QC_KEYS) or any(
                value not in {True, False, None} for value in qc_checklist.values()
            ):
                raise WebAppError("The QC checklist contains invalid fields or values.", status_code=422)
            checklist.update(qc_checklist)
        notes = row["reviewer_notes"] if reviewer_notes is None else reviewer_notes.strip()
        if len(notes) > 20_000:
            raise WebAppError("Reviewer notes are too long.", status_code=422)
        new_status = status or row["review_status"]
        state = row["state"]
        metrics = decode_json(row["metrics_json"], None)
        if new_status in {"Ready for review", "Approved"}:
            if not all(checklist.get(key) is True for key in QC_KEYS):
                raise WebAppError("Complete all four QC checks before review approval.", status_code=409)
            _, metrics = self._validate_latest(row)
            if new_status == "Approved" and row["validation_status"] != "VERIFIED":
                raise WebAppError("Only a verified source volume can be approved.", status_code=409)
            state = "APPROVED" if new_status == "Approved" else "READY_FOR_REVIEW"
        elif new_status == "Needs changes":
            state = "NEEDS_CHANGES"
        elif new_status == "Draft":
            state = "ANNOTATING" if row["latest_revision"] is not None else "READY"
        now = utc_now()
        with self.store.connect() as connection:
            connection.execute(
                """
                UPDATE cases SET state = ?, review_status = ?, qc_checklist_json = ?,
                    reviewer_notes = ?, metrics_json = ?, updated_at = ? WHERE id = ?
                """,
                (
                    state,
                    new_status,
                    encode_json(checklist),
                    notes,
                    encode_json(metrics) if metrics is not None else None,
                    now,
                    case_id,
                ),
            )
        return self.get_case(case_id)

    def export_case(self, case_id: str) -> Path:
        with self.store.connect() as connection:
            row = self._get_case_row(connection, case_id)
        revision, metrics = self._validate_latest(row)
        protocol = load_protocol()
        manifest = {
            "schema_version": 1,
            "case_label": row["case_label"],
            "source_type": row["source_type"],
            "source_sha256": row["source_sha256"],
            "shape": decode_json(row["shape_json"], []),
            "voxel_spacing_mm": decode_json(row["spacing_json"], []),
            "validation_status": row["validation_status"],
            "annotation_revision": revision["revision"],
            "annotation_sha256": revision["mask_sha256"],
            "protocol_version": row["protocol_version"],
            "training_eligibility": "PROVISIONAL_NOT_FOR_TRAINING",
        }
        review = {
            "case_label": row["case_label"],
            "qc_checklist": decode_json(row["qc_checklist_json"], _empty_checklist()),
            "reviewer_notes": row["reviewer_notes"],
            "review_status": row["review_status"],
        }
        destination = self.root / "exports" / f"{row['case_label']}_annotation.zip"
        handle, name = tempfile.mkstemp(prefix=f".{destination.stem}.", suffix=".zip", dir=destination.parent)
        os.close(handle)
        temporary = Path(name)
        try:
            with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as archive:
                archive.write(self._path(revision["mask_relpath"]), "annotation.nii.gz")
                for filename, value in (
                    ("manifest.json", manifest),
                    ("protocol.json", protocol),
                    ("review.json", review),
                    ("metrics.json", metrics),
                ):
                    archive.writestr(
                        filename,
                        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
                    )
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)
        return destination

    def delete_draft(self, case_id: str) -> None:
        with self.store.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = self._get_case_row(connection, case_id)
            if row["review_status"] != "Draft" or row["state"] == "APPROVED":
                raise WebAppError("Only draft cases can be deleted.", status_code=409)
            active_run = connection.execute(
                "SELECT 1 FROM count_runs WHERE case_id=? AND status IN ('queued','running')",
                (case_id,),
            ).fetchone()
            if active_run is not None:
                raise WebAppError("Wait for the active count run before deleting this case.", status_code=409)
            connection.execute("DELETE FROM cases WHERE id = ?", (case_id,))
        case_dir = (self.root / "cases" / case_id).resolve()
        case_dir.relative_to((self.root / "cases").resolve())
        shutil.rmtree(case_dir, ignore_errors=False)
