"""Isolated native-grid PanSeg watershed/GASP research backend.

No references, clinical cores, model downloads or alternative decoder enter
this interface. The CUDA caller exchanges hashed files with a CPU subprocess.
"""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
import uuid
import zipfile
from urllib.parse import urlparse

# Direct invocation of a conda Python on Windows does not activate its DLL path.
_DLL_DIRECTORY = None
if os.name == "nt" and (Path(sys.prefix) / "Library" / "bin").is_dir():
    _DLL_DIRECTORY = os.add_dll_directory(str(Path(sys.prefix) / "Library" / "bin"))

import numpy as np

VERSION = "native-boundary-panseg-average-1"
CONFIG = {"membership_threshold": .5, "threshold": .5, "sigma_seeds": 1.,
          "sigma_weights": 0., "min_size": 0, "alpha": 1.,
          "stacked": False, "blockwise": False, "apply_nonmax_suppression": False,
          "n_threads": 1, "beta": .5, "post_minsize": 0,
          "linkage_criteria": "average", "minimum_native_voxels": 4}
MEMORY_BUDGET = 2 * 1024**3
TIMEOUT_SECONDS = 120
REQUEST_FIELDS = {"schema_version", "probabilities_path", "probabilities_sha256",
                  "affine", "runtime_manifest_path", "runtime_manifest_sha256"}
RUNTIME_FIELDS = {"schema_version", "backend_version", "python_executable", "python_sha256",
    "python_version", "adapter_sha256", "panseg_version", "panseg_commit", "panseg_source_root",
    "source_archive", "source_archive_sha256", "panseg_python_sha256", "conda_lock_path",
    "conda_lock_sha256", "explicit_lock_path", "explicit_lock_sha256", "package_archive_audit_path",
    "package_archive_audit_sha256", "conda_metadata_sha256", "packages", "jobs_directory", "panseg_home"}


class PartitionError(ValueError):
    """Fixed codes may cross the protected-artifact boundary."""


def _require(condition, code):
    if not condition:
        raise PartitionError(code)


def _sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _external(path, *, exists=True):
    try:
        candidate = Path(path)
        _require(candidate.is_absolute(), "PARTITION_EXTERNAL_PATH_REQUIRED")
        candidate = candidate.resolve(strict=exists)
        _require(not any((p / ".git").exists() for p in (candidate, *candidate.parents)),
                 "PARTITION_EXTERNAL_PATH_REQUIRED")
        return candidate
    except (OSError, TypeError):
        raise PartitionError("PARTITION_EXTERNAL_PATH_REQUIRED") from None


def _read(path):
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8-sig"),
                           parse_constant=lambda _: (_ for _ in ()).throw(ValueError()))
        _require(isinstance(value, dict), "PARTITION_JSON_INVALID")
        return value
    except (OSError, ValueError):
        raise PartitionError("PARTITION_JSON_INVALID") from None


def _write(path, value):
    with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def _tree_hashes(root):
    return {str(path.relative_to(root)).replace("\\", "/"): _sha(path)
            for path in sorted(Path(root).rglob("*.py"))}


def _validate_arrays(probabilities, affine):
    values = np.asarray(probabilities)
    try:
        matrix = np.asarray(affine, dtype=np.float64)
    except (TypeError, ValueError):
        raise PartitionError("PARTITION_AFFINE_INVALID") from None
    _require(values.ndim == 4 and values.shape[0] == 2 and all(n >= 3 for n in values.shape[1:])
             and values.dtype.kind == "f", "PARTITION_PROBABILITIES_INVALID")
    estimated = 128 * 1024**2 + int(np.prod(values.shape[1:])) * 384
    _require(estimated <= MEMORY_BUDGET, "PARTITION_MEMORY_BUDGET_EXCEEDED")
    if os.name == "nt":
        class MemoryStatus(ctypes.Structure):
            _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong)] + [
                (name, ctypes.c_ulonglong) for name in ("total_physical", "available_physical",
                    "total_pagefile", "available_pagefile", "total_virtual", "available_virtual", "extended")]
        memory = MemoryStatus(); memory.length = ctypes.sizeof(memory)
        _require(ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(memory)) != 0,
                 "PARTITION_MEMORY_PREFLIGHT_FAILED")
        _require(estimated <= min(memory.available_physical, memory.available_pagefile),
                 "PARTITION_MEMORY_UNAVAILABLE")
    _require(np.isfinite(values).all() and np.all((values >= 0) & (values <= 1)),
             "PARTITION_PROBABILITIES_INVALID")
    _require(matrix.shape == (4, 4) and np.isfinite(matrix).all()
             and np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-8, rtol=0),
             "PARTITION_AFFINE_INVALID")
    pitch = np.linalg.norm(matrix[:3, :3], axis=0)
    _require(np.all(pitch > 0) and np.allclose((matrix[:3, :3] / pitch).T @
             (matrix[:3, :3] / pitch), np.eye(3), atol=1e-5, rtol=0),
             "PARTITION_AFFINE_INVALID")
    return np.ascontiguousarray(values, dtype=np.float32), matrix, pitch, estimated


def _runtime(runtime_manifest):
    if isinstance(runtime_manifest, dict):
        _require(set(runtime_manifest) == {"manifest_path"}, "PARTITION_RUNTIME_PROTOCOL_MISMATCH")
        runtime_manifest = runtime_manifest.get("manifest_path")
    try:
        path = _external(runtime_manifest)
        record = _read(path)
        _require(set(record) == RUNTIME_FIELDS and record.get("schema_version") == 1
                 and record.get("backend_version") == VERSION,
                 "PARTITION_RUNTIME_PROTOCOL_MISMATCH")
        python = _external(record["python_executable"])
        source = _external(record["panseg_source_root"])
        prefix = python.parent
        _require(_sha(python) == record["python_sha256"]
                 and _sha(__file__) == record["adapter_sha256"], "PARTITION_RUNTIME_HASH_MISMATCH")
        metadata = {p.name: _sha(p) for p in sorted((prefix / "conda-meta").glob("*.json"))}
        _require(metadata and metadata == record["conda_metadata_sha256"]
                 and _tree_hashes(source / "panseg") == record["panseg_python_sha256"],
                 "PARTITION_RUNTIME_HASH_MISMATCH")
        _require(_sha(_external(record["source_archive"])) == record["source_archive_sha256"],
                 "PARTITION_RUNTIME_HASH_MISMATCH")
        _require(_sha(record["conda_lock_path"]) == record["conda_lock_sha256"]
                 and _sha(record["explicit_lock_path"]) == record["explicit_lock_sha256"]
                 and _sha(_external(record["package_archive_audit_path"])) == record["package_archive_audit_sha256"],
                 "PARTITION_RUNTIME_HASH_MISMATCH")
        return path, record
    except (KeyError, OSError, TypeError):
        raise PartitionError("PARTITION_RUNTIME_UNAVAILABLE") from None


def _offline_guard():
    def denied(*args, **kwargs):
        raise PartitionError("PARTITION_NETWORK_DISABLED")
    socket.create_connection = denied
    socket.getaddrinfo = denied
    socket.socket.connect = denied
    socket.socket.connect_ex = denied
    socket.socket.sendto = denied
    for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"):
        os.environ[key] = "1"


def _decode(values, pitch):
    from scipy import ndimage as ndi
    from skimage.measure import label
    from panseg.functionals.segmentation import dt_watershed
    from panseg.functionals.segmentation.utils import shift_affinities
    from elf.segmentation import GaspFromAffinities
    from elf.segmentation.gasp import SegmentationFeeder

    foreground = values[0] >= CONFIG["membership_threshold"]
    if not foreground.any():
        return np.zeros(foreground.shape, np.int32), {"watershed_regions": 0,
            "raw_gasp_regions": 0, "removed_small_components": 0, "removed_small_voxels": 0,
            "missing_foreground_voxels": 0, "foreground_components": 0, "final_regions": 0}
    boundary = np.where(foreground, values[1], 1).astype(np.float32)
    watershed = dt_watershed(boundary, threshold=.5, sigma_seeds=1., stacked=False,
        sigma_weights=0., min_size=0, alpha=1., pixel_pitch=tuple(map(float, pitch)),
        apply_nonmax_suppression=False, n_threads=1, blockwise=False, mask=foreground)
    _require(watershed.shape == foreground.shape and np.all(watershed[foreground] > 0)
             and np.all(watershed[~foreground] == 0), "PARTITION_WATERSHED_COVERAGE_FAILED")
    offsets = [[0, 0, 1], [0, 1, 0], [1, 0, 0]]
    affinities = 1 - shift_affinities(np.stack([boundary] * 3), offsets)
    # PanSeg's gasp wrapper omits foreground_mask. ELF's feeder converts every
    # background superpixel to its ignored -1 sentinel before building the RAG.
    decoder = GaspFromAffinities(offsets, superpixel_generator=SegmentationFeeder(),
        run_GASP_kwargs={"linkage_criteria": "average", "add_cannot_link_constraints": False,
                         "use_efficient_implementations": False}, n_threads=1, beta_bias=.5)
    raw, _ = decoder(affinities, watershed, foreground_mask=foreground)
    _require(raw.shape == foreground.shape and np.isfinite(raw).all(),
             "PARTITION_DECODER_OUTPUT_INVALID")
    raw = np.where(foreground, raw.astype(np.int64), -1)
    components = label(raw, background=-1, connectivity=1)
    sizes = np.bincount(components.ravel())
    keep = sizes >= CONFIG["minimum_native_voxels"]
    keep[0] = False
    removed = (sizes > 0) & ~keep
    removed[0] = False
    lut = np.cumsum(keep, dtype=np.int32)
    lut[~keep] = 0
    labels = lut[components]
    _require(np.all(labels[~foreground] == 0), "PARTITION_BACKGROUND_LEAK")
    _, foreground_components = ndi.label(foreground)
    return labels, {"watershed_regions": int(np.unique(watershed[foreground]).size),
        "raw_gasp_regions": int(np.unique(raw[foreground]).size),
        "foreground_components": int(foreground_components),
        "removed_small_components": int(removed.sum()),
        "removed_small_voxels": int(sizes[removed].sum()),
        "missing_foreground_voxels": int(np.sum(foreground & (labels == 0))),
        "final_regions": int(labels.max())}


def worker(request_path, output_path):
    _offline_guard()
    request_path, output = _external(request_path), _external(output_path, exists=False)
    _require(not output.exists() and output.parent.is_dir(), "PARTITION_FRESH_OUTPUT_REQUIRED")
    request_hash = _sha(request_path)
    request = _read(request_path)
    _require(set(request) == REQUEST_FIELDS and request["schema_version"] == 1,
             "PARTITION_REQUEST_INVALID")
    manifest_path, runtime = _runtime(request["runtime_manifest_path"])
    _require(_sha(manifest_path) == request["runtime_manifest_sha256"]
             and Path(sys.executable).resolve() == Path(runtime["python_executable"]).resolve(),
             "PARTITION_RUNTIME_PROTOCOL_MISMATCH")
    values_path = _external(request["probabilities_path"])
    _require(_sha(values_path) == request["probabilities_sha256"], "PARTITION_INPUT_HASH_MISMATCH")
    values, affine, pitch, estimated = _validate_arrays(
        np.load(values_path, allow_pickle=False, mmap_mode="r"), request["affine"])
    started = time.monotonic()
    labels, diagnostics = _decode(values, pitch)
    _require(_sha(request_path) == request_hash and _sha(values_path) == request["probabilities_sha256"],
             "PARTITION_INPUT_CHANGED")
    output.mkdir()
    labels_path = output / "labels.local.npy"
    np.save(labels_path, labels, allow_pickle=False)
    result = {"status": "COMPLETE", "version": VERSION, "worker_identity": str(uuid.uuid4()),
        "worker_pid": os.getpid(), "config": CONFIG, "shape": list(labels.shape),
        "affine": affine.tolist(), "pixel_pitch": pitch.tolist(), "native_grid": True,
        "estimated_peak_bytes": estimated, "memory_budget_bytes": MEMORY_BUDGET,
        "runtime_manifest_sha256": _sha(manifest_path), "request_sha256": request_hash,
        "probabilities_sha256": request["probabilities_sha256"], "labels_path": str(labels_path),
        "labels_sha256": _sha(labels_path), "array_sha256": hashlib.sha256(labels.tobytes()).hexdigest(),
        "elapsed_seconds": time.monotonic() - started, "offline_guard": True,
        "cannot_link_constraints": False, **diagnostics}
    _write(output / "result.local.json", result)
    return result


def partition(probabilities, affine, runtime_manifest):
    """Return native int32 labels and hashed diagnostics from one CPU worker.

    runtime_manifest is its absolute JSON path, or {"manifest_path": path}.
    Each call retains a fresh protected request/result job outside Git.
    """
    values, matrix, _, _ = _validate_arrays(probabilities, affine)
    manifest_path, runtime = _runtime(runtime_manifest)
    jobs = _external(runtime["jobs_directory"])
    job = jobs / str(uuid.uuid4())
    job.mkdir()
    inputs = job / "probabilities.local.npy"
    np.save(inputs, values, allow_pickle=False)
    request_path = job / "request.local.json"
    _write(request_path, {"schema_version": 1, "probabilities_path": str(inputs),
        "probabilities_sha256": _sha(inputs), "affine": matrix.tolist(),
        "runtime_manifest_path": str(manifest_path), "runtime_manifest_sha256": _sha(manifest_path)})
    output = job / "result"
    env = {**os.environ, "PYTHONNOUSERSITE": "1", "PYTHONDONTWRITEBYTECODE": "1",
           "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
           "CUDA_VISIBLE_DEVICES": "", "PANSEG_HOME": runtime["panseg_home"],
           "PYTHONPATH": os.pathsep.join([str(Path(__file__).resolve().parents[2]), runtime["panseg_source_root"]])}
    # Conda's Windows DLL directories are needed without changing the caller PATH.
    prefix = Path(runtime["python_executable"]).parent
    env["PATH"] = os.pathsep.join([str(prefix), str(prefix / "Library" / "bin"),
                                  str(prefix / "Scripts"), env.get("PATH", "")])
    try:
        completed = subprocess.run([runtime["python_executable"], "-s", "-m",
            "ethmoid_pipeline.counting.boundary_partition", "--request", str(request_path),
            "--output", str(output)], env=env, capture_output=True, timeout=TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired:
        _write(job / "failure.local.json", {"status": "FAILED", "error_code": "PARTITION_TIMEOUT"})
        raise PartitionError("PARTITION_TIMEOUT") from None
    if completed.returncode != 0:
        code = completed.stdout.decode("utf-8", errors="replace").strip().splitlines()
        code = code[-1] if code and code[-1].startswith("PARTITION_") and code[-1].replace("_", "").isalnum() else "PARTITION_EXECUTION_FAILED"
        _write(job / "failure.local.json", {"status": "FAILED", "error_code": code})
        raise PartitionError(code)
    result = _read(output / "result.local.json")
    labels_path = _external(result["labels_path"])
    _require(result.get("status") == "COMPLETE" and result.get("version") == VERSION
             and result.get("request_sha256") == _sha(request_path)
             and result.get("runtime_manifest_sha256") == _sha(manifest_path)
             and labels_path == output / "labels.local.npy"
             and _sha(labels_path) == result.get("labels_sha256"), "PARTITION_RESULT_HASH_MISMATCH")
    labels = np.load(labels_path, allow_pickle=False)
    _require(labels.shape == values.shape[1:] and labels.dtype == np.int32
             and np.all(labels >= 0) and np.all(labels[values[0] < .5] == 0),
             "PARTITION_DECODER_OUTPUT_INVALID")
    return labels, {**result, "job_directory": str(job), "result_path": str(output / "result.local.json"),
                    "result_sha256": _sha(output / "result.local.json")}


def seal_runtime(root, conda_lock):
    """Bootstrap-only sealing of public source and installed package metadata."""
    root = _external(root)
    source_lock = _read(Path(conda_lock).with_name("boundary-partition-source.lock.json"))
    prefix = root / "env"
    _require(Path(sys.executable).resolve() == (prefix / "python.exe").resolve()
             and sys.version_info[:2] == (3, 12), "PARTITION_RUNTIME_PROTOCOL_MISMATCH")
    lock = _read(conda_lock)
    metadata = sorted((prefix / "conda-meta").glob("*.json"))
    installed = sorted(({key: _read(p).get(key) for key in ("name", "version", "build", "sha256")}
                        for p in metadata), key=lambda p: p["name"])
    _require(installed == lock["packages"], "PARTITION_PACKAGE_LOCK_MISMATCH")
    archives = []
    for path in metadata:
        package = _read(path)
        url = urlparse(package["url"])
        _require(url.scheme == "https" and url.netloc == "conda.anaconda.org"
                 and url.path.startswith("/conda-forge/"), "PARTITION_PACKAGE_SOURCE_INVALID")
        archive = Path(package["package_tarball_full_path"]).with_name(Path(url.path).name)
        _require(_sha(archive) == package["sha256"], "PARTITION_PACKAGE_ARCHIVE_HASH_MISMATCH")
        archives.append({"name": package["name"], "url": package["url"], "sha256": _sha(archive)})
    audit_path = root / ("package-archive-audit-" + str(uuid.uuid4()) + ".local.json")
    _write(audit_path, {"status": "PASS", "packages": archives})
    source = root / "source" / ("panseg-" + source_lock["panseg_commit"])
    archive = root / ("panseg-" + source_lock["panseg_commit"] + ".zip")
    _require(_sha(archive) == source_lock["panseg_archive_sha256"], "PARTITION_SOURCE_HASH_MISMATCH")
    with zipfile.ZipFile(archive) as zipped:
        prefix_in_zip = "panseg-" + source_lock["panseg_commit"] + "/panseg/"
        expected = {name[len(prefix_in_zip):]: hashlib.sha256(zipped.read(name)).hexdigest()
                    for name in zipped.namelist() if name.startswith(prefix_in_zip) and name.endswith(".py")}
    _require(expected and _tree_hashes(source / "panseg") == expected, "PARTITION_SOURCE_HASH_MISMATCH")
    jobs, home = root / "jobs", root / "home"
    jobs.mkdir(exist_ok=True); home.mkdir(exist_ok=True)
    record = {"schema_version": 1, "backend_version": VERSION,
        "python_executable": str(prefix / "python.exe"), "python_sha256": _sha(prefix / "python.exe"),
        "python_version": sys.version, "adapter_sha256": _sha(__file__),
        "panseg_version": source_lock["panseg_version"], "panseg_commit": source_lock["panseg_commit"],
        "panseg_source_root": str(source), "source_archive": str(archive),
        "source_archive_sha256": _sha(archive), "panseg_python_sha256": _tree_hashes(source / "panseg"),
        "conda_lock_path": str(Path(conda_lock).resolve()), "conda_lock_sha256": _sha(conda_lock),
        "explicit_lock_path": str(Path(conda_lock).with_name("boundary-partition-win-64.explicit.txt").resolve()),
        "explicit_lock_sha256": _sha(Path(conda_lock).with_name("boundary-partition-win-64.explicit.txt")),
        "package_archive_audit_path": str(audit_path), "package_archive_audit_sha256": _sha(audit_path),
        "conda_metadata_sha256": {p.name: _sha(p) for p in metadata},
        "packages": installed, "jobs_directory": str(jobs), "panseg_home": str(home)}
    _write(root / "runtime-manifest.local.json", record)


def main():
    parser = argparse.ArgumentParser(description="Offline isolated PanSeg research partition worker")
    parser.add_argument("--request"); parser.add_argument("--output")
    parser.add_argument("--seal-runtime"); parser.add_argument("--conda-lock")
    parser.add_argument("--verify-runtime")
    args = parser.parse_args()
    try:
        if args.seal_runtime and args.conda_lock and not any((args.request, args.output, args.verify_runtime)):
            seal_runtime(args.seal_runtime, args.conda_lock)
        elif args.verify_runtime and not any((args.request, args.output, args.seal_runtime, args.conda_lock)):
            _runtime(args.verify_runtime)
        elif args.request and args.output and not any((args.seal_runtime, args.conda_lock, args.verify_runtime)):
            worker(args.request, args.output)
        else:
            raise PartitionError("PARTITION_ARGUMENTS_INVALID")
    except Exception as error:
        code = str(error) if isinstance(error, PartitionError) else "PARTITION_EXECUTION_FAILED"
        print(code)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
