"""Synthetic contracts, including actual isolated PanSeg/ELF integration."""
from __future__ import annotations

import os
from pathlib import Path
import json
import subprocess
import textwrap

import numpy as np
import pytest

from ethmoid_pipeline.counting import boundary_partition as backend


def boxes():
    values = np.zeros((2, 20, 20, 20), np.float32)
    values[0, 2:8, 2:8, 2:8] = 1
    values[0, 12:18, 12:18, 12:18] = 1
    return values


@pytest.mark.parametrize("change", ["nan", "range", "channels", "integer"])
def test_probability_contract(change):
    values = boxes()
    if change == "nan": values[0, 0, 0, 0] = np.nan
    elif change == "range": values[0, 0, 0, 0] = -1
    elif change == "channels": values = values[:1]
    else: values = values.astype(np.int32)
    with pytest.raises(backend.PartitionError, match="PARTITION_PROBABILITIES_INVALID"):
        backend._validate_arrays(values, np.eye(4))


def test_affine_requires_physical_orthogonal_axes_and_preserves_anisotropic_pitch():
    affine = np.diag([.3, .6, 1.2, 1])
    _, observed, pitch, _ = backend._validate_arrays(boxes(), affine)
    assert np.array_equal(observed, affine)
    assert np.array_equal(pitch, [.3, .6, 1.2])
    affine[0, 1] = .2
    with pytest.raises(backend.PartitionError, match="PARTITION_AFFINE_INVALID"):
        backend._validate_arrays(boxes(), affine)


def test_runtime_missing_fails_without_fallback(tmp_path):
    with pytest.raises(backend.PartitionError):
        backend.partition(boxes(), np.eye(4), tmp_path / "not-installed.json")


def test_memory_preflight_rejects_large_views_without_resizing():
    large = np.broadcast_to(np.zeros(1, np.float32), (2, 200, 200, 200))
    with pytest.raises(backend.PartitionError, match="PARTITION_MEMORY_BUDGET_EXCEEDED"):
        backend._validate_arrays(large, np.eye(4))


def test_real_connected_watershed_preserves_coverage_ids_and_face_connectivity(real_runtime):
    code = textwrap.dedent(r"""
        import json, sys
        import numpy as np
        from ethmoid_pipeline.counting import boundary_partition as backend
        backend._offline_guard()
        backend._runtime(sys.argv[1])
        from scipy import ndimage as ndi

        watershed = np.zeros((6, 6, 6), np.uint64)
        watershed[1:3, 1:3, 1:3] = 7
        watershed[3, 3, 3] = 7  # Diagonal contact is not six-connectivity.
        watershed[1:3, 1:3, 3] = 9  # Adjacent IDs must remain distinct.
        before = watershed.copy()
        connected = backend._connected_watershed(watershed)
        assert connected.dtype == np.uint64
        assert np.array_equal(watershed, before)
        assert np.array_equal(connected > 0, watershed > 0)
        assert connected.max() == 3
        for fragment in range(1, int(connected.max()) + 1):
            mask = connected == fragment
            assert ndi.label(mask)[1] == 1
            assert np.unique(watershed[mask]).size == 1
        print(json.dumps({"six_connected_fragments": 3}))
    """)
    assert _run_real_runtime(code, real_runtime) == {"six_connected_fragments": 3}


def test_strict_worker_schema_rejects_clinical_references_before_runtime(tmp_path):
    request = tmp_path / "request.json"
    backend._write(request, {"schema_version": 1, "reference_counts": [2, 2]})
    # Run a subprocess so the permanent offline guard cannot affect pytest.
    import subprocess, sys
    result = subprocess.run([sys.executable, "-m", backend.__name__, "--request", str(request),
                             "--output", str(tmp_path / "output")], capture_output=True, text=True)
    assert result.returncode == 1
    assert result.stdout.strip() == "PARTITION_REQUEST_INVALID"
    assert not (tmp_path / "output").exists()


def test_worker_guard_blocks_dns_and_connection_before_network_access():
    import subprocess, sys
    code = ("from ethmoid_pipeline.counting.boundary_partition import _offline_guard, PartitionError; "
            "import socket; _offline_guard(); "
            "socket.create_connection(('example.com',443))")
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode != 0
    assert "PARTITION_NETWORK_DISABLED" in result.stderr


@pytest.fixture
def real_runtime():
    value = os.environ.get("ETHMOID_BOUNDARY_RUNTIME_MANIFEST")
    if not value:
        pytest.skip("Set ETHMOID_BOUNDARY_RUNTIME_MANIFEST for real isolated CPU integration")
    assert Path(value).is_absolute() and Path(value).is_file()
    return value


def _run_real_runtime(code, real_runtime):
    manifest, runtime = backend._runtime(real_runtime)
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "1",
           "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1",
           "PANSEG_HOME": runtime["panseg_home"],
           "PYTHONPATH": os.pathsep.join([str(Path(backend.__file__).resolve().parents[2]),
                                           runtime["panseg_source_root"]])}
    prefix = Path(runtime["python_executable"]).parent
    env["PATH"] = os.pathsep.join([str(prefix / "Library" / "bin"), env.get("PATH", "")])
    completed = subprocess.run([runtime["python_executable"], "-s", "-c", code, str(manifest)],
                               env=env, capture_output=True, text=True, timeout=backend.TIMEOUT_SECONDS)
    assert completed.returncode == 0, completed.stderr
    return json.loads(completed.stdout.strip().splitlines()[-1])


def test_real_panseg_disconnected_foreground_cannot_bridge_and_repeats(real_runtime):
    values = boxes()
    before = values.copy()
    affine = np.diag([.3, .6, 1.2, 1])
    first, a = backend.partition(values, affine, real_runtime)
    second, b = backend.partition(values, affine, real_runtime)
    assert first.dtype == np.int32 and first.shape == values.shape[1:]
    assert np.array_equal(first, second) and np.array_equal(values, before)
    assert first.max() == 2 and a["foreground_components"] == 2
    assert np.all(first[values[0] == 0] == 0)
    assert first[4, 4, 4] != first[14, 14, 14]
    assert a["worker_identity"] != b["worker_identity"]
    assert a["worker_pid"] != b["worker_pid"]
    assert a["array_sha256"] == b["array_sha256"]
    assert a["pixel_pitch"] == [.3, .6, 1.2]
    assert a["native_grid"] and a["offline_guard"] and not a["cannot_link_constraints"]
    assert a["config"]["connected_watershed_superpixels"] is True
    assert a["watershed_regions"] >= a["watershed_regions_before_split"]
    assert a["watershed_fragments_added"] == a["watershed_regions"] - a["watershed_regions_before_split"]
    assert Path(a["result_path"]).is_file()


def test_real_gasp_disconnected_superpixels_cannot_couple_remote_costs(real_runtime):
    # Deliberately aliased superpixels isolate the graph failure: changing only
    # a disconnected remote region must not change the local clustering.
    code = textwrap.dedent(r"""
        import json, sys
        import numpy as np
        from ethmoid_pipeline.counting import boundary_partition as backend
        backend._offline_guard()
        backend._runtime(sys.argv[1])
        from elf.segmentation import GaspFromAffinities
        from elf.segmentation.gasp import SegmentationFeeder
        from panseg.functionals.segmentation.utils import shift_affinities

        shape = (16, 8, 8)
        watershed = np.zeros(shape, np.uint64)
        watershed[1:3, 1:5, 1:5] = 1
        watershed[3:5, 1:5, 1:5] = 2
        watershed[10:12, 1:5, 1:5] = 1
        watershed[12:14, 1:5, 1:5] = 2
        foreground = watershed > 0
        connected = backend._connected_watershed(watershed)
        assert connected.max() == 4
        assert np.array_equal(connected > 0, foreground)
        offsets = [[0, 0, 1], [0, 1, 0], [1, 0, 0]]

        def local_joined(boundary, superpixels):
            affinities = 1 - shift_affinities(np.stack([boundary] * 3), offsets)
            decoder = GaspFromAffinities(offsets, superpixel_generator=SegmentationFeeder(),
                run_GASP_kwargs={"linkage_criteria": "average", "add_cannot_link_constraints": False,
                                 "use_efficient_implementations": False}, n_threads=1, beta_bias=.5)
            raw, _ = decoder(affinities, superpixels, foreground_mask=foreground)
            assert np.all(raw[~foreground] == 0)
            return bool(raw[1, 2, 2] == raw[4, 2, 2])

        aliased, safe = [], []
        for remote_boundary in (.95, .05):
            boundary = np.ones(shape, np.float32)
            boundary[1:5, 1:5, 1:5] = .25
            boundary[10:14, 1:5, 1:5] = remote_boundary
            aliased.append(local_joined(boundary, watershed))
            safe.append(local_joined(boundary, connected))
        assert aliased == [False, True]  # The fixture actually exposes the failure.
        assert safe == [True, True]
        print(json.dumps({"aliased": aliased, "connected": safe, "offline_guard": True}))
    """)
    result = _run_real_runtime(code, real_runtime)
    assert result == {"aliased": [False, True], "connected": [True, True], "offline_guard": True}


def test_real_panseg_excluded_separator_and_connected_channel(real_runtime):
    values = np.zeros((2, 20, 20, 20), np.float32)
    values[0, 2:18, 2:18, 2:18] = 1
    values[0, 10] = 0
    values[1, 10, 2:18, 2:18] = 1
    labels, diagnostic = backend.partition(values, np.eye(4), real_runtime)
    assert labels[5, 10, 10] != labels[14, 10, 10]
    assert diagnostic["final_regions"] == 2
    values[0, 10, 2:18, 2:18] = 1
    values[1] = 0
    joined, diagnostic = backend.partition(values, np.eye(4), real_runtime)
    assert joined.max() == 1 and diagnostic["missing_foreground_voxels"] == 0


def test_real_average_gasp_does_not_promise_hard_separator_constraints(real_runtime):
    # Measured limitation: these contradictory heads produce two watershed
    # regions but one average-GASP region. Never advertise a cannot-link gate.
    values = np.zeros((2, 20, 20, 20), np.float32)
    values[0, 2:18, 2:18, 2:18] = 1
    values[1, 10, 2:18, 2:18] = 1
    labels, diagnostic = backend.partition(values, np.eye(4), real_runtime)
    assert diagnostic["watershed_regions"] == 2
    assert diagnostic["raw_gasp_regions"] == 1 and labels.max() == 1
    assert not diagnostic["cannot_link_constraints"]


def test_real_empty_foreground_and_small_components_are_reported(real_runtime):
    values = np.zeros((2, 20, 20, 20), np.float32)
    empty, diagnostic = backend.partition(values, np.eye(4), real_runtime)
    assert empty.max() == 0 and diagnostic["final_regions"] == 0
    assert diagnostic["watershed_regions_before_split"] == 0
    assert diagnostic["watershed_regions"] == diagnostic["watershed_fragments_added"] == 0
    values[0, 2:8, 2:8, 2:8] = 1
    values[0, 15, 15, 15:18] = 1
    labels, diagnostic = backend.partition(values, np.eye(4), real_runtime)
    assert labels.max() == 1
    assert diagnostic["removed_small_components"] == 1
    assert diagnostic["removed_small_voxels"] == 3
    assert diagnostic["missing_foreground_voxels"] == 3
