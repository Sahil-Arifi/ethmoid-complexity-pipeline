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


def test_voxel_boundary_contributes_to_both_incident_faces_in_all_axes():
    boundary = np.zeros((5, 6, 7), np.float32)
    center = (2, 3, 4)
    boundary[center] = .8
    before = boundary.copy()
    affinities = backend._voxel_boundary_affinities(boundary)
    assert affinities.shape == (3, *boundary.shape) and affinities.dtype == np.float32
    assert np.array_equal(boundary, before)
    for channel, axis in enumerate((2, 1, 0)):
        low = list(center); low[axis] -= 1
        assert affinities[(channel, *low)] == pytest.approx(.2)
        assert affinities[(channel, *center)] == pytest.approx(.2)
        assert affinities[channel, 0, 0, 0] == 1
        last = [slice(None)] * 3; last[axis] = -1
        assert np.all(affinities[(channel, *last)] == 0)


def test_voxel_boundary_faces_do_not_wrap_or_cross_excluded_background():
    boundary = np.zeros((5, 6, 7), np.float32)
    boundary[-1, 2, 3] = 1
    affinities = backend._voxel_boundary_affinities(boundary)
    assert affinities[2, 0, 2, 3] == 1  # No wrap from the opposite edge.
    assert affinities[2, -2, 2, 3] == 0
    boundary[:] = 1
    boundary[1:4, 1:5, 1:6] = 0  # Adapter supplies boundary1 outside foreground.
    affinities = backend._voxel_boundary_affinities(boundary)
    for channel, axis in enumerate((2, 1, 0)):
        low = [2, 2, 2]; low[axis] = 0
        high = [2, 2, 2]; high[axis] = boundary.shape[axis] - 2
        interior = [2, 2, 2]; interior[axis] = 1
        assert affinities[(channel, *low)] == 0
        assert affinities[(channel, *high)] == 0
        assert affinities[(channel, *interior)] == 1


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


_GASP_REGRESSION_SETUP = textwrap.dedent(r"""
    import json, sys
    import numpy as np
    from ethmoid_pipeline.counting import boundary_partition as backend
    backend._offline_guard()
    backend._runtime(sys.argv[1])
    from elf.segmentation import GaspFromAffinities
    from elf.segmentation.gasp import SegmentationFeeder
    import elf.segmentation.gasp as gasp

    captured = []
    original_gasp = gasp.run_GASP
    def capture(graph, weights, *args, **kwargs):
        captured.append((np.asarray(weights).copy(), np.asarray(kwargs['edge_sizes']).copy()))
        return original_gasp(graph, weights, *args, **kwargs)
    gasp.run_GASP = capture

    def cluster(boundary, superpixels):
        decoder = GaspFromAffinities([[0,0,1], [0,1,0], [1,0,0]],
            superpixel_generator=SegmentationFeeder(),
            run_GASP_kwargs={"linkage_criteria": "average", "add_cannot_link_constraints": False,
                             "use_efficient_implementations": False}, n_threads=1, beta_bias=.5)
        raw, _ = decoder(backend._voxel_boundary_affinities(boundary), superpixels,
                         foreground_mask=superpixels > 0)
        return raw, captured[-1]
""")


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


def test_real_complete_voxel_wall_survives_fragment_ownership_and_reflection(real_runtime):
    # A one-sided projection used to miss a complete wall when its voxels
    # belonged to the high-coordinate fragment. This isolates projection from
    # averaging, with one RAG edge and 256 uniformly repulsive contacts.
    code = _GASP_REGRESSION_SETUP + textwrap.dedent(r"""
        fixed_cases = 0
        full_cases = 0
        for axis in range(3):
            foreground = np.zeros((20,20,20), bool)
            foreground[2:18,2:18,2:18] = True
            coordinates = np.indices(foreground.shape)[axis]
            boundary = np.where(foreground, 0, 1).astype(np.float32)
            boundary[coordinates == 10] = 1
            for split in (10, 11):  # Wall voxels owned by either fragment.
                superpixels = np.where(foreground, np.where(coordinates < split, 1, 2), 0)
                for reflected in (False, True):
                    bd = np.flip(boundary, axis).copy() if reflected else boundary.copy()
                    ws = np.flip(superpixels, axis).copy() if reflected else superpixels.copy()
                    raw, (weights, sizes) = cluster(bd, ws)
                    assert weights.tolist() == [-.5] and sizes.tolist() == [256.]
                    assert np.unique(raw[ws == 1]).size == np.unique(raw[ws == 2]).size == 1
                    assert raw[ws == 1][0] != raw[ws == 2][0]
                    fixed_cases += 1
            for reflected in (False, True):
                bd = np.flip(boundary, axis).copy() if reflected else boundary.copy()
                fg = np.flip(foreground, axis).copy() if reflected else foreground.copy()
                labels, diagnostic = backend._decode(np.stack([fg.astype(np.float32), bd]), [1.,1.,1.])
                assert diagnostic['watershed_regions'] == diagnostic['raw_gasp_regions'] == 2
                assert labels.max() == 2 and np.all(labels[fg] > 0)
                labels = np.flip(labels, axis) if reflected else labels
                # Watershed tie ownership of the wall itself need not reflect;
                # both nonboundary interiors must retain the same partition.
                lo = labels[foreground & (coordinates < 10)]
                hi = labels[foreground & (coordinates > 10)]
                assert np.unique(lo).size == np.unique(hi).size == 1 and lo[0] != hi[0]
                full_cases += 1
        print(json.dumps({'fixed_fragment_cases': fixed_cases, 'full_pipeline_cases': full_cases}))
    """)
    assert _run_real_runtime(code, real_runtime) == {"fixed_fragment_cases": 12, "full_pipeline_cases": 6}


def test_real_average_gasp_partial_boundary_remains_soft_evidence(real_runtime):
    # A real average-aggregation property: 64 repulsive and 192 attractive
    # contacts yield mean affinity .75, not a hard cannot-link constraint.
    code = _GASP_REGRESSION_SETUP + textwrap.dedent(r"""
        foreground = np.zeros((20,20,20), bool)
        foreground[2:18,2:18,2:18] = True
        coordinates = np.indices(foreground.shape)[0]
        ws = np.where(foreground, np.where(coordinates < 10, 1, 2), 0)
        boundary = np.where(foreground, 0, 1).astype(np.float32)
        boundary[10,2:6,2:18] = 1
        raw, (weights, sizes) = cluster(boundary, ws)
        assert weights.tolist() == [.25] and sizes.tolist() == [256.]
        assert np.unique(raw[foreground]).size == 1
        print(json.dumps({'mean_affinity': .75, 'groups': 1}))
    """)
    assert _run_real_runtime(code, real_runtime) == {"mean_affinity": .75, "groups": 1}


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
