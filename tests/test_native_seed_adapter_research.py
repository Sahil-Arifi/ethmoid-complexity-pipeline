"""Source-only adapter contracts; no clinical fixtures or counting closure."""
import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("scipy")
pytest.importorskip("skimage")
from scipy import ndimage as ndi
from skimage.morphology import h_maxima

from ethmoid_pipeline.counting import native_seed_adapter_research as adapter


SYNTHETIC_SOURCE = '''def marker_field(distance, free, air_distance, air_lumen, cfg, ridge, spacing):
    maxima = h_maxima(distance.astype(np.float32), cfg.seed_prominence_mm) & free
    if air_lumen.any():
        air_maxima = h_maxima(air_distance.astype(np.float32), cfg.seed_prominence_mm) & air_lumen
        maxima |= air_maxima
    return maxima
'''


@pytest.fixture
def synthetic_source_pin(monkeypatch):
    # Exercise the same closed transformation with a public, nonclinical fixture.
    monkeypatch.setattr(adapter, "FROZEN_V3_SHA256", hashlib.sha256(SYNTHETIC_SOURCE.encode()).hexdigest())


@pytest.mark.parametrize("source", ["", "print('untrusted')\n", b"not source text"])
def test_only_exact_frozen_source_is_accepted(source):
    assert adapter.FROZEN_V3_SHA256 == "2411b5d9b5220852e9ae26e537e28873a883cff799dd3ba7a7fbe72c32549b5d"
    with pytest.raises(adapter.SeedAdapterError, match="^SEED_SOURCE_MISMATCH$"):
        adapter.build_seed_candidate_source(source)


def test_normalized_port_keeps_both_marker_fields_boolean(synthetic_source_pin):
    generated = adapter.build_seed_candidate_source(SYNTHETIC_SOURCE)
    assert generated == adapter.build_seed_candidate_source(SYNTHETIC_SOURCE.replace("\n", "\r\n"))
    assert generated == adapter.build_seed_candidate_source(SYNTHETIC_SOURCE.replace("\n", "\r"))
    namespace = {"np": np, "ndi": ndi, "h_maxima": h_maxima}
    exec(compile(generated, "<synthetic-marker-port>", "exec"), namespace)
    free = np.zeros((17, 13, 13), bool)
    free[1:-1, 1:-1, 1:-1] = True
    ridge = np.zeros_like(free)
    ridge[8, 1:-1, 1:-1] = True
    air = free & ~ridge
    edt = lambda mask: ndi.distance_transform_edt(np.pad(mask, 1))[1:-1, 1:-1, 1:-1]
    markers = namespace["marker_field"](edt(free & ~ridge), free, edt(air), air,
                                         SimpleNamespace(seed_prominence_mm=1), ridge, np.ones(3))
    assert markers.dtype == np.dtype(bool)
    assert ndi.label(markers)[1] == 2
    assert not (markers & ridge).any()


def test_cli_never_executes_rejected_source(tmp_path, capsys):
    source, output, sentinel = tmp_path / "input.py", tmp_path / "output.py", tmp_path / "executed"
    source.write_text(f"from pathlib import Path\nPath({str(sentinel)!r}).touch()\n")
    assert adapter.main(["--source", str(source), "--output", str(output)]) == 2
    assert not sentinel.exists() and not output.exists()
    assert json.loads(capsys.readouterr().out)["failure_code"] == "SEED_SOURCE_MISMATCH"


def test_cli_writes_fresh_output_and_refuses_overwrite(tmp_path, capsys, synthetic_source_pin):
    source, output = tmp_path / "input.py", tmp_path / "output.py"
    source.write_text(SYNTHETIC_SOURCE)
    arguments = ["--source", str(source), "--output", str(output)]
    assert adapter.main(arguments) == 0
    expected = adapter.build_seed_candidate_source(SYNTHETIC_SOURCE).encode()
    assert output.read_bytes() == expected
    assert json.loads(capsys.readouterr().out)["candidate_sha256"] == hashlib.sha256(expected).hexdigest()
    assert adapter.main(arguments) == 2
    assert output.read_bytes() == expected
    assert json.loads(capsys.readouterr().out)["failure_code"] == "SEED_OUTPUT_EXISTS"
