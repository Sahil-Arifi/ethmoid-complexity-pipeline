"""Synthetic paths and captured Windows ACL commands; no patient material."""
from types import SimpleNamespace

import pytest

pytest.importorskip("scipy")

from ethmoid_pipeline.counting import boundary_io
from ethmoid_pipeline.counting import instance_experiment as legacy


def test_windows_acl_uses_sid_and_system_without_shell_or_private_output(tmp_path, monkeypatch, capsys):
    calls = []

    def run(command, **options):
        calls.append((command, options))
        return SimpleNamespace(returncode=0, stdout='"synthetic-account","S-1-5-21-123-456-789-1001"\n')

    monkeypatch.setattr(boundary_io.subprocess, "run", run)
    boundary_io._windows_acl(tmp_path)
    assert calls[0][0] == ["whoami", "/user", "/fo", "csv", "/nh"]
    assert calls[1][0] == ["icacls", str(tmp_path), "/inheritance:r", "/grant:r",
                           "*S-1-5-21-123-456-789-1001:(OI)(CI)F", "*S-1-5-18:(OI)(CI)F"]
    assert all(call[1]["capture_output"] and "shell" not in call[1] for call in calls)
    assert capsys.readouterr().out == ""


@pytest.mark.parametrize("stdout,returncode", [("private invalid identity", 0),
    ('"synthetic-account","S-1-5-21-123-456-789-1001"\n', 1)])
def test_windows_identity_failure_is_sanitized(tmp_path, monkeypatch, stdout, returncode):
    monkeypatch.setattr(boundary_io.subprocess, "run", lambda *a, **k: SimpleNamespace(stdout=stdout, returncode=returncode))
    with pytest.raises(legacy.ExperimentError, match="^PROTECTED_OUTPUT_ACL_FAILED$"):
        boundary_io._windows_acl(tmp_path)


def test_fresh_output_is_removed_on_acl_failure(tmp_path, monkeypatch):
    monkeypatch.setattr(boundary_io.os, "name", "nt")
    def fail(_):
        raise RuntimeError("synthetic-private-diagnostic")
    monkeypatch.setattr(boundary_io, "_windows_acl", fail)
    target = tmp_path / "never-populated"
    with pytest.raises(legacy.ExperimentError, match="^PROTECTED_OUTPUT_ACL_FAILED$"):
        boundary_io.new_output(target)
    assert not target.exists()


def test_fresh_output_preserves_external_and_no_overwrite_guards(tmp_path):
    output = boundary_io.new_output(tmp_path / "protected")
    assert output.is_dir()
    with pytest.raises(legacy.ExperimentError, match="FRESH_OUTPUT_REQUIRED"):
        boundary_io.new_output(output)
