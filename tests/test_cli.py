from __future__ import annotations

from pathlib import Path

from ethmoid_pipeline import cli


def test_unexpected_failure_does_not_echo_exception_or_source_path(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    source = tmp_path / "source"
    work = tmp_path / "work"
    source.mkdir()

    def fail(*_args, **_kwargs):
        raise RuntimeError("synthetic sensitive value and source path")

    monkeypatch.setattr(cli, "build_inventory", fail)
    result = cli.main(
        [
            "inventory",
            "--dicom-root",
            str(source),
            "--work-root",
            str(work),
        ]
    )
    captured = capsys.readouterr()
    assert result == 3
    assert "synthetic sensitive value" not in captured.err
    assert str(source) not in captured.err
    assert "not displayed" in captured.err
