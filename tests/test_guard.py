from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from ethmoid_pipeline.guard import _blocked_text_content, scan_staged

from .synthetic import write_ct_series


def _git(repo: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True)


def _repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init")
    (repo / "src").mkdir()
    return repo


def _dicom_blob() -> bytes:
    return b"\x00" * 128 + b"DICM" + b"synthetic-only"


@pytest.mark.parametrize("generation", [4, 5])
def test_research_protocol_path_still_checks_medical_content(tmp_path: Path, generation: int) -> None:
    repo = _repo(tmp_path)
    path = repo / f"src/ethmoid_pipeline/counting/protocol.research.v{generation}.json"
    path.parent.mkdir(parents=True)
    path.write_text('{"protocol_version": "ethmoid-count-research-v4"}', encoding="utf-8")
    _git(repo, "add", ".")
    assert scan_staged(repo) == []
    path.write_bytes(_dicom_blob())
    _git(repo, "add", ".")
    assert any("DICOM" in failure for failure in scan_staged(repo))


def test_guard_reads_staged_blob_not_worktree(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    path = repo / "src" / "safe.py"
    path.write_bytes(_dicom_blob())
    _git(repo, "add", "src/safe.py")
    path.write_text("VALUE = 'safe worktree'\n", encoding="utf-8")
    assert any("DICOM" in failure for failure in scan_staged(repo))


def test_guard_allows_staged_source_when_worktree_is_dangerous(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    path = repo / "src" / "safe.py"
    path.write_text("VALUE = 'safe staged content'\n", encoding="utf-8")
    _git(repo, "add", "src/safe.py")
    path.write_bytes(_dicom_blob())
    assert scan_staged(repo) == []


def test_guard_rejects_nifti_magic_lfs_and_disallowed_sidecar(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    nifti = bytearray(352)
    nifti[344:348] = b"n+1\x00"
    (repo / "src" / "image.py").write_bytes(nifti)
    (repo / "src" / "pointer.py").write_text(
        "version https://git-lfs.github.com/spec/v1\n"
        "oid sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\n"
        "size 123\n",
        encoding="utf-8",
    )
    (repo / "metadata.json").write_text('{"PatientName": "SYNTHETIC"}', encoding="utf-8")
    _git(repo, "add", "-f", ".")
    failures = scan_staged(repo)
    assert any("NIfTI" in failure for failure in failures)
    assert any("Git LFS" in failure for failure in failures)
    assert any("allowlist" in failure for failure in failures)


def test_guard_rejects_valid_dicom_without_part10_preamble(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    dicom_path = write_ct_series(tmp_path / "synthetic-dicom", positions=[0.0])[0]
    blob = dicom_path.read_bytes()[132:]
    (repo / "src" / "payload.py").write_bytes(blob)
    _git(repo, "add", "src/payload.py")
    assert any("without a Part-10 preamble" in failure for failure in scan_staged(repo))


def test_guard_rejects_nifti2_uid_and_absolute_path_text(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    (repo / "docs").mkdir()
    nifti2 = bytearray(540)
    nifti2[0:4] = (540).to_bytes(4, "little")
    nifti2[4:12] = b"n+2\x00\r\n\x1a\n"
    (repo / "src" / "nifti2.py").write_bytes(nifti2)
    (repo / "docs" / "uid.md").write_text(
        "synthetic UID "
        + ".".join(["1", "2", "826", "0", "1", "3680043", "10", "543", "999"]),
        encoding="utf-8",
    )
    separator = chr(92)
    path_payloads = [
        "C:" + separator + "Synthetic" + separator + "Private",
        "/" * 2 + "synthetic-host/share/private",
        separator * 2 + "." + separator + "SyntheticDevice",
    ]
    assert all(
        _blocked_text_content(("synthetic path " + payload).encode())
        == "an absolute Windows/UNC path"
        for payload in path_payloads
    )
    for index, payload in enumerate(path_payloads):
        (repo / "docs" / f"path_{index}.md").write_text(
            "synthetic path " + payload,
            encoding="utf-8",
        )
    _git(repo, "add", ".")
    failures = scan_staged(repo)
    assert any("NIfTI-2" in failure for failure in failures)
    assert any("UID-like" in failure for failure in failures)
    assert any("Windows/UNC" in failure for failure in failures)


def test_guard_allows_web_source_but_rejects_web_image_content(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    web = repo / "web" / "src"
    web.mkdir(parents=True)
    (web / "App.tsx").write_text(
        "export function App() { return <main>Ethmoid viewer</main>; }\n",
        encoding="utf-8",
    )
    (repo / "web" / "package.json").write_text(
        '{"name":"synthetic-web","private":true}\n', encoding="utf-8"
    )
    _git(repo, "add", "web/src/App.tsx", "web/package.json")
    assert scan_staged(repo) == []

    (web / "hidden.ts").write_bytes(b"\x89PNG\r\n\x1a\nsynthetic-only")
    _git(repo, "add", "web/src/hidden.ts")
    assert any("image content" in failure for failure in scan_staged(repo))


def test_all_tracked_mode_checks_previously_committed_blobs(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    path = repo / "src" / "hidden.py"
    path.write_bytes(_dicom_blob())
    _git(repo, "add", "src/hidden.py")
    _git(repo, "-c", "user.name=Synthetic", "-c", "user.email=test@example.invalid", "commit", "-m", "Synthetic unsafe fixture")
    assert scan_staged(repo) == []
    assert any("DICOM" in failure for failure in scan_staged(repo, all_tracked=True))
