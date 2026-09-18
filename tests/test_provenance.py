import importlib.metadata

import pytest

from ethmoid_pipeline import provenance


def test_uv_environment_without_pip_keeps_processing_provenance(monkeypatch):
    actual_version = importlib.metadata.version

    def version(name):
        if name == 'pip':
            raise importlib.metadata.PackageNotFoundError(name)
        return actual_version(name)

    monkeypatch.setattr(provenance.importlib.metadata, 'version', version)
    result = provenance.current_code_provenance(require_clean=False)
    assert result['dependencies']['pip'] is None
    for name in ('pydicom', 'numpy', 'pylibjpeg', 'pylibjpeg-libjpeg', 'pylibjpeg-openjpeg', 'pyjpegls'):
        assert result['dependencies'][name] == actual_version(name)
    assert len(result['git_commit']) == 40
    assert len(result['requirements_lock_sha256']) == 64


def test_missing_processing_dependency_is_not_treated_as_optional(monkeypatch):
    actual_version = importlib.metadata.version

    def version(name):
        if name == 'pydicom':
            raise importlib.metadata.PackageNotFoundError(name)
        return actual_version(name)

    monkeypatch.setattr(provenance.importlib.metadata, 'version', version)
    with pytest.raises(importlib.metadata.PackageNotFoundError):
        provenance.current_code_provenance(require_clean=False)
