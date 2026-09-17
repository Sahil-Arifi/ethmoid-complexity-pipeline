"""Protected local source configuration; never accepted through a browser path."""
from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path

from ..util import atomic_write_json, repo_root


@dataclass(frozen=True)
class CatalogConfig:
    work_root: Path
    source_root: Path
    auto_prepare: bool = True
    minimum_free_bytes: int = 2 * 1024 ** 3

    def validate(self, app_root: Path) -> None:
        source, work, app, code = (p.resolve() for p in (self.source_root, self.work_root, app_root, repo_root()))
        if not source.is_dir() or not work.is_dir() or not app.is_relative_to(work):
            raise ValueError("CATALOG_CONFIGURATION_INVALID")
        if any(a.is_relative_to(b) or b.is_relative_to(a) for a, b in ((source, work), (source, code), (work, code))):
            raise ValueError("CATALOG_ROOTS_OVERLAP")


def load_local_config(app_root: Path) -> CatalogConfig | None:
    base = os.environ.get("LOCALAPPDATA")
    if not base:
        return None
    protected = (Path(base) / "EthmoidComplexity/work").resolve()
    # Test/custom app roots never implicitly discover or alter the real catalog.
    if app_root.resolve() != protected / "webapp":
        return None
    config_path = protected / "catalog.local.json"
    if not config_path.exists():
        mapping_path = protected / "inventory/uid_map.local.json"
        if not mapping_path.is_file():
            return None
        mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
        config = CatalogConfig(protected, Path(mapping["source_root"]))
        config.validate(app_root)
        atomic_write_json(config_path, {"schema_version": 1, "source_root": str(config.source_root.resolve()), "auto_prepare": True})
        return config
    data = json.loads(config_path.read_text(encoding="utf-8"))
    if data.get("schema_version") != 1 or type(data.get("auto_prepare", True)) is not bool:
        raise ValueError("CATALOG_CONFIGURATION_INVALID")
    config = CatalogConfig(protected, Path(data["source_root"]), data.get("auto_prepare", True))
    config.validate(app_root)
    return config
