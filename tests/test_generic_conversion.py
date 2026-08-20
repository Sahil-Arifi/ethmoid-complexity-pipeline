from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from ethmoid_pipeline.conversion import convert_pilot, convert_series
from ethmoid_pipeline.inventory import build_inventory
from ethmoid_pipeline.util import PipelineError, default_dcm2niix, sha256_file
from ethmoid_pipeline.verification import verify_conversion, verify_pilot

from .synthetic import rewrite, write_ct_series


@pytest.mark.integration
def test_generic_conversion_verifies_arbitrary_hu_scaling_without_inventory(
    tmp_path: Path,
) -> None:
    executable = Path(os.environ.get("ETHMOID_DCM2NIIX", default_dcm2niix()))
    if not executable.is_file():
        pytest.skip("pinned dcm2niix is not installed")

    source = tmp_path / "source"
    work = tmp_path / "work"
    paths = write_ct_series(
        source,
        positions=[0.0, 0.8, 1.6, 2.4, 3.2],
        instance_numbers=[5, 4, 3, 2, 1],
    )
    for path in paths:
        rewrite(
            path,
            SliceThickness=0.8,
            SpacingBetweenSlices=0.8,
            RescaleSlope=2,
            RescaleIntercept=-900,
        )
    before = [sha256_file(path) for path in paths]
    provenance = {"kind": "synthetic-test", "version": 1, "git_dirty": True}

    converted = convert_series(
        source,
        work,
        paths,
        "Case001",
        "Series001",
        executable,
        pipeline_provenance=provenance,
    )
    unrelated_inventory = work / "inventory" / "uid_map.local.json"
    unrelated_inventory.parent.mkdir(parents=True)
    unrelated_inventory.write_text('{"unrelated": true}\n', encoding="utf-8")
    report = verify_conversion(
        source,
        work,
        paths,
        "Case001",
        "Series001",
        pipeline_provenance=provenance,
    )

    conversion_report = json.loads(
        (converted["output_dir"] / "conversion_report.json").read_text(encoding="utf-8")
    )
    assert report["status"] == "AUTOMATED_PASS_PENDING_MANUAL"
    assert report["issues"] == []
    assert report["metrics"]["max_absolute_hu_error"] == pytest.approx(0.0)
    assert "pipeline provenance records uncommitted source changes" in report["warnings"]
    provenance_check = next(
        item for item in report["checks"] if item["name"] == "pipeline provenance"
    )
    assert "uncommitted changes were recorded" in provenance_check["summary"]
    assert "clean committed" not in provenance_check["summary"]
    assert "inventory_uid_map_sha256" not in conversion_report
    assert conversion_report["ordered_source_manifest_sha256"] == converted[
        "ordered_source_manifest_sha256"
    ]
    assert not any(item["name"] == "NIfTI dtype and scaling" for item in report["checks"])
    assert "complete 5-slice stack" in (
        converted["output_dir"] / "SLICER_QC_CHECKLIST.md"
    ).read_text(encoding="utf-8")
    assert [sha256_file(path) for path in paths] == before


def test_legacy_pilot_wrappers_remain_restricted_to_audited_selection(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    work = tmp_path / "work"
    write_ct_series(source)
    build_inventory(source, work)

    with pytest.raises(PipelineError, match="restricted to audited Case003/Series007"):
        convert_pilot(
            source,
            work,
            "Case001",
            "Series001",
            tmp_path / "not-used.exe",
        )
    with pytest.raises(PipelineError, match="restricted to audited Case003/Series007"):
        verify_pilot(source, work, "Case001", "Series001")
