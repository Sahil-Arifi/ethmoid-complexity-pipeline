"""Frozen experimental localization policies; these are not clinical evidence."""
from __future__ import annotations

from .profiles import build_processing_plan, sealed

METHOD_VERSION = "research-native-0.4.0"
PROTOCOL_VERSION = "ethmoid-count-research-v4"
PLAN_VERSION = "research-localization-plan-1"
LOCALIZATION_POLICY = {"sphenoid_seed_clearance_mm": 1.0, "physical_ray_extent": True}
ANATOMY_PROXY_POLICY = {
    "maximum_voxels": 64_000_000,
    "minimum_spacing_mm": 0.5,
    "maximum_spacing_mm": 1.0,
    "spacing_increment_mm": 0.125,
    "trigger": "full_and_bounded_native_spacing_anatomy_exceed_budget",
    "extent": "complete_source_half_voxel_physical_extent",
    "purpose": "anatomical_localization_only_never_final_cell_separation",
}


def build_research_plan(profile: dict) -> dict:
    baseline = build_processing_plan(profile)
    return sealed({**baseline, "version": PLAN_VERSION,
        "localization_backend": "research_geometric_baseline",
        "localization_policy": "physical_seed_with_bounded_anatomy_proxy",
        "localization_parameters": LOCALIZATION_POLICY,
        "anatomy_proxy_fallback": ANATOMY_PROXY_POLICY,
        "clinical_accuracy": "unmeasured",
        "automatic_completion_guaranteed": False,
        "cell_separation_parameters": "frozen_v2_unchanged"})
