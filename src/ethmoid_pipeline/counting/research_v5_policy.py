"""Opt-in seed fallback experiment; software regression control, not accuracy."""
from __future__ import annotations

from .profiles import sealed
from .research_policy import build_research_plan as build_v4_plan

METHOD_VERSION = "research-native-0.5.0"
PROTOCOL_VERSION = "ethmoid-count-research-v5"
PLAN_VERSION = "research-localization-plan-2"
LOCALIZATION_POLICY = {
    "sphenoid_seed_mode": "preserve_legacy_then_thick_empty_fallback",
    "sphenoid_fallback_clearance_mm": 1.0,
    "fallback_requires_max_spacing_gt_mm": 1.0,
    "physical_ray_extent": True,
}


def build_research_plan(profile: dict) -> dict:
    return sealed({**build_v4_plan(profile), "version": PLAN_VERSION,
        "localization_policy": "legacy_seed_preserving_thick_empty_fallback",
        "localization_parameters": LOCALIZATION_POLICY,
        "seed_spacing_reference": "current_anatomical_localization_grid_not_nominal_slice_thickness",
        "supported_legacy_seed_policy": "preserve_exact_existing_voxels",
        "regression_control": "engineering_only_not_anatomical_validation"})
