"""Separate research v5 method; v4 and application defaults remain available."""
from __future__ import annotations

from pathlib import Path

from .engine import analyze_volume
from .research import ResearchStrategy
from .research_v5_policy import LOCALIZATION_POLICY, METHOD_VERSION, PROTOCOL_VERSION, build_research_plan


class ResearchV5Strategy(ResearchStrategy):
    localization_policy = LOCALIZATION_POLICY

    def initial_metadata(self):
        result = super().initial_metadata()
        result.update(method_version=METHOD_VERSION, protocol_version=PROTOCOL_VERSION)
        return result

    def build_plan(self, profile):
        return build_research_plan(profile)


def analyze_research_volume(volume_path, output_dir, landmarks_dir=None, config=None, *, acquisition=None,
                            processing_plan=None, progress_path=None):
    strategy = ResearchV5Strategy(acquisition=acquisition, processing_plan=processing_plan,
        progress_path=Path(progress_path) if progress_path else None)
    return analyze_volume(volume_path, output_dir, landmarks_dir, config, acquisition=acquisition, _strategy=strategy)
