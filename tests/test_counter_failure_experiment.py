"""The experiment must reject a persuasive but false numerical recovery."""
from pathlib import Path
import sys

import pytest

pytest.importorskip('scipy')
pytest.importorskip('skimage')
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from counter_failure_experiment import rounding_challenge, summarize


def test_positive_aperture_rounding_twin_blocks_promotion():
    challenge = rounding_challenge()
    assert challenge['identical_full_observation']
    assert challenge['baseline']['status'] == 'withheld'
    assert challenge['candidate']['counts']['total'] == 2
    assert challenge['physical_count_open'] == 1
    assert challenge['wrong_accepted']
    result = summarize([], challenge)
    assert not result['gates']['numerical_adversary_safe']
    assert result['promotion'] == 'REJECTED'
