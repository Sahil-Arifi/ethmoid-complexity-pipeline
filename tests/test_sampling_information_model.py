"""Independent geometry and conditional-decision checks, using development shapes."""
from itertools import permutations, product
from pathlib import Path
import sys

import numpy as np
import pytest

ndimage = pytest.importorskip('scipy.ndimage')

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from sampling_information_model import classify, render_pair


@pytest.mark.parametrize('aperture', [.125, .25])
@pytest.mark.parametrize('shift', [0., .125])
def test_exact_sampling_matches_independent_physical_predicates(aperture, shift):
    # Every physical boundary here is exactly on the independent fine grid.
    size, factor, fine_spacing = 128, 16, .03125
    centers = -2. + (np.arange(size)+.5)*fine_spacing
    x = (centers+shift)[:, None, None]
    y, z = centers[None, :, None], centers[None, None, :]
    inside = lambda coordinate, lower, upper: (coordinate >= lower) & (coordinate < upper)
    chamber = inside(x, -1., 1.) & inside(y, -1., 1.) & inside(z, -1., 1.)
    wall = inside(x, -.5, .5)
    through = wall & inside(y, -aperture/2, aperture/2) & inside(z, -aperture/2, aperture/2)
    recesses = ((inside(x, -.5, -.25) | inside(x, .25, .5))
                & inside(y, -aperture, aperture) & inside(z, -aperture/2, aperture/2))
    rendered = render_pair(aperture, .25, 0., shift)
    for count, cutout, image in zip([1, 2], [through, recesses], rendered):
        air = chamber & (~wall | cutout)
        assert ndimage.label(air)[1] == count
        assert air.sum()*fine_spacing**3 == pytest.approx(4.+aperture**2, abs=1e-12)
        expected = np.where(air, -900., 800.).reshape(
            8, factor, 8, factor, 8, factor).mean(axis=(1, 3, 5))
        np.testing.assert_array_equal(image, expected)
        assert image.shape == (8, 8, 8) and image.dtype == np.float64
        assert not image.flags.writeable
        assert np.sum((800.-image)/1700.)*.5**3 == pytest.approx(4.+aperture**2, abs=1e-12)


@pytest.mark.parametrize('aperture', [.125, .25])
def test_development_twins_are_identical_until_new_samples_are_supplied(aperture):
    original = render_pair(aperture, .25, 0.)
    assert original[0].tobytes() == original[1].tobytes()
    assert classify(original[0], original)['status'] == 'ambiguous'
    shifted = render_pair(aperture, .25, 0., .125)
    assert np.max(np.abs(shifted[0]-shifted[1])) > 10.
    for index in range(2):
        assert classify(shifted[index], shifted)['conditional_count'] == index+1


def test_positive_blur_is_integrated_before_sampling():
    from scipy.integrate import quad
    from scipy.special import ndtr
    aperture, depth, sigma, shift = .25, .25, .2, .125
    pair = render_pair(aperture, depth, sigma, shift)
    edges = [-2. + np.arange(9)*.5 + phase for phase in [shift, 0., 0.]]

    def independent_axis(axis, lower, upper):
        return np.array([quad(lambda physical: ndtr((right-physical)/sigma)
                             - ndtr((left-physical)/sigma), lower, upper,
                             epsabs=1e-12, epsrel=1e-12)[0]/(right-left)
                         for left, right in zip(edges[axis][:-1], edges[axis][1:])])

    def box(bounds):
        axes = [independent_axis(axis, *interval) for axis, interval in enumerate(bounds)]
        return axes[0][:, None, None]*axes[1][None, :, None]*axes[2][None, None, :]

    chambers = box([(-1., -.5), (-1., 1.), (-1., 1.)]) + box([(.5, 1.), (-1., 1.), (-1., 1.)])
    through = box([(-.5, .5), (-aperture/2, aperture/2), (-aperture/2, aperture/2)])
    recesses = (box([(-.5, -.5+depth), (-aperture, aperture), (-aperture/2, aperture/2)])
                + box([(.5-depth, .5), (-aperture, aperture), (-aperture/2, aperture/2)]))
    for image, holes in zip(pair, [through, recesses]):
        np.testing.assert_allclose(image, 800.-1700.*(chambers+holes), atol=2e-10, rtol=0.)


@pytest.mark.parametrize('kwargs', [
    {'aperture_mm': 0.}, {'aperture_mm': -.1}, {'aperture_mm': np.nan},
    {'aperture_mm': True}, {'aperture_mm': [.125]}, {'aperture_mm': '0.125'},
    {'aperture_mm': 1.1},  # at depth .25 the recess would exceed the chamber
    {'recess_depth_mm': 0.}, {'recess_depth_mm': -.1}, {'recess_depth_mm': .5},
    {'recess_depth_mm': np.inf}, {'recess_depth_mm': False},
    {'sigma_mm': -.1}, {'sigma_mm': np.nan}, {'sigma_mm': True},
    {'shift_x_mm': np.inf}, {'shift_x_mm': False}, {'shift_x_mm': [0.]},
])
def test_invalid_shape_and_sampling_parameters_are_rejected(kwargs):
    arguments = dict(aperture_mm=.125, recess_depth_mm=.25, sigma_mm=0., shift_x_mm=0.)
    with pytest.raises(ValueError):
        render_pair(**(arguments | kwargs))


def test_declared_model_classification_and_inclusive_error_boundary():
    first, second = np.zeros((2, 3, 4)), np.full((2, 3, 4), 10.)
    assert classify(first, [first, second]) == {
        'compatible_indices': [0], 'residual_linf': [0., 10.],
        'conditional_count': 1, 'status': 'unique'}
    assert classify(second, [first, second])['conditional_count'] == 2
    midpoint = classify(np.full(first.shape, 5.), [first, second])
    assert midpoint == {'compatible_indices': [0, 1], 'residual_linf': [5., 5.],
                        'conditional_count': None, 'status': 'ambiguous'}
    assert classify(np.full(first.shape, 20.), [first, second]) == {
        'compatible_indices': [], 'residual_linf': [20., 10.],
        'conditional_count': None, 'status': 'no_fit'}


def test_bounded_error_never_excludes_the_generating_hypothesis():
    pair = render_pair(.25, .25, 0., .125)
    perturbation = np.where(np.indices(pair[0].shape).sum(axis=0) % 2, 5., -5.)
    for index, truth in enumerate(pair):
        result = classify(truth+perturbation, pair)
        assert index in result['compatible_indices']
        assert result['conditional_count'] in (None, index+1)


@pytest.mark.parametrize('length', [0, 1, 3])
def test_exactly_two_hypotheses_are_required(length):
    image = np.zeros((2, 2, 2))
    with pytest.raises(ValueError, match='two'):
        classify(image, [image]*length)


@pytest.mark.parametrize('observation, models, bound', [
    (np.zeros((2, 2)), [np.zeros((2, 2))]*2, 5.),
    (np.zeros((2, 2, 2)), [np.zeros((2, 2, 2)), np.zeros((2, 2, 3))], 5.),
    (np.full((2, 2, 2), np.nan), [np.zeros((2, 2, 2))]*2, 5.),
    (np.zeros((2, 2, 2)), [np.full((2, 2, 2), np.inf)]*2, 5.),
    (np.zeros((2, 2, 2)), [np.zeros((2, 2, 2))]*2, -1.),
    (np.zeros((2, 2, 2)), [np.zeros((2, 2, 2))]*2, True),
])
def test_classifier_reuses_strict_numeric_and_geometry_validation(observation, models, bound):
    with pytest.raises(ValueError):
        classify(observation, models, bound)


def test_all_signed_reindexings_preserve_decisions_and_inputs():
    pair = render_pair(.25, .25, .2, .125)
    originals = [array.copy() for array in pair]
    expected = classify(pair[0], pair)
    for order, signs in product(permutations(range(3)), product([False, True], repeat=3)):
        axes = tuple(axis for axis, flipped in enumerate(signs) if flipped)
        reindexed = [np.flip(array.transpose(order), axis=axes) for array in pair]
        assert classify(reindexed[0], reindexed) == expected
    for array, before in zip(pair, originals):
        np.testing.assert_array_equal(array, before)
        assert not array.flags.writeable


def test_readonly_observation_is_not_mutated():
    pair = render_pair(.25, .25, 0., .125)
    observed = pair[0].copy()
    observed[0, 0, 0] += 3.
    observed.setflags(write=False)
    before = observed.tobytes()
    assert classify(observed, pair) == classify(observed.copy(), pair)
    assert observed.tobytes() == before and not observed.flags.writeable
