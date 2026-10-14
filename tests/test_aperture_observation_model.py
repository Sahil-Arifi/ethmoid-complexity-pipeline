"""Independent analytic and numerical checks for a diagnostic image model."""
from itertools import permutations, product
from pathlib import Path
import sys

import numpy as np
import pytest

integrate = pytest.importorskip('scipy.integrate')
special = pytest.importorskip('scipy.special')
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from aperture_observation_model import aperture_delta_bound, compatibility, interval_fraction, render


BASE = dict(shape=(16, 14, 12), spacing=(.5, .6, .8), origin=(-3.75, -3.9, -4.4),
            half_extent=(2.5, 2.4, 2.4), wall_mm=1., aperture_mm=0.,
            aperture_center=(0., 0.), sigma=0., air=-900., bone=800.)


def test_interval_exact_overlap_and_zero_width():
    edges = np.array([-3., -1., 0., .5, 2., 4.])
    assert np.array_equal(interval_fraction(edges, -.5, 1.5, 0.), [0., .5, 1., 2/3, 0.])
    assert np.array_equal(interval_fraction(edges, .2, .2, .3), np.zeros(5))
    assert np.array_equal(edges, [-3., -1., 0., .5, 2., 4.])


@pytest.mark.parametrize('sigma', [.03, .3, 2.])
@pytest.mark.parametrize('bounds', [(-.7, 1.4), (.1, .1001), (-8., -7.9)])
def test_gaussian_interval_matches_independent_quadrature(sigma, bounds):
    edges = np.array([-9., -1.3, -.2, 0., .3, 2.1, 9.])
    lower, upper = bounds
    # Integrate over the physical interval, not the implementation's antiderivative.
    expected = [integrate.quad(lambda t: special.ndtr((right-t)/sigma)
                              - special.ndtr((left-t)/sigma), lower, upper,
                              epsabs=1e-12, epsrel=1e-12)[0] / (right-left)
                for left, right in zip(edges[:-1], edges[1:])]
    assert np.allclose(interval_fraction(edges, lower, upper, sigma), expected,
                       rtol=2e-8, atol=2e-14)


def test_gaussian_limit_and_conservation():
    edges = np.linspace(-10., 10., 401)
    unblurred = interval_fraction(edges, -.73, 1.29, 0.)
    assert np.allclose(interval_fraction(edges, -.73, 1.29, 1e-7), unblurred, atol=1e-6)
    for sigma in [0., .3, 1.]:
        measured = interval_fraction(edges, -.73, 1.29, sigma)
        assert np.all((measured >= 0) & (measured <= 1))
        assert np.dot(measured, np.diff(edges)) == pytest.approx(2.02, abs=1e-12)


@pytest.mark.parametrize('aperture', [0., .03125, .25, 1., 4.8])
def test_zero_blur_volume_is_exact(aperture):
    hu = render(**(BASE | {'aperture_mm': aperture}))
    air_volume = np.sum((800. - hu) / 1700.) * np.prod(BASE['spacing'])
    assert air_volume == pytest.approx((5.-1.)*4.8*4.8 + aperture**2, abs=2e-12)
    assert hu.shape == BASE['shape'] and hu.dtype == np.float64
    assert hu.min() >= -900. and hu.max() <= 800.


def test_exact_voxel_means_for_half_wall_aperture():
    settings = dict(shape=(4, 4, 4), spacing=(1., 1., 1.), origin=(-1.5, -1.5, -1.5),
                    half_extent=(2., 2., 2.), wall_mm=1., aperture_mm=1.,
                    aperture_center=(0., 0.), sigma=0., air=0., bone=1.)
    hu = render(**settings)
    expected = np.zeros((4, 4, 4))
    expected[1:3] = .5
    expected[1:3, 1:3, 1:3] = .375
    assert np.array_equal(hu, expected)


def test_sampling_phase_preserves_exact_integrated_volume_and_changes_samples():
    original = render(**(BASE | {'aperture_mm': .25}))
    shifted_origin = np.array(BASE['origin']) + [.137, -.173, .219]
    shifted_origin.setflags(write=False)
    shifted = render(**(BASE | {'aperture_mm': .25, 'origin': shifted_origin}))
    assert not np.array_equal(original, shifted)
    for hu in [original, shifted]:
        assert np.sum((800.-hu)/1700.)*np.prod(BASE['spacing']) == pytest.approx(
            4.*4.8*4.8+.25**2, abs=2e-12)
    assert np.array_equal(shifted_origin, np.array(BASE['origin']) + [.137, -.173, .219])


def test_identical_hypotheses_remain_simultaneously_compatible():
    observed = render(**BASE)
    result = compatibility(observed, [observed, observed.copy()], 0.)
    assert result == {'residual_linf': [0., 0.], 'compatible_indices': [0, 1]}
    assert aperture_delta_bound(1700., 1., 0., BASE['spacing']) == 0.


def test_opposed_blind_recesses_are_an_exact_observation_twin_of_a_through_aperture():
    from scipy import ndimage
    fine_spacing, native_spacing = .03125, .5
    size, factor = 96, 16
    centers = -1.5 + (np.arange(size)+.5)*fine_spacing
    x, y, z = centers[:, None, None], centers[None, :, None], centers[None, None, :]
    inside = lambda coordinates, lower, upper: (coordinates >= lower) & (coordinates < upper)
    chamber = inside(x, -1., 1.) & inside(y, -1., 1.) & inside(z, -1., 1.)
    wall = inside(x, -.5, .5)
    through_hole = wall & inside(y, -.0625, .0625) & inside(z, -.0625, .0625)
    blind_recesses = ((inside(x, -.5, -.25) | inside(x, .25, .5))
                     & inside(y, -.125, .125) & inside(z, -.0625, .0625))
    through_air = chamber & (~wall | through_hole)
    blind_air = chamber & (~wall | blind_recesses)
    assert native_spacing == factor*fine_spacing
    assert ndimage.label(through_air)[1] == 1
    assert ndimage.label(blind_air)[1] == 2
    assert not blind_air[inside(centers, -.25, .25)].any()
    observations = []
    for air in [through_air, blind_air]:
        # The recess doubles the Y footprint and halves depth inside each X voxel.
        # Its retained central septum changes topology without changing any voxel mean.
        fine_hu = np.where(air, -900., 800.)
        observations.append(fine_hu.reshape(6, factor, 6, factor, 6, factor).mean(axis=(1, 3, 5)))
    assert observations[0].tobytes() == observations[1].tobytes()
    assert compatibility(observations[0], observations, 0.) == {
        'residual_linf': [0., 0.], 'compatible_indices': [0, 1]}


@pytest.mark.parametrize('sigma', [0., .3, 1.])
@pytest.mark.parametrize('center', [(0., 0.), (.37, -.21)])
def test_smaller_apertures_have_smaller_positive_bounded_signal(sigma, center):
    closed = render(**(BASE | {'sigma': sigma}))
    maxima = []
    previous = np.full(closed.shape, np.inf)
    for aperture in [.5, .25, .125, .03125, .001]:
        opened = render(**(BASE | {'sigma': sigma, 'aperture_mm': aperture,
                                 'aperture_center': center}))
        change = closed - opened
        bound = aperture_delta_bound(1700., 1., aperture, BASE['spacing'])
        assert np.all(change >= -1e-12) and np.all(change <= previous + 1e-12)
        assert 0 < change.max() <= bound + 1e-10
        maxima.append(change.max())
        previous = change
    assert all(a > b for a, b in zip(maxima, maxima[1:]))


def test_compatibility_returns_residuals_only_and_keeps_all_inputs():
    observation = np.array([[[1., 2.], [3., 4.]]])
    models = [observation.copy(), observation + .5, observation - 2.]
    frozen = [array.copy() for array in [observation, *models]]
    for array in [observation, *models]:
        array.setflags(write=False)
    result = compatibility(observation, models, .5)
    assert result == {'residual_linf': [0., .5, 2.], 'compatible_indices': [0, 1]}
    for array, before in zip([observation, *models], frozen):
        assert np.array_equal(array, before)


def test_all_signed_reindexings_preserve_exact_compatibility():
    sealed = render(**BASE)
    opened = render(**(BASE | {'aperture_mm': .25}))
    expected = compatibility(sealed, [sealed, opened], 3.)
    for order, signs in product(permutations(range(3)), product([False, True], repeat=3)):
        axes = tuple(axis for axis, flipped in enumerate(signs) if flipped)
        transformed = [np.flip(array.transpose(order), axis=axes) for array in [sealed, opened]]
        assert compatibility(transformed[0], transformed, 3.) == expected


@pytest.mark.parametrize('args', [([0, 1], 1, 0, 0), ([0, 1], 0, 1, -1),
                                ([0, 1], 0, 1, True), ([0, 1], 0, np.inf, 0),
                                ([0, 1], 0j, 1, 0), ([0, 0, 1], 0, 1, 0),
                                ([0, np.nan], 0, 1, 0), ([True, False], 0, 1, 0),
                                ([[0, 1]], 0, 1, 0), ([0], 0, 1, 0)])
def test_invalid_interval_inputs(args):
    with pytest.raises(ValueError):
        interval_fraction(*args)


@pytest.mark.parametrize('updates', [
    {'shape': (16., 14, 12)}, {'shape': (True, 14, 12)}, {'shape': (0, 14, 12)},
    {'shape': (16, 14)}, {'spacing': (0., .6, .8)}, {'spacing': (.5, np.inf, .8)},
    {'spacing': (True, .6, .8)}, {'origin': (0., np.nan, 0.)}, {'origin': (0j, 0., 0.)},
    {'half_extent': (0., 2.4, 2.4)}, {'wall_mm': 0.}, {'wall_mm': 5.},
    {'wall_mm': True}, {'aperture_mm': -.1}, {'aperture_mm': 4.81},
    {'aperture_center': (2.5, 0.)}, {'aperture_center': (0., 0., 0.)},
    {'aperture_mm': .5, 'aperture_center': (2.3, 0.)}, {'sigma': -.3},
    {'sigma': np.nan}, {'sigma': True}, {'air': 800.}, {'bone': np.inf}, {'air': '0'},
])
def test_invalid_render_inputs(updates):
    with pytest.raises(ValueError):
        render(**(BASE | updates))


@pytest.mark.parametrize('args', [(0., 1., .1, (.5, .6, .8)),
                                (1700., -1., .1, (.5, .6, .8)),
                                (1700., 1., -.1, (.5, .6, .8)),
                                (1700., 1., np.nan, (.5, .6, .8)),
                                (True, 1., .1, (.5, .6, .8)),
                                (1700., 1., .1, (True, .6, .8))])
def test_invalid_bound_inputs(args):
    with pytest.raises(ValueError):
        aperture_delta_bound(*args)


@pytest.mark.parametrize('observation,models,error', [
    (np.zeros((2, 2, 2)), [], 0.),
    (np.zeros((2, 2, 2)), [np.zeros((2, 2))], 0.),
    (np.zeros((2, 2)), [np.zeros((2, 2))], 0.),
    (np.zeros((0, 2, 2)), [np.zeros((0, 2, 2))], 0.),
    (np.zeros((2, 2, 2), bool), [np.zeros((2, 2, 2))], 0.),
    (np.zeros((2, 2, 2)), [np.full((2, 2, 2), np.nan)], 0.),
    (np.zeros((2, 2, 2)), [np.zeros((2, 2, 2), complex)], 0.),
    (np.zeros((2, 2, 2)), [np.zeros((2, 2, 2))], -.1),
    (np.zeros((2, 2, 2)), [np.zeros((2, 2, 2))], True),
    (np.zeros((2, 2, 2)), [np.zeros((2, 2, 2))], np.inf),
])
def test_invalid_compatibility_inputs(observation, models, error):
    with pytest.raises(ValueError):
        compatibility(observation, models, error)
