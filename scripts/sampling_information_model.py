"""Conditional two-shape sampling diagnostic; not a clinical counting model."""
import numpy as np

from aperture_observation_model import _scalar, compatibility, interval_fraction, render


def render_pair(aperture_mm, recess_depth_mm, sigma_mm, shift_x_mm=0.):
    """Return known through-opening and blind-recess predictions, in that order.

    Dimensions and material values are prescribed rather than inferred. Continuous
    Gaussian blur precedes exact averaging on an 8 cubed grid of 0.5 mm voxels.
    A positive X shift changes the physical sampling grid; it does not move the
    structures. The wall is fixed at 1 mm, so recess Y width is aperture/(2*depth)
    in the stated millimeter units. Both shapes have equal air volume and different
    physical topology.
    """
    aperture = _scalar(aperture_mm, 'aperture_mm', 0., strict=True)
    depth = _scalar(recess_depth_mm, 'recess_depth_mm', 0., strict=True)
    sigma = _scalar(sigma_mm, 'sigma_mm', 0.)
    shift = _scalar(shift_x_mm, 'shift_x_mm')
    recess_width = aperture/(2.*depth)
    if depth >= .5 or aperture > 2. or recess_width > 2.:
        raise ValueError('Both openings must fit the chamber and leave a positive blind septum.')
    settings = dict(shape=(8, 8, 8), spacing=(.5, .5, .5),
                    origin=(-1.75+shift, -1.75, -1.75), half_extent=(1., 1., 1.),
                    wall_mm=1., aperture_center=(0., 0.), sigma=sigma)
    through = render(**settings, aperture_mm=aperture)
    sealed = render(**settings, aperture_mm=0.)
    edges = [-2. + np.arange(9)*.5 + phase for phase in [shift, 0., 0.]]
    recess_x = (interval_fraction(edges[0], -.5, -.5+depth, sigma)
                + interval_fraction(edges[0], .5-depth, .5, sigma))
    recess_y = interval_fraction(edges[1], -recess_width/2., recess_width/2., sigma)
    recess_z = interval_fraction(edges[2], -aperture/2., aperture/2., sigma)
    blind = sealed - 1700.*recess_x[:, None, None]*recess_y[None, :, None]*recess_z[None, None, :]
    for image in [through, blind]:
        image.setflags(write=False)
    return through, blind


def classify(observation, hypothesis_arrays, error_bound=5.):
    """Select a count only within the two supplied, calibrated shape hypotheses.

    Index zero denotes a through-opening (one space), index one blind recesses
    (two spaces). A unique fit is conditional on this model, not clinical proof.
    """
    hypotheses = list(hypothesis_arrays)
    if len(hypotheses) != 2:
        raise ValueError('Exactly two hypotheses, through then blind, are required.')
    result = compatibility(observation, hypotheses, error_bound)
    compatible = result['compatible_indices']
    return result | {
        'conditional_count': compatible[0]+1 if len(compatible) == 1 else None,
        'status': 'unique' if len(compatible) == 1 else ('ambiguous' if compatible else 'no_fit')}
