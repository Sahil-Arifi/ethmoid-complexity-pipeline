"""Exact voxel averages of idealized rectangular air spaces; diagnostic only.

The background is constant bone, the chamber is centered at physical zero, and
its separating wall is normal to X. No scanner calibration or counts are inferred.
"""
import numpy as np
from scipy.special import ndtr


def _real(value, name):
    array = np.asarray(value)
    if (array.dtype.kind not in 'iuf' or
            any(isinstance(item, (bool, np.bool_)) for item in np.asarray(value, dtype=object).flat)):
        raise ValueError(f'{name} must contain real numbers, excluding booleans.')
    array = array.astype(np.float64, copy=False)
    if not np.isfinite(array).all():
        raise ValueError(f'{name} must be finite.')
    return array


def _scalar(value, name, minimum=None, strict=False):
    array = _real(value, name)
    if array.shape != ():
        raise ValueError(f'{name} must be scalar.')
    result = float(array)
    if minimum is not None and (result <= minimum if strict else result < minimum):
        raise ValueError(f'{name} is outside its allowed range.')
    return result


def _vector(value, name, size=3, positive=False):
    array = _real(value, name)
    if array.shape != (size,) or (positive and np.any(array <= 0)):
        raise ValueError(f'{name} must have {size} valid coordinates.')
    return array


def interval_fraction(edges, lower, upper, sigma):
    """Average a Gaussian-blurred interval over each voxel between adjacent edges."""
    edges = _real(edges, 'edges')
    lower, upper = _scalar(lower, 'lower'), _scalar(upper, 'upper')
    sigma = _scalar(sigma, 'sigma', 0.)
    if (edges.ndim != 1 or len(edges) < 2 or np.any(np.diff(edges) <= 0)
            or lower > upper or not np.isfinite(upper-lower)):
        raise ValueError('Edges must increase strictly and interval bounds must be ordered.')
    widths = np.diff(edges)
    if not np.isfinite(widths).all():
        raise ValueError('Voxel widths must be finite.')
    overlap = np.maximum(0., np.minimum(edges[1:], upper)-np.maximum(edges[:-1], lower))
    if sigma == 0 or lower == upper:
        return overlap / widths
    # G(t)=max(t,0)+C(t). Separating the linear part avoids cancellation in far tails.
    def correction(t):
        magnitude = np.abs(t)
        if not np.isfinite(magnitude).all():
            raise ValueError('Interval offsets must be finite.')
        with np.errstate(over='ignore', under='ignore'):
            scaled = magnitude / sigma
            return sigma*np.exp(-.5*scaled*scaled)/np.sqrt(2*np.pi)-magnitude*ndtr(-scaled)
    correction_edges = correction(edges-lower)-correction(edges-upper)
    result = (overlap + np.diff(correction_edges)) / widths
    # Floating-point tail cancellation can yield a tiny negative occupancy.
    return np.clip(result, 0., np.minimum(1., (upper-lower)/widths))


def render(shape, spacing, origin, half_extent, wall_mm, aperture_mm,
           aperture_center, sigma, air=-900., bone=800.):
    """Return float64 HU; origin is the first voxel center in physical millimeters."""
    shape_values = np.asarray(shape, dtype=object)
    if (shape_values.shape != (3,) or any(
            isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
            or value <= 0 for value in shape_values)):
        raise ValueError('shape must contain three positive integers.')
    shape = tuple(int(value) for value in shape_values)
    spacing = _vector(spacing, 'spacing', positive=True)
    origin = _vector(origin, 'origin')
    half_extent = _vector(half_extent, 'half_extent', positive=True)
    wall = _scalar(wall_mm, 'wall_mm', 0., strict=True)
    aperture = _scalar(aperture_mm, 'aperture_mm', 0.)
    center = _vector(aperture_center, 'aperture_center', size=2)
    sigma = _scalar(sigma, 'sigma', 0.)
    air, bone = _scalar(air, 'air'), _scalar(bone, 'bone')
    if (wall >= 2*half_extent[0] or np.any(np.abs(center)+aperture/2 > half_extent[1:])
            or bone <= air or not np.isfinite(bone-air)):
        raise ValueError('Wall/aperture must fit the chamber and bone HU must exceed air HU.')
    edges = [origin[axis] + (np.arange(shape[axis]+1)-.5)*spacing[axis] for axis in range(3)]
    box_axes = [interval_fraction(edge, -extent, extent, sigma)
                for edge, extent in zip(edges, half_extent)]
    wall_x = interval_fraction(edges[0], -wall/2, wall/2, sigma)
    box = box_axes[0][:, None, None]*box_axes[1][None, :, None]*box_axes[2][None, None, :]
    wall_fraction = wall_x[:, None, None]*box_axes[1][None, :, None]*box_axes[2][None, None, :]
    aperture_yz = [interval_fraction(edge, location-aperture/2, location+aperture/2, sigma)
                   for edge, location in zip(edges[1:], center)]
    hole = wall_x[:, None, None]*aperture_yz[0][None, :, None]*aperture_yz[1][None, None, :]
    air_fraction = np.clip(box-wall_fraction+hole, 0., 1.)
    return bone-(bone-air)*air_fraction


def aperture_delta_bound(contrast, wall_mm, aperture_mm, spacing):
    """Conservative max HU change for normalized Gaussian blur and voxel averaging."""
    contrast = _scalar(contrast, 'contrast', 0., strict=True)
    wall = _scalar(wall_mm, 'wall_mm', 0., strict=True)
    aperture = _scalar(aperture_mm, 'aperture_mm', 0.)
    spacing = _vector(spacing, 'spacing', positive=True)
    return float(contrast*np.prod(np.minimum(np.array([wall, aperture, aperture])/spacing, 1.)))


def compatibility(observation, hypothesis_arrays, error_bound):
    """Compare supplied observations, never labels; compatibility is not a count."""
    observed = _real(observation, 'observation')
    bound = _scalar(error_bound, 'error_bound', 0.)
    if observed.ndim != 3 or any(size == 0 for size in observed.shape):
        raise ValueError('observation must be a nonempty 3D array.')
    residuals = []
    for hypothesis in hypothesis_arrays:
        expected = _real(hypothesis, 'hypothesis')
        if expected.shape != observed.shape:
            raise ValueError('Every hypothesis must match the observation shape.')
        with np.errstate(over='ignore'):
            residual = float(np.max(np.abs(observed-expected)))
        if not np.isfinite(residual):
            raise ValueError('The HU residual must be finite.')
        residuals.append(residual)
    if not residuals:
        raise ValueError('At least one hypothesis is required.')
    return {'residual_linf': residuals,
            'compatible_indices': [index for index, residual in enumerate(residuals) if residual <= bound]}
