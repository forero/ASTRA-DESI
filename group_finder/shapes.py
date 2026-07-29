from dataclasses import dataclass
import json, os
from numbers import Integral, Real
from pathlib import Path
import tempfile
import numpy as np


DEFAULT_MIN_RANDOM_MEMBERS = 10
DEFAULT_N_BOOTSTRAP = 200
DEFAULT_BOOTSTRAP_SEED = 12345
DEFAULT_MAX_RELATIVE_R_EFF_SIGMA = 0.25
DEFAULT_MAX_ELLIPTICITY_SIGMA = 0.10
DEFAULT_MIN_VALID_BOOTSTRAP_FRACTION = 0.80

QUALITY_VALID = 'valid'
QUALITY_NO_RANDOM_MEMBERS = 'no_random_members'
QUALITY_INSUFFICIENT_RANDOM_MEMBERS = 'insufficient_random_members'
QUALITY_DEGENERATE_MOMENT = 'degenerate_moment'
QUALITY_INSUFFICIENT_VALID_BOOTSTRAP = 'insufficient_valid_bootstrap'
QUALITY_UNSTABLE_R_EFF = 'unstable_r_eff'
QUALITY_UNSTABLE_ELLIPTICITY = 'unstable_ellipticity'
QUALITY_UNSTABLE_BOTH = 'unstable_r_eff_and_ellipticity'
_QUALITY_REASON_DTYPE = '<U40'


@dataclass(frozen=True)
class VoidShapes:
    group_id: np.ndarray
    n_members: np.ndarray
    n_data: np.ndarray
    n_random: np.ndarray
    center: np.ndarray
    lambda_values: np.ndarray
    semi_axes: np.ndarray
    r_eff: np.ndarray
    ellipticity: np.ndarray
    r_eff_sigma: np.ndarray
    relative_r_eff_sigma: np.ndarray
    ellipticity_sigma: np.ndarray
    n_bootstrap_valid: np.ndarray
    bootstrap_valid_fraction: np.ndarray
    quality_reason: np.ndarray
    valid_shape: np.ndarray


def _positions_array(values):
    positions = np.asarray(values, dtype=np.float64)
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError('positions must have shape (n_points, 3).')
    if not np.all(np.isfinite(positions)):
        raise ValueError('positions contains non-finite coordinates.')
    return positions


def _boolean_labels(values, n_points):
    labels = np.asarray(values)
    if labels.ndim != 1 or len(labels) != n_points:
        raise ValueError(
            'is_data must be one-dimensional and match positions.')
    if labels.dtype.kind != 'b':
        raise TypeError('is_data must have boolean dtype.')
    return labels


def _group_labels(values, n_points):
    labels = np.asarray(values)
    if labels.ndim != 1 or len(labels) != n_points:
        raise ValueError(
            'group_ids must be one-dimensional and match positions.')
    if labels.dtype.kind not in 'iu':
        raise TypeError('group_ids must have integer dtype.')
    return labels.astype(np.int64, copy=False)


def _positive_scale(value):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError('coordinate_scale must be a positive real number.')
    scale = float(value)
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError('coordinate_scale must be finite and positive.')
    return scale


def _minimum_random_members(value):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError('min_random_members must be an integer of at least 4.')
    minimum = int(value)
    if minimum < 4:
        raise ValueError('min_random_members must be at least 4.')
    return minimum


def _integer_at_least(value, name, minimum):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError(f'{name} must be an integer of at least {minimum}.')
    result = int(value)
    if result < minimum:
        raise ValueError(f'{name} must be at least {minimum}.')
    return result


def _nonnegative_seed(value):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError('bootstrap_seed must be a non-negative integer.')
    seed = int(value)
    if seed < 0:
        raise ValueError('bootstrap_seed must be non-negative.')
    return seed


def _nonnegative_finite(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f'{name} must be a non-negative real number.')
    result = float(value)
    if not np.isfinite(result) or result < 0.0:
        raise ValueError(f'{name} must be finite and non-negative.')
    return result


def _fraction(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f'{name} must be a real number in (0, 1].')
    result = float(value)
    if not np.isfinite(result) or not 0.0 < result <= 1.0:
        raise ValueError(f'{name} must lie in (0, 1].')
    return result


def _empty_shapes() -> VoidShapes:
    empty_int = np.empty(0, dtype=np.int64)
    empty_float = np.empty(0, dtype=np.float64)
    return VoidShapes(group_id=empty_int.copy(),
                      n_members=empty_int.copy(),
                      n_data=empty_int.copy(),
                      n_random=empty_int.copy(),
                      center=np.empty((0, 3), dtype=np.float64),
                      lambda_values=np.empty((0, 3), dtype=np.float64),
                      semi_axes=np.empty((0, 3), dtype=np.float64),
                      r_eff=empty_float.copy(),
                      ellipticity=empty_float.copy(),
                      r_eff_sigma=empty_float.copy(),
                      relative_r_eff_sigma=empty_float.copy(),
                      ellipticity_sigma=empty_float.copy(),
                      n_bootstrap_valid=empty_int.copy(),
                      bootstrap_valid_fraction=empty_float.copy(),
                      quality_reason=np.empty(0, dtype=_QUALITY_REASON_DTYPE),
                      valid_shape=np.empty(0, dtype=bool))


def _moment_measurements(positions):
    group_center = np.mean(positions, axis=0, dtype=np.float64)
    centered = positions - group_center
    shape = centered.T @ centered / float(len(positions))
    shape = 0.5 * (shape + shape.T)
    eigenvalues = np.linalg.eigvalsh(shape)[::-1]

    machine_epsilon = np.finfo(np.float64).eps
    eigenvalue_scale = max(float(np.max(np.abs(eigenvalues))),
                           np.finfo(np.float64).tiny)
    negative_tolerance = 64.0 * machine_epsilon * eigenvalue_scale
    if np.any(eigenvalues < -negative_tolerance):
        return None
    eigenvalues = np.clip(eigenvalues, 0.0, None)

    rank_tolerance = 3.0 * machine_epsilon * eigenvalues[0]
    if eigenvalues[0] <= 0.0 or eigenvalues[2] <= rank_tolerance:
        return None

    axes = np.sqrt(5.0 * eigenvalues)

    # sqrt(5) * (lambda_1 * lambda_2 * lambda_3)**(1/6).
    effective_radius = np.sqrt(5.0) * np.exp(
        np.sum(np.log(eigenvalues)) / 6.0)
    moment_ratio = ((eigenvalues[2] + eigenvalues[1]) /
                    (eigenvalues[1] + eigenvalues[0]))
    moment_ratio = float(np.clip(moment_ratio, 0.0, 1.0))
    ellipticity = 1.0 - moment_ratio ** 0.25
    return (group_center, eigenvalues, axes, float(effective_radius), float(ellipticity))


def _group_rng(seed, group_id) -> np.random.Generator:
    identifier = int(group_id)
    low = identifier & 0xFFFFFFFF
    high = (identifier >> 32) & 0xFFFFFFFF
    sequence = np.random.SeedSequence(seed, spawn_key=(low, high))
    return np.random.default_rng(sequence)


def _bootstrap_uncertainties(positions, group_id, n_bootstrap, bootstrap_seed):
    member_order = np.lexsort((positions[:, 2], positions[:, 1], positions[:, 0]))
    positions = positions[member_order]
    n_random = len(positions)
    rng = _group_rng(bootstrap_seed, group_id)
    radius_samples = np.empty(n_bootstrap, dtype=np.float64)
    ellipticity_samples = np.empty(n_bootstrap, dtype=np.float64)
    n_valid = 0

    for _ in range(n_bootstrap):
        indices = rng.integers(0, n_random, size=n_random)
        measurements = _moment_measurements(positions[indices])
        if measurements is None:
            continue
        radius_samples[n_valid] = measurements[3]
        ellipticity_samples[n_valid] = measurements[4]
        n_valid += 1

    if n_valid < 2:
        return n_valid, np.nan, np.nan
    return (n_valid,
            float(np.std(radius_samples[:n_valid], ddof=1)),
            float(np.std(ellipticity_samples[:n_valid], ddof=1)))


def compute_void_shapes(positions, is_data, group_ids, coordinate_scale=1.0,
                        min_random_members=DEFAULT_MIN_RANDOM_MEMBERS,
                        n_bootstrap=DEFAULT_N_BOOTSTRAP,
                        bootstrap_seed=DEFAULT_BOOTSTRAP_SEED,
                        max_relative_r_eff_sigma=DEFAULT_MAX_RELATIVE_R_EFF_SIGMA,
                        max_ellipticity_sigma=DEFAULT_MAX_ELLIPTICITY_SIGMA,
                        min_valid_bootstrap_fraction=DEFAULT_MIN_VALID_BOOTSTRAP_FRACTION) -> VoidShapes:

    xyz = _positions_array(positions)
    data_labels = _boolean_labels(is_data, len(xyz))
    groups = _group_labels(group_ids, len(xyz))
    scale = _positive_scale(coordinate_scale)
    minimum = _minimum_random_members(min_random_members)
    bootstrap_count = _integer_at_least(
        n_bootstrap, 'n_bootstrap', minimum=2)
    seed = _nonnegative_seed(bootstrap_seed)
    maximum_relative_radius_sigma = _nonnegative_finite(
        max_relative_r_eff_sigma, 'max_relative_r_eff_sigma')
    maximum_ellipticity_sigma = _nonnegative_finite(
        max_ellipticity_sigma, 'max_ellipticity_sigma')
    minimum_valid_fraction = _fraction(
        min_valid_bootstrap_fraction, 'min_valid_bootstrap_fraction')
    minimum_valid_bootstraps = max(2, int(np.ceil(minimum_valid_fraction * bootstrap_count)))

    retained_group_ids = np.unique(groups[groups >= 0])
    if len(retained_group_ids) == 0:
        return _empty_shapes()

    retained_group_ids = retained_group_ids.astype(np.int64, copy=False)
    n_groups = len(retained_group_ids)
    n_members = np.zeros(n_groups, dtype=np.int64)
    n_data = np.zeros(n_groups, dtype=np.int64)
    n_random = np.zeros(n_groups, dtype=np.int64)
    center = np.full((n_groups, 3), np.nan, dtype=np.float64)
    lambda_values = np.full((n_groups, 3), np.nan, dtype=np.float64)
    semi_axes = np.full((n_groups, 3), np.nan, dtype=np.float64)

    r_eff = np.full(n_groups, np.nan, dtype=np.float64)
    ellipticity = np.full(n_groups, np.nan, dtype=np.float64)
    r_eff_sigma = np.full(n_groups, np.nan, dtype=np.float64)
    relative_r_eff_sigma = np.full(n_groups, np.nan, dtype=np.float64)

    ellipticity_sigma = np.full(n_groups, np.nan, dtype=np.float64)
    n_bootstrap_valid = np.zeros(n_groups, dtype=np.int64)
    bootstrap_valid_fraction = np.zeros(n_groups, dtype=np.float64)
    quality_reason = np.full(n_groups,
                             QUALITY_NO_RANDOM_MEMBERS,
                             dtype=_QUALITY_REASON_DTYPE)
    valid_shape = np.zeros(n_groups, dtype=bool)

    scaled_positions = xyz * scale

    for row, group_id in enumerate(retained_group_ids):
        member_mask = groups == group_id
        data_mask = member_mask & data_labels
        random_mask = member_mask & ~data_labels

        n_members[row] = np.count_nonzero(member_mask)
        n_data[row] = np.count_nonzero(data_mask)
        n_random[row] = np.count_nonzero(random_mask)
        if n_random[row] == 0:
            continue

        random_positions = scaled_positions[random_mask]
        group_center = np.mean(random_positions, axis=0, dtype=np.float64)
        center[row] = group_center
        if n_random[row] < minimum:
            quality_reason[row] = QUALITY_INSUFFICIENT_RANDOM_MEMBERS
            continue

        measurements = _moment_measurements(random_positions)
        if measurements is None:
            quality_reason[row] = QUALITY_DEGENERATE_MOMENT
            continue

        (_, eigenvalues, axes, effective_radius, shape_ellipticity) = measurements
        lambda_values[row] = eigenvalues
        semi_axes[row] = axes
        r_eff[row] = effective_radius
        ellipticity[row] = shape_ellipticity

        (n_valid_bootstraps, radius_sigma, shape_ellipticity_sigma) = _bootstrap_uncertainties(random_positions,
                                                                                               group_id=int(group_id),
                                                                                               n_bootstrap=bootstrap_count,
                                                                                               bootstrap_seed=seed)
        n_bootstrap_valid[row] = n_valid_bootstraps
        bootstrap_valid_fraction[row] = (
            n_valid_bootstraps / float(bootstrap_count))
        r_eff_sigma[row] = radius_sigma
        ellipticity_sigma[row] = shape_ellipticity_sigma
        if np.isfinite(radius_sigma):
            relative_r_eff_sigma[row] = radius_sigma / effective_radius

        if n_valid_bootstraps < minimum_valid_bootstraps:
            quality_reason[row] = QUALITY_INSUFFICIENT_VALID_BOOTSTRAP
            continue

        radius_stable = (
            relative_r_eff_sigma[row] <= maximum_relative_radius_sigma)
        ellipticity_stable = (
            ellipticity_sigma[row] <= maximum_ellipticity_sigma)
        if not radius_stable and not ellipticity_stable:
            quality_reason[row] = QUALITY_UNSTABLE_BOTH
            continue
        if not radius_stable:
            quality_reason[row] = QUALITY_UNSTABLE_R_EFF
            continue
        if not ellipticity_stable:
            quality_reason[row] = QUALITY_UNSTABLE_ELLIPTICITY
            continue

        quality_reason[row] = QUALITY_VALID
        valid_shape[row] = True

    return VoidShapes(group_id=retained_group_ids,
                      n_members=n_members,
                      n_data=n_data,
                      n_random=n_random,
                      center=center,
                      lambda_values=lambda_values,
                      semi_axes=semi_axes,
                      r_eff=r_eff,
                      ellipticity=ellipticity,
                      r_eff_sigma=r_eff_sigma,
                      relative_r_eff_sigma=relative_r_eff_sigma,
                      ellipticity_sigma=ellipticity_sigma,
                      n_bootstrap_valid=n_bootstrap_valid,
                      bootstrap_valid_fraction=bootstrap_valid_fraction,
                      quality_reason=quality_reason,
                      valid_shape=valid_shape)


def write_void_shapes_npz(path, shapes: VoidShapes, metadata, overwrite=False):
    path = Path(path)
    if path.suffix.lower() != '.npz':
        raise ValueError('path must use the .npz suffix.')
    if not isinstance(shapes, VoidShapes):
        raise TypeError('shapes must be a VoidShapes instance.')
    if not isinstance(overwrite, (bool, np.bool_)):
        raise TypeError('overwrite must be boolean.')
    if path.exists() and not overwrite:
        raise FileExistsError(f'Output already exists: {path}. Set overwrite=True to replace it.')

    metadata_json = json.dumps(metadata, sort_keys=True, allow_nan=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f'.{path.name}.',
                                                  suffix='.tmp',
                                                  dir=path.parent)

    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, 'wb') as stream:
            np.savez_compressed(stream,
                                group_id=np.asarray(shapes.group_id),
                                n_members=np.asarray(shapes.n_members),
                                n_data=np.asarray(shapes.n_data),
                                n_random=np.asarray(shapes.n_random),
                                center=np.asarray(shapes.center),
                                lambda_values=np.asarray(shapes.lambda_values),
                                semi_axes=np.asarray(shapes.semi_axes),
                                r_eff=np.asarray(shapes.r_eff),
                                ellipticity=np.asarray(shapes.ellipticity),
                                r_eff_sigma=np.asarray(shapes.r_eff_sigma),
                                relative_r_eff_sigma=np.asarray(
                                    shapes.relative_r_eff_sigma),
                                ellipticity_sigma=np.asarray(shapes.ellipticity_sigma),
                                n_bootstrap_valid=np.asarray(shapes.n_bootstrap_valid),
                                bootstrap_valid_fraction=np.asarray(
                                    shapes.bootstrap_valid_fraction),
                                quality_reason=np.asarray(shapes.quality_reason),
                                valid_shape=np.asarray(shapes.valid_shape),
                                metadata=np.asarray(metadata_json))

        if path.exists() and not overwrite:
            raise FileExistsError(
                f'Output already exists: {path}. '
                'Set overwrite=True to replace it.')
        os.replace(temporary, path)

    finally:
        if temporary.exists():
            temporary.unlink()
    return path