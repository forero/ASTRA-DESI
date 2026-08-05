from dataclasses import dataclass
import json
import os
from numbers import Real
from pathlib import Path
import tempfile

import numpy as np


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
                      ellipticity=empty_float.copy())


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
    effective_radius = np.sqrt(5.0) * np.exp(
        np.sum(np.log(eigenvalues)) / 6.0)
    moment_ratio = ((eigenvalues[2] + eigenvalues[1]) /
                    (eigenvalues[1] + eigenvalues[0]))
    moment_ratio = float(np.clip(moment_ratio, 0.0, 1.0))
    ellipticity = 1.0 - moment_ratio ** 0.25
    return (group_center, eigenvalues, axes,
            float(effective_radius), float(ellipticity))


def compute_void_shapes(positions, is_data, group_ids,
                        coordinate_scale=1.0) -> VoidShapes:
    """Measure every post-mask group without statistical quality cuts."""

    xyz = _positions_array(positions)
    data_labels = _boolean_labels(is_data, len(xyz))
    groups = _group_labels(group_ids, len(xyz))
    scale = _positive_scale(coordinate_scale)

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
        center[row] = np.mean(random_positions, axis=0, dtype=np.float64)
        measurements = _moment_measurements(random_positions)
        if measurements is None:
            continue

        (_, eigenvalues, axes, effective_radius,
         shape_ellipticity) = measurements
        lambda_values[row] = eigenvalues
        semi_axes[row] = axes
        r_eff[row] = effective_radius
        ellipticity[row] = shape_ellipticity

    return VoidShapes(group_id=retained_group_ids,
                      n_members=n_members,
                      n_data=n_data,
                      n_random=n_random,
                      center=center,
                      lambda_values=lambda_values,
                      semi_axes=semi_axes,
                      r_eff=r_eff,
                      ellipticity=ellipticity)


def write_void_shapes_npz(path, shapes: VoidShapes, metadata, overwrite=False):
    path = Path(path)
    if path.suffix.lower() != '.npz':
        raise ValueError('path must use the .npz suffix.')
    if not isinstance(shapes, VoidShapes):
        raise TypeError('shapes must be a VoidShapes instance.')
    if not isinstance(overwrite, (bool, np.bool_)):
        raise TypeError('overwrite must be boolean.')
    if path.exists() and not overwrite:
        raise FileExistsError(
            f'Output already exists: {path}. Set overwrite=True to replace it.')

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
