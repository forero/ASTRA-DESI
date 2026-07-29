import hashlib, json, os, re
import tempfile
from dataclasses import dataclass, replace
from numbers import Integral, Real
from pathlib import Path

import fitsio
import healpy as hp
import numpy as np

from .astra import (GroupFinderResult, UNASSIGNED)


DEFAULT_HEALPIX_NSIDE = 128
DEFAULT_MIN_RANDOMS_PER_PIXEL = 3
DEFAULT_MIN_RANDOMS_PER_RADIAL_BIN = 3
DEFAULT_RADIAL_BIN_WIDTH = 10.0
DEFAULT_EDGE_CHUNK_SIZE = 250_000
DEFAULT_MASK_CACHE = Path('temp/group_finder/healpix_masks')
_CACHE_VERSION = 2


@dataclass(frozen=True)
class RandomHealpixMask:

    random_counts: np.ndarray
    nside: int
    nest: bool
    min_randoms_per_pixel: int
    n_random_iterations: int
    radial_bin_edges: np.ndarray
    radial_counts: np.ndarray
    min_randoms_per_radial_bin: int
    source: str
    metadata: dict

    def __post_init__(self):
        counts = np.asarray(self.random_counts)
        nside = _positive_integer(self.nside, 'nside')
        minimum = _positive_integer(self.min_randoms_per_pixel,
                                    'min_randoms_per_pixel')
        n_iterations = _positive_integer(self.n_random_iterations, 'n_random_iterations')
        radial_edges = np.asarray(self.radial_bin_edges, dtype=np.float64)
        radial_counts = np.asarray(self.radial_counts)
        radial_minimum = _positive_integer(self.min_randoms_per_radial_bin,
                                           'min_randoms_per_radial_bin')

        if counts.ndim != 1 or counts.size != 12 * nside * nside:
            raise ValueError('random_counts must have 12*nside^2 one-dimensional values.')
        if counts.dtype.kind not in 'iu':
            raise TypeError('random_counts must have integer dtype.')
        if np.any(counts < 0):
            raise ValueError('random_counts cannot be negative.')
        if (radial_edges.ndim != 1 or radial_edges.size < 2
                                   or not np.all(np.isfinite(radial_edges))
                                   or radial_edges[0] < 0.0
                                   or np.any(np.diff(radial_edges) <= 0.0)):
            raise ValueError('radial_bin_edges must be finite, non-negative, and strictly increasing.')
        if (radial_counts.ndim != 1 or len(radial_counts) != len(radial_edges) - 1):
            raise ValueError('radial_counts must have len(radial_bin_edges)-1 values.')
        if radial_counts.dtype.kind not in 'iu':
            raise TypeError('radial_counts must have integer dtype.')
        if np.any(radial_counts < 0):
            raise ValueError('radial_counts cannot be negative.')
        object.__setattr__(self, 'random_counts', counts.astype(np.int64, copy=False))

        object.__setattr__(self, 'nside', nside)
        object.__setattr__(self, 'nest', bool(self.nest))
        object.__setattr__(self, 'min_randoms_per_pixel', minimum)
        object.__setattr__(self, 'n_random_iterations', n_iterations)
        object.__setattr__(self, 'radial_bin_edges', radial_edges)
        object.__setattr__(self, 'radial_counts', radial_counts.astype(np.int64, copy=False))
        object.__setattr__(self, 'min_randoms_per_radial_bin', radial_minimum)

    @property
    def mean_random_counts(self):
        return self.random_counts / float(self.n_random_iterations)

    @property
    def valid_pixels(self):
        required_total = (self.min_randoms_per_pixel * self.n_random_iterations)
        return self.random_counts >= required_total

    @property
    def mean_radial_counts(self):
        return self.radial_counts / float(self.n_random_iterations)

    @property
    def valid_radial_bins(self):
        required_total = (self.min_randoms_per_radial_bin * self.n_random_iterations)
        return self.radial_counts >= required_total


@dataclass(frozen=True)
class EdgeMaskApplication:

    result: GroupFinderResult
    group_ids_before_mask: np.ndarray
    edge_group_removed: np.ndarray
    low_count_random_member: np.ndarray
    invalid_angular_member: np.ndarray
    invalid_radial_member: np.ndarray
    healpix_pixel: np.ndarray
    radial_bin: np.ndarray
    internal_group_edge: np.ndarray
    angular_edge_crossing: np.ndarray
    radial_edge_crossing: np.ndarray
    angular_member_group_ids: np.ndarray
    radial_member_group_ids: np.ndarray
    angular_edge_group_ids: np.ndarray
    radial_edge_group_ids: np.ndarray
    edge_group_ids: np.ndarray
    selection_pruned_member: np.ndarray
    invalid_member_pruned: np.ndarray
    disconnected_member: np.ndarray
    small_component_member: np.ndarray
    audited_group_ids: np.ndarray
    seed_point: np.ndarray
    group_size_before_pruning: np.ndarray
    seed_component_size: np.ndarray
    group_size_after_pruning: np.ndarray
    pruned_group_ids: np.ndarray
    seed_invalid_group_ids: np.ndarray
    undersized_component_group_ids: np.ndarray
    discarded_group_ids: np.ndarray


def _positive_integer(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError(f'{name} must be a positive integer.')
    value = int(value)
    if value < 1:
        raise ValueError(f'{name} must be a positive integer.')
    return value


def _positive_finite(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f'{name} must be a positive finite number.')
    value = float(value)
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f'{name} must be a positive finite number.')
    return value


def _decode_text(value):
    if isinstance(value, (bytes, np.bytes_)):
        return value.decode('utf-8').strip()
    return str(value).strip()


def _row_key(hdu, row):
    record = hdu.read(columns=['TRACERTYPE', 'RANDITER'], rows=[int(row)])[0]
    return _decode_text(record['TRACERTYPE']), int(record['RANDITER'])


def _lower_bound(hdu, target):
    lower = 0
    upper = int(hdu.get_nrows())
    while lower < upper:
        middle = (lower + upper) // 2
        if _row_key(hdu, middle) < target:
            lower = middle + 1
        else:
            upper = middle
    return lower


def _raw_identity(path):
    path = Path(path).resolve()
    stat = path.stat()
    return {'path': str(path),
            'size': int(stat.st_size),
            'mtime_ns': int(stat.st_mtime_ns)}


def _safe_label(value):
    return re.sub(r'[^A-Za-z0-9_.-]+', '_', str(value).strip()) or 'UNKNOWN'


def _cache_file(cache_path, raw_path, tracer, nside, nest, radial_bin_width):
    if cache_path is None:
        return None
    cache_path = Path(cache_path)

    if cache_path.suffix.lower() == '.npz':
        return cache_path

    ordering = 'nest' if nest else 'ring'
    radial_tag = format(float(radial_bin_width), '.12g').replace('.', 'p')
    raw_identity_tag = hashlib.sha256(str(Path(raw_path).resolve()).encode('utf-8')).hexdigest()[:12]
    return (cache_path / (f'{_safe_label(Path(raw_path).name)}.{raw_identity_tag}.'
                          f'{_safe_label(tracer)}.nside{int(nside)}.{ordering}.'
                          f'radial{radial_tag}.all-random-selection-counts.npz'))


def _expected_cache_metadata(raw_path, tracer, nside, nest, radial_bin_width):
    return {'cache_version': _CACHE_VERSION,
            'raw': _raw_identity(raw_path),
            'tracer': str(tracer),
            'nside': int(nside),
            'nest': bool(nest),
            'radial_bin_width': float(radial_bin_width),
            'radial_coordinate_units': 'raw Cartesian coordinate units',
            'random_iterations': 'all non-negative RANDITER values',
            'count_normalization': 'mean per distinct RANDITER'}


def _load_cache(path, expected, angular_minimum, radial_minimum) -> RandomHealpixMask | None:
    if path is None or not path.is_file():
        return None

    try:
        with np.load(path, allow_pickle=False) as payload:
            metadata = json.loads(str(payload['metadata'].item()))
            if any(metadata.get(key) != value for key, value in expected.items()):
                return None
            counts = np.asarray(payload['random_counts'], dtype=np.int64)
            radial_edges = np.asarray(payload['radial_bin_edges'], dtype=np.float64)
            radial_counts = np.asarray(payload['radial_counts'], dtype=np.int64)
        returned_metadata = _threshold_metadata(metadata={**metadata, 'kind': 'cache',
                                                          'cache_path': str(path)},
                                                random_counts=counts,
                                                radial_counts=radial_counts,
                                                n_random_iterations=metadata['n_random_iterations'],
                                                angular_minimum=angular_minimum,
                                                radial_minimum=radial_minimum)
        return RandomHealpixMask(random_counts=counts,
                                 nside=expected['nside'],
                                 nest=expected['nest'],
                                 min_randoms_per_pixel=angular_minimum,
                                 n_random_iterations=metadata['n_random_iterations'],
                                 radial_bin_edges=radial_edges,
                                 radial_counts=radial_counts,
                                 min_randoms_per_radial_bin=radial_minimum,
                                 source=str(path),
                                 metadata=returned_metadata)
    except (OSError, TypeError, ValueError, KeyError, OverflowError, json.JSONDecodeError):
        return None


def _write_cache(path, counts, radial_bin_edges, radial_counts, metadata):
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f'.{path.name}.', suffix='.tmp', dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, 'wb') as stream:
            np.savez_compressed(stream, random_counts=np.asarray(counts, dtype=np.int64),
                                radial_bin_edges=np.asarray(radial_bin_edges, dtype=np.float64),
                                radial_counts=np.asarray(radial_counts, dtype=np.int64),
                                metadata=np.asarray(json.dumps(metadata, sort_keys=True)))
        os.replace(temporary, path)

    finally:
        if temporary.exists():
            temporary.unlink()


def _threshold_metadata(metadata, random_counts, radial_counts, n_random_iterations,
                        angular_minimum, radial_minimum):
    n_iterations = _positive_integer(n_random_iterations, 'n_random_iterations')
    angular_minimum = _positive_integer(angular_minimum, 'angular_minimum')
    radial_minimum = _positive_integer(radial_minimum, 'radial_minimum')

    random_counts = np.asarray(random_counts, dtype=np.int64)
    radial_counts = np.asarray(radial_counts, dtype=np.int64)
    angular_required = angular_minimum * n_iterations
    radial_required = radial_minimum * n_iterations
    return {**metadata,
            'min_randoms_per_pixel_per_iteration': int(
                angular_minimum),
            'min_randoms_per_radial_bin_per_iteration': int(
                radial_minimum),
            'angular_required_accumulated_count': int(angular_required),
            'radial_required_accumulated_count': int(radial_required),
            'n_valid_pixels': int(np.count_nonzero(
                random_counts >= angular_required)),
            'n_valid_radial_bins': int(
                np.count_nonzero(radial_counts >= radial_required))}


def build_all_random_healpix_mask(raw_path, tracer, nside=DEFAULT_HEALPIX_NSIDE,
                                  min_randoms_per_pixel=DEFAULT_MIN_RANDOMS_PER_PIXEL,
                                  min_randoms_per_radial_bin=DEFAULT_MIN_RANDOMS_PER_RADIAL_BIN,
                                  radial_bin_width=DEFAULT_RADIAL_BIN_WIDTH,
                                  nest=False,
                                  cache_path=DEFAULT_MASK_CACHE,
                                  chunk_size=1_000_000,
                                  force=False) -> RandomHealpixMask:

    raw_path = Path(raw_path)
    tracer = str(tracer).strip()
    if not raw_path.is_file():
        raise FileNotFoundError(f'Raw FITS does not exist: {raw_path}.')
    if not tracer:
        raise ValueError('tracer cannot be empty.')
    nside = _positive_integer(nside, 'nside')

    minimum = _positive_integer(min_randoms_per_pixel, 'min_randoms_per_pixel')
    radial_minimum = _positive_integer(min_randoms_per_radial_bin, 'min_randoms_per_radial_bin')

    radial_bin_width = _positive_finite(radial_bin_width, 'radial_bin_width')

    chunk_size = _positive_integer(chunk_size, 'chunk_size')
    if not hp.isnsideok(nside):
        raise ValueError(f'Invalid HEALPix nside={nside}.')

    cache_file = _cache_file(cache_path, raw_path, tracer, nside, bool(nest), radial_bin_width)
    expected = _expected_cache_metadata(raw_path, tracer, nside, bool(nest), radial_bin_width)
    if not force:
        cached = _load_cache(cache_file,
                             expected,
                             angular_minimum=minimum,
                             radial_minimum=radial_minimum)
        if cached is not None:
            return cached

    counts = np.zeros(hp.nside2npix(nside), dtype=np.int64)
    radial_counts_by_index: dict[int, int] = {}
    random_iterations: set[int] = set()
    n_rows = 0
    n_valid_angular = 0
    n_valid_radial = 0
    n_valid_joint = 0
    with fitsio.FITS(str(raw_path)) as raw:
        hdu = raw[1]
        columns = {_decode_text(name).upper() for name in hdu.get_colnames()}
        required_columns = {'RA', 'DEC', 'XCART', 'YCART', 'ZCART', 'RANDITER'}
        missing = required_columns.difference(columns)

        if missing:
            raise KeyError(
                f'Raw FITS is missing mask columns: {", ".join(sorted(missing))}.')

        start = _lower_bound(hdu, (tracer, 0))
        stop = _lower_bound(hdu, (tracer, np.iinfo(np.int64).max))
        if start == stop:
            raise ValueError(f'No random rows found for TRACERTYPE={tracer!r}.')

        first_iteration = _row_key(hdu, start)[1]
        last_iteration = _row_key(hdu, stop - 1)[1]
        for chunk_start in range(start, stop, chunk_size):
            chunk_stop = min(chunk_start + chunk_size, stop)
            rows = np.arange(chunk_start, chunk_stop, dtype=np.int64)
            chunk = hdu.read(columns=['RA', 'DEC', 'XCART', 'YCART', 'ZCART', 'RANDITER'],
                             rows=rows)

            ra = np.asarray(chunk['RA'], dtype=np.float64)
            dec = np.asarray(chunk['DEC'], dtype=np.float64)
            x = np.asarray(chunk['XCART'], dtype=np.float64)
            y = np.asarray(chunk['YCART'], dtype=np.float64)
            z = np.asarray(chunk['ZCART'], dtype=np.float64)
            iterations = np.asarray(chunk['RANDITER'], dtype=np.int64)
            if np.any(iterations < 0):
                raise ValueError('The selected all-random row range contains a negative RANDITER.')
            random_iterations.update(int(value) for value in np.unique(iterations))

            angular_valid = (np.isfinite(ra)
                             & np.isfinite(dec)
                             & (dec >= -90.0)
                             & (dec <= 90.0))

            if np.any(angular_valid):
                pixels = hp.ang2pix(nside,
                                    np.radians(90.0 - dec[angular_valid]),
                                    np.radians(np.mod(ra[angular_valid], 360.0)),
                                    nest=bool(nest))
                counts += np.bincount(pixels, minlength=counts.size).astype(np.int64, copy=False)

            radius = np.hypot(np.hypot(x, y), z)
            radial_valid = np.isfinite(radius) & (radius > 0.0)
            if np.any(radial_valid):
                radial_indices = np.floor(radius[radial_valid] / radial_bin_width).astype(np.int64)
                unique_indices, unique_counts = np.unique(radial_indices, return_counts=True)

                for index, count in zip(unique_indices, unique_counts):
                    key = int(index)
                    radial_counts_by_index[key] = (radial_counts_by_index.get(key, 0) + int(count))

            n_rows += len(chunk)
            n_valid_angular += int(np.count_nonzero(angular_valid))
            n_valid_radial += int(np.count_nonzero(radial_valid))
            n_valid_joint += int(np.count_nonzero(
                angular_valid & radial_valid))

    if not random_iterations:
        raise ValueError(f'No non-negative random iterations found for {tracer!r}.')
    if not radial_counts_by_index:
        raise ValueError(f'No valid random Cartesian radii found for {tracer!r}.')

    first_radial_index = min(radial_counts_by_index)
    last_radial_index = max(radial_counts_by_index)
    n_radial_bins = last_radial_index - first_radial_index + 1
    if n_radial_bins > 10_000_000:
        raise ValueError('The random Cartesian radius range would require more than '
                         '10,000,000 radial bins; check the raw coordinates or increase '
                         'radial_bin_width.')

    radial_counts = np.zeros(n_radial_bins, dtype=np.int64)
    for index, count in radial_counts_by_index.items():
        radial_counts[index - first_radial_index] = count
    radial_bin_edges = (np.arange(first_radial_index, last_radial_index + 2, dtype=np.float64) * radial_bin_width)
    iteration_values = sorted(random_iterations)
    n_iterations = len(iteration_values)

    metadata = {**expected,
                'kind': 'all-random-angular-radial-counts',
                'row_range': {'start': int(start), 'stop': int(stop)},
                'first_randiter': int(first_iteration),
                'last_randiter': int(last_iteration),
                'random_iteration_values': iteration_values,
                'n_random_iterations': int(n_iterations),
                'n_random_rows': int(n_rows),

                'n_valid_random_rows': int(n_valid_angular),
                'n_valid_angular_random_rows': int(n_valid_angular),
                'n_valid_radial_random_rows': int(n_valid_radial),
                'n_valid_joint_random_rows': int(n_valid_joint),
                'n_nonempty_pixels': int(np.count_nonzero(counts)),
                'n_radial_bins': int(len(radial_counts)),
                'n_nonempty_radial_bins': int(np.count_nonzero(radial_counts)),
                'radial_min': float(radial_bin_edges[0]),
                'radial_max': float(radial_bin_edges[-1]),
                'ordering': 'NESTED' if nest else 'RING'}

    _write_cache(cache_file, counts, radial_bin_edges, radial_counts, metadata)
    returned_metadata = _threshold_metadata(metadata=metadata, random_counts=counts,
                                            radial_counts=radial_counts,
                                            n_random_iterations=n_iterations,
                                            angular_minimum=minimum, radial_minimum=radial_minimum)

    return RandomHealpixMask(random_counts=counts,
                             nside=nside, nest=bool(nest),
                             min_randoms_per_pixel=minimum,
                             n_random_iterations=n_iterations,
                             radial_bin_edges=radial_bin_edges,
                             radial_counts=radial_counts,
                             min_randoms_per_radial_bin=radial_minimum,
                             source=str(cache_file) if cache_file is not None else str(raw_path),
                             metadata=returned_metadata)


def cartesian_healpix_pixels(positions, nside, nest=False):

    positions = np.asarray(positions, dtype=np.float64)
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError('positions must have shape (n_points, 3).')
    nside = _positive_integer(nside, 'nside')
    radius = np.linalg.norm(positions, axis=1)

    valid = (np.all(np.isfinite(positions), axis=1)
             & np.isfinite(radius)
             & (radius > 0.0))

    pixels = np.full(len(positions), -1, dtype=np.int64)
    if np.any(valid):
        unit = positions[valid] / radius[valid, None]
        pixels[valid] = hp.vec2pix(nside,
                                   unit[:, 0],
                                   unit[:, 1],
                                   unit[:, 2],
                                   nest=bool(nest))
    return pixels, valid


def radec_healpix_pixels(ra, dec, nside, nest=False):

    ra = np.asarray(ra, dtype=np.float64)
    dec = np.asarray(dec, dtype=np.float64)
    if ra.ndim != 1 or dec.ndim != 1 or ra.shape != dec.shape:
        raise ValueError('ra and dec must be matching one-dimensional arrays.')

    nside = _positive_integer(nside, 'nside')
    valid = (np.isfinite(ra)
             & np.isfinite(dec)
             & (dec >= -90.0)
             & (dec <= 90.0))
    pixels = np.full(len(ra), -1, dtype=np.int64)

    if np.any(valid):
        pixels[valid] = hp.ang2pix(nside,
                                   np.radians(90.0 - dec[valid]),
                                   np.radians(np.mod(ra[valid], 360.0)),
                                   nest=bool(nest))
    return pixels, valid


def cartesian_radial_bins(positions, radial_bin_edges):

    positions = np.asarray(positions, dtype=np.float64)
    edges = np.asarray(radial_bin_edges, dtype=np.float64)
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError('positions must have shape (n_points, 3).')
    if (edges.ndim != 1 or len(edges) < 2 or not np.all(np.isfinite(edges)) or np.any(np.diff(edges) <= 0.0)):
        raise ValueError('radial_bin_edges must be finite and strictly increasing.')

    radius = np.linalg.norm(positions, axis=1)
    valid = (np.all(np.isfinite(positions), axis=1)
             & np.isfinite(radius)
             & (radius > 0.0)
             & (radius >= edges[0])
             & (radius <= edges[-1]))

    bins = np.full(len(positions), -1, dtype=np.int64)
    if np.any(valid):
        indices = np.searchsorted(edges, radius[valid], side='right') - 1
        indices[indices == len(edges) - 1] = len(edges) - 2
        bins[valid] = indices
    return bins, valid


def _group_ids_for_members(group_ids, selected):
    values = np.asarray(group_ids, dtype=np.int64)[selected]
    values = values[values >= 0]
    return np.unique(values).astype(np.int64, copy=False)


def _group_ids_for_edges(group_ids, edges, selected):
    if not np.any(selected):
        return np.empty(0, dtype=np.int64)
    values = np.asarray(group_ids, dtype=np.int64)[np.asarray(edges, dtype=np.int64)[selected, 0]]
    return np.unique(values[values >= 0]).astype(np.int64, copy=False)


def _angular_edge_crossings(positions, edges, internal, point_pixels, point_angular_valid,
                            mask: RandomHealpixMask, sample_step, chunk_size):

    positions = np.asarray(positions, dtype=np.float64)
    edges = np.asarray(edges, dtype=np.int64)
    crossings = np.zeros(len(edges), dtype=bool)
    internal_indices = np.flatnonzero(internal)
    if not len(internal_indices):
        return crossings

    valid_mask = mask.valid_pixels
    tiny = np.finfo(np.float64).tiny
    antipodal_tolerance = 1.0e-10
    for chunk_start in range(0, len(internal_indices), chunk_size):
        chunk_indices = internal_indices[chunk_start:chunk_start + chunk_size]

        chunk_edges = edges[chunk_indices]
        first = chunk_edges[:, 0]
        second = chunk_edges[:, 1]
        p0 = positions[first]
        p1 = positions[second]
        radius0 = np.linalg.norm(p0, axis=1)
        radius1 = np.linalg.norm(p1, axis=1)
        usable = (np.asarray(point_angular_valid, dtype=bool)[first]
                  & np.asarray(point_angular_valid, dtype=bool)[second]
                  & (radius0 > tiny)
                  & (radius1 > tiny))

        endpoint_ok = np.zeros(len(chunk_edges), dtype=bool)
        if np.any(usable):
            endpoint_ok[usable] = (valid_mask[point_pixels[first[usable]]]
                                   & valid_mask[point_pixels[second[usable]]])
        keep = usable & endpoint_ok

        u0 = np.zeros_like(p0)
        u1 = np.zeros_like(p1)
        if np.any(keep):
            u0[keep] = p0[keep] / radius0[keep, None]
            u1[keep] = p1[keep] / radius1[keep, None]
            active = np.flatnonzero(keep)
            dot = np.einsum('ij,ij->i', u0[active], u1[active])
            angles = np.arccos(np.clip(dot, -1.0, 1.0))
            unique_arc = angles < (np.pi - antipodal_tolerance)
            keep[active[~unique_arc]] = False
            active = active[unique_arc]
            angles = angles[unique_arc]

            if len(active):
                n_segments = np.maximum(1, np.ceil(angles / sample_step).astype(np.int64))
                sin_angles = np.sin(angles)
                max_segments = int(np.max(n_segments))
                for sample_index in range(1, max_segments):
                    sample_active = ((n_segments > sample_index) & keep[active])
                    if not np.any(sample_active):
                        continue

                    local = np.flatnonzero(sample_active)
                    rows = active[local]
                    fractions = sample_index / n_segments[local]
                    local_angles = angles[local]
                    local_sin = sin_angles[local]
                    sample = np.empty((len(local), 3), dtype=np.float64)

                    regular = np.abs(local_sin) >= 1.0e-12
                    if np.any(regular):
                        w0 = np.sin((1.0 - fractions[regular]) * local_angles[regular]) / local_sin[regular]
                        w1 = np.sin(fractions[regular] * local_angles[regular] ) / local_sin[regular]
                        sample[regular] = (w0[:, None] * u0[rows[regular]] + w1[:, None] * u1[rows[regular]])

                    nearly_zero = ~regular
                    if np.any(nearly_zero):
                        sample[nearly_zero] = ((1.0 - fractions[nearly_zero, None])
                                                * u0[rows[nearly_zero]]
                                                + fractions[nearly_zero, None]
                                                * u1[rows[nearly_zero]])

                    sample_radius = np.linalg.norm(sample, axis=1)
                    sample_valid = sample_radius > tiny
                    sample_ok = np.zeros(len(local), dtype=bool)
                    if np.any(sample_valid):
                        sample[sample_valid] /= sample_radius[sample_valid, None]

                        sample_pixels = hp.vec2pix(mask.nside,
                                                   sample[sample_valid, 0],
                                                   sample[sample_valid, 1],
                                                   sample[sample_valid, 2],
                                                   nest=mask.nest)

                        sample_ok[sample_valid] = valid_mask[sample_pixels]
                    keep[rows] &= sample_ok
        crossings[chunk_indices] = ~keep
    return crossings


def _radial_edge_crossings(positions, edges, internal, mask: RandomHealpixMask, chunk_size):

    positions = np.asarray(positions, dtype=np.float64)
    edges = np.asarray(edges, dtype=np.int64)
    crossings = np.zeros(len(edges), dtype=bool)
    internal_indices = np.flatnonzero(internal)
    if not len(internal_indices):
        return crossings

    radial_edges = mask.radial_bin_edges
    radial_valid = mask.valid_radial_bins
    invalid_prefix = np.concatenate((np.zeros(1, dtype=np.int64),
                                     np.cumsum(~radial_valid, dtype=np.int64)))

    for chunk_start in range(0, len(internal_indices), chunk_size):
        chunk_indices = internal_indices[chunk_start:chunk_start + chunk_size]
        chunk_edges = edges[chunk_indices]
        p0 = positions[chunk_edges[:, 0]]
        p1 = positions[chunk_edges[:, 1]]
        finite = (np.all(np.isfinite(p0), axis=1)
                  & np.all(np.isfinite(p1), axis=1))

        delta = p1 - p0
        delta_sq = np.einsum('ij,ij->i', delta, delta)
        t_min = np.zeros(len(chunk_edges), dtype=np.float64)
        nonzero = delta_sq > 0.0
        t_min[nonzero] = -np.einsum('ij,ij->i', p0[nonzero],
                                    delta[nonzero]) / delta_sq[nonzero]

        np.clip(t_min, 0.0, 1.0, out=t_min)
        closest = p0 + t_min[:, None] * delta
        radius_min = np.linalg.norm(closest, axis=1)
        radius_max = np.maximum(
            np.linalg.norm(p0, axis=1),
            np.linalg.norm(p1, axis=1))

        in_support = (finite
                      & np.isfinite(radius_max)
                      & np.isfinite(radius_min)
                      & (radius_min > 0.0)
                      & (radius_min >= radial_edges[0])
                      & (radius_max <= radial_edges[-1]))

        keep = np.zeros(len(chunk_edges), dtype=bool)
        if np.any(in_support):
            rows = np.flatnonzero(in_support)
            first_bin = np.searchsorted(radial_edges,
                                        radius_min[rows],
                                        side='right') - 1
            last_bin = np.searchsorted(radial_edges,
                                       radius_max[rows],
                                       side='right') - 1
            first_bin[first_bin == len(radial_edges) - 1] = (len(radial_edges) - 2)
            last_bin[last_bin == len(radial_edges) - 1] = (len(radial_edges) - 2)
            invalid_count = (invalid_prefix[last_bin + 1] - invalid_prefix[first_bin])
            keep[rows] = invalid_count == 0

        crossings[chunk_indices] = ~keep
    return crossings


@dataclass(frozen=True)
class _SeedTopologyPruning:

    group_ids: np.ndarray
    selection_pruned_member: np.ndarray
    invalid_member_pruned: np.ndarray
    disconnected_member: np.ndarray
    small_component_member: np.ndarray
    audited_group_ids: np.ndarray
    seed_point: np.ndarray
    group_size_before_pruning: np.ndarray
    seed_component_size: np.ndarray
    group_size_after_pruning: np.ndarray
    pruned_group_ids: np.ndarray
    seed_invalid_group_ids: np.ndarray
    undersized_component_group_ids: np.ndarray
    discarded_group_ids: np.ndarray


def _prune_to_seed_connected_components(group_ids, edges, internal_group_edge,
                                        invalid_member, invalid_edge, processing_order,
                                        min_members) -> _SeedTopologyPruning:

    group_ids = np.asarray(group_ids, dtype=np.int64)
    edges = np.asarray(edges, dtype=np.int64)
    internal_group_edge = np.asarray(internal_group_edge, dtype=bool)
    invalid_member = np.asarray(invalid_member, dtype=bool)
    invalid_edge = np.asarray(invalid_edge, dtype=bool)
    processing_order = np.asarray(processing_order, dtype=np.int64)
    n_points = len(group_ids)
    retained = group_ids >= 0
    if invalid_member.shape != (n_points,):
        raise ValueError('invalid_member must match group_ids.')
    if (internal_group_edge.shape != (len(edges),)
            or invalid_edge.shape != (len(edges),)):
        raise ValueError('edge masks must match edges.')
    if processing_order.ndim != 1:
        raise ValueError('processing_order must be one-dimensional.')
    if len(processing_order) and (np.min(processing_order) < 0
                                  or np.max(processing_order) >= n_points):
        raise IndexError('processing_order contains an invalid point index.')

    audited_group_ids, group_size_before = np.unique(group_ids[retained],
                                                     return_counts=True)

    audited_group_ids = audited_group_ids.astype(np.int64, copy=False)
    group_size_before = group_size_before.astype(np.int64, copy=False)
    n_groups = len(audited_group_ids)

    if n_groups == 0:
        empty_points = np.zeros(n_points, dtype=bool)
        empty_groups = np.empty(0, dtype=np.int64)
        return _SeedTopologyPruning(group_ids=group_ids.copy(),
                                    selection_pruned_member=empty_points.copy(),
                                    invalid_member_pruned=empty_points.copy(),
                                    disconnected_member=empty_points.copy(),
                                    small_component_member=empty_points.copy(),
                                    audited_group_ids=empty_groups.copy(),
                                    seed_point=empty_groups.copy(),
                                    group_size_before_pruning=empty_groups.copy(),
                                    seed_component_size=empty_groups.copy(),
                                    group_size_after_pruning=empty_groups.copy(),
                                    pruned_group_ids=empty_groups.copy(),
                                    seed_invalid_group_ids=empty_groups.copy(),
                                    undersized_component_group_ids=empty_groups.copy(),
                                    discarded_group_ids=empty_groups.copy())

    seed_point = np.full(n_groups, -1, dtype=np.int64)
    group_row = {int(group_id): row for row, group_id in enumerate(audited_group_ids)}
    for point in processing_order:
        group_id = int(group_ids[point])
        if group_id < 0:
            continue
        row = group_row[group_id]
        if seed_point[row] < 0:
            seed_point[row] = point
    if np.any(seed_point < 0):
        missing = audited_group_ids[seed_point < 0].tolist()
        raise ValueError('Every retained group must have a member in processing_order; '
                         f'missing group IDs: {missing}.')

    valid_member = retained & ~invalid_member
    parent = np.arange(n_points, dtype=np.int64)
    rank = np.zeros(n_points, dtype=np.int8)

    def find(point):
        point = int(point)
        while parent[point] != point:
            parent[point] = parent[parent[point]]
            point = int(parent[point])
        return point

    def union(first, second):
        root_first = find(first)
        root_second = find(second)
        if root_first == root_second:
            return
        if rank[root_first] < rank[root_second]:
            root_first, root_second = root_second, root_first
        parent[root_second] = root_first
        if rank[root_first] == rank[root_second]:
            rank[root_first] += 1

    if len(edges):
        valid_internal_edge = (internal_group_edge
                               & ~invalid_edge
                               & valid_member[edges[:, 0]]
                               & valid_member[edges[:, 1]])

        for first, second in edges[valid_internal_edge]:
            union(first, second)

    valid_points = np.flatnonzero(valid_member)
    roots = np.full(n_points, -1, dtype=np.int64)
    for point in valid_points:
        roots[point] = find(point)

    row_for_point = np.full(n_points, -1, dtype=np.int64)
    if np.any(retained):
        row_for_point[retained] = np.searchsorted(audited_group_ids,
                                                  group_ids[retained])

    seed_invalid = invalid_member[seed_point]
    seed_root = np.full(n_groups, -1, dtype=np.int64)
    valid_seed_rows = np.flatnonzero(~seed_invalid)
    seed_root[valid_seed_rows] = roots[seed_point[valid_seed_rows]]

    in_seed_component = np.zeros(n_points, dtype=bool)
    if len(valid_points):
        point_rows = row_for_point[valid_points]
        in_seed_component[valid_points] = (~seed_invalid[point_rows]
                                           & (roots[valid_points] == seed_root[point_rows]))

    seed_component_size = np.bincount(row_for_point[in_seed_component],
                                      minlength=n_groups).astype(np.int64, copy=False)

    undersized = ~seed_invalid & (seed_component_size < min_members)
    accepted_group = ~seed_invalid & ~undersized
    kept_member = (in_seed_component & accepted_group[row_for_point.clip(min=0)])

    invalid_member_pruned = retained & invalid_member
    disconnected_member = (valid_member & ~in_seed_component & ~seed_invalid[row_for_point.clip(min=0)])
    small_component_member = (in_seed_component & undersized[row_for_point.clip(min=0)])

    selection_pruned_member = retained & ~kept_member
    filtered_group_ids = group_ids.copy()
    filtered_group_ids[selection_pruned_member] = UNASSIGNED

    group_size_after = seed_component_size.copy()
    group_size_after[~accepted_group] = 0
    pruned = group_size_after < group_size_before
    discarded = group_size_after == 0
    return _SeedTopologyPruning(group_ids=filtered_group_ids,
                                selection_pruned_member=selection_pruned_member,
                                invalid_member_pruned=invalid_member_pruned,
                                disconnected_member=disconnected_member,
                                small_component_member=small_component_member,
                                audited_group_ids=audited_group_ids,
                                seed_point=seed_point,
                                group_size_before_pruning=group_size_before,
                                seed_component_size=seed_component_size,
                                group_size_after_pruning=group_size_after,
                                pruned_group_ids=audited_group_ids[pruned],
                                seed_invalid_group_ids=audited_group_ids[seed_invalid],
                                undersized_component_group_ids=audited_group_ids[undersized],
                                discarded_group_ids=audited_group_ids[discarded])


def apply_random_healpix_edge_mask(result: GroupFinderResult, mask: RandomHealpixMask,
                                   random_ra=None, random_dec=None, angular_sample_step=None,
                                   edge_chunk_size=DEFAULT_EDGE_CHUNK_SIZE, min_members=4) -> EdgeMaskApplication:

    if not isinstance(result, GroupFinderResult):
        raise TypeError('result must be a GroupFinderResult.')
    if not isinstance(mask, RandomHealpixMask):
        raise TypeError('mask must be a RandomHealpixMask.')

    edge_chunk_size = _positive_integer(edge_chunk_size, 'edge_chunk_size')

    min_members = _positive_integer(min_members, 'min_members')
    if angular_sample_step is None:
        angular_sample_step = (0.5 * np.sqrt(4.0 * np.pi / len(mask.random_counts)))
    else:
        angular_sample_step = _positive_finite(angular_sample_step, 'angular_sample_step')

    group_ids_before = np.asarray(result.grouping.group_ids, dtype=np.int64).copy()
    pixels, valid_pixels = cartesian_healpix_pixels(result.graph.positions,
                                                    nside=mask.nside,
                                                    nest=mask.nest)

    if (random_ra is None) != (random_dec is None):
        raise ValueError('random_ra and random_dec must be supplied together.')

    if random_ra is not None:
        expected_randoms = int(result.graph.n_random)
        angular_pixels, angular_valid = radec_healpix_pixels(random_ra,
                                                             random_dec,
                                                             nside=mask.nside,
                                                             nest=mask.nest)
        if len(angular_pixels) != expected_randoms:
            raise ValueError('random_ra/random_dec must match result.graph.n_random.')

        random_start = int(result.graph.n_data)
        pixels[random_start:] = angular_pixels
        valid_pixels[random_start:] = angular_valid

    retained_members = group_ids_before >= 0
    invalid_angular_member = retained_members & ~valid_pixels
    angular_usable_members = retained_members & valid_pixels

    if np.any(angular_usable_members):
        member_indices = np.flatnonzero(angular_usable_members)
        invalid_angular_member[member_indices] = (~mask.valid_pixels[pixels[member_indices]])

    radial_bins, valid_radii = cartesian_radial_bins(result.graph.positions,
                                                     mask.radial_bin_edges)

    invalid_radial_member = retained_members & ~valid_radii
    radial_usable_members = retained_members & valid_radii
    if np.any(radial_usable_members):
        member_indices = np.flatnonzero(radial_usable_members)
        invalid_radial_member[member_indices] = (~mask.valid_radial_bins[radial_bins[member_indices]])

    random_members = (~np.asarray(result.graph.is_data, dtype=bool)
                      & retained_members)
    low_count_random_member = (invalid_angular_member & random_members)

    edges = np.asarray(result.graph.edges, dtype=np.int64)
    if edges.ndim != 2 or edges.shape[1] != 2:
        raise ValueError('result.graph.edges must have shape (n_edges, 2).')
    if len(edges):
        if np.min(edges) < 0 or np.max(edges) >= len(group_ids_before):
            raise IndexError('result.graph.edges contains an index outside the graph.')
        internal_group_edge = ((group_ids_before[edges[:, 0]] >= 0) &
                               (group_ids_before[edges[:, 0]] == group_ids_before[edges[:, 1]]))
    else:
        internal_group_edge = np.zeros(0, dtype=bool)

    angular_edge_crossing = _angular_edge_crossings(positions=result.graph.positions,
                                                    edges=edges,
                                                    internal=internal_group_edge,
                                                    point_pixels=pixels,
                                                    point_angular_valid=valid_pixels,
                                                    mask=mask,
                                                    sample_step=angular_sample_step,
                                                    chunk_size=edge_chunk_size)
    radial_edge_crossing = _radial_edge_crossings(positions=result.graph.positions,
                                                  edges=edges,
                                                  internal=internal_group_edge,
                                                  mask=mask,
                                                  chunk_size=edge_chunk_size)

    angular_member_group_ids = _group_ids_for_members(group_ids_before,
                                                      invalid_angular_member)
    radial_member_group_ids = _group_ids_for_members(group_ids_before,
                                                     invalid_radial_member)
    angular_edge_group_ids = _group_ids_for_edges(group_ids_before, edges,
                                                  angular_edge_crossing)
    radial_edge_group_ids = _group_ids_for_edges(group_ids_before,
                                                 edges,
                                                 radial_edge_crossing)
    cause_group_ids = (angular_member_group_ids,
                       radial_member_group_ids,
                       angular_edge_group_ids,
                       radial_edge_group_ids)
    nonempty_causes = [values for values in cause_group_ids if len(values)]
    edge_group_ids = (np.unique(np.concatenate(nonempty_causes)).astype(np.int64, copy=False,)
                      if nonempty_causes else np.empty(0, dtype=np.int64))
    pruning = _prune_to_seed_connected_components(group_ids=group_ids_before,
                                                  edges=edges,
                                                  internal_group_edge=internal_group_edge,
                                                  invalid_member=invalid_angular_member | invalid_radial_member,
                                                  invalid_edge=angular_edge_crossing | radial_edge_crossing,
                                                  processing_order=result.grouping.processing_order,
                                                  min_members=min_members)
    filtered_group_ids = pruning.group_ids

    edge_group_removed = pruning.selection_pruned_member.copy()
    retained = filtered_group_ids >= 0
    if np.any(retained):
        unique, sizes = np.unique(filtered_group_ids[retained],
                                  return_counts=True)
        group_sizes = {int(group_id): int(size)
                       for group_id, size in zip(unique, sizes)}
    else:
        group_sizes = {}

    filtered_grouping = replace(result.grouping,
                                group_ids=filtered_group_ids,
                                group_sizes=group_sizes,
                                retained=retained)
    filtered_result = GroupFinderResult(graph=result.graph,
                                        contrast=result.contrast,
                                        grouping=filtered_grouping)

    return EdgeMaskApplication(result=filtered_result,
                               group_ids_before_mask=group_ids_before,
                               edge_group_removed=edge_group_removed,
                               low_count_random_member=low_count_random_member,
                               invalid_angular_member=invalid_angular_member,
                               invalid_radial_member=invalid_radial_member,
                               healpix_pixel=pixels,
                               radial_bin=radial_bins,
                               internal_group_edge=internal_group_edge,
                               angular_edge_crossing=angular_edge_crossing,
                               radial_edge_crossing=radial_edge_crossing,
                               angular_member_group_ids=angular_member_group_ids,
                               radial_member_group_ids=radial_member_group_ids,
                               angular_edge_group_ids=angular_edge_group_ids,
                               radial_edge_group_ids=radial_edge_group_ids,
                               edge_group_ids=edge_group_ids,
                               selection_pruned_member=pruning.selection_pruned_member,
                               invalid_member_pruned=pruning.invalid_member_pruned,
                               disconnected_member=pruning.disconnected_member,
                               small_component_member=pruning.small_component_member,
                               audited_group_ids=pruning.audited_group_ids,
                               seed_point=pruning.seed_point,
                               group_size_before_pruning=pruning.group_size_before_pruning,
                               seed_component_size=pruning.seed_component_size,
                               group_size_after_pruning=pruning.group_size_after_pruning,
                               pruned_group_ids=pruning.pruned_group_ids,
                               seed_invalid_group_ids=pruning.seed_invalid_group_ids,
                               undersized_component_group_ids=(
                                   pruning.undersized_component_group_ids),
                               discarded_group_ids=pruning.discarded_group_ids)
