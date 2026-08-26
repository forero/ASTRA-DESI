from dataclasses import dataclass
from collections import deque
import gc
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import time
import traceback

import fitsio
import numpy as np
from scipy.integrate import cumulative_trapezoid

from group_finder.astra import (run_group_finder, warmup_accelerators as warmup_astra)
from group_finder.consensus import (consensus_output_paths, consensus_outputs_complete,
                                    run_consensus)
from group_finder.make_cat import (build_random_membership_catalog, build_void_catalogs,
                                   compute_void_shapes,
                                   void_catalog_has_required_columns,
                                   write_membership_catalog, write_void_catalog)
from group_finder.read_data import (TRACER_CODES, TRACER_DISPLAY, normalize_tracer,
                                    normalize_zone)
from group_finder.watershed import (apply_random_healpix_edge_mask,
                                    build_random_healpix_mask,
                                    warmup_accelerators as warmup_watershed)

SPEED_OF_LIGHT_KM_S = 299_792.458
DEFAULT_H = 0.6736
DEFAULT_OMEGA_M = 0.315
DEFAULT_RA_MIN = 83.0
DEFAULT_RA_MAX = 302.0
DEFAULT_R_THRESHOLD = -0.25
DEFAULT_MIN_MEMBERS = 4
DEFAULT_NSIDE = 128
DEFAULT_MIN_RANDOMS_PER_PIXEL = 3
DEFAULT_MIN_RANDOMS_PER_RADIAL_BIN = 3
DEFAULT_RADIAL_BIN_WIDTH = 5.0
DEFAULT_MEMORY_FRACTION = 0.60
DEFAULT_MEMORY_BYTES_PER_POINT = 1500
DEFAULT_WORKER_RETRIES = 2

_REQUIRED_SKY_COLUMNS = ('TARGETID', 'RA', 'DEC', 'Z')

_SHARED_CASE = None
_SHARED_CONFIG = None


@dataclass(frozen=True)
class SkySample:
    targetid: np.ndarray
    ra: np.ndarray
    dec: np.ndarray
    redshift: np.ndarray

    def __post_init__(self):
        n_rows = len(self.targetid)
        for name in ('ra', 'dec', 'redshift'):
            if np.asarray(getattr(self, name)).shape != (n_rows,):
                raise ValueError(f'{name} must contain one value per row.')

    def __len__(self):
        return len(self.targetid)


@dataclass(frozen=True)
class CartesianCase:
    object_positions: np.ndarray
    random_positions: np.ndarray
    random_targetid: np.ndarray
    random_ra: np.ndarray
    random_dec: np.ndarray
    random_redshift: np.ndarray
    random_source_index: np.ndarray


@dataclass(frozen=True)
class IterationConfig:
    case_root: str
    dataset: str
    tracer: str
    zone: str
    random_source: str
    random_sources: tuple
    random_source_indices: tuple
    random_pool_signature: str
    mask_cache: str
    base_seed: int
    random_factor: float
    r_threshold: float
    min_members: int
    healpix_nside: int
    min_randoms_per_pixel: int
    min_randoms_per_radial_bin: int
    radial_bin_width: float
    mask_chunk_size: int
    edge_chunk_size: int
    h: float
    omega_m: float
    include_membership: bool
    overwrite: bool


def parse_iteration_tokens(tokens, default=(0, 100)):
    """Parse values such as ``0-99``, ``0:100`` or ``0 3 8``."""
    if not tokens:
        return tuple(range(*default))
    values = []
    for token in tokens:
        text = str(token).strip()
        if not text:
            continue
        if ':' in text:
            parts = text.split(':')
            if len(parts) not in (2, 3):
                raise ValueError(f'Invalid iteration range {text!r}.')
            start = int(parts[0]) if parts[0] else 0
            stop = int(parts[1])
            step = int(parts[2]) if len(parts) == 3 else 1
            if step <= 0:
                raise ValueError('Iteration range steps must be positive.')
            values.extend(range(start, stop, step))
        elif '-' in text[1:]:
            start_text, stop_text = text.split('-', 1)
            start, stop = int(start_text), int(stop_text)
            if stop < start:
                raise ValueError(f'Invalid iteration range {text!r}.')
            values.extend(range(start, stop + 1))
        else:
            values.append(int(text))
    result = tuple(dict.fromkeys(values))
    if not result:
        raise ValueError('No realizations were selected.')
    if min(result) < 0 or max(result) >= 1000:
        raise ValueError('Realization indices must lie in [0, 1000).')
    return result


def normalize_catalog_tracer(value):
    """Normalize both DESI on-disk names and display aliases."""
    text = str(value).strip().upper()
    aliases = {'BGS': 'BGS',
               'BGS_ANY': 'BGS',
               'BGS_BRIGHT': 'BGS',
               'LRG': 'LRG',
               'ELG': 'ELG',
               'ELGNOTQSO': 'ELG',
               'ELG_NOTQSO': 'ELG',
               'ELG_LOPNOTQSO': 'ELG',
               'QSO': 'QSO',
               'QSOS': 'QSO'}
    if text not in aliases:
        raise ValueError('tracer must be BGS, LRG, ELG, or QSO.')
    return aliases[text]


def _cap_mask(ra, cap, ra_min, ra_max):
    cap = normalize_zone(cap)
    north = (ra >= float(ra_min)) & (ra <= float(ra_max))
    return north if cap == 'NGC' else ~north


def _read_columns(path, columns, chunk_size):
    """Yield selected FITS columns in bounded row chunks."""
    with fitsio.FITS(str(path)) as catalog:
        hdu = catalog[1]
        available = set(hdu.get_colnames())
        missing = set(columns).difference(available)
        if missing:
            raise KeyError(f'{path} is missing columns: ' + ', '.join(sorted(missing)))
        n_rows = int(hdu.get_nrows())
        selected_columns = hdu[list(columns)]
        for start in range(0, n_rows, int(chunk_size)):
            yield selected_columns[start:min(n_rows, start + int(chunk_size))]


def read_sky_sample(path,
                    cap=None,
                    ra_min=DEFAULT_RA_MIN,
                    ra_max=DEFAULT_RA_MAX,
                    z_min=None,
                    z_max=None,
                    magnitude_limit=None,
                    chunk_size=2_000_000):
    """Read, validate, cut, and cap-split a sky catalogue without full-table I/O."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    with fitsio.FITS(str(path)) as catalog:
        available = set(catalog[1].get_colnames())
    columns = list(_REQUIRED_SKY_COLUMNS)
    if magnitude_limit is not None:
        if 'R_MAG_ABS' not in available:
            raise KeyError(f'{path} has no R_MAG_ABS column required by '
                           '--bgs-mr-limit.')
        columns.append('R_MAG_ABS')

    ids = []
    ras = []
    decs = []
    redshifts = []
    for chunk in _read_columns(path, columns, chunk_size):
        ra = np.asarray(chunk['RA'], dtype=np.float64)
        dec = np.asarray(chunk['DEC'], dtype=np.float64)
        redshift = np.asarray(chunk['Z'], dtype=np.float64)
        selected = (np.isfinite(ra) & np.isfinite(dec)
                    & np.isfinite(redshift) & (redshift >= 0.0)
                    & (dec >= -90.0) & (dec <= 90.0))
        if cap is not None:
            selected &= _cap_mask(ra, cap, ra_min, ra_max)
        if z_min is not None:
            selected &= redshift >= float(z_min)
        if z_max is not None:
            selected &= redshift <= float(z_max)
        if magnitude_limit is not None:
            magnitude = np.asarray(chunk['R_MAG_ABS'], dtype=np.float64)
            selected &= (np.isfinite(magnitude) & (magnitude <= float(magnitude_limit)))
        if not np.any(selected):
            continue
        ids.append(np.asarray(chunk['TARGETID'][selected], dtype=np.int64))
        ras.append(ra[selected])
        decs.append(dec[selected])
        redshifts.append(redshift[selected])

    if not ids:
        label = f' for cap {cap}' if cap is not None else ''
        raise ValueError(f'No rows remain in {path}{label}.')
    return SkySample(targetid=np.concatenate(ids),
                     ra=np.concatenate(ras),
                     dec=np.concatenate(decs),
                     redshift=np.concatenate(redshifts))


def concatenate_sky_samples(samples):
    """Concatenate already-filtered catalogues without structured tables."""
    samples = tuple(samples)
    if not samples:
        raise ValueError('At least one sky sample is required.')
    return SkySample(targetid=np.concatenate([sample.targetid for sample in samples]),
                     ra=np.concatenate([sample.ra for sample in samples]),
                     dec=np.concatenate([sample.dec for sample in samples]),
                     redshift=np.concatenate([sample.redshift for sample in samples]))


def random_pool_signature(paths, source_indices=None):
    """Hash the ordered source identities used to build a random pool."""
    paths = tuple(Path(path).expanduser().resolve() for path in paths)
    if not paths:
        raise ValueError('A random pool needs at least one source path.')
    if source_indices is None:
        source_indices = tuple(range(len(paths)))
    source_indices = tuple(int(value) for value in source_indices)
    if len(source_indices) != len(paths):
        raise ValueError('source_indices must match the random source paths.')
    if len(set(source_indices)) != len(source_indices):
        raise ValueError('Random source indices must be unique.')
    identities = []
    for source_index, path in zip(source_indices, paths):
        if not path.is_file():
            raise FileNotFoundError(path)
        stat = path.stat()
        identities.append({'index': source_index,
                           'path': str(path),
                           'size': int(stat.st_size),
                           'mtime_ns': int(stat.st_mtime_ns)})
    payload = json.dumps(identities, sort_keys=True, separators=(',', ':'))
    return hashlib.sha256(payload.encode('utf-8')).hexdigest()


def comoving_distance_mpc_h(redshift, omega_m, grid_size=131_073):
    """Fast flat-LambdaCDM comoving distance in Mpc/h.

    In Mpc/h the Hubble parameter cancels: ``chi*h = c/100 integral dz/E``.
    A dense cumulative integral followed by interpolation avoids millions of
    independent quadratures for the large DESI random pools.
    """
    redshift = np.asarray(redshift, dtype=np.float64)
    omega_m = float(omega_m)
    if (redshift.ndim != 1 or not np.all(np.isfinite(redshift))
            or np.any(redshift < 0.0)):
        raise ValueError('redshift must be a finite non-negative vector.')
    if not np.isfinite(omega_m) or not 0.0 < omega_m < 1.0:
        raise ValueError('omega_m must lie within (0, 1).')
    if not len(redshift):
        return np.empty(0, dtype=np.float64)
    observed_maximum = float(np.max(redshift))
    if observed_maximum == 0.0:
        return np.zeros(len(redshift), dtype=np.float64)
    # A fixed DESI-wide grid makes the transformation identical for the data
    # and random arrays.  It is also cheap enough to rebuild for each of the
    # three cosmologies and avoids a per-row cosmology call.
    maximum = max(6.0, observed_maximum)
    grid = np.linspace(0.0, maximum, int(grid_size), dtype=np.float64)
    expansion = np.sqrt(omega_m * (1.0 + grid)**3 + (1.0 - omega_m))
    distance = ((SPEED_OF_LIGHT_KM_S / 100.0) *
                cumulative_trapezoid(1.0 / expansion, grid, initial=0.0))
    return np.interp(redshift, grid, distance)


def cartesian_positions(sample, omega_m):
    radius = comoving_distance_mpc_h(sample.redshift, omega_m)
    ra = np.radians(sample.ra)
    dec = np.radians(sample.dec)
    cos_dec = np.cos(dec)
    return np.ascontiguousarray(np.column_stack(
        (radius * cos_dec * np.cos(ra), radius * cos_dec * np.sin(ra),
         radius * np.sin(dec))),
                                dtype=np.float64)


def make_cartesian_case(objects, randoms, omega_m, random_source_index=None):
    """Convert one cap's invariant data and random pool once per cosmology."""
    if random_source_index is None:
        random_source_index = np.zeros(len(randoms), dtype=np.int16)
    random_source_index = np.asarray(random_source_index)
    if random_source_index.shape != (len(randoms),):
        raise ValueError('random_source_index must match the random pool.')
    if not np.issubdtype(random_source_index.dtype, np.integer):
        raise TypeError('random_source_index must contain integers.')
    return CartesianCase(object_positions=cartesian_positions(objects, omega_m),
                         random_positions=cartesian_positions(randoms, omega_m),
                         random_targetid=np.asarray(randoms.targetid, dtype=np.int64),
                         random_ra=np.asarray(randoms.ra, dtype=np.float64),
                         random_dec=np.asarray(randoms.dec, dtype=np.float64),
                         random_redshift=np.asarray(randoms.redshift, dtype=np.float64),
                         random_source_index=np.asarray(random_source_index,
                                                        dtype=np.int16))


def iteration_paths(case_root, tracer, zone, iteration, include_membership=False):
    label = TRACER_DISPLAY[normalize_tracer(tracer)].lower()
    zone = normalize_zone(zone).lower()
    root = Path(case_root) / label / zone / f'iter{int(iteration):02d}'
    paths = {'all': root / 'all.fits',
             'clean': root / 'clean.fits',
             'summary': root / 'summary.json'}
    if include_membership:
        paths['membership'] = root / 'membership.fits'
    return paths


def _write_json(path, payload, overwrite=False):
    path = Path(path)
    if path.exists() and not overwrite:
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + '\n',
            encoding='utf-8')
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return path


def _iteration_seed(config, iteration):
    tracer = normalize_tracer(config.tracer)
    zone_code = 1 if normalize_zone(config.zone) == 'NGC' else 2
    sequence = np.random.SeedSequence([int(config.base_seed),
                                       int(iteration),
                                       int(TRACER_CODES[tracer]),
                                       zone_code])
    # Keep the seed representable by ordinary FITS integer header cards.
    return int(sequence.generate_state(1, dtype=np.uint32)[0])


def _sample_random_indices(config, iteration, n_available, n_data):
    n_target = int(round(float(config.random_factor) * int(n_data)))
    if n_target < 1:
        raise ValueError('Each realization needs at least one random point.')
    if n_target > n_available:
        raise ValueError(f'Not enough randoms for {config.tracer} {config.zone}: '
                         f'requested {n_target:,}, available {n_available:,}.')
    seed = _iteration_seed(config, iteration)
    if n_target == n_available:
        return np.arange(n_available, dtype=np.int64), seed
    indices = np.random.default_rng(seed).choice(n_available,
                                                 size=n_target,
                                                 replace=False,
                                                 shuffle=False)
    return np.asarray(indices, dtype=np.int64), seed


def _random_records(case, indices, iteration):
    dtype = [('TARGETID', 'i8'), ('RA', 'f8'), ('DEC', 'f8'), ('Z', 'f8'),
             ('XCART', 'f8'), ('YCART', 'f8'), ('ZCART', 'f8'), ('RANDITER', 'i4'),
             ('RANSRCIDX', 'i2')]
    records = np.empty(len(indices), dtype=dtype)
    records['TARGETID'] = case.random_targetid[indices]
    records['RA'] = case.random_ra[indices]
    records['DEC'] = case.random_dec[indices]
    records['Z'] = case.random_redshift[indices]
    records['XCART'], records['YCART'], records['ZCART'] = (
        case.random_positions[indices].T)
    records['RANDITER'] = int(iteration)
    records['RANSRCIDX'] = case.random_source_index[indices]
    return records


def _mask_cache_tag(config):
    payload = {'dataset': config.dataset,
               'tracer': config.tracer,
               'zone': config.zone,
               'base_seed': config.base_seed,
               'random_factor': config.random_factor,
               'omega_m': config.omega_m,
               'random_pool_signature': config.random_pool_signature}
    digest = hashlib.sha256(json.dumps(payload,
                                       sort_keys=True).encode()).hexdigest()[:12]
    return f'{config.tracer}-{config.zone}-{digest}'


def _run_iteration(iteration):
    case = _SHARED_CASE
    config = _SHARED_CONFIG
    if case is None or config is None:
        raise RuntimeError('Catalogue worker was not initialized.')
    iteration = int(iteration)
    started = time.time()
    paths = iteration_paths(config.case_root, config.tracer, config.zone, iteration,
                            config.include_membership)

    indices, random_seed = _sample_random_indices(config, iteration,
                                                  len(case.random_positions),
                                                  len(case.object_positions))
    selected_source_indices, selected_source_counts = np.unique(
        case.random_source_index[indices], return_counts=True)
    selected_counts = {
        str(int(source_index)): int(count)
        for source_index, count in zip(selected_source_indices, selected_source_counts)}
    randoms = _random_records(case, indices, iteration)
    random_positions = np.ascontiguousarray(case.random_positions[indices],
                                            dtype=np.float64)
    del indices

    selection = build_random_healpix_mask(
        raw_path=config.random_source,
        tracer=_mask_cache_tag(config),
        iteration=iteration,
        nside=config.healpix_nside,
        min_randoms_per_pixel=config.min_randoms_per_pixel,
        min_randoms_per_radial_bin=config.min_randoms_per_radial_bin,
        radial_bin_width=config.radial_bin_width,
        cache_path=config.mask_cache,
        chunk_size=config.mask_chunk_size,
        random_records=randoms)

    unmasked = run_group_finder(object_positions=case.object_positions,
                                random_positions=random_positions,
                                r_threshold=config.r_threshold,
                                min_members=config.min_members)
    del random_positions
    masked = apply_random_healpix_edge_mask(unmasked,
                                            selection,
                                            random_ra=randoms['RA'],
                                            random_dec=randoms['DEC'],
                                            edge_chunk_size=config.edge_chunk_size,
                                            min_members=config.min_members,
                                            retain_edge_diagnostics=False)
    result = masked.result

    positions = result.graph.positions
    is_data = result.graph.is_data
    grouping = result.grouping
    n_data = int(result.graph.n_data)
    edge_group_ids = np.asarray(masked.edge_group_ids).copy()
    n_edges = int(len(result.graph.edges))
    n_groups_before_mask = int(len(unmasked.grouping.group_sizes))
    n_groups_pruned = int(len(masked.pruned_group_ids))
    n_groups_discarded = int(len(masked.discarded_group_ids))
    if config.include_membership:
        group_ids_before_random = np.asarray(
            masked.group_ids_before_mask[n_data:]).copy()
        selection_pruned_random = np.asarray(
            masked.selection_pruned_member[n_data:]).copy()
    else:
        group_ids_before_random = None
        selection_pruned_random = None
        del randoms
    del result, masked, unmasked, selection
    gc.collect()

    shapes = compute_void_shapes(positions=positions,
                                 is_data=is_data,
                                 group_ids=grouping.group_ids,
                                 coordinate_scale=1.0)
    catalogs = build_void_catalogs(shapes,
                                   border_group_ids=edge_group_ids,
                                   tracer=config.tracer,
                                   zone=config.zone,
                                   iteration=iteration,
                                   h=config.h)
    metadata = {
        'DATASET':
        config.dataset[:68],
        'OMEGA_M':
        float(config.omega_m),
        'RANSRC': (Path(config.random_source).name[:58] if len(config.random_sources)
                   == 1 else f'pool:{len(config.random_sources)} sources'),
        'NRANSRC':
        len(config.random_sources),
        'RANPOOL':
        config.random_pool_signature[:16],
        'RSEED':
        int(random_seed)}
    catalogs.all_voids.meta.update(metadata)
    catalogs.clean_voids.meta.update(metadata)
    n_defined_shapes = int(
        np.count_nonzero(np.isfinite(shapes.r_eff) & np.isfinite(shapes.ellipticity)))
    del shapes, positions, is_data
    gc.collect()

    write_void_catalog(paths['all'], catalogs.all_voids, overwrite=config.overwrite)
    write_void_catalog(paths['clean'], catalogs.clean_voids, overwrite=config.overwrite)
    membership_rows = None
    if config.include_membership:
        membership = build_random_membership_catalog(
            randoms=randoms,
            group_ids=grouping.group_ids[n_data:],
            group_ids_before_mask=group_ids_before_random,
            r_values=grouping.r_values[n_data:],
            threshold_selected=grouping.threshold_selected[n_data:],
            selection_pruned_member=selection_pruned_random,
            border_group_ids=edge_group_ids,
            tracer=config.tracer,
            zone=config.zone,
            iteration=iteration)
        write_membership_catalog(paths['membership'],
                                 membership,
                                 overwrite=config.overwrite)
        membership_rows = len(membership)
        del membership, randoms

    summary = {'algorithm': 'ASTRA literal lowest-index Delaunay watershed',
               'dataset': config.dataset,
               'tracer': config.tracer,
               'zone': config.zone,
               'iteration': iteration,
               'random_seed': random_seed,
               'random_source': config.random_source,
               'random_sources': list(config.random_sources),
               'random_source_indices': list(config.random_source_indices),
               'random_pool_signature': config.random_pool_signature,
               'random_source_counts_selected': selected_counts,
               'random_factor': config.random_factor,
               'base_seed': config.base_seed,
               'h': config.h,
               'omega_m': config.omega_m,
               'r_threshold': config.r_threshold,
               'min_members': config.min_members,
               'n_data': len(case.object_positions),
               'n_random': len(grouping.group_ids) - n_data,
               'n_edges': n_edges,
               'n_groups_before_mask': n_groups_before_mask,
               'n_groups_pruned': n_groups_pruned,
               'n_groups_discarded': n_groups_discarded,
               'n_groups_after_mask': len(grouping.group_sizes),
               'n_defined_shapes': n_defined_shapes,
               'n_catalog_all': len(catalogs.all_voids),
               'n_catalog_border': int(np.count_nonzero(catalogs.all_voids['BORDER'])),
               'n_catalog_clean': len(catalogs.clean_voids),
               'membership_rows': membership_rows,
               'outputs': {name: str(path)
                           for name, path in paths.items() if name != 'summary'},
               'elapsed_seconds': float(time.time() - started),}
    _write_json(paths['summary'], summary, overwrite=config.overwrite)
    return summary


def _init_worker(case, config):
    global _SHARED_CASE, _SHARED_CONFIG
    _SHARED_CASE = case
    _SHARED_CONFIG = config


def _worker_entry(iteration):
    try:
        return _run_iteration(iteration)
    except Exception as exc:
        raise RuntimeError(f'Iteration {iteration} failed: {exc}\n'
                           f'{traceback.format_exc()}') from exc


def _process_worker_entry(iteration, connection):
    """Run one realization in a disposable process and report atomically."""
    try:
        connection.send(('ok', _worker_entry(iteration)))
    except BaseException as exc:
        try:
            connection.send(('error', f'{exc}\n{traceback.format_exc()}'))
        except (BrokenPipeError, EOFError, OSError):
            pass
    finally:
        connection.close()


def _available_memory_bytes():
    candidates = []
    try:
        import psutil
        candidates.append(int(psutil.virtual_memory().available))
    except (ImportError, OSError, ValueError):
        pass
    try:
        maximum = Path('/sys/fs/cgroup/memory.max').read_text().strip()
        current = int(Path('/sys/fs/cgroup/memory.current').read_text().strip())
        if maximum != 'max':
            candidates.append(max(0, int(maximum) - current))
    except (OSError, ValueError):
        pass
    return min((value for value in candidates if value > 0), default=1 << 60)


def _allocated_cpus():
    value = os.environ.get('SLURM_CPUS_PER_TASK', '').strip()
    if value:
        try:
            return max(1, int(value))
        except ValueError:
            pass
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except (AttributeError, OSError):
        return max(1, os.cpu_count() or 1)


def resolve_workers(requested,
                    n_tasks,
                    n_points,
                    memory_fraction=DEFAULT_MEMORY_FRACTION,
                    memory_bytes_per_point=DEFAULT_MEMORY_BYTES_PER_POINT):
    cpu_limit = min(int(n_tasks), _allocated_cpus())
    estimate = 2 * 1024**3 + int(n_points) * int(memory_bytes_per_point)
    usable = int(_available_memory_bytes() * float(memory_fraction))
    memory_limit = max(1, usable // max(1, estimate))
    text = str(requested).strip().lower()
    if text == 'auto':
        return max(1, min(cpu_limit, memory_limit)), estimate
    workers = int(text)
    if workers < 1:
        raise ValueError('workers must be "auto" or a positive integer.')
    return min(workers, cpu_limit), estimate


def iteration_complete(config, iteration):
    paths = iteration_paths(config.case_root, config.tracer, config.zone, iteration,
                            config.include_membership)
    if not all(path.is_file() and path.stat().st_size > 0 for path in paths.values()):
        return False
    if not all(void_catalog_has_required_columns(paths[name])
               for name in ('all', 'clean')):
        return False
    try:
        summary = json.loads(paths['summary'].read_text(encoding='utf-8'))
    except (OSError, ValueError, json.JSONDecodeError):
        return False

    stored_signature = summary.get('random_pool_signature')
    if stored_signature is not None:
        return stored_signature == config.random_pool_signature

    # Backwards compatibility for completed mock runs written before pool
    # signatures were introduced.  Multi-file DR2 configurations must never
    # reuse those old single-source products.
    if len(config.random_sources) != 1:
        return False
    stored_source = summary.get('random_source')
    if not stored_source:
        return False
    return (Path(stored_source).expanduser().resolve() == Path(
        config.random_sources[0]).expanduser().resolve())


def _stop_processes(active):
    for process, _, connection in active.values():
        connection.close()
        if process.is_alive():
            process.terminate()
    for process, _, _ in active.values():
        process.join()


def _run_parallel_realizations(case, config, pending, worker_count, quiet=False):
    """Run disposable fork workers without losing tasks after an OOM kill.

    ``multiprocessing.Pool`` does not requeue a task if the operating system
    kills its worker.  On large Delaunay graphs that leaves ``imap`` waiting
    forever.  This small scheduler owns one process per iteration, so it knows
    exactly which task died, retries it, and lowers concurrency before
    launching more work.
    """
    context = mp.get_context('fork')
    queued = deque(int(value) for value in pending)
    active = {}
    retries = {int(value): 0 for value in pending}
    summaries = []
    concurrency = int(worker_count)
    _init_worker(case, config)

    def launch(iteration):
        receive, send = context.Pipe(duplex=False)
        process = context.Process(target=_process_worker_entry,
                                  args=(iteration, send),
                                  name=f'astra-iter{iteration:02d}')
        process.start()
        send.close()
        active[process.pid] = (process, iteration, receive)

    try:
        while queued or active:
            while queued and len(active) < concurrency:
                launch(queued.popleft())

            failed = []
            reported_error = None
            for pid, (process, iteration, connection) in list(active.items()):
                message = None
                try:
                    if connection.poll():
                        message = connection.recv()
                except (EOFError, OSError):
                    message = None

                if message is not None:
                    status, payload = message
                    process.join(timeout=5.0)
                    if process.is_alive():
                        process.terminate()
                        process.join()
                    connection.close()
                    del active[pid]
                    if status == 'ok':
                        summaries.append(payload)
                        if not quiet:
                            print(f'[{config.dataset} {config.tracer} '
                                  f'{config.zone}] iter{iteration:02d} complete '
                                  f'({len(summaries)}/{len(pending)})',
                                  flush=True)
                    else:
                        reported_error = RuntimeError(payload)
                    continue

                if not process.is_alive():
                    process.join()
                    exitcode = process.exitcode
                    connection.close()
                    del active[pid]
                    failed.append((iteration, exitcode))

            if reported_error is not None:
                raise reported_error

            if failed:
                old_concurrency = concurrency
                concurrency = max(1, int(np.floor(concurrency * 0.60)))
                if concurrency == old_concurrency and concurrency > 1:
                    concurrency -= 1
                labels = []
                for iteration, exitcode in failed:
                    retries[iteration] += 1
                    labels.append(f'iter{iteration:02d}(exit={exitcode})')
                    if retries[iteration] > DEFAULT_WORKER_RETRIES:
                        raise RuntimeError(f'Iteration {iteration} worker '
                                           f'died {retries[iteration]} times '
                                           f'(last exit code {exitcode}). '
                                           'Completed outputs are reusable with '
                                           '--resume.')
                    queued.appendleft(iteration)
                if not quiet:
                    print('[parallel] worker failure detected: ' + ', '.join(labels) +
                          f'; reducing workers {old_concurrency} -> '
                          f'{concurrency} and retrying',
                          flush=True)

            if active and not failed:
                time.sleep(0.05)
    except BaseException:
        _stop_processes(active)
        raise
    finally:
        _init_worker(None, None)
    return summaries


def build_case_consensus(case_root,
                         tracer,
                         zone,
                         iterations,
                         resume=False,
                         overwrite=False,
                         keep_all=False,
                         vol_frac=0.5,
                         v_cut=0.5,
                         query_workers='auto',
                         quiet=False):
    # The independent DR2/mock BGS files contain the BGS_ANY selection even
    # though their on-disk basename and realization directory are simply
    # ``BGS``/``bgs``.  Preserve that scientific selection in consensus
    # filenames, FITS headers, and JSON metadata.
    consensus_tracer = ('BGS_ANY'
                        if normalize_catalog_tracer(tracer) == 'BGS' else tracer)
    output_root = Path(case_root) / 'consensus'
    paths = consensus_output_paths(output_root,
                                   consensus_tracer,
                                   zone,
                                   len(iterations),
                                   keep_all=keep_all)
    complete = consensus_outputs_complete(paths)
    if complete and resume and not overwrite:
        if not quiet:
            print(f'[resume] consensus {consensus_tracer} {zone} complete', flush=True)
        return paths
    partial = any(path.exists() for path in paths.values())
    if partial and not resume and not overwrite:
        raise FileExistsError('Consensus outputs already exist: '
                              + ', '.join(str(path)
                                          for path in paths.values()
                                          if path.exists())
                              + '. Use --resume or --overwrite.')
    workers = (min(8, _allocated_cpus())
               if str(query_workers).strip().lower() == 'auto' else int(query_workers))
    if workers < 1:
        raise ValueError('consensus workers must be "auto" or a positive integer.')
    _, written = run_consensus(input_root=case_root,
                               output_root=output_root,
                               tracer=consensus_tracer,
                               zone=zone,
                               iterations=iterations,
                               layout='iteration',
                               vol_frac=vol_frac,
                               v_cut=v_cut,
                               keep_all=keep_all,
                               query_workers=workers,
                               overwrite=(overwrite or (resume and partial)),
                               verbose=not quiet)
    return written


def run_realizations(case,
                     config,
                     iterations,
                     workers='auto',
                     resume=False,
                     memory_fraction=DEFAULT_MEMORY_FRACTION,
                     memory_bytes_per_point=DEFAULT_MEMORY_BYTES_PER_POINT,
                     consensus=True,
                     consensus_only=False,
                     consensus_keep_all=False,
                     consensus_vol_frac=0.5,
                     consensus_v_cut=0.5,
                     consensus_workers='auto',
                     quiet=False):
    """Run one dataset/tracer/cap/cosmology case and optionally aggregate it."""
    iterations = tuple(int(value) for value in iterations)
    if consensus_only:
        return build_case_consensus(config.case_root,
                                    config.tracer,
                                    config.zone,
                                    iterations,
                                    resume=resume,
                                    overwrite=config.overwrite,
                                    keep_all=consensus_keep_all,
                                    vol_frac=consensus_vol_frac,
                                    v_cut=consensus_v_cut,
                                    query_workers=consensus_workers,
                                    quiet=quiet)

    n_target = int(round(config.random_factor * len(case.object_positions)))
    if n_target < 1:
        raise ValueError('Each realization needs at least one random point.')
    if n_target > len(case.random_positions):
        raise ValueError(f'Not enough randoms for {config.tracer} '
                         f'{config.zone}: requested {n_target:,}, available '
                         f'{len(case.random_positions):,}.')

    manifest_path = (Path(config.case_root) /
                     TRACER_DISPLAY[normalize_tracer(config.tracer)].lower() /
                     normalize_zone(config.zone).lower() / 'run_summary.json')

    pending = []
    summaries = []
    for iteration in iterations:
        if resume and iteration_complete(config, iteration):
            if not quiet:
                print(f'[resume] {config.dataset} {config.tracer} '
                      f'{config.zone} iter{iteration:02d} complete',
                      flush=True)
            summary_path = iteration_paths(config.case_root, config.tracer, config.zone,
                                           iteration,
                                           config.include_membership)['summary']
            try:
                summaries.append(json.loads(summary_path.read_text(encoding='utf-8')))
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                raise ValueError(f'Cannot reuse invalid iteration summary '
                                 f'{summary_path}: {exc}') from exc
            continue
        pending.append(iteration)

    if not resume and not config.overwrite:
        existing = []
        for iteration in pending:
            existing.extend(path for path in iteration_paths(
                config.case_root, config.tracer, config.zone, iteration,
                config.include_membership).values() if path.exists())
        if existing:
            preview = ', '.join(str(path) for path in existing[:5])
            suffix = '' if len(existing) <= 5 else f' (+{len(existing) - 5} more)'
            raise FileExistsError(f'Realization outputs already exist: '
                                  f'{preview}{suffix}. '
                                  'Use --resume or --overwrite.')
        if manifest_path.exists():
            raise FileExistsError(f'Run manifest already exists: {manifest_path}. '
                                  'Use --resume or --overwrite.')

    if consensus and not resume and not config.overwrite:
        consensus_paths = consensus_output_paths(Path(config.case_root) / 'consensus',
                                                 config.tracer,
                                                 config.zone,
                                                 len(iterations),
                                                 keep_all=consensus_keep_all)
        existing = [path for path in consensus_paths.values() if path.exists()]
        if existing:
            raise FileExistsError('Consensus outputs already exist: ' +
                                  ', '.join(str(path) for path in existing) +
                                  '. Use --resume or --overwrite.')

    if pending:
        worker_count, estimate = resolve_workers(
            workers,
            len(pending),
            len(case.object_positions) + n_target,
            memory_fraction=memory_fraction,
            memory_bytes_per_point=memory_bytes_per_point)
        if not quiet:
            print(f'[{config.dataset} {config.tracer} {config.zone}] '
                  f'{len(case.object_positions):,} data, '
                  f'{len(case.random_positions):,} random pool, '
                  f'{n_target:,}/iteration; workers={worker_count}; '
                  f'estimated peak/worker={estimate / 1024**3:.1f} GiB',
                  flush=True)
        # Preflight has established ownership of every pending path.  Workers
        # may therefore replace their own partial products after a retry.
        worker_config = IterationConfig(**{**config.__dict__, 'overwrite': True})
        warmup_astra()
        warmup_watershed()
        if worker_count == 1:
            _init_worker(case, worker_config)
            try:
                for iteration in pending:
                    summary = _worker_entry(iteration)
                    summaries.append(summary)
                    if not quiet:
                        print(f'[{config.dataset} {config.tracer} {config.zone}] '
                              f'iter{iteration:02d} complete',
                              flush=True)
            finally:
                _init_worker(None, None)
        else:
            summaries.extend(_run_parallel_realizations(case,
                                                        worker_config,
                                                        pending,
                                                        worker_count,
                                                        quiet=quiet))

    manifest = {
        'dataset': config.dataset,
        'tracer': config.tracer,
        'zone': config.zone,
        'h': config.h,
        'omega_m': config.omega_m,
        'random_sources': list(config.random_sources),
        'random_source_indices': list(config.random_source_indices),
        'random_pool_signature': config.random_pool_signature,
        'iterations_requested': list(iterations),
        'iterations_completed': sorted(int(item['iteration']) for item in summaries),
        'cases': {f'iter{item["iteration"]:02d}': item
                  for item in summaries},}
    _write_json(manifest_path, manifest, overwrite=(config.overwrite or resume))

    if consensus:
        return build_case_consensus(config.case_root,
                                    config.tracer,
                                    config.zone,
                                    iterations,
                                    resume=resume,
                                    overwrite=bool(config.overwrite or pending),
                                    keep_all=consensus_keep_all,
                                    vol_frac=consensus_vol_frac,
                                    v_cut=consensus_v_cut,
                                    query_workers=consensus_workers,
                                    quiet=quiet)
    return summaries


def validate_common_options(
        iterations,
        random_factor,
        r_threshold,
        min_members,
        h,
        omega_m,
        memory_fraction,
        memory_bytes_per_point,
        healpix_nside=DEFAULT_NSIDE,
        min_randoms_per_pixel=DEFAULT_MIN_RANDOMS_PER_PIXEL,
        min_randoms_per_radial_bin=DEFAULT_MIN_RANDOMS_PER_RADIAL_BIN,
        radial_bin_width=DEFAULT_RADIAL_BIN_WIDTH,
        mask_chunk_size=1_000_000,
        edge_chunk_size=250_000,
        input_chunk_size=2_000_000,
        consensus_vol_frac=0.5,
        consensus_v_cut=0.5,
        consensus_workers='auto',
        ra_min=DEFAULT_RA_MIN,
        ra_max=DEFAULT_RA_MAX,
        z_min=None,
        z_max=None,
        workers='auto',
        seed=0):
    if not iterations:
        raise ValueError('At least one iteration is required.')
    if not np.isfinite(random_factor) or random_factor <= 0.0:
        raise ValueError('--random-factor must be positive.')
    if not np.isfinite(r_threshold) or not -1.0 <= r_threshold <= 1.0:
        raise ValueError('--r-threshold must lie within [-1, 1].')
    if int(min_members) < 1:
        raise ValueError('--min-members must be positive.')
    if not np.isfinite(h) or h <= 0.0:
        raise ValueError('--h must be positive.')
    if not np.isfinite(omega_m) or not 0.0 < omega_m < 1.0:
        raise ValueError('omega_m must lie within (0, 1).')
    if not 0.0 < float(memory_fraction) <= 1.0:
        raise ValueError('--memory-fraction must lie within (0, 1].')
    if int(memory_bytes_per_point) < 1:
        raise ValueError('--memory-bytes-per-point must be positive.')
    if isinstance(seed, (bool, np.bool_)) or int(seed) < 0:
        raise ValueError('--seed must be a non-negative integer.')
    if str(workers).strip().lower() != 'auto':
        try:
            if int(workers) < 1:
                raise ValueError
        except ValueError as exc:
            raise ValueError('--workers must be "auto" or a positive integer.') from exc
    nside = int(healpix_nside)
    if nside < 1 or nside & (nside - 1):
        raise ValueError('--healpix-nside must be a positive power of two.')
    for value, name in ((min_randoms_per_pixel, '--min-randoms-per-pixel'),
                        (min_randoms_per_radial_bin, '--min-randoms-per-radial-bin'),
                        (mask_chunk_size, '--mask-chunk-size'),
                        (edge_chunk_size, '--edge-chunk-size'), (input_chunk_size,
                                                                 '--input-chunk-size')):
        if int(value) < 1:
            raise ValueError(f'{name} must be positive.')
    if not np.isfinite(radial_bin_width) or radial_bin_width <= 0.0:
        raise ValueError('--radial-bin-width must be positive.')
    if (not np.isfinite(consensus_vol_frac)
            or not 0.5 <= float(consensus_vol_frac) <= 1.0):
        raise ValueError('--consensus-vol-frac must lie within [0.5, 1].')
    if (not np.isfinite(consensus_v_cut) or not 0.0 <= float(consensus_v_cut) <= 1.0):
        raise ValueError('--consensus-v-cut must lie within [0, 1].')
    if str(consensus_workers).strip().lower() != 'auto':
        try:
            if int(consensus_workers) < 1:
                raise ValueError
        except ValueError as exc:
            raise ValueError('--consensus-workers must be "auto" or a positive '
                             'integer.') from exc
    if (not np.isfinite(ra_min) or not np.isfinite(ra_max)
            or not 0.0 <= float(ra_min) <= 360.0 or not 0.0 <= float(ra_max) <= 360.0
            or float(ra_min) >= float(ra_max)):
        raise ValueError('--ra-min/--ra-max must satisfy 0 <= min < max <= 360.')
    if z_min is not None and (not np.isfinite(z_min) or z_min < 0.0):
        raise ValueError('--z-min must be finite and non-negative.')
    if z_max is not None and (not np.isfinite(z_max) or z_max < 0.0):
        raise ValueError('--z-max must be finite and non-negative.')
    if z_min is not None and z_max is not None and z_min > z_max:
        raise ValueError('--z-min must not exceed --z-max.')


__all__ = ['CartesianCase',
           'DEFAULT_H',
           'DEFAULT_MEMORY_BYTES_PER_POINT',
           'DEFAULT_MEMORY_FRACTION',
           'DEFAULT_MIN_MEMBERS',
           'DEFAULT_MIN_RANDOMS_PER_PIXEL',
           'DEFAULT_MIN_RANDOMS_PER_RADIAL_BIN',
           'DEFAULT_NSIDE',
           'DEFAULT_OMEGA_M',
           'DEFAULT_RADIAL_BIN_WIDTH',
           'DEFAULT_RA_MAX',
           'DEFAULT_RA_MIN',
           'DEFAULT_R_THRESHOLD',
           'IterationConfig',
           'SkySample',
           'build_case_consensus',
           'cartesian_positions',
           'comoving_distance_mpc_h',
           'concatenate_sky_samples',
           'iteration_paths',
           'make_cartesian_case',
           'normalize_catalog_tracer',
           'parse_iteration_tokens',
           'random_pool_signature',
           'read_sky_sample',
           'run_realizations',
           'validate_common_options']
