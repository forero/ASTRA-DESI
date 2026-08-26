from dataclasses import dataclass
import datetime
import json
import os
from pathlib import Path
import time

import fitsio
import numpy as np
from scipy.spatial import cKDTree

from .make_cat import (EIGENVALUE_COLUMNS, EIGENVECTOR_COLUMNS, VOID_SHAPE_COLUMNS)
from .read_data import (TRACER_DISPLAY, TRACER_OUTPUT_LABELS, normalize_tracer,
                        normalize_zone)

DEFAULT_CONSENSUS_VOL_FRAC = 0.5
DEFAULT_CONSENSUS_V_CUT = 0.5

CONSENSUS_DTYPE = np.dtype([('VOID_ID', 'i8'),
                            ('SRC_ITER', 'i4'),
                            ('X', 'f8'),
                            ('Y', 'f8'),
                            ('Z', 'f8'),
                            ('R_EFF', 'f8'),
                            ('ELLIP', 'f8')]
                           + [(name, 'f8') for name in VOID_SHAPE_COLUMNS]
                           + [('V', 'i4'),
                              ('FRAC_V', 'f8'),
                              ('N_ABSORBED', 'i4')])

_INPUT_COLUMNS = (('VOID_ID', 'XCART', 'YCART', 'ZCART', 'R_EFF', 'ELLIP') +
                  VOID_SHAPE_COLUMNS + ('BORDER',))


@dataclass(frozen=True)
class PooledVoids:
    """Compact aligned arrays for all usable voids in a set of runs."""

    centers: np.ndarray
    r_eff: np.ndarray
    ellipticity: np.ndarray
    eigenvalues: np.ndarray
    eigenvectors: np.ndarray
    source_iteration: np.ndarray
    void_id: np.ndarray
    iterations: tuple[int, ...]
    input_paths: tuple[Path, ...]
    n_read: int
    n_border: int
    n_undefined_shape: int


@dataclass(frozen=True)
class ConsensusResult:
    """Consensus rows and scalar diagnostics used by FITS/JSON writers."""

    catalog: np.ndarray
    n_pooled: int
    n_groups: int
    n_representatives: int
    n_after_pruning: int
    n_after_support_cut: int
    elapsed_seconds: float


def _validate_iterations(iterations):
    values = []
    for value in iterations:
        if isinstance(value,
                      (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
            raise TypeError('Consensus iterations must be integers.')
        value = int(value)
        if value < 0 or value >= 1000:
            raise ValueError('Consensus iterations must lie in [0, 1000).')
        values.append(value)
    result = tuple(dict.fromkeys(values))
    if not result:
        raise ValueError('At least one consensus iteration is required.')
    if len(result) != len(values):
        raise ValueError('Consensus iterations must not contain duplicates.')
    return result


def _validate_consensus_thresholds(vol_frac, v_cut):
    vol_frac = float(vol_frac)
    v_cut = float(v_cut)
    if not np.isfinite(vol_frac) or not 0.5 <= vol_frac <= 1.0:
        raise ValueError('vol_frac must lie within [0.5, 1].')
    if not np.isfinite(v_cut) or not 0.0 <= v_cut <= 1.0:
        raise ValueError('v_cut must lie within [0, 1].')
    return vol_frac, v_cut


def iteration_catalog_path(input_root, tracer, zone, iteration, layout='iteration'):
    """Return the per-run ``all.fits`` path for either pipeline layout."""
    tracer = normalize_tracer(tracer)
    zone = normalize_zone(zone)
    iteration = _validate_iterations((iteration,))[0]
    label = TRACER_OUTPUT_LABELS[tracer]
    # Independent-catalog runners historically write every BGS realization
    # under ``bgs``.  Their input sample is BGS_ANY, whereas the modular
    # pipeline keeps BGS_ANY and BGS_BRIGHT in explicit separate directories.
    # Prefer the explicit label, then accept the historical path.
    legacy_labels = ('BGS',) if tracer in {'BGS_ANY', 'BGS_BRIGHT'} else ()
    labels = tuple(dict.fromkeys((label, TRACER_DISPLAY[tracer]) + legacy_labels))
    root = Path(input_root)
    if layout == 'iteration':
        candidates = tuple(root / candidate.lower() / zone.lower() /
                           f'iter{iteration:02d}' / 'all.fits' for candidate in labels)
    elif layout == 'legacy':
        candidates = tuple(root / 'catalogs' / candidate / zone /
                           f'{candidate}_{zone}_iter{iteration:03d}_all.fits'
                           for candidate in labels)
    else:
        raise ValueError(f'Unknown output layout {layout!r}.')
    return next((path for path in candidates if path.is_file()), candidates[0])


def consensus_output_paths(output_root, tracer, zone, n_runs, keep_all=False):
    """Return the atomic catalogue, NumPy, and summary destinations."""
    tracer = normalize_tracer(tracer)
    zone = normalize_zone(zone)
    if isinstance(n_runs, (bool, np.bool_)) or int(n_runs) < 1:
        raise ValueError('n_runs must be a positive integer.')
    suffix = '_all' if keep_all else ''
    stem = (f'voids_{TRACER_OUTPUT_LABELS[tracer]}_{zone}_n{int(n_runs)}'
            f'{suffix}')
    root = Path(output_root)
    return {'fits': root / f'{stem}.fits',
            'npy': root / f'{stem}.npy',
            'summary': root / f'{stem}.json'}


def consensus_outputs_complete(paths) -> bool:
    """Return whether all products exist and carry the current shape schema."""
    paths = {name: Path(path) for name, path in paths.items()}
    if set(paths) != {'fits', 'npy', 'summary'}:
        return False
    if not all(path.is_file() and path.stat().st_size > 0 for path in paths.values()):
        return False
    try:
        with fitsio.FITS(str(paths['fits'])) as catalog:
            fits_names = tuple(catalog[1].get_colnames())
        npy_names = np.load(paths['npy'], mmap_mode='r', allow_pickle=False).dtype.names
    except (OSError, IndexError, TypeError, ValueError):
        return False
    required = set(CONSENSUS_DTYPE.names)
    return (required.issubset(fits_names) and required.issubset(npy_names or ()))


def load_pooled_voids(input_root,
                      tracer,
                      zone,
                      iterations,
                      layout='iteration',
                      verbose=True):
    """Read and pool clean, finite-shape voids from the requested runs.

    FITS I/O is restricted to the columns used by the algorithm and copied to
    the representative rows.  Other diagnostic columns in ``all.fits`` do
    not affect peak memory.
    """
    tracer = normalize_tracer(tracer)
    zone = normalize_zone(zone)
    iterations = _validate_iterations(iterations)

    centers = []
    radii = []
    ellipticities = []
    eigenvalues = []
    eigenvectors = []
    origins = []
    void_ids = []
    input_paths = []
    n_read = 0
    n_border = 0
    n_undefined_shape = 0

    for run_index, iteration in enumerate(iterations):
        path = iteration_catalog_path(input_root,
                                      tracer,
                                      zone,
                                      iteration,
                                      layout=layout)
        if not path.is_file():
            raise FileNotFoundError(f'Consensus input does not exist for '
                                    f'iteration {iteration}: '
                                    f'{path}')
        try:
            data = fitsio.read(str(path), ext=1, columns=list(_INPUT_COLUMNS))
        except (OSError, ValueError) as exc:
            raise ValueError(f'Cannot read consensus columns from {path}: '
                             f'{exc}') from exc

        finite_shape = (np.isfinite(data['R_EFF']) & np.isfinite(data['ELLIP']))
        for name in VOID_SHAPE_COLUMNS:
            finite_shape &= np.isfinite(data[name])
        border = np.asarray(data['BORDER'], dtype=bool)
        usable = finite_shape & ~border

        if np.any(np.asarray(data['R_EFF'][usable]) <= 0.0):
            raise ValueError(f'Usable voids must have R_EFF > 0 in {path}.')
        center = np.column_stack((data['XCART'][usable], data['YCART'][usable],
                                  data['ZCART'][usable])).astype(np.float64, copy=False)
        if not np.all(np.isfinite(center)):
            raise ValueError(f'Usable void centers must be finite in {path}.')

        count = int(np.count_nonzero(usable))
        centers.append(center)
        radii.append(np.asarray(data['R_EFF'][usable], dtype=np.float64))
        ellipticities.append(np.asarray(data['ELLIP'][usable], dtype=np.float64))
        eigenvalues.append(
            np.column_stack([np.asarray(data[name][usable], dtype=np.float64)
                             for name in EIGENVALUE_COLUMNS]))
        eigenvectors.append(
            np.stack([
                np.column_stack([
                    np.asarray(data[f'EIGVEC_{axis}_{component}'][usable],
                               dtype=np.float64) for component in ('X', 'Y', 'Z')])
                for axis in range(1, 4)],
                     axis=1))
        origins.append(np.full(count, run_index, dtype=np.int32))
        void_ids.append(np.asarray(data['VOID_ID'][usable], dtype=np.int64))
        input_paths.append(path)
        n_read += len(data)
        n_border += int(np.count_nonzero(border))
        n_undefined_shape += int(np.count_nonzero(~finite_shape))

        if verbose and ((run_index + 1) % 20 == 0 or run_index + 1 == len(iterations)):
            print(f'[consensus] read {run_index + 1}/{len(iterations)} runs',
                  flush=True)

    n_usable = sum(len(values) for values in radii)
    if n_usable == 0:
        raise ValueError('No non-border voids with finite shapes were found.')

    return PooledVoids(centers=np.ascontiguousarray(np.concatenate(centers),
                                                    dtype=np.float64),
                       r_eff=np.concatenate(radii),
                       ellipticity=np.concatenate(ellipticities),
                       eigenvalues=np.ascontiguousarray(np.concatenate(eigenvalues),
                                                        dtype=np.float64),
                       eigenvectors=np.ascontiguousarray(np.concatenate(eigenvectors),
                                                         dtype=np.float64),
                       source_iteration=np.concatenate(origins),
                       void_id=np.concatenate(void_ids),
                       iterations=iterations,
                       input_paths=tuple(input_paths),
                       n_read=int(n_read),
                       n_border=int(n_border),
                       n_undefined_shape=int(n_undefined_shape))


def contained_volume_fraction(distance, container_radius, sphere_radius):
    """Fraction of one sphere contained in another using the exact lens.

    The operation is intentionally asymmetric: the denominator is the volume
    of ``sphere_radius``.  Scalar inputs still return a one-dimensional array,
    matching vector calls made by the neighbour search.
    """
    distance, container_radius, sphere_radius = np.broadcast_arrays(
        np.asarray(distance, dtype=np.float64),
        np.asarray(container_radius, dtype=np.float64),
        np.asarray(sphere_radius, dtype=np.float64))
    distance = np.atleast_1d(distance)
    container_radius = np.atleast_1d(container_radius)
    sphere_radius = np.atleast_1d(sphere_radius)
    if (np.any(distance < 0.0) or np.any(container_radius <= 0.0)
            or np.any(sphere_radius <= 0.0)):
        raise ValueError('Distances must be non-negative and radii positive.')

    fraction = np.zeros(distance.shape, dtype=np.float64)
    nested = distance <= np.abs(container_radius - sphere_radius)
    if np.any(nested):
        inner = sphere_radius[nested]
        outer = container_radius[nested]
        fraction[nested] = np.where(inner <= outer, 1.0, (outer / inner)**3)

    lens = (~nested) & (distance < container_radius + sphere_radius)
    if np.any(lens):
        d = distance[lens]
        big = container_radius[lens]
        small = sphere_radius[lens]
        intersection = (np.pi * (big + small - d)**2 *
                        (d**2 + 2.0 * d * small - 3.0 * small**2 + 2.0 * d * big +
                         6.0 * small * big - 3.0 * big**2) / (12.0 * d))
        fraction[lens] = (intersection / ((4.0 / 3.0) * np.pi * small**3))
    return np.clip(fraction, 0.0, 1.0)


def _qualifying_neighbors(tree, centers, radii, seed, vol_frac, candidates=None):
    """Return all pooled spheres passing the asymmetric overlap threshold."""
    # For vol_frac >= 0.5 a passing sphere's centre must lie inside the seed
    # radius.  This exact bound avoids a much larger candidate query.
    if candidates is None:
        candidates = tree.query_ball_point(centers[seed],
                                           radii[seed],
                                           return_sorted=False)
    near = np.asarray(candidates, dtype=np.int64)
    if near.size == 1 and near[0] == seed:
        return near
    if near.size:
        delta = centers[near] - centers[seed]
        distance = np.einsum('ij,ij->i', delta, delta)
        np.sqrt(distance, out=distance)
        fraction = contained_volume_fraction(distance, radii[seed], radii[near])
        group = near[fraction > vol_frac]
    else:
        group = near

    # Self-overlap is one analytically.  Force the seed to remain present in
    # case a future spatial backend excludes a boundary/self result.
    if not np.any(group == seed):
        group = np.append(group, np.int64(seed))
    return group


def _votes_and_medoid(group, origins, radii):
    """Choose each run's largest group member and its median-radius vote."""
    if len(group) == 1:
        return group, int(group[0])
    group_origins = origins[group]
    group_radii = radii[group]

    # Run first, then decreasing radius, then pooled index.  Taking the first
    # member of every run therefore gives a deterministic largest-radius vote.
    order = np.lexsort((group, -group_radii, group_origins))
    sorted_origins = group_origins[order]
    first = np.empty(len(order), dtype=bool)
    first[0] = True
    first[1:] = sorted_origins[1:] != sorted_origins[:-1]
    votes = group[order[first]]

    radius_order = votes[np.lexsort((votes, radii[votes]))]
    medoid = int(radius_order[(len(radius_order) - 1) // 2])
    return votes, medoid


def _query_batches(tree, centers, radii, order, absorbed, batch_size, query_workers):
    """Batch variable-radius KD queries while preserving seed order."""
    for start in range(0, len(order), batch_size):
        stop = min(len(order), start + batch_size)
        seeds = order[start:stop]
        # Do not query points absorbed by an earlier batch.  A seed absorbed
        # within this batch is checked again by the consumer and skipped.
        seeds = seeds[~absorbed[seeds]]
        if not len(seeds):
            yield stop, seeds, ()
            continue
        candidates = tree.query_ball_point(centers[seeds],
                                           radii[seeds],
                                           return_sorted=False,
                                           workers=query_workers)
        yield stop, seeds, candidates


def _group_largest_first(centers, radii, origins, vol_frac, verbose, query_workers,
                         query_batch_size):
    """Steps 2--4 without retaining per-group member or vote lists."""
    tree = cKDTree(centers)
    # Match the reference implementation exactly: descending radius with the
    # lower pooled index first when radii tie.
    order = np.argsort(-radii, kind='stable')
    absorbed = np.zeros(len(radii), dtype=bool)
    medoids = np.empty(len(radii), dtype=np.int64)
    support = np.empty(len(radii), dtype=np.int32)
    n_groups = 0

    next_progress = 1_000_000
    for visited, seeds, candidates in _query_batches(tree, centers, radii, order,
                                                     absorbed, query_batch_size,
                                                     query_workers):
        for seed_value, near in zip(seeds, candidates):
            seed = int(seed_value)
            if absorbed[seed]:
                continue
            group = _qualifying_neighbors(tree,
                                          centers,
                                          radii,
                                          seed,
                                          vol_frac,
                                          candidates=near)
            votes, medoid = _votes_and_medoid(group, origins, radii)
            medoids[n_groups] = medoid
            support[n_groups] = len(votes)
            n_groups += 1
            absorbed[group] = True
        if verbose and visited >= next_progress:
            print(f'[consensus] grouping: visited {visited:,}/'
                  f'{len(order):,} pooled voids; {n_groups:,} objects',
                  flush=True)
            next_progress += 1_000_000

    result_medoids = medoids[:n_groups].copy()
    result_support = support[:n_groups].copy()
    return result_medoids, result_support, tree


def _prune_representatives(centers, radii, vol_frac, verbose, query_workers,
                           query_batch_size):
    """Step 5, returning the largest member of every surviving seed group.

    Membership remains inclusive, exactly as in the reference implementation:
    a representative already absorbed by an earlier seed may still be present
    in a later group.  The published catalogue chooses ``argmax(R_EFF)`` from
    each such group and reports that representative's original support.
    """
    tree = cKDTree(centers)
    order = np.argsort(-radii, kind='stable')
    absorbed = np.zeros(len(radii), dtype=bool)
    keep = np.empty(len(radii), dtype=np.int64)
    n_absorbed = np.empty(len(radii), dtype=np.int32)
    n_keep = 0

    next_progress = 250_000
    for visited, seeds, candidates in _query_batches(tree, centers, radii, order,
                                                     absorbed, query_batch_size,
                                                     query_workers):
        for seed_value, near in zip(seeds, candidates):
            seed = int(seed_value)
            if absorbed[seed]:
                continue
            group = _qualifying_neighbors(tree,
                                          centers,
                                          radii,
                                          seed,
                                          vol_frac,
                                          candidates=near)
            keep[n_keep] = group[np.argmax(radii[group])]
            n_absorbed[n_keep] = max(0, len(group) - 1)
            n_keep += 1
            absorbed[group] = True
        if verbose and visited >= next_progress:
            print(f'[consensus] pruning: visited {visited:,}/'
                  f'{len(order):,} representatives; {n_keep:,} kept',
                  flush=True)
            next_progress += 250_000

    return keep[:n_keep].copy(), n_absorbed[:n_keep].copy()


def build_consensus_catalog(pool,
                            vol_frac=DEFAULT_CONSENSUS_VOL_FRAC,
                            v_cut=DEFAULT_CONSENSUS_V_CUT,
                            keep_all=False,
                            query_workers=1,
                            query_batch_size=4096,
                            verbose=True):
    """Execute the six consensus steps on a :class:`PooledVoids` instance."""
    if not isinstance(pool, PooledVoids):
        raise TypeError('pool must be a PooledVoids instance.')
    n_voids = len(pool.r_eff)
    if n_voids < 1:
        raise ValueError('The pooled catalogue must contain at least one void.')
    if np.asarray(pool.r_eff).shape != (n_voids,):
        raise ValueError('pool.r_eff must be one-dimensional.')
    if np.asarray(pool.centers).shape != (n_voids, 3):
        raise ValueError('pool.centers must have shape (n_voids, 3).')
    for name in ('ellipticity', 'source_iteration', 'void_id'):
        if np.asarray(getattr(pool, name)).shape != (n_voids,):
            raise ValueError(f'pool.{name} must have one value per void.')
    if (not np.all(np.isfinite(pool.centers)) or not np.all(np.isfinite(pool.r_eff))
            or not np.all(np.isfinite(pool.ellipticity))):
        raise ValueError('Pooled centers, radii, and ellipticities must be finite.')
    if np.any(np.asarray(pool.r_eff) <= 0.0):
        raise ValueError('Pooled effective radii must be positive.')
    if np.asarray(pool.source_iteration).dtype.kind not in 'iu':
        raise TypeError('pool.source_iteration must contain integer indices.')
    if np.asarray(pool.void_id).dtype.kind not in 'iu':
        raise TypeError('pool.void_id must contain integer identifiers.')
    iterations = _validate_iterations(pool.iterations)
    origins = np.asarray(pool.source_iteration)
    if np.any(origins < 0) or np.any(origins >= len(iterations)):
        raise ValueError('pool.source_iteration contains an unknown run index.')
    vol_frac, v_cut = _validate_consensus_thresholds(vol_frac, v_cut)
    if isinstance(query_workers, (bool, np.bool_)):
        raise TypeError('query_workers must be -1 or a positive integer.')
    query_workers = int(query_workers)
    if query_workers == 0 or query_workers < -1:
        raise ValueError('query_workers must be -1 or a positive integer.')
    if isinstance(query_batch_size, (bool, np.bool_)):
        raise TypeError('query_batch_size must be a positive integer.')
    query_batch_size = int(query_batch_size)
    if query_batch_size < 1:
        raise ValueError('query_batch_size must be a positive integer.')
    started = time.time()

    if verbose:
        print(f'[consensus] pooled {len(pool.r_eff):,} usable voids; '
              f'grouping largest-first',
              flush=True)
    medoids, support, pool_tree = _group_largest_first(pool.centers, pool.r_eff,
                                                       pool.source_iteration, vol_frac,
                                                       verbose, query_workers,
                                                       query_batch_size)
    # Release the large pooled tree before constructing the representative
    # tree; this reduces the peak for BGS by hundreds of MB.
    del pool_tree

    representative_centers = np.ascontiguousarray(pool.centers[medoids])
    representative_radii = np.asarray(pool.r_eff[medoids], dtype=np.float64)
    keep, n_absorbed = _prune_representatives(representative_centers,
                                              representative_radii, vol_frac, verbose,
                                              query_workers, query_batch_size)

    pooled_rows = medoids[keep]
    catalog = np.zeros(len(keep), dtype=CONSENSUS_DTYPE)
    catalog['VOID_ID'] = pool.void_id[pooled_rows]
    catalog['SRC_ITER'] = np.asarray(pool.iterations,
                                     dtype=np.int32)[pool.source_iteration[pooled_rows]]
    catalog['X'], catalog['Y'], catalog['Z'] = pool.centers[pooled_rows].T
    catalog['R_EFF'] = pool.r_eff[pooled_rows]
    catalog['ELLIP'] = pool.ellipticity[pooled_rows]
    for axis, name in enumerate(EIGENVALUE_COLUMNS):
        catalog[name] = pool.eigenvalues[pooled_rows, axis]
    for axis in range(3):
        for component, label in enumerate(('X', 'Y', 'Z')):
            catalog[f'EIGVEC_{axis + 1}_{label}'] = (pool.eigenvectors[pooled_rows,
                                                                       axis, component])
    catalog['V'] = support[keep]
    catalog['FRAC_V'] = support[keep] / float(len(pool.iterations))
    catalog['N_ABSORBED'] = n_absorbed

    n_after_pruning = len(catalog)
    if not keep_all:
        catalog = catalog[catalog['FRAC_V'] > v_cut].copy()

    if len(np.unique(catalog['VOID_ID'])) != len(catalog):
        raise RuntimeError('Consensus construction produced duplicate VOID_IDs.')
    if verbose:
        action = 'support cut skipped' if keep_all else f'V/n > {v_cut:g}'
        print(f'[consensus] {len(medoids):,} representatives -> '
              f'{n_after_pruning:,} after pruning -> {len(catalog):,} '
              f'after {action}',
              flush=True)

    return ConsensusResult(catalog=catalog,
                           n_pooled=len(pool.r_eff),
                           n_groups=len(medoids),
                           n_representatives=len(medoids),
                           n_after_pruning=n_after_pruning,
                           n_after_support_cut=len(catalog),
                           elapsed_seconds=float(time.time() - started))


def _summary_payload(pool, result, tracer, zone, vol_frac, v_cut, keep_all):
    catalog = result.catalog
    if len(catalog):
        percentile = np.percentile(catalog['R_EFF'], (16.0, 84.0))
        catalog_statistics = {'median_r_eff': float(np.median(catalog['R_EFF'])),
                              'median_ellip': float(np.median(catalog['ELLIP'])),
                              'r_eff_p16': float(percentile[0]),
                              'r_eff_p84': float(percentile[1])}
    else:
        catalog_statistics = None
    return {
        'algorithm':
        'six-step ASTRA consensus catalogue',
        'tracer':
        normalize_tracer(tracer),
        'display_tracer':
        TRACER_OUTPUT_LABELS[normalize_tracer(tracer)],
        'zone':
        normalize_zone(zone),
        'iterations':
        list(pool.iterations),
        'n_iterations':
        len(pool.iterations),
        'vol_frac':
        float(vol_frac),
        'v_cut':
        None if keep_all else float(v_cut),
        'support_policy':
        ('all post-pruning representatives' if keep_all else 'strictly FRAC_V > v_cut'),
        'representative':
        'lower median-radius vote; one largest vote per run',
        'inputs': [str(path) for path in pool.input_paths],
        'n_voids_read':
        pool.n_read,
        'n_border':
        pool.n_border,
        'n_undefined_shape':
        pool.n_undefined_shape,
        'n_pooled':
        result.n_pooled,
        'n_groups':
        result.n_groups,
        'n_after_pruning':
        result.n_after_pruning,
        'n_after_support_cut':
        result.n_after_support_cut,
        'catalog_statistics':
        catalog_statistics,
        'elapsed_seconds':
        result.elapsed_seconds,}


def write_consensus_outputs(paths,
                            pool,
                            result,
                            tracer,
                            zone,
                            vol_frac=DEFAULT_CONSENSUS_VOL_FRAC,
                            v_cut=DEFAULT_CONSENSUS_V_CUT,
                            keep_all=False,
                            overwrite=False):
    """Write FITS, NPY, and JSON products through same-directory temporaries."""
    vol_frac, v_cut = _validate_consensus_thresholds(vol_frac, v_cut)
    destinations = {name: Path(path) for name, path in paths.items()}
    missing = {'fits', 'npy', 'summary'} - set(destinations)
    if missing:
        raise ValueError('Missing consensus output paths: ' +
                         ', '.join(sorted(missing)))
    existing = [path for path in destinations.values() if path.exists()]
    if existing and not overwrite:
        raise FileExistsError('Consensus outputs already exist: ' +
                              ', '.join(str(path) for path in existing))

    root = destinations['fits'].parent
    if any(path.parent != root for path in destinations.values()):
        raise ValueError('Consensus outputs must share one output directory.')
    root.mkdir(parents=True, exist_ok=True)
    pid = os.getpid()
    temporary = {name: path.with_name(f'.{path.name}.{pid}.tmp')
                 for name, path in destinations.items()}
    # Preserve recognized suffixes for writers which infer the format.
    temporary['fits'] = root / f'.{destinations["fits"].stem}.{pid}.tmp.fits'
    temporary['npy'] = root / f'.{destinations["npy"].stem}.{pid}.tmp.npy'
    temporary['summary'] = root / (f'.{destinations["summary"].stem}.{pid}.tmp.json')

    catalog = result.catalog
    header = [{'name': 'PLAN',
               'value': 'B',
               'comment': 'six-step ASTRA consensus'},
              {'name': 'TRACER',
               'value': TRACER_OUTPUT_LABELS[normalize_tracer(tracer)]},
              {'name': 'CAP',
               'value': normalize_zone(zone)},
              {'name': 'NITER',
               'value': len(pool.iterations)},
              {'name': 'VOLFRAC',
               'value': vol_frac},
              {'name': 'AGGREGAT',
               'value': 'medoid'},
              {'name': 'EIGORDER',
               'value': 'EIGVAL_1 >= EIGVAL_2 >= EIGVAL_3'},
              {'name': 'LOSANGLE',
               'value': 'acos(abs(EIGVEC_1 dot center/|center|))'},
              {'name': 'VCUT',
               'value': -1.0 if keep_all else v_cut},
              {'name': 'SELCAL',
               'value': 'calibrated on BGS vs DESIVAST V2 ZOBOV'},
              {'name': 'NPOOL',
               'value': result.n_pooled},
              {'name': 'NROWS',
               'value': len(catalog)},
              {'name': 'DATE',
               'value':
               datetime.datetime.now().astimezone().isoformat(timespec='seconds')},]
    units = [{'X': 'Mpc/h',
              'Y': 'Mpc/h',
              'Z': 'Mpc/h',
              'R_EFF': 'Mpc/h',
              'EIGVAL_1': '(Mpc/h)^2',
              'EIGVAL_2': '(Mpc/h)^2',
              'EIGVAL_3': '(Mpc/h)^2'}.get(name, '') for name in catalog.dtype.names]
    summary = _summary_payload(pool, result, tracer, zone, vol_frac, v_cut, keep_all)
    try:
        with fitsio.FITS(str(temporary['fits']), 'rw', clobber=True) as output:
            output.write(catalog, header=header, units=units, extname='VOIDS')
        np.save(temporary['npy'], catalog, allow_pickle=False)
        temporary['summary'].write_text(
            json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + '\n',
            encoding='utf-8')

        if not overwrite:
            late_existing = [path for path in destinations.values() if path.exists()]
            if late_existing:
                raise FileExistsError('Consensus outputs appeared while writing: ' +
                                      ', '.join(str(path) for path in late_existing))
        for name in ('fits', 'npy', 'summary'):
            os.replace(temporary[name], destinations[name])
    finally:
        for path in temporary.values():
            if path.exists():
                path.unlink()
    return destinations


def run_consensus(input_root,
                  output_root,
                  tracer,
                  zone,
                  iterations,
                  layout='iteration',
                  vol_frac=DEFAULT_CONSENSUS_VOL_FRAC,
                  v_cut=DEFAULT_CONSENSUS_V_CUT,
                  keep_all=False,
                  query_workers=1,
                  query_batch_size=4096,
                  overwrite=False,
                  verbose=True):
    """Read, build, and write one tracer/cap consensus catalogue."""
    iterations = _validate_iterations(iterations)
    if verbose:
        print(f'[consensus] {TRACER_OUTPUT_LABELS[normalize_tracer(tracer)]} '
              f'{normalize_zone(zone)} from {len(iterations)} runs',
              flush=True)
    pool = load_pooled_voids(input_root,
                             tracer,
                             zone,
                             iterations,
                             layout=layout,
                             verbose=verbose)
    result = build_consensus_catalog(pool,
                                     vol_frac=vol_frac,
                                     v_cut=v_cut,
                                     keep_all=keep_all,
                                     query_workers=query_workers,
                                     query_batch_size=query_batch_size,
                                     verbose=verbose)
    paths = consensus_output_paths(output_root,
                                   tracer,
                                   zone,
                                   len(iterations),
                                   keep_all=keep_all)
    written = write_consensus_outputs(paths,
                                      pool,
                                      result,
                                      tracer,
                                      zone,
                                      vol_frac=vol_frac,
                                      v_cut=v_cut,
                                      keep_all=keep_all,
                                      overwrite=overwrite)
    if verbose:
        print(f'[consensus] saved {written["fits"]}', flush=True)
    return result, written


__all__ = ['CONSENSUS_DTYPE',
           'ConsensusResult',
           'DEFAULT_CONSENSUS_VOL_FRAC',
           'DEFAULT_CONSENSUS_V_CUT',
           'PooledVoids',
           'build_consensus_catalog',
           'consensus_output_paths',
           'consensus_outputs_complete',
           'contained_volume_fraction',
           'iteration_catalog_path',
           'load_pooled_voids',
           'run_consensus',
           'write_consensus_outputs']
