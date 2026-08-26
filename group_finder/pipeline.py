import argparse
import gc
import json
import multiprocessing as mp
import os
from pathlib import Path
import time
import traceback

from astropy.cosmology import Planck18
from astropy.table import Table
import numpy as np

from .astra import (run_group_finder, warmup_accelerators as warmup_astra_accelerators)
from .consensus import (DEFAULT_CONSENSUS_VOL_FRAC, DEFAULT_CONSENSUS_V_CUT,
                        consensus_output_paths, consensus_outputs_complete,
                        iteration_catalog_path, run_consensus)
from .make_cat import (ELLIPTICITY_DEFINITION, R_EFF_DEFINITION,
                       build_random_membership_catalog, build_void_catalogs,
                       compute_void_shapes, void_catalog_has_required_columns,
                       write_membership_catalog, write_void_catalog)
from .plotting import plot_all_tracers
from .read_data import (TRACER_LABELS, TRACER_OUTPUT_LABELS, ZONES, cartesian_positions,
                        normalize_tracer, normalize_zone, raw_zone_path,
                        read_raw_object_positions, read_raw_random_realization,
                        read_raw_realization)
from .watershed import (apply_random_healpix_edge_mask, build_random_healpix_mask,
                        warmup_accelerators as warmup_watershed_accelerators)

DEFAULT_R_THRESHOLD = -0.25
DEFAULT_MIN_MEMBERS = 4
DEFAULT_NSIDE = 128
DEFAULT_MIN_RANDOMS_PER_PIXEL = 3
DEFAULT_MIN_RANDOMS_PER_RADIAL_BIN = 3
DEFAULT_RADIAL_BIN_WIDTH = 5.0
DEFAULT_TRACERS = tuple(TRACER_LABELS)
DEFAULT_ZONES = ZONES

_SHARED_OBJECT_POSITIONS = None
_SHARED_OBJECT_CONTEXT = None


def _iteration_root(output_root, tracer, zone, iteration, layout='legacy'):
    tracer = normalize_tracer(tracer)
    zone = normalize_zone(zone)
    label = TRACER_OUTPUT_LABELS[tracer]
    if layout == 'iteration':
        return (Path(output_root) / label.lower() / zone.lower() /
                f'iter{int(iteration):02d}')
    if layout != 'legacy':
        raise ValueError(f'Unknown output layout {layout!r}.')
    return Path(output_root) / 'catalogs' / label / zone


def _catalog_paths(output_root, tracer, zone, iteration, layout='legacy'):
    tracer = normalize_tracer(tracer)
    zone = normalize_zone(zone)
    label = TRACER_OUTPUT_LABELS[tracer]
    base = _iteration_root(output_root, tracer, zone, iteration, layout)
    if layout == 'iteration':
        return {'all': base / 'all.fits',
                'clean': base / 'clean.fits',
                'membership': base / 'membership.fits'}
    stem = f'{label}_{zone}_iter{int(iteration):03d}'
    return {'all': base / f'{stem}_all.fits',
            'clean': base / f'{stem}_clean.fits',
            'membership': base / f'{stem}_membership.fits'}


def _required_case_paths(args, tracer, zone, iteration):
    """Files that make an iteration complete for the requested products."""
    paths = _catalog_paths(args.output_root, tracer, zone, iteration,
                           getattr(args, 'output_layout', 'legacy'))
    required = [paths['all'], paths['clean']]
    if not getattr(args, 'no_membership', False):
        required.append(paths['membership'])
    return required


def _plot_path(output_root, iteration, tracer=None, zone=None, layout='legacy'):
    if layout == 'iteration' and tracer is not None and zone is not None:
        return (_iteration_root(output_root, tracer, zone, iteration, layout) /
                'R_EFF_ELLIP.png')
    return (Path(output_root) / 'plots' /
            f'all_tracers_zones_iter{int(iteration):03d}_R_EFF_ELLIP.png')


def _summary_path(output_root, iteration, tracer=None, zone=None, layout='legacy'):
    if layout == 'iteration' and tracer is not None and zone is not None:
        return (_iteration_root(output_root, tracer, zone, iteration, layout) /
                'summary.json')
    return (Path(output_root) / f'run_iter{int(iteration):03d}_summary.json')


def _json_default(value):
    if isinstance(value, os.PathLike):
        return os.fspath(value)
    raise TypeError(f'Object of type {value.__class__.__name__} '
                    'is not JSON serializable')


def _write_json(path, payload, overwrite=False):
    path = Path(path)
    if path.exists() and not overwrite:
        raise FileExistsError(f'Output already exists: {path}. Use --overwrite.')
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    try:
        temporary.write_text(json.dumps(
            payload, indent=2, sort_keys=True, allow_nan=False, default=_json_default) +
                             '\n',
                             encoding='utf-8')
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return path


def _normalized_selection(args):
    tracers = []
    for value in args.tracers:
        tracer = normalize_tracer(value)
        if tracer not in tracers:
            tracers.append(tracer)
    zones = []
    for value in args.zones:
        zone = normalize_zone(value)
        if zone not in zones:
            zones.append(zone)
    return tuple(tracers), tuple(zones)


def _validate_args(args):
    if args.iteration < 0:
        raise ValueError('--iteration must be non-negative.')
    if not -1.0 <= args.r_threshold <= 1.0:
        raise ValueError('--r-threshold must lie within [-1, 1].')
    for name in ('min_members', 'healpix_nside', 'min_randoms_per_pixel',
                 'min_randoms_per_radial_bin', 'plot_bootstrap_samples', 'ellip_bins',
                 'reff_bins', 'min_combined_count'):

        if int(getattr(args, name)) < 1:
            raise ValueError(f'--{name.replace("_", "-")} must be positive.')
    if args.plot_bootstrap_samples < 2:
        raise ValueError('--plot-bootstrap-samples must be at least 2.')
    if args.radial_bin_width <= 0.0:
        raise ValueError('--radial-bin-width must be positive.')
    if args.h <= 0.0:
        raise ValueError('--h must be positive.')
    if getattr(args, 'memory_fraction', 0.0) <= 0.0 or args.memory_fraction > 1.0:
        raise ValueError('--memory-fraction must lie within (0, 1].')
    if getattr(args, 'memory_bytes_per_point', 0) < 1:
        raise ValueError('--memory-bytes-per-point must be positive.')
    if getattr(args, 'consensus', False) and getattr(args, 'plot_only', False):
        raise ValueError('--consensus and --plot-only cannot be used together.')
    if (getattr(args, 'consensus_only', False) and getattr(args, 'plot_only', False)):
        raise ValueError('--consensus-only and --plot-only cannot be used together.')
    if getattr(args, 'consensus_only', False):
        args.consensus = True
    consensus_vol_frac = float(
        getattr(args, 'consensus_vol_frac', DEFAULT_CONSENSUS_VOL_FRAC))
    if (not np.isfinite(consensus_vol_frac) or not 0.5 <= consensus_vol_frac <= 1.0):
        raise ValueError('--consensus-vol-frac must lie within [0.5, 1].')
    consensus_v_cut = float(getattr(args, 'consensus_v_cut', DEFAULT_CONSENSUS_V_CUT))
    if (not np.isfinite(consensus_v_cut) or not 0.0 <= consensus_v_cut <= 1.0):
        raise ValueError('--consensus-v-cut must lie within [0, 1].')
    consensus_workers = str(getattr(args, 'consensus_workers', 'auto')).lower()
    if consensus_workers != 'auto':
        try:
            if int(consensus_workers) < 1:
                raise ValueError
        except ValueError as exc:
            raise ValueError('--consensus-workers must be "auto" or a '
                             'positive integer.') from exc
    if int(getattr(args, 'consensus_query_batch_size', 4096)) < 1:
        raise ValueError('--consensus-query-batch-size must be positive.')


def _preflight(args, tracers, zones, iterations=None):
    iterations = ([args.iteration] if iterations is None else list(iterations))
    paths = []
    layout = getattr(args, 'output_layout', 'legacy')
    for iteration in iterations:
        for tracer in tracers:
            for zone in zones:
                paths.extend(_required_case_paths(args, tracer, zone, iteration))
                paths.append(
                    _summary_path(args.output_root, iteration, tracer, zone, layout))
        if not getattr(args, 'no_plot', False):
            paths.append(_plot_path(args.output_root, iteration,
                                    tracers[0] if len(tracers) == 1 else None,
                                    zones[0] if len(zones) == 1 else None, layout))
    existing = [str(path) for path in paths if path.exists()]
    if existing and not args.overwrite and not getattr(args, 'resume', False):
        raise FileExistsError('Outputs already exist; use --overwrite or another '
                              '--output-root: ' + ', '.join(existing))


def _consensus_root(args):
    value = getattr(args, 'consensus_output_root', None)
    return (Path(value) if value else Path(args.output_root) / 'consensus')


def _preflight_consensus(args, tracers, zones, iterations):
    if not getattr(args, 'consensus', False):
        return
    existing = []
    for tracer in tracers:
        for zone in zones:
            paths = consensus_output_paths(_consensus_root(args),
                                           tracer,
                                           zone,
                                           len(iterations),
                                           keep_all=getattr(args, 'consensus_keep_all',
                                                            False))
            existing.extend(str(path) for path in paths.values() if path.exists())
    if existing and not args.overwrite and not getattr(args, 'resume', False):
        raise FileExistsError('Consensus outputs already exist; use '
                              '--overwrite, --resume, or '
                              'another --consensus-output-root: ' + ', '.join(existing))


def _catalog_samples(table):
    radius = np.asarray(table['R_EFF'], dtype=np.float64)
    ellipticity = np.asarray(table['ELLIP'], dtype=np.float64)
    finite = np.isfinite(radius) & np.isfinite(ellipticity)
    return {'R_EFF': radius[finite], 'ELLIP': ellipticity[finite]}


def _plot_existing_catalogs(args, tracers, zones, iteration=None):
    iteration = args.iteration if iteration is None else int(iteration)
    layout = getattr(args, 'output_layout', 'legacy')
    samples = {}
    missing = []
    for tracer in tracers:
        for zone in zones:
            path = _catalog_paths(args.output_root, tracer, zone, iteration,
                                  layout)['all']
            if not path.is_file():
                missing.append(str(path))
                continue
            samples[(tracer, zone)] = _catalog_samples(Table.read(path))
    if missing:
        raise FileNotFoundError('Cannot use --plot-only; catalog files are missing: ' +
                                ', '.join(missing))

    nonempty_samples = {key: values
                        for key, values in samples.items()
                        if len(values['R_EFF']) and len(values['ELLIP'])}
    if not nonempty_samples:
        raise ValueError('No measurable voids are available for the comparison plot.')

    figure_path = _plot_path(args.output_root, iteration,
                             tracers[0] if len(tracers) == 1 else None,
                             zones[0] if len(zones) == 1 else None, layout)
    if figure_path.exists() and not args.overwrite:
        raise FileExistsError(f'Output already exists: {figure_path}. Use --overwrite.')
    return plot_all_tracers(nonempty_samples,
                            figure_path,
                            iteration=iteration,
                            r_threshold=args.r_threshold,
                            ellip_bins=args.ellip_bins,
                            reff_bins=args.reff_bins,
                            n_bootstrap=args.plot_bootstrap_samples,
                            seed=args.plot_seed,
                            min_combined_count=args.min_combined_count,
                            use_tex=not args.no_tex)


def run_case(args, tracer, zone, iteration=None):

    tracer = normalize_tracer(tracer)
    zone = normalize_zone(zone)
    iteration = args.iteration if iteration is None else int(iteration)
    input_path = raw_zone_path(args.raw_dir, zone, tracer=tracer)
    started = time.time()
    print(f'[{TRACER_OUTPUT_LABELS[tracer]} {zone}] reading objects and '
          f'RANDITER={iteration}',
          flush=True)

    shared_context = (str(Path(input_path).resolve()), tracer, zone)
    if (_SHARED_OBJECT_POSITIONS is not None
            and _SHARED_OBJECT_CONTEXT == shared_context):
        object_positions = _SHARED_OBJECT_POSITIONS
        randoms, raw_metadata = read_raw_random_realization(input_path,
                                                            tracer=tracer,
                                                            iteration=iteration)
    else:
        objects, randoms, raw_metadata = read_raw_realization(input_path,
                                                              tracer=tracer,
                                                              iteration=iteration)
        object_positions = cartesian_positions(objects)
        del objects
    random_positions = cartesian_positions(randoms)

    print(f'[{TRACER_OUTPUT_LABELS[tracer]} {zone}] RANDITER={iteration} mask '
          f'(NSIDE={args.healpix_nside}, count >= '
          f'{args.min_randoms_per_pixel})',
          flush=True)
    selection = build_random_healpix_mask(
        raw_path=input_path,
        tracer=tracer,
        iteration=iteration,
        nside=args.healpix_nside,
        min_randoms_per_pixel=args.min_randoms_per_pixel,
        min_randoms_per_radial_bin=args.min_randoms_per_radial_bin,
        radial_bin_width=args.radial_bin_width,
        cache_path=args.mask_cache,
        chunk_size=args.mask_chunk_size,
        random_records=randoms)

    print(f'[{TRACER_OUTPUT_LABELS[tracer]} {zone}] Delaunay: '
          f'{len(object_positions):,} data + {len(randoms):,} random',
          flush=True)
    unmasked = run_group_finder(object_positions=object_positions,
                                random_positions=random_positions,
                                r_threshold=args.r_threshold,
                                min_members=args.min_members)
    del random_positions
    masked = apply_random_healpix_edge_mask(unmasked,
                                            selection,
                                            random_ra=np.asarray(randoms['RA'],
                                                                 dtype=np.float64),
                                            random_dec=np.asarray(randoms['DEC'],
                                                                  dtype=np.float64),
                                            edge_chunk_size=args.edge_chunk_size,
                                            min_members=args.min_members,
                                            retain_edge_diagnostics=False)
    result = masked.result

    # Retain only the arrays needed by catalog construction.  Connectivity is
    # by far the largest graph allocation (hundreds of millions of indices for
    # BGS) and is no longer needed after topology pruning.
    positions = result.graph.positions
    is_data = result.graph.is_data
    grouping = result.grouping
    n_data = int(result.graph.n_data)
    n_random = int(result.graph.n_random)
    n_edges = int(len(result.graph.edges))
    n_groups_before_mask = int(len(unmasked.grouping.group_sizes))
    edge_group_ids = np.asarray(masked.edge_group_ids).copy()
    n_groups_affected = int(len(edge_group_ids))
    n_groups_pruned = int(len(masked.pruned_group_ids))
    n_groups_discarded = int(len(masked.discarded_group_ids))
    mask_random_iterations = int(selection.n_random_iterations)
    write_membership = not getattr(args, 'no_membership', False)
    if write_membership:
        group_ids_before_random = np.asarray(
            masked.group_ids_before_mask[n_data:]).copy()
        selection_pruned_random = np.asarray(
            masked.selection_pruned_member[n_data:]).copy()
    else:
        group_ids_before_random = None
        selection_pruned_random = None
        # Raw random records are needed only for membership construction; the
        # Cartesian graph arrays already contain everything shape measurement
        # needs.
        del randoms
    del result, masked, unmasked, selection
    gc.collect()

    shapes = compute_void_shapes(positions=positions,
                                 is_data=is_data,
                                 group_ids=grouping.group_ids,
                                 coordinate_scale=args.h)
    catalogs = build_void_catalogs(shapes,
                                   border_group_ids=edge_group_ids,
                                   tracer=tracer,
                                   zone=zone,
                                   iteration=iteration,
                                   h=args.h)
    n_defined_shapes = int(
        np.count_nonzero(np.isfinite(shapes.r_eff) & np.isfinite(shapes.ellipticity)))
    del shapes, positions, is_data
    gc.collect()
    layout = getattr(args, 'output_layout', 'legacy')
    paths = _catalog_paths(args.output_root, tracer, zone, iteration, layout)
    write_void_catalog(paths['all'], catalogs.all_voids, overwrite=args.overwrite)
    write_void_catalog(paths['clean'], catalogs.clean_voids, overwrite=args.overwrite)
    if write_membership:
        membership = build_random_membership_catalog(
            randoms=randoms,
            group_ids=grouping.group_ids[n_data:],
            group_ids_before_mask=group_ids_before_random,
            r_values=grouping.r_values[n_data:],
            threshold_selected=grouping.threshold_selected[n_data:],
            selection_pruned_member=selection_pruned_random,
            border_group_ids=edge_group_ids,
            tracer=tracer,
            zone=zone,
            iteration=iteration)
        write_membership_catalog(paths['membership'],
                                 membership,
                                 overwrite=args.overwrite)
        n_random_membership_rows = int(len(membership))
        n_random_assigned = int(np.count_nonzero(membership['MEMBER']))
        del membership
        del randoms
    else:
        n_random_membership_rows = None
        n_random_assigned = None

    summary = {
        'tracer': tracer,
        'display_tracer': TRACER_OUTPUT_LABELS[tracer],
        'zone': zone,
        'iteration': int(iteration),
        'input': input_path,
        'release': raw_metadata['release'],
        'source_data_tracer': raw_metadata['source_data_tracer'],
        'source_random_tracer': raw_metadata['source_random_tracer'],
        'n_data': n_data,
        'n_random': n_random,
        'n_edges': n_edges,
        'n_threshold_selected': int(np.count_nonzero(grouping.threshold_selected)),
        'n_groups_before_mask': n_groups_before_mask,
        'n_groups_affected_by_mask': n_groups_affected,
        'n_groups_pruned': n_groups_pruned,
        'n_groups_discarded': n_groups_discarded,
        'n_groups_after_mask': int(len(grouping.group_sizes)),
        'n_defined_shapes': n_defined_shapes,
        'n_catalog_all': int(len(catalogs.all_voids)),
        'n_catalog_border': int(np.count_nonzero(catalogs.all_voids['BORDER'])),
        'n_catalog_clean': int(len(catalogs.clean_voids)),
        'all_catalog': paths['all'],
        'clean_catalog': paths['clean'],
        'membership_catalog': (paths['membership'] if write_membership else None),
        'membership_written': bool(write_membership),
        'n_random_membership_rows': n_random_membership_rows,
        'n_random_assigned': n_random_assigned,
        'mask_random_iterations': mask_random_iterations,
        'elapsed_seconds': float(time.time() - started)}

    if layout == 'iteration':
        case_summary = {'algorithm':
                        'ASTRA literal lowest-index Delaunay watershed',
                        'mask': ('single-RANDITER angular/radial count selection with '
                                 'seed-connected topology pruning'),
                        'r_threshold':
                        float(args.r_threshold),
                        'min_members':
                        int(args.min_members),
                        'h':
                        float(args.h),
                        'r_eff_definition':
                        R_EFF_DEFINITION,
                        'ellipticity_definition':
                        ELLIPTICITY_DEFINITION,
                        'case':
                        summary,}
        _write_json(_summary_path(args.output_root, iteration, tracer, zone, layout),
                    case_summary,
                    overwrite=(args.overwrite or getattr(args, 'resume', False)))

    print(f'[{TRACER_OUTPUT_LABELS[tracer]} {zone}] '
          f'groups={summary["n_groups_after_mask"]:,}, '
          f'all={summary["n_catalog_all"]:,}, '
          f'border={summary["n_catalog_border"]:,}, '
          f'clean={summary["n_catalog_clean"]:,}',
          flush=True)
    return catalogs, summary


def _parse_iterations(args):
    values = getattr(args, 'iterations', None)
    if not values:
        return (int(args.iteration),)

    result = []
    for token in values:
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
            result.extend(range(start, stop, step))
        elif '-' in text[1:]:
            start_text, stop_text = text.split('-', 1)
            start, stop = int(start_text), int(stop_text)
            if stop < start:
                raise ValueError(f'Invalid iteration range {text!r}.')
            result.extend(range(start, stop + 1))
        else:
            result.append(int(text))

    unique = tuple(dict.fromkeys(result))
    if not unique:
        raise ValueError('--iterations did not select any realization.')
    if min(unique) < 0 or max(unique) >= 1000:
        raise ValueError('iterations must lie in [0, 1000).')
    return unique


def _iteration_complete(args, tracer, zone, iteration):
    layout = getattr(args, 'output_layout', 'legacy')
    paths = _required_case_paths(args, tracer, zone, iteration)
    paths.append(_summary_path(args.output_root, iteration, tracer, zone, layout))
    if not all(path.is_file() and path.stat().st_size > 0 for path in paths):
        return False
    catalog_paths = _catalog_paths(args.output_root, tracer, zone, iteration, layout)
    return all(void_catalog_has_required_columns(catalog_paths[name])
               for name in ('all', 'clean'))


def _available_memory_bytes():
    candidates = []
    try:
        import psutil
        candidates.append(int(psutil.virtual_memory().available))
    except (ImportError, OSError, ValueError):
        pass

    cgroup_root = Path('/sys/fs/cgroup')
    try:
        maximum = (cgroup_root / 'memory.max').read_text().strip()
        current = int((cgroup_root / 'memory.current').read_text().strip())
        if maximum != 'max':
            candidates.append(max(0, int(maximum) - current))
    except (OSError, ValueError):
        pass

    if not candidates:
        return 1 << 60
    return min(value for value in candidates if value > 0)


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


def _resolve_worker_count(args, n_tasks, n_data):
    requested = str(args.workers).strip().lower()
    cpu_limit = min(int(n_tasks), _allocated_cpus())
    combined_points = 2 * int(n_data)
    estimate = (2 * 1024**3 + combined_points * int(args.memory_bytes_per_point))
    usable_memory = int(_available_memory_bytes() * float(args.memory_fraction))
    memory_limit = max(1, usable_memory // max(1, estimate))

    if requested == 'auto':
        workers = max(1, min(cpu_limit, memory_limit))
    else:
        workers = int(requested)
        if workers < 1:
            raise ValueError('--workers must be "auto" or a positive integer.')
        workers = min(workers, cpu_limit)
        if workers > memory_limit:
            print('[parallel] warning: requested workers exceed the conservative '
                  f'memory estimate ({workers} requested, '
                  f'{memory_limit} estimated safe).',
                  flush=True)

    print('[parallel] workers='
          f'{workers}/{n_tasks}, CPUs={_allocated_cpus()}, '
          f'available_memory={_available_memory_bytes() / 1024 ** 3:.1f} GiB, '
          f'estimated_peak_per_worker={estimate / 1024 ** 3:.1f} GiB',
          flush=True)
    return workers


def _init_parallel_worker(object_positions, context):
    global _SHARED_OBJECT_POSITIONS, _SHARED_OBJECT_CONTEXT
    _SHARED_OBJECT_POSITIONS = object_positions
    _SHARED_OBJECT_CONTEXT = context


def _run_iteration_worker(payload):
    args, tracer, zone, iteration, collect_samples = payload
    try:
        catalogs, summary = run_case(args, tracer, zone, iteration=iteration)
        samples = (_catalog_samples(catalogs.all_voids) if collect_samples else None)
        del catalogs
        gc.collect()
        return {'iteration': int(iteration), 'summary': summary, 'samples': samples}
    except Exception as exc:
        detail = traceback.format_exc()
        raise RuntimeError(f'{TRACER_OUTPUT_LABELS[normalize_tracer(tracer)]} {zone} '
                           f'iteration {iteration} failed: {exc}\n{detail}') from exc


def _run_parallel_case(args, tracer, zone, iterations):
    tracer = normalize_tracer(tracer)
    zone = normalize_zone(zone)
    input_path = raw_zone_path(args.raw_dir, zone, tracer=tracer)
    pending = []
    completed = []
    for iteration in iterations:
        if args.resume and _iteration_complete(args, tracer, zone, iteration):
            print(f'[resume] {TRACER_OUTPUT_LABELS[tracer]} {zone} '
                  f'iter{iteration:02d} complete; skipping',
                  flush=True)
            continue
        pending.append(iteration)

    if not pending:
        return completed

    print(f'[parallel] preloading shared data coordinates for '
          f'{TRACER_OUTPUT_LABELS[tracer]} {zone}',
          flush=True)
    object_positions, _ = read_raw_object_positions(input_path,
                                                    tracer,
                                                    reference_iteration=pending[0])
    context = (str(Path(input_path).resolve()), tracer, zone)
    workers = _resolve_worker_count(args, len(pending), len(object_positions))

    worker_args = argparse.Namespace(**vars(args))
    if args.resume:
        worker_args.overwrite = True
    collect_samples = not args.no_plot
    payloads = [(worker_args, tracer, zone, iteration, collect_samples)
                for iteration in pending]

    warmup_astra_accelerators()
    warmup_watershed_accelerators()
    if workers == 1:
        _init_parallel_worker(object_positions, context)
        try:
            for payload in payloads:
                completed.append(_run_iteration_worker(payload))
        finally:
            _init_parallel_worker(None, None)
    else:
        pool_context = mp.get_context('fork')
        with pool_context.Pool(processes=workers,
                               initializer=_init_parallel_worker,
                               initargs=(object_positions, context),
                               maxtasksperchild=1) as pool:
            for result in pool.imap_unordered(_run_iteration_worker,
                                              payloads,
                                              chunksize=1):
                completed.append(result)
                print(f'[parallel] {TRACER_OUTPUT_LABELS[tracer]} {zone} '
                      f'iter{result["iteration"]:02d} complete '
                      f'({len(completed)}/{len(pending)})',
                      flush=True)

    completed.sort(key=lambda item: item['iteration'])
    manifest = {
        'tracer': tracer,
        'display_tracer': TRACER_OUTPUT_LABELS[tracer],
        'zone': zone,
        'iterations_requested': [int(value) for value in iterations],
        'iterations_completed_this_run': [item['iteration'] for item in completed],
        'workers': int(workers),
        'output_layout': args.output_layout,
        'cases': {f'iter{item["iteration"]:02d}': item['summary']
                  for item in completed},}
    manifest_path = (Path(args.output_root) / TRACER_OUTPUT_LABELS[tracer].lower() /
                     zone.lower() / 'run_summary.json')
    _write_json(manifest_path, manifest, overwrite=(args.overwrite or args.resume))
    return completed


def _run_consensus_cases(args, tracers, zones, iterations):
    """Build requested final catalogues, with resume-safe partial handling."""
    output_root = _consensus_root(args)
    query_workers = (min(8, _allocated_cpus()) if str(args.consensus_workers).lower()
                     == 'auto' else int(args.consensus_workers))
    if not args.consensus_quiet:
        print(f'[consensus] KD-tree query workers={query_workers}, '
              f'batch={args.consensus_query_batch_size:,}',
              flush=True)
    written = []
    for tracer in tracers:
        for zone in zones:
            paths = consensus_output_paths(output_root,
                                           tracer,
                                           zone,
                                           len(iterations),
                                           keep_all=args.consensus_keep_all)
            complete = consensus_outputs_complete(paths)
            if complete and args.resume and not args.overwrite:
                print(f'[resume] consensus {TRACER_OUTPUT_LABELS[tracer]} {zone} '
                      f'already complete; skipping',
                      flush=True)
                written.append(paths)
                continue

            partial = any(path.exists() for path in paths.values())
            _, case_paths = run_consensus(
                input_root=args.output_root,
                output_root=output_root,
                tracer=tracer,
                zone=zone,
                iterations=iterations,
                layout=args.output_layout,
                vol_frac=args.consensus_vol_frac,
                v_cut=args.consensus_v_cut,
                keep_all=args.consensus_keep_all,
                query_workers=query_workers,
                query_batch_size=args.consensus_query_batch_size,
                overwrite=(args.overwrite or (args.resume and partial)),
                verbose=not args.consensus_quiet)
            written.append(case_paths)
            gc.collect()
    return written


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw-dir', default='temp/raw')
    parser.add_argument('--output-root', default='temp/group_finder')
    parser.add_argument('--tracers', nargs='+', default=list(DEFAULT_TRACERS))
    parser.add_argument('--zones', nargs='+', default=list(DEFAULT_ZONES))
    parser.add_argument('--iteration', type=int, default=0)
    parser.add_argument('--iterations',
                        nargs='+',
                        default=None,
                        help='Iterations to process (e.g. 0-99, 0:100, or 0 1 2).')
    parser.add_argument(
        '--workers',
        default='auto',
        help='Parallel iteration workers, or "auto" for a memory-aware value.')
    parser.add_argument('--output-layout',
                        choices=('legacy', 'iteration'),
                        default='legacy',
                        help='Use tracer/zone/iterNN directories with "iteration".')
    parser.add_argument(
        '--memory-fraction',
        type=float,
        default=0.85,
        help='Fraction of currently available memory usable by auto workers.')
    parser.add_argument(
        '--memory-bytes-per-point',
        type=int,
        default=1100,
        help='Conservative peak-memory model for one combined Delaunay point.')
    parser.add_argument('--r-threshold', type=float, default=DEFAULT_R_THRESHOLD)
    parser.add_argument('--min-members', type=int, default=DEFAULT_MIN_MEMBERS)
    parser.add_argument('--healpix-nside', type=int, default=DEFAULT_NSIDE)
    parser.add_argument('--min-randoms-per-pixel',
                        type=int,
                        default=DEFAULT_MIN_RANDOMS_PER_PIXEL)
    parser.add_argument('--min-randoms-per-radial-bin',
                        dest='min_randoms_per_radial_bin',
                        type=int,
                        default=DEFAULT_MIN_RANDOMS_PER_RADIAL_BIN)
    parser.add_argument('--radial-bin-width',
                        type=float,
                        default=DEFAULT_RADIAL_BIN_WIDTH)
    parser.add_argument('--mask-cache', default='temp/group_finder/healpix_masks')
    parser.add_argument('--mask-chunk-size', type=int, default=1_000_000)
    parser.add_argument('--edge-chunk-size', type=int, default=250_000)
    parser.add_argument('--h', type=float, default=float(Planck18.h))
    parser.add_argument('--ellip-bins', type=int, default=30)
    parser.add_argument('--reff-bins', type=int, default=30)
    parser.add_argument('--plot-bootstrap-samples', type=int, default=2000)
    parser.add_argument('--plot-seed', type=int, default=12345)
    parser.add_argument('--min-combined-count', type=int, default=5)
    parser.add_argument('--no-tex', action='store_true')
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--resume',
                        action='store_true',
                        help='Skip complete iterations and rebuild partial ones.')
    parser.add_argument('--no-plot',
                        action='store_true',
                        help='Skip diagnostic plots (recommended for 100x production).')
    parser.add_argument('--no-membership',
                        action='store_true',
                        help=('Do not build membership.fits (recommended when '
                              'the per-run catalogues are only consensus inputs).'))
    parser.add_argument('--consensus',
                        action='store_true',
                        help='Build one six-step consensus catalogue after all runs.')
    parser.add_argument(
        '--consensus-only',
        action='store_true',
        help='Build consensus from existing all.fits files; run no ASTRA cases.')
    parser.add_argument('--consensus-output-root',
                        default=None,
                        help='Consensus directory (default: OUTPUT_ROOT/consensus).')
    parser.add_argument(
        '--consensus-vol-frac',
        type=float,
        default=DEFAULT_CONSENSUS_VOL_FRAC,
        help='Asymmetric sphere-volume threshold; must be at least 0.5.')
    parser.add_argument('--consensus-v-cut',
                        type=float,
                        default=DEFAULT_CONSENSUS_V_CUT,
                        help='Keep final objects with V/n strictly above this value.')
    parser.add_argument('--consensus-keep-all',
                        action='store_true',
                        help='Skip the final V/n support cut for diagnostics.')
    parser.add_argument('--consensus-workers',
                        default='auto',
                        help='Parallel KD-tree query threads, or "auto" (up to 8).')
    parser.add_argument('--consensus-query-batch-size',
                        type=int,
                        default=4096,
                        help='Number of ordered seeds per batched KD-tree query.')
    parser.add_argument('--consensus-quiet',
                        action='store_true',
                        help='Suppress per-case consensus progress messages.')
    parser.add_argument('--plot-only', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    _validate_args(args)
    tracers, zones = _normalized_selection(args)
    iterations = _parse_iterations(args)

    if args.consensus_only:
        _preflight_consensus(args, tracers, zones, iterations)
        if args.dry_run:
            for tracer in tracers:
                for zone in zones:
                    inputs = [iteration_catalog_path(args.output_root,
                                                     tracer,
                                                     zone,
                                                     iteration,
                                                     layout=args.output_layout)
                              for iteration in iterations]
                    outputs = consensus_output_paths(_consensus_root(args),
                                                     tracer,
                                                     zone,
                                                     len(iterations),
                                                     keep_all=args.consensus_keep_all)
                    print(f'{TRACER_OUTPUT_LABELS[tracer]} {zone}: '
                          f'{len(inputs)} all.fits inputs '
                          f'({inputs[0]} ... {inputs[-1]}) -> '
                          f'{outputs["fits"]}')
            return 0
        _run_consensus_cases(args, tracers, zones, iterations)
        return 0

    if args.plot_only:
        if args.dry_run:
            for iteration in iterations:
                for tracer in tracers:
                    for zone in zones:
                        path = _catalog_paths(args.output_root, tracer, zone, iteration,
                                              args.output_layout)['all']
                        print(f'{TRACER_OUTPUT_LABELS[tracer]} {zone} '
                              f'iter{iteration:02d}: {path}')
            return 0
        for iteration in iterations:
            figure_path = _plot_existing_catalogs(args,
                                                  tracers,
                                                  zones,
                                                  iteration=iteration)
            print(f'Figure: {figure_path}', flush=True)
        return 0

    _preflight(args, tracers, zones, iterations=iterations)
    _preflight_consensus(args, tracers, zones, iterations)

    if args.dry_run:
        for tracer in tracers:
            for zone in zones:
                source = raw_zone_path(args.raw_dir, zone, tracer=tracer)
                for iteration in iterations:
                    paths = _catalog_paths(args.output_root, tracer, zone, iteration,
                                           args.output_layout)
                    print(f'{TRACER_OUTPUT_LABELS[tracer]} {zone} '
                          f'iter{iteration:02d}: {source} -> '
                          f'{paths["all"]}, {paths["clean"]}' +
                          ('' if args.no_membership else f', {paths["membership"]}'))
        if args.consensus:
            for tracer in tracers:
                for zone in zones:
                    outputs = consensus_output_paths(_consensus_root(args),
                                                     tracer,
                                                     zone,
                                                     len(iterations),
                                                     keep_all=args.consensus_keep_all)
                    print(f'{TRACER_OUTPUT_LABELS[tracer]} {zone} consensus -> '
                          f'{outputs["fits"]}')
        return 0

    parallel_mode = (len(iterations) > 1 or args.output_layout == 'iteration')
    if parallel_mode:
        started = time.time()
        for tracer in tracers:
            for zone in zones:
                _run_parallel_case(args, tracer, zone, iterations)
        if not args.no_plot:
            for iteration in iterations:
                figure_path = _plot_existing_catalogs(args,
                                                      tracers,
                                                      zones,
                                                      iteration=iteration)
                print(f'Figure: {figure_path}', flush=True)
        if args.consensus:
            _run_consensus_cases(args, tracers, zones, iterations)
        print(f'[parallel] all requested cases finished in '
              f'{(time.time() - started) / 60.0:.2f} min',
              flush=True)
        return 0

    started = time.time()
    samples = {}
    cases = {}
    for tracer in tracers:
        for zone in zones:
            catalogs, summary = run_case(args, tracer, zone)
            samples[(tracer, zone)] = _catalog_samples(catalogs.all_voids)
            cases[f'{TRACER_OUTPUT_LABELS[tracer]}_{zone}'] = summary
            del catalogs
            gc.collect()

    figure_path = None
    if not args.no_plot:
        nonempty_samples = {key: values
                            for key, values in samples.items()
                            if len(values['R_EFF']) and len(values['ELLIP'])}
        if not nonempty_samples:
            raise ValueError('No finite post-mask void shapes remain for the '
                             'comparison plot.')
        figure_path = plot_all_tracers(nonempty_samples,
                                       _plot_path(args.output_root, args.iteration),
                                       iteration=args.iteration,
                                       r_threshold=args.r_threshold,
                                       ellip_bins=args.ellip_bins,
                                       reff_bins=args.reff_bins,
                                       n_bootstrap=args.plot_bootstrap_samples,
                                       seed=args.plot_seed,
                                       min_combined_count=args.min_combined_count,
                                       use_tex=not args.no_tex)
    summary = {
        'algorithm':
        'ASTRA literal lowest-index Delaunay watershed',
        'mask': ('single-RANDITER angular/radial count selection with '
                 'seed-connected topology pruning'),
        'iteration':
        int(args.iteration),
        'r_threshold':
        float(args.r_threshold),
        'min_members':
        int(args.min_members),
        'tracers': [TRACER_OUTPUT_LABELS[value] for value in tracers],
        'raw_tracers':
        list(tracers),
        'zones':
        list(zones),
        'h':
        float(args.h),
        'r_eff_definition':
        R_EFF_DEFINITION,
        'ellipticity_definition':
        ELLIPTICITY_DEFINITION,
        'catalog_policy': {'all': ('all post-mask groups; undefined shapes are NaN and '
                                   'BORDER marks groups that touched the selection'),
                           'clean':
                           'post-mask survivors with BORDER=False'},
        'cases':
        cases,
        'figure':
        figure_path,
        'elapsed_seconds':
        float(time.time() - started)}
    summary_path = _write_json(_summary_path(args.output_root, args.iteration),
                               summary,
                               overwrite=args.overwrite)
    if figure_path is not None:
        print(f'Figure: {figure_path}', flush=True)
    print(f'Summary: {summary_path}', flush=True)
    if args.consensus:
        _run_consensus_cases(args, tracers, zones, iterations)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
