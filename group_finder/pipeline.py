import argparse
import gc
import json
import os
from pathlib import Path
import time

from astropy.cosmology import Planck18
from astropy.table import Table
import numpy as np

from .astra import run_group_finder
from .make_cat import (ELLIPTICITY_DEFINITION,
                       R_EFF_DEFINITION,
                       build_random_membership_catalog,
                       build_void_catalogs,
                       compute_void_shapes,
                       write_membership_catalog,
                       write_void_catalog)
from .plotting import plot_all_tracers
from .read_data import (TRACER_DISPLAY,
                        TRACER_LABELS,
                        ZONES,
                        cartesian_positions,
                        normalize_tracer,
                        normalize_zone,
                        raw_zone_path,
                        read_raw_realization)
from .watershed import (apply_random_healpix_edge_mask,
                        build_all_random_healpix_mask)


DEFAULT_R_THRESHOLD = -0.25
DEFAULT_MIN_MEMBERS = 4
DEFAULT_NSIDE = 128
DEFAULT_MIN_RANDOMS_PER_PIXEL = 3
DEFAULT_MIN_RANDOMS_PER_RADIAL_BIN = 3
DEFAULT_RADIAL_BIN_WIDTH = 5.0
DEFAULT_TRACERS = tuple(TRACER_LABELS)
DEFAULT_ZONES = ZONES


def _catalog_paths(output_root,tracer, zone, iteration):
    tracer = normalize_tracer(tracer)
    zone = normalize_zone(zone)
    label = TRACER_DISPLAY[tracer]
    base = Path(output_root) / 'catalogs' / label / zone
    stem = f'{label}_{zone}_iter{int(iteration):03d}'
    return {'all': base / f'{stem}_all.fits',
            'clean': base / f'{stem}_clean.fits',
            'membership': base / f'{stem}_membership.fits'}


def _plot_path(output_root, iteration):
    return (Path(output_root) / 'plots'
            / f'all_tracers_zones_iter{int(iteration):03d}_R_EFF_ELLIP.png')


def _summary_path(output_root, iteration):
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
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True,
                                        allow_nan=False, default=_json_default)
                             + '\n', encoding='utf-8')
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
    for name in ('min_members',
                 'healpix_nside',
                 'min_randoms_per_pixel',
                 'min_randoms_per_radial_bin',
                 'plot_bootstrap_samples',
                 'ellip_bins',
                 'reff_bins',
                 'min_combined_count'):

        if int(getattr(args, name)) < 1:
            raise ValueError(f'--{name.replace("_", "-")} must be positive.')
    if args.plot_bootstrap_samples < 2:
        raise ValueError('--plot-bootstrap-samples must be at least 2.')
    if args.radial_bin_width <= 0.0:
        raise ValueError('--radial-bin-width must be positive.')
    if args.h <= 0.0:
        raise ValueError('--h must be positive.')


def _preflight(args, tracers, zones):
    paths = []
    for tracer in tracers:
        for zone in zones:
            paths.extend(_catalog_paths(args.output_root, tracer, zone, args.iteration).values())
    paths.extend((_plot_path(args.output_root, args.iteration),
                  _summary_path(args.output_root, args.iteration)))
    existing = [str(path) for path in paths if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError('Outputs already exist; use --overwrite or another '
                              '--output-root: ' + ', '.join(existing))


def _catalog_samples(table):
    radius = np.asarray(table['R_EFF'], dtype=np.float64)
    ellipticity = np.asarray(table['ELLIP'], dtype=np.float64)
    finite = np.isfinite(radius) & np.isfinite(ellipticity)
    return {'R_EFF': radius[finite],
            'ELLIP': ellipticity[finite]}


def _plot_existing_catalogs(args, tracers, zones):
    samples = {}
    missing = []
    for tracer in tracers:
        for zone in zones:
            path = _catalog_paths(args.output_root, tracer, zone, args.iteration)['all']
            if not path.is_file():
                missing.append(str(path))
                continue
            samples[(tracer, zone)] = _catalog_samples(Table.read(path))
    if missing:
        raise FileNotFoundError('Cannot use --plot-only; catalog files are missing: '
                                + ', '.join(missing))

    nonempty_samples = {key: values for key, values in samples.items()
                        if len(values['R_EFF']) and len(values['ELLIP'])}
    if not nonempty_samples:
        raise ValueError('No measurable voids are available for the comparison plot.')

    figure_path = _plot_path(args.output_root, args.iteration)
    if figure_path.exists() and not args.overwrite:
        raise FileExistsError(f'Output already exists: {figure_path}. Use --overwrite.')
    return plot_all_tracers(
        nonempty_samples,
        figure_path,
        iteration=args.iteration,
        r_threshold=args.r_threshold,
        ellip_bins=args.ellip_bins,
        reff_bins=args.reff_bins,
        n_bootstrap=args.plot_bootstrap_samples,
        seed=args.plot_seed,
        min_combined_count=args.min_combined_count,
        use_tex=not args.no_tex)


def run_case(args, tracer, zone):

    tracer = normalize_tracer(tracer)
    zone = normalize_zone(zone)
    input_path = raw_zone_path(args.raw_dir, zone, tracer=tracer)
    started = time.time()
    print(f'[{TRACER_DISPLAY[tracer]} {zone}] reading objects and '
          f'RANDITER={args.iteration}', flush=True)
    objects, randoms, raw_metadata = read_raw_realization(input_path,
                                                          tracer=tracer,
                                                          iteration=args.iteration)
    object_positions = cartesian_positions(objects)
    random_positions = cartesian_positions(randoms)

    print(f'[{TRACER_DISPLAY[tracer]} {zone}] all-RANDITER mask '
          f'(NSIDE={args.healpix_nside}, mean count >= '
          f'{args.min_randoms_per_pixel})', flush=True)
    selection = build_all_random_healpix_mask(raw_path=input_path,
                                              tracer=tracer,
                                              nside=args.healpix_nside,
                                              min_randoms_per_pixel=args.min_randoms_per_pixel,
                                              min_randoms_per_radial_bin=args.min_randoms_per_radial_bin,
                                              radial_bin_width=args.radial_bin_width,
                                              cache_path=args.mask_cache,
                                              chunk_size=args.mask_chunk_size)

    print(f'[{TRACER_DISPLAY[tracer]} {zone}] Delaunay: '
          f'{len(objects):,} data + {len(randoms):,} random', flush=True)
    unmasked = run_group_finder(object_positions=object_positions,
                                random_positions=random_positions,
                                r_threshold=args.r_threshold,
                                min_members=args.min_members)
    masked = apply_random_healpix_edge_mask(unmasked, selection,
                                            random_ra=np.asarray(randoms['RA'], dtype=np.float64),
                                            random_dec=np.asarray(randoms['DEC'], dtype=np.float64),
                                            edge_chunk_size=args.edge_chunk_size,
                                            min_members=args.min_members)
    result = masked.result

    shapes = compute_void_shapes(positions=result.graph.positions,
                                 is_data=result.graph.is_data,
                                 group_ids=result.grouping.group_ids,
                                 coordinate_scale=args.h)
    catalogs = build_void_catalogs(shapes,
                                   border_group_ids=masked.edge_group_ids,
                                   tracer=tracer,
                                   zone=zone,
                                   iteration=args.iteration,
                                   h=args.h)
    random_start = int(result.graph.n_data)
    membership = build_random_membership_catalog(
        randoms=randoms,
        group_ids=result.grouping.group_ids[random_start:],
        group_ids_before_mask=masked.group_ids_before_mask[random_start:],
        r_values=result.grouping.r_values[random_start:],
        threshold_selected=result.grouping.threshold_selected[random_start:],
        selection_pruned_member=masked.selection_pruned_member[random_start:],
        border_group_ids=masked.edge_group_ids,
        tracer=tracer,
        zone=zone,
        iteration=args.iteration)
    paths = _catalog_paths(args.output_root, tracer, zone, args.iteration)
    write_void_catalog(paths['all'], catalogs.all_voids, overwrite=args.overwrite)
    write_void_catalog(paths['clean'], catalogs.clean_voids, overwrite=args.overwrite)
    write_membership_catalog(paths['membership'], membership,
                             overwrite=args.overwrite)

    summary = {'tracer': tracer,
               'display_tracer': TRACER_DISPLAY[tracer],
               'zone': zone,
               'iteration': int(args.iteration),
               'input': input_path,
               'release': raw_metadata['release'],
               'source_data_tracer': raw_metadata['source_data_tracer'],
               'source_random_tracer': raw_metadata['source_random_tracer'],
               'n_data': int(result.graph.n_data),
               'n_random': int(result.graph.n_random),
               'n_edges': int(len(result.graph.edges)),
               'n_threshold_selected': int(np.count_nonzero(
                   result.grouping.threshold_selected)),
               'n_groups_before_mask': int(len(
                   unmasked.grouping.group_sizes)),
               'n_groups_affected_by_mask': int(len(
                   masked.edge_group_ids)),
               'n_groups_pruned': int(len(masked.pruned_group_ids)),
               'n_groups_discarded': int(len(masked.discarded_group_ids)),
               'n_groups_after_mask': int(len(result.grouping.group_sizes)),
               'n_defined_shapes': int(np.count_nonzero(
                   np.isfinite(shapes.r_eff) & np.isfinite(shapes.ellipticity))),
               'n_catalog_all': int(len(catalogs.all_voids)),
               'n_catalog_border': int(np.count_nonzero(
                   catalogs.all_voids['BORDER'])),
               'n_catalog_clean': int(len(catalogs.clean_voids)),
               'all_catalog': paths['all'],
               'clean_catalog': paths['clean'],
               'membership_catalog': paths['membership'],
               'n_random_membership_rows': int(len(membership)),
               'n_random_assigned': int(np.count_nonzero(membership['MEMBER'])),
               'mask_random_iterations': int(selection.n_random_iterations),
               'elapsed_seconds': float(time.time() - started)}

    print(f'[{TRACER_DISPLAY[tracer]} {zone}] '
          f'groups={summary["n_groups_after_mask"]:,}, '
          f'all={summary["n_catalog_all"]:,}, '
          f'border={summary["n_catalog_border"]:,}, '
          f'clean={summary["n_catalog_clean"]:,}',
          flush=True)
    return catalogs, summary


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw-dir', default='temp/raw')
    parser.add_argument('--output-root', default='temp/group_finder')
    parser.add_argument('--tracers', nargs='+', default=list(DEFAULT_TRACERS))
    parser.add_argument('--zones', nargs='+', default=list(DEFAULT_ZONES))
    parser.add_argument('--iteration', type=int, default=0)
    parser.add_argument('--r-threshold', type=float, default=DEFAULT_R_THRESHOLD)
    parser.add_argument('--min-members', type=int, default=DEFAULT_MIN_MEMBERS)
    parser.add_argument('--healpix-nside', type=int, default=DEFAULT_NSIDE)
    parser.add_argument('--min-randoms-per-pixel', type=int, default=DEFAULT_MIN_RANDOMS_PER_PIXEL)
    parser.add_argument('--min-randoms-per-radial-bin', dest='min_randoms_per_radial_bin', type=int, default=DEFAULT_MIN_RANDOMS_PER_RADIAL_BIN)
    parser.add_argument('--radial-bin-width', type=float, default=DEFAULT_RADIAL_BIN_WIDTH)
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
    parser.add_argument('--plot-only', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    _validate_args(args)
    tracers, zones = _normalized_selection(args)

    if args.plot_only:
        if args.dry_run:
            for tracer in tracers:
                for zone in zones:
                    path = _catalog_paths(
                        args.output_root, tracer, zone, args.iteration)['all']
                    print(f'{TRACER_DISPLAY[tracer]} {zone}: {path}')
            print(f'Figure: {_plot_path(args.output_root, args.iteration)}')
            return 0
        figure_path = _plot_existing_catalogs(args, tracers, zones)
        print(f'Figure: {figure_path}', flush=True)
        return 0

    _preflight(args, tracers, zones)

    if args.dry_run:
        for tracer in tracers:
            for zone in zones:
                paths = _catalog_paths(args.output_root, tracer, zone, args.iteration)
                print(f'{TRACER_DISPLAY[tracer]} {zone}: '
                      f'{raw_zone_path(args.raw_dir, zone, tracer=tracer)} -> '
                      f'{paths["all"]}, {paths["clean"]}, '
                      f'{paths["membership"]}')
        print(f'Figure: {_plot_path(args.output_root, args.iteration)}')
        return 0

    started = time.time()
    samples = {}
    cases = {}
    for tracer in tracers:
        for zone in zones:
            catalogs, summary = run_case(args, tracer, zone)
            samples[(tracer, zone)] = _catalog_samples(
                catalogs.all_voids)
            cases[f'{TRACER_DISPLAY[tracer]}_{zone}'] = summary
            del catalogs
            gc.collect()

    nonempty_samples = {key: values for key, values in samples.items()
                        if len(values['R_EFF']) and len(values['ELLIP'])}
    if not nonempty_samples:
        raise ValueError('No finite post-mask void shapes remain for the comparison plot.')
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
    summary = {'algorithm': 'ASTRA literal lowest-index Delaunay watershed',
               'mask': ('all-RANDITER mean-count angular/radial selection with '
                        'seed-connected topology pruning'),
              'iteration': int(args.iteration),
              'r_threshold': float(args.r_threshold),
              'min_members': int(args.min_members),
              'tracers': [TRACER_DISPLAY[value] for value in tracers],
              'raw_tracers': list(tracers),
              'zones': list(zones),
              'h': float(args.h),
              'r_eff_definition': R_EFF_DEFINITION,
              'ellipticity_definition': ELLIPTICITY_DEFINITION,
              'catalog_policy': {
              'all': ('all post-mask groups; undefined shapes are NaN and '
                      'BORDER marks groups that touched the selection'),
              'clean': 'post-mask survivors with BORDER=False'},
              'cases': cases,
              'figure': figure_path,
              'elapsed_seconds': float(time.time() - started)}
    summary_path = _write_json(_summary_path(args.output_root, args.iteration),
                               summary, overwrite=args.overwrite)
    print(f'Figure: {figure_path}', flush=True)
    print(f'Summary: {summary_path}', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
