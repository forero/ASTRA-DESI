import argparse
import gc
import os
from pathlib import Path
import re
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('NUMEXPR_NUM_THREADS', '1')

import numpy as np

try:
    from .astra_catalog_pipeline import (DEFAULT_H, DEFAULT_MEMORY_BYTES_PER_POINT,
                                         DEFAULT_MEMORY_FRACTION, DEFAULT_MIN_MEMBERS,
                                         DEFAULT_MIN_RANDOMS_PER_PIXEL,
                                         DEFAULT_MIN_RANDOMS_PER_RADIAL_BIN,
                                         DEFAULT_NSIDE, DEFAULT_RADIAL_BIN_WIDTH,
                                         DEFAULT_RA_MAX, DEFAULT_RA_MIN,
                                         DEFAULT_R_THRESHOLD, IterationConfig,
                                         build_case_consensus, concatenate_sky_samples,
                                         make_cartesian_case, normalize_catalog_tracer,
                                         parse_iteration_tokens, random_pool_signature,
                                         read_sky_sample, run_realizations,
                                         validate_common_options)
    from .publish_final_consensus import (FINAL_DATASET_NAMES,
                                          publish_consensus_products)
except ImportError:
    from astra_catalog_pipeline import (DEFAULT_H, DEFAULT_MEMORY_BYTES_PER_POINT,
                                        DEFAULT_MEMORY_FRACTION, DEFAULT_MIN_MEMBERS,
                                        DEFAULT_MIN_RANDOMS_PER_PIXEL,
                                        DEFAULT_MIN_RANDOMS_PER_RADIAL_BIN,
                                        DEFAULT_NSIDE, DEFAULT_RADIAL_BIN_WIDTH,
                                        DEFAULT_RA_MAX, DEFAULT_RA_MIN,
                                        DEFAULT_R_THRESHOLD, IterationConfig,
                                        build_case_consensus, concatenate_sky_samples,
                                        make_cartesian_case, normalize_catalog_tracer,
                                        parse_iteration_tokens, random_pool_signature,
                                        read_sky_sample, run_realizations,
                                        validate_common_options)
    from publish_final_consensus import (FINAL_DATASET_NAMES,
                                         publish_consensus_products)

DEFAULT_DATA_DIR = ('/global/cfs/cdirs/desi/survey/catalogs/DA2/LSS/'
                    'loa-v1/LSScats/v1.1/nonKP')
DEFAULT_OUTPUT_ROOT = '/pscratch/sd/v/vtorresg/void_catalog'
COSMOLOGIES = (('DR2_Om_1_Om0p301_h0p6736', 0.301), ('DR2_Om_2_Om0p315_h0p6736', 0.315),
               ('DR2_Om_3_Om0p329_h0p6736', 0.329))
COSMOLOGY_NAMES = ('low', 'default', 'high')
COSMOLOGY_BY_NAME = dict(zip(COSMOLOGY_NAMES, COSMOLOGIES))
DISK_TRACERS = {'BGS': 'BGS_ANY', 'LRG': 'LRG', 'ELG': 'ELGnotqso', 'QSO': 'QSO'}


def _tracer_argument(value):
    try:
        return normalize_catalog_tracer(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('tracer', type=_tracer_argument)
    parser.add_argument('--data-dir', default=DEFAULT_DATA_DIR)
    parser.add_argument('--output-root', default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        '--final-catalog-root',
        default=None,
        help=('Optional compact final-catalog root. Production jobs use '
              '/pscratch/sd/v/vtorresg/void_catalog_dr2_new.'))
    parser.add_argument('--caps',
                        nargs='+',
                        choices=('NGC', 'SGC'),
                        default=['NGC', 'SGC'])
    parser.add_argument('--cosmologies',
                        nargs='+',
                        choices=COSMOLOGY_NAMES,
                        default=list(COSMOLOGY_NAMES),
                        help=('Cosmologies to process: low (Omega_m=0.301), default '
                              '(0.315), and/or high (0.329). Default: all three.'))
    parser.add_argument('--iterations', nargs='+', default=['0-99'])
    random_selection = parser.add_mutually_exclusive_group()
    random_selection.add_argument(
        '--random-indices',
        nargs='+',
        default=None,
        help='Random files to pool (for example 0-17). Default: all available.')
    random_selection.add_argument(
        '--random-index',
        type=int,
        default=None,
        help='Use one random file only (legacy/debug option).')
    parser.add_argument('--workers', default='auto')
    parser.add_argument('--memory-fraction',
                        type=float,
                        default=DEFAULT_MEMORY_FRACTION)
    parser.add_argument('--memory-bytes-per-point',
                        type=int,
                        default=DEFAULT_MEMORY_BYTES_PER_POINT)
    parser.add_argument('--h', type=float, default=DEFAULT_H)
    parser.add_argument('--ra-min', type=float, default=DEFAULT_RA_MIN)
    parser.add_argument('--ra-max', type=float, default=DEFAULT_RA_MAX)
    parser.add_argument('--z-min', type=float, default=None)
    parser.add_argument('--z-max', type=float, default=None)
    parser.add_argument('--seed', type=int, default=12345)
    parser.add_argument('--random-factor', type=float, default=1.0)
    parser.add_argument('--r-threshold', type=float, default=DEFAULT_R_THRESHOLD)
    parser.add_argument('--min-members', type=int, default=DEFAULT_MIN_MEMBERS)
    parser.add_argument('--healpix-nside', type=int, default=DEFAULT_NSIDE)
    parser.add_argument('--min-randoms-per-pixel',
                        type=int,
                        default=DEFAULT_MIN_RANDOMS_PER_PIXEL)
    parser.add_argument('--min-randoms-per-radial-bin',
                        type=int,
                        default=DEFAULT_MIN_RANDOMS_PER_RADIAL_BIN)
    parser.add_argument('--radial-bin-width',
                        type=float,
                        default=DEFAULT_RADIAL_BIN_WIDTH)
    parser.add_argument('--mask-cache', default=None)
    parser.add_argument('--mask-chunk-size', type=int, default=1_000_000)
    parser.add_argument('--edge-chunk-size', type=int, default=250_000)
    parser.add_argument('--input-chunk-size', type=int, default=2_000_000)
    parser.add_argument('--include-membership', action='store_true')
    consensus_mode = parser.add_mutually_exclusive_group()
    consensus_mode.add_argument('--no-consensus', action='store_true')
    consensus_mode.add_argument('--consensus-only', action='store_true')
    parser.add_argument('--consensus-keep-all', action='store_true')
    parser.add_argument('--consensus-vol-frac', type=float, default=0.5)
    parser.add_argument('--consensus-v-cut', type=float, default=0.5)
    parser.add_argument('--consensus-workers', default='auto')
    output_mode = parser.add_mutually_exclusive_group()
    output_mode.add_argument('--resume', action='store_true')
    output_mode.add_argument('--overwrite', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--quiet', action='store_true')
    return parser.parse_args(argv)


def parse_random_indices(tokens=None, single_index=None):
    """Return selected source indices, or ``None`` for all discovered files."""
    if single_index is not None:
        if int(single_index) < 0:
            raise ValueError('--random-index must be non-negative.')
        return (int(single_index),)
    if not tokens:
        return None
    normalized = [str(value).strip().lower() for value in tokens]
    if 'all' in normalized:
        if len(normalized) != 1:
            raise ValueError('Use --random-indices all by itself.')
        return None
    try:
        return parse_iteration_tokens(tokens, default=(0, 0))
    except ValueError as exc:
        raise ValueError(f'Invalid --random-indices: {exc}') from exc


def select_cosmologies(names):
    """Resolve CLI names to unique cosmologies in the requested order."""
    selected = []
    for name in names:
        if name not in COSMOLOGY_BY_NAME:
            raise ValueError(f'Unknown cosmology {name!r}.')
        cosmology = COSMOLOGY_BY_NAME[name]
        if cosmology not in selected:
            selected.append(cosmology)
    return tuple(selected)


def _indexed_random_files(data_dir, prefix):
    pattern = re.compile(rf'^{re.escape(prefix)}_(\d+)_clustering\.ran\.fits$')
    found = {}
    for path in Path(data_dir).glob(f'{prefix}_*_clustering.ran.fits'):
        match = pattern.match(path.name)
        if match:
            found[int(match.group(1))] = path
    return found


def dr2_data_path(data_dir, tracer, cap):
    """Return the data path and an optional cap cut for combined files."""
    data_dir = Path(data_dir).expanduser()
    disk = DISK_TRACERS[normalize_catalog_tracer(tracer)]
    cap_data = data_dir / f'{disk}_{cap}_clustering.dat.fits'
    if cap_data.is_file():
        return cap_data, None
    combined_data = data_dir / f'{disk}_clustering.dat.fits'
    return combined_data, cap


def dr2_random_paths(data_dir, tracer, cap, requested_indices=None):
    """Resolve an ordered multi-file random pool for one survey cap."""
    data_dir = Path(data_dir).expanduser()
    disk = DISK_TRACERS[normalize_catalog_tracer(tracer)]
    cap_files = _indexed_random_files(data_dir, f'{disk}_{cap}')
    if cap_files:
        available = cap_files
        cap_cut = None
    else:
        available = _indexed_random_files(data_dir, disk)
        cap_cut = cap
    if not available:
        raise FileNotFoundError(f'No indexed random catalogues found for '
                                f'{disk} {cap} in '
                                f'{data_dir}.')
    indices = (tuple(sorted(available)) if requested_indices is None else tuple(
        int(value) for value in requested_indices))
    missing = [value for value in indices if value not in available]
    if missing:
        raise FileNotFoundError(f'Random indices not found for {disk} {cap}: '
                                f'{missing}. '
                                f'Available: {sorted(available)}')
    return tuple(available[value] for value in indices), indices, cap_cut


def _config(args, case_root, label, omega_m, cap, random_paths, random_indices,
            pool_signature):
    if args.mask_cache:
        mask_cache = Path(args.mask_cache) / label
    else:
        mask_cache = Path(case_root) / 'mask_cache'
    return IterationConfig(
        case_root=str(case_root),
        dataset=label,
        tracer=args.tracer,
        zone=cap,
        random_source=str(random_paths[0]),
        random_sources=tuple(str(Path(path).resolve()) for path in random_paths),
        random_source_indices=tuple(int(value) for value in random_indices),
        random_pool_signature=pool_signature,
        mask_cache=str(mask_cache),
        base_seed=args.seed,
        random_factor=args.random_factor,
        r_threshold=args.r_threshold,
        min_members=args.min_members,
        healpix_nside=args.healpix_nside,
        min_randoms_per_pixel=args.min_randoms_per_pixel,
        min_randoms_per_radial_bin=args.min_randoms_per_radial_bin,
        radial_bin_width=args.radial_bin_width,
        mask_chunk_size=args.mask_chunk_size,
        edge_chunk_size=args.edge_chunk_size,
        h=args.h,
        omega_m=omega_m,
        include_membership=args.include_membership,
        overwrite=args.overwrite)


def main(argv=None):
    args = parse_args(argv)
    iterations = parse_iteration_tokens(args.iterations)
    cosmologies = select_cosmologies(args.cosmologies)
    requested_random_indices = parse_random_indices(args.random_indices,
                                                    args.random_index)
    if args.final_catalog_root and args.consensus_keep_all:
        raise ValueError('--final-catalog-root is reserved for the standard '
                         'support-cut '
                         'consensus; do not combine it with --consensus-keep-all.')
    for _, omega_m in cosmologies:
        validate_common_options(
            iterations,
            args.random_factor,
            args.r_threshold,
            args.min_members,
            args.h,
            omega_m,
            args.memory_fraction,
            args.memory_bytes_per_point,
            healpix_nside=args.healpix_nside,
            min_randoms_per_pixel=args.min_randoms_per_pixel,
            min_randoms_per_radial_bin=args.min_randoms_per_radial_bin,
            radial_bin_width=args.radial_bin_width,
            mask_chunk_size=args.mask_chunk_size,
            edge_chunk_size=args.edge_chunk_size,
            input_chunk_size=args.input_chunk_size,
            consensus_vol_frac=args.consensus_vol_frac,
            consensus_v_cut=args.consensus_v_cut,
            consensus_workers=args.consensus_workers,
            ra_min=args.ra_min,
            ra_max=args.ra_max,
            z_min=args.z_min,
            z_max=args.z_max,
            workers=args.workers,
            seed=args.seed)
    output_root = Path(args.output_root).expanduser().resolve()
    if args.dry_run:
        print(f'DR2 source: {Path(args.data_dir).expanduser().resolve()}')
        print(f'Cosmologies: {", ".join(args.cosmologies)}')
        print(f'Realizations: {iterations[0]}..{iterations[-1]} '
              f'({len(iterations)})')
        for cap in args.caps:
            data_path, _ = dr2_data_path(args.data_dir, args.tracer, cap)
            random_paths, random_indices, _ = dr2_random_paths(
                args.data_dir, args.tracer, cap, requested_random_indices)
            print(f'{cap}: data={data_path}')
            print(f'  random pool={len(random_paths)} files; '
                  f'indices={list(random_indices)}')
            for label, _ in cosmologies:
                case_root = output_root / label
                print(f'{label} {cap}: pool -> '
                      f'{case_root}/{args.tracer.lower()}/{cap.lower()}/'
                      'iterNN/all.fits')
                if not args.no_consensus:
                    print(f'  consensus: {case_root}/consensus/'
                          f'voids_{args.tracer}_{cap}_n{len(iterations)}.fits')
                    if args.final_catalog_root:
                        dataset = FINAL_DATASET_NAMES[label]
                        print(f'  final: {Path(args.final_catalog_root) / dataset}/'
                              f'voids_{args.tracer}_{cap}.fits')
        return 0

    for cap in args.caps:
        if args.consensus_only:
            for label, omega_m in cosmologies:
                consensus_paths = build_case_consensus(
                    output_root / label,
                    args.tracer,
                    cap,
                    iterations,
                    resume=args.resume,
                    overwrite=args.overwrite,
                    keep_all=args.consensus_keep_all,
                    vol_frac=args.consensus_vol_frac,
                    v_cut=args.consensus_v_cut,
                    query_workers=args.consensus_workers,
                    quiet=args.quiet)
                if args.final_catalog_root:
                    published = publish_consensus_products(consensus_paths,
                                                           args.final_catalog_root,
                                                           FINAL_DATASET_NAMES[label],
                                                           args.tracer,
                                                           cap,
                                                           omega_m=omega_m,
                                                           resume=args.resume,
                                                           overwrite=args.overwrite)
                    if not args.quiet:
                        print(f'[final] {published["fits"]}', flush=True)
            continue

        data_path, data_cap_cut = dr2_data_path(args.data_dir, args.tracer, cap)
        random_paths, random_indices, random_cap_cut = dr2_random_paths(
            args.data_dir, args.tracer, cap, requested_random_indices)
        if not data_path.is_file():
            raise FileNotFoundError(data_path)
        for random_path in random_paths:
            if not random_path.is_file():
                raise FileNotFoundError(random_path)

        if not args.quiet:
            print(f'[DR2 {args.tracer} {cap}] reading {data_path.name} and '
                  f'{len(random_paths)} random files '
                  f'({random_indices[0]}..{random_indices[-1]})',
                  flush=True)
        objects = read_sky_sample(data_path,
                                  data_cap_cut,
                                  args.ra_min,
                                  args.ra_max,
                                  args.z_min,
                                  args.z_max,
                                  chunk_size=args.input_chunk_size)
        random_parts = []
        random_counts = []
        for position, (source_index,
                       random_path) in enumerate(zip(random_indices, random_paths),
                                                 start=1):
            if not args.quiet:
                print(f'[DR2 {args.tracer} {cap}] random source '
                      f'{position}/{len(random_paths)}: index={source_index} '
                      f'{random_path.name}',
                      flush=True)
            part = read_sky_sample(random_path,
                                   random_cap_cut,
                                   args.ra_min,
                                   args.ra_max,
                                   args.z_min,
                                   args.z_max,
                                   chunk_size=args.input_chunk_size)
            random_parts.append(part)
            random_counts.append(len(part))
        randoms = concatenate_sky_samples(random_parts)
        random_source_index = np.repeat(np.asarray(random_indices, dtype=np.int16),
                                        np.asarray(random_counts, dtype=np.int64))
        del random_parts
        gc.collect()
        pool_signature = random_pool_signature(random_paths, random_indices)
        if not args.quiet:
            print(f'[DR2 {args.tracer} {cap}] pooled '
                  f'{len(randoms):,} random rows from '
                  f'{len(random_paths)} sources',
                  flush=True)

        for label, omega_m in cosmologies:
            case_root = output_root / label
            config = _config(args, case_root, label, omega_m, cap, random_paths,
                             random_indices, pool_signature)
            if not args.quiet:
                print(f'[{label} {args.tracer} {cap}] Cartesian conversion', flush=True)
            case = make_cartesian_case(objects,
                                       randoms,
                                       omega_m,
                                       random_source_index=random_source_index)
            consensus_paths = run_realizations(
                case,
                config,
                iterations,
                workers=args.workers,
                resume=args.resume,
                memory_fraction=args.memory_fraction,
                memory_bytes_per_point=args.memory_bytes_per_point,
                consensus=not args.no_consensus,
                consensus_keep_all=args.consensus_keep_all,
                consensus_vol_frac=args.consensus_vol_frac,
                consensus_v_cut=args.consensus_v_cut,
                consensus_workers=args.consensus_workers,
                quiet=args.quiet)
            if args.final_catalog_root and not args.no_consensus:
                published = publish_consensus_products(consensus_paths,
                                                       args.final_catalog_root,
                                                       FINAL_DATASET_NAMES[label],
                                                       args.tracer,
                                                       cap,
                                                       omega_m=omega_m,
                                                       resume=args.resume,
                                                       overwrite=args.overwrite)
                if not args.quiet:
                    print(f'[final] {published["fits"]}', flush=True)
            del case
        del objects, randoms, random_source_index
        gc.collect()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
