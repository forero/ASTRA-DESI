import argparse
import os
from pathlib import Path
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
                                         DEFAULT_NSIDE, DEFAULT_OMEGA_M,
                                         DEFAULT_RADIAL_BIN_WIDTH, DEFAULT_RA_MAX,
                                         DEFAULT_RA_MIN, DEFAULT_R_THRESHOLD,
                                         IterationConfig, build_case_consensus,
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
                                        DEFAULT_NSIDE, DEFAULT_OMEGA_M,
                                        DEFAULT_RADIAL_BIN_WIDTH, DEFAULT_RA_MAX,
                                        DEFAULT_RA_MIN, DEFAULT_R_THRESHOLD,
                                        IterationConfig, build_case_consensus,
                                        make_cartesian_case, normalize_catalog_tracer,
                                        parse_iteration_tokens, random_pool_signature,
                                        read_sky_sample, run_realizations,
                                        validate_common_options)
    from publish_final_consensus import (FINAL_DATASET_NAMES,
                                         publish_consensus_products)

DEFAULT_MOCK_DIR = '/pscratch/sd/h/hrincon/LSScats/testfibers'
DEFAULT_OUTPUT_DIR = '/pscratch/sd/v/vtorresg/void_catalog/fiber_assignment'


def _tracer_argument(value):
    try:
        return normalize_catalog_tracer(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def mock_path(mock_dir, tracer, mock_kind):
    return Path(mock_dir).expanduser() / f'{tracer}_{mock_kind}.fits'


def random_path(mock_dir, tracer):
    return Path(mock_dir).expanduser() / f'{tracer}_randoms.fits'


def build_parser(default_output=DEFAULT_OUTPUT_DIR, fixed_kind=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('tracer', type=_tracer_argument)
    parser.add_argument('--mock-dir', default=DEFAULT_MOCK_DIR)
    parser.add_argument('--output-dir', default=default_output)
    parser.add_argument(
        '--final-catalog-root',
        default=None,
        help=('Optional compact final-catalog root. Production jobs use '
              '/pscratch/sd/v/vtorresg/void_catalog_dr2_new.'))
    if fixed_kind is None:
        parser.add_argument('--mock-kind',
                            choices=('altmtl', 'complete'),
                            default='altmtl')
    parser.add_argument('--caps',
                        nargs='+',
                        choices=('NGC', 'SGC'),
                        default=['NGC', 'SGC'])
    parser.add_argument('--iterations', nargs='+', default=['0-99'])
    parser.add_argument('--workers', default='auto')
    parser.add_argument('--memory-fraction',
                        type=float,
                        default=DEFAULT_MEMORY_FRACTION)
    parser.add_argument('--memory-bytes-per-point',
                        type=int,
                        default=DEFAULT_MEMORY_BYTES_PER_POINT)
    parser.add_argument('--ra-min', type=float, default=DEFAULT_RA_MIN)
    parser.add_argument('--ra-max', type=float, default=DEFAULT_RA_MAX)
    parser.add_argument('--z-min', type=float, default=None)
    parser.add_argument('--z-max', type=float, default=None)
    parser.add_argument('--bgs-mr-limit', type=float, default=None)
    parser.add_argument('--h', type=float, default=DEFAULT_H)
    parser.add_argument('--omega-m', type=float, default=DEFAULT_OMEGA_M)
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
    return parser


def _config(args, case_root, mock_kind, cap, randoms_path):
    mask_cache = (Path(args.mask_cache) if args.mask_cache else Path(case_root) /
                  'mask_cache')
    pool_signature = random_pool_signature((randoms_path,), (0,))
    return IterationConfig(case_root=str(case_root),
                           dataset=f'mock-{mock_kind}',
                           tracer=args.tracer,
                           zone=cap,
                           random_source=str(randoms_path),
                           random_sources=(str(Path(randoms_path).resolve()),),
                           random_source_indices=(0,),
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
                           omega_m=args.omega_m,
                           include_membership=args.include_membership,
                           overwrite=args.overwrite)


def run_mock_cli(argv=None, default_output=DEFAULT_OUTPUT_DIR, fixed_kind=None):
    parser = build_parser(default_output=default_output, fixed_kind=fixed_kind)
    args = parser.parse_args(argv)
    mock_kind = fixed_kind or args.mock_kind
    iterations = parse_iteration_tokens(args.iterations)
    if args.final_catalog_root and args.consensus_keep_all:
        raise ValueError('--final-catalog-root is reserved for the standard '
                         'support-cut '
                         'consensus; do not combine it with --consensus-keep-all.')
    validate_common_options(iterations,
                            args.random_factor,
                            args.r_threshold,
                            args.min_members,
                            args.h,
                            args.omega_m,
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
    if (args.bgs_mr_limit is not None and not np.isfinite(args.bgs_mr_limit)):
        raise ValueError('--bgs-mr-limit must be finite.')

    data_path = mock_path(args.mock_dir, args.tracer, mock_kind)
    randoms_path = random_path(args.mock_dir, args.tracer)
    case_root = Path(args.output_dir).expanduser().resolve() / mock_kind

    if args.dry_run:
        print(f'Data: {data_path}')
        print(f'Random pool: {randoms_path}')
        print(f'Realizations: {iterations[0]}..{iterations[-1]} '
              f'({len(iterations)})')
        for cap in args.caps:
            print(f'{cap}: {case_root}/{args.tracer.lower()}/'
                  f'{cap.lower()}/iterNN/all.fits')
            if not args.no_consensus:
                print(f'{cap} consensus: {case_root}/consensus/'
                      f'voids_{args.tracer}_{cap}_n{len(iterations)}.fits')
                if args.final_catalog_root:
                    dataset = FINAL_DATASET_NAMES[mock_kind]
                    print(f'{cap} final: '
                          f'{Path(args.final_catalog_root) / dataset}/'
                          f'voids_{args.tracer}_{cap}.fits')
        return 0

    if not args.consensus_only and not data_path.is_file():
        raise FileNotFoundError(data_path)
    if not args.consensus_only and not randoms_path.is_file():
        raise FileNotFoundError(randoms_path)

    for cap in args.caps:
        if args.consensus_only:
            consensus_paths = build_case_consensus(case_root,
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
                                                       FINAL_DATASET_NAMES[mock_kind],
                                                       args.tracer,
                                                       cap,
                                                       omega_m=args.omega_m,
                                                       resume=args.resume,
                                                       overwrite=args.overwrite)
                if not args.quiet:
                    print(f'[final] {published["fits"]}', flush=True)
            continue

        config = _config(args, case_root, mock_kind, cap, randoms_path)
        magnitude_limit = (args.bgs_mr_limit if args.tracer == 'BGS' else None)
        if not args.quiet:
            print(f'[{mock_kind} {args.tracer} {cap}] reading inputs', flush=True)
        objects = read_sky_sample(data_path,
                                  cap,
                                  args.ra_min,
                                  args.ra_max,
                                  args.z_min,
                                  args.z_max,
                                  magnitude_limit,
                                  chunk_size=args.input_chunk_size)
        randoms = read_sky_sample(randoms_path,
                                  cap,
                                  args.ra_min,
                                  args.ra_max,
                                  args.z_min,
                                  args.z_max,
                                  magnitude_limit,
                                  chunk_size=args.input_chunk_size)
        case = make_cartesian_case(objects, randoms, args.omega_m)
        del objects, randoms
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
                                                   FINAL_DATASET_NAMES[mock_kind],
                                                   args.tracer,
                                                   cap,
                                                   omega_m=args.omega_m,
                                                   resume=args.resume,
                                                   overwrite=args.overwrite)
            if not args.quiet:
                print(f'[final] {published["fits"]}', flush=True)
        del case
    return 0


def main(argv=None):
    return run_mock_cli(argv=argv)


if __name__ == '__main__':
    raise SystemExit(main())
