import argparse, json, os, sys, time

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np

from group_finder.astra import build_cosmology
from run_fiber_assignment_voids import (DEFAULT_H,
                                        DEFAULT_MOCK_DIR,
                                        DEFAULT_OMEGA_M,
                                        DEFAULT_RA_MAX,
                                        DEFAULT_RA_MIN,
                                        log_message,
                                        mock_path,
                                        normalize_tracer,
                                        output_path,
                                        random_path,
                                        read_mock_table,
                                        run_region,
                                        split_regions,
                                        subsample_randoms)


DEFAULT_OUTPUT_DIR = '/pscratch/sd/v/vtorresg/void_catalog/complete_targets'
MOCK_KIND = 'complete'


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('tracer', type=normalize_tracer)
    parser.add_argument('--mock-dir', default=DEFAULT_MOCK_DIR)
    parser.add_argument('--output-dir', default=DEFAULT_OUTPUT_DIR)
    parser.add_argument('--log-dir', default=None)
    parser.add_argument('--split-caps', dest='split_caps', action='store_true', default=True)
    parser.add_argument('--no-split-caps', dest='split_caps', action='store_false')
    parser.add_argument('--caps', nargs='+', default=['NGC', 'SGC'], choices=['NGC', 'SGC'])
    parser.add_argument('--ra-min', type=float, default=DEFAULT_RA_MIN)
    parser.add_argument('--ra-max', type=float, default=DEFAULT_RA_MAX)
    parser.add_argument('--z-min', type=float, default=None)
    parser.add_argument('--z-max', type=float, default=None)
    parser.add_argument('--bgs-mr-limit', type=float, default=None)
    parser.add_argument('--h', type=float, default=DEFAULT_H)
    parser.add_argument('--omega-m', type=float, default=DEFAULT_OMEGA_M)
    parser.add_argument('--seed', type=int, default=12345)
    parser.add_argument('--random-factor', type=float, default=1.0)
    parser.add_argument('--r-threshold', type=float, default=-0.25)
    parser.add_argument('--seed-threshold', type=float, default=-0.85)
    parser.add_argument('--min-group-size', type=int, default=4)
    parser.add_argument('--min-rand-for-shape', type=int, default=3)
    parser.add_argument('--healpix-edge-nside', type=int, default=256)
    parser.add_argument('--healpix-edge-min-randoms', type=int, default=3)
    parser.add_argument('--healpix-edge-min-data-ngc', type=int, default=3)
    parser.add_argument('--healpix-edge-min-data-sgc', type=int, default=4)
    parser.add_argument('--disable-healpix-edge-data-cut', action='store_true',
                        default=False)
    parser.add_argument('--mode', choices=['underdense', 'overdense'], default='underdense')
    parser.add_argument('--include-membership', action='store_true')
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--quiet', action='store_true')
    return parser.parse_args()


def main():
    args = parse_args()
    args.mock_kind = MOCK_KIND
    verbose = not args.quiet
    args.mock_dir = os.path.abspath(os.path.expanduser(args.mock_dir))
    args.output_dir = os.path.abspath(os.path.expanduser(args.output_dir))

    data_path = mock_path(args.mock_dir, args.tracer, MOCK_KIND)
    randoms_path = random_path(args.mock_dir, args.tracer)
    regions = args.caps if args.split_caps else ['ALL']
    outputs = [output_path(args.output_dir, args.tracer, MOCK_KIND, region)
               for region in regions]

    if args.dry_run:
        print(f'Input data:    {data_path}')
        print(f'Input randoms: {randoms_path}')
        print('Planned output FITS files:')
        for path in outputs:
            print(path)
        return

    log_dir = args.log_dir or os.path.join(args.output_dir, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir,
                            f'run_complete_targets_{args.tracer}_'
                            f'{time.strftime("%Y%m%d_%H%M%S", time.gmtime())}.log')

    with open(log_path, 'a', encoding='utf-8') as log_fh:
        t0 = time.time()
        log_message(log_fh, f'run start log_file={log_path}', verbose=verbose)
        log_message(log_fh, f'config={json.dumps(vars(args), sort_keys=True)}',
                    verbose=verbose)

        if not os.path.exists(data_path):
            raise FileNotFoundError(data_path)
        if not os.path.exists(randoms_path):
            raise FileNotFoundError(randoms_path)

        step = time.time()
        data = read_mock_table(data_path, args, is_random=False)
        randoms = read_mock_table(randoms_path, args, is_random=True)
        log_message(log_fh, f'loaded inputs elapsed_s={time.time() - step:.3f} '
                            f'n_data={len(data)} n_random_available={len(randoms)}',
                    verbose=verbose)

        data_regions = split_regions(data, args)
        random_regions = split_regions(randoms, args)
        rng = np.random.default_rng(args.seed)
        cosmo = build_cosmology(h=args.h, omega_m=args.omega_m)

        written = []
        for region, out_path in zip(regions, outputs):
            region_seed = int(rng.integers(0, np.iinfo(np.int32).max))
            region_rng = np.random.default_rng(region_seed)
            rand_sub = subsample_randoms(random_regions[region],
                                         n_data=len(data_regions[region]),
                                         factor=args.random_factor,
                                         rng=region_rng)
            log_message(log_fh, f'region={region} data={len(data_regions[region])} '
                                f'random_available={len(random_regions[region])} '
                                f'random_used={len(rand_sub)} seed={region_seed}',
                        verbose=verbose)
            written.append(run_region(data_table=data_regions[region],
                                      random_table=rand_sub,
                                      tracer=args.tracer,
                                      mock_kind=MOCK_KIND,
                                      region=region,
                                      data_path=data_path,
                                      randoms_path=randoms_path,
                                      output=out_path,
                                      cosmo=cosmo,
                                      args=args,
                                      log_fh=log_fh,
                                      verbose=verbose))

        log_message(log_fh, f'run complete elapsed_s={time.time() - t0:.3f}',
                    verbose=verbose)
        log_message(log_fh, 'outputs=' + json.dumps(written, indent=2),
                    verbose=verbose)


if __name__ == '__main__':
    main()
