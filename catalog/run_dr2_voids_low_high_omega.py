import argparse, json, os, sys, time

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from group_finder.astra import build_cosmology
from group_finder.read_data import (DEFAULT_COLUMNS,
                                    DEFAULT_RA_MAX,
                                    DEFAULT_RA_MIN,
                                    load_all_tracer_samples)

from run_dr2_voids_three_cosmologies import (DEFAULT_DATA_DIR,
                                             DEFAULT_H,
                                             DEFAULT_OUTPUT_ROOT,
                                             log_message,
                                             normalize_tracer,
                                             run_case)


COSMOLOGIES = (('low_omega', 0.301),
               ('high_omega', 0.329))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('tracer', type=normalize_tracer)
    parser.add_argument('--data-dir', default=DEFAULT_DATA_DIR)
    parser.add_argument('--output-root', default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument('--log-dir', default=None)
    parser.add_argument('--caps', nargs='+', default=['NGC', 'SGC'], choices=['NGC', 'SGC'])
    parser.add_argument('--random-index', type=int, default=0)
    parser.add_argument('--seed', type=int, default=12345)
    parser.add_argument('--h', type=float, default=DEFAULT_H)
    parser.add_argument('--ra-min', type=float, default=DEFAULT_RA_MIN)
    parser.add_argument('--ra-max', type=float, default=DEFAULT_RA_MAX)
    parser.add_argument('--r-threshold', type=float, default=-0.25)
    parser.add_argument('--seed-threshold', type=float, default=-0.85)
    parser.add_argument('--merge-threshold', type=float, default=-0.80)
    parser.add_argument('--min-group-size', type=int, default=4)
    parser.add_argument('--min-rand-for-shape', type=int, default=3)
    parser.add_argument('--healpix-edge-nside', type=int, default=256)
    parser.add_argument('--healpix-edge-min-randoms', type=int, default=3)
    parser.add_argument('--mode', choices=['underdense', 'overdense'], default='underdense')
    parser.add_argument('--include-membership', action='store_true')
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--quiet', action='store_true')
    return parser.parse_args()


def output_path_for(output_root, cosmo_label, tracer, cap):
    return os.path.join(output_root, cosmo_label, f'voids_{tracer}_{cap}.fits')


def main():
    args = parse_args()
    verbose = not args.quiet
    output_root = os.path.abspath(os.path.expanduser(args.output_root))

    planned = []
    for cosmo_label, _omega_m in COSMOLOGIES:
        for cap in args.caps:
            planned.append(output_path_for(output_root, cosmo_label,
                                           args.tracer, cap))

    if args.dry_run:
        print('Planned output FITS files:')
        for path in planned:
            print(path)
        return

    log_dir = args.log_dir or os.path.join(output_root, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir,
                            f'run_dr2_voids_low_high_{args.tracer}_rand{args.random_index:02d}_'
                            f'{time.strftime("%Y%m%d_%H%M%S", time.gmtime())}.log')

    with open(log_path, 'a', encoding='utf-8') as log_fh:
        t0 = time.time()
        log_message(log_fh, f'run start log_file={log_path}', verbose=verbose)
        log_message(log_fh, f'config={json.dumps(vars(args), sort_keys=True)}',
                    verbose=verbose)
        log_message(log_fh, f'cosmologies={json.dumps(COSMOLOGIES)}',
                    verbose=verbose)

        step = time.time()
        all_data = load_all_tracer_samples(data_dir=args.data_dir,
                                           tracers=[args.tracer],
                                           random_index=args.random_index,
                                           columns=DEFAULT_COLUMNS,
                                           ra_min=args.ra_min,
                                           ra_max=args.ra_max,
                                           seed=args.seed,
                                           caps=args.caps,
                                           release='dr2',
                                           tracer_aliases=None,
                                           mask_dir=None,
                                           verbose=verbose)
        log_message(log_fh, f'loaded tracer={args.tracer} '
                            f'elapsed_s={time.time() - step:.3f}',
                    verbose=verbose)

        outputs = []
        for cosmo_label, omega_m in COSMOLOGIES:
            cosmo = build_cosmology(h=args.h, omega_m=omega_m)
            for cap in args.caps:
                key = f'{args.tracer}_{cap}'
                rand_key = f'{args.tracer}_RAND_{cap}'
                output_path = output_path_for(output_root, cosmo_label,
                                              args.tracer, cap)
                outputs.append(run_case(data_table=all_data[key],
                                        rand_table=all_data[rand_key],
                                        tracer=args.tracer,
                                        cap=cap,
                                        cosmo_label=cosmo_label,
                                        omega_m=omega_m,
                                        cosmo=cosmo,
                                        output_path=output_path,
                                        args=args,
                                        log_fh=log_fh,
                                        verbose=verbose))

        log_message(log_fh, f'run complete elapsed_s={time.time() - t0:.3f}',
                    verbose=verbose)
        log_message(log_fh, 'outputs=' + json.dumps(outputs, indent=2),
                    verbose=verbose)


if __name__ == '__main__':
    main()
