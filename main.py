import argparse
import json, os, time

from group_finder.astra import (add_cartesian_to_all,
                          add_neighbor_columns_to_tables,
                          build_cosmology,
                          compute_neighbor_statistics)
from group_finder.make_cat import (build_point_membership_table,
                             consolidate_group_info,
                             write_group_table_fits)
from group_finder.read_data import (DEFAULT_CAPS_DR1,
                              DEFAULT_COLUMNS,
                              DEFAULT_DR1_MASK_DIR,
                              DEFAULT_RA_MAX,
                              DEFAULT_RA_MIN,
                              DEFAULT_TRACER_ALIASES_DR1,
                              DEFAULT_TRACERS_ALL,
                              DEFAULT_TRACERS_DR1,
                              DEFAULT_TRACERS_DR2,
                              load_all_tracer_samples,
                              validate_required_keys)
from group_finder.watershed import assign_group_ids_to_tables, run_watershed


DEFAULT_DATA_DIR_DR2 = '/global/cfs/cdirs/desi/survey/catalogs/DA2/LSS/loa-v1/LSScats/v1.1/nonKP/'
DEFAULT_DATA_DIR_DR1 = '/global/cfs/cdirs/desi/public/dr1/vac/dr1/lss/guadalupe/v1.0/LSScats/clustering'
DEFAULT_OUTPUT_DIR = '/pscratch/sd/v/vtorresg/cosmic-web/dr1/void-cat'
DEFAULT_LOG_DIR = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'logs')
DEFAULT_CAPS_BY_RELEASE = {'dr2': ['NGC', 'SGC'],
                           'dr1': list(DEFAULT_CAPS_DR1)}


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--release', choices=['dr2', 'dr1'], default='dr1')
    parser.add_argument('--data-dir', default=None)
    parser.add_argument('--output-dir', default=DEFAULT_OUTPUT_DIR)
    parser.add_argument('--log-dir', default=DEFAULT_LOG_DIR)
    parser.add_argument('--mask-dir', default=None)

    parser.add_argument('--tracers', nargs='+', default=None, choices=list(DEFAULT_TRACERS_ALL))
    parser.add_argument('--caps', nargs='+', default=None, choices=['NGC', 'SGC'])

    parser.add_argument('--random-index', type=int, default=0)
    parser.add_argument('--seed', type=int, default=12345)
    parser.add_argument('--ra-min', type=float, default=DEFAULT_RA_MIN)
    parser.add_argument('--ra-max', type=float, default=DEFAULT_RA_MAX)

    parser.add_argument('--h', type=float, default=0.6736)
    parser.add_argument('--omega-m', type=float, default=0.315)

    parser.add_argument('--r-threshold', type=float, default=-0.25)
    parser.add_argument('--seed-threshold', type=float, default=None)
    parser.add_argument('--merge-threshold', type=float, default=None)
    parser.add_argument('--min-group-size', type=int, default=4)
    parser.add_argument('--mode', choices=['underdense', 'overdense'], default='underdense')
    parser.add_argument('--edge-radial-buffer', type=float, default=20.0)
    parser.add_argument('--edge-angular-buffer', type=float, default=1.0)
    parser.add_argument('--edge-cartesian-buffer', type=float, default=None)
    parser.add_argument('--healpix-edge-nside', type=int, default=256)
    parser.add_argument('--healpix-edge-min-randoms', type=int, default=3)

    parser.add_argument('--include-membership', dest='include_membership', action='store_true', default=True)
    parser.add_argument('--no-membership', dest='include_membership',
                        action='store_false')
    parser.add_argument('--overwrite', action='store_true', default=False)
    parser.add_argument('--quiet', action='store_true', default=False)

    return parser.parse_args()


def _utc_timestamp():
    return time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime())


def _log(log_fh, message, verbose):
    log_fh.write(f'[{_utc_timestamp()}] {message}\n')
    log_fh.flush()
    if verbose:
        print(message)


def _resolve_release_output_dir(base_output_dir, release):
    norm_base = os.path.normpath(base_output_dir)
    if os.path.basename(norm_base) == release:
        return base_output_dir
    return os.path.join(base_output_dir, release)


def _configure_release_args(args):
    if args.data_dir is None:
        if args.release == 'dr1':
            args.data_dir = DEFAULT_DATA_DIR_DR1
        else:
            args.data_dir = DEFAULT_DATA_DIR_DR2

    if args.tracers is None:
        if args.release == 'dr1':
            args.tracers = list(DEFAULT_TRACERS_DR1)
        else:
            args.tracers = list(DEFAULT_TRACERS_DR2)

    if args.caps is None:
        args.caps = list(DEFAULT_CAPS_BY_RELEASE[args.release])

    args.output_dir = _resolve_release_output_dir(args.output_dir, args.release)

    if args.release == 'dr1':
        valid_tracers = tuple(dict.fromkeys(DEFAULT_TRACERS_DR1 +
                                            tuple(DEFAULT_TRACER_ALIASES_DR1)))
        invalid_tracers = [tracer for tracer in args.tracers if tracer not in valid_tracers]
        if invalid_tracers:
            raise ValueError(f'Invalid tracers for DR1: {invalid_tracers}. Expected subset of {list(valid_tracers)}')

        invalid_caps = [cap for cap in args.caps if cap not in DEFAULT_CAPS_BY_RELEASE['dr1']]
        if invalid_caps:
            raise ValueError('Invalid caps for DR1: '
                             f'{invalid_caps}. Expected subset of {DEFAULT_CAPS_BY_RELEASE["dr1"]}')
        args.cap_cuts = None
        args.tracer_aliases = dict(DEFAULT_TRACER_ALIASES_DR1)
        if args.mask_dir is None:
            args.mask_dir = os.environ.get('ASTRA_DR1_MASK_DIR', DEFAULT_DR1_MASK_DIR)
    else:
        invalid_tracers = [tracer for tracer in args.tracers if tracer not in DEFAULT_TRACERS_DR2]
        if invalid_tracers:
            raise ValueError('Invalid tracers for DR2: '
                             f'{invalid_tracers}. Expected subset of {list(DEFAULT_TRACERS_DR2)}')

        invalid_caps = [cap for cap in args.caps if cap not in DEFAULT_CAPS_BY_RELEASE['dr2']]
        if invalid_caps:
            raise ValueError('Invalid caps for DR2: '
                             f'{invalid_caps}. Expected subset of {DEFAULT_CAPS_BY_RELEASE["dr2"]}')
        args.cap_cuts = None
        args.tracer_aliases = None
        args.mask_dir = None


def run_pipeline(args):
    verbose = not args.quiet

    if args.ra_min >= args.ra_max:
        raise ValueError(f'RA bounds must satisfy ra_min < ra_max, got {args.ra_min} >= {args.ra_max}')
    if args.min_group_size < 1:
        raise ValueError(f'min_group_size must be >= 1, got {args.min_group_size}')
    if args.edge_radial_buffer < 0:
        raise ValueError(f'edge_radial_buffer must be >= 0, got {args.edge_radial_buffer}')
    if args.edge_angular_buffer is not None and args.edge_angular_buffer < 0:
        args.edge_angular_buffer = None
    if args.edge_cartesian_buffer is not None and args.edge_cartesian_buffer < 0:
        raise ValueError(f'edge_cartesian_buffer must be >= 0, got {args.edge_cartesian_buffer}')

    os.makedirs(args.log_dir, exist_ok=True)
    run_tag = time.strftime('%Y%m%d_%H%M%S', time.gmtime())
    log_path = os.path.join(args.log_dir, f'pipeline_{run_tag}.log')
    os.makedirs(args.output_dir, exist_ok=True)

    with open(log_path, 'a', encoding='utf-8') as log_fh:
        t0 = time.time()
        _log(log_fh, f'Run start. log_file={log_path}', verbose=verbose)
        _log(log_fh, f'Run config: {json.dumps(vars(args), sort_keys=True)}', verbose=verbose)

        t_step = time.time()
        _log(log_fh, 'Step=load_catalogs start', verbose=verbose)
        all_data = load_all_tracer_samples(data_dir=args.data_dir,
                                           tracers=args.tracers,
                                           random_index=args.random_index,
                                           columns=DEFAULT_COLUMNS,
                                           ra_min=args.ra_min,
                                           ra_max=args.ra_max,
                                           caps=args.caps,
                                           cap_cuts=args.cap_cuts,
                                           release=args.release,
                                           tracer_aliases=args.tracer_aliases,
                                           mask_dir=args.mask_dir,
                                           seed=args.seed,
                                           verbose=verbose)
        validate_required_keys(all_data, tracers=args.tracers, caps=args.caps)
        _log(log_fh, f'Step=load_catalogs done elapsed_s={time.time() - t_step:.3f}', verbose=verbose)

        t_step = time.time()
        _log(log_fh, 'Step=cartesian_coordinates start', verbose=verbose)
        cosmo = build_cosmology(h=args.h, omega_m=args.omega_m)
        add_cartesian_to_all(all_data, cosmo=cosmo, h=args.h)
        _log(log_fh, f'Step=cartesian_coordinates done elapsed_s={time.time() - t_step:.3f}', verbose=verbose)

        for tracer in args.tracers:
            t_tracer = time.time()
            _log(log_fh, f'Tracer={tracer} start', verbose=verbose)

            for cap in args.caps:
                key = f'{tracer}_{cap}'
                rand_key = f'{tracer}_RAND_{cap}'
                output_path = os.path.join(args.output_dir, f'voids_{tracer}_{cap}.fits')

                if os.path.exists(output_path) and not args.overwrite:
                    _log(log_fh, f'Case={key} skipped output_exists path={output_path}', verbose=verbose)
                    continue

                t_case = time.time()
                _log(log_fh, f'Case={key} start', verbose=verbose)

                data_tbl = all_data[key]
                rand_tbl = all_data[rand_key]

                t_step = time.time()
                stats = compute_neighbor_statistics(data_tbl, rand_tbl)
                add_neighbor_columns_to_tables(data_tbl, rand_tbl, stats)
                _log(log_fh, f'Case={key} Step=neighbor_stats done elapsed_s={time.time() - t_step:.3f}',
                     verbose=verbose)

                rvals = stats['r_values']
                _log(log_fh, f'Case={key} R_stats n={len(rvals)} min={float(rvals.min()):.3f} '
                             f'max={float(rvals.max()):.3f}',
                     verbose=verbose)

                t_step = time.time()
                ws = run_watershed(neighbors=stats['neighbors'],
                                   r_values=stats['r_values'],
                                   r_threshold=args.r_threshold,
                                   min_group_size=args.min_group_size,
                                   mode=args.mode,
                                   seed_threshold=args.seed_threshold,
                                   merge_threshold=args.merge_threshold)
                assign_group_ids_to_tables(data_tbl, rand_tbl, ws['group_of'], group_col='GROUPID')
                _log(log_fh, f'Case={key} Step=watershed done elapsed_s={time.time() - t_step:.3f} '
                             f'groups={ws["n_groups"]} assigned={ws["n_assigned"]} '
                             f'boundary={ws["n_boundary_nodes"]} unassigned={ws["n_unassigned"]} '
                             f'total_nodes={len(ws["group_of"])}',
                     verbose=verbose)

                t_step = time.time()
                group_table = consolidate_group_info(data_table=data_tbl,
                                                     rand_table=rand_tbl,
                                                     cosmo=cosmo, h=args.h,
                                                     group_col='GROUPID',
                                                     min_rand_for_shape=3,
                                                     edge_radial_buffer=args.edge_radial_buffer,
                                                     edge_angular_buffer_deg=args.edge_angular_buffer,
                                                     edge_cartesian_buffer=args.edge_cartesian_buffer,
                                                     healpix_edge_nside=args.healpix_edge_nside,
                                                     healpix_edge_min_randoms=args.healpix_edge_min_randoms)
                if 'FOOTPRINT_EDGE' in group_table.colnames:
                    n_edge = int(sum(group_table['FOOTPRINT_EDGE']))
                    edge_msg = f' footprint_edge={n_edge} footprint_clean={len(group_table) - n_edge}'
                else:
                    edge_msg = ''
                _log(log_fh, f'Case={key} Step=consolidate_groups done elapsed_s={time.time() - t_step:.3f} '
                             f'n_voids={len(group_table)}{edge_msg}',
                     verbose=verbose)

                point_table = None
                if args.include_membership:
                    t_step = time.time()
                    point_table = build_point_membership_table(data_table=data_tbl,
                                                               rand_table=rand_tbl,
                                                               group_col='GROUPID')
                    _log(log_fh, f'Case={key} Step=build_point_membership done elapsed_s={time.time() - t_step:.3f} '
                                 f'n_points={len(point_table)}',
                         verbose=verbose)
                else:
                    _log(log_fh, f'Case={key} Step=build_point_membership skipped', verbose=verbose)

                t_step = time.time()
                write_group_table_fits(group_table=group_table,
                                       output_path=output_path,
                                       tracer=tracer,
                                       cap=cap, h=args.h,
                                       omega_m=args.omega_m,
                                       r_threshold=args.r_threshold,
                                       mode=args.mode,
                                       point_table=point_table,
                                       seed_threshold=args.seed_threshold,
                                       merge_threshold=args.merge_threshold,
                                       boundary_id=ws['boundary_id'],
                                       watershed_stats=ws,
                                       overwrite=args.overwrite)
                _log(log_fh, f'Case={key} Step=write_fits done elapsed_s={time.time() - t_step:.3f} '
                             f'output={output_path}',
                     verbose=verbose)

                _log(log_fh, f'Case={key} done elapsed_s={time.time() - t_case:.3f}', verbose=verbose)

            _log(log_fh, f'Tracer={tracer} done elapsed_s={time.time() - t_tracer:.3f}', verbose=verbose)

        elapsed = time.time() - t0
        _log(log_fh, f'Run completed elapsed_s={elapsed:.3f}', verbose=verbose)
        _log(log_fh, f'Run end. log_file={log_path}', verbose=verbose)


def main():
    args = parse_args()
    _configure_release_args(args)
    run_pipeline(args)


if __name__ == '__main__':
    main()