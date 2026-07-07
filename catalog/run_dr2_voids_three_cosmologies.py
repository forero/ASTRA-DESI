import argparse, json, os, sys, time

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('NUMEXPR_NUM_THREADS', '1')

import numpy as np
from astropy.io import fits
from astropy.table import Table

from group_finder.astra import (add_cartesian_columns,
                                build_cosmology,
                                compute_neighbor_statistics,
                                add_neighbor_columns_to_tables)
from group_finder.make_cat import (AXIS_VECTOR_COLUMNS,
                                   build_point_membership_table,
                                   consolidate_group_info,
                                   ellipticity_from_axes,
                                   ELLIPTICITY_DEFINITION,
                                   J1J3_DEFINITION)
from group_finder.read_data import (DEFAULT_COLUMNS,
                                    DEFAULT_RA_MAX,
                                    DEFAULT_RA_MIN,
                                    DEFAULT_TRACERS_DR2,
                                    load_all_tracer_samples)
from group_finder.watershed import BOUNDARY_ID, assign_group_ids_to_tables, run_watershed


DEFAULT_DATA_DIR = ('/global/cfs/cdirs/desi/survey/catalogs/DA2/LSS/loa-v1/LSScats/v1.1/nonKP/')
DEFAULT_OUTPUT_ROOT = '/pscratch/sd/v/vtorresg/void_catalog'
DEFAULT_H = 0.6736
COSMOLOGIES = (('DR2_Om_1_Om0p301_h0p6736', 0.301),
               ('DR2_Om_2_Om0p315_h0p6736', 0.315),
               ('DR2_Om_3_Om0p329_h0p6736', 0.329))


def utc_timestamp():
    return time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime())


def log_message(log_fh, message, verbose=True):
    line = f'[{utc_timestamp()}] {message}'
    log_fh.write(line + '\n')
    log_fh.flush()
    if verbose:
        print(message, flush=True)


def normalize_tracer(value):
    aliases = {t.upper(): t for t in DEFAULT_TRACERS_DR2}
    aliases['BGS'] = 'BGS_ANY'
    aliases['BGS_BRIGHT'] = 'BGS_ANY'
    aliases['ELGNOTQSO'] = 'ELGnotqso'
    aliases['ELG_NOTQSO'] = 'ELGnotqso'
    aliases['ELG'] = 'ELGnotqso'
    key = str(value).strip().upper()
    if key not in aliases:
        allowed = ', '.join(DEFAULT_TRACERS_DR2)
        raise argparse.ArgumentTypeError(f'Invalid DR2 tracer {value!r}. Expected one of: {allowed}')
    return aliases[key]


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
    parser.add_argument('--seed-threshold', type=float, default=None)
    parser.add_argument('--merge-threshold', type=float, default=None)
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


def common_void_table(group_table, filter_edge=False, filter_footprint_edge=None):
    if filter_footprint_edge is None:
        filter_footprint_edge = bool(filter_edge)

    source = group_table
    if filter_footprint_edge:
        if 'FOOTPRINT_EDGE' in group_table.colnames:
            footprint_edge = np.asarray(group_table['FOOTPRINT_EDGE'], dtype=bool)
            source = group_table[~footprint_edge]
        elif 'EDGE' in group_table.colnames:
            footprint_edge = np.asarray(group_table['EDGE'], dtype=bool)
            source = group_table[~footprint_edge]

    out = Table()
    out['VOID_ID'] = np.asarray(source['VOID_ID'], dtype=np.int32)
    out['RA'] = np.asarray(source['RA'], dtype=np.float64)
    out['DEC'] = np.asarray(source['DEC'], dtype=np.float64)
    out['REDSHIFT'] = np.asarray(source['REDSHIFT'], dtype=np.float64)
    out['R_EFF'] = np.asarray(source['R_EFF'], dtype=np.float64)
    out['ELLIP'] = ellipticity_from_axes(source)

    extra_cols = ('N_DATA_IN_GROUP', 'N_RAND_IN_GROUP',
                  'LAMBDA_1', 'LAMBDA_2', 'LAMBDA_3',
                  'GEOM_BAD',
                  'EDGE', 'FOOTPRINT_EDGE',
                  'TOUCHES_RADIAL_EDGE', 'CENTER_NEAR_RADIAL_EDGE',
                  'TOUCHES_RA_EDGE', 'TOUCHES_DEC_EDGE',
                  'TOUCHES_HEALPIX_EDGE', 'CENTER_NEAR_HEALPIX_EDGE',
                  'TOUCHES_CART_EDGE', 'CENTER_NEAR_CART_EDGE',
                  'X', 'Y', 'Z',
                  'SEMI_AXIS_A', 'SEMI_AXIS_B', 'SEMI_AXIS_C',
                  *AXIS_VECTOR_COLUMNS)
    for col in extra_cols:
        if col in source.colnames:
            out[col] = np.asarray(source[col])
    if 'FOOTPRINT_EDGE' not in out.colnames:
        if 'EDGE' in source.colnames:
            out['FOOTPRINT_EDGE'] = np.asarray(source['EDGE'], dtype=np.bool_)
        else:
            out['FOOTPRINT_EDGE'] = np.zeros(len(out), dtype=np.bool_)
    if 'EDGE' in out.colnames:
        out['EDGE'] = np.zeros(len(out), dtype=np.bool_)
    else:
        out['EDGE'] = np.zeros(len(out), dtype=np.bool_)
    return out


def write_common_void_fits(group_table, output_path, tracer, cap, cosmo_label,
                           omega_m, args, point_table=None):
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    if 'FOOTPRINT_EDGE' in group_table.colnames:
        footprint_edge_flags = np.asarray(group_table['FOOTPRINT_EDGE'], dtype=bool)
        n_footprint_edge = int(np.count_nonzero(footprint_edge_flags))
        n_footprint_clean = int(np.count_nonzero(~footprint_edge_flags))
    elif 'EDGE' in group_table.colnames:
        footprint_edge_flags = np.asarray(group_table['EDGE'], dtype=bool)
        n_footprint_edge = int(np.count_nonzero(footprint_edge_flags))
        n_footprint_clean = int(np.count_nonzero(~footprint_edge_flags))
    else:
        n_footprint_edge = 0
        n_footprint_clean = len(group_table)
    voids = common_void_table(group_table)
    n_edge = int(np.count_nonzero(np.asarray(voids['EDGE'], dtype=bool)))

    primary = fits.PrimaryHDU()
    hdr = primary.header
    hdr['RELEASE'] = ('DR2', 'DESI data release')
    hdr['TRACER'] = (tracer, 'Tracer type')
    hdr['CAP'] = (cap, 'Sky cap')
    hdr['COSMO'] = (cosmo_label, 'Cosmology label')
    hdr['H'] = (float(args.h), 'h = H0 / 100')
    hdr['OMEGA_M'] = (float(omega_m), 'Matter density parameter')
    hdr['RANDIDX'] = (int(args.random_index), 'DR2 random index')
    hdr['SEED'] = (int(args.seed), 'Random subsampling seed')
    hdr['RTHRESH'] = (float(args.r_threshold), 'Watershed R threshold')
    if args.seed_threshold is not None:
        hdr['SEEDTHR'] = (float(args.seed_threshold), 'Watershed seed threshold')
    if getattr(args, 'merge_threshold', None) is not None:
        hdr['MERGETHR'] = (float(args.merge_threshold), 'Watershed saddle merge threshold')
    hdr['MINGRP'] = (int(args.min_group_size), 'Minimum watershed group size')
    hdr['MINRSHAP'] = (int(args.min_rand_for_shape), 'Min randoms for axes')
    hdr['WMODE'] = (args.mode, 'Watershed mode')
    hdr['NVOIDS'] = (len(voids), 'Number of GROUPID>=0 voids written')
    if 'GEOM_BAD' in voids.colnames:
        geom_flags = np.asarray(voids['GEOM_BAD'], dtype=bool)
        hdr['NGEOMBAD'] = (int(np.count_nonzero(geom_flags)), 'Number of GEOM_BAD=True voids')
    hdr['UNITSXYZ'] = ('Mpc/h', 'Units for R_EFF, X/Y/Z, semi-axes')
    hdr['REFFDEF'] = ('sqrt(5*<|r-r_cm|^2>/3)', 'R_EFF definition')
    hdr['LAMDEF'] = ('eig(<dx_i dx_j>)', 'LAMBDA_1..3 definition')
    hdr['AXDEF'] = ('SEMI_AXIS_j=sqrt(5*LAMBDA_j)', 'Semi-axis definition')
    hdr['UNITSAX'] = ('unitless', 'Units for X1..Z3 axis-vector columns')
    hdr['AXVEC'] = ('Xj,Yj,Zj', 'Unit-vector components for axis j')
    hdr['UNITSANG'] = ('deg', 'Units for RA and DEC')
    hdr['ZUNIT'] = ('redshift', 'Units for REDSHIFT')
    hdr['ELLIPDEF'] = (ELLIPTICITY_DEFINITION, 'Ellipticity definition')
    hdr['J1J3'] = (J1J3_DEFINITION, 'Stored second-moment axis values')
    hdr['GEOMDEF'] = ('1-C/A>0.9', 'GEOM_BAD definition')
    hdr['EDGEDEF'] = ('GROUPID==boundary_id', 'EDGE=True means watershed boundary')
    hdr['FPEDDEF'] = ('survey footprint/mask boundary', 'FOOTPRINT_EDGE definition')
    hdr['NEDGE'] = (n_edge, 'EDGE=True rows in VOIDS')
    hdr['NFPEDGE'] = (n_footprint_edge, 'FOOTPRINT_EDGE=True voids written')
    hdr['NFPCLN'] = (n_footprint_clean, 'FOOTPRINT_EDGE=False voids written')
    hdr['GIDM1'] = (-1, 'GROUPID=-1 means unassigned point')
    hdr['GIDM2'] = (int(BOUNDARY_ID), 'GROUPID for watershed boundary point')
    if 'SURVEY_VOL' in group_table.meta and np.isfinite(group_table.meta['SURVEY_VOL']):
        hdr['SURVVOL'] = (float(group_table.meta['SURVEY_VOL']), 'Survey volume in (Mpc/h)^3')
    if 'SURVEY_OMG' in group_table.meta and np.isfinite(group_table.meta['SURVEY_OMG']):
        hdr['SURVOMG'] = (float(group_table.meta['SURVEY_OMG']), 'Survey solid angle in sr')
    if 'RAND_DENS' in group_table.meta and np.isfinite(group_table.meta['RAND_DENS']):
        hdr['RANDDENS'] = (float(group_table.meta['RAND_DENS']), 'Mean random density h^3/Mpc^3')
    if 'NRAND_DENS' in group_table.meta:
        hdr['NRANDDEN'] = (int(group_table.meta['NRAND_DENS']), 'Randoms used for mean density')
    if 'HPX_EDGE' in group_table.meta:
        hdr['HPXEDGE'] = (bool(group_table.meta['HPX_EDGE']), 'HEALPix angular edge enabled')
    if 'HPX_NSIDE' in group_table.meta:
        hdr['HPXNSIDE'] = (int(group_table.meta['HPX_NSIDE']), 'HEALPix edge NSIDE')
    if 'HPX_NEST' in group_table.meta:
        hdr['HPXNEST'] = (bool(group_table.meta['HPX_NEST']), 'HEALPix NESTED ordering')
    if 'HPX_MINR' in group_table.meta:
        hdr['HPXMINR'] = (int(group_table.meta['HPX_MINR']), 'Min randoms per HEALPix pixel')
    if 'HPX_EBUF' in group_table.meta:
        hdr['HPXEBUF'] = (float(group_table.meta['HPX_EBUF']), 'HEALPix edge buffer in deg')
    if 'HPX_NOBS' in group_table.meta:
        hdr['HPXNOBS'] = (int(group_table.meta['HPX_NOBS']), 'Observed HEALPix pixels')
    if 'HPX_NEDGE' in group_table.meta:
        hdr['HPXNEDG'] = (int(group_table.meta['HPX_NEDGE']), 'Angular edge HEALPix pixels')
    if 'HPX_NBUF' in group_table.meta:
        hdr['HPXNBUF'] = (int(group_table.meta['HPX_NBUF']), 'Buffered angular edge HEALPix pixels')
    if point_table is not None:
        hdr['NPOINTS'] = (len(point_table), 'Rows in POINT_MEMBERSHIP')
        point_gids = np.asarray(point_table['GROUPID'], dtype=np.int32)
        hdr['NPTASGN'] = (int(np.count_nonzero(point_gids >= 0)), 'Assigned points')
        hdr['NPTUNASN'] = (int(np.count_nonzero(point_gids == -1)), 'Unassigned points')
        hdr['NPTBND'] = (int(np.count_nonzero(point_gids == int(BOUNDARY_ID))),
                         'Watershed boundary points')

    hdus = [primary, fits.BinTableHDU(data=voids.as_array(), name='VOIDS')]
    if point_table is not None:
        hdus.append(fits.BinTableHDU(data=point_table.as_array(),
                                     name='POINT_MEMBERSHIP'))

    fits.HDUList(hdus).writeto(output_path, overwrite=args.overwrite)
    return output_path


def read_void_catalog(path):
    with fits.open(path, memmap=True) as hdul:
        if 'VOIDS' in hdul:
            table = Table(hdul['VOIDS'].data)
        else:
            table = Table(hdul[1].data)
    required = ('VOID_ID', 'RA', 'DEC', 'REDSHIFT', 'R_EFF')
    missing = [col for col in required if col not in table.colnames]
    if missing:
        raise KeyError(f'{path} missing required columns: {missing}')
    return table


def run_case(data_table, rand_table, tracer, cap, cosmo_label, omega_m,
             cosmo, output_path, args, log_fh, verbose):
    if os.path.exists(output_path) and not args.overwrite:
        log_message(log_fh, f'skip existing {output_path}', verbose=verbose)
        return output_path

    data_tbl = data_table.copy(copy_data=True)
    rand_tbl = rand_table.copy(copy_data=True)

    t0 = time.time()
    log_message(log_fh, f'case start cosmo={cosmo_label} tracer={tracer} cap={cap} '
                        f'n_data={len(data_tbl)} n_rand={len(rand_tbl)}',
                        verbose=verbose)

    step = time.time()
    add_cartesian_columns(data_tbl, cosmo=cosmo, h=args.h)
    add_cartesian_columns(rand_tbl, cosmo=cosmo, h=args.h)
    log_message(log_fh, f'case={cosmo_label}/{tracer}/{cap} cartesian '
                        f'elapsed_s={time.time() - step:.3f}',
                        verbose=verbose)

    step = time.time()
    stats = compute_neighbor_statistics(data_tbl, rand_tbl)
    add_neighbor_columns_to_tables(data_tbl, rand_tbl, stats)
    rvals = stats['r_values']
    log_message(log_fh, f'case={cosmo_label}/{tracer}/{cap} neighbors '
                        f'elapsed_s={time.time() - step:.3f} '
                        f'n={len(rvals)} min={float(rvals.min()):.3f} '
                        f'max={float(rvals.max()):.3f}',
                        verbose=verbose)

    step = time.time()
    ws = run_watershed(neighbors=stats['neighbors'],
                       r_values=stats['r_values'],
                       r_threshold=args.r_threshold,
                       min_group_size=args.min_group_size,
                       mode=args.mode,
                       seed_threshold=args.seed_threshold,
                       merge_threshold=getattr(args, 'merge_threshold', None))
    assign_group_ids_to_tables(data_tbl, rand_tbl, ws['group_of'],
                               group_col='GROUPID')
    log_message(log_fh, f"case={cosmo_label}/{tracer}/{cap} watershed "
                        f"elapsed_s={time.time() - step:.3f} "
                        f"groups={ws['n_groups']} assigned={ws['n_assigned']} "
                        f"boundary={ws['n_boundary_nodes']} "
                        f"unassigned={ws['n_unassigned']}",
                verbose=verbose)

    step = time.time()
    group_table = consolidate_group_info(data_table=data_tbl,
                                         rand_table=rand_tbl,
                                         cosmo=cosmo,
                                         h=args.h,
                                         group_col='GROUPID',
                                         min_rand_for_shape=args.min_rand_for_shape,
                                         healpix_edge_nside=getattr(args, 'healpix_edge_nside', 256),
                                         healpix_edge_min_randoms=getattr(args, 'healpix_edge_min_randoms', 3))
    log_message(log_fh, f'case={cosmo_label}/{tracer}/{cap} consolidate '
                        f'elapsed_s={time.time() - step:.3f} '
                        f'n_voids={len(group_table)}',
                        verbose=verbose)

    point_table = None
    if args.include_membership:
        step = time.time()
        point_table = build_point_membership_table(data_tbl, rand_tbl,
                                                   group_col='GROUPID')
        log_message(log_fh, f'case={cosmo_label}/{tracer}/{cap} membership '
                            f'elapsed_s={time.time() - step:.3f} '
                            f'n_points={len(point_table)}',
                            verbose=verbose)

    step = time.time()
    write_common_void_fits(group_table=group_table,
                           output_path=output_path,
                           tracer=tracer,
                           cap=cap,
                           cosmo_label=cosmo_label,
                           omega_m=omega_m,
                           args=args,
                           point_table=point_table)
    log_message(log_fh, f'case={cosmo_label}/{tracer}/{cap} write '
                        f'elapsed_s={time.time() - step:.3f} '
                        f'output={output_path}',
                        verbose=verbose)
    log_message(log_fh, f'case done cosmo={cosmo_label} tracer={tracer} cap={cap} '
                        f'elapsed_s={time.time() - t0:.3f}',
                        verbose=verbose)
    return output_path


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
                            f'run_dr2_voids_{args.tracer}_rand{args.random_index:02d}_'
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