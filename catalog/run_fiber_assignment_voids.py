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
try:
    import fitsio
except ImportError:
    fitsio = None

from group_finder.astra import (add_cartesian_columns,
                                add_neighbor_columns_to_tables,
                                build_cosmology,
                                compute_neighbor_statistics)
from group_finder.make_cat import (build_point_membership_table,
                                   consolidate_group_info)
from group_finder.watershed import BOUNDARY_ID, assign_group_ids_to_tables, run_watershed
try:
    from .run_dr2_voids_three_cosmologies import (common_void_table,
                                                  ELLIPTICITY_DEFINITION,
                                                  J1J3_DEFINITION,
                                                  REFF_DEFINITION)
except ImportError:
    from run_dr2_voids_three_cosmologies import (common_void_table,
                                                 ELLIPTICITY_DEFINITION,
                                                 J1J3_DEFINITION,
                                                 REFF_DEFINITION)


DEFAULT_MOCK_DIR = '/pscratch/sd/h/hrincon/LSScats/testfibers'
DEFAULT_OUTPUT_DIR = '/pscratch/sd/v/vtorresg/void_catalog/fiber_assignment'
DEFAULT_H = 0.6736
DEFAULT_OMEGA_M = 0.315
DEFAULT_RA_MIN = 83.0
DEFAULT_RA_MAX = 302.0
BASE_COLUMNS = ('TARGETID', 'RA', 'DEC', 'Z')
TRACERS = ('BGS', 'LRG', 'ELG', 'QSO')


def utc_timestamp():
    return time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime())


def log_message(log_fh, message, verbose=True):
    line = f'[{utc_timestamp()}] {message}'
    log_fh.write(line + '\n')
    log_fh.flush()
    if verbose:
        print(message, flush=True)


def normalize_tracer(value):
    aliases = {'BGS': 'BGS',
               'BGS_ANY': 'BGS',
               'BGS_BRIGHT': 'BGS',
               'LRG': 'LRG',
               'ELG': 'ELG',
               'ELGNOTQSO': 'ELG',
               'ELG_NOTQSO': 'ELG',
               'ELG_LOPNOTQSO': 'ELG',
               'QSO': 'QSO'}
    key = str(value).strip().upper()
    if key not in aliases:
        allowed = ', '.join(TRACERS)
        raise argparse.ArgumentTypeError(f'Invalid tracer {value!r}. Expected one of: {allowed}')
    return aliases[key]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('tracer', type=normalize_tracer)
    parser.add_argument('--mock-dir', default=DEFAULT_MOCK_DIR)
    parser.add_argument('--output-dir', default=DEFAULT_OUTPUT_DIR)
    parser.add_argument('--log-dir', default=None)
    parser.add_argument('--mock-kind', choices=['altmtl', 'complete'], default='altmtl')
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
    parser.add_argument('--merge-threshold', type=float, default=-0.85)
    parser.add_argument('--min-group-size', type=int, default=4)
    parser.add_argument('--min-rand-for-shape', type=int, default=4)
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


def mock_path(mock_dir, tracer, mock_kind):
    return os.path.join(mock_dir, f'{tracer}_{mock_kind}.fits')


def random_path(mock_dir, tracer):
    return os.path.join(mock_dir, f'{tracer}_randoms.fits')


def output_path(output_dir, tracer, mock_kind, region):
    if region == 'ALL':
        return os.path.join(output_dir, f'voids_{tracer}_{mock_kind}.fits')
    return os.path.join(output_dir, f'voids_{tracer}_{mock_kind}_{region}.fits')


def healpix_edge_min_data_for_region(args, tracer, region):
    if getattr(args, 'disable_healpix_edge_data_cut', False):
        return None
    if str(tracer).upper() != 'LRG':
        return None

    region_upper = str(region).upper()
    if region_upper == 'NGC':
        return getattr(args, 'healpix_edge_min_data_ngc', 3)
    if region_upper == 'SGC':
        return getattr(args, 'healpix_edge_min_data_sgc', 4)
    return None


def columns_to_read(path, extra=()):
    with fits.open(path, memmap=True) as hdul:
        names = set(hdul[1].columns.names)
    cols = [col for col in BASE_COLUMNS if col in names]
    missing = [col for col in BASE_COLUMNS if col not in names]
    if missing:
        raise KeyError(f'{path} missing required columns: {missing}')
    for col in extra:
        if col in names and col not in cols:
            cols.append(col)
    return cols


def read_fits_columns(path, columns):
    if fitsio is not None:
        return Table(fitsio.read(path, ext=1, columns=list(columns)))

    with fits.open(path, memmap=True) as hdul:
        data = hdul[1].data
        out = Table()
        for col in columns:
            out[col] = data[col]
        return out


def read_mock_table(path, args, is_random=False):
    extra = ['R_MAG_ABS'] if args.tracer == 'BGS' else []
    cols = columns_to_read(path, extra=extra)
    table = read_fits_columns(path, cols)

    mask = (np.isfinite(np.asarray(table['RA'], dtype=np.float64)) &
            np.isfinite(np.asarray(table['DEC'], dtype=np.float64)) &
            np.isfinite(np.asarray(table['Z'], dtype=np.float64)))
    z = np.asarray(table['Z'], dtype=np.float64)
    if args.z_min is not None:
        mask &= z >= float(args.z_min)
    if args.z_max is not None:
        mask &= z <= float(args.z_max)

    if args.tracer == 'BGS' and args.bgs_mr_limit is not None:
        if 'R_MAG_ABS' not in table.colnames:
            kind = 'random' if is_random else 'data'
            raise KeyError(f'BGS {kind} table {path} has no R_MAG_ABS column')
        mr = np.asarray(table['R_MAG_ABS'], dtype=np.float64)
        mask &= np.isfinite(mr) & (mr <= float(args.bgs_mr_limit))

    return table[mask]


def split_regions(table, args):
    if not args.split_caps:
        return {'ALL': table}

    ra = np.asarray(table['RA'], dtype=np.float64)
    ngc = (ra >= args.ra_min) & (ra <= args.ra_max)
    all_regions = {'NGC': table[ngc], 'SGC': table[~ngc]}
    return {cap: all_regions[cap] for cap in args.caps}


def subsample_randoms(random_table, n_data, factor, rng):
    n_target = int(round(float(factor) * int(n_data)))
    if n_target < 0:
        raise ValueError(f'--random-factor must be non-negative, got {factor}')
    if n_target == 0:
        return random_table[:0].copy()
    if len(random_table) < n_target:
        raise ValueError('Not enough random points after cuts: '
                         f'requested {n_target}, available {len(random_table)}')
    if len(random_table) == n_target:
        return random_table.copy(copy_data=True)
    idx = rng.choice(len(random_table), size=n_target, replace=False)
    return random_table[idx].copy(copy_data=True)


def write_mock_void_fits(group_table, output, tracer, mock_kind, region,
                         data_path, randoms_path, omega_m, args,
                         point_table=None):
    os.makedirs(os.path.dirname(output) or '.', exist_ok=True)
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
    n_footprint_written = int(np.count_nonzero(np.asarray(voids['FOOTPRINT_EDGE'], dtype=bool)))

    primary = fits.PrimaryHDU()
    hdr = primary.header
    hdr['RELEASE'] = ('MOCK', 'Catalog class')
    hdr['MOCKTYPE'] = (mock_kind, 'Input mock type')
    hdr['TRACER'] = (tracer, 'Mock tracer')
    hdr['CAP'] = (region, 'ALL, NGC, or SGC')
    hdr['H'] = (float(args.h), 'h = H0 / 100')
    hdr['OMEGA_M'] = (float(omega_m), 'Matter density parameter')
    hdr['SEED'] = (int(args.seed), 'Random subsampling seed')
    hdr['RANFAC'] = (float(args.random_factor), 'Random/data count ratio')
    hdr['RTHRESH'] = (float(args.r_threshold), 'Watershed R threshold')
    if args.seed_threshold is not None:
        hdr['SEEDTHR'] = (float(args.seed_threshold), 'Watershed seed threshold')
    if args.merge_threshold is not None:
        hdr['MERGETHR'] = (float(args.merge_threshold), 'Watershed saddle merge threshold')
    hdr['MINGRP'] = (int(args.min_group_size), 'Minimum watershed group size')
    hdr['MINRSHAP'] = (int(args.min_rand_for_shape), 'Min randoms for axes')
    hdr['WMODE'] = (args.mode, 'Watershed mode')
    hdr['NVOIDS'] = (len(voids), 'Clean voids written')
    hdr['NVOIDRAW'] = (len(group_table), 'Voids before footprint cut')
    if 'GEOM_BAD' in voids.colnames:
        geom_flags = np.asarray(voids['GEOM_BAD'], dtype=bool)
        hdr['NGEOMBAD'] = (int(np.count_nonzero(geom_flags)), 'Number of GEOM_BAD=True voids')
    hdr['UNITSXYZ'] = ('Mpc/h', 'Units for R_EFF, X/Y/Z, semi-axes')
    hdr['REFFDEF'] = (REFF_DEFINITION, 'R_EFF')
    hdr['LAMDEF'] = ('eig(<dx_i dx_j>)', 'LAMBDA_1..3 definition')
    hdr['AXDEF'] = ('SEMI_AXIS_j=sqrt(5*LAMBDA_j)', 'Semi-axis definition')
    hdr['UNITSAX'] = ('unitless', 'Units for X1..Z3 axis-vector columns')
    hdr['AXVEC'] = ('Xj,Yj,Zj', 'Unit-vector components for axis j')
    hdr['UNITSANG'] = ('deg', 'Units for RA and DEC')
    hdr['ZUNIT'] = ('redshift', 'Units for REDSHIFT')
    hdr['ELLIPDEF'] = (ELLIPTICITY_DEFINITION, 'Ellipticity definition')
    hdr['J1J3'] = (J1J3_DEFINITION, 'Moment ratio')
    hdr['GEOMDEF'] = ('1-C/A>0.9', 'GEOM_BAD definition')
    hdr['EDGEDEF'] = ('GROUPID==boundary_id', 'EDGE=True means watershed boundary')
    hdr['FPEDDEF'] = ('HEALPix low-data mask', 'FOOTPRINT_EDGE definition')
    hdr['FPCUT'] = (True, 'Drop FOOTPRINT_EDGE rows')
    hdr['NEDGE'] = (n_edge, 'EDGE=True rows in VOIDS')
    hdr['NFPEDGE'] = (n_footprint_edge, 'Footprint-edge rows dropped')
    hdr['NFPCLN'] = (n_footprint_clean, 'Clean voids written')
    hdr['NFPWRT'] = (n_footprint_written, 'Footprint-edge rows written')
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
    if 'HPX_MINDATA' in group_table.meta:
        hdr['HPXMIND'] = (int(group_table.meta['HPX_MINDATA']), 'N_data/Npix threshold')
    if 'HPX_NLOWDATA' in group_table.meta:
        hdr['HPXNLOW'] = (int(group_table.meta['HPX_NLOWDATA']), 'Pixels failing N_data/Npix cut')
    hdr['IN_DATA'] = (os.path.basename(data_path), 'Input mock file')
    hdr['IN_RAND'] = (os.path.basename(randoms_path), 'Input random file')
    if args.z_min is not None:
        hdr['ZMIN'] = (float(args.z_min), 'Minimum redshift cut')
    if args.z_max is not None:
        hdr['ZMAX'] = (float(args.z_max), 'Maximum redshift cut')
    if args.bgs_mr_limit is not None:
        hdr['MRLIM'] = (float(args.bgs_mr_limit), 'BGS R_MAG_ABS upper cut')
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
    fits.HDUList(hdus).writeto(output, overwrite=args.overwrite)


def run_region(data_table, random_table, tracer, mock_kind, region,
               data_path, randoms_path, output, cosmo, args, log_fh, verbose):
    if os.path.exists(output) and not args.overwrite:
        log_message(log_fh, f'skip existing {output}', verbose=verbose)
        return output

    t0 = time.time()
    data_tbl = data_table.copy(copy_data=True)
    rand_tbl = random_table.copy(copy_data=True)
    log_message(log_fh, f'case start tracer={tracer} mock={mock_kind} '
                        f'region={region} n_data={len(data_tbl)} '
                        f'n_rand={len(rand_tbl)}',
                        verbose=verbose)

    step = time.time()
    add_cartesian_columns(data_tbl, cosmo=cosmo, h=args.h)
    add_cartesian_columns(rand_tbl, cosmo=cosmo, h=args.h)
    log_message(log_fh, f'case={tracer}/{mock_kind}/{region} cartesian '
                        f'elapsed_s={time.time() - step:.3f}',
                        verbose=verbose)

    step = time.time()
    stats = compute_neighbor_statistics(data_tbl, rand_tbl)
    add_neighbor_columns_to_tables(data_tbl, rand_tbl, stats)
    rvals = stats['r_values']
    log_message(log_fh, f'case={tracer}/{mock_kind}/{region} neighbors '
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
                       merge_threshold=args.merge_threshold)
    assign_group_ids_to_tables(data_tbl, rand_tbl, ws['group_of'],
                               group_col='GROUPID')
    log_message(log_fh, f"case={tracer}/{mock_kind}/{region} watershed "
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
                                         healpix_edge_min_randoms=getattr(args, 'healpix_edge_min_randoms', 3),
                                         healpix_edge_min_data_per_pix=healpix_edge_min_data_for_region(
                                             args, tracer, region))
    log_message(log_fh, f'case={tracer}/{mock_kind}/{region} consolidate '
                        f'elapsed_s={time.time() - step:.3f} '
                        f'n_voids={len(group_table)}',
                        verbose=verbose)

    point_table = None
    if args.include_membership:
        step = time.time()
        point_table = build_point_membership_table(data_tbl, rand_tbl,
                                                   group_col='GROUPID')
        log_message(log_fh, f'case={tracer}/{mock_kind}/{region} membership '
                            f'elapsed_s={time.time() - step:.3f} '
                            f'n_points={len(point_table)}',
                            verbose=verbose)

    step = time.time()
    write_mock_void_fits(group_table=group_table,
                         output=output,
                         tracer=tracer,
                         mock_kind=mock_kind,
                         region=region,
                         data_path=data_path,
                         randoms_path=randoms_path,
                         omega_m=args.omega_m,
                         args=args,
                         point_table=point_table)
    log_message(log_fh, f'case={tracer}/{mock_kind}/{region} write '
                        f'elapsed_s={time.time() - step:.3f} output={output}',
                        verbose=verbose)
    log_message(log_fh, f'case done tracer={tracer} mock={mock_kind} '
                        f'region={region} elapsed_s={time.time() - t0:.3f}',
                        verbose=verbose)
    return output


def main():
    args = parse_args()
    verbose = not args.quiet
    args.mock_dir = os.path.abspath(os.path.expanduser(args.mock_dir))
    args.output_dir = os.path.abspath(os.path.expanduser(args.output_dir))

    data_path = mock_path(args.mock_dir, args.tracer, args.mock_kind)
    randoms_path = random_path(args.mock_dir, args.tracer)
    regions = args.caps if args.split_caps else ['ALL']
    outputs = [output_path(args.output_dir, args.tracer, args.mock_kind, region)
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
                            f'run_fiber_assignment_{args.tracer}_{args.mock_kind}_'
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
        for region, output in zip(regions, outputs):
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
                                      mock_kind=args.mock_kind,
                                      region=region,
                                      data_path=data_path,
                                      randoms_path=randoms_path,
                                      output=output,
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