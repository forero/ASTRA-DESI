import argparse, glob, json, os, re, time
import numpy as np
import fitsio
from astropy.io import fits
from astropy.table import Table, join, vstack

from group_finder.astra import (add_cartesian_columns,
                                add_neighbor_columns_to_tables,
                                build_cosmology,
                                compute_neighbor_statistics)
from group_finder.make_cat import (build_point_membership_table,
                                   consolidate_group_info,
                                   write_group_table_fits)
from group_finder.read_data import (DEFAULT_DR1_MASK_DIR,
                                    load_dr1_healpix_masks,
                                    split_caps_by_healpix_masks)
from group_finder.watershed import assign_group_ids_to_tables, run_watershed


DEFAULT_DATA_DIR = '/global/cfs/cdirs/desi/public/dr1/vac/dr1/lss/guadalupe/v1.0/LSScats/clustering'
DEFAULT_FASTSPEC = '/global/cfs/cdirs/desi/public/dr1/vac/dr1/fastspecfit/guadalupe/v3.1/catalogs/fastspec-guadalupe-main-bright.fits'
DEFAULT_OUTPUT_DIR = '/pscratch/sd/v/vtorresg/cosmic-web/dr1/void-cat-mr20-z024-100iter-group-finder-randomzmatch'
TRACER = 'BGS_BRIGHT'
DATA_COLUMNS = ('TARGETID', 'RA', 'DEC', 'Z')
FSF_COLUMNS = ('TARGETID', 'ABSMAG01_SDSS_R', 'ABSMAG01_IVAR_SDSS_R')
CAP_TO_HEMI = {'NGC': 'N', 'SGC': 'S'}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', default=DEFAULT_DATA_DIR)
    parser.add_argument('--fastspec-path', default=DEFAULT_FASTSPEC)
    parser.add_argument('--mask-dir', default=DEFAULT_DR1_MASK_DIR)
    parser.add_argument('--output-dir', default=DEFAULT_OUTPUT_DIR)
    parser.add_argument('--log-dir', default=None)

    parser.add_argument('--iteration', type=int, default=0)
    parser.add_argument('--random-index', type=int, default=None)
    parser.add_argument('--seed-base', type=int, default=12345)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--caps', nargs='+', default=['NGC', 'SGC'], choices=['NGC', 'SGC'])
    parser.add_argument('--cap-split', choices=['file', 'mask'], default='file')

    parser.add_argument('--sample-tag', default='Mr20_z0p24')
    parser.add_argument('--z-max', type=float, default=0.24)
    parser.add_argument('--mr-limit', type=float, default=-20.0)
    parser.add_argument('--e-corr-slope', type=float, default=0.97)
    parser.add_argument('--z-pivot', type=float, default=0.1)
    parser.add_argument('--random-z-mode', choices=['original', 'data-z'], default='data-z')

    parser.add_argument('--h', type=float, default=0.6736)
    parser.add_argument('--omega-m', type=float, default=0.315)
    parser.add_argument('--r-threshold', type=float, default=-0.25)
    parser.add_argument('--seed-threshold', type=float, default=-0.85)
    parser.add_argument('--merge-threshold', type=float, default=-0.80)
    parser.add_argument('--min-group-size', type=int, default=4)
    parser.add_argument('--min-rand-for-shape', type=int, default=3)
    parser.add_argument('--healpix-edge-nside', type=int, default=256)
    parser.add_argument('--healpix-edge-min-randoms', type=int, default=3)
    parser.add_argument('--mode', choices=['underdense', 'overdense'], default='underdense')

    parser.add_argument('--overwrite', action='store_true', default=False)
    parser.add_argument('--quiet', action='store_true', default=False)
    parser.add_argument('--count-only', action='store_true', default=False)
    return parser.parse_args()


def utc_timestamp():
    return time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime())


def log_message(log_fh, message, verbose=True):
    line = f'[{utc_timestamp()}] {message}'
    log_fh.write(line + '\n')
    log_fh.flush()
    if verbose:
        print(message, flush=True)


def read_table_columns(path, columns):
    data = fitsio.read(path, ext=1, columns=list(columns))
    return Table(data)


def read_lss_data(data_dir):
    tables = []
    for hemi in ('N', 'S'):
        path = os.path.join(data_dir, f'{TRACER}_{hemi}_clustering.dat.fits')
        if not os.path.exists(path):
            raise FileNotFoundError(f'---> Missing LSS data catalog: {path}')
        tables.append(read_table_columns(path, DATA_COLUMNS))
    return vstack(tables, metadata_conflicts='silent')


def read_lss_data_for_cap(data_dir, cap):
    hemi = CAP_TO_HEMI[cap]
    path = os.path.join(data_dir, f'{TRACER}_{hemi}_clustering.dat.fits')
    if not os.path.exists(path):
        raise FileNotFoundError(f'---> Missing LSS data catalog: {path}')
    return read_table_columns(path, DATA_COLUMNS)


def read_fastspec(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f'---> Missing fastspecfit catalog: {path}')
    return read_table_columns(path, FSF_COLUMNS)


def filter_lss_with_fastspec(lss, fsf, args):
    '''
    Filter the LSS catalog by joining with the fastspecfit catalog on TARGETID,
    then applying cuts on absolute magnitude and redshift.
    The Mr cut is applied after an evolution correction that depends on redshift,
    with a pivot redshift and a slope specified by args (DESIVAST paper).
    '''
    cat = join(lss, fsf, keys='TARGETID', join_type='inner',
               table_names=['lss', 'fsf'], metadata_conflicts='silent')

    z = np.asarray(cat['Z'], dtype=np.float64)
    mr = np.asarray(cat['ABSMAG01_SDSS_R'], dtype=np.float64)
    mr_ivar = np.asarray(cat['ABSMAG01_IVAR_SDSS_R'], dtype=np.float64)
    mr_ecorr = mr - args.e_corr_slope * (z - args.z_pivot)

    mask = (np.isfinite(mr) & np.isfinite(z) &
            (mr_ivar > 0) & (z <= args.z_max) &
            (mr_ecorr <= args.mr_limit))

    out = cat[mask]
    out['MR_ECORR'] = mr_ecorr[mask].astype(np.float32)
    keep_cols = list(DATA_COLUMNS) + ['ABSMAG01_SDSS_R',
                                      'ABSMAG01_IVAR_SDSS_R',
                                      'MR_ECORR']
    return out[keep_cols], {'n_lss': len(lss),
                            'n_join': len(cat),
                            'n_filtered': len(out)}


def build_filtered_data_by_cap(args, fsf, log_fh, verbose):
    '''
    Build the filtered data tables for each cap, either by reading separate files
    for each cap or by reading a single file and splitting by healpix masks.
    '''
    sample_counts = {'n_fastspec': len(fsf),
                     'n_lss': 0,
                     'n_join': 0,
                     'n_filtered': 0}

    if args.cap_split == 'file':
        data_caps = {}
        for cap in args.caps:
            lss = read_lss_data_for_cap(args.data_dir, cap)
            data, counts = filter_lss_with_fastspec(lss, fsf, args)
            data_caps[cap] = data
            sample_counts['n_lss'] += counts['n_lss']
            sample_counts['n_join'] += counts['n_join']
            sample_counts['n_filtered'] += counts['n_filtered']
            sample_counts[f'n_{cap.lower()}_lss'] = counts['n_lss']
            sample_counts[f'n_{cap.lower()}_join'] = counts['n_join']
            sample_counts[f'n_{cap.lower()}_filtered'] = counts['n_filtered']
        return data_caps, sample_counts

    lss = read_lss_data(args.data_dir)
    data_all, counts = filter_lss_with_fastspec(lss, fsf, args)
    sample_counts['n_lss'] = counts['n_lss']
    sample_counts['n_join'] = counts['n_join']
    sample_counts['n_filtered'] = counts['n_filtered']

    mask_cache = load_dr1_healpix_masks(mask_dir=args.mask_dir,
                                        caps=args.caps,
                                        programs=('bright',))
    masks_by_cap = mask_cache['masks']['bright']
    nside = mask_cache['nside']
    data_caps = split_caps_by_healpix_masks(data_all, masks_by_cap, nside, caps=args.caps)
    log_message(log_fh, f'Step=split_data_caps mask_dir={mask_cache["mask_dir"]} nside={nside}',
                verbose=verbose)
    return data_caps, sample_counts


def available_random_indices(data_dir):
    '''
    Look for random catalog files in the data directory and extract the available
    random index values for both N and S hemispheres. Return the sorted list of
    indices that have files for both hemispheres.
    '''
    pattern = os.path.join(data_dir, f'{TRACER}_N_*_clustering.ran.fits')
    n_files = glob.glob(pattern)
    regex = re.compile(r'_N_(\d+)_clustering\.ran\.fits$')
    n_idx = {int(regex.search(path).group(1))
             for path in n_files if regex.search(path)}

    pattern = os.path.join(data_dir, f'{TRACER}_S_*_clustering.ran.fits')
    s_files = glob.glob(pattern)
    regex = re.compile(r'_S_(\d+)_clustering\.ran\.fits$')
    s_idx = {int(regex.search(path).group(1))
             for path in s_files if regex.search(path)}

    return sorted(n_idx & s_idx)


def resolve_random_index(data_dir, iteration, requested_index):
    '''
    Determine the random catalog index to use for this run.
    '''
    if requested_index is not None:
        return int(requested_index), available_random_indices(data_dir)

    indices = available_random_indices(data_dir)
    if not indices:
        raise FileNotFoundError(f'No {TRACER} random catalogs found in {data_dir}')
    return int(indices[iteration % len(indices)]), indices


def read_randoms(data_dir, random_index, z_max=None):
    '''
    Read the random catalogs for both hemispheres for the given random index,
    and optionally apply a redshift cut.
    Return a single table with all random points.
    '''
    tables = []
    for hemi in ('N', 'S'):
        path = os.path.join(data_dir, f'{TRACER}_{hemi}_{random_index}_clustering.ran.fits')
        if not os.path.exists(path):
            raise FileNotFoundError(f'---> Missing LSS random catalog: {path}')
        tables.append(read_table_columns(path, DATA_COLUMNS))
    rand = vstack(tables, metadata_conflicts='silent')
    if z_max is None:
        return rand
    z = np.asarray(rand['Z'], dtype=np.float64)
    return rand[np.isfinite(z) & (z <= z_max)]


def read_random_for_cap(data_dir, cap, random_index, z_max=None):
    '''
    Read the random catalog for the given cap and random index,
    and optionally apply a redshift cut.
    Return the table of random points for this cap.
    '''
    hemi = CAP_TO_HEMI[cap]
    path = os.path.join(data_dir, f'{TRACER}_{hemi}_{random_index}_clustering.ran.fits')
    if not os.path.exists(path):
        raise FileNotFoundError(f'---> Missing LSS random catalog: {path}')
    rand = read_table_columns(path, DATA_COLUMNS)
    if z_max is None:
        return rand
    z = np.asarray(rand['Z'], dtype=np.float64)
    return rand[np.isfinite(z) & (z <= z_max)]


def subsample_random(random_table, n_target, seed):
    '''
    Subsample the random table to have n_target rows, using the given seed for reproducibility.
    '''
    if n_target < 0:
        raise ValueError(f'---- n_target must be >= 0, got {n_target}')
    if n_target == 0:
        return random_table[:0].copy()
    if len(random_table) < n_target:
        raise ValueError('--------- Not enough random points after cuts and cap mask: '
                         f'requested {n_target}, available {len(random_table)}')
    if len(random_table) == n_target:
        return random_table.copy()

    rng = np.random.default_rng(seed)
    idx = rng.choice(len(random_table), size=n_target, replace=False)
    return random_table[idx].copy()


def match_random_redshifts_to_data(random_table, data_table, args, seed):
    '''
    Adjust the redshifts of the random table to match the distribution of the data table,
    according to the mode specified in args.random_z_mode.
     - 'original': keep the original random redshifts (after any z cut applied on read)
     - 'data-z': assign the data redshifts in random order when counts match,
       otherwise draw data redshifts with replacement
     In all cases, the returned table has the same number of rows as random_table,
     but the Z column may be modified to match the data distribution.
    '''
    if args.random_z_mode == 'original':
        return random_table.copy(copy_data=True)
    if args.random_z_mode != 'data-z':
        raise ValueError()

    out = random_table.copy(copy_data=True)
    if len(out) == 0:
        return out

    data_z = np.asarray(data_table['Z'], dtype=np.float64)
    data_z = data_z[np.isfinite(data_z) & (data_z >= 0.0) & (data_z <= args.z_max)]
    if len(data_z) == 0:
        raise RuntimeError('------------- Cant assign random Z values: data table has no finite Z in range.')
    rng = np.random.default_rng(seed)
    if len(out) == len(data_z):
        z_new = rng.permutation(data_z).astype(np.float64)
    else:
        z_new = rng.choice(data_z, size=len(out), replace=True).astype(np.float64)

    out['Z'] = z_new
    return out


def finite_mean_or_nan(values):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return np.nan
    return float(np.mean(values))


def output_path_for(args, cap):
    iter_dir = os.path.join(args.output_dir, f'iter_{args.iteration:03d}')
    filename = f'voids_{TRACER}_{args.sample_tag}_{cap}_iter{args.iteration:03d}.fits'
    return os.path.join(iter_dir, filename)


def patch_header(path, args, cap, random_index, seed, data_count, rand_count,
                 sample_counts, available_indices):
    with fits.open(path, mode='update') as hdul:
        hdr = hdul[0].header
        hdr['SAMPLE'] = (args.sample_tag, 'Input sample tag')
        hdr['CAPMODE'] = (args.cap_split, 'Cap split mode')
        hdr['ITER'] = (int(args.iteration), 'Realization index')
        hdr['RANDIDX'] = (int(random_index), 'LSS random catalog index')
        hdr['SEED'] = (int(seed), 'Random subsampling seed')
        hdr['MRLIM'] = (float(args.mr_limit), 'Mr ecorr cut')
        hdr['ZMAX'] = (float(args.z_max), 'Maximum redshift cut')
        hdr['ECORR'] = (float(args.e_corr_slope), 'Mr evolution correction slope')
        hdr['ZPIVOT'] = (float(args.z_pivot), 'Mr evolution correction pivot redshift')
        hdr['RZMODE'] = (args.random_z_mode, 'Rand Z mode')
        hdr['NDATCAP'] = (int(data_count), 'Data points in this cap after cuts')
        hdr['NRANCAP'] = (int(rand_count), 'Random points used in this cap')
        hdr['NLSS'] = (int(sample_counts['n_lss']), 'Input LSS BGS rows')
        hdr['NFSF'] = (int(sample_counts['n_fastspec']), 'Input fastspecfit rows')
        hdr['NJOIN'] = (int(sample_counts['n_join']), 'Rows after TARGETID join')
        hdr['NFILT'] = (int(sample_counts['n_filtered']), 'Rows after Mr/z cuts')
        hdr['NRANDAV'] = (int(len(available_indices)), 'Available N/S random indices')
        hdul.flush()


def run_case(args, cap, data_table, rand_table, cosmo, random_index, seed,
             sample_counts, available_indices, log_fh, verbose):
    '''
    Run the full group finding process for a single cap, including:
        - Adding Cartesian coordinates to data and random tables
        - Computing neighbor statistics and adding to tables
        - Running watershed to assign group IDs
        - Consolidating group info into a group table
        - Building point membership table
        - Writing output FITS file with group and point tables
    '''
    output_path = output_path_for(args, cap)
    if os.path.exists(output_path) and not args.overwrite:
        log_message(log_fh, f'Case={cap} skipped output_exists path={output_path}',
                    verbose=verbose)
        return

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    data_tbl = data_table.copy(copy_data=True)
    rand_tbl = rand_table.copy(copy_data=True)

    t_case = time.time()
    log_message(log_fh, f'Case={cap} start n_data={len(data_tbl)} n_rand={len(rand_tbl)}',
                verbose=verbose)

    t_step = time.time()
    add_cartesian_columns(data_tbl, cosmo=cosmo, h=args.h)
    add_cartesian_columns(rand_tbl, cosmo=cosmo, h=args.h)
    log_message(log_fh, f'Case={cap} Step=cartesian done elapsed_s={time.time() - t_step:.3f}',
                verbose=verbose)

    t_step = time.time()
    stats = compute_neighbor_statistics(data_tbl, rand_tbl)
    add_neighbor_columns_to_tables(data_tbl, rand_tbl, stats)
    rvals = stats['r_values']
    log_message(log_fh, f'Case={cap} Step=neighbor_stats done elapsed_s={time.time() - t_step:.3f} '
                        f'n={len(rvals)} min={float(rvals.min()):.3f} max={float(rvals.max()):.3f}',
                verbose=verbose)

    t_step = time.time()
    ws = run_watershed(neighbors=stats['neighbors'],
                       r_values=stats['r_values'],
                       r_threshold=args.r_threshold,
                       min_group_size=args.min_group_size,
                       mode=args.mode,
                       seed_threshold=args.seed_threshold,
                       merge_threshold=args.merge_threshold)
    assign_group_ids_to_tables(data_tbl, rand_tbl, ws['group_of'],
                               group_col='GROUPID')
    log_message(log_fh, f'Case={cap} Step=watershed done elapsed_s={time.time() - t_step:.3f} '
                        f'groups={ws["n_groups"]} assigned={ws["n_assigned"]} '
                        f'boundary={ws["n_boundary_nodes"]} '
                        f'unassigned={ws["n_unassigned"]}',
                verbose=verbose)

    t_step = time.time()
    group_table = consolidate_group_info(data_table=data_tbl,
                                         rand_table=rand_tbl,
                                         cosmo=cosmo,
                                         h=args.h,
                                         group_col='GROUPID',
                                         min_rand_for_shape=args.min_rand_for_shape,
                                         healpix_edge_nside=args.healpix_edge_nside,
                                         healpix_edge_min_randoms=args.healpix_edge_min_randoms)
    log_message(log_fh, f'Case={cap} Step=consolidate_groups done elapsed_s={time.time() - t_step:.3f} '
                        f'n_voids={len(group_table)}',
                verbose=verbose)

    t_step = time.time()
    point_table = build_point_membership_table(data_table=data_tbl,
                                               rand_table=rand_tbl,
                                               group_col='GROUPID')
    write_group_table_fits(group_table=group_table,
                           output_path=output_path,
                           tracer=f'{TRACER}_{args.sample_tag}',
                           cap=cap,
                           h=args.h,
                           omega_m=args.omega_m,
                           r_threshold=args.r_threshold,
                           mode=args.mode,
                           point_table=point_table,
                           seed_threshold=args.seed_threshold,
                           boundary_id=ws['boundary_id'],
                           watershed_stats=ws,
                           overwrite=args.overwrite)
    patch_header(output_path, args, cap, random_index, seed, len(data_tbl), len(rand_tbl),
                 sample_counts, available_indices)
    log_message(log_fh, f'Case={cap} Step=write_fits done elapsed_s={time.time() - t_step:.3f} '
                        f'output={output_path}',
                verbose=verbose)
    log_message(log_fh, f'Case={cap} done elapsed_s={time.time() - t_case:.3f}',
                verbose=verbose)


def main():
    args = parse_args()

    verbose = not args.quiet
    args.output_dir = os.path.abspath(os.path.expanduser(args.output_dir))
    log_dir = args.log_dir or os.path.join(args.output_dir, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f'voids_{args.sample_tag}_iter{args.iteration:03d}.log')

    with open(log_path, 'a', encoding='utf-8') as log_fh:
        t0 = time.time()
        random_index, available_indices = resolve_random_index(args.data_dir,
                                                               args.iteration,
                                                               args.random_index)
        seed = int(args.seed if args.seed is not None else args.seed_base + args.iteration)

        output_paths = [output_path_for(args, cap) for cap in args.caps]
        if all(os.path.exists(path) for path in output_paths) and not args.overwrite:
            log_message(log_fh, f'Iteration={args.iteration} skipped all outputs exist',
                        verbose=verbose)
            return

        config = vars(args).copy()
        config['random_index_resolved'] = random_index
        config['available_random_indices'] = available_indices
        config['seed_resolved'] = seed
        log_message(log_fh, f'Run start. log_file={log_path}', verbose=verbose)
        log_message(log_fh, f'Run config: {json.dumps(config, sort_keys=True)}',
                    verbose=verbose)

        t_step = time.time()
        fsf = read_fastspec(args.fastspec_path)
        data_caps, sample_counts = build_filtered_data_by_cap(args, fsf, log_fh, verbose)
        log_message(log_fh, f'Step=filter_data done elapsed_s={time.time() - t_step:.3f} '
                            f'counts={json.dumps(sample_counts, sort_keys=True)}',
                    verbose=verbose)

        t_step = time.time()
        random_read_z_max = args.z_max if args.random_z_mode == 'original' else None
        if args.cap_split == 'file':
            random_caps = {cap: read_random_for_cap(args.data_dir, cap, random_index,
                                                    random_read_z_max)
                           for cap in args.caps}
            rand_count_msg = ' '.join(f'{cap}={len(random_caps[cap])}' for cap in args.caps)
            log_message(log_fh, f'Step=read_randoms done elapsed_s={time.time() - t_step:.3f} '
                                f'random_index={random_index} random_z_mode={args.random_z_mode} '
                                f'zcut_for_randoms={random_read_z_max is not None} '
                                f'counts={rand_count_msg}',
                        verbose=verbose)
        else:
            random_all = read_randoms(args.data_dir, random_index=random_index,
                                      z_max=random_read_z_max)
            mask_cache = load_dr1_healpix_masks(mask_dir=args.mask_dir,
                                                caps=args.caps,
                                                programs=('bright',))
            masks_by_cap = mask_cache['masks']['bright']
            nside = mask_cache['nside']
            random_caps = split_caps_by_healpix_masks(random_all, masks_by_cap, nside, caps=args.caps)
            log_message(log_fh, f'Step=read_randoms_and_split_caps done elapsed_s={time.time() - t_step:.3f} '
                                f'random_index={random_index} random_z_mode={args.random_z_mode} '
                                f'zcut_for_randoms={random_read_z_max is not None} '
                                f'n_random_input={len(random_all)} '
                                f'mask_dir={mask_cache["mask_dir"]} nside={nside}',
                        verbose=verbose)

        cap_randoms = {}
        cap_seeds = {}
        for i, cap in enumerate(args.caps):
            cap_seed = seed + 1009 * i
            cap_seeds[cap] = cap_seed
            random_subsample = subsample_random(random_caps[cap],
                                                n_target=len(data_caps[cap]),
                                                seed=cap_seed)
            cap_randoms[cap] = match_random_redshifts_to_data(random_subsample,
                                                              data_caps[cap],
                                                              args,
                                                              seed=cap_seed + 50021)
            data_z = np.asarray(data_caps[cap]['Z'], dtype=np.float64)
            rand_z = np.asarray(cap_randoms[cap]['Z'], dtype=np.float64)
            log_message(log_fh, f'Cap={cap} counts data={len(data_caps[cap])} '
                                f'random_available={len(random_caps[cap])} '
                                f'random_used={len(cap_randoms[cap])} seed={cap_seed} '
                                f'random_z_mode={args.random_z_mode} '
                                f'data_z_mean={finite_mean_or_nan(data_z):.5f} '
                                f'random_z_mean={finite_mean_or_nan(rand_z):.5f}',
                        verbose=verbose)

        if args.count_only:
            log_message(log_fh, f'--- Count-only complete elapsed_s={time.time() - t0:.3f}',
                        verbose=verbose)
            return

        cosmo = build_cosmology(h=args.h, omega_m=args.omega_m)
        for cap in args.caps:
            run_case(args, cap, data_caps[cap], cap_randoms[cap], cosmo,
                     random_index, cap_seeds[cap], sample_counts, available_indices,
                     log_fh, verbose)

        log_message(log_fh, f'!------> -> -> Run completed elapsed_s={time.time() - t0:.3f}',
                    verbose=verbose)


if __name__ == '__main__':
    main()
