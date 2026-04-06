import os
import numpy as np
from astropy.table import Table, vstack

DEFAULT_TRACERS_DR2 = ('BGS_ANY', 'LRG', 'ELGnotqso', 'QSO')
DEFAULT_TRACERS_DR1 = ('BGS_BRIGHT', 'LRG', 'ELG_LOPnotqso', 'QSO')
DEFAULT_TRACER_ALIASES_DR1 = {'BGS_ANY': 'BGS_BRIGHT',
                              'ELGnotqso': 'ELG_LOPnotqso'}
DEFAULT_TRACERS = DEFAULT_TRACERS_DR2
DEFAULT_TRACERS_ALL = tuple(dict.fromkeys(DEFAULT_TRACERS_DR2 + DEFAULT_TRACERS_DR1 +
                                          tuple(DEFAULT_TRACER_ALIASES_DR1)))
DEFAULT_COLUMNS = ('TARGETID', 'RA', 'DEC', 'Z')
DEFAULT_RA_MIN = 83.0
DEFAULT_RA_MAX = 302.0
DEFAULT_CUTS = {'NGC1': {'RA_min': 110, 'RA_max': 260, 'DEC_min': -10, 'DEC_max': 8},
                'NGC2': {'RA_min': 180, 'RA_max': 260, 'DEC_min': 30, 'DEC_max': 40}}


def build_catalog_paths(data_dir, tracer, random_index=0, release='dr2',
                        tracer_aliases=None):
    '''
    Build file paths for the data and random catalogs based on the tracer and random index.

    Parameters:
        - data_dir: Directory where the catalogs are stored.
        - tracer: Name of the tracer (e.g., 'BGS_ANY', 'LRG', 'ELGnotqso', 'QSO').
        - random_index: Index for the random catalog, used to differentiate between multiple
                        random samples if needed.
        - release: Data release identifier ('dr2' or 'dr1').
        - tracer_aliases: Optional dict mapping logical tracer names to on-disk tracer names.
    Returns:
        - data_files: List of data catalog FITS paths.
        - rand_files: List of random catalog FITS paths.
        - tracer_on_disk: Tracer name used for on-disk path resolution.
    '''
    tracer_on_disk = tracer
    if tracer_aliases is not None and tracer in tracer_aliases:
        tracer_on_disk = tracer_aliases[tracer]

    if release == 'dr1':
        data_files = [os.path.join(data_dir, f'{tracer_on_disk}_N_clustering.dat.fits'),
                      os.path.join(data_dir, f'{tracer_on_disk}_S_clustering.dat.fits')]
        rand_files = [os.path.join(data_dir, f'{tracer_on_disk}_N_{random_index}_clustering.ran.fits'),
                      os.path.join(data_dir, f'{tracer_on_disk}_S_{random_index}_clustering.ran.fits')]
    else:
        data_files = [os.path.join(data_dir, f'{tracer_on_disk}_clustering.dat.fits')]
        rand_files = [os.path.join(data_dir, f'{tracer_on_disk}_{random_index}_clustering.ran.fits')]

    return data_files, rand_files, tracer_on_disk


def split_ngc_sgc_by_ra(table, ra_min=DEFAULT_RA_MIN, ra_max=DEFAULT_RA_MAX):
    '''
    Split the input table into NGC and SGC based on RA boundaries.

    Parameters:
        - table: Astropy Table containing the data points with an 'RA' column.
        - ra_min: Minimum RA value to define the NGC boundary.
        - ra_max: Maximum RA value to define the NGC boundary.
    Returns:
        - A dict with keys 'NGC' and 'SGC', containing the respective subsets of the input table.
    '''
    ngc_mask = (table['RA'] >= ra_min) & (table['RA'] <= ra_max)
    return {'NGC': table[ngc_mask],
            'SGC': table[~ngc_mask]}


def split_caps_by_cuts(table, cuts):
    '''
    Split the input table into arbitrary caps defined by RA/DEC cuts.

    Parameters:
        - table: Astropy Table containing the data points with 'RA' and 'DEC' columns.
        - cuts: Dict with cap names as keys and bounds dicts as values.
                Each bounds dict must define RA_min, RA_max, DEC_min, DEC_max.
    Returns:
        - A dict with one entry per cap in cuts, each containing the respective subset.
    '''
    out = {}
    for cap, bounds in cuts.items():
        cap_mask = ((table['RA'] >= bounds['RA_min']) &
                    (table['RA'] <= bounds['RA_max']) &
                    (table['DEC'] >= bounds['DEC_min']) &
                    (table['DEC'] <= bounds['DEC_max']))
        out[cap] = table[cap_mask]
    return out


def subsample_random(random_table, n_target, rng):
    '''
    Subsample the random table to match the number of data points in the corresponding cap.

    Parameters:
        - random_table: Astropy Table containing the random points.
        - n_target: Number of random points to subsample, typically equal to the number of data
                    points in the corresponding cap.
        - rng: Random number generator instance (e.g., numpy.random.Generator or numpy.random.RandomState)
               to use for reproducibility.
    Returns:
        - Subsampled Astropy Table containing n_target random points.
    '''
    if n_target < 0:
        raise ValueError(f'n_target must be >= 0, got {n_target}')
    if n_target == 0:
        return random_table[:0]
    if len(random_table) < n_target:
        raise ValueError('Not enough random points after cap split: '
                         f'requested {n_target}, available {len(random_table)}')
    idx = rng.permutation(len(random_table))[:n_target]
    return random_table[idx]


def _read_and_concat_tables(files, keep_cols):
    tables = [Table.read(path)[keep_cols] for path in files]
    if len(tables) == 1:
        return tables[0]
    return vstack(tables, metadata_conflicts='silent')


def load_tracer_samples(data_dir, tracer, random_index, columns, ra_min,
                        ra_max, rng, caps=('NGC', 'SGC'),
                        cap_cuts=None, release='dr2',
                        tracer_aliases=None, verbose=True):
    '''
    Load data and random samples for a given tracer, split them into NGC and SGC based on RA,
    and subsample the randoms to match the data counts.

    Parameters:
        - data_dir: Directory where the catalogs are stored.
        - tracer: Name of the tracer (e.g., 'BGS_ANY', 'LRG', 'ELGnotqso', 'QSO').
        - random_index: Index for the random catalog, used to differentiate between multiple random samples if needed.
        - columns: Tuple of column names to read from the catalogs (e.g., ('TARGETID', 'RA', 'DEC', 'Z')).
        - ra_min: Minimum RA value to define the NGC boundary.
        - ra_max: Maximum RA value to define the NGC boundary.
        - rng: Random number generator instance (e.g., numpy.random.Generator or numpy.random.RandomState)
               to use for reproducibility.
        - caps: Tuple or list of cap names to process.
        - cap_cuts: Optional dict with per-cap RA/DEC bounds. If None, use NGC/SGC RA split.
        - release: Data release identifier ('dr2' or 'dr1').
        - tracer_aliases: Optional dict mapping logical tracer names to on-disk tracer names.
        - verbose: If True, print the number of data and random points in NGC and SGC after processing.
    '''
    data_files, rand_files, tracer_on_disk = build_catalog_paths(data_dir, tracer,
                                                                 random_index=random_index,
                                                                 release=release,
                                                                 tracer_aliases=tracer_aliases)

    for data_file in data_files:
        if not os.path.exists(data_file):
            raise FileNotFoundError(f'Missing data catalog: {data_file}')
    for rand_file in rand_files:
        if not os.path.exists(rand_file):
            raise FileNotFoundError(f'Missing random catalog: {rand_file}')

    keep_cols = list(columns)
    data_table = _read_and_concat_tables(data_files, keep_cols=keep_cols)
    rand_table = _read_and_concat_tables(rand_files, keep_cols=keep_cols)

    if cap_cuts is None:
        data_caps = split_ngc_sgc_by_ra(data_table, ra_min=ra_min, ra_max=ra_max)
        rand_caps = split_ngc_sgc_by_ra(rand_table, ra_min=ra_min, ra_max=ra_max)
    else:
        data_caps = split_caps_by_cuts(data_table, cuts=cap_cuts)
        rand_caps = split_caps_by_cuts(rand_table, cuts=cap_cuts)

    out = {}
    data_sizes = {}
    rand_sizes = {}
    for cap in caps:
        if cap not in data_caps:
            raise KeyError(f'Cap {cap} not found in data split for tracer {tracer}')
        if cap not in rand_caps:
            raise KeyError(f'Cap {cap} not found in random split for tracer {tracer}')

        data_cap = data_caps[cap]
        rand_cap = subsample_random(rand_caps[cap], len(data_cap), rng)

        out[f'{tracer}_{cap}'] = data_cap
        out[f'{tracer}_RAND_{cap}'] = rand_cap
        data_sizes[cap] = len(data_cap)
        rand_sizes[cap] = len(rand_cap)

    if verbose:
        data_msg = ' '.join([f'{cap}={data_sizes[cap]}' for cap in caps])
        rand_msg = ' '.join([f'{cap}={rand_sizes[cap]}' for cap in caps])
        if tracer_on_disk == tracer:
            print(f'[{tracer}] data {data_msg} - rand {rand_msg}')
        else:
            print(f'[{tracer}] on_disk={tracer_on_disk} data {data_msg} - rand {rand_msg}')

    return out


def load_all_tracer_samples(data_dir, tracers=DEFAULT_TRACERS, random_index=0,
                            columns=DEFAULT_COLUMNS, ra_min=DEFAULT_RA_MIN,
                            ra_max=DEFAULT_RA_MAX, seed=12345,
                            caps=('NGC', 'SGC'), cap_cuts=None,
                            release='dr2', tracer_aliases=None, verbose=True):
    '''
    Load data and random samples for all specified tracers, split them into NGC and SGC based on RA,
    and subsample the randoms to match the data counts.

    Parameters:
        - data_dir: Directory where the catalogs are stored.
        - tracers: Tuple of tracer names to load (e.g., ('BGS_ANY', 'LRG', 'ELGnotqso', 'QSO')).
        - random_index: Index for the random catalog, used to differentiate between multiple random samples if needed.
        - columns: Tuple of column names to read from the catalogs (e.g., ('TARGETID', 'RA', 'DEC', 'Z')).
        - ra_min: Minimum RA value to define the NGC boundary.
        - ra_max: Maximum RA value to define the NGC boundary.
        - seed: Random seed for reproducibility when subsampling randoms.
        - caps: Tuple or list of cap names to process.
        - cap_cuts: Optional dict with per-cap RA/DEC bounds. If None, use NGC/SGC RA split.
        - release: Data release identifier ('dr2' or 'dr1').
        - tracer_aliases: Optional dict mapping logical tracer names to on-disk tracer names.
        - verbose: If True, print the number of data and random points in NGC and SGC for each tracer after processing.
    Returns:
        - A dict containing the loaded and processed tables for each tracer and cap, with keys like
          'BGS_ANY_NGC', 'BGS_ANY_SGC', 'BGS_ANY_RAND_NGC', 'BGS_ANY_RAND_SGC', etc.
    '''
    if hasattr(np.random, 'default_rng'):
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.RandomState(seed)
    all_data = {}

    for tracer in tracers:
        tracer_tables = load_tracer_samples(data_dir=data_dir, tracer=tracer,
                                            random_index=random_index, columns=columns,
                                            ra_min=ra_min, ra_max=ra_max, rng=rng,
                                            caps=caps, cap_cuts=cap_cuts,
                                            release=release,
                                            tracer_aliases=tracer_aliases,
                                            verbose=verbose)
        all_data.update(tracer_tables)

    return all_data


def validate_required_keys(all_data, tracers, caps):
    '''
    Validate that all required keys for the specified tracers and caps are present in the all_data dict.

    Parameters:
        - all_data: Dict containing the loaded data tables for each tracer and cap.
        - tracers: Tuple of tracer names (e.g., ('BGS_ANY', 'LRG', 'ELGnotqso', 'QSO')).
        - caps: Tuple of cap names (e.g., ('NGC', 'SGC')).
    Raises:
        - KeyError: If any required key is missing from the all_data dict, with a
    '''
    missing = []
    for tracer in tracers:
        for cap in caps:
            key = f'{tracer}_{cap}'
            rand_key = f'{tracer}_RAND_{cap}'
            if key not in all_data:
                missing.append(key)
            if rand_key not in all_data:
                missing.append(rand_key)

    if missing:
        missing_str = ', '.join(missing)
        raise KeyError(f'Missing required tables in all_data: {missing_str}')
