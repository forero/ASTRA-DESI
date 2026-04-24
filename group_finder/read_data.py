import glob, os, re
import numpy as np
from astropy.io import fits
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
DEFAULT_CAPS_DR1 = ('NGC', 'SGC')
DEFAULT_DR1_MASK_DIR = '/pscratch/sd/v/vtorresg/cosmic-web/dr1/masks/bright_dark'
DR1_TRACER_MASK_PROGRAM = {'BGS_BRIGHT': 'bright',
                           'BGS_ANY': 'bright',
                           'LRG': 'dark',
                           'ELG_LOPnotqso': 'dark',
                           'ELGnotqso': 'dark',
                           'QSO': 'dark'}
DR1_MASK_PROGRAMS = ('bright', 'dark')
DR1_MASK_ZONE_SUFFIX = {'NGC': 'ngc', 'SGC': 'sgc'}
DR1_MASK_NSIDE_RE = re.compile(r'_nside(?P<nside>\d+)_')
_HEALPY = None
_HEALPY_CHECKED = False


def _get_healpy():
    global _HEALPY, _HEALPY_CHECKED
    if _HEALPY_CHECKED:
        return _HEALPY
    _HEALPY_CHECKED = True
    try:
        import healpy as hp
    except ImportError:
        _HEALPY = None
    else:
        _HEALPY = hp
    return _HEALPY


def _nside_from_npix(npix):
    nside = int(round((int(npix) / 12.0) ** 0.5))
    if 12 * nside * nside != int(npix):
        raise RuntimeError(f'Invalid HEALPix map size: npix={npix}')
    return nside


def _read_healpix_map_astropy(path):
    with fits.open(path, memmap=True) as hdul:
        hdu = hdul[1]
        header = hdu.header
        ordering = str(header.get('ORDERING', 'RING')).strip().upper()
        if ordering != 'RING':
            raise RuntimeError(f'Only RING HEALPix masks are supported without healpy: {path}')

        nside = header.get('NSIDE')
        data = hdu.data
        if data is None:
            raise RuntimeError(f'HEALPix mask has no table data: {path}')

        names = list(data.names or [])
        if names:
            value_name = None
            for candidate in ('T', 'TEMPERATURE', 'SIGNAL', 'MASK'):
                if candidate in names:
                    value_name = candidate
                    break
            if value_name is None:
                value_name = next((name for name in names if name.upper() not in ('PIXEL', 'PIX')),
                                  names[0])
            arr = np.asarray(data[value_name]).reshape(-1)
        else:
            arr = np.asarray(data).reshape(-1)

        if nside is None:
            nside = _nside_from_npix(arr.size)
        nside = int(nside)
        expected_npix = 12 * nside * nside
        if arr.size != expected_npix:
            raise RuntimeError(f'HEALPix mask {path} has {arr.size} pixels, expected {expected_npix}')
        return arr, nside


def _read_healpix_map(path):
    hp = _get_healpy()
    if hp is not None:
        values = hp.read_map(path, dtype=np.int16, verbose=False)
        arr = np.asarray(values)
        return arr, hp.get_nside(arr)
    return _read_healpix_map_astropy(path)


def _ang2pix_ring(nside, theta, phi):
    theta = np.asarray(theta, dtype=np.float64)
    phi = np.asarray(phi, dtype=np.float64)
    scalar = theta.ndim == 0 and phi.ndim == 0
    theta, phi = np.broadcast_arrays(theta, phi)

    z = np.cos(theta)
    za = np.abs(z)
    tt = np.mod(phi, 2.0 * np.pi) / (0.5 * np.pi)

    nside = int(nside)
    ncap = 2 * nside * (nside - 1)
    npix = 12 * nside * nside
    pix = np.empty(theta.shape, dtype=np.int64)

    equatorial = za <= (2.0 / 3.0)
    if np.any(equatorial):
        temp1 = nside * (0.5 + tt[equatorial])
        temp2 = nside * z[equatorial] * 0.75
        jp = np.floor(temp1 - temp2).astype(np.int64)
        jm = np.floor(temp1 + temp2).astype(np.int64)
        ir = nside + 1 + jp - jm
        kshift = 1 - (ir & 1)
        ip = np.floor((jp + jm - nside + kshift + 1) / 2.0).astype(np.int64) + 1
        ip = np.where(ip > 4 * nside, ip - 4 * nside, ip)
        pix[equatorial] = ncap + (ir - 1) * 4 * nside + ip - 1

    polar = ~equatorial
    if np.any(polar):
        tp = tt[polar] - np.floor(tt[polar])
        tmp = nside * np.sqrt(3.0 * (1.0 - za[polar]))
        jp = np.floor(tp * tmp).astype(np.int64)
        jm = np.floor((1.0 - tp) * tmp).astype(np.int64)
        ir = jp + jm + 1
        ip = np.floor(tt[polar] * ir).astype(np.int64) + 1
        ip = np.where(ip > 4 * ir, ip - 4 * ir, ip)

        north = z[polar] > 0.0
        pix_north = 2 * ir * (ir - 1) + ip
        pix_south = npix - 2 * ir * (ir + 1) + ip
        pix[polar] = np.where(north, pix_north, pix_south) - 1

    if scalar:
        return int(pix.item())
    return pix


def _ang2pix(nside, theta, phi):
    hp = _get_healpy()
    if hp is not None:
        return hp.ang2pix(nside, theta, phi)
    return _ang2pix_ring(nside, theta, phi)


def _normalize_release(release):
    return str(release).strip().lower()


def _normalize_cap_label(cap):
    return str(cap).strip().upper()


def _normalize_caps(caps):
    return tuple(_normalize_cap_label(cap) for cap in caps)


def _validate_dr1_caps(caps):
    caps = _normalize_caps(caps)
    invalid = [cap for cap in caps if cap not in DR1_MASK_ZONE_SUFFIX]
    if invalid:
        allowed = ', '.join(DEFAULT_CAPS_DR1)
        raise ValueError(f'DR1 caps must be HEALPix mask labels ({allowed}); got {invalid}')
    return caps


def _dr1_mask_dir(mask_dir=None):
    env_mask_dir = os.environ.get('ASTRA_DR1_MASK_DIR')
    selected = env_mask_dir or mask_dir or DEFAULT_DR1_MASK_DIR
    return os.path.abspath(os.path.expanduser(selected))


def _extract_nside_from_path(path):
    match = DR1_MASK_NSIDE_RE.search(os.path.basename(path))
    if match is None:
        return -1
    return int(match.group('nside'))


def _dr1_mask_program_for_tracer(tracer):
    program = DR1_TRACER_MASK_PROGRAM.get(str(tracer))
    if program is None:
        known = ', '.join(sorted(DR1_TRACER_MASK_PROGRAM))
        raise KeyError(f'No DR1 HEALPix mask program configured for tracer {tracer}. Known: {known}')
    return program


def load_dr1_healpix_masks(mask_dir=None, caps=DEFAULT_CAPS_DR1,
                           programs=DR1_MASK_PROGRAMS):
    '''
    Load DR1 bright/dark HEALPix masks for NGC/SGC.

    Args:
        mask_dir: Directory containing dr1_mask_<program>_nside*_<cap>.fits.
              ASTRA_DR1_MASK_DIR takes priority when set.
        caps: Cap labels to load, usually ('NGC', 'SGC').
        programs: Mask programs to load, usually ('bright', 'dark').
    Returns:
        Dict with keys masks, paths, nside, mask_dir.
    '''
    caps = _validate_dr1_caps(caps)
    programs = tuple(str(program).strip().lower() for program in programs)
    selected_dir = _dr1_mask_dir(mask_dir)

    masks = {program: {} for program in programs}
    paths = {program: {} for program in programs}
    expected_nside = None
    expected_npix = None

    for program in programs:
        for cap in caps:
            suffix = DR1_MASK_ZONE_SUFFIX[cap]
            pattern = os.path.join(selected_dir, f'dr1_mask_{program}_nside*_{suffix}.fits')
            candidates = glob.glob(pattern)
            if not candidates:
                raise FileNotFoundError(f'No DR1 HEALPix mask file matches {pattern}')

            selected = max(candidates, key=_extract_nside_from_path)
            arr, nside = _read_healpix_map(selected)
            npix = arr.size

            if expected_nside is None:
                expected_nside = nside
                expected_npix = npix
            elif nside != expected_nside or npix != expected_npix:
                raise RuntimeError('DR1 HEALPix mask maps have inconsistent NSIDE/npix')

            masks[program][cap] = arr > 0
            paths[program][cap] = selected

    return {'masks': masks,
            'paths': paths,
            'nside': expected_nside,
            'mask_dir': selected_dir}


def build_catalog_paths(data_dir, tracer, random_index=0, release='dr2',
                        tracer_aliases=None):
    '''
    Build file paths for the data and random catalogs based on the tracer and random index.

    Args:
        data_dir: Directory where the catalogs are stored.
        tracer: Name of the tracer (e.g., 'BGS_ANY', 'LRG', 'ELGnotqso', 'QSO').
        random_index: Index for the random catalog, used to differentiate between multiple
                  random samples if needed.
        release: Data release identifier ('dr2' or 'dr1').
        tracer_aliases: Optional dict mapping logical tracer names to on-disk tracer names.
    Returns:
        data_files: List of data catalog FITS paths.
        rand_files: List of random catalog FITS paths.
        tracer_on_disk: Tracer name used for on-disk path resolution.
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

    Args:
        table: Astropy Table containing the data points with an 'RA' column.
        ra_min: Minimum RA value to define the NGC boundary.
        ra_max: Maximum RA value to define the NGC boundary.
    Returns:
        A dict with keys 'NGC' and 'SGC', containing the respective subsets of the input table.
    '''
    ngc_mask = (table['RA'] >= ra_min) & (table['RA'] <= ra_max)
    return {'NGC': table[ngc_mask],
            'SGC': table[~ngc_mask]}


def split_caps_by_cuts(table, cuts):
    '''
    Split the input table into arbitrary caps defined by RA/DEC cuts.

    Args:
        table: Astropy Table containing the data points with 'RA' and 'DEC' columns.
        cuts: Dict with cap names as keys and bounds dicts as values.
                Each bounds dict must define RA_min, RA_max, DEC_min, DEC_max.
    Returns:
        A dict with one entry per cap in cuts, each containing the respective subset.
    '''
    out = {}
    for cap, bounds in cuts.items():
        cap_mask = ((table['RA'] >= bounds['RA_min']) &
                    (table['RA'] <= bounds['RA_max']) &
                    (table['DEC'] >= bounds['DEC_min']) &
                    (table['DEC'] <= bounds['DEC_max']))
        out[cap] = table[cap_mask]
    return out


def mask_table_by_healpix(table, pixel_mask, nside):
    '''
    Filter rows using a HEALPix boolean pixel mask and RA/DEC coordinates.

    Args:
        table: Astropy Table containing RA and DEC columns.
        pixel_mask: Boolean HEALPix map where True pixels are retained.
        nside: HEALPix NSIDE for pixel_mask.
    Returns:
        Table subset containing only rows inside the mask.
    '''
    if len(table) == 0:
        return table

    ra = np.asarray(table['RA'], dtype=np.float64)
    dec = np.asarray(table['DEC'], dtype=np.float64)
    valid = np.isfinite(ra) & np.isfinite(dec)
    keep = np.zeros(len(table), dtype=bool)

    if np.any(valid):
        theta = np.radians(90.0 - dec[valid])
        phi = np.radians(np.mod(ra[valid], 360.0))
        pix = _ang2pix(nside, theta, phi)
        keep[valid] = pixel_mask[pix]

    return table[keep]


def split_caps_by_healpix_masks(table, masks_by_cap, nside, caps=DEFAULT_CAPS_DR1):
    '''
    Split a table into DR1 NGC/SGC caps using HEALPix masks.

    Args:
        table: Astropy Table containing RA and DEC columns.
        masks_by_cap: Dict mapping cap labels to boolean HEALPix masks.
        nside: HEALPix NSIDE for the masks.
        caps: Cap labels to return.
    Returns:
        Dict with one masked table per cap.
    '''
    out = {}
    for cap in _normalize_caps(caps):
        if cap not in masks_by_cap:
            raise KeyError(f'Missing HEALPix mask for DR1 cap {cap}')
        out[cap] = mask_table_by_healpix(table, masks_by_cap[cap], nside)
    return out


def subsample_random(random_table, n_target, rng):
    '''
    Subsample the random table to match the number of data points in the corresponding cap.

    Args:
        random_table: Astropy Table containing the random points.
        n_target: Number of random points to subsample, typically equal to the number of data
              points in the corresponding cap.
        rng: Random number generator instance (e.g., numpy.random.Generator or numpy.random.RandomState)
               to use for reproducibility.
    Returns:
        Subsampled Astropy Table containing n_target random points.
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
                        tracer_aliases=None, mask_dir=None,
                        dr1_masks=None, verbose=True):
    '''
    Load data and random samples for a given tracer, split them into sky caps,
    and subsample the randoms to match the data counts.

    DR1 uses the release HEALPix masks for NGC/SGC. Other releases keep the
    historical RA split unless explicit cap_cuts are provided.

    Args:
        data_dir: Directory where the catalogs are stored.
        tracer: Name of the tracer (e.g., 'BGS_ANY', 'LRG', 'ELGnotqso', 'QSO').
        random_index: Index for the random catalog, used to differentiate between multiple random samples if needed.
        columns: Tuple of column names to read from the catalogs (e.g., ('TARGETID', 'RA', 'DEC', 'Z')).
        ra_min: Minimum RA value to define the NGC boundary.
        ra_max: Maximum RA value to define the NGC boundary.
        rng: Random number generator instance (e.g., numpy.random.Generator or numpy.random.RandomState)
               to use for reproducibility.
        caps: Tuple or list of cap names to process.
        cap_cuts: Optional dict with per-cap RA/DEC bounds for non-DR1 releases.
        release: Data release identifier ('dr2' or 'dr1').
        tracer_aliases: Optional dict mapping logical tracer names to on-disk tracer names.
        mask_dir: Optional DR1 mask directory. ASTRA_DR1_MASK_DIR takes priority.
        dr1_masks: Optional cache from load_dr1_healpix_masks.
        verbose: If True, print the number of data and random points in NGC and SGC after processing.
    '''
    release_norm = _normalize_release(release)
    caps = _normalize_caps(caps)
    data_files, rand_files, tracer_on_disk = build_catalog_paths(data_dir, tracer,
                                                                 random_index=random_index,
                                                                 release=release_norm,
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

    if release_norm == 'dr1':
        if cap_cuts is not None:
            raise ValueError('DR1 no longer supports rectangular cap_cuts; use HEALPix NGC/SGC masks')
        caps = _validate_dr1_caps(caps)
        program = _dr1_mask_program_for_tracer(tracer_on_disk)
        mask_cache = dr1_masks
        if mask_cache is None:
            mask_cache = load_dr1_healpix_masks(mask_dir=mask_dir, caps=caps,
                                                programs=(program,))
        masks_by_cap = mask_cache['masks'][program]
        nside = mask_cache['nside']
        data_caps = split_caps_by_healpix_masks(data_table, masks_by_cap, nside, caps=caps)
        rand_caps = split_caps_by_healpix_masks(rand_table, masks_by_cap, nside, caps=caps)
    elif cap_cuts is None:
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
                            release='dr2', tracer_aliases=None,
                            mask_dir=None, verbose=True):
    '''
    Load data and random samples for all specified tracers, split them into sky caps,
    and subsample the randoms to match the data counts.

    DR1 uses the release HEALPix masks for NGC/SGC. Other releases keep the
    historical RA split unless explicit cap_cuts are provided.

    Args:
        data_dir: Directory where the catalogs are stored.
        tracers: Tuple of tracer names to load (e.g., ('BGS_ANY', 'LRG', 'ELGnotqso', 'QSO')).
        random_index: Index for the random catalog, used to differentiate between multiple random samples if needed.
        columns: Tuple of column names to read from the catalogs (e.g., ('TARGETID', 'RA', 'DEC', 'Z')).
        ra_min: Minimum RA value to define the NGC boundary.
        ra_max: Maximum RA value to define the NGC boundary.
        seed: Random seed for reproducibility when subsampling randoms.
        caps: Tuple or list of cap names to process.
        cap_cuts: Optional dict with per-cap RA/DEC bounds for non-DR1 releases.
        release: Data release identifier ('dr2' or 'dr1').
        tracer_aliases: Optional dict mapping logical tracer names to on-disk tracer names.
        mask_dir: Optional DR1 mask directory. ASTRA_DR1_MASK_DIR takes priority.
        verbose: If True, print the number of data and random points in NGC and SGC for each tracer after processing.
    Returns:
        A dict containing the loaded and processed tables for each tracer and cap, with keys like
          'BGS_ANY_NGC', 'BGS_ANY_SGC', 'BGS_ANY_RAND_NGC', 'BGS_ANY_RAND_SGC', etc.
    '''
    release_norm = _normalize_release(release)
    caps = _normalize_caps(caps)
    if hasattr(np.random, 'default_rng'):
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.RandomState(seed)
    all_data = {}
    dr1_masks = None

    if release_norm == 'dr1':
        if cap_cuts is not None:
            raise ValueError('DR1 no longer supports rectangular cap_cuts; use HEALPix NGC/SGC masks')
        caps = _validate_dr1_caps(caps)
        programs = []
        seen_programs = set()
        for tracer in tracers:
            tracer_on_disk = tracer_aliases.get(tracer, tracer) if tracer_aliases else tracer
            program = _dr1_mask_program_for_tracer(tracer_on_disk)
            if program not in seen_programs:
                programs.append(program)
                seen_programs.add(program)
        dr1_masks = load_dr1_healpix_masks(mask_dir=mask_dir, caps=caps,
                                           programs=programs)
        if verbose:
            print(f'[dr1] using HEALPix masks from {dr1_masks["mask_dir"]} '
                  f'nside={dr1_masks["nside"]}')
            for program in programs:
                for cap in caps:
                    pix = int(dr1_masks['masks'][program][cap].sum())
                    path = dr1_masks['paths'][program][cap]
                    print(f'[dr1] mask {program}/{cap}: pixels={pix} path={path}')

    for tracer in tracers:
        tracer_tables = load_tracer_samples(data_dir=data_dir, tracer=tracer,
                                            random_index=random_index, columns=columns,
                                            ra_min=ra_min, ra_max=ra_max, rng=rng,
                                            caps=caps, cap_cuts=cap_cuts,
                                            release=release_norm,
                                            tracer_aliases=tracer_aliases,
                                            mask_dir=mask_dir,
                                            dr1_masks=dr1_masks,
                                            verbose=verbose)
        all_data.update(tracer_tables)

    return all_data


def validate_required_keys(all_data, tracers, caps):
    '''
    Validate that all required keys for the specified tracers and caps are present in the all_data dict.

    Args:
        all_data: Dict containing the loaded data tables for each tracer and cap.
        tracers: Tuple of tracer names (e.g., ('BGS_ANY', 'LRG', 'ELGnotqso', 'QSO')).
        caps: Tuple of cap names (e.g., ('NGC', 'SGC')).
    Raises:
        KeyError: If any required key is missing from the all_data dict, with a
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