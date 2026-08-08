import fcntl, glob, json, os, re
import astropy.units as u
import healpy as hp
import numpy as np
from argparse import Namespace
from astropy.coordinates import SkyCoord
from astropy.cosmology import Planck18
from astropy.io import fits
import fitsio
from astropy.table import Column, Table, vstack

from desiproc.implement_astra import register_tracer_mapping
from desiproc.paths import safe_tag, zone_tag

from .dr2 import build_raw_dr2_zone

from .base import ReleaseConfig


TRACERS = ['BGS_ANY', 'BGS_BRIGHT', 'ELG_LOPnotqso', 'LRG', 'QSO']
LOCAL_TRACERS = ['BGS_ANY', 'BGS_BRIGHT', 'BGS_BRIGHT-21.5',
                 'ELG_LOPnotqso', 'LRG', 'QSO']
REAL_SUFFIX = {'N': '_NGC_clustering.dat.fits', 'S': '_SGC_clustering.dat.fits'}
RANDOM_SUFFIX = {'N': '_NGC_{i}_clustering.ran.fits', 'S': '_SGC_{i}_clustering.ran.fits'}
N_RANDOM_FILES = 18
REAL_COLUMNS = ['TARGETID', 'RA', 'DEC', 'Z']
RANDOM_COLUMNS = REAL_COLUMNS
DEFAULT_ZONES = ['NGC', 'SGC']
ZONE_ALIASES = {'NGC': 'NGC', 'SGC': 'SGC'}
ZONE_VALUES = {'NGC': 1001, 'SGC': 1002}
TRACER_ALIAS = {'bgs': 'BGS_BRIGHT', 'bgs_any': 'BGS_ANY',
                'bgs_bright': 'BGS_BRIGHT', 'elg': 'ELG_LOPnotqso',
                'lrg': 'LRG', 'qso': 'QSO'}
LOCAL_TRACER_ALIAS = {'bgs': 'BGS_BRIGHT',
                      'bgs_any': 'BGS_ANY',
                      'bgs_bright': 'BGS_BRIGHT',
                      'bgs_bright-21.5': 'BGS_BRIGHT-21.5',
                      'bgs_bright_21.5': 'BGS_BRIGHT-21.5',
                      'elg': 'ELG_LOPnotqso',
                      'elg_lopnotqso': 'ELG_LOPnotqso',
                      'lrg': 'LRG',
                      'qso': 'QSO'}
TRACER_MASK_PROGRAM = {'BGS_BRIGHT': 'bright',
                       'ELG_LOPnotqso': 'dark',
                       'LRG': 'dark',
                       'QSO': 'dark'}
MASK_PROGRAMS = ('bright', 'dark')
MASK_ZONE_SUFFIX = {'NGC': 'ngc', 'SGC': 'sgc'}
MASK_NSIDE_RE = re.compile(r'_nside(?P<nside>\d+)_')
EMLINE_CATALOG_PATH = ('/global/cfs/cdirs/desi/public/dr1/vac/dr1/stellar-mass-emline/'
                       'v1.0/dr1_galaxy_stellarmass_lineinfo_v1.0.fits')
EMLINE_REQUIRED_COLUMNS = ('TARGETID', 'ZERR', 'FLUX_G', 'FLUX_R')
EMLINE_OUTPUT_MAP = {'SED_SFR': ('SED_SFR', 'SFR_CG'),
                     'SED_MASS': ('SED_MASS', 'MASS_CG'),
                     'FLUX_G': ('FLUX_G',),
                     'FLUX_R': ('FLUX_R',)}
PROPERTY_COLUMNS = ('TARGETID', 'SED_SFR', 'SED_MASS', 'FLUX_G', 'FLUX_R')
_EMLINE_BEST_CACHE = None


def _float_with_nan(column):
    """
    Convert an input column to float64, replacing masked values with NaN.

    Args:
        column: Input column, which can be a masked array or regular array.
    Returns:
        A numpy array of type float64 with masked values replaced by NaN.
    """
    if np.ma.isMaskedArray(column):
        return np.asarray(np.ma.filled(column, np.nan), dtype=np.float64)
    arr = np.asarray(column)
    return np.asarray(arr, dtype=np.float64)


def _load_emline_best(catalog_path=EMLINE_CATALOG_PATH, targetids=None):
    """
    Load the DR1 emline catalogue and keep one row per TARGETID with minimum ZERR.

    Args:
        catalog_path: Path to the DR1 emline catalogue FITS file.
        targetids: Optional TARGETIDs to retain before sorting/deduplicating. This
            avoids materialising irrelevant rows from the 49 GB VAC.
    Returns:
        A table containing the best emline entries per TARGETID.
    Raises:
        FileNotFoundError: If the catalogue file does not exist.
        KeyError: If required columns are missing from the catalogue.
    """
    global _EMLINE_BEST_CACHE
    if targetids is None and _EMLINE_BEST_CACHE is not None:
        return _EMLINE_BEST_CACHE

    if not os.path.exists(catalog_path):
        raise FileNotFoundError(f'DR1 emline catalogue not found: {catalog_path}')

    desired = None
    if targetids is not None:
        desired = np.unique(np.asarray(targetids, dtype=np.int64))

    with fits.open(catalog_path, memmap=True, lazy_load_hdus=True) as hdul:
        available = list(hdul[1].columns.names)
        missing = [name for name in EMLINE_REQUIRED_COLUMNS if name not in available]
        if missing:
            raise KeyError(f'DR1 emline catalogue missing columns: {missing}')

        optional_cols = []
        for candidates in EMLINE_OUTPUT_MAP.values():
            for name in candidates:
                if name in available:
                    optional_cols.append(name)

        selected_cols = list(EMLINE_REQUIRED_COLUMNS)
        for name in optional_cols:
            if name not in selected_cols:
                selected_cols.append(name)

    row_index = None
    if desired is not None:
        if fitsio is not None:
            target_rows = fitsio.read(catalog_path, ext=1, columns=['TARGETID'])
            vac_targetid = np.asarray(target_rows['TARGETID'], dtype=np.int64)
        else:
            with fits.open(catalog_path, memmap=True, lazy_load_hdus=True) as hdul:
                vac_targetid = np.asarray(hdul[1].data['TARGETID'], dtype=np.int64)
        positions = np.searchsorted(desired, vac_targetid, side='left')
        matched = positions < desired.size
        matched[matched] &= desired[positions[matched]] == vac_targetid[matched]
        row_index = np.flatnonzero(matched)

    if fitsio is not None:
        read_kwargs = {'ext': 1, 'columns': selected_cols}
        if row_index is not None:
            read_kwargs['rows'] = row_index
        emline = Table(fitsio.read(catalog_path, **read_kwargs))
    else:
        fits_rows = slice(None) if row_index is None else row_index
        with fits.open(catalog_path, memmap=True, lazy_load_hdus=True) as hdul:
            data = hdul[1].data
            emline = Table()
            for name in selected_cols:
                emline[name] = np.asarray(data[name][fits_rows])

    if len(emline) == 0:
        if targetids is None:
            _EMLINE_BEST_CACHE = emline
        return emline

    score = _float_with_nan(emline['ZERR'])
    score = np.where(np.isfinite(score), score, np.inf)
    order = np.lexsort((score, np.asarray(emline['TARGETID'], dtype=np.int64)))
    emline_sorted = emline[order]

    targetid_sorted = np.asarray(emline_sorted['TARGETID'], dtype=np.int64)
    keep = np.ones(len(emline_sorted), dtype=bool)
    keep[1:] = targetid_sorted[1:] != targetid_sorted[:-1]
    result = emline_sorted[keep]
    if targetids is None:
        _EMLINE_BEST_CACHE = result

    print(f'[dr1] emline rows={len(emline)} unique-targetid={len(result)}', flush=True)
    return result


def _properties_for_targetids(targetids, emline_best):
    """Return the requested DR1 properties for sorted, unique TARGETIDs."""
    targetids = np.unique(np.asarray(targetids, dtype=np.int64))
    result = Table()
    result['TARGETID'] = targetids

    best_tid = np.asarray(emline_best['TARGETID'], dtype=np.int64)
    idx = np.searchsorted(best_tid, targetids, side='left')
    valid = idx < best_tid.size
    valid[valid] &= best_tid[idx[valid]] == targetids[valid]

    mapping_used = {}
    for out_name, candidates in EMLINE_OUTPUT_MAP.items():
        src_name = next((name for name in candidates if name in emline_best.colnames), None)
        values_out = np.full(targetids.size, np.nan, dtype=np.float64)
        if src_name is not None:
            values_in = _float_with_nan(emline_best[src_name])
            values_out[valid] = values_in[idx[valid]]
            mapping_used[out_name] = src_name
        else:
            mapping_used[out_name] = 'nan'
        result[out_name] = values_out

    print(f'[dr1] property matches={int(valid.sum())}/{len(result)} mapping={mapping_used}',
          flush=True)
    return result[list(PROPERTY_COLUMNS)]


def _merge_properties(existing, incoming):
    """Merge two property tables by TARGETID, preferring finite incoming values."""
    old_tid = np.asarray(existing['TARGETID'], dtype=np.int64)
    new_tid = np.asarray(incoming['TARGETID'], dtype=np.int64)
    all_tid = np.union1d(old_tid, new_tid)

    merged = Table()
    merged['TARGETID'] = all_tid
    old_pos = np.searchsorted(all_tid, old_tid)
    new_pos = np.searchsorted(all_tid, new_tid)
    for name in PROPERTY_COLUMNS[1:]:
        values = np.full(all_tid.size, np.nan, dtype=np.float64)
        old_values = _float_with_nan(existing[name])
        new_values = _float_with_nan(incoming[name])
        values[old_pos] = old_values
        finite_new = np.isfinite(new_values)
        values[new_pos[finite_new]] = new_values[finite_new]
        merged[name] = values
    return merged


def _read_real_targetids(base_dir, tracers, zone_label):
    """Read and deduplicate TARGETIDs from native DR1 real catalogues."""
    parts = []
    for tracer in tracers:
        path = os.path.join(base_dir, f'{tracer}_{zone_label}_clustering.dat.fits')
        if not os.path.exists(path):
            raise FileNotFoundError(f'DR1 real catalogue not found: {path}')
        with fits.open(path, memmap=True, lazy_load_hdus=True) as hdul:
            names = list(hdul[1].data.columns.names)
            if 'TARGETID' not in names:
                raise KeyError(f'DR1 real catalogue missing TARGETID: {path}')
            parts.append(np.asarray(hdul[1].data['TARGETID'], dtype=np.int64).copy())
    if not parts:
        return np.empty(0, dtype=np.int64)
    return np.unique(np.concatenate(parts))


def write_zone_properties(base_dir, properties_root, zone_label, tracers,
                          release_tag='DR1', catalog_path=EMLINE_CATALOG_PATH):
    """
    Create/update ``properties/zone_REGION_properties.fits.gz`` for real DR1 rows.

    Successive per-tracer executions are merged under an advisory file lock, so a
    regional file contains the union of all TARGETIDs processed so far.
    """
    zone_label = _normalize_zone_label(zone_label)
    properties_dir = os.path.join(properties_root, 'properties')
    os.makedirs(properties_dir, exist_ok=True)
    output_path = os.path.join(properties_dir, f'zone_{zone_label}_properties.fits.gz')
    lock_path = output_path + '.lock'
    requested_tid = _read_real_targetids(base_dir, tracers, zone_label)

    with open(lock_path, 'a', encoding='utf-8') as lock_handle:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        existing = None
        missing_tid = requested_tid
        if os.path.exists(output_path):
            existing = Table.read(output_path, memmap=True)
            missing_cols = [name for name in PROPERTY_COLUMNS if name not in existing.colnames]
            if missing_cols:
                raise KeyError(f'Existing DR1 properties file missing columns: {missing_cols}')
            existing = existing[list(PROPERTY_COLUMNS)]
            existing_tid = np.unique(np.asarray(existing['TARGETID'], dtype=np.int64))
            missing_tid = np.setdiff1d(requested_tid, existing_tid, assume_unique=True)

        if missing_tid.size == 0:
            print(f'[dr1] reuse complete properties {output_path}', flush=True)
            return output_path

        emline_best = _load_emline_best(catalog_path, targetids=missing_tid)
        incoming = _properties_for_targetids(missing_tid, emline_best)
        properties = incoming if existing is None else _merge_properties(existing, incoming)
        properties.meta['ZONE'] = zone_label
        properties.meta['RELEASE'] = str(release_tag)

        tmp_path = f'{output_path}.tmp.{os.getpid()}.fits.gz'
        try:
            properties.write(tmp_path, format='fits', overwrite=True)
            os.replace(tmp_path, output_path)
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

        print(f'[dr1] wrote properties rows={len(properties)} path={output_path}', flush=True)
        return output_path


def _append_emline_columns(raw_table, emline_best):
    """
    Add SED_SFR, SED_MASS, FLUX_G and FLUX_R to raw rows by TARGETID.

    Args:
        raw_table: The input raw table to enrich.
        emline_best: The table containing the best emline entries per TARGETID.
    Returns:
        The enriched raw table with emline columns added.
    Raises:
        KeyError: If 'TARGETID' is missing from the raw table or required
        emline columns are missing from the emline table.
    """
    if 'TARGETID' not in raw_table.colnames:
        raise KeyError("Raw table does not contain 'TARGETID'")

    raw_tid = np.asarray(raw_table['TARGETID'], dtype=np.int64)
    best_tid = np.asarray(emline_best['TARGETID'], dtype=np.int64)

    idx = np.searchsorted(best_tid, raw_tid, side='left')
    valid = idx < best_tid.size
    valid[valid] &= best_tid[idx[valid]] == raw_tid[valid]

    mapping_used = {}

    for out_name, candidates in EMLINE_OUTPUT_MAP.items():
        src_name = None
        for cand in candidates:
            if cand in emline_best.colnames:
                src_name = cand
                break

        out = np.full(len(raw_table), np.nan, dtype=np.float64)
        if src_name is not None:
            values = _float_with_nan(emline_best[src_name])
            out[valid] = values[idx[valid]]
            mapping_used[out_name] = src_name
        else:
            mapping_used[out_name] = 'nan'

        if out_name in raw_table.colnames:
            raw_table.remove_column(out_name)
        raw_table[out_name] = out

    print(f'[dr1] enriched raw with emline columns matches={int(valid.sum())}/{len(raw_table)}', flush=True)
    print(f'[dr1] emline mapping: {mapping_used}', flush=True)
    return raw_table


def _normalize_zone_label(zone):
    """
    Normalize a user-provided DR1 zone token to ``NGC`` or ``SGC``.

    Args:
        zone: Input zone token.
    Returns:
        str: Normalized zone label.
    Raises:
        RuntimeError: If the label is unknown.
    """
    key = str(zone).strip().upper()
    label = ZONE_ALIASES.get(key)
    if label is None:
        known = ', '.join(sorted(ZONE_ALIASES))
        raise RuntimeError(f'Unknown DR1 zone "{zone}". Allowed labels: {known}')
    return label


def _compute_cartesian(tbl, dtype=np.float64):
    """
    Add Cartesian coordinates (XCART/YCART/ZCART) to ``tbl``.

    Args:
        tbl: Input table with RA, DEC, Z columns.
        dtype: Desired data type for the Cartesian columns.
    Returns:
        Table: The input table with added Cartesian coordinate columns.
    """
    z = np.asarray(tbl['Z'], dtype=float)
    dist = Planck18.comoving_distance(z)
    ra = np.asarray(tbl['RA'], dtype=float) * u.deg
    dec = np.asarray(tbl['DEC'], dtype=float) * u.deg
    sc = SkyCoord(ra=ra, dec=dec, distance=dist)
    tbl['XCART'] = np.asarray(sc.cartesian.x.value, dtype=dtype)
    tbl['YCART'] = np.asarray(sc.cartesian.y.value, dtype=dtype)
    tbl['ZCART'] = np.asarray(sc.cartesian.z.value, dtype=dtype)
    return tbl


def _ensure_zone_column(tbl, zone_value):
    """
    Overwrite/create ``ZONE`` column with a constant synthetic zone value.

    Args:
        tbl: Input table to modify.
        zone_value: Integer value to assign to the ZONE column.
    Returns:
        Table: The input table with the ZONE column set to the specified value.
    """
    if 'ZONE' in tbl.colnames:
        tbl.remove_column('ZONE')
    tbl.add_column(Column(np.full(len(tbl), int(zone_value), dtype=np.int32), name='ZONE'))
    return tbl


def _extract_nside_from_path(path):
    """
    Return the NSIDE encoded in a DR1 mask filename.

    Args:
        path (str): The path to the mask file.
    Returns:
        int: The NSIDE value or -1 if not found.
    """
    match = MASK_NSIDE_RE.search(os.path.basename(path))
    if match is None:
        return -1
    return int(match.group('nside'))


def _resolve_mask_dir(parsed_args, user_cfg):
    """
    Resolve the DR1 mask directory.
    Priority: ``ASTRA_DR1_MASK_DIR`` env > config ``mask_dir`` > sibling of ``raw_out``.

    Args:
        parsed_args: Parsed command line arguments.
        user_cfg: User configuration dictionary loaded from JSON config file.
    Returns:
        str: The resolved absolute path to the DR1 mask directory.
    """
    env_mask_dir = os.environ.get('ASTRA_DR1_MASK_DIR')
    if env_mask_dir:
        return os.path.abspath(os.path.expanduser(env_mask_dir))

    cfg_mask_dir = user_cfg.get('mask_dir')
    if isinstance(cfg_mask_dir, str) and cfg_mask_dir.strip():
        return os.path.abspath(os.path.expanduser(cfg_mask_dir))

    raw_parent = os.path.abspath(os.path.join(parsed_args.raw_out, os.pardir))
    return os.path.join(raw_parent, 'masks', 'bright_dark')


def _load_dr1_masks(mask_dir):
    """
    Load bright/dark NGC/SGC HEALPix masks generated by dr1_mask.py.

    Args:
        mask_dir: Directory containing the DR1 mask files.
    Returns:     A tuple (masks, paths, nside) where:
        - masks: A nested dictionary of boolean arrays with structure masks[program][zone_label].
        - paths: A nested dictionary of file paths with structure paths[program][zone_label].
        - nside: The NSIDE value of the loaded HEALPix maps (must be consistent across all masks).
    Raises:
        FileNotFoundError: If no mask files are found for a program/zone combination.
        RuntimeError: If the loaded masks have inconsistent NSIDE or npix values.
    """
    masks = {program: {} for program in MASK_PROGRAMS}
    paths = {program: {} for program in MASK_PROGRAMS}
    expected_nside = None
    expected_npix = None

    for program in MASK_PROGRAMS:
        for zone_label, zone_suffix in MASK_ZONE_SUFFIX.items():
            pattern = os.path.join(mask_dir, f'dr1_mask_{program}_nside*_{zone_suffix}.fits')
            candidates = glob.glob(pattern)
            if not candidates:
                raise FileNotFoundError(f'No DR1 mask file matches {pattern}')
            selected = max(candidates, key=_extract_nside_from_path)
            values = hp.read_map(selected, dtype=np.int16)
            arr = np.asarray(values)
            nside = hp.get_nside(arr)
            npix = arr.size

            if expected_nside is None:
                expected_nside = nside
                expected_npix = npix
            elif (nside != expected_nside) or (npix != expected_npix):
                raise RuntimeError('DR1 mask maps have inconsistent NSIDE/npix')

            mask_bool = arr > 0
            masks[program][zone_label] = mask_bool
            paths[program][zone_label] = selected

    return masks, paths, expected_nside


def _mask_table_rows(tbl, pixel_mask, nside):
    """
    Filter rows by HEALPix pixel mask using RA/DEC.

    Args:
        tbl: Input table with 'RA' and 'DEC' columns.
        pixel_mask: Boolean array where True indicates pixels to keep.
        nside: NSIDE of the HEALPix pixelization used for the mask.
    Returns:
        Table: A subset of the input table containing only rows that fall within the True pixels of the mask.
    """
    if len(tbl) == 0:
        return tbl

    ra = np.asarray(tbl['RA'], dtype=np.float64)
    dec = np.asarray(tbl['DEC'], dtype=np.float64)
    valid = np.isfinite(ra) & np.isfinite(dec)
    keep = np.zeros(len(tbl), dtype=bool)

    if np.any(valid):
        theta = np.radians(90.0 - dec[valid])
        phi = np.radians(np.mod(ra[valid], 360.0))
        pix = hp.ang2pix(nside, theta, phi)
        keep[valid] = pixel_mask[pix]

    return tbl[keep]


def _collect_real_region_table(real_tables, tracer, region):
    """
    Merge hemisphere real tables into one region table.

    Args:
        real_tables: Dictionary with real tables per tracer and hemisphere.
        tracer: Tracer name to collect.
        region: Region label (e.g. 'N', 'S', 'ALL').
    Returns:
        Table: The combined real table for the specified tracer and region.
    Raises:
        KeyError: If no data is found for the specified tracer and region.
    """
    region = str(region).upper()
    if region == 'ALL':
        parts = []
        for hemi in ('N', 'S'):
            tbl = real_tables[tracer].get(hemi)
            if tbl is not None:
                parts.append(tbl)
        if not parts:
            raise KeyError(f'No data for tracer {tracer} in any hemisphere')
        return vstack(parts, metadata_conflicts='silent') if len(parts) > 1 else parts[0]
    return real_tables[tracer][region]


def _collect_random_region_tables(random_tables, tracer, region):
    """
    Collect random tables for one tracer and region.

    Args:
        random_tables: Dictionary with random tables per tracer and hemisphere.
        tracer: Tracer name to collect.
        region: Region label (e.g. 'N', 'S', 'ALL').
    Returns:
        list of Table: A list of random tables for the specified tracer and region.
    """
    region = str(region).upper()
    if region == 'ALL':
        tables = []
        hemi_dict = random_tables[tracer]
        for hemi in ('N', 'S'):
            tables.extend(list(hemi_dict.get(hemi, {}).values()))
        return tables
    return list(random_tables[tracer][region].values())


def _process_real_region_masked(real_tables, tracer, region, pixel_mask, nside, zone_value):
    """
    Return masked DR1 real table for one tracer.

    Args:
        real_tables: Dictionary with real tables per tracer and hemisphere.
        tracer: Tracer name to process.
        region: Region label (e.g. 'N', 'S', 'ALL').
        pixel_mask: Boolean array where True indicates pixels to keep.
        nside: NSIDE of the HEALPix pixelization used for the mask.
        zone_value: Integer value to assign to the ZONE column.
    Returns:
        Table: The processed real table for the specified tracer and region,
                with HEALPix mask applied and Cartesian coordinates computed.
    """
    base_tbl = _collect_real_region_table(real_tables, tracer, region)
    sel = _mask_table_rows(base_tbl, pixel_mask, nside)
    if len(sel) == 0:
        raise ValueError(f'No entries for {tracer} in region {region} after HEALPix mask')
    sel = _ensure_zone_column(sel.copy(), zone_value)
    sel = _compute_cartesian(sel)
    sel['TRACERTYPE'] = tracer
    sel['RANDITER'] = np.full(len(sel), -1, dtype=np.int32)
    return sel


def _generate_randoms_region_masked(random_tables, tracer, region, pixel_mask, nside,
                                    n_random, real_count, zone_value):
    """
    Return random catalogues sampled from the masked DR1 random pool.

    Args:
        random_tables: Dictionary with random tables per tracer and hemisphere.
        tracer: Tracer name to process.
        region: Region label (e.g. 'N', 'S', 'ALL').
        pixel_mask: Boolean array where True indicates pixels to keep.
        nside: NSIDE of the HEALPix pixelization used for the mask.
        n_random: Number of random catalogues to generate.
        real_count: Number of random entries to sample per catalogue (should match the real count).
        zone_value: Integer value to assign to the ZONE column.
    Returns:
        Table: The combined random table for the specified tracer and region, with HEALPix
               mask applied and Cartesian coordinates computed.
     Raises:
        KeyError: If no random tables are found for the specified tracer and region.
        ValueError: If no random entries remain after masking or if the available
                    randoms are fewer than the real count.
    """
    tables = _collect_random_region_tables(random_tables, tracer, region)
    if not tables:
        raise KeyError(f'No random tables for {tracer} in region {region}')

    zone_tables = []
    total_after_mask = 0
    for tbl in tables:
        sel = _mask_table_rows(tbl, pixel_mask, nside)
        if len(sel) == 0:
            continue
        sel = _ensure_zone_column(sel.copy(), zone_value)
        zone_tables.append(sel)
        total_after_mask += len(sel)

    if total_after_mask == 0:
        raise ValueError(f'No random entries for {tracer} in region {region} after HEALPix mask')
    if total_after_mask < real_count:
        raise ValueError(f'Region {region} randoms have only {total_after_mask} points after mask (< {real_count})')

    zone_tables_xyz = []
    for sel in zone_tables:
        zone_tables_xyz.append(_compute_cartesian(sel.copy()))
    pool = vstack(zone_tables_xyz, metadata_conflicts='silent')

    samples = []
    for j in range(n_random):
        rng = np.random.default_rng(j)
        rows = rng.choice(len(pool), real_count, replace=False)
        samp = pool[rows]
        samp['TRACERTYPE'] = tracer
        samp['RANDITER'] = np.full(len(samp), j, dtype=np.int32)
        samples.append(samp)

    return vstack(samples, metadata_conflicts='silent')


def build_raw_region(zone_label, region, tracers, real_tables, random_tables,
                     output_raw, n_random, zone_value, out_tag, release_tag,
                     zone_masks, nside):
    """
    Build and persist the DR1 raw table for ``zone_label`` applying HEALPix masks.

    Args:
        zone_label: Label for the zone being processed.
        region: Region label (e.g. 'N', 'S', 'ALL').
        tracers: List of tracers to process.
        real_tables: Dictionary with real tables per tracer.
        random_tables: Dictionary with random tables per tracer.
        output_raw: Path to the output raw directory.
        n_random: Number of randoms per data object.
        zone_value: Integer value to assign to the ZONE column.
        out_tag: Optional tag to append to the output file name.
        release_tag: Release tag string or None.
        zone_masks: Mapping ``{'bright': bool[npix], 'dark': bool[npix]}``.
        nside: HEALPix NSIDE for ``zone_masks``.
    Returns:
        Table: The combined table written to disk.
    """
    parts = []
    skipped = []
    for tr in tracers:
        program = TRACER_MASK_PROGRAM.get(tr)
        if program is None:
            raise RuntimeError(f'No DR1 mask program configured for tracer {tr}')
        pixel_mask = zone_masks[program]

        try:
            rt = _process_real_region_masked(real_tables, tr, region, pixel_mask, nside, zone_value=zone_value)
        except ValueError as exc:
            print(f'[warn] {tr} empty after mask in zone {zone_label} ({program}): {exc}')
            skipped.append(tr)
            continue
        parts.append(rt)
        count = len(rt)
        rpt = _generate_randoms_region_masked(random_tables, tr, region, pixel_mask, nside,
                                              n_random, count, zone_value=zone_value)
        parts.append(rpt)

    if not parts:
        raise ValueError(f'No data in region {region} for zone {zone_label} (tracers tried: {tracers})')

    tbl = vstack(parts)
    if 'RANDITER' in tbl.colnames:
        tbl['RANDITER'] = np.asarray(tbl['RANDITER'], dtype=np.int32)
    tbl = _append_emline_columns(tbl, _load_emline_best())

    tag_suffix = safe_tag(out_tag)
    out_path = os.path.join(output_raw, f'zone_{zone_label}{tag_suffix}.fits.gz')
    tmp_path = out_path + '.tmp'

    tbl_out = tbl.copy()
    if 'ZONE' in tbl_out.colnames:
        tbl_out.remove_column('ZONE')

    tbl_out.meta['ZONE'] = zone_tag(zone_label)
    tbl_out.meta['RELEASE'] = str(release_tag) if release_tag is not None else 'UNKNOWN'

    tbl_out.write(tmp_path, format='fits', overwrite=True)
    os.replace(tmp_path, out_path)

    if skipped:
        print(f'[info] In {zone_label} skipped tracers (empty): {", ".join(skipped)}')
    return tbl


def create_config(args):
    """
    Create the release configuration from command line arguments.

    Args:
        args: Parsed command line arguments.
    Returns:
        The release configuration object.
    """
    user_cfg = {}
    if args.config:
        with open(args.config, 'r', encoding='utf-8') as handle:
            loaded = json.load(handle)
        if not isinstance(loaded, dict):
            raise RuntimeError('--config for DR1 must be a JSON object')
        user_cfg = loaded

    if args.zones is not None:
        zones = [_normalize_zone_label(z) for z in args.zones]
    elif isinstance(user_cfg.get('zones'), list):
        zones = [_normalize_zone_label(z) for z in user_cfg['zones']]
    else:
        zones = list(DEFAULT_ZONES)

    dedup = []
    seen = set()
    for zone in zones:
        if zone in seen:
            continue
        seen.add(zone)
        dedup.append(zone)
    zones = dedup

    use_extended_tracers = bool(getattr(args, 'local_zone_files', False)
                                or user_cfg.get('local_zone_files', False))
    available_tracers = LOCAL_TRACERS if use_extended_tracers else TRACERS
    tracer_alias = LOCAL_TRACER_ALIAS if use_extended_tracers else TRACER_ALIAS
    tracer_ids = {name: idx for idx, name in enumerate(available_tracers)}
    tracer_full_labels = {}
    for tracer_name, tracer_idx in tracer_ids.items():
        tracer_full_labels[(tracer_idx, True)] = tracer_name.encode('ascii')
        tracer_full_labels[(tracer_idx, False)] = tracer_name.encode('ascii')
    register_tracer_mapping(tracer_ids, tracer_full_labels)

    def _build(zone, real_tables, random_tables, sel_tracers,
               parsed_args, release_tag):
        label = _normalize_zone_label(zone)
        zone_value = ZONE_VALUES.get(label, 1999)
        raw = build_raw_dr2_zone(
            label, sel_tracers, real_tables, random_tables,
            parsed_args.raw_out, parsed_args.n_random, zone_value,
            out_tag=parsed_args.out_tag, release_tag=release_tag,
            tracer_ids=tracer_ids, tracer_full_labels=tracer_full_labels,
            log_label='dr1')
        property_tracers = [
            tracer for tracer in available_tracers
            if os.path.exists(os.path.join(
                parsed_args.base_dir, f'{tracer}_{label}_clustering.dat.fits'))
        ]
        if not property_tracers:
            property_tracers = list(sel_tracers)
        write_zone_properties(parsed_args.base_dir, parsed_args.class_out, label,
                              property_tracers, release_tag=release_tag)
        return raw

    preload_kwargs = {
        'real_template': '{tracer}_{zone}_clustering.dat.fits',
        'random_template': '{tracer}_{zone}_{idx}_clustering.ran.fits',
        'log_label': 'dr1',
        'zones_to_keep': zones,
    }
    return ReleaseConfig(
        name='DR1', release_tag='DR1', tracers=available_tracers,
        tracer_alias=tracer_alias, real_suffix=None, random_suffix=None,
        n_random_files=N_RANDOM_FILES, real_columns=REAL_COLUMNS,
        random_columns=RANDOM_COLUMNS, use_dr2_preload=True,
        preload_kwargs=preload_kwargs, zones=zones,
        build_raw=_build, combine_outputs=False)