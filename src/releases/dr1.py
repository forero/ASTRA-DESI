import json, os
import numpy as np
from argparse import Namespace
from astropy.table import Table, vstack

from desiproc.read_data import generate_randoms_region, process_real_region
from desiproc.paths import safe_tag, zone_tag

from .base import ReleaseConfig


TRACERS = ['BGS_BRIGHT', 'ELG_LOPnotqso', 'LRG', 'QSO']
REAL_SUFFIX = {'N': '_N_clustering.dat.fits', 'S': '_S_clustering.dat.fits'}
RANDOM_SUFFIX = {'N': '_N_{i}_clustering.ran.fits', 'S': '_S_{i}_clustering.ran.fits'}
N_RANDOM_FILES = 18
REAL_COLUMNS = ['TARGETID', 'RA', 'DEC', 'Z']
RANDOM_COLUMNS = REAL_COLUMNS
DEFAULT_CUTS = {'NGC1': {'RA_min': 110, 'RA_max': 260, 'DEC_min': -10, 'DEC_max': 8},
                'NGC2': {'RA_min': 180, 'RA_max': 260, 'DEC_min': 30, 'DEC_max': 40}}
ZONE_VALUES = {'NGC1': 1001, 'NGC2': 1002}
TRACER_ALIAS = {'bgs': 'BGS_BRIGHT', 'elg': 'ELG_LOPnotqso', 'lrg': 'LRG', 'qso': 'QSO'}
EMLINE_CATALOG_PATH = ('/global/cfs/cdirs/desi/public/dr1/vac/dr1/stellar-mass-emline/'
                       'v1.0/dr1_galaxy_stellarmass_lineinfo_v1.0.fits')
EMLINE_REQUIRED_COLUMNS = ('TARGETID', 'ZERR', 'FLUX_G', 'FLUX_R')
EMLINE_OUTPUT_MAP = {'SED_SFR': ('SED_SFR', 'SFR_CG'),
                     'SED_MASS': ('SED_MASS', 'MASS_CG'),
                     'FLUX_G': ('FLUX_G',),
                     'FLUX_R': ('FLUX_R',)}
_EMLINE_BEST_CACHE = None


def _float_with_nan(column):
    """
    Convert an input column to float64, replacing masked values with NaN.

    Args:
        column: Input column, which can be a masked array or regular array.
    Returns:
        A numpy array of type float64 with masked values replaced by NaN.
    """
    arr = np.asarray(column)
    if np.ma.isMaskedArray(arr):
        return np.asarray(arr.filled(np.nan), dtype=np.float64)
    return np.asarray(arr, dtype=np.float64)


def _load_emline_best(catalog_path=EMLINE_CATALOG_PATH):
    """
    Load the DR1 emline catalogue and keep one row per TARGETID with minimum ZERR.

    Args:
        catalog_path: Path to the DR1 emline catalogue FITS file.
    Returns:
        A table containing the best emline entries per TARGETID.
    Raises:
        FileNotFoundError: If the catalogue file does not exist.
        KeyError: If required columns are missing from the catalogue.
    """
    global _EMLINE_BEST_CACHE
    if _EMLINE_BEST_CACHE is not None:
        return _EMLINE_BEST_CACHE

    if not os.path.exists(catalog_path):
        raise FileNotFoundError(f'DR1 emline catalogue not found: {catalog_path}')

    emline = Table.read(catalog_path, memmap=True)
    missing = [name for name in EMLINE_REQUIRED_COLUMNS if name not in emline.colnames]
    if missing:
        raise KeyError(f'DR1 emline catalogue missing columns: {missing}')

    optional_cols = []
    for candidates in EMLINE_OUTPUT_MAP.values():
        for name in candidates:
            if name in emline.colnames:
                optional_cols.append(name)

    selected_cols = list(EMLINE_REQUIRED_COLUMNS)
    for name in optional_cols:
        if name not in selected_cols:
            selected_cols.append(name)
    emline = emline[selected_cols]
    if len(emline) == 0:
        _EMLINE_BEST_CACHE = emline
        return _EMLINE_BEST_CACHE

    score = _float_with_nan(emline['ZERR'])
    order = np.lexsort((score, np.asarray(emline['TARGETID'], dtype=np.int64)))
    emline_sorted = emline[order]

    targetid_sorted = np.asarray(emline_sorted['TARGETID'], dtype=np.int64)
    keep = np.ones(len(emline_sorted), dtype=bool)
    keep[1:] = targetid_sorted[1:] != targetid_sorted[:-1]
    _EMLINE_BEST_CACHE = emline_sorted[keep]

    print(f'[dr1] emline rows={len(emline)} unique-targetid={len(_EMLINE_BEST_CACHE)}', flush=True)
    return _EMLINE_BEST_CACHE


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


def build_raw_region(zone_label, cuts, region, tracers, real_tables, random_tables,
                     output_raw, n_random, zone_value, out_tag, release_tag):
    """
    Build and persist the DR1 raw table for ``zone_label`` applying ``cuts``.

    Args:
        zone_label: Label for the zone being processed.
        cuts: Dictionary with the cuts to apply.
        region: Region label (e.g. 'N', 'S', 'ALL').
        tracers: List of tracers to process.
        real_tables: Dictionary with real tables per tracer.
        random_tables: Dictionary with random tables per tracer.
        output_raw: Path to the output raw directory.
        n_random: Number of randoms per data object.
        zone_value: Integer value to assign to the ZONE column.
        out_tag: Optional tag to append to the output file name.
        release_tag: Release tag string or None.
    Returns:
        The combined table written to disk.
    """
    parts = []
    skipped = []
    for tr in tracers:
        try:
            rt = process_real_region(real_tables, tr, region, cuts, zone_value=zone_value)
        except ValueError as exc:
            print(f'[warn] {tr} empty after cuts in region {region}: {exc}')
            skipped.append(tr)
            continue
        parts.append(rt)
        count = len(rt)
        rpt = generate_randoms_region(random_tables, tr, region, cuts, n_random, count, zone_value=zone_value)
        parts.append(rpt)

    if not parts:
        raise ValueError(f'No data in region {region} for cuts {cuts} (tracers tried: {tracers})')

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
    cuts = {label: values.copy() for label, values in DEFAULT_CUTS.items()}
    if args.config:
        with open(args.config, 'r', encoding='utf-8') as handle:
            user_cuts = json.load(handle)
        cuts.update({str(k): {key: float(val) for key, val in v.items()} for k, v in user_cuts.items()})

    if args.zones is not None:
        zones = [str(z) for z in args.zones]
    else:
        zones = sorted(cuts.keys())

    missing = [z for z in zones if z not in cuts]
    if missing:
        raise RuntimeError(f'No cuts configured: {", ".join(missing)}')

    def _build(zone, real_tables, random_tables, sel_tracers, parsed_args, release_tag):
        """
        Build the raw table for a given zone.

        Args:
            zone: Zone label.
            real_tables: Dictionary with real tables per tracer.
            random_tables: Dictionary with random tables per tracer.
            sel_tracers: List of selected tracers to process.
            parsed_args: Parsed command line arguments.
            release_tag: Release tag string or None.
        Returns:
            The combined table written to disk.
        """
        label = str(zone)
        zone_value = ZONE_VALUES.get(label, 9999)
        zone_cuts = cuts[label]
        return build_raw_region(label, zone_cuts, 'ALL', sel_tracers, real_tables, random_tables,
                                parsed_args.raw_out, parsed_args.n_random, zone_value,
                                out_tag=parsed_args.out_tag, release_tag=release_tag)

    return ReleaseConfig(name='DR1', release_tag='DR1', tracers=TRACERS, tracer_alias=TRACER_ALIAS,
                         real_suffix=REAL_SUFFIX, random_suffix=RANDOM_SUFFIX,
                         n_random_files=N_RANDOM_FILES, real_columns=REAL_COLUMNS,
                         random_columns=RANDOM_COLUMNS, use_dr2_preload=False,
                         preload_kwargs={}, zones=zones, build_raw=_build)