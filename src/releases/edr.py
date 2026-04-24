import os
import numpy as np
from astropy.table import Table, vstack

from desiproc.read_data import generate_randoms, process_real
from desiproc.paths import safe_tag, zone_tag

from .base import ReleaseConfig


TRACERS = ['BGS_ANY', 'ELG', 'LRG', 'QSO']
REAL_SUFFIX = {'N': '_N_clustering.dat.fits', 'S': '_S_clustering.dat.fits'}
RANDOM_SUFFIX = {'N': '_N_{i}_clustering.ran.fits', 'S': '_S_{i}_clustering.ran.fits'}
N_RANDOM_FILES = 18
REAL_COLUMNS = ['TARGETID', 'ROSETTE_NUMBER', 'RA', 'DEC', 'Z']
RANDOM_COLUMNS = REAL_COLUMNS
N_ZONES = 20
NORTH_ROSETTES = {3, 6, 7, 11, 12, 13, 14, 15, 18, 19}
TRACER_ALIAS = {'bgs': 'BGS_ANY', 'elg': 'ELG', 'lrg': 'LRG', 'qso': 'QSO'}
EMLINE_CATALOG_PATH = ('/global/cfs/cdirs/desi/public/edr/vac/edr/stellar-mass-emline/'
                       'v1.0/edr_galaxy_stellarmass_lineinfo_v1.0.fits')
EMLINE_REQUIRED_COLUMNS = ('TARGETID', 'ZERR', 'SED_SFR', 'SED_MASS', 'FLUX_G', 'FLUX_R')
EMLINE_OUTPUT_COLUMNS = ('SED_SFR', 'SED_MASS', 'FLUX_G', 'FLUX_R')
_EMLINE_BEST_CACHE = None


def _float_with_nan(column):
    """
    Convert an input column to float64, replacing masked values with NaN.
    """
    arr = np.asarray(column)
    if np.ma.isMaskedArray(arr):
        return np.asarray(arr.filled(np.nan), dtype=np.float64)
    return np.asarray(arr, dtype=np.float64)


def _load_emline_best(catalog_path=EMLINE_CATALOG_PATH):
    """
    Load the EDR emline catalogue and keep one row per TARGETID with minimum ZERR.

    Args:
        catalog_path: Path to the EDR emline catalogue FITS file.
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
        raise FileNotFoundError(f'EDR emline catalogue not found: {catalog_path}')

    emline = Table.read(catalog_path, memmap=True)
    missing = [name for name in EMLINE_REQUIRED_COLUMNS if name not in emline.colnames]
    if missing:
        raise KeyError(f'EDR emline catalogue missing columns: {missing}')
    emline = emline[list(EMLINE_REQUIRED_COLUMNS)]
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

    print(f'[edr] emline rows={len(emline)} unique-targetid={len(_EMLINE_BEST_CACHE)}', flush=True)
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
        KeyError: If 'TARGETID' is missing from the raw table or if required
        emline columns are missing from the emline_best table.
    """
    if 'TARGETID' not in raw_table.colnames:
        raise KeyError("Raw table does not contain 'TARGETID'")

    raw_tid = np.asarray(raw_table['TARGETID'], dtype=np.int64)
    best_tid = np.asarray(emline_best['TARGETID'], dtype=np.int64)

    idx = np.searchsorted(best_tid, raw_tid, side='left')
    valid = idx < best_tid.size
    valid[valid] &= best_tid[idx[valid]] == raw_tid[valid]

    for name in EMLINE_OUTPUT_COLUMNS:
        values = _float_with_nan(emline_best[name])
        out = np.full(len(raw_table), np.nan, dtype=np.float64)
        out[valid] = values[idx[valid]]
        if name in raw_table.colnames:
            raw_table.remove_column(name)
        raw_table[name] = out

    print(f'[edr] enriched raw with emline columns matches={int(valid.sum())}/{len(raw_table)}', flush=True)
    return raw_table


def build_raw_table(zone, real_tables, random_tables, output_raw, n_random, tracers,
                    north_rosettes, out_tag, release_tag):
    """
    Build and persist the EDR raw table for ``zone``.

    Args:
        zone: Zone number (0-19).
        real_tables: Preloaded real tables.
        random_tables: Preloaded random tables.
        output_raw: Output directory for raw tables.
        n_random: Number of randoms to generate per real.
        tracers: List of tracers to include.
        north_rosettes: Set of rosette numbers in the North.
        out_tag: Optional tag to append to output filename.
        release_tag: Optional release tag to include in metadata.
    Returns:
        The combined raw table for the specified zone.
    """
    parts = []
    for tr in tracers:
        rt = process_real(real_tables, tr, zone, north_rosettes)
        parts.append(rt)
        count = len(rt)
        rpt = generate_randoms(random_tables, tr, zone, north_rosettes, n_random, count)
        parts.append(rpt)

    tbl = vstack(parts)
    if 'RANDITER' in tbl.colnames:
        tbl['RANDITER'] = np.asarray(tbl['RANDITER'], dtype=np.int32)
    tbl = _append_emline_columns(tbl, _load_emline_best())

    tag_suffix = safe_tag(out_tag)
    out_path = os.path.join(output_raw, f'zone_{zone:02d}{tag_suffix}.fits.gz')
    tmp_path = out_path + '.tmp'

    tbl_out = tbl.copy()
    if 'ZONE' in tbl_out.colnames:
        tbl_out.remove_column('ZONE')

    tbl_out.meta['ZONE'] = zone_tag(zone)
    tbl_out.meta['RELEASE'] = str(release_tag) if release_tag is not None else 'UNKNOWN'

    tbl_out.write(tmp_path, format='fits', overwrite=True)
    os.replace(tmp_path, out_path)
    return tbl


def create_config(args):
    """
    Create the EDR release configuration.

    Args:
        args: Parsed command-line arguments.
    Returns:
        The EDR release configuration.
    Raises:
        RuntimeError: If a specified zone is out of range.
    """
    if args.zone is not None:
        if not 0 <= int(args.zone) < N_ZONES:
            raise RuntimeError(f"Zone {args.zone} out of range (0-{N_ZONES-1})")
        zones = [int(args.zone)]
    else:
        zones = list(range(N_ZONES))

    def _build(zone, real_tables, random_tables, sel_tracers, parsed_args, release_tag):
        return build_raw_table(int(zone), real_tables, random_tables, parsed_args.raw_out,
                               parsed_args.n_random, sel_tracers, NORTH_ROSETTES,
                               out_tag=parsed_args.out_tag, release_tag=release_tag)

    return ReleaseConfig(name='EDR', release_tag='EDR', tracers=TRACERS,
                         tracer_alias=TRACER_ALIAS, real_suffix=REAL_SUFFIX,
                         random_suffix=RANDOM_SUFFIX, n_random_files=N_RANDOM_FILES,
                         real_columns=REAL_COLUMNS, random_columns=RANDOM_COLUMNS,
                         use_dr2_preload=False, preload_kwargs={}, zones=zones,
                         build_raw=_build)