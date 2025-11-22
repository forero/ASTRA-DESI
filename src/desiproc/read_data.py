import os
import random
import warnings

import astropy.units as u
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.cosmology import Planck18
from astropy.table import Column, Table, vstack
from astropy.units import UnitsWarning

warnings.filterwarnings('ignore', category=UnitsWarning)


def _progress(message):
    """
    Emit a progress message when verbose progress is enabled.
    """
    if os.environ.get('ASTRA_PROGRESS'):
        print(f'[progress] {message}', flush=True)


def load_table(path, columns):
    """
    Return a FITS table containing only the requested columns.

    The function renames ``ROSETTE_NUMBER`` to ``ZONE`` when the source column is
    present so downstream code can rely on a consistent column name.

    Args:
        path (str): Path to the FITS catalogue.
        columns (list): Column names to retain.
    Returns:
        Table: Table containing the selected columns.
    Raises:
        IOError: If the file cannot be read.
        KeyError: If any requested column is missing.
        RuntimeError: If the filtered table cannot be produced.
    """
    try:
        tbl = Table.read(path, memmap=True)
    except Exception as e:
        raise IOError(f'Error reading {path}: {e}') from e

    missing = set(columns) - set(tbl.colnames)
    if missing:
        raise KeyError(f'Missing columns {missing} in file {path}')

    try:
        tbl = tbl[columns]
        if 'ROSETTE_NUMBER' in tbl.colnames:
            tbl.rename_column('ROSETTE_NUMBER', 'ZONE')
        return tbl
    except Exception as e:
        raise RuntimeError(f'Error processing table columns for {path}: {e}') from e


def _compute_cartesian(tbl, dtype=np.float64):
    """
    Add Cartesian coordinates (``XCART``, ``YCART``, ``ZCART``) to ``tbl``.

    Args:
        tbl (Table): Input table with ``RA``, ``DEC``, and ``Z`` columns.
        dtype (np.dtype or type): Floating dtype for Cartesian columns.
    Returns:
        Table: The same table with Cartesian coordinates appended.
    Raises:
        RuntimeError: If the coordinate transformation fails.
    """
    try:
        z = np.asarray(tbl['Z'], dtype=float)
        dist = Planck18.comoving_distance(z)
        ra = np.asarray(tbl['RA'], dtype=float) * u.deg
        dec = np.asarray(tbl['DEC'], dtype=float) * u.deg
        sc = SkyCoord(ra=ra, dec=dec, distance=dist)
        tbl['XCART'] = np.asarray(sc.cartesian.x.value, dtype=dtype)
        tbl['YCART'] = np.asarray(sc.cartesian.y.value, dtype=dtype)
        tbl['ZCART'] = np.asarray(sc.cartesian.z.value, dtype=dtype)
        return tbl
    except Exception as e:
        raise RuntimeError(f'Error computing Cartesian coordinates: {e}') from e


def _ensure_zone_column(tbl, zone_value):
    """
    Ensure that ``tbl`` contains a ``ZONE`` column with a constant value.

    Args:
        tbl (Table): Table to validate.
        zone_value (int): Value to assign when the column is missing.
    Returns:
        Table: Table with an explicit ``ZONE`` column.
    Raises:
        RuntimeError: If the column cannot be created.
    """
    try:
        if 'ZONE' not in tbl.colnames:
            tbl.add_column(Column(np.full(len(tbl), int(zone_value), dtype=np.int32), name='ZONE'))
        else:
            col = tbl['ZONE']
            if col.dtype != np.int32:
                tbl['ZONE'] = np.asarray(col, dtype=np.int32)
        return tbl
    except Exception as e:
        raise RuntimeError(f"Error ensuring 'ZONE' column: {e}") from e


def _build_fixed_string_array(size, value, min_length=4):
    """
    Return an ASCII array filled with ``value`` using a fixed-width dtype.

    Args:
        size (int): Number of elements.
        value (str): Value to fill.
        min_length (int): Minimum field width.
    Returns:
        np.ndarray: Array of shape ``(size,)`` with dtype ``S``.
    """
    encoded = str(value).encode('ascii')
    width = max(len(encoded), int(min_length))
    dtype = np.dtype(f'S{width}')
    arr = np.empty(size, dtype=dtype)
    arr[...] = encoded
    return arr


def _filter_by_box(tbl, ra_min, ra_max, dec_min, dec_max, z_min=None, z_max=None):
    """
    Return rows within rectangular RA/DEC limits optionally clipped in redshift.

    Args:
        tbl (Table): Table containing ``RA``/``DEC`` (and optionally ``Z``).
        ra_min (float): Lower RA bound in degrees.
        ra_max (float): Upper RA bound in degrees.
        dec_min (float): Lower DEC bound in degrees.
        dec_max (float): Upper DEC bound in degrees.
        z_min (float, optional): Lower redshift bound.
        z_max (float, optional): Upper redshift bound.
    Returns:
        Table: Filtered table view.
    Raises:
        RuntimeError: If the filtering operation fails.
    """
    try:
        ra = np.asarray(tbl['RA'], dtype=float)
        dec = np.asarray(tbl['DEC'], dtype=float)
        mask = (ra > ra_min) & (ra < ra_max) & (dec > dec_min) & (dec < dec_max)
        return tbl[mask]
    except Exception as e:
        raise RuntimeError(f"Error filtering by box: {e}") from e


def get_hemisphere(zone, north_rosettes):
    """
    Return the survey hemisphere (``'N'`` or ``'S'``) for ``zone``.

    Args:
        zone (int): DESI zone identifier.
        north_rosettes (set): Indices considered part of the northern footprint.
    Returns:
        str: ``'N'`` when the zone is northern, otherwise ``'S'``.
    Raises:
        RuntimeError: If the hemisphere cannot be determined.
    """
    try:
        return ('S', 'N')[zone in north_rosettes]
    except Exception as e:
        raise RuntimeError(f'Error determining hemisphere for zone {zone}: {e}') from e


def process_real(real_tables, tracer, zone, north_rosettes):
    """
    Return real objects for a zone with derived Cartesian coordinates.

    Args:
        real_tables (dict): Preloaded real tables keyed by tracer and hemisphere.
        tracer (str): Tracer identifier (e.g., ``'BGS_ANY'``).
        zone (int): Zone identifier.
        north_rosettes (set): Indices corresponding to the northern footprint.
    Returns:
        Table: Real objects annotated with tracer type and ``RANDITER``.
    Raises:
        KeyError: If the tracer has no data in the required hemisphere.
        ValueError: If the zone is empty for the tracer.
        RuntimeError: If processing fails.
    """
    try:
        hemi = get_hemisphere(zone, north_rosettes)
        tbl = real_tables[tracer][hemi]
        sel = tbl[tbl['ZONE'] == zone]
        if len(sel) == 0:
            raise ValueError(f'No entries for zone {zone} in tracer {tracer} ({hemi})')
        sel = _compute_cartesian(sel)
        sel['TRACERTYPE'] = f'{tracer}_DATA'
        sel['RANDITER'] = np.full(len(sel), -1, dtype=np.int32)
        return sel
    except KeyError:
        raise
    except ValueError:
        raise
    except Exception as e:
        raise RuntimeError(f'Error processing real data for tracer {tracer}, zone {zone}: {e}') from e


def generate_randoms(random_tables, tracer, zone, north_rosettes, n_random, real_count):
    """
    Return random catalogues that mirror a zone's real sample size.

    Each iteration selects a random file deterministically via ``random.Random``
    seeded by the iteration index and samples ``real_count`` rows without
    replacement using ``numpy.random.default_rng`` for reproducibility.

    Args:
        random_tables (dict): Preloaded random tables keyed by tracer and hemisphere.
        tracer (str): Tracer identifier (e.g., ``'BGS_ANY'``).
        zone (int): Zone identifier.
        north_rosettes (set): Indices corresponding to the northern sky footprint.
        n_random (int): Number of random realizations to generate.
        real_count (int): Number of objects in the matching real sample.
    Returns:
        Table: Concatenated random sample table with Cartesian coordinates.
    Raises:
        KeyError: If the tracer lacks random tables for the required hemisphere.
        ValueError: If any random table provides fewer than ``real_count`` rows.
        RuntimeError: If the sampling process fails.
    """
    try:
        hemi = get_hemisphere(zone, north_rosettes)
        tables = random_tables[tracer][hemi].values()
        zone_tables = []
        for tbl in tables:
            sel = tbl[tbl['ZONE'] == zone]
            if len(sel) < real_count:
                raise ValueError(f'Zone {zone} has only {len(sel)} random points (< {real_count})')
            zone_tables.append(sel)
        n_files = len(zone_tables)

        zone_tables_xyz = []
        for _sel in zone_tables:
            _sel_xyz = _compute_cartesian(_sel.copy())
            zone_tables_xyz.append(_sel_xyz)

        samples, used = [], set()
        for j in range(n_random):
            if len(used) == n_files:
                used.clear()
            choices = [i for i in range(n_files) if i not in used]
            idx = random.Random(j).choice(choices)
            used.add(idx)
            sel = zone_tables_xyz[idx]
            rows = np.random.default_rng(j).choice(len(sel), real_count, replace=False)
            samp = sel[rows]
            samp['TRACERTYPE'] = f'{tracer}_RAND'
            samp['RANDITER'] = np.full(len(samp), j, dtype=np.int32)
            samples.append(samp)

        return vstack(samples)
    except KeyError:
        raise
    except ValueError:
        raise
    except Exception as e:
        raise RuntimeError(f"Error generating randoms for tracer {tracer}, zone {zone}: {e}") from e


def process_real_region(real_tables, tracer, region, cuts, zone_value=9001):
    """
    Return real objects within a sky box for a DR1-style region selection.

    Args:
        real_tables (dict): Preloaded real tables keyed by tracer and hemisphere.
        tracer (str): Tracer identifier (e.g., ``'BGS_BRIGHT'``).
        region (str): Hemisphere label (``'N'``, ``'S'``, or ``'ALL'``).
        cuts (dict): Bounding box limits with ``RA``/``DEC`` (and optionally ``Z``).
        zone_value (int): Synthetic zone identifier to insert when missing.
    Returns:
        Table: Real objects rolled into a single table with Cartesian coordinates.
    Raises:
        KeyError: If the tracer lacks data in the requested region.
        ValueError: If filtering yields no rows.
        RuntimeError: If the extraction or coordinate conversion fails.
    """
    try:
        region = str(region).upper()
        if region == 'ALL':
            tN = real_tables[tracer].get('N')
            tS = real_tables[tracer].get('S')
            if (tN is None) and (tS is None):
                raise KeyError(f'No data for tracer {tracer} in any hemisphere')
            if (tN is not None) and (tS is not None):
                tbl = vstack([tN, tS])
            else:
                tbl = tN if tN is not None else tS
        else:
            tbl = real_tables[tracer][region]
        sel = _filter_by_box(tbl,
                             cuts['RA_min'], cuts['RA_max'],
                             cuts['DEC_min'], cuts['DEC_max'])
        if len(sel) == 0:
            raise ValueError(f'No entries for {tracer} in region {region} after cuts {cuts}')
        sel = _ensure_zone_column(sel, zone_value)
        sel = _compute_cartesian(sel)
        sel['TRACERTYPE'] = f'{tracer}_DATA'
        sel['RANDITER'] = np.full(len(sel), -1, dtype=np.int32)
        return sel
    except KeyError:
        raise
    except ValueError:
        raise
    except Exception as e:
        raise RuntimeError(f'Error processing real data for tracer {tracer} in region {region}: {e}') from e


def generate_randoms_region(random_tables, tracer, region, cuts, n_random, real_count, zone_value=9001):
    """
    Return random catalogues matching the regional real-data selection.

    Args:
        random_tables (dict): Preloaded random tables keyed by tracer and hemisphere.
        tracer (str): Tracer identifier.
        region (str): Hemisphere label (``'N'``, ``'S'``, or ``'ALL'``).
        cuts (dict): Bounding box limits with ``RA``/``DEC`` (and optionally ``Z``).
        n_random (int): Number of random realizations to generate.
        real_count (int): Number of objects per realization.
        zone_value (int): Synthetic zone identifier in case the source lacks one.
    Returns:
        Table: Concatenated random sample table with Cartesian coordinates.
    Raises:
        KeyError: If the tracer lacks random tables for the requested region.
        ValueError: If the filtered random pool is too small.
        RuntimeError: If sampling fails.
    """
    try:
        region = str(region).upper()
        if region == 'ALL':
            hemi_dict = random_tables[tracer]
            tables = []
            for hemi in ('N', 'S'):
                sub = hemi_dict.get(hemi, {})
                tables.extend(list(sub.values()))
        else:
            tables = list(random_tables[tracer][region].values())
        if len(tables) == 0:
            raise KeyError(f'No random tables for {tracer} in region {region}')

        zone_tables = []
        total_after_cuts = 0
        for tbl in tables:
            sel = _filter_by_box(tbl,
                                 cuts['RA_min'], cuts['RA_max'],
                                 cuts['DEC_min'], cuts['DEC_max'],
                                 cuts.get('Z_min', None), cuts.get('Z_max', None))
            if len(sel) == 0:
                continue
            sel = _ensure_zone_column(sel, zone_value)
            zone_tables.append(sel)
            total_after_cuts += len(sel)

        if total_after_cuts == 0:
            raise ValueError(f'No random entries for {tracer} in region {region} after cuts {cuts}')
        if total_after_cuts < real_count:
            raise ValueError(f'Region {region} randoms total have only {total_after_cuts} points after cuts (< {real_count})')

        zone_tables_xyz = []
        for _sel in zone_tables:
            _sel_xyz = _compute_cartesian(_sel.copy())
            zone_tables_xyz.append(_sel_xyz)
        pool = vstack(zone_tables_xyz)

        samples = []
        for j in range(n_random):
            rng = np.random.default_rng(j)
            rows = rng.choice(len(pool), real_count, replace=False)
            samp = pool[rows]
            samp['TRACERTYPE'] = f'{tracer}_RAND'
            samp['RANDITER'] = np.full(len(samp), j, dtype=np.int32)
            samples.append(samp)

        return vstack(samples)
    except KeyError:
        raise
    except ValueError:
        raise
    except Exception as e:
        raise RuntimeError(f'Error generating randoms for tracer {tracer} in region {region}: {e}') from e


def _split_table_by_ra(tbl, ra_min, ra_max, include_edges=True):
    """Split a table into two subsets based on an RA window.

    Args:
        tbl (Table): Input table containing an ``RA`` column in degrees.
        ra_min (float): Lower bound of the window assigned to the first subset.
        ra_max (float): Upper bound of the window assigned to the first subset.
        include_edges (bool): When ``True``, boundary values belong to the first subset.
    Returns:
        tuple[Table, Table]: Subsets inside and outside the RA window.
    Raises:
        RuntimeError: If the RA column cannot be processed.
    """
    try:
        ra = np.asarray(tbl['RA'], dtype=float)
    except Exception as e:
        raise RuntimeError(f"Error accessing RA column for RA split: {e}") from e

    if include_edges:
        mask = (ra >= ra_min) & (ra <= ra_max)
    else:
        mask = (ra > ra_min) & (ra < ra_max)

    return tbl[mask], tbl[~mask]


def preload_dr2_tables(base_dir, tracers, real_columns, random_columns, n_random_files,
                       ra_min=90.0, ra_max=300.0, include_edges=True):
    """
    Preload DR2 full catalogues and split them into NGC/SGC by RA.

    Args:
        base_dir (str): Directory containing DR2 catalogues.
        tracers (list[str]): Tracer identifiers to load.
        real_columns (list[str]): Columns to load from real catalogues.
        random_columns (list[str]): Columns to load from random catalogues.
        n_random_files (int): Number of random catalogues per tracer.
        ra_min (float): Minimum RA assigned to the NGC subset.
        ra_max (float): Maximum RA assigned to the NGC subset.
        include_edges (bool): When ``True``, boundary values belong to NGC.
    Returns:
        tuple[dict, dict]: Real and random table dictionaries keyed by tracer and zone label.
    Raises:
        RuntimeError: If any catalogue fails to load or split.
    """
    try:
        real_tables = {t: {'NGC': None, 'SGC': None} for t in tracers}
        rand_tables = {t: {'NGC': {}, 'SGC': {}} for t in tracers}

        for tracer in tracers:
            real_path = os.path.join(base_dir, f'{tracer}_clustering.dat.fits')
            print(f"[dr2] loading real catalogue {real_path}", flush=True)
            requested_cols = list(real_columns)
            real_tbl = load_table(real_path, requested_cols)
            if 'Z' not in real_tbl.colnames:
                raise KeyError(f"Missing 'Z' column for tracer {tracer} in DR2 real table")
            ngc_real, sgc_real = _split_table_by_ra(real_tbl, ra_min, ra_max, include_edges=include_edges)
            real_tables[tracer]['NGC'] = ngc_real
            real_tables[tracer]['SGC'] = sgc_real
            print(f"[dr2] tracer={tracer} real rows -> NGC={len(ngc_real)} SGC={len(sgc_real)}", flush=True)

            for idx in range(n_random_files):
                rand_path = os.path.join(base_dir, f'{tracer}_{idx}_clustering.ran.fits')
                print(f"[dr2] loading random catalogue {rand_path}", flush=True)
                rand_tbl = load_table(rand_path, random_columns)
                ngc_rand, sgc_rand = _split_table_by_ra(rand_tbl, ra_min, ra_max, include_edges=include_edges)
                rand_tables[tracer]['NGC'][idx] = ngc_rand
                rand_tables[tracer]['SGC'][idx] = sgc_rand
                print(f"[dr2] tracer={tracer} rand file={idx} rows -> NGC={len(ngc_rand)} SGC={len(sgc_rand)}", flush=True)

        return real_tables, rand_tables
    except Exception as e:
        raise RuntimeError(f'Error preloading DR2 tables: {e}') from e


def process_real_dr2(real_tables, tracer, zone_label, zone_value=2001,
                     tracer_id=None, include_tracertype=True, downcast=True):
    """
    Return DR2 real objects for the requested zone label.
    
    Args:
        real_tables (dict): Preloaded real tables keyed by tracer and zone label.
        tracer (str): Tracer identifier (e.g., ``'BGS_ANY'``).
        zone_label (str): Zone label (``'NGC'`` or ``'SGC'``).
        zone_value (int): Synthetic zone identifier to insert when missing.
        tracer_id (int | None): Optional numeric tracer identifier stored in the ``TRACER_ID`` column.
        include_tracertype (bool): When ``True`` the ``TRACERTYPE`` string column is attached.
        downcast (bool): When ``True`` reduce floating-point precision to ``float32``.
    Returns:
        Table: Real objects annotated with tracer type and ``RANDITER``.
    Raises:
        KeyError: If the tracer has no data in the required zone.
        ValueError: If the zone is empty for the tracer.
        RuntimeError: If processing fails.
    """
    try:
        tbl = real_tables[tracer][zone_label]
        if tbl is None or len(tbl) == 0:
            raise ValueError(f'No entries for tracer {tracer} in zone {zone_label}')
        sel = tbl.copy()
        sel = _ensure_zone_column(sel, zone_value)
        sel = _compute_cartesian(sel, dtype=np.float32 if downcast else np.float64)
        if downcast:
            for col in ('RA', 'DEC', 'Z'):
                if col in sel.colnames:
                    sel[col] = np.asarray(sel[col], dtype=np.float32)
        sel['RANDITER'] = Column(np.full(len(sel), -1, dtype=np.int16 if downcast else np.int32),
                                 name='RANDITER')
        if tracer_id is not None:
            sel['TRACER_ID'] = Column(np.full(len(sel), int(tracer_id), dtype=np.uint8), name='TRACER_ID')
        if include_tracertype:
            sel['TRACERTYPE'] = Column(_build_fixed_string_array(len(sel), f'{tracer}_DATA'),
                                       name='TRACERTYPE')
        return sel
    except KeyError:
        raise
    except ValueError:
        raise
    except Exception as e:
        raise RuntimeError(f'Error processing DR2 real data for tracer {tracer}, zone {zone_label}: {e}') from e


def generate_randoms_dr2(random_tables, tracer, zone_label, n_random, real_table,
                         zone_value=2001, tracer_id=None, include_tracertype=True,
                         downcast=True):
    """
    Return DR2 random catalogues sampled directly from the zone-specific pool.
    
    Args:
        random_tables (dict): Preloaded random tables keyed by tracer and zone label.
        tracer (str): Tracer identifier (e.g., ``'BGS_ANY'``).
        zone_label (str): Zone label (``'NGC'`` or ``'SGC'``).
        n_random (int): Number of random realizations to generate.
        real_table (Table): Real data table used to determine the target sample size.
        zone_value (int): Synthetic zone identifier to insert when missing.
        tracer_id (int | None): Optional numeric tracer identifier stored in the ``TRACER_ID`` column.
        include_tracertype (bool): When ``True`` the ``TRACERTYPE`` column is populated.
        downcast (bool): When ``True`` floating columns are stored as ``float32``.
    Returns:
        Table: Concatenated random sample table with Cartesian coordinates.
    Raises:
        KeyError: If the tracer lacks random tables for the requested zone.
        ValueError: If the filtered random pool is too small.
        RuntimeError: If sampling fails.
    """
    try:
        zone_dict = random_tables[tracer][zone_label]
        if not zone_dict:
            raise KeyError(f'No random tables for tracer {tracer} in zone {zone_label}')

        real_count = len(real_table)
        if real_count == 0:
            raise ValueError(f'Real table for tracer {tracer} zone {zone_label} is empty')
        _progress(f'zone {zone_label}: generating randoms for tracer {tracer} '
                  f'(target {n_random} iterations, {real_count} rows each)')

        base_numeric_cols = ('TARGETID', 'RA', 'DEC', 'Z')
        keep_columns = [name for name in real_table.colnames
                        if name not in ('TRACERTYPE', 'RANDITER', 'TRACER_ID')]
        extra_columns = [col for col in keep_columns if col not in ('TARGETID', 'RA', 'DEC', 'Z',
                                                                    'XCART', 'YCART', 'ZCART', 'ZONE')]

        required_source_cols = set(base_numeric_cols) | set(extra_columns)

        zone_tables = []
        lengths = []

        for tbl in zone_dict.values():
            if tbl is None or len(tbl) == 0:
                continue

            has_cart = all(comp in tbl.colnames for comp in ('XCART', 'YCART', 'ZCART'))
            if has_cart:
                required_source_cols.update(('XCART', 'YCART', 'ZCART'))

            missing = [col for col in required_source_cols if col not in tbl.colnames]
            if missing:
                raise KeyError(f"Missing columns {missing} in random tables for tracer {tracer}, zone {zone_label}")

            arrays = {}
            for col in required_source_cols:
                data = tbl[col]
                if isinstance(data, np.ma.MaskedArray):
                    arrays[col] = np.asarray(data.filled(0))
                else:
                    arrays[col] = np.asarray(data)
            zone_tables.append(arrays)
            lengths.append(len(tbl))

        if not zone_tables:
            raise ValueError(f'No random entries for tracer {tracer} in zone {zone_label}')

        total_rows = int(np.sum(lengths, dtype=np.int64))
        if total_rows < real_count:
            raise ValueError(f'Zone {zone_label} randoms have {total_rows} rows (< {real_count})')
        _progress(f'zone {zone_label}: tracer {tracer} random pool size {total_rows} rows '
                  f'({len(zone_tables)} files)')

        offsets = np.zeros(len(lengths) + 1, dtype=np.int64)
        offsets[1:] = np.cumsum(lengths, dtype=np.int64)

        float_dtype = np.float32 if downcast else np.float64
        total_out = real_count * n_random

        output_numeric = {
            'TARGETID': np.empty(total_out, dtype=np.int64),
            'RA': np.empty(total_out, dtype=float_dtype),
            'DEC': np.empty(total_out, dtype=float_dtype),
            'Z': np.empty(total_out, dtype=float_dtype),
            'XCART': np.empty(total_out, dtype=float_dtype),
            'YCART': np.empty(total_out, dtype=float_dtype),
            'ZCART': np.empty(total_out, dtype=float_dtype),
        }

        output_extra = {}
        for col in extra_columns:
            output_extra[col] = np.empty(total_out, dtype=real_table[col].dtype)

        zone_col = None
        if 'ZONE' in real_table.colnames:
            zone_col = np.full(total_out, int(zone_value), dtype=np.int32)

        tracer_id_col = None
        tracer_code = int(tracer_id) if tracer_id is not None else 255
        if ('TRACER_ID' in real_table.colnames) or tracer_id is not None:
            tracer_id_col = np.full(total_out, tracer_code, dtype=np.uint8)

        tracertype = _build_fixed_string_array(total_out, f'{tracer}_RAND', min_length=8) if include_tracertype else None

        randiter_dtype = np.int16 if downcast else np.int32
        randiter = np.empty(total_out, dtype=randiter_dtype)

        have_cartesian = all(all(comp in arrays for comp in ('XCART', 'YCART', 'ZCART'))
                             for arrays in zone_tables)

        print(f"[dr2] tracer={tracer} zone={zone_label} random pool rows={total_rows} target={real_count}", flush=True)
        for j in range(n_random):
            start = j * real_count
            end = start + real_count
            rng = np.random.default_rng(j)
            rows = rng.choice(total_rows, real_count, replace=False)
            table_idx = np.searchsorted(offsets, rows, side='right') - 1
            table_idx = np.asarray(table_idx, dtype=np.int16)
            local_idx = rows - offsets[table_idx]

            unique_tables = np.unique(table_idx)
            for idx_table in unique_tables:
                positions = np.where(table_idx == idx_table)[0]
                if positions.size == 0:
                    continue
                dest_pos = start + positions
                src_rows = local_idx[positions]
                arrays = zone_tables[int(idx_table)]

                output_numeric['TARGETID'][dest_pos] = arrays['TARGETID'][src_rows].astype(np.int64, copy=False)
                output_numeric['RA'][dest_pos] = arrays['RA'][src_rows].astype(float_dtype, copy=False)
                output_numeric['DEC'][dest_pos] = arrays['DEC'][src_rows].astype(float_dtype, copy=False)
                output_numeric['Z'][dest_pos] = arrays['Z'][src_rows].astype(float_dtype, copy=False)

                for col in extra_columns:
                    output_extra[col][dest_pos] = arrays[col][src_rows]

            randiter[start:end] = j

            if have_cartesian:
                for idx_table in unique_tables:
                    positions = np.where(table_idx == idx_table)[0]
                    if positions.size == 0:
                        continue
                    dest_pos = start + positions
                    src_rows = local_idx[positions]
                    arrays = zone_tables[int(idx_table)]
                    output_numeric['XCART'][dest_pos] = arrays['XCART'][src_rows].astype(float_dtype, copy=False)
                    output_numeric['YCART'][dest_pos] = arrays['YCART'][src_rows].astype(float_dtype, copy=False)
                    output_numeric['ZCART'][dest_pos] = arrays['ZCART'][src_rows].astype(float_dtype, copy=False)
            else:
                ra_chunk = output_numeric['RA'][start:end].astype(np.float64, copy=False)
                dec_chunk = output_numeric['DEC'][start:end].astype(np.float64, copy=False)
                z_chunk = output_numeric['Z'][start:end].astype(np.float64, copy=False)
                dist = Planck18.comoving_distance(z_chunk).value
                sc = SkyCoord(ra=ra_chunk * u.deg, dec=dec_chunk * u.deg, distance=dist * u.Mpc)
                output_numeric['XCART'][start:end] = np.asarray(sc.cartesian.x.value, dtype=float_dtype)
                output_numeric['YCART'][start:end] = np.asarray(sc.cartesian.y.value, dtype=float_dtype)
                output_numeric['ZCART'][start:end] = np.asarray(sc.cartesian.z.value, dtype=float_dtype)

            if (j + 1) % 10 == 0 or (j + 1) == n_random:
                print(f"[dr2] tracer={tracer} zone={zone_label} generated random iteration {j+1}/{n_random}", flush=True)

        rand_tbl = Table()
        for col in real_table.colnames:
            if col == 'TRACERTYPE':
                if include_tracertype and tracertype is not None:
                    rand_tbl[col] = Column(tracertype, name=col)
            elif col == 'RANDITER':
                rand_tbl[col] = Column(randiter, name=col)
            elif col == 'TRACER_ID':
                if tracer_id_col is not None:
                    rand_tbl[col] = Column(tracer_id_col, name=col)
            elif col == 'ZONE':
                if zone_col is not None:
                    rand_tbl[col] = Column(zone_col, name=col)
            elif col in output_numeric:
                rand_tbl[col] = Column(output_numeric[col], name=col)
            elif col in output_extra:
                rand_tbl[col] = Column(output_extra[col], name=col)

        if 'TRACER_ID' not in rand_tbl.colnames and tracer_id_col is not None:
            rand_tbl['TRACER_ID'] = Column(tracer_id_col, name='TRACER_ID')

        _progress(f'zone {zone_label}: tracer {tracer} random tables generated ({len(rand_tbl)} rows)')
        return rand_tbl
    except KeyError:
        raise
    except ValueError:
        raise
    except Exception as e:
        raise RuntimeError(f'Error generating DR2 randoms for tracer {tracer}, zone {zone_label}: {e}') from e