import random
import warnings

import astropy.units as u
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.cosmology import Planck18
from astropy.table import Column, Table, vstack
from astropy.units import UnitsWarning

warnings.filterwarnings('ignore', category=UnitsWarning)


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


def _compute_cartesian(tbl):
    """
    Add Cartesian coordinates (``XCART``, ``YCART``, ``ZCART``) to ``tbl``.

    Args:
        tbl (Table): Input table with ``RA``, ``DEC``, and ``Z`` columns.
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
        tbl['XCART'] = sc.cartesian.x.value
        tbl['YCART'] = sc.cartesian.y.value
        tbl['ZCART'] = sc.cartesian.z.value
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
            tbl.add_column(Column(np.full(len(tbl), int(zone_value), dtype=int), name='ZONE'))
        return tbl
    except Exception as e:
        raise RuntimeError(f"Error ensuring 'ZONE' column: {e}") from e


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