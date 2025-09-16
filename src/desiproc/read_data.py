from astropy.table import Table, vstack, Column
from astropy.cosmology import Planck18
from astropy.coordinates import SkyCoord
import astropy.units as u
import numpy as np, random

import warnings
from astropy.units import UnitsWarning
warnings.filterwarnings('ignore', category=UnitsWarning)


def load_table(path, columns):
    """
    Read the FITS file and return a Table with specified columns.
    Renames 'ROSETTE_NUMBER' to 'ZONE' if present.

    Args:
        path (str): Path to the FITS file.
        columns (list): List of column names to select.
    Returns:
        Table: Astropy Table with selected columns.
    Raises:
        IOError: If the file cannot be read.
        KeyError: If any requested column is missing.
        RuntimeError: If any other exception occurs during loading.
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
    Compute Cartesian coordinates (XCART, YCART, ZCART) from RA, DEC, and Z.

    Args:
        tbl (Table): Astropy Table with 'RA', 'DEC', and 'Z' columns.
    Returns:
        Table: Input table with added Cartesian coordinates.
    Raises:
        RuntimeError: If coordinate conversion fails.
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
    Ensure a 'ZONE' column exists. If missing, create it as a constant integer.
    This helps keep downstream code compatible when working with DR1 (no rosettes).

    Args:
        tbl (Table): Astropy Table.
        zone_value (int): Constant integer value to assign to ZONE.
    Returns:
        Table: Table guaranteed to have a 'ZONE' column.
    Raises:
        RuntimeError: If any exception occurs during the operation.
    """
    try:
        if 'ZONE' not in tbl.colnames:
            tbl.add_column(Column(np.full(len(tbl), int(zone_value), dtype=int), name='ZONE'))
        return tbl
    except Exception as e:
        raise RuntimeError(f"Error ensuring 'ZONE' column: {e}") from e


def _filter_by_box(tbl, ra_min, ra_max, dec_min, dec_max, z_min=None, z_max=None):
    """
    Filter a table by rectangular cuts in RA, DEC, and optionally Z.

    Args:
        tbl (Table): Input table with columns 'RA', 'DEC' and optionally 'Z'.
        ra_min, ra_max (float): RA limits in degrees.
        dec_min, dec_max (float): DEC limits in degrees.
        z_min, z_max (float or None): Redshift limits; if None, Z is not filtered.
    Returns:
        Table: Filtered view of the input table.
    Raises:
        RuntimeError: If filtering fails.
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
    Determine the hemisphere ('N' or 'S') based on the zone number
    and a set of known northern rosettes.

    Args:
        zone (int): Zone number.
        north_rosettes (set): Set of northern rosette numbers.
    Returns:
        str: 'N' for northern hemisphere, 'S' for southern hemisphere.
    Raises:
        RuntimeError: If determination fails.
    """
    try:
        return ('S','N')[zone in north_rosettes]
    except Exception as e:
        raise RuntimeError(f'Error determining hemisphere for zone {zone}: {e}') from e


def process_real(real_tables, tracer, zone, north_rosettes):
    """
    Process real data for a given tracer and zone, computing Cartesian coordinates.

    Args:
        real_tables (dict): Dictionary of preloaded real data tables.
        tracer (str): Tracer type (e.g., 'BGS_ANY', 'ELG', 'LRG', 'QSO').
        zone (int): Zone number.
        north_rosettes (set): Set of northern rosette numbers.
    Returns:
        Table: Processed table with Cartesian coordinates and tracer type.
    Raises:
        KeyError: If no data for the tracer in the specified hemisphere.
        ValueError: If no entries for the zone in the tracer data.
        RuntimeError: If any other exception occurs during processing.
    """
    try:
        hemi = get_hemisphere(zone, north_rosettes)
        tbl = real_tables[tracer][hemi]
        sel = tbl[tbl['ZONE'] == zone]
        if len(sel) == 0:
            raise ValueError(f'No entries for zone {zone} in tracer {tracer} ({hemi})')
        sel = _compute_cartesian(sel)
        sel['TRACERTYPE'] = f'{tracer}_DATA'
        sel['RANDITER'] = -1
        return sel
    except KeyError:
        raise
    except ValueError:
        raise
    except Exception as e:
        raise RuntimeError(f'Error processing real data for tracer {tracer}, zone {zone}: {e}') from e


def generate_randoms(random_tables, tracer, zone, north_rosettes, n_random, real_count):
    """
    Generate n_random reproducible random samples for zone, matching real_count.
        - Prefilter each random table by zone and validate size.
        - In a single loop, pick a table via random.Random(j) (seeded),
        then sample rows via np.random.default_rng(j).

    Args:
        random_tables (dict): Dictionary of preloaded random data tables.
        tracer (str): Tracer type (e.g., 'BGS_ANY', 'ELG', 'LRG', 'QSO').
        zone (int): Zone number.
        north_rosettes (set): Set of northern rosette numbers.
        n_random (int): Number of random samples to generate.
        real_count (int): Number of real points to match.
    Returns:
        Table: Concatenated table of random samples with Cartesian coordinates.
    Raises:
        KeyError: If no random data for the tracer in the specified hemisphere.
        ValueError: If zone has fewer random points than real_count.
        RuntimeError: If any other exception occurs during sampling.
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
            samp['RANDITER'] = j
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
    DR1-friendly: process real data by REGION ('N' or 'S') and rectangular sky/redshift cuts.
    If region is 'ALL', first concatenate N and S, then apply the cuts.
    Adds a synthetic 'ZONE' column if missing so downstream code stays compatible.

    Args:
        real_tables (dict): Preloaded real data tables, indexed as real_tables[tracer][region].
        tracer (str): Tracer type (e.g., 'BGS_BRIGHT', 'ELG_LOPnotqso', 'LRG', 'QSO').
        region (str): 'N', 'S', or 'ALL' to combine both hemispheres.
        cuts (dict): {'RA_min','RA_max','DEC_min','DEC_max','Z_min','Z_max'} (Z limits optional).
        zone_value (int): Constant to assign to 'ZONE' if it does not exist (synthetic zone id).
    Returns:
        Table: Processed table with Cartesian coordinates and tracer type, filtered to the box.
    Raises:
        KeyError: If tracer/region is not present.
        ValueError: If no entries after filtering.
        RuntimeError: If any other exception occurs during processing.
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
        sel['RANDITER'] = -1
        return sel
    except KeyError:
        raise
    except ValueError:
        raise
    except Exception as e:
        raise RuntimeError(f'Error processing real data for tracer {tracer} in region {region}: {e}') from e


def generate_randoms_region(random_tables, tracer, region, cuts, n_random, real_count, zone_value=9001):
    """
    DR1-friendly: generate randoms by REGION ('N' or 'S') and rectangular sky/redshift cuts.
    If region is 'ALL', use random tables from both hemispheres.
    Creates a synthetic 'ZONE' column (constant) if missing to keep compatibility.

    Args:
        random_tables (dict): Preloaded random data tables; random_tables[tracer][region] is a dict of tables.
        tracer (str): Tracer type (e.g., 'BGS_BRIGHT', 'ELG_LOPnotqso', 'LRG', 'QSO').
        region (str): 'N', 'S', or 'ALL'.
        cuts (dict): {'RA_min','RA_max','DEC_min','DEC_max','Z_min','Z_max'} (Z limits optional).
        n_random (int): Number of random realizations to generate.
        real_count (int): Number of real points to match per realization.
        zone_value (int): Constant to assign to 'ZONE' if it does not exist (synthetic zone id).
    Returns:
        Table: Concatenated table of random samples with Cartesian coordinates.
    Raises:
        KeyError: If no random data for the tracer/region.
        ValueError: If filtered randoms have fewer points than real_count.
        RuntimeError: If any other exception occurs during sampling.
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
            samp['RANDITER'] = j
            samples.append(samp)

        return vstack(samples)
    except KeyError:
        raise
    except ValueError:
        raise
    except Exception as e:
        raise RuntimeError(f'Error generating randoms for tracer {tracer} in region {region}: {e}') from e