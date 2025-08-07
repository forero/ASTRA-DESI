from astropy.table import Table, vstack
from astropy.cosmology import Planck18
from astropy.coordinates import SkyCoord
import astropy.units as u
import numpy as np, random
from itertools import cycle

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
        tbl = Table.read(path)
    except Exception as e:
        raise IOError(f"Error reading {path}: {e}") from e

    missing = set(columns) - set(tbl.colnames)
    if missing:
        raise KeyError(f"Missing columns {missing} in file {path}")

    try:
        tbl = tbl[columns]
        if "ROSETTE_NUMBER" in tbl.colnames:
            tbl.rename_column("ROSETTE_NUMBER", "ZONE")
        return tbl
    except Exception as e:
        raise RuntimeError(f"Error processing table columns for {path}: {e}") from e


def compute_cartesian(tbl):
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
        dist = Planck18.comoving_distance(np.asarray(tbl["Z"], float))
        ra = np.asarray(tbl["RA"], dtype=float) * u.deg
        dec = np.asarray(tbl["DEC"], dtype=float) * u.deg
        sc = SkyCoord(ra=ra, dec=dec, distance=dist)
        tbl["XCART"] = sc.cartesian.x.value
        tbl["YCART"] = sc.cartesian.y.value
        tbl["ZCART"] = sc.cartesian.z.value
        return tbl
    except Exception as e:
        raise RuntimeError(f"Error computing Cartesian coordinates: {e}") from e


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
        raise RuntimeError(f"Error determining hemisphere for zone {zone}: {e}") from e


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
            raise ValueError(f"No entries for zone {zone} in tracer '{tracer}' ({hemi})")
        sel = compute_cartesian(sel)
        sel['TRACERTYPE'] = f"{tracer}_DATA"
        sel['RANDITER'] = -1
        return sel
    except KeyError:
        raise
    except ValueError:
        raise
    except Exception as e:
        raise RuntimeError(f"Error processing real data for tracer '{tracer}', zone {zone}: {e}") from e


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
            sel = tbl[tbl["ZONE"] == zone]
            if len(sel) < real_count:
                raise ValueError(f"Zone {zone} has only {len(sel)} random points (< {real_count})")
            zone_tables.append(sel)
        n_files = len(zone_tables)

        samples, used = [], set()
        for j in range(n_random):
            if len(used) == n_files:
                used.clear()
            choices = [i for i in range(n_files) if i not in used]
            idx = random.Random(j).choice(choices)
            used.add(idx)
            sel = zone_tables[idx]
            rows = np.random.default_rng(j).choice(len(sel), real_count, replace=False)
            samp = compute_cartesian(sel[rows])
            samp["TRACERTYPE"] = f"{tracer}_RAND"
            samp["RANDITER"] = j
            samples.append(samp)

        return vstack(samples)
    except KeyError:
        raise
    except ValueError:
        raise
    except Exception as e:
        raise RuntimeError(f"Error generating randoms for tracer '{tracer}', zone {zone}: {e}") from e