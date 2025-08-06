import os, argparse
from astropy.table import vstack
from read_data import load_table, process_real, generate_randoms
from implement_astra import (generate_pairs,
                             save_pairs_fits,
                             save_classification_fits,
                             save_probability_fits)


TRACERS = ["BGS_ANY", "ELG", "LRG", "QSO"]
REAL_SUFFIX = {"N": "_N_clustering.dat.fits", "S": "_S_clustering.dat.fits"}
RANDOM_SUFFIX = {"N": "_N_{i}_clustering.ran.fits", "S": "_S_{i}_clustering.ran.fits"}
N_RANDOM_FILES = 18
N_ZONES = 20
NORTH_ROSETTES = {3, 6, 7, 11, 12, 13, 14, 15, 18, 19}
REAL_COLUMNS = ["TARGETID", "ROSETTE_NUMBER", "RA", "DEC", "Z"]
RANDOM_COLUMNS = REAL_COLUMNS

#! TODO  - clustering, fof? implementation

def preload_all_tables(base_dir, tracers, real_suffix, random_suffix, real_columns, random_columns, n_random_files):
    """
    Preload all real and random tables for the specified tracers.
    Args:
        base_dir (str): Base directory containing the data files.
        tracers (list): List of tracer types to process.
        real_suffix (dict): Dictionary with suffixes for real data files.
        random_suffix (dict): Dictionary with suffixes for random data files.
        real_columns (list): List of columns to load from real data files.
        random_columns (list): List of columns to load from random data files.
        n_random_files (int): Number of random files per tracer.
    Returns:
        tuple: Two dictionaries containing preloaded real and random tables.
    Raises:
        RuntimeError: If any table fails to load.
    """
    try:
        real_tables = {t: {} for t in tracers}
        rand_tables = {t: {'N': {}, 'S': {}} for t in tracers}

        for tr in tracers:
            for hemi in ('N', 'S'):
                real_path = os.path.join(base_dir, tr + real_suffix[hemi])
                real_tables[tr][hemi] = load_table(real_path, real_columns)

                for i in range(n_random_files):
                    fname = random_suffix[hemi].format(i=i)
                    path = os.path.join(base_dir, tr + fname)
                    rand_tables[tr][hemi][i] = load_table(path, random_columns)

        return real_tables, rand_tables
    except Exception as e:
        raise RuntimeError(f"Error preloading tables: {e}") from e


def build_raw_table(zone, real_tables, random_tables, output_raw, n_random):
    """
    Build a raw table for a specific zone by combining real and random data.

    Args:
        zone (int): Zone number (0-19).
        real_tables (dict): Preloaded real data tables.
        random_tables (dict): Preloaded random data tables.
        output_raw (str): Output directory for the raw table.
        n_random (int): Number of randoms per real object.
    Returns:
        Astropy Table: Combined table with real and random data for the specified zone.
    Raises:
        RuntimeError: If building or saving the raw table fails.
    """
    try:
        parts = []
        for tr in TRACERS:
            rt = process_real(real_tables, tr, zone, NORTH_ROSETTES)
            parts.append(rt)
            count = len(rt)
            rpt = generate_randoms(random_tables, tr, zone, NORTH_ROSETTES, n_random, count)
            parts.append(rpt)
        tbl = vstack(parts)
        out = os.path.join(output_raw, f"zone_{zone:02d}.fits.gz")
        tbl.write(out, format="fits", overwrite=True)#, compression="gzip")
        return tbl
    except Exception as e:
        raise RuntimeError(f"Error building raw table for zone {zone}: {e}") from e


def classify_zone(zone, tbl, output_class, n_random):
    """
    Classify a zone by generating pairs, classification, and probability files.
    Saves the generated files in the specified output directory.

    Args:
        zone (int): Zone number (0-19).
        tbl (Astropy Table): Input table with real and random data.
        output_class (str): Output directory for classification files.
        n_random (int): Number of randoms per real object.
    Raises:
        RuntimeError: If classification or saving files fails.
    """
    try:
        base = f"zone_{zone:02d}"
        pr, cr, rdict = generate_pairs(tbl, n_random)
        save_pairs_fits(pr, os.path.join(output_class, f"{base}_pairs.fits.gz"))
        save_classification_fits(cr, os.path.join(output_class, f"{base}_class.fits.gz"))
        save_probability_fits(rdict, os.path.join(output_class, f"{base}_probability.fits.gz"))
    except Exception as e:
        raise RuntimeError(f"Error classifying zone {zone}: {e}") from e


def main():
    """
    Main function to parse arguments and run the classification process for specified zones.
    Raises:
        RuntimeError: If any step in the main workflow fails.
    """
    try:
        p = argparse.ArgumentParser()
        p.add_argument("--base-dir", required=True, help="DESI base dir")
        p.add_argument("--raw-out", required=True, help="Raw output folder")
        p.add_argument("--class-out", required=True, help="Classification output folder")
        p.add_argument("--n-random", type=int, default=100, help="Number of randoms per real object")
        p.add_argument("--zone", type=int, default=1, help="Single zone to run (0...19)")
        args = p.parse_args()

        os.makedirs(args.raw_out, exist_ok=True)
        os.makedirs(args.class_out, exist_ok=True)

        real_tables, random_tables = preload_all_tables(args.base_dir, TRACERS,
                                                        REAL_SUFFIX, RANDOM_SUFFIX,
                                                        REAL_COLUMNS, RANDOM_COLUMNS,
                                                        N_RANDOM_FILES)

        zones = [args.zone] if args.zone is not None else range(N_ZONES)
        for z in zones:
            tbl = build_raw_table(z, real_tables, random_tables, args.raw_out, args.n_random)
            classify_zone(z, tbl, args.class_out, args.n_random)
    except Exception as e:
        raise RuntimeError(f"Pipeline execution failed: {e}") from e

if __name__=="__main__":
    main()
