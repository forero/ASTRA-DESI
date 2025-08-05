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


def preload_real_tables(base_dir):
    """
    Preload real data tables for all tracers and hemispheres.
    
    Args:
      base_dir: Base directory containing the real data files.
    Returns:
      A dictionary with tracer names as keys and another dictionary as values,
      where the inner dictionary has hemispheres ('N' or 'S') as keys and
      Astropy Tables as values.
    """
    real = {t:{} for t in TRACERS}
    for tr in TRACERS:
        for hemi in ("N","S"):
            path = os.path.join(base_dir, tr + REAL_SUFFIX[hemi])
            real[tr][hemi] = load_table(path, REAL_COLUMNS)
    return real


def preload_random_tables(base_dir):
    """
    Preload random data tables for all tracers and hemispheres.

    Args:
      base_dir: Base directory containing the random data files.
    Returns:
      A dictionary with tracer names as keys and another dictionary as values,
      where the inner dictionary has hemispheres ('N' or 'S') as keys and
      a dictionary of random iterations as keys and Astropy Tables as values.
    """
    rand = {t:{"N":{}, "S":{}} for t in TRACERS}
    for tr in TRACERS:
        for hemi in ("N","S"):
            for i in range(N_RANDOM_FILES):
                fname = RANDOM_SUFFIX[hemi].format(i=i)
                path = os.path.join(base_dir, tr + fname)
                rand[tr][hemi][i] = load_table(path, RANDOM_COLUMNS)
    return rand


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
    """
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


def classify_zone(zone, tbl, output_class, n_random):
    """
    Classify a zone by generating pairs, classification, and probability files.
    Saves the generated files in the specified output directory.

    Args:
        zone (int): Zone number (0-19).
        tbl (Astropy Table): Input table with real and random data.
        output_class (str): Output directory for classification files.
        n_random (int): Number of randoms per real object.
    """
    base = f"zone_{zone:02d}"
    pr, cr, rdict = generate_pairs(tbl, n_random)
    save_pairs_fits(pr, os.path.join(output_class, f"{base}_pairs.fits.gz"))
    save_classification_fits(cr, os.path.join(output_class, f"{base}_class.fits.gz"))
    save_probability_fits(rdict, os.path.join(output_class, f"{base}_probability.fits.gz"))

def main():
    """
    Main function to parse arguments and run the classification process for specified zones.
    """
    p = argparse.ArgumentParser()
    p.add_argument("--base-dir", required=True, help="DESI base dir")
    p.add_argument("--raw-out", required=True, help="Raw output folder")
    p.add_argument("--class-out", required=True, help="Classification output folder")
    p.add_argument("--n-random", type=int, default=10, help="Number of randoms per real object")
    p.add_argument("--zone", type=int, default=1, help="Single zone to run (0...19)")
    args = p.parse_args()

    os.makedirs(args.raw_out, exist_ok=True)
    os.makedirs(args.class_out, exist_ok=True)

    real_tables = preload_real_tables(args.base_dir)
    random_tables = preload_random_tables(args.base_dir)

    zones = [args.zone] if args.zone is not None else range(N_ZONES)
    for z in zones:
        print(f"[Zone {z}] building raw table")
        tbl = build_raw_table(z, real_tables, random_tables, args.raw_out, args.n_random)
        print(f"[Zone {z}] classifying")
        classify_zone(z, tbl, args.class_out, args.n_random)

if __name__=="__main__":
    main()