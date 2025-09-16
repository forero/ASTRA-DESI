import os, argparse, json, time, numpy as np
from astropy.table import vstack, join, Table
import re

from desiproc.read_data import *
from desiproc import implement_astra as astra
from plot.plot_groups import *
from desiproc.gen_groups import process_zone


def _zone_tag(z):
    """
    Convert a zone number to a zero-padded string.

    Args:
        z (int or str): Zone number (0-99) or label (e.g., 'NGC1').
    Returns:
        str: Zero-padded zone number as a string.
    """
    try:
        return f'{int(z):02d}'
    except Exception:
        return str(z)


def _safe_tag(tag):
    """
    Make a safe tag string for filenames by replacing unsafe characters with underscores.
    
    Args:
        tag (str or None): Input tag string.
    Returns:
        str: Safe tag string prefixed with underscore, or empty string if tag is None or empty.
    """
    if tag is None:
        return ''
    safe = re.sub(r"[^A-Za-z0-9_\-\.\+]+", "_", str(tag))
    return f'_{safe}' if safe else ''


def _read_groups_compat(groups_dir, zone, webtype, out_tag=None):
    """
    Read groups FITS using a zone tag that supports numeric or string labels (e.g., 'NGC1').
    
    Args:
        groups_dir (str): Directory containing groups files.
        zone (int or str): Zone number (0-99) or label (e.g., 'NGC1').
        webtype (str): Webtype to read ('void','sheet','filament','knot').
        out_tag (str or None): Optional tag appended to filenames.
    Returns:
        Astropy Table: Table read from the specified groups file.
    Raises:
        TypeError: If the file cannot be read.
    """
    tag = _zone_tag(zone)
    tsuf = _safe_tag(out_tag)
    path = os.path.join(groups_dir, f'zone_{tag}{tsuf}_groups_fof_{webtype}.fits.gz')
    try:
        return Table.read(path, memmap=True)
    except TypeError:
        return Table.read(path)


def _read_raw_min_compat(raw_dir, class_dir, zone, out_tag=None):
    """
    Read raw (and optionally class) data with a zone tag that supports numeric or string labels.

    Args:
        raw_dir (str): Directory containing raw files.
        class_dir (str): Directory containing classification files (not used here).
        zone (int or str): Zone number (0-99) or label (e.g., 'NGC1').
        out_tag (str or None): Optional tag appended to filenames.
    Returns:
        Astropy Table: Table read from the specified raw file with selected columns.
    Raises:
        Exception: If the file cannot be read.
    """
    tag = _zone_tag(zone)
    tsuf = _safe_tag(out_tag)
    raw_path = os.path.join(raw_dir, f'zone_{tag}{tsuf}.fits.gz')
    cols = ['TARGETID','TRACERTYPE','RANDITER','RA','DEC','Z']
    try:
        raw = Table.read(raw_path, hdu=1, include_names=cols, memmap=True)
        return raw
    except Exception:
        raw = Table.read(raw_path, memmap=True)
        present = [c for c in cols if c in raw.colnames]
        return raw[present]


def preload_all_tables(base_dir, tracers, real_suffix, random_suffix, real_columns,
                       random_columns, n_random_files):
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
        raise RuntimeError(f'Error preloading tables: {e}') from e


def build_raw_region(zone_label, cuts, region, tracers, real_tables, random_tables, output_raw,
                     n_random, zone_value, out_tag=None):
    """
    Build a raw table for a DR1 sub-region (e.g., 'NGC1', 'NGC2') by combining real and random data
    filtered by RA/DEC/Z cuts. Tracers that yield no rows after cuts are skipped (warning), so the
    pipeline does not fail if a given tracer is empty in that box.
    
    Args:
        zone_label (str): Zone label (e.g., 'NGC1', 'NGC2').
        cuts (dict): Dictionary with RA/DEC/Z cuts (keys: RA_min, RA_max, DEC_min, DEC_max, Z_min, Z_max).
        region (str): 'N' or 'S' for the hemisphere.
        tracers (list): List of tracers to process.
        real_tables (dict): Preloaded real data tables.
        random_tables (dict): Preloaded random data tables.
        output_raw (str): Output directory for the raw table.
        n_random (int): Number of randoms per real object.
        zone_value (int): Integer value to assign to the ZONE column for this region.
    Returns:
        Astropy Table: Combined table with real and random data for the specified region.
    Raises:
        RuntimeError: If building or saving the raw table fails.
        ValueError: If no data is found for any tracer in the specified region and cuts.
    """
    try:
        parts = []
        skipped = []
        for tr in tracers:
            try:
                rt = process_real_region(real_tables, tr, region, cuts, zone_value=zone_value)
            except ValueError as e:
                print(f'[warn] {tr} empty after cuts in region {region}: {e}')
                skipped.append(tr)
                continue
            parts.append(rt)
            count = len(rt)
            rpt = generate_randoms_region(random_tables, tr, region, cuts, n_random, count, zone_value=zone_value)
            parts.append(rpt)

        if not parts:
            raise ValueError(f'No data in region {region} for cuts {cuts} (tracers tried: {tracers})')

        tbl = vstack(parts)
        tsuf = _safe_tag(out_tag)
        out = os.path.join(output_raw, f'zone_{zone_label}{tsuf}.fits.gz')
        tmp = out + '.tmp'
        tbl.write(tmp, format='fits', overwrite=True)
        os.replace(tmp, out)
        if skipped:
            print(f'[info] In {zone_label} skipped tracers (empty): {", ".join(skipped)}')
        return tbl
    except Exception as e:
        raise RuntimeError(f'Error building raw table for region {zone_label}: {e}') from e


def build_raw_table(zone, real_tables, random_tables, output_raw, n_random, tracers, north_rosettes, out_tag=None):
    """
    Build a raw table for a specific zone by combining real and random data.

    Args:
        zone (int): Zone number (0-19).
        real_tables (dict): Preloaded real data tables.
        random_tables (dict): Preloaded random data tables.
        output_raw (str): Output directory for the raw table.
        n_random (int): Number of randoms per real object.
        tracers (list): List of tracers to process.
        north_rosettes (set): Set of north rosette indices.
    Returns:
        Astropy Table: Combined table with real and random data for the specified zone.
    Raises:
        RuntimeError: If building or saving the raw table fails.
    """
    try:
        parts = []
        for tr in tracers:
            rt = process_real(real_tables, tr, zone, north_rosettes)
            parts.append(rt)
            count = len(rt)
            rpt = generate_randoms(random_tables, tr, zone, north_rosettes, n_random, count)
            parts.append(rpt)
        tbl = vstack(parts)
        tsuf = _safe_tag(out_tag)
        out = os.path.join(output_raw, f'zone_{zone:02d}{tsuf}.fits.gz')
        tmp = out + '.tmp'
        tbl.write(tmp, format='fits', overwrite=True)
        os.replace(tmp, out)
        return tbl
    except Exception as e:
        raise RuntimeError(f'Error building raw table for zone {zone}: {e}') from e


def classify_zone(zone, tbl, output_class, n_random, out_tag=None):
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
        tsuf = _safe_tag(out_tag)
        base = f'zone_{(f"{zone:02d}" if isinstance(zone, int) else str(zone))}{tsuf}'
        pairs_path = os.path.join(output_class, f'{base}_pairs.fits.gz')
        class_path = os.path.join(output_class, f'{base}_class.fits.gz')
        prob_path = os.path.join(output_class, f'{base}_probability.fits.gz')

        if os.path.exists(pairs_path):
            print(f'[classify] Reusing existing pairs: {pairs_path}')
            need_class = not os.path.exists(class_path)
            need_prob = not os.path.exists(prob_path)
            if need_class or need_prob:
                try:
                    ptbl = astra.load_pairs_fits(pairs_path)
                    cr = astra.build_class_rows_from_pairs(tbl, ptbl, n_random)
                    astra.save_classification_fits(cr, class_path)
                    astra.save_probability_fits(cr, prob_path, r_limit=0.9)
                except Exception as e:
                    print(f'[classify] Warning: failed to read existing pairs ({e}); recomputing pairs for {base}')
                    pr, cr, _ = astra.generate_pairs(tbl, n_random)
                    astra.save_pairs_fits(pr, pairs_path)
                    astra.save_classification_fits(cr, class_path)
                    astra.save_probability_fits(cr, prob_path, r_limit=0.9)
            else:
                print(f'[classify] Found class and probability files; skipping rebuild for {base}')
        else:
            pr, cr, _ = astra.generate_pairs(tbl, n_random)
            astra.save_pairs_fits(pr, pairs_path)
            astra.save_classification_fits(cr, class_path)
            astra.save_probability_fits(cr, prob_path, r_limit=0.9)
    except Exception as e:
        raise RuntimeError(f'Error classifying zone {zone}: {e}') from e
    

def plot_zone_wedges_for_args(z, args, plot_dir):
    """
    Generate and save the wedge plot for zone z using the parameters from args.
    Does not return anything; prints the PNG path if generated.

    Args:
        z (int or str): Zone number or label.
        args (argparse.Namespace): Command-line arguments.
        plot_dir (str): Directory to save plots.
    """
    tag = _zone_tag(z)
    try:
        groups = _read_groups_compat(args.groups_out, z, args.webtype, getattr(args, 'out_tag', None))
    except Exception as e:
        print(f'[plot] skip zone {tag}: cannot read groups ({e})')
        return

    try:
        raw = _read_raw_min_compat(args.raw_out, args.class_out, z, getattr(args, 'out_tag', None))
    except Exception as e:
        print(f'[plot] skip zone {tag}: cannot read raw/class ({e})')
        return

    gm = mask_source(np.asarray(groups['RANDITER']), args.source)
    groups = groups[gm]

    rm = mask_source(np.asarray(raw['RANDITER']), args.source)
    raw = raw[rm]

    jtbl = join(groups, raw, keys=['TARGETID','TRACERTYPE','RANDITER'], join_type='inner')

    available = tracer_prefixes(np.asarray(jtbl['TRACERTYPE']).astype(str))
    tracers = pick_tracers(available, args.plot_tracers)

    tsuf = _safe_tag(getattr(args, 'out_tag', None))
    out_png = os.path.join(plot_dir, f'groups_wedges_zone_{tag}{tsuf}_{args.webtype}.png')
    plot_wedges(jtbl, tracers, z, args.webtype, out_png, args.plot_smin, args.plot_max_z, connect_lines=args.connect_lines)
    # print(f'----- [plot] Saved {out_png}')


def main():
    try:
        p = argparse.ArgumentParser()
        p.add_argument('--base-dir', required=False, help='DESI base dir (required unless --only-plot)')
        p.add_argument('--raw-out', required=True, help='Raw output folder')
        p.add_argument('--class-out', required=True, help='Classification output folder')
        p.add_argument('--groups-out', required=True, help='Groups output folder')
        p.add_argument('--n-random', type=int, default=100, help='Number of randoms per real object')

        p.add_argument('--webtype', choices=['void','sheet','filament','knot'], default='filament', help='Webtype to group')
        p.add_argument('--source', choices=['data','rand','both'], default='data', help='Use data, randoms, or both for FoF')
        p.add_argument('--r-limit', type=float, default=0.9, help='r threshold to classify webtype')
        p.add_argument('--linking', type=str, default='{"BGS_ANY":10,"LRG":20,"ELG":20,"QSO":55,"default":10}', help='JSON-type dict of linking lengths per tracer')

        p.add_argument('--zone', type=int, default=None, help='Single zone to run (0...19)')
        p.add_argument('--plot', action='store_true', help='Generate wedge plots after grouping')
        p.add_argument('--plot-output', default=None, help='Directory to save plots (defaults to --groups-out)')
        p.add_argument('--plot-tracers', nargs='*', default=None, help='Subset of tracer prefixes to plot (e.g., BGS_ANY ELG)')
        p.add_argument('--plot-smin', type=int, default=1, help='Minimum marker size for scatter')
        p.add_argument('--plot-max-z', type=float, default=None, help='Max redshift to include in plot')
        p.add_argument('--connect-lines', action='store_true', help='Connect points in groups plot')
        p.add_argument('--only-plot', action='store_true', help='Skip preproc and only plot')
        p.add_argument('--out-tag', type=str, default=None, help='Tag appended to filenames (e.g., tracer)')
        p.add_argument('--tracers', nargs='+', default=None, help='Process only these tracers (e.g., BGS_ANY ELG LRG QSO for EDR; BGS_BRIGHT ELG_LOPnotqso LRG QSO for DR1)')

        p.add_argument('--release', choices=['EDR','DR1'], default='EDR', help='Data release: EDR (by rosette) or DR1 (by NGC/SGC)')
        p.add_argument('--region', choices=['N','S'], default='N', help='Region for DR1 (N=NGC, S=SGC). Ignored for EDR.')
        p.add_argument('--zones', nargs='+', type=str, default=None, help='For DR1: zone labels to run (e.g., NGC1 NGC2). For EDR, ignored if --zone is given.')
        p.add_argument('--config', type=str, default=None, help='Optional JSON file with cuts per label for DR1 (keys like NGC1/NGC2).')

        args = p.parse_args()

        if args.release.upper() == 'EDR':
            TRACERS = ['BGS_ANY', 'ELG', 'LRG', 'QSO']
            REAL_SUFFIX = {'N': '_N_clustering.dat.fits', 'S': '_S_clustering.dat.fits'}
            RANDOM_SUFFIX = {'N': '_N_{i}_clustering.ran.fits', 'S': '_S_{i}_clustering.ran.fits'}
            N_RANDOM_FILES = 18
            N_ZONES = 20
            NORTH_ROSETTES = {3, 6, 7, 11, 12, 13, 14, 15, 18, 19}
            REAL_COLUMNS = ['TARGETID', 'ROSETTE_NUMBER', 'RA', 'DEC', 'Z']
            RANDOM_COLUMNS = REAL_COLUMNS
        else:
            # DR1 or DR2
            TRACERS = ['BGS_BRIGHT', 'ELG_LOPnotqso', 'LRG', 'QSO']
            REAL_SUFFIX = {'N': '_N_clustering.dat.fits', 'S': '_S_clustering.dat.fits'}
            RANDOM_SUFFIX = {'N': '_N_{i}_clustering.ran.fits', 'S': '_S_{i}_clustering.ran.fits'}
            N_RANDOM_FILES = 18

            REAL_COLUMNS = ['TARGETID', 'RA', 'DEC', 'Z']
            RANDOM_COLUMNS = REAL_COLUMNS

            DEFAULT_CUTS = {'NGC1': {'RA_min':110, 'RA_max':260, 'DEC_min':-10, 'DEC_max':34},
                            'NGC2': {'RA_min':180, 'RA_max':260, 'DEC_min':30, 'DEC_max':40}}
            if args.config: # load external config json if prov
                with open(args.config, 'r') as f:
                    user_cuts = json.load(f)
                DEFAULT_CUTS.update(user_cuts)

        if args.tracers is not None:
            req = [str(t).strip() for t in args.tracers]
            rel = args.release.upper()
            alias = {}
            if rel == 'EDR':
                alias = {'bgs': 'BGS_ANY', 'elg': 'ELG', 'lrg': 'LRG', 'qso': 'QSO'}
            else:
                alias = {'bgs': 'BGS_BRIGHT', 'elg': 'ELG_LOPnotqso', 'lrg': 'LRG', 'qso': 'QSO'}
            norm = []
            for t in req:
                t_low = t.lower()
                if t in TRACERS:
                    norm.append(t)
                elif t.upper() in TRACERS:
                    norm.append(t.upper())
                elif t_low in alias:
                    norm.append(alias[t_low])
                else:
                    raise RuntimeError(f"Unknown tracer '{t}'. Available: {', '.join(TRACERS)}")

            seen = set()
            SEL_TRACERS = [x for x in norm if not (x in seen or seen.add(x))]
        else:
            SEL_TRACERS = TRACERS

        if not args.only_plot and not args.base_dir:
            raise RuntimeError('--base-dir is required unless --only-plot is specified')

        os.makedirs(args.raw_out, exist_ok=True)
        os.makedirs(args.class_out, exist_ok=True)
        os.makedirs(args.groups_out, exist_ok=True)

        plot_dir = args.plot_output or args.groups_out
        os.makedirs(plot_dir, exist_ok=True)

        i_t = time.time()

        if args.release.upper() == 'EDR':
            zones = [args.zone] if args.zone is not None else range(N_ZONES)
        else:
            zones = args.zones if args.zones is not None else ['NGC1', 'NGC2']

        if args.only_plot:
            for z in zones:
                plot_zone_wedges_for_args(z, args, plot_dir)
            print(f'--- [pipeline] only-plot elapsed t {time.time()-i_t:.2f} s')
            return

        real_tables, random_tables = preload_all_tables(args.base_dir, SEL_TRACERS,
                                                        REAL_SUFFIX, RANDOM_SUFFIX,
                                                        REAL_COLUMNS, RANDOM_COLUMNS,
                                                        N_RANDOM_FILES)

        linklen_map = json.loads(args.linking)

        for z in zones:
            if args.release.upper() == 'EDR':
                tbl = build_raw_table(int(z), real_tables, random_tables, args.raw_out, args.n_random, SEL_TRACERS, NORTH_ROSETTES, args.out_tag)
            else:
                zone_value = {'NGC1': 1001, 'NGC2': 1002}.get(str(z), 9999)
                cuts = DEFAULT_CUTS[str(z)]
                tbl = build_raw_region(str(z), cuts, 'ALL', SEL_TRACERS, real_tables, random_tables,
                                       args.raw_out, args.n_random, zone_value, args.out_tag)

            classify_zone(z, tbl, args.class_out, args.n_random, args.out_tag)

            out_groups = process_zone(z, args.raw_out, args.class_out,
                                      args.groups_out, args.webtype, args.source,
                                      linklen_map, args.r_limit, out_tag=args.out_tag)
            if out_groups is not None:
                tag = f'{z:02d}' if isinstance(z, int) else str(z)
                print(f'[groups] zone {tag} in -> {out_groups}')
                if args.plot:
                    plot_zone_wedges_for_args(z, args, plot_dir)
            else:
                tag = f'{z:02d}' if isinstance(z, int) else str(z)
                print(f'[groups] zone {tag}: no objects with WEBTYPE={args.webtype} for {args.source} source')
    except Exception as e:
        raise RuntimeError(f'Pipeline failed with: {e}') from e

    print(f'Pipeline -> elapsed t {time.time()-i_t:.2f} s')


if __name__=="__main__":
    main()