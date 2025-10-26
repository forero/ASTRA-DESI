import os, glob, argparse, time, numpy as np
from astropy.table import vstack, join, Table

from desiproc.read_data import *
from desiproc import implement_astra as astra
from plot.plot_wedges import plot_wedges, mask_source, pick_tracers, tracer_prefixes, ALL_WEBTYPES
from desiproc.gen_groups import process_zone, classify_by_probability
from desiproc.paths import (zone_tag, safe_tag, zone_prefix,
                            classification_path, probability_path, pairs_path,
                            ensure_release_subdirs, normalize_release_dir)
from releases import RELEASE_FACTORIES


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
    tag = zone_tag(zone)
    tsuf = safe_tag(out_tag)
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
    tag = zone_tag(zone)
    tsuf = safe_tag(out_tag)
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




def classify_zone(zone, tbl, output_class, n_random, r_lower, r_upper,
                  out_tag=None, release_tag=None):
    """
    Classify a zone by generating pairs, classification, and probability files.
    Saves the generated files in the specified output directory.

    Args:
        zone (int): Zone number (0-19).
        tbl (Astropy Table): Input table with real and random data.
        output_class (str): Output directory for classification files.
        n_random (int): Number of randoms per real object.
        r_lower (float): Lower ``r`` threshold (negative).
        r_upper (float): Upper ``r`` threshold (positive).
    Raises:
        RuntimeError: If classification or saving files fails.
    """
    try:
        prefix = zone_prefix(zone, out_tag)
        pairs_file = pairs_path(output_class, zone, out_tag)
        class_file = classification_path(output_class, zone, out_tag)
        prob_file = probability_path(output_class, zone, out_tag)

        zone_header = zone_tag(zone)
        meta = {'ZONE': zone_header, 'RELEASE': str(release_tag) if release_tag is not None else 'UNKNOWN'}

        for path in (pairs_file, class_file, prob_file):
            os.makedirs(os.path.dirname(path), exist_ok=True)

        if os.path.exists(pairs_file):
            print(f'[classify] Reusing existing pairs: {pairs_file}')
            need_class = not os.path.exists(class_file)
            need_prob = not os.path.exists(prob_file)
            if need_class or need_prob:
                try:
                    ptbl = astra.load_pairs_fits(pairs_file)
                    cr = astra.build_class_rows_from_pairs(tbl, ptbl, n_random)
                    astra.save_classification_fits(cr, class_file, meta=meta)
                    astra.save_probability_fits(cr, tbl, prob_file, r_lower=r_lower, r_upper=r_upper, meta=meta)
                except Exception as e:
                    print(f'[classify] Warning: failed to read existing pairs ({e}); recomputing pairs for {prefix}')
                    pr, cr, _ = astra.generate_pairs(tbl, n_random)
                    try:
                        astra.save_pairs_fits(pr, pairs_file, meta=meta)
                        astra.save_classification_fits(cr, class_file, meta=meta)
                        astra.save_probability_fits(cr, tbl, prob_file, r_lower=r_lower, r_upper=r_upper, meta=meta)
                    finally:
                        for store in (pr, cr):
                            cleanup = getattr(store, 'cleanup', None)
                            if callable(cleanup):
                                cleanup()
            else:
                print(f'[classify] Found class and probability files; skipping rebuild for {prefix}')
        else:
            pr, cr, _ = astra.generate_pairs(tbl, n_random)
            try:
                astra.save_pairs_fits(pr, pairs_file, meta=meta)
                astra.save_classification_fits(cr, class_file, meta=meta)
                astra.save_probability_fits(cr, tbl, prob_file, r_lower=r_lower, r_upper=r_upper, meta=meta)
            finally:
                for store in (pr, cr):
                    cleanup = getattr(store, 'cleanup', None)
                    if callable(cleanup):
                        cleanup()
    except Exception as e:
        raise RuntimeError(f'Error classifying zone {zone}: {e}') from e
    

def plot_zone_wedges_for_args(z, args, plot_dir):
    """
    Generate the trio of wedge plots (groups, types, structure) for a given zone.

    Args:
        z (int | str): Zone identifier.
        args (argparse.Namespace): Command-line arguments.
        plot_dir (str): Directory to save the resulting PNG files.
    """
    tag = zone_tag(z)
    out_tag = getattr(args, 'out_tag', None)
    tsuf = safe_tag(out_tag)

    try:
        raw = _read_raw_min_compat(args.raw_out, args.class_out, z, out_tag)
    except Exception as e:
        print(f'[plot] skip zone {tag}: cannot read raw/class ({e})')
        return

    rm = mask_source(np.asarray(raw['RANDITER']), args.source)
    raw = raw[rm]
    if len(raw) == 0:
        print(f'[plot] skip zone {tag}: raw catalogue empty after filtering source={args.source}')
        return

    def _select_tracers(table):
        available = tracer_prefixes(np.asarray(table['TRACERTYPE']).astype(str))
        selected = pick_tracers(available, args.plot_tracers)
        if isinstance(selected, np.ndarray):
            values = selected.tolist()
        else:
            values = list(selected)
        return [str(val) for val in values if str(val)]

    raw_tracers = _select_tracers(raw)
    if not raw_tracers:
        print(f'[plot] skip zone {tag}: none of the requested tracers present in raw catalogue')
        return

    structure_png = os.path.join(plot_dir, f'structure_wedges_zone_{tag}{tsuf}.png')
    plot_wedges(raw, raw_tracers, z, 'all', structure_png, args.plot_smin, args.plot_max_z,
                connect_lines=args.connect_lines, color_mode='mono', title=f'Zone {tag}')

    try:
        groups = _read_groups_compat(args.groups_out, z, args.webtype, out_tag)
    except Exception as e:
        print(f'[plot] skip zone {tag}: cannot read groups ({e})')
    else:
        gm = mask_source(np.asarray(groups['RANDITER']), args.source)
        groups = groups[gm]
        if len(groups) == 0:
            print(f'[plot] skip zone {tag}: groups table empty after source={args.source}')
        else:
            join_keys = ['TARGETID', 'TRACERTYPE', 'RANDITER']
            missing_cols = [key for key in join_keys if key not in groups.colnames]
            if missing_cols:
                missing = ', '.join(missing_cols)
                print(f'[plot] skip zone {tag}: groups table missing columns: {missing}')
            else:
                jtbl = join(groups, raw, keys=join_keys, join_type='inner')
                if len(jtbl) == 0:
                    print(f'[plot] skip zone {tag}: join between groups and raw is empty')
                else:
                    group_tracers = _select_tracers(jtbl)
                    if not group_tracers:
                        print(f'[plot] skip zone {tag}: requested tracers missing in groups join')
                    else:
                        groups_png = os.path.join(
                            plot_dir,
                            f'groups_wedges_zone_{tag}{tsuf}_{args.webtype}.png',
                        )
                        plot_wedges(jtbl, group_tracers, z, args.webtype, groups_png,
                                    args.plot_smin, args.plot_max_z,
                                    connect_lines=args.connect_lines, color_mode='group',
                                    title=f'{str(args.webtype).capitalize()}s in zone {tag}')

    try:
        prob_path = probability_path(args.class_out, z, out_tag)
        prob_tbl = Table.read(prob_path, memmap=True)
    except Exception as e:
        print(f'[plot] skip zone {tag}: cannot read probabilities ({e})')
    else:
        prob_tbl = classify_by_probability(prob_tbl)
        pm = mask_source(np.asarray(prob_tbl['RANDITER']), args.source)
        prob_tbl = prob_tbl[pm]
        if len(prob_tbl) == 0:
            print(f'[plot] skip zone {tag}: probability table empty after source={args.source}')
        else:
            join_keys = ['TARGETID', 'RANDITER']
            prob_cols = set(prob_tbl.colnames)
            missing_cols = [key for key in join_keys if key not in prob_cols]
            if missing_cols:
                missing = ', '.join(missing_cols)
                print(f'[plot] skip zone {tag}: probability table missing columns: {missing}')
            elif 'WEBTYPE' not in prob_cols:
                print(f'[plot] skip zone {tag}: probability table missing WEBTYPE column')
            else:
                keep_cols = [c for c in ('TARGETID', 'RANDITER', 'WEBTYPE') if c in prob_cols]
                types_join = join(raw, prob_tbl[keep_cols], keys=join_keys, join_type='inner')
                if len(types_join) == 0:
                    print(f'[plot] skip zone {tag}: join between raw and probability tables is empty')
                else:
                    type_tracers = _select_tracers(types_join)
                    if not type_tracers:
                        print(f'[plot] skip zone {tag}: requested tracers missing in probability join')
                    else:
                        types_png = os.path.join(plot_dir, f'types_wedges_zone_{tag}{tsuf}.png')
                        plot_wedges(types_join, type_tracers, z, 'all', types_png, args.plot_smin,
                                    args.plot_max_z, connect_lines=args.connect_lines, color_mode='webtype',
                                    webtype_order=list(ALL_WEBTYPES), title=f'Web types in zone {tag}')


def _base_tracer_labels(tracer_col):
    """
    Return base tracer labels (e.g., 'LRG', 'ELG_LOPnotqso') from a TRACERTYPE column.

    Args:
        tracer_col (iterable): Sequence of TRACERTYPE values (strings or bytes).
    Returns:
        set[str]: Set of upper-case tracer identifiers without the DATA/RAND suffix.
    """
    if tracer_col is None:
        return set()
    labels = set()
    for raw in tracer_col:
        name = str(raw)
        if not name:
            continue
        base = name.rsplit('_', 1)[0]
        labels.add(base.upper())
    return labels


def _stack_tables(parts):
    """
    Vertically stack astropy tables ignoring ``None`` entries.

    Args:
        parts (list[Table | None]): Tables to stack.
    Returns:
        Table | None: Stacked table or ``None`` when no parts are left.
    """
    valid = [t for t in parts if t is not None]
    if not valid:
        return None
    if len(valid) == 1:
        return valid[0]
    return vstack(valid, metadata_conflicts='silent')


def _read_table_if_exists(path):
    """
    Read astropy table if exists.
    
    Args:
        path (str or None): Path to the FITS file.
    Returns:
        Table | None: Table read from the file or ``None`` if the file does not exist.
    Raises:
        Exception: If reading the file fails for reasons other than non-existence.
    """
    if path is None or not os.path.exists(path):
        return None
    try:
        return Table.read(path, memmap=True)
    except TypeError:
        return Table.read(path)


def _write_table_with_meta(tbl, path, zone, release_tag):
    """
    Persist ``tbl`` to ``path`` ensuring standard metadata is present.
    
    Args:
        tbl (Table | None): Table to write.
        path (str): Output FITS file path.
        zone (int | str): Zone identifier for metadata.
        release_tag (str | None): Release label for metadata.
    Returns:
        None
    """
    if tbl is None:
        return
    tbl.meta['ZONE'] = zone_tag(zone)
    tbl.meta['RELEASE'] = str(release_tag) if release_tag is not None else ''
    tmp = f'{path}.tmp'
    tbl.write(tmp, format='fits', overwrite=True)
    os.replace(tmp, path)


def combine_zone_products(zone, args, release_tag):
    """
    Merge all existing per-tracer outputs for ``zone`` into the shared EDR-style files.

    Args:
        zone (int | str): Zone identifier being processed.
        args (argparse.Namespace): Parsed CLI arguments.
        release_tag (str): Release label stored in FITS headers.
    """
    ztag = zone_tag(zone)
    base_prefix = f'zone_{ztag}'

    def _collect_paths(directory, suffix):
        if not directory or not os.path.isdir(directory):
            return []
        paths = []
        for ext in ('.fits.gz', '.fits'):
            pattern = os.path.join(directory, f'{base_prefix}_*{suffix}{ext}')
            paths.extend(sorted(glob.glob(pattern)))
        return paths

    raw_paths = _collect_paths(args.raw_out, '')
    if not raw_paths:
        return

    raw_tables = [_read_table_if_exists(path) for path in raw_paths]
    merged_raw = _stack_tables(raw_tables)
    if merged_raw is not None:
        combined_raw_path = os.path.join(args.raw_out, f'{zone_prefix(zone)}.fits.gz')
        _write_table_with_meta(merged_raw, combined_raw_path, zone, release_tag)

    class_dir = os.path.join(args.class_out, 'classification')
    class_paths = _collect_paths(class_dir, '_classified')
    if class_paths:
        class_tables = [_read_table_if_exists(path) for path in class_paths]
        merged_class = _stack_tables(class_tables)
        if merged_class is not None:
            combined_class_path = classification_path(args.class_out, zone, None)
            _write_table_with_meta(merged_class, combined_class_path, zone, release_tag)

    prob_dir = os.path.join(args.class_out, 'probabilities')
    prob_paths = _collect_paths(prob_dir, '_probability')
    if prob_paths:
        prob_tables = [_read_table_if_exists(path) for path in prob_paths]
        merged_prob = _stack_tables(prob_tables)
        if merged_prob is not None:
            combined_prob_path = probability_path(args.class_out, zone, None)
            _write_table_with_meta(merged_prob, combined_prob_path, zone, release_tag)

    pairs_dir = os.path.join(args.class_out, 'pairs')
    pairs_paths = _collect_paths(pairs_dir, '_pairs')
    if pairs_paths:
        pairs_tables = [_read_table_if_exists(path) for path in pairs_paths]
        merged_pairs = _stack_tables(pairs_tables)
        if merged_pairs is not None:
            combined_pairs_path = pairs_path(args.class_out, zone, None)
            _write_table_with_meta(merged_pairs, combined_pairs_path, zone, release_tag)

    groups_paths = []
    if os.path.isdir(args.groups_out):
        suffix = f'_groups_fof_{args.webtype}'
        for ext in ('.fits.gz', '.fits'):
            pattern = os.path.join(args.groups_out, f'{base_prefix}_*{suffix}{ext}')
            groups_paths.extend(sorted(glob.glob(pattern)))
    if groups_paths:
        groups_tables = [_read_table_if_exists(path) for path in groups_paths]
        merged_groups = _stack_tables(groups_tables)
        if merged_groups is not None:
            combined_groups_path = os.path.join(args.groups_out, f'{base_prefix}{suffix}.fits.gz')
            _write_table_with_meta(merged_groups, combined_groups_path, zone, release_tag)

    tracers_list = ', '.join(sorted(_base_tracer_labels(merged_raw['TRACERTYPE'])) if merged_raw is not None else [])
    print(f'[combine] zone {ztag}: combined raw tables from tracers {tracers_list or "unknown"}')


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
        p.add_argument('--r-lower', type=float, default=-0.9,
                       help='Lower r threshold used to classify web types (default: -0.9)')
        p.add_argument('--r-upper', type=float, default=0.9,
                       help='Upper r threshold used to classify web types (default: 0.9)')
        p.add_argument('--r-limit', type=float, default=None,
                       help='[Deprecated] Symmetric absolute threshold; overrides --r-lower/--r-upper when set')

        p.add_argument('--zone', type=int, default=None, help='Single zone to run (0...19)')
        p.add_argument('--plot', action='store_true', help='Generate wedge plots after grouping')
        p.add_argument('--plot-output', default=None, help='Directory to save plots (defaults to --groups-out)')
        p.add_argument('--plot-tracers', nargs='*', default=None, help='Subset of tracer prefixes to plot (e.g., BGS_ANY ELG)')
        p.add_argument('--plot-smin', type=int, default=1, help='Minimum marker size for scatter')
        p.add_argument('--plot-max-z', type=float, default=None, help='Max redshift to include in plot')
        p.add_argument('--connect-lines', action='store_true', help='Connect points in groups plot')
        p.add_argument('--only-plot', action='store_true', help='Skip preproc and only plot')
        p.add_argument('--combine-only', action='store_true', help='Skip processing and only merge per-tracer outputs into combined files')
        p.add_argument('--out-tag', type=str, default=None, help='Tag appended to filenames (e.g., tracer)')
        p.add_argument('--tracers', nargs='+', default=None,
                       help='Process only these tracers (e.g., BGS_ANY ELG LRG QSO for EDR; BGS_BRIGHT ELG_LOPnotqso LRG QSO for DR1)')

        p.add_argument('--release', choices=['EDR','DR1','DR2'], default='EDR',
                       help='Data release: EDR (rosettes), DR1 (NGC1/NGC2 boxes), DR2 (full-sky NGC/SGC split)')
        p.add_argument('--region', choices=['N','S'], default='N', help='Region for DR1 (N=NGC, S=SGC). Ignored for EDR.')
        p.add_argument('--zones', nargs='+', type=str, default=None,
                       help='For DR1: zone labels to run (e.g., NGC1 NGC2). For EDR, ignored if --zone is given.')
        p.add_argument('--config', type=str, default=None, help='Optional JSON file with cuts per label for DR1 (keys like NGC1/NGC2).')

        args = p.parse_args()

        if args.r_limit is not None:
            sym = float(abs(args.r_limit))
            args.r_lower = -sym
            args.r_upper = sym
        if args.r_lower >= 0 or args.r_upper <= 0:
            raise ValueError('--r-lower must be negative and --r-upper must be positive.')

        release = args.release.upper()

        config_factory = RELEASE_FACTORIES.get(release)
        if config_factory is None:
            raise RuntimeError(f"Unsupported release '{args.release}'")

        release_config = config_factory(args)
        available_tracers = list(release_config.tracers)
        tracer_alias = dict(release_config.tracer_alias)

        if args.tracers is not None:
            req = [str(t).strip() for t in args.tracers]
            norm = []
            for t in req:
                t_low = t.lower()
                t_up = t.upper()
                if t in available_tracers:
                    norm.append(t)
                elif t_up in available_tracers:
                    norm.append(t_up)
                elif t_low in tracer_alias:
                    norm.append(tracer_alias[t_low])
                else:
                    raise RuntimeError(f"Unknown tracer '{t}'. Available: {', '.join(available_tracers)}")

            seen = set()
            SEL_TRACERS = [x for x in norm if not (x in seen or seen.add(x))]
        else:
            SEL_TRACERS = list(available_tracers)

        if not args.only_plot and not args.combine_only and not args.base_dir:
            raise RuntimeError('--base-dir is required unless --only-plot is specified')

        release_tag = release_config.release_tag

        os.makedirs(args.raw_out, exist_ok=True)
        class_root = normalize_release_dir(args.class_out)
        os.makedirs(class_root, exist_ok=True)
        ensure_release_subdirs(class_root)
        os.makedirs(args.groups_out, exist_ok=True)

        args.class_out = class_root

        plot_dir = args.plot_output or args.groups_out
        os.makedirs(plot_dir, exist_ok=True)

        print(f'--- [pipeline] Starting release {release} with tracers: {", ".join(SEL_TRACERS)}')
        pipeline_start = time.time()

        zones = list(release_config.zones)

        if args.only_plot:
            for z in zones:
                plot_zone_wedges_for_args(z, args, plot_dir)
            print(f'--- [pipeline] only-plot elapsed t {time.time()-pipeline_start:.2f} s')
            return

        if args.combine_only:
            if release_config.combine_outputs:
                for z in zones:
                    combine_zone_products(z, args, release_tag)
                print(f'--- [pipeline] combine-only elapsed t {time.time()-pipeline_start:.2f} s')
            else:
                print(f'[combine] skipping combine-only operation for release {release} (disabled)')
            return

        real_tables, random_tables = {}, {}
        need_preload = True
        if release_config.use_dr2_preload and release_config.name.upper() == 'DR2':
            suffix = safe_tag(args.out_tag)
            def _raw_exists(zone):
                ztag = zone_tag(zone)
                candidates = [os.path.join(args.raw_out, f'zone_{ztag}{suffix}{ext}')
                              for ext in ('.fits.gz', '.fits')]
                for candidate in candidates:
                    if os.path.exists(candidate):
                        return candidate
                return None

            missing = [z for z in zones if _raw_exists(z) is None]
            if not missing:
                need_preload = False
                print('[dr2] all requested raw files already exist; skipping preload step', flush=True)

        if release_config.use_dr2_preload and need_preload:
            real_tables, random_tables = preload_dr2_tables(args.base_dir,
                                                            SEL_TRACERS,
                                                            release_config.real_columns,
                                                            release_config.random_columns,
                                                            release_config.n_random_files,
                                                            **release_config.preload_kwargs)
        elif not release_config.use_dr2_preload:
            real_tables, random_tables = preload_all_tables(args.base_dir,
                                                            SEL_TRACERS,
                                                            release_config.real_suffix,
                                                            release_config.random_suffix,
                                                            release_config.real_columns,
                                                            release_config.random_columns,
                                                            release_config.n_random_files)

        for z in zones:
            stage_start = time.time()
            tbl = release_config.build_raw(z, real_tables, random_tables, SEL_TRACERS,
                                           args, release_tag)
            print(f'-- [pipeline] Built raw zone {z} with {len(tbl)} rows in {time.time()-stage_start:.2f} s')

            stage_start = time.time()
            classify_zone(z, tbl, args.class_out, args.n_random,
                          args.r_lower, args.r_upper,
                          out_tag=args.out_tag, release_tag=release_tag)
            print(f'-- [pipeline] Classified zone {z} in {time.time()-stage_start:.2f} s')

            stage_start = time.time()
            outputs = process_zone(z, args.raw_out, args.class_out, args.groups_out,
                                   args.webtype, args.source, args.r_lower, args.r_upper,
                                   release_tag=release_tag, out_tag=args.out_tag)
            print(f'-- [pipeline] Grouped zone {z} in {time.time()-stage_start:.2f} s')

            if release_config.combine_outputs:
                combine_zone_products(z, args, release_tag)

            if outputs:
                tag = f'{z:02d}' if isinstance(z, int) else str(z)
                if args.plot:
                    stage_start = time.time()
                    plot_zone_wedges_for_args(z, args, plot_dir)
                    print(f'--- [pipeline] Plotted zone {tag} in {time.time()-stage_start:.2f} s')
            else:
                tag = f'{z:02d}' if isinstance(z, int) else str(z)
            
    except Exception as e:
        raise RuntimeError(f'Pipeline failed with: {e}') from e

    print(f'Pipeline -> elapsed t {time.time()-pipeline_start:.2f} s')


if __name__=="__main__":
    main()