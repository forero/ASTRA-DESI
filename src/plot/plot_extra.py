import argparse, os, re
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import pandas as pd
from astropy.table import Table

import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['text.usetex'] = True

from desiproc.paths import (classification_path, locate_classification_file, locate_probability_file,
                            normalize_release_dir, probability_path, safe_tag, zone_tag)

try:
    from .common import (load_probability_dataframe, load_raw_dataframe, resolve_class_path,
                         resolve_probability_path, resolve_raw_path)
except ImportError:
    import sys
    from pathlib import Path

    src_root = Path(__file__).resolve().parents[1]
    if str(src_root) not in sys.path:
        sys.path.append(str(src_root))
    from plot.common import (load_probability_dataframe, load_raw_dataframe, resolve_class_path,
                             resolve_probability_path, resolve_raw_path)

PLOT_DPI = 360
CACHE_VERSION = 'v1'


def _max_mtime(paths):
    """
    Return the maximum modification time among existing files in *paths*.
    
    Args:
        paths (list[str]): List of file paths.
    Returns:
        float | None: Maximum modification time in seconds since epoch, or None if no files exist.
    """
    times = [os.path.getmtime(p) for p in paths if p and os.path.exists(p)]
    return max(times) if times else None


def _cache_file(cache_dir, key):
    """
    Return the path to a cache file for *key* in *cache_dir*.
    
    Args:
        cache_dir (str | None): Directory to store cache files, or None to disable caching.
        key (str): Unique key identifying the cache file.
    Returns:
        str | None: Full path to the cache file, or None if caching is disabled.
    """
    if cache_dir is None:
        return None
    os.makedirs(cache_dir, exist_ok=True)
    fname = f'{CACHE_VERSION}_{key}.pkl'
    return os.path.join(cache_dir, fname)


def _load_or_build_df(cache_dir, key, source_paths, builder, progress=False):
    """
    Load a DataFrame from cache or build it using *builder*.
    Cache invalidates automatically if any source path is newer than the cache file.
    
    Args:
        cache_dir (str | None): Directory to store cache files, or None to disable caching.
        key (str): Unique key identifying the cache file.
        source_paths (list[str]): List of source file paths that the DataFrame depends on.
        builder (callable): Function that returns a pd.DataFrame when called.
        progress (bool): If True, print progress messages.
    Returns:
        pd.DataFrame: The loaded or newly built DataFrame.
    """
    cache_path = _cache_file(cache_dir, key)
    src_mtime = _max_mtime(source_paths)
    if cache_path and os.path.exists(cache_path):
        cache_mtime = os.path.getmtime(cache_path)
        if src_mtime is None or cache_mtime >= src_mtime:
            if progress:
                print(f'[cache hit] {key}')
            return pd.read_pickle(cache_path)

    df = builder()
    if cache_path:
        tmp_path = f'{cache_path}.tmp'
        try:
            df.to_pickle(tmp_path)
            os.replace(tmp_path, cache_path)
            if progress:
                print(f'[cache write] {key}')
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
    return df


def _zone_cache_key(kind, zone, tag_list):
    """
    Create a cache key for a zone and list of tags.
    
    Args:
        kind (str): Kind of data (e.g., 'raw', 'class', 'r').
        zone (int|str): Zone identifier (e.g., 0, 01, 12, or 'NGC1').
        tag_list (list[str]): List of out-tags ('' represents the combined run).
    Returns:
        str: Cache key string.
    """
    tag_suffix = ''.join(safe_tag(t) for t in tag_list if t)
    return f'{kind}_zone_{zone_tag(zone)}{tag_suffix}'



def get_zone_paths(raw_dir, class_dir, zone, out_tag=None):
    """
    Get the paths to raw and classification files for a specific zone.
    
    Args:
        raw_dir (str): Path to the raw data directory.
        class_dir (str): Path to the classification directory.
        zone (int|str): Zone identifier (e.g., 0, 01, 12, or 'NGC1').
        out_tag (str|None): Optional output tag for the files.
    Returns:
        tuple[str, str]: Paths to the raw and classification files.
    """
    raw_path = resolve_raw_path(raw_dir, zone, out_tag)
    class_path = resolve_class_path(class_dir, zone, out_tag)
    return raw_path, class_path


def get_prob_path(raw_dir, class_dir, zone, out_tag=None):
    """
    Get the path to the probability file for a specific zone.
    
    Args:
        raw_dir (str): Path to the raw data directory.
        class_dir (str): Path to the classification directory.
        zone (int|str): Zone identifier (e.g., 0, 01, 12, or 'NGC1').
        out_tag (str|None): Optional output tag for the file.
    Returns:
        str: Path to the probability file.
    """
    return resolve_probability_path(class_dir, zone, out_tag)


def _expected_class_path(class_dir, zone, tag):
    """
    Get the expected classification file path, handling legacy layout.
    
    Args:
        class_dir (str): Release directory containing the `classification` subfolder.
        zone (int|str): Zone identifier (e.g., 0, 01, 12, or 'NGC1').
        tag (str|None): Optional output tag for the file.
    Returns:
        str: Path to the classification file.
    Raises:
        FileNotFoundError: If the classification file cannot be found.
    """
    try:
        return locate_classification_file(class_dir, zone, tag)
    except FileNotFoundError:
        return classification_path(class_dir, zone, tag)


def _expected_prob_path(class_dir, zone, tag):
    """
    Get the expected probability file path, handling legacy layout.
    
    Args:
        class_dir (str): Release directory containing the `classification` subfolder.
        zone (int|str): Zone identifier (e.g., 0, 01, 12, or 'NGC1').
        tag (str|None): Optional output tag for the file.
    Returns:
        str: Path to the probability file.
    Raises:
        FileNotFoundError: If the probability file cannot be found.
    """
    try:
        return locate_probability_file(class_dir, zone, tag)
    except FileNotFoundError:
        return probability_path(class_dir, zone, tag)


def _raw_candidate(raw_dir, zone, tag):
    """
    Get the candidate raw file path, handling legacy layout.
    
    Args:
        raw_dir (str): Raw directory path (used to validate raw files exist).
        zone (int|str): Zone identifier (e.g., 0 or 'NGC1').
        tag (str|None): Optional output tag for the file.
    Returns:
        str: Path to the raw file.
    Raises:
        FileNotFoundError: If the raw file cannot be found.
    """
    try:
        return resolve_raw_path(raw_dir, zone, tag)
    except FileNotFoundError:
        zone_str = zone_tag(zone)
        suffix = safe_tag(tag)
        return os.path.join(raw_dir, f'zone_{zone_str}{suffix}.fits.gz')


def list_zone_out_tags(raw_dir, class_dir, zone):
    """
    Discover available out-tags for a zone based on the new release layout.

    Args:
        raw_dir (str): Raw directory path (used to validate raw files exist).
        class_dir (str): Release directory containing the `classification` subfolder.
        zone (int|str): Zone identifier (e.g., 00 or 'NGC1').
    Returns:
        list[str]: Sorted list of out-tags ('' represents the combined run).
    """
    ztag = zone_tag(zone)
    class_root = os.path.join(class_dir, 'classification')
    if not os.path.isdir(class_root):
        return []

    pattern = re.compile(rf'^zone_{re.escape(ztag)}(?:_(?P<tag>.+))?_classified\.fits\.gz$')
    discovered = []
    for fname in os.listdir(class_root):
        match = pattern.match(fname)
        if not match:
            continue
        tag_part = match.group('tag') or ''
        tag = '' if tag_part in {'', 'combined'} else tag_part
        raw_path = _raw_candidate(raw_dir, zone, tag or None)
        if os.path.exists(raw_path):
            discovered.append(tag)

    if not discovered:
        combined_path = classification_path(class_dir, zone, None)
        if os.path.exists(combined_path):
            return ['']
        return []

    ordered = []
    seen = set()
    for tag in sorted(discovered):
        key = tag or ''
        if key not in seen:
            seen.add(key)
            ordered.append(key)
    return ordered


def infer_zones(raw_dir, provided):
    """
    Infer available zones from raw directory if not provided.

    Supports both numeric zones (e.g., 0, 01, 12) and string labels
    (e.g., 'NGC1'). Returns a list of zone identifiers as strings
    or ints depending on input in `provided`. When inferred from
    filenames, returns the raw tag from filenames without casting.

    Args:
        raw_dir (str): Directory containing raw data files.
        provided (list or None): List of zone identifiers if provided.
    Returns:
        list: List of zone identifiers (strings like 'NGC1' or numeric-like strings).
    Raises:
        ValueError: If provided is not a list or None.
    """
    if provided:
        return list(dict.fromkeys(provided))

    pat = re.compile(r"^zone_([^_.]+)(?:_.+)?\.fits(?:\.gz)?$")
    tags = set()
    for f in os.listdir(raw_dir):
        m = pat.match(f)
        if m:
            tags.add(m.group(1))

    def _key(t):
        try:
            return (0, int(t))
        except Exception:
            return (1, str(t))

    return sorted(tags, key=_key)


def make_output_dirs(base):
    """
    Create output directories for plots.

    Args:
        base (str): Base output directory.
    Returns:
        dict: Dictionary with paths for different plot types.
    """
    out = {'z': os.path.join(base, 'z_histograms'),
           'radial': os.path.join(base, 'radial'),
           'cdf': os.path.join(base, 'cdf')}
    for d in out.values():
        os.makedirs(d, exist_ok=True)
    return out


def load_raw_df(path):
    """
    Load raw data from FITS file into a pandas DataFrame.
    
    Args:
        path (str): Path to the FITS file.
    Returns:
        pd.DataFrame: DataFrame containing the raw data.
    """
    return load_raw_dataframe(path)


def load_class_df(path):
    """
    Load classification data from FITS file into a pandas DataFrame.

    Args:
        path (str): Path to the FITS file.
    Returns:
        pd.DataFrame: DataFrame containing the classification data.
    """
    try:
        tbl = Table.read(path, hdu=1, include_names=['TARGETID','NDATA','NRAND','ISDATA'], memmap=True)
    except Exception:
        tbl = Table.read(path, memmap=True)
    df = tbl.to_pandas()
    return df[['TARGETID','NDATA','NRAND','ISDATA']]


def load_prob_df(path):
    """
    Load probability data from FITS file into a pandas DataFrame.
    
    Args:
        path (str): Path to the FITS file.
    Returns:
        pd.DataFrame: DataFrame containing the probability data with CLASS column.
    """

    frame = load_probability_dataframe(path)
    prob_cols = ['PVOID', 'PSHEET', 'PFILAMENT', 'PKNOT']
    frame['CLASS'] = frame[prob_cols].idxmax(axis=1).str[1:].str.lower()
    return frame[['TARGETID', 'CLASS', 'PVOID', 'PSHEET', 'PFILAMENT', 'PKNOT']]


def _concat_existing(paths, loader):
    """
    Concatenate DataFrames loaded from existing files in *paths* using *loader*.
    
    Args:
        paths (list[str]): List of file paths.
        loader (callable): Function that takes a file path and returns a pd.DataFrame.
    Returns:
        pd.DataFrame: Concatenated DataFrame from existing files.
    """
    dfs = []
    for p in paths:
        if os.path.exists(p):
            dfs.append(loader(p))
    if not dfs:
        raise FileNotFoundError(f'None of the expected files exist: {paths}')
    if len(dfs) == 1:
        return dfs[0]
    return pd.concat(dfs, ignore_index=True, copy=False)


def compute_r(df):
    """
    Compute the r for each entry in the DataFrame.

    Args:
        df (pd.DataFrame): DataFrame containing 'NDATA' and 'NRAND' columns.
    Returns:
        pd.DataFrame: DataFrame with an additional 'r' column.
    """
    df = df.copy()
    denom = (df['NDATA'] + df['NRAND']).to_numpy()
    num = (df['NDATA'] - df['NRAND']).to_numpy()
    r = np.full_like(num, np.nan, dtype=float)
    np.divide(num, denom, out=r, where=(np.isfinite(denom) & (denom > 0)))
    df['r'] = r
    return df


def plot_z_histogram(df, zone, bins, out_dir):
    """
    Plot histogram of Z values for real and random data.

    Args:
        df (pd.DataFrame): DataFrame containing 'Z' and 'ISDATA' columns.
        zone (int): Zone number for title and filename.
        bins (int): Number of bins for the histogram.
        out_dir (str): Output directory to save the plot.
    """
    real, rand = df[df['ISDATA']]['Z'], df[~df['ISDATA']]['Z']

    fig, ax = plt.subplots()
    ax.grid(linewidth=0.7)
    ax.hist([real, rand], bins=bins, label=['Data','Random'], alpha=0.7)
    ax.set(xlabel='$z$', ylabel='Count', title=f'Zone {zone}')
    ax.legend()

    path = os.path.join(out_dir, f'z_hist_zone_{zone}.png')
    fig.savefig(path, dpi=PLOT_DPI); plt.close(fig)


def plot_radial_distribution(raw_df, zone, tracers, out_dir, bins):
    """
    Plot radial distribution histograms for specified tracers.

    Args:
        raw_df (pd.DataFrame): DataFrame containing raw data with 'XCART', 
                    'YCART', 'ZCART', 'TRACERTYPE', and 'RANDITER' columns.
        zone (int): Zone number for title and filename.
        tracers (list): List of tracer types to plot.
        out_dir (str): Output directory to save the plots.
        bins (int): Number of bins for the histograms.
    """
    fig, axes = plt.subplots(1, len(tracers), figsize=(4*len(tracers),4))
    if len(tracers) == 1:
        axes = [axes]

    for ax, tracer in zip(axes, tracers):
        tr_prefix = 'BGS' if tracer == 'BGS_BRIGHT' else tracer
        sub = raw_df[raw_df['TRACERTYPE'].str.startswith(tr_prefix)]
        real = sub[sub['RANDITER'] == -1]
        rand_pool = sub[sub['RANDITER'] != -1]
        if len(rand_pool) == 0 or len(real) == 0:
            ax.set(title=tr_prefix, xlabel=r'$r$', ylabel='Count')
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            ax.grid(linewidth=0.7)
            continue
        nsamp = min(len(rand_pool), len(real))
        rand = rand_pool.sample(nsamp, random_state=0, replace=False)

        r_real = np.linalg.norm(real[['XCART','YCART','ZCART']].to_numpy(), axis=1)
        r_rand = np.linalg.norm(rand[['XCART','YCART','ZCART']].to_numpy(), axis=1)

        ax.grid(linewidth=0.7)
        ax.hist([r_real, r_rand], bins, label=['Real','Random'], alpha=0.7)
        ax.set(title=tr_prefix, xlabel=r'$r$', ylabel='Count')
        ax.legend(fontsize=6)

    fig.suptitle(f'Radial Zone {zone}')
    fig.tight_layout()
    path = os.path.join(out_dir, f'radial_zone_{zone}.png')
    fig.savefig(path, dpi=PLOT_DPI)
    plt.close(fig)


def plot_cdf(df_r, zone, tracers, out_dir):
    """
    Plot CDF of r values for specified tracers using NumPy ECDF.
    
    Args:
        df_r (pd.DataFrame): DataFrame containing 'r' values and 'TRACERTYPE' columns.
        zone (int): Zone number for title and filename.
        tracers (list): List of tracer types to plot.
        out_dir (str): Output directory to save the plot.
    """
    cmap = {'BGS_BRIGHT':'blue','ELG':'red','LRG':'green','QSO':'purple'}
    fig, ax = plt.subplots()
    ax.grid(linewidth=0.7)

    def ecdf(arr):
        a = np.asarray(arr, dtype=float)
        a = a[~np.isnan(a)]
        if a.size == 0:
            return None, None
        a = np.sort(a)
        y = np.arange(1, a.size + 1) / a.size
        return a, y

    for tr in tracers:
        tr_key = str(tr).upper()
        tr_prefix = 'BGS' if tr_key.startswith('BGS') else tr_key
        sub = df_r[df_r['TRACERTYPE'].str.startswith(tr_prefix)]
        xr, yr = ecdf(sub[sub['ISDATA']]['r'])
        if xr is not None:
            ax.plot(xr, yr, label=f'{tr_prefix} real', linewidth=1, color=cmap.get(tr_key, 'black'))
        xr, yr = ecdf(sub[~sub['ISDATA']]['r'])
        if xr is not None:
            ax.plot(xr, yr, label=f'{tr_prefix} rand', linewidth=1, linestyle='--', color=cmap.get(tr_key, 'black'))

    ax.set(xlabel='$r$', ylabel='CDF', title=f'CDF Zone {zone}')
    ax.legend(fontsize=6)
    path = os.path.join(out_dir, f'cdf_zone_{zone}.png')
    fig.savefig(path, dpi=PLOT_DPI)
    plt.close(fig)


def plot_cdf_dispersion(raw_dir, class_dir, zones, out_dir, tracers=None, xbins=400,
                        subsample_per_zone=None, progress=False, out_tags=None, cache_dir=None, tags_map=None):
    """
    Plot the dispersion (percentile band) of CDFs over multiple zones in one figure.
    
    Args:
        raw_dir (str): Directory with raw zone files.
        class_dir (str): Directory with class zone files.
        zones (list[int]): Zone numbers to include (e.g., range(20)).
        out_dir (str): Base output directory (the figure is written under `<out_dir>/cdf`).
        tracers (list[str] or None): Tracers to include. Default: ['BGS_BRIGHT','ELG','LRG','QSO'].
        xbins (int): Number of points in the common x-grid for interpolation.
        subsample_per_zone (int or None): Max samples per (zone,tracer,real/rand) for ECDF calculation.
        progress (bool): Print progress logs.
        out_tags (list[str] or None): Override tag selection per zone.
        cache_dir (str or None): Directory used to persist intermediate DataFrames.
        tags_map (dict|None): Precomputed tag list per zone (overrides discovery when present).
    """
    if tracers is None:
        tracers = ['BGS_BRIGHT','ELG','LRG','QSO']

    def _ecdf_interp(arr, xgrid):
        a = np.asarray(arr, dtype=float)
        a = a[~np.isnan(a)]
        if a.size == 0:
            return np.full_like(xgrid, np.nan, dtype=float)
        a.sort()
        y = np.arange(1, a.size + 1) / a.size
        return np.interp(xgrid, a, y, left=0.0, right=1.0)

    xgrid = np.linspace(-1.0, 1.0, xbins)

    if subsample_per_zone is not None and subsample_per_zone <= 0:
        subsample_per_zone = None

    per_tracer_real = {t: [] for t in tracers}
    per_tracer_rand = {t: [] for t in tracers}

    for i, z in enumerate(zones):
        if tags_map and z in tags_map:
            tag_list = tags_map[z]
        else:
            tags = out_tags if out_tags else list_zone_out_tags(raw_dir, class_dir, z)
            tag_list = tags if tags else ['']
        if progress:
            print(f'[cdf-disp] zone {z}: tags={tag_list if any(tag_list) else "(legacy single-file)"}')
        ztag = zone_tag(z)
        raw_paths = [_raw_candidate(raw_dir, z, t or None) for t in tag_list]
        cls_paths = [_expected_class_path(class_dir, z, t or None) for t in tag_list]

        raw_df = _load_or_build_df(cache_dir, _zone_cache_key('raw', z, tag_list), raw_paths,
                                   lambda: _concat_existing(raw_paths, load_raw_df), progress=progress)
        cls_df = _load_or_build_df(cache_dir, _zone_cache_key('class', z, tag_list), cls_paths,
                                   lambda: _concat_existing(cls_paths, load_class_df), progress=progress)
        cls_df = cls_df[cls_df['ISDATA'] == True]

        def _build_r():
            merged = raw_df.merge(cls_df[['TARGETID','NDATA','NRAND']], on='TARGETID', how='left')
            return compute_r(merged)

        r_df = _load_or_build_df(cache_dir, _zone_cache_key('r', z, tag_list), raw_paths + cls_paths,
                                 _build_r, progress=progress)

        for tr in tracers:
            tr_prefix = 'BGS' if tr == 'BGS_BRIGHT' else tr
            sub = r_df[r_df['TRACERTYPE'].str.startswith(tr_prefix)]
            real_r = sub[sub['ISDATA']]['r'].to_numpy()
            rand_r = sub[~sub['ISDATA']]['r'].to_numpy()

            if subsample_per_zone is not None:
                if real_r.size > subsample_per_zone:
                    idx = np.random.default_rng(0).choice(real_r.size, subsample_per_zone, replace=False)
                    real_r = real_r[idx]
                if rand_r.size > subsample_per_zone:
                    idx = np.random.default_rng(0).choice(rand_r.size, subsample_per_zone, replace=False)
                    rand_r = rand_r[idx]
            y_real = _ecdf_interp(real_r, xgrid)
            y_rand = _ecdf_interp(rand_r, xgrid)
            if np.isfinite(y_real).any():
                per_tracer_real[tr].append(y_real)
            if np.isfinite(y_rand).any():
                per_tracer_rand[tr].append(y_rand)

    fig, ax = plt.subplots(figsize=(9, 7))
    ax.grid(linewidth=0.7)

    tracer_color_map = {'BGS_BRIGHT': 'blue', 'ELG': 'red', 'LRG': 'green', 'QSO': 'orange'}

    for tr in tracers:
        color = tracer_color_map.get(tr, 'black')

        if len(per_tracer_real[tr]) > 0:
            Y = np.vstack(per_tracer_real[tr])
            p16, p50, p84 = np.nanpercentile(Y, [16,50,84], axis=0)
            ax.fill_between(xgrid, p16, p84, alpha=0.15, label=f'{tr} real ±1σ', color=color)
            ax.plot(xgrid, p50, linewidth=1.5, color=color, label=f'{tr} real median')

        if len(per_tracer_rand[tr]) > 0:
            Y = np.vstack(per_tracer_rand[tr])
            p16, p50, p84 = np.nanpercentile(Y, [16,50,84], axis=0)
            ax.fill_between(xgrid, p16, p84, alpha=0.10, label=f'{tr} rand ±1σ', color=color)
            ax.plot(xgrid, p50, linewidth=1.2, linestyle='--', color=color, label=f'{tr} rand median')

    ax.set_ylabel('CDF')
    ax.set_xlabel(r"$r = \frac{N_{\mathrm{data}} - N_{\mathrm{rand}}}{N_{\mathrm{data}} + N_{\mathrm{rand}}}$")
    ax.set_title('Dispersion of CDF across zones', fontsize=16)
    ax.legend(fontsize=9, ncol=2)

    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, 'cdf/cdf_dispersion_zones.png')
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=PLOT_DPI)
    plt.close(fig)


def entropy(df):
    """
    Compute the Shannon entropy for each row in the DataFrame.
    
    Args:
        df (pd.DataFrame): The input DataFrame containing probability columns.
    Returns:
        pd.DataFrame: The input DataFrame with an additional column for entropy.
    """
    cols = ['PVOID', 'PSHEET', 'PFILAMENT', 'PKNOT']
    P = np.column_stack([np.asarray(df[c], dtype=float) for c in cols])
    out = df.copy()

    # H = - sum p_i log2 p_i / log2(4)
    with np.errstate(divide='ignore', invalid='ignore'):
        terms = P * np.log2(P)
    terms[~np.isfinite(terms)] = 0.0
    H = -np.sum(terms, axis=1) / np.log2(4.0)

    out['H'] = H.astype(np.float32)
    return out


def _read_raw_df_min(raw_path):
    """
    Read the raw FITS file and return a minimal DataFrame.
    
    Args:
        raw_path (str): The path to the raw FITS file.
    Returns:
        pd.DataFrame: A minimal DataFrame containing relevant columns.
    """
    tbl = Table.read(raw_path, memmap=True)
    df = tbl.to_pandas()
    df['TRACERTYPE'] = df['TRACERTYPE'].apply(lambda x: x.decode('utf-8')
                                              if isinstance(x, (bytes, bytearray)) else x)
    df['BASE'] = df['TRACERTYPE'].str.replace(r'_(DATA|RAND)$','', regex=True)
    df['ISDATA'] = df['TRACERTYPE'].str.endswith('_DATA')
    return df[['TARGETID','TRACERTYPE','BASE','ISDATA']]


def _targets_of_tracer_real(raw_df, tracer_prefix):
    """
    Get the set of TARGETIDs for a specific tracer prefix from the raw dataframe.
    
    Args:
        raw_df (pd.DataFrame): The raw dataframe containing tracer information.
        tracer_prefix (str): The tracer prefix to filter by.
    Returns:
        set: A set of TARGETIDs matching the tracer prefix.
    """
    if tracer_prefix == 'BGS_BRIGHT':
        tracer_prefix = 'BGS'
    m = raw_df['ISDATA'] & raw_df['BASE'].str.startswith(tracer_prefix)
    return set(raw_df.loc[m, 'TARGETID'].to_numpy(dtype=np.int64))


def plot_pdf_entropy(raw_dir, class_dir, zones, tracers, out_path, bins=25, cache_dir=None, tags_map=None):
    """
    Plot the PDF of the normalized Shannon entropy H for specified tracers across zones.
    
    Args:
        raw_dir (str): Directory containing raw data files.
        class_dir (str): Directory containing class data files.
        zones (list): List of zone identifiers.
        tracers (list): List of tracer identifiers.
        out_path (str): Output path for the plot.
        bins (int): Number of bins for the histogram.
        cache_dir (str or None): Directory used to reuse intermediate results.
        tags_map (dict|None): Optional mapping of zone -> tag list.
    """
    colors = plt.cm.tab20(np.linspace(0, 1, max(20, len(zones))))
    fig, axes = plt.subplots(2, 2, figsize=(10,10), sharey=True, sharex=True)
    axes = np.ravel(axes)

    for ax, tracer in zip(axes, tracers):
        ax.grid(True, alpha=0.3)
        ax.set_title(tracer.replace('_ANY', ''))

        for iz, z in enumerate(zones):
            if tags_map and z in tags_map:
                tag_list = tags_map[z]
            else:
                tags = list_zone_out_tags(raw_dir, class_dir, z)
                tag_list = tags if tags else ['']
            ztag = zone_tag(z)
            raw_paths = [_raw_candidate(raw_dir, z, t or None) for t in tag_list]
            prob_paths = [_expected_prob_path(class_dir, z, t or None) for t in tag_list]

            raw_df = _load_or_build_df(cache_dir, _zone_cache_key('raw_min', z, tag_list), raw_paths,
                                       lambda: _concat_existing(raw_paths, _read_raw_df_min))

            def _load_prob_entropy():
                probs = _concat_existing(prob_paths, load_prob_df)
                return entropy(probs)

            probs_df = _load_or_build_df(cache_dir, _zone_cache_key('prob_entropy', z, tag_list), prob_paths,
                                         _load_prob_entropy)

            tids_tr = _targets_of_tracer_real(raw_df, tracer.split('_', 1)[0])
            tids = probs_df['TARGETID'].to_numpy(dtype=np.int64, copy=False)
            m = np.isin(tids, list(tids_tr))
            v = probs_df.loc[m, 'H'].to_numpy(dtype=float, copy=False)

            hist, edges = np.histogram(v, bins=bins, range=(0, 0.6), density=True)
            centers = 0.5 * (edges[:-1] + edges[1:])
            label = f'Zone {zone_tag(z)}' if ax is axes[0] else None
            ax.plot(centers, hist, color=colors[iz], label=label)

        if ax in (axes[0], axes[2]):
            ax.set_ylabel('PDF')
        if ax in (axes[2], axes[3]):
            ax.set_xlabel('$H$')

    handles, labels = [], []
    for ax in axes:
        h, l = ax.get_legend_handles_labels()
        for hh, ll in zip(h, l):
            if ll not in labels and ll is not None:
                handles.append(hh); labels.append(ll)

    if handles:
        fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.01),
                   bbox_transform=fig.transFigure, ncol=min(7, len(labels)), frameon=False)
    fig.subplots_adjust(bottom=0.14, top=0.85)
    plt.suptitle('Normalized Shannon Entropy', y=0.94)

    path = f'{out_path}/entropy'
    os.makedirs(path, exist_ok=True)
    fig.savefig(f'{path}/pdf_entropy.png', dpi=PLOT_DPI)
    return fig, axes


def _process_zone(zone, config):
    """
    Process a single zone: load data, compute r, and generate requested plots.
    
    Args:
        zone (int): Zone identifier.
        config (dict): Configuration dictionary with keys:
    Returns:     
        dict: Summary of processing results.
    """
    tag_list = config['tags_map'][zone]
    raw_paths = [_raw_candidate(config['raw_dir'], zone, t or None) for t in tag_list]
    cls_paths = [_expected_class_path(config['class_dir'], zone, t or None) for t in tag_list]
    progress = config['progress']
    cache_dir = config['cache_dir']

    raw_df = _load_or_build_df(cache_dir, _zone_cache_key('raw', zone, tag_list), raw_paths,
                               lambda: _concat_existing(raw_paths, load_raw_df), progress=progress)
    if progress:
        print(f'Zone {zone}: {len(raw_df)} total objects in raw data')

    cls_full = _load_or_build_df(cache_dir, _zone_cache_key('class', zone, tag_list), cls_paths,
                                 lambda: _concat_existing(cls_paths, load_class_df), progress=progress)
    if progress:
        print(f'Zone {zone}: {len(cls_full)} total objects in class data')
    cls_df = cls_full[cls_full['ISDATA'] == True]
    if progress:
        print(f'Zone {zone}: {len(cls_df)} total objects in class data (ISDATA=True)')

    def _build_r():
        merged = raw_df.merge(cls_df[['TARGETID','NDATA','NRAND']], on='TARGETID', how='left')
        return compute_r(merged)

    r_df = _load_or_build_df(cache_dir, _zone_cache_key('r', zone, tag_list), raw_paths + cls_paths,
                             _build_r, progress=progress)
    if progress:
        print(f'Zone {zone}: {len(r_df)} total objects after merging raw and class data')

    if config['plot_z']:
        plot_z_histogram(raw_df, zone, config['bins'], config['outdirs']['z'])
        if progress:
            print(f'Zone {zone}: plotted z histogram')
    if config['plot_cdf']:
        plot_cdf(r_df, zone, config['tracers'], config['outdirs']['cdf'])
        if progress:
            print(f'Zone {zone}: plotted CDF')
    if config['plot_radial']:
        plot_radial_distribution(raw_df, zone, config['tracers'], config['outdirs']['radial'], config['bins'])
        if progress:
            print(f'Zone {zone}: plotted radial distribution')

    return {'zone': zone, 'raw_rows': len(raw_df), 'class_rows': len(cls_df), 'r_rows': len(r_df),
            'tags': tag_list}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--raw-dir', default='/pscratch/sd/v/vtorresg/cosmic-web/dr1/raw')
    p.add_argument('--class-dir', default='/pscratch/sd/v/vtorresg/cosmic-web/dr1')
    p.add_argument('--output', default='/pscratch/sd/v/vtorresg/cosmic-web/dr1/figs')

    p.add_argument('--zones', nargs='+', default=None)
    p.add_argument('--bins', type=int, default=10)
    p.add_argument('--tracers', nargs='+', default=['BGS_BRIGHT','LRG','ELG','QSO'])

    p.add_argument('--plot-z', action='store_true', default=False)
    p.add_argument('--plot-radial', action='store_true', default=False)
    p.add_argument('--plot-cdf', action='store_true', default=False)
    p.add_argument('--plot-cdf-dispersion', action='store_true', default=False)
    p.add_argument('--plot-entropy-cdf', action='store_true', default=False)
    p.add_argument('--all', action='store_true', help='Enable all plots (heavy)')
    p.add_argument('--usetex', action='store_true', help='Use LaTeX text rendering (heavy)')
    p.add_argument('--dpi', type=int, default=50, help='Figure DPI')
    p.add_argument('--xbins', type=int, default=200,
                   help='Number of x grid points for CDF interpolation (default: 200)')
    p.add_argument('--subsample-per-zone', type=int, default=10000,
                   help='Max samples per (zone,tracer,real/rand) for dispersion plot')
    p.add_argument('--progress', action='store_true', default=False, help='Print simple progress logs')
    p.add_argument('--out-tags', nargs='*', default=None,
                   help='Limit to these out-tags (e.g., LRG ELG). If omitted, auto-discovers tags per zone.')

    default_workers = os.cpu_count()
    p.add_argument('--workers', type=int, default=default_workers,
                   help='Number of worker processes for per-zone plots (default: CPU count - 1)')
    p.add_argument('--cache-dir', default=None,
                   help='Directory for cached intermediate data (defaults to <output>/_cache)')
    p.add_argument('--no-cache', action='store_true', help='Disable on-disk caching of intermediate data')

    return p.parse_args()


def main():
    args = parse_args()
    global PLOT_DPI
    PLOT_DPI = args.dpi

    args.class_dir = normalize_release_dir(args.class_dir)

    if args.usetex:
        matplotlib.rcParams['text.usetex'] = True
    if args.all:
        args.plot_z = True
        args.plot_radial = True
        args.plot_cdf = True
        args.plot_cdf_dispersion = True
        args.plot_entropy_cdf = True

    zones = infer_zones(args.raw_dir, args.zones)
    outdirs = make_output_dirs(args.output)

    cache_dir = None if args.no_cache else (args.cache_dir or os.path.join(args.output, '_cache'))
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)

    tags_map = {}
    for zone in zones:
        if args.out_tags:
            tags = list(args.out_tags)
        else:
            tags = list_zone_out_tags(args.raw_dir, args.class_dir, zone)
        tag_list = tags if tags else ['']
        tags_map[zone] = tag_list
        if args.progress:
            non_empty = [t for t in tag_list if t]
            pretty = ','.join(non_empty) if non_empty else '(legacy single-file)'
            print(f'[plot] zone {zone}: tags={pretty}')

    zone_config = {'raw_dir': args.raw_dir,
                   'class_dir': args.class_dir,
                   'outdirs': outdirs,
                   'bins': args.bins,
                   'tracers': args.tracers,
                   'plot_z': args.plot_z,
                   'plot_radial': args.plot_radial,
                   'plot_cdf': args.plot_cdf,
                   'progress': args.progress,
                   'cache_dir': cache_dir,
                   'tags_map': tags_map}

    has_zone_work = any((args.plot_z, args.plot_radial, args.plot_cdf))
    zone_results = []
    workers = max(1, args.workers)
    if zones:
        workers = min(workers, len(zones))

    if has_zone_work and zones:
        if workers == 1:
            for zone in zones:
                zone_results.append(_process_zone(zone, zone_config))
        else:
            with ProcessPoolExecutor(max_workers=workers) as pool:
                futures = {pool.submit(_process_zone, zone, zone_config): zone for zone in zones}
                for future in as_completed(futures):
                    zone_results.append(future.result())
    elif not zones:
        print('No zones found to process.')

    if zone_results and not args.progress:
        print(f'Generated per-zone plots for {len(zone_results)} zones using {workers} worker(s).')

    if args.plot_cdf_dispersion:
        plot_cdf_dispersion(args.raw_dir, args.class_dir, zones, args.output, args.tracers,
                            xbins=args.xbins, subsample_per_zone=args.subsample_per_zone,
                            progress=args.progress, out_tags=args.out_tags,
                            cache_dir=cache_dir, tags_map=tags_map)
        print(f'Plotted CDF dispersion for zones: {list(zones)}')
    if args.plot_entropy_cdf:
        plot_pdf_entropy(args.raw_dir, args.class_dir, zones, args.tracers, args.output, args.bins,
                         cache_dir=cache_dir, tags_map=tags_map)
        print(f'Plotted PDF of entropy for zones: {list(zones)}')


if __name__ == '__main__':
    main()