import argparse, os, re
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import pandas as pd
from astropy.table import Table

import matplotlib
import matplotlib.pyplot as plt

import sys
from pathlib import Path

matplotlib.rcParams['text.usetex'] = True

if __package__ is None or __package__ == '':
    src_root = Path(__file__).resolve().parents[1]
    if str(src_root) not in sys.path:
        sys.path.append(str(src_root))
    from desiproc.paths import (classification_path, locate_classification_file, locate_probability_file,
                                normalize_release_dir, probability_path, safe_tag, zone_tag)
    from plot.common import (load_probability_dataframe, load_raw_dataframe, resolve_class_path,
                             resolve_probability_path, resolve_raw_path)
else:
    from desiproc.paths import (classification_path, locate_classification_file, locate_probability_file,
                                normalize_release_dir, probability_path, safe_tag, zone_tag)
    from .common import (load_probability_dataframe, load_raw_dataframe, resolve_class_path,
                         resolve_probability_path, resolve_raw_path)

PLOT_DPI = 360
CACHE_VERSION = 'v2'

_PANDAS_NA = getattr(pd, 'NA', None)
_NP_BYTES_TYPE = getattr(np, 'bytes_', None)
_BYTES_TYPES = (bytes, bytearray)
if _NP_BYTES_TYPE is not None and _NP_BYTES_TYPE not in _BYTES_TYPES:
    _BYTES_TYPES = _BYTES_TYPES + (_NP_BYTES_TYPE,)

_TRACER_PREFIX_ALIASES = {'BGS': ('BGS',),
                          'BGS_BRIGHT': ('BGS',),
                          'BGS_ANY': ('BGS',),
                          'ELG': ('ELG', 'ELG_LOPnotqso'),
                          'ELG_LOPNOTQSO': ('ELG_LOPnotqso',),
                          'LRG': ('LRG',),
                          'QSO': ('QSO',)}

_TRACER_COLORS = {'BGS': 'blue',
                  'BGS_BRIGHT': 'blue',
                  'BGS_ANY': 'blue',
                  'ELG': 'red',
                  'ELG_LOPNOTQSO': 'red',
                  'LRG': 'green',
                  'QSO': 'orange'}


def _tracer_key(tracer):
    """
    Return a normalized key for *tracer* (uppercase string or empty string if None).
    
    Args:
        tracer (str | None): Tracer label.
    Returns:
        str: Uppercase tracer label or empty string if None.
    """
    return str(tracer).upper() if tracer is not None else ''


def _tracer_prefixes(tracer):
    """
    Return a tuple of catalogue prefixes for *tracer*.

    Args:
        tracer (str | None): Tracer label.
    Returns:
        tuple[str]: Tuple of catalogue prefixes.
    """
    key = _tracer_key(tracer)
    prefixes = _TRACER_PREFIX_ALIASES.get(key)
    if prefixes:
        return prefixes
    value = str(tracer)
    if key != value.upper():
        return (value, value.upper())
    return (value,)


def _tracer_display_name(tracer):
    """
    Return a display name for *tracer*.
    
    Args:
        tracer (str | None): Tracer label.
    """
    key = _tracer_key(tracer)
    if key in ('BGS_BRIGHT', 'BGS_ANY'):
        return 'BGS'
    if key == 'ELG_LOPNOTQSO':
        return 'ELG'
    return str(tracer)


def _tracer_color(tracer):
    """
    Return a color for *tracer*.
    
    Args:
        tracer (str | None): Tracer label.
    Returns:
        str: Color name or hex code.
    """
    key = _tracer_key(tracer)
    color = _TRACER_COLORS.get(key)
    if color:
        return color
    display_key = _tracer_display_name(tracer).upper()
    return _TRACER_COLORS.get(display_key, 'black')


def _is_text_like(value):
    """
    Return True if *value* is None, pd.NA, NaN, or a string.
    
    Args:
        value: Any value.
    Returns:
        bool: True if *value* is considered text-like.
    """
    if value is None:
        return True
    if _PANDAS_NA is not None and value is _PANDAS_NA:
        return True
    if isinstance(value, str):
        return True
    try:
        if pd.isna(value):
            return True
    except Exception:
        pass
    return False


def _coerce_text(value):
    """
    Coerce *value* to a string, handling bytes and None/pd.NA.
    
    Args:
        value: Any value.
    Returns:
        str | None: String representation of *value*, or None/pd.NA if applicable.
    """
    if value is None:
        return None
    if _PANDAS_NA is not None and value is _PANDAS_NA:
        return value
    if isinstance(value, str):
        return value
    if isinstance(value, _BYTES_TYPES):
        try:
            return value.decode('utf-8')
        except Exception:
            try:
                return value.decode('utf-8', errors='ignore')
            except Exception:
                return str(value)
    return str(value)


def _ensure_text_column(df, column):
    """
    Ensure that the specified *column* in *df* contains only text-like values.
    
    Args:
        df (pd.DataFrame): The DataFrame to modify.
        column (str): The column name to ensure text-like values.
    Returns:
        pd.DataFrame: The modified DataFrame with text-like values in *column*.
    """
    if column not in df.columns or df.empty:
        return df
    col = df[column]
    mask = ~col.map(_is_text_like)
    if mask.any():
        df.loc[mask, column] = col.loc[mask].map(_coerce_text)
    return df


def _sanitize_tracer_columns(df):
    """
    Ensure that tracer-related columns in *df* contain only text-like values.
    
    Args:
        df (pd.DataFrame | None): The DataFrame to sanitize, or None.
    Returns:
        pd.DataFrame | None: The sanitized DataFrame, or None if input was None.
    """
    if df is None:
        return df
    for column in ('TRACERTYPE', 'BASE', 'BASE_CORE'):
        _ensure_text_column(df, column)
    return df


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
    df = load_raw_dataframe(path)
    return _sanitize_tracer_columns(df)


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
    zone_label = zone_tag(zone)

    fig, ax = plt.subplots()
    ax.grid(linewidth=0.7, zorder=-10)
    ax.hist([real, rand], bins=bins, label=['Data','Random'], alpha=0.9,
            color=['royalblue', 'navy'], zorder=10, edgecolor='black', lw=0.3)
    ax.set(xlabel='Z', ylabel='Count', title=f'Zone {zone_label}')
    ax.legend()

    path = os.path.join(out_dir, f'z_hist_zone_{zone_label}.png')
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
    zone_label = zone_tag(zone)
    fig, axes = plt.subplots(1, len(tracers), figsize=(4*len(tracers),4))
    if len(tracers) == 1:
        axes = [axes]

    for ax, tracer in zip(axes, tracers):
        prefixes = _tracer_prefixes(tracer)
        display = _tracer_display_name(tracer)
        sub = raw_df[raw_df['TRACERTYPE'].str.startswith(prefixes)]
        real = sub[sub['RANDITER'] == -1]
        rand_pool = sub[sub['RANDITER'] != -1]
        if len(rand_pool) == 0 or len(real) == 0:
            ax.set(title=display, xlabel=r'$r$', ylabel='Count')
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            ax.grid(linewidth=0.7, zorder=-10)
            continue
        nsamp = min(len(rand_pool), len(real))
        rand = rand_pool.sample(nsamp, random_state=0, replace=False)

        r_real = np.linalg.norm(real[['XCART','YCART','ZCART']].to_numpy(), axis=1)
        r_rand = np.linalg.norm(rand[['XCART','YCART','ZCART']].to_numpy(), axis=1)

        ax.grid(linewidth=0.7, zorder=-10)
        ax.hist([r_real, r_rand], bins, label=['Real','Random'], alpha=0.9,
                color=['royalblue', 'navy'], zorder=10, edgecolor='black', lw=0.05)
        ax.set(title=display, xlabel=r'$r$', ylabel='Count')
        ax.legend(fontsize=6)

    fig.suptitle(f'Radial Zone {zone_label}')
    fig.tight_layout()
    path = os.path.join(out_dir, f'radial_zone_{zone_label}.png')
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
    fig, ax = plt.subplots()
    ax.grid(linewidth=0.7, zorder=-10)

    def ecdf(arr):
        a = np.asarray(arr, dtype=float)
        a = a[~np.isnan(a)]
        if a.size == 0:
            return None, None
        a = np.sort(a)
        y = np.arange(1, a.size + 1) / a.size
        return a, y

    for tr in tracers:
        prefixes = _tracer_prefixes(tr)
        display = _tracer_display_name(tr)
        color = _tracer_color(tr)
        sub = df_r[df_r['TRACERTYPE'].str.startswith(prefixes)]
        xr, yr = ecdf(sub[sub['ISDATA']]['r'])
        if xr is not None:
            ax.plot(xr, yr, label=f'{display} real', linewidth=1, color=color, zorder=10)
        xr, yr = ecdf(sub[~sub['ISDATA']]['r'])
        if xr is not None:
            ax.plot(xr, yr, label=f'{display} rand', linewidth=1, linestyle='--', color=color, zorder=10)

    zone_label = zone_tag(zone)
    ax.set(xlabel='$r$', ylabel='CDF', title=f'CDF Zone {zone_label}')
    ax.legend(fontsize=6)
    path = os.path.join(out_dir, f'cdf_zone_{zone_label}.png')
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
        _sanitize_tracer_columns(raw_df)
        cls_df = _load_or_build_df(cache_dir, _zone_cache_key('class', z, tag_list), cls_paths,
                                   lambda: _concat_existing(cls_paths, load_class_df), progress=progress)
        cls_df = cls_df[cls_df['ISDATA'] == True]

        def _build_r():
            merged = raw_df.merge(cls_df[['TARGETID','NDATA','NRAND']], on='TARGETID', how='left')
            return compute_r(merged)

        r_df = _load_or_build_df(cache_dir, _zone_cache_key('r', z, tag_list), raw_paths + cls_paths,
                                 _build_r, progress=progress)
        _sanitize_tracer_columns(r_df)

        for tr in tracers:
            prefixes = _tracer_prefixes(tr)
            sub = r_df[r_df['TRACERTYPE'].str.startswith(prefixes)]
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

    fig, ax = plt.subplots()
    ax.grid(linewidth=0.7, zorder=-10)

    for tr in tracers:
        color = _tracer_color(tr)
        display = _tracer_display_name(tr)

        if len(per_tracer_real[tr]) > 0:
            Y = np.vstack(per_tracer_real[tr])
            p16, p50, p84 = np.nanpercentile(Y, [16,50,84], axis=0)
            ax.fill_between(xgrid, p16, p84, alpha=0.25,
                            edgecolor=color, linewidth=0.6,
                            label=f'{display} real '+r'$\pm 1\sigma$', color=color)
            ax.plot(xgrid, p50, linewidth=1.5, color=color, label=f'{display} real median', zorder=10)
            ax.plot(xgrid, p16, linewidth=0.7, color=color, linestyle=':', zorder=8)
            ax.plot(xgrid, p84, linewidth=0.7, color=color, linestyle=':', zorder=8)

        if len(per_tracer_rand[tr]) > 0:
            Y = np.vstack(per_tracer_rand[tr])
            p16, p50, p84 = np.nanpercentile(Y, [16,50,84], axis=0)
            ax.fill_between(xgrid, p16, p84, alpha=0.18,
                            edgecolor=color, linewidth=0.5, linestyle='--',
                            label=f'{display} rand '+r'$\pm 1\sigma$', color=color)
            ax.plot(xgrid, p50, linewidth=1.2, linestyle='--', color=color, label=f'{display} rand median', zorder=10)
            ax.plot(xgrid, p16, linewidth=0.6, color=color, linestyle='--', dashes=(4,2), zorder=8)
            ax.plot(xgrid, p84, linewidth=0.6, color=color, linestyle='--', dashes=(4,2), zorder=8)

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
    df = _sanitize_tracer_columns(df)
    return df[['TARGETID','TRACERTYPE','BASE','ISDATA']]


def _targets_of_tracer_real(raw_df, tracer):
    """
    Get the set of TARGETIDs for a specific tracer prefix from the raw dataframe.
    
    Args:
        raw_df (pd.DataFrame): The raw dataframe containing tracer information.
        tracer (str): Tracer label used to determine matching prefixes.
    Returns:
        set: A set of TARGETIDs matching the tracer's catalogue prefixes.
    """
    prefixes = _tracer_prefixes(tracer)
    m = raw_df['ISDATA'] & raw_df['BASE'].str.startswith(prefixes)
    return set(raw_df.loc[m, 'TARGETID'].to_numpy(dtype=np.int64))


def plot_pdf_entropy(raw_dir, class_dir, zones, tracers, out_path, bins=25, cache_dir=None, tags_map=None):
    """
    Plot the PDF of the normalized Shannon entropy H for each tracer, summarizing
    the dispersion across the provided zones.

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

    bin_edges = np.linspace(0, 0.6, bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    per_tracer_hist = {tr: [] for tr in tracers}

    for z in zones:
        if tags_map and z in tags_map:
            tag_list = tags_map[z]
        else:
            tags = list_zone_out_tags(raw_dir, class_dir, z)
            tag_list = tags if tags else ['']

        raw_paths = [_raw_candidate(raw_dir, z, t or None) for t in tag_list]
        prob_paths = [_expected_prob_path(class_dir, z, t or None) for t in tag_list]

        raw_df = _load_or_build_df(cache_dir, _zone_cache_key('raw_min', z, tag_list), raw_paths,
                                   lambda: _concat_existing(raw_paths, _read_raw_df_min))
        _sanitize_tracer_columns(raw_df)

        def _load_prob_entropy():
            probs = _concat_existing(prob_paths, load_prob_df)
            return entropy(probs)

        probs_df = _load_or_build_df(cache_dir, _zone_cache_key('prob_entropy', z, tag_list), prob_paths,
                                     _load_prob_entropy)

        tids = probs_df['TARGETID'].to_numpy(dtype=np.int64, copy=False)

        for tracer in tracers:
            tids_tr = _targets_of_tracer_real(raw_df, tracer)
            if not tids_tr:
                continue

            mask = np.isin(tids, list(tids_tr))
            values = probs_df.loc[mask, 'H'].to_numpy(dtype=float, copy=False)
            if values.size == 0:
                continue

            hist, _ = np.histogram(values, bins=bin_edges, density=True)
            per_tracer_hist[tracer].append(hist)

    fig, ax = plt.subplots()
    ax.grid(linewidth=0.7, zorder=-10)

    for tracer in tracers:
        series = per_tracer_hist.get(tracer, [])
        if not series:
            continue

        color = _tracer_color(tracer)
        display = _tracer_display_name(tracer)
        Y = np.vstack(series)
        p16, p50, p84 = np.nanpercentile(Y, [16, 50, 84], axis=0)

        ax.fill_between(bin_centers, p16, p84, alpha=0.25,
                        edgecolor=color, linewidth=0.6,
                        color=color, label=f'{display} '+r'$\pm 1\sigma$')
        ax.plot(bin_centers, p50, linewidth=1.5, color=color, label=f'{display} median', zorder=10)
        ax.plot(bin_centers, p16, linewidth=0.7, color=color, linestyle=':', zorder=8)
        ax.plot(bin_centers, p84, linewidth=0.7, color=color, linestyle=':', zorder=8)

    ax.set_xlabel('$H$')
    ax.set_ylabel('PDF')
    ax.set_title('Normalized Shannon Entropy', fontsize=16)
    ax.legend(fontsize=9, ncol=2)

    path = f'{out_path}/entropy'
    os.makedirs(path, exist_ok=True)
    fig.savefig(f'{path}/pdf_entropy.png', dpi=PLOT_DPI)
    return fig, ax


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
    _sanitize_tracer_columns(raw_df)
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
    _sanitize_tracer_columns(r_df)
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
    p.add_argument('--dpi', type=int, default=360, help='Figure DPI')
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