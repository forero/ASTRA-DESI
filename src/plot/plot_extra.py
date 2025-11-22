import argparse, os, re
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import pandas as pd
from astropy.table import Table
from astropy.io import fits

import matplotlib
import matplotlib.pyplot as plt

import sys
from pathlib import Path

# matplotlib.rcParams['text.usetex'] = True

if __package__ is None or __package__ == '':
    src_root = Path(__file__).resolve().parents[1]
    if str(src_root) not in sys.path:
        sys.path.append(str(src_root))
    from desiproc.paths import (classification_path, locate_classification_file, locate_probability_file,
                                normalize_release_dir, probability_path, safe_tag, zone_tag)
    from plot.common import (load_probability_dataframe, load_raw_dataframe, resolve_class_path,
                             resolve_probability_path, resolve_raw_path)
    from plot.color_theme import load_theme, apply_matplotlib_theme
else:
    from desiproc.paths import (classification_path, locate_classification_file, locate_probability_file,
                                normalize_release_dir, probability_path, safe_tag, zone_tag)
    from .common import (load_probability_dataframe, load_raw_dataframe, resolve_class_path,
                         resolve_probability_path, resolve_raw_path)
    from .color_theme import load_theme, apply_matplotlib_theme

PLOT_DPI = 50
CACHE_VERSION = 'v5'
try:
    _DEFAULT_RAW_LIMIT = int(os.environ.get('PLOT_EXTRA_RAW_LIMIT', '3000000'))
except ValueError:
    _DEFAULT_RAW_LIMIT = 3000000
if _DEFAULT_RAW_LIMIT <= 0:
    _DEFAULT_RAW_LIMIT = None
RAW_LOAD_OPTIONS = {'columns': None,
                    'downcast': True,
                    'limit': _DEFAULT_RAW_LIMIT,
                    'randomize': False,
                    'seed': 0}
CLASS_LOAD_OPTIONS = {'chunk_rows': 500_000}
CACHE_CONTEXT_SUFFIX = ''

_PANDAS_NA = getattr(pd, 'NA', None)
_NP_BYTES_TYPE = getattr(np, 'bytes_', None)
_BYTES_TYPES = (bytes, bytearray)
if _NP_BYTES_TYPE is not None and _NP_BYTES_TYPE not in _BYTES_TYPES:
    _BYTES_TYPES = _BYTES_TYPES + (_NP_BYTES_TYPE,)

THEME_NAME, THEME = load_theme('PLOT_EXTRA_THEME', default='dark')
apply_matplotlib_theme(THEME)

TEXT_COLOR = THEME['text']
PRIMARY_COLOR = THEME['primary']
SECONDARY_COLOR = THEME['secondary']
SCATTER_EDGE_COLOR = THEME['scatter_edge']
HIGHLIGHT_EDGE_COLOR = THEME['highlight_edge']
GRID_COLOR = SECONDARY_COLOR

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
    return _TRACER_COLORS.get(display_key, PRIMARY_COLOR)


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
    mask_values = np.fromiter((not _is_text_like(v) for v in col), dtype=bool, count=len(col))
    if not mask_values.any():
        return df
    if pd.api.types.is_categorical_dtype(col.dtype):
        df[column] = col.astype('string')
        col = df[column]
    mask = pd.Series(mask_values, index=df.index, dtype=bool)
    sanitized = [_coerce_text(v) for v in col.loc[mask]]
    df.loc[mask, column] = sanitized
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
            pd.to_pickle(df, tmp_path)
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
    context = CACHE_CONTEXT_SUFFIX
    if context:
        return f'{kind}_zone_{zone_tag(zone)}{tag_suffix}_{context}'
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


def _normalize_discovered_tag(tag):
    """
    Normalize a tag discovered from filenames, collapsing legacy markers.

    Args:
        tag (str | None): Raw tag extracted from a filename.
    Returns:
        str: Normalized tag ('' represents the combined catalogue).
    """
    if tag is None:
        return ''
    value = str(tag).strip('_')
    if value.lower() == 'combined':
        return ''
    return value


def discover_zone_entries(raw_dir, class_dir, zone):
    """
    Inspect available files for a zone and tracer combination.

    Args:
        raw_dir (str): Directory containing raw zone files.
        class_dir (str): Release root directory with classification/probability subdirs.
        zone (int | str): Zone identifier (e.g., 0, 'NGC').
    Returns:
        list[dict]: Ordered list of per-tag dictionaries with availability flags.
    """
    ztag = zone_tag(zone)
    entries = {}
    raw_by_tag = {}

    def _entry_for(tag_value):
        key = _normalize_discovered_tag(tag_value)
        if key not in entries:
            entries[key] = {'tag': key,
                            'raw': None,
                            'has_raw': False,
                            'class': None,
                            'class_iter': [],
                            'has_class': False,
                            'prob': None,
                            'prob_iter': [],
                            'prob_data': None,
                            'has_prob': False,
                            'has_prob_data': False}
        return entries[key]

    def _ensure_raw(entry, base_tag):
        if base_tag is None:
            key = ''
        else:
            key = _normalize_discovered_tag(base_tag)
        if entry.get('has_raw'):
            return
        candidate = None
        if key in raw_by_tag:
            candidate = raw_by_tag[key]
        else:
            try:
                candidate = _raw_candidate(raw_dir, zone, base_tag)
            except FileNotFoundError:
                candidate = None
            if candidate and os.path.exists(candidate):
                raw_by_tag[key] = candidate
        if candidate and os.path.exists(candidate):
            entry['raw'] = candidate
            entry['has_raw'] = True

    if raw_dir and os.path.isdir(raw_dir):
        raw_pattern = re.compile(rf'^zone_{re.escape(ztag)}(?:_(?P<tag>.+))?\.fits(?:\.gz)?$', re.IGNORECASE)
        for fname in os.listdir(raw_dir):
            match = raw_pattern.match(fname)
            if not match:
                continue
            entry = _entry_for(match.group('tag'))
            path = os.path.join(raw_dir, fname)
            entry['raw'] = path
            entry['has_raw'] = True
            raw_by_tag[entry['tag']] = path

    class_root = os.path.join(class_dir, 'classification') if class_dir else None
    if class_root and os.path.isdir(class_root):
        class_pattern = re.compile(rf'^zone_{re.escape(ztag)}(?:_(?P<tag>.+))?_classified\.fits(?:\.gz)?$', re.IGNORECASE)
        iter_pattern = re.compile(rf'^zone_{re.escape(ztag)}_(?P<tracer>[^_]+)_iter(?P<iter>\d+)\.fits(?:\.gz)?$', re.IGNORECASE)
        for root, _, files in os.walk(class_root):
            for fname in files:
                path = os.path.join(root, fname)
                match = class_pattern.match(fname)
                if match:
                    entry = _entry_for(match.group('tag'))
                    entry['class'] = path
                    entry['has_class'] = True
                    continue
                iter_match = iter_pattern.match(fname)
                if iter_match:
                    tracer_tag_value = iter_match.group('tracer')
                    entry = _entry_for(tracer_tag_value)
                    entry['class_iter'].append(path)
                    entry['has_class'] = True
                    _ensure_raw(entry, tracer_tag_value)

    prob_root = os.path.join(class_dir, 'probabilities') if class_dir else None
    if prob_root and os.path.isdir(prob_root):
        prob_pattern = re.compile(rf'^zone_{re.escape(ztag)}(?:_(?P<tag>.+))?_probability\.fits(?:\.gz)?$', re.IGNORECASE)
        iter_pattern = re.compile(rf'^zone_{re.escape(ztag)}_(?P<tracer>[^_]+)_probability_iter(?P<iter>\d+)\.fits(?:\.gz)?$', re.IGNORECASE)
        iterdata_pattern = re.compile(rf'^zone_{re.escape(ztag)}_(?P<tracer>[^_]+)_probability_iterdata\.fits(?:\.gz)?$', re.IGNORECASE)
        for root, _, files in os.walk(prob_root):
            for fname in files:
                path = os.path.join(root, fname)
                match = prob_pattern.match(fname)
                if match:
                    entry = _entry_for(match.group('tag'))
                    entry['prob'] = path
                    entry['has_prob'] = True
                    continue
                iterdata_match = iterdata_pattern.match(fname)
                if iterdata_match:
                    tracer_tag_value = iterdata_match.group('tracer')
                    entry = _entry_for(tracer_tag_value)
                    entry['prob_data'] = path
                    entry['has_prob'] = True
                    entry['has_prob_data'] = True
                    _ensure_raw(entry, tracer_tag_value)
                    continue
                iter_match = iter_pattern.match(fname)
                if iter_match:
                    tracer_tag_value = iter_match.group('tracer')
                    entry = _entry_for(tracer_tag_value)
                    entry['prob_iter'].append(path)
                    entry['has_prob'] = True
                    _ensure_raw(entry, tracer_tag_value)

    for entry in entries.values():
        if entry['class_iter']:
            entry['class_iter'].sort()
        if entry['prob_iter']:
            entry['prob_iter'].sort()

    if not entries:
        entry = _entry_for(None)
        raw_path = _raw_candidate(raw_dir, zone, None)
        entry['raw'] = raw_path
        entry['has_raw'] = os.path.exists(raw_path)
        class_path = _expected_class_path(class_dir, zone, None)
        if os.path.exists(class_path):
            entry['class'] = class_path
            entry['has_class'] = True
        prob_path = _expected_prob_path(class_dir, zone, None)
        if os.path.exists(prob_path):
            entry['prob'] = prob_path
            entry['has_prob'] = True

    def _order_key(tag_value):
        return (tag_value != '', str(tag_value).upper())

    ordered_tags = sorted(entries.keys(), key=_order_key)
    return [entries[tag] for tag in ordered_tags]


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
    entries = discover_zone_entries(raw_dir, class_dir, zone)
    return [entry['tag'] for entry in entries if entry.get('has_raw')]


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
    df = load_raw_dataframe(path,
                            columns=RAW_LOAD_OPTIONS.get('columns'),
                            downcast=RAW_LOAD_OPTIONS.get('downcast', True),
                            row_limit=RAW_LOAD_OPTIONS.get('limit'),
                            randomize=RAW_LOAD_OPTIONS.get('randomize', False),
                            seed=RAW_LOAD_OPTIONS.get('seed'))
    return _sanitize_tracer_columns(df)


def load_class_df(path, target_ids=None, chunk_rows=None, downcast=True):
    """
    Load classification data from FITS file into a pandas DataFrame.

    Args:
        path (str): Path to the FITS file.
        target_ids (Collection[int] | None): Optional set/list of TARGETIDs to retain.
        chunk_rows (int | None): Number of rows per chunk when streaming the FITS file.
        downcast (bool): Whether to downcast numeric columns to reduce memory.
    Returns:
        pd.DataFrame: DataFrame containing the classification data.
    """
    include = ['TARGETID', 'NDATA', 'NRAND', 'ISDATA']
    chunk_size = chunk_rows or CLASS_LOAD_OPTIONS.get('chunk_rows') or 500_000
    chunk_size = max(int(chunk_size), 1)

    if target_ids is not None and len(target_ids) == 0:
        return pd.DataFrame(columns=include)

    with fits.open(path, memmap=True) as hdul:
        if len(hdul) < 2:
            raise ValueError(f'Classification file {path} does not contain HDU 1')
        hdu = hdul[1]
        available = list(hdu.columns.names)
        present = [col for col in include if col in available]
        missing = [col for col in include if col not in available]
        total_rows = int(hdu.header.get('NAXIS2', 0))

        def _chunk_to_frame(start=None, stop=None):
            if total_rows == 0:
                return pd.DataFrame(columns=present)
            data = hdu.data
            if data is None:
                return pd.DataFrame(columns=present)
            if start is not None or stop is not None:
                data = data[start:stop]
            tbl = Table(data, copy=False)
            if present:
                tbl = tbl[present]
            if len(tbl) == 0:
                return pd.DataFrame(columns=present)
            return tbl.to_pandas()

        if target_ids:
            remaining = set(int(v) for v in target_ids)
            frames = []
            start = 0
            while remaining and start < total_rows:
                stop = min(start + chunk_size, total_rows)
                frame = _chunk_to_frame(start, stop)
                if frame.empty:
                    start = stop
                    continue
                mask = frame['TARGETID'].isin(remaining)
                if mask.any():
                    matched = frame.loc[mask]
                    frames.append(matched)
                    remaining.difference_update(int(v) for v in matched['TARGETID'].to_numpy(dtype=np.int64))
                start = stop
            if frames:
                df = pd.concat(frames, ignore_index=True, copy=False)
            else:
                df = pd.DataFrame(columns=include)
        else:
            df = _chunk_to_frame(None, None)

    if not df.empty:
        df['TARGETID'] = df['TARGETID'].astype(np.int64, copy=False)
        if downcast:
            for col in ('NDATA', 'NRAND'):
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], downcast='integer')
        if 'ISDATA' in df.columns:
            df['ISDATA'] = df['ISDATA'].astype(bool, copy=False)
    for col in missing:
        if col == 'ISDATA':
            df[col] = True
        else:
            df[col] = 0
    if 'ISDATA' not in df.columns:
        df['ISDATA'] = True

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


def _load_entry_class(entry, zone, cache_dir, target_ids=None, progress=False,
                      path_override=None, cache_token=None, allow_cache=True):
    """Load (and cache) the classification table for *entry* limited to *target_ids*."""
    path = path_override or entry.get('class')
    if not path or not os.path.exists(path):
        return pd.DataFrame(columns=['TARGETID', 'NDATA', 'NRAND', 'ISDATA'])
    tag_bits = [entry.get('tag') or 'combined']
    if cache_token:
        tag_bits.append(cache_token)
    chunk_rows = CLASS_LOAD_OPTIONS.get('chunk_rows')

    def _loader():
        return load_class_df(path, target_ids=target_ids, chunk_rows=chunk_rows)

    if allow_cache and cache_dir:
        return _load_or_build_df(cache_dir, _zone_cache_key('class', zone, tag_bits),
                                 [path], _loader, progress=progress)
    return _loader()


def _load_entry_prob_entropy(entry, zone, cache_dir, progress=False,
                             path_override=None, cache_token=None, allow_cache=True):
    """Load probabilities for *entry* and attach entropy column."""
    path = path_override or entry.get('prob')
    if not path or not os.path.exists(path):
        return pd.DataFrame(columns=['TARGETID', 'H'])
    tag_bits = [entry.get('tag') or 'combined']
    if cache_token:
        tag_bits.append(cache_token)

    def _loader():
        probs = load_prob_df(path)
        return entropy(probs)

    if allow_cache and cache_dir:
        return _load_or_build_df(cache_dir, _zone_cache_key('prob_entropy', zone, tag_bits),
                                 [path], _loader, progress=progress)
    return _loader()


def _entry_class_paths(entry):
    paths = entry.get('class_iter') or []
    if paths:
        for p in paths:
            yield p, os.path.basename(p)
    elif entry.get('class'):
        yield entry['class'], None

def _entry_prob_data_path(entry):
    if entry.get('prob_data'):
        return entry['prob_data']
    return entry.get('prob')


class _CDFSeriesAccumulator:
    """Incremental ECDF builder that tracks counts on a fixed *xgrid*."""

    __slots__ = ('xgrid', 'counts', 'total')

    def __init__(self, xgrid):
        self.xgrid = np.asarray(xgrid, dtype=float)
        self.counts = np.zeros_like(self.xgrid, dtype=np.int64)
        self.total = 0

    def add(self, values):
        arr = np.asarray(values, dtype=float)
        if arr.size == 0:
            return
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            return
        arr.sort()
        incr = np.searchsorted(arr, self.xgrid, side='right')
        self.counts += incr.astype(np.int64, copy=False)
        self.total += int(arr.size)

    def result(self):
        if self.total == 0:
            return None
        return self.counts.astype(float) / float(self.total)


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
    if 'Z' not in df.columns or 'ISDATA' not in df.columns:
        zone_label = zone_tag(zone)
        print(f'Z histogram skipped for zone {zone_label}: missing columns Z or ISDATA')
        return

    real, rand = df[df['ISDATA']]['Z'], df[~df['ISDATA']]['Z']
    zone_label = zone_tag(zone)

    fig, ax = plt.subplots()
    ax.grid(color=GRID_COLOR, alpha=0.3, linewidth=0.7, zorder=-10)
    ax.hist([real, rand], bins=bins, label=['Data','Random'], alpha=0.9,
            color=['royalblue', 'navy'], zorder=10, edgecolor=HIGHLIGHT_EDGE_COLOR, lw=0.3)
    ax.set(xlabel='Z', ylabel='Count', title=f'Zone {zone_label}')
    leg = ax.legend()
    if leg:
        for text in leg.get_texts():
            text.set_color(TEXT_COLOR)

    path = os.path.join(out_dir, f'z_hist_zone_{zone_label}.png')
    fig.savefig(path, dpi=PLOT_DPI); plt.close(fig)


def plot_radial_distribution(raw_df, zone, tracers, out_dir, bins):
    """
    Plot radial distribution histograms for specified tracers.

    Args:
        raw_df (pd.DataFrame): DataFrame containing raw data with 'XCART',
                    'YCART', 'ZCART', 'TRACERTYPE', and 'ISDATA' columns.
        zone (int): Zone number for title and filename.
        tracers (list): List of tracer types to plot.
        out_dir (str): Output directory to save the plots.
        bins (int): Number of bins for the histograms.
    """
    zone_label = zone_tag(zone)
    required = {'TRACERTYPE', 'ISDATA', 'XCART', 'YCART', 'ZCART'}
    missing = [col for col in required if col not in raw_df.columns]
    if missing:
        missing_str = ', '.join(missing)
        print(f'Radial plot skipped for zone {zone_label}: missing columns {missing_str}')
        return

    fig, axes = plt.subplots(1, len(tracers), figsize=(4*len(tracers),4))
    if len(tracers) == 1:
        axes = [axes]

    tracer_series = raw_df['TRACERTYPE'].astype(str)

    for ax, tracer in zip(axes, tracers):
        prefixes = _tracer_prefixes(tracer)
        display = _tracer_display_name(tracer)
        mask = tracer_series.str.startswith(prefixes)
        sub = raw_df.loc[mask]
        real = sub[sub['ISDATA']]
        rand_pool = sub[~sub['ISDATA']]
        if len(rand_pool) == 0 or len(real) == 0:
            ax.set(title=display, xlabel=r'$r$', ylabel='Count')
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes, color=TEXT_COLOR)
            ax.grid(color=GRID_COLOR, alpha=0.3, linewidth=0.7, zorder=-10)
            continue
        nsamp = min(len(rand_pool), len(real))
        rand = rand_pool.sample(nsamp, random_state=0, replace=False)

        r_real = np.linalg.norm(real[['XCART','YCART','ZCART']].to_numpy(), axis=1)
        r_rand = np.linalg.norm(rand[['XCART','YCART','ZCART']].to_numpy(), axis=1)

        ax.grid(color=GRID_COLOR, alpha=0.3, linewidth=0.7, zorder=-10)
        ax.hist([r_real, r_rand], bins, label=['Real','Random'], alpha=0.9,
                color=['royalblue', 'navy'], zorder=10, edgecolor=HIGHLIGHT_EDGE_COLOR, lw=0.05)
        ax.set(title=display, xlabel=r'$r$', ylabel='Count')
        leg = ax.legend(fontsize=6)
        if leg:
            for text in leg.get_texts():
                text.set_color(TEXT_COLOR)

    fig.suptitle(f'Radial Zone {zone_label}', color=TEXT_COLOR)
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
    ax.grid(color=GRID_COLOR, alpha=0.3, linewidth=0.7, zorder=-10)

    def ecdf(arr):
        a = np.asarray(arr, dtype=float)
        a = a[~np.isnan(a)]
        if a.size == 0:
            return None, None
        a = np.sort(a)
        y = np.arange(1, a.size + 1) / a.size
        return a, y

    if 'TRACERTYPE' not in df_r.columns or 'ISDATA' not in df_r.columns:
        zone_label = zone_tag(zone)
        print(f'CDF plot skipped for zone {zone_label}: missing TRACERTYPE or ISDATA columns')
        return

    tracer_series = df_r['TRACERTYPE'].astype(str)

    for tr in tracers:
        prefixes = _tracer_prefixes(tr)
        display = _tracer_display_name(tr)
        color = _tracer_color(tr)
        mask = tracer_series.str.startswith(prefixes)
        sub = df_r.loc[mask]
        xr, yr = ecdf(sub[sub['ISDATA']]['r'])
        if xr is not None:
            ax.plot(xr, yr, label=f'{display} real', linewidth=1, color=color, zorder=10)
        xr, yr = ecdf(sub[~sub['ISDATA']]['r'])
        if xr is not None:
            ax.plot(xr, yr, label=f'{display} rand', linewidth=1, linestyle='--', color=color, zorder=10)

    zone_label = zone_tag(zone)
    ax.set(xlabel='$r$', ylabel='CDF', title=f'CDF Zone {zone_label}')
    leg = ax.legend(fontsize=6)
    if leg:
        for text in leg.get_texts():
            text.set_color(TEXT_COLOR)
    path = os.path.join(out_dir, f'cdf_zone_{zone_label}.png')
    fig.savefig(path, dpi=PLOT_DPI)
    plt.close(fig)


def plot_cdf_dispersion(raw_dir, class_dir, zones, out_dir, tracers=None, xbins=400,
                        subsample_per_zone=None, progress=False, out_tags=None,
                        cache_dir=None, tags_map=None, zone_files_map=None):
    """
    Plot the dispersion (percentile band) of CDFs over multiple zones in one figure.
    
    Args:
        raw_dir (str): Directory with raw zone files.
        class_dir (str): Directory with class zone files.
        zones (list[int]): Zone numbers to include (e.g., range(20)).
        out_dir (str): Base output directory (the figure is written under `<out_dir>/cdf`).
        tracers (list[str] or None): Tracers to include. Default: ['BGS_BRIGHT','ELG','LRG','QSO'].
        xbins (int): Number of points in the common x-grid for interpolation.
        subsample_per_zone (int or None): Legacy sampling control (ignored by the streamed ECDF
            builder, which always uses the full sample per zone).
        progress (bool): Print progress logs.
        out_tags (list[str] or None): Override tag selection per zone.
        cache_dir (str or None): Directory used to persist intermediate DataFrames.
        tags_map (dict|None): Precomputed tag list per zone (overrides discovery when present).
        zone_files_map (dict|None): Mapping of zone -> discovered file metadata.
    """
    if tracers is None:
        tracers = ['BGS_BRIGHT','ELG','LRG','QSO']

    xgrid = np.linspace(-1.0, 1.0, xbins)

    if subsample_per_zone is not None and subsample_per_zone <= 0:
        subsample_per_zone = None
    if subsample_per_zone is not None and progress:
        print('[cdf-disp] subsample_per_zone is ignored when streaming ECDF data; using full samples.')
    subsample_per_zone = None

    per_tracer_real = {t: [] for t in tracers}
    per_tracer_rand = {t: [] for t in tracers}

    def _build_zone_cdf_state(zone, usable_entries):
        tracer_state = {tr: {'real': _CDFSeriesAccumulator(xgrid),
                             'rand': _CDFSeriesAccumulator(xgrid)}
                        for tr in tracers}

        data_seen = set()

        for entry in usable_entries:
            class_paths = list(_entry_class_paths(entry))
            if not class_paths:
                continue

            entry_key = entry.get('tag') or '(combined)'

            for cls_path, cache_token in class_paths:
                cls_df = _load_entry_class(entry, zone, cache_dir,
                                           target_ids=None,
                                           progress=progress,
                                           path_override=cls_path,
                                           cache_token=cache_token,
                                           allow_cache=False)
                if cls_df.empty:
                    continue

                _sanitize_tracer_columns(cls_df)

                if entry_key in data_seen:
                    cls_df = cls_df[cls_df.get('ISDATA') == False]
                else:
                    data_seen.add(entry_key)

                if cls_df.empty:
                    continue

                r_df = compute_r(cls_df)
                if 'TRACERTYPE' not in r_df.columns or 'ISDATA' not in r_df.columns or 'r' not in r_df.columns:
                    continue

                tracer_series = r_df['TRACERTYPE'].astype(str)
                for tr in tracers:
                    prefixes = _tracer_prefixes(tr)
                    mask = tracer_series.str.startswith(prefixes)
                    if not mask.any():
                        continue
                    subset = r_df.loc[mask]
                    if subset.empty:
                        continue
                    tracer_state[tr]['real'].add(subset.loc[subset['ISDATA'], 'r'].to_numpy(dtype=float, copy=False))
                    tracer_state[tr]['rand'].add(subset.loc[~subset['ISDATA'], 'r'].to_numpy(dtype=float, copy=False))

        packed = {}
        for tr in tracers:
            packed[tr] = {
                'real': {'counts': tracer_state[tr]['real'].counts.copy(),
                         'total': tracer_state[tr]['real'].total},
                'rand': {'counts': tracer_state[tr]['rand'].counts.copy(),
                         'total': tracer_state[tr]['rand'].total}
            }
        return {'tracers': packed}

    def _state_to_cdf(state_part):
        if not state_part:
            return None
        counts = np.asarray(state_part.get('counts'))
        total = int(state_part.get('total', 0))
        if total <= 0 or counts.size == 0:
            return None
        return counts.astype(float) / float(total)

    for i, z in enumerate(zones):
        entries = None
        if zone_files_map and z in zone_files_map:
            entries = zone_files_map[z]
        else:
            entries = discover_zone_entries(raw_dir, class_dir, z)

        if tags_map and z in tags_map:
            tag_list = tags_map[z]
        else:
            base_tags = [entry['tag'] for entry in entries if entry.get('has_class')]
            if out_tags:
                requested = {str(tag).upper() for tag in out_tags}
                tag_list = [tag for tag in base_tags if (tag or '').upper() in requested]
            else:
                tag_list = base_tags

        if not tag_list:
            if progress:
                print(f'[cdf-disp] zone {z}: skipped (no classification tags available)')
            continue

        entry_map = {entry['tag']: entry for entry in entries}
        selected_entries = [entry_map[tag] for tag in tag_list if tag in entry_map]
        usable_entries = [entry for entry in selected_entries if entry.get('has_class')]

        if not usable_entries:
            if progress:
                pretty_tags = ','.join(tag or '(combined)' for tag in tag_list)
                print(f'[cdf-disp] zone {z}: skipped (missing raw/class files for tags {pretty_tags})')
            continue

        cache_key = f"{_zone_cache_key('cdf_counts', z, tag_list)}_xb{xbins}"
        source_paths = []
        for entry in usable_entries:
            for cls_path, _ in _entry_class_paths(entry):
                source_paths.append(cls_path)

        state = _load_or_build_df(cache_dir, cache_key, source_paths,
                                   lambda z=z, entries=usable_entries: _build_zone_cdf_state(z, entries),
                                   progress=progress)
        tracer_state = (state or {}).get('tracers', {})

        for tr in tracers:
            tr_state = tracer_state.get(tr)
            if not tr_state:
                continue
            y_real = _state_to_cdf(tr_state.get('real'))
            y_rand = _state_to_cdf(tr_state.get('rand'))
            if y_real is not None and np.isfinite(y_real).any():
                per_tracer_real[tr].append(y_real)
            if y_rand is not None and np.isfinite(y_rand).any():
                per_tracer_rand[tr].append(y_rand)

    fig, ax = plt.subplots()
    ax.grid(color=GRID_COLOR, alpha=0.3, linewidth=0.7, zorder=-10)

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
    ax.set_title('Dispersion of CDF across zones', fontsize=16, color=TEXT_COLOR)
    leg = ax.legend(fontsize=9, ncol=2)
    if leg:
        for text in leg.get_texts():
            text.set_color(TEXT_COLOR)

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


def plot_pdf_entropy(raw_dir, class_dir, zones, tracers, out_path, bins=25, cache_dir=None,
                     tags_map=None, zone_files_map=None, progress=False):
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
        zone_files_map (dict|None): Mapping of zone -> discovered file metadata.
        progress (bool): Whether to print progress information.
    """

    bin_edges = np.linspace(0, 0.6, bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    bin_widths = np.diff(bin_edges)

    per_tracer_hist = {tr: [] for tr in tracers}

    def _build_zone_entropy_state(zone, usable_entries):
        tracer_state = {tr: {'counts': np.zeros(len(bin_edges) - 1, dtype=float),
                             'total': 0}
                        for tr in tracers}

        for entry in usable_entries:
            data_path = _entry_prob_data_path(entry)
            if not data_path:
                continue

            probs_df = _load_entry_prob_entropy(entry, zone, cache_dir,
                                                progress=progress,
                                                path_override=data_path,
                                                cache_token=os.path.basename(data_path) if data_path else None,
                                                allow_cache=False)
            if probs_df is None or probs_df.empty:
                continue

            if 'ISDATA' in probs_df.columns:
                probs_df = probs_df[probs_df['ISDATA'] == True]
            if probs_df.empty or 'H' not in probs_df.columns:
                continue

            _sanitize_tracer_columns(probs_df)
            tracer_series = probs_df['TRACERTYPE'].astype(str) if 'TRACERTYPE' in probs_df.columns else None

            for tracer in tracers:
                if tracer_series is not None:
                    prefixes = _tracer_prefixes(tracer)
                    mask = tracer_series.str.startswith(prefixes)
                else:
                    mask = slice(None)
                subset = probs_df.loc[mask] if isinstance(mask, pd.Series) else probs_df
                if subset.empty:
                    continue
                values = subset['H'].to_numpy(dtype=float, copy=False)
                values = values[np.isfinite(values)]
                if values.size == 0:
                    continue
                counts, _ = np.histogram(values, bins=bin_edges, density=False)
                tracer_state[tracer]['counts'] += counts
                tracer_state[tracer]['total'] += int(values.size)

        packed = {tr: {'counts': tracer_state[tr]['counts'].copy(),
                       'total': tracer_state[tr]['total']}
                  for tr in tracers}
        return {'tracers': packed}

    def _counts_to_density(state_part):
        if not state_part:
            return None
        counts = np.asarray(state_part.get('counts'))
        total = float(state_part.get('total', 0))
        if total <= 0 or counts.size == 0:
            return None
        # density=True equivalent: counts / (total * bin_width)
        return np.divide(counts.astype(float), total * bin_widths, out=np.full_like(bin_widths, np.nan, dtype=float), where=bin_widths>0)

    for z in zones:
        if zone_files_map and z in zone_files_map:
            entries = zone_files_map[z]
        else:
            entries = discover_zone_entries(raw_dir, class_dir, z)

        if tags_map and z in tags_map:
            tag_list = tags_map[z]
        else:
            tag_list = [entry['tag'] for entry in entries if (entry.get('has_prob') or entry.get('has_prob_data'))]

        if not tag_list:
            continue

        entry_map = {entry['tag']: entry for entry in entries}
        selected_entries = [entry_map[tag] for tag in tag_list if tag in entry_map]
        usable_entries = []
        for entry in selected_entries:
            if _entry_prob_data_path(entry):
                usable_entries.append(entry)

        if not usable_entries:
            continue

        cache_key = f"{_zone_cache_key('entropy_hist', z, tag_list)}_b{bins}"
        source_paths = []
        for entry in usable_entries:
            data_path = _entry_prob_data_path(entry)
            if data_path:
                source_paths.append(data_path)

        state = _load_or_build_df(cache_dir, cache_key, source_paths,
                                   lambda z=z, entries=usable_entries: _build_zone_entropy_state(z, entries),
                                   progress=progress)
        tracer_state = (state or {}).get('tracers', {})

        for tracer in tracers:
            tr_state = tracer_state.get(tracer)
            density = _counts_to_density(tr_state)
            if density is not None and np.isfinite(density).any():
                per_tracer_hist[tracer].append(density)

    fig, ax = plt.subplots()
    ax.grid(color=GRID_COLOR, alpha=0.3, linewidth=0.7, zorder=-10)

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
    ax.set_title('Normalized Shannon Entropy', fontsize=16, color=TEXT_COLOR)
    leg = ax.legend(fontsize=9, ncol=2)
    if leg:
        for text in leg.get_texts():
            text.set_color(TEXT_COLOR)

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
    tags_map = config['tags_map']
    tag_list = tags_map.get(zone, [])
    progress = config['progress']
    cache_dir = config['cache_dir']

    if not tag_list:
        if progress:
            print(f'Zone {zone}: no raw tags available; skipping.')
        return {'zone': zone, 'raw_rows': 0, 'class_rows': 0, 'r_rows': 0, 'tags': []}

    entries = config.get('zone_files_map', {}).get(zone)
    if entries is None:
        entries = discover_zone_entries(config['raw_dir'], config['class_dir'], zone)

    entry_map = {entry['tag']: entry for entry in entries}
    ordered_entries = []
    missing_entries = []
    for tag in tag_list:
        entry = entry_map.get(tag)
        if entry:
            ordered_entries.append(entry)
        else:
            missing_entries.append(tag)

    if missing_entries and progress:
        pretty_missing = ','.join(t or '(combined)' for t in missing_entries)
        print(f'Zone {zone}: missing file listings for tags {pretty_missing}; they will be ignored.')

    raw_entries = [entry for entry in ordered_entries if entry.get('has_raw')]
    if not raw_entries:
        if progress:
            print(f'Zone {zone}: no raw files found for requested tags; skipping.')
        return {'zone': zone, 'raw_rows': 0, 'class_rows': 0, 'r_rows': 0, 'tags': tag_list}

    raw_paths = [entry['raw'] for entry in raw_entries]
    raw_tags = [entry['tag'] for entry in raw_entries]

    raw_df = _load_or_build_df(cache_dir, _zone_cache_key('raw', zone, raw_tags), raw_paths,
                               lambda: _concat_existing(raw_paths, load_raw_df), progress=progress)
    _sanitize_tracer_columns(raw_df)
    if progress:
        print(f'Zone {zone}: {len(raw_df)} total objects in raw data across tags {raw_tags}')

    cls_full = pd.DataFrame(columns=['TARGETID', 'NDATA', 'NRAND', 'ISDATA'])
    cls_df = cls_full
    r_df = None

    data_target_ids = set()
    if not raw_df.empty and 'TARGETID' in raw_df.columns and 'ISDATA' in raw_df.columns:
        data_target_ids = set(raw_df.loc[raw_df['ISDATA'], 'TARGETID'].to_numpy(dtype=np.int64))

    iter_only_class = [entry for entry in ordered_entries if entry.get('has_class') and not entry.get('class')]
    if iter_only_class and progress:
        pretty_iter = ','.join(entry['tag'] or '(combined)' for entry in iter_only_class)
        print(f'Zone {zone}: classification only available as iterative chunks for tags {pretty_iter}; '
              'per-zone classification plots are skipped.')

    cls_entries = [entry for entry in ordered_entries if entry.get('has_class') and entry.get('class')]
    class_tags = [entry['tag'] for entry in cls_entries]
    if cls_entries:
        cls_paths = [entry['class'] for entry in cls_entries]
        cls_loader = lambda: _concat_existing(
            cls_paths,
            lambda p: load_class_df(p,
                                    target_ids=data_target_ids,
                                    chunk_rows=CLASS_LOAD_OPTIONS.get('chunk_rows')))

        cls_full = _load_or_build_df(cache_dir, _zone_cache_key('class', zone, class_tags), cls_paths,
                                     cls_loader, progress=progress)
        if progress:
            print(f'Zone {zone}: {len(cls_full)} total objects in class data (tags {class_tags})')
        if not cls_full.empty and 'ISDATA' in cls_full.columns:
            cls_df = cls_full[cls_full['ISDATA'] == True]
        else:
            cls_df = cls_full
        if progress:
            print(f'Zone {zone}: {len(cls_df)} total objects in class data (ISDATA=True)')

        def _build_r():
            merged = raw_df.merge(cls_df[['TARGETID', 'NDATA', 'NRAND']], on='TARGETID', how='left')
            return compute_r(merged)

        r_df = _load_or_build_df(cache_dir, _zone_cache_key('r', zone, raw_tags), raw_paths + cls_paths,
                                 _build_r, progress=progress)
        _sanitize_tracer_columns(r_df)
        if progress:
            print(f'Zone {zone}: {len(r_df)} total objects after merging raw and class data')
        missing_cls_tags = [tag for tag in raw_tags if tag not in class_tags]
        if missing_cls_tags and progress:
            pretty_missing_cls = ','.join(t or '(combined)' for t in missing_cls_tags)
            print(f'Zone {zone}: no classification files found for tags {pretty_missing_cls}')
    elif progress:
        print(f'Zone {zone}: no classification files found; skipping classification-based plots.')

    if config['plot_z']:
        plot_z_histogram(raw_df, zone, config['bins'], config['outdirs']['z'])
        if progress:
            print(f'Zone {zone}: plotted z histogram')
    if config['plot_cdf'] and r_df is not None:
        plot_cdf(r_df, zone, config['tracers'], config['outdirs']['cdf'])
        if progress:
            print(f'Zone {zone}: plotted CDF')
    elif config['plot_cdf'] and progress:
        print(f'Zone {zone}: skipped CDF plot (missing classification data).')
    if config['plot_radial']:
        plot_radial_distribution(raw_df, zone, config['tracers'], config['outdirs']['radial'], config['bins'])
        if progress:
            print(f'Zone {zone}: plotted radial distribution')

    return {'zone': zone,
            'raw_rows': len(raw_df),
            'class_rows': len(cls_df),
            'r_rows': len(r_df) if r_df is not None else 0,
            'tags': tag_list}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--raw-dir', default='/pscratch/sd/v/vtorresg/cosmic-web/dr2/raw')
    p.add_argument('--class-dir', default='/pscratch/sd/v/vtorresg/cosmic-web/dr2/classification')
    p.add_argument('--output', default='/pscratch/sd/v/vtorresg/cosmic-web/dr2/figs/')

    p.add_argument('--zones', nargs='+', default=None)
    p.add_argument('--bins', type=int, default=10)
    p.add_argument('--tracers', nargs='+', default=['BGS_ANY','LRG','ELG','QSO'])

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
                   help='Legacy max-sample cap for the dispersion plot (ignored in streamed mode)')
    p.add_argument('--progress', action='store_true', default=False, help='Print simple progress logs')
    p.add_argument('--out-tags', nargs='*', default=None,
                   help='Limit to these out-tags (e.g., LRG ELG). If omitted, auto-discovers tags per zone.')
    p.add_argument('--raw-columns', nargs='+', default=None,
                   help=('Extra raw columns to load (defaults to minimal set). '
                         'Use "all" to keep every column from the FITS table.'))
    p.add_argument('--no-downcast', action='store_true',
                   help='Disable float/int downcasting when loading raw tables (uses more RAM).')
    p.add_argument('--raw-limit', type=int, default=None,
                   help=('Cap the number of rows read per raw FITS file (sampled as a contiguous block). '
                         'Default comes from $PLOT_EXTRA_RAW_LIMIT (3000000 if unset, <=0 disables).'))
    p.add_argument('--raw-random', action='store_true',
                   help='With --raw-limit, choose a random contiguous block instead of the first rows.')
    p.add_argument('--raw-seed', type=int, default=0,
                   help='Seed used for deterministic raw sampling (default: 0).')
    p.add_argument('--class-chunk', type=int, default=500000,
                   help='Chunk size when streaming classification FITS tables (default: 500000).')

    default_workers = 1#os.cpu_count()
    p.add_argument('--workers', type=int, default=default_workers,
                   help='Number of worker processes for per-zone plots (default: CPU count - 1)')
    p.add_argument('--cache-dir', default=None,
                   help='Directory for cached intermediate data (defaults to <output>/_cache)')
    p.add_argument('--no-cache', action='store_true', help='Disable on-disk caching of intermediate data')

    return p.parse_args()


def main():
    args = parse_args()
    global PLOT_DPI, RAW_LOAD_OPTIONS, CACHE_CONTEXT_SUFFIX
    PLOT_DPI = args.dpi

    if args.raw_columns:
        if len(args.raw_columns) == 1 and args.raw_columns[0].lower() in ('all', '*'):
            RAW_LOAD_OPTIONS['columns'] = 'all'
        else:
            RAW_LOAD_OPTIONS['columns'] = args.raw_columns
    else:
        RAW_LOAD_OPTIONS['columns'] = None
    RAW_LOAD_OPTIONS['downcast'] = not args.no_downcast
    if args.raw_limit is not None:
        RAW_LOAD_OPTIONS['limit'] = int(args.raw_limit) if args.raw_limit > 0 else None
    limit_active = RAW_LOAD_OPTIONS['limit'] is not None
    RAW_LOAD_OPTIONS['randomize'] = bool(args.raw_random and limit_active)
    RAW_LOAD_OPTIONS['seed'] = int(args.raw_seed or 0)

    CLASS_LOAD_OPTIONS['chunk_rows'] = max(int(args.class_chunk), 1)

    col_setting = RAW_LOAD_OPTIONS['columns']
    if col_setting in (None, 'all', '*'):
        col_label = 'default' if col_setting is None else 'all'
    elif isinstance(col_setting, str):
        col_label = col_setting
    else:
        col_label = f'sel{len(col_setting)}'

    limit_label = RAW_LOAD_OPTIONS['limit'] if RAW_LOAD_OPTIONS['limit'] else 'all'
    random_label = 'rand' if RAW_LOAD_OPTIONS['randomize'] else 'head'
    down_label = 'dc1' if RAW_LOAD_OPTIONS['downcast'] else 'dc0'
    chunk_label = CLASS_LOAD_OPTIONS['chunk_rows']
    seed_label = RAW_LOAD_OPTIONS['seed']
    CACHE_CONTEXT_SUFFIX = safe_tag(f'{col_label}_{limit_label}_{random_label}_{down_label}_{chunk_label}_{seed_label}')

    if args.progress:
        if col_setting is None:
            cols_msg = 'TARGETID,TRACERTYPE,RANDITER,Z,XCART,YCART,ZCART'
        elif isinstance(col_setting, str):
            cols_msg = col_setting
        else:
            cols_msg = ','.join(col_setting)
        print(f'[loader] raw columns = {cols_msg} (downcast={RAW_LOAD_OPTIONS["downcast"]})')
        if RAW_LOAD_OPTIONS['limit']:
            block_msg = 'random block' if RAW_LOAD_OPTIONS['randomize'] else 'leading block'
            print(f'[loader] raw row limit = {RAW_LOAD_OPTIONS["limit"]} ({block_msg}, seed={RAW_LOAD_OPTIONS["seed"]})')
        else:
            print('[loader] raw row limit = all rows')
        print(f'[loader] class chunk rows = {CLASS_LOAD_OPTIONS["chunk_rows"]}')

    args.class_dir = normalize_release_dir(args.class_dir)

    # if args.usetex:
        # matplotlib.rcParams['text.usetex'] = True
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

    zone_files_map = {}
    tags_map = {}
    skipped_no_raw = []
    skipped_missing_tags = []
    for zone in zones:
        entries = discover_zone_entries(args.raw_dir, args.class_dir, zone)
        zone_files_map[zone] = entries

        has_any_raw = any(entry.get('has_raw') for entry in entries)

        if args.out_tags:
            requested = {str(tag).upper() for tag in args.out_tags}
            available_raw = {(entry['tag'] or '').upper() for entry in entries if entry.get('has_raw')}
            filtered_entries = [entry for entry in entries
                                if entry.get('has_raw') and (entry['tag'] or '').upper() in requested]
            missing_requested = requested - available_raw
            if missing_requested and args.progress:
                pretty_missing = ','.join(sorted(missing_requested))
                print(f'[plot] zone {zone}: missing requested tags {pretty_missing}')
        else:
            filtered_entries = [entry for entry in entries if entry.get('has_raw')]

        if not filtered_entries:
            if args.out_tags and has_any_raw:
                skipped_missing_tags.append(zone)
                if args.progress:
                    print(f'[plot] zone {zone}: requested tags not available; skipping.')
            else:
                skipped_no_raw.append(zone)
                if args.progress:
                    print(f'[plot] zone {zone}: no raw files found; skipping.')
            continue

        tag_list = [entry['tag'] for entry in filtered_entries]
        tags_map[zone] = tag_list
        if args.progress:
            non_empty = [t for t in tag_list if t]
            pretty = ','.join(non_empty) if non_empty else '(combined)'
            print(f'[plot] zone {zone}: tags={pretty}')

    available_zones = [zone for zone in zones if zone in tags_map]
    if skipped_no_raw and not args.progress:
        print(f'Skipping zones without raw files: {skipped_no_raw}')
    if skipped_missing_tags and not args.progress:
        print(f'Skipping zones (requested tags unavailable): {skipped_missing_tags}')
    zones = available_zones

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
                   'tags_map': tags_map,
                   'zone_files_map': zone_files_map}

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
                            cache_dir=cache_dir, tags_map=tags_map,
                            zone_files_map=zone_files_map)
        print(f'Plotted CDF dispersion for zones: {list(zones)}')
    if args.plot_entropy_cdf:
        plot_pdf_entropy(args.raw_dir, args.class_dir, zones, args.tracers, args.output, args.bins,
                         cache_dir=cache_dir, tags_map=tags_map,
                         zone_files_map=zone_files_map, progress=args.progress)
        print(f'Plotted PDF of entropy for zones: {list(zones)}')


if __name__ == '__main__':
    main()
