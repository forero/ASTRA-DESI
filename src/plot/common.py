import glob
import hashlib
import os
from typing import Iterable, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from astropy.table import Table
from astropy.io import fits

from desiproc.paths import ( locate_classification_file, locate_probability_file,
                            safe_tag, zone_tag,)

__all__ = ["resolve_raw_path",
           "resolve_class_path",
           "resolve_probability_path",
           "load_raw_dataframe",
           "load_probability_dataframe",
           "table_row_count"]

_MINIMUM_RAW_COLUMNS = ("TARGETID", "TRACERTYPE")
_DEFAULT_RAW_COLUMNS = _MINIMUM_RAW_COLUMNS + ("RANDITER", "Z", "XCART", "YCART", "ZCART")


def resolve_raw_path(raw_dir, zone, out_tag=None):
    """
    Return the path to the raw FITS file for ``zone``.

    Args:
        raw_dir (str): Directory containing raw zone files.
        zone (object): Zone identifier.
        out_tag (object | None): Optional suffix used during generation.
    Returns:
        str: Path to the raw FITS file.
    Raises:
        FileNotFoundError: If the expected file does not exist.
    """
    zone_str = zone_tag(zone)
    suffix = safe_tag(out_tag)
    base = os.path.join(raw_dir, f"zone_{zone_str}{suffix}")
    candidates = (f"{base}.fits.gz", f"{base}.fits")
    for path in candidates:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(f"No raw table found for zone {zone_str} (expected {candidates[0]} or {candidates[1]})")


def resolve_class_path(release_dir, zone, out_tag=None):
    """
    Return the path to the classification FITS file for ``zone``.
    
    Args:
        release_dir (str): Directory containing classification files.
        zone (object): Zone identifier.
        out_tag (object | None): Optional suffix used during generation.
    Returns:
        str: Path to the classification FITS file.
    """
    return locate_classification_file(release_dir, zone, out_tag)


def resolve_probability_path(release_dir, zone, out_tag=None):
    """
    Return the path to the probability FITS file for ``zone``.
    
    Args:
        release_dir (str): Directory containing probability files.
        zone (object): Zone identifier.
        out_tag (object | None): Optional suffix used during generation.
    Returns:
        str: Path to the probability FITS file.
    """
    try:
        return locate_probability_file(release_dir, zone, out_tag)
    except FileNotFoundError:
        pass

    base = os.path.join(release_dir, 'probabilities')
    zone_str = zone_tag(zone)
    tsuf = safe_tag(out_tag)
    candidates = [
        os.path.join(base, f"zone_{zone_str}{tsuf}_probability.fits.gz"),
        os.path.join(base, f"zone_{zone_str}{tsuf}_probability.fits"),
    ]

    if not tsuf:
        pattern = os.path.join(base, f"zone_{zone_str}_probability*.fits*")
        candidates.extend(sorted(glob.glob(pattern)))

    for path in candidates:
        if os.path.exists(path):
            return path

    raise FileNotFoundError(candidates[0])


def table_row_count(path: str, hdu: int = 1) -> int:
    """
    Return the number of rows stored in the requested HDU of a FITS table.
    """
    with fits.open(path, memmap=True) as hdul:
        return int(hdul[hdu].header.get("NAXIS2", 0))


def _stable_int_from_path(path: str) -> int:
    """
    Compute a stable 32-bit integer hash from a filesystem path.
    """
    digest = hashlib.blake2b(os.fsencode(os.path.abspath(path)), digest_size=8).digest()
    return int.from_bytes(digest, byteorder="little", signed=False) & 0xFFFFFFFF


def _uniq_seq(values: Iterable[str]) -> Tuple[str, ...]:
    seen = set()
    ordered = []
    for value in values:
        if value in seen:
            continue
        ordered.append(value)
        seen.add(value)
    return tuple(ordered)


def load_raw_dataframe(raw_path,
                       columns: Optional[Iterable[str]] = None,
                       downcast: bool = True,
                       row_limit: Optional[int] = None,
                       randomize: bool = False,
                       seed: Optional[int] = None):
    """
    Load a raw FITS catalogue into a pandas DataFrame while minimising memory usage.

    The loader keeps only a small, plot-oriented column set by default. Additional
    column names can be supplied via ``columns``; ``"all"`` or ``"*"`` forces the
    full table to be materialised. Regardless of the selection, convenience columns
    are added:

    * ``ISDATA``: Bool flag indicating real data rows (``RANDITER == -1`` when available)
    * ``BASE`` / ``BASE_CORE``: Normalised tracer labels derived from ``TRACERTYPE``
    * ``TARGETID`` cast to ``int64``

    Args:
        raw_path (str): Path to the raw FITS file.
        columns (Iterable[str] | str | None): Additional columns to keep. ``None`` keeps
            the default minimalist set, ``"all"``/``"*"`` keeps every column.
        downcast (bool): If True, attempt to downcast float/int columns to reduce RAM.
    Returns:
        pd.DataFrame: DataFrame containing the requested columns.
    Raises:
        ValueError: If mandatory columns such as ``TRACERTYPE`` or ``TARGETID`` are missing.
    """
    include = None
    if isinstance(columns, str):
        if columns.lower() in ("all", "*"):
            include = None
        else:
            include = _uniq_seq(list(_MINIMUM_RAW_COLUMNS) + [columns])
    elif columns:
        include = _uniq_seq(list(_MINIMUM_RAW_COLUMNS) + list(columns))
    else:
        include = _DEFAULT_RAW_COLUMNS

    with fits.open(raw_path, memmap=True) as hdul:
        if len(hdul) < 2:
            raise ValueError(f"Raw file {raw_path} does not contain HDU 1")
        hdu = hdul[1]
        available = list(hdu.columns.names)

        if include is not None:
            missing = [name for name in include if name not in available]
            if missing:
                raise ValueError(f"Raw file {raw_path} is missing columns: {missing}")

        total_rows = int(hdu.header.get("NAXIS2", 0))
        row_slice = None
        if row_limit is not None and row_limit > 0 and total_rows > 0:
            limit = min(int(row_limit), total_rows)
            if limit < total_rows:
                if randomize:
                    rng_seed = (_stable_int_from_path(raw_path) + (seed or 0)) & 0xFFFFFFFF
                    rng = np.random.default_rng(rng_seed)
                    max_start = total_rows - limit
                    start = int(rng.integers(0, max_start + 1))
                else:
                    start = 0
                row_slice = slice(start, start + limit)

        data = hdu.data
        if data is None:
            raise ValueError(f"Raw file {raw_path} has no table data")
        if row_slice is not None:
            data = data[row_slice]

        table = Table(data, copy=False)
        if include is not None:
            table = table[list(include)]

    frame = table.to_pandas()

    if 'TRACERTYPE' not in frame.columns:
        raise ValueError(f"Raw file {raw_path} is missing column: TRACERTYPE")

    frame['TRACERTYPE'] = frame['TRACERTYPE'].apply(_normalize_tracertype)
    frame['TRACERTYPE'] = pd.Categorical(frame['TRACERTYPE'])
    frame['BASE'] = frame['TRACERTYPE'].astype(str).str.replace(r'_(DATA|RAND)$', '', regex=True)
    frame['BASE'] = pd.Categorical(frame['BASE'])
    frame['BASE_CORE'] = frame['BASE'].astype(str).str.rsplit('_', n=1).str[0]
    frame['BASE_CORE'] = pd.Categorical(frame['BASE_CORE'])

    if 'TARGETID' in frame.columns:
        frame['TARGETID'] = frame['TARGETID'].astype(np.int64, copy=False)
    else:
        raise ValueError(f"Raw file {raw_path} is missing column: TARGETID")

    if 'RANDITER' in frame.columns:
        frame['RANDITER'] = pd.to_numeric(frame['RANDITER'], downcast='integer')
        frame['ISDATA'] = frame['RANDITER'].to_numpy() == -1
    elif 'ISDATA' in frame.columns:
        frame['ISDATA'] = frame['ISDATA'].astype(bool, copy=False)
    else:
        frame['ISDATA'] = frame['TRACERTYPE'].astype(str).str.endswith('_DATA')

    frame['ISDATA'] = frame['ISDATA'].astype(bool, copy=False)

    if downcast:
        float_cols = frame.select_dtypes(include=['float', 'float64']).columns
        for col in float_cols:
            frame[col] = pd.to_numeric(frame[col], downcast='float')
        int_cols = [col for col in frame.select_dtypes(include=['int', 'int64', 'int32']).columns
                    if col != 'TARGETID']
        for col in int_cols:
            frame[col] = pd.to_numeric(frame[col], downcast='integer')

    return frame


def load_probability_dataframe(prob_path, include_random=False):
    """
    Load the probability FITS table into a pandas DataFrame.

    Args:
        prob_path (str): Path to the probability FITS file.
        include_random (bool): Whether to retain random rows and duplicates.
    Returns:
        pd.DataFrame: DataFrame containing the probability table with columns:
            'TARGETID' (int64), 'PVOID' (float), 'PSHEET' (float),
            'PFILAMENT' (float), 'PKNOT' (float).
    """
    table = Table.read(prob_path, memmap=True)
    frame = table.to_pandas()
    if not include_random:
        if 'ISDATA' in frame.columns:
            frame = frame[frame['ISDATA'] == True]
        if 'TARGETID' in frame.columns:
            frame = frame.drop_duplicates(subset=['TARGETID'], keep='first')
    for column in ('TARGETID', 'PVOID', 'PSHEET', 'PFILAMENT', 'PKNOT'):
        if column not in frame.columns:
            frame[column] = 0.0 if column != 'TARGETID' else frame.get('TARGETID', pd.Series(dtype=np.int64))
    frame['TARGETID'] = frame['TARGETID'].astype(np.int64, copy=False)
    if include_random:
        cols = [c for c in ('TARGETID', 'TRACERTYPE', 'RANDITER', 'ISDATA', 'PVOID', 'PSHEET', 'PFILAMENT', 'PKNOT') if c in frame.columns]
        return frame[cols]
    return frame[['TARGETID', 'PVOID', 'PSHEET', 'PFILAMENT', 'PKNOT']]


def _normalize_tracertype(value) -> str:
    """
    Normalize tracer type strings to a standard format.
    
    Args:
        value (str | bytes | bytearray): Input tracer type.
    Returns:
        str: Normalized tracer type string.
    """
    if isinstance(value, (bytes, bytearray)):
        try:
            value = value.decode('utf-8', errors='ignore')
        except Exception:
            value = str(value)
    return str(value).strip()
