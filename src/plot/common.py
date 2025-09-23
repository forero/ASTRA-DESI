import os
from typing import Iterable, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from astropy.table import Table

from desiproc.paths import ( locate_classification_file, locate_probability_file,
                            safe_tag, zone_tag,)

__all__ = [
    "resolve_raw_path",
    "resolve_class_path",
    "resolve_probability_path",
    "load_raw_dataframe",
    "load_probability_dataframe",
]


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
    return locate_probability_file(release_dir, zone, out_tag)


def load_raw_dataframe(raw_path):
    """
    Load a raw FITS catalogue into a pandas DataFrame.

    The function annotates the frame with convenience columns:

    * ``ISDATA``: Bool flag indicating real data rows (``RANDITER == -1``)
    * ``BASE`` / ``BASE_CORE``: Normalized tracer labels
    * ``TARGETID`` cast to ``int64``

    Args:
        raw_path (str): Path to the raw FITS file.
    Returns:
        pd.DataFrame: DataFrame containing the raw catalogue.
    Raises:
        ValueError: If required columns are missing.
    """
    table = Table.read(raw_path, memmap=True)
    frame = table.to_pandas()

    if 'RANDITER' in frame.columns:
        frame['ISDATA'] = frame['RANDITER'].to_numpy() == -1
    else:
        frame['ISDATA'] = True

    if 'TRACERTYPE' in frame.columns:
        frame['BASE'] = frame['TRACERTYPE'].apply(_normalize_tracertype)
        frame['BASE_CORE'] = frame['BASE'].str.rsplit('_', n=1).str[0]
    else:
        raise ValueError(f"Raw file {raw_path} is missing column: TRACERTYPE")

    if 'TARGETID' in frame.columns:
        frame['TARGETID'] = frame['TARGETID'].astype(np.int64)

    required = ('RA', 'DEC', 'Z', 'TARGETID')
    missing = [col for col in required if col not in frame.columns]
    if missing:
        raise ValueError(f"Raw file {raw_path} is missing columns: {missing}")

    return frame


def load_probability_dataframe(prob_path):
    """
    Load the probability FITS table into a pandas DataFrame.
    
    Args:
        prob_path (str): Path to the probability FITS file.
    Returns:
        pd.DataFrame: DataFrame containing the probability table with columns:
            'TARGETID' (int64), 'PVOID' (float), 'PSHEET' (float),
            'PFILAMENT' (float), 'PKNOT' (float).
    """
    table = Table.read(prob_path, memmap=True)
    frame = table.to_pandas()
    for column in ('TARGETID', 'PVOID', 'PSHEET', 'PFILAMENT', 'PKNOT'):
        if column not in frame.columns:
            frame[column] = 0.0 if column != 'TARGETID' else frame.get('TARGETID', pd.Series(dtype=np.int64))
    frame['TARGETID'] = frame['TARGETID'].astype(np.int64, copy=False)
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