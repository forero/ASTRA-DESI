import argparse
import gzip
import os
import shutil
import sys
import time as t
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.table import Table
from scipy.spatial import ConvexHull, QhullError
from sklearn.cluster import DBSCAN

if __package__ is None or __package__ == '':
    pkg_root = Path(__file__).resolve().parent
    if str(pkg_root) not in sys.path:
        sys.path.append(str(pkg_root))
    parent_root = pkg_root.parent
    if str(parent_root) not in sys.path:
        sys.path.append(str(parent_root))
    from implement_astra import TempTableStore
    from paths import locate_classification_file, safe_tag, zone_tag
else:
    from .implement_astra import TempTableStore
    from .paths import locate_classification_file, safe_tag, zone_tag


RAW_COLS = ['TRACERTYPE','TRACER_ID','RANDITER','TARGETID','XCART','YCART','ZCART']
CLASS_COLS = ['TARGETID','RANDITER','ISDATA','NDATA','NRAND','TRACERTYPE','TRACER_ID']
WEBTYPE_MAPPING = np.array(['void', 'sheet', 'filament', 'knot'], dtype='U8')
HULL_SAMPLE_CAP = int(os.environ.get('ASTRA_GROUPS_HULL_CAP', '50000'))
_GROUP_TRACER_DTYPE = 'S32'
_GROUP_WEBTYPE_DTYPE = 'S8'
_GROUP_ROW_DTYPE = np.dtype([('TRACERTYPE', _GROUP_TRACER_DTYPE),
                             ('TARGETID', np.int64),
                             ('RANDITER', np.int32),
                             ('WEBTYPE', _GROUP_WEBTYPE_DTYPE),
                             ('GROUPID', np.int32),
                             ('NPTS', np.int32),
                             ('XCM', np.float32),
                             ('YCM', np.float32),
                             ('ZCM', np.float32),
                             ('A', np.float32),
                             ('B', np.float32),
                             ('C', np.float32),
                             ('LINKLEN', np.float32),
                             ('ISDATA', np.bool_)])

_GROUP_FITS_COLUMNS = (('TRACERTYPE', '32A'),
                       ('TARGETID', 'K'),
                       ('RANDITER', 'J'),
                       ('WEBTYPE', '8A'),
                       ('GROUPID', 'J'),
                       ('NPTS', 'J'),
                       ('XCM', 'E'),
                       ('YCM', 'E'),
                       ('ZCM', 'E'),
                       ('A', 'E'),
                       ('B', 'E'),
                       ('C', 'E'),
                       ('LINKLEN', 'E'),
                       ('ISDATA', 'L'))


def _group_chunk_rows():
    """
    Get the number of rows per chunk for group output.
    
    Returns:
        int: Number of rows per chunk.
    """
    try:
        return max(1, int(os.environ.get('ASTRA_GROUP_CHUNK_ROWS', '250000')))
    except Exception:
        return 250000


def _group_spill_dir():
    """
    Get the temporary spill directory for group processing.
    
    Returns:
        str or None: Path to the spill directory, or None if not set.
    """
    return (os.environ.get('ASTRA_GROUP_SPILL_DIR')
            or os.environ.get('ASTRA_TMPDIR')
            or os.environ.get('PSCRATCH')
            or os.environ.get('TMPDIR'))


def _ascii_fill(value, size, dtype):
    """
    Create an array of fixed-size ASCII strings filled with the given value.
    
    Args:
        value: Input value to encode.
        size (int): Number of elements in the output array.
        dtype: Numpy dtype for the output array (e.g., 'S32').
    Returns:
        np.ndarray: Array of fixed-size ASCII strings.
    """
    arr = np.empty(size, dtype=np.dtype(dtype))
    encoded = str(value).encode('ascii', errors='ignore')[:arr.itemsize]
    arr[...] = encoded
    return arr


def _normalize_tracer_array(values):
    """
    Vectorised normalisation of tracer labels to strip _DATA/_RAND suffixes.

    Args:
        values (array-like): Input tracer labels.
    Returns:
        np.ndarray: Array of normalised tracer prefixes.
    """
    arr = np.asarray(values).astype('U32', copy=False)
    head, sep, tail = np.char.rpartition(arr, '_')
    mask = (sep != '') & np.isin(np.char.upper(tail), ('DATA', 'RAND'))
    result = arr.copy()
    if np.any(mask):
        result[mask] = head[mask]
    return result


def _compute_tracer_codes(raw_labels, sel_labels):
    """
    Generate aligned integer codes for raw and selected tracer labels.

    Args:
        raw_labels (array-like): Raw tracer labels (strings).
        sel_labels (array-like): Selected tracer labels (strings).
    Returns:
        tuple[np.ndarray, np.ndarray]: Raw codes array and selected codes array.
    """
    raw_base = _normalize_tracer_array(raw_labels)
    sel_base = _normalize_tracer_array(sel_labels)

    raw_unique, raw_codes = np.unique(raw_base, return_inverse=True)
    sel_unique, sel_inverse = np.unique(sel_base, return_inverse=True)

    lookup = np.searchsorted(raw_unique, sel_unique)
    matched = (lookup < raw_unique.size) & (raw_unique[lookup] == sel_unique)
    sel_code_map = np.full(sel_unique.size, -1, dtype=np.int32)
    sel_code_map[matched] = lookup[matched]
    sel_codes = sel_code_map[sel_inverse]
    return raw_codes.astype(np.int32, copy=False), sel_codes.astype(np.int32, copy=False)


def classify_by_probability(prob_tbl):
    """
    Annotate ``prob_tbl`` rows with the most likely web type.

    Args:
        prob_tbl (Table): Probability table containing columns ``PVOID``,
            ``PSHEET``, ``PFILAMENT``, ``PKNOT``.
    Returns:
        Table: The input table with a ``WEBTYPE`` column containing the
            maximum-probability label per row (empty string when undefined).
    Raises:
        TypeError: When ``prob_tbl`` is not an Astropy Table.
        KeyError: When required probability columns are missing.
    """
    if not isinstance(prob_tbl, Table):
        raise TypeError('classify_by_probability expects an astropy Table')

    required = ('PVOID', 'PSHEET', 'PFILAMENT', 'PKNOT')
    missing = [col for col in required if col not in prob_tbl.colnames]
    if missing:
        raise KeyError(f'Probability table missing columns: {missing}')

    n_rows = len(prob_tbl)
    if 'WEBTYPE' in prob_tbl.colnames:
        prob_tbl.remove_column('WEBTYPE')

    if n_rows == 0:
        prob_tbl['WEBTYPE'] = np.empty(0, dtype='U8')
        return prob_tbl

    cols = []
    for name in required:
        data = prob_tbl[name]
        if isinstance(data, np.ma.MaskedArray):
            values = data.filled(np.nan)
        else:
            values = np.asarray(data)
        cols.append(np.asarray(values, dtype=np.float64))

    arr = np.column_stack(cols)
    arr = np.nan_to_num(arr, nan=-np.inf, copy=False)
    idx = np.argmax(arr, axis=1)
    webtypes = WEBTYPE_MAPPING[idx].astype('U8', copy=False)
    invalid = ~np.isfinite(arr).any(axis=1)
    if np.any(invalid):
        webtypes = webtypes.copy()
        webtypes[invalid] = ''

    prob_tbl['WEBTYPE'] = webtypes
    return prob_tbl


def _to_tracer_text(value):
    """
    Decode tracer values to plain strings.

    Args:
        value: Input value, possibly bytes.
    Returns:
        str: Decoded and stripped string.
    """
    if isinstance(value, (bytes, bytearray)):
        try:
            return value.decode('utf-8', errors='ignore').strip()
        except Exception:
            return value.decode('latin-1', errors='ignore').strip()
    return str(value).strip()


def _normalize_tracer_label(value):
    """
    Remove _DATA/_RAND suffixes from tracer labels.

    Args:
        value: Input tracer label.
    Returns:
        str: Normalized tracer prefix.
    """
    text = _to_tracer_text(value)
    if not text:
        return ''
    head, sep, tail = text.rpartition('_')
    if sep and tail.upper() in {'DATA', 'RAND'}:
        return head
    return text


def _read_fits_columns(path, cols):
    """
    Read specific columns from a FITS file.

    Args:
        path (str): Path to the FITS file.
        cols (list[str]): List of column names to read.
    Returns:
        Table: A table containing the requested columns.
    """
    with fits.open(path, memmap=True) as hdul:
        data = hdul[1].data
        available = set(data.columns.names)
        subset = {}
        for name in cols:
            if name not in available:
                continue
            col = data[name]
            subset[name] = np.array(col, copy=False)
    if not subset:
        raise KeyError(f'None of the requested columns {cols} present in {path}')
    return Table(subset, copy=False)


def _get_zone_paths(raw_dir, class_dir, zone, out_tag=None):
    """
    Get file paths for a given zone number or label.
    
    Args:
        raw_dir (str): Directory containing raw data files.
        class_dir (str): Directory containing classification data files.
        zone (int or str): Zone number (int) or label (str).
        out_tag (str or None): Optional tag to append to filenames.
    Returns:
        Tuple[str, str]: Paths to the raw and classification files for the zone.
    """
    ztag = zone_tag(zone)
    tsuf = safe_tag(out_tag)
    raw_base = os.path.join(raw_dir, f'zone_{ztag}{tsuf}')
    raw_candidates = (f'{raw_base}.fits.gz', f'{raw_base}.fits')
    for raw_path in raw_candidates:
        if os.path.exists(raw_path):
            break
    else:
        raise FileNotFoundError(f'Raw table not found for zone {zone} with tag {out_tag}')

    class_path = locate_classification_file(class_dir, zone, out_tag)
    return raw_path, class_path


def _read_zone_tables(raw_path, class_path):
    """
    Read the raw and class tables for a given zone.
    
    Args:
        raw_path (str): Path to the raw data file.
        class_path (str): Path to the classification data file.
    Returns:
        Tuple[Table, Table]: Tuple containing the raw and classification tables.
    """
    raw = _read_fits_columns(raw_path, RAW_COLS)
    cls = _read_fits_columns(class_path, CLASS_COLS)
    return raw, cls


def _split_blocks(tracer_labels, randiters):
    """
    Yield index blocks grouped by tracer label and random iteration.

    Args:
        tracer_labels (np.ndarray): Tracer labels (strings) for each row.
        randiters (np.ndarray): Random iteration numbers per row.
    Yields:
        tuple[str, int, np.ndarray]: Tracer label, iteration, and indices for the block.
    """
    tr = np.asarray(tracer_labels)
    ri = np.asarray(randiters)
    if tr.size == 0:
        return
    order = np.lexsort((ri, tr))
    tr_sorted = tr[order]
    ri_sorted = ri[order]
    change = np.empty(order.size, dtype=bool)
    change[0] = True
    change[1:] = (tr_sorted[1:] != tr_sorted[:-1]) | (ri_sorted[1:] != ri_sorted[:-1])
    start_idx = np.nonzero(change)[0]
    end_idx = np.r_[start_idx[1:], order.size]
    for start, end in zip(start_idx, end_idx):
        idxs = order[start:end]
        yield tr_sorted[start], int(ri_sorted[start]), idxs


def _max_pairwise_distance(coords, block_size=2048):
    """
    Compute the maximum pairwise distance without building a dense NxN array.

    Args:
        coords (np.ndarray): Array with shape (N, 3).
        block_size (int): Number of points processed per block.
    Returns:
        float: Maximum Euclidean distance between any two points.
    """
    npts = coords.shape[0]
    if npts <= 1:
        return 0.0

    coords64 = coords.astype(np.float64, copy=False)
    max_dist2 = 0.0

    for i_start in range(0, npts, block_size):
        i_end = min(i_start + block_size, npts)
        block_i = coords64[i_start:i_end]
        norm_i = np.sum(block_i * block_i, axis=1)

        for j_start in range(i_start, npts, block_size):
            j_end = min(j_start + block_size, npts)
            block_j = coords64[j_start:j_end]
            norm_j = np.sum(block_j * block_j, axis=1)

            cross = np.dot(block_i, block_j.T)
            dist2 = norm_i[:, None] + norm_j[None, :] - 2.0 * cross
            np.maximum(dist2, 0.0, out=dist2)

            if j_start == i_start:
                if dist2.shape[0] < 2:
                    continue
                tri_mask = np.triu_indices(dist2.shape[0], k=1)
                block_max = float(np.max(dist2[tri_mask]))
            else:
                block_max = float(np.max(dist2))

            if block_max > max_dist2:
                max_dist2 = block_max

    return float(np.sqrt(max_dist2)) if max_dist2 > 0 else 0.0


def length(data_raw, hull_cap=None, rng_seed=0):
    """
    Estimate a linking length based on the convex hull of the points.
    
    Args:
        data_raw (Table): Table containing the raw data with 'XCART', 'YCART', 'ZCART' columns.
    Returns:
        float: Estimated linking length, or 0.0 if it cannot be computed.
    """
    x_all = np.asarray(data_raw['XCART'])
    y_all = np.asarray(data_raw['YCART'])
    z_all = np.asarray(data_raw['ZCART'])

    npts = x_all.size
    if npts <= 1:
        return float(np.finfo(np.float32).eps)

    cap = HULL_SAMPLE_CAP if hull_cap is None else hull_cap
    idx = None
    if cap and npts > cap:
        rs = np.random.RandomState(rng_seed)
        idx = rs.choice(npts, size=cap, replace=False)

    if idx is None:
        sample = np.column_stack((x_all, y_all, z_all)).astype(np.float32, copy=False)
    else:
        sample = np.column_stack((x_all[idx], y_all[idx], z_all[idx])).astype(np.float32, copy=False)

    try:
        hull = ConvexHull(sample)
        vol = float(hull.volume)
    except QhullError:
        dx = float(np.max(x_all) - np.min(x_all))
        dy = float(np.max(y_all) - np.min(y_all))
        dz = float(np.max(z_all) - np.min(z_all))
        diag_full = np.sqrt(dx*dx + dy*dy + dz*dz)
        dist_sample = _max_pairwise_distance(sample)
        return float(max(dist_sample, diag_full, np.finfo(np.float32).eps))

    if vol <= 0:
        dx = float(np.max(x_all) - np.min(x_all))
        dy = float(np.max(y_all) - np.min(y_all))
        dz = float(np.max(z_all) - np.min(z_all))
        diag_full = np.sqrt(dx*dx + dy*dy + dz*dz)
        dist_sample = _max_pairwise_distance(sample)
        return float(max(dist_sample, diag_full, np.finfo(np.float32).eps))

    return float(np.cbrt(vol / float(npts)))


def _dbscan_labels(coords, eps):
    """
    Applies DBSCAN (same as FoF) to the given coordinates to find clusters.
    
    Args:
        coords (np.ndarray): Array of shape (N, 3) containing 3D coordinates.
        eps (float): The maximum distance between two samples for one to be
                     considered as in the neighborhood of the other.
    Returns:
        np.ndarray: Array of cluster labels, where -1 indicates noise.
    """
    lab = DBSCAN(eps=eps, min_samples=1, metric='euclidean', algorithm='ball_tree').fit(coords).labels_
    return lab.astype(np.int32)


def _grouped_sum(values, labels, ngrp):
    """
    Computes the sum of values grouped by labels.
    
    Args:
        values (np.ndarray): Array of values to be summed.
        labels (np.ndarray): Array of labels corresponding to the values.
        ngrp (int): Number of unique groups.
    Returns:
        np.ndarray: Array of sums for each group.
    """
    order = np.argsort(labels, kind='mergesort')
    vals = values[order]
    lab_o = labels[order]
    cuts = np.r_[0, np.cumsum(np.bincount(lab_o, minlength=ngrp))]
    return np.add.reduceat(vals, cuts[:-1])


def _group_inertia(coords, labels):
    """
    Computes the inertia of each group based on its 3D coordinates.
    
    Args:
        coords (np.ndarray): Array of shape (N, 3) containing 3D coordinates.
        labels (np.ndarray): Array of cluster labels for the coordinates.
    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        Returns labels, counts of points in each group, and the center of mass
        (xcm, ycm, zcm) and the semi-axis lengths (A, B, C) of the inertia ellipsoid.
    """
    labs, counts = np.unique(labels, return_counts=True)
    ngrp = labs.size

    x, y, z = coords[:,0], coords[:,1], coords[:,2]
    sx = _grouped_sum(x, labels, ngrp)
    sy = _grouped_sum(y, labels, ngrp)
    sz = _grouped_sum(z, labels, ngrp)

    invn = 1.0 / counts
    xcm, ycm, zcm = sx*invn, sy*invn, sz*invn

    x0 = x - xcm[labels]
    y0 = y - ycm[labels]
    z0 = z - zcm[labels]

    r2 = x0*x0 + y0*y0 + z0*z0
    Ixx, Iyy, Izz = r2 - x0*x0, r2 - y0*y0, r2 - z0*z0
    Ixy, Ixz, Iyz = -(x0*y0), -(x0*z0), -(y0*z0)

    Sxx = _grouped_sum(Ixx, labels, ngrp)
    Syy = _grouped_sum(Iyy, labels, ngrp)
    Szz = _grouped_sum(Izz, labels, ngrp)
    Sxy = _grouped_sum(Ixy, labels, ngrp)
    Sxz = _grouped_sum(Ixz, labels, ngrp)
    Syz = _grouped_sum(Iyz, labels, ngrp)

    M = np.zeros((ngrp, 3, 3), float)
    M[:,0,0] = Sxx; M[:,1,1] = Syy; M[:,2,2] = Szz
    M[:,0,1] = Sxy; M[:,1,0] = Sxy
    M[:,0,2] = Sxz; M[:,2,0] = Sxz
    M[:,1,2] = Syz; M[:,2,1] = Syz

    vals = np.linalg.eigvalsh(M)
    vals = np.clip(vals, 0, None)
    A = np.sqrt(vals[:,2]); B = np.sqrt(vals[:,1]); C = np.sqrt(vals[:,0])

    return labs.astype(np.int32), counts.astype(np.int32), xcm, ycm, zcm, A, B, C


def _build_block_rows(ttype, webtype, tids, randiters, isdata,
                      labels, labs, counts, xcm, ycm, zcm, A, B, C, link_len):
    """
    Create structured probability rows for a block.
    
    Args:
        ttype (str): Tracer type.
        webtype (str): Web type.
        tids (np.ndarray): Target IDs.
        randiters (np.ndarray): Random iteration numbers.
        isdata (np.ndarray): Boolean array indicating data points.
        labels (np.ndarray): Cluster labels for each point.
        labs (np.ndarray): Unique cluster labels.
        counts (np.ndarray): Counts of points in each cluster.
        xcm (np.ndarray): X center of mass for each cluster.
        ycm (np.ndarray): Y center of mass for each cluster.
        zcm (np.ndarray): Z center of mass for each cluster.
        A (np.ndarray): Semi-axis A for each cluster.
        B (np.ndarray): Semi-axis B for each cluster.
        C (np.ndarray): Semi-axis C for each cluster.
        link_len (float): Linking length used in clustering.
    Returns:
        np.ndarray: Structured array of group rows.
    """
    n = tids.size
    rows = np.empty(n, dtype=_GROUP_ROW_DTYPE)
    rows['TRACERTYPE'] = _ascii_fill(ttype, n, _GROUP_TRACER_DTYPE)
    rows['TARGETID'] = tids.astype(np.int64, copy=False)
    rows['RANDITER'] = randiters.astype(np.int32, copy=False)
    rows['WEBTYPE'] = _ascii_fill(webtype, n, _GROUP_WEBTYPE_DTYPE)
    rows['GROUPID'] = labels.astype(np.int32, copy=False)

    idx = np.searchsorted(labs, labels)
    rows['NPTS'] = counts[idx].astype(np.int32, copy=False)
    rows['XCM'] = xcm[idx].astype(np.float32, copy=False)
    rows['YCM'] = ycm[idx].astype(np.float32, copy=False)
    rows['ZCM'] = zcm[idx].astype(np.float32, copy=False)
    rows['A'] = A[idx].astype(np.float32, copy=False)
    rows['B'] = B[idx].astype(np.float32, copy=False)
    rows['C'] = C[idx].astype(np.float32, copy=False)
    rows['LINKLEN'] = np.full(n, float(link_len), dtype=np.float32)
    rows['ISDATA'] = isdata.astype(bool, copy=False)
    return rows


def _write_chunked_fits(columns, total_rows, chunk_iter, output_path, meta=None):
    """
    Write a FITS binary table in chunks.
    
    Args:
        columns (list of tuple): List of (name, format) for each column.
        total_rows (int): Total number of rows in the table.
        chunk_iter (iterator): Iterator yielding chunks of data as structured arrays.
        output_path (str): Path to write the FITS file.
        meta (dict, optional): Metadata to add to the FITS header.
    """
    coldefs = fits.ColDefs([fits.Column(name=name, format=fmt) for name, fmt in columns])
    hdu = fits.BinTableHDU.from_columns(coldefs, nrows=int(total_rows))
    if meta:
        for key, value in meta.items():
            try:
                hdu.header[key] = value
            except Exception:
                pass
    hdu.writeto(output_path, overwrite=True)
    with fits.open(output_path, mode='update', memmap=True) as hdul:
        data = hdul[1].data
        start = 0
        for chunk in chunk_iter:
            length = len(chunk)
            if length == 0:
                continue
            end = start + length
            for name, _ in columns:
                data[name][start:end] = chunk[name]
            start = end
        hdul.flush()


def _write_groups_fits(store, out_dir, zone, webtype, out_tag=None, release_tag=None):
    """
    Write the groups stored in `store` to a compressed FITS file.
    
    Args:
        store (TempTableStore): Store containing group rows.
        out_dir (str): Output directory.
        zone (int | str): Zone identifier.
        webtype (str): Cosmic web type.
        out_tag (str, optional): Tag appended to filenames.
        release_tag (str, optional): Release label for metadata.
    Returns:
        str: Path to the compressed FITS file.
    """
    os.makedirs(out_dir, exist_ok=True)
    tsuf = safe_tag(out_tag)
    zone_str = zone_tag(zone)
    base = os.path.join(out_dir, f'zone_{zone_str}{tsuf}_groups_fof_{webtype}.fits')
    tmp_path = base + '.tmp'
    meta = {'ZONE': zone_str, 'RELEASE': str(release_tag) if release_tag is not None else ''}
    chunk_iter = store.iter_arrays(_group_chunk_rows())
    _write_chunked_fits(_GROUP_FITS_COLUMNS, store.total, chunk_iter, tmp_path, meta=meta)

    compressed = base + '.gz'
    tmp_compressed = compressed + '.tmp'
    with open(tmp_path, 'rb') as fi, gzip.open(tmp_compressed, 'wb') as fo:
        shutil.copyfileobj(fi, fo)
    os.remove(tmp_path)
    os.replace(tmp_compressed, compressed)
    return compressed


def process_zone(zone, raw_dir, class_dir, out_dir, webtype, source,
                 r_lower, r_upper, release_tag=None, out_tag=None):
    """
    Generate group catalogues for a zone.

    Args:
        zone (int | str): Zone identifier to process.
        raw_dir (str): Directory containing raw catalogues.
        class_dir (str): Release directory containing classification products.
        out_dir (str): Destination directory for groups.
        webtype (str): Desired cosmic web type (e.g., ``'filament'``).
        source (str): Source selection (``'data'``, ``'rand'``, or ``'both'``).
        r_lower (float): Lower r threshold (negative).
        r_upper (float): Upper r threshold (positive).
        release_tag (str, optional): Release label stored in output metadata.
        out_tag (str, optional): Tag appended to filenames.
    Returns:
        list[str]: Paths to generated groups FITS files. Empty when no objects
        meet the criteria.
    """
    raw_path, class_path = _get_zone_paths(raw_dir, class_dir, zone, out_tag=out_tag)
    raw_tbl, class_tbl = _read_zone_tables(raw_path, class_path)

    if len(class_tbl) == 0:
        return []

    r_num = np.asarray(class_tbl['NDATA'], dtype=np.float64) - np.asarray(class_tbl['NRAND'], dtype=np.float64)
    r_den = np.asarray(class_tbl['NDATA'], dtype=np.float64) + np.asarray(class_tbl['NRAND'], dtype=np.float64)
    r_val = np.full(len(class_tbl), np.nan, dtype=np.float64)
    np.divide(r_num, r_den, out=r_val, where=(r_den > 0))

    valid = np.isfinite(r_val)
    bins = np.array([r_lower, 0.0, r_upper], dtype=float)
    webtypes = np.full(len(class_tbl), '', dtype='U8')
    if np.any(valid):
        idx = np.clip(np.digitize(r_val[valid], bins, right=False), 0, 3)
        webtypes_valid = WEBTYPE_MAPPING[idx]
        webtypes[valid] = webtypes_valid

    isdata_cls = np.asarray(class_tbl['ISDATA'], dtype=bool)
    mask = valid & (webtypes == webtype)
    if source == 'data':
        mask &= isdata_cls
    elif source == 'rand':
        mask &= ~isdata_cls

    if not np.any(mask):
        return []

    class_sel = class_tbl[mask]
    isdata_sel = np.asarray(class_sel['ISDATA'], dtype=bool)
    rand_sel = np.asarray(class_sel['RANDITER'], dtype=np.int32)
    rand_sel = np.where(isdata_sel, -1, rand_sel)
    tid_sel = np.asarray(class_sel['TARGETID'], dtype=np.int64)
    tracer_labels = np.asarray(class_sel['TRACERTYPE']).astype('U32')

    raw_tid = np.asarray(raw_tbl['TARGETID'], dtype=np.int64)
    raw_iter = np.asarray(raw_tbl['RANDITER'], dtype=np.int32)

    if 'TRACER_ID' in raw_tbl.colnames and 'TRACER_ID' in class_sel.colnames:
        raw_tracer_codes = np.asarray(raw_tbl['TRACER_ID'], dtype=np.int32)
        sel_tracer_codes = np.asarray(class_sel['TRACER_ID'], dtype=np.int32)
    else:
        raw_tracer_codes, sel_tracer_codes = _compute_tracer_codes(raw_tbl['TRACERTYPE'],
                                                                   class_sel['TRACERTYPE'])

    key_dtype = np.dtype([('TARGETID', np.int64),
                          ('RANDITER', np.int32),
                          ('TRACER', np.int32)])

    raw_keys = np.empty(raw_tid.size, dtype=key_dtype)
    raw_keys['TARGETID'] = raw_tid
    raw_keys['RANDITER'] = raw_iter
    raw_keys['TRACER'] = raw_tracer_codes

    sel_keys = np.empty(tid_sel.size, dtype=key_dtype)
    sel_keys['TARGETID'] = tid_sel
    sel_keys['RANDITER'] = rand_sel
    sel_keys['TRACER'] = sel_tracer_codes

    sorter = np.argsort(raw_keys, order=('TARGETID', 'RANDITER', 'TRACER'))
    raw_sorted = raw_keys[sorter]

    pos = np.searchsorted(raw_sorted, sel_keys)
    within = pos < raw_sorted.size
    keep = np.zeros(sel_keys.size, dtype=bool)
    if np.any(within):
        matches = raw_sorted[pos[within]] == sel_keys[within]
        if np.any(matches):
            keep_indices = np.where(within)[0][matches]
            keep[keep_indices] = True

    if not np.any(keep):
        return []

    if not np.all(keep):
        tid_sel = tid_sel[keep]
        rand_sel = rand_sel[keep]
        isdata_sel = isdata_sel[keep]
        tracer_labels = tracer_labels[keep]
        sel_keys = sel_keys[keep]

    matched_indices = sorter[pos[keep]]

    x_raw = np.array(raw_tbl['XCART'], copy=False)
    y_raw = np.array(raw_tbl['YCART'], copy=False)
    z_raw = np.array(raw_tbl['ZCART'], copy=False)

    x_sel = x_raw[matched_indices].astype(np.float32, copy=False)
    y_sel = y_raw[matched_indices].astype(np.float32, copy=False)
    z_sel = z_raw[matched_indices].astype(np.float32, copy=False)

    outputs = []
    store = TempTableStore(_GROUP_ROW_DTYPE, prefix='groups', base_dir=_group_spill_dir())

    try:
        for ttype, randit, idxs in _split_blocks(tracer_labels, rand_sel):
            print(f'Processing zone {zone}, TRACERTYPE={ttype}, RANDITER={randit}, NPTS={len(idxs)}')
            idxs = np.asarray(idxs, dtype=np.int64)
            tids_block = tid_sel[idxs]
            rand_block = rand_sel[idxs].astype(np.int32, copy=False)
            isdata_block = isdata_sel[idxs]
            x_block = x_sel[idxs]
            y_block = y_sel[idxs]
            z_block = z_sel[idxs]

            eps = float(length({'XCART': x_block, 'YCART': y_block, 'ZCART': z_block}))
            if not np.isfinite(eps) or eps <= 0.0:
                eps = np.finfo(float).eps

            coords = np.column_stack((x_block, y_block, z_block)).astype(np.float32, copy=False)
            labels = _dbscan_labels(coords, eps)
            labs, counts, xcm, ycm, zcm, A, B, C = _group_inertia(coords, labels)

            rows = _build_block_rows(ttype, webtype,
                                     tids_block, rand_block, isdata_block,
                                     labels, labs, counts, xcm, ycm, zcm, A, B, C,
                                     link_len=eps)
            store.append(rows)

        if store.total:
            outputs.append(_write_groups_fits(store, out_dir, zone, webtype,
                                              out_tag=out_tag, release_tag=release_tag))
    finally:
        store.cleanup()

    return outputs


def _default_zones_for_release(release_tag):
    """
    Get default zones for a given release tag.
    
    Args:
        release_tag (str): Release tag (e.g., 'dr1').
    Returns:
        list[str]: Default zone labels.
    """
    rel = str(release_tag).lower()
    if rel.startswith('dr') or 'ngc' in rel:
        return ['NGC1', 'NGC2']
    return [f"{i:02d}" for i in range(20)]


def parse_args():
    release_default = os.environ.get('RELEASE', 'dr2_res')

    p = argparse.ArgumentParser()
    p.add_argument('--raw-dir', default=os.path.join('/pscratch/sd/v/vtorresg/cosmic-web', release_default, 'raw'),
                   help='Raw data dir')
    p.add_argument('--class-dir', default=os.path.join('/pscratch/sd/v/vtorresg/cosmic-web', release_default),
                   help='Release dir containing classification/probabilities/pairs')
    p.add_argument('--groups-dir', default=os.path.join('/pscratch/sd/v/vtorresg/cosmic-web', release_default, 'groups'),
                   help='Output groups dir')
    p.add_argument('--zones', nargs='+', type=str, default=_default_zones_for_release(release_default),
                   help='Zone numbers or labels (e.g., 00 01 ... or NGC1 NGC2)')
    p.add_argument('--webtype', choices=['void','sheet','filament','knot'], default='void')
    p.add_argument('--source', choices=['data','rand','both'], default='rand')
    p.add_argument('--out-tag', type=str, default=None, help='Tag appended to filenames')
    p.add_argument('--release', default=release_default.upper(), help='Release tag stored in FITS metadata')
    p.add_argument('--r-lower', type=float, default=-0.9, help='Lower r threshold (must be negative)')
    p.add_argument('--r-upper', type=float, default=0.9, help='Upper r threshold (must be positive)')
    return p.parse_args()


def main():
    args = parse_args()
    if args.r_lower >= 0 or args.r_upper <= 0:
        raise ValueError('r-lower must be negative and r-upper must be positive.')
    release_tag = str(args.release).upper()
    init = t.time()
    for z in args.zones:
        outputs = process_zone(z, raw_dir=args.raw_dir, class_dir=args.class_dir,
                               out_dir=args.groups_dir, webtype=args.webtype,
                               source=args.source,
                               r_lower=args.r_lower, r_upper=args.r_upper,
                               release_tag=release_tag,
                               out_tag=args.out_tag)
        if outputs:
            for out in outputs:
                print(f'---- zone {z} done: {out}')
        else:
            print(f'---- zone {z} no objects with WEBTYPE={args.webtype} for "{args.source}".')
    print(f'Elapsed: {(t.time() - init)/60:.2f} min')

if __name__ == '__main__':
    main()