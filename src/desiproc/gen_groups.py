import argparse
import gzip
import os
import shutil
import sys
import time as t
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.table import Table, join, vstack
from scipy.spatial import ConvexHull, QhullError
from sklearn.cluster import DBSCAN

if __package__ is None or __package__ == '':
    pkg_root = Path(__file__).resolve().parent
    if str(pkg_root) not in sys.path:
        sys.path.append(str(pkg_root))
    from paths import locate_classification_file, safe_tag, zone_tag
else:
    from .paths import locate_classification_file, safe_tag, zone_tag


RAW_COLS = ['TRACERTYPE','RANDITER','TARGETID','XCART','YCART','ZCART']
CLASS_COLS = ['TARGETID','RANDITER','ISDATA','NDATA','NRAND','TRACERTYPE']
WEBTYPE_MAPPING = np.array(['void', 'sheet', 'filament', 'knot'], dtype='U8')


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
        subset = {c: np.asarray(data[c]) for c in cols}
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


def _split_blocks(raw_sub):
    """
    Splits the raw data into blocks based on the tracer type and random iteration.

    Args:
        raw_sub (Table): Subset of the raw data table.
    Returns:
        Generator yielding tuples of (tracer type, random iteration, indices).
    """
    tr = np.asarray(raw_sub['TRACERTYPE'], dtype=str)
    ri = np.asarray(raw_sub['RANDITER'])
    keys = np.char.add(np.char.add(tr, '|'), ri.astype(str))
    uniq, inv = np.unique(keys, return_inverse=True)
    for k_idx, key in enumerate(uniq):
        idxs = np.nonzero(inv == k_idx)[0]
        ttype = tr[idxs[0]]
        randit = int(ri[idxs[0]])
        yield ttype, randit, idxs


def length(data_raw):
    """
    Estimate a linking length based on the convex hull of the points.
    
    Args:
        data_raw (Table): Table containing the raw data with 'XCART', 'YCART', 'ZCART' columns.
    Returns:
        float: Estimated linking length, or 0.0 if it cannot be computed.
    """
    coords = np.column_stack([np.asarray(data_raw[col], dtype=float) for col in ('XCART','YCART','ZCART')])
    try:
        hull = ConvexHull(coords)
        vol = float(hull.volume)
    except QhullError:
        diffs = coords[:, None, :] - coords[None, :, :]
        return float(np.max(np.linalg.norm(diffs, axis=-1)))

    if vol <= 0:
        diffs = coords[:, None, :] - coords[None, :, :]
        return float(np.max(np.linalg.norm(diffs, axis=-1)))

    num_galaxies = coords.shape[0]
    return float(np.cbrt(vol / num_galaxies))


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


def _build_block_tables(ttype, randit, webtype, tids, labels, labs,
                       counts, xcm, ycm, zcm, A, B, C, link_len):
    """
    Builds the tables for the groups based on the computed labels and properties.
    
    Args:
        ttype (str): Tracer type.
        randit (int): Random iteration number.
        webtype (str): Type of web structure.
        tids (np.ndarray): Array of target IDs.
        labels (np.ndarray): Array of cluster labels.
        labs (np.ndarray): Unique labels for the groups.
        counts (np.ndarray): Number of points in each group.
        xcm (np.ndarray): X coordinate of the center of mass for each group.
        ycm (np.ndarray): Y coordinate of the center of mass for each group.
        zcm (np.ndarray): Z coordinate of the center of mass for each group.
        A (np.ndarray): Semi-axis length A of the inertia ellipsoid.
        B (np.ndarray): Semi-axis length B of the inertia ellipsoid.
        C (np.ndarray): Semi-axis length C of the inertia ellipsoid.
        link_len (float): Linking length used in the FoF algorithm.
    Returns:
        Table: Astropy Table containing the group properties.
    """
    pts = Table({'TRACERTYPE': np.full(tids.size, ttype),
                 'TARGETID': tids, 'RANDITER': np.full(tids.size, randit, dtype=np.int32),
                 'WEBTYPE': np.full(tids.size, webtype), 'GROUPID': labels})
    
    props = Table({'TRACERTYPE': np.full(labs.size, ttype),
                   'RANDITER': np.full(labs.size, randit, dtype=np.int32),
                   'WEBTYPE': np.full(labs.size, webtype), 'GROUPID': labs,
                   'NPTS': counts, 'XCM': xcm, 'YCM': ycm, 'ZCM': zcm,
                   'A': A, 'B': B, 'C': C,
                   'LINKLEN': np.full(labs.size, float(link_len))})
    return join(pts, props, keys=['TRACERTYPE','RANDITER','WEBTYPE','GROUPID'],
                    join_type='left')


def write_fits_gz(tbl, out_dir, zone, webtype, out_tag=None, release_tag=None):
    """
    Write ``tbl`` to a gzipped FITS file with zone metadata.

    Args:
        tbl (Table): Group catalogue to persist.
        out_dir (str): Destination directory.
        zone (int | str): Zone identifier used in the filename and metadata.
        webtype (str): Cosmic web type for the output file name.
        out_tag (str, optional): Additional tag appended to the filename.
        release_tag (str, optional): Release label stored in the FITS header.
    Returns:
        str: Path to the compressed FITS file.
    """
    os.makedirs(out_dir, exist_ok=True)
    tsuf = safe_tag(out_tag)
    zone_str = zone_tag(zone)
    uncompressed = os.path.join(out_dir, f'zone_{zone_str}{tsuf}_groups_fof_{webtype}.fits')
    compressed = uncompressed + '.gz'
    tmp_compressed = compressed + '.tmp'
    tbl.meta['ZONE'] = zone_str
    tbl.meta['RELEASE'] = str(release_tag) if release_tag is not None else ''
    tbl.write(uncompressed, overwrite=True)
    with open(uncompressed, 'rb') as fi, gzip.open(tmp_compressed, 'wb') as fo:
        shutil.copyfileobj(fi, fo)
    os.remove(uncompressed)
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
    isdata_sel = isdata_cls[mask]
    rand_sel = np.asarray(class_sel['RANDITER'], dtype=np.int32)
    tid_sel = np.asarray(class_sel['TARGETID'], dtype=np.int64)
    tracer_sel = np.array([_normalize_tracer_label(v) for v in class_sel['TRACERTYPE']], dtype='U32')

    raw_tid = np.asarray(raw_tbl['TARGETID'], dtype=np.int64)
    raw_iter = np.asarray(raw_tbl['RANDITER'], dtype=np.int32)
    raw_tracer_base = np.array([_normalize_tracer_label(v) for v in raw_tbl['TRACERTYPE']], dtype='U32')
    coords = np.column_stack([np.asarray(raw_tbl['XCART'], dtype=np.float64),
                             np.asarray(raw_tbl['YCART'], dtype=np.float64),
                             np.asarray(raw_tbl['ZCART'], dtype=np.float64)])

    data_mask = raw_iter == -1
    rand_mask = ~data_mask
    data_idx = np.where(data_mask)[0]
    rand_idx = np.where(rand_mask)[0]

    data_lookup = {(int(raw_tid[i]), raw_tracer_base[i]): i for i in data_idx}
    rand_lookup = {(int(raw_tid[i]), raw_tracer_base[i], int(raw_iter[i])): i for i in rand_idx}

    keep = np.ones(tid_sel.size, dtype=bool)
    coords_sel = np.empty((tid_sel.size, 3), dtype=np.float64)
    for i in range(tid_sel.size):
        if isdata_sel[i]:
            key = (int(tid_sel[i]), tracer_sel[i])
            idx_lookup = data_lookup.get(key)
        else:
            key = (int(tid_sel[i]), tracer_sel[i], int(rand_sel[i]))
            idx_lookup = rand_lookup.get(key)
        if idx_lookup is None:
            keep[i] = False
        else:
            coords_sel[i] = coords[idx_lookup]

    if not np.any(keep):
        return []

    if not np.all(keep):
        tid_sel = tid_sel[keep]
        rand_sel = rand_sel[keep]
        isdata_sel = isdata_sel[keep]
        tracer_sel = tracer_sel[keep]
        coords_sel = coords_sel[keep]

    raw_sub = Table({'TRACERTYPE': tracer_sel.astype('U32'),
                     'TARGETID': tid_sel,
                     'RANDITER': rand_sel,
                     'XCART': coords_sel[:,0],
                     'YCART': coords_sel[:,1],
                     'ZCART': coords_sel[:,2],
                     'WEBTYPE': np.full(tid_sel.size, webtype, dtype='U8'),
                     'ISDATA': isdata_sel})

    block_tables = []

    for ttype, randit, idxs in _split_blocks(raw_sub):
        print(f'Processing zone {zone}, TRACERTYPE={ttype}, RANDITER={randit}, NPTS={len(idxs)}')
        block_data = raw_sub[idxs]
        eps = float(length(block_data))
        if not np.isfinite(eps) or eps <= 0.0:
            eps = np.finfo(float).eps

        coords = np.column_stack([np.asarray(block_data[col], dtype=float)
                                  for col in ('XCART', 'YCART', 'ZCART')])
        labels = _dbscan_labels(coords, eps)  # FoF is the same as DBSCAN with min_samples=1
        labs, counts, xcm, ycm, zcm, A, B, C = _group_inertia(coords, labels)

        tids = np.asarray(block_data['TARGETID'], dtype=np.int64)
        block_tbl = _build_block_tables(ttype, randit, webtype,
                                        tids=tids,
                                        labels=labels, labs=labs, counts=counts,
                                        xcm=xcm, ycm=ycm, zcm=zcm, A=A, B=B, C=C,
                                        link_len=eps)
        if 'ISDATA' in block_data.colnames:
            block_tbl['ISDATA'] = np.asarray(block_data['ISDATA'], dtype=bool)

        block_tables.append(block_tbl)

    outputs = []

    if block_tables:
        merged = vstack(block_tables, metadata_conflicts='silent')
        outputs.append(write_fits_gz(merged, out_dir, zone, webtype,
                                     out_tag=out_tag, release_tag=release_tag))

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