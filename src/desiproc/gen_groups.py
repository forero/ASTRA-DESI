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
    from paths import locate_probability_file, safe_tag, zone_tag
else:
    from .paths import locate_probability_file, safe_tag, zone_tag


RAW_COLS = ['TRACERTYPE','RANDITER','TARGETID','XCART','YCART','ZCART']
PROB_COLS = ['TARGETID','RANDITER','ISDATA','PVOID','PSHEET','PFILAMENT','PKNOT']


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

    prob_path = locate_probability_file(class_dir, zone, out_tag)
    return raw_path, prob_path


def _read_zone_tables(raw_path, prob_path):
    """
    Read the raw and class tables for a given zone.
    
    Args:
        raw_path (str): Path to the raw data file.
        class_path (str): Path to the classification data file.
    Returns:
        Tuple[Table, Table]: Tuple containing the raw and classification tables.
    """
    raw = _read_fits_columns(raw_path, RAW_COLS)
    prob = _read_fits_columns(prob_path, PROB_COLS)
    return raw, prob


def classify_by_probability(prob_tbl):
    """
    Assign WEBTYPE per row using the maximum of the probability columns.
    
    Args:
        prob_tbl (Table): Table containing probability columns.
    Returns:
        Table: Updated table with a new 'WEBTYPE' column.
    """
    prob_tbl = prob_tbl.copy()

    if 'RANDITER' not in prob_tbl.colnames:
        prob_tbl['RANDITER'] = np.full(len(prob_tbl), -1, dtype=np.int32)
    prob_tbl['RANDITER'] = np.asarray(prob_tbl['RANDITER'], dtype=np.int32)

    if 'ISDATA' not in prob_tbl.colnames:
        prob_tbl['ISDATA'] = (np.asarray(prob_tbl['RANDITER']) == -1)

    if 'TARGETID' not in prob_tbl.colnames:
        raise KeyError('Probability table missing TARGETID column')

    prob_tbl['TARGETID'] = np.asarray(prob_tbl['TARGETID'], dtype=np.int64)

    columns = []
    for name in ('PVOID', 'PSHEET', 'PFILAMENT', 'PKNOT'):
        if name in prob_tbl.colnames:
            columns.append(np.asarray(prob_tbl[name], dtype=float))
        else:
            columns.append(np.zeros(len(prob_tbl), dtype=float))

    arr = np.vstack(columns).T if columns else np.zeros((len(prob_tbl), 4), dtype=float)
    arr = np.nan_to_num(arr, nan=-np.inf)
    idx = np.argmax(arr, axis=1)
    mapping = np.array(['void', 'sheet', 'filament', 'knot'], dtype='U8')
    prob_tbl['WEBTYPE'] = mapping[idx]

    return prob_tbl


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


def lenght(data_raw):
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
                       counts, xcm, ycm, zcm, A, B, C):
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
                   'A': A, 'B': B, 'C': C})
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
                 release_tag=None, out_tag=None):
    """
    Generate group catalogues for a zone.

    Args:
        zone (int | str): Zone identifier to process.
        raw_dir (str): Directory containing raw catalogues.
        class_dir (str): Release directory containing classification products.
        out_dir (str): Destination directory for groups.
        webtype (str): Desired cosmic web type (e.g., ``'filament'``).
        source (str): Source selection (``'data'``, ``'rand'``, or ``'both'``).
        release_tag (str, optional): Release label stored in output metadata.
        out_tag (str, optional): Tag appended to filenames.
    Returns:
        list[str]: Paths to generated groups FITS files. Empty when no objects
        meet the criteria.
    """
    raw_path, prob_path = _get_zone_paths(raw_dir, class_dir, zone, out_tag=out_tag)
    raw_tbl, prob_tbl = _read_zone_tables(raw_path, prob_path)
    prob_tbl = classify_by_probability(prob_tbl)

    if source == 'data':
        prob_tbl = prob_tbl[prob_tbl['ISDATA'] == True]
    elif source == 'rand':
        prob_tbl = prob_tbl[prob_tbl['ISDATA'] == False]

    if len(prob_tbl) == 0:
        return []

    prob_tbl = prob_tbl[prob_tbl['WEBTYPE'] == webtype]
    if len(prob_tbl) == 0:
        return []

    keep_cols = ['TARGETID','RANDITER','WEBTYPE','ISDATA']
    extra_cols = [c for c in prob_tbl.colnames if c in keep_cols]
    prob_keep = prob_tbl[extra_cols]

    joined = join(raw_tbl, prob_keep, keys=['TARGETID','RANDITER'], join_type='inner')
    if len(joined) == 0:
        return []

    if source == 'data':
        joined = joined[joined['ISDATA'] == True]
    elif source == 'rand':
        joined = joined[joined['ISDATA'] == False]

    if len(joined) == 0:
        return []

    raw_sub = joined

    data_blocks = []
    rand_blocks = {}

    for ttype, randit, idxs in _split_blocks(raw_sub):
        is_data = (randit == -1)
        if (source == 'data' and not is_data) or (source == 'rand' and is_data):
            continue

        block_data = raw_sub[idxs]
        eps = float(lenght(block_data))
        if not np.isfinite(eps) or eps < 0.0:
            eps = 0.0

        coords = np.column_stack([
            np.asarray(block_data[col], dtype=float)
            for col in ('XCART', 'YCART', 'ZCART')
        ])
        labels = _dbscan_labels(coords, eps)  # FoF is the same as DBSCAN with min_samples=1
        labs, counts, xcm, ycm, zcm, A, B, C = _group_inertia(coords, labels)

        tids = np.asarray(block_data['TARGETID'], dtype=np.int64)
        block_tbl = _build_block_tables(
            ttype, randit, webtype,
            tids=tids,
            labels=labels, labs=labs, counts=counts,
            xcm=xcm, ycm=ycm, zcm=zcm, A=A, B=B, C=C
        )
        if is_data:
            data_blocks.append(block_tbl)
        else:
            rand_blocks.setdefault(randit, []).append(block_tbl)

    outputs = []

    if data_blocks:
        merged = vstack(data_blocks, metadata_conflicts='silent')
        outputs.append(write_fits_gz(merged, out_dir, zone, webtype,
                                     out_tag=out_tag, release_tag=release_tag))

    if rand_blocks:
        for randit, tables in sorted(rand_blocks.items()):
            merged = vstack(tables, metadata_conflicts='silent')
            merged.meta['RANDITER'] = int(randit)
            if out_tag is None:
                iter_tag = f'{randit:02d}'
            else:
                iter_tag = f'{out_tag}_{randit:02d}'
            outputs.append(write_fits_gz(merged, out_dir, zone, webtype,
                                         out_tag=iter_tag, release_tag=release_tag))

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
    release_default = os.environ.get('RELEASE', 'dr1')

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
    return p.parse_args()

def main():
    args = parse_args()
    release_tag = str(args.release).upper()
    init = t.time()
    for z in args.zones:
        outputs = process_zone(z, raw_dir=args.raw_dir, class_dir=args.class_dir,
                               out_dir=args.groups_dir, webtype=args.webtype,
                               source=args.source,
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