import argparse
import gzip
import json
import os
import shutil
import time as t

import numpy as np
from astropy.io import fits
from astropy.table import Table, join, vstack
from sklearn.cluster import DBSCAN

from .paths import locate_classification_file, safe_tag, zone_tag


RAW_COLS = ['TRACERTYPE','RANDITER','TARGETID','XCART','YCART','ZCART']
CLASS_COLS = ['TARGETID','RANDITER','NDATA','NRAND']


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
    klass = _read_fits_columns(class_path, CLASS_COLS)
    return raw, klass


def classify_by_r(klass, r_lower, r_upper):
    """
    Classify rows according to the ratio of real to random counts.

    Args:
        klass (Table): Classification table with ``NDATA`` and ``NRAND`` columns.
        r_lower (float): Lower threshold (should be negative).
        r_upper (float): Upper threshold (should be positive).
    Returns:
        Table: Classification table with an updated ``WEBTYPE`` column.
    Raises:
        ValueError: If the thresholds do not straddle zero.
    """

    if r_lower >= 0 or r_upper <= 0:
        raise ValueError('r_lower must be negative and r_upper must be positive.')

    ndata = np.asarray(klass['NDATA'], float)
    nrand = np.asarray(klass['NRAND'], float)
    denom = ndata + nrand
    r = np.divide(ndata - nrand, denom, out=np.zeros_like(denom, dtype=float),
                  where=denom > 0,)

    web = np.empty(r.size, dtype='U8')
    web[r <= r_lower] = 'void'
    web[(r > r_lower) & (r < 0.0)] = 'sheet'
    web[(r >= 0.0) & (r < r_upper)] = 'filament'
    web[r >= r_upper] = 'knot'

    klass['WEBTYPE'] = web
    return klass


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


def process_zone(zone, raw_dir, class_dir, out_dir, webtype, source, linklen_map,
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
        linklen_map (dict): Mapping of tracer to linking length.
        r_lower (float): Lower threshold applied when classifying by ``r``.
        r_upper (float): Upper threshold applied when classifying by ``r``.
        release_tag (str, optional): Release label stored in output metadata.
        out_tag (str, optional): Tag appended to filenames.
    Returns:
        str | None: Path to the generated groups FITS file, or ``None`` when no
        objects meet the criteria.
    """
    raw_path, class_path = _get_zone_paths(raw_dir, class_dir, zone, out_tag=out_tag)
    raw_tbl, cls_tbl = _read_zone_tables(raw_path, class_path)
    cls_tbl = classify_by_r(cls_tbl, r_lower, r_upper)

    ids_web = np.asarray(cls_tbl['TARGETID'][cls_tbl['WEBTYPE'] == webtype], dtype=np.int64)
    if ids_web.size == 0:
        return None
    mask_web = np.isin(raw_tbl['TARGETID'], ids_web, assume_unique=False)
    raw_sub = raw_tbl[mask_web]
    if len(raw_sub) == 0:
        return None

    tid_all = np.asarray(raw_sub['TARGETID'], dtype=np.int64)
    x = np.asarray(raw_sub['XCART'], dtype=float)
    y = np.asarray(raw_sub['YCART'], dtype=float)
    z = np.asarray(raw_sub['ZCART'], dtype=float)

    blocks = []

    for ttype, randit, idxs in _split_blocks(raw_sub):
        is_data = (randit == -1)
        if (source == 'data' and not is_data) or (source == 'rand' and is_data):
            continue

        tracer = ttype.split('_', 1)[0]
        eps = float(linklen_map.get(tracer, linklen_map.get('default', 30.0)))

        coords = np.column_stack((x[idxs], y[idxs], z[idxs]))
        labels = _dbscan_labels(coords, eps)  # FoF is the same as DBSCAN with min_samples=1
        labs, counts, xcm, ycm, zcm, A, B, C = _group_inertia(coords, labels)

        block_tbl = _build_block_tables(
            ttype, randit, webtype,
            tids=tid_all[idxs],
            labels=labels, labs=labs, counts=counts,
            xcm=xcm, ycm=ycm, zcm=zcm, A=A, B=B, C=C
        )
        blocks.append(block_tbl)

    if blocks:
        merged = vstack(blocks, metadata_conflicts='silent')
        return write_fits_gz(merged, out_dir, zone, webtype,
                             out_tag=out_tag, release_tag=release_tag)
    return None


def parse_args():
    release_default = os.environ.get('RELEASE', 'edr')

    p = argparse.ArgumentParser()
    p.add_argument('--raw-dir', default=os.path.join('/pscratch/sd/v/vtorresg/cosmic-web', release_default, 'raw'), help='Raw data dir')
    p.add_argument('--class-dir', default=os.path.join('/pscratch/sd/v/vtorresg/cosmic-web', release_default), help='Release dir containing classification/probabilities/pairs')
    p.add_argument('--groups-dir', default=os.path.join('/pscratch/sd/v/vtorresg/cosmic-web', release_default, 'groups'), help='Output groups dir')
    p.add_argument('--zones', nargs='+', type=str, default=[f"{i:02d}" for i in range(20)], help='Zone numbers or labels (e.g., 00 01 ... or NGC1 NGC2)')
    p.add_argument('--webtype', choices=['void','sheet','filament','knot'], default='filament')
    p.add_argument('--source', choices=['data','rand','both'], default='data')
    p.add_argument('--r-lower', type=float, default=-0.9,
                   help='Lower r threshold used to classify web types (default: -0.9)')
    p.add_argument('--r-upper', type=float, default=0.9,
                   help='Upper r threshold used to classify web types (default: 0.9)')
    p.add_argument('--r-limit', type=float, default=None,
                   help='[Deprecated] Symmetric absolute threshold; overrides --r-lower/--r-upper when set')
    p.add_argument('--out-tag', type=str, default=None, help='Tag appended to filenames')
    p.add_argument('--linking', type=str, default='{"BGS_ANY":10,"ELG":10,"LRG":10,"QSO":10,"default":10}')
    p.add_argument('--release', default=release_default.upper(), help='Release tag stored in FITS metadata')
    return p.parse_args()

def main():
    args = parse_args()
    linklen_map = json.loads(args.linking)
    if args.r_limit is not None:
        sym = float(abs(args.r_limit))
        args.r_lower = -sym
        args.r_upper = sym
    if args.r_lower >= 0 or args.r_upper <= 0:
        raise ValueError('r_lower must be negative and r_upper must be positive.')
    release_tag = str(args.release).upper()
    init = t.time()
    for z in args.zones:
        out = process_zone(z, raw_dir=args.raw_dir, class_dir=args.class_dir,
                           out_dir=args.groups_dir, webtype=args.webtype,
                           source=args.source, linklen_map=linklen_map,
                           r_lower=args.r_lower, r_upper=args.r_upper,
                           release_tag=release_tag,
                           out_tag=args.out_tag)
        if out is not None:
            print(f'---- zone {z} done: {out}')
        else:
            print(f'---- zone {z} no objects with WEBTYPE={args.webtype} for "{args.source}".')
    print(f'Elapsed: {(t.time() - init)/60:.2f} min')

if __name__ == '__main__':
    main()