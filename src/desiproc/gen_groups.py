import os, argparse, gzip, shutil, json
import numpy as np
from astropy.table import Table, vstack, join
from sklearn.cluster import DBSCAN
from astropy.io import fits
import time as t


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


def _get_zone_paths(raw_dir, class_dir, zone):
    """
    Get file paths for a given zone number.

    Args:
        raw_dir (str): Directory containing raw data files.
        class_dir (str): Directory containing classification files.
        zone (int): Zone number.
    Returns:
        tuple: Paths to the raw data file and classification file for the zone.
    """
    z2 = f"{zone:02d}"
    return (os.path.join(raw_dir, f"zone_{z2}.fits.gz"),
            os.path.join(class_dir, f"zone_{z2}_class.fits.gz"),)


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


def classify_by_r(klass, r_limit):
    """
    Classify entries in the classification table based on the ratio of real
    data to random data.
    
    Args:
        klass (Table): Classification table with columns 'NDATA' and 'NRAND'.
        r_limit (float): Threshold for classification.
    Returns:
        Table: Updated classification table with a new 'WEBTYPE' column.
    """
    ndata = np.asarray(klass['NDATA'], float)
    nrand = np.asarray(klass['NRAND'], float)
    r = (ndata - nrand) / (ndata + nrand)

    web = np.empty(r.size, dtype='U8')
    web[(r >= -1.0) & (r <= -r_limit)] = 'void'
    web[(r > -r_limit) & (r <= 0.0)] = 'sheet'
    web[(r > 0.0) & (r <= r_limit)] = 'filament'
    web[(r > r_limit) & (r <= 1.0)] = 'knot'
    klass['WEBTYPE'] = web
    return klass


def _split_blocks(raw_sub):
    """
    Splits the raw data into blocks based on the tracer type and random iteration.
    
    Args:
        raw_sub (Table): Subset of the raw table containing tracer data.
    Yields:
        Tuple[str, int, np.ndarray]: Yields tracer type, random iteration, and
        a boolean mask for the subset.
    """
    tr = np.asarray(raw_sub['TRACERTYPE']).astype(str)
    ri = np.asarray(raw_sub['RANDITER'])
    # keys = np.core.defchararray.add(tr, '|') + ri.astype(str)
    keys = np.char.add(np.char.add(tr, '|'), ri.astype(str))
    uniq = np.unique(keys)
    for key in uniq:
        m = (keys == key)
        ttype  = str(np.unique(tr[m])[0])
        randit = int(np.unique(ri[m])[0])
        yield ttype, randit, m


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
    lab = DBSCAN(eps=eps, min_samples=1, metric='euclidean').fit(coords).labels_
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
    vals  = values[order]
    lab_o = labels[order]
    cuts  = np.r_[0, np.cumsum(np.bincount(lab_o, minlength=ngrp))]
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


def write_fits_gz(tbl, out_dir, zone, webtype):
    """
    Writes the table to a FITS file and compresses it with gzip.
    
    Args:
        tbl (Table): Astropy Table to be written.
        out_dir (str): Output directory for the FITS file.
        zone (int): Zone number for naming the file.
        webtype (str): Type of web structure for naming the file.
    Returns:
        str: Path to the compressed FITS file.
    """
    os.makedirs(out_dir, exist_ok=True)
    uncompressed = os.path.join(out_dir, f'zone_{zone:02d}_groups_fof_{webtype}.fits')
    compressed = uncompressed + '.gz'
    tbl.write(uncompressed, overwrite=True)
    with open(uncompressed, 'rb') as fi, gzip.open(compressed, 'wb') as fo:
        shutil.copyfileobj(fi, fo)
    os.remove(uncompressed)
    return compressed


def process_zone(zone, raw_dir, class_dir, out_dir, webtype, source, linklen_map, r_limit):
    """
    Processes a single zone to build groups based on the provided parameters.
    
    Args:
        zone (int): Zone number to process.
        raw_dir (str): Directory containing raw data files.
        class_dir (str): Directory containing classification data files.
        out_dir (str): Directory to save the output groups files.
        webtype (str): Type of web structure to filter.
        source (str): Source of data, either 'data', 'rand', or 'both'.
        linklen_map (dict): Mapping of tracer types to linking lengths.
        r_limit (float): Ratio limit for classification.
    Returns:
        str: Path to the compressed FITS file containing the groups for the zone,
        or None if no valid groups were found.
    """
    raw_path, class_path = _get_zone_paths(raw_dir, class_dir, zone)
    raw_tbl, cls_tbl = _read_zone_tables(raw_path, class_path)
    cls_tbl = classify_by_r(cls_tbl, r_limit)

    ids_web = set(np.asarray(cls_tbl['TARGETID'][cls_tbl['WEBTYPE'] == webtype], dtype=np.int64))
    raw_sub = raw_tbl[np.isin(raw_tbl['TARGETID'], list(ids_web))]

    blocks = []
    tr_all = np.asarray(raw_sub['TRACERTYPE']).astype(str)
    ri_all = np.asarray(raw_sub['RANDITER'])
    tid_all = np.asarray(raw_sub['TARGETID'])
    x = np.asarray(raw_sub['XCART'], float)
    y = np.asarray(raw_sub['YCART'], float)
    z = np.asarray(raw_sub['ZCART'], float)

    for ttype, randit, mask in _split_blocks(raw_sub):
        is_data = (randit == -1)
        if (source == 'data' and not is_data) or (source == 'rand' and is_data):
            continue

        tracer = ttype.split('_', 1)[0]
        eps = float(linklen_map.get(tracer, linklen_map.get('default', 30.0)))

        coords = np.column_stack((x[mask], y[mask], z[mask]))
        labels = _dbscan_labels(coords, eps)# fof same as dbscan with min_samples=1
        labs, counts, xcm, ycm, zcm, A, B, C = _group_inertia(coords, labels)

        block_tbl = _build_block_tables(ttype, randit, webtype, tids=tid_all[mask],
                                       labels=labels,labs=labs, counts=counts,
                                       xcm=xcm, ycm=ycm, zcm=zcm, A=A, B=B, C=C)
        blocks.append(block_tbl)

    if blocks:
        merged = vstack(blocks, metadata_conflicts='silent')
        return write_fits_gz(merged, out_dir, zone, webtype)
    return None


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--raw-dir', default='/pscratch/sd/v/vtorresg/cosmic-web/edr/raw', help='Raw data dir')
    p.add_argument('--class-dir', default='/pscratch/sd/v/vtorresg/cosmic-web/edr/class', help='Classification dir')
    p.add_argument('--groups-dir', default='/pscratch/sd/v/vtorresg/cosmic-web/edr/groups', help='Output groups dir')
    p.add_argument('--zones', type=int, nargs='+', default=list(range(20)))
    p.add_argument('--webtype', choices=['void','sheet','filament','knot'], default='filament')
    p.add_argument('--source', choices=['data','rand','both'], default='data')
    p.add_argument('--r-limit', type=float, default=0.9)
    p.add_argument('--linking', type=str, default='{"BGS_ANY":10,"ELG":10,"LRG":10,"QSO":10,"default":10}')
    return p.parse_args()

def main():
    args = parse_args()
    linklen_map = json.loads(args.linking)
    init = t.time()
    for z in args.zones:
        out = process_zone(z, raw_dir=args.raw_dir, class_dir=args.class_dir,
                           out_dir=args.groups_dir, webtype=args.webtype,
                           source=args.source, linklen_map=linklen_map,
                           r_limit=args.r_limit)
        if out is not None:
            print(f'---- zone {z:02d} done: {out}')
        else:
            print(f'---- zone {z:02d} no objects with WEBTYPE={args.webtype} for "{args.source}".')
    print(f'Elapsed: {(t.time() - init)/60:.2f} min') #~5 min total without enhan

if __name__ == '__main__':
    main()