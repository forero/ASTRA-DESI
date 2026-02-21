"""
gen_watershed.py — Watershed void finder for ASTRA-DESI.

Identifies connected underdense regions in ASTRA classification catalogues
using a graph-based watershed algorithm on the Delaunay pair graph.

Points are sorted by R = (NDATA - NRAND) / (NDATA + NRAND) from most
underdense (R → -1) to least, then assigned to existing connected groups
or used to seed new groups.  Only points with R ≤ --r-threshold are
considered.

Output files follow the naming convention:
    zone_{zone}_groups_watershed.fits.gz
"""

import argparse
import gzip
import os
import shutil
import sys
import time as t
from collections import defaultdict
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.table import Table
from scipy.spatial import Delaunay

if __package__ is None or __package__ == '':
    pkg_root = Path(__file__).resolve().parent
    if str(pkg_root) not in sys.path:
        sys.path.append(str(pkg_root))
    parent_root = pkg_root.parent
    if str(parent_root) not in sys.path:
        sys.path.append(str(parent_root))
    from paths import locate_classification_file, safe_tag, zone_tag
else:
    from .paths import locate_classification_file, safe_tag, zone_tag


# --------------------------------------------------------------------------- #
#  Output schema
# --------------------------------------------------------------------------- #

_WS_ROW_DTYPE = np.dtype([
    ('TRACERTYPE', 'S32'),
    ('TARGETID',   np.int64),
    ('RANDITER',   np.int32),
    ('WEBTYPE',    'S8'),
    ('GROUPID',    np.int32),
    ('NPTS',       np.int32),
    ('XCM',        np.float32),
    ('YCM',        np.float32),
    ('ZCM',        np.float32),
    ('R_EFF',      np.float32),
    ('R_MEAN',     np.float32),
    ('R_MIN',      np.float32),
    ('R_MAX',      np.float32),
    ('ISDATA',     np.bool_),
])

_WS_FITS_COLUMNS = (
    ('TRACERTYPE', '32A'),
    ('TARGETID',   'K'),
    ('RANDITER',   'J'),
    ('WEBTYPE',    '8A'),
    ('GROUPID',    'J'),
    ('NPTS',       'J'),
    ('XCM',        'E'),
    ('YCM',        'E'),
    ('ZCM',        'E'),
    ('R_EFF',      'E'),
    ('R_MEAN',     'E'),
    ('R_MIN',      'E'),
    ('R_MAX',      'E'),
    ('ISDATA',     'L'),
)

RAW_COLS   = ['TRACERTYPE', 'RANDITER', 'TARGETID', 'XCART', 'YCART', 'ZCART']
CLASS_COLS = ['TARGETID', 'RANDITER', 'ISDATA', 'NDATA', 'NRAND', 'TRACERTYPE']
PAIRS_COLS = ['TARGETID1', 'TARGETID2', 'RANDITER']


# --------------------------------------------------------------------------- #
#  I/O helpers
# --------------------------------------------------------------------------- #

def _read_fits_columns(path, cols):
    """Read specific columns from a FITS file, skipping missing ones."""
    with fits.open(path, memmap=True) as hdul:
        data = hdul[1].data
        available = set(data.columns.names)
        subset = {name: np.array(data[name], copy=False)
                  for name in cols if name in available}
    if not subset:
        raise KeyError(f'None of {cols} found in {path}')
    return Table(subset, copy=False)


def _find_raw_path(raw_dir, zone, out_tag=None):
    ztag = zone_tag(zone)
    tsuf = safe_tag(out_tag)
    base = os.path.join(raw_dir, f'zone_{ztag}{tsuf}')
    for ext in ('.fits.gz', '.fits'):
        p = base + ext
        if os.path.exists(p):
            return p
    raise FileNotFoundError(
        f'Raw table not found for zone {zone}: {base}[.fits.gz|.fits]'
    )


def _find_pairs_path(pairs_dir, zone, out_tag=None):
    ztag = zone_tag(zone)
    tsuf = safe_tag(out_tag)
    base = os.path.join(pairs_dir, f'zone_{ztag}{tsuf}_pairs')
    for ext in ('.fits.gz', '.fits'):
        p = base + ext
        if os.path.exists(p):
            return p
    raise FileNotFoundError(
        f'Pairs file not found for zone {zone}: {base}[.fits.gz|.fits]'
    )


# --------------------------------------------------------------------------- #
#  Tracer normalisation
# --------------------------------------------------------------------------- #

def _strip_data_rand_suffix(arr):
    """Remove _DATA or _RAND suffixes from tracer label arrays."""
    arr = np.asarray(arr).astype('U32')
    head, sep, tail = np.char.rpartition(arr, '_')
    mask = (sep != '') & np.isin(np.char.upper(tail), ('DATA', 'RAND'))
    result = arr.copy()
    result[mask] = head[mask]
    return result


# --------------------------------------------------------------------------- #
#  Delaunay pair recomputation
# --------------------------------------------------------------------------- #

def _compute_delaunay_pairs(raw_tbl, rand_iter, verbose=False):
    """
    Recompute Delaunay neighbour pairs for one random iteration.

    Combines data points (RANDITER == -1) with the requested random iteration
    (RANDITER == rand_iter) and builds a 3-D Delaunay triangulation, then
    extracts all unique edges from the resulting tetrahedra.

    Returns
    -------
    np.ndarray, shape (N, 2), dtype int64
        Columns are TARGETID1, TARGETID2.
    """
    ri = np.asarray(raw_tbl['RANDITER'], dtype=np.int32)
    mask = (ri == -1) | (ri == rand_iter)
    sub = raw_tbl[mask]
    if len(sub) < 4:
        return np.empty((0, 2), dtype=np.int64)

    points = np.column_stack([
        np.asarray(sub['XCART']),
        np.asarray(sub['YCART']),
        np.asarray(sub['ZCART']),
    ])
    tids = np.asarray(sub['TARGETID'], dtype=np.int64)

    if verbose:
        print(f'  Recomputing Delaunay pairs for RANDITER={rand_iter} '
              f'({len(sub)} points)…')

    tri = Delaunay(points)
    pairs_set = set()
    for simplex in tri.simplices:
        for i in range(4):
            for j in range(i + 1, 4):
                a, b = int(simplex[i]), int(simplex[j])
                if a > b:
                    a, b = b, a
                pairs_set.add((a, b))

    if not pairs_set:
        return np.empty((0, 2), dtype=np.int64)

    idx_pairs = np.array(list(pairs_set), dtype=np.int64)
    return np.column_stack([tids[idx_pairs[:, 0]], tids[idx_pairs[:, 1]]])


# --------------------------------------------------------------------------- #
#  Watershed algorithm
# --------------------------------------------------------------------------- #

def _build_adjacency(tid1_arr, tid2_arr):
    """Build an {int → set} adjacency map from pair arrays."""
    graph = defaultdict(set)
    for a, b in zip(tid1_arr.tolist(), tid2_arr.tolist()):
        graph[a].add(b)
        graph[b].add(a)
    return graph


def _watershed_grouping(graph, sel_tid, sel_r):
    """
    Assign group IDs via the watershed algorithm.

    Processes points in ascending R order (most underdense first).  Each point
    is assigned to the first existing group it is connected to, or starts a new
    group if it has no such neighbour.

    Parameters
    ----------
    graph : dict {int: set}
        Adjacency map covering at least the nodes in ``sel_tid``.
    sel_tid : np.ndarray, int64
        Target IDs of the points to group.
    sel_r : np.ndarray, float64
        R values corresponding to ``sel_tid`` (same order).

    Returns
    -------
    np.ndarray, int32
        Group ID for each point in ``sel_tid`` (-1 if unassigned, but in
        practice every point receives a group).
    """
    order = np.argsort(sel_r, kind='stable')
    sorted_tid = sel_tid[order]

    group_of = {}            # targetid → group_id
    group_members = defaultdict(set)   # group_id → set of targetids
    current_max = -1

    for tid in sorted_tid.tolist():
        neighbours = graph.get(tid, set())
        assigned = False
        for gid in range(current_max + 1):
            if neighbours & group_members[gid]:
                group_of[tid] = gid
                group_members[gid].add(tid)
                assigned = True
                break
        if not assigned:
            current_max += 1
            group_of[tid] = current_max
            group_members[current_max].add(tid)

    groupids = np.array([group_of[int(tid)] for tid in sel_tid], dtype=np.int32)
    return groupids


# --------------------------------------------------------------------------- #
#  Per-block processing
# --------------------------------------------------------------------------- #

def _process_block(ttype, rand_iter, raw_block, class_block, pairs_block,
                   r_threshold, min_group_size, verbose):
    """
    Run the watershed algorithm for one (tracer, rand_iter) block.

    Returns a structured array of ``_WS_ROW_DTYPE`` rows, or ``None`` if no
    groups meet the size criterion.
    """
    # ---- compute R values ------------------------------------------------- #
    ndata = np.asarray(class_block['NDATA'], dtype=np.float64)
    nrand = np.asarray(class_block['NRAND'], dtype=np.float64)
    denom = ndata + nrand
    r_val = np.full(len(class_block), np.nan)
    valid = denom > 0
    r_val[valid] = (ndata[valid] - nrand[valid]) / denom[valid]

    sel_mask = np.isfinite(r_val) & (r_val <= r_threshold)
    if not np.any(sel_mask):
        return None

    cls_tid    = np.asarray(class_block['TARGETID'], dtype=np.int64)
    cls_isdata = np.asarray(class_block['ISDATA'],   dtype=bool)

    sel_tid    = cls_tid[sel_mask]
    sel_r      = r_val[sel_mask]
    sel_isdata = cls_isdata[sel_mask]

    # ---- build adjacency graph from pairs --------------------------------- #
    if len(pairs_block) == 0:
        return None
    ptid1 = np.asarray(pairs_block['TARGETID1'], dtype=np.int64)
    ptid2 = np.asarray(pairs_block['TARGETID2'], dtype=np.int64)
    graph = _build_adjacency(ptid1, ptid2)

    # ---- run watershed ---------------------------------------------------- #
    groupids = _watershed_grouping(graph, sel_tid, sel_r)

    # ---- filter by minimum group size ------------------------------------- #
    unique_grps, inverse, counts = np.unique(
        groupids, return_inverse=True, return_counts=True
    )
    size_ok   = counts >= min_group_size
    keep_mask = size_ok[inverse]

    if not np.any(keep_mask):
        return None

    sel_tid    = sel_tid[keep_mask]
    sel_r      = sel_r[keep_mask]
    sel_isdata = sel_isdata[keep_mask]
    groupids   = groupids[keep_mask]

    unique_grps, inverse, counts = np.unique(
        groupids, return_inverse=True, return_counts=True
    )

    # ---- map 3-D positions from raw_block --------------------------------- #
    raw_tid = np.asarray(raw_block['TARGETID'], dtype=np.int64)
    sorter  = np.argsort(raw_tid)
    sorted_raw = raw_tid[sorter]

    pos      = np.searchsorted(sorted_raw, sel_tid)
    in_range = pos < sorted_raw.size
    matched  = np.zeros(len(sel_tid), dtype=bool)
    matched[in_range] = (sorted_raw[pos[in_range]] == sel_tid[in_range])

    if not np.any(matched):
        return None

    raw_idx = sorter[pos[matched]]
    x_pts = np.asarray(raw_block['XCART'])[raw_idx].astype(np.float64)
    y_pts = np.asarray(raw_block['YCART'])[raw_idx].astype(np.float64)
    z_pts = np.asarray(raw_block['ZCART'])[raw_idx].astype(np.float64)

    sel_tid    = sel_tid[matched]
    sel_r      = sel_r[matched]
    sel_isdata = sel_isdata[matched]
    groupids   = groupids[matched]

    # recompute after matched-point filter
    unique_grps, inverse, counts = np.unique(
        groupids, return_inverse=True, return_counts=True
    )

    # drop groups that shrank below min_group_size after position matching
    size_ok   = counts >= min_group_size
    keep_mask = size_ok[inverse]
    if not np.any(keep_mask):
        return None

    sel_tid    = sel_tid[keep_mask]
    sel_r      = sel_r[keep_mask]
    sel_isdata = sel_isdata[keep_mask]
    groupids   = groupids[keep_mask]
    x_pts      = x_pts[keep_mask]
    y_pts      = y_pts[keep_mask]
    z_pts      = z_pts[keep_mask]

    unique_grps, inverse, counts = np.unique(
        groupids, return_inverse=True, return_counts=True
    )
    ngrp = len(unique_grps)

    # ---- compute per-group statistics ------------------------------------- #
    xcm   = np.zeros(ngrp, dtype=np.float64)
    ycm   = np.zeros(ngrp, dtype=np.float64)
    zcm   = np.zeros(ngrp, dtype=np.float64)
    r_eff  = np.zeros(ngrp, dtype=np.float64)
    r_mean = np.zeros(ngrp, dtype=np.float64)
    r_min  = np.zeros(ngrp, dtype=np.float64)
    r_max  = np.zeros(ngrp, dtype=np.float64)

    for gi in range(ngrp):
        mg      = inverse == gi
        xcm[gi] = x_pts[mg].mean()
        ycm[gi] = y_pts[mg].mean()
        zcm[gi] = z_pts[mg].mean()
        dx = x_pts[mg] - xcm[gi]
        dy = y_pts[mg] - ycm[gi]
        dz = z_pts[mg] - zcm[gi]
        r_eff[gi]  = np.sqrt((dx**2 + dy**2 + dz**2).mean())
        rv = sel_r[mg]
        r_mean[gi] = rv.mean()
        r_min[gi]  = rv.min()
        r_max[gi]  = rv.max()

    # renumber groups 0..ngrp-1
    new_ids  = np.arange(ngrp, dtype=np.int32)
    groupids = new_ids[inverse]

    if verbose:
        n_members = len(sel_tid)
        print(f'    {ttype} RANDITER={rand_iter}: {ngrp} groups, '
              f'{n_members} members')

    # ---- pack output rows ------------------------------------------------- #
    n = len(sel_tid)
    rows = np.empty(n, dtype=_WS_ROW_DTYPE)
    # Data points use RANDITER=-1 (project convention); randoms keep rand_iter.
    randiter_vals = np.where(sel_isdata, -1, rand_iter).astype(np.int32)

    rows['TRACERTYPE'] = str(ttype).encode('ascii', errors='ignore')[:32]
    rows['TARGETID']   = sel_tid.astype(np.int64,   copy=False)
    rows['RANDITER']   = randiter_vals
    rows['WEBTYPE']    = b'void'
    rows['GROUPID']    = groupids
    rows['NPTS']       = counts[inverse].astype(np.int32, copy=False)
    rows['XCM']        = xcm[inverse].astype(np.float32, copy=False)
    rows['YCM']        = ycm[inverse].astype(np.float32, copy=False)
    rows['ZCM']        = zcm[inverse].astype(np.float32, copy=False)
    rows['R_EFF']      = r_eff[inverse].astype(np.float32, copy=False)
    rows['R_MEAN']     = r_mean[inverse].astype(np.float32, copy=False)
    rows['R_MIN']      = r_min[inverse].astype(np.float32, copy=False)
    rows['R_MAX']      = r_max[inverse].astype(np.float32, copy=False)
    rows['ISDATA']     = sel_isdata.astype(bool, copy=False)
    return rows


# --------------------------------------------------------------------------- #
#  FITS output
# --------------------------------------------------------------------------- #

def _write_watershed_fits(rows_list, out_dir, zone, out_tag=None,
                          release_tag=None, r_threshold=None):
    """
    Concatenate row blocks and write a compressed FITS file.

    Returns
    -------
    str
        Path to the written ``.fits.gz`` file.
    """
    os.makedirs(out_dir, exist_ok=True)
    all_rows = np.concatenate(rows_list)
    total    = len(all_rows)

    ztag = zone_tag(zone)
    tsuf = safe_tag(out_tag)
    base = os.path.join(out_dir, f'zone_{ztag}{tsuf}_groups_watershed.fits')
    tmp_path = base + '.tmp'

    coldefs = fits.ColDefs([
        fits.Column(name=name, format=fmt) for name, fmt in _WS_FITS_COLUMNS
    ])
    hdu = fits.BinTableHDU.from_columns(coldefs, nrows=total)
    hdu.header['ZONE']    = ztag
    hdu.header['RELEASE'] = str(release_tag) if release_tag else ''
    if r_threshold is not None:
        hdu.header['RTHRESH'] = float(r_threshold)

    hdu.writeto(tmp_path, overwrite=True)
    with fits.open(tmp_path, mode='update', memmap=True) as hdul:
        data = hdul[1].data
        for name, _ in _WS_FITS_COLUMNS:
            data[name][:] = all_rows[name]
        hdul.flush()

    compressed     = base + '.gz'
    tmp_compressed = compressed + '.tmp'
    with open(tmp_path, 'rb') as fi, gzip.open(tmp_compressed, 'wb') as fo:
        shutil.copyfileobj(fi, fo)
    os.remove(tmp_path)
    os.replace(tmp_compressed, compressed)
    return compressed


# --------------------------------------------------------------------------- #
#  Public zone-processing function
# --------------------------------------------------------------------------- #

def process_zone(zone, raw_dir, class_dir, pairs_dir, out_dir,
                 r_threshold=-0.7, min_group_size=4,
                 rand_iters=None, tracer=None,
                 recompute_pairs=False,
                 release_tag=None, out_tag=None, verbose=False):
    """
    Run the watershed void finder for a single zone.

    Parameters
    ----------
    zone : int or str
        Zone identifier (e.g., 1, "01", "NGC1").
    raw_dir : str
        Directory containing raw zone FITS files.
    class_dir : str
        Release root containing classification products.
    pairs_dir : str or None
        Directory containing pairs FITS files.  Required unless
        ``recompute_pairs`` is True.
    out_dir : str
        Output directory for watershed group files.
    r_threshold : float
        Maximum R value to include (must be negative, default -0.7).
    min_group_size : int
        Minimum group membership to retain (default 4).
    rand_iters : list[int] or None
        Random iterations to process.  None processes all found in the raw
        table.
    tracer : str or None
        Restrict to this tracer prefix (e.g., ``'BGS_ANY'``).
    recompute_pairs : bool
        If True, rebuild Delaunay pairs from raw positions instead of loading
        from ``pairs_dir``.
    release_tag : str or None
        Release label stored in output FITS metadata.
    out_tag : str or None
        Tag appended to input/output filenames.
    verbose : bool
        Print progress messages.

    Returns
    -------
    str or None
        Path to the written ``.fits.gz`` file, or None if no groups were
        found.
    """
    # ---- load tables ------------------------------------------------------ #
    raw_path = _find_raw_path(raw_dir, zone, out_tag)
    if verbose:
        print(f'Zone {zone}: loading raw  → {raw_path}')
    raw_tbl = _read_fits_columns(raw_path, RAW_COLS)

    class_path = locate_classification_file(class_dir, zone, out_tag)
    if verbose:
        print(f'Zone {zone}: loading classification → {class_path}')
    class_tbl = _read_fits_columns(class_path, CLASS_COLS)

    if len(class_tbl) == 0:
        if verbose:
            print(f'Zone {zone}: empty classification table — skipping.')
        return None

    # ---- optional tracer filter ------------------------------------------- #
    raw_tracer_base = _strip_data_rand_suffix(raw_tbl['TRACERTYPE'])
    cls_tracer_base = _strip_data_rand_suffix(class_tbl['TRACERTYPE'])

    if tracer is not None:
        up = tracer.upper()
        raw_tbl   = raw_tbl[np.char.upper(raw_tracer_base) == up]
        class_tbl = class_tbl[np.char.upper(cls_tracer_base) == up]
        if verbose:
            print(f'Zone {zone}: filtered to tracer={tracer} → '
                  f'{len(raw_tbl)} raw / {len(class_tbl)} class rows')

    if len(class_tbl) == 0:
        if verbose:
            print(f'Zone {zone}: no rows after tracer filter — skipping.')
        return None

    # ---- determine which rand_iters to process ---------------------------- #
    available_iters = np.unique(
        np.asarray(raw_tbl['RANDITER'], dtype=np.int32)
    )
    # exclude data rows (RANDITER == -1) from top-level iteration;
    # they are included inside _compute_delaunay_pairs when needed
    available_iters = available_iters[available_iters >= 0]

    if rand_iters is not None:
        available_iters = np.intersect1d(
            available_iters, np.asarray(rand_iters, dtype=np.int32)
        )

    if len(available_iters) == 0:
        if verbose:
            print(f'Zone {zone}: no matching RANDITER values — skipping.')
        return None

    if verbose:
        print(f'Zone {zone}: processing RANDITER values {available_iters.tolist()}')

    # ---- load pairs once (unless recomputing per-iter) -------------------- #
    pairs_tbl_full = None
    pairs_has_randiter = False
    if not recompute_pairs:
        pairs_path = _find_pairs_path(pairs_dir, zone, out_tag)
        if verbose:
            print(f'Zone {zone}: loading pairs → {pairs_path}')
        pairs_tbl_full = _read_fits_columns(pairs_path, PAIRS_COLS)
        pairs_has_randiter = 'RANDITER' in pairs_tbl_full.colnames

    # ---- iterate over rand_iters and tracers ------------------------------ #
    all_rows = []

    for rand_iter in available_iters.tolist():
        if verbose:
            print(f'  RANDITER={rand_iter}')

        # filter raw and class to this iteration
        raw_ri  = np.asarray(raw_tbl['RANDITER'], dtype=np.int32)
        cls_ri  = np.asarray(class_tbl['RANDITER'], dtype=np.int32)
        # raw_block: data rows (RANDITER=-1) + this random iter
        raw_block  = raw_tbl[(raw_ri == -1) | (raw_ri == rand_iter)]
        class_iter = class_tbl[cls_ri == rand_iter]

        if len(class_iter) == 0:
            continue

        # get or recompute pairs for this iteration
        if recompute_pairs:
            tid_pairs = _compute_delaunay_pairs(raw_tbl, rand_iter, verbose=verbose)
            if len(tid_pairs) == 0:
                continue
            pairs_block = Table({
                'TARGETID1': tid_pairs[:, 0],
                'TARGETID2': tid_pairs[:, 1],
            })
        else:
            if pairs_has_randiter:
                pmask = (
                    np.asarray(pairs_tbl_full['RANDITER'], dtype=np.int32) == rand_iter
                )
                pairs_block = pairs_tbl_full[pmask]
            else:
                pairs_block = pairs_tbl_full

        if len(pairs_block) == 0:
            if verbose:
                print(f'    No pairs for RANDITER={rand_iter} — skipping.')
            continue

        # iterate per tracer within this rand_iter
        cls_tracers    = _strip_data_rand_suffix(class_iter['TRACERTYPE'])
        unique_tracers = np.unique(cls_tracers)

        for ttype in unique_tracers.tolist():
            class_block = class_iter[cls_tracers == ttype]
            rows = _process_block(
                ttype, rand_iter,
                raw_block, class_block, pairs_block,
                r_threshold, min_group_size, verbose,
            )
            if rows is not None and len(rows) > 0:
                all_rows.append(rows)

    if not all_rows:
        if verbose:
            print(f'Zone {zone}: no watershed groups found.')
        return None

    out_path = _write_watershed_fits(
        all_rows, out_dir, zone,
        out_tag=out_tag,
        release_tag=release_tag,
        r_threshold=r_threshold,
    )
    if verbose:
        print(f'Zone {zone}: wrote {out_path}')
    return out_path


# --------------------------------------------------------------------------- #
#  Plotting
# --------------------------------------------------------------------------- #

_PLOT_RAW_COLS = ['TARGETID', 'RANDITER', 'RA', 'Z', 'TRACERTYPE']
_PLOT_GRP_COLS = ['TARGETID', 'RANDITER', 'TRACERTYPE', 'GROUPID', 'NPTS']


def _find_watershed_path(out_dir, zone, out_tag=None):
    """Return the path to an existing watershed groups file."""
    ztag = zone_tag(zone)
    tsuf = safe_tag(out_tag)
    path = os.path.join(out_dir, f'zone_{ztag}{tsuf}_groups_watershed.fits.gz')
    if not os.path.exists(path):
        raise FileNotFoundError(f'Watershed groups file not found: {path}')
    return path


def plot_watershed_zone(zone, raw_dir, out_dir, out_png,
                        out_tag=None, rand_iter=0, min_npts=4,
                        max_z=None, tracers=None, verbose=False):
    """
    Generate a wedge plot of watershed void groups for one zone.

    Joins the watershed groups file with the raw catalogue to obtain RA and Z
    coordinates for each member point, then calls ``plot_wedges``.

    Parameters
    ----------
    zone : int or str
        Zone identifier.
    raw_dir : str
        Directory containing raw zone FITS files (must include RA and Z).
    out_dir : str
        Directory containing the watershed groups file.
    out_png : str
        Output PNG path.
    out_tag : str or None
        Tag used when generating the files.
    rand_iter : int
        Random iteration to show (default 0).  Data points (RANDITER=-1) are
        always included.
    min_npts : int
        Minimum group size to display (default 4).
    max_z : float or None
        Maximum redshift to include in the plot.
    tracers : list[str] or None
        Tracer prefixes to plot (e.g. ['BGS']).  None plots all found.
    verbose : bool
        Print progress messages.
    """
    # lazy import so matplotlib is only loaded when --plot is used
    import sys
    from pathlib import Path as _Path

    _src = str(_Path(__file__).resolve().parents[1])
    if _src not in sys.path:
        sys.path.insert(0, _src)

    from plot.plot_wedges import plot_wedges, tracer_prefixes, ORDERED_TRACERS

    # ---- load raw (need RA, Z) -------------------------------------------- #
    raw_path = _find_raw_path(raw_dir, zone, out_tag)
    if verbose:
        print(f'Plot zone {zone}: loading raw → {raw_path}')
    raw_tbl = _read_fits_columns(raw_path, _PLOT_RAW_COLS)

    # keep data rows and the requested rand_iter
    raw_ri = np.asarray(raw_tbl['RANDITER'], dtype=np.int32)
    raw_mask = (raw_ri == -1) | (raw_ri == rand_iter)
    raw_tbl = raw_tbl[raw_mask]

    # ---- load watershed groups -------------------------------------------- #
    grp_path = _find_watershed_path(out_dir, zone, out_tag)
    if verbose:
        print(f'Plot zone {zone}: loading groups → {grp_path}')
    grp_tbl = _read_fits_columns(grp_path, _PLOT_GRP_COLS)

    # keep the same rand_iter selection
    grp_ri = np.asarray(grp_tbl['RANDITER'], dtype=np.int32)
    grp_mask = (grp_ri == -1) | (grp_ri == rand_iter)
    grp_tbl = grp_tbl[grp_mask]

    if len(grp_tbl) == 0:
        if verbose:
            print(f'Plot zone {zone}: no groups for RANDITER={rand_iter} — skipping.')
        return None

    # ---- join on TARGETID + RANDITER -------------------------------------- #
    from astropy.table import join as _join
    joined = _join(grp_tbl, raw_tbl, keys=['TARGETID', 'RANDITER'], join_type='inner')

    if len(joined) == 0:
        if verbose:
            print(f'Plot zone {zone}: join returned no rows — skipping.')
        return None

    if verbose:
        print(f'Plot zone {zone}: {len(joined)} points after join')

    # ---- resolve tracers -------------------------------------------------- #
    tr_types  = np.asarray(joined['TRACERTYPE_1'
                                  if 'TRACERTYPE_1' in joined.colnames
                                  else 'TRACERTYPE']).astype(str)
    # expose as plain TRACERTYPE for plot_wedges
    if 'TRACERTYPE' not in joined.colnames:
        joined['TRACERTYPE'] = tr_types
    available_pref = np.unique(tracer_prefixes(tr_types))
    if tracers:
        want = {str(t).split('_', 1)[0].upper() for t in tracers}
        plot_tracers = [t for t in ORDERED_TRACERS if t in want and t in set(available_pref.tolist())]
    else:
        avail_set = set(available_pref.tolist())
        plot_tracers = [t for t in ORDERED_TRACERS if t in avail_set]

    if not plot_tracers:
        if verbose:
            print(f'Plot zone {zone}: no matching tracers — skipping.')
        return None

    os.makedirs(os.path.dirname(os.path.abspath(out_png)), exist_ok=True)

    plot_wedges(
        joined, plot_tracers, zone, 'void', out_png,
        smin=2, max_z=max_z,
        min_npts=min_npts,
        color_mode='group',
        title=f'Watershed voids — zone {zone_tag(zone)}',
    )
    if verbose:
        print(f'Plot zone {zone}: saved → {out_png}')
    return out_png


# --------------------------------------------------------------------------- #
#  CLI
# --------------------------------------------------------------------------- #

def _default_release():
    return os.environ.get('RELEASE', 'EDR').upper()


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            'Watershed void finder for ASTRA-DESI classification catalogues.\n\n'
            'Identifies connected underdense regions (R ≤ r-threshold) using a\n'
            'graph-based watershed algorithm on the Delaunay pair graph.'
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    io = p.add_argument_group('I/O paths')
    io.add_argument('--raw-dir', required=True,
                    help='Directory containing raw zone FITS files.')
    io.add_argument('--class-dir', required=True,
                    help='Release root containing classification products '
                         '(must have a classification/ sub-directory).')
    io.add_argument('--pairs-dir', default=None,
                    help='Directory containing pairs FITS files.  Required '
                         'unless --recompute-pairs is set.')
    io.add_argument('--out-dir', required=True,
                    help='Output directory for watershed group files.')

    sel = p.add_argument_group('Zone / tracer selection')
    sel.add_argument('--zones', nargs='+', required=True,
                     help='Zone identifiers to process '
                          '(e.g., 01 02 or NGC1 NGC2).')
    sel.add_argument('--tracer', type=str, default=None,
                     help='Restrict to a single tracer prefix (e.g., BGS_ANY). '
                          'Default: process all tracers found in the files.')
    sel.add_argument('--rand-iters', nargs='+', type=int, default=None,
                     help='Random iterations to process.  Default: all found '
                          'in the raw table.')

    algo = p.add_argument_group('Algorithm parameters')
    algo.add_argument('--r-threshold', type=float, default=-0.7,
                      help='Maximum R = (NDATA-NRAND)/(NDATA+NRAND) to '
                           'include.  Must be negative.  (default: -0.7)')
    algo.add_argument('--min-group-size', type=int, default=4,
                      help='Minimum members for a group to appear in output. '
                           '(default: 4)')
    algo.add_argument('--recompute-pairs', action='store_true',
                      help='Recompute Delaunay pairs from raw positions '
                           'instead of reading from --pairs-dir.')

    meta = p.add_argument_group('Metadata / tagging')
    meta.add_argument('--release', default=_default_release(),
                      help='Release tag stored in output FITS header '
                           '(default: $RELEASE env var or "EDR").')
    meta.add_argument('--out-tag', type=str, default=None,
                      help='Optional tag appended to input/output filenames '
                           '(e.g., a tracer suffix).')

    viz = p.add_argument_group('Visualisation')
    viz.add_argument('--plot', action='store_true',
                     help='Generate a wedge plot of the watershed groups after '
                          'processing each zone.')
    viz.add_argument('--plot-output', type=str, default=None,
                     help='Directory for plot PNG files.  Defaults to --out-dir.')
    viz.add_argument('--plot-rand-iter', type=int, default=0,
                     help='Random iteration shown in the plot.  Data points '
                          '(RANDITER=-1) are always included.  (default: 0)')
    viz.add_argument('--plot-min-npts', type=int, default=4,
                     help='Minimum group size to display in the plot. '
                          '(default: 4)')
    viz.add_argument('--plot-max-z', type=float, default=None,
                     help='Maximum redshift shown in the plot.')
    viz.add_argument('--plot-tracers', nargs='+', default=None,
                     help='Tracer prefixes to include in the plot '
                          '(e.g., BGS LRG).  Default: all available.')

    p.add_argument('--verbose', action='store_true',
                   help='Print progress information.')

    return p.parse_args()


def main():
    args = parse_args()

    if args.r_threshold >= 0:
        raise SystemExit('Error: --r-threshold must be negative.')

    if not args.recompute_pairs and args.pairs_dir is None:
        raise SystemExit(
            'Error: provide --pairs-dir or use --recompute-pairs.'
        )

    release_tag = str(args.release).upper()
    init = t.time()

    for zone in args.zones:
        result = process_zone(
            zone,
            raw_dir=args.raw_dir,
            class_dir=args.class_dir,
            pairs_dir=args.pairs_dir,
            out_dir=args.out_dir,
            r_threshold=args.r_threshold,
            min_group_size=args.min_group_size,
            rand_iters=args.rand_iters,
            tracer=args.tracer,
            recompute_pairs=args.recompute_pairs,
            release_tag=release_tag,
            out_tag=args.out_tag,
            verbose=args.verbose,
        )
        if result:
            print(f'---- zone {zone} done: {result}')
        else:
            print(f'---- zone {zone}: no groups found.')

        if args.plot and result:
            plot_dir = args.plot_output or args.out_dir
            ztag = zone_tag(zone)
            tsuf = safe_tag(args.out_tag)
            out_png = os.path.join(plot_dir, f'watershed_zone_{ztag}{tsuf}.png')
            try:
                plot_watershed_zone(
                    zone,
                    raw_dir=args.raw_dir,
                    out_dir=args.out_dir,
                    out_png=out_png,
                    out_tag=args.out_tag,
                    rand_iter=args.plot_rand_iter,
                    min_npts=args.plot_min_npts,
                    max_z=args.plot_max_z,
                    tracers=args.plot_tracers,
                    verbose=args.verbose,
                )
            except Exception as exc:
                print(f'---- zone {zone} plot failed: {exc}', file=sys.stderr)

    print(f'Elapsed: {(t.time() - init) / 60:.2f} min')


if __name__ == '__main__':
    main()
