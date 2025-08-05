import numpy as np
from astropy.table import Table
from scipy.spatial import Delaunay
from itertools import combinations
from collections import defaultdict


def extract_tracer_blocks(tbl):
    """
    Split the input Table by tracer prefix.

    Args:
      tbl: Astropy Table with columns 'TRACERTYPE', 'RANDITER', 'TARGETID', 'XCART', 'YCART', 'ZCART'
    Returns:
      A dict mapping each tracer prefix to a dict with:
        - 'tids': ndarray of TARGETID
        - 'rand': ndarray of RANDITER
        - 'coords': ndarray of shape (N,3) with XCART, YCART, ZCART
        - 'is_data': boolean mask where RANDITER == -1
    """
    tracertypes = tbl['TRACERTYPE'].astype(str)
    randiters = np.asarray(tbl['RANDITER'])
    targetids = np.asarray(tbl['TARGETID'])
    coords_all = np.column_stack((tbl['XCART'], tbl['YCART'], tbl['ZCART']))
    prefixes = {s.split('_', 1)[0] for s in tracertypes}

    blocks = {}
    for tracer in prefixes:
        mask = np.char.startswith(tracertypes, tracer)
        idxs = np.nonzero(mask)[0]
        blocks[tracer] = {'tids':targetids[idxs],
                          'rand':randiters[idxs],
                          'coords':coords_all[idxs],
                          'is_data':randiters[idxs] == -1}
    return blocks


def compute_delaunay_adjacency(pts):
    """
    Given a set of points, compute the Delaunay triangulation
    and return the adjacency list and pairs of indices.

    Args:
      pts: ndarray of shape (N, 3) with coordinates of points
    Returns:
      neighbors: List[set] where neighbors[i] contains indices of neighbors for point i
      pairs: List[tuple] of pairs of indices (i, j) for each edge
    """
    tri = Delaunay(pts)
    neighbors = [set() for _ in pts]
    pairs = []
    for simplex in tri.simplices:
        for a, b in combinations(simplex, 2):
            neighbors[a].add(b)
            neighbors[b].add(a)
            pairs.append((a, b))
    return neighbors, pairs


def process_neighbors(neighbors, pairs, tids, is_data, iteration):
    """
    Process the neighbors to generate pair_rows and class_rows.

    Args:
      neighbors: List[set] of neighbors for each point
      pairs: List[tuple] of pairs of indices
      tids: ndarray of TARGETID for the points
      is_data: boolean mask indicating which points are data
      iteration: current random iteration index
    Returns:
      pair_rows: List[(int, int, int)] with pairs of TARGETID and iteration
      class_rows: List[(int, int, bool, int, int)] with TARGETID, iteration, is_data flag, ndata, nrand
      r_updates: Dict[int, List[float]] with TARGETID and corresponding r values
    """
    pair_rows = [(int(tids[a]), int(tids[b]), iteration) for a, b in pairs]
    class_rows = []
    r_updates = defaultdict(list)

    for i, nbrs in enumerate(neighbors):
        tid = int(tids[i])
        ndata = sum(is_data[list(nbrs)])
        nrand = len(nbrs) - ndata
        flag = bool(is_data[i])
        class_rows.append((tid, iteration, flag, ndata, nrand))
        if flag and (ndata + nrand) > 0:
            r = (ndata - nrand) / (ndata + nrand)
            r_updates[tid].append(r)

    return pair_rows, class_rows, r_updates


def generate_pairs(tbl, n_random):
    """
    Generate pairs, class rows, and r values from the input Table.

    Args:
      tbl: Astropy Table with columns 'TRACERTYPE', 'RANDITER', 'TARGETID', 'XCART', 'YCART', 'ZCART'
      n_random: Number of random iterations to process
    Returns:
      pair_rows: List of tuples (TARGETID1, TARGETID2, RANDITER)
      class_rows: List of tuples (TARGETID, RANDITER, is_data_flag, ndata, nrand)
      r_by_tid: Dict[int, List[float]] mapping TARGETID to list of r values
    """
    pair_rows, class_rows, r_by_tid = [], [], defaultdict(list)

    blocks = extract_tracer_blocks(tbl)
    for tracer, data in blocks.items():
        tids, rand_sub, coords, is_data = data['tids'], data['rand'], data['coords'], data['is_data']

        for j in range(n_random):
            mask = is_data | (rand_sub == j)
            sel_idxs = np.nonzero(mask)[0]
            pts, tids_sel, is_data_j = coords[sel_idxs], tids[sel_idxs], is_data[sel_idxs]

            neighbors, pairs = compute_delaunay_adjacency(pts)
            pr, cr, rupd = process_neighbors(neighbors, pairs, tids_sel, is_data_j, j)

            pair_rows.extend(pr)
            class_rows.extend(cr)
            for tid, rs in rupd.items():
                r_by_tid[tid].extend(rs)

    return pair_rows, class_rows, r_by_tid


def build_pairs_table(rows):
    """
    Build an Astropy Table for pair rows.

    Args:
      rows: List of tuples (TARGETID1, TARGETID2, RANDITER)
    Returns:
      An Astropy Table with columns 'TARGETID1', 'TARGETID2', 'RAND
    """
    return Table( rows=rows, names=('TARGETID1','TARGETID2','RANDITER'), dtype=('i8','i8','i4'))

def save_pairs_fits(rows, output_path):
    """
    Save pair_rows to a gzipped FITS file.

    Args:
      rows: List of tuples (TARGETID1, TARGETID2, RANDITER)
      output_path: Path to save the FITS file
    """
    tbl = build_pairs_table(rows)
    tbl.write(output_path, format='fits', overwrite=True)#, compression='gzip')


def build_class_table(rows):
    """
    Build an Astropy Table for classification rows.

    Args:
      rows: List of tuples (TARGETID, RANDITER, is_data_flag, ndata, nrand)
    Returns:
      An Astropy Table with columns 'TARGETID', 'RANDITER', 'ISDATA', 'NDATA', 'NRAND'
    """
    return Table(rows=rows, names=('TARGETID','RANDITER','ISDATA','NDATA','NRAND'),
                 dtype=('i8','i4','bool','i4','i4'))


def save_classification_fits(rows, output_path):
    """
    Save class_rows to a gzipped FITS file.

    Args:
      rows: List of tuples (TARGETID, RANDITER, is_data_flag, ndata, nrand)
      output_path: Path to save the FITS file
    """
    tbl = build_class_table(rows)
    tbl.write(output_path, format='fits', overwrite=True)#, compression='gzip')


def classify_type(r):
    """
    Map r to an index for ['void','sheet','filament','knot'] based on its value.

    Args:
      r: float value in the range [-1.0, 1.0]
    Returns:
      An integer index corresponding to the classification type.
    Raises:
      ValueError if r is out of bounds.
    """
    if -1.0 <= r <= -0.9: return 0
    elif -0.9 < r <= 0.0: return 1
    elif 0.0 < r <= 0.9: return 2
    elif 0.9 < r <= 1.0: return 3
    else: raise ValueError(f"r out of bounds: {r}")


def build_probability_table(r_by_tid):
    """
    Build an Astropy Table of probabilities per TARGETID.

    Args:
      r_by_tid: Dictionary mapping TARGETID to list of r values
    Returns:
      An Astropy Table with columns 'TARGETID', 'PVOID', 'PSHEET', 'PFILAMENT', 'PKNOT'
    """
    n = len(r_by_tid)
    data = np.zeros(n, dtype=[('TARGETID','i8'), ('PVOID','f4'),
                              ('PSHEET','f4'), ('PFILAMENT','f4'),
                              ('PKNOT','f4')])
    
    for i, (tid, rlist) in enumerate(r_by_tid.items()):
        counts = [0,0,0,0]
        for r in rlist:
            counts[classify_type(r)] += 1
        total = len(rlist)
        data[i] = (tid, counts[0]/total, counts[1]/total, counts[2]/total, counts[3]/total)
    return Table(data)


def save_probability_fits(r_by_tid, output_path):
    """
    Save probability table to a gzipped FITS file.

    Args:
      r_by_tid: Dictionary mapping TARGETID to list of r values
      output_path: Path to save the FITS file
    """
    tbl = build_probability_table(r_by_tid)
    tbl.write(output_path, format='fits', overwrite=True)#, compression='gzip')