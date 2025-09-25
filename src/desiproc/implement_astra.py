import gc
import multiprocessing as mp
import os
from collections import defaultdict

import numpy as np
from astropy.table import Table
from scipy.spatial import Delaunay

_GP_SHARED = None


def _gp_init_worker(tids, rand_sub, coords, is_data, tracer):
    """
    Initialise shared worker state for multiprocessing jobs.

    Args:
        tids (np.ndarray): Target identifiers.
        rand_sub (np.ndarray): Random iteration numbers for each row.
        coords (np.ndarray): Cartesian coordinates with shape ``(N, 3)``.
        is_data (np.ndarray): Boolean mask identifying real data rows.
        tracer (str): Tracer label associated with the block.
    """
    global _GP_SHARED
    _GP_SHARED = (tids, rand_sub, coords, is_data, tracer)


def extract_tracer_blocks(tbl):
    """
    Group table rows by tracer prefix and return per-tracer blocks.

    Args:
        tbl (Table): Input table containing ``TRACERTYPE`` and coordinate columns.
    Returns:
        dict: Mapping from tracer prefix to arrays of IDs, iterations, coordinates,
        and data masks.
    """
    tracertypes = tbl['TRACERTYPE'].astype(str)
    randiters = np.asarray(tbl['RANDITER'], dtype=np.int32)
    targetids = np.asarray(tbl['TARGETID'], dtype=np.int64)
    coords_all = np.column_stack((np.asarray(tbl['XCART'], dtype=np.float32),
                                  np.asarray(tbl['YCART'], dtype=np.float32),
                                  np.asarray(tbl['ZCART'], dtype=np.float32)))

    base_tracer = np.asarray([t.rsplit('_', 1)[0] for t in tracertypes], dtype=object)
    prefixes = np.unique(base_tracer)
    blocks = {}
    for tracer in prefixes:
        mask = (base_tracer == tracer)
        idxs = np.nonzero(mask)[0]
        blocks[tracer] = {'tids': targetids[idxs],
                          'rand': randiters[idxs],
                          'coords': coords_all[idxs],
                          'is_data': randiters[idxs] == -1}
    return blocks


def compute_delaunay_pairs(pts):
    """
    Return unique edges from the Delaunay triangulation of ``pts``.

    The implementation leverages the CSR-like neighbour representation exposed
    by SciPy to avoid creating and deduplicating all simplex edges explicitly.

    Args:
        pts (np.ndarray): Cartesian coordinates with shape ``(N, 3)``.
    Returns:
        np.ndarray: Array of index pairs ``(i, j)`` with ``i < j``.
    """
    tri = Delaunay(pts)
    indptr, indices = tri.vertex_neighbor_vertices
    n = pts.shape[0]
    out = []
    for i in range(n):
        start, end = indptr[i], indptr[i+1]
        nbrs = indices[start:end]
        for j in nbrs:
            if j > i:
                out.append((i, j))
    del tri
    gc.collect()
    if not out:
        return np.empty((0,2), dtype=np.int64)
    return np.asarray(out, dtype=np.int64)


def process_delaunay(pts, tids, is_data, iteration, tracer):
    """
    Generate pair and classification rows for a single tracer iteration.

    Args:
        pts (np.ndarray): Cartesian coordinates with shape ``(N, 3)``.
        tids (np.ndarray): Target identifiers aligned with ``pts``.
        is_data (np.ndarray): Boolean mask identifying data rows.
        iteration (int): Random iteration identifier.
        tracer (str): Tracer prefix.
    Returns:
        tuple[list, list, dict]: Pair rows, classification rows, and per-target
        random ``r`` updates.
    """
    pairs = compute_delaunay_pairs(pts)
    idx0, idx1 = pairs[:,0], pairs[:,1]
    n = len(tids)

    total_count = np.bincount(idx0, minlength=n) + np.bincount(idx1, minlength=n)
    mask0 = is_data[idx0]
    mask1 = is_data[idx1]
    data_count = (np.bincount(idx0[mask1], minlength=n) + np.bincount(idx1[mask0], minlength=n))

    tid0 = tids[idx0].astype(np.int64, copy=False)
    tid1 = tids[idx1].astype(np.int64, copy=False)
    iters = np.full(tid0.shape[0], iteration, dtype=np.int32)
    pair_rows = list(zip(tid0.tolist(), tid1.tolist(), iters.tolist()))

    class_rows = []
    r_updates = defaultdict(list)

    valid = np.nonzero(is_data & (total_count>0))[0]
    r_vals = (data_count[valid] - (total_count[valid] - data_count[valid])) / total_count[valid]
    for i, r in zip(valid, r_vals):
        r_updates[int(tids[i])].append(float(r))

    for i in range(n):
        class_rows.append((int(tids[i]), iteration, bool(is_data[i]),
                        int(data_count[i]), int(total_count[i] - data_count[i]),
                        str(tracer)))

    return pair_rows, class_rows, r_updates


def _gp_process_iter(j):
    """
    Process a single random iteration within a worker process.

    Args:
        j (int): Random iteration identifier.
    Returns:
        tuple[list, list, dict]: Pair rows, classification rows, and ``r`` updates.
    """
    tids, rand_sub, coords, is_data, tracer = _GP_SHARED
    mask = is_data | (rand_sub == j)
    if not mask.any():
        return [], [], {}
    return process_delaunay(coords[mask], tids[mask], is_data[mask], j, tracer)


def generate_pairs(tbl, n_random, n_jobs=None):
    """
    Run the pair-generation pipeline for all tracers in ``tbl``.

    Args:
        tbl (Table): Input table containing data and random catalogues.
        n_random (int): Total number of random iterations available.
        n_jobs (int, optional): Parallel worker count; defaults to available CPUs.
    Returns:
        tuple[list, list, defaultdict]: Pair rows, classification rows, and
        cached ``r`` updates keyed by target id.
    """
    pair_rows, class_rows = [], []
    r_by_tid = defaultdict(list)

    if n_jobs is None:
        env_val = os.environ.get('SLURM_CPUS_PER_TASK', '').strip()
        if env_val:
            try:
                cpu_env = int(env_val)
            except Exception:
                cpu_env = 1
        else:
            cpu_env = os.cpu_count() or 1
        n_jobs = max(1, min(cpu_env, int(n_random)))
        cap_val = os.environ.get('PAIR_NJOBS_CAP', '').strip()
        if cap_val:
            try:
                cap = int(cap_val)
                if cap > 0:
                    n_jobs = max(1, min(n_jobs, cap))
            except Exception:
                pass

    blocks = extract_tracer_blocks(tbl)
    for tracer, data in blocks.items():
        tids, rand_sub, coords, is_data = (data['tids'], data['rand'], data['coords'], data['is_data'])

        if n_jobs > 1:
            with mp.get_context('fork').Pool(processes=n_jobs,
                                             initializer=_gp_init_worker,
                                             initargs=(tids, rand_sub, coords, is_data, tracer)) as pool:
                for pr, cr, ru in pool.imap_unordered(_gp_process_iter, range(n_random)):
                    if pr:
                        pair_rows.extend(pr)
                    if cr:
                        class_rows.extend(cr)
                    for tid, rs in ru.items():
                        r_by_tid[tid].extend(rs)
        else:
            for j in range(n_random):
                mask = is_data | (rand_sub == j)
                if not mask.any():
                    continue
                pr, cr, ru = process_delaunay(coords[mask], tids[mask], is_data[mask], j, tracer)
                pair_rows.extend(pr)
                class_rows.extend(cr)
                for tid, rs in ru.items():
                    r_by_tid[tid].extend(rs)

    return pair_rows, class_rows, r_by_tid


def build_pairs_table(rows):
    """
    Construct a pairs table from ``rows``.

    Args:
        rows (list): Sequence of ``(targetid1, targetid2, iteration)`` tuples.
    Returns:
        Table: Table with ``TARGETID1``, ``TARGETID2``, and ``RANDITER`` columns.
    """

    return Table(
        rows=rows,
        names=('TARGETID1', 'TARGETID2', 'RANDITER'),
        dtype=('i8', 'i8', 'i4'),
    )


def save_pairs_fits(rows, output_path, meta=None):
    """
    Write the pairs table to ``output_path`` as a FITS file.

    Args:
        rows (list): Sequence of pair tuples.
        output_path (str): Destination path for the FITS file.
        meta (dict | None): Optional metadata inserted into the FITS header.
    """
    tbl = build_pairs_table(rows)
    if meta:
        for key, value in meta.items():
            tbl.meta[key] = value
    _tmp = f"{output_path}.tmp"
    try:
        tbl.write(_tmp, format='fits', overwrite=True)
        os.replace(_tmp, output_path)
    except Exception:
        try:
            if os.path.exists(_tmp):
                os.remove(_tmp)
        except Exception:
            pass
        raise


def load_pairs_fits(path):
    """
    Load a previously written pairs FITS file.

    Args:
        path (str): Path to the FITS file containing ``TARGETID1``, ``TARGETID2``,
            and ``RANDITER`` columns.
    Returns:
        Table: Table with pair information.
    """
    try:
        return Table.read(path, memmap=True)
    except TypeError:
        return Table.read(path)


def build_class_rows_from_pairs(tbl, pairs_tbl, n_random):
    """
    Reconstruct classification rows from previously saved pairs.

    Args:
        tbl (Table): Raw table with ``TARGETID``, ``RANDITER``, and ``TRACERTYPE``.
        pairs_tbl (Table): Table containing pair information.
        n_random (int): Number of random iterations present in the dataset.
    Returns:
        list[tuple]: Classification tuples ``(TARGETID, RANDITER, ISDATA, NDATA, NRAND, TRACERTYPE)``.
    """
    tids = np.asarray(tbl['TARGETID'], dtype=np.int64)
    randiter = np.asarray(tbl['RANDITER'], dtype=np.int32)
    trtype = np.asarray(tbl['TRACERTYPE']).astype('U24')

    is_data_map = {int(t): (ri == -1) for t, ri in zip(tids, randiter)}
    tracer_map = {int(t): str(tt) for t, tt in zip(tids, trtype)}

    rand_ids_by_j = {}
    for j in range(n_random):
        mask = (randiter == j)
        if mask.any():
            rand_ids_by_j[j] = np.asarray(tids[mask], dtype=np.int64)
        else:
            rand_ids_by_j[j] = np.empty(0, dtype=np.int64)

    data_ids = np.asarray(tids[randiter == -1], dtype=np.int64)

    p_tid1 = np.asarray(pairs_tbl['TARGETID1'], dtype=np.int64)
    p_tid2 = np.asarray(pairs_tbl['TARGETID2'], dtype=np.int64)
    p_j = np.asarray(pairs_tbl['RANDITER'], dtype=np.int32)

    class_rows = []

    present_js = np.unique(p_j)
    for j in range(n_random):
        if j in present_js:
            sel = (p_j == j)
            a = p_tid1[sel]
            b = p_tid2[sel]

            if a.size > 0:
                all_pairs_ids = np.concatenate([a, b])
                uniq, inv = np.unique(all_pairs_ids, return_inverse=True)
                idx_a = inv[:a.size]
                idx_b = inv[a.size:]

                total = np.zeros(uniq.size, dtype=np.int64)
                np.add.at(total, idx_a, 1)
                np.add.at(total, idx_b, 1)

                is_data_uniq = np.fromiter((1 if is_data_map.get(int(t), False) else 0 for t in uniq),
                                            dtype=np.int8, count=uniq.size)

                ndata = np.zeros(uniq.size, dtype=np.int64)
                np.add.at(ndata, idx_a, is_data_uniq[idx_b].astype(np.int64))
                np.add.at(ndata, idx_b, is_data_uniq[idx_a].astype(np.int64))

                total_map = {int(t): int(c) for t, c in zip(uniq.tolist(), total.tolist())}
                ndata_map = {int(t): int(c) for t, c in zip(uniq.tolist(), ndata.tolist())}
            else:
                total_map, ndata_map = {}, {}
        else:
            total_map, ndata_map = {}, {}

        for t in data_ids.tolist():
            nd = int(ndata_map.get(int(t), 0))
            tt = int(total_map.get(int(t), 0))
            nr = tt - nd
            class_rows.append((int(t), int(j), True, nd, nr, tracer_map.get(int(t), 'UNKNOWN')))

        rids = rand_ids_by_j.get(j, np.empty(0, dtype=np.int64))
        for t in rids.tolist():
            nd = int(ndata_map.get(int(t), 0))
            tt = int(total_map.get(int(t), 0))
            nr = tt - nd
            class_rows.append((int(t), int(j), False, nd, nr, tracer_map.get(int(t), 'UNKNOWN')))

    return class_rows


def build_class_table(rows):
    """
    Construct the classification table from tuple rows.

    Args:
        rows (list): Sequence of classification tuples.
    Returns:
        Table: Table with ``TARGETID``, ``RANDITER``, ``ISDATA``, ``NDATA``, ``NRAND``,
        and ``TRACERTYPE`` columns.
    """

    return Table(rows=rows, names=('TARGETID', 'RANDITER', 'ISDATA', 'NDATA', 'NRAND', 'TRACERTYPE'),
                 dtype=('i8', 'i4', 'bool', 'i4', 'i4', 'U24'))


def save_classification_fits(rows, output_path, meta=None):
    """
    Write the classification table to ``output_path`` as a FITS file.

    Args:
        rows (list): Classification tuples.
        output_path (str): Destination path for the FITS file.
        meta (dict | None): Optional metadata inserted into the FITS header.
    """
    tbl = build_class_table(rows)
    if meta:
        for key, value in meta.items():
            tbl.meta[key] = value
    _tmp = f"{output_path}.tmp"
    try:
        tbl.write(_tmp, format='fits', overwrite=True)
        os.replace(_tmp, output_path)
    except Exception:
        try:
            if os.path.exists(_tmp):
                os.remove(_tmp)
        except Exception:
            pass
        raise


def build_probability_table(class_rows, r_lower=-0.9, r_upper=0.9):
    """
    Build a probability table from classification rows.

    Args:
        class_rows (list): Classification tuples.
        r_lower (float): Lower ``r`` threshold (negative).
        r_upper (float): Upper ``r`` threshold (positive).
    Returns:
        Table: Probability table containing ``PVOID``, ``PSHEET``, ``PFILAMENT``, ``PKNOT``.
    Raises:
        ValueError: If the thresholds do not straddle zero.
    """
    if r_lower >= 0 or r_upper <= 0:
        raise ValueError('r_lower must be negative and r_upper must be positive.')
    arr = np.asarray(class_rows, dtype=[('TARGETID','i8'), ('RANDITER','i4'),
                                        ('ISDATA','?'), ('NDATA','i4'),
                                        ('NRAND','i4'), ('TRACERTYPE','U24')])

    m = arr['ISDATA']
    tids = arr['TARGETID'][m]
    trcs = arr['TRACERTYPE'][m].astype('U24')
    ndata = arr['NDATA'][m].astype(np.float64, copy=False)
    nrand = arr['NRAND'][m].astype(np.float64, copy=False)

    denom = ndata + nrand
    r = np.zeros_like(denom, dtype=np.float64)
    np.divide(ndata - nrand, denom, out=r, where=(denom > 0))

    classes = np.digitize(r, bins=[r_lower, 0.0, r_upper])
    keys = np.empty(tids.size, dtype=[('TARGETID','i8'),('TRACERTYPE','U24')])
    keys['TARGETID'] = tids
    keys['TRACERTYPE'] = trcs
    uniq, inv = np.unique(keys, return_inverse=True)
    counts = np.zeros((uniq.size, 4), dtype=np.int64)
    np.add.at(counts, (inv, classes), 1)

    total = counts.sum(axis=1, keepdims=True).astype(np.float32)
    probs = counts.astype(np.float32)
    np.divide(probs, total, out=probs, where=(total > 0))

    return Table({'TARGETID': uniq['TARGETID'], 'TRACERTYPE': uniq['TRACERTYPE'],
                  'PVOID':probs[:, 0], 'PSHEET':probs[:, 1],
                  'PFILAMENT':probs[:, 2], 'PKNOT':probs[:, 3]})


def save_probability_fits(class_rows, output_path, r_lower=-0.9, r_upper=0.9, meta=None):
    """
    Saves the probability table to a FITS file.

    Args:
        class_rows (list): List of tuples containing classification data.
        output_path (str): Path to save the FITS file.
        r_lower (float, optional): Lower ``r`` threshold (default: -0.9).
        r_upper (float, optional): Upper ``r`` threshold (default: 0.9).
        meta (dict | None): Optional metadata to inject into the FITS header.
    """
    tbl = build_probability_table(class_rows, r_lower=r_lower, r_upper=r_upper)
    if meta:
        for key, value in meta.items():
            tbl.meta[key] = value
    _tmp = f"{output_path}.tmp"
    try:
        tbl.write(_tmp, format='fits', overwrite=True)
        os.replace(_tmp, output_path)
    except Exception:
        try:
            if os.path.exists(_tmp):
                os.remove(_tmp)
        except Exception:
            pass
        raise