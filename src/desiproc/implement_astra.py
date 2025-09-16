import numpy as np
from astropy.table import Table
from scipy.spatial import Delaunay
from collections import defaultdict
import multiprocessing as mp, os, gc


_GP_SHARED = None

def _gp_init_worker(tids, rand_sub, coords, is_data, tracer):
    """
    Initializes global variables for worker processes.
    
    Args:
    	tids (np.ndarray): Array of target IDs.
    	rand_sub (np.ndarray): Array of random iteration indices.
    	coords (np.ndarray): Array of 3D coordinates.
    	is_data (np.ndarray): Boolean array indicating real data points.
    	tracer (str): Tracer type string.
    """
    global _GP_SHARED
    _GP_SHARED = (tids, rand_sub, coords, is_data, tracer)


def extract_tracer_blocks(tbl):
    """
    Extracts blocks of data for each tracer type from the table.

    Args:
    	tbl (Table): Astropy Table containing the data.
    Returns:
    	dict: Dictionary with tracer types as keys and their corresponding data blocks.
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
    Computes unique pairs of points from a set of 3D coordinates using Delaunay triangulation.

    Optimized: uses the CSR-like neighbor representation from SciPy's Delaunay
    (vertex_neighbor_vertices) to avoid materializing and uniquifying all
    simplex edges, greatly reducing memory and time for large inputs.

    Args:
        pts (np.ndarray): Array of shape (N, 3) containing 3D coordinates.
    Returns:
        np.ndarray: Array of unique pairs (i,j) with i<j.
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
    Processes 3D coordinates to compute pairs, counts, and classification for a given iteration.

    Args:
    	pts (np.ndarray): Array of shape (N, 3) containing 3D coordinates.
    	tids (np.ndarray): Array of target IDs corresponding to the points.
    	is_data (np.ndarray): Boolean array indicating whether each point is real data.
    	iteration (int): Current iteration number for random data.
    Returns:
    	tuple: Contains:
    		- pair_rows (list): List of tuples representing pairs of target IDs and iteration.
    		- class_rows (list): List of tuples for classification data.
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
    for i,r in zip(valid, r_vals):
    	r_updates[int(tids[i])].append(float(r))

    for i in range(n):
        class_rows.append((int(tids[i]), iteration, bool(is_data[i]),
                        int(data_count[i]), int(total_count[i] - data_count[i]),
                        str(tracer)))

    return pair_rows, class_rows, r_updates


def _gp_process_iter(j):
    """
    Worker function to process a specific random iteration.
    
    Args:
    	j (int): Random iteration index.
    Returns:
    	tuple: Contains:
    		- pair_rows (list): List of tuples representing pairs of target IDs and iteration.
    		- class_rows (list): List of tuples for classification data.
    """
    tids, rand_sub, coords, is_data, tracer = _GP_SHARED
    mask = is_data | (rand_sub == j)
    if not mask.any():
        return [], [], {}
    return process_delaunay(coords[mask], tids[mask], is_data[mask], j, tracer)


def generate_pairs(tbl, n_random, n_jobs=None):
    """
    Generates pairs of target IDs and classification data from the input table.

    Args:
        tbl (Table): Astropy Table containing the data.
        n_random (int): Number of random iterations to process.
    Returns:
        tuple: Contains:
            - pair_rows (list): List of tuples representing pairs of target IDs and iteration.
            - class_rows (list): List of tuples for classification data.
            - r_by_tid (defaultdict): Dictionary mapping target IDs to random values.
    """
    pair_rows, class_rows = [], []
    r_by_tid = defaultdict(list)

    if n_jobs is None:
        try:
            cpu_env = int(os.environ.get('SLURM_CPUS_PER_TASK', '1'))
        except Exception:
            cpu_env = 1
        n_jobs = max(1, min(cpu_env, int(n_random)))
        try:
            cap = int(os.environ.get('PAIR_NJOBS_CAP', '0'))
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
    Builds a pairs table from the provided rows.

	Args:
		rows (list): List of tuples containing pairs data.
	Returns:
		Table: Astropy Table containing pairs data with columns:
			- TARGETID1: First target ID in the pair
			- TARGETID2: Second target ID in the pair
			- RANDITER: Random iteration number
    """
    return Table(rows=rows, names=('TARGETID1','TARGETID2','RANDITER'),
				 dtype=('i8','i8','i4'))


def save_pairs_fits(rows, output_path):
	"""
	Saves the pairs table to a FITS file.

	Args:
		rows (list): List of tuples containing pairs data.
		output_path (str): Path to save the FITS file.
	"""
	tbl = build_pairs_table(rows)
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
    Load an existing pairs FITS file.

    Args:
        path (str): Path to the FITS file containing pairs with columns
            TARGETID1, TARGETID2, RANDITER.
    Returns:
        Table: Astropy Table with the pairs data.
    """
    try:
        return Table.read(path, memmap=True)
    except TypeError:
        return Table.read(path)


def build_class_rows_from_pairs(tbl, pairs_tbl, n_random):
    """
    Rebuild classification rows (NDATA/NRAND counts) from an existing pairs table.

    This avoids recomputing Delaunay triangulations when the pairs have already
    been generated and saved. It uses the input raw table `tbl` to determine
    which TARGETIDs are data (RANDITER == -1) and the TRACERTYPE of each
    TARGETID, and uses the pairs to count per-iteration degrees.

    Args:
        tbl (Table): Raw table with columns TARGETID, RANDITER, TRACERTYPE.
        pairs_tbl (Table): Pairs table with TARGETID1, TARGETID2, RANDITER.
        n_random (int): Number of random iterations used to generate pairs.
    Returns:
        list: Classification rows tuples (TARGETID, RANDITER, ISDATA, NDATA, NRAND, TRACERTYPE).
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
	Builds a classification table from the provided rows.

	Args:
		rows (list): List of tuples containing classification data.
	Returns:
		Table: Astropy Table containing classification data with columns:
			- TARGETID: Target ID
			- RANDITER: Random iteration number
			- ISDATA: Boolean indicating if the entry is real data
			- NDATA: Number of real data points
			- NRAND: Number of random points
	"""
	return Table(rows=rows, names=('TARGETID','RANDITER','ISDATA','NDATA','NRAND','TRACERTYPE'),
                 dtype=('i8','i4','bool','i4','i4','U24'))


def save_classification_fits(rows, output_path):
    """
    Saves the classification table to a FITS file.

	Args:
		rows (list): List of tuples containing classification data.
		output_path (str): Path to save the FITS file.
	"""
    tbl = build_class_table(rows)
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


def build_probability_table(class_rows, r_limit=0.9):
    """
    Builds a probability table from the classification rows.

    Args:
        class_rows (list): List of tuples containing classification data.
        r_limit (float): Upper limit for the r value bins.
    Returns:
        Table: Astropy Table containing target IDs and their corresponding probabilities.
    """
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

    classes = np.digitize(r, bins=[-r_limit, 0.0, r_limit])
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


def save_probability_fits(class_rows, output_path, r_limit=0.9):
	"""
	Saves the probability table to a FITS file.

	Args:
		class_rows (list): List of tuples containing classification data.
		output_path (str): Path to save the FITS file.
		r_limit (float, optional): Upper limit threshold used to bin r into
		    classes when computing probabilities. Defaults to 0.9.
	"""
	tbl = build_probability_table(class_rows, r_limit=r_limit)
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