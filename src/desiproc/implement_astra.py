import numpy as np
from astropy.table import Table
from scipy.spatial import Delaunay
from collections import defaultdict
import gc


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

    Args:
    	pts (np.ndarray): Array of shape (N, 3) containing 3D coordinates.
    Returns:
    	np.ndarray: Array of unique pairs of indices representing edges in the triangulation.
    """
    try:
        tri = Delaunay(pts)
        simps = tri.simplices
    finally:
        try:
            del tri
        except NameError:
            pass
    comb = np.array([(0,1),(0,2),(0,3),(1,2),(1,3),(2,3)], dtype=np.int64)
    edges = simps[:, comb].reshape(-1, 2)
    edges.sort(axis=1)
    pairs = np.unique(edges, axis=0)
    gc.collect()
    return pairs


def process_delaunay(pts, tids, is_data, iteration):
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
                        int(data_count[i]), int(total_count[i] - data_count[i])))

    return pair_rows, class_rows, r_updates


def generate_pairs(tbl, n_random):
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

	blocks = extract_tracer_blocks(tbl)
	for data in blocks.values():
		tids, rand_sub, coords, is_data = (data['tids'], data['rand'], data['coords'], data['is_data'])

		for j in range(n_random):
			mask = is_data | (rand_sub == j)
			if not mask.any():
				continue
			pr, cr, ru = process_delaunay(coords[mask], tids[mask], is_data[mask], j)
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
	tbl.write(output_path, format='fits', overwrite=True)


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
	return Table(rows=rows, names=('TARGETID','RANDITER','ISDATA','NDATA','NRAND'),
                 dtype=('i8','i4','bool','i4','i4'))


def save_classification_fits(rows, output_path):
    """
    Saves the classification table to a FITS file.

	Args:
		rows (list): List of tuples containing classification data.
		output_path (str): Path to save the FITS file.
	"""
    tbl = build_class_table(rows)
    tbl.write(output_path, format='fits', overwrite=True)


def build_probability_table(r_by_tid):
	"""
	Builds a probability table from the random values associated with target IDs.

	Args:
		r_by_tid (defaultdict): Dictionary mapping target IDs to lists of random values.
	Returns:
		Table: Astropy Table containing target IDs and their corresponding probabilities.
	"""
	n = len(r_by_tid)
	dtype = [('TARGETID','i8'), ('PVOID','f4'),
             ('PSHEET','f4'), ('PFILAMENT','f4'),
             ('PKNOT','f4')]
	data = np.zeros(n, dtype=dtype)

	for i,(tid, rlist) in enumerate(r_by_tid.items()):
		arr = np.asarray(rlist, dtype=np.float32)
		if arr.size:
			void_m = (arr <= -0.9)
			sheet_m = (arr > -0.9) & (arr < 0.0)
			fila_m = (arr >= 0.0) & (arr < 0.9)
			knot_m = (arr >= 0.9)
			counts = np.array([void_m.sum(), sheet_m.sum(), fila_m.sum(), knot_m.sum()], dtype=np.int64)
			total = counts.sum()
			if total > 0:
				probs = counts / total
			else:
				probs = np.zeros(4, dtype=np.float32)
		else:
			probs = np.zeros(4, dtype=np.float32)
		data[i] = (int(tid), float(probs[0]), float(probs[1]), float(probs[2]), float(probs[3]))
	return Table(data)


def save_probability_fits(r_by_tid, output_path):
	"""
	Saves the probability table to a FITS file.

	Args:
		r_by_tid (defaultdict): Dictionary mapping target IDs to lists of random values.
		output_path (str): Path to save the FITS file.
	"""
	tbl = build_probability_table(r_by_tid)
	tbl.write(output_path, format='fits', overwrite=True)