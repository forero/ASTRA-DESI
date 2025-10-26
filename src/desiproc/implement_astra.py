import gc
import multiprocessing as mp
import os
import shutil
import tempfile

import numpy as np
from astropy.table import Table
from scipy.spatial import Delaunay

_GP_SHARED = None

#! --- helpers for memory-mapped storage of intermediate results ---
_PAIR_ROW_DTYPE = np.dtype([('TARGETID1', np.int64),
                            ('TARGETID2', np.int64),
                            ('RANDITER', np.int32)])
_CLASS_ROW_DTYPE = np.dtype([('TARGETID', np.int64),
                             ('RANDITER', np.int32),
                             ('ISDATA', np.bool_),
                             ('NDATA', np.int32),
                             ('NRAND', np.int32),
                             ('TRACER_ID', np.uint8),
                             ('TRACERTYPE', 'S24')])
_TRACER_ASCII_DTYPE = _CLASS_ROW_DTYPE.fields['TRACERTYPE'][0]

_TRACER_NAME_TO_ID = {}
_TRACER_ID_TO_NAME = {}
_TRACER_ID_TO_FULL = {}


def register_tracer_mapping(name_to_id, full_labels=None):
    """
    Register a tracer mapping so in-memory operations can avoid repeated string handling.

    Args:
        name_to_id: Mapping from tracer base labels (e.g., ``'BGS_ANY'``) to integer identifiers.
        full_labels: Optional mapping from ``(tracer_id, is_data)`` to ASCII bytes representing
            the full tracer label (e.g., ``b'BGS_ANY_DATA'``). When omitted, labels are generated
            on the fly from ``name_to_id``.
    """
    global _TRACER_NAME_TO_ID, _TRACER_ID_TO_NAME, _TRACER_ID_TO_FULL
    _TRACER_NAME_TO_ID = dict(name_to_id or {})
    _TRACER_ID_TO_NAME = {int(v): str(k) for k, v in _TRACER_NAME_TO_ID.items()}
    if full_labels:
        _TRACER_ID_TO_FULL = {(int(k[0]), bool(k[1])): bytes(v)
                              for k, v in full_labels.items()}
    else:
        _TRACER_ID_TO_FULL = {}


def _tracer_id_from_label(label):
    """
    Return the integer tracer ID for a given tracer base label.
    """
    return _TRACER_NAME_TO_ID.get(str(label), -1)


def _full_tracer_label(tracer_id, is_data) -> bytes:
    """
    Return the full tracer label for a given tracer ID and data/random flag.
    """
    key = (int(tracer_id), bool(is_data))
    if key in _TRACER_ID_TO_FULL:
        return _TRACER_ID_TO_FULL[key]
    base = _TRACER_ID_TO_NAME.get(int(tracer_id))
    if base is None:
        return b'UNKNOWN'
    suffix = b'DATA' if is_data else b'RAND'
    return f'{base}_{suffix.decode()}'.encode('ascii', errors='ignore')


class TempTableStore:
    """
    Persist rows to temporary on-disk chunks and expose a contiguous memmap when finalised.

    This avoids keeping huge Python lists resident in memory while we accumulate
    pair/classification rows across many iterations.
    """
    def __init__(self, dtype, prefix='astra_tmp', base_dir=None):
        self.dtype = np.dtype(dtype)
        if base_dir is None:
            base_dir = (os.environ.get('ASTRA_TMPDIR')
                        or os.environ.get('PSCRATCH')
                        or os.environ.get('TMPDIR'))
        if base_dir:
            os.makedirs(base_dir, exist_ok=True)
        self._tmpdir = tempfile.mkdtemp(prefix=f'{prefix}_', dir=base_dir)
        self._chunks = []
        self._total = 0
        self._final_path = None

    @property
    def total(self):
        return self._total

    @property
    def path(self):
        return self._final_path

    def append(self, rows):
        """
        Append rows to the store, spilling them into a new on-disk chunk.

        Args:
            rows: Sequence or numpy structured array convertible to ``self.dtype``.
        """
        if rows is None:
            return
        if isinstance(rows, np.ndarray) and rows.dtype == self.dtype:
            arr = rows
        else:
            arr = np.asarray(rows, dtype=self.dtype)
        if arr.size == 0:
            return
        chunk_path = os.path.join(self._tmpdir, f'chunk_{len(self._chunks):04d}.npy')
        np.save(chunk_path, arr, allow_pickle=False)
        self._chunks.append((chunk_path, arr.shape[0]))
        self._total += arr.shape[0]

    def _ensure_combined(self):
        if self._final_path is not None:
            return
        if self._total == 0:
            empty_path = os.path.join(self._tmpdir, 'empty.npy')
            np.save(empty_path, np.empty(0, dtype=self.dtype), allow_pickle=False)
            self._final_path = empty_path
            self._chunks.clear()
            return

        combined_path = os.path.join(self._tmpdir, 'combined.npy')
        dest = np.lib.format.open_memmap(combined_path,
                                         mode='w+',
                                         dtype=self.dtype,
                                         shape=(self._total,))
        offset = 0
        for chunk_path, length in self._chunks:
            chunk = np.load(chunk_path, mmap_mode='r')
            dest[offset:offset+length] = chunk
            offset += length
            del chunk
            os.remove(chunk_path)
        dest.flush()
        del dest
        self._chunks.clear()
        self._final_path = combined_path

    def as_array(self):
        """
        Return the data as a read-only memmap array with dtype ``self.dtype``.
        """
        self._ensure_combined()
        return np.load(self._final_path, mmap_mode='r')

    def cleanup(self):
        """
        Remove all temporary files created by the store.
        """
        try:
            shutil.rmtree(self._tmpdir)
        except Exception:
            pass


def _ascii_fill(value, size, dtype='S24'):
    """
    Return an array of length ``size`` filled with ``value`` encoded as ASCII.
    """
    target_dtype = np.dtype(dtype)
    arr = np.empty(size, dtype=target_dtype)
    encoded = str(value).encode('ascii', errors='ignore')
    arr[...] = encoded
    return arr
#! --- helpers for memory-mapped storage of intermediate results ---


def _to_tracer_text(value):
    """
    Decode tracer values to plain Python strings.
    
    Args:
        value: Input value, possibly bytes or bytearray.
    Returns:
        str: Decoded and stripped string, or empty string if input is None or empty.
    Raises:
        UnicodeDecodeError: If decoding fails for both UTF-8 and Latin-1.
    """

    if isinstance(value, (bytes, bytearray)):
        try:
            return value.decode('utf-8', errors='ignore').strip()
        except Exception:
            return value.decode('latin-1', errors='ignore').strip()
    return str(value).strip()


def _normalize_tracertype_label(value):
    """
    Normalize tracer type labels by removing '_DATA' or '_RAND' suffixes.
    
    Args:
        value: Input tracer type label.
    Returns:
        str: Normalized tracer prefix without suffixes.
    """

    text = _to_tracer_text(value)
    if not text:
        return ''
    head, sep, tail = text.rpartition('_')
    if sep and tail.upper() in {'DATA', 'RAND'}:
        return head
    return text


def _gp_init_worker(tids, rand_sub, coords, is_data, tracer, tracer_id):
    """
    Initialise shared worker state for multiprocessing jobs.

    Args:
        tids (np.ndarray): Target identifiers.
        rand_sub (np.ndarray): Random iteration numbers for each row.
        coords (np.ndarray): Cartesian coordinates with shape ``(N, 3)``.
        is_data (np.ndarray): Boolean mask identifying real data rows.
        tracer (str): Tracer label associated with the block.
        tracer_id (int): Integer tracer identifier.
    """
    global _GP_SHARED
    _GP_SHARED = (tids, rand_sub, coords, is_data, tracer, int(tracer_id))


def extract_tracer_blocks(tbl):
    """
    Group table rows by tracer prefix and return per-tracer blocks.

    Args:
        tbl (Table): Input table containing ``TRACERTYPE`` and coordinate columns.
    Returns:
        dict: Mapping from tracer prefix to arrays of IDs, iterations, coordinates,
        and data masks.
    """
    randiters = np.asarray(tbl['RANDITER'], dtype=np.int32)
    targetids = np.asarray(tbl['TARGETID'], dtype=np.int64)
    coords_all = np.column_stack((np.asarray(tbl['XCART'], dtype=np.float32),
                                  np.asarray(tbl['YCART'], dtype=np.float32),
                                  np.asarray(tbl['ZCART'], dtype=np.float32)))

    tracer_ids = None
    if 'TRACER_ID' in tbl.colnames:
        tracer_ids = np.asarray(tbl['TRACER_ID'], dtype=np.int32)
        unique_ids = np.unique(tracer_ids)
        prefixes = []
        label_by_id = {}
        for tid in unique_ids:
            label = _TRACER_ID_TO_NAME.get(int(tid))
            if label is None and 'TRACERTYPE' in tbl.colnames and len(tbl):
                # Fallback: derive from first matching entry.
                first_idx = int(np.nonzero(tracer_ids == tid)[0][0])
                first_label = _to_tracer_text(tbl['TRACERTYPE'][first_idx])
                label = _normalize_tracertype_label(first_label)
            if label is not None:
                label_by_id[int(tid)] = label
                prefixes.append(int(tid))
    else:
        tracertypes = tbl['TRACERTYPE'].astype(str)
        base_tracer = np.asarray([t.rsplit('_', 1)[0] for t in tracertypes], dtype=object)
        prefixes = np.unique(base_tracer)
        label_by_id = None

    blocks = {}
    if tracer_ids is not None and len(prefixes):
        for tracer_id in prefixes:
            mask = (tracer_ids == tracer_id)
            idxs = np.nonzero(mask)[0]
            label = label_by_id.get(int(tracer_id), str(tracer_id))
            blocks[label] = {'tids': targetids[idxs],
                             'rand': randiters[idxs],
                             'coords': coords_all[idxs],
                             'is_data': randiters[idxs] == -1,
                             'label': label,
                             'tracer_id': int(tracer_id)}
    else:
        tracertypes = tbl['TRACERTYPE'].astype(str)
        base_tracer = np.asarray([t.rsplit('_', 1)[0] for t in tracertypes], dtype=object)
        for tracer in prefixes:
            mask = (base_tracer == tracer)
            idxs = np.nonzero(mask)[0]
            blocks[tracer] = {'tids': targetids[idxs],
                              'rand': randiters[idxs],
                              'coords': coords_all[idxs],
                              'is_data': randiters[idxs] == -1,
                              'label': tracer,
                              'tracer_id': _tracer_id_from_label(tracer)}
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


def process_delaunay(pts, tids, is_data, iteration, tracer, tracer_id=None):
    """
    Generate pair and classification arrays for a single tracer iteration.

    Args:
        pts (np.ndarray): Cartesian coordinates with shape ``(N, 3)``.
        tids (np.ndarray): Target identifiers aligned with ``pts``.
        is_data (np.ndarray): Boolean mask identifying data rows.
        iteration (int): Random iteration identifier.
        tracer (str): Tracer prefix.
    Returns:
        tuple[np.ndarray, np.ndarray]: Pair rows and classification rows as structured arrays.
    """
    pairs = compute_delaunay_pairs(pts)
    n = len(tids)

    if pairs.size == 0:
        pair_rows = np.empty(0, dtype=_PAIR_ROW_DTYPE)
        total_count = np.zeros(n, dtype=np.int32)
        data_count = np.zeros(n, dtype=np.int32)
    else:
        idx0, idx1 = pairs[:, 0], pairs[:, 1]
        total_count = np.bincount(idx0, minlength=n) + np.bincount(idx1, minlength=n)
        mask0 = is_data[idx0]
        mask1 = is_data[idx1]
        data_count = (np.bincount(idx0[mask1], minlength=n) +
                      np.bincount(idx1[mask0], minlength=n))

        pair_rows = np.empty(pairs.shape[0], dtype=_PAIR_ROW_DTYPE)
        pair_rows['TARGETID1'] = tids[idx0].astype(np.int64, copy=False)
        pair_rows['TARGETID2'] = tids[idx1].astype(np.int64, copy=False)
        pair_rows['RANDITER'] = iteration

    class_rows = np.empty(n, dtype=_CLASS_ROW_DTYPE)
    class_rows['TARGETID'] = tids.astype(np.int64, copy=False)
    class_rows['RANDITER'] = iteration
    class_rows['ISDATA'] = is_data.astype(bool, copy=False)
    class_rows['NDATA'] = data_count.astype(np.int32, copy=False)
    class_rows['NRAND'] = (total_count - data_count).astype(np.int32, copy=False)
    tracer_code = tracer_id if tracer_id is not None and tracer_id >= 0 else _tracer_id_from_label(tracer)
    if tracer_code < 0:
        tracer_code = 255
    class_rows['TRACER_ID'] = np.full(n, tracer_code, dtype=np.uint8)
    class_rows['TRACERTYPE'] = _ascii_fill(tracer, n, dtype=_TRACER_ASCII_DTYPE)

    return pair_rows, class_rows


def _gp_process_iter(j):
    """
    Process a single random iteration within a worker process.

    Args:
        j (int): Random iteration identifier.
    Returns:
        tuple[np.ndarray | None, np.ndarray | None]: Pair and classification rows.
    """
    tids, rand_sub, coords, is_data, tracer, tracer_id = _GP_SHARED
    print(f'[astra] tracer={tracer} iter={j}', flush=True)
    mask = is_data | (rand_sub == j)
    if not mask.any():
        return None, None
    return process_delaunay(coords[mask], tids[mask], is_data[mask], j, tracer,
                            tracer_id=tracer_id)


def generate_pairs(tbl, n_random, n_jobs=None, spill_dir=None):
    """
    Run the pair-generation pipeline for all tracers in ``tbl``.

    Args:
        tbl (Table): Input table containing data and random catalogues.
        n_random (int): Total number of random iterations available.
        n_jobs (int, optional): Parallel worker count; defaults to available CPUs.
        spill_dir (str | None): Optional directory where temporary chunks are stored.
    Returns:
        tuple[TempTableStore, TempTableStore, dict]: Disk-backed pair and classification
        stores plus an (empty) placeholder for backwards compatibility.
    """
    pair_store = TempTableStore(_PAIR_ROW_DTYPE, prefix='pairs', base_dir=spill_dir)
    class_store = TempTableStore(_CLASS_ROW_DTYPE, prefix='class', base_dir=spill_dir)

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
        tids, rand_sub, coords, is_data, tracer_label = (
            data['tids'],
            data['rand'],
            data['coords'],
            data['is_data'],
            data['label'],
        )
        tracer_id = data.get('tracer_id', _tracer_id_from_label(tracer_label))

        if n_jobs > 1:
            with mp.get_context('fork').Pool(processes=n_jobs,
                                             initializer=_gp_init_worker,
                                             initargs=(tids, rand_sub, coords, is_data,
                                                       tracer_label, tracer_id)) as pool:
                for pr, cr in pool.imap_unordered(_gp_process_iter, range(n_random)):
                    if pr is not None and pr.size:
                        pair_store.append(pr)
                    if cr is not None and cr.size:
                        class_store.append(cr)
        else:
            for j in range(n_random):
                print(f'[astra] tracer={tracer_label} iter={j}', flush=True)
                mask = is_data | (rand_sub == j)
                if not mask.any():
                    continue
                pr, cr = process_delaunay(coords[mask], tids[mask], is_data[mask], j,
                                          tracer_label, tracer_id=tracer_id)
                if pr.size:
                    pair_store.append(pr)
                if cr.size:
                    class_store.append(cr)

    return pair_store, class_store, {}


def _coerce_structured_rows(rows, dtype):
    """
    Convert ``rows`` to a structured numpy array with dtype ``dtype`` without unnecessary copies.
    
    Args:
        rows: Sequence or disk-backed store of structured rows.
        dtype: Target numpy structured dtype.
    Returns:
        np.ndarray: Structured numpy array with the specified dtype.
    """
    if isinstance(rows, TempTableStore):
        arr = rows.as_array()
        if arr.dtype != dtype:
            arr = arr.astype(dtype, copy=False)
        return arr
    if isinstance(rows, np.memmap):
        if rows.dtype == dtype:
            return rows
        return rows.astype(dtype, copy=False)
    if isinstance(rows, np.ndarray):
        if rows.dtype == dtype:
            return rows
        return rows.astype(dtype, copy=False)
    return np.asarray(rows, dtype=dtype)


def build_pairs_table(rows):
    """
    Construct a pairs table from ``rows``.

    Args:
        rows: Sequence or disk-backed store of pair rows.
    Returns:
        Table: Table with ``TARGETID1``, ``TARGETID2``, and ``RANDITER`` columns.
    """
    return Table(_coerce_structured_rows(rows, _PAIR_ROW_DTYPE), copy=False)


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
    if 'TRACER_ID' in tbl.colnames:
        tracer_ids = np.asarray(tbl['TRACER_ID'], dtype=np.uint8)
    else:
        tracer_ids = np.full(tids.size, 255, dtype=np.uint8)
    trtype = np.asarray(tbl['TRACERTYPE']).astype('U24') if 'TRACERTYPE' in tbl.colnames else np.array(
        [_TRACER_ID_TO_NAME.get(int(code), 'UNKNOWN') for code in tracer_ids], dtype='U24')

    is_data_map = {int(t): (ri == -1) for t, ri in zip(tids, randiter)}
    tracer_map = {int(t): str(tt) for t, tt in zip(tids, trtype)}
    tracer_id_map = {int(t): int(code) for t, code in zip(tids, tracer_ids)}

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
            class_rows.append((int(t), int(j), True, nd, nr,
                               tracer_id_map.get(int(t), 255),
                               tracer_map.get(int(t), 'UNKNOWN')))

        rids = rand_ids_by_j.get(j, np.empty(0, dtype=np.int64))
        for t in rids.tolist():
            nd = int(ndata_map.get(int(t), 0))
            tt = int(total_map.get(int(t), 0))
            nr = tt - nd
            class_rows.append((int(t), int(j), False, nd, nr,
                               tracer_id_map.get(int(t), 255),
                               tracer_map.get(int(t), 'UNKNOWN')))

    return class_rows


def build_class_table(rows):
    """
    Construct the classification table from tuple rows.

    Args:
        rows: Sequence or disk-backed store of classification rows.
    Returns:
        Table: Table with ``TARGETID``, ``RANDITER``, ``ISDATA``, ``NDATA``, ``NRAND``,
        and ``TRACERTYPE`` columns.
    """
    arr = _coerce_structured_rows(rows, _CLASS_ROW_DTYPE)
    tbl = Table(arr, copy=False)
    if tbl['TRACERTYPE'].dtype.kind == 'S':
        tbl['TRACERTYPE'] = tbl['TRACERTYPE'].astype('U24')
    return tbl


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


def build_probability_table(class_rows, raw_table, r_lower=-0.9, r_upper=0.9):
    """
    Build a probability table aligned with the raw table rows.

    Args:
        class_rows (list): Classification tuples.
        raw_table (Table): Raw table used to generate ``class_rows``.
        r_lower (float): Lower ``r`` threshold (negative).
        r_upper (float): Upper ``r`` threshold (positive).
    Returns:
        Table: Probability table containing probabilities for each row in ``raw_table``.
    Raises:
        ValueError: If the thresholds do not straddle zero.
    """
    if r_lower >= 0 or r_upper <= 0:
        raise ValueError('r_lower must be negative and r_upper must be positive.')

    arr = _coerce_structured_rows(class_rows, _CLASS_ROW_DTYPE)

    raw_len = len(raw_table)
    target_raw = np.asarray(raw_table['TARGETID'], dtype=np.int64)
    randiter_raw = (np.asarray(raw_table['RANDITER'], dtype=np.int32)
                    if 'RANDITER' in raw_table.colnames
                    else np.full(raw_len, -1, dtype=np.int32))
    use_tracer_id = ('TRACER_ID' in raw_table.colnames) and ('TRACER_ID' in arr.dtype.names)

    if not use_tracer_id:
        tracers_raw = (np.array([_to_tracer_text(v) for v in raw_table['TRACERTYPE']], dtype='U32')
                       if raw_len else np.asarray([], dtype='U32'))
        isdata_raw = randiter_raw == -1

        if arr.size == 0:
            zeros = np.zeros(raw_len, dtype=np.float32)
            return Table({'TARGETID': target_raw,
                          'RANDITER': randiter_raw,
                          'ISDATA': isdata_raw,
                          'TRACERTYPE': tracers_raw,
                          'PVOID': zeros.copy(),
                          'PSHEET': zeros.copy(),
                          'PFILAMENT': zeros.copy(),
                          'PKNOT': zeros.copy()})

        ndata = arr['NDATA'].astype(np.float64, copy=False)
        nrand = arr['NRAND'].astype(np.float64, copy=False)
        denom = ndata + nrand
        r = np.zeros_like(denom, dtype=np.float64)
        np.divide(ndata - nrand, denom, out=r, where=(denom > 0))

        classes = np.digitize(r, bins=[r_lower, 0.0, r_upper])
        classes = np.clip(classes, 0, 3)

        mask_data = arr['ISDATA']
        data_map = {}
        if mask_data.any():
            data_keys = np.empty(mask_data.sum(), dtype=[('TARGETID', 'i8'), ('TRACERTYPE', 'U24')])
            data_keys['TARGETID'] = arr['TARGETID'][mask_data]
            data_keys['TRACERTYPE'] = arr['TRACERTYPE'][mask_data]
            uniq_data, inv_data = np.unique(data_keys, return_inverse=True)
            counts_data = np.zeros((uniq_data.size, 4), dtype=np.int64)
            np.add.at(counts_data, (inv_data, classes[mask_data]), 1)
            totals_data = counts_data.sum(axis=1, keepdims=True).astype(np.float32)
            probs_data = counts_data.astype(np.float32)
            np.divide(probs_data, totals_data, out=probs_data, where=(totals_data > 0))
            data_map = {(int(tid), _to_tracer_text(tracer)): probs_data[idx]
                        for idx, (tid, tracer) in enumerate(zip(uniq_data['TARGETID'], uniq_data['TRACERTYPE']))}

        mask_rand = ~mask_data
        rand_map = {}
        if mask_rand.any():
            rand_keys = np.empty(mask_rand.sum(), dtype=[('TARGETID', 'i8'), ('TRACERTYPE', 'U24'), ('RANDITER', 'i4')])
            rand_keys['TARGETID'] = arr['TARGETID'][mask_rand]
            rand_keys['TRACERTYPE'] = arr['TRACERTYPE'][mask_rand]
            rand_keys['RANDITER'] = arr['RANDITER'][mask_rand]
            uniq_rand, inv_rand = np.unique(rand_keys, return_inverse=True)
            counts_rand = np.zeros((uniq_rand.size, 4), dtype=np.int64)
            np.add.at(counts_rand, (inv_rand, classes[mask_rand]), 1)
            totals_rand = counts_rand.sum(axis=1, keepdims=True).astype(np.float32)
            probs_rand = counts_rand.astype(np.float32)
            np.divide(probs_rand, totals_rand, out=probs_rand, where=(totals_rand > 0))
            rand_map = {(int(tid), _to_tracer_text(tracer), int(rj)): probs_rand[idx]
                        for idx, (tid, tracer, rj) in enumerate(zip(uniq_rand['TARGETID'],
                                                                   uniq_rand['TRACERTYPE'],
                                                                   uniq_rand['RANDITER']))}

        pvoid = np.zeros(raw_len, dtype=np.float32)
        psheet = np.zeros(raw_len, dtype=np.float32)
        pfilament = np.zeros(raw_len, dtype=np.float32)
        pknot = np.zeros(raw_len, dtype=np.float32)

        for idx in range(raw_len):
            tid = int(target_raw[idx])
            tracer_full = tracers_raw[idx]
            base = _normalize_tracertype_label(tracer_full)
            if isdata_raw[idx]:
                probs = data_map.get((tid, base))
            else:
                probs = rand_map.get((tid, base, int(randiter_raw[idx])))
            if probs is not None:
                pvoid[idx], psheet[idx], pfilament[idx], pknot[idx] = probs

        return Table({'TARGETID': target_raw,
                      'RANDITER': randiter_raw,
                      'ISDATA': isdata_raw,
                      'TRACERTYPE': tracers_raw,
                      'PVOID': pvoid,
                      'PSHEET': psheet,
                      'PFILAMENT': pfilament,
                      'PKNOT': pknot})

    tracer_id_raw = np.asarray(raw_table['TRACER_ID'], dtype=np.uint8)
    isdata_raw = randiter_raw == -1

    if arr.size == 0:
        zeros = np.zeros(raw_len, dtype=np.float32)
        tracer_strings = np.full(raw_len, b'UNKNOWN', dtype=_TRACER_ASCII_DTYPE)
        for tid in np.unique(tracer_id_raw):
            mask = tracer_id_raw == tid
            if not mask.any():
                continue
            data_mask = mask & isdata_raw
            rand_mask = mask & ~isdata_raw
            if data_mask.any():
                tracer_strings[data_mask] = _full_tracer_label(int(tid), True)
            if rand_mask.any():
                tracer_strings[rand_mask] = _full_tracer_label(int(tid), False)
        return Table({'TARGETID': target_raw,
                      'RANDITER': randiter_raw,
                      'ISDATA': isdata_raw,
                      'TRACER_ID': tracer_id_raw,
                      'TRACERTYPE': tracer_strings.astype('U24'),
                      'PVOID': zeros.copy(),
                      'PSHEET': zeros.copy(),
                      'PFILAMENT': zeros.copy(),
                      'PKNOT': zeros.copy()})

    ndata = arr['NDATA'].astype(np.float64, copy=False)
    nrand = arr['NRAND'].astype(np.float64, copy=False)
    denom = ndata + nrand
    r = np.zeros_like(denom, dtype=np.float64)
    np.divide(ndata - nrand, denom, out=r, where=(denom > 0))

    classes = np.digitize(r, bins=[r_lower, 0.0, r_upper])
    classes = np.clip(classes, 0, 3)

    mask_data = arr['ISDATA']
    tracer_id_arr = arr['TRACER_ID']

    if mask_data.any():
        data_keys = np.empty(mask_data.sum(), dtype=[('TARGETID', 'i8'), ('TRACER_ID', 'u1')])
        data_keys['TARGETID'] = arr['TARGETID'][mask_data]
        data_keys['TRACER_ID'] = tracer_id_arr[mask_data]
        uniq_data, inv_data = np.unique(data_keys, return_inverse=True)
        counts_data = np.zeros((uniq_data.size, 4), dtype=np.int64)
        np.add.at(counts_data, (inv_data, classes[mask_data]), 1)
        totals_data = counts_data.sum(axis=1, keepdims=True).astype(np.float32)
        probs_data = counts_data.astype(np.float32)
        np.divide(probs_data, totals_data, out=probs_data, where=(totals_data > 0))
    else:
        uniq_data = np.empty(0, dtype=[('TARGETID', 'i8'), ('TRACER_ID', 'u1')])
        probs_data = np.empty((0, 4), dtype=np.float32)

    mask_rand = ~mask_data
    if mask_rand.any():
        rand_keys = np.empty(mask_rand.sum(), dtype=[('TARGETID', 'i8'), ('TRACER_ID', 'u1'), ('RANDITER', 'i4')])
        rand_keys['TARGETID'] = arr['TARGETID'][mask_rand]
        rand_keys['TRACER_ID'] = tracer_id_arr[mask_rand]
        rand_keys['RANDITER'] = arr['RANDITER'][mask_rand]
        uniq_rand, inv_rand = np.unique(rand_keys, return_inverse=True)
        counts_rand = np.zeros((uniq_rand.size, 4), dtype=np.int64)
        np.add.at(counts_rand, (inv_rand, classes[mask_rand]), 1)
        totals_rand = counts_rand.sum(axis=1, keepdims=True).astype(np.float32)
        probs_rand = counts_rand.astype(np.float32)
        np.divide(probs_rand, totals_rand, out=probs_rand, where=(totals_rand > 0))
    else:
        uniq_rand = np.empty(0, dtype=[('TARGETID', 'i8'), ('TRACER_ID', 'u1'), ('RANDITER', 'i4')])
        probs_rand = np.empty((0, 4), dtype=np.float32)

    pvoid = np.zeros(raw_len, dtype=np.float32)
    psheet = np.zeros(raw_len, dtype=np.float32)
    pfilament = np.zeros(raw_len, dtype=np.float32)
    pknot = np.zeros(raw_len, dtype=np.float32)

    if uniq_data.size:
        raw_data_mask = isdata_raw
        if raw_data_mask.any():
            raw_data_keys = np.empty(raw_data_mask.sum(), dtype=uniq_data.dtype)
            raw_data_keys['TARGETID'] = target_raw[raw_data_mask]
            raw_data_keys['TRACER_ID'] = tracer_id_raw[raw_data_mask]
            pos = np.searchsorted(uniq_data, raw_data_keys)
            valid = pos < uniq_data.size
            valid_idx = np.nonzero(valid)[0]
            if valid_idx.size:
                same = uniq_data[pos[valid_idx]] == raw_data_keys[valid_idx]
                valid_idx = valid_idx[same]
                if valid_idx.size:
                    idxs = np.where(raw_data_mask)[0][valid_idx]
                    pos_valid = pos[valid_idx]
                    pvoid[idxs] = probs_data[pos_valid, 0]
                    psheet[idxs] = probs_data[pos_valid, 1]
                    pfilament[idxs] = probs_data[pos_valid, 2]
                    pknot[idxs] = probs_data[pos_valid, 3]

    if uniq_rand.size:
        raw_rand_mask = ~isdata_raw
        if raw_rand_mask.any():
            raw_rand_keys = np.empty(raw_rand_mask.sum(), dtype=uniq_rand.dtype)
            raw_rand_keys['TARGETID'] = target_raw[raw_rand_mask]
            raw_rand_keys['TRACER_ID'] = tracer_id_raw[raw_rand_mask]
            raw_rand_keys['RANDITER'] = randiter_raw[raw_rand_mask]
            pos = np.searchsorted(uniq_rand, raw_rand_keys)
            valid = pos < uniq_rand.size
            valid_idx = np.nonzero(valid)[0]
            if valid_idx.size:
                same = uniq_rand[pos[valid_idx]] == raw_rand_keys[valid_idx]
                valid_idx = valid_idx[same]
                if valid_idx.size:
                    idxs = np.where(raw_rand_mask)[0][valid_idx]
                    pos_valid = pos[valid_idx]
                    pvoid[idxs] = probs_rand[pos_valid, 0]
                    psheet[idxs] = probs_rand[pos_valid, 1]
                    pfilament[idxs] = probs_rand[pos_valid, 2]
                    pknot[idxs] = probs_rand[pos_valid, 3]

    tracer_strings = np.full(raw_len, b'UNKNOWN', dtype=_TRACER_ASCII_DTYPE)
    for tid in np.unique(tracer_id_raw):
        mask = tracer_id_raw == tid
        if not mask.any():
            continue
        data_mask = mask & isdata_raw
        rand_mask = mask & ~isdata_raw
        if data_mask.any():
            tracer_strings[data_mask] = _full_tracer_label(int(tid), True)
        if rand_mask.any():
            tracer_strings[rand_mask] = _full_tracer_label(int(tid), False)

    return Table({'TARGETID': target_raw,
                  'RANDITER': randiter_raw,
                  'ISDATA': isdata_raw,
                  'TRACER_ID': tracer_id_raw,
                  'TRACERTYPE': tracer_strings.astype('U24'),
                  'PVOID': pvoid,
                  'PSHEET': psheet,
                  'PFILAMENT': pfilament,
                  'PKNOT': pknot})

def save_probability_fits(class_rows, raw_table, output_path, r_lower=-0.9, r_upper=0.9, meta=None):
    """
    Saves the probability table to a FITS file.

    Args:
        class_rows (list): List of tuples containing classification data.
        output_path (str): Path to save the FITS file.
        r_lower (float, optional): Lower ``r`` threshold (default: -0.9).
        r_upper (float, optional): Upper ``r`` threshold (default: 0.9).
        meta (dict | None): Optional metadata to inject into the FITS header.
    Raises:
        ValueError: If the thresholds do not straddle zero.
        TypeError: If the input types are incorrect.
        RuntimeError: If the FITS file cannot be written.
    """
    tbl = build_probability_table(class_rows, raw_table, r_lower=r_lower, r_upper=r_upper)
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