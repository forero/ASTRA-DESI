import gc
import multiprocessing as mp
import os
import shutil
import tempfile

import numpy as np
from astropy.io import fits
from astropy.table import Table
from scipy.spatial import Delaunay

__all__ = ['TempTableStore',
           'register_tracer_mapping',
           'generate_pairs',
           'save_pairs_fits',
           'load_pairs_fits',
           'build_class_rows_from_pairs',
           'save_classification_fits',
           'build_probability_table',
           'save_probability_fits',
           'PAIR_ROW_DTYPE',
           'CLASS_ROW_DTYPE',
           'PROB_ROW_DTYPE']

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
_PROB_ROW_DTYPE = np.dtype([('TARGETID', np.int64),
                            ('RANDITER', np.int32),
                            ('ISDATA', np.bool_),
                            ('TRACER_ID', np.uint8),
                            ('TRACERTYPE', 'S24'),
                            ('PVOID', np.float32),
                            ('PSHEET', np.float32),
                            ('PFILAMENT', np.float32),
                            ('PKNOT', np.float32)])

_DATA_KEY_DTYPE = np.dtype([('TRACER_ID', np.int16), ('TARGETID', np.int64)])

try:
    _DEFAULT_CHUNK_ROWS = max(1, int(os.environ.get('ASTRA_CHUNK_ROWS', '1000000')))
except Exception:
    _DEFAULT_CHUNK_ROWS = 1_000_000

PAIR_ROW_DTYPE = _PAIR_ROW_DTYPE
CLASS_ROW_DTYPE = _CLASS_ROW_DTYPE
PROB_ROW_DTYPE = _PROB_ROW_DTYPE

_TRACER_NAME_TO_ID = {}
_TRACER_ID_TO_NAME = {}
_TRACER_ID_TO_FULL = {}


def _bool_env(name, default=False):
    """
    Return the boolean value of an environment variable.
    
    Args:
        name (str): Name of the environment variable.
        default (bool): Default value if the variable is not set.
    Returns:
        bool: Boolean value of the environment variable.
    """
    value = os.environ.get(name)
    if value is None:
        return default
    return str(value).strip().lower() in {'1', 'true', 'yes', 'on'}


def _split_iter_path(base_path, iteration):
    """
    Generate a path for a specific iteration based on a base path.
    
    Args:
        base_path (str): The base file path.
        iteration (int or str): The iteration identifier.
    Returns:
        str: The generated file path for the given iteration.
    """
    root, ext = os.path.splitext(base_path)
    if ext == '.gz':
        root2, ext2 = os.path.splitext(root)
        base_root = root2
        extension = ext2 + ext
    else:
        base_root = root
        extension = ext

    if base_root.endswith('_classified'):
        base_root = base_root[:-len('_classified')]

    tag = iteration
    if not isinstance(tag, str):
        val = int(tag)
        tag = f"{val:03d}" if val >= 0 else f"m{abs(val):03d}"
    return f"{base_root}_iter{tag}{extension}"


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
    
    Args:
        label (str): Tracer base label.
    Returns:
        int: Integer tracer ID, or -1 if not found.
    """
    return _TRACER_NAME_TO_ID.get(str(label), -1)


def _full_tracer_label(tracer_id, is_data) -> bytes:
    """
    Return the full tracer label for a given tracer ID and data/random flag.
    
    Args:
        tracer_id (int): Integer tracer ID.
        is_data (bool): Flag indicating data (True) or random (False).
    Returns:
        bytes: Full tracer label as ASCII bytes.
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

    @classmethod
    def from_directory(cls, directory, dtype):
        """
        Reconstruct a store from an existing spill directory produced during a previous run.

        Args:
            directory (str): Path to the spill directory containing chunk_XXXX.npy files
                             and/or a combined.npy file.
            dtype (np.dtype): Structured dtype describing the stored rows.
        Returns:
            TempTableStore: Rehydrated store referencing the on-disk chunks.
        """
        obj = cls.__new__(cls)
        obj.dtype = np.dtype(dtype)
        obj._tmpdir = os.path.abspath(directory)
        obj._chunks = []
        obj._total = 0
        obj._final_path = None

        combined = os.path.join(obj._tmpdir, 'combined.npy')
        if os.path.exists(combined):
            mm = np.load(combined, mmap_mode='r')
            obj._total = mm.shape[0]
            obj._final_path = combined
            del mm
        if obj._final_path is None:
            chunk_files = sorted(f for f in os.listdir(obj._tmpdir)
                                 if f.startswith('chunk_') and f.endswith('.npy'))
            for fname in chunk_files:
                path = os.path.join(obj._tmpdir, fname)
                mm = np.load(path, mmap_mode='r')
                obj._chunks.append((path, mm.shape[0]))
                obj._total += mm.shape[0]
                del mm
        if obj._total == 0 and obj._final_path is None:
            raise FileNotFoundError(f'No chunk files found in spill directory {directory}')
        return obj

    @property
    def total(self):
        return self._total

    @property
    def tmpdir(self):
        return self._tmpdir

    def iter_arrays(self, chunk_rows=None):
        """
        Yield numpy arrays (memmap views) covering the stored data.

        Args:
            chunk_rows (int | None): Optional maximum rows per yielded chunk. When
                provided, large chunks are subdivided into smaller slices.
        Yields:
            np.ndarray: View over a portion of the stored rows.
        """
        limit = int(chunk_rows) if chunk_rows else None
        if self._final_path is not None and not self._chunks:
            arr = np.load(self._final_path, mmap_mode='r')
            if limit and limit > 0:
                for start in range(0, arr.shape[0], limit):
                    yield arr[start:start+limit]
            else:
                yield arr
            return

        for chunk_path, length in self._chunks:
            arr = np.load(chunk_path, mmap_mode='r')
            if limit and limit > 0 and length > limit:
                for start in range(0, length, limit):
                    yield arr[start:start+limit]
            else:
                yield arr

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
        """
        Combine all existing chunks into a single memmap file if not already done.
        """
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
    if spill_dir is None:
        spill_dir = (os.environ.get('ASTRA_PAIR_SPILL_DIR')
                     or os.environ.get('ASTRA_TMPDIR')
                     or os.environ.get('PSCRATCH')
                     or os.environ.get('TMPDIR'))

    pair_store = TempTableStore(_PAIR_ROW_DTYPE, prefix='pairs', base_dir=spill_dir)
    class_store = TempTableStore(_CLASS_ROW_DTYPE, prefix='class', base_dir=spill_dir)
    print(f'[astra] pair chunks -> {pair_store.tmpdir}')
    print(f'[astra] class chunks -> {class_store.tmpdir}')

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
        tids, rand_sub, coords, is_data, tracer_label = (data['tids'],
                                                         data['rand'],
                                                         data['coords'],
                                                         data['is_data'],
                                                         data['label'])
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


def _iter_structured_chunks(rows, dtype, chunk_rows=None):
    """
    Return an iterator over structured arrays for ``rows`` with optional chunking.
    
    Args:
        rows: Sequence or disk-backed store of structured rows.
        dtype: Target numpy structured dtype.
        chunk_rows: Optional number of rows per chunk.
    Returns:
        Iterator over structured arrays and total number of rows.
    """
    if isinstance(rows, TempTableStore):
        return rows.iter_arrays(chunk_rows), rows.total
    arr = _coerce_structured_rows(rows, dtype)
    total = len(arr)
    if chunk_rows and chunk_rows > 0 and total > chunk_rows:
        def _gen():
            for start in range(0, total, chunk_rows):
                yield arr[start:start+chunk_rows]
        return _gen(), total
    else:
        def _gen_single():
            if total:
                yield arr
        return _gen_single(), total


def _write_fits_table(columns, total_rows, chunk_iter, output_path, meta=None):
    """
    Write a FITS binary table from chunked structured arrays.

    Args:
        columns (list): List of column definitions.
        total_rows (int): Total number of rows in the table.
        chunk_iter (iterator): Iterator over chunks of structured arrays.
        output_path (str): Path to the output FITS file.
        meta (dict | None): Optional metadata inserted into the FITS header.
    """
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    coldefs = fits.ColDefs([fits.Column(name=name, format=fmt) for name, fmt in columns])
    hdu = fits.BinTableHDU.from_columns(coldefs, nrows=int(total_rows))
    if meta:
        for key, value in meta.items():
            try:
                hdu.header[key] = value
            except Exception:
                pass
    tmp = f"{output_path}.tmp"
    hdu.writeto(tmp, overwrite=True)
    with fits.open(tmp, mode='update', memmap=True) as hdul:
        data = hdul[1].data
        start = 0
        for chunk in chunk_iter:
            length = len(chunk)
            if length == 0:
                continue
            end = start + length
            for name, _ in columns:
                data[name][start:end] = chunk[name]
            start = end
        hdul.flush()
    os.replace(tmp, output_path)


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
    chunk_iter, total_rows = _iter_structured_chunks(rows, _PAIR_ROW_DTYPE, chunk_rows=_DEFAULT_CHUNK_ROWS)
    columns = (('TARGETID1', 'K'),
               ('TARGETID2', 'K'),
               ('RANDITER', 'J'))
    _write_fits_table(columns, total_rows, chunk_iter, output_path, meta=meta)


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


def build_class_rows_from_pairs(tbl, pairs_tbl, n_random, spill_dir=None):
    """
    Reconstruct classification rows from previously saved pairs.

    Args:
        tbl (Table): Raw table with ``TARGETID``, ``RANDITER``, and ``TRACERTYPE``.
        pairs_tbl (Table): Table containing pair information.
        n_random (int): Number of random iterations present in the dataset.
        spill_dir (str | None): Optional directory for temporary spill files.
    Returns:
        TempTableStore: Disk-backed store of classification rows.
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
    tracer_bytes_map = {int(t): str(tt).encode('ascii', errors='ignore')
                        for t, tt in tracer_map.items()}

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
    if spill_dir is None:
        spill_dir = (os.environ.get('ASTRA_PAIR_SPILL_DIR')
                     or os.environ.get('ASTRA_TMPDIR')
                     or os.environ.get('PSCRATCH')
                     or os.environ.get('TMPDIR'))
    store = TempTableStore(_CLASS_ROW_DTYPE, prefix='class_rebuild', base_dir=spill_dir)

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

        if data_ids.size:
            nd_vals = np.fromiter((ndata_map.get(int(t), 0) for t in data_ids),
                                  dtype=np.int32, count=data_ids.size)
            total_vals = np.fromiter((total_map.get(int(t), 0) for t in data_ids),
                                     dtype=np.int32, count=data_ids.size)
            rand_vals = total_vals - nd_vals

            arr_data = np.empty(data_ids.size, dtype=_CLASS_ROW_DTYPE)
            arr_data['TARGETID'] = data_ids
            arr_data['RANDITER'] = np.full(data_ids.size, j, dtype=np.int32)
            arr_data['ISDATA'] = True
            arr_data['NDATA'] = nd_vals
            arr_data['NRAND'] = rand_vals
            arr_data['TRACER_ID'] = np.fromiter((tracer_id_map.get(int(t), 255) for t in data_ids),
                                                dtype=np.int16, count=data_ids.size).astype(np.uint8, copy=False)
            arr_data['TRACERTYPE'] = np.asarray([
                tracer_bytes_map.get(int(t), b'UNKNOWN') for t in data_ids
            ], dtype='S24')
            store.append(arr_data)

        rids = rand_ids_by_j.get(j, np.empty(0, dtype=np.int64))
        if rids.size:
            nd_vals_r = np.fromiter((ndata_map.get(int(t), 0) for t in rids),
                                    dtype=np.int32, count=rids.size)
            total_vals_r = np.fromiter((total_map.get(int(t), 0) for t in rids),
                                       dtype=np.int32, count=rids.size)
            rand_vals_r = total_vals_r - nd_vals_r

            arr_rand = np.empty(rids.size, dtype=_CLASS_ROW_DTYPE)
            arr_rand['TARGETID'] = rids
            arr_rand['RANDITER'] = np.full(rids.size, j, dtype=np.int32)
            arr_rand['ISDATA'] = False
            arr_rand['NDATA'] = nd_vals_r
            arr_rand['NRAND'] = rand_vals_r
            arr_rand['TRACER_ID'] = np.fromiter((tracer_id_map.get(int(t), 255) for t in rids),
                                                dtype=np.int16, count=rids.size).astype(np.uint8, copy=False)
            arr_rand['TRACERTYPE'] = np.asarray([
                tracer_bytes_map.get(int(t), b'UNKNOWN') for t in rids
            ], dtype='S24')
            store.append(arr_rand)

    return store


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
    chunk_iter, total_rows = _iter_structured_chunks(rows, _CLASS_ROW_DTYPE, chunk_rows=_DEFAULT_CHUNK_ROWS)
    columns = (('TARGETID', 'K'),
               ('RANDITER', 'J'),
               ('ISDATA', 'L'),
               ('NDATA', 'J'),
               ('NRAND', 'J'),
               ('TRACER_ID', 'B'),
               ('TRACERTYPE', '24A'))
    split_iter = _bool_env('ASTRA_CLASS_SPLIT_ITER', default=True)
    skip_combined = _bool_env('ASTRA_CLASS_SKIP_COMBINED', default=False)
    if not split_iter and skip_combined:
        return

    class _SplitCollector:
        __slots__ = ('base_iter', 'stores')

        def __init__(self, base_iter):
            self.base_iter = base_iter
            self.stores = {}

        def __iter__(self):
            for chunk in self.base_iter:
                if chunk.size == 0:
                    yield chunk
                    continue
                rand_vals = np.asarray(chunk['RANDITER'], dtype=np.int32, copy=False)
                unique_vals = np.unique(rand_vals)
                for val in unique_vals:
                    mask = (rand_vals == val)
                    if not mask.any():
                        continue
                    store = self.stores.get(int(val))
                    if store is None:
                        store = TempTableStore(_CLASS_ROW_DTYPE, prefix=f'class_iter_{int(val):03d}',
                                               base_dir=os.path.dirname(output_path))
                        self.stores[int(val)] = store
                    store.append(chunk[mask])
                yield chunk

    if split_iter:
        collector = _SplitCollector(chunk_iter)
        iterator = iter(collector)
    else:
        collector = None
        iterator = chunk_iter

    if not skip_combined:
        _write_fits_table(columns, total_rows, iterator, output_path, meta=meta)
    else:
        for _ in iterator:
            pass

    if split_iter and collector is not None:
        for iteration, store in collector.stores.items():
            iter_path = _split_iter_path(output_path, iteration)
            iter_chunk_iter, iter_total = _iter_structured_chunks(store, _CLASS_ROW_DTYPE, chunk_rows=_DEFAULT_CHUNK_ROWS)
            _write_fits_table(columns, iter_total, iter_chunk_iter, iter_path, meta=meta)
            store.cleanup()


def _prepare_dense_data_accumulator(raw_table):
    """
    Construct a dense accumulator for data-target probability counts using ``raw_table``.

    Args:
        raw_table (Table | None): Raw table containing TARGETID, RANDITER, and tracer columns.
    Returns:
        dict | None: Dense accumulator dictionary or ``None`` when not applicable.
    """
    if raw_table is None:
        return None
    required = {'TARGETID', 'RANDITER'}
    if not required.issubset(set(raw_table.colnames)):
        return None

    randiters = np.asarray(raw_table['RANDITER'])
    data_mask = randiters < 0
    if not np.any(data_mask):
        return None

    targetids = np.asarray(raw_table['TARGETID'][data_mask], dtype=np.int64)

    if 'TRACER_ID' in raw_table.colnames:
        tracer_ids = np.asarray(raw_table['TRACER_ID'][data_mask], dtype=np.int16)
    else:
        tracer_vals = np.asarray(raw_table['TRACERTYPE'][data_mask]).astype(str)
        tracer_ids = np.fromiter((_tracer_id_from_label(val) for val in tracer_vals),
                                 dtype=np.int16, count=tracer_vals.size)

    tracer_ids = np.asarray(tracer_ids, dtype=np.int16)
    if tracer_ids.size:
        neg_mask = tracer_ids < 0
        if np.any(neg_mask):
            tracer_ids = tracer_ids.copy()
            tracer_ids[neg_mask] = 255

    tracer_ids_clipped = np.clip(tracer_ids, 0, 255).astype(np.uint8, copy=False)

    keys = np.empty(targetids.size, dtype=_DATA_KEY_DTYPE)
    keys['TRACER_ID'] = tracer_ids.astype(np.int16, copy=False)
    keys['TARGETID'] = targetids
    if keys.size:
        sorter = np.argsort(keys, order=('TRACER_ID', 'TARGETID'))
        keys_sorted = keys[sorter]
        tracer_ids_sorted = tracer_ids_clipped[sorter]
    else:
        keys_sorted = keys
        tracer_ids_sorted = tracer_ids_clipped

    counts = np.zeros((keys_sorted.size, 4), dtype=np.uint16)
    return {'keys': keys_sorted,
            'counts': counts,
            'tracer_ids': tracer_ids_sorted}


def _finalize_dense_data_records(acc):
    """
    Convert a dense accumulator into probability data records.

    Args:
        acc (dict | None): Dense accumulator dictionary.
    Returns:
        np.ndarray: Structured probability rows for data targets.
    """
    if acc is None or acc['keys'].size == 0:
        return np.empty(0, dtype=_PROB_ROW_DTYPE)

    n = acc['keys'].shape[0]
    data_records = np.empty(n, dtype=_PROB_ROW_DTYPE)
    data_records['TARGETID'] = acc['keys']['TARGETID']
    data_records['RANDITER'] = -1
    data_records['ISDATA'] = True
    data_records['TRACER_ID'] = acc['tracer_ids']
    tracer_ids = acc['tracer_ids']
    labels = np.empty(n, dtype=_TRACER_ASCII_DTYPE)
    if n:
        unique_ids = np.unique(tracer_ids)
        for tid in unique_ids:
            mask = tracer_ids == tid
            labels[mask] = _full_tracer_label(int(tid), True)
    data_records['TRACERTYPE'] = labels
    data_records['PVOID'] = 0.0
    data_records['PSHEET'] = 0.0
    data_records['PFILAMENT'] = 0.0
    data_records['PKNOT'] = 0.0

    counts_float = acc['counts'].astype(np.float32, copy=False)
    totals = counts_float.sum(axis=1)
    nonzero = totals > 0
    if np.any(nonzero):
        nz_idx = np.nonzero(nonzero)[0]
        data_records['PVOID'][nz_idx] = counts_float[nz_idx, 0] / totals[nz_idx]
        data_records['PSHEET'][nz_idx] = counts_float[nz_idx, 1] / totals[nz_idx]
        data_records['PFILAMENT'][nz_idx] = counts_float[nz_idx, 2] / totals[nz_idx]
        data_records['PKNOT'][nz_idx] = counts_float[nz_idx, 3] / totals[nz_idx]

    acc['keys'] = None
    acc['counts'] = None
    acc['tracer_ids'] = None
    return data_records


def _iter_dense_data_chunks(acc, chunk_rows):
    """
    Yield probability records for dense accumulator data in ``chunk_rows`` slices.

    Args:
        acc (dict | None): Dense accumulator dictionary.
        chunk_rows (int): Maximum rows per yielded chunk.
    Yields:
        np.ndarray: Structured probability rows for data targets.
    """
    if acc is None:
        return
    keys = acc.get('keys')
    counts = acc.get('counts')
    tracer_ids = acc.get('tracer_ids')
    if keys is None or counts is None or tracer_ids is None:
        return

    total = keys.shape[0]
    if total == 0:
        return

    step = int(chunk_rows) if chunk_rows and chunk_rows > 0 else total
    try:
        for start in range(0, total, step):
            end = min(start + step, total)
            size = end - start
            chunk = np.empty(size, dtype=_PROB_ROW_DTYPE)
            chunk['TARGETID'] = keys['TARGETID'][start:end]
            chunk['RANDITER'] = -1
            chunk['ISDATA'] = True
            tracer_subset = tracer_ids[start:end]
            chunk['TRACER_ID'] = tracer_subset

            labels = np.empty(size, dtype=_TRACER_ASCII_DTYPE)
            if size:
                unique_ids = np.unique(tracer_subset)
                for tid in unique_ids:
                    mask = tracer_subset == tid
                    labels[mask] = _full_tracer_label(int(tid), True)
            chunk['TRACERTYPE'] = labels
            chunk['PVOID'] = 0.0
            chunk['PSHEET'] = 0.0
            chunk['PFILAMENT'] = 0.0
            chunk['PKNOT'] = 0.0

            counts_slice = counts[start:end].astype(np.float32, copy=False)
            totals = counts_slice.sum(axis=1)
            nonzero = totals > 0
            if np.any(nonzero):
                idx = np.nonzero(nonzero)[0]
                chunk['PVOID'][idx] = counts_slice[idx, 0] / totals[idx]
                chunk['PSHEET'][idx] = counts_slice[idx, 1] / totals[idx]
                chunk['PFILAMENT'][idx] = counts_slice[idx, 2] / totals[idx]
                chunk['PKNOT'][idx] = counts_slice[idx, 3] / totals[idx]
            yield chunk
    finally:
        acc['keys'] = None
        acc['counts'] = None
        acc['tracer_ids'] = None


def _build_data_records_from_dict(counts_dict, tracer_dict):
    """
    Convert a dictionary-based accumulator into probability data records.

    Args:
        counts_dict (dict): Mapping ``(TARGETID, TRACER_ID) -> counts``.
        tracer_dict (dict): Mapping ``(TARGETID, TRACER_ID) -> tracer bytes``.
    Returns:
        np.ndarray: Structured probability rows.
    """
    if not counts_dict:
        return np.empty(0, dtype=_PROB_ROW_DTYPE)

    data_records = np.empty(len(counts_dict), dtype=_PROB_ROW_DTYPE)
    data_records['RANDITER'] = -1
    data_records['ISDATA'] = True
    data_records['PVOID'] = 0.0
    data_records['PSHEET'] = 0.0
    data_records['PFILAMENT'] = 0.0
    data_records['PKNOT'] = 0.0
    for idx, (key, counts) in enumerate(counts_dict.items()):
        tid, tracer_id = key
        data_records[idx]['TARGETID'] = tid
        if 0 <= int(tracer_id) < 256:
            data_records[idx]['TRACER_ID'] = np.uint8(tracer_id)
        else:
            data_records[idx]['TRACER_ID'] = np.uint8(255)
        tracer_bytes = tracer_dict.get(key)
        if tracer_bytes is None:
            tracer_bytes = _full_tracer_label(tracer_id, True)
        data_records[idx]['TRACERTYPE'] = tracer_bytes
        counts_float = np.asarray(counts, dtype=np.float32)
        total = counts_float.sum()
        if total > 0:
            data_records[idx]['PVOID'] = counts_float[0] / total
            data_records[idx]['PSHEET'] = counts_float[1] / total
            data_records[idx]['PFILAMENT'] = counts_float[2] / total
            data_records[idx]['PKNOT'] = counts_float[3] / total
    return data_records


def _compute_probability_components(class_rows, raw_table=None, r_lower=-0.9, r_upper=0.9,
                                    chunk_rows=None, spill_dir=None):
    """
    Aggregate probability information from classification rows.
    
    Args:
        class_rows: Sequence or disk-backed store of classification rows.
        r_lower (float): Lower ratio threshold for classification.
        r_upper (float): Upper ratio threshold for classification.
        chunk_rows (int | None): Optional maximum rows per processing chunk.
        spill_dir (str | None): Optional directory where temporary chunks are stored.
    Returns:
        tuple[np.ndarray, TempTableStore]: Data records array and random records store.
    """
    if chunk_rows is None or chunk_rows <= 0:
        chunk_rows = _DEFAULT_CHUNK_ROWS
    if isinstance(class_rows, TempTableStore):
        chunk_iter = class_rows.iter_arrays(chunk_rows)
    else:
        arr = _coerce_structured_rows(class_rows, _CLASS_ROW_DTYPE)
        total = len(arr)
        if chunk_rows and chunk_rows > 0 and total > chunk_rows:
            def _gen():
                for start in range(0, total, chunk_rows):
                    yield arr[start:start+chunk_rows]
            chunk_iter = _gen()
        else:
            def _single():
                if total:
                    yield arr
            chunk_iter = _single()

    dense_acc = _prepare_dense_data_accumulator(raw_table)
    if dense_acc is not None and dense_acc['keys'].size == 0:
        dense_acc = None
    fallback_counts = {}
    fallback_tracer = {}
    random_store = TempTableStore(_PROB_ROW_DTYPE, prefix='prob', base_dir=spill_dir)

    for chunk in chunk_iter:
        if chunk.size == 0:
            continue
        ndata = chunk['NDATA'].astype(np.float32, copy=False)
        nrand = chunk['NRAND'].astype(np.float32, copy=False)
        denom = ndata + nrand
        diff = ndata - nrand
        ratios = np.zeros_like(denom, dtype=np.float32)
        valid = denom > 0
        np.divide(diff, denom, out=ratios, where=valid)

        classes = np.zeros(chunk.shape[0], dtype=np.uint8)
        mid_mask = (ratios >= r_lower) & (ratios < 0.0)
        pos_mask = (ratios >= 0.0) & (ratios < r_upper)
        high_mask = ratios >= r_upper
        classes[mid_mask] = 1
        classes[pos_mask] = 2
        classes[high_mask] = 3

        mask_data = chunk['ISDATA']
        if mask_data.any():
            tids = chunk['TARGETID'][mask_data].astype(np.int64, copy=False)
            tracer_ids_full = chunk['TRACER_ID'][mask_data].astype(np.int16, copy=False)
            tracers = chunk['TRACERTYPE'][mask_data]
            class_vals = classes[mask_data]

            dense_used = False
            if dense_acc is not None:
                dense_keys = dense_acc['keys']
                if dense_keys.size:
                    dense_used = True
                    chunk_keys = np.empty(tids.size, dtype=_DATA_KEY_DTYPE)
                    chunk_keys['TRACER_ID'] = tracer_ids_full
                    chunk_keys['TARGETID'] = tids
                    positions = np.searchsorted(dense_keys, chunk_keys)
                    matched = np.zeros_like(positions, dtype=bool)
                    valid = positions < dense_keys.shape[0]
                    if np.any(valid):
                        matched_valid = dense_keys[positions[valid]] == chunk_keys[valid]
                        matched[valid] = matched_valid
                    if np.any(matched):
                        np.add.at(dense_acc['counts'],
                                  (positions[matched].astype(np.intp, copy=False),
                                   class_vals[matched].astype(np.intp, copy=False)),
                                  1)
                    if not np.all(matched):
                        unmatched_idx = np.nonzero(~matched)[0]
                        for tid, tracer_id, tracer_type, cls in zip(
                                tids[unmatched_idx],
                                tracer_ids_full[unmatched_idx],
                                tracers[unmatched_idx],
                                class_vals[unmatched_idx]):
                            key = (int(tid), int(tracer_id))
                            counts = fallback_counts.get(key)
                            if counts is None:
                                counts = np.zeros(4, dtype=np.uint16)
                                fallback_counts[key] = counts
                                fallback_tracer[key] = bytes(tracer_type)
                            counts[int(cls)] += 1
            if dense_acc is None or not dense_used:
                for tid, tracer_id, tracer_type, cls in zip(tids, tracer_ids_full, tracers, class_vals):
                    key = (int(tid), int(tracer_id))
                    counts = fallback_counts.get(key)
                    if counts is None:
                        counts = np.zeros(4, dtype=np.uint16)
                        fallback_counts[key] = counts
                        fallback_tracer[key] = bytes(tracer_type)
                    counts[int(cls)] += 1

        mask_rand = ~mask_data
        if mask_rand.any():
            class_vals_rand = classes[mask_rand]
            rec_size = mask_rand.sum()
            rec = np.empty(rec_size, dtype=_PROB_ROW_DTYPE)
            rec['TARGETID'] = chunk['TARGETID'][mask_rand].astype(np.int64, copy=False)
            rec['RANDITER'] = chunk['RANDITER'][mask_rand].astype(np.int32, copy=False)
            rec['ISDATA'] = False
            rec['TRACER_ID'] = chunk['TRACER_ID'][mask_rand].astype(np.uint8, copy=False)
            rec['TRACERTYPE'] = chunk['TRACERTYPE'][mask_rand]
            rec['PVOID'] = 0.0
            rec['PSHEET'] = 0.0
            rec['PFILAMENT'] = 0.0
            rec['PKNOT'] = 0.0
            if rec_size:
                idx_void = (class_vals_rand == 0)
                if np.any(idx_void):
                    rec['PVOID'][idx_void] = 1.0
                idx_sheet = (class_vals_rand == 1)
                if np.any(idx_sheet):
                    rec['PSHEET'][idx_sheet] = 1.0
                idx_fil = (class_vals_rand == 2)
                if np.any(idx_fil):
                    rec['PFILAMENT'][idx_fil] = 1.0
                idx_knot = (class_vals_rand == 3)
                if np.any(idx_knot):
                    rec['PKNOT'][idx_knot] = 1.0
            random_store.append(rec)

    fallback_records = _build_data_records_from_dict(fallback_counts, fallback_tracer)
    if dense_acc is not None and dense_acc.get('keys') is None:
        dense_acc = None

    return dense_acc, fallback_records, random_store


def _chain_probability_chunks(dense_acc, fallback_records, random_store, chunk_rows):
    """
    Yield probability rows, prioritising data records then random chunks.
    
    Args:
        dense_acc (dict | None): Dense accumulator for data probabilities.
        fallback_records (np.ndarray | None): Fallback data records array.
        random_store (TempTableStore | np.ndarray | None): Random records store or array.
        chunk_rows (int): Maximum rows per yielded chunk.
    Returns:
        generator[np.ndarray]: Generator yielding structured arrays of probability rows.
    """
    yielded = False
    for chunk in _iter_dense_data_chunks(dense_acc, chunk_rows):
        if chunk.size:
            yielded = True
            yield chunk

    if fallback_records is not None:
        total = int(fallback_records.shape[0]) if hasattr(fallback_records, 'shape') else len(fallback_records)
        if total:
            yielded = True
            if chunk_rows and chunk_rows > 0 and total > chunk_rows:
                for start in range(0, total, int(chunk_rows)):
                    yield fallback_records[start:start+int(chunk_rows)]
            else:
                yield fallback_records

    if isinstance(random_store, TempTableStore):
        for chunk in random_store.iter_arrays(chunk_rows):
            if chunk.size:
                yielded = True
                yield chunk
    elif random_store is not None and len(random_store):
        yielded = True
        yield random_store
    if not yielded:
        yield np.empty(0, dtype=_PROB_ROW_DTYPE)


def build_probability_table(class_rows, raw_table=None, r_lower=-0.9, r_upper=0.9):
    """
    Return an in-memory probability table (suitable for small datasets).
    
    Args:
        class_rows: Sequence or disk-backed store of classification rows.
        raw_table (Table | None): Optional raw input table for reference.
        r_lower (float): Lower ratio threshold for classification.
        r_upper (float): Upper ratio threshold for classification.
    Returns:
        Table: Table with probability information.
    """
    if r_lower >= 0 or r_upper <= 0:
        raise ValueError('r_lower must be negative and r_upper must be positive.')

    dense_acc, fallback_records, random_store = _compute_probability_components(class_rows,
                                                                                raw_table=raw_table,
                                                                                r_lower=r_lower,
                                                                                r_upper=r_upper)

    arrays = []
    if dense_acc is not None:
        arrays.append(_finalize_dense_data_records(dense_acc))
    if fallback_records is not None and getattr(fallback_records, 'size', 0):
        arrays.append(fallback_records)
    if isinstance(random_store, TempTableStore):
        arrays.extend(list(random_store.iter_arrays()))
        random_store.cleanup()
    elif random_store is not None:
        arrays.append(random_store)

    if arrays:
        combined = np.concatenate(arrays, axis=0) if len(arrays) > 1 else arrays[0]
    else:
        combined = np.empty(0, dtype=_PROB_ROW_DTYPE)

    return Table(combined, copy=False)

def save_probability_fits(class_rows, raw_table=None, output_path=None, r_lower=-0.9, r_upper=0.9, meta=None):
    """
    Write the probability table to ``output_path`` with minimal peak memory.
    
    Args:
        class_rows: Sequence or disk-backed store of classification rows.
        raw_table (Table | None): Optional raw input table for reference.
        output_path (str): Destination path for the FITS file.
        r_lower (float): Lower ratio threshold for classification.
        r_upper (float): Upper ratio threshold for classification.
        meta (dict | None): Optional metadata inserted into the FITS header.
    """
    if output_path is None:
        raise ValueError('output_path must be provided')
    if r_lower >= 0 or r_upper <= 0:
        raise ValueError('r_lower must be negative and r_upper must be positive.')

    spill_dir = None
    if isinstance(class_rows, TempTableStore):
        spill_dir = os.path.dirname(class_rows.tmpdir)
    dense_acc, fallback_records, random_store = _compute_probability_components(class_rows,
                                                                                raw_table=raw_table,
                                                                                r_lower=r_lower,
                                                                                r_upper=r_upper,
                                                                                chunk_rows=_DEFAULT_CHUNK_ROWS,
                                                                                spill_dir=spill_dir)

    total_rows = 0
    if dense_acc is not None:
        keys = dense_acc.get('keys')
        if keys is not None:
            total_rows += int(keys.shape[0])
    if fallback_records is not None:
        total_rows += int(len(fallback_records))
    if isinstance(random_store, TempTableStore):
        total_rows += random_store.total
    elif random_store is not None:
        total_rows += len(random_store)

    columns = (('TARGETID', 'K'),
               ('RANDITER', 'J'),
               ('ISDATA', 'L'),
               ('TRACER_ID', 'B'),
               ('TRACERTYPE', '24A'),
               ('PVOID', 'E'),
               ('PSHEET', 'E'),
               ('PFILAMENT', 'E'),
               ('PKNOT', 'E'))

    split_iter = _bool_env('ASTRA_PROB_SPLIT_ITER', default=True)
    skip_combined = _bool_env('ASTRA_PROB_SKIP_COMBINED', default=False)
    if not split_iter and skip_combined:
        if isinstance(random_store, TempTableStore):
            random_store.cleanup()
        return

    random_split_stores = {}
    data_store = None
    base_dir = os.path.dirname(output_path)

    def _append_random_split(chunk):
        """
        Append random chunk rows to per-iteration stores.
        
        Args:
            chunk (np.ndarray): Structured array of probability rows.
        """
        if not split_iter or chunk.size == 0:
            return
        vals = np.asarray(chunk['RANDITER'], dtype=np.int32, copy=False)
        unique_vals = np.unique(vals)
        for val in unique_vals:
            mask = (vals == val)
            if not mask.any():
                continue
            store = random_split_stores.get(int(val))
            if store is None:
                store = TempTableStore(_PROB_ROW_DTYPE, prefix=f'prob_iter_{int(val):03d}', base_dir=base_dir)
                random_split_stores[int(val)] = store
            store.append(chunk[mask])

    def _prob_chunk_generator():
        """
        Generator yielding probability chunks.
        
        Yields:
            np.ndarray: Structured arrays of probability rows.
        """
        nonlocal data_store
        if dense_acc is not None:
            dense_records_arr = _finalize_dense_data_records(dense_acc)
            if dense_records_arr.size:
                if split_iter:
                    data_store = TempTableStore(_PROB_ROW_DTYPE, prefix='prob_data', base_dir=base_dir)
                    data_store.append(dense_records_arr)
                yield dense_records_arr
        if fallback_records is not None and getattr(fallback_records, 'size', 0):
            if split_iter:
                if data_store is None:
                    data_store = TempTableStore(_PROB_ROW_DTYPE, prefix='prob_data', base_dir=base_dir)
                data_store.append(fallback_records)
            yield fallback_records
        if isinstance(random_store, TempTableStore):
            for chunk in random_store.iter_arrays(_DEFAULT_CHUNK_ROWS):
                if chunk.size:
                    _append_random_split(chunk)
                    yield chunk
        elif random_store is not None:
            _append_random_split(random_store)
            yield random_store

    iterator = _prob_chunk_generator()

    if not skip_combined:
        _write_fits_table(columns, total_rows, iterator, output_path, meta=meta)
    else:
        for _ in iterator:
            pass

    if isinstance(random_store, TempTableStore):
        random_store.cleanup()

    if split_iter:
        if data_store is not None and data_store.total:
            data_path = _split_iter_path(output_path, 'data')
            data_iter, data_total = _iter_structured_chunks(data_store, _PROB_ROW_DTYPE, chunk_rows=_DEFAULT_CHUNK_ROWS)
            _write_fits_table(columns, data_total, data_iter, data_path, meta=meta)
            data_store.cleanup()
        for iteration, store in random_split_stores.items():
            iter_path = _split_iter_path(output_path, iteration)
            iter_chunk_iter, iter_total = _iter_structured_chunks(store, _PROB_ROW_DTYPE, chunk_rows=_DEFAULT_CHUNK_ROWS)
            _write_fits_table(columns, iter_total, iter_chunk_iter, iter_path, meta=meta)
            store.cleanup()