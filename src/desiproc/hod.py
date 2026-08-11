from __future__ import annotations

import fcntl, gc, json
import multiprocessing as mp
import os, re, time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, Optional, Sequence, Tuple

import numpy as np
import fitsio
from .implement_astra import compute_delaunay_neighbor_counts


DEFAULT_INPUT_ROOT = '/pscratch/sd/n/ntbfin/emulator/hods/z0.5/yuan23_prior'
DEFAULT_N_ITERATIONS = 5
DEFAULT_COORD_COLUMNS = ('X_RSD', 'Y_RSD', 'Z_RSD')
DEFAULT_ID_COLUMN = 'ID'
DEFAULT_BOX_ORIGIN = (-1000.0, -1000.0, -1000.0)
DEFAULT_BOX_SIZE = 2000.0

_CLASSIFICATION_DTYPE = np.dtype([('TARGETID', np.int64),
                                  ('RANDITER', np.int32),
                                  ('NDATA', np.int32),
                                  ('NRAND', np.int32)])
_PROBABILITY_DTYPE = np.dtype([('TARGETID', np.int64),
                               ('PSHEET', np.float32),
                               ('PVOID', np.float32),
                               ('PFILAMENT', np.float32),
                               ('PKNOT', np.float32)])
_COSMO_RE = re.compile(r'^[cC]?(?P<number>\d+)(?:_[pP][hH](?P<phase>\d+))?$')
_HOD_RE = re.compile(r'^(?:[hH][oO][dD])?(?P<number>\d+)(?:\.fits)?$')
_WORKER_STATE = None


@dataclass(frozen=True)
class HODRunConfig:
    input_root: str
    output_root: str
    cosmology: str
    hod: str
    phase: int = 0
    simulation_seed: int = 0
    n_iterations: int = DEFAULT_N_ITERATIONS
    random_seed: int = 0
    coordinate_columns: Tuple[str, str, str] = DEFAULT_COORD_COLUMNS
    id_column: str = DEFAULT_ID_COLUMN
    box_origin: Tuple[float, float, float] = DEFAULT_BOX_ORIGIN
    box_size: float = DEFAULT_BOX_SIZE
    r_lower: float = -0.25
    r_med: float = 0.25
    r_upper: float = 0.65
    iteration_workers: Optional[int] = None
    count_chunk_vertices: int = 250_000
    io_chunk_rows: int = 500_000
    qhull_options: Optional[str] = None
    save_classification: bool = True
    force: bool = False


@dataclass(frozen=True)
class HODRunPaths:
    source: Path
    run_dir: Path
    raw_dir: Path
    classification_dir: Path
    probability_dir: Path
    raw_link: Path
    probability: Path
    metadata: Path

    def classification(self, hod: str, iteration: int) -> Path:
        return self.classification_dir / f'{hod}_iter{int(iteration):03d}_classified.fits'


def normalize_cosmology(value: str) -> Tuple[str, Optional[int]]:
    """Return ``(cXXX, optional_phase)`` for a cosmology token."""
    match = _COSMO_RE.fullmatch(str(value).strip())
    if match is None:
        raise ValueError(f"Invalid cosmology '{value}'; expected cXXX or cXXX_phXXX")
    number = int(match.group('number'))
    phase_text = match.group('phase')
    phase = None if phase_text is None else int(phase_text)
    return f'c{number:03d}', phase


def normalize_hod(value: str) -> str:
    """Return a normalized ``hodXXX`` token."""
    match = _HOD_RE.fullmatch(str(value).strip())
    if match is None:
        raise ValueError(f"Invalid HOD realization '{value}'; expected hodXXX")
    return f"hod{int(match.group('number')):03d}"


def resolve_run_paths(config: HODRunConfig) -> HODRunPaths:
    """Resolve the source catalogue and requested output hierarchy."""
    cosmology, embedded_phase = normalize_cosmology(config.cosmology)
    hod = normalize_hod(config.hod)
    phase = int(config.phase if embedded_phase is None else embedded_phase)
    source = (Path(config.input_root).expanduser().resolve()
              / f'{cosmology}_ph{phase:03d}'
              / f'seed{int(config.simulation_seed)}'
              / f'{hod}.fits')
    run_dir = Path(config.output_root).expanduser().resolve() / cosmology / hod
    raw_dir = run_dir / 'raw'
    classification_dir = run_dir / 'classification'
    probability_dir = run_dir / 'probabilities'
    return HODRunPaths(source=source,
                       run_dir=run_dir,
                       raw_dir=raw_dir,
                       classification_dir=classification_dir,
                       probability_dir=probability_dir,
                       raw_link=raw_dir / f'{hod}.fits',
                       probability=probability_dir / f'{hod}_probability.fits',
                       metadata=run_dir / 'run.json')


def read_manifest_entry(path: str, task_index: int) -> Tuple[str, str]:
    """Read a zero-based ``cosmology HOD`` entry from a manifest."""
    requested = int(task_index)
    if requested < 0:
        raise ValueError('task_index must be non-negative')
    current = -1
    with open(path, 'r', encoding='utf-8') as stream:
        for line in stream:
            text = line.strip()
            if not text or text.startswith('#'):
                continue
            current += 1
            if current != requested:
                continue
            fields = text.split()
            if len(fields) < 2:
                raise ValueError(f'Manifest entry {requested} must contain cosmology and HOD')
            cosmology, _ = normalize_cosmology(fields[0])
            return cosmology, normalize_hod(fields[1])
    raise IndexError(f'Manifest task index {requested} is out of range for {path}')


def _validate_config(config: HODRunConfig) -> None:
    if int(config.n_iterations) <= 0:
        raise ValueError('n_iterations must be greater than zero')
    if int(config.n_iterations) > np.iinfo(np.uint16).max:
        raise ValueError('n_iterations cannot exceed 65535')
    if len(config.coordinate_columns) != 3:
        raise ValueError('Exactly three coordinate columns are required')
    if len(config.box_origin) != 3:
        raise ValueError('box_origin must contain three values')
    if not np.isfinite(config.box_size) or float(config.box_size) <= 0:
        raise ValueError('box_size must be finite and greater than zero')
    if (config.r_lower >= 0 or config.r_upper <= 0
            or not (config.r_lower < config.r_med < config.r_upper)):
        raise ValueError('Thresholds must satisfy r_lower < r_med < r_upper with r_lower < 0 < r_upper')
    if int(config.count_chunk_vertices) <= 0 or int(config.io_chunk_rows) <= 0:
        raise ValueError('Chunk sizes must be greater than zero')


def _inspect_catalog(path: Path, coordinate_columns: Sequence[str], id_column: str) -> Dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(f'HOD catalogue not found: {path}')
    stat = path.stat()
    with fitsio.FITS(str(path), mode='r') as hdus:
        if len(hdus) < 2:
            raise ValueError(f'HOD catalogue has no binary table extension: {path}')
        table = hdus[1]
        names = set(table.get_colnames())
        required = [id_column, *coordinate_columns]
        missing = [name for name in required if name not in names]
        if missing:
            raise KeyError(f'HOD catalogue {path} is missing columns: {missing}')
        n_rows = int(table.get_nrows())
    if n_rows < 3:
        raise ValueError(f'HOD catalogue needs at least three rows; found {n_rows}')
    return {'rows': n_rows,
            'size': int(stat.st_size),
            'mtime_ns': int(stat.st_mtime_ns)}


def _read_column(path: Path, name: str) -> np.ndarray:
    with fitsio.FITS(str(path), mode='r') as hdus:
        values = hdus[1].read(columns=[name])[name]
    return values


def _read_target_ids(path: Path, id_column: str, expected_rows: int) -> np.ndarray:
    target_ids = np.asarray(_read_column(path, id_column), dtype=np.int64)
    if target_ids.shape != (expected_rows,):
        raise ValueError(f'Unexpected {id_column} shape {target_ids.shape} in {path}')
    return np.ascontiguousarray(target_ids)


def _read_coordinates(config: HODRunConfig, paths: HODRunPaths,
                      source_info: Dict[str, object]) -> np.ndarray:
    n_rows = int(source_info['rows'])
    coords = np.empty((n_rows, 3), dtype=np.float64)
    lower = np.asarray(config.box_origin, dtype=np.float64)
    upper = lower + float(config.box_size)
    for axis, name in enumerate(config.coordinate_columns):
        values = np.asarray(_read_column(paths.source, name), dtype=np.float64)
        if values.shape != (n_rows,):
            raise ValueError(f'Unexpected {name} shape {values.shape} in {paths.source}')
        if not np.all(np.isfinite(values)):
            raise ValueError(f'Coordinate column {name} contains non-finite values')
        col_min = float(np.min(values))
        col_max = float(np.max(values))
        tolerance = max(1.0, abs(lower[axis]), abs(upper[axis])) * 1e-10
        if col_min < lower[axis] - tolerance or col_max > upper[axis] + tolerance:
            raise ValueError(f'{name} range [{col_min}, {col_max}] lies outside configured '
                             f'box [{lower[axis]}, {upper[axis]}]. Adjust --box-origin/--box-size.')
        coords[:, axis] = values
        del values
    return coords


def _base_header(config: HODRunConfig, paths: HODRunPaths,
                 source_info: Dict[str, object]) -> Dict[str, object]:
    cosmology, embedded_phase = normalize_cosmology(config.cosmology)
    phase = int(config.phase if embedded_phase is None else embedded_phase)
    hod = normalize_hod(config.hod)
    lower = np.asarray(config.box_origin, dtype=np.float64)
    upper = lower + float(config.box_size)
    return {'MODE': 'HODBOX',
            'VERSION': 1,
            'COSMO': cosmology,
            'HOD': hod,
            'PHASE': phase,
            'SIMSEED': int(config.simulation_seed),
            'BASESEED': int(config.random_seed),
            'SRCFILE': paths.source.name,
            'SRCROWS': int(source_info['rows']),
            'SRCSIZE': int(source_info['size']),
            'SRCMTIME': int(source_info['mtime_ns']),
            'IDCOL': str(config.id_column),
            'ROWALIGN': 'SOURCE',
            'COORDX': str(config.coordinate_columns[0]),
            'COORDY': str(config.coordinate_columns[1]),
            'COORDZ': str(config.coordinate_columns[2]),
            'BOXLOX': float(lower[0]),
            'BOXLOY': float(lower[1]),
            'BOXLOZ': float(lower[2]),
            'BOXHIX': float(upper[0]),
            'BOXHIY': float(upper[1]),
            'BOXHIZ': float(upper[2])}


def _classification_header(config: HODRunConfig, paths: HODRunPaths,
                           source_info: Dict[str, object], iteration: int) -> Dict[str, object]:
    header = _base_header(config, paths, source_info)
    header.update({'ITER': int(iteration), 'ISDATA': 1, 'PAIRFILE': 0})
    return header


def _probability_header(config: HODRunConfig, paths: HODRunPaths,
                        source_info: Dict[str, object]) -> Dict[str, object]:
    header = _base_header(config, paths, source_info)
    header.update({'NITER': int(config.n_iterations),
                   'RLOWER': float(config.r_lower),
                   'RMED': float(config.r_med),
                   'RUPPER': float(config.r_upper),
                   'PAIRFILE': 0})
    return header


def _header_value(header, key: str):
    try:
        return header[key]
    except Exception:
        return None


def _headers_match(actual, expected: Dict[str, object]) -> Tuple[bool, str]:
    for key, wanted in expected.items():
        found = _header_value(actual, key)
        if isinstance(wanted, float):
            try:
                matches = np.isclose(float(found), wanted, rtol=0.0, atol=1e-10)
            except (TypeError, ValueError):
                matches = False
        else:
            if isinstance(found, bytes):
                found = found.decode('ascii', errors='ignore').strip()
            matches = found == wanted
        if not matches:
            return False, f'{key}: expected {wanted!r}, found {found!r}'
    return True, ''


def _fits_product_matches(path: Path, expected_header: Dict[str, object],
                          expected_rows: int) -> Tuple[bool, str]:
    if not path.is_file():
        return False, 'missing'
    try:
        with fitsio.FITS(str(path), mode='r') as hdus:
            if len(hdus) < 2:
                return False, 'missing table HDU'
            if int(hdus[1].get_nrows()) != int(expected_rows):
                return False, 'row count differs'
            return _headers_match(hdus[1].read_header(), expected_header)
    except Exception as exc:
        return False, str(exc)


def _write_fits_chunks(path: Path, dtype: np.dtype, chunks: Iterable[np.ndarray],
                       header: Dict[str, object]) -> None:
    """Atomically append structured chunks to one FITS binary table."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'.{path.name}.tmp.{os.getpid()}')
    try:
        with fitsio.FITS(str(temporary), mode='rw', clobber=True) as hdus:
            hdus.create_table_hdu(dtype=dtype, extname='ASTRA')
            table = hdus[-1]
            table.write_keys(header)
            for chunk in chunks:
                if len(chunk):
                    table.append(np.asarray(chunk, dtype=dtype))
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _classification_chunks(target_ids: np.ndarray, iteration: int,
                           ndata: np.ndarray, nrand: np.ndarray,
                           chunk_rows: int) -> Iterator[np.ndarray]:
    total = len(target_ids)
    for start in range(0, total, int(chunk_rows)):
        stop = min(start + int(chunk_rows), total)
        chunk = np.empty(stop - start, dtype=_CLASSIFICATION_DTYPE)
        chunk['TARGETID'] = target_ids[start:stop]
        chunk['RANDITER'] = int(iteration)
        chunk['NDATA'] = ndata[start:stop]
        chunk['NRAND'] = nrand[start:stop]
        yield chunk


def _probability_chunks(target_ids: np.ndarray, counts: np.ndarray,
                        n_iterations: int, chunk_rows: int) -> Iterator[np.ndarray]:
    total = len(target_ids)
    inverse = np.float32(1.0 / float(n_iterations))
    fields = ('PVOID', 'PSHEET', 'PFILAMENT', 'PKNOT')
    for start in range(0, total, int(chunk_rows)):
        stop = min(start + int(chunk_rows), total)
        chunk = np.empty(stop - start, dtype=_PROBABILITY_DTYPE)
        chunk['TARGETID'] = target_ids[start:stop]
        for class_index, field in enumerate(fields):
            chunk[field] = counts[class_index, start:stop] * inverse
        yield chunk


def classify_neighbor_counts(ndata: np.ndarray, nrand: np.ndarray,
                             r_lower: float, r_med: float,
                             r_upper: float) -> np.ndarray:
    """Return ASTRA class codes 0=void, 1=sheet, 2=filament, 3=knot."""
    data_float = np.asarray(ndata, dtype=np.float32)
    rand_float = np.asarray(nrand, dtype=np.float32)
    denominator = data_float + rand_float
    ratios = np.zeros(denominator.shape, dtype=np.float32)
    np.divide(data_float - rand_float, denominator, out=ratios, where=denominator > 0)
    classes = np.zeros(denominator.shape, dtype=np.uint8)
    classes[(ratios >= r_lower) & (ratios < r_med)] = 1
    classes[(ratios >= r_med) & (ratios < r_upper)] = 2
    classes[ratios >= r_upper] = 3
    return classes


def _read_class_codes(path: Path, config: HODRunConfig) -> np.ndarray:
    with fitsio.FITS(str(path), mode='r') as hdus:
        values = hdus[1].read(columns=['NDATA', 'NRAND'])
    return classify_neighbor_counts(values['NDATA'], values['NRAND'],
                                    config.r_lower, config.r_med, config.r_upper)


def _accumulate_classes(counts: np.ndarray, classes: np.ndarray) -> None:
    for class_index in range(4):
        counts[class_index] += (classes == class_index)


def _seed_sequence(config: HODRunConfig, iteration: int) -> np.random.SeedSequence:
    cosmology, embedded_phase = normalize_cosmology(config.cosmology)
    hod = normalize_hod(config.hod)
    phase = int(config.phase if embedded_phase is None else embedded_phase)
    return np.random.SeedSequence([int(config.random_seed),
                                   int(cosmology[1:]),
                                   int(hod[3:]),
                                   phase,
                                   int(config.simulation_seed),
                                   int(iteration)])


def _init_worker(state: Dict[str, object]) -> None:
    global _WORKER_STATE
    _WORKER_STATE = state


def _run_iteration_worker(iteration: int):
    state = _WORKER_STATE
    if state is None:
        raise RuntimeError('HOD worker state was not initialized')
    config = state['config']
    paths = state['paths']
    source_info = state['source_info']
    data_coords = state['coordinates']
    target_ids = state['target_ids']
    n_data = int(data_coords.shape[0])
    start_time = time.time()
    print(f'hod-astra --> {config.cosmology}/{config.hod} iteration={iteration:03d} points={2 * n_data:,}',
          flush=True)

    points = np.empty((2 * n_data, 3), dtype=np.float64)
    points[:n_data] = data_coords
    random_points = points[n_data:]
    rng = np.random.default_rng(_seed_sequence(config, iteration))
    rng.random(random_points.shape, dtype=np.float64, out=random_points)
    random_points *= float(config.box_size)
    random_points += np.asarray(config.box_origin, dtype=np.float64)

    ndata, nrand = compute_delaunay_neighbor_counts(points, n_data=n_data,
                                                    n_vertices=n_data,
                                                    chunk_vertices=config.count_chunk_vertices,
                                                    qhull_options=config.qhull_options)
    del random_points, points
    gc.collect()

    classes = classify_neighbor_counts(ndata, nrand, config.r_lower, config.r_med, config.r_upper)
    if config.save_classification:
        class_path = paths.classification(normalize_hod(config.hod), iteration)
        header = _classification_header(config, paths, source_info, iteration)
        _write_fits_chunks(class_path, _CLASSIFICATION_DTYPE,
                           _classification_chunks(target_ids, iteration, ndata, nrand, config.io_chunk_rows), header)
    del ndata, nrand
    elapsed = time.time() - start_time
    print(f'hod-astra --> {config.cosmology}/{config.hod} iteration={iteration:03d} done in {elapsed:.1f}s',
          flush=True)
    return int(iteration), classes, float(elapsed)


def _resolve_workers(config: HODRunConfig, pending_count: int) -> int:
    if pending_count <= 1:
        return 1
    requested = config.iteration_workers
    if requested is None:
        env_value = os.environ.get('HOD_ITER_WORKERS', '').strip()
        if env_value:
            requested = int(env_value)
        elif os.environ.get('SLURM_JOB_ID'):
            requested = int(os.environ.get('SLURM_CPUS_PER_TASK', '1'))
        else:
            requested = 1
    if int(requested) <= 0:
        raise ValueError('iteration_workers must be greater than zero')
    return min(int(requested), int(pending_count))


def _ensure_raw_link(paths: HODRunPaths, force: bool) -> None:
    paths.raw_dir.mkdir(parents=True, exist_ok=True)
    desired = str(paths.source)
    if os.path.lexists(paths.raw_link):
        if paths.raw_link.is_symlink() and os.path.realpath(paths.raw_link) == desired:
            return
        if not force:
            raise FileExistsError(f'Raw output already exists and is not the expected symlink: {paths.raw_link}')
        paths.raw_link.unlink()
    temporary = paths.raw_link.with_name(f'.{paths.raw_link.name}.tmp.{os.getpid()}')
    try:
        os.symlink(desired, temporary)
        os.replace(temporary, paths.raw_link)
    finally:
        if os.path.lexists(temporary):
            temporary.unlink()


def _write_metadata(path: Path, payload: Dict[str, object]) -> None:
    temporary = path.with_name(f'.{path.name}.tmp.{os.getpid()}')
    try:
        with open(temporary, 'w', encoding='utf-8') as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write('\n')
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _run_locked(config: HODRunConfig, paths: HODRunPaths) -> HODRunPaths:
    source_info = _inspect_catalog(paths.source, config.coordinate_columns, config.id_column)
    paths.raw_dir.mkdir(parents=True, exist_ok=True)
    paths.classification_dir.mkdir(parents=True, exist_ok=True)
    paths.probability_dir.mkdir(parents=True, exist_ok=True)
    _ensure_raw_link(paths, config.force)

    probability_header = _probability_header(config, paths, source_info)
    probability_matches, _ = _fits_product_matches(paths.probability, probability_header, int(source_info['rows']),)
    if probability_matches and not config.force:
        print(f'hod-astra --> complete; reusing {paths.probability}', flush=True)
        return paths

    n_rows = int(source_info['rows'])
    probability_counts = np.zeros((4, n_rows), dtype=np.uint16)
    pending = []
    reused = 0
    if config.save_classification and not config.force:
        for iteration in range(int(config.n_iterations)):
            class_path = paths.classification(normalize_hod(config.hod), iteration)
            expected = _classification_header(config, paths, source_info, iteration)
            matches, reason = _fits_product_matches(class_path, expected, n_rows)
            if matches:
                classes = _read_class_codes(class_path, config)
                _accumulate_classes(probability_counts, classes)
                del classes
                reused += 1
            elif class_path.exists():
                raise RuntimeError(f'Existing classification is incompatible ({reason}): {class_path}. Use --force to replace it.')
            else:
                pending.append(iteration)
    else:
        pending = list(range(int(config.n_iterations)))

    print(f'hod-astra --> {normalize_cosmology(config.cosmology)[0]}/{normalize_hod(config.hod)} '
          f'rows={n_rows:,} iterations={config.n_iterations} reused={reused} pending={len(pending)} pairs=disabled',
          flush=True)

    target_ids = _read_target_ids(paths.source, config.id_column, n_rows)
    iteration_times = {}
    if pending:
        coordinates = _read_coordinates(config, paths, source_info)
        worker_count = _resolve_workers(config, len(pending))
        print(f'hod-astra --> iteration workers={worker_count}', flush=True)
        state = {'config': config,
                 'paths': paths,
                 'source_info': source_info,
                 'coordinates': coordinates,
                 'target_ids': target_ids}
        if worker_count > 1:
            context = mp.get_context('fork')
            with context.Pool(
                    processes=worker_count,
                    initializer=_init_worker,
                    initargs=(state,)) as pool:
                for iteration, classes, elapsed in pool.imap_unordered(
                        _run_iteration_worker, pending):
                    _accumulate_classes(probability_counts, classes)
                    iteration_times[str(iteration)] = elapsed
                    del classes
        else:
            _init_worker(state)
            for iteration in pending:
                iteration, classes, elapsed = _run_iteration_worker(iteration)
                _accumulate_classes(probability_counts, classes)
                iteration_times[str(iteration)] = elapsed
                del classes
        del coordinates
        gc.collect()

    _write_fits_chunks(paths.probability, _PROBABILITY_DTYPE, _probability_chunks(target_ids,
                                                                                  probability_counts,
                                                                                  int(config.n_iterations),
                                                                                  int(config.io_chunk_rows)),
                       probability_header)
    _write_metadata(paths.metadata, {'box_origin': list(map(float, config.box_origin)),
                                     'box_size': float(config.box_size),
                                     'classification_saved': bool(config.save_classification),
                                     'coordinate_columns': list(config.coordinate_columns),
                                     'cosmology': normalize_cosmology(config.cosmology)[0],
                                     'hod': normalize_hod(config.hod),
                                     'id_column': config.id_column,
                                     'input': str(paths.source),
                                     'iteration_seconds': iteration_times,
                                     'n_iterations': int(config.n_iterations),
                                     'n_rows': n_rows,
                                     'pair_files_saved': False,
                                     'probability': str(paths.probability),
                                     'random_seed': int(config.random_seed),
                                     'randoms': 'uniform independent realization per iteration',
                                     'r_thresholds': [float(config.r_lower), float(config.r_med), float(config.r_upper)],
                                     'raw': str(paths.raw_link),
                                     'raw_storage': 'symlink',
                                     'row_alignment': 'output row i corresponds to input FITS row i'})
    print(f'hod-astra --> wrote {paths.probability}', flush=True)
    return paths


def run_hod_pipeline(config: HODRunConfig) -> HODRunPaths:
    """Run or resume ASTRA for one cosmology/HOD catalogue."""
    _validate_config(config)
    paths = resolve_run_paths(config)
    paths.run_dir.mkdir(parents=True, exist_ok=True)
    lock_path = paths.run_dir / '.run.lock'
    with open(lock_path, 'a', encoding='utf-8') as lock_stream:
        try:
            fcntl.flock(lock_stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f'Another process is already running {paths.run_dir}') from exc
        return _run_locked(config, paths)


__all__ = ['DEFAULT_BOX_ORIGIN',
           'DEFAULT_BOX_SIZE',
           'DEFAULT_COORD_COLUMNS',
           'DEFAULT_ID_COLUMN',
           'DEFAULT_INPUT_ROOT',
           'DEFAULT_N_ITERATIONS',
           'HODRunConfig',
           'HODRunPaths',
           'classify_neighbor_counts',
           'normalize_cosmology',
           'normalize_hod',
           'read_manifest_entry',
           'resolve_run_paths',
           'run_hod_pipeline']