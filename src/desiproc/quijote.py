from __future__ import annotations

import fcntl
import gc
import importlib
import importlib.util
import multiprocessing as mp
import os
import struct
import time
import zlib
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Dict, Iterable, Iterator, Optional, Tuple

import fitsio
import numpy as np

from .hod import classify_neighbor_counts
from .implement_astra import compute_delaunay_neighbor_counts


DEFAULT_INPUT_ROOT = '/pscratch/sd/v/vtorresg/quijotes/Halos/FoF'
DEFAULT_OUTPUT_ROOT = '/pscratch/sd/v/vtorresg/quijotes/ASTRA/FoF'
DEFAULT_N_ITERATIONS = 5
DEFAULT_BOX_ORIGIN = (0.0, 0.0, 0.0)
DEFAULT_BOX_SIZE = 1000.0
SNAPSHOT = 3
REDSHIFT = 0.5

_FOF_HEADER = struct.Struct('<iiiQI')
_PROBABILITY_DTYPE = np.dtype([('PVOID', np.float32),
                               ('PSHEET', np.float32),
                               ('PFILAMENT', np.float32),
                               ('PKNOT', np.float32)])
_RANDOM_VOID_DTYPE = np.dtype([('RANDITER', np.int32),
                               ('RANDINDEX', np.int32),
                               ('X', np.float32),
                               ('Y', np.float32),
                               ('Z', np.float32)])
_WORKER_STATE = None


@dataclass(frozen=True)
class QuijoteRunConfig:
    input_root: str
    output_root: str
    parameter: str
    realization: int
    n_iterations: int = DEFAULT_N_ITERATIONS
    random_seed: int = 0
    box_origin: Tuple[float, float, float] = DEFAULT_BOX_ORIGIN
    box_size: float = DEFAULT_BOX_SIZE
    r_lower: float = -0.25
    r_med: float = 0.25
    r_upper: float = 0.65
    iteration_workers: Optional[int] = None
    count_chunk_vertices: int = 250_000
    io_chunk_rows: int = 500_000
    qhull_options: Optional[str] = None
    readfof_path: Optional[str] = None
    force: bool = False


@dataclass(frozen=True)
class QuijoteRunPaths:
    catalogue_root: Path
    group_dir: Path
    output_dir: Path
    probability: Path
    random_voids: Path
    lock: Path


def normalize_parameter(value: str) -> str:
    """Validate a single Quijote parameter-directory name."""
    parameter = str(value).strip()
    if (not parameter or parameter in {'.', '..'}
            or '/' in parameter or '\\' in parameter):
        raise ValueError(f"Invalid Quijote parameter directory: {value!r}")
    return parameter


def resolve_run_paths(config: QuijoteRunConfig) -> QuijoteRunPaths:
    """Resolve the fixed-z=0.5 input catalogue and compact output path."""
    parameter = normalize_parameter(config.parameter)
    realization = int(config.realization)
    catalogue_root = (Path(config.input_root).expanduser().resolve()
                      / parameter / str(realization))
    group_dir = catalogue_root / f'groups_{SNAPSHOT:03d}'
    output_dir = (Path(config.output_root).expanduser().resolve()
                  / parameter / str(realization))
    return QuijoteRunPaths(
        catalogue_root=catalogue_root,
        group_dir=group_dir,
        output_dir=output_dir,
        probability=output_dir / f'group_{SNAPSHOT:03d}_probability.fits.gz',
        random_voids=output_dir / f'group_{SNAPSHOT:03d}_random_voids.fits.gz',
        lock=output_dir / '.run.lock')


def read_manifest_entry(path: str, task_index: int) -> Tuple[str, int]:
    """Read one zero-based ``parameter realization`` task from a manifest."""
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
                raise ValueError(
                    f'Manifest entry {requested} must contain parameter and realization')
            return normalize_parameter(fields[0]), int(fields[1])
    raise IndexError(f'Manifest task index {requested} is out of range for {path}')


def _validate_config(config: QuijoteRunConfig) -> None:
    normalize_parameter(config.parameter)
    if int(config.realization) < 0:
        raise ValueError('realization must be non-negative')
    if int(config.n_iterations) <= 0:
        raise ValueError('n_iterations must be greater than zero')
    if int(config.n_iterations) > np.iinfo(np.uint16).max:
        raise ValueError('n_iterations cannot exceed 65535')
    if not (0 <= int(config.random_seed) <= np.iinfo(np.uint32).max):
        raise ValueError('random_seed must be between 0 and 4294967295')
    if len(config.box_origin) != 3:
        raise ValueError('box_origin must contain three values')
    if not np.all(np.isfinite(np.asarray(config.box_origin, dtype=np.float64))):
        raise ValueError('box_origin values must be finite')
    if not np.isfinite(config.box_size) or float(config.box_size) <= 0:
        raise ValueError('box_size must be finite and greater than zero')
    if (config.r_lower >= 0 or config.r_upper <= 0
            or not (config.r_lower < config.r_med < config.r_upper)):
        raise ValueError(
            'Thresholds must satisfy r_lower < r_med < r_upper '
            'with r_lower < 0 < r_upper')
    if int(config.count_chunk_vertices) <= 0 or int(config.io_chunk_rows) <= 0:
        raise ValueError('Chunk sizes must be greater than zero')


def _inspect_catalog(paths: QuijoteRunPaths) -> Dict[str, int]:
    first_file = paths.group_dir / f'group_tab_{SNAPSHOT:03d}.0'
    if not first_file.is_file():
        raise FileNotFoundError(f'Quijote FoF catalogue not found: {first_file}')
    with open(first_file, 'rb') as stream:
        raw_header = stream.read(_FOF_HEADER.size)
    if len(raw_header) != _FOF_HEADER.size:
        raise ValueError(f'Incomplete FoF header in {first_file}')
    _, total_groups, _, _, n_files = _FOF_HEADER.unpack(raw_header)
    if total_groups < 3:
        raise ValueError(
            f'Quijote catalogue needs at least three halos; found {total_groups}')
    if total_groups > np.iinfo(np.int32).max:
        raise ValueError('Quijote halo count exceeds the RANDINDEX int32 range')
    if n_files <= 0:
        raise ValueError(f'Invalid FoF file count {n_files} in {first_file}')

    files = [paths.group_dir / f'group_tab_{SNAPSHOT:03d}.{index}'
             for index in range(int(n_files))]
    missing = [str(path) for path in files if not path.is_file()]
    if missing:
        raise FileNotFoundError(f'Missing Quijote FoF file parts: {missing}')
    stats = [path.stat() for path in files]
    return {'rows': int(total_groups),
            'n_files': int(n_files),
            'size': int(sum(stat.st_size for stat in stats)),
            'mtime_ns': int(max(stat.st_mtime_ns for stat in stats))}


def _readfof_candidates(explicit_path: Optional[str]) -> Iterable[Path]:
    if explicit_path:
        supplied = Path(explicit_path).expanduser()
        yield supplied / 'readfof.py' if supplied.is_dir() else supplied
        return

    configured = os.environ.get('PYLIANS_READFOF', '').strip()
    if configured:
        supplied = Path(configured).expanduser()
        yield supplied / 'readfof.py' if supplied.is_dir() else supplied

    venv = Path.home() / 'venvs' / 'pylians'
    yield from sorted(venv.glob('lib*/python*/site-packages/readfof.py'))


def _import_readfof(explicit_path: Optional[str] = None) -> ModuleType:
    """Import Pylians' pure-Python readfof module without importing its old NumPy."""
    if not explicit_path and not os.environ.get('PYLIANS_READFOF'):
        try:
            return importlib.import_module('readfof')
        except ModuleNotFoundError:
            pass

    checked = []
    for candidate in _readfof_candidates(explicit_path):
        candidate = candidate.resolve()
        checked.append(str(candidate))
        if not candidate.is_file():
            continue
        spec = importlib.util.spec_from_file_location('_astra_pylians_readfof', candidate)
        if spec is None or spec.loader is None:
            continue
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    detail = ', '.join(checked) if checked else 'no candidate paths'
    raise ModuleNotFoundError(
        'Could not import Pylians readfof. Pass --readfof-path pointing to readfof.py '
        f'or set PYLIANS_READFOF. Checked: {detail}')


def _read_positions(config: QuijoteRunConfig, paths: QuijoteRunPaths,
                    source_info: Dict[str, int]) -> np.ndarray:
    readfof = _import_readfof(config.readfof_path)
    catalogue = readfof.FoF_catalog(
        str(paths.catalogue_root), SNAPSHOT, long_ids=False,
        swap=False, SFR=False, read_IDs=False)
    source_positions = np.asarray(catalogue.GroupPos)
    expected_shape = (int(source_info['rows']), 3)
    if source_positions.shape != expected_shape:
        raise ValueError(
            f'Unexpected GroupPos shape {source_positions.shape}; expected {expected_shape}')

    positions = np.empty(expected_shape, dtype=np.float64)
    np.divide(source_positions, 1e3, out=positions)
    del source_positions, catalogue
    gc.collect()

    if not np.all(np.isfinite(positions)):
        raise ValueError('GroupPos contains non-finite values')
    lower = np.asarray(config.box_origin, dtype=np.float64)
    upper = lower + float(config.box_size)
    minimum = np.min(positions, axis=0)
    maximum = np.max(positions, axis=0)
    tolerance = np.maximum(1.0, np.maximum(np.abs(lower), np.abs(upper))) * 1e-10
    if np.any(minimum < lower - tolerance) or np.any(maximum > upper + tolerance):
        raise ValueError(
            f'GroupPos range {minimum.tolist()}..{maximum.tolist()} lies outside '
            f'configured box {lower.tolist()}..{upper.tolist()}')
    return np.ascontiguousarray(positions)


def _base_header(config: QuijoteRunConfig,
                 source_info: Dict[str, int]) -> Dict[str, object]:
    lower = np.asarray(config.box_origin, dtype=np.float64)
    upper = lower + float(config.box_size)
    return {'MODE': 'QUIJOTE',
            'VERSION': 1,
            'PARAM': normalize_parameter(config.parameter),
            'REALIZ': int(config.realization),
            'SNAPNUM': SNAPSHOT,
            'REDSHIFT': REDSHIFT,
            'BASESEED': int(config.random_seed),
            'SRCROWS': int(source_info['rows']),
            'SRCNFILE': int(source_info['n_files']),
            'SRCSIZE': int(source_info['size']),
            'SRCMTIME': int(source_info['mtime_ns']),
            'ROWALIGN': 'GROUPPOS',
            'POSUNIT': 'Mpc/h',
            'BOXLOX': float(lower[0]),
            'BOXLOY': float(lower[1]),
            'BOXLOZ': float(lower[2]),
            'BOXHIX': float(upper[0]),
            'BOXHIY': float(upper[1]),
            'BOXHIZ': float(upper[2]),
            'NITER': int(config.n_iterations),
            'RLOWER': float(config.r_lower),
            'RMED': float(config.r_med),
            'RUPPER': float(config.r_upper),
            'PAIRFILE': 0,
            'CLASSFIL': 0,
            'RANDSAVE': 0}


def _random_void_header(config: QuijoteRunConfig,
                        source_info: Dict[str, int]) -> Dict[str, object]:
    header = _base_header(config, source_info)
    header.update({'PRODUCT': 'RANDVOID',
                   'WEBCLASS': 'VOID',
                   'ISDATA': 0,
                   'ROWALIGN': 'RANDITER',
                   'CLASSFIL': 1,
                   'RANDSAVE': 1})
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


def _probability_product_matches(path: Path, expected_header: Dict[str, object],
                                 expected_rows: int) -> Tuple[bool, str]:
    if not path.is_file():
        return False, 'missing'
    try:
        with fitsio.FITS(str(path), mode='r') as hdus:
            if len(hdus) < 2:
                return False, 'missing table HDU'
            table = hdus[1]
            if int(table.get_nrows()) != int(expected_rows):
                return False, 'row count differs'
            if tuple(table.get_colnames()) != tuple(_PROBABILITY_DTYPE.names):
                return False, 'probability columns differ'
            return _headers_match(table.read_header(), expected_header)
    except Exception as exc:
        return False, str(exc)


def _random_void_product_matches(path: Path,
                                 expected_header: Dict[str, object]) -> Tuple[bool, str]:
    if not path.is_file():
        return False, 'missing'
    try:
        with fitsio.FITS(str(path), mode='r') as hdus:
            if len(hdus) < 2:
                return False, 'missing table HDU'
            table = hdus[1]
            if tuple(table.get_colnames()) != tuple(_RANDOM_VOID_DTYPE.names):
                return False, 'random-void columns differ'
            return _headers_match(table.read_header(), expected_header)
    except Exception as exc:
        return False, str(exc)


def _probability_chunks(counts: np.ndarray, n_iterations: int,
                        chunk_rows: int) -> Iterator[np.ndarray]:
    total = counts.shape[1]
    inverse = np.float32(1.0 / float(n_iterations))
    fields = ('PVOID', 'PSHEET', 'PFILAMENT', 'PKNOT')
    for start in range(0, total, int(chunk_rows)):
        stop = min(start + int(chunk_rows), total)
        chunk = np.empty(stop - start, dtype=_PROBABILITY_DTYPE)
        for class_index, field in enumerate(fields):
            chunk[field] = counts[class_index, start:stop] * inverse
        yield chunk


def _write_probability(path: Path, counts: np.ndarray, n_iterations: int,
                       chunk_rows: int, header: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        f'.{path.stem}.tmp.{os.getpid()}{"".join(path.suffixes)}')
    try:
        with fitsio.FITS(str(temporary), mode='rw', clobber=True) as hdus:
            hdus.create_table_hdu(dtype=_PROBABILITY_DTYPE, extname='ASTRA')
            table = hdus[-1]
            table.write_keys(header)
            for chunk in _probability_chunks(counts, n_iterations, chunk_rows):
                if len(chunk):
                    table.append(chunk)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


@contextmanager
def _random_void_writer(path: Path, header: Dict[str, object], enabled: bool):
    """Atomically stream selected random-void rows into one compressed FITS."""
    if not enabled:
        yield None
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        f'.{path.stem}.tmp.{os.getpid()}{"".join(path.suffixes)}')
    hdus = None
    try:
        hdus = fitsio.FITS(str(temporary), mode='rw', clobber=True)
        hdus.create_table_hdu(dtype=_RANDOM_VOID_DTYPE, extname='RANDOM_VOIDS')
        table = hdus[-1]
        table.write_keys(header)
        yield table
        hdus.close()
        hdus = None
        os.replace(temporary, path)
    finally:
        if hdus is not None:
            hdus.close()
        if temporary.exists():
            temporary.unlink()


def _seed_sequence(config: QuijoteRunConfig,
                   iteration: int) -> np.random.SeedSequence:
    parameter_code = zlib.crc32(
        normalize_parameter(config.parameter).encode('utf-8'))
    return np.random.SeedSequence([int(config.random_seed),
                                   int(parameter_code),
                                   int(config.realization),
                                   SNAPSHOT,
                                   int(iteration)])


def _init_worker(state: Dict[str, object]) -> None:
    global _WORKER_STATE
    _WORKER_STATE = state


def _run_iteration_worker(iteration: int):
    state = _WORKER_STATE
    if state is None:
        raise RuntimeError('Quijote worker state was not initialized')
    config = state['config']
    data_coords = state['coordinates']
    n_data = int(data_coords.shape[0])
    start_time = time.time()
    print(f'quijote-astra --> {config.parameter}/{config.realization} '
          f'iteration={iteration:03d} points={2 * n_data:,}', flush=True)

    points = state.get('points')
    if points is None:
        points = np.empty((2 * n_data, 3), dtype=np.float64)
        points[:n_data] = data_coords
        state['points'] = points
    random_points = points[n_data:]
    rng = np.random.default_rng(_seed_sequence(config, iteration))
    rng.random(random_points.shape, dtype=np.float64, out=random_points)
    random_points *= float(config.box_size)
    random_points += np.asarray(config.box_origin, dtype=np.float64)

    ndata, nrand = compute_delaunay_neighbor_counts(
        points, n_data=n_data, n_vertices=2 * n_data,
        chunk_vertices=config.count_chunk_vertices,
        qhull_options=config.qhull_options)
    classes = classify_neighbor_counts(
        ndata, nrand, config.r_lower, config.r_med, config.r_upper)
    del ndata, nrand

    data_classes = None
    if state['need_probability']:
        data_classes = classes[:n_data].copy()

    random_void_rows = None
    if state['need_random_voids']:
        random_void_indices = np.flatnonzero(classes[n_data:] == 0)
        random_void_rows = np.empty(len(random_void_indices),
                                    dtype=_RANDOM_VOID_DTYPE)
        random_void_rows['RANDITER'] = int(iteration)
        random_void_rows['RANDINDEX'] = random_void_indices
        selected = random_points[random_void_indices]
        random_void_rows['X'] = selected[:, 0]
        random_void_rows['Y'] = selected[:, 1]
        random_void_rows['Z'] = selected[:, 2]
        del selected, random_void_indices
    del classes
    elapsed = time.time() - start_time
    print(f'quijote-astra --> {config.parameter}/{config.realization} '
          f'iteration={iteration:03d} done in {elapsed:.1f}s', flush=True)
    return int(iteration), data_classes, random_void_rows, float(elapsed)


def _resolve_workers(config: QuijoteRunConfig) -> int:
    if int(config.n_iterations) <= 1:
        return 1
    requested = config.iteration_workers
    if requested is None:
        env_value = os.environ.get('QUIJOTE_ITER_WORKERS', '').strip()
        if env_value:
            requested = int(env_value)
        elif os.environ.get('SLURM_JOB_ID'):
            requested = int(os.environ.get('SLURM_CPUS_PER_TASK', '1'))
        else:
            requested = 1
    if int(requested) <= 0:
        raise ValueError('iteration_workers must be greater than zero')
    return min(int(requested), int(config.n_iterations))


def _accumulate_classes(counts: np.ndarray, classes: np.ndarray) -> None:
    for class_index in range(4):
        counts[class_index] += (classes == class_index)


def _consume_iteration_results(results, probability_counts,
                               random_void_table) -> None:
    for _, data_classes, random_void_rows, _ in results:
        if data_classes is not None:
            _accumulate_classes(probability_counts, data_classes)
        if random_void_table is not None and len(random_void_rows):
            random_void_table.append(random_void_rows)
        del data_classes, random_void_rows


def _run_locked(config: QuijoteRunConfig,
                paths: QuijoteRunPaths) -> QuijoteRunPaths:
    global _WORKER_STATE
    source_info = _inspect_catalog(paths)
    probability_header = _base_header(config, source_info)
    random_void_header = _random_void_header(config, source_info)
    probability_matches, probability_reason = _probability_product_matches(
        paths.probability, probability_header, int(source_info['rows']))
    random_voids_match, random_voids_reason = _random_void_product_matches(
        paths.random_voids, random_void_header)
    if probability_matches and random_voids_match and not config.force:
        print(f'quijote-astra --> complete; reusing {paths.probability} '
              f'and {paths.random_voids}', flush=True)
        return paths
    if paths.probability.exists() and not probability_matches and not config.force:
        raise RuntimeError(
            f'Existing probability product is incompatible ({probability_reason}): '
            f'{paths.probability}. Use --force to replace it.')
    if paths.random_voids.exists() and not random_voids_match and not config.force:
        raise RuntimeError(
            f'Existing random-void product is incompatible ({random_voids_reason}): '
            f'{paths.random_voids}. Use --force to replace it.')

    need_probability = bool(config.force or not probability_matches)
    need_random_voids = bool(config.force or not random_voids_match)

    n_rows = int(source_info['rows'])
    print(f'quijote-astra --> {config.parameter}/{config.realization} '
          f'rows={n_rows:,} iterations={config.n_iterations} '
          f'probability={"compute" if need_probability else "reuse"} '
          f'random-voids={"compute" if need_random_voids else "reuse"} '
          'pairs=disabled',
          flush=True)
    coordinates = _read_positions(config, paths, source_info)
    count_dtype = np.uint8 if int(config.n_iterations) <= 255 else np.uint16
    probability_counts = (np.zeros((4, n_rows), dtype=count_dtype)
                          if need_probability else None)
    worker_count = _resolve_workers(config)
    print(f'quijote-astra --> iteration workers={worker_count}', flush=True)
    state = {'config': config,
             'coordinates': coordinates,
             'points': None,
             'need_probability': need_probability,
             'need_random_voids': need_random_voids}
    iterations = range(int(config.n_iterations))
    if worker_count > 1:
        context = mp.get_context('fork')
        with context.Pool(processes=worker_count,
                          initializer=_init_worker,
                          initargs=(state,)) as pool:
            with _random_void_writer(paths.random_voids, random_void_header,
                                     need_random_voids) as random_void_table:
                results = pool.imap(_run_iteration_worker, iterations)
                _consume_iteration_results(results, probability_counts,
                                           random_void_table)
    else:
        _init_worker(state)
        with _random_void_writer(paths.random_voids, random_void_header,
                                 need_random_voids) as random_void_table:
            results = map(_run_iteration_worker, iterations)
            _consume_iteration_results(results, probability_counts,
                                       random_void_table)

    del coordinates
    state.clear()
    _WORKER_STATE = None
    gc.collect()
    if need_probability:
        _write_probability(paths.probability, probability_counts,
                           int(config.n_iterations), int(config.io_chunk_rows),
                           probability_header)
        print(f'quijote-astra --> wrote {paths.probability}', flush=True)
    if need_random_voids:
        print(f'quijote-astra --> wrote {paths.random_voids}', flush=True)
    return paths


def run_quijote_pipeline(config: QuijoteRunConfig) -> QuijoteRunPaths:
    """Run ASTRA for one Quijote z=0.5 FoF catalogue."""
    _validate_config(config)
    paths = resolve_run_paths(config)
    paths.output_dir.mkdir(parents=True, exist_ok=True)
    with open(paths.lock, 'a', encoding='utf-8') as lock_stream:
        try:
            fcntl.flock(lock_stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(
                f'Another process is already running {paths.output_dir}') from exc
        return _run_locked(config, paths)


__all__ = ['DEFAULT_BOX_ORIGIN',
           'DEFAULT_BOX_SIZE',
           'DEFAULT_INPUT_ROOT',
           'DEFAULT_N_ITERATIONS',
           'DEFAULT_OUTPUT_ROOT',
           'QuijoteRunConfig',
           'QuijoteRunPaths',
           'REDSHIFT',
           'SNAPSHOT',
           'normalize_parameter',
           'read_manifest_entry',
           'resolve_run_paths',
           'run_quijote_pipeline']
