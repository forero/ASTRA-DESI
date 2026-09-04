from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from contextlib import redirect_stderr, redirect_stdout
import datetime as dt
import json
import multiprocessing as mp
import os
from pathlib import Path
import sys
import time
import traceback
from types import SimpleNamespace


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_INPUT_ROOT = Path('/pscratch/sd/v/vtorresg/quijotes/void_finder/FoF')
DEFAULT_PROCESS_CAP = 64
DEFAULT_IO_WORKERS = 8
DEFAULT_SCAN_CAP = 64

_WORKER_CONFIG = None
_IO_GATE = None
_QUIJOTE = None
_BUILD_CONSENSUS = None


def _positive_int(value: str) -> int:
    result = int(value)
    if result < 1:
        raise argparse.ArgumentTypeError('value must be a positive integer')
    return result


def _nonnegative_int(value: str) -> int:
    result = int(value)
    if result < 0:
        raise argparse.ArgumentTypeError('value must be non-negative')
    return result


def _parameter(value: str) -> str:
    value = value.strip()
    if not value or value in {'.', '..'} or '/' in value or '\\' in value:
        raise argparse.ArgumentTypeError(f'invalid parameter name: {value!r}')
    return value


def _parse_iterations(values: list[str]) -> tuple[int, ...]:
    iterations = []
    for value in values:
        for token in value.split(','):
            token = token.strip()
            if not token:
                continue
            if '-' in token:
                pieces = token.split('-', 1)
                try:
                    start, stop = (int(piece) for piece in pieces)
                except ValueError as exc:
                    raise ValueError(f'invalid iteration range: {token!r}') from exc
                if start < 0 or stop < start or stop >= 1000:
                    raise ValueError(f'invalid iteration range: {token!r}')
                iterations.extend(range(start, stop + 1))
            else:
                try:
                    iteration = int(token)
                except ValueError as exc:
                    raise ValueError(f'invalid iteration: {token!r}') from exc
                if iteration < 0 or iteration >= 1000:
                    raise ValueError('iterations must lie in [0, 1000)')
                iterations.append(iteration)
    result = tuple(dict.fromkeys(iterations))
    if not result:
        raise ValueError('at least one iteration is required')
    return result


def _allocated_cpus() -> int:
    for name in ('SLURM_CPUS_PER_TASK', 'SLURM_CPUS_ON_NODE'):
        raw = os.environ.get(name, '').split('(', 1)[0]
        if raw.isdigit() and int(raw) > 0:
            return int(raw)
    return os.cpu_count() or 1


def _available_memory_gib() -> float | None:
    try:
        with open('/proc/meminfo', encoding='ascii') as handle:
            fields = {line.split(':', 1)[0]: line.split()[1] for line in handle}
        return int(fields['MemAvailable']) / 1024**2
    except (OSError, KeyError, ValueError):
        return None


def _directory_names(path: Path) -> set[str] | None:
    """Return entry names with one getdents call and no per-file stat calls."""
    try:
        with os.scandir(path) as entries:
            return {entry.name for entry in entries}
    except OSError:
        return None


def _candidate_cases(input_root: Path,
                     parameters: set[str] | None,
                     realization_min: int,
                     realization_max: int) -> list[tuple[str, int]]:
    cases = []
    try:
        parameter_entries = list(os.scandir(input_root))
    except OSError as exc:
        raise RuntimeError(f'cannot scan input root {input_root}: {exc}') from exc
    for parameter_entry in sorted(parameter_entries, key=lambda item: item.name):
        parameter = parameter_entry.name
        if parameters is not None and parameter not in parameters:
            continue
        try:
            realization_entries = os.scandir(parameter_entry.path)
        except OSError:
            continue
        with realization_entries:
            for realization_entry in realization_entries:
                if not realization_entry.name.isdigit():
                    continue
                realization = int(realization_entry.name)
                if realization_min <= realization <= realization_max:
                    cases.append((parameter, realization))
    return sorted(cases, key=lambda item: (item[0], item[1]))


def _quick_output_complete(case_root: Path,
                           parameter: str,
                           realization: int,
                           snapshot: int,
                           iterations: tuple[int, ...],
                           vol_frac: float,
                           v_cut: float,
                           keep_all: bool) -> bool:
    suffix = '_all' if keep_all else ''
    output_root = case_root / f'consensus_n{len(iterations)}{suffix}'
    names = _directory_names(output_root)
    if names is None or not {
            'voids_consensus.fits', 'voids_consensus.npy', 'summary.json'
    }.issubset(names):
        return False
    try:
        with open(output_root / 'summary.json', encoding='utf-8') as handle:
            summary = json.load(handle)
    except (OSError, TypeError, ValueError):
        return False
    expected = {
        'consensus_version': 1,
        'parameter': parameter,
        'realization': realization,
        'snapshot': snapshot,
        'iterations': list(iterations),
        'vol_frac': float(vol_frac),
        'v_cut': None if keep_all else float(v_cut),
        'keep_all': bool(keep_all),
    }
    return all(summary.get(name) == value for name, value in expected.items())


def _scan_case(case: tuple[str, int], config: dict) -> tuple[str, str, int, int | None]:
    parameter, realization = case
    case_root = (Path(config['input_root']) / parameter / str(realization) /
                 f"groups_{config['snapshot']:03d}")
    for iteration in config['iterations']:
        names = _directory_names(case_root / f'random_{iteration:03d}')
        if names is None or not {'summary.json', 'voids_all.fits'}.issubset(names):
            return 'incomplete', parameter, realization, iteration
    if _quick_output_complete(case_root, parameter, realization,
                              config['snapshot'], config['iterations'],
                              config['vol_frac'], config['v_cut'],
                              config['keep_all']):
        return 'existing', parameter, realization, None
    return 'ready', parameter, realization, None


def _scan_cases(cases: list[tuple[str, int]],
                config: dict,
                scan_workers: int):
    ready = []
    existing = []
    incomplete = []
    with ThreadPoolExecutor(max_workers=scan_workers) as executor:
        futures = [executor.submit(_scan_case, case, config) for case in cases]
        for future in as_completed(futures):
            status, parameter, realization, missing = future.result()
            record = (parameter, realization)
            if status == 'ready':
                ready.append(record)
            elif status == 'existing':
                existing.append(record)
            else:
                incomplete.append((parameter, realization, missing))
    ready.sort(key=lambda item: (item[0], item[1]))
    existing.sort(key=lambda item: (item[0], item[1]))
    incomplete.sort(key=lambda item: (item[0], item[1]))
    return ready, existing, incomplete


def _write_manifests(state_root: Path,
                     stamp: str,
                     ready: list[tuple[str, int]],
                     existing: list[tuple[str, int]],
                     incomplete: list[tuple[str, int, int | None]]) -> dict[str, Path]:
    state_root.mkdir(parents=True, exist_ok=True)
    paths = {
        'ready': state_root / f'ready_{stamp}.txt',
        'existing': state_root / f'existing_{stamp}.txt',
        'incomplete': state_root / f'skipped_incomplete_{stamp}.txt',
    }
    paths['ready'].write_text(
        ''.join(f'{parameter} {realization}\n' for parameter, realization in ready),
        encoding='utf-8')
    paths['existing'].write_text(
        ''.join(f'{parameter} {realization}\n' for parameter, realization in existing),
        encoding='utf-8')
    paths['incomplete'].write_text(
        ''.join(f'{parameter} {realization} first_missing={missing}\n'
                for parameter, realization, missing in incomplete),
        encoding='utf-8')
    return paths


def _init_worker(config: dict, io_gate) -> None:
    global _WORKER_CONFIG, _IO_GATE, _QUIJOTE, _BUILD_CONSENSUS
    _WORKER_CONFIG = config
    _IO_GATE = io_gate
    from catalog import run_quijote_consensus as quijote
    from group_finder.consensus import build_consensus_catalog
    _QUIJOTE = quijote
    _BUILD_CONSENSUS = build_consensus_catalog


def _warm_worker_modules() -> None:
    """Import heavy modules and build shared caches once before forking."""
    global _QUIJOTE, _BUILD_CONSENSUS
    from catalog import run_quijote_consensus as quijote
    from group_finder.consensus import build_consensus_catalog
    _QUIJOTE = quijote
    _BUILD_CONSENSUS = build_consensus_catalog


def _worker_arguments(parameter: str, realization: int):
    config = _WORKER_CONFIG
    return SimpleNamespace(
        parameter=parameter,
        realization=realization,
        input_root=config['input_root'],
        snapshot=config['snapshot'],
        iterations=[str(value) for value in config['iterations']],
        output_dir=None,
        vol_frac=config['vol_frac'],
        v_cut=config['v_cut'],
        keep_all=config['keep_all'],
        query_workers=config['query_workers'],
        query_batch_size=config['query_batch_size'],
        resume=True,
        repair_incomplete=True,
        overwrite=False,
        quiet=False,
    )


def _run_case(case: tuple[str, int]) -> tuple[str, str, int, float, str]:
    parameter, realization = case
    config = _WORKER_CONFIG
    log_path = Path(config['log_root']) / parameter / f'{realization}.log'
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    timestamp = dt.datetime.now(dt.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
    try:
        with open(log_path, 'a', encoding='utf-8', buffering=1) as log_handle:
            with redirect_stdout(log_handle), redirect_stderr(log_handle):
                print(f'[{timestamp}] START parameter={parameter} '
                      f'realization={realization}', flush=True)
                args = _worker_arguments(parameter, realization)
                iterations = tuple(config['iterations'])
                output_dir = _QUIJOTE._output_dir(args, iterations)
                paths = _QUIJOTE._output_paths(output_dir)
                expected = _QUIJOTE._expected_output_config(args, iterations)
                state = _QUIJOTE._output_state(paths, expected)
                if state == 'complete':
                    print(f'[{timestamp}] SKIP complete output', flush=True)
                    return ('existing', parameter, realization,
                            time.monotonic() - started, str(log_path))
                if state == 'incompatible':
                    raise RuntimeError(f'incompatible existing output: {output_dir}')
                case_root = (Path(config['input_root']) / parameter /
                             str(realization) / f"groups_{config['snapshot']:03d}")
                with _IO_GATE:
                    pool, input_config = _QUIJOTE.load_quijote_pool(
                        case_root, args, iterations)
                result = _BUILD_CONSENSUS(
                    pool,
                    vol_frac=config['vol_frac'],
                    v_cut=config['v_cut'],
                    keep_all=config['keep_all'],
                    query_workers=config['query_workers'],
                    query_batch_size=config['query_batch_size'],
                    verbose=True)
                elapsed = time.monotonic() - started
                _QUIJOTE._write_outputs(
                    paths, args, iterations, pool, result, input_config, elapsed,
                    overwrite=(state == 'incomplete'))
                finished = dt.datetime.now(dt.timezone.utc).strftime(
                    '%Y-%m-%dT%H:%M:%SZ')
                print(f'[{finished}] DONE rows={len(result.catalog)} '
                      f'elapsed={elapsed:.1f}s', flush=True)
        return 'ok', parameter, realization, time.monotonic() - started, str(log_path)
    except BaseException:
        try:
            with open(log_path, 'a', encoding='utf-8') as log_handle:
                traceback.print_exc(file=log_handle)
                print(f'FAILED elapsed={time.monotonic() - started:.1f}s',
                      file=log_handle, flush=True)
        except OSError:
            pass
        return 'failed', parameter, realization, time.monotonic() - started, str(log_path)


def _configure_environment() -> None:
    for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                 'NUMEXPR_NUM_THREADS', 'NUMBA_NUM_THREADS'):
        os.environ[name] = '1'
    os.environ['MALLOC_ARENA_MAX'] = '2'
    cache_label = os.environ.get('SLURM_JOB_ID', str(os.getpid()))
    cache_root = Path('/tmp') / f'astra-quijote-consensus-{cache_label}'
    os.environ['MPLCONFIGDIR'] = str(cache_root / 'matplotlib')
    os.environ['XDG_CACHE_HOME'] = str(cache_root / 'cache')
    Path(os.environ['MPLCONFIGDIR']).mkdir(parents=True, exist_ok=True)
    Path(os.environ['XDG_CACHE_HOME']).mkdir(parents=True, exist_ok=True)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--input-root', type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument('--snapshot', type=_nonnegative_int, default=3)
    parser.add_argument('--iterations', nargs='+', default=['0-99'],
                        metavar='N|START-STOP')
    parser.add_argument('--parameters', nargs='+', type=_parameter,
                        help='only these cosmology directories (default: all)')
    parser.add_argument('--realization-min', type=_nonnegative_int, default=0)
    parser.add_argument('--realization-max', type=_nonnegative_int,
                        default=2_147_483_647)
    parser.add_argument('--workers', type=_positive_int,
                        help='consensus processes (default: min(CPUs, 64))')
    parser.add_argument('--io-workers', type=_positive_int,
                        default=DEFAULT_IO_WORKERS,
                        help='simultaneous 100-FITS readers (default: 8)')
    parser.add_argument('--scan-workers', type=_positive_int,
                        help='parallel readiness checks (default: min(CPUs, 64))')
    parser.add_argument('--query-workers', type=_positive_int,
                        help='cKDTree threads per consensus (default: auto)')
    parser.add_argument('--query-batch-size', type=_positive_int, default=4096)
    parser.add_argument('--progress-every', type=_positive_int, default=10)
    parser.add_argument('--vol-frac', type=float, default=0.5)
    parser.add_argument('--v-cut', type=float, default=0.5)
    parser.add_argument('--keep-all', action='store_true')
    parser.add_argument('--state-root', type=Path)
    parser.add_argument('--dry-run', action='store_true')
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if not os.environ.get('SLURM_JOB_ID'):
        parser.error('run inside a Slurm interactive allocation')
    try:
        iterations = _parse_iterations(args.iterations)
    except ValueError as exc:
        parser.error(str(exc))
    if args.realization_max < args.realization_min:
        parser.error('--realization-max must be >= --realization-min')
    if not 0.5 <= args.vol_frac <= 1.0:
        parser.error('--vol-frac must lie in [0.5, 1]')
    if not 0.0 <= args.v_cut <= 1.0:
        parser.error('--v-cut must lie in [0, 1]')

    input_root = args.input_root.expanduser().resolve()
    state_root = ((args.state_root.expanduser().resolve()) if args.state_root
                  else input_root / 'consensus_run_state_py')
    _configure_environment()

    cpus = _allocated_cpus()
    workers = args.workers or min(cpus, DEFAULT_PROCESS_CAP)
    scan_workers = args.scan_workers or min(cpus, DEFAULT_SCAN_CAP)
    io_workers = min(args.io_workers, workers)
    query_workers = args.query_workers or max(1, min(4, cpus // workers))
    memory_gib = _available_memory_gib()
    if workers > cpus:
        parser.error(f'--workers={workers} exceeds visible CPUs={cpus}')
    if workers * query_workers > cpus:
        parser.error('--workers * --query-workers exceeds visible CPUs; '
                     'reduce one of them')

    config = {
        'input_root': str(input_root),
        'snapshot': args.snapshot,
        'iterations': iterations,
        'vol_frac': args.vol_frac,
        'v_cut': args.v_cut,
        'keep_all': args.keep_all,
        'query_workers': query_workers,
        'query_batch_size': args.query_batch_size,
        'log_root': str(state_root / 'logs'),
    }
    parameters = set(args.parameters) if args.parameters else None
    print(f'[setup] CPUs={cpus}; processes={workers}; '
          f'FITS readers={io_workers}; query threads/process={query_workers}',
          flush=True)
    if memory_gib is not None:
        print(f'[setup] available RAM={memory_gib:.1f} GiB', flush=True)
    print(f'[scan] searching {input_root} for all {len(iterations)} iterations...',
          flush=True)
    cases = _candidate_cases(input_root, parameters, args.realization_min,
                             args.realization_max)
    ready, existing, incomplete = _scan_cases(cases, config, scan_workers)
    stamp = dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    manifests = _write_manifests(state_root, stamp, ready, existing, incomplete)
    print(f'[scan] candidates={len(cases)} ready_to_run={len(ready)} '
          f'already_complete={len(existing)} skipped_incomplete={len(incomplete)}',
          flush=True)
    print(f'[scan] ready manifest: {manifests["ready"]}', flush=True)
    print(f'[scan] existing manifest: {manifests["existing"]}', flush=True)
    print(f'[scan] skipped manifest: {manifests["incomplete"]}', flush=True)
    if args.dry_run or not ready:
        print('[done] dry run; nothing was executed.' if args.dry_run
              else '[done] no pending ready consensus catalogues.', flush=True)
        return 0

    # Scanning threads are already joined here.  Import NumPy/SciPy/fitsio once
    # in the single-threaded parent, then fork: worker processes share clean
    # module pages copy-on-write and avoid 64 simultaneous import/cache storms.
    print('[setup] warming scientific modules once before worker startup...',
          flush=True)
    _warm_worker_modules()
    ctx = mp.get_context('fork')
    io_gate = ctx.BoundedSemaphore(io_workers)
    started = time.monotonic()
    ok = failed = skipped = 0
    total = len(ready)
    print(f'[run] launching {total} consensus catalogues; progress every '
          f'{args.progress_every} completions.', flush=True)
    with ProcessPoolExecutor(max_workers=workers,
                             mp_context=ctx,
                             initializer=_init_worker,
                             initargs=(config, io_gate)) as executor:
        futures = [executor.submit(_run_case, case) for case in ready]
        try:
            for completed, future in enumerate(as_completed(futures), start=1):
                status, parameter, realization, elapsed, log_path = future.result()
                if status == 'ok':
                    ok += 1
                elif status == 'existing':
                    skipped += 1
                else:
                    failed += 1
                    print(f'[failed] {parameter}/{realization}; log={log_path}',
                          flush=True)
                if completed % args.progress_every == 0 or completed == total:
                    wall = time.monotonic() - started
                    rate = completed / wall if wall else 0.0
                    eta = ((total - completed) / rate) if rate else float('inf')
                    overall = len(existing) + completed
                    print(f'[progress] {completed}/{total} submitted-ready done; '
                          f'{overall}/{len(existing) + total} total complete-or-done; '
                          f'ok={ok} resumed={skipped} failed={failed}; '
                          f'rate={rate:.2f}/s ETA={eta / 60:.1f} min', flush=True)
        except KeyboardInterrupt:
            print('[interrupt] stopping new work and waiting for active atomic '
                  'catalogue writes to finish safely...', flush=True)
            for future in futures:
                future.cancel()
            return 130

    wall = time.monotonic() - started
    print(f'[done] new={ok} resumed={skipped} failed={failed} '
          f'already_complete={len(existing)} elapsed={wall / 60:.1f} min',
          flush=True)
    return 1 if failed else 0


if __name__ == '__main__':
    raise SystemExit(main())
