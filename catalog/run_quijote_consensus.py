import argparse
import datetime
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('NUMEXPR_NUM_THREADS', '1')
os.environ.setdefault('MPLCONFIGDIR', '/tmp/astra-desi-matplotlib')
os.environ.setdefault('XDG_CACHE_HOME', '/tmp/astra-desi-cache')

import fitsio
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from group_finder.consensus import (CONSENSUS_DTYPE, PooledVoids,
                                    build_consensus_catalog)
from group_finder.make_cat import EIGENVALUE_COLUMNS, VOID_SHAPE_COLUMNS


DEFAULT_INPUT_ROOT = '/pscratch/sd/v/vtorresg/quijotes/void_finder/FoF'
DEFAULT_SNAPSHOT = 3
DEFAULT_VOL_FRAC = 0.5
DEFAULT_V_CUT = 0.5
_CONSENSUS_VERSION = 1
_INPUT_COLUMNS = (('VOID_ID', 'XCART', 'YCART', 'ZCART', 'R_EFF', 'ELLIP') +
                  VOID_SHAPE_COLUMNS + ('BORDER',))
_COMMON_SUMMARY_KEYS = ('box_origin_mpc_h', 'box_size_mpc_h', 'n_data',
                        'n_random', 'random_factor', 'random_seed', 'r_threshold',
                        'min_members', 'min_halo_mass_msun_h',
                        'max_halo_mass_msun_h', 'min_halo_particles',
                        'boundary_buffer_requested', 'boundary_buffer_mpc_h',
                        'source_total_groups', 'source_total_ids')


def _parameter_name(value):
    value = str(value).strip()
    if not value or value in ('.', '..') or '/' in value or '\\' in value:
        raise argparse.ArgumentTypeError(f'Invalid Quijote parameter: {value!r}')
    return value


def _nonnegative_int(value):
    result = int(value)
    if result < 0:
        raise argparse.ArgumentTypeError('value must be non-negative')
    return result


def _parse_iteration_tokens(values):
    iterations = []
    for value in values:
        for token in str(value).split(','):
            token = token.strip()
            if not token:
                continue
            if '-' in token:
                try:
                    start, stop = map(int, token.split('-', 1))
                except ValueError:
                    raise ValueError(f'Invalid iteration range: {token!r}')
                if start < 0 or stop < start or stop >= 1000:
                    raise ValueError(f'Invalid iteration range: {token!r}')
                iterations.extend(range(start, stop + 1))
            else:
                try:
                    iteration = int(token)
                except ValueError:
                    raise ValueError(f'Invalid iteration: {token!r}')
                if iteration < 0 or iteration >= 1000:
                    raise ValueError('iterations must lie in [0, 1000)')
                iterations.append(iteration)
    result = tuple(dict.fromkeys(iterations))
    if not result:
        raise ValueError('at least one iteration is required')
    return result


def _output_dir(args, iterations):
    if args.output_dir:
        return Path(args.output_dir).expanduser().resolve()
    suffix = '_all' if args.keep_all else ''
    return (Path(args.input_root).expanduser().resolve() / args.parameter /
            str(args.realization) / f'groups_{args.snapshot:03d}' /
            f'consensus_n{len(iterations)}{suffix}')


def _output_paths(output_dir):
    return {'fits': output_dir / 'voids_consensus.fits',
            'npy': output_dir / 'voids_consensus.npy',
            'summary': output_dir / 'summary.json'}


def _expected_output_config(args, iterations):
    return {'consensus_version': _CONSENSUS_VERSION,
            'parameter': args.parameter,
            'realization': args.realization,
            'snapshot': args.snapshot,
            'iterations': list(iterations),
            'vol_frac': float(args.vol_frac),
            'v_cut': None if args.keep_all else float(args.v_cut),
            'keep_all': bool(args.keep_all)}


def _output_state(paths, expected):
    existing = [path for path in paths.values() if path.exists()]
    if not existing:
        return 'missing'
    try:
        summary = json.loads(paths['summary'].read_text(encoding='utf-8'))
    except (OSError, TypeError, ValueError):
        return 'incomplete'
    if not all(summary.get(key) == value for key, value in expected.items()):
        return 'incompatible'
    if not all(path.is_file() and path.stat().st_size > 0 for path in paths.values()):
        return 'incomplete'
    try:
        with fitsio.FITS(str(paths['fits'])) as catalog:
            fits_names = set(catalog[1].get_colnames())
        npy_names = set(np.load(paths['npy'], mmap_mode='r',
                                allow_pickle=False).dtype.names or ())
    except (OSError, IndexError, TypeError, ValueError):
        return 'incomplete'
    required = set(CONSENSUS_DTYPE.names)
    return 'complete' if required.issubset(fits_names & npy_names) else 'incomplete'


def _iteration_paths(case_root, iteration):
    root = case_root / f'random_{iteration:03d}'
    return root / 'voids_all.fits', root / 'summary.json'


def _read_iteration_summary(path, args, iteration):
    if not path.is_file():
        raise FileNotFoundError(f'Missing iteration summary: {path}')
    try:
        summary = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, TypeError, ValueError) as exc:
        raise ValueError(f'Invalid iteration summary {path}: {exc}')
    expected = {'parameter': args.parameter,
                'realization': args.realization,
                'snapshot': args.snapshot,
                'random_iteration': iteration}
    mismatches = [key for key, value in expected.items()
                  if summary.get(key) != value]
    if mismatches:
        raise ValueError(f'Iteration summary {path} has incompatible fields: '
                         + ', '.join(mismatches))
    missing = [key for key in _COMMON_SUMMARY_KEYS if key not in summary]
    if missing:
        raise ValueError(f'Iteration summary {path} is missing: '
                         + ', '.join(missing))
    return summary


def _validate_common_config(reference, candidate, path):
    mismatches = [key for key in _COMMON_SUMMARY_KEYS
                  if candidate.get(key) != reference.get(key)]
    if mismatches:
        raise ValueError(f'Iteration {path} is scientifically incompatible in: '
                         + ', '.join(mismatches))


def load_quijote_pool(case_root, args, iterations):
    centers = []
    radii = []
    ellipticities = []
    eigenvalues = []
    eigenvectors = []
    origins = []
    void_ids = []
    input_paths = []
    n_read = 0
    n_border = 0
    n_undefined_shape = 0
    reference_summary = None

    for run_index, iteration in enumerate(iterations):
        catalog_path, summary_path = _iteration_paths(case_root, iteration)
        summary = _read_iteration_summary(summary_path, args, iteration)
        if reference_summary is None:
            reference_summary = summary
        else:
            _validate_common_config(reference_summary, summary, summary_path)
        if not catalog_path.is_file() or catalog_path.stat().st_size <= 0:
            raise FileNotFoundError(f'Missing complete iteration catalogue: '
                                    f'{catalog_path}')
        try:
            data = fitsio.read(str(catalog_path), ext=1,
                               columns=list(_INPUT_COLUMNS))
        except (OSError, ValueError) as exc:
            raise ValueError(f'Cannot read consensus input {catalog_path}: {exc}')

        finite_shape = np.isfinite(data['R_EFF']) & np.isfinite(data['ELLIP'])
        for name in VOID_SHAPE_COLUMNS:
            finite_shape &= np.isfinite(data[name])
        border = np.asarray(data['BORDER'], dtype=bool)
        usable = finite_shape & ~border
        if np.any(np.asarray(data['R_EFF'][usable]) <= 0.0):
            raise ValueError(f'Usable voids must have R_EFF > 0 in {catalog_path}')

        count = int(np.count_nonzero(usable))
        centers.append(np.column_stack((data['XCART'][usable],
                                        data['YCART'][usable],
                                        data['ZCART'][usable])).astype(
                                            np.float64, copy=False))
        radii.append(np.asarray(data['R_EFF'][usable], dtype=np.float64))
        ellipticities.append(np.asarray(data['ELLIP'][usable], dtype=np.float64))
        eigenvalues.append(np.column_stack([
            np.asarray(data[name][usable], dtype=np.float64)
            for name in EIGENVALUE_COLUMNS]))
        eigenvectors.append(np.stack([
            np.column_stack([
                np.asarray(data[f'EIGVEC_{axis}_{component}'][usable],
                           dtype=np.float64) for component in ('X', 'Y', 'Z')])
            for axis in range(1, 4)], axis=1))
        origins.append(np.full(count, run_index, dtype=np.int32))

        local_ids = np.asarray(data['VOID_ID'][usable], dtype=np.int64)
        if np.any(local_ids < 0) or np.any(local_ids >= 100_000_000):
            raise ValueError(f'Local VOID_ID outside [0, 100000000) in '
                             f'{catalog_path}')
        if len(np.unique(local_ids)) != len(local_ids):
            raise ValueError(f'Duplicate local VOID_ID values in {catalog_path}')
        void_ids.append(iteration * 100_000_000 + local_ids)
        input_paths.append(catalog_path)
        n_read += len(data)
        n_border += int(np.count_nonzero(border))
        n_undefined_shape += int(np.count_nonzero(~finite_shape))
        if not args.quiet and ((run_index + 1) % 20 == 0
                               or run_index + 1 == len(iterations)):
            print(f'[Quijote consensus] read {run_index + 1}/'
                  f'{len(iterations)} iterations', flush=True)

    n_usable = sum(len(values) for values in radii)
    if n_usable == 0:
        raise ValueError('No finite non-border voids found in requested iterations')
    pool = PooledVoids(
        centers=np.ascontiguousarray(np.concatenate(centers), dtype=np.float64),
        r_eff=np.concatenate(radii),
        ellipticity=np.concatenate(ellipticities),
        eigenvalues=np.ascontiguousarray(np.concatenate(eigenvalues),
                                         dtype=np.float64),
        eigenvectors=np.ascontiguousarray(np.concatenate(eigenvectors),
                                          dtype=np.float64),
        source_iteration=np.concatenate(origins),
        void_id=np.concatenate(void_ids),
        iterations=tuple(iterations),
        input_paths=tuple(input_paths),
        n_read=int(n_read),
        n_border=int(n_border),
        n_undefined_shape=int(n_undefined_shape))
    return pool, reference_summary


def _catalog_statistics(catalog):
    if not len(catalog):
        return None
    percentile = np.percentile(catalog['R_EFF'], (16.0, 84.0))
    return {'median_r_eff': float(np.median(catalog['R_EFF'])),
            'median_ellip': float(np.median(catalog['ELLIP'])),
            'r_eff_p16': float(percentile[0]),
            'r_eff_p84': float(percentile[1])}


def _summary_payload(args, iterations, pool, result, input_config, elapsed):
    payload = _expected_output_config(args, iterations)
    payload.update({
        'algorithm': 'six-step ASTRA consensus catalogue',
        'support_policy': ('all post-pruning representatives' if args.keep_all
                           else 'strictly FRAC_V > v_cut'),
        'representative': 'lower median-radius vote; one largest vote per run',
        'void_id_policy': 'random_iteration * 100000000 + local VOID_ID',
        'input_configuration': {key: input_config[key]
                                for key in _COMMON_SUMMARY_KEYS},
        'inputs': [str(path) for path in pool.input_paths],
        'n_iterations': len(iterations),
        'n_voids_read': pool.n_read,
        'n_border': pool.n_border,
        'n_undefined_shape': pool.n_undefined_shape,
        'n_pooled': result.n_pooled,
        'n_groups': result.n_groups,
        'n_after_pruning': result.n_after_pruning,
        'n_after_support_cut': result.n_after_support_cut,
        'catalog_statistics': _catalog_statistics(result.catalog),
        'elapsed_seconds': float(elapsed)})
    return payload


def _write_outputs(paths, args, iterations, pool, result, input_config,
                   elapsed, overwrite=False):
    output_dir = paths['fits'].parent
    output_dir.mkdir(parents=True, exist_ok=True)
    pid = os.getpid()
    temporary = {'fits': output_dir / f'.voids_consensus.{pid}.tmp.fits',
                 'npy': output_dir / f'.voids_consensus.{pid}.tmp.npy',
                 'summary': output_dir / f'.summary.{pid}.tmp.json'}
    header = [{'name': 'MODE', 'value': 'QUIJOTE'},
              {'name': 'PARAM', 'value': args.parameter},
              {'name': 'REALIZ', 'value': args.realization},
              {'name': 'SNAPNUM', 'value': args.snapshot},
              {'name': 'NITER', 'value': len(iterations)},
              {'name': 'VOLFRAC', 'value': args.vol_frac},
              {'name': 'VCUT', 'value': -1.0 if args.keep_all else args.v_cut},
              {'name': 'NPOOL', 'value': result.n_pooled},
              {'name': 'NROWS', 'value': len(result.catalog)},
              {'name': 'DATE', 'value': datetime.datetime.now().astimezone().isoformat(
                  timespec='seconds')}]
    units = [{'X': 'Mpc/h', 'Y': 'Mpc/h', 'Z': 'Mpc/h',
              'R_EFF': 'Mpc/h', 'EIGVAL_1': '(Mpc/h)^2',
              'EIGVAL_2': '(Mpc/h)^2', 'EIGVAL_3': '(Mpc/h)^2'}.get(name, '')
             for name in result.catalog.dtype.names]
    summary = _summary_payload(args, iterations, pool, result, input_config,
                               elapsed)
    try:
        with fitsio.FITS(str(temporary['fits']), 'rw', clobber=True) as output:
            output.write(result.catalog, header=header, units=units, extname='VOIDS')
        np.save(temporary['npy'], result.catalog, allow_pickle=False)
        temporary['summary'].write_text(
            json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + '\n',
            encoding='utf-8')
        if not overwrite:
            existing = [path for path in paths.values() if path.exists()]
            if existing:
                raise FileExistsError('Consensus outputs already exist: ' +
                                      ', '.join(str(path) for path in existing))
        for name in ('fits', 'npy', 'summary'):
            os.replace(temporary[name], paths[name])
    finally:
        for path in temporary.values():
            if path.exists():
                path.unlink()


def build_parser():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('parameter', type=_parameter_name)
    parser.add_argument('realization', type=_nonnegative_int)
    parser.add_argument('--input-root', default=DEFAULT_INPUT_ROOT)
    parser.add_argument('--snapshot', type=_nonnegative_int, default=DEFAULT_SNAPSHOT)
    parser.add_argument('--iterations', nargs='+', default=['0-99'],
                        metavar='N|START-STOP')
    parser.add_argument('--output-dir', default=None)
    parser.add_argument('--vol-frac', type=float, default=DEFAULT_VOL_FRAC)
    parser.add_argument('--v-cut', type=float, default=DEFAULT_V_CUT)
    parser.add_argument('--keep-all', action='store_true',
                        help='skip the final FRAC_V support cut')
    parser.add_argument('--query-workers', type=int, default=1)
    parser.add_argument('--query-batch-size', type=int, default=4096)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--repair-incomplete', action='store_true')
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--quiet', action='store_true')
    return parser


def _validate_args(args):
    if not np.isfinite(args.vol_frac) or not 0.5 <= args.vol_frac <= 1.0:
        raise ValueError('--vol-frac must lie in [0.5, 1]')
    if not np.isfinite(args.v_cut) or not 0.0 <= args.v_cut <= 1.0:
        raise ValueError('--v-cut must lie in [0, 1]')
    if args.query_workers == 0 or args.query_workers < -1:
        raise ValueError('--query-workers must be -1 or a positive integer')
    if args.query_batch_size < 1:
        raise ValueError('--query-batch-size must be positive')
    if args.resume and args.overwrite:
        raise ValueError('--resume and --overwrite cannot be combined')
    if args.repair_incomplete and not args.resume:
        raise ValueError('--repair-incomplete requires --resume')


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        iterations = _parse_iteration_tokens(args.iterations)
    except ValueError as exc:
        parser.error(str(exc))
    _validate_args(args)

    output_dir = _output_dir(args, iterations)
    paths = _output_paths(output_dir)
    expected = _expected_output_config(args, iterations)
    state = _output_state(paths, expected)
    overwrite_output = bool(args.overwrite)
    if state == 'complete' and args.resume:
        if not args.quiet:
            print(f'[Quijote consensus] complete; skipping {output_dir}', flush=True)
        return 0
    if state != 'missing' and not args.overwrite:
        if state == 'complete':
            raise FileExistsError('Consensus outputs already exist; use --resume '
                                  'to skip them or --overwrite to replace them')
        if state == 'incompatible':
            raise RuntimeError('Existing consensus has incompatible parameters; '
                               'use another --output-dir or --overwrite')
        if not (args.resume and args.repair_incomplete):
            raise RuntimeError('Existing consensus is incomplete; use --resume '
                               '--repair-incomplete or --overwrite')
        overwrite_output = True
        if not args.quiet:
            print(f'[Quijote consensus] repairing incomplete {output_dir}', flush=True)

    started = time.time()
    case_root = (Path(args.input_root).expanduser().resolve() / args.parameter /
                 str(args.realization) / f'groups_{args.snapshot:03d}')
    if not args.quiet:
        print(f'[Quijote consensus] {args.parameter}/{args.realization} from '
              f'{len(iterations)} iterations', flush=True)
    pool, input_config = load_quijote_pool(case_root, args, iterations)
    result = build_consensus_catalog(
        pool, vol_frac=args.vol_frac, v_cut=args.v_cut, keep_all=args.keep_all,
        query_workers=args.query_workers, query_batch_size=args.query_batch_size,
        verbose=not args.quiet)
    elapsed = time.time() - started
    _write_outputs(paths, args, iterations, pool, result, input_config, elapsed,
                   overwrite=overwrite_output)
    if not args.quiet:
        print(f'[Quijote consensus] wrote {len(result.catalog):,} rows to '
              f'{paths["fits"]}', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
