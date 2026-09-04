import argparse
from functools import lru_cache
import gc
import json
import os
from pathlib import Path
import struct
import sys
import time
import zlib

os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('NUMEXPR_NUM_THREADS', '1')
os.environ.setdefault('MPLCONFIGDIR', '/tmp/astra-desi-matplotlib')
os.environ.setdefault('XDG_CACHE_HOME', '/tmp/astra-desi-cache')

import numpy as np
from astropy.table import Table

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from group_finder.astra import run_group_finder
from group_finder.make_cat import write_void_catalog
from group_finder.shapes import compute_void_shapes


DEFAULT_INPUT_ROOT = '/pscratch/sd/v/vtorresg/quijotes/Halos/FoF'
DEFAULT_OUTPUT_ROOT = '/pscratch/sd/v/vtorresg/quijotes/void_finder/FoF'
DEFAULT_BOX_SIZE = 1000.0
DEFAULT_SNAPSHOT = 3

# Quijote's Gadget FoF header is followed by 84 bytes per group:
# GroupLen, GroupOffset, GroupMass, GroupPos[3], GroupVel[3],
# GroupLenType[6], and GroupMassType[6].
_FOF_HEADER = struct.Struct('<iiiQI')
_FOF_BYTES_PER_GROUP = 84


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


def _positive_float(value):
    result = float(value)
    if not np.isfinite(result) or result <= 0.0:
        raise argparse.ArgumentTypeError('value must be finite and positive')
    return result


def _boundary_buffer(value):
    if str(value).strip().lower() == 'auto':
        return None
    result = float(value)
    if not np.isfinite(result) or result < 0.0:
        raise argparse.ArgumentTypeError('boundary buffer must be "auto" or non-negative')
    return result


def _parse_iteration_tokens(values):
    iterations = []
    for value in values:
        for token in str(value).split(','):
            token = token.strip()
            if not token:
                continue
            if '-' in token:
                fields = token.split('-', 1)
                try:
                    start, stop = map(int, fields)
                except ValueError:
                    raise ValueError(f'Invalid random-iteration range: {token!r}')
                if start < 0 or stop < start:
                    raise ValueError(f'Invalid random-iteration range: {token!r}')
                iterations.extend(range(start, stop + 1))
            else:
                try:
                    iteration = int(token)
                except ValueError:
                    raise ValueError(f'Invalid random iteration: {token!r}')
                if iteration < 0:
                    raise ValueError('random iterations must be non-negative')
                iterations.append(iteration)
    if not iterations:
        raise ValueError('at least one random iteration is required')
    return tuple(dict.fromkeys(iterations))


def _read_exact_array(stream, dtype, count, path, label):
    values = np.fromfile(stream, dtype=dtype, count=int(count))
    if len(values) != int(count):
        raise ValueError(f'Incomplete {label} array in {path}')
    return values


def inspect_fof_catalog(group_dir, snapshot):
    """Return and validate the common header of a split Quijote FoF catalogue."""
    group_dir = Path(group_dir)
    first = group_dir / f'group_tab_{int(snapshot):03d}.0'
    if not first.is_file():
        raise FileNotFoundError(f'Quijote FoF catalogue not found: {first}')
    with first.open('rb') as stream:
        raw = stream.read(_FOF_HEADER.size)
    if len(raw) != _FOF_HEADER.size:
        raise ValueError(f'Incomplete FoF header in {first}')
    n_groups, total_groups, n_ids, total_ids, n_files = _FOF_HEADER.unpack(raw)
    if n_groups < 0 or total_groups < 1 or n_ids < 0 or total_ids < 0 or n_files < 1:
        raise ValueError(f'Invalid FoF header in {first}')
    files = tuple(group_dir / f'group_tab_{int(snapshot):03d}.{index}'
                  for index in range(n_files))
    missing = [str(path) for path in files if not path.is_file()]
    if missing:
        raise FileNotFoundError('Missing FoF file parts: ' + ', '.join(missing))
    return {'total_groups': int(total_groups),
            'total_ids': int(total_ids),
            'n_files': int(n_files),
            'files': files}


@lru_cache(maxsize=1)
def read_fof_halos(group_dir, snapshot):
    """Read position, total mass, and particle count without requiring Pylians."""
    info = inspect_fof_catalog(group_dir, snapshot)
    total = info['total_groups']
    positions = np.empty((total, 3), dtype=np.float64)
    masses = np.empty(total, dtype=np.float64)
    lengths = np.empty(total, dtype=np.uint32)
    cursor = 0

    for part_index, path in enumerate(info['files']):
        with path.open('rb') as stream:
            raw = stream.read(_FOF_HEADER.size)
            if len(raw) != _FOF_HEADER.size:
                raise ValueError(f'Incomplete FoF header in {path}')
            n_groups, part_total, _, _, n_files = _FOF_HEADER.unpack(raw)
            if part_total != total or n_files != info['n_files'] or n_groups < 0:
                raise ValueError(f'Inconsistent FoF header in {path}')
            expected_size = _FOF_HEADER.size + _FOF_BYTES_PER_GROUP * n_groups
            if path.stat().st_size != expected_size:
                raise ValueError(
                    f'Unexpected FoF layout in {path}: found {path.stat().st_size} '
                    f'bytes, expected {expected_size}')

            stop = cursor + n_groups
            if stop > total:
                raise ValueError(f'FoF row count exceeds total in part {part_index}')
            lengths[cursor:stop] = _read_exact_array(
                stream, '<u4', n_groups, path, 'GroupLen')
            stream.seek(4 * n_groups, os.SEEK_CUR)  # GroupOffset
            masses[cursor:stop] = _read_exact_array(
                stream, '<f4', n_groups, path, 'GroupMass')
            part_positions = _read_exact_array(
                stream, '<f4', 3 * n_groups, path, 'GroupPos').reshape(n_groups, 3)
            positions[cursor:stop] = part_positions
            cursor = stop

    if cursor != total:
        raise ValueError(f'FoF parts contain {cursor} rows but header declares {total}')

    positions /= 1_000.0       # kpc/h -> Mpc/h
    masses *= 1.0e10           # 1e10 Msun/h -> Msun/h
    if not np.all(np.isfinite(positions)) or not np.all(np.isfinite(masses)):
        raise ValueError('FoF catalogue contains non-finite position or mass values')
    info = {key: value for key, value in info.items() if key != 'files'}
    return positions, masses, lengths, info


def _random_seed(base_seed, parameter, realization, snapshot, random_iteration):
    parameter_code = zlib.crc32(parameter.encode('utf-8'))
    return np.random.SeedSequence([int(base_seed), int(parameter_code),
                                   int(realization), int(snapshot),
                                   int(random_iteration)])


def generate_box_randoms(n_random, box_origin, box_size, seed_sequence):
    """Draw a reproducible uniform random catalogue in the simulation cube."""
    rng = np.random.default_rng(seed_sequence)
    randoms = rng.random((int(n_random), 3), dtype=np.float64)
    randoms *= float(box_size)
    randoms += np.asarray(box_origin, dtype=np.float64)
    return randoms


def _border_group_ids(positions, group_ids, box_origin, box_size, buffer):
    if buffer <= 0.0:
        return np.empty(0, dtype=np.int64)
    assigned = group_ids >= 0
    if not np.any(assigned):
        return np.empty(0, dtype=np.int64)
    lower = np.asarray(box_origin, dtype=np.float64)
    upper = lower + float(box_size)
    xyz = positions[assigned]
    near_face = np.any((xyz - lower <= buffer) | (upper - xyz <= buffer), axis=1)
    return np.unique(group_ids[assigned][near_face]).astype(np.int64, copy=False)


def _void_table(shapes, border_group_ids, metadata, clean=False):
    border = np.isin(shapes.group_id, np.asarray(border_group_ids, dtype=np.int64))
    keep = ~border if clean else np.ones(len(border), dtype=bool)
    table = Table()
    table['VOID_ID'] = np.asarray(shapes.group_id[keep], dtype=np.int64)
    table['XCART'] = np.asarray(shapes.center[keep, 0], dtype=np.float64)
    table['YCART'] = np.asarray(shapes.center[keep, 1], dtype=np.float64)
    table['ZCART'] = np.asarray(shapes.center[keep, 2], dtype=np.float64)
    table['R_EFF'] = np.asarray(shapes.r_eff[keep], dtype=np.float64)
    table['ELLIP'] = np.asarray(shapes.ellipticity[keep], dtype=np.float64)
    table['N_MEMBERS'] = np.asarray(shapes.n_members[keep], dtype=np.int64)
    table['N_DATA'] = np.asarray(shapes.n_data[keep], dtype=np.int64)
    table['N_RANDOM'] = np.asarray(shapes.n_random[keep], dtype=np.int64)
    for axis in range(3):
        table[f'EIGVAL_{axis + 1}'] = np.asarray(
            shapes.lambda_values[keep, axis], dtype=np.float64)
    for axis in range(3):
        for component, label in enumerate(('X', 'Y', 'Z')):
            table[f'EIGVEC_{axis + 1}_{label}'] = np.asarray(
                shapes.eigenvectors[keep, axis, component], dtype=np.float64)
    if not clean:
        table['BORDER'] = border
    table.meta.update(metadata)
    table.meta['CAT_KIND'] = 'clean' if clean else 'all'
    table.meta['CENTER'] = 'mean retained random-member position'
    table.meta['SHAPEPTS'] = 'retained random members'
    return table


def _membership_table(randoms, grouping, n_data, border_group_ids, metadata):
    group_ids = np.asarray(grouping.group_ids[n_data:], dtype=np.int64)
    member = group_ids >= 0
    border = member & np.isin(group_ids, np.asarray(border_group_ids, dtype=np.int64))
    table = Table()
    table['RANDINDEX'] = np.arange(len(randoms), dtype=np.int64)
    table['XCART'] = randoms[:, 0]
    table['YCART'] = randoms[:, 1]
    table['ZCART'] = randoms[:, 2]
    table['R_VALUE'] = np.asarray(grouping.r_values[n_data:], dtype=np.float64)
    table['THRESHOLD_SELECTED'] = np.asarray(
        grouping.threshold_selected[n_data:], dtype=bool)
    table['GROUP_ID'] = group_ids
    table['VOID_ID'] = np.where(member, group_ids, -1)
    table['MEMBER'] = member
    table['BORDER'] = border
    table.meta.update(metadata)
    table.meta['CAT_KIND'] = 'random_membership'
    return table


def _write_table_atomic(path, table, overwrite=False):
    path = Path(path)
    if path.exists() and not overwrite:
        raise FileExistsError(f'Output already exists: {path}. Use --overwrite.')
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'.{path.name}.{os.getpid()}.tmp.fits')
    try:
        table.write(temporary, format='fits', overwrite=True)
        if path.exists() and not overwrite:
            raise FileExistsError(f'Output already exists: {path}')
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _write_json_atomic(path, payload, overwrite=False):
    path = Path(path)
    if path.exists() and not overwrite:
        raise FileExistsError(f'Output already exists: {path}. Use --overwrite.')
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    try:
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n',
                             encoding='utf-8')
        if path.exists() and not overwrite:
            raise FileExistsError(f'Output already exists: {path}')
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _output_state(paths, args):
    required = [paths['all'], paths['clean'], paths['summary']]
    if args.include_membership:
        required.append(paths['membership'])
    existing = [path for path in required if path.exists()]
    if not existing:
        return 'missing'
    try:
        summary = json.loads(paths['summary'].read_text(encoding='utf-8'))
    except (OSError, ValueError, TypeError):
        return 'incomplete'
    expected = {'parameter': args.parameter,
                'realization': args.realization,
                'snapshot': args.snapshot,
                'random_seed': args.random_seed,
                'random_iteration': args.random_iteration,
                'box_origin_mpc_h': list(map(float, args.box_origin)),
                'box_size_mpc_h': float(args.box_size),
                'min_halo_mass_msun_h': args.min_halo_mass,
                'max_halo_mass_msun_h': args.max_halo_mass,
                'min_halo_particles': args.min_halo_particles,
                'random_factor': args.random_factor,
                'r_threshold': args.r_threshold,
                'min_members': args.min_members,
                'boundary_buffer_requested': (
                    'auto' if args.boundary_buffer is None
                    else float(args.boundary_buffer))}
    if not all(summary.get(key) == value for key, value in expected.items()):
        return 'incompatible'
    if not all(path.is_file() and path.stat().st_size > 0 for path in required):
        return 'incomplete'
    return 'complete'


def build_parser():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('parameter', type=_parameter_name,
                        help='Quijote parameter directory, e.g. fiducial or Om_m')
    parser.add_argument('realization', type=_nonnegative_int)
    parser.add_argument('--input-root', default=DEFAULT_INPUT_ROOT)
    parser.add_argument('--output-root', default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument('--snapshot', type=_nonnegative_int, default=DEFAULT_SNAPSHOT)
    parser.add_argument('--box-origin', nargs=3, type=float, default=(0.0, 0.0, 0.0),
                        metavar=('X0', 'Y0', 'Z0'))
    parser.add_argument('--box-size', type=_positive_float, default=DEFAULT_BOX_SIZE,
                        help='simulation-box side in Mpc/h (default: 1000)')
    parser.add_argument('--random-factor', type=_positive_float, default=1.0,
                        help='N_random / N_selected_halos (default: 1)')
    parser.add_argument('--random-seed', type=_nonnegative_int, default=0)
    parser.add_argument('--random-iteration', nargs='+', default=['0'],
                        metavar='N|START-STOP',
                        help=('one or more iterations, including ranges such as 0-99; '
                              'each is mixed into the seed and output directory'))
    parser.add_argument('--min-halo-mass', type=float, default=None,
                        help='optional minimum GroupMass in Msun/h')
    parser.add_argument('--max-halo-mass', type=float, default=None,
                        help='optional maximum GroupMass in Msun/h')
    parser.add_argument('--min-halo-particles', type=_nonnegative_int, default=0)
    parser.add_argument('--r-threshold', type=float, default=-0.25)
    parser.add_argument('--min-members', type=int, default=4)
    parser.add_argument(
        '--boundary-buffer', type=_boundary_buffer, default=None, metavar='MPC_H|auto',
        help=('distance from a box face used to flag BORDER voids; "auto" uses '
              'one mean random separation (default: auto), 0 disables flagging'))
    parser.add_argument('--include-membership', action='store_true')
    parser.add_argument('--resume', action='store_true',
                        help='skip a complete existing iteration')
    parser.add_argument('--repair-incomplete', action='store_true',
                        help=('with --resume, replace only an incomplete or '
                              'incompatible existing iteration'))
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    return parser


def _validate_args(args):
    origin = np.asarray(args.box_origin, dtype=np.float64)
    if origin.shape != (3,) or not np.all(np.isfinite(origin)):
        raise ValueError('--box-origin must contain three finite values')
    if not np.isfinite(args.r_threshold) or not -1.0 <= args.r_threshold <= 1.0:
        raise ValueError('--r-threshold must lie in [-1, 1]')
    if args.min_members < 1:
        raise ValueError('--min-members must be positive')
    for name in ('min_halo_mass', 'max_halo_mass'):
        value = getattr(args, name)
        if value is not None and (not np.isfinite(value) or value < 0.0):
            raise ValueError(f'--{name.replace("_", "-")} must be finite and non-negative')
    if (args.min_halo_mass is not None and args.max_halo_mass is not None
            and args.min_halo_mass > args.max_halo_mass):
        raise ValueError('--min-halo-mass cannot exceed --max-halo-mass')
    if args.repair_incomplete and not args.resume:
        raise ValueError('--repair-incomplete requires --resume')


def _run_one(args):
    started = time.time()

    catalogue_root = (Path(args.input_root).expanduser().resolve() /
                      args.parameter / str(args.realization))
    group_dir = catalogue_root / f'groups_{args.snapshot:03d}'
    output_dir = (Path(args.output_root).expanduser().resolve() /
                  args.parameter / str(args.realization) /
                  f'groups_{args.snapshot:03d}' /
                  f'random_{args.random_iteration:03d}')
    paths = {'all': output_dir / 'voids_all.fits',
             'clean': output_dir / 'voids_clean.fits',
             'membership': output_dir / 'random_membership.fits',
             'summary': output_dir / 'summary.json'}
    requested_paths = [paths['all'], paths['clean'], paths['summary']]
    if args.include_membership:
        requested_paths.append(paths['membership'])
    existing = [str(path) for path in requested_paths if path.exists()]
    overwrite_output = bool(args.overwrite)
    if args.resume and args.overwrite:
        raise ValueError('--resume and --overwrite cannot be used together')
    output_state = _output_state(paths, args) if args.resume else None
    if output_state == 'complete':
        print(f'[Quijote {args.parameter}/{args.realization}] random iteration '
              f'{args.random_iteration} already complete; skipping {output_dir}',
              flush=True)
        return 0
    if existing and not args.overwrite:
        if args.resume:
            if output_state == 'incompatible':
                raise RuntimeError(
                    'Existing iteration was produced with incompatible parameters; '
                    'use a different --output-root or explicitly use --overwrite: ' +
                    ', '.join(existing))
            if not args.repair_incomplete:
                raise RuntimeError(
                    'Existing iteration is incomplete; inspect it, '
                    'use --resume --repair-incomplete, or use --overwrite: ' +
                    ', '.join(existing))
            overwrite_output = True
            print(f'[Quijote {args.parameter}/{args.realization}] random iteration '
                  f'{args.random_iteration} is incomplete; repairing',
                  flush=True)
        else:
            raise FileExistsError(
                'Outputs already exist; use --resume to skip a complete iteration '
                'or --overwrite to replace it: ' + ', '.join(existing))

    print(f'[Quijote {args.parameter}/{args.realization}] reading {group_dir}',
          flush=True)
    positions, masses, lengths, source_info = read_fof_halos(group_dir, args.snapshot)
    lower = np.asarray(args.box_origin, dtype=np.float64)
    upper = lower + args.box_size
    tolerance = max(1.0, float(args.box_size)) * 1.0e-8
    if np.any(positions < lower - tolerance) or np.any(positions >= upper + tolerance):
        minimum = np.min(positions, axis=0).tolist()
        maximum = np.max(positions, axis=0).tolist()
        raise ValueError(f'GroupPos range {minimum}..{maximum} is outside the box')

    selection = lengths >= args.min_halo_particles
    if args.min_halo_mass is not None:
        selection &= masses >= args.min_halo_mass
    if args.max_halo_mass is not None:
        selection &= masses <= args.max_halo_mass
    positions = np.ascontiguousarray(positions[selection], dtype=np.float64)
    selected_mass_min = float(np.min(masses[selection])) if len(positions) else None
    selected_mass_max = float(np.max(masses[selection])) if len(positions) else None
    del masses, lengths, selection
    if len(positions) < 4:
        raise ValueError(f'Halo selection leaves only {len(positions)} objects')

    n_random = max(4, int(round(args.random_factor * len(positions))))
    seed = _random_seed(args.random_seed, args.parameter, args.realization,
                        args.snapshot, args.random_iteration)
    randoms = generate_box_randoms(n_random, args.box_origin, args.box_size, seed)
    buffer = (args.box_size / np.cbrt(n_random)
              if args.boundary_buffer is None else args.boundary_buffer)
    print(f'[Quijote {args.parameter}/{args.realization}] '
          f'{len(positions):,} halos + {n_random:,} uniform box randoms; '
          f'boundary buffer={buffer:.3f} Mpc/h', flush=True)

    if args.dry_run:
        print(f'Output: {output_dir}')
        print(f'Selected halo mass range: {selected_mass_min:.6g}..'
              f'{selected_mass_max:.6g} Msun/h')
        return 0

    result = run_group_finder(object_positions=positions,
                              random_positions=randoms,
                              r_threshold=args.r_threshold,
                              min_members=args.min_members)
    n_data = int(result.graph.n_data)
    n_edges = int(len(result.graph.edges))
    graph_positions = result.graph.positions
    is_data = result.graph.is_data
    grouping = result.grouping
    border_ids = _border_group_ids(graph_positions, grouping.group_ids,
                                   args.box_origin, args.box_size, buffer)
    shapes = compute_void_shapes(graph_positions, is_data, grouping.group_ids,
                                 coordinate_scale=1.0)
    del result, positions, graph_positions, is_data
    gc.collect()

    metadata = {'MODE': 'QUIJOTE',
                'PARAM': args.parameter,
                'REALIZ': args.realization,
                'SNAPNUM': args.snapshot,
                'RANDITER': args.random_iteration,
                'BASESEED': args.random_seed,
                'N_DATA': n_data,
                'N_RANDOM': n_random,
                'RANFACT': args.random_factor,
                'RTHRESH': args.r_threshold,
                'MINMEM': args.min_members,
                'BOXSIZE': args.box_size,
                'BOXLOX': lower[0], 'BOXLOY': lower[1], 'BOXLOZ': lower[2],
                'BORDERBF': buffer,
                'XYZUNIT': 'Mpc/h', 'REFFUNIT': 'Mpc/h',
                'MASSUNIT': 'Msun/h'}
    all_voids = _void_table(shapes, border_ids, metadata, clean=False)
    clean_voids = _void_table(shapes, border_ids, metadata, clean=True)
    write_void_catalog(paths['all'], all_voids, overwrite=overwrite_output)
    write_void_catalog(paths['clean'], clean_voids, overwrite=overwrite_output)
    if args.include_membership:
        membership = _membership_table(randoms, grouping, n_data, border_ids, metadata)
        _write_table_atomic(paths['membership'], membership,
                            overwrite=overwrite_output)
        del membership

    n_defined = int(np.count_nonzero(np.isfinite(shapes.r_eff)))
    summary = {'parameter': args.parameter,
               'realization': args.realization,
               'snapshot': args.snapshot,
               'input': str(group_dir),
               'output': str(output_dir),
               'box_origin_mpc_h': list(map(float, args.box_origin)),
               'box_size_mpc_h': float(args.box_size),
               'source_total_groups': source_info['total_groups'],
               'source_total_ids': source_info['total_ids'],
               'source_n_files': source_info['n_files'],
               'n_data': n_data,
               'selected_mass_min_msun_h': selected_mass_min,
               'selected_mass_max_msun_h': selected_mass_max,
               'min_halo_mass_msun_h': args.min_halo_mass,
               'max_halo_mass_msun_h': args.max_halo_mass,
               'min_halo_particles': args.min_halo_particles,
               'n_random': n_random,
               'random_factor': args.random_factor,
               'random_seed': args.random_seed,
               'random_iteration': args.random_iteration,
               'r_threshold': args.r_threshold,
               'min_members': args.min_members,
               'boundary_buffer_requested': (
                   'auto' if args.boundary_buffer is None
                   else float(args.boundary_buffer)),
               'boundary_buffer_mpc_h': float(buffer),
               'n_edges': n_edges,
               'n_threshold_selected': int(np.count_nonzero(
                   grouping.threshold_selected)),
               'n_groups': int(len(grouping.group_sizes)),
               'n_border_groups': int(len(border_ids)),
               'n_defined_shapes': n_defined,
               'n_catalog_all': int(len(all_voids)),
               'n_catalog_clean': int(len(clean_voids)),
               'all_catalog': str(paths['all']),
               'clean_catalog': str(paths['clean']),
               'membership_catalog': (str(paths['membership'])
                                      if args.include_membership else None),
               'elapsed_seconds': float(time.time() - started)}
    _write_json_atomic(paths['summary'], summary, overwrite=overwrite_output)
    print(f'[Quijote {args.parameter}/{args.realization}] wrote '
          f'{len(all_voids):,} voids ({len(clean_voids):,} clean) to {output_dir}',
          flush=True)
    return 0


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        iterations = _parse_iteration_tokens(args.random_iteration)
    except ValueError as exc:
        parser.error(str(exc))
    _validate_args(args)
    for iteration in iterations:
        args.random_iteration = int(iteration)
        status = _run_one(args)
        if status:
            return status
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
