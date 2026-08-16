import argparse
import gc, os, shutil, subprocess, sys, time
from pathlib import Path

import fitsio
import MAS_library as MASL
import numpy as np
import Pk_library as PKL
import readfof
import redshift_space_library as RSL


DEFAULT_INPUT_ROOT = '/pscratch/sd/v/vtorresg/quijotes/Halos/FoF'
DEFAULT_ASTRA_ROOT = '/pscratch/sd/v/vtorresg/quijotes/ASTRA/FoF'
DEFAULT_OUTPUT_ROOT = '/pscratch/sd/v/vtorresg/quijotes/PowerSpectrum/FoF'
SNAPSHOT = 3
CLASS_NAMES = ('void', 'sheet', 'filament', 'knot')
PROBABILITY_COLUMNS = ('PVOID', 'PSHEET', 'PFILAMENT', 'PKNOT')
SAMPLE_NAMES = ('halo_void', 'halo_sheet', 'halo_filament', 'halo_knot',
                'random_void', 'halo_all')


def _read_manifest_entry(path, task_index):
    if int(task_index) < 0:
        raise ValueError('--task-index must be non-negative')
    data_index = 0
    with open(str(Path(path).expanduser().resolve()), 'r') as stream:
        for line_number, line in enumerate(stream, 1):
            stripped = line.strip()
            if not stripped or stripped.startswith('#'):
                continue
            fields = stripped.split()
            if len(fields) != 4:
                raise ValueError('Manifest line {} must contain: parameter realization '
                                 'omega_m w'.format(line_number))
            try:
                entry = (fields[0], int(fields[1]), float(fields[2]),
                         float(fields[3]))
            except ValueError as error:
                raise ValueError('Invalid manifest line {}: {}'.format(
                    line_number, stripped)) from error
            if data_index == int(task_index):
                return entry
            data_index += 1
    raise IndexError('--task-index {} is outside a {}-row manifest'.format(
        task_index, data_index))


def _default_threads():
    try:
        allocated = int(os.environ.get('SLURM_CPUS_PER_TASK', '8'))
    except ValueError:
        allocated = 8
    return max(1, min(32, allocated))


def _paths(args):
    relative = Path(args.parameter) / str(int(args.realization))
    catalogue_root = Path(args.input_root).expanduser().resolve() / relative
    astra_root = Path(args.astra_root).expanduser().resolve() / relative
    output_root = Path(args.output_root).expanduser().resolve() / relative
    return {'catalogue_root': catalogue_root,
            'probability': astra_root / 'group_003_probability.fits.gz',
            'random_voids': astra_root / 'group_003_random_voids.fits.gz',
            'output_root': output_root}


def _read_header(path):
    with fitsio.FITS(str(path), mode='r') as hdus:
        if len(hdus) < 2:
            raise ValueError('Missing binary-table extension in {}'.format(path))
        return hdus[1].read_header(), int(hdus[1].get_nrows())


def _validate_products(args, paths):
    for key in ('probability', 'random_voids'):
        if not paths[key].is_file():
            raise FileNotFoundError('Missing ASTRA product: {}'.format(paths[key]))

    probability_header, probability_rows = _read_header(paths['probability'])
    random_header, random_rows = _read_header(paths['random_voids'])
    expected = {'MODE': 'QUIJOTE',
                'PARAM': str(args.parameter),
                'REALIZ': int(args.realization),
                'SNAPNUM': SNAPSHOT}

    for label, header in (('probability', probability_header),
                          ('random void', random_header)):
        for key, wanted in expected.items():
            if header.get(key) != wanted:
                raise ValueError('{} header mismatch for {}: expected {!r}, found {!r}'
                                 .format(label, key, wanted, header.get(key)))
    if probability_rows != int(probability_header.get('SRCROWS', probability_rows)):
        raise ValueError('Probability row count does not match SRCROWS')
    if random_header.get('PRODUCT') != 'RANDVOID':
        raise ValueError('Random catalogue is not an ASTRA RANDVOID product')
    return probability_header, probability_rows, random_rows


def _read_probabilities(path, expected_rows):
    rows = fitsio.read(str(path), ext=1, columns=list(PROBABILITY_COLUMNS))
    if len(rows) != int(expected_rows):
        raise ValueError('Probability row count changed while reading {}'.format(path))
    probabilities = np.empty((len(rows), 4), dtype=np.float32)
    for index, name in enumerate(PROBABILITY_COLUMNS):
        probabilities[:, index] = rows[name]
    del rows
    sums = np.sum(probabilities, axis=1, dtype=np.float32)
    if not np.allclose(sums, 1.0, rtol=0.0, atol=2e-5):
        raise ValueError('ASTRA probabilities do not sum to one')
    maxima = np.max(probabilities, axis=1)
    tie_count = int(np.count_nonzero(
        np.sum(probabilities == maxima[:, None], axis=1) > 1))
    labels = np.argmax(probabilities, axis=1).astype(np.uint8)
    counts = np.bincount(labels, minlength=4)
    del maxima, sums, probabilities
    print('pk-quijote --> hard classes by argmax: {} ties={:,}'.format(
        ', '.join('{}={:,}'.format(name, int(counts[index]))
                  for index, name in enumerate(CLASS_NAMES)), tie_count), flush=True)
    return labels, counts, tie_count


def _compute_hubble(args):
    if args.hubble is not None:
        return float(args.hubble)
    dark_energy = 1.0 - float(args.omega_m)
    expansion2 = (float(args.omega_m) * (1.0 + args.redshift) ** 3
                  + dark_energy * (1.0 + args.redshift) **
                  (3.0 * (1.0 + args.w)))
    if expansion2 <= 0.0:
        raise ValueError('The requested cosmology gives H(z)^2 <= 0')
    return float(100.0 * np.sqrt(expansion2))


def _read_halos(args, catalogue_root, expected_rows, hubble):
    catalogue = readfof.FoF_catalog(
        str(catalogue_root), SNAPSHOT, long_ids=False,
        swap=False, SFR=False, read_IDs=False)
    positions = np.array(catalogue.GroupPos, dtype=np.float32, copy=True)
    velocities = np.array(catalogue.GroupVel, dtype=np.float32, copy=True)
    del catalogue
    gc.collect()
    if positions.shape != (int(expected_rows), 3):
        raise ValueError('GroupPos has shape {}; expected ({}, 3)'.format(
            positions.shape, expected_rows))
    if velocities.shape != positions.shape:
        raise ValueError('GroupVel shape does not match GroupPos')
    positions *= np.float32(1e-3)  # kpc/h -> Mpc/h
    velocities *= np.float32(1.0 + args.redshift)

    if not args.real_space:
        print('pk-quijote --> applying RSD: axis={} z={} H(z)={:.6f} '
              'km/s/(Mpc/h)'.format(args.axis, args.redshift, hubble), flush=True)
        RSL.pos_redshift_space(positions, velocities, float(args.box_size),
                               float(hubble), float(args.redshift), int(args.axis))
    else:
        print('pk-quijote --> using real-space halo positions', flush=True)
    del velocities
    gc.collect()
    return np.ascontiguousarray(positions)


def _read_random_void_positions(path, expected_rows):
    print('pk-quijote --> loading {:,} stacked random-void points'.format(
        int(expected_rows)), flush=True)
    rows = fitsio.read(str(path), ext=1, columns=['X', 'Y', 'Z'])
    if len(rows) != int(expected_rows):
        raise ValueError('Random-void row count changed while reading {}'.format(path))

    names = rows.dtype.names
    offsets = tuple(rows.dtype.fields[name][1] for name in names)
    can_view = (names == ('X', 'Y', 'Z') and rows.dtype.itemsize == 12
                and offsets == (0, 4, 8) and rows.flags.c_contiguous)
    if can_view:
        byteorders = tuple(rows.dtype.fields[name][0].byteorder for name in names)
        if any(order == '>' for order in byteorders):
            rows.byteswap(True)
        positions = rows.view(np.float32).reshape(len(rows), 3)
    else:
        positions = np.empty((len(rows), 3), dtype=np.float32)
        positions[:, 0] = rows['X']
        positions[:, 1] = rows['Y']
        positions[:, 2] = rows['Z']
        del rows
    return np.ascontiguousarray(positions)


def _compute_multipoles(sample, positions, args):
    count = int(len(positions))
    if count < 2:
        raise ValueError('{} contains fewer than two points'.format(sample))
    positions = np.ascontiguousarray(positions, dtype=np.float32)
    if not np.all(np.isfinite(positions)):
        raise ValueError('{} positions contain NaN or infinity'.format(sample))
    box_size32 = np.float32(args.box_size)
    outside = (positions < 0.0) | (positions >= box_size32)
    if np.any(outside):
        minimum = float(np.min(positions))
        maximum = float(np.max(positions))
        tolerance = 2.0 * float(np.spacing(box_size32))
        if minimum < -tolerance or maximum > float(args.box_size) + tolerance:
            raise ValueError('{} positions lie materially outside [0, BoxSize): '
                             'min={} max={}'.format(sample, minimum, maximum))
        coordinate_count = int(np.count_nonzero(outside))
        positions[outside] = np.remainder(positions[outside], box_size32)
        print('pk-quijote --> wrapped {} {} boundary coordinate(s) '
              'periodically into [0, BoxSize)'.format(
                  coordinate_count, sample), flush=True)
        del outside

    start = time.time()
    delta = np.zeros((args.grid, args.grid, args.grid), dtype=np.float32)
    MASL.MA(positions, delta, float(args.box_size), args.mas, verbose=args.verbose)
    mean = float(np.mean(delta, dtype=np.float64))
    if not np.isfinite(mean) or mean <= 0.0:
        raise ValueError('{} produced an invalid density-field mean'.format(sample))
    delta /= np.float32(mean)
    delta -= np.float32(1.0)

    spectrum = PKL.Pk(delta, float(args.box_size), int(args.axis), args.mas,
                      int(args.threads), args.verbose)
    k = np.asarray(spectrum.k3D, dtype=np.float64).copy()
    multipoles = np.asarray(spectrum.Pk, dtype=np.float64)
    if multipoles.ndim != 2 or multipoles.shape[1] < 3:
        raise ValueError('Pylians did not return P0, P2, and P4')
    p0 = multipoles[:, 0].copy()
    p2 = multipoles[:, 1].copy()
    p4 = multipoles[:, 2].copy()
    nmodes = np.asarray(spectrum.Nmodes3D, dtype=np.float64).copy()
    shot_noise = float(args.box_size) ** 3 / float(count)
    rebinned = _rebin_multipoles(
        k, p0, p2, p4, nmodes, float(args.box_size),
        int(args.bin_width_kf), float(args.kmax))
    rebinned['Pk0_raw'] = rebinned.pop('Pk0')
    rebinned['Pk0_shot_subtracted'] = rebinned['Pk0_raw'] - shot_noise
    elapsed = time.time() - start
    print('pk-quijote --> {} objects={:,} bins={} delta_k={:.8f} '
          'shot_noise={:.6e} time={:.1f}s'.format(
              sample, count, len(rebinned['k']), rebinned['delta_k'],
              shot_noise, elapsed), flush=True)
    del spectrum, multipoles, delta
    gc.collect()
    rebinned.update({'shot_noise': shot_noise, 'n_objects': count,
                     'seconds': elapsed})
    return rebinned


def _rebin_multipoles(k, p0, p2, p4, nmodes, box_size,
                      bin_width_kf, kmax):
    arrays = tuple(np.asarray(value, dtype=np.float64)
                   for value in (k, p0, p2, p4, nmodes))
    if any(value.ndim != 1 or len(value) != len(arrays[0])
           for value in arrays):
        raise ValueError('Pylians multipole arrays have inconsistent shapes')

    fundamental_mode = 2.0 * np.pi / float(box_size)
    delta_k = float(bin_width_kf) * fundamental_mode
    valid = (np.isfinite(arrays[0]) & np.isfinite(arrays[1])
             & np.isfinite(arrays[2]) & np.isfinite(arrays[3])
             & np.isfinite(arrays[4]) & (arrays[0] > 0.0)
             & (arrays[0] <= float(kmax)) & (arrays[4] > 0.0))
    if not np.any(valid):
        raise ValueError('No Fourier modes lie at or below kmax={}'.format(kmax))

    selected = tuple(value[valid] for value in arrays)
    bin_index = np.floor(selected[0] / delta_k).astype(np.int64)
    size = int(np.max(bin_index)) + 1
    mode_sum = np.bincount(bin_index, weights=selected[4], minlength=size)
    populated = mode_sum > 0.0
    populated_index = np.nonzero(populated)[0]

    def weighted_mean(values):
        total = np.bincount(
            bin_index, weights=values * selected[4], minlength=size)
        return total[populated] / mode_sum[populated]

    return {'k': weighted_mean(selected[0]),
            'k_center': (populated_index.astype(np.float64) + 0.5) * delta_k,
            'k_bin_low': populated_index.astype(np.float64) * delta_k,
            'k_bin_high': (populated_index.astype(np.float64) + 1.0) * delta_k,
            'Pk0': weighted_mean(selected[1]),
            'Pk2': weighted_mean(selected[2]),
            'Pk4': weighted_mean(selected[3]),
            'Nmodes': mode_sum[populated],
            'fundamental_mode': fundamental_mode,
            'delta_k': delta_k}


def _output_path(paths, sample, args):
    space = _sample_space(sample, args, filename=True)
    name = 'group_003_pk_{}_{}_{}_N{}_{}.npz'.format(
        sample, space, args.mas, int(args.grid), _binning_tag(args))
    return paths['output_root'] / name


def _legacy_output_path(paths, sample, args):
    space = _sample_space(sample, args, filename=True)
    name = 'group_003_pk_{}_{}_{}_N{}.npz'.format(
        sample, space, args.mas, int(args.grid))
    return paths['output_root'] / name


def _binning_tag(args):
    kmax = '{:.8g}'.format(float(args.kmax)).replace('-', 'm').replace('.', 'p')
    return 'dk{}kf_kmax{}'.format(int(args.bin_width_kf), kmax)


def _sample_space(sample, args, filename=False):
    if sample == 'random_void':
        return 'real_los{}'.format(args.axis) if filename else 'real'
    if args.real_space:
        return 'real_los{}'.format(args.axis) if filename else 'real'
    return 'rsd{}'.format(args.axis) if filename else 'redshift'


def _write_result(path, result, sample, args, probability_header,
                  hubble, tie_count):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name('.{}.tmp.{}.npz'.format(path.stem, os.getpid()))
    try:
        np.savez_compressed(str(temporary),
                            k=result['k'], k_center=result['k_center'],
                            k_bin_low=result['k_bin_low'], k_bin_high=result['k_bin_high'],
                            Pk0=result['Pk0_raw'], Pk0_raw=result['Pk0_raw'],
                            Pk0_shot_subtracted=result['Pk0_shot_subtracted'],
                            Pk2=result['Pk2'], Pk4=result['Pk4'], Nmodes=result['Nmodes'],
                            shot_noise=np.float64(result['shot_noise']),
                            n_objects=np.int64(result['n_objects']),
                            seconds=np.float64(result['seconds']),
                            sample=np.asarray(sample), parameter=np.asarray(args.parameter),
                            realization=np.int64(args.realization), snapshot=np.int64(SNAPSHOT),
                            redshift=np.float64(args.redshift), box_size=np.float64(args.box_size),
                            grid=np.int64(args.grid), axis=np.int64(args.axis),
                            mas=np.asarray(args.mas),
                            space=np.asarray(_sample_space(sample, args)),
                            omega_m=np.float64(args.omega_m), w=np.float64(args.w),
                            hubble=np.float64(hubble),
                            fundamental_mode=np.float64(result['fundamental_mode']),
                            delta_k=np.float64(result['delta_k']),
                            bin_width_kf=np.int64(args.bin_width_kf),
                            kmax=np.float64(args.kmax),
                            n_bins=np.int64(len(result['k'])),
                            binning=np.asarray('Nmodes_weighted_Pylians_kf_shells'),
                            probability_iterations=np.int64(probability_header.get('NITER', -1)),
                            class_assignment=np.asarray('argmax_void_sheet_filament_knot'),
                            probability_ties=np.int64(tie_count),
                            shot_noise_products=np.asarray('raw_and_subtracted'),
                            shot_noise_subtracted=np.bool_(False))
        os.replace(str(temporary), str(path))
    finally:
        if temporary.exists():
            temporary.unlink()
    print('pk-quijote --> wrote {}'.format(path), flush=True)


def _existing_result_matches(path, sample, args, probability_header, hubble):
    expected = {'sample': str(sample),
                'parameter': str(args.parameter),
                'realization': int(args.realization),
                'snapshot': SNAPSHOT,
                'redshift': float(args.redshift),
                'box_size': float(args.box_size),
                'grid': int(args.grid),
                'axis': int(args.axis),
                'mas': str(args.mas),
                'space': _sample_space(sample, args),
                'omega_m': float(args.omega_m),
                'w': float(args.w),
                'hubble': float(hubble),
                'fundamental_mode': 2.0 * np.pi / float(args.box_size),
                'delta_k': (2.0 * np.pi / float(args.box_size)
                            * int(args.bin_width_kf)),
                'bin_width_kf': int(args.bin_width_kf),
                'kmax': float(args.kmax),
                'binning': 'Nmodes_weighted_Pylians_kf_shells',
                'probability_iterations': int(probability_header.get('NITER', -1)),
                'shot_noise_products': 'raw_and_subtracted',
                'shot_noise_subtracted': False}

    required_arrays = ('k', 'k_center', 'k_bin_low', 'k_bin_high', 'Pk0', 'Pk0_raw',
                       'Pk0_shot_subtracted', 'Pk2', 'Pk4', 'Nmodes')

    try:
        with np.load(str(path), allow_pickle=False) as stored:
            for key in required_arrays:
                if key not in stored:
                    return False, 'missing {}'.format(key)
            for key, wanted in expected.items():
                if key not in stored:
                    return False, 'missing {}'.format(key)
                found = stored[key].item()
                if isinstance(wanted, float):
                    matches = np.isclose(float(found), wanted, rtol=0.0, atol=1e-10)
                else:
                    matches = found == wanted
                if not matches:
                    return False, '{} expected {!r}, found {!r}'.format(
                        key, wanted, found)
    except Exception as error:
        return False, str(error)
    return True, ''


def _legacy_result_matches(path, sample, args, probability_header, hubble):
    expected = {'sample': str(sample),
                'parameter': str(args.parameter),
                'realization': int(args.realization),
                'snapshot': SNAPSHOT,
                'redshift': float(args.redshift),
                'box_size': float(args.box_size),
                'grid': int(args.grid),
                'axis': int(args.axis),
                'mas': str(args.mas),
                'space': _sample_space(sample, args),
                'omega_m': float(args.omega_m),
                'w': float(args.w),
                'hubble': float(hubble),
                'probability_iterations': int(probability_header.get('NITER', -1)),
                'shot_noise_subtracted': False}

    required_arrays = ('k', 'Pk0', 'Pk2', 'Pk4', 'Nmodes', 'shot_noise',
                       'n_objects')
    try:
        with np.load(str(path), allow_pickle=False) as stored:
            for key in required_arrays:
                if key not in stored:
                    return False, 'missing {}'.format(key)
            for key, wanted in expected.items():
                if key not in stored:
                    return False, 'missing {}'.format(key)
                found = stored[key].item()
                if isinstance(wanted, float):
                    matches = np.isclose(float(found), wanted, rtol=0.0, atol=1e-10)
                else:
                    matches = found == wanted
                if not matches:
                    return False, '{} expected {!r}, found {!r}'.format(
                        key, wanted, found)
    except Exception as error:
        return False, str(error)
    return True, ''


def _rebin_legacy_result(source, output, sample, args, probability_header, hubble):
    print('pk-quijote --> rebinning existing k_f shells from {}'.format(source),
          flush=True)
    with np.load(str(source), allow_pickle=False) as stored:
        result = _rebin_multipoles(stored['k'], stored['Pk0'], stored['Pk2'], stored['Pk4'],
                                   stored['Nmodes'], float(args.box_size), int(args.bin_width_kf),
                                   float(args.kmax))
        result['Pk0_raw'] = result.pop('Pk0')
        result['shot_noise'] = float(stored['shot_noise'].item())
        result['Pk0_shot_subtracted'] = (result['Pk0_raw'] - result['shot_noise'])
        result['n_objects'] = int(stored['n_objects'].item())
        result['seconds'] = float(stored['seconds'].item()) if 'seconds' in stored else 0.0
        tie_count = (int(stored['probability_ties'].item())
                     if 'probability_ties' in stored else -1)
    _write_result(output, result, sample, args, probability_header,
                  hubble, tie_count)


def _should_compute_sample(sample, args, paths, probability_header, hubble):
    output = _output_path(paths, sample, args)
    if output.exists() and not args.force:
        matches, reason = _existing_result_matches(output, sample, args, probability_header, hubble)
        if matches:
            print('pk-quijote --> reusing {}'.format(output), flush=True)
            return False
        raise RuntimeError('Existing power spectrum is incompatible ({}): {}. '
                           'Use --force to replace it.'.format(reason, output))
    if not args.force:
        legacy = _legacy_output_path(paths, sample, args)
        if legacy.is_file():
            matches, reason = _legacy_result_matches(
                legacy, sample, args, probability_header, hubble)
            if matches:
                _rebin_legacy_result(legacy, output, sample, args, probability_header, hubble)
                return False
            print('pk-quijote --> cannot reuse legacy {} ({})'.format(
                legacy, reason), flush=True)
    return True


def _run_sample(sample, positions, args, paths, probability_header,
                hubble, tie_count):
    output = _output_path(paths, sample, args)
    result = _compute_multipoles(sample, positions, args)
    _write_result(output, result, sample, args, probability_header,
                  hubble, tie_count)
    del result
    gc.collect()


def _python_has_matplotlib(executable):
    try:
        completed = subprocess.run([str(executable), '-c', 'import matplotlib'],
                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except (OSError, TypeError):
        return False
    return completed.returncode == 0


def _find_plot_python(explicit=None):
    candidates = []
    if explicit:
        candidates.append(explicit)
    environment_python = os.environ.get('ASTRA_PLOT_PYTHON')
    if environment_python:
        candidates.append(environment_python)
    path_python = shutil.which('python3.12')
    if path_python:
        candidates.append(path_python)

    nersc_root = Path('/global/common/software/nersc/pe/conda-envs')
    if nersc_root.is_dir():
        pattern = '*/python-3.12/nersc-python/bin/python3.12'
        candidates.extend(str(path) for path in
                          sorted(nersc_root.glob(pattern), reverse=True))
    candidates.append(sys.executable)

    seen = set()
    for candidate in candidates:
        candidate = str(Path(candidate).expanduser())
        if candidate in seen:
            continue
        seen.add(candidate)
        if _python_has_matplotlib(candidate):
            return candidate

    if explicit:
        raise RuntimeError(f'The --plot-python executable cannot import matplotlib: {explicit}')
    raise RuntimeError('Plotting needs a Python executable containing matplotlib. Set '
                       'ASTRA_PLOT_PYTHON or pass --plot-python explicitly.')


def _plot_results(args, paths):
    inputs = [_output_path(paths, sample, args) for sample in args.samples]
    missing = [str(path) for path in inputs if not path.is_file()]
    if missing:
        raise FileNotFoundError('Cannot plot because power-spectrum products are missing: {}'.format(missing))

    plot_python = _find_plot_python(args.plot_python)
    print('pk-quijote --> plotting with {}'.format(plot_python), flush=True)
    space = 'real' if args.real_space else 'rsd{}'.format(args.axis)
    output_base = paths['output_root'] / ('group_003_pk_multipoles_{}_{}_N{}_{}'.format(
            space, args.mas, int(args.grid), _binning_tag(args)))
    if args.plot_shot_noise == 'subtracted':
        output_base = output_base.with_name(output_base.name + '_shot_subtracted')
    helper = Path(__file__).resolve().with_name('plot_power_quijote.py')
    command = [str(plot_python), str(helper), '--output-base', str(output_base),
               '--shot-noise', args.plot_shot_noise]
    command.extend(str(path) for path in inputs)
    subprocess.check_call(command)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-root', default=DEFAULT_INPUT_ROOT)
    parser.add_argument('--astra-root', default=DEFAULT_ASTRA_ROOT)
    parser.add_argument('--output-root', default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument('--parameter')
    parser.add_argument('--realization', type=int)
    parser.add_argument('--manifest', help='Four-column bulk manifest: parameter realization omega_m w')
    parser.add_argument('--task-index', type=int, help='Zero-based data-row index in --manifest')
    parser.add_argument('--grid', type=int, default=512)
    parser.add_argument('--mas', choices=('NGP', 'CIC', 'TSC', 'PCS'), default='CIC')
    parser.add_argument('--threads', type=int, default=_default_threads())
    parser.add_argument('--axis', type=int, choices=(0, 1, 2), default=2, help='Line-of-sight axis for RSD and multipoles (default: 2)')
    parser.add_argument('--box-size', type=float, default=1000.0)
    parser.add_argument('--bin-width-kf', type=int, default=2, help='Bin width in integer multiples of k_f (default: 2)')
    parser.add_argument('--kmax', type=float, default=0.5, help='Maximum effective k in h/Mpc (default: 0.5)')
    parser.add_argument('--redshift', type=float, default=0.5)
    parser.add_argument('--omega-m', type=float, default=0.3175, help='Omega_m used for H(z); override for varied cosmologies')
    parser.add_argument('--w', type=float, default=-1.0, help='Constant dark-energy equation of state used for H(z)')
    parser.add_argument('--hubble', type=float, default=None, help='Explicit H(z) in km/s/(Mpc/h), overriding Omega_m and w')
    parser.add_argument('--real-space', action='store_true', help='Do not displace halos into redshift space')
    parser.add_argument('--plot-shot-noise', choices=('raw', 'subtracted'), default='raw', help='Monopole variant shown in the plot (default: raw)')
    parser.add_argument('--subtract-shot-noise', action='store_true', help='Both monopoles are always saved')
    parser.add_argument('--samples', nargs='+', choices=SAMPLE_NAMES,  default=list(SAMPLE_NAMES))
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--no-plot', action='store_true')
    parser.add_argument('--plot-python', default=None, help='Python executable containing matplotlib; defaults to python3.12')
    parser.add_argument('--force', action='store_true')
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    using_manifest = args.manifest is not None or args.task_index is not None
    if using_manifest:
        if args.manifest is None or args.task_index is None:
            parser.error('--manifest and --task-index must be supplied together')
        if args.parameter is not None or args.realization is not None:
            parser.error('do not combine --manifest with --parameter/--realization')
        try:
            (args.parameter, args.realization,
             args.omega_m, args.w) = _read_manifest_entry(
                 args.manifest, args.task_index)
        except (OSError, ValueError, IndexError) as error:
            parser.error(str(error))
        print('pk-quijote --> manifest task={} parameter={} realization={} '
              'Omega_m={} w={}'.format(
                  args.task_index, args.parameter, args.realization,
                  args.omega_m, args.w), flush=True)
    elif args.parameter is None or args.realization is None:
        parser.error('supply --parameter and --realization, or '
                     '--manifest and --task-index')
    if args.realization < 0:
        parser.error('--realization must be non-negative')
    if (args.grid <= 1 or args.threads <= 0 or args.box_size <= 0.0
            or args.bin_width_kf <= 0 or args.kmax <= 0.0):
        parser.error('--grid, --threads, --box-size, --bin-width-kf, and '
                     '--kmax must be positive')
    if not (0.0 < args.omega_m < 1.0):
        parser.error('--omega-m must lie between zero and one')
    if args.subtract_shot_noise:
        args.plot_shot_noise = 'subtracted'
    args.samples = tuple(dict.fromkeys(args.samples))

    paths = _paths(args)
    probability_header, halo_rows, random_rows = _validate_products(args, paths)
    hubble = _compute_hubble(args)
    pending = {}
    for sample in args.samples:
        pending[sample] = _should_compute_sample(
            sample, args, paths, probability_header, hubble)
    if not any(pending.values()):
        if not args.no_plot:
            _plot_results(args, paths)
        return

    labels, _, tie_count = _read_probabilities(paths['probability'], halo_rows)
    halo_pending = any(pending.get(sample, False)
                       for sample in SAMPLE_NAMES if sample != 'random_void')
    if halo_pending:
        halo_positions = _read_halos(args, paths['catalogue_root'], halo_rows, hubble)
        for class_index, class_name in enumerate(CLASS_NAMES):
            sample = 'halo_{}'.format(class_name)
            if pending.get(sample, False):
                selected = halo_positions[labels == class_index]
                _run_sample(sample, selected, args, paths, probability_header,
                            hubble, tie_count)
                del selected
                gc.collect()

        if pending.get('halo_all', False):
            _run_sample('halo_all', halo_positions, args, paths,
                        probability_header, hubble, tie_count)
        del halo_positions
        gc.collect()

    if pending.get('random_void', False):
        random_positions = _read_random_void_positions(paths['random_voids'], random_rows)
        _run_sample('random_void', random_positions, args, paths,
                    probability_header, hubble, tie_count)
        del random_positions
        gc.collect()
    del labels
    if not args.no_plot:
        _plot_results(args, paths)


if __name__ == '__main__':
    main()