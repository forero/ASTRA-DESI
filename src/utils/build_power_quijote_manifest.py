import argparse
import os
from pathlib import Path


DEFAULT_INPUT_ROOT = '/pscratch/sd/v/vtorresg/quijotes/Halos/FoF'
DEFAULT_ASTRA_ROOT = '/pscratch/sd/v/vtorresg/quijotes/ASTRA/FoF'
DEFAULT_OUTPUT_ROOT = '/pscratch/sd/v/vtorresg/quijotes/PowerSpectrum/FoF'
SNAPSHOT = 3
SAMPLES = ('halo_void', 'halo_sheet', 'halo_filament', 'halo_knot',
           'random_void', 'halo_all')

# Quijote parameter directories whose background expansion differs from the
# fiducial model. The remaining standard suites use Omega_m=0.3175 and w=-1.
BACKGROUND_COSMOLOGY = {'Om_m': (0.3075, -1.0),
                        'Om_p': (0.3275, -1.0),
                        'w_m': (0.3175, -0.95),
                        'w_p': (0.3175, -1.05)}
FIDUCIAL_BACKGROUND_PARAMETERS = {'fiducial', 'fiducial_ZA',
                                  'Mnu_p', 'Mnu_pp', 'Mnu_ppp',
                                  'Ob2_m', 'Ob2_p',
                                  'h_m', 'h_p',
                                  'ns_m', 'ns_p',
                                  's8_m', 's8_p'}


def _cosmology(parameter):
    if parameter in BACKGROUND_COSMOLOGY:
        return BACKGROUND_COSMOLOGY[parameter]
    if parameter in FIDUCIAL_BACKGROUND_PARAMETERS:
        return 0.3175, -1.0
    raise ValueError('No background cosmology is registered for parameter directory {!r}. '
                     'Add its Omega_m and w to build_power_quijote_manifest.py before '
                     'computing its redshift-space spectrum.'.format(parameter))


def _binning_tag(bin_width_kf, kmax):
    kmax_text = '{:.8g}'.format(float(kmax)).replace('-', 'm').replace('.', 'p')
    return 'dk{}kf_kmax{}'.format(int(bin_width_kf), kmax_text)


def _spectrum_paths(output_root, parameter, realization, grid, mas, axis,
                    bin_width_kf, kmax, real_space):
    root = Path(output_root).expanduser().resolve() / parameter / str(realization)
    tag = _binning_tag(bin_width_kf, kmax)
    paths = []
    for sample in SAMPLES:
        if sample == 'random_void' or real_space:
            space = 'real_los{}'.format(axis)
        else:
            space = 'rsd{}'.format(axis)
        name = 'group_{:03d}_pk_{}_{}_{}_N{}_{}.npz'.format(
            SNAPSHOT, sample, space, mas, grid, tag)
        paths.append(root / name)
    return paths


def _discover(args):
    input_root = Path(args.input_root).expanduser().resolve()
    astra_root = Path(args.astra_root).expanduser().resolve()
    selected = set(args.parameter) if args.parameter else None
    pattern = '*/*/groups_{0:03d}/group_tab_{0:03d}.0'.format(SNAPSHOT)
    entries = []
    missing_astra = 0
    complete = 0
    for first_file in input_root.glob(pattern):
        relative = first_file.relative_to(input_root)
        if len(relative.parts) != 4:
            continue
        parameter, realization_text = relative.parts[:2]
        if selected is not None and parameter not in selected:
            continue
        try:
            realization = int(realization_text)
        except ValueError:
            continue
        products = astra_root / parameter / realization_text
        if not ((products / 'group_003_probability.fits.gz').is_file()
                and (products / 'group_003_random_voids.fits.gz').is_file()):
            missing_astra += 1
            continue
        omega_m, w = _cosmology(parameter)
        outputs = _spectrum_paths(
            args.output_root, parameter, realization, args.grid, args.mas,
            args.axis, args.bin_width_kf, args.kmax, args.real_space)
        if args.skip_complete and all(path.is_file() for path in outputs):
            complete += 1
            continue
        entries.append((parameter, realization, omega_m, w))
    entries.sort(key=lambda item: (item[0], item[1]))
    return entries, missing_astra, complete


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-root', default=DEFAULT_INPUT_ROOT)
    parser.add_argument('--astra-root', default=DEFAULT_ASTRA_ROOT)
    parser.add_argument('--output-root', default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument('--output', required=True)
    parser.add_argument('--parameter', action='append', default=None, help='Only include this parameter directory; repeatable')
    parser.add_argument('--skip-complete', action='store_true', help='Omit simulations having all six requested spectra')
    parser.add_argument('--grid', type=int, default=512)
    parser.add_argument('--mas', choices=('NGP', 'CIC', 'TSC', 'PCS'), default='CIC')
    parser.add_argument('--axis', type=int, choices=(0, 1, 2), default=2)
    parser.add_argument('--bin-width-kf', type=int, default=2)
    parser.add_argument('--kmax', type=float, default=0.5)
    parser.add_argument('--real-space', action='store_true')
    args = parser.parse_args()
    if args.grid <= 1 or args.bin_width_kf <= 0 or args.kmax <= 0.0:
        parser.error('--grid, --bin-width-kf, and --kmax must be positive')

    try:
        entries, missing_astra, complete = _discover(args)
    except ValueError as error:
        parser.error(str(error))
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name('.{}.tmp.{}'.format(output.name, os.getpid()))
    try:
        with open(str(temporary), 'w') as stream:
            stream.write('# zero-based rows: parameter realization omega_m w\n')
            for parameter, realization, omega_m, w in entries:
                stream.write('{} {} {:.8g} {:.8g}\n'.format(
                    parameter, realization, omega_m, w))
        os.replace(str(temporary), str(output))
    finally:
        if temporary.exists():
            temporary.unlink()
    print('wrote {} power-spectrum tasks to {} '
          '(missing_astra={}, skipped_complete={})'.format(
              len(entries), output, missing_astra, complete))


if __name__ == '__main__':
    main()