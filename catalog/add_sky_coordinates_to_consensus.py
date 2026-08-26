import argparse, json, os
from pathlib import Path
import re, shutil, sys
import tempfile

from astropy.io import fits
import numpy as np
from scipy.integrate import cumulative_trapezoid

DEFAULT_ROOT = Path('/pscratch/sd/v/vtorresg/void_catalog')
SPEED_OF_LIGHT_KM_S = 299_792.458
SKY_COLUMNS = ('RA', 'DEC', 'REDSHIFT')
_OMEGA_DIRECTORY_PATTERN = re.compile(r'(?:^|_)Om(?P<value>[0-9]+p[0-9]+)(?:_|$)')


def _validate_omega_m(value):
    omega_m = float(value)
    if not np.isfinite(omega_m) or not 0.0 < omega_m < 1.0:
        raise ValueError(f'Omega_m must lie within (0, 1), got {value!r}.')
    return omega_m


def _omega_from_header(header):
    for key in ('OMEGA_M', 'OMEGAM', 'OM0'):
        if key in header:
            return _validate_omega_m(header[key])
    return None


def infer_omega_m(catalog_path, hdu=1, override=None):
    if override is not None:
        return _validate_omega_m(override)

    catalog_path = Path(catalog_path)
    with fits.open(catalog_path, memmap=True) as hdul:
        omega_m = _omega_from_header(hdul[hdu].header)
    if omega_m is not None:
        return omega_m

    summary_path = catalog_path.with_suffix('.json')
    if summary_path.is_file():
        summary = json.loads(summary_path.read_text(encoding='utf-8'))
        for source_name in summary.get('inputs', ()):
            source_path = Path(source_name)
            if not source_path.is_file():
                continue
            with fits.open(source_path, memmap=True) as hdul:
                omega_m = _omega_from_header(hdul[hdu].header)
            if omega_m is not None:
                return omega_m

    for component in reversed(catalog_path.parts):
        match = _OMEGA_DIRECTORY_PATTERN.search(component)
        if match:
            value = match.group('value').replace('p', '.')
            return _validate_omega_m(value)

    raise ValueError(f'Cannot determine Omega_m for {catalog_path}. Its '
                     'FITS header has no OMEGA_M, no readable companion JSON '
                     'input supplies it, and the path '
                     'does not encode it. Pass --omega-m explicitly.')


def redshift_from_comoving_distance(distance, omega_m, z_max=6.0, grid_size=131_073):
    distance = np.asarray(distance, dtype=np.float64)
    omega_m = _validate_omega_m(omega_m)
    z_max = float(z_max)
    grid_size = int(grid_size)
    if distance.ndim != 1 or not np.all(np.isfinite(distance)):
        raise ValueError('Comoving distances must be a finite 1D array.')
    if np.any(distance < 0.0):
        raise ValueError('Comoving distances cannot be negative.')
    if not np.isfinite(z_max) or z_max <= 0.0:
        raise ValueError('z_max must be finite and positive.')
    if grid_size < 2:
        raise ValueError('grid_size must be at least 2.')
    if not len(distance):
        return np.empty(0, dtype=np.float64)

    redshift_grid = np.linspace(0.0, z_max, grid_size, dtype=np.float64)
    expansion = np.sqrt(omega_m * (1.0 + redshift_grid)**3 + (1.0 - omega_m))
    distance_grid = ((SPEED_OF_LIGHT_KM_S / 100.0) *
                     cumulative_trapezoid(1.0 / expansion, redshift_grid, initial=0.0))

    maximum = float(np.max(distance))
    if maximum > distance_grid[-1]:
        raise ValueError(f'Largest radius is {maximum:.6g} Mpc/h, beyond z={z_max:g} '
                         f'({distance_grid[-1]:.6g} Mpc/h). Increase --z-max.')
    return np.interp(distance, distance_grid, redshift_grid)


def cartesian_to_sky(x, y, z, omega_m, z_max=6.0, grid_size=131_073):
    x, y, z = np.broadcast_arrays(np.asarray(x, dtype=np.float64),
                                  np.asarray(y, dtype=np.float64),
                                  np.asarray(z, dtype=np.float64))
    if x.ndim != 1 or not (np.all(np.isfinite(x)) and np.all(np.isfinite(y))
                           and np.all(np.isfinite(z))):
        raise ValueError('X, Y, and Z must be finite one-dimensional arrays.')

    transverse = np.hypot(x, y)
    radius = np.hypot(transverse, z)
    if np.any(radius == 0.0):
        raise ValueError('A void center lies at the Cartesian origin, where RA and DEC '
                         'are undefined.')

    ra = np.mod(np.degrees(np.arctan2(y, x)), 360.0)
    dec = np.degrees(np.arctan2(z, transverse))
    redshift = redshift_from_comoving_distance(radius,
                                               omega_m=omega_m,
                                               z_max=z_max,
                                               grid_size=grid_size)
    return ra, dec, redshift


def _output_path(input_path, suffix, in_place):
    input_path = Path(input_path)
    if in_place:
        return input_path
    return input_path.with_name(f'{input_path.stem}{suffix}{input_path.suffix}')


def augment_fits_catalog(input_path,
                         output_path,
                         omega_m,
                         overwrite=False,
                         hdu=1,
                         z_max=6.0,
                         grid_size=131_073):
    input_path = Path(input_path)
    output_path = Path(output_path)
    same_path = input_path.resolve() == output_path.resolve()
    if output_path.exists() and not same_path and not overwrite:
        raise FileExistsError(f'Output exists: {output_path}. Pass '
                              '--overwrite to replace it.')

    omega_m = _validate_omega_m(omega_m)
    with fits.open(input_path, memmap=True) as hdul:
        if hdu >= len(hdul) or not isinstance(hdul[hdu], fits.BinTableHDU):
            raise ValueError(f'HDU {hdu} in {input_path} is not a binary table.')
        table_hdu = hdul[hdu]
        names = tuple(table_hdu.columns.names)
        missing = [name for name in ('X', 'Y', 'Z') if name not in names]
        if missing:
            raise ValueError(f'{input_path} is missing Cartesian columns: ' +
                             ', '.join(missing))
        present_sky = [name for name in SKY_COLUMNS if name in names]
        if present_sky:
            if len(present_sky) == len(SKY_COLUMNS):
                return False
            raise ValueError(f'{input_path} contains only some sky columns: ' +
                             ', '.join(present_sky))

        data = table_hdu.data
        row_count = len(data)
        ra, dec, redshift = cartesian_to_sky(data['X'],
                                             data['Y'],
                                             data['Z'],
                                             omega_m=omega_m,
                                             z_max=z_max,
                                             grid_size=grid_size)
        added = fits.ColDefs([
            fits.Column(name='RA', format='D', unit='deg', array=ra),
            fits.Column(name='DEC', format='D', unit='deg', array=dec),
            fits.Column(name='REDSHIFT', format='D', array=redshift)])
        augmented = fits.BinTableHDU.from_columns(table_hdu.columns + added,
                                                  header=table_hdu.header,
                                                  name=table_hdu.name)
        augmented.header['OMEGA_M'] = (omega_m, 'Matter density for distance inversion')
        augmented.header['SKYFRAME'] = ('ICRS', 'Coordinate frame for RA and DEC')
        augmented.header['ZUNIT'] = ('dimensionless', 'Unit of REDSHIFT')
        augmented.header.add_history(
            'RA, DEC, REDSHIFT derived from Cartesian void centers.')
        output_hdus = fits.HDUList([augmented if index == hdu else original.copy()
                                    for index, original in enumerate(hdul)])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f'.{output_path.name}.',
                                                  suffix='.tmp.fits',
                                                  dir=output_path.parent)
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    try:
        output_hdus.writeto(temporary_path, overwrite=True, checksum=True)
        shutil.copymode(input_path, temporary_path)
        with fits.open(temporary_path, memmap=True) as check:
            written_names = tuple(check[hdu].columns.names)
            if any(name not in written_names for name in SKY_COLUMNS):
                raise RuntimeError(f'Output verification failed for {temporary_path}.')
            if len(check[hdu].data) != row_count:
                raise RuntimeError(f'Row-count verification failed for '
                                   f'{temporary_path}.')
        if output_path.exists() and not same_path and not overwrite:
            raise FileExistsError(f'Output appeared while writing: {output_path}.')
        os.replace(temporary_path, output_path)
    finally:
        output_hdus.close()
        if temporary_path.exists():
            temporary_path.unlink()
    return True


def discover_catalogs(paths):
    requested = [Path(path).expanduser() for path in paths]
    if not requested:
        requested = [DEFAULT_ROOT]
    found = []
    for path in requested:
        if path.is_file():
            found.append(path.resolve())
        elif path.is_dir():
            found.extend(candidate.resolve() for candidate in path.rglob('voids_*.fits')
                         if candidate.parent.name == 'consensus')
        else:
            raise FileNotFoundError(path)
    unique = sorted(set(found), key=str)
    if not unique:
        raise FileNotFoundError('No consensus files matching '
                                'consensus/voids_*.fits were found.')
    return unique


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('paths', nargs='*')
    parser.add_argument('--in-place', action='store_true')
    parser.add_argument('--suffix', default='_sky')
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--omega-m', type=float, default=None)
    parser.add_argument('--z-max', type=float, default=6.0)
    parser.add_argument('--grid-size', type=int, default=131_073)
    parser.add_argument('--dry-run', action='store_true')
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if not args.in_place and not args.suffix:
        raise ValueError('--suffix cannot be empty unless --in-place is used.')
    catalogs = discover_catalogs(args.paths)
    failures, written, skipped = 0, 0, 0

    for input_path in catalogs:
        try:
            omega_m = infer_omega_m(input_path, override=args.omega_m)
            output_path = _output_path(input_path,
                                       suffix=args.suffix,
                                       in_place=args.in_place)
            if args.dry_run:
                print(f'[dry-run] {input_path} -> {output_path} '
                      f'(Omega_m={omega_m:g})')
                continue
            changed = augment_fits_catalog(input_path,
                                           output_path,
                                           omega_m=omega_m,
                                           overwrite=args.overwrite,
                                           z_max=args.z_max,
                                           grid_size=args.grid_size)
            if changed:
                written += 1
                print(f'[written] {output_path} (Omega_m={omega_m:g})')
            else:
                skipped += 1
                print(f'[skip] {input_path} already has RA, DEC, REDSHIFT')
        except Exception as exc:
            failures += 1
            print(f'[error] {input_path}: {exc}', file=sys.stderr)
    print(f'Finished: {written} written, {skipped} skipped, '
          f'{failures} failed, {len(catalogs)} total.')
    return 1 if failures else 0


if __name__ == '__main__':
    raise SystemExit(main())
