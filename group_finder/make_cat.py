from dataclasses import dataclass
import os
from pathlib import Path

import astropy.units as u
from astropy.cosmology import Planck18
from astropy.table import Table
import numpy as np

from .shapes import (DEFAULT_BOOTSTRAP_SEED,
                     DEFAULT_MAX_ELLIPTICITY_SIGMA,
                     DEFAULT_MAX_RELATIVE_R_EFF_SIGMA,
                     DEFAULT_MIN_RANDOM_MEMBERS,
                     DEFAULT_MIN_VALID_BOOTSTRAP_FRACTION,
                     DEFAULT_N_BOOTSTRAP,
                     VoidShapes,
                     compute_void_shapes)

from .read_data import TRACER_CODES, normalize_tracer, normalize_zone


REQUIRED_COLUMNS = ('VOID_ID', 'RA', 'DEC', 'REDSHIFT', 'R_EFF', 'ELLIP')
R_EFF_DEFINITION = ('sqrt(5)*(lambda_1*lambda_2*lambda_3)**(1/6)')
ELLIPTICITY_DEFINITION = ('1-((lambda_3+lambda_2)/(lambda_2+lambda_1))**(1/4)')


@dataclass(frozen=True)
class VoidCatalogs:
    all_voids: Table
    clean_voids: Table


def comoving_distance_to_redshift(distance_mpc, cosmology=Planck18, n_grid: int = 20_000):

    distance = np.asarray(distance_mpc, dtype=np.float64)
    result = np.full(distance.shape, np.nan, dtype=np.float64)
    finite = np.isfinite(distance)
    if np.any(distance[finite] < 0.0):
        raise ValueError('Comoving distances cannot be negative.')
    if not np.any(finite):
        return result
    maximum = float(np.max(distance[finite]))
    if maximum == 0.0:
        result[finite] = 0.0
        return result

    n_grid = max(int(n_grid), 2)
    z_max = 2.0
    z_grid = np.linspace(0.0, z_max, n_grid)
    chi_grid = cosmology.comoving_distance(z_grid).to_value(u.Mpc)
    while chi_grid[-1] < maximum:
        z_max *= 2.0
        z_grid = np.linspace(0.0, z_max, n_grid)
        chi_grid = cosmology.comoving_distance(z_grid).to_value(u.Mpc)
    result[finite] = np.interp(distance[finite], chi_grid, z_grid)
    return result


def _global_void_ids(group_ids, tracer, zone, iteration):
    group_ids = np.asarray(group_ids, dtype=np.int64)
    if np.any(group_ids < 0) or np.any(group_ids >= 100_000_000):
        raise ValueError('Group IDs must lie in [0, 100000000).')

    tracer = normalize_tracer(tracer)
    zone = normalize_zone(zone)

    iteration = int(iteration)
    if iteration < 0 or iteration >= 1000:
        raise ValueError('iteration must lie in [0, 1000).')

    zone_code = 1 if zone == 'NGC' else 2
    prefix = (int(TRACER_CODES[tracer]) * 1_000_000_000_000
              + zone_code * 100_000_000_000 + iteration * 100_000_000)
    return prefix + group_ids


def _empty_catalog(include_border):
    table = Table()
    table['VOID_ID'] = np.empty(0, dtype=np.int64)
    table['RA'] = np.empty(0, dtype=np.float64)
    table['DEC'] = np.empty(0, dtype=np.float64)
    table['REDSHIFT'] = np.empty(0, dtype=np.float64)
    table['R_EFF'] = np.empty(0, dtype=np.float64)
    table['ELLIP'] = np.empty(0, dtype=np.float64)
    if include_border:
        table['BORDER'] = np.empty(0, dtype=bool)
    return table


def _set_catalog_metadata(table, tracer, zone, iteration, h, kind):
    table.meta.update({'TRACER': normalize_tracer(tracer),
                       'ZONE': normalize_zone(zone),
                       'RANDITER': int(iteration),
                       'HUBBLE_H': float(h),
                       'CAT_KIND': str(kind),
                       'REFF_DEF': R_EFF_DEFINITION,
                       'ELLIPDF': ELLIPTICITY_DEFINITION,
                       'REFFUNIT': 'Mpc/h',
                       'CENTER': 'mean retained random-member Cartesian position',
                       'SHAPEPTS': 'retained random members'})
    table['RA'].unit = u.deg
    table['DEC'].unit = u.deg


def build_void_catalogs(shapes: VoidShapes, border_group_ids, tracer, zone, iteration,
                        h = float(Planck18.h), cosmology=Planck18) -> VoidCatalogs:

    if not isinstance(shapes, VoidShapes):
        raise TypeError('shapes must be a VoidShapes instance.')
    h = float(h)
    if not np.isfinite(h) or h <= 0.0:
        raise ValueError('h must be finite and positive.')
    tracer = normalize_tracer(tracer)
    zone = normalize_zone(zone)
    iteration = int(iteration)

    valid = (np.asarray(shapes.valid_shape, dtype=bool)
             & np.isfinite(shapes.r_eff)
             & (np.asarray(shapes.r_eff) > 0.0)
             & np.isfinite(shapes.ellipticity)
             & np.all(np.isfinite(shapes.center), axis=1))

    if not np.any(valid):
        all_voids = _empty_catalog(include_border=True)
        clean_voids = _empty_catalog(include_border=False)
        _set_catalog_metadata(
            all_voids, tracer, zone, iteration, h, 'all_survivors')
        _set_catalog_metadata(
            clean_voids, tracer, zone, iteration, h, 'clean')
        return VoidCatalogs(all_voids, clean_voids)

    group_ids = np.asarray(shapes.group_id[valid], dtype=np.int64)
    centers_mpc_h = np.asarray(shapes.center[valid], dtype=np.float64)
    radius_mpc_h = np.linalg.norm(centers_mpc_h, axis=1)
    nonzero = radius_mpc_h > 0.0
    if not np.all(nonzero):
        raise ValueError('A valid void center cannot lie at the origin.')

    ra = np.degrees(np.arctan2(centers_mpc_h[:, 1], centers_mpc_h[:, 0],)) % 360.0
    dec = np.degrees(np.arcsin(np.clip(centers_mpc_h[:, 2] / radius_mpc_h,
                                       -1.0, 1.0)))
    redshift = comoving_distance_to_redshift(radius_mpc_h / h, cosmology=cosmology)
    border_ids = np.asarray(border_group_ids, dtype=np.int64).reshape(-1)
    border = np.isin(group_ids, border_ids)

    all_voids = Table()
    all_voids['VOID_ID'] = _global_void_ids(group_ids, tracer=tracer, zone=zone, iteration=iteration)
    all_voids['RA'] = ra
    all_voids['DEC'] = dec
    all_voids['REDSHIFT'] = redshift
    all_voids['R_EFF'] = np.asarray(shapes.r_eff[valid], dtype=np.float64)
    all_voids['ELLIP'] = np.asarray(shapes.ellipticity[valid], dtype=np.float64)
    all_voids['BORDER'] = border
    _set_catalog_metadata(
        all_voids, tracer, zone, iteration, h, 'all_survivors')

    clean_voids = all_voids[~border].copy(copy_data=True)
    clean_voids.remove_column('BORDER')
    _set_catalog_metadata(
        clean_voids, tracer, zone, iteration, h, 'clean')

    if len(np.unique(all_voids['VOID_ID'])) != len(all_voids):
        raise RuntimeError('VOID_ID generation produced duplicate IDs.')
    return VoidCatalogs(all_voids, clean_voids)


def write_void_catalog(path, table: Table, overwrite: bool = False) -> Path:
    path = Path(path)
    if path.suffix.lower() not in ('.fits', '.fit'):
        raise ValueError('Void catalog output must use .fits or .fit.')
    if path.exists() and not overwrite:
        raise FileExistsError(f'Output already exists: {path}. Use --overwrite to replace it.')
    missing = [name for name in REQUIRED_COLUMNS if name not in table.colnames]
    if missing:
        raise ValueError('Void catalog is missing required columns: ' + ', '.join(missing))

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'.{path.name}.{os.getpid()}.tmp.fits')
    try:
        table.write(temporary, format='fits', overwrite=True)
        if path.exists() and not overwrite:
            raise FileExistsError(f'Output already exists: {path}.')
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return path


__all__ = ['DEFAULT_BOOTSTRAP_SEED',
           'DEFAULT_MAX_ELLIPTICITY_SIGMA',
           'DEFAULT_MAX_RELATIVE_R_EFF_SIGMA',
           'DEFAULT_MIN_RANDOM_MEMBERS',
           'DEFAULT_MIN_VALID_BOOTSTRAP_FRACTION',
           'DEFAULT_N_BOOTSTRAP',
           'ELLIPTICITY_DEFINITION',
           'REQUIRED_COLUMNS',
           'R_EFF_DEFINITION',
           'VoidCatalogs',
           'VoidShapes',
           'build_void_catalogs',
           'comoving_distance_to_redshift',
           'compute_void_shapes',
           'write_void_catalog']