from dataclasses import dataclass
import os
from pathlib import Path

import astropy.units as u
from astropy.cosmology import Planck18
from astropy.table import Table
import numpy as np

from .shapes import (VoidShapes,
                     compute_void_shapes)

from .read_data import TRACER_CODES, normalize_tracer, normalize_zone


REQUIRED_COLUMNS = ('VOID_ID', 'XCART', 'YCART', 'ZCART', 'R_EFF', 'ELLIP')
MEMBERSHIP_REQUIRED_COLUMNS = ('TARGETID', 'RA', 'DEC', 'Z', 'RANDITER',
                               'GROUP_ID', 'VOID_ID', 'MEMBER')
R_EFF_DEFINITION = ('sqrt(5)*(lambda_1*lambda_2*lambda_3)**(1/6)')
ELLIPTICITY_DEFINITION = ('1-((lambda_3+lambda_2)/(lambda_2+lambda_1))**(1/4)')


@dataclass(frozen=True)
class VoidCatalogs:
    all_voids: Table
    clean_voids: Table


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
    table['XCART'] = np.empty(0, dtype=np.float64)
    table['YCART'] = np.empty(0, dtype=np.float64)
    table['ZCART'] = np.empty(0, dtype=np.float64)
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
                       'XYZUNIT': 'Mpc/h',
                       'CENTER': 'mean retained random-member Cartesian position',
                       'SHAPEPTS': 'retained random members'})


def build_void_catalogs(shapes: VoidShapes, border_group_ids, tracer, zone, iteration,
                        h = float(Planck18.h)) -> VoidCatalogs:

    if not isinstance(shapes, VoidShapes):
        raise TypeError('shapes must be a VoidShapes instance.')
    h = float(h)
    if not np.isfinite(h) or h <= 0.0:
        raise ValueError('h must be finite and positive.')
    tracer = normalize_tracer(tracer)
    zone = normalize_zone(zone)
    iteration = int(iteration)

    if len(shapes.group_id) == 0:
        all_voids = _empty_catalog(include_border=True)
        clean_voids = _empty_catalog(include_border=False)
        _set_catalog_metadata(
            all_voids, tracer, zone, iteration, h, 'all_survivors')
        _set_catalog_metadata(
            clean_voids, tracer, zone, iteration, h, 'clean')
        return VoidCatalogs(all_voids, clean_voids)

    group_ids = np.asarray(shapes.group_id, dtype=np.int64)
    centers_mpc_h = np.asarray(shapes.center, dtype=np.float64)
    border_ids = np.asarray(border_group_ids, dtype=np.int64).reshape(-1)
    border = np.isin(group_ids, border_ids)

    all_voids = Table()
    all_voids['VOID_ID'] = _global_void_ids(group_ids, tracer=tracer, zone=zone, iteration=iteration)
    all_voids['XCART'] = centers_mpc_h[:, 0]
    all_voids['YCART'] = centers_mpc_h[:, 1]
    all_voids['ZCART'] = centers_mpc_h[:, 2]
    all_voids['R_EFF'] = np.asarray(shapes.r_eff, dtype=np.float64)
    all_voids['ELLIP'] = np.asarray(shapes.ellipticity, dtype=np.float64)
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


def build_random_membership_catalog(randoms, group_ids, group_ids_before_mask,
                                    r_values, threshold_selected,
                                    selection_pruned_member, border_group_ids,
                                    tracer, zone, iteration) -> Table:
    """Build one row per input random, including unassigned randoms."""

    required_input = ('TARGETID', 'RA', 'DEC', 'Z', 'XCART', 'YCART', 'ZCART')
    names = getattr(getattr(randoms, 'dtype', None), 'names', None)
    if names is None:
        raise TypeError('randoms must be a structured array.')
    missing = [name for name in required_input if name not in names]
    if missing:
        raise ValueError('Random input is missing membership columns: '
                         + ', '.join(missing))

    n_random = len(randoms)

    def aligned(values, name, dtype):
        result = np.asarray(values, dtype=dtype)
        if result.shape != (n_random,):
            raise ValueError(f'{name} must have one value per random point.')
        return result

    group_ids = aligned(group_ids, 'group_ids', np.int64)
    pre_mask_ids = aligned(group_ids_before_mask,
                           'group_ids_before_mask', np.int64)
    r_values = aligned(r_values, 'r_values', np.float64)
    threshold_selected = aligned(threshold_selected,
                                 'threshold_selected', bool)
    selection_pruned = aligned(selection_pruned_member,
                               'selection_pruned_member', bool)
    if not np.all(np.isfinite(r_values)):
        raise ValueError('r_values must be finite.')
    if np.any(group_ids < -1) or np.any(pre_mask_ids < -1):
        raise ValueError('Group IDs must be -1 or non-negative.')

    tracer = normalize_tracer(tracer)
    zone = normalize_zone(zone)
    iteration = int(iteration)
    member = group_ids >= 0
    void_ids = np.full(n_random, -1, dtype=np.int64)
    void_ids[member] = _global_void_ids(group_ids[member], tracer=tracer,
                                        zone=zone, iteration=iteration)

    border_ids = np.asarray(border_group_ids, dtype=np.int64).reshape(-1)
    touched_border = ((pre_mask_ids >= 0)
                      & np.isin(pre_mask_ids, border_ids))
    table = Table()
    table['TARGETID'] = np.asarray(randoms['TARGETID'])
    table['RA'] = np.asarray(randoms['RA'], dtype=np.float64)
    table['DEC'] = np.asarray(randoms['DEC'], dtype=np.float64)
    table['Z'] = np.asarray(randoms['Z'], dtype=np.float64)
    table['XCART'] = np.asarray(randoms['XCART'], dtype=np.float64)
    table['YCART'] = np.asarray(randoms['YCART'], dtype=np.float64)
    table['ZCART'] = np.asarray(randoms['ZCART'], dtype=np.float64)
    table['RANDITER'] = np.full(n_random, iteration, dtype=np.int32)
    table['R_VALUE'] = r_values
    table['THRESHOLD_SELECTED'] = threshold_selected
    table['GROUP_ID_PREMASK'] = pre_mask_ids
    table['GROUP_ID'] = group_ids
    table['VOID_ID'] = void_ids
    table['MEMBER'] = member
    table['PRUNED_BY_MASK'] = selection_pruned
    table['BORDER'] = touched_border
    table.meta.update({'TRACER': tracer,
                       'ZONE': zone,
                       'RANDITER': iteration,
                       'CAT_KIND': 'random_membership',
                       'UNASSIGN': -1,
                       'GRPSTATE': 'GROUP_ID is final post-mask membership',
                       'PREMASK': 'GROUP_ID_PREMASK is membership before selection pruning',
                       'BORDER': 'pre-mask group touched angular/radial selection'})
    table['RA'].unit = u.deg
    table['DEC'].unit = u.deg
    return table


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


def write_membership_catalog(path, table: Table,
                             overwrite: bool = False) -> Path:
    path = Path(path)
    if path.suffix.lower() not in ('.fits', '.fit'):
        raise ValueError('Membership catalog output must use .fits or .fit.')
    if path.exists() and not overwrite:
        raise FileExistsError(
            f'Output already exists: {path}. Use --overwrite to replace it.')
    missing = [name for name in MEMBERSHIP_REQUIRED_COLUMNS
               if name not in table.colnames]
    if missing:
        raise ValueError('Membership catalog is missing required columns: '
                         + ', '.join(missing))

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


__all__ = ['ELLIPTICITY_DEFINITION',
           'MEMBERSHIP_REQUIRED_COLUMNS',
           'REQUIRED_COLUMNS',
           'R_EFF_DEFINITION',
           'VoidCatalogs',
           'VoidShapes',
           'build_random_membership_catalog',
           'build_void_catalogs',
           'compute_void_shapes',
           'write_membership_catalog',
           'write_void_catalog']
