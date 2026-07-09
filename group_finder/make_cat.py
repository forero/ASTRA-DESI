import os
import numpy as np
from astropy.io import fits
from astropy.table import Table
import astropy.units as u

try:
    from .watershed import BOUNDARY_ID, compute_semi_axes
except ImportError:
    from watershed import BOUNDARY_ID, compute_semi_axes


GROUP_COLUMNS = ('VOID_ID', 'N_DATA_IN_GROUP', 'N_RAND_IN_GROUP',
                 'RA', 'DEC', 'REDSHIFT', 'X', 'Y', 'Z',
                 'R_EFF',
                 'LAMBDA_1', 'LAMBDA_2', 'LAMBDA_3',
                 'SEMI_AXIS_A', 'SEMI_AXIS_B', 'SEMI_AXIS_C',
                 'CHI_MIN_GROUP', 'CHI_MAX_GROUP', 'CHI_CENTER',
                 'D_RADIAL_EDGE', 'CHI_MIN_SAMPLE', 'CHI_MAX_SAMPLE',
                 'X_MIN_GROUP', 'X_MAX_GROUP', 'Y_MIN_GROUP', 'Y_MAX_GROUP',
                 'Z_MIN_GROUP', 'Z_MAX_GROUP', 'D_CART_EDGE',
                 'X_MIN_SAMPLE', 'X_MAX_SAMPLE', 'Y_MIN_SAMPLE', 'Y_MAX_SAMPLE',
                 'Z_MIN_SAMPLE', 'Z_MAX_SAMPLE',
                 'GEOM_BAD',
                 'EDGE', 'FOOTPRINT_EDGE',
                 'TOUCHES_RADIAL_EDGE', 'CENTER_NEAR_RADIAL_EDGE',
                 'TOUCHES_RA_EDGE', 'TOUCHES_DEC_EDGE',
                 'TOUCHES_HEALPIX_EDGE', 'CENTER_NEAR_HEALPIX_EDGE',
                 'TOUCHES_CART_EDGE', 'CENTER_NEAR_CART_EDGE')

GROUP_DTYPES = (np.int32, np.int32, np.int32,
                np.float64, np.float64, np.float64,
                np.float64, np.float64, np.float64,
                np.float64,
                np.float64, np.float64, np.float64,
                np.float64, np.float64,
                np.float64,
                np.float64, np.float64, np.float64,
                np.float64, np.float64, np.float64,
                np.float64, np.float64, np.float64, np.float64,
                np.float64, np.float64, np.float64,
                np.float64, np.float64, np.float64, np.float64,
                np.float64, np.float64,
                np.bool_,
                np.bool_,
                np.bool_, np.bool_, np.bool_,
                np.bool_, np.bool_,
                np.bool_, np.bool_,
                np.bool_, np.bool_)

ELLIPTICITY_DEFINITION = '1-((C^2+B^2)/(B^2+A^2))**0.25'
J1J3_DEFINITION = 'J_1/J_3=(C^2+B^2)/(B^2+A^2)'
REFF_DEFINITION = 'sqrt(5)*(LAMBDA_1*LAMBDA_2*LAMBDA_3)**(1/6)'
AXIS_VECTOR_COLUMNS = ('X1', 'X2', 'X3',
                       'Y1', 'Y2', 'Y3',
                       'Z1', 'Z2', 'Z3')


def _empty_group_table():
    '''
    Create an empty group table with the correct columns and data types.
    This is used when no valid groups are found, to ensure consistent output format.

    Returns:
         - An empty Astropy Table with columns defined by GROUP_COLUMNS and GROUP_DTYPES.
    '''
    return Table(names=list(GROUP_COLUMNS), dtype=list(GROUP_DTYPES))


def comoving_distance_to_redshift(dist_mpc, cosmo, z_max_init=2.0, n_grid=20000):
    '''
    Convert comoving distance in Mpc to redshift using the provided cosmology.
    This implementation inverts the distance-redshift relation with a dense
    interpolation grid. Calling z_at_value once per void is prohibitively slow
    for watershed catalogs with O(1e5) groups.

    Parameters:
        - dist_mpc: Array of comoving distances in Mpc to convert to redshift.
        - cosmo: Astropy cosmology instance to use for distance-redshift conversion.
        - z_max_init: Initial maximum redshift for the interpolation grid.
        - n_grid: Number of grid samples used for interpolation.
    Returns:
        - Array of redshift values corresponding to the input comoving distances.
    Raises:
        - ValueError: If any finite comoving distance is negative.
    '''
    dist_mpc = np.asarray(dist_mpc, dtype=np.float64)
    if dist_mpc.size == 0:
        return np.array([], dtype=np.float64)

    negative_mask = np.isfinite(dist_mpc) & (dist_mpc < 0)
    if np.any(negative_mask):
        raise ValueError('Comoving distance must be non-negative')

    z_vals = np.full(dist_mpc.shape, np.nan, dtype=np.float64)
    finite = np.isfinite(dist_mpc)
    if not np.any(finite):
        return z_vals

    max_dist = float(np.nanmax(dist_mpc[finite]))
    if max_dist == 0.0:
        z_vals[finite] = 0.0
        return z_vals

    z_max = max(float(z_max_init), 1.0e-4)
    n_grid = max(int(n_grid), 2)
    z_grid = np.linspace(0.0, z_max, n_grid, dtype=np.float64)
    chi_grid = cosmo.comoving_distance(z_grid).to_value(u.Mpc)
    while chi_grid[-1] < max_dist:
        z_max *= 2.0
        z_grid = np.linspace(0.0, z_max, n_grid, dtype=np.float64)
        chi_grid = cosmo.comoving_distance(z_grid).to_value(u.Mpc)

    z_vals[finite] = np.interp(dist_mpc[finite], chi_grid, z_grid)

    return z_vals


def _finite_xyz_mask(table):
    '''
    Return finite Cartesian-coordinate mask for an Astropy table.
    '''
    x = np.asarray(table['X_CART'], dtype=np.float64)
    y = np.asarray(table['Y_CART'], dtype=np.float64)
    z = np.asarray(table['Z_CART'], dtype=np.float64)
    return np.isfinite(x) & np.isfinite(y) & np.isfinite(z)


def _sample_geometry_bounds(data_table, rand_table):
    '''
    Estimate survey-geometry bounds from randoms, falling back to data if needed.

    Random catalogues are preferred because they trace the angular/radial mask more
    uniformly than the observed galaxies.
    '''
    source_table = rand_table
    source_name = 'randoms'
    finite = _finite_xyz_mask(source_table)
    if not np.any(finite):
        source_table = data_table
        source_name = 'data'
        finite = _finite_xyz_mask(source_table)

    if not np.any(finite):
        return {'source': source_name,
                'chi_min': np.nan, 'chi_max': np.nan,
                'x_min': np.nan, 'x_max': np.nan,
                'y_min': np.nan, 'y_max': np.nan,
                'z_min': np.nan, 'z_max': np.nan,
                'ra_min': np.nan, 'ra_max': np.nan,
                'dec_min': np.nan, 'dec_max': np.nan}

    x = np.asarray(source_table['X_CART'], dtype=np.float64)[finite]
    y = np.asarray(source_table['Y_CART'], dtype=np.float64)[finite]
    z = np.asarray(source_table['Z_CART'], dtype=np.float64)[finite]
    chi = np.sqrt(x * x + y * y + z * z)

    bounds = {'source': source_name,
              'chi_min': float(np.nanmin(chi)),
              'chi_max': float(np.nanmax(chi)),
              'x_min': float(np.nanmin(x)),
              'x_max': float(np.nanmax(x)),
              'y_min': float(np.nanmin(y)),
              'y_max': float(np.nanmax(y)),
              'z_min': float(np.nanmin(z)),
              'z_max': float(np.nanmax(z)),
              'ra_min': np.nan, 'ra_max': np.nan,
              'dec_min': np.nan, 'dec_max': np.nan}

    if 'RA' in source_table.colnames:
        ra = np.asarray(source_table['RA'], dtype=np.float64)[finite]
        ra = ra[np.isfinite(ra)]
        if ra.size:
            bounds['ra_min'] = float(np.nanmin(ra))
            bounds['ra_max'] = float(np.nanmax(ra))

    if 'DEC' in source_table.colnames:
        dec = np.asarray(source_table['DEC'], dtype=np.float64)[finite]
        dec = dec[np.isfinite(dec)]
        if dec.size:
            bounds['dec_min'] = float(np.nanmin(dec))
            bounds['dec_max'] = float(np.nanmax(dec))

    return bounds


def _solid_angle_from_bounds(bounds):
    '''
    Solid angle of a simple RA/DEC rectangle in steradians.
    '''
    if not all(np.isfinite(bounds[name])
               for name in ('ra_min', 'ra_max', 'dec_min', 'dec_max')):
        return np.nan

    ra_min = float(bounds['ra_min']) % 360.0
    ra_max = float(bounds['ra_max']) % 360.0
    ra_width = (ra_max - ra_min) % 360.0
    if ra_width == 0.0 and float(bounds['ra_max']) > float(bounds['ra_min']):
        ra_width = 360.0
    dec_min = np.radians(np.clip(float(bounds['dec_min']), -90.0, 90.0))
    dec_max = np.radians(np.clip(float(bounds['dec_max']), -90.0, 90.0))
    omega = np.radians(ra_width) * (np.sin(dec_max) - np.sin(dec_min))
    return float(omega) if omega > 0.0 else np.nan


def _solid_angle_from_randoms(rand_table, n_ra=360, n_sin_dec=180):
    '''
    Estimate angular footprint area from occupied equal-area RA/sin(DEC) cells.
    '''
    if 'RA' not in rand_table.colnames or 'DEC' not in rand_table.colnames:
        return np.nan

    ra = np.asarray(rand_table['RA'], dtype=np.float64)
    dec = np.asarray(rand_table['DEC'], dtype=np.float64)
    finite = np.isfinite(ra) & np.isfinite(dec) & (dec >= -90.0) & (dec <= 90.0)
    if not np.any(finite):
        return np.nan

    n_ra = max(int(n_ra), 1)
    n_sin_dec = max(int(n_sin_dec), 1)
    ra = np.mod(ra[finite], 360.0)
    sin_dec = np.sin(np.radians(dec[finite]))

    ra_idx = np.floor(ra / 360.0 * n_ra).astype(np.int64)
    dec_idx = np.floor((sin_dec + 1.0) * 0.5 * n_sin_dec).astype(np.int64)
    ra_idx = np.clip(ra_idx, 0, n_ra - 1)
    dec_idx = np.clip(dec_idx, 0, n_sin_dec - 1)
    cell_ids = ra_idx * n_sin_dec + dec_idx
    n_occupied = np.unique(cell_ids).size
    omega_cell = 4.0 * np.pi / float(n_ra * n_sin_dec)
    return float(n_occupied * omega_cell)


def _estimate_survey_volume(rand_table, bounds):
    '''
    Estimate the comoving survey volume in (Mpc/h)^3 for uniform randoms.
    '''
    chi_min = float(bounds['chi_min'])
    chi_max = float(bounds['chi_max'])
    if not (np.isfinite(chi_min) and np.isfinite(chi_max) and chi_max > chi_min):
        return np.nan, np.nan

    omega = _solid_angle_from_randoms(rand_table)
    if not np.isfinite(omega) or omega <= 0.0:
        omega = _solid_angle_from_bounds(bounds)
    if not np.isfinite(omega) or omega <= 0.0:
        return np.nan, np.nan

    volume = omega * (chi_max ** 3 - chi_min ** 3) / 3.0
    return float(volume), float(omega)


def _flag_radial_edge(chi_members, chi_center, edge_scale, bounds,
                      edge_radial_buffer):
    '''
    Compute radial edge diagnostics for one void group.
    '''
    chi_members = np.asarray(chi_members, dtype=np.float64)
    finite = np.isfinite(chi_members)
    if not np.any(finite):
        return np.nan, np.nan, np.nan, False, False

    chi_min_group = float(np.nanmin(chi_members[finite]))
    chi_max_group = float(np.nanmax(chi_members[finite]))
    chi_min_sample = float(bounds['chi_min'])
    chi_max_sample = float(bounds['chi_max'])

    if not (np.isfinite(chi_min_sample) and np.isfinite(chi_max_sample)):
        return chi_min_group, chi_max_group, np.nan, False, False

    radial_buffer = 0.0 if edge_radial_buffer is None else max(float(edge_radial_buffer), 0.0)
    touches_radial = ((chi_min_group <= chi_min_sample + radial_buffer) or
                      (chi_max_group >= chi_max_sample - radial_buffer))

    d_radial = float(min(chi_center - chi_min_sample,
                         chi_max_sample - chi_center))
    center_near = bool(np.isfinite(d_radial) and np.isfinite(edge_scale) and
                       (d_radial < float(edge_scale)))

    return chi_min_group, chi_max_group, d_radial, bool(touches_radial), center_near


def _flag_cartesian_edge(x_members, y_members, z_members,
                         x_center, y_center, z_center,
                         edge_scale, bounds, edge_cartesian_buffer):
    '''
    Compute Cartesian edge diagnostics for one void group.
    '''
    x_members = np.asarray(x_members, dtype=np.float64)
    y_members = np.asarray(y_members, dtype=np.float64)
    z_members = np.asarray(z_members, dtype=np.float64)
    finite = np.isfinite(x_members) & np.isfinite(y_members) & np.isfinite(z_members)
    if not np.any(finite):
        return (np.nan, np.nan, np.nan, np.nan, np.nan, np.nan,
                np.nan, False, False)

    x_min_group = float(np.nanmin(x_members[finite]))
    x_max_group = float(np.nanmax(x_members[finite]))
    y_min_group = float(np.nanmin(y_members[finite]))
    y_max_group = float(np.nanmax(y_members[finite]))
    z_min_group = float(np.nanmin(z_members[finite]))
    z_max_group = float(np.nanmax(z_members[finite]))

    sample_limits = (bounds['x_min'], bounds['x_max'],
                     bounds['y_min'], bounds['y_max'],
                     bounds['z_min'], bounds['z_max'])
    if not np.all(np.isfinite(sample_limits)):
        return (x_min_group, x_max_group, y_min_group, y_max_group,
                z_min_group, z_max_group, np.nan, False, False)

    cart_buffer = max(float(edge_cartesian_buffer), 0.0)
    touches_cart = (x_min_group <= bounds['x_min'] + cart_buffer or
                    x_max_group >= bounds['x_max'] - cart_buffer or
                    y_min_group <= bounds['y_min'] + cart_buffer or
                    y_max_group >= bounds['y_max'] - cart_buffer or
                    z_min_group <= bounds['z_min'] + cart_buffer or
                    z_max_group >= bounds['z_max'] - cart_buffer)

    d_cart = float(min(x_center - bounds['x_min'],
                       bounds['x_max'] - x_center,
                       y_center - bounds['y_min'],
                       bounds['y_max'] - y_center,
                       z_center - bounds['z_min'],
                       bounds['z_max'] - z_center))
    center_near_cart = bool(np.isfinite(d_cart) and np.isfinite(edge_scale) and
                            (d_cart < float(edge_scale)))

    return (x_min_group, x_max_group, y_min_group, y_max_group,
            z_min_group, z_max_group, d_cart,
            bool(touches_cart), center_near_cart)


def _flag_angular_edge(rand_table, member_idx, bounds, edge_angular_buffer_deg):
    '''
    Optionally flag groups touching simple min/max RA or DEC bounds.

    This is a linear RA check and is intended as a diagnostic. It is disabled
    when edge_angular_buffer_deg is None.
    '''
    if edge_angular_buffer_deg is None:
        return False, False

    angular_buffer = max(float(edge_angular_buffer_deg), 0.0)
    touches_ra = False
    touches_dec = False

    if ('RA' in rand_table.colnames and
            np.isfinite(bounds['ra_min']) and np.isfinite(bounds['ra_max'])):
        ra = np.asarray(rand_table['RA'], dtype=np.float64)[member_idx]
        ra = ra[np.isfinite(ra)]
        if ra.size:
            touches_ra = bool((float(np.nanmin(ra)) <= bounds['ra_min'] + angular_buffer) or
                              (float(np.nanmax(ra)) >= bounds['ra_max'] - angular_buffer))

    if ('DEC' in rand_table.colnames and
            np.isfinite(bounds['dec_min']) and np.isfinite(bounds['dec_max'])):
        dec = np.asarray(rand_table['DEC'], dtype=np.float64)[member_idx]
        dec = dec[np.isfinite(dec)]
        if dec.size:
            touches_dec = bool((float(np.nanmin(dec)) <= bounds['dec_min'] + angular_buffer) or
                               (float(np.nanmax(dec)) >= bounds['dec_max'] - angular_buffer))

    return touches_ra, touches_dec


def _healpix_pixels(hp, nside, ra_deg, dec_deg, nest=False):
    '''
    Convert RA/DEC arrays to HEALPix pixels, returning pixels and a finite mask.
    '''
    ra = np.asarray(ra_deg, dtype=np.float64)
    dec = np.asarray(dec_deg, dtype=np.float64)
    valid = np.isfinite(ra) & np.isfinite(dec) & (dec >= -90.0) & (dec <= 90.0)
    pix = np.full(ra.shape, -1, dtype=np.int64)
    if np.any(valid):
        theta = np.radians(90.0 - dec[valid])
        phi = np.radians(np.mod(ra[valid], 360.0))
        pix[valid] = hp.ang2pix(int(nside), theta, phi, nest=nest)
    return pix, valid


def _build_healpix_edge_mask(rand_table, nside=256, min_randoms_per_pix=3,
                             edge_buffer_deg=1.0, nest=False):
    '''
    Build a buffered angular-edge HEALPix mask from random catalogue positions.
    '''
    try:
        import healpy as hp
    except ImportError as exc:
        raise ImportError('HEALPix footprint-edge masking requires healpy. '
                          'Load an environment with healpy installed before '
                          'running catalogue generation.') from exc

    if nside is None or int(nside) <= 0:
        raise ValueError('HEALPix edge masking is required; healpix_edge_nside must be a positive integer.')
    if 'RA' not in rand_table.colnames or 'DEC' not in rand_table.colnames:
        raise KeyError('HEALPix edge masking requires RA and DEC columns in the random catalogue.')

    nside = int(nside)
    min_randoms_per_pix = max(int(min_randoms_per_pix), 1)
    npix = hp.nside2npix(nside)
    pix, valid = _healpix_pixels(hp, nside, rand_table['RA'], rand_table['DEC'],
                                 nest=nest)
    if not np.any(valid):
        raise RuntimeError('HEALPix edge masking failed: no finite random RA/DEC values.')

    counts = np.bincount(pix[valid], minlength=npix)
    mask = counts >= min_randoms_per_pix
    observed = np.flatnonzero(mask)
    if observed.size == 0:
        raise RuntimeError('HEALPix edge masking failed: no observed pixels passed '
                           f'min_randoms_per_pix={min_randoms_per_pix}. Lower '
                           '--healpix-edge-min-randoms or use a lower --healpix-edge-nside.')

    neighbors = hp.get_all_neighbours(nside, observed, nest=nest)
    valid_neighbors = neighbors >= 0
    safe_neighbors = np.where(valid_neighbors, neighbors, observed[None, :])
    edge_flags = np.any(valid_neighbors & ~mask[safe_neighbors], axis=0)
    edge_pix = np.zeros(npix, dtype=bool)
    edge_pix[observed] = edge_flags

    edge_buffered = edge_pix.copy()
    buffer_deg = 0.0 if edge_buffer_deg is None else max(float(edge_buffer_deg), 0.0)
    if buffer_deg > 0.0 and np.any(edge_pix):
        radius = np.radians(buffer_deg)
        for pix_id in np.flatnonzero(edge_pix):
            vec = hp.pix2vec(nside, int(pix_id), nest=nest)
            nearby = hp.query_disc(nside, vec, radius, nest=nest)
            edge_buffered[nearby] = True

    return {'hp': hp,
            'nside': nside,
            'nest': bool(nest),
            'min_randoms_per_pix': min_randoms_per_pix,
            'edge_buffer_deg': buffer_deg,
            'mask': mask,
            'edge_pix': edge_pix,
            'edge_buffered': edge_buffered,
            'n_observed_pix': int(np.count_nonzero(mask)),
            'n_edge_pix': int(np.count_nonzero(edge_pix)),
            'n_buffered_edge_pix': int(np.count_nonzero(edge_buffered))}


def consolidate_group_info(data_table, rand_table, cosmo, h,
                           group_col='GROUPID', min_rand_for_shape=3,
                           edge_radial_buffer=20.0,
                           edge_angular_buffer_deg=1.0,
                           edge_cartesian_buffer=None,
                           healpix_edge_nside=256,
                           healpix_edge_min_randoms=3,
                           healpix_edge_nest=False):
    '''
    Consolidate group information from data and random tables to create a group table with properties.

    Parameters:
        - data_table: Astropy Table containing the data points with group IDs.
        - rand_table: Astropy Table containing the random points with group IDs.
        - cosmo: Astropy cosmology instance for distance-redshift conversion.
        - h: Dimensionless Hubble parameter (H0 / 100) for distance unit conversion.
        - group_col: Name of the column in both tables that contains the group IDs.
        - min_rand_for_shape: Minimum number of random members required to compute
                              shape parameters (semi-axes). If a group has fewer random members than
                              this threshold, its semi-axes will be set to NaN.
        - edge_radial_buffer: Buffer in Mpc/h used to flag groups whose random
                              members touch the radial survey boundary.
        - edge_angular_buffer_deg: Optional angular buffer in degrees for simple
                                   RA/DEC edge flags. If None, angular edge
                                   flags are disabled.
        - edge_cartesian_buffer: Cartesian X/Y/Z buffer in Mpc/h for edge
                                 flags. If None, uses edge_radial_buffer.
        - healpix_edge_nside: Required HEALPix NSIDE for angular footprint
                              edge flags. Must be positive.
        - healpix_edge_min_randoms: Minimum randoms per pixel for the angular
                                    footprint mask.
        - healpix_edge_nest: Use NESTED HEALPix ordering if True.
    Returns:
        - An Astropy Table containing consolidated group information, including VOID_ID, N_DATA_IN_GROUP,
          N_RAND_IN_GROUP, RA, DEC, REDSHIFT, X, Y, Z, R_EFF,
          LAMBDA_1, LAMBDA_2, LAMBDA_3, SEMI_AXIS_A, SEMI_AXIS_B,
          SEMI_AXIS_C, EDGE, FOOTPRINT_EDGE, and GEOM_BAD for each group.
          EDGE is the standard watershed edge flag and is False for consolidated
          GROUPID >= 0 void rows; FOOTPRINT_EDGE marks objects touching the
          survey footprint/mask boundary. The shape
          eigenvalues are the eigenvalues of the central second-moment tensor
          <(x-x_cm)_i (x-x_cm)_j>. semi-axis_j = sqrt(5 * LAMBDA_j),
          and R_EFF is the radius of the sphere with the same volume as the
          ellipsoid: R_EFF = sqrt(5) * (LAMBDA_1 * LAMBDA_2 * LAMBDA_3)**(1/6).
    '''
    data_gids = np.asarray(data_table[group_col], dtype=np.int32)
    rand_gids = np.asarray(rand_table[group_col], dtype=np.int32)

    valid_data = data_gids >= 0
    valid_rand = rand_gids >= 0

    if not np.any(valid_data) and not np.any(valid_rand):
        return _empty_group_table()

    if np.any(valid_data):
        data_sizes = np.bincount(data_gids[valid_data])
    else:
        data_sizes = np.array([], dtype=np.int64)

    rand_valid_idx = np.flatnonzero(valid_rand)
    if len(rand_valid_idx) == 0:
        return _empty_group_table()
    rand_valid_gid = rand_gids[rand_valid_idx]
    rand_order = np.argsort(rand_valid_gid, kind='stable')
    rand_idx_sorted = rand_valid_idx[rand_order]
    rand_gid_sorted = rand_valid_gid[rand_order]
    unique_rand_gid, rand_start, rand_count = np.unique(rand_gid_sorted,
                                                        return_index=True,
                                                        return_counts=True)

    bounds = _sample_geometry_bounds(data_table=data_table, rand_table=rand_table)
    cart_buffer = edge_radial_buffer if edge_cartesian_buffer is None else edge_cartesian_buffer
    if cart_buffer is None:
        cart_buffer = 0.0

    x_rand_all = np.asarray(rand_table['X_CART'], dtype=np.float64)
    y_rand_all = np.asarray(rand_table['Y_CART'], dtype=np.float64)
    z_rand_all = np.asarray(rand_table['Z_CART'], dtype=np.float64)
    x_members = x_rand_all[rand_idx_sorted]
    y_members = y_rand_all[rand_idx_sorted]
    z_members = z_rand_all[rand_idx_sorted]

    n_group = len(unique_rand_gid)
    n_rand = rand_count.astype(np.int32)
    n_float = rand_count.astype(np.float64)

    n_data = np.zeros(n_group, dtype=np.int32)
    in_data_sizes = unique_rand_gid < len(data_sizes)
    if np.any(in_data_sizes):
        n_data[in_data_sizes] = data_sizes[unique_rand_gid[in_data_sizes]].astype(np.int32)

    sum_x = np.add.reduceat(x_members, rand_start)
    sum_y = np.add.reduceat(y_members, rand_start)
    sum_z = np.add.reduceat(z_members, rand_start)
    sum_x2 = np.add.reduceat(x_members * x_members, rand_start)
    sum_y2 = np.add.reduceat(y_members * y_members, rand_start)
    sum_z2 = np.add.reduceat(z_members * z_members, rand_start)
    sum_xy = np.add.reduceat(x_members * y_members, rand_start)
    sum_xz = np.add.reduceat(x_members * z_members, rand_start)
    sum_yz = np.add.reduceat(y_members * z_members, rand_start)

    x_cm = sum_x / n_float
    y_cm = sum_y / n_float
    z_cm = sum_z / n_float

    center_r2 = x_cm * x_cm + y_cm * y_cm + z_cm * z_cm
    mean_centered_r2 = np.clip((sum_x2 + sum_y2 + sum_z2) / n_float - center_r2,
                               0.0, None)
    r_moment_eff = np.sqrt(5.0 * mean_centered_r2 / 3.0)

    survey_volume, survey_omega = _estimate_survey_volume(rand_table, bounds)
    n_rand_density = int(np.count_nonzero(_finite_xyz_mask(rand_table)))
    rand_density = np.nan
    if (np.isfinite(survey_volume) and survey_volume > 0.0 and
            n_rand_density > 0):
        rand_density = float(n_rand_density) / float(survey_volume)

    dx2 = np.clip(sum_x2 - n_float * x_cm * x_cm, 0.0, None)
    dy2 = np.clip(sum_y2 - n_float * y_cm * y_cm, 0.0, None)
    dz2 = np.clip(sum_z2 - n_float * z_cm * z_cm, 0.0, None)
    dxy = sum_xy - n_float * x_cm * y_cm
    dxz = sum_xz - n_float * x_cm * z_cm
    dyz = sum_yz - n_float * y_cm * z_cm

    lambda_axes = np.full((n_group, 3), np.nan, dtype=np.float64)
    semi_axes = np.full((n_group, 3), np.nan, dtype=np.float64)
    valid_shape = n_rand >= int(min_rand_for_shape)
    if np.any(valid_shape):
        shape = np.empty((int(np.count_nonzero(valid_shape)), 3, 3),
                         dtype=np.float64)
        inv_n = 1.0 / n_float[valid_shape]
        shape[:, 0, 0] = dx2[valid_shape] * inv_n
        shape[:, 1, 1] = dy2[valid_shape] * inv_n
        shape[:, 2, 2] = dz2[valid_shape] * inv_n
        shape[:, 0, 1] = shape[:, 1, 0] = dxy[valid_shape] * inv_n
        shape[:, 0, 2] = shape[:, 2, 0] = dxz[valid_shape] * inv_n
        shape[:, 1, 2] = shape[:, 2, 1] = dyz[valid_shape] * inv_n
        eigenvalues = np.linalg.eigvalsh(shape)[:, ::-1]
        eigenvalues = np.clip(eigenvalues, 0.0, None)
        lambda_axes[valid_shape] = eigenvalues
        semi_axes[valid_shape] = np.sqrt(5.0 * eigenvalues)

    lambda_1 = lambda_axes[:, 0]
    lambda_2 = lambda_axes[:, 1]
    lambda_3 = lambda_axes[:, 2]
    semi_axis_a = semi_axes[:, 0]
    semi_axis_b = semi_axes[:, 1]
    semi_axis_c = semi_axes[:, 2]

    r_eff = np.full(n_group, np.nan, dtype=np.float64)
    valid_reff = (np.isfinite(semi_axis_a) & np.isfinite(semi_axis_b) &
                  np.isfinite(semi_axis_c) &
                  (semi_axis_a >= 0.0) & (semi_axis_b >= 0.0) &
                  (semi_axis_c >= 0.0))
    r_eff[valid_reff] = np.cbrt(
        semi_axis_a[valid_reff] *
        semi_axis_b[valid_reff] *
        semi_axis_c[valid_reff])

    axis_ellip = np.full(n_group, np.nan, dtype=np.float64)
    valid_axis_ratio = (np.isfinite(semi_axis_a) & np.isfinite(semi_axis_c) &
                        (semi_axis_a > 0.0))
    axis_ellip[valid_axis_ratio] = 1.0 - semi_axis_c[valid_axis_ratio] / semi_axis_a[valid_axis_ratio]
    bad_axis = np.isfinite(axis_ellip) & (axis_ellip > 0.9)
    geom_bad = bad_axis

    r_cm = np.sqrt(center_r2)
    ra_cm = np.full(n_group, np.nan, dtype=np.float64)
    dec_cm = np.full(n_group, np.nan, dtype=np.float64)
    nonzero_center = r_cm > 0.0
    ra_cm[nonzero_center] = (
        np.degrees(np.arctan2(y_cm[nonzero_center],
                              x_cm[nonzero_center])) % 360.0)
    dec_cm[nonzero_center] = np.degrees(
        np.arcsin(np.clip(z_cm[nonzero_center] / r_cm[nonzero_center],
                          -1.0, 1.0)))

    chi_members = np.sqrt(x_members * x_members +
                          y_members * y_members +
                          z_members * z_members)
    chi_min_group = np.minimum.reduceat(chi_members, rand_start)
    chi_max_group = np.maximum.reduceat(chi_members, rand_start)

    radial_buffer = 0.0 if edge_radial_buffer is None else max(float(edge_radial_buffer), 0.0)
    chi_min_sample = float(bounds['chi_min'])
    chi_max_sample = float(bounds['chi_max'])
    d_radial_edge = np.full(n_group, np.nan, dtype=np.float64)
    touches_radial = np.zeros(n_group, dtype=bool)
    if np.isfinite(chi_min_sample) and np.isfinite(chi_max_sample):
        touches_radial = ((chi_min_group <= chi_min_sample + radial_buffer) |
                          (chi_max_group >= chi_max_sample - radial_buffer))
        d_radial_edge = np.minimum(r_cm - chi_min_sample, chi_max_sample - r_cm)

    edge_scale = np.where(np.isfinite(semi_axis_a) & (semi_axis_a > 0.0),
                          semi_axis_a, r_moment_eff)
    center_near_radial = (np.isfinite(d_radial_edge) &
                          np.isfinite(edge_scale) &
                          (d_radial_edge < edge_scale))

    x_min_group = np.minimum.reduceat(x_members, rand_start)
    x_max_group = np.maximum.reduceat(x_members, rand_start)
    y_min_group = np.minimum.reduceat(y_members, rand_start)
    y_max_group = np.maximum.reduceat(y_members, rand_start)
    z_min_group = np.minimum.reduceat(z_members, rand_start)
    z_max_group = np.maximum.reduceat(z_members, rand_start)

    cart_buffer = max(float(cart_buffer), 0.0)
    sample_limits = (bounds['x_min'], bounds['x_max'],
                     bounds['y_min'], bounds['y_max'],
                     bounds['z_min'], bounds['z_max'])
    d_cart_edge = np.full(n_group, np.nan, dtype=np.float64)
    touches_cart = np.zeros(n_group, dtype=bool)
    center_near_cart = np.zeros(n_group, dtype=bool)
    if np.all(np.isfinite(sample_limits)):
        touches_cart = (
            (x_min_group <= bounds['x_min'] + cart_buffer) |
            (x_max_group >= bounds['x_max'] - cart_buffer) |
            (y_min_group <= bounds['y_min'] + cart_buffer) |
            (y_max_group >= bounds['y_max'] - cart_buffer) |
            (z_min_group <= bounds['z_min'] + cart_buffer) |
            (z_max_group >= bounds['z_max'] - cart_buffer))
        d_cart_edge = np.minimum.reduce([
            x_cm - bounds['x_min'],
            bounds['x_max'] - x_cm,
            y_cm - bounds['y_min'],
            bounds['y_max'] - y_cm,
            z_cm - bounds['z_min'],
            bounds['z_max'] - z_cm])
        center_near_cart = (np.isfinite(d_cart_edge) &
                            np.isfinite(edge_scale) &
                            (d_cart_edge < edge_scale))

    touches_ra = np.zeros(n_group, dtype=bool)
    touches_dec = np.zeros(n_group, dtype=bool)
    if edge_angular_buffer_deg is not None:
        angular_buffer = max(float(edge_angular_buffer_deg), 0.0)
        if ('RA' in rand_table.colnames and
                np.isfinite(bounds['ra_min']) and np.isfinite(bounds['ra_max'])):
            ra_members = np.asarray(rand_table['RA'], dtype=np.float64)[rand_idx_sorted]
            finite_ra = np.isfinite(ra_members)
            ra_min_group = np.minimum.reduceat(np.where(finite_ra, ra_members, np.inf),
                                               rand_start)
            ra_max_group = np.maximum.reduceat(np.where(finite_ra, ra_members, -np.inf),
                                               rand_start)
            has_ra = np.isfinite(ra_min_group) & np.isfinite(ra_max_group)
            touches_ra = (has_ra &
                          ((ra_min_group <= bounds['ra_min'] + angular_buffer) |
                           (ra_max_group >= bounds['ra_max'] - angular_buffer)))
        if ('DEC' in rand_table.colnames and
                np.isfinite(bounds['dec_min']) and np.isfinite(bounds['dec_max'])):
            dec_members = np.asarray(rand_table['DEC'], dtype=np.float64)[rand_idx_sorted]
            finite_dec = np.isfinite(dec_members)
            dec_min_group = np.minimum.reduceat(np.where(finite_dec, dec_members, np.inf),
                                                rand_start)
            dec_max_group = np.maximum.reduceat(np.where(finite_dec, dec_members, -np.inf),
                                                rand_start)
            has_dec = np.isfinite(dec_min_group) & np.isfinite(dec_max_group)
            touches_dec = (has_dec &
                           ((dec_min_group <= bounds['dec_min'] + angular_buffer) |
                            (dec_max_group >= bounds['dec_max'] - angular_buffer)))

    healpix_edge = _build_healpix_edge_mask(
        rand_table=rand_table,
        nside=healpix_edge_nside,
        min_randoms_per_pix=healpix_edge_min_randoms,
        edge_buffer_deg=edge_angular_buffer_deg,
        nest=healpix_edge_nest)
    if healpix_edge is None:
        raise RuntimeError('HEALPix edge masking is required but mask construction returned None.')
    touches_healpix = np.zeros(n_group, dtype=bool)
    center_near_healpix = np.zeros(n_group, dtype=bool)
    hp = healpix_edge['hp']
    nside = healpix_edge['nside']
    nest = healpix_edge['nest']
    edge_buffered = healpix_edge['edge_buffered']

    member_pix, member_pix_valid = _healpix_pixels(
        hp, nside,
        np.asarray(rand_table['RA'], dtype=np.float64)[rand_idx_sorted],
        np.asarray(rand_table['DEC'], dtype=np.float64)[rand_idx_sorted],
        nest=nest)
    member_on_edge = np.zeros(member_pix.shape, dtype=bool)
    member_on_edge[member_pix_valid] = edge_buffered[member_pix[member_pix_valid]]
    touches_healpix = np.maximum.reduceat(member_on_edge, rand_start)

    center_pix, center_pix_valid = _healpix_pixels(hp, nside, ra_cm, dec_cm,
                                                   nest=nest)
    center_near_healpix[center_pix_valid] = edge_buffered[center_pix[center_pix_valid]]

    footprint_edge = (touches_radial | center_near_radial |
                      touches_healpix | center_near_healpix |
                      touches_ra | touches_dec |
                      touches_cart | center_near_cart)

    z_vals = comoving_distance_to_redshift(r_cm / h, cosmo=cosmo)

    group_table = Table(data=[unique_rand_gid.astype(np.int32),
                              n_data.astype(np.int32), n_rand.astype(np.int32),
                              ra_cm.astype(np.float64), dec_cm.astype(np.float64), z_vals.astype(np.float64),
                              x_cm.astype(np.float64), y_cm.astype(np.float64), z_cm.astype(np.float64),
                              r_eff.astype(np.float64),
                              lambda_1.astype(np.float64),
                              lambda_2.astype(np.float64),
                              lambda_3.astype(np.float64),
                              semi_axis_a.astype(np.float64),
                              semi_axis_b.astype(np.float64),
                              semi_axis_c.astype(np.float64),
                              chi_min_group.astype(np.float64),
                              chi_max_group.astype(np.float64),
                              r_cm.astype(np.float64),
                              d_radial_edge.astype(np.float64),
                              np.full(n_group, chi_min_sample, dtype=np.float64),
                              np.full(n_group, chi_max_sample, dtype=np.float64),
                              x_min_group.astype(np.float64),
                              x_max_group.astype(np.float64),
                              y_min_group.astype(np.float64),
                              y_max_group.astype(np.float64),
                              z_min_group.astype(np.float64),
                              z_max_group.astype(np.float64),
                              d_cart_edge.astype(np.float64),
                              np.full(n_group, float(bounds['x_min']), dtype=np.float64),
                              np.full(n_group, float(bounds['x_max']), dtype=np.float64),
                              np.full(n_group, float(bounds['y_min']), dtype=np.float64),
                              np.full(n_group, float(bounds['y_max']), dtype=np.float64),
                              np.full(n_group, float(bounds['z_min']), dtype=np.float64),
                              np.full(n_group, float(bounds['z_max']), dtype=np.float64),
                              geom_bad.astype(np.bool_),
                              np.zeros(n_group, dtype=np.bool_),
                              footprint_edge.astype(np.bool_),
                              touches_radial.astype(np.bool_),
                              center_near_radial.astype(np.bool_),
                              touches_ra.astype(np.bool_),
                              touches_dec.astype(np.bool_),
                              touches_healpix.astype(np.bool_),
                              center_near_healpix.astype(np.bool_),
                              touches_cart.astype(np.bool_),
                              center_near_cart.astype(np.bool_)], names=list(GROUP_COLUMNS))
    group_table.meta['EDGE_RBUF'] = float(edge_radial_buffer) if edge_radial_buffer is not None else 0.0
    group_table.meta['EDGE_ABUF'] = (float(edge_angular_buffer_deg)
                                     if edge_angular_buffer_deg is not None else np.nan)
    group_table.meta['EDGE_CBUF'] = float(cart_buffer) if cart_buffer is not None else 0.0
    group_table.meta['EDGE_SRC'] = str(bounds['source'])
    group_table.meta['HPX_EDGE'] = True
    group_table.meta['HPX_NSIDE'] = int(healpix_edge['nside'])
    group_table.meta['HPX_NEST'] = bool(healpix_edge['nest'])
    group_table.meta['HPX_MINR'] = int(healpix_edge['min_randoms_per_pix'])
    group_table.meta['HPX_EBUF'] = float(healpix_edge['edge_buffer_deg'])
    group_table.meta['HPX_NOBS'] = int(healpix_edge['n_observed_pix'])
    group_table.meta['HPX_NEDGE'] = int(healpix_edge['n_edge_pix'])
    group_table.meta['HPX_NBUF'] = int(healpix_edge['n_buffered_edge_pix'])
    group_table.meta['CHI_MIN'] = float(bounds['chi_min'])
    group_table.meta['CHI_MAX'] = float(bounds['chi_max'])
    group_table.meta['X_MIN'] = float(bounds['x_min'])
    group_table.meta['X_MAX'] = float(bounds['x_max'])
    group_table.meta['Y_MIN'] = float(bounds['y_min'])
    group_table.meta['Y_MAX'] = float(bounds['y_max'])
    group_table.meta['Z_MIN'] = float(bounds['z_min'])
    group_table.meta['Z_MAX'] = float(bounds['z_max'])
    group_table.meta['SURVEY_VOL'] = float(survey_volume) if np.isfinite(survey_volume) else np.nan
    group_table.meta['SURVEY_OMG'] = float(survey_omega) if np.isfinite(survey_omega) else np.nan
    group_table.meta['RAND_DENS'] = float(rand_density) if np.isfinite(rand_density) else np.nan
    group_table.meta['NRAND_DENS'] = int(n_rand_density)

    return group_table


def _stack_column(data_table, rand_table, col_name, dtype, fill_value):
    '''
    Stack a column from the data and random tables into a single array, ensuring consistent dtype and
    handling missing columns.

    Parameters:
        - data_table: Astropy Table containing the data points.
        - rand_table: Astropy Table containing the random points.
        - col_name: Name of the column to stack from both tables.
        - dtype: Desired data type for the output array.
        - fill_value: Value to use for filling the output array if the column is missing in either table.
    Returns:
        - A single array containing the values from the specified column in both tables, with missing
          columns filled with fill_value.
    Raises:
        - ValueError: If the specified column exists in one table but not the other, which is not allowed.
                      The column must either exist in both tables or in neither table.
    '''
    has_data = col_name in data_table.colnames
    has_rand = col_name in rand_table.colnames

    n_data = len(data_table)
    n_rand = len(rand_table)

    if has_data and has_rand:
        return np.concatenate([np.asarray(data_table[col_name], dtype=dtype),
                               np.asarray(rand_table[col_name], dtype=dtype)])
    if not has_data and not has_rand:
        return np.full(n_data + n_rand, fill_value, dtype=dtype)
    raise ValueError(f'Column "{col_name}" must exist in both tables or in neither table.')


def build_point_membership_table(data_table, rand_table, group_col='GROUPID',
                                 boundary_col=None, boundary_id=BOUNDARY_ID):
    '''
    Build a point membership table that combines information from the data and
    random tables, including group IDs, watershed boundary flags, void
    classifications, and relevant columns for analysis.

    Parameters:
        - data_table: Astropy Table containing the data points with a GROUPID column.
        - rand_table: Astropy Table containing the random points with a GROUPID column.
        - group_col: Name of the column in both tables that contains the group IDs.
        - boundary_col: Optional column containing explicit boundary flags.
        - boundary_id: GROUPID value used for watershed boundary points.
    If class columns are missing, defaults are used in the output.
    Returns:
        - An Astropy Table containing the combined point membership information.
          EDGE=True marks points with GROUPID equal to boundary_id.
    '''
    n_data = len(data_table)
    n_rand = len(rand_table)
    n_total = n_data + n_rand

    targetid = np.concatenate([np.asarray(data_table['TARGETID']),
                               np.asarray(rand_table['TARGETID'])])

    groupid = np.concatenate([np.asarray(data_table[group_col], dtype=np.int32),
                              np.asarray(rand_table[group_col], dtype=np.int32)])
    is_boundary = (groupid == int(boundary_id)).astype(np.int8)
    if boundary_col is not None:
        explicit_boundary = np.zeros(n_total, dtype=np.int8)
        if boundary_col in data_table.colnames:
            explicit_boundary[:n_data] = np.asarray(data_table[boundary_col],
                                                    dtype=np.int8)
        if boundary_col in rand_table.colnames:
            explicit_boundary[n_data:] = np.asarray(rand_table[boundary_col],
                                                    dtype=np.int8)
        is_boundary = ((is_boundary != 0) | (explicit_boundary != 0)).astype(np.int8)

    class_label = np.full(n_total, 'unknown', dtype='U13')
    class_id = np.full(n_total, -1, dtype=np.int8)

    is_data = np.zeros(n_total, dtype=np.int8)
    is_data[:n_data] = 1
    node_index = np.arange(n_total, dtype=np.int32)

    r_vals = _stack_column(data_table, rand_table, 'R', np.float32, np.nan)
    n_data_neighbors = _stack_column(data_table, rand_table, 'N_DATA', np.int32, -1)
    n_rand_neighbors = _stack_column(data_table, rand_table, 'N_RAND', np.int32, -1)
    ra_vals = _stack_column(data_table, rand_table, 'RA', np.float64, np.nan)
    dec_vals = _stack_column(data_table, rand_table, 'DEC', np.float64, np.nan)
    z_vals = _stack_column(data_table, rand_table, 'Z', np.float64, np.nan)
    x_cart = _stack_column(data_table, rand_table, 'X_CART', np.float64, np.nan)
    y_cart = _stack_column(data_table, rand_table, 'Y_CART', np.float64, np.nan)
    z_cart = _stack_column(data_table, rand_table, 'Z_CART', np.float64, np.nan)

    out = Table()
    out['NODE_INDEX'] = node_index
    out['TARGETID'] = targetid
    out['IS_DATA'] = is_data
    out['GROUPID'] = groupid
    out['EDGE'] = is_boundary.astype(np.bool_)
    out['IS_BOUNDARY'] = is_boundary
    out['VOID_CLASS_ID'] = class_id
    out['VOID_CLASS'] = class_label
    out['R'] = r_vals
    out['N_DATA'] = n_data_neighbors
    out['N_RAND'] = n_rand_neighbors
    out['RA'] = ra_vals
    out['DEC'] = dec_vals
    out['Z'] = z_vals
    out['X_CART'] = x_cart
    out['Y_CART'] = y_cart
    out['Z_CART'] = z_cart
    return out


def ellipticity_from_axes(group_table):
    '''
    Compute void ellipticity from stored second-moment axis values.

    The definition is 1 - (J_1 / J_3)**0.25. For principal
    axis lengths a <= b <= c, J_1 / J_3 = (a^2 + b^2) /
    (b^2 + c^2). In this catalog SEMI_AXIS_A >= SEMI_AXIS_B >=
    SEMI_AXIS_C, so a=C, b=B, and c=A.
    Invalid or non-positive axes return NaN.
    '''
    ellip = np.full(len(group_table), np.nan, dtype=np.float32)
    if len(group_table) == 0:
        return ellip

    needed = ('SEMI_AXIS_A', 'SEMI_AXIS_B', 'SEMI_AXIS_C')
    if any(col not in group_table.colnames for col in needed):
        return ellip

    semi_a = np.asarray(group_table['SEMI_AXIS_A'], dtype=np.float64)
    semi_b = np.asarray(group_table['SEMI_AXIS_B'], dtype=np.float64)
    semi_c = np.asarray(group_table['SEMI_AXIS_C'], dtype=np.float64)
    valid = (np.isfinite(semi_a) & np.isfinite(semi_b) & np.isfinite(semi_c) &
             (semi_a > 0.0) & (semi_b > 0.0) & (semi_c > 0.0))
    if not np.any(valid):
        return ellip

    numerator = semi_c[valid] * semi_c[valid] + semi_b[valid] * semi_b[valid]
    denominator = semi_b[valid] * semi_b[valid] + semi_a[valid] * semi_a[valid]
    ratio = numerator / denominator
    ratio = np.clip(ratio, 0.0, 1.0)
    ellip[valid] = (1.0 - np.power(ratio, 0.25)).astype(np.float32)
    return ellip


def add_ellipticity_column(group_table, output_col='ELLIP'):
    '''
    Add an ellipticity column to the group table based on the semi-axes lengths.

    Parameters:
        - group_table: Astropy Table containing the group information, including SEMI_AXIS_A,
                       SEMI_AXIS_B, and SEMI_AXIS_C columns.
        - output_col: Name of the column to store the computed ellipticity values. The ellipticity
                      is defined as 1 - (J_1/J_3)**0.25, where
                      J_1/J_3 = (SEMI_AXIS_C^2 + SEMI_AXIS_B^2) /
                      (SEMI_AXIS_B^2 + SEMI_AXIS_A^2).
                      If the necessary semi-axis columns are missing or contain non-positive values,
                      the ellipticity will be set to NaN for those groups.
    Returns:
        - An Astropy Table with the added ellipticity column.
    '''
    out = group_table.copy()

    out[output_col] = ellipticity_from_axes(out)
    return out


def write_group_table_fits(group_table, output_path, tracer, cap,
                           h, omega_m, r_threshold, mode,
                           point_table=None, overwrite=False,
                           seed_threshold=None, boundary_id=BOUNDARY_ID,
                           watershed_stats=None, merge_threshold=None):
    '''
    Write the group table to a FITS file with appropriate metadata in the header.

    Parameters:
        - group_table: Astropy Table containing the consolidated group information. Rows with
                       FOOTPRINT_EDGE=True are excluded from the VOIDS HDU by default.
        - output_path: File path for the output FITS file.
        - tracer: Name of the tracer (e.g., 'BGS_ANY', 'LRG', 'ELGnotqso', 'QSO') to include in the metadata.
        - cap: Name of the sky cap (e.g., 'NGC', 'SGC') to include in the metadata.
        - h: Dimensionless Hubble parameter (H0 / 100) to include in the metadata.
        - omega_m: Matter density parameter to include in the metadata.
        - r_threshold: R threshold used in the watershed algorithm to include in the metadata.
        - mode: Watershed mode used (e.g., 'underdense' or 'overdense') to include in the metadata.
        - point_table: Optional Astropy Table containing point membership information to be included
                       as a separate HDU in the FITS file. If provided, the metadata will also include
                       statistics about the point classifications.
        - overwrite: If True, overwrite the output file if it already exists. If False and the file
                     exists, a FileExistsError will be raised.
        - seed_threshold: Optional seed threshold used by the watershed.
        - boundary_id: GROUPID value used for watershed boundary points.
        - watershed_stats: Optional dict of watershed summary values for the header.
        - merge_threshold: Optional saddle-density threshold used by the
                           watershed to merge neighboring basins.
    Returns:
        - The file path of the written FITS file.
    '''
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)

    out_table = group_table.copy()
    if 'FOOTPRINT_EDGE' not in out_table.colnames:
        if 'EDGE' in out_table.colnames:
            out_table['FOOTPRINT_EDGE'] = np.asarray(out_table['EDGE'], dtype=np.bool_)
        else:
            out_table['FOOTPRINT_EDGE'] = np.zeros(len(out_table), dtype=np.bool_)
    if 'EDGE' not in out_table.colnames:
        out_table['EDGE'] = np.zeros(len(out_table), dtype=np.bool_)
    else:
        out_table['EDGE'] = np.zeros(len(out_table), dtype=np.bool_)

    n_void_raw = len(out_table)
    footprint_edge_flags = np.asarray(out_table['FOOTPRINT_EDGE'], dtype=bool)
    n_footprint_edge = int(np.count_nonzero(footprint_edge_flags))
    n_footprint_clean = int(np.count_nonzero(~footprint_edge_flags))
    out_table = out_table[~footprint_edge_flags]
    out_table = add_ellipticity_column(out_table, output_col='ELLIP')
    edge_flags = np.asarray(out_table['EDGE'], dtype=bool)
    n_edge = int(np.count_nonzero(edge_flags))
    footprint_edge_flags_written = np.asarray(out_table['FOOTPRINT_EDGE'], dtype=bool)

    primary_hdu = fits.PrimaryHDU()
    hdr = primary_hdu.header
    hdr['TRACER'] = (tracer, 'Tracer type')
    hdr['CAP'] = (cap, 'Sky cap')
    hdr['NVOIDS'] = (len(out_table), 'Clean voids written')
    hdr['NVOIDRAW'] = (n_void_raw, 'Voids before footprint cut')
    hdr['H'] = (float(h), 'h = H0 / 100')
    hdr['OMEGA_M'] = (float(omega_m), 'Matter density parameter')
    hdr['RTHRESH'] = (float(r_threshold), 'R threshold for watershed')
    hdr['WMODE'] = (mode, 'Watershed mode')
    hdr['UNITSXYZ'] = ('Mpc/h', 'XYZ, R_EFF, semi-axis units')
    hdr['REFFDEF'] = (REFF_DEFINITION, 'R_EFF')
    hdr['LAMDEF'] = ('eig(<dx_i dx_j>)', 'LAMBDA_1..3 definition')
    hdr['AXDEF'] = ('SEMI_AXIS_j=sqrt(5*LAMBDA_j)', 'Semi-axis definition')
    hdr['UNITSANG'] = ('deg', 'Units for RA and DEC')
    hdr['ELLIPDEF'] = (ELLIPTICITY_DEFINITION, 'Ellipticity definition')
    hdr['J1J3'] = (J1J3_DEFINITION, 'Moment ratio')
    hdr['GEOMDEF'] = ('1-C/A>0.9', 'GEOM_BAD definition')
    hdr['PTUNITSX'] = ('Mpc/h', 'POINT_MEMBERSHIP XYZ units')
    hdr['PTUNITSR'] = ('dimensionless', 'POINT_MEMBERSHIP R units')
    hdr['PTUNITSZ'] = ('redshift', 'POINT_MEMBERSHIP Z units')
    hdr['GIDM1'] = (-1, 'GROUPID=-1 means unassigned point')
    hdr['GIDM2'] = (int(boundary_id), 'GROUPID for watershed boundary point')
    if seed_threshold is not None:
        hdr['SEEDTHR'] = (float(seed_threshold), 'Watershed seed threshold')
    if merge_threshold is not None:
        hdr['MERGETHR'] = (float(merge_threshold), 'Watershed saddle merge threshold')
    hdr['EDGEDEF'] = ('GROUPID==boundary_id', 'EDGE=True means watershed boundary')
    hdr['FPEDDEF'] = ('survey footprint/mask boundary', 'FOOTPRINT_EDGE definition')
    hdr['FPCUT'] = (True, 'Drop FOOTPRINT_EDGE rows')
    hdr['NEDGE'] = (n_edge, 'EDGE=True rows in VOIDS')
    hdr['NFPEDGE'] = (n_footprint_edge, 'Footprint-edge rows dropped')
    hdr['NFPCLN'] = (n_footprint_clean, 'Clean voids written')
    hdr['NFPWRT'] = (int(np.count_nonzero(footprint_edge_flags_written)),
                     'Footprint-edge rows written')
    if 'GEOM_BAD' in out_table.colnames:
        geom_flags = np.asarray(out_table['GEOM_BAD'], dtype=bool)
        hdr['NGEOMBAD'] = (int(np.count_nonzero(geom_flags)), 'Number of GEOM_BAD=True voids')
    if 'EDGE_RBUF' in group_table.meta:
        hdr['ERADBUF'] = (float(group_table.meta['EDGE_RBUF']), 'Radial edge buffer in Mpc/h')
    if 'EDGE_ABUF' in group_table.meta and np.isfinite(group_table.meta['EDGE_ABUF']):
        hdr['EANGBUF'] = (float(group_table.meta['EDGE_ABUF']), 'Angular edge buffer in deg')
    if 'EDGE_CBUF' in group_table.meta:
        hdr['ECARBUF'] = (float(group_table.meta['EDGE_CBUF']), 'Cartesian edge buffer in Mpc/h')
    if 'EDGE_SRC' in group_table.meta:
        hdr['EDGESRC'] = (str(group_table.meta['EDGE_SRC']), 'Sample used for survey edge bounds')
    if 'HPX_EDGE' in group_table.meta:
        hdr['HPXEDGE'] = (bool(group_table.meta['HPX_EDGE']), 'HEALPix angular edge enabled')
    if 'HPX_NSIDE' in group_table.meta:
        hdr['HPXNSIDE'] = (int(group_table.meta['HPX_NSIDE']), 'HEALPix edge NSIDE')
    if 'HPX_NEST' in group_table.meta:
        hdr['HPXNEST'] = (bool(group_table.meta['HPX_NEST']), 'HEALPix NESTED ordering')
    if 'HPX_MINR' in group_table.meta:
        hdr['HPXMINR'] = (int(group_table.meta['HPX_MINR']), 'Min randoms per HEALPix pixel')
    if 'HPX_EBUF' in group_table.meta:
        hdr['HPXEBUF'] = (float(group_table.meta['HPX_EBUF']), 'HEALPix edge buffer in deg')
    if 'HPX_NOBS' in group_table.meta:
        hdr['HPXNOBS'] = (int(group_table.meta['HPX_NOBS']), 'Observed HEALPix pixels')
    if 'HPX_NEDGE' in group_table.meta:
        hdr['HPXNEDG'] = (int(group_table.meta['HPX_NEDGE']), 'Angular edge HEALPix pixels')
    if 'HPX_NBUF' in group_table.meta:
        hdr['HPXNBUF'] = (int(group_table.meta['HPX_NBUF']), 'Buffered angular edge HEALPix pixels')
    if 'SURVEY_VOL' in group_table.meta and np.isfinite(group_table.meta['SURVEY_VOL']):
        hdr['SURVVOL'] = (float(group_table.meta['SURVEY_VOL']), 'Survey volume in (Mpc/h)^3')
    if 'SURVEY_OMG' in group_table.meta and np.isfinite(group_table.meta['SURVEY_OMG']):
        hdr['SURVOMG'] = (float(group_table.meta['SURVEY_OMG']), 'Survey solid angle in sr')
    if 'RAND_DENS' in group_table.meta and np.isfinite(group_table.meta['RAND_DENS']):
        hdr['RANDDENS'] = (float(group_table.meta['RAND_DENS']), 'Mean random density h^3/Mpc^3')
    if 'NRAND_DENS' in group_table.meta:
        hdr['NRANDDEN'] = (int(group_table.meta['NRAND_DENS']), 'Randoms used for mean density')
    if 'CHI_MIN' in group_table.meta and np.isfinite(group_table.meta['CHI_MIN']):
        hdr['CHIMIN'] = (float(group_table.meta['CHI_MIN']), 'Sample minimum chi in Mpc/h')
    if 'CHI_MAX' in group_table.meta and np.isfinite(group_table.meta['CHI_MAX']):
        hdr['CHIMAX'] = (float(group_table.meta['CHI_MAX']), 'Sample maximum chi in Mpc/h')
    if point_table is not None:
        hdr['NPOINTS'] = (len(point_table), 'Number of points in POINT_MEMBERSHIP table')
        point_gids = np.asarray(point_table['GROUPID'], dtype=np.int32)
        hdr['NPTASGN'] = (int(np.count_nonzero(point_gids >= 0)), 'Assigned points')
        hdr['NPTUNASN'] = (int(np.count_nonzero(point_gids == -1)), 'Unassigned points')
        hdr['NPTBND'] = (int(np.count_nonzero(point_gids == int(boundary_id))),
                         'Watershed boundary points')
    if watershed_stats is not None:
        if 'n_boundary_nodes' in watershed_stats:
            hdr['WBNODES'] = (int(watershed_stats['n_boundary_nodes']),
                              'Watershed boundary nodes')
        if 'n_unassigned' in watershed_stats:
            hdr['WUNASGN'] = (int(watershed_stats['n_unassigned']),
                              'Watershed unassigned nodes')

    table_hdu = fits.BinTableHDU(data=out_table.as_array(), name='VOIDS')
    hdus = [primary_hdu, table_hdu]
    if point_table is not None:
        point_hdu = fits.BinTableHDU(data=point_table.as_array(), name='POINT_MEMBERSHIP')
        hdus.append(point_hdu)
    hdul = fits.HDUList(hdus)
    hdul.writeto(output_path, overwrite=overwrite)
    return output_path
