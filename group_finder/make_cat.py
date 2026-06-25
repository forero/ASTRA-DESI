import os
import numpy as np
from astropy.io import fits
from astropy.table import Table
from astropy.cosmology import z_at_value
import astropy.units as u

try:
    from .watershed import compute_semi_axes
except ImportError:
    from watershed import compute_semi_axes


GROUP_COLUMNS = ('VOID_ID', 'N_DATA_IN_GROUP', 'N_RAND_IN_GROUP',
                 'RA', 'DEC', 'REDSHIFT', 'X', 'Y', 'Z',
                 'R_EFF', 'SEMI_AXIS_A', 'SEMI_AXIS_B', 'SEMI_AXIS_C',
                 'CHI_MIN_GROUP', 'CHI_MAX_GROUP', 'CHI_CENTER',
                 'D_RADIAL_EDGE', 'CHI_MIN_SAMPLE', 'CHI_MAX_SAMPLE',
                 'X_MIN_GROUP', 'X_MAX_GROUP', 'Y_MIN_GROUP', 'Y_MAX_GROUP',
                 'Z_MIN_GROUP', 'Z_MAX_GROUP', 'D_CART_EDGE',
                 'X_MIN_SAMPLE', 'X_MAX_SAMPLE', 'Y_MIN_SAMPLE', 'Y_MAX_SAMPLE',
                 'Z_MIN_SAMPLE', 'Z_MAX_SAMPLE',
                 'EDGE', 'TOUCHES_RADIAL_EDGE', 'CENTER_NEAR_RADIAL_EDGE',
                 'TOUCHES_RA_EDGE', 'TOUCHES_DEC_EDGE',
                 'TOUCHES_CART_EDGE', 'CENTER_NEAR_CART_EDGE')

GROUP_DTYPES = (np.int32, np.int32, np.int32,
                np.float64, np.float64, np.float64,
                np.float64, np.float64, np.float64,
                np.float64, np.float64, np.float64,
                np.float64,
                np.float64, np.float64, np.float64,
                np.float64, np.float64, np.float64,
                np.float64, np.float64, np.float64, np.float64,
                np.float64, np.float64, np.float64,
                np.float64, np.float64, np.float64, np.float64,
                np.float64, np.float64,
                np.bool_, np.bool_, np.bool_,
                np.bool_, np.bool_,
                np.bool_, np.bool_)


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
    This implementation uses direct inversion with z_at_value (no interpolation grid).

    Parameters:
        - dist_mpc: Array of comoving distances in Mpc to convert to redshift.
        - cosmo: Astropy cosmology instance to use for distance-redshift conversion.
        - z_max_init: Unused (kept only for backward compatibility).
        - n_grid: Unused (kept only for backward compatibility).
    Returns:
        - Array of redshift values corresponding to the input comoving distances.
    Raises:
        - ValueError: If any finite comoving distance is negative.
    '''
    _ = (z_max_init, n_grid)

    dist_mpc = np.asarray(dist_mpc, dtype=np.float64)
    if dist_mpc.size == 0:
        return np.array([], dtype=np.float64)

    negative_mask = np.isfinite(dist_mpc) & (dist_mpc < 0)
    if np.any(negative_mask):
        raise ValueError('Comoving distance must be non-negative')

    z_vals = np.full(dist_mpc.shape, np.nan, dtype=np.float64)
    finite_idx = np.where(np.isfinite(dist_mpc))[0]
    for i in finite_idx:
        z_vals[i] = float(z_at_value(cosmo.comoving_distance, dist_mpc[i] * u.Mpc))

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


def consolidate_group_info(data_table, rand_table, cosmo, h,
                           group_col='GROUPID', min_rand_for_shape=3,
                           edge_radial_buffer=20.0,
                           edge_angular_buffer_deg=1.0,
                           edge_cartesian_buffer=None):
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
    Returns:
        - An Astropy Table containing consolidated group information, including VOID_ID, N_DATA_IN_GROUP,
          N_RAND_IN_GROUP, RA, DEC, REDSHIFT, X, Y, Z, R_EFF, SEMI_AXIS_A, SEMI_AXIS_B, and SEMI_AXIS_C for
          each group.
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

    rows = []
    dist_mpc_list = []
    bounds = _sample_geometry_bounds(data_table=data_table, rand_table=rand_table)
    cart_buffer = edge_radial_buffer if edge_cartesian_buffer is None else edge_cartesian_buffer
    if cart_buffer is None:
        cart_buffer = 0.0

    x_rand_all = np.asarray(rand_table['X_CART'], dtype=np.float64)
    y_rand_all = np.asarray(rand_table['Y_CART'], dtype=np.float64)
    z_rand_all = np.asarray(rand_table['Z_CART'], dtype=np.float64)

    for i, gid in enumerate(unique_rand_gid.tolist()):
        start = int(rand_start[i])
        stop = start + int(rand_count[i])
        member_idx = rand_idx_sorted[start:stop]

        n_rand = int(stop - start)
        n_data = int(data_sizes[gid]) if gid < len(data_sizes) else 0

        x_members = x_rand_all[member_idx]
        y_members = y_rand_all[member_idx]
        z_members = z_rand_all[member_idx]
        chi_members = np.sqrt(x_members * x_members +
                              y_members * y_members +
                              z_members * z_members)

        x_cm = float(np.mean(x_members))
        y_cm = float(np.mean(y_members))
        z_cm = float(np.mean(z_members))

        dx = x_members - x_cm
        dy = y_members - y_cm
        dz = z_members - z_cm
        r_eff = float(np.sqrt(np.mean(dx * dx + dy * dy + dz * dz)))

        if n_rand >= min_rand_for_shape:
            semi_axes = compute_semi_axes(x_members=x_members,
                                          y_members=y_members,
                                          z_members=z_members,
                                          x_cm=x_cm,
                                          y_cm=y_cm,
                                          z_cm=z_cm)
        else:
            semi_axes = np.array([np.nan, np.nan, np.nan], dtype=np.float64)

        r_cm = float(np.sqrt(x_cm * x_cm + y_cm * y_cm + z_cm * z_cm))
        if r_cm > 0:
            ra_cm = float(np.degrees(np.arctan2(y_cm, x_cm)) % 360.0)
            dec_cm = float(np.degrees(np.arcsin(np.clip(z_cm / r_cm, -1.0, 1.0))))
        else:
            ra_cm = np.nan
            dec_cm = np.nan

        semi_axis_a = float(semi_axes[0])
        semi_axis_b = float(semi_axes[1])
        semi_axis_c = float(semi_axes[2])
        edge_scale = semi_axis_a if np.isfinite(semi_axis_a) and semi_axis_a > 0.0 else r_eff

        chi_min_group, chi_max_group, d_radial_edge, touches_radial, center_near_radial = (
            _flag_radial_edge(chi_members=chi_members,
                              chi_center=r_cm,
                              edge_scale=edge_scale,
                              bounds=bounds,
                              edge_radial_buffer=edge_radial_buffer))
        (x_min_group, x_max_group, y_min_group, y_max_group,
         z_min_group, z_max_group, d_cart_edge,
         touches_cart, center_near_cart) = _flag_cartesian_edge(
            x_members=x_members, y_members=y_members, z_members=z_members,
            x_center=x_cm, y_center=y_cm, z_center=z_cm,
            edge_scale=edge_scale, bounds=bounds,
            edge_cartesian_buffer=cart_buffer)
        touches_ra, touches_dec = _flag_angular_edge(rand_table=rand_table,
                                                     member_idx=member_idx,
                                                     bounds=bounds,
                                                     edge_angular_buffer_deg=edge_angular_buffer_deg)
        edge = bool(touches_radial or center_near_radial or
                    touches_ra or touches_dec or
                    touches_cart or center_near_cart)

        rows.append((gid, n_data, n_rand, ra_cm, dec_cm,
                     0.0, x_cm, y_cm, z_cm, r_eff,
                     semi_axis_a, semi_axis_b, semi_axis_c,
                     chi_min_group, chi_max_group, r_cm,
                     d_radial_edge, float(bounds['chi_min']), float(bounds['chi_max']),
                     x_min_group, x_max_group, y_min_group, y_max_group,
                     z_min_group, z_max_group, d_cart_edge,
                     float(bounds['x_min']), float(bounds['x_max']),
                     float(bounds['y_min']), float(bounds['y_max']),
                     float(bounds['z_min']), float(bounds['z_max']),
                     edge, touches_radial, center_near_radial,
                     touches_ra, touches_dec,
                     touches_cart, center_near_cart))
        dist_mpc_list.append(r_cm / h)

    if not rows:
        return _empty_group_table()

    group_table = Table(rows=rows, names=list(GROUP_COLUMNS), dtype=list(GROUP_DTYPES))
    group_table.meta['EDGE_RBUF'] = float(edge_radial_buffer) if edge_radial_buffer is not None else 0.0
    group_table.meta['EDGE_ABUF'] = (float(edge_angular_buffer_deg)
                                     if edge_angular_buffer_deg is not None else np.nan)
    group_table.meta['EDGE_CBUF'] = float(cart_buffer) if cart_buffer is not None else 0.0
    group_table.meta['EDGE_SRC'] = str(bounds['source'])
    group_table.meta['CHI_MIN'] = float(bounds['chi_min'])
    group_table.meta['CHI_MAX'] = float(bounds['chi_max'])
    group_table.meta['X_MIN'] = float(bounds['x_min'])
    group_table.meta['X_MAX'] = float(bounds['x_max'])
    group_table.meta['Y_MIN'] = float(bounds['y_min'])
    group_table.meta['Y_MAX'] = float(bounds['y_max'])
    group_table.meta['Z_MIN'] = float(bounds['z_min'])
    group_table.meta['Z_MAX'] = float(bounds['z_max'])

    z_vals = comoving_distance_to_redshift(np.asarray(dist_mpc_list, dtype=np.float64), cosmo=cosmo)
    group_table['REDSHIFT'] = z_vals.astype(np.float64)

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


def build_point_membership_table(data_table, rand_table, group_col='GROUPID'):
    '''
    Build a point membership table that combines information from the data and random tables,
    including group IDs, void classifications, and relevant columns for analysis.

    Parameters:
        - data_table: Astropy Table containing the data points with a GROUPID column.
        - rand_table: Astropy Table containing the random points with a GROUPID column.
        - group_col: Name of the column in both tables that contains the group IDs.
    If class columns are missing, defaults are used in the output.
    Returns:
        - An Astropy Table containing the combined point membership information, including TARGETID, IS_DATA
    '''
    n_data = len(data_table)
    n_rand = len(rand_table)
    n_total = n_data + n_rand

    targetid = np.concatenate([np.asarray(data_table['TARGETID']),
                               np.asarray(rand_table['TARGETID'])])

    groupid = np.concatenate([np.asarray(data_table[group_col], dtype=np.int32),
                              np.asarray(rand_table[group_col], dtype=np.int32)])

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


def add_ellipticity_column(group_table, output_col='ELLIP'):
    '''
    Add an ellipticity column to the group table based on the semi-axes lengths.

    Parameters:
        - group_table: Astropy Table containing the group information, including SEMI_AXIS_A and
                       SEMI_AXIS_C columns.
        - output_col: Name of the column to store the computed ellipticity values. The ellipticity
                      is defined as 1 - (C/A), where A is the longest semi-axis and C is the
                      shortest semi-axis. If the necessary semi-axis columns are missing or
                      contain non-positive values, the ellipticity will be set to NaN for those groups.
    Returns:
        - An Astropy Table with the added ellipticity column.
    '''
    out = group_table.copy()

    if len(out) == 0:
        out[output_col] = np.array([], dtype=np.float32)
        return out

    semi_a = np.asarray(out['SEMI_AXIS_A'], dtype=np.float64)
    semi_c = np.asarray(out['SEMI_AXIS_C'], dtype=np.float64)

    ellip = np.full(len(out), np.nan, dtype=np.float64)
    valid = np.isfinite(semi_a) & np.isfinite(semi_c) & (semi_a > 0)
    ellip[valid] = 1.0 - (semi_c[valid] / semi_a[valid])

    out[output_col] = ellip.astype(np.float32)
    return out


def write_group_table_fits(group_table, output_path, tracer, cap,
                           h, omega_m, r_threshold, mode,
                           point_table=None, overwrite=False):
    '''
    Write the group table to a FITS file with appropriate metadata in the header.

    Parameters:
        - group_table: Astropy Table containing the consolidated group information to be written to the FITS file.
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
    Returns:
        - The file path of the written FITS file.
    '''
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)

    out_table = add_ellipticity_column(group_table, output_col='ELLIP')

    primary_hdu = fits.PrimaryHDU()
    hdr = primary_hdu.header
    hdr['TRACER'] = (tracer, 'Tracer type')
    hdr['CAP'] = (cap, 'Sky cap')
    hdr['NVOIDS'] = (len(out_table), 'Number of voids')
    hdr['H'] = (float(h), 'h = H0 / 100')
    hdr['OMEGA_M'] = (float(omega_m), 'Matter density parameter')
    hdr['RTHRESH'] = (float(r_threshold), 'R threshold for watershed')
    hdr['WMODE'] = (mode, 'Watershed mode')
    hdr['UNITSXYZ'] = ('Mpc/h', 'XYZ, R_EFF, semi-axis units')
    hdr['UNITSANG'] = ('deg', 'Units for RA and DEC')
    hdr['ELLIPDEF'] = ('1-C/A', 'Ellipticity definition')
    hdr['PTUNITSX'] = ('Mpc/h', 'POINT_MEMBERSHIP XYZ units')
    hdr['PTUNITSR'] = ('dimensionless', 'POINT_MEMBERSHIP R units')
    hdr['PTUNITSZ'] = ('redshift', 'POINT_MEMBERSHIP Z units')
    hdr['GIDM1'] = (-1, 'GROUPID=-1 means unassigned point')
    hdr['EDGEDEF'] = ('radial_or_angular_or_cartesian', 'Primary EDGE flag')
    if 'EDGE' in out_table.colnames:
        edge_flags = np.asarray(out_table['EDGE'], dtype=bool)
        hdr['NEDGE'] = (int(np.count_nonzero(edge_flags)), 'Number of EDGE=True voids')
        hdr['NCLEAN'] = (int(np.count_nonzero(~edge_flags)), 'Number of EDGE=False voids')
    if 'EDGE_RBUF' in group_table.meta:
        hdr['ERADBUF'] = (float(group_table.meta['EDGE_RBUF']), 'Radial edge buffer in Mpc/h')
    if 'EDGE_ABUF' in group_table.meta and np.isfinite(group_table.meta['EDGE_ABUF']):
        hdr['EANGBUF'] = (float(group_table.meta['EDGE_ABUF']), 'Angular edge buffer in deg')
    if 'EDGE_CBUF' in group_table.meta:
        hdr['ECARBUF'] = (float(group_table.meta['EDGE_CBUF']), 'Cartesian edge buffer in Mpc/h')
    if 'EDGE_SRC' in group_table.meta:
        hdr['EDGESRC'] = (str(group_table.meta['EDGE_SRC']), 'Sample used for survey edge bounds')
    if 'CHI_MIN' in group_table.meta and np.isfinite(group_table.meta['CHI_MIN']):
        hdr['CHIMIN'] = (float(group_table.meta['CHI_MIN']), 'Sample minimum chi in Mpc/h')
    if 'CHI_MAX' in group_table.meta and np.isfinite(group_table.meta['CHI_MAX']):
        hdr['CHIMAX'] = (float(group_table.meta['CHI_MAX']), 'Sample maximum chi in Mpc/h')
    if point_table is not None:
        hdr['NPOINTS'] = (len(point_table), 'Number of points in POINT_MEMBERSHIP table')

    table_hdu = fits.BinTableHDU(data=out_table.as_array(), name='VOIDS')
    hdus = [primary_hdu, table_hdu]
    if point_table is not None:
        point_hdu = fits.BinTableHDU(data=point_table.as_array(), name='POINT_MEMBERSHIP')
        hdus.append(point_hdu)
    if 'EDGE' in out_table.colnames:
        clean_table = out_table[~np.asarray(out_table['EDGE'], dtype=bool)]
        clean_hdu = fits.BinTableHDU(data=clean_table.as_array(), name='VOIDS_CLEAN')
        hdus.append(clean_hdu)
    hdul = fits.HDUList(hdus)
    hdul.writeto(output_path, overwrite=overwrite)
    return output_path