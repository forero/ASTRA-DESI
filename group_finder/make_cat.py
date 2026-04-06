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
                 'R_EFF', 'SEMI_AXIS_A', 'SEMI_AXIS_B', 'SEMI_AXIS_C')

GROUP_DTYPES = (np.int32, np.int32, np.int32,
                np.float64, np.float64, np.float64,
                np.float64, np.float64, np.float64,
                np.float64, np.float64, np.float64,
                np.float64)


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


def consolidate_group_info(data_table, rand_table, cosmo, h,
                           group_col='GROUPID', min_rand_for_shape=3):
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

        rows.append((gid, n_data, n_rand, ra_cm, dec_cm,
                     0.0, x_cm, y_cm, z_cm, r_eff,
                     float(semi_axes[0]), float(semi_axes[1]),
                     float(semi_axes[2])))
        dist_mpc_list.append(r_cm / h)

    if not rows:
        return _empty_group_table()

    group_table = Table(rows=rows, names=list(GROUP_COLUMNS), dtype=list(GROUP_DTYPES))

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
    hdr['UNITSXYZ'] = ('Mpc/h', 'Units for X,Y,Z,R_EFF,SEMI_AXIS_* columns')
    hdr['UNITSANG'] = ('deg', 'Units for RA and DEC')
    hdr['ELLIPDEF'] = ('1-C/A', 'Ellipticity definition')
    hdr['PTUNITSX'] = ('Mpc/h', 'Units for X_CART,Y_CART,Z_CART in POINT_MEMBERSHIP')
    hdr['PTUNITSR'] = ('dimensionless', 'Units for R in POINT_MEMBERSHIP')
    hdr['PTUNITSZ'] = ('redshift', 'Units for Z in POINT_MEMBERSHIP')
    hdr['GIDM1'] = (-1, 'GROUPID=-1 means unassigned point')
    if point_table is not None:
        hdr['NPOINTS'] = (len(point_table), 'Number of points in POINT_MEMBERSHIP table')

    table_hdu = fits.BinTableHDU(data=out_table.as_array(), name='VOIDS')
    hdus = [primary_hdu, table_hdu]
    if point_table is not None:
        point_hdu = fits.BinTableHDU(data=point_table.as_array(), name='POINT_MEMBERSHIP')
        hdus.append(point_hdu)
    hdul = fits.HDUList(hdus)
    hdul.writeto(output_path, overwrite=overwrite)
    return output_path