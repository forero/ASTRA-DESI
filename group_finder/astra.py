import astropy.units as u
import numpy as np
from astropy.cosmology import FlatLambdaCDM
from scipy.spatial import Delaunay

DEFAULT_H = 0.6736
DEFAULT_OMEGA_M = 0.315


EDGE_PAIRS_3D = np.array([[0, 1], [0, 2],
                          [0, 3], [1, 2],
                          [1, 3], [2, 3]], dtype=np.int64)


def build_cosmology(h=DEFAULT_H, omega_m=DEFAULT_OMEGA_M):
    '''
    Build a FlatLambdaCDM cosmology instance based on the provided Hubble parameter and matter density.

    Parameters:
        - h: Hubble parameter (dimensionless, e.g., 0.6736).
        - omega_m: Matter density parameter (e.g., 0.315).
    Returns:
        - An instance of astropy.cosmology.FlatLambdaCDM configured with the specified parameters.
    '''
    return FlatLambdaCDM(H0=h * 100.0, Om0=omega_m)


def radec_z_to_cartesian(ra_deg, dec_deg, redshift, cosmo, h):
    '''
    Convert RA, DEC, and redshift to Cartesian coordinates (X, Y, Z) in comoving Mpc/h.

    Parameters:
        - ra_deg: Right Ascension in degrees.
        - dec_deg: Declination in degrees.
        - redshift: Redshift of the object.
        - cosmo: An instance of astropy.cosmology to compute comoving distances.
        - h: Hubble parameter (dimensionless) to convert distances to Mpc/h.
    Returns:
        - x, y, z_cart: Arrays of Cartesian coordinates in comoving Mpc/h. The z coordinate
                        is named z_cart to avoid confusion with redshift.
    '''
    ra = np.radians(np.asarray(ra_deg, dtype=np.float64))
    dec = np.radians(np.asarray(dec_deg, dtype=np.float64))
    z = np.asarray(redshift, dtype=np.float64)

    r_mpc_h = cosmo.comoving_distance(z).to(u.Mpc).value * h

    x = r_mpc_h * np.cos(dec) * np.cos(ra)
    y = r_mpc_h * np.cos(dec) * np.sin(ra)
    z_cart = r_mpc_h * np.sin(dec)
    return x, y, z_cart


def add_cartesian_columns(table, cosmo, h):
    '''
    Add Cartesian coordinate columns (X_CART, Y_CART, Z_CART) to the input table based on its
    RA, DEC, and Z columns.

    Parameters:
        - table: An Astropy Table containing 'RA', 'DEC', and 'Z' columns.
        - cosmo: An instance of astropy.cosmology to compute comoving distances.
        - h: Hubble parameter (dimensionless) to convert distances to Mpc/h.
    '''
    x, y, z = radec_z_to_cartesian(table['RA'],
                                   table['DEC'],
                                   table['Z'],
                                   cosmo=cosmo, h=h)
    table['X_CART'] = x
    table['Y_CART'] = y
    table['Z_CART'] = z


def add_cartesian_to_all(all_data, cosmo, h):
    '''
    Add Cartesian coordinate columns to all tables in the all_data dict.

    Parameters:
        - all_data: Dict containing the loaded data tables for each tracer and cap.
        - cosmo: An instance of astropy.cosmology to compute comoving distances.
        - h: Hubble parameter (dimensionless) to convert distances to Mpc/h.
    '''
    for key, table in all_data.items():
        add_cartesian_columns(table, cosmo=cosmo, h=h)


def _build_unique_edges_from_simplices(simplices):
    '''
    Build a unique set of edges from the given simplices of a Delaunay triangulation.

    Parameters:
        - simplices: An array of shape (n_simplices, 4) containing the indices of
                     the vertices of each simplex (tetrahedron) in 3D.
    Returns:
        - An array of shape (n_edges, 2) containing the unique edges as pairs of
          vertex indices, sorted in ascending order within each edge and with no
          duplicate edges.
    '''
    n_simplices = simplices.shape[0]
    n_pairs = EDGE_PAIRS_3D.shape[0]
    edges = np.empty((n_simplices * n_pairs, 2), dtype=np.int64)
    for i, pair in enumerate(EDGE_PAIRS_3D):
        start = i * n_simplices
        stop = start + n_simplices
        edges[start:stop] = simplices[:, pair]
    edges = np.sort(edges, axis=1)
    edges = np.unique(edges, axis=0)
    return edges


def compute_neighbor_statistics(data_table, rand_table):
    '''
    Compute neighbor statistics for the combined data and random tables using Delaunay triangulation.

    Parameters:
        - data_table: Astropy Table containing the data points with Cartesian coordinates.
        - rand_table: Astropy Table containing the random points with Cartesian coordinates.
    Returns:
        - A dict containing the computed neighbor statistics, including:
          'coords': Combined Cartesian coordinates of data and random points.
          'is_data': Boolean array indicating which points are data vs random.
          'targetid': Combined TARGETID array for data and random points.
          'edges': Array of unique edges from the Delaunay triangulation.
          'neighbors': List of neighbor indices for each point.
          'n_data_neighbors': Array of counts of data neighbors for each point.
          'n_rand_neighbors': Array of counts of random neighbors for each point.
          'r_values': Array of R values computed as (N_DATA - N_RAND) / (N_DATA + N_RAND) for each point.
    Raises:
        - ValueError: If there are fewer than 4 total points (data + random), which is the min
                      required for 3D Delaunay triangulation, or if any points have
    '''
    n_data = len(data_table)
    n_rand = len(rand_table)
    n_total = n_data + n_rand

    if n_total < 4:
        raise ValueError(f'Need at least 4 total points for 3D Delaunay, got {n_total}')

    coords = np.empty((n_total, 3), dtype=np.float64)
    coords[:n_data, 0] = np.asarray(data_table['X_CART'], dtype=np.float64)
    coords[:n_data, 1] = np.asarray(data_table['Y_CART'], dtype=np.float64)
    coords[:n_data, 2] = np.asarray(data_table['Z_CART'], dtype=np.float64)
    coords[n_data:, 0] = np.asarray(rand_table['X_CART'], dtype=np.float64)
    coords[n_data:, 1] = np.asarray(rand_table['Y_CART'], dtype=np.float64)
    coords[n_data:, 2] = np.asarray(rand_table['Z_CART'], dtype=np.float64)

    is_data = np.zeros(n_total, dtype=bool)
    is_data[:n_data] = True

    tri = Delaunay(coords)
    edges = _build_unique_edges_from_simplices(tri.simplices.astype(np.int64, copy=False))

    n_data_neighbors = np.zeros(n_total, dtype=np.int32)
    n_rand_neighbors = np.zeros(n_total, dtype=np.int32)

    u_idx = edges[:, 0]
    v_idx = edges[:, 1]

    u_is_data = is_data[u_idx].astype(np.int32)
    v_is_data = is_data[v_idx].astype(np.int32)

    np.add.at(n_data_neighbors, u_idx, v_is_data)
    np.add.at(n_rand_neighbors, u_idx, 1 - v_is_data)

    np.add.at(n_data_neighbors, v_idx, u_is_data)
    np.add.at(n_rand_neighbors, v_idx, 1 - u_is_data)

    denom = n_data_neighbors + n_rand_neighbors
    if np.any(denom == 0):
        zero_nodes = np.where(denom == 0)[0]
        raise ValueError('Some nodes have no neighbors after triangulation. '
                         f'First problematic indices: {zero_nodes[:10].tolist()}')

    r_values = (n_data_neighbors - n_rand_neighbors) / denom

    neighbors = [[] for _ in range(n_total)]
    for i in range(edges.shape[0]):
        a = int(edges[i, 0])
        b = int(edges[i, 1])
        neighbors[a].append(b)
        neighbors[b].append(a)

    targetid = np.concatenate([np.asarray(data_table['TARGETID']),
                               np.asarray(rand_table['TARGETID'])])

    return {'coords': coords, 'is_data': is_data, 'targetid': targetid,
            'edges': edges, 'neighbors': neighbors,
            'n_data_neighbors': n_data_neighbors,
            'n_rand_neighbors': n_rand_neighbors,
            'r_values': r_values.astype(np.float32),
            'n_data': n_data, 'n_rand': n_rand}


def add_neighbor_columns_to_tables(data_table, rand_table, stats):
    '''
    Add neighbor statistics columns (N_DATA, N_RAND, R) to the data and random tables
    based on the computed stats.

    Parameters:
        - data_table: Astropy Table containing the data points.
        - rand_table: Astropy Table containing the random points.
        - stats: Dict containing the neighbor statistics computed by compute_neighbor_statistics,
                 including: 'n_data', 'n_rand', 'n_data_neighbors', 'n_rand_neighbors', and 'r_values'.
    Raises:
        - ValueError: If the length of stats['r_values'] does not match the total number of data
                      and random points, which indicates a mismatch in the computed statistics
                      and the input tables.
    '''
    n_data = stats['n_data']
    n_rand = stats['n_rand']
    n_total = n_data + n_rand

    if len(stats['r_values']) != n_total:
        raise ValueError('stats["r_values"] length does not match n_data+n_rand')

    n_data_neighbors = stats['n_data_neighbors']
    n_rand_neighbors = stats['n_rand_neighbors']
    r_values = stats['r_values']

    data_table['N_DATA'] = n_data_neighbors[:n_data]
    data_table['N_RAND'] = n_rand_neighbors[:n_data]
    data_table['R'] = r_values[:n_data]

    rand_table['N_DATA'] = n_data_neighbors[n_data:]
    rand_table['N_RAND'] = n_rand_neighbors[n_data:]
    rand_table['R'] = r_values[n_data:]