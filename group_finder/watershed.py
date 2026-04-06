import numpy as np


def watershed_grouping(neighbors, r_values, r_threshold, mode='underdense'):
    '''
    Perform watershed grouping based on the provided neighbors and r_values.

    Parameters:
        - neighbors: List of lists, where neighbors[i] contains the indices of neighboring
                     nodes for node i.
        - r_values: Array of r values for each node, used to determine the order of processing.
        - r_threshold: Threshold for r values to select nodes for grouping.
        - mode: 'underdense' to group nodes with r <= r_threshold, 'overdense' to group nodes
                 with r >= r_threshold.
    Returns:
        - group_of: Array of group IDs for each node, where -1 indicates unassigned nodes.
    Raises:
        - ValueError: If mode is not 'underdense' or 'overdense', or if the length of r_values
                      does not match the number of nodes.
    '''
    r_values = np.asarray(r_values, dtype=np.float64)
    n_total = len(neighbors)

    if len(r_values) != n_total:
        raise ValueError(f'Length mismatch: len(r_values)={len(r_values)} vs n_nodes={n_total}')

    if mode == 'underdense':
        selected = np.where(r_values <= r_threshold)[0]
        order = selected[np.argsort(r_values[selected], kind='stable')]
    elif mode == 'overdense':
        selected = np.where(r_values >= r_threshold)[0]
        order = selected[np.argsort(r_values[selected], kind='stable')[::-1]]
    else:
        raise ValueError(f'mode must be "underdense" or "overdense", got "{mode}"')

    group_of = np.full(n_total, -1, dtype=np.int32)
    if len(order) == 0:
        return group_of

    current_max = -1

    for node in order:
        node_idx = int(node)
        min_gid = -1
        for nbr in neighbors[node_idx]:
            gid = group_of[nbr]
            if gid >= 0 and (min_gid < 0 or gid < min_gid):
                min_gid = int(gid)
                if min_gid == 0:
                    break
        if min_gid >= 0:
            group_of[node_idx] = min_gid
        else:
            current_max += 1
            group_of[node_idx] = current_max

    return group_of


def filter_groups_by_size(group_of, min_group_size=4):
    '''
    Filter groups based on their size and return the filtered group assignments and group sizes.

    Parameters:
        - group_of: Array of group IDs for each node, where -1 indicates unassigned nodes.
        - min_group_size: Minimum size of groups to keep. Groups smaller than this will be
                          set to -1 (unassigned).
    Returns:
        - filtered: Array of group IDs after filtering, where -1 indicates unassigned nodes.
        - group_sizes: Dict mapping group ID to its size for groups that meet the
                       size threshold.
    '''
    if min_group_size <= 1:
        valid = group_of >= 0
        if np.any(valid):
            sizes = np.bincount(group_of[valid])
            group_sizes = {gid: int(size) for gid, size in enumerate(sizes) if size > 0}
        else:
            group_sizes = {}
        return group_of.astype(np.int32, copy=True), group_sizes

    filtered = group_of.astype(np.int32, copy=True)
    assigned = filtered >= 0

    if not np.any(assigned):
        return filtered, {}

    sizes = np.bincount(filtered[assigned])
    keep_group = sizes >= min_group_size
    assigned_idx = np.flatnonzero(assigned)
    keep_assigned = keep_group[filtered[assigned_idx]]
    if np.any(~keep_assigned):
        filtered[assigned_idx[~keep_assigned]] = -1

    assigned_filtered = filtered >= 0
    if np.any(assigned_filtered):
        final_sizes = np.bincount(filtered[assigned_filtered])
        group_sizes = {gid: int(size) for gid, size in enumerate(final_sizes) if size > 0}
    else:
        group_sizes = {}

    return filtered, group_sizes


def run_watershed(neighbors, r_values, r_threshold=-0.3, min_group_size=4,
                  mode='underdense'):
    '''
    Run the watershed grouping algorithm and filter groups by size.

    Parameters:
        - neighbors: Array of neighbor indices for each node.
        - r_values: Array of r-values for each node.
        - r_threshold: Threshold for determining group membership.
        - min_group_size: Minimum size of groups to keep.
        - mode: Mode for grouping ('underdense' or 'overdense').
    Returns:
        - Dict containing the filtered group assignments and summary statistics.
    '''
    raw_groups = watershed_grouping(neighbors=neighbors, r_values=r_values,
                                    r_threshold=r_threshold, mode=mode)
    filtered_groups, group_sizes = filter_groups_by_size(raw_groups, min_group_size=min_group_size)

    n_assigned = int(np.count_nonzero(filtered_groups >= 0))
    return {'group_of': filtered_groups,
            'group_sizes': group_sizes,
            'n_groups': len(group_sizes),
            'n_assigned': n_assigned}


def assign_group_ids_to_tables(data_table, rand_table, group_of, group_col='GROUPID'):
    '''
    Assign group IDs to data and random tables.

    Parameters:
        - data_table: DataFrame containing the data nodes.
        - rand_table: DataFrame containing the random nodes.
        - group_of: Array of group IDs for each node.
        - group_col: Name of the column to store group IDs.
    '''
    n_data = len(data_table)
    n_rand = len(rand_table)

    if len(group_of) != (n_data + n_rand):
        raise ValueError(f'group_of length mismatch: got {len(group_of)}, expected {n_data + n_rand}')

    data_table[group_col] = np.asarray(group_of[:n_data], dtype=np.int32)
    rand_table[group_col] = np.asarray(group_of[n_data:], dtype=np.int32)


def compute_semi_axes(x_members, y_members, z_members, x_cm, y_cm, z_cm):
    '''
    Compute principal semi-axes from the inertia tensor of group members.

    Parameters:
        - x_members, y_members, z_members: Lists of x, y, z coordinates of group members.
        - x_cm, y_cm, z_cm: Coordinates of the center of-mass of the group.
    Returns:
        - semi_axes: Array of the three principal semi-axes lengths, sorted in descending order.
    '''
    dx = np.asarray(x_members, dtype=np.float64) - x_cm
    dy = np.asarray(y_members, dtype=np.float64) - y_cm
    dz = np.asarray(z_members, dtype=np.float64) - z_cm

    n = len(dx)
    if n == 0:
        return np.array([np.nan, np.nan, np.nan], dtype=np.float64)

    r2 = dx * dx + dy * dy + dz * dz

    i_xx = np.sum(r2 - dx * dx)
    i_yy = np.sum(r2 - dy * dy)
    i_zz = np.sum(r2 - dz * dz)
    i_xy = np.sum(-dx * dy)
    i_xz = np.sum(-dx * dz)
    i_yz = np.sum(-dy * dz)

    inertia = np.array([[i_xx, i_xy, i_xz],
                        [i_xy, i_yy, i_yz],
                        [i_xz, i_yz, i_zz]], dtype=np.float64)

    eigenvalues = np.linalg.eigvalsh(inertia)
    eigenvalues = np.sort(eigenvalues)[::-1]
    eigenvalues = np.clip(eigenvalues, 0.0, None)

    semi_axes = np.sqrt(eigenvalues / n)
    return semi_axes