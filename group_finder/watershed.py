from collections import deque

import numpy as np


UNASSIGNED_ID = -1
BOUNDARY_ID = -2


def watershed_grouping(neighbors, r_values, r_threshold, mode='underdense',
                       seed_threshold=None, boundary_id=BOUNDARY_ID):
    '''
    Perform watershed grouping based on the provided neighbors and r_values.

    Parameters:
        - neighbors: List of lists, where neighbors[i] contains the indices of neighboring
                     nodes for node i.
        - r_values: Array of r values for each node, used to determine the order of processing.
        - r_threshold: Growth threshold for r values to select nodes for grouping.
        - mode: 'underdense' to group nodes with r <= r_threshold, 'overdense' to group nodes
                 with r >= r_threshold.
        - seed_threshold: Optional threshold for nodes that are allowed to create new groups.
                          If None, r_threshold is used, matching the original single-threshold
                          behavior.
        - boundary_id: Negative group ID assigned to watershed boundary/saddle nodes.
    Returns:
        - group_of: Array of group IDs for each node, where -1 indicates unassigned nodes
                    and boundary_id indicates watershed boundaries.
    Raises:
        - ValueError: If mode is not 'underdense' or 'overdense', or if the length of r_values
                      does not match the number of nodes.
    '''
    r_values = np.asarray(r_values, dtype=np.float64)
    n_total = len(neighbors)

    if len(r_values) != n_total:
        raise ValueError(f'Length mismatch: len(r_values)={len(r_values)} vs n_nodes={n_total}')

    if boundary_id >= 0:
        raise ValueError(f'boundary_id must be negative, got {boundary_id}')
    if boundary_id == UNASSIGNED_ID:
        raise ValueError(f'boundary_id must be distinct from {UNASSIGNED_ID}')

    grow_threshold = float(r_threshold)
    if seed_threshold is None:
        seed_threshold = grow_threshold
    seed_threshold = float(seed_threshold)

    if mode == 'underdense':
        if seed_threshold > grow_threshold:
            raise ValueError('For underdense mode, seed_threshold must be <= r_threshold')
        selected = np.where(r_values <= grow_threshold)[0]
        can_seed = r_values <= seed_threshold
        order = selected[np.argsort(r_values[selected], kind='stable')]
    elif mode == 'overdense':
        if seed_threshold < grow_threshold:
            raise ValueError('For overdense mode, seed_threshold must be >= r_threshold')
        selected = np.where(r_values >= grow_threshold)[0]
        can_seed = r_values >= seed_threshold
        order = selected[np.argsort(r_values[selected], kind='stable')[::-1]]
    else:
        raise ValueError(f'mode must be "underdense" or "overdense", got "{mode}"')

    group_of = np.full(n_total, UNASSIGNED_ID, dtype=np.int32)
    if len(order) == 0:
        return group_of
    selected_mask = np.zeros(n_total, dtype=bool)
    selected_mask[selected] = True

    current_max = -1

    for node in order:
        node_idx = int(node)
        neighbor_groups = set()
        for nbr in neighbors[node_idx]:
            gid = group_of[nbr]
            if gid >= 0:
                neighbor_groups.add(int(gid))
                if len(neighbor_groups) > 1:
                    break

        if len(neighbor_groups) == 0:
            if not can_seed[node_idx]:
                continue
            current_max += 1
            group_of[node_idx] = current_max
        elif len(neighbor_groups) == 1:
            group_of[node_idx] = next(iter(neighbor_groups))
        else:
            group_of[node_idx] = boundary_id

    pending = np.flatnonzero(selected_mask & (group_of == UNASSIGNED_ID))
    if len(pending) == 0:
        return group_of

    queue = deque(int(node) for node in pending)
    in_queue = np.zeros(n_total, dtype=bool)
    in_queue[pending] = True

    while queue:
        node_idx = queue.popleft()
        in_queue[node_idx] = False

        if group_of[node_idx] != UNASSIGNED_ID:
            continue

        neighbor_groups = set()
        for nbr in neighbors[node_idx]:
            gid = group_of[nbr]
            if gid >= 0:
                neighbor_groups.add(int(gid))
                if len(neighbor_groups) > 1:
                    break

        if len(neighbor_groups) == 0:
            continue
        if len(neighbor_groups) == 1:
            group_of[node_idx] = next(iter(neighbor_groups))
        else:
            group_of[node_idx] = boundary_id

        for nbr in neighbors[node_idx]:
            nbr_idx = int(nbr)
            if (selected_mask[nbr_idx] and
                    group_of[nbr_idx] == UNASSIGNED_ID and
                    not in_queue[nbr_idx]):
                queue.append(nbr_idx)
                in_queue[nbr_idx] = True

    return group_of


def filter_groups_by_size(group_of, min_group_size=4):
    '''
    Filter groups based on their size and return the filtered group assignments and group sizes.

    Parameters:
        - group_of: Array of group IDs for each node, where -1 indicates unassigned nodes
                    and negative values below -1 are preserved as non-member labels.
        - min_group_size: Minimum size of groups to keep. Groups smaller than this will be
                          set to -1 (unassigned).
    Returns:
        - filtered: Array of group IDs after filtering, where -1 indicates unassigned nodes
                    and watershed boundary labels remain negative.
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
                  mode='underdense', seed_threshold=None,
                  boundary_id=BOUNDARY_ID):
    '''
    Run the watershed grouping algorithm and filter groups by size.

    Parameters:
        - neighbors: Array of neighbor indices for each node.
        - r_values: Array of r-values for each node.
        - r_threshold: Growth threshold for determining group membership.
        - min_group_size: Minimum size of groups to keep.
        - mode: Mode for grouping ('underdense' or 'overdense').
        - seed_threshold: Optional threshold for nodes that can create new groups.
        - boundary_id: Negative group ID used for watershed boundaries.
    Returns:
        - Dict containing the filtered group assignments and summary statistics.
    '''
    raw_groups = watershed_grouping(neighbors=neighbors, r_values=r_values,
                                    r_threshold=r_threshold, mode=mode,
                                    seed_threshold=seed_threshold,
                                    boundary_id=boundary_id)
    filtered_groups, group_sizes = filter_groups_by_size(raw_groups, min_group_size=min_group_size)

    n_assigned = int(np.count_nonzero(filtered_groups >= 0))
    n_boundary = int(np.count_nonzero(filtered_groups == boundary_id))
    n_unassigned = int(np.count_nonzero(filtered_groups == UNASSIGNED_ID))
    return {'group_of': filtered_groups,
            'group_sizes': group_sizes,
            'n_groups': len(group_sizes),
            'n_assigned': n_assigned,
            'n_boundary_nodes': n_boundary,
            'n_unassigned': n_unassigned,
            'boundary_id': int(boundary_id),
            'seed_threshold': seed_threshold}


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


def _fix_eigenvector_signs(eigenvectors):
    '''
    Apply a deterministic sign convention to eigenvectors.

    Eigenvectors are directions, so v and -v are mathematically equivalent. For
    catalog output, choose the sign where the largest absolute component is
    positive so reruns do not flip signs arbitrarily for the same tensor.
    '''
    vectors = np.asarray(eigenvectors, dtype=np.float64).copy()
    for j in range(vectors.shape[1]):
        col = vectors[:, j]
        anchor = int(np.argmax(np.abs(col)))
        if np.isfinite(col[anchor]) and col[anchor] < 0.0:
            vectors[:, j] *= -1.0
    return vectors


def compute_semi_axes(x_members, y_members, z_members, x_cm, y_cm, z_cm,
                      return_vectors=False):
    '''
    Compute principal semi-axes from the inertia tensor of group members.

    Parameters:
        - x_members, y_members, z_members: Lists of x, y, z coordinates of group members.
        - x_cm, y_cm, z_cm: Coordinates of the center of-mass of the group.
        - return_vectors: If True, also return the unit eigenvectors associated
                          with the sorted semi-axes.
    Returns:
        - semi_axes: Array of the three principal semi-axes lengths, sorted in descending order.
        - axis_vectors: Optional array with shape (3, 3). Column j is the
                        Cartesian unit vector for semi_axes[j].
    '''
    dx = np.asarray(x_members, dtype=np.float64) - x_cm
    dy = np.asarray(y_members, dtype=np.float64) - y_cm
    dz = np.asarray(z_members, dtype=np.float64) - z_cm

    n = len(dx)
    if n == 0:
        semi_axes = np.array([np.nan, np.nan, np.nan], dtype=np.float64)
        if return_vectors:
            return semi_axes, np.full((3, 3), np.nan, dtype=np.float64)
        return semi_axes

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

    eigenvalues, eigenvectors = np.linalg.eigh(inertia)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = _fix_eigenvector_signs(eigenvectors[:, order])
    eigenvalues = np.clip(eigenvalues, 0.0, None)

    semi_axes = np.sqrt(eigenvalues / n)
    if return_vectors:
        return semi_axes, eigenvectors
    return semi_axes