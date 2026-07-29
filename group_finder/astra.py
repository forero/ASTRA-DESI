"""
1. combine object and random Cartesian positions;
2. construct their three-dimensional Delaunay graph;
3. count data and random neighbours and compute ``r``;
4. optionally average already aligned per-point ``r`` realizations;
5. retain points with ``r <= r_threshold``;
6. process them in ascending ``r`` order;
7. assign each point to the lowest-indexed existing group that it touches, or
   seed a group when no assigned neighbour exists; and
8. discard groups smaller than ``min_members``.
"""

from dataclasses import dataclass
from itertools import combinations
from numbers import Integral, Real

import numpy as np
from scipy.spatial import Delaunay, QhullError


DEFAULT_R_THRESHOLD = -0.25
DEFAULT_MIN_MEMBERS = 4
UNASSIGNED = -1


@dataclass(frozen=True)
class DelaunayGraph:
    positions: np.ndarray
    is_data: np.ndarray
    edges: np.ndarray
    neighbors: tuple[tuple[int, ...], ...]
    n_data: int
    n_random: int


@dataclass(frozen=True)
class DensityContrast:
    n_data_neighbors: np.ndarray
    n_random_neighbors: np.ndarray
    r_values: np.ndarray


@dataclass(frozen=True)
class VoidGrouping:
    group_ids: np.ndarray
    group_sizes: dict[int, int]
    threshold_selected: np.ndarray
    retained: np.ndarray
    processing_order: np.ndarray
    r_values: np.ndarray


@dataclass(frozen=True)
class GroupFinderResult:
    graph: DelaunayGraph
    contrast: DensityContrast
    grouping: VoidGrouping


def _positions_array(values, name):
    positions = np.asarray(values, dtype=np.float64)
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError(f'{name} must have shape (n_points, 3).')
    if len(positions) == 0:
        raise ValueError(f'{name} must contain at least one point.')
    if not np.all(np.isfinite(positions)):
        raise ValueError(f'{name} contains non-finite coordinates.')
    return positions


def _positive_integer(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError(f'{name} must be a positive integer.')
    value = int(value)
    if value < 1:
        raise ValueError(f'{name} must be a positive integer.')
    return value


def _finite_unit_interval_number(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f'{name} must be a real number within [-1, 1].')
    value = float(value)
    if not np.isfinite(value) or not -1.0 <= value <= 1.0:
        raise ValueError(f'{name} must lie within [-1, 1].')
    return value


def neighbors_from_edges(n_points, edges):

    n_points = _positive_integer(n_points, 'n_points')

    edge_array = np.asarray(edges)
    if edge_array.size == 0:
        return tuple(() for _ in range(n_points))
    if edge_array.ndim != 2 or edge_array.shape[1] != 2:
        raise ValueError('edges must have shape (n_edges, 2).')
    if not np.issubdtype(edge_array.dtype, np.integer):
        raise TypeError('edges must contain integer vertex indices.')

    edge_array = edge_array.astype(np.int64, copy=False)
    if np.min(edge_array) < 0 or np.max(edge_array) >= n_points:
        raise IndexError('edges contains a vertex outside the point set.')
    if np.any(edge_array[:, 0] == edge_array[:, 1]):
        raise ValueError('Delaunay graph edges cannot be self-loops.')

    edge_array = np.sort(edge_array, axis=1)
    edge_array = np.unique(edge_array, axis=0)
    adjacency = [set() for _ in range(n_points)]
    for first, second in edge_array:
        first = int(first)
        second = int(second)
        adjacency[first].add(second)
        adjacency[second].add(first)
    return tuple(tuple(sorted(row)) for row in adjacency)


def _unique_delaunay_edges(simplices):
    vertex_pairs = tuple(combinations(range(4), 2))
    edges = np.concatenate([simplices[:, pair] for pair in vertex_pairs], axis=0)
    edges = np.sort(edges.astype(np.int64, copy=False), axis=1)
    return np.unique(edges, axis=0)


def build_delaunay_graph(object_positions, random_positions) -> DelaunayGraph:

    objects = _positions_array(object_positions, 'object_positions')
    randoms = _positions_array(random_positions, 'random_positions')
    positions = np.concatenate((objects, randoms), axis=0)
    if len(positions) < 4:
        raise ValueError('A three-dimensional Delaunay triangulation needs at least four combined points.')
    if len(np.unique(positions, axis=0)) != len(positions):
        raise ValueError('Combined object and random positions must be distinct.')

    try:
        triangulation = Delaunay(positions)
    except QhullError as exc:
        raise ValueError('The combined positions do not define a valid 3-D Delaunay triangulation.') from exc

    edges = _unique_delaunay_edges(triangulation.simplices)
    neighbors = neighbors_from_edges(len(positions), edges)
    if any(len(row) == 0 for row in neighbors):
        raise ValueError('The Delaunay graph contains a point with no neighbours, so r '
                         'is undefined for that point.')

    is_data = np.zeros(len(positions), dtype=bool)
    is_data[:len(objects)] = True

    return DelaunayGraph(positions=positions, is_data=is_data, edges=edges,
                         neighbors=neighbors, n_data=len(objects), n_random=len(randoms))


def _normalized_neighbors(neighbors):

    n_points = len(neighbors)
    if n_points < 1:
        raise ValueError('neighbors must contain at least one point.')
    normalized = []
    for point, row in enumerate(neighbors):
        array = np.asarray(tuple(row))
        if array.size == 0:
            normalized.append(())
            continue
        if array.ndim != 1 or not np.issubdtype(array.dtype, np.integer):
            raise TypeError('neighbors must contain one-dimensional integers.')
        array = np.unique(array.astype(np.int64, copy=False))
        if np.min(array) < 0 or np.max(array) >= n_points:
            raise IndexError(
                f'neighbors[{point}] contains an index outside the graph.')
        if np.any(array == point):
            raise ValueError('neighbors cannot contain self-neighbours.')
        normalized.append(tuple(int(value) for value in array))

    normalized = tuple(normalized)
    for point, row in enumerate(normalized):
        for other in row:
            if point not in normalized[other]:
                raise ValueError('neighbors must describe an undirected graph.')
    isolated = [point for point, row in enumerate(normalized) if not row]
    if isolated:
        raise ValueError(f'A Delaunay grouping graph cannot contain points without neighbors; indices={isolated}.')
    return normalized


def compute_density_contrast(neighbors, is_data) -> DensityContrast:

    graph = _normalized_neighbors(neighbors)
    labels = np.asarray(is_data)
    if labels.ndim != 1 or len(labels) != len(graph):
        raise ValueError('is_data must be one-dimensional and match the graph size.')
    if labels.dtype.kind != 'b':
        raise TypeError('is_data must have boolean dtype.')

    n_data_neighbors = np.fromiter((sum(bool(labels[neighbor]) for neighbor in row) for row in graph),
                                   dtype=np.int64, count=len(graph))
    degrees = np.fromiter((len(row) for row in graph), dtype=np.int64, count=len(graph))
    if np.any(degrees == 0):
        indices = np.flatnonzero(degrees == 0)
        raise ValueError(f'r is undefined for points without Delaunay neighbours - indices={indices.tolist()}.')

    n_random_neighbors = degrees - n_data_neighbors
    r_values = ((n_data_neighbors - n_random_neighbors) / degrees.astype(np.float64))
    return DensityContrast(n_data_neighbors=n_data_neighbors, n_random_neighbors=n_random_neighbors,
                           r_values=r_values)


def average_r_values(r_values_by_realization, point_ids_by_realization=None):

    values = np.asarray(r_values_by_realization, dtype=np.float64)
    if values.ndim == 1:
        values = values[None, :]
    if values.ndim != 2 or values.shape[0] < 1 or values.shape[1] < 1:
        raise ValueError('r_values_by_realization must have shape (n_realizations, n_points).')
    if not np.all(np.isfinite(values)):
        raise ValueError('Every aligned r value must be finite.')
    if np.any((values < -1.0) | (values > 1.0)):
        raise ValueError('r values must lie within [-1, 1].')

    if point_ids_by_realization is not None:
        point_ids = np.asarray(point_ids_by_realization)
        if point_ids.shape != values.shape:
            raise ValueError('point_ids_by_realization must match the r-value shape.')
        if len(np.unique(point_ids[0])) != values.shape[1]:
            raise ValueError('Point IDs must be unique within each realization.')
        if not np.all(point_ids == point_ids[0][None, :]):
            raise ValueError('Point IDs must be identically aligned in every realization.')
    elif values.shape[0] > 1:
        raise ValueError('Multiple realizations require identically aligned point_ids_by_realization.')
    return np.mean(values, axis=0)


def group_void_points(neighbors, r_values, r_threshold = DEFAULT_R_THRESHOLD, min_members = DEFAULT_MIN_MEMBERS) -> VoidGrouping:

    graph = _normalized_neighbors(neighbors)
    r_values = np.asarray(r_values, dtype=np.float64)
    if r_values.ndim != 1 or len(r_values) != len(graph):
        raise ValueError('r_values must be one-dimensional and match the graph size.')
    if not np.all(np.isfinite(r_values)):
        raise ValueError('r_values must be finite.')
    if np.any((r_values < -1.0) | (r_values > 1.0)):
        raise ValueError('r_values must lie within [-1, 1].')

    r_threshold = _finite_unit_interval_number(r_threshold, 'r_threshold')
    min_members = _positive_integer(min_members, 'min_members')

    selected = r_values <= r_threshold
    selected_indices = np.flatnonzero(selected)
    stable_order = np.argsort(r_values[selected_indices], kind='stable')
    processing_order = selected_indices[stable_order]

    group_ids = np.full(len(graph), UNASSIGNED, dtype=np.int64)
    next_group_id = 0
    for point in processing_order:
        point = int(point)
        neighboring_groups = tuple(sorted({int(group_ids[neighbor]) for neighbor in graph[point]
                                           if group_ids[neighbor] >= 0}))
        if not neighboring_groups:
            group_ids[point] = next_group_id
            next_group_id += 1
        else:
            group_ids[point] = min(neighboring_groups)

    assigned = group_ids >= 0
    if np.any(assigned):
        sizes = np.bincount(group_ids[assigned], minlength=next_group_id)
        discard = sizes < min_members
        assigned_indices = np.flatnonzero(assigned)
        discard_members = discard[group_ids[assigned_indices]]
        group_ids[assigned_indices[discard_members]] = UNASSIGNED

    retained = group_ids >= 0
    if np.any(retained):
        retained_sizes = np.bincount(group_ids[retained], minlength=next_group_id)
        group_sizes = {int(group_id): int(size) for group_id, size in enumerate(retained_sizes)
                       if size > 0}
    else:
        group_sizes = {}

    return VoidGrouping(group_ids=group_ids, group_sizes=group_sizes, threshold_selected=selected,
                        retained=group_ids >= 0, processing_order=processing_order, r_values=r_values.copy())


def group_aligned_realizations(neighbors, r_values_by_realization, point_ids_by_realization=None,
                               graph_point_ids=None, r_threshold = DEFAULT_R_THRESHOLD,
                               min_members = DEFAULT_MIN_MEMBERS) -> VoidGrouping:

    realization_values = np.asarray(r_values_by_realization, dtype=np.float64)
    n_realizations = (1 if realization_values.ndim == 1
                      else realization_values.shape[0]
                      if realization_values.ndim == 2
                      else 0)
    averaged = average_r_values(realization_values, point_ids_by_realization=point_ids_by_realization)

    if graph_point_ids is not None:
        graph_ids = np.asarray(graph_point_ids)
        if graph_ids.ndim != 1 or len(graph_ids) != len(averaged):
            raise ValueError('graph_point_ids must be one-dimensional and match the grouping graph size.')
        if point_ids_by_realization is not None:
            aligned_ids = np.asarray(point_ids_by_realization)
            if aligned_ids.ndim == 1:
                aligned_ids = aligned_ids[None, :]
            if not np.all(graph_ids == aligned_ids[0]):
                raise ValueError('graph_point_ids must match the aligned point-ID order.')

    elif n_realizations > 1:
        raise ValueError('Multiple realizations require graph_point_ids so each averaged '
                         'column is tied to the corresponding graph vertex.')
    return group_void_points(neighbors=neighbors, r_values=averaged,
                             r_threshold=r_threshold, min_members=min_members)


def run_group_finder(object_positions, random_positions, r_threshold = DEFAULT_R_THRESHOLD,
                     min_members = DEFAULT_MIN_MEMBERS) -> GroupFinderResult:

    graph = build_delaunay_graph(object_positions, random_positions)
    contrast = compute_density_contrast(neighbors=graph.neighbors, is_data=graph.is_data)
    grouping = group_void_points(neighbors=graph.neighbors, r_values=contrast.r_values,
                                 r_threshold=r_threshold, min_members=min_members)
    return GroupFinderResult(graph=graph, contrast=contrast, grouping=grouping)