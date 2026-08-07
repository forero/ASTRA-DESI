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
import gc
from itertools import combinations
from numbers import Integral, Real

import numpy as np
from scipy.spatial import Delaunay, QhullError
from numba import njit


DEFAULT_R_THRESHOLD = -0.25
DEFAULT_MIN_MEMBERS = 4
UNASSIGNED = -1


@dataclass(frozen=True)
class CSRNeighbors:
    """Compact read-only adjacency lists for a large undirected graph."""

    offsets: np.ndarray
    indices: np.ndarray

    def __post_init__(self):
        offsets = np.asarray(self.offsets)
        indices = np.asarray(self.indices)
        if offsets.ndim != 1 or len(offsets) < 2:
            raise ValueError('CSR offsets must contain at least two values.')
        if indices.ndim != 1:
            raise ValueError('CSR indices must be one-dimensional.')
        if offsets.dtype.kind not in 'iu' or indices.dtype.kind not in 'iu':
            raise TypeError('CSR offsets and indices must contain integers.')
        if int(offsets[0]) != 0 or int(offsets[-1]) != len(indices):
            raise ValueError('CSR offsets do not span the indices array.')
        if np.any(np.diff(offsets) < 0):
            raise ValueError('CSR offsets must be non-decreasing.')
        n_points = len(offsets) - 1
        if len(indices) and (int(np.min(indices)) < 0
                             or int(np.max(indices)) >= n_points):
            raise IndexError('CSR contains a vertex outside the point set.')
        object.__setattr__(self, 'offsets', offsets)
        object.__setattr__(self, 'indices', indices)

    def __len__(self):
        return len(self.offsets) - 1

    def __getitem__(self, point):
        if not isinstance(point, (int, np.integer)):
            raise TypeError('CSRNeighbors indices must be integers.')
        point = int(point)
        if point < 0:
            point += len(self)
        if point < 0 or point >= len(self):
            raise IndexError(point)
        return self.indices[int(self.offsets[point]):int(self.offsets[point + 1])]

    def __iter__(self):
        for point in range(len(self)):
            yield self[point]

    @property
    def n_edges(self):
        return len(self.indices) // 2


def _python_csr_upper_edges(offsets, indices):
    n_points = len(offsets) - 1
    n_edges = sum(np.count_nonzero(
        indices[int(offsets[point]):int(offsets[point + 1])] > point)
        for point in range(n_points))
    edges = np.empty((n_edges, 2), dtype=indices.dtype)
    cursor = 0
    for point in range(n_points):
        row = indices[int(offsets[point]):int(offsets[point + 1])]
        upper = row[row > point]
        stop = cursor + len(upper)
        edges[cursor:stop, 0] = point
        edges[cursor:stop, 1] = upper
        cursor = stop
    return edges


def _python_group_csr(offsets, indices, processing_order):
    group_ids = np.full(len(offsets) - 1, UNASSIGNED, dtype=np.int32)
    next_group_id = 0
    for point_value in processing_order:
        point = int(point_value)
        smallest = np.iinfo(np.int32).max
        for neighbor in indices[int(offsets[point]):int(offsets[point + 1])]:
            group_id = int(group_ids[int(neighbor)])
            if 0 <= group_id < smallest:
                smallest = group_id
        if smallest == np.iinfo(np.int32).max:
            group_ids[point] = next_group_id
            next_group_id += 1
        else:
            group_ids[point] = smallest
    return group_ids, next_group_id


if njit is not None:
    @njit(cache=True)
    def _csr_upper_edges_compiled(offsets, indices):
        n_points = len(offsets) - 1
        n_edges = 0
        for point in range(n_points):
            for cursor in range(offsets[point], offsets[point + 1]):
                if indices[cursor] > point:
                    n_edges += 1
        edges = np.empty((n_edges, 2), dtype=indices.dtype)
        output = 0
        for point in range(n_points):
            for cursor in range(offsets[point], offsets[point + 1]):
                neighbor = indices[cursor]
                if neighbor > point:
                    edges[output, 0] = point
                    edges[output, 1] = neighbor
                    output += 1
        return edges

    @njit(cache=True)
    def _group_csr_compiled(offsets, indices, processing_order):
        group_ids = np.full(len(offsets) - 1, UNASSIGNED,
                            dtype=np.int32)
        next_group_id = 0
        sentinel = np.iinfo(np.int32).max
        for order_index in range(len(processing_order)):
            point = processing_order[order_index]
            smallest = sentinel
            for cursor in range(offsets[point], offsets[point + 1]):
                group_id = group_ids[indices[cursor]]
                if group_id >= 0 and group_id < smallest:
                    smallest = group_id
            if smallest == sentinel:
                group_ids[point] = next_group_id
                next_group_id += 1
            else:
                group_ids[point] = smallest
        return group_ids, next_group_id
else:
    _csr_upper_edges_compiled = _python_csr_upper_edges
    _group_csr_compiled = _python_group_csr


def warmup_accelerators():
    """Compile the two small Numba kernels before worker processes are forked."""
    if njit is None:
        return
    offsets = np.array([0, 1, 2], dtype=np.int32)
    indices = np.array([1, 0], dtype=np.int32)
    _csr_upper_edges_compiled(offsets, indices)
    _group_csr_compiled(offsets, indices, np.array([0, 1], dtype=np.int32))


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
    # A full unique() of tens of millions of 3-D points adds a large sort and
    # several GiB of peak memory.  Keep the explicit diagnostic for small
    # inputs; large inputs are checked by the isolated-vertex test below.
    if len(positions) <= 1_000_000 and len(np.unique(positions, axis=0)) != len(positions):
        raise ValueError('Combined object and random positions must be distinct.')

    try:
        triangulation = Delaunay(positions)
    except QhullError as exc:
        raise ValueError('The combined positions do not define a valid 3-D Delaunay triangulation.') from exc

    offsets, indices = triangulation.vertex_neighbor_vertices
    offsets = np.asarray(offsets)
    indices = np.asarray(indices)
    if offsets.dtype.itemsize > 4 and int(offsets[-1]) <= np.iinfo(np.int32).max:
        offsets = offsets.astype(np.int32)
    if indices.dtype.itemsize > 4 and len(positions) <= np.iinfo(np.int32).max:
        indices = indices.astype(np.int32)
    neighbors = CSRNeighbors(offsets=offsets, indices=indices)
    if np.any(np.diff(neighbors.offsets) == 0):
        raise ValueError('The Delaunay graph contains a point with no neighbours, so r '
                         'is undefined for that point.')

    # The CSR arrays are independent NumPy allocations.  Releasing the
    # triangulation here drops simplices, neighbouring simplices and Qhull
    # bookkeeping before the unique undirected edge list is allocated.
    del triangulation
    gc.collect()
    edges = _csr_upper_edges_compiled(neighbors.offsets, neighbors.indices)

    is_data = np.zeros(len(positions), dtype=bool)
    is_data[:len(objects)] = True

    return DelaunayGraph(positions=positions, is_data=is_data, edges=edges,
                         neighbors=neighbors, n_data=len(objects), n_random=len(randoms))


def _normalized_neighbors(neighbors):

    if isinstance(neighbors, CSRNeighbors):
        if len(neighbors) < 1:
            raise ValueError('neighbors must contain at least one point.')
        isolated = np.flatnonzero(np.diff(neighbors.offsets) == 0)
        if len(isolated):
            raise ValueError('A Delaunay grouping graph cannot contain points '
                             f'without neighbors; indices={isolated.tolist()}.')
        return neighbors

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

    if isinstance(graph, CSRNeighbors):
        degrees = np.diff(graph.offsets).astype(np.int32, copy=False)
        adjacent_is_data = labels[graph.indices]
        n_data_neighbors = np.add.reduceat(
            adjacent_is_data.view(np.uint8), graph.offsets[:-1]).astype(
                np.int32, copy=False)
        del adjacent_is_data
    else:
        n_data_neighbors = np.fromiter(
            (sum(bool(labels[neighbor]) for neighbor in row) for row in graph),
            dtype=np.int64, count=len(graph))
        degrees = np.fromiter((len(row) for row in graph),
                              dtype=np.int64, count=len(graph))
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
    if len(graph) <= np.iinfo(np.int32).max:
        processing_order = processing_order.astype(np.int32, copy=False)

    if isinstance(graph, CSRNeighbors):
        group_ids, next_group_id = _group_csr_compiled(
            graph.offsets, graph.indices, processing_order)
    else:
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

    return VoidGrouping(group_ids=group_ids, group_sizes=group_sizes,
                        threshold_selected=selected,
                        retained=group_ids >= 0,
                        processing_order=processing_order,
                        r_values=r_values)


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