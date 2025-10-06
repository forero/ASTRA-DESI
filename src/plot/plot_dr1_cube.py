import argparse
import json
import os
from pathlib import Path

import numpy as np
from astropy.table import Table, join

try:
    import pyvista as pv
except ImportError:
    pv = None

try:
    from scipy.ndimage import gaussian_filter
except ImportError:
    gaussian_filter = None

try:
    from scipy.spatial import cKDTree
except ImportError:
    cKDTree = None

try:
    import imageio_ffmpeg
    HAS_FFMPEG = True
except ImportError:
    HAS_FFMPEG = False

DEFAULT_BASE_DIR = Path("/pscratch/sd/v/vtorresg/cosmic-web/dr1")
DEFAULT_MOVIE = Path("dr1_cube.mp4")
SUPPORTED_MODES = ("points", "volume", "graph")
TRACER_NAME = "BGS"
PROB_TEMPLATE = "probabilities/zone_{zone}_probability.fits.gz"
WEBTYPE_LABELS = ("void", "sheet", "filament", "knot")


def _find_column_name(table, base):
    for suffix in ("", "_1", "_2", "_raw", "_prob"):
        name = f"{base}{suffix}" if suffix else base
        if name in table.colnames:
            return name
    raise KeyError()


def _normalize_tracer(value):
    text = str(value).strip().upper()
    if not text:
        return ""
    core = text.split("_", 1)[0]
    return core


def _classify_webtypes(table):
    prob_cols = [_find_column_name(table, "PVOID"),
                 _find_column_name(table, "PSHEET"),
                 _find_column_name(table, "PFILAMENT"),
                 _find_column_name(table, "PKNOT")]
    probs = np.vstack([np.asarray(table[col], dtype=float) for col in prob_cols]).T
    finite = np.isfinite(probs).any(axis=1)
    safe = np.where(np.isfinite(probs), probs, -np.inf)
    idx = np.argmax(safe, axis=1)
    labels = np.array(WEBTYPE_LABELS, dtype="U8")[idx]
    labels[~finite] = ""
    return labels


def _load_xyz(base_dir, zone, tracer, webtype, groups_path, with_groups=False):
    raw_path = base_dir / "raw" / f"zone_{zone}.fits.gz"
    if not raw_path.exists():
        raise FileNotFoundError()
    table = Table.read(raw_path)

    tracer_col = _find_column_name(table, "TRACERTYPE")
    tracers = np.array([_normalize_tracer(val) for val in table[tracer_col]], dtype="U8")
    tracer_mask = tracers == tracer.upper()
    if not np.any(tracer_mask):
        raise ValueError()
    table = table[tracer_mask]

    try:
        randiter_col = _find_column_name(table, "RANDITER")
    except KeyError:
        randiter_col = None
    if randiter_col:
        data_mask = np.asarray(table[randiter_col]) == -1
        if np.any(data_mask):
            table = table[data_mask]

    working = table
    webtype_norm = None
    if webtype is not None:
        prob_path = base_dir / PROB_TEMPLATE.format(zone=zone)
        if not prob_path.exists():
            raise FileNotFoundError()

        prob_table = Table.read(prob_path)
        working = join(working, prob_table, keys="TARGETID", join_type="inner")

        randiter_col = _find_column_name(working, "RANDITER")
        data_mask = np.asarray(working[randiter_col]) == -1
        if not np.any(data_mask):
            raise ValueError()
        working = working[data_mask]

        labels = _classify_webtypes(working)
        webtype_norm = webtype.strip().lower()
        if webtype_norm not in WEBTYPE_LABELS:
            raise ValueError()
        class_mask = labels == webtype_norm
        if not np.any(class_mask):
            raise ValueError()
        working = working[class_mask]

    groups_arr = None
    if with_groups:
        inferred = "filament" if webtype_norm is None else webtype_norm
        default_groups_path = base_dir / "groups" / f"zone_{zone}_groups_fof_{inferred}.fits.gz"
        target_groups_path = groups_path or default_groups_path
        if not target_groups_path.exists():
            raise FileNotFoundError()
        groups_table = Table.read(target_groups_path)
        target_col = _find_column_name(groups_table, "TARGETID")
        group_col = _find_column_name(groups_table, "GROUPID")
        groups_table = groups_table[[target_col, group_col]]
        groups_table.rename_columns([target_col, group_col], ["TARGETID", "GROUPID"])
        working = join(working, groups_table, keys="TARGETID", join_type="inner")

        tracer_col_joined = _find_column_name(working, "TRACERTYPE")
        tracers_joined = np.array([_normalize_tracer(val) for val in working[tracer_col_joined]], dtype="U8")
        tracer_mask_joined = tracers_joined == tracer.upper()
        if not np.any(tracer_mask_joined):
            raise ValueError()
        working = working[tracer_mask_joined]
        groups_arr = np.asarray(working["GROUPID"], dtype=np.int64)

    x = np.asarray(working[_find_column_name(working, "XCART")], dtype=float)
    y = np.asarray(working[_find_column_name(working, "YCART")], dtype=float)
    z = np.asarray(working[_find_column_name(working, "ZCART")], dtype=float)
    target_ids = np.asarray(working[_find_column_name(working, "TARGETID")], dtype=np.int64)
    return x, y, z, target_ids, groups_arr


def _select_cube(x, y, z, side, center, extras=None,):
    if center is None:
        cx, cy, cz = (np.median(arr) for arr in (x, y, z))
    else:
        cx, cy, cz = center
    half = 0.5 * side

    mask = ((x >= cx - half) & (x <= cx + half) &
            (y >= cy - half) & (y <= cy + half) &
            (z >= cz - half) & (z <= cz + half))
    if not np.any(mask):
        raise ValueError()

    extras_sel = None
    if extras:
        extras_sel = tuple(arr[mask] for arr in extras)

    return x[mask] - cx, y[mask] - cy, z[mask] - cz, (cx, cy, cz), extras_sel


def _build_histogram(x, y, z, side, grid, sigma):
    half = 0.5 * side
    edges = np.linspace(-half, half, grid + 1)
    hist, _ = np.histogramdd(np.column_stack((x, y, z)), bins=(edges, edges, edges))
    if sigma and sigma > 0:
        if gaussian_filter is None:
            raise RuntimeError()
        hist = gaussian_filter(hist, sigma=sigma, mode="constant", cval=0.0)
    return hist


def _volume_plotter(rho, side, iso_fracs, color, cmap, opacity, lighting):
    grid_cls = getattr(pv, "UniformGrid", None) or getattr(pv, "ImageData", None)
    if grid_cls is None:
        raise RuntimeError()

    dims = np.array(rho.shape, dtype=int)
    if dims.size != 3:
        raise ValueError()

    grid = grid_cls()
    grid.dimensions = dims + 1
    spacing = side / dims.astype(float)
    grid.spacing = tuple(spacing)
    grid.origin = (-0.5 * side, -0.5 * side, -0.5 * side)
    grid.cell_data["rho"] = rho.ravel(order="F")

    dataset = grid
    if "rho" not in dataset.point_data:
        dataset = dataset.cell_data_to_point_data()

    scalar_min, scalar_max = dataset.get_data_range("rho")
    if not np.isfinite(scalar_max) or scalar_max <= 0:
        raise ValueError()

    levels = [frac * scalar_max if 0 < frac <= 1 else frac for frac in iso_fracs]

    mesh = dataset.contour(isosurfaces=levels, scalars="rho")
    if mesh.n_points == 0:
        raise ValueError()
    plotter = pv.Plotter(off_screen=True, lighting=lighting)
    constant_color = color is not None and color.lower() not in {"", "auto", "density"}
    if constant_color:
        plotter.add_mesh(mesh,
                         color=color,
                         opacity=opacity,
                         show_scalar_bar=False)
    else:
        plotter.add_mesh(mesh,
                         opacity=opacity,
                         cmap=cmap,
                         clim=[scalar_min, scalar_max],
                         show_scalar_bar=False)
    plotter.add_axes(line_width=1, labels_off=False)
    plotter.set_background("black")

    for actor in plotter.renderer.GetActors():
        prop = actor.GetProperty()
        line_width = prop.GetLineWidth()
        if line_width and line_width < 0.5:
            prop.SetLineWidth(0.5)

    return plotter


def _points_plotter(x, y, z, point_size, color, scalars=None, cmap=None):
    data = np.column_stack((x, y, z))
    cloud = pv.PolyData(data)
    if scalars is not None:
        cloud.point_data["group"] = scalars
    plotter = pv.Plotter(off_screen=True)
    mesh_kwargs = dict(render_points_as_spheres=True, point_size=point_size, show_scalar_bar=False)
    if scalars is not None:
        plotter.add_mesh(cloud, scalars="group", cmap=cmap or "tab20", **mesh_kwargs)
    else:
        plotter.add_mesh(cloud, color=color, **mesh_kwargs)
    plotter.add_axes(line_width=1, labels_off=False)
    plotter.set_background("black")
    return plotter


def _build_graph_lines(points, neighbors, radius, max_length, context="vecinos"):
    if cKDTree is None:
        raise RuntimeError()

    if points.shape[0] < 2:
        raise ValueError("At least two points are required to build graph lines.")

    tree = cKDTree(points)
    pairs = set()

    use_radius = radius is not None and radius > 0
    if use_radius:
        for i, pt in enumerate(points):
            indices = tree.query_ball_point(pt, r=radius)
            for j in indices:
                if j <= i:
                    continue
                pairs.add((i, j))
    else:
        k = max(1, neighbors)
        dists, idxs = tree.query(points, k=k + 1)
        # idxs shape (N, k+1)... include self at idxs[:,0]
        for i, row in enumerate(idxs):
            for j in row[1:]:
                if j < 0:
                    continue
                if j <= i:
                    continue
                pairs.add((i, int(j)))

    if not pairs:
        raise ValueError("No valid pairs found.")

    pairs_arr = np.array(sorted(pairs), dtype=np.int64)
    pairs_arr, limit_used, removed = _filter_pairs_by_length(pairs_arr, points, max_length, context)
    if removed > 0 and (max_length is None or max_length <= 0):
        print(f"Discarded {removed} long edges (> {limit_used:.1f} Mpc) generated by {context}.")

    lines = np.empty(pairs_arr.shape[0] * 3, dtype=np.int64)
    lines[0::3] = 2
    lines[1::3] = pairs_arr[:, 0]
    lines[2::3] = pairs_arr[:, 1]
    return lines, limit_used


def _build_lines_from_pairs(target_ids, groups, pairs_path, allow_cross_groups, points, max_length):
    if not pairs_path.exists():
        raise FileNotFoundError()

    pairs_table = Table.read(pairs_path)
    tid1_col = _find_column_name(pairs_table, "TARGETID1")
    tid2_col = _find_column_name(pairs_table, "TARGETID2")
    tid1 = np.asarray(pairs_table[tid1_col], dtype=np.int64)
    tid2 = np.asarray(pairs_table[tid2_col], dtype=np.int64)

    try:
        randiter_col = _find_column_name(pairs_table, "RANDITER")
    except KeyError:
        randiter_col = None

    if randiter_col is not None:
        rand_mask = np.asarray(pairs_table[randiter_col]) == -1
        if np.any(rand_mask):
            tid1 = tid1[rand_mask]
            tid2 = tid2[rand_mask]

    index_map = {tid: idx for idx, tid in enumerate(target_ids)}
    pairs = set()

    for a, b in zip(tid1, tid2):
        ia = index_map.get(int(a))
        ib = index_map.get(int(b))
        if ia is None or ib is None:
            continue
        if ia == ib:
            continue
        if groups is not None and not allow_cross_groups:
            if groups[ia] != groups[ib]:
                continue
        if ia > ib:
            ia, ib = ib, ia
        pairs.add((ia, ib))

    if not pairs:
        raise ValueError

    pairs_arr = np.array(sorted(pairs), dtype=np.int64)
    pairs_arr, limit_used, removed = _filter_pairs_by_length(
        pairs_arr, points, max_length, context="pares FOF"
    )
    if removed > 0:
        print(f"Discarded {removed} long edges (> {limit_used:.1f} Mpc) from pairs catalog.")

    lines = np.empty(pairs_arr.shape[0] * 3, dtype=np.int64)
    lines[0::3] = 2
    lines[1::3] = pairs_arr[:, 0]
    lines[2::3] = pairs_arr[:, 1]
    return lines, limit_used


def _graph_plotter(x, y, z, lines, point_size, line_width, point_color, edge_color,
                   edge_opacity, edge_radius, scalars, cmap):
    points = np.column_stack((x, y, z))

    node_mesh = pv.PolyData(points)
    if scalars is not None:
        node_mesh.point_data["group"] = scalars

    edge_mesh = pv.PolyData()
    edge_mesh.points = points
    edge_mesh.lines = lines

    plotter = pv.Plotter(off_screen=True)
    node_kwargs = dict(render_points_as_spheres=True, point_size=point_size, show_scalar_bar=False)

    line_width = max(float(line_width), 0.1)
    edge_opacity = float(np.clip(edge_opacity, 0.0, 1.0))

    mesh_for_edges = edge_mesh
    add_kwargs = dict(opacity=edge_opacity)
    if edge_radius and edge_radius > 0:
        try:
            mesh_for_edges = edge_mesh.tube(radius=edge_radius)
            add_kwargs.update(smooth_shading=False, lighting=False)
        except Exception:
            mesh_for_edges = edge_mesh
    else:
        mesh_for_edges = edge_mesh

    if scalars is not None:
        plotter.add_mesh(node_mesh, scalars="group", cmap=cmap or "tab20", **node_kwargs)
        plotter.add_mesh(mesh_for_edges,
                         color=edge_color,
                         line_width=line_width,
                         render_lines_as_tubes=edge_radius <= 0,
                         **add_kwargs)
    else:
        plotter.add_mesh(node_mesh, color=point_color, **node_kwargs)
        plotter.add_mesh(mesh_for_edges,
                         color=edge_color,
                         line_width=line_width,
                         render_lines_as_tubes=edge_radius <= 0,
                         **add_kwargs)
    plotter.add_axes(line_width=1, labels_off=False)
    plotter.set_background("black")
    return plotter


def _filter_pairs_by_length(pairs_arr, points, max_length, context):
    if pairs_arr.size == 0:
        return pairs_arr, None, 0

    if points is None or points.size == 0:
        return pairs_arr, (float(max_length) if max_length and max_length > 0 else None), 0

    diffs = points[pairs_arr[:, 0]] - points[pairs_arr[:, 1]]
    distances = np.linalg.norm(diffs, axis=1)

    if max_length and max_length > 0:
        limit_used = float(max_length)
    else:
        if distances.size == 0:
            return pairs_arr, None, 0
        p95 = float(np.percentile(distances, 95))
        median = float(np.median(distances))
        limit_used = max(p95, median * 2.5)

    if not limit_used or limit_used <= 0:
        return pairs_arr, None, 0

    mask = distances <= limit_used
    removed = int(pairs_arr.shape[0] - np.count_nonzero(mask))
    if removed == pairs_arr.shape[0]:
        raise ValueError

    return pairs_arr[mask], limit_used, removed


def _record_movie(plotter, output, frames, framerate, azimuth, elevation):
    output.parent.mkdir(parents=True, exist_ok=True)

    suffix = output.suffix.lower()
    needs_ffmpeg = suffix in {".mp4", ".m4v", ".mov", ".avi", ".mkv"}
    if needs_ffmpeg and not HAS_FFMPEG:
        raise RuntimeError

    plotter.open_movie(str(output), framerate=framerate)
    plotter.render()
    plotter.write_frame()
    camera = plotter.camera

    def _rotate(attr, angle):
        if not angle:
            return
        attr_lower = attr.lower()
        func = getattr(camera, attr_lower, None)
        if callable(func):
            func(angle)
            return
        vtk_func = getattr(camera, attr_lower.capitalize(), None)
        if callable(vtk_func):
            vtk_func(angle)
            return

    for _ in range(frames - 1):
        _rotate("azimuth", azimuth)
        _rotate("elevation", elevation)
        plotter.render()
        plotter.write_frame()
    plotter.close()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--zone", default="NGC2")
    parser.add_argument("--base-dir", type=Path, default=DEFAULT_BASE_DIR)
    parser.add_argument("--tracer", default=TRACER_NAME)
    parser.add_argument("--cube-size", type=float, default=900.0)
    parser.add_argument("--grid", type=int, default=256)
    parser.add_argument("--sigma", type=float, default=1.2)
    parser.add_argument("--mode", choices=SUPPORTED_MODES, default="volume")
    parser.add_argument("--iso-levels", type=float, nargs="*", default=(0.3, 0.6, 0.9))
    parser.add_argument("--opacity", type=float, default=0.55)
    parser.add_argument("--lighting", default="three lights")
    parser.add_argument("--cmap", default="viridis")
    parser.add_argument("--color", default="white")
    parser.add_argument("--color-groups", action="store_true")
    parser.add_argument("--groups-file", type=Path, default=None)
    parser.add_argument("--group-cmap", default="tab20")
    parser.add_argument("--point-size", type=float, default=1.5)
    parser.add_argument("--edge-color", default="white")
    parser.add_argument("--edge-opacity", type=float, default=0.6)
    parser.add_argument("--edge-radius", type=float, default=0.0)
    parser.add_argument("--graph-point-size", type=float, default=3.0)
    parser.add_argument("--graph-line-width", type=float, default=1.2)
    parser.add_argument("--graph-neighbors", type=int, default=3)
    parser.add_argument("--graph-radius", type=float, default=0.0)
    parser.add_argument("--graph-from-pairs", action="store_true")
    parser.add_argument("--graph-cross-groups", action="store_true")
    parser.add_argument("--pairs-file", type=Path, default=None)
    parser.add_argument("--graph-max-length", type=float, default=0.0)
    parser.add_argument("--group-ids", type=int, nargs="+")
    parser.add_argument("--frames", type=int, default=180)
    parser.add_argument("--framerate", type=int, default=18)
    parser.add_argument("--azimuth", type=float, default=2.0)
    parser.add_argument("--elevation", type=float, default=0.0)
    parser.add_argument("--zoom", type=float, default=1.0)
    parser.add_argument("--movie", type=Path, default=DEFAULT_MOVIE)
    parser.add_argument("--snapshot", type=Path, default=None)
    parser.add_argument("--center", type=json.loads, default=None)
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--webtype", type=lambda s: s.lower(), choices=WEBTYPE_LABELS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if pv is None:
        raise RuntimeError()

    if not os.environ.get("DISPLAY"):
        os.environ.setdefault("PYVISTA_OFF_SCREEN", "true")
        os.environ.setdefault("PYVISTA_USE_OSMESA", "true")
        try:
            pv.OFF_SCREEN = True
        except AttributeError:
            pass
        try:
            pv.global_theme.rendering_backend = "osmesa"
        except AttributeError:
            pass

    if args.color_groups and args.mode == "volume":
        raise ValueError("--color-groups")

    groups_path = args.groups_file
    if groups_path is not None and not isinstance(groups_path, Path):
        groups_path = Path(groups_path)
    if args.color_groups and groups_path is None:
        inferred = (args.webtype or "filament").lower()
        groups_path = args.base_dir / "groups" / f"zone_{args.zone}_groups_fof_{inferred}.fits.gz"

    need_groups = (args.color_groups
                   or (args.mode == "graph" and not args.graph_cross_groups)
                   or args.graph_from_pairs
                   or bool(args.group_ids))

    x, y, z, target_ids, groups = _load_xyz(args.base_dir, args.zone, args.tracer, args.webtype,
                                            groups_path=groups_path, with_groups=need_groups)
    extras_tuple = []
    extras_tuple.append(target_ids)
    if groups is not None:
        extras_tuple.append(groups)
    extras = tuple(extras_tuple)
    x_cube, y_cube, z_cube, center, extras_sel = _select_cube(x, y, z, args.cube_size,
                                                              args.center, extras=extras)
    target_ids_cube = extras_sel[0] if extras_sel else None
    groups_cube = extras_sel[1] if extras_sel and len(extras_sel) > 1 else None

    if args.group_ids:
        if groups_cube is None or groups_cube.size == 0:
            raise ValueError()
        requested = np.array(args.group_ids, dtype=np.int64)
        group_mask = np.isin(groups_cube, requested)
        if not np.any(group_mask):
            raise ValueError()
        x_cube = x_cube[group_mask]
        y_cube = y_cube[group_mask]
        z_cube = z_cube[group_mask]
        target_ids_cube = target_ids_cube[group_mask]
        groups_cube = groups_cube[group_mask]

    total = x.size
    selected = x_cube.size
    fraction = 100.0 * selected / max(total, 1)
    half = 0.5 * args.cube_size
    print(f"Cubic ROI: {selected} / {total} galaxies ({fraction:.2f}%) in |x|,|y|,|z| <= {half:.1f} Mpc")

    group_count = None
    group_sample = None
    group_scalars = None
    if args.color_groups:
        if groups_cube is None or groups_cube.size == 0:
            raise ValueError()
        unique_groups = np.unique(groups_cube)
        group_count = unique_groups.size
        group_sample = unique_groups[:6]
        preview = ", ".join(str(int(val)) for val in group_sample)
        if group_count > 6:
            preview += ", ..."
        print(f"FOF groups in cube: {group_count} (IDs: {preview})")
        index_map = {gid: idx for idx, gid in enumerate(unique_groups)}
        group_scalars = np.array([index_map[gid] for gid in groups_cube], dtype=float)

    pairs_path_used: Optional[Path] = None
    graph_limit_used: Optional[float] = None

    if args.mode == "volume":
        rho = _build_histogram(x_cube, y_cube, z_cube, args.cube_size, args.grid, args.sigma)
        plotter = _volume_plotter(rho, args.cube_size, args.iso_levels, args.color,  args.cmap,
                                  args.opacity, args.lighting)
    elif args.mode == "points":
        point_color = args.color
        if point_color is None or point_color.strip().lower() in {"", "auto", "density"}:
            point_color = "white"
        plotter = _points_plotter(x_cube, y_cube, z_cube, args.point_size, point_color,
                                  scalars=group_scalars,
                                  cmap=args.group_cmap if args.color_groups else None)
    else:  # graph
        if x_cube.size < 2:
            raise ValueError()
        limit_used_collected: Optional[float] = None
        if args.graph_from_pairs or args.pairs_file is not None:
            pairs_path = args.pairs_file
            if pairs_path is not None and not isinstance(pairs_path, Path):
                pairs_path = Path(pairs_path)
            if pairs_path is None:
                pairs_path = args.base_dir / "pairs" / f"zone_{args.zone}_pairs.fits.gz"
            pairs_path_used = pairs_path
            groups_for_edges = None if args.graph_cross_groups else groups_cube
            lines, limit_used_collected = _build_lines_from_pairs(target_ids_cube,
                                                                  groups_for_edges,
                                                                  pairs_path,
                                                                  allow_cross_groups=args.graph_cross_groups,
                                                                  points=np.column_stack((x_cube, y_cube, z_cube)),
                                                                  max_length=args.graph_max_length)
            graph_limit_used = limit_used_collected
        else:
            points = np.column_stack((x_cube, y_cube, z_cube))
            if not args.graph_cross_groups and groups_cube is not None:
                pair_list = []
                unique_groups = np.unique(groups_cube)
                for gid in unique_groups:
                    mask = groups_cube == gid
                    if np.count_nonzero(mask) < 2:
                        continue
                    local_lines, limit_local = _build_graph_lines(
                        points[mask], args.graph_neighbors, args.graph_radius, args.graph_max_length
                    )
                    if local_lines.size == 0:
                        continue
                    idxs = np.nonzero(mask)[0]
                    local_pairs = local_lines.reshape(-1, 3)
                    local_pairs = local_pairs[:, 1:].astype(np.int64)
                    global_pairs = [(idxs[a], idxs[b]) for a, b in local_pairs]
                    pair_list.extend(global_pairs)
                    if limit_local:
                        limit_used_collected = (
                            limit_local
                            if limit_used_collected is None
                            else max(limit_used_collected, limit_local)
                        )
                if not pair_list:
                    raise ValueError()
                pair_array = np.array(sorted({tuple(sorted(p)) for p in pair_list}), dtype=np.int64)
                lines = np.empty(pair_array.shape[0] * 3, dtype=np.int64)
                lines[0::3] = 2
                lines[1::3] = pair_array[:, 0]
                lines[2::3] = pair_array[:, 1]
                graph_limit_used = limit_used_collected
            else:
                lines, limit_used_collected = _build_graph_lines(
                    points, args.graph_neighbors, args.graph_radius, args.graph_max_length
                )
                graph_limit_used = limit_used_collected
        point_color = args.color if args.color not in {None, "", "auto", "density"} else "white"
        plotter = _graph_plotter(x_cube, y_cube, z_cube, lines, args.graph_point_size,
                                 args.graph_line_width, point_color, args.edge_color,
                                 args.edge_opacity, args.edge_radius, scalars=group_scalars,
                                 cmap=args.group_cmap if args.color_groups else None)

    plotter.camera_position = "iso"
    if args.zoom and args.zoom != 1.0:
        try:
            plotter.camera.zoom(args.zoom)
        except AttributeError:
            pass

    title_parts = [f"Zona {args.zone}", f"Tracer {args.tracer.upper()}"]
    if args.webtype:
        title_parts.append(args.webtype.title())
    plotter.add_text(" | ".join(title_parts), font_size=14, color="white", position="upper_left")
    # plotter.add_text(
    #     f"Cut: |x|,|y|,|z| <= {half:.1f} Mpc",
    #     font_size=12,
    #     color="white",
    #     position="upper_right",
    # )

    if args.snapshot:
        args.snapshot.parent.mkdir(parents=True, exist_ok=True)
        plotter.screenshot(str(args.snapshot), window_size=[1024, 1024])

    _record_movie(plotter, args.movie, args.frames, args.framerate, args.azimuth, args.elevation)

    if args.show:
        plotter.show()

    cx, cy, cz = center
    info = {"zone": args.zone,
            "tracer": args.tracer,
            "cube_size_mpch": args.cube_size,
            "grid": args.grid,
            "sigma": args.sigma,
            "center": [float(cx), float(cy), float(cz)],
            "mode": args.mode,
            "movie": str(args.movie.resolve()),
            "webtype": args.webtype,
            "color": args.color}
    if args.group_ids:
        info["group_ids"] = [int(val) for val in args.group_ids]
    if args.color_groups:
        info["groups"] = {"path": str(groups_path.resolve()) if groups_path is not None else None,
                          "count": int(group_count or 0)}
        if group_sample is not None:
            info["groups"]["sample_ids"] = [int(val) for val in group_sample]
    if args.mode == "graph":
        info["graph"] = {"neighbors": int(args.graph_neighbors),
                         "radius": float(args.graph_radius),
                         "max_length": float(args.graph_max_length),
                         "line_width": float(args.graph_line_width),
                         "point_size": float(args.graph_point_size),
                         "edge_opacity": float(args.edge_opacity),
                         "edge_radius": float(args.edge_radius),
                         "from_pairs": bool(args.graph_from_pairs or args.pairs_file is not None)}
        if pairs_path_used is not None:
            info["graph"]["pairs_file"] = str(pairs_path_used.resolve())
        if graph_limit_used is not None:
            info["graph"]["length_limit_used"] = float(graph_limit_used)
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()