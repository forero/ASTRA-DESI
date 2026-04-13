import argparse, hashlib, json, os, time
from pathlib import Path
from typing import Optional

import numpy as np
from astropy.table import Table, join

_PYVISTA_IMPORT_ERROR = None
try:
    import pyvista as pv
except Exception as exc:
    pv = None
    _PYVISTA_IMPORT_ERROR = exc

try:
    from scipy.ndimage import gaussian_filter
except ImportError:
    gaussian_filter = None

try:
    from scipy.spatial import cKDTree
except ImportError:
    cKDTree = None

try:
    import imageio
    HAS_IMAGEIO = True
except ImportError:
    HAS_IMAGEIO = False

try:
    import imageio_ffmpeg
    HAS_FFMPEG = True
except ImportError:
    HAS_FFMPEG = False

DEFAULT_BASE_DIR = Path('/pscratch/sd/v/vtorresg/cosmic-web/dr2')
DEFAULT_MOVIE = Path('dr2_cube.mp4')
DEFAULT_CACHE_DIR = Path('.cache/animate_points')
DEFAULT_WINDOW_SIZE = (2160, 2160)
SUPPORTED_MODES = ('points', 'volume', 'graph')
TRACER_NAME = 'BGS'
WEBTYPE_LABELS = ('void', 'sheet', 'filament', 'knot')
WEBTYPE_COLORS = {'void': 'deepskyblue',
                  'sheet': 'darkorange',
                  'filament': 'green',
                  'knot': 'magenta'}
TRACER_CHOICES = ('BGS', 'LRG', 'ELG', 'QSO')
ZONE_CHOICES = ('NGC', 'SGC')
DR2_ZONE_VALUES = {'NGC': 2001, 'SGC': 2002}
DR2_RA_MIN = 90.0
DR2_RA_MAX = 300.0


def _log_step_timing(step, t0, timings):
    elapsed = time.perf_counter() - t0
    timings[step] = float(elapsed)
    print(f'[timing] {step}: {elapsed:.2f} s')
    return elapsed


def _configure_plotter_quality(plotter):
    if not os.environ.get('DISPLAY'):
        return
    if str(os.environ.get('PYVISTA_USE_OSMESA', '')).strip().lower() in {'1', 'true', 'yes', 'on'}:
        return
    try:
        plotter.enable_anti_aliasing('ssaa')
    except Exception:
        try:
            plotter.enable_anti_aliasing()
        except Exception:
            pass


def _as_text(value):
    if isinstance(value, (bytes, np.bytes_)):
        return value.decode('utf-8', errors='ignore')
    return str(value)


def _find_column_name(table, base):
    for suffix in ('', '_1', '_2', '_raw', '_prob'):
        name = f'{base}{suffix}' if suffix else base
        if name in table.colnames:
            return name
    raise KeyError()


def _normalize_tracer(value):
    text = _as_text(value).strip().upper()
    if not text:
        return ''
    core = text.split('_', 1)[0]
    return core


def _normalize_zone(value):
    text = _as_text(value).strip().upper()
    if not text:
        return ''
    if text in {'N', 'NORTH', '2001'}:
        return 'NGC'
    if text in {'S', 'SOUTH', '2002'}:
        return 'SGC'
    return text


def _parse_zone_arg(value):
    zone = _normalize_zone(value)
    if zone not in ZONE_CHOICES:
        raise argparse.ArgumentTypeError('Zone must be NGC or SGC for DR2 raw files.')
    return zone


def _parse_tracer_arg(value):
    tracer = _normalize_tracer(value)
    if tracer not in TRACER_CHOICES:
        raise argparse.ArgumentTypeError('Tracer must be one of: BGS, LRG, ELG, QSO.')
    return tracer


def _resolve_zone_file(directory, zone, stem_suffix=''):
    if not directory.exists():
        raise FileNotFoundError(f'Directory not found: {directory}')

    zone_tag = _normalize_zone(zone)
    candidates = [directory / f'zone_{zone_tag}{stem_suffix}.fits.gz',
                  directory / f'zone_{zone_tag}{stem_suffix}.fits',
                  directory / f'zone_{zone_tag.lower()}{stem_suffix}.fits.gz',
                  directory / f'zone_{zone_tag.lower()}{stem_suffix}.fits',]
    for path in candidates:
        if path.exists():
            return path

    patterns = [f'zone_{zone_tag}_*{stem_suffix}.fits.gz',
                f'zone_{zone_tag}_*{stem_suffix}.fits',
                f'zone_{zone_tag.lower()}_*{stem_suffix}.fits.gz',
                f'zone_{zone_tag.lower()}_*{stem_suffix}.fits',]
    tagged = []
    for pattern in patterns:
        tagged.extend(directory.glob(pattern))
    if tagged:
        tagged = sorted(set(tagged), key=lambda path: path.name)
        return tagged[0]

    raise FileNotFoundError(f'No file found for zone={zone_tag} in {directory} with suffix "{stem_suffix}"')


def _resolve_probability_path(base_dir, zone, tracer):
    zone_tag = _normalize_zone(zone)
    prob_dir = base_dir / 'probabilities'
    tracer_tag = _normalize_tracer(tracer).upper()

    search_dirs = [
        prob_dir / _normalize_tracer(tracer).lower() / zone_tag.lower(),
        prob_dir,
    ]

    patterns = [f'zone_{zone_tag}_{tracer_tag}_probability*.fits.gz',
                f'zone_{zone_tag}_{tracer_tag}_probability*.fits',
                f'zone_{zone_tag}_*{tracer_tag}*probability*.fits.gz',
                f'zone_{zone_tag}_*{tracer_tag}*probability*.fits',
                f'zone_{zone_tag}_probability*.fits.gz',
                f'zone_{zone_tag}_probability*.fits',
                f'zone_{zone_tag}_*probability*.fits.gz',
                f'zone_{zone_tag}_*probability*.fits',
                f'zone_{zone_tag.lower()}_*probability*.fits.gz',
                f'zone_{zone_tag.lower()}_*probability*.fits']

    attempted = []
    for directory in search_dirs:
        attempted.append(str(directory))
        if not directory.exists():
            continue
        matches = []
        for pattern in patterns:
            matches.extend(directory.glob(pattern))
        if not matches:
            continue
        matches = sorted(set(matches), key=lambda path: path.name)
        tracer_matches = [path for path in matches if tracer_tag in path.name.upper()]
        if tracer_matches:
            return tracer_matches[0]
        return matches[0]

    raise FileNotFoundError('No probability file found for '
                            f'zone={zone_tag}, tracer={tracer_tag}. Checked: {', '.join(attempted)}')


def _build_cache_path(cache_dir, zone, tracer, webtype, with_groups, with_webtypes, source_paths):
    payload = {'zone': str(zone),
               'tracer': str(tracer),
               'webtype': str(webtype) if webtype is not None else None,
               'with_groups': bool(with_groups),
               'with_webtypes': bool(with_webtypes),
               'sources': [],}
    for path in source_paths:
        stat = path.stat()
        payload['sources'].append({'path': str(path.resolve()),
                                   'mtime_ns': int(stat.st_mtime_ns),
                                   'size': int(stat.st_size),})

    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode('utf-8')).hexdigest()[:20]
    zone_token = _normalize_zone(zone).lower()
    tracer_token = _normalize_tracer(tracer).lower()
    web_token = (str(webtype).lower() if webtype is not None else 'all')
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f'loadxyz_{zone_token}_{tracer_token}_{web_token}_{digest}.npz'


def _load_cached_xyz(cache_path):
    try:
        with np.load(cache_path, allow_pickle=False) as data:
            x = np.asarray(data['x'], dtype=float)
            y = np.asarray(data['y'], dtype=float)
            z = np.asarray(data['z'], dtype=float)
            target_ids = np.asarray(data['target_ids'], dtype=np.int64)
            has_groups = bool(np.asarray(data['has_groups'], dtype=np.uint8)[0])
            has_webtypes = bool(np.asarray(data['has_webtypes'], dtype=np.uint8)[0])
            groups = np.asarray(data['groups'], dtype=np.int64) if has_groups else None
            webtypes = np.asarray(data['webtypes'], dtype='U8') if has_webtypes else None
        return x, y, z, target_ids, groups, webtypes
    except Exception as exc:
        print(f'[cache] warning: failed to read {cache_path}; rebuilding ({exc})')
        return None


def _save_cached_xyz(cache_path, x, y, z, target_ids, groups, webtypes):
    groups_arr = np.asarray(groups, dtype=np.int64) if groups is not None else np.array([], dtype=np.int64)
    webtypes_arr = np.asarray(webtypes, dtype='U8') if webtypes is not None else np.array([], dtype='U8')
    np.savez_compressed(cache_path,
                        x=np.asarray(x, dtype=np.float32),
                        y=np.asarray(y, dtype=np.float32),
                        z=np.asarray(z, dtype=np.float32),
                        target_ids=np.asarray(target_ids, dtype=np.int64),
                        has_groups=np.array([groups is not None], dtype=np.uint8),
                        groups=groups_arr,
                        has_webtypes=np.array([webtypes is not None], dtype=np.uint8),
                        webtypes=webtypes_arr,)


def _extract_tracers(table):
    try:
        tracer_col = _find_column_name(table, 'TRACERTYPE')
    except KeyError:
        tracer_col = None

    if tracer_col is not None:
        return np.array([_normalize_tracer(val) for val in table[tracer_col]], dtype='U8')

    tracer_id_col = _find_column_name(table, 'TRACER_ID')
    tracer_ids = np.asarray(table[tracer_id_col], dtype=np.int16)
    tracer_map = {0: 'BGS', 1: 'ELG', 2: 'LRG', 3: 'QSO'}
    return np.array([tracer_map.get(int(tid), '') for tid in tracer_ids], dtype='U8')


def _zone_mask(table, zone):
    zone_norm = _normalize_zone(zone)
    if zone_norm not in ZONE_CHOICES:
        return np.ones(len(table), dtype=bool)

    try:
        zone_col = _find_column_name(table, 'ZONE')
    except KeyError:
        zone_col = None

    if zone_col is not None:
        values = np.asarray(table[zone_col])
        if values.dtype.kind in {'U', 'S', 'O'}:
            labels = np.array([_normalize_zone(val) for val in values], dtype='U8')
            return labels == zone_norm
        zone_value = DR2_ZONE_VALUES.get(zone_norm)
        if zone_value is not None:
            return np.asarray(values, dtype=np.int64) == zone_value

    try:
        ra_col = _find_column_name(table, 'RA')
    except KeyError:
        return np.ones(len(table), dtype=bool)

    ra = np.asarray(table[ra_col], dtype=float)
    ngc_mask = (ra >= DR2_RA_MIN) & (ra <= DR2_RA_MAX)
    return ngc_mask if zone_norm == 'NGC' else ~ngc_mask


def _classify_webtypes(table):
    prob_cols = [_find_column_name(table, 'PVOID'),
                 _find_column_name(table, 'PSHEET'),
                 _find_column_name(table, 'PFILAMENT')]
    try:
        prob_cols.append(_find_column_name(table, 'PKNOT'))
        has_pknot = True
    except KeyError:
        has_pknot = False
    probs = np.vstack([np.asarray(table[col], dtype=float) for col in prob_cols]).T
    if not has_pknot:
        probs = np.column_stack((probs, np.zeros(probs.shape[0], dtype=float)))
    finite = np.isfinite(probs).any(axis=1)
    safe = np.where(np.isfinite(probs), probs, -np.inf)
    idx = np.argmax(safe, axis=1)
    labels = np.array(WEBTYPE_LABELS, dtype='U8')[idx]
    labels[~finite] = ''
    return labels


def _load_xyz(base_dir, zone, tracer, webtype, groups_path, with_groups=False, with_webtypes=False,
              cache_dir=None, use_cache=True, refresh_cache=False):
    zone_norm = _normalize_zone(zone)
    tracer_norm = _normalize_tracer(tracer)
    webtype_norm = webtype.strip().lower() if webtype is not None else None
    need_probabilities = (webtype is not None) or bool(with_webtypes)

    raw_path = _resolve_zone_file(base_dir / 'raw', zone_norm)
    prob_path = _resolve_probability_path(base_dir, zone_norm, tracer_norm) if need_probabilities else None

    target_groups_path = None
    if with_groups:
        inferred = 'filament' if webtype_norm is None else webtype_norm
        default_groups_path = _resolve_zone_file(base_dir / 'groups', zone_norm, f'_groups_fof_{inferred}')
        target_groups_path = groups_path or default_groups_path
        if not target_groups_path.exists():
            raise FileNotFoundError(f'Groups file not found: {target_groups_path}')

    cache_path = None
    if use_cache and cache_dir is not None:
        source_paths = [raw_path]
        if prob_path is not None:
            source_paths.append(prob_path)
        if target_groups_path is not None:
            source_paths.append(target_groups_path)
        cache_path = _build_cache_path(Path(cache_dir), zone_norm, tracer_norm, webtype_norm,
                                       with_groups=with_groups, with_webtypes=with_webtypes,
                                       source_paths=source_paths,)
        if not refresh_cache and cache_path.exists():
            cached = _load_cached_xyz(cache_path)
            if cached is not None:
                print(f'[cache] hit: {cache_path}')
                return cached

    table = Table.read(raw_path)

    zone_rows = _zone_mask(table, zone_norm)
    if not np.any(zone_rows):
        raise ValueError(f'No rows found for zone={zone_norm} in {raw_path}')
    table = table[zone_rows]

    tracers = _extract_tracers(table)
    tracer_mask = tracers == tracer_norm
    if not np.any(tracer_mask):
        raise ValueError(f'No rows found for tracer={tracer_norm} in {raw_path}')
    table = table[tracer_mask]

    try:
        randiter_col = _find_column_name(table, 'RANDITER')
    except KeyError:
        randiter_col = None
    if randiter_col:
        data_mask = np.asarray(table[randiter_col]) == -1
        if np.any(data_mask):
            table = table[data_mask]

    working = table
    web_labels = None
    if need_probabilities:
        prob_table = Table.read(prob_path)
        working = join(working, prob_table, keys='TARGETID', join_type='inner')

        try:
            randiter_col = _find_column_name(working, 'RANDITER')
        except KeyError:
            randiter_col = None

        if randiter_col:
            data_mask = np.asarray(working[randiter_col]) == -1
            if not np.any(data_mask):
                raise ValueError('No data rows (RANDITER = -1) after joining probabilities.')
            working = working[data_mask]

    groups_arr = None
    if with_groups:
        groups_table = Table.read(target_groups_path)
        target_col = _find_column_name(groups_table, 'TARGETID')
        group_col = _find_column_name(groups_table, 'GROUPID')
        groups_table = groups_table[[target_col, group_col]]
        groups_table.rename_columns([target_col, group_col], ['TARGETID', 'GROUPID'])
        working = join(working, groups_table, keys='TARGETID', join_type='inner')

        tracers_joined = _extract_tracers(working)
        tracer_mask_joined = tracers_joined == tracer_norm
        if not np.any(tracer_mask_joined):
            raise ValueError(f'No group rows found for tracer={tracer_norm}')
        working = working[tracer_mask_joined]
        groups_arr = np.asarray(working['GROUPID'], dtype=np.int64)

    if need_probabilities:
        web_labels = _classify_webtypes(working)
        if webtype is not None:
            if webtype_norm not in WEBTYPE_LABELS:
                raise ValueError(f'Invalid webtype "{webtype_norm}"')
            class_mask = web_labels == webtype_norm
            if not np.any(class_mask):
                raise ValueError(f'No rows found for webtype={webtype_norm}')
            working = working[class_mask]
            web_labels = web_labels[class_mask]

    x = np.asarray(working[_find_column_name(working, 'XCART')], dtype=float)
    y = np.asarray(working[_find_column_name(working, 'YCART')], dtype=float)
    z = np.asarray(working[_find_column_name(working, 'ZCART')], dtype=float)
    target_ids = np.asarray(working[_find_column_name(working, 'TARGETID')], dtype=np.int64)

    if cache_path is not None:
        try:
            _save_cached_xyz(cache_path, x, y, z, target_ids, groups_arr, web_labels)
            print(f'[cache] saved: {cache_path}')
        except Exception as exc:
            print(f'[cache] warning: failed to save {cache_path} ({exc})')

    return x, y, z, target_ids, groups_arr, web_labels


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
        hist = gaussian_filter(hist, sigma=sigma, mode='constant', cval=0.0)
    return hist


def _volume_plotter(rho, side, iso_fracs, color, cmap, opacity, lighting, window_size):
    grid_cls = getattr(pv, 'UniformGrid', None) or getattr(pv, 'ImageData', None)
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
    grid.cell_data['rho'] = rho.ravel(order='F')

    dataset = grid
    if 'rho' not in dataset.point_data:
        dataset = dataset.cell_data_to_point_data()

    scalar_min, scalar_max = dataset.get_data_range('rho')
    if not np.isfinite(scalar_max) or scalar_max <= 0:
        raise ValueError()

    levels = [frac * scalar_max if 0 < frac <= 1 else frac for frac in iso_fracs]

    mesh = dataset.contour(isosurfaces=levels, scalars='rho')
    if mesh.n_points == 0:
        raise ValueError()
    plotter = pv.Plotter(off_screen=True, lighting=lighting, window_size=list(window_size))
    _configure_plotter_quality(plotter)
    constant_color = color is not None and color.lower() not in {'', 'auto', 'density'}
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
    plotter.set_background('black')

    for actor in plotter.renderer.GetActors():
        prop = actor.GetProperty()
        line_width = prop.GetLineWidth()
        if line_width and line_width < 0.5:
            prop.SetLineWidth(0.5)

    return plotter


def _add_webtype_points(plotter, points, web_labels, point_size):
    mesh_kwargs = dict(render_points_as_spheres=True,
                       point_size=point_size,
                       show_scalar_bar=False,
                       lighting=False)
    legend_entries = []
    for label in WEBTYPE_LABELS:
        mask = web_labels == label
        if not np.any(mask):
            continue
        cloud = pv.PolyData(points[mask])
        color = WEBTYPE_COLORS[label]
        plotter.add_mesh(cloud, color=color, **mesh_kwargs)
        legend_entries.append([label.title(), color])
    if legend_entries:
        try:
            plotter.add_legend(legend_entries,
                               bcolor='black',
                               border=False,
                               loc='upper right',
                               size=(0.06, 0.06))
        except Exception:
            pass


def _points_plotter(x, y, z, point_size, color, scalars=None, cmap=None, web_labels=None,
                    window_size=(1024, 1024)):
    data = np.column_stack((x, y, z))
    plotter = pv.Plotter(off_screen=True, window_size=list(window_size))
    _configure_plotter_quality(plotter)
    if web_labels is not None:
        _add_webtype_points(plotter, data, np.asarray(web_labels, dtype='U8'), point_size)
    else:
        cloud = pv.PolyData(data)
        if scalars is not None:
            cloud.point_data['group'] = scalars
        mesh_kwargs = dict(render_points_as_spheres=True,
                           point_size=point_size,
                           show_scalar_bar=False,
                           lighting=False)
        if scalars is not None:
            plotter.add_mesh(cloud, scalars='group', cmap=cmap or 'tab20', **mesh_kwargs)
        else:
            plotter.add_mesh(cloud, color=color, **mesh_kwargs)
    plotter.set_background('black')
    return plotter


def _build_graph_lines(points, neighbors, radius, max_length, context='vecinos'):
    if cKDTree is None:
        raise RuntimeError()

    if points.shape[0] < 2:
        raise ValueError('At least two points are required to build graph lines.')

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
        raise ValueError('No valid pairs found.')

    pairs_arr = np.array(sorted(pairs), dtype=np.int64)
    pairs_arr, limit_used, removed = _filter_pairs_by_length(pairs_arr, points, max_length, context)
    if removed > 0 and (max_length is None or max_length <= 0):
        print(f'Discarded {removed} long edges (> {limit_used:.1f} Mpc) generated by {context}.')

    lines = np.empty(pairs_arr.shape[0] * 3, dtype=np.int64)
    lines[0::3] = 2
    lines[1::3] = pairs_arr[:, 0]
    lines[2::3] = pairs_arr[:, 1]
    return lines, limit_used


def _build_lines_from_pairs(target_ids, groups, pairs_path, allow_cross_groups, points, max_length):
    if not pairs_path.exists():
        raise FileNotFoundError()

    pairs_table = Table.read(pairs_path)
    tid1_col = _find_column_name(pairs_table, 'TARGETID1')
    tid2_col = _find_column_name(pairs_table, 'TARGETID2')
    tid1 = np.asarray(pairs_table[tid1_col], dtype=np.int64)
    tid2 = np.asarray(pairs_table[tid2_col], dtype=np.int64)

    try:
        randiter_col = _find_column_name(pairs_table, 'RANDITER')
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
        raise ValueError('No valid pairs found.')

    pairs_arr = np.array(sorted(pairs), dtype=np.int64)
    pairs_arr, limit_used, removed = _filter_pairs_by_length(pairs_arr, points, max_length, context='pairs FOF')
    if removed > 0:
        print(f'Discarded {removed} long edges (> {limit_used:.1f} Mpc) from pairs catalog.')

    lines = np.empty(pairs_arr.shape[0] * 3, dtype=np.int64)
    lines[0::3] = 2
    lines[1::3] = pairs_arr[:, 0]
    lines[2::3] = pairs_arr[:, 1]
    return lines, limit_used


def _graph_plotter(x, y, z, lines, point_size, line_width, point_color, edge_color,
                   edge_opacity, edge_radius, scalars, cmap, web_labels=None,
                   window_size=(1024, 1024)):
    points = np.column_stack((x, y, z))

    edge_mesh = pv.PolyData()
    edge_mesh.points = points
    edge_mesh.lines = lines

    plotter = pv.Plotter(off_screen=True, window_size=list(window_size))
    _configure_plotter_quality(plotter)
    node_kwargs = dict(render_points_as_spheres=True,
                       point_size=point_size,
                       show_scalar_bar=False,
                       lighting=False)

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

    if web_labels is not None:
        _add_webtype_points(plotter, points, np.asarray(web_labels, dtype='U8'), point_size)
        plotter.add_mesh(mesh_for_edges,
                         color=edge_color,
                         line_width=line_width,
                         render_lines_as_tubes=edge_radius <= 0,
                         **add_kwargs)
    else:
        node_mesh = pv.PolyData(points)
        if scalars is not None:
            node_mesh.point_data['group'] = scalars
            plotter.add_mesh(node_mesh, scalars='group', cmap=cmap or 'tab20', **node_kwargs)
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
    plotter.set_background('black')
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


def _record_movie(plotter, output, frames, framerate, azimuth, elevation, quality=10):
    output.parent.mkdir(parents=True, exist_ok=True)

    suffix = output.suffix.lower()
    needs_ffmpeg = suffix in {'.mp4', '.m4v', '.mov', '.avi', '.mkv'}
    if needs_ffmpeg:
        missing = []
        if not HAS_IMAGEIO:
            missing.append('imageio')
        if not HAS_FFMPEG:
            missing.append('imageio-ffmpeg')
        if missing:
            raise RuntimeError('Missing dependencies to write movie files. '
                                f'{', '.join(missing)} ')

    quality = int(np.clip(int(quality), 0, 10))
    try:
        plotter.open_movie(str(output), framerate=framerate, quality=quality)
    except TypeError:
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
        _rotate('azimuth', azimuth)
        _rotate('elevation', elevation)
        plotter.render()
        plotter.write_frame()
    plotter.close()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--zone', type=_parse_zone_arg, default='NGC')
    parser.add_argument('--base-dir', type=Path, default=DEFAULT_BASE_DIR)
    parser.add_argument('--tracer', type=_parse_tracer_arg, default=TRACER_NAME)
    parser.add_argument('--cube-size', type=float, default=900.0)
    parser.add_argument('--grid', type=int, default=256)
    parser.add_argument('--sigma', type=float, default=1.2)
    parser.add_argument('--mode', choices=SUPPORTED_MODES, default='volume')
    parser.add_argument('--iso-levels', type=float, nargs='*', default=(0.3, 0.6, 0.9))
    parser.add_argument('--opacity', type=float, default=0.55)
    parser.add_argument('--lighting', default='three lights')
    parser.add_argument('--cmap', default='viridis')
    parser.add_argument('--color', default='white')
    parser.add_argument('--color-groups', action='store_true')
    parser.add_argument('--color-webtype', action='store_true')
    parser.add_argument('--groups-file', type=Path, default=None)
    parser.add_argument('--group-cmap', default='tab20')
    parser.add_argument('--point-size', type=float, default=1.5)
    parser.add_argument('--edge-color', default='white')
    parser.add_argument('--edge-opacity', type=float, default=0.6)
    parser.add_argument('--edge-radius', type=float, default=0.0)
    parser.add_argument('--graph-point-size', type=float, default=3.0)
    parser.add_argument('--graph-line-width', type=float, default=1.2)
    parser.add_argument('--graph-neighbors', type=int, default=3)
    parser.add_argument('--graph-radius', type=float, default=0.0)
    parser.add_argument('--graph-from-pairs', action='store_true')
    parser.add_argument('--graph-cross-groups', action='store_true')
    parser.add_argument('--pairs-file', type=Path, default=None)
    parser.add_argument('--graph-max-length', type=float, default=0.0)
    parser.add_argument('--group-ids', type=int, nargs='+')
    parser.add_argument('--frames', type=int, default=180)
    parser.add_argument('--framerate', type=int, default=18)
    parser.add_argument('--movie-quality', type=int, default=10)
    parser.add_argument('--window-size', type=int, nargs=2, metavar=('WIDTH', 'HEIGHT'),
                        default=DEFAULT_WINDOW_SIZE)
    parser.add_argument('--azimuth', type=float, default=2.0)
    parser.add_argument('--elevation', type=float, default=0.0)
    parser.add_argument('--zoom', type=float, default=1.0)
    parser.add_argument('--cache-dir', type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument('--no-cache', action='store_true')
    parser.add_argument('--refresh-cache', action='store_true')
    parser.add_argument('--movie', type=Path, default=DEFAULT_MOVIE)
    parser.add_argument('--snapshot', type=Path, default=None)
    parser.add_argument('--center', type=json.loads, default=None)
    parser.add_argument('--show', action='store_true')
    parser.add_argument('--webtype', type=lambda s: s.lower(), choices=WEBTYPE_LABELS)
    return parser.parse_args()


def main() -> None:
    t_total = time.perf_counter()
    timings = {}

    t_parse = time.perf_counter()
    args = parse_args()
    args.zone = _parse_zone_arg(args.zone)
    args.tracer = _parse_tracer_arg(args.tracer)
    args.window_size = (int(args.window_size[0]), int(args.window_size[1]))
    if args.window_size[0] < 256 or args.window_size[1] < 256:
        raise ValueError('--window-size must be at least 256 256.')
    _log_step_timing('parse_args', t_parse, timings)

    if pv is None:
        detail = (f'Original import error: {type(_PYVISTA_IMPORT_ERROR).__name__}: {_PYVISTA_IMPORT_ERROR}'
                  if _PYVISTA_IMPORT_ERROR is not None else 'Original import error unavailable.')
        raise RuntimeError()

    movie_suffix = args.movie.suffix.lower()
    if movie_suffix in {'.mp4', '.m4v', '.mov', '.avi', '.mkv'}:
        missing = []
        if not HAS_IMAGEIO:
            missing.append('imageio')
        if not HAS_FFMPEG:
            missing.append('imageio-ffmpeg')
        if missing:
            raise RuntimeError()

    if not os.environ.get('DISPLAY'):
        os.environ.setdefault('PYVISTA_OFF_SCREEN', 'true')
        os.environ.setdefault('PYVISTA_USE_OSMESA', 'true')
        try:
            pv.OFF_SCREEN = True
        except AttributeError:
            pass
        try:
            pv.global_theme.rendering_backend = 'osmesa'
        except AttributeError:
            pass

    if args.color_groups and args.mode == 'volume':
        raise ValueError('--color-groups is not compatible with --mode volume.')
    if args.color_webtype and args.mode == 'volume':
        raise ValueError('--color-webtype is not compatible with --mode volume.')
    if args.color_groups and args.color_webtype:
        raise ValueError('Use either --color-groups or --color-webtype, not both.')

    groups_path = args.groups_file
    if groups_path is not None and not isinstance(groups_path, Path):
        groups_path = Path(groups_path)
    if args.color_groups and groups_path is None:
        inferred = (args.webtype or 'filament').lower()
        groups_path = _resolve_zone_file(args.base_dir / 'groups', args.zone, f'_groups_fof_{inferred}')

    need_groups = (args.color_groups
                   or (args.mode == 'graph' and not args.graph_cross_groups)
                   or args.graph_from_pairs
                   or bool(args.group_ids))

    t_load = time.perf_counter()
    x, y, z, target_ids, groups, webtypes = _load_xyz(args.base_dir,
                                                      args.zone,
                                                      args.tracer,
                                                      args.webtype,
                                                      groups_path=groups_path,
                                                      with_groups=need_groups,
                                                      with_webtypes=args.color_webtype,
                                                      cache_dir=args.cache_dir,
                                                      use_cache=not args.no_cache,
                                                      refresh_cache=args.refresh_cache,)
    _log_step_timing('load_data', t_load, timings)
    extras_tuple = []
    extras_tuple.append(target_ids)
    if groups is not None:
        extras_tuple.append(groups)
    if webtypes is not None:
        extras_tuple.append(webtypes)
    extras = tuple(extras_tuple)
    t_cube = time.perf_counter()
    x_cube, y_cube, z_cube, center, extras_sel = _select_cube(x, y, z, args.cube_size,
                                                              args.center, extras=extras)
    _log_step_timing('select_cube', t_cube, timings)
    if extras_sel is None:
        raise RuntimeError('Unexpected empty selection payload.')
    extra_idx = 0
    target_ids_cube = extras_sel[extra_idx]
    extra_idx += 1
    groups_cube = None
    if groups is not None:
        groups_cube = extras_sel[extra_idx]
        extra_idx += 1
    webtypes_cube = None
    if webtypes is not None:
        webtypes_cube = np.asarray(extras_sel[extra_idx], dtype='U8')

    if args.group_ids:
        t_group_filter = time.perf_counter()
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
        if webtypes_cube is not None:
            webtypes_cube = webtypes_cube[group_mask]
        _log_step_timing('filter_group_ids', t_group_filter, timings)

    total = x.size
    selected = x_cube.size
    fraction = 100.0 * selected / max(total, 1)
    half = 0.5 * args.cube_size
    print(f'Cubic ROI: {selected} / {total} galaxies ({fraction:.2f}%) in |x|,|y|,|z| <= {half:.1f} Mpc')

    t_colors = time.perf_counter()
    group_count = None
    group_sample = None
    group_scalars = None
    if args.color_groups:
        if groups_cube is None or groups_cube.size == 0:
            raise ValueError()
        unique_groups = np.unique(groups_cube)
        group_count = unique_groups.size
        group_sample = unique_groups[:6]
        preview = ', '.join(str(int(val)) for val in group_sample)
        if group_count > 6:
            preview += ', ...'
        print(f'FOF groups in cube: {group_count} (IDs: {preview})')
        index_map = {gid: idx for idx, gid in enumerate(unique_groups)}
        group_scalars = np.array([index_map[gid] for gid in groups_cube], dtype=float)

    webtype_counts = None
    if args.color_webtype:
        if webtypes_cube is None or webtypes_cube.size == 0:
            raise ValueError('No webtype labels available to color points by class.')
        valid_mask = np.isin(webtypes_cube, WEBTYPE_LABELS)
        if not np.any(valid_mask):
            raise ValueError('No valid webtype labels found in selected cube.')
        if not np.all(valid_mask):
            x_cube = x_cube[valid_mask]
            y_cube = y_cube[valid_mask]
            z_cube = z_cube[valid_mask]
            target_ids_cube = target_ids_cube[valid_mask]
            webtypes_cube = webtypes_cube[valid_mask]
            if groups_cube is not None:
                groups_cube = groups_cube[valid_mask]
            if group_scalars is not None:
                group_scalars = group_scalars[valid_mask]

        webtype_counts = {label: int(np.count_nonzero(webtypes_cube == label))
                          for label in WEBTYPE_LABELS
                          if np.count_nonzero(webtypes_cube == label) > 0}
        summary = ', '.join(f'{label}:{count}' for label, count in webtype_counts.items())
        print(f'Webtype classes in cube: {summary}')
    _log_step_timing('prepare_coloring', t_colors, timings)

    pairs_path_used: Optional[Path] = None
    graph_limit_used: Optional[float] = None

    if args.mode == 'volume':
        t_build = time.perf_counter()
        rho = _build_histogram(x_cube, y_cube, z_cube, args.cube_size, args.grid, args.sigma)
        plotter = _volume_plotter(rho, args.cube_size, args.iso_levels, args.color,  args.cmap,
                                  args.opacity, args.lighting, args.window_size)
        _log_step_timing('build_volume_plotter', t_build, timings)
    elif args.mode == 'points':
        t_build = time.perf_counter()
        point_color = args.color
        if point_color is None or point_color.strip().lower() in {'', 'auto', 'density'}:
            point_color = 'white'
        if args.point_size < 1.5 and max(args.window_size) >= 1800:
            print('[warn] point-size < 1.5 with high resolution can look black/tiny; try --point-size 2.5 or higher.')
        plotter = _points_plotter(x_cube, y_cube, z_cube, args.point_size, point_color,
                                  scalars=group_scalars,
                                  cmap=args.group_cmap if args.color_groups else None,
                                  web_labels=webtypes_cube if args.color_webtype else None,
                                  window_size=args.window_size)
        _log_step_timing('build_points_plotter', t_build, timings)
    else:
        t_build = time.perf_counter()
        if x_cube.size < 2:
            raise ValueError()
        limit_used_collected: Optional[float] = None
        if args.graph_from_pairs or args.pairs_file is not None:
            pairs_path = args.pairs_file
            if pairs_path is not None and not isinstance(pairs_path, Path):
                pairs_path = Path(pairs_path)
            if pairs_path is None:
                pairs_path = _resolve_zone_file(args.base_dir / 'pairs', args.zone, '_pairs')
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
                    local_lines, limit_local = _build_graph_lines(points[mask],
                                                                  args.graph_neighbors,
                                                                  args.graph_radius,
                                                                  args.graph_max_length)
                    if local_lines.size == 0:
                        continue
                    idxs = np.nonzero(mask)[0]
                    local_pairs = local_lines.reshape(-1, 3)
                    local_pairs = local_pairs[:, 1:].astype(np.int64)
                    global_pairs = [(idxs[a], idxs[b]) for a, b in local_pairs]
                    pair_list.extend(global_pairs)
                    if limit_local:
                        limit_used_collected = (limit_local
                                                if limit_used_collected is None
                                                else max(limit_used_collected, limit_local))
                if not pair_list:
                    raise ValueError()
                pair_array = np.array(sorted({tuple(sorted(p)) for p in pair_list}), dtype=np.int64)
                lines = np.empty(pair_array.shape[0] * 3, dtype=np.int64)
                lines[0::3] = 2
                lines[1::3] = pair_array[:, 0]
                lines[2::3] = pair_array[:, 1]
                graph_limit_used = limit_used_collected
            else:
                lines, limit_used_collected = _build_graph_lines(points, args.graph_neighbors,
                                                                 args.graph_radius,
                                                                 args.graph_max_length)
                graph_limit_used = limit_used_collected
        point_color = args.color if args.color not in {None, '', 'auto', 'density'} else 'white'
        plotter = _graph_plotter(x_cube, y_cube, z_cube, lines, args.graph_point_size,
                                 args.graph_line_width, point_color, args.edge_color,
                                 args.edge_opacity, args.edge_radius, scalars=group_scalars,
                                 cmap=args.group_cmap if args.color_groups else None,
                                 web_labels=webtypes_cube if args.color_webtype else None,
                                 window_size=args.window_size)
        _log_step_timing('build_graph_plotter', t_build, timings)

    plotter.camera_position = 'iso'
    try:
        plotter.reset_camera()
    except Exception:
        pass
    try:
        plotter.camera.SetClippingRange(0.1, 1e8)
    except Exception:
        try:
            plotter.renderer.ResetCameraClippingRange()
        except Exception:
            pass
    if args.zoom and args.zoom != 1.0:
        try:
            plotter.camera.zoom(args.zoom)
        except AttributeError:
            pass

    if args.snapshot:
        t_snapshot = time.perf_counter()
        args.snapshot.parent.mkdir(parents=True, exist_ok=True)
        plotter.screenshot(str(args.snapshot), window_size=list(args.window_size))
        _log_step_timing('snapshot', t_snapshot, timings)

    t_movie = time.perf_counter()
    _record_movie(plotter, args.movie, args.frames, args.framerate, args.azimuth,
                  args.elevation, quality=args.movie_quality)
    _log_step_timing('record_movie', t_movie, timings)

    if args.show:
        t_show = time.perf_counter()
        plotter.show()
        _log_step_timing('show_window', t_show, timings)

    cx, cy, cz = center
    info = {'zone': args.zone,
            'tracer': args.tracer,
            'cube_size_mpch': args.cube_size,
            'grid': args.grid,
            'sigma': args.sigma,
            'center': [float(cx), float(cy), float(cz)],
            'mode': args.mode,
            'movie': str(args.movie.resolve()),
            'webtype': args.webtype,
            'color': args.color,
            'color_webtype': bool(args.color_webtype),
            'window_size': [int(args.window_size[0]), int(args.window_size[1])],
            'movie_quality': int(args.movie_quality),
            'cache': {'enabled': bool(not args.no_cache),
                      'refresh': bool(args.refresh_cache),
                      'dir': str(args.cache_dir.resolve())}}
    if args.group_ids:
        info['group_ids'] = [int(val) for val in args.group_ids]
    if args.color_groups:
        info['groups'] = {'path': str(groups_path.resolve()) if groups_path is not None else None,
                          'count': int(group_count or 0)}
        if group_sample is not None:
            info['groups']['sample_ids'] = [int(val) for val in group_sample]
    if args.color_webtype and webtype_counts is not None:
        info['webtype_counts'] = {label: int(count) for label, count in webtype_counts.items()}
    if args.mode == 'graph':
        info['graph'] = {'neighbors': int(args.graph_neighbors),
                         'radius': float(args.graph_radius),
                         'max_length': float(args.graph_max_length),
                         'line_width': float(args.graph_line_width),
                         'point_size': float(args.graph_point_size),
                         'edge_opacity': float(args.edge_opacity),
                         'edge_radius': float(args.edge_radius),
                         'from_pairs': bool(args.graph_from_pairs or args.pairs_file is not None)}
        if pairs_path_used is not None:
            info['graph']['pairs_file'] = str(pairs_path_used.resolve())
        if graph_limit_used is not None:
            info['graph']['length_limit_used'] = float(graph_limit_used)
    info['timings_seconds'] = {key: float(val) for key, val in timings.items()}
    info['timings_seconds']['total'] = float(time.perf_counter() - t_total)
    print(f'[timing] total: {info['timings_seconds']['total']:.2f} s')
    print(json.dumps(info, indent=2))


if __name__ == '__main__':
    main()