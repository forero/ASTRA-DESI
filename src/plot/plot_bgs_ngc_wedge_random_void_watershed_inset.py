import argparse, os, sys
from pathlib import Path

import fitsio
import numpy as np
from astropy.table import Table
from scipy.spatial import Delaunay, QhullError

os.environ.setdefault('MPLCONFIGDIR', '/tmp/matplotlib-cache')
os.environ.setdefault('XDG_CACHE_HOME', '/tmp')
Path(os.environ['MPLCONFIGDIR']).mkdir(parents=True, exist_ok=True)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

plt.style.use('dark_background')
plt.rcParams.update({'text.usetex': True})

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from .plot_bgs_ngc_wedge_void_groups_inset import (DEFAULT_CLASS_GLOB,
                                                       DEFAULT_RANDOM_VOID_CACHE,
                                                       DEFAULT_RAW,
                                                       _build_void_color_map,
                                                       _ellipse_theta_r,
                                                       add_void_inset,
                                                       empty_random_void_array,
                                                       find_first_non_data_row,
                                                       load_or_build_random_void_cache,
                                                       parse_iteration_spec,
                                                       select_auto_dec0)
except ImportError:
    from plot_bgs_ngc_wedge_void_groups_inset import (DEFAULT_CLASS_GLOB,
                                                      DEFAULT_RANDOM_VOID_CACHE,
                                                      DEFAULT_RAW,
                                                      _build_void_color_map,
                                                      _ellipse_theta_r,
                                                      add_void_inset,
                                                      empty_random_void_array,
                                                      find_first_non_data_row,
                                                      load_or_build_random_void_cache,
                                                      parse_iteration_spec,
                                                      select_auto_dec0)

from group_finder_v2.astra import build_cosmology, radec_z_to_cartesian
from group_finder_v2.make_cat import (build_point_membership_table,
                                      consolidate_group_info,
                                      write_group_table_fits)
from group_finder_v2.watershed import BOUNDARY_ID, run_watershed


DEFAULT_OUTPUT_CATALOG = 'cache/zone_NGC_BGS_random_void_watershed_quick.fits'
DEFAULT_OUTPUT_PLOT = 'plots/bgs_ngc_wedge_random_void_watershed_inset.png'
RANDOM_VOID_COLUMNS = ['TARGETID', 'RANDITER', 'RA', 'DEC', 'Z', 'R_VALUE']
EDGE_PAIRS_3D = np.array([[0, 1], [0, 2], [0, 3],
                          [1, 2], [1, 3], [2, 3]], dtype=np.int64)


def _optional_float(value):
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in ('none', 'null', 'nan'):
        return None
    return float(value)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--raw-input', default=DEFAULT_RAW)
    p.add_argument('--classification-glob', default=DEFAULT_CLASS_GLOB)
    p.add_argument('--random-void-cache', default=DEFAULT_RANDOM_VOID_CACHE)
    p.add_argument('--rebuild-random-void-cache', action='store_true')
    p.add_argument('--random-iterations', default='all')
    p.add_argument('--watershed-iterations', default='0')
    p.add_argument('--random-chunk-size', type=int, default=1_000_000)
    p.add_argument('--max-points', type=int, default=60_000)
    p.add_argument('--sample-seed', type=int, default=12345)
    p.add_argument('--output-catalog', default=DEFAULT_OUTPUT_CATALOG)
    p.add_argument('--output', default=DEFAULT_OUTPUT_PLOT)

    p.add_argument('--slice-width-deg', type=float, default=None)
    p.add_argument('--dec0', type=float, default=None)
    p.add_argument('--dec-scan-step', type=float, default=0.25)
    p.add_argument('--ra-center-deg', type=float, default=180.0)
    p.add_argument('--rmax-mpc', type=float, default=1250.0)
    p.add_argument('--data-randiter', type=int, default=-1)

    p.add_argument('--h', type=float, default=0.6736)
    p.add_argument('--omega-m', type=float, default=0.315)
    p.add_argument('--r-threshold', '--growth-threshold', dest='r_threshold',
                   type=float, default=-0.25)
    p.add_argument('--seed-threshold', type=_optional_float, default=-0.45)
    p.add_argument('--merge-strategy', choices=['none', 'threshold', 'depth'],
                   default='depth')
    p.add_argument('--merge-threshold', type=_optional_float, default=-0.35)
    p.add_argument('--delta-merge', type=_optional_float, default=0.16)
    p.add_argument('--boundary-policy', choices=['keep', 'unassigned'],
                   default='unassigned')
    p.add_argument('--seed-plateau-tolerance', type=float, default=0.0)
    p.add_argument('--min-group-size', type=int, default=4)
    p.add_argument('--min-rand-for-shape', type=int, default=3)

    p.add_argument('--max-groups', type=int, default=3000)
    p.add_argument('--plot-members', dest='plot_members', action='store_true')
    p.add_argument('--no-members', dest='plot_members', action='store_false')
    p.set_defaults(plot_members=True)
    p.add_argument('--random-void-color', default='#f2f2f2')
    p.add_argument('--random-void-point-size', type=float, default=0.015)
    p.add_argument('--random-void-alpha', type=float, default=0.14)
    p.add_argument('--member-point-size', type=float, default=0.08)
    p.add_argument('--member-alpha', type=float, default=0.72)
    p.add_argument('--void-center-size', type=float, default=1.1)
    p.add_argument('--void-alpha', type=float, default=0.95)
    p.add_argument('--void-ellipse-alpha', type=float, default=0.9)
    p.add_argument('--void-ellipse-lw', type=float, default=0.42)
    p.add_argument('--void-ellipse-npts', type=int, default=200)
    p.add_argument('--void-cmap', default='turbo')
    p.add_argument('--bg-color', default='#000000')
    p.add_argument('--dpi', type=int, default=500)
    p.add_argument('--hide-legend', action='store_true')

    p.add_argument('--zoom-ra-min', type=float, default=None)
    p.add_argument('--zoom-ra-max', type=float, default=None)
    p.add_argument('--zoom-r-min', type=float, default=0.0)
    p.add_argument('--zoom-r-max', type=float, default=300.0)
    p.add_argument('--zoom-ra-tick-step', type=float, default=15.0)
    p.add_argument('--zoom-r-tick-step', type=float, default=100.0)
    p.add_argument('--zoom-point-scale', type=float, default=2.2)
    p.add_argument('--zoom-random-void-point-scale', type=float, default=1.8)
    p.add_argument('--zoom-alpha-boost', type=float, default=0.15)
    p.add_argument('--inset-left', type=float, default=0.49)
    p.add_argument('--inset-bottom', type=float, default=0.42)
    p.add_argument('--inset-width', type=float, default=0.40)
    p.add_argument('--inset-height', type=float, default=0.40)

    return p.parse_args()


def _read_cache_dec_bounds(cache_path):
    if not cache_path.exists():
        return None
    with fitsio.FITS(str(cache_path)) as fobj:
        header = fobj[1].read_header()
    if 'DECLO' in header and 'DECHI' in header:
        return float(header['DECLO']), float(header['DECHI'])
    return None


def _resolve_dec_bounds(args, cache_path):
    cache_bounds = _read_cache_dec_bounds(cache_path)
    if args.dec0 is None and args.slice_width_deg is None and cache_bounds is not None:
        dec_lo, dec_hi = cache_bounds
        return dec_lo, dec_hi, 0.5 * (dec_lo + dec_hi), dec_hi - dec_lo, 'cache-header'

    if args.slice_width_deg is None:
        args.slice_width_deg = 2.0

    if args.dec0 is None:
        with fitsio.FITS(str(args.raw_input)) as fobj:
            hdu = fobj[1]
            n_data = find_first_non_data_row(hdu, data_randiter=args.data_randiter)
            rows = np.arange(n_data, dtype=np.int64)
            dec = hdu.read(columns=['DEC'], rows=rows)['DEC']
        finite = np.isfinite(dec)
        dec0, _, _ = select_auto_dec0(dec=np.asarray(dec[finite], dtype=np.float64),
                                      width_deg=args.slice_width_deg,
                                      step_deg=args.dec_scan_step)
        source = 'raw-auto'
    else:
        dec0 = float(args.dec0)
        source = 'args'

    half = 0.5 * float(args.slice_width_deg)
    return dec0 - half, dec0 + half, dec0, float(args.slice_width_deg), source


def _validate_cache_covers_dec_bounds(cache_path, dec_bounds, rebuild):
    if rebuild or (not cache_path.exists()):
        return
    cache_bounds = _read_cache_dec_bounds(cache_path)
    if cache_bounds is None:
        return
    tol = 1.0e-6
    if dec_bounds[0] < cache_bounds[0] - tol or dec_bounds[1] > cache_bounds[1] + tol:
        raise RuntimeError(f'{cache_path} was built for DEC={cache_bounds}, '
                           f'but this run needs DEC={dec_bounds}. Use the cache '
                           'bounds, --rebuild-random-void-cache, or a different '
                           '--random-void-cache.')


def _reservoir_add(sample, n_seen, rows, max_points, rng):
    if len(rows) == 0:
        return sample, n_seen

    if sample is None:
        sample = np.empty(max_points, dtype=rows.dtype)

    n_rows = len(rows)
    if n_seen < max_points:
        n_fill = min(max_points - n_seen, n_rows)
        sample[n_seen:n_seen + n_fill] = rows[:n_fill]
        n_seen += n_fill
        rows = rows[n_fill:]
        n_rows = len(rows)

    if n_rows == 0:
        return sample, n_seen

    counts = np.arange(n_seen + 1, n_seen + n_rows + 1, dtype=np.int64)
    draws = rng.integers(0, counts)
    hits = draws < max_points
    for target_idx, row in zip(draws[hits].tolist(), rows[hits]):
        sample[int(target_idx)] = row
    n_seen += n_rows
    return sample, n_seen


def _read_random_void_selection(cache_path, dec_bounds, iteration_spec, chunk_size,
                                max_points, sample_seed):
    wanted_iterations = parse_iteration_spec(iteration_spec)
    max_points = int(max_points)
    use_reservoir = max_points > 0
    rng = np.random.default_rng(sample_seed)

    out = None
    chunks = []
    n_seen = 0
    n_scanned = 0
    n_matching = 0

    with fitsio.FITS(str(cache_path)) as fobj:
        hdu = fobj[1]
        nrows = hdu.get_nrows()
        for start in range(0, nrows, chunk_size):
            stop = min(start + chunk_size, nrows)
            rows = np.arange(start, stop, dtype=np.int64)
            arr = hdu.read(columns=RANDOM_VOID_COLUMNS, rows=rows)
            n_scanned += len(arr)

            mask = (np.isfinite(arr['RA'])
                    & np.isfinite(arr['DEC'])
                    & np.isfinite(arr['Z'])
                    & np.isfinite(arr['R_VALUE']))
            mask &= (arr['DEC'] >= dec_bounds[0]) & (arr['DEC'] <= dec_bounds[1])
            if wanted_iterations is not None:
                mask &= np.isin(arr['RANDITER'], np.fromiter(wanted_iterations, dtype=np.int32))

            if not np.any(mask):
                continue

            selected = arr[mask]
            n_matching += len(selected)
            if use_reservoir:
                out, n_seen = _reservoir_add(out, n_seen, selected,
                                             max_points=max_points, rng=rng)
            else:
                chunks.append(selected.copy())

    if use_reservoir:
        if out is None:
            out = empty_random_void_array()
        else:
            out = out[:min(n_seen, max_points)].copy()
    elif chunks:
        out = np.concatenate(chunks)
    else:
        out = empty_random_void_array()

    return out, {'n_scanned': n_scanned,
                 'n_matching': n_matching,
                 'n_returned': len(out),
                 'sampled': use_reservoir and n_matching > len(out)}


def _unique_coordinate_filter(coords):
    if len(coords) == 0:
        return np.array([], dtype=np.int64)
    _, unique_idx = np.unique(coords, axis=0, return_index=True)
    return np.sort(unique_idx.astype(np.int64, copy=False))


def _build_unique_edges_from_simplices(simplices):
    n_simplices = simplices.shape[0]
    edges = np.empty((n_simplices * len(EDGE_PAIRS_3D), 2), dtype=np.int64)
    for i, pair in enumerate(EDGE_PAIRS_3D):
        start = i * n_simplices
        stop = start + n_simplices
        edges[start:stop] = simplices[:, pair]
    edges = np.sort(edges, axis=1)
    return np.unique(edges, axis=0)


def _build_delaunay_neighbors(coords):
    if len(coords) < 4:
        raise ValueError(f'Need at least 4 points for 3D Delaunay, got {len(coords)}')
    try:
        tri = Delaunay(coords)
    except QhullError:
        tri = Delaunay(coords, qhull_options='QJ Qbb Qc Qz Q12')

    edges = _build_unique_edges_from_simplices(tri.simplices.astype(np.int64, copy=False))
    valid_edge = (edges[:, 0] < len(coords)) & (edges[:, 1] < len(coords))
    edges = edges[valid_edge]

    neighbors = [[] for _ in range(len(coords))]
    for a, b in edges:
        ai = int(a)
        bi = int(b)
        neighbors[ai].append(bi)
        neighbors[bi].append(ai)

    degree = np.asarray([len(item) for item in neighbors], dtype=np.int32)
    if np.any(degree == 0):
        bad = np.flatnonzero(degree == 0)[:10].tolist()
        raise RuntimeError(f'Delaunay produced isolated nodes. First indices: {bad}')
    return neighbors, degree, edges


def _make_empty_data_table():
    table = Table()
    table['TARGETID'] = np.array([], dtype=np.int64)
    table['RA'] = np.array([], dtype=np.float64)
    table['DEC'] = np.array([], dtype=np.float64)
    table['Z'] = np.array([], dtype=np.float64)
    table['X_CART'] = np.array([], dtype=np.float64)
    table['Y_CART'] = np.array([], dtype=np.float64)
    table['Z_CART'] = np.array([], dtype=np.float64)
    table['GROUPID'] = np.array([], dtype=np.int32)
    table['IS_BOUNDARY'] = np.array([], dtype=np.int8)
    table['R'] = np.array([], dtype=np.float32)
    table['N_DATA'] = np.array([], dtype=np.int32)
    table['N_RAND'] = np.array([], dtype=np.int32)
    return table


def _make_random_table(random_arr, coords, group_of, is_boundary, r_values, degree):
    table = Table()
    table['TARGETID'] = np.asarray(random_arr['TARGETID'], dtype=np.int64)
    table['RA'] = np.asarray(random_arr['RA'], dtype=np.float64)
    table['DEC'] = np.asarray(random_arr['DEC'], dtype=np.float64)
    table['Z'] = np.asarray(random_arr['Z'], dtype=np.float64)
    table['X_CART'] = coords[:, 0].astype(np.float64)
    table['Y_CART'] = coords[:, 1].astype(np.float64)
    table['Z_CART'] = coords[:, 2].astype(np.float64)
    table['GROUPID'] = np.asarray(group_of, dtype=np.int32)
    table['IS_BOUNDARY'] = np.asarray(is_boundary, dtype=np.int8)
    table['R'] = np.asarray(r_values, dtype=np.float32)
    table['N_DATA'] = np.zeros(len(table), dtype=np.int32)
    table['N_RAND'] = np.asarray(degree, dtype=np.int32)
    return table


def _select_groups_for_plot(group_table, rmax_mpc, h, max_groups):
    if len(group_table) == 0:
        return group_table[:0], np.array([], dtype=bool)

    x = np.asarray(group_table['X'], dtype=np.float64)
    y = np.asarray(group_table['Y'], dtype=np.float64)
    z = np.asarray(group_table['Z'], dtype=np.float64)
    r_mpc = np.sqrt(x * x + y * y + z * z) / h
    valid = (np.isfinite(group_table['RA'])
             & np.isfinite(group_table['DEC'])
             & np.isfinite(r_mpc)
             & (r_mpc <= rmax_mpc)
             & (np.asarray(group_table['VOID_ID'], dtype=np.int64) >= 0))
    idx = np.flatnonzero(valid)
    if max_groups > 0 and len(idx) > max_groups:
        sizes = np.asarray(group_table['N_RAND_IN_GROUP'], dtype=np.int64)
        order = np.argsort(sizes[idx], kind='stable')[::-1]
        idx = idx[order[:max_groups]]
        idx = np.sort(idx)

    keep = np.zeros(len(group_table), dtype=bool)
    keep[idx] = True
    return group_table[keep], keep


def _plot_random_void_wedge(args, random_arr, group_table, group_of, dec0,
                            slice_width_deg, stats):
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    h = float(args.h)
    ra = np.asarray(random_arr['RA'], dtype=np.float64)
    dec = np.asarray(random_arr['DEC'], dtype=np.float64)
    random_r_mpc = stats['r_mpc']
    random_in_r = random_r_mpc <= args.rmax_mpc
    random_theta = np.deg2rad(ra - args.ra_center_deg)

    plot_groups, _ = _select_groups_for_plot(group_table=group_table,
                                             rmax_mpc=args.rmax_mpc,
                                             h=h,
                                             max_groups=args.max_groups)
    void_ids = np.asarray(plot_groups['VOID_ID'], dtype=np.int64)
    void_ra = np.asarray(plot_groups['RA'], dtype=np.float64)
    void_x = np.asarray(plot_groups['X'], dtype=np.float64)
    void_y = np.asarray(plot_groups['Y'], dtype=np.float64)
    void_zcart = np.asarray(plot_groups['Z'], dtype=np.float64)
    void_r_mpc = np.sqrt(void_x * void_x + void_y * void_y + void_zcart * void_zcart) / h
    void_a = np.asarray(plot_groups['SEMI_AXIS_A'], dtype=np.float64) / h
    void_b = np.asarray(plot_groups['SEMI_AXIS_B'], dtype=np.float64) / h
    void_reff = np.asarray(plot_groups['R_EFF'], dtype=np.float64) / h
    void_theta = np.deg2rad(void_ra - args.ra_center_deg)

    theta_min_deg = float(np.rad2deg(np.min(random_theta[random_in_r])))
    theta_max_deg = float(np.rad2deg(np.max(random_theta[random_in_r])))
    if len(void_theta) > 0:
        theta_min_deg = min(theta_min_deg, float(np.rad2deg(np.min(void_theta))))
        theta_max_deg = max(theta_max_deg, float(np.rad2deg(np.max(void_theta))))

    plotted_gid = set(void_ids.tolist())
    member_mask = (group_of >= 0) & np.isin(group_of, void_ids)
    member_mask &= random_in_r
    if args.plot_members and np.any(member_mask):
        member_theta = random_theta[member_mask]
        member_r = random_r_mpc[member_mask]
        member_gid = group_of[member_mask]
    else:
        member_theta = None
        member_r = None
        member_gid = None

    color_ids = void_ids
    if member_gid is not None and len(member_gid) > 0:
        color_ids = np.concatenate([void_ids, member_gid])
    void_colors = _build_void_color_map(color_ids, args.void_cmap)
    if member_gid is not None and len(member_gid) > 0:
        member_colors = np.array([void_colors.get(int(g), (1.0, 1.0, 1.0, 0.9))
                                  for g in member_gid])
    else:
        member_colors = None

    fig = plt.figure(figsize=(11.0, 7.2), dpi=args.dpi, facecolor=args.bg_color)
    ax = fig.add_subplot(111, projection='polar', facecolor=args.bg_color)
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(1)

    ax.scatter(random_theta[random_in_r],
               random_r_mpc[random_in_r],
               s=args.random_void_point_size,
               c=args.random_void_color,
               alpha=args.random_void_alpha,
               linewidths=0,
               rasterized=True,
               zorder=0)

    if member_theta is not None and member_colors is not None:
        ax.scatter(member_theta, member_r,
                   s=args.member_point_size,
                   c=member_colors,
                   alpha=args.member_alpha,
                   linewidths=0,
                   rasterized=True,
                   zorder=2)

    for i in range(len(void_ids)):
        vid = int(void_ids[i])
        color = void_colors.get(vid, (1.0, 1.0, 1.0, 0.9))
        theta0 = float(void_theta[i])
        r0 = float(void_r_mpc[i])
        a = float(void_a[i]) if np.isfinite(void_a[i]) and void_a[i] > 0 else float(void_reff[i])
        b = float(void_b[i]) if np.isfinite(void_b[i]) and void_b[i] > 0 else float(void_reff[i])
        if not np.isfinite(a) or a <= 0:
            a = float(void_reff[i]) if np.isfinite(void_reff[i]) and void_reff[i] > 0 else 0.0
        if not np.isfinite(b) or b <= 0:
            b = float(void_reff[i]) if np.isfinite(void_reff[i]) and void_reff[i] > 0 else 0.0
        if r0 <= 0 or (a <= 0 and b <= 0):
            continue

        ax.scatter([theta0], [r0], s=args.void_center_size,
                   c=[color], alpha=args.void_alpha,
                   linewidths=0, zorder=5)
        theta_e, r_e = _ellipse_theta_r(theta0=theta0, r0=r0, a=a, b=b,
                                        npts=args.void_ellipse_npts)
        ax.plot(theta_e, r_e, color=color, linewidth=args.void_ellipse_lw,
                alpha=args.void_ellipse_alpha, zorder=4)

    ax.set_thetamin(theta_min_deg)
    ax.set_thetamax(theta_max_deg)
    ax.set_ylim(0.0, args.rmax_mpc)
    ax.grid(color='0.50', alpha=0.30, linewidth=0.7)
    ax.spines['polar'].set_color('0.75')
    ax.spines['polar'].set_linewidth(1.0)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.tick_params(axis='both', which='both', length=0)

    ax.set_title(rf'$\mathrm{{BGS\ NGC\ random\ void\ watershed}},\ '
                 rf'\delta={slice_width_deg:.1f}^\circ'
                 rf'\ (\mathrm{{DEC\ center}}={dec0:.2f}^\circ)$',
                 fontsize=17, pad=28, color='white')
    ax.text(0.2, 0.215, rf'$\delta = {slice_width_deg:.1f}^\circ$',
            transform=ax.transAxes, ha='left', va='center',
            fontsize=20, color='white', rotation=-7, clip_on=False)

    theta_zoom_min_deg = theta_min_deg if args.zoom_ra_min is None else args.zoom_ra_min - args.ra_center_deg
    theta_zoom_max_deg = theta_max_deg if args.zoom_ra_max is None else args.zoom_ra_max - args.ra_center_deg
    if theta_zoom_min_deg >= theta_zoom_max_deg:
        raise ValueError('Inset RA range must satisfy --zoom-ra-min < --zoom-ra-max.')

    add_void_inset(fig=fig, ax=ax, void_theta=void_theta, void_r=void_r_mpc,
                   void_ids=void_ids, void_a=void_a, void_b=void_b,
                   void_reff=void_reff, void_colors=void_colors,
                   ra_center_deg=args.ra_center_deg,
                   theta_zoom_min_deg=theta_zoom_min_deg,
                   theta_zoom_max_deg=theta_zoom_max_deg,
                   r_zoom_min=args.zoom_r_min, r_zoom_max=args.zoom_r_max,
                   inset_rect=(args.inset_left, args.inset_bottom,
                               args.inset_width, args.inset_height),
                   point_size=args.void_center_size * args.zoom_point_scale,
                   alpha=min(1.0, args.void_alpha + args.zoom_alpha_boost),
                   ellipse_lw=args.void_ellipse_lw,
                   ellipse_alpha=min(1.0, args.void_ellipse_alpha + args.zoom_alpha_boost),
                   ellipse_npts=args.void_ellipse_npts,
                   ra_tick_step=args.zoom_ra_tick_step,
                   rtick_step=args.zoom_r_tick_step,
                   bg_color=args.bg_color,
                   plot_ellipses=True,
                   member_theta=member_theta,
                   member_r=member_r,
                   member_colors=member_colors,
                   member_point_size=args.member_point_size * args.zoom_point_scale,
                   member_alpha=min(1.0, args.member_alpha + args.zoom_alpha_boost),
                   random_void_theta=random_theta,
                   random_void_r=random_r_mpc,
                   random_void_color=args.random_void_color,
                   random_void_point_size=args.random_void_point_size * args.zoom_random_void_point_scale,
                   random_void_alpha=min(1.0, args.random_void_alpha + args.zoom_alpha_boost))

    if not args.hide_legend:
        handles = [Line2D([0], [0], marker='o', linestyle='', markersize=5,
                        markerfacecolor=args.random_void_color,
                        markeredgecolor='none',
                        label='Random void points',
                        alpha=min(0.8, max(args.random_void_alpha * 3.0, 0.25))),
                   Line2D([0], [0], marker='o', linestyle='', markersize=7,
                        markerfacecolor='none', markeredgecolor='white',
                        label='Watershed groups', alpha=0.9)]
        if member_theta is not None:
            handles.insert(1, Line2D([0], [0], marker='o', linestyle='',
                                     markersize=7, markerfacecolor='none',
                                     markeredgecolor='white',
                                     label='Assigned members', alpha=0.9))
        leg = fig.legend(handles=handles,
                         labels=[hnd.get_label() for hnd in handles],
                         loc='upper right',
                         bbox_to_anchor=(0.985, 0.965),
                         framealpha=0.6,
                         ncol=1,
                         fontsize=10)
        for text in leg.get_texts():
            text.set_color('white')

    fig.subplots_adjust(left=0.04, right=0.98, bottom=0.06, top=0.90)
    fig.savefig(out_path, dpi=args.dpi)
    plt.close(fig)

    return {'output_plot': out_path,
            'n_groups_drawn': len(void_ids),
            'n_members_drawn': 0 if member_gid is None else len(member_gid),
            'n_random_in_rmax': int(np.count_nonzero(random_in_r))}


def main():
    args = parse_args()
    cache_path = Path(args.random_void_cache)
    output_catalog = Path(args.output_catalog)
    output_catalog.parent.mkdir(parents=True, exist_ok=True)

    dec_lo, dec_hi, dec0, slice_width_deg, dec_source = _resolve_dec_bounds(args, cache_path)
    dec_bounds = (dec_lo, dec_hi)
    _validate_cache_covers_dec_bounds(cache_path=cache_path,
                                      dec_bounds=dec_bounds,
                                      rebuild=args.rebuild_random_void_cache)

    if args.rebuild_random_void_cache or (not cache_path.exists()):
        load_or_build_random_void_cache(raw_path=Path(args.raw_input),
                                        class_pattern=args.classification_glob,
                                        iteration_spec=args.random_iterations,
                                        cache_path=cache_path,
                                        data_randiter=args.data_randiter,
                                        chunk_size=args.random_chunk_size,
                                        dec_bounds=dec_bounds,
                                        rebuild=True)

    random_arr, selection_stats = _read_random_void_selection(
        cache_path=cache_path,
        dec_bounds=dec_bounds,
        iteration_spec=args.watershed_iterations,
        chunk_size=args.random_chunk_size,
        max_points=args.max_points,
        sample_seed=args.sample_seed)
    if len(random_arr) < 4:
        raise RuntimeError(f'Need at least 4 random void points after selection, got {len(random_arr)}')

    cosmo = build_cosmology(h=args.h, omega_m=args.omega_m)
    x, y, zcart = radec_z_to_cartesian(random_arr['RA'], random_arr['DEC'], random_arr['Z'],
                                       cosmo=cosmo, h=args.h)
    coords = np.vstack([x, y, zcart]).T
    r_mpc = np.sqrt(x * x + y * y + zcart * zcart) / args.h
    keep_r = np.isfinite(r_mpc) & (r_mpc <= args.rmax_mpc)
    if not np.any(keep_r):
        raise RuntimeError(f'No random void points left inside r <= {args.rmax_mpc:.1f} Mpc.')
    random_arr = random_arr[keep_r]
    coords = coords[keep_r]
    r_mpc = r_mpc[keep_r]

    unique_idx = _unique_coordinate_filter(coords)
    n_duplicate = len(coords) - len(unique_idx)
    if n_duplicate > 0:
        random_arr = random_arr[unique_idx]
        coords = coords[unique_idx]
        r_mpc = r_mpc[unique_idx]

    r_values = np.asarray(random_arr['R_VALUE'], dtype=np.float32)
    neighbors, degree, edges = _build_delaunay_neighbors(coords)
    ws = run_watershed(neighbors=neighbors,
                       r_values=r_values,
                       r_threshold=args.r_threshold,
                       min_group_size=args.min_group_size,
                       mode='underdense',
                       seed_threshold=args.seed_threshold,
                       merge_strategy=args.merge_strategy,
                       merge_threshold=args.merge_threshold,
                       delta_merge=args.delta_merge,
                       boundary_policy=args.boundary_policy,
                       boundary_id=BOUNDARY_ID,
                       seed_plateau_tolerance=args.seed_plateau_tolerance)

    rand_table = _make_random_table(random_arr=random_arr,
                                    coords=coords,
                                    group_of=ws['group_of'],
                                    is_boundary=ws['is_boundary'],
                                    r_values=r_values,
                                    degree=degree)
    data_table = _make_empty_data_table()
    group_table = consolidate_group_info(data_table=data_table,
                                         rand_table=rand_table,
                                         cosmo=cosmo,
                                         h=args.h,
                                         group_col='GROUPID',
                                         min_rand_for_shape=args.min_rand_for_shape)
    point_table = build_point_membership_table(data_table=data_table,
                                               rand_table=rand_table,
                                               group_col='GROUPID',
                                               boundary_col='IS_BOUNDARY',
                                               boundary_id=BOUNDARY_ID)
    write_group_table_fits(group_table=group_table,
                           output_path=str(output_catalog),
                           tracer='BGS_RVOID',
                           cap='NGC',
                           h=args.h,
                           omega_m=args.omega_m,
                           r_threshold=args.r_threshold,
                           mode='underdense',
                           point_table=point_table,
                           overwrite=True,
                           seed_threshold=args.seed_threshold,
                           merge_strategy=args.merge_strategy,
                           merge_threshold=args.merge_threshold,
                           delta_merge=args.delta_merge,
                           boundary_id=BOUNDARY_ID,
                           watershed_stats=ws['stats'])

    plot_stats = _plot_random_void_wedge(
        args=args,
        random_arr=random_arr,
        group_table=group_table,
        group_of=ws['group_of'],
        dec0=dec0,
        slice_width_deg=slice_width_deg,
        stats={'r_mpc': r_mpc})

    print('--- BGS NGC Random-Void Watershed Wedge Summary ---')
    print(f'Random void cache: {cache_path}')
    print(f'DEC slice: [{dec_lo:.3f}, {dec_hi:.3f}] deg '
          f'(center={dec0:.3f}, width={slice_width_deg:.3f}, source={dec_source})')
    print(f'Watershed RANDITER selection: {args.watershed_iterations}')
    print(f'Cache rows scanned={selection_stats["n_scanned"]:,} '
          f'matching={selection_stats["n_matching"]:,} '
          f'returned={selection_stats["n_returned"]:,} '
          f'sampled={selection_stats["sampled"]}')
    print(f'Points after rmax/duplicate filtering: {len(random_arr):,} '
          f'(duplicates removed={n_duplicate:,})')
    print(f'Delaunay: edges={len(edges):,} median_degree={float(np.median(degree)):.1f}')
    print(f'Watershed: groups={ws["n_groups"]:,} assigned={ws["n_assigned"]:,} '
          f'boundary={ws["n_boundary_nodes"]:,} '
          f'unassigned={ws["n_unassigned_after_size_filter"]:,}')
    print(f'Group catalog rows: {len(group_table):,}')
    print(f'Groups drawn: {plot_stats["n_groups_drawn"]:,} '
          f'| assigned members drawn={plot_stats["n_members_drawn"]:,} '
          f'| random points in rmax={plot_stats["n_random_in_rmax"]:,}')
    print(f'Output catalog: {output_catalog}')
    print(f'Output PNG: {plot_stats["output_plot"]}')


if __name__ == '__main__':
    main()