import argparse, os, re
from pathlib import Path

import fitsio
import numpy as np
from astropy.cosmology import Planck18

os.environ.setdefault('MPLCONFIGDIR', '/tmp/matplotlib-cache')
os.environ.setdefault('XDG_CACHE_HOME', '/tmp')
Path(os.environ['MPLCONFIGDIR']).mkdir(parents=True, exist_ok=True)

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import ConnectionPatch

plt.style.use('dark_background')
plt.rcParams.update({'text.usetex': True})

DEFAULT_VOIDS = ('/pscratch/sd/v/vtorresg/void_catalog/'
                 'DR2_Om_2_Om0p315_h0p6736/dr2/voids_BGS_ANY_NGC.fits')
DEFAULT_OUTPUT = 'plots/bgs_ngc_wedge_void_members_boundary_inset.png'


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--void-input', default=DEFAULT_VOIDS)
    p.add_argument('--void-ext', default='VOIDS')
    p.add_argument('--output', default=DEFAULT_OUTPUT)
    p.add_argument('--distance-mode', choices=['auto', 'xyz', 'redshift'], default='auto')
    p.add_argument('--h', type=float, default=None)
    p.add_argument('--slice-width-deg', type=float, default=1.0)
    p.add_argument('--dec0', type=float, default=None)
    p.add_argument('--dec-scan-step', type=float, default=0.25)
    p.add_argument('--ra-center-deg', type=float, default=180.0)
    p.add_argument('--rmax-mpc', type=float, default=1250.0)
    p.add_argument('--chunk-size', type=int, default=1_000_000)

    p.add_argument('--member-point-size', type=float, default=0.08)
    p.add_argument('--member-alpha', type=float, default=0.75)
    p.add_argument('--members-include-rand', action='store_true')
    p.add_argument('--boundary-id', type=int, default=None)
    p.add_argument('--plot-boundary', dest='plot_boundary', action='store_true')
    p.add_argument('--no-boundary', dest='plot_boundary', action='store_false')
    p.set_defaults(plot_boundary=True)
    p.add_argument('--boundary-color', default='white')
    p.add_argument('--boundary-point-size', type=float, default=0.05)
    p.add_argument('--boundary-alpha', type=float, default=0.45)

    p.add_argument('--void-center-size', type=float, default=1.0)
    p.add_argument('--void-alpha', type=float, default=0.95)
    p.add_argument('--plot-void-centers', dest='plot_void_centers', action='store_true')
    p.add_argument('--no-void-centers', dest='plot_void_centers', action='store_false')
    p.set_defaults(plot_void_centers=True)
    p.add_argument('--void-ellipse-alpha', type=float, default=0.9)
    p.add_argument('--void-ellipse-lw', type=float, default=0.2)
    p.add_argument('--void-ellipse-npts', type=int, default=200)
    p.add_argument('--plot-void-ellipses', dest='plot_void_ellipses', action='store_true')
    p.add_argument('--no-void-ellipses', dest='plot_void_ellipses', action='store_false')
    p.set_defaults(plot_void_ellipses=True)
    p.add_argument('--void-cmap', default='turbo')

    p.add_argument('--plot-inset', dest='plot_inset', action='store_true')
    p.add_argument('--no-inset', dest='plot_inset', action='store_false')
    p.set_defaults(plot_inset=True)
    p.add_argument('--zoom-ra-min', type=float, default=None)
    p.add_argument('--zoom-ra-max', type=float, default=None)
    p.add_argument('--zoom-r-min', type=float, default=0.0)
    p.add_argument('--zoom-r-max', type=float, default=300.0)
    p.add_argument('--zoom-ra-tick-step', type=float, default=15.0)
    p.add_argument('--zoom-r-tick-step', type=float, default=100.0)
    p.add_argument('--zoom-point-scale', type=float, default=2.2)
    p.add_argument('--zoom-alpha-boost', type=float, default=0.15)
    p.add_argument('--inset-left', type=float, default=0.49)
    p.add_argument('--inset-bottom', type=float, default=0.42)
    p.add_argument('--inset-width', type=float, default=0.40)
    p.add_argument('--inset-height', type=float, default=0.40)

    p.add_argument('--dpi', type=int, default=500)
    p.add_argument('--ra-tick-step', type=float, default=15.0)
    p.add_argument('--rtick-step', type=float, default=250.0)
    p.add_argument('--bg-color', default='#000000')
    p.add_argument('--hide-legend', action='store_true')
    return p.parse_args()


def _hdu_by_name_or_index(fobj, ext):
    try:
        return fobj[ext]
    except Exception:
        try:
            return fobj[int(ext)]
        except Exception as exc:
            raise RuntimeError(f'Could not open FITS extension {ext!r}.') from exc


def _infer_h(header, path, override):
    if override is not None:
        if not np.isfinite(override) or override <= 0.0:
            raise ValueError('--h must be positive.')
        return float(override), '--h'

    for key in ('H', 'H0'):
        if key not in header:
            continue
        try:
            value = float(header[key])
            if key == 'H0' and value > 5.0:
                value /= 100.0
            if np.isfinite(value) and value > 0.0:
                return value, key
        except Exception:
            pass

    match = re.search(r'h(\d+(?:p\d+|\.\d+)?)', str(path))
    if match:
        try:
            value = float(match.group(1).replace('p', '.'))
            if np.isfinite(value) and value > 0.0:
                return value, 'path'
        except Exception:
            pass

    return 1.0, 'default'


def _infer_boundary_id(header, override):
    if override is not None:
        return int(override), '--boundary-id'
    try:
        return int(header['GIDM2']), 'GIDM2'
    except Exception:
        return -2, 'default'


def _resolve_distance_mode(cols, requested, xyz_cols, redshift_cols):
    has_xyz = all(col in cols for col in xyz_cols)
    redshift_col = next((col for col in redshift_cols if col in cols), None)
    if requested == 'auto':
        if has_xyz:
            return 'xyz', None
        if redshift_col is not None:
            return 'redshift', redshift_col
    elif requested == 'xyz':
        if has_xyz:
            return 'xyz', None
    elif requested == 'redshift':
        if redshift_col is not None:
            return 'redshift', redshift_col

    raise RuntimeError(f'Cannot use distance mode {requested!r}; available columns are {cols}.')


def _read_void_catalog(path, ext, requested_distance_mode, h_override):
    with fitsio.FITS(str(path)) as fobj:
        header = fobj[0].read_header()
        h, h_source = _infer_h(header, path, h_override)
        boundary_id, boundary_source = _infer_boundary_id(header, None)
        hdu = _hdu_by_name_or_index(fobj, ext)
        cols = hdu.get_colnames()
        mode, redshift_col = _resolve_distance_mode(cols=cols,
                                                    requested=requested_distance_mode,
                                                    xyz_cols=('X', 'Y', 'Z'),
                                                    redshift_cols=('REDSHIFT',))

        use_cols = ['VOID_ID', 'RA', 'DEC', 'R_EFF',
                    'SEMI_AXIS_A', 'SEMI_AXIS_B', 'SEMI_AXIS_C']
        if mode == 'xyz':
            use_cols.extend(['X', 'Y', 'Z'])
        else:
            use_cols.append(redshift_col)
        missing = [col for col in use_cols if col not in cols]
        if missing:
            raise RuntimeError(f'Missing columns in {ext}: {missing}')
        data = hdu.read(columns=use_cols)

    if mode == 'xyz':
        x = np.asarray(data['X'], dtype=np.float64)
        y = np.asarray(data['Y'], dtype=np.float64)
        zcart = np.asarray(data['Z'], dtype=np.float64)
        radius_mpc = np.sqrt(x * x + y * y + zcart * zcart) / h
    else:
        redshift = np.asarray(data[redshift_col], dtype=np.float64)
        radius_mpc = Planck18.comoving_distance(redshift).value

    scale = 1.0 / h
    voids = {'id': np.asarray(data['VOID_ID'], dtype=np.int64),
             'ra': np.asarray(data['RA'], dtype=np.float64),
             'dec': np.asarray(data['DEC'], dtype=np.float64),
             'r': radius_mpc,
             'reff': np.asarray(data['R_EFF'], dtype=np.float64) * scale,
             'a': np.asarray(data['SEMI_AXIS_A'], dtype=np.float64) * scale,
             'b': np.asarray(data['SEMI_AXIS_B'], dtype=np.float64) * scale,
             'c': np.asarray(data['SEMI_AXIS_C'], dtype=np.float64) * scale}
    return voids, h, h_source, boundary_id, boundary_source, mode


def _select_auto_dec0(dec, width_deg, step_deg):
    half = 0.5 * width_deg
    dec = np.asarray(dec, dtype=np.float64)
    dec = dec[np.isfinite(dec)]
    if len(dec) == 0:
        raise RuntimeError('No finite DEC values available for automatic slice selection.')

    dec_min = float(np.min(dec))
    dec_max = float(np.max(dec))
    cmin = dec_min + half
    cmax = dec_max - half
    if cmin > cmax:
        raise ValueError('Invalid DEC range for the chosen slice width.')

    centers = np.arange(cmin, cmax + 0.5 * step_deg, step_deg, dtype=np.float64)
    dec_sorted = np.sort(dec)
    left = np.searchsorted(dec_sorted, centers - half, side='left')
    right = np.searchsorted(dec_sorted, centers + half, side='right')
    counts = right - left
    best_idx = int(np.argmax(counts))
    order = np.argsort(counts)[::-1][:5]
    top5 = [(float(centers[i]), int(counts[i])) for i in order]
    return float(centers[best_idx]), top5


def _read_dec_for_auto(path, include_rand, chunk_size):
    values = []
    with fitsio.FITS(str(path)) as fobj:
        try:
            hdu = fobj['POINT_MEMBERSHIP']
        except Exception as exc:
            raise RuntimeError('POINT_MEMBERSHIP extension is required when --dec0 is omitted.') from exc
        cols = hdu.get_colnames()
        read_cols = ['DEC']
        if (not include_rand) and ('IS_DATA' in cols):
            read_cols.append('IS_DATA')
        nrows = hdu.get_nrows()
        for start in range(0, nrows, chunk_size):
            stop = min(start + chunk_size, nrows)
            rows = np.arange(start, stop, dtype=np.int64)
            arr = hdu.read(columns=read_cols, rows=rows)
            dec = np.asarray(arr['DEC'], dtype=np.float64)
            mask = np.isfinite(dec)
            if (not include_rand) and ('IS_DATA' in read_cols):
                mask &= np.asarray(arr['IS_DATA'], dtype=np.int8) == 1
            if np.any(mask):
                values.append(dec[mask])
    if not values:
        return np.array([], dtype=np.float64)
    return np.concatenate(values)


def _membership_distance(arr, mode, h):
    if mode == 'xyz':
        x = np.asarray(arr['X_CART'], dtype=np.float64)
        y = np.asarray(arr['Y_CART'], dtype=np.float64)
        zcart = np.asarray(arr['Z_CART'], dtype=np.float64)
        finite = np.isfinite(x) & np.isfinite(y) & np.isfinite(zcart)
        radius = np.sqrt(x * x + y * y + zcart * zcart) / h
        return radius, finite

    redshift = np.asarray(arr['Z'], dtype=np.float64)
    finite = np.isfinite(redshift)
    radius = np.empty(len(redshift), dtype=np.float64)
    radius[:] = np.nan
    if np.any(finite):
        radius[finite] = Planck18.comoving_distance(redshift[finite]).value
    return radius, finite


def _read_membership_points(path, mode, h, dec_lo, dec_hi, rmax_mpc, ra_center_deg,
                            include_rand, boundary_id, plotted_void_ids,
                            chunk_size, plot_boundary):
    member_theta_parts = []
    member_r_parts = []
    member_gid_parts = []
    boundary_theta_parts = []
    boundary_r_parts = []
    stats = {'rows': 0,
             'candidate_rows': 0,
             'members_plotted': 0,
             'boundary_plotted': 0,
             'unassigned_ignored': 0}

    with fitsio.FITS(str(path)) as fobj:
        try:
            hdu = fobj['POINT_MEMBERSHIP']
        except Exception as exc:
            raise RuntimeError('POINT_MEMBERSHIP extension is required for this plot.') from exc
        cols = hdu.get_colnames()
        member_mode, _ = _resolve_distance_mode(cols=cols,
                                                requested=mode,
                                                xyz_cols=('X_CART', 'Y_CART', 'Z_CART'),
                                                redshift_cols=('Z',))

        read_cols = ['GROUPID', 'RA', 'DEC']
        if 'IS_DATA' in cols:
            read_cols.append('IS_DATA')
        if 'IS_BOUNDARY' in cols:
            read_cols.append('IS_BOUNDARY')
        if member_mode == 'xyz':
            read_cols.extend(['X_CART', 'Y_CART', 'Z_CART'])
        else:
            read_cols.append('Z')
        read_cols = list(dict.fromkeys(read_cols))

        nrows = hdu.get_nrows()
        plotted_void_ids = None if plotted_void_ids is None else np.asarray(plotted_void_ids, dtype=np.int64)
        for start in range(0, nrows, chunk_size):
            stop = min(start + chunk_size, nrows)
            rows = np.arange(start, stop, dtype=np.int64)
            arr = hdu.read(columns=read_cols, rows=rows)

            gid = np.asarray(arr['GROUPID'], dtype=np.int64)
            ra = np.asarray(arr['RA'], dtype=np.float64)
            dec = np.asarray(arr['DEC'], dtype=np.float64)
            radius, finite_radius = _membership_distance(arr, member_mode, h)

            base = np.isfinite(ra) & np.isfinite(dec) & finite_radius
            if (not include_rand) and ('IS_DATA' in read_cols):
                base &= np.asarray(arr['IS_DATA'], dtype=np.int8) == 1
            base &= (dec >= dec_lo) & (dec <= dec_hi)
            base &= radius <= rmax_mpc

            stats['rows'] += len(gid)
            stats['candidate_rows'] += int(np.count_nonzero(base))
            stats['unassigned_ignored'] += int(np.count_nonzero(base & (gid == -1)))

            member_mask = base & (gid >= 0)
            if np.any(member_mask):
                member_idx = np.flatnonzero(member_mask)
                if plotted_void_ids is not None:
                    member_idx = member_idx[np.isin(gid[member_idx], plotted_void_ids)]
                if len(member_idx) > 0:
                    member_theta_parts.append(np.deg2rad(ra[member_idx] - ra_center_deg))
                    member_r_parts.append(radius[member_idx])
                    member_gid_parts.append(gid[member_idx])
                    stats['members_plotted'] += len(member_idx)

            if plot_boundary:
                boundary_mask = base & (gid == int(boundary_id))
                if 'IS_BOUNDARY' in read_cols:
                    boundary_mask |= base & (np.asarray(arr['IS_BOUNDARY'], dtype=np.int8) != 0)
                if np.any(boundary_mask):
                    boundary_idx = np.flatnonzero(boundary_mask)
                    boundary_theta_parts.append(np.deg2rad(ra[boundary_idx] - ra_center_deg))
                    boundary_r_parts.append(radius[boundary_idx])
                    stats['boundary_plotted'] += len(boundary_idx)

    members = {'theta': _concat_or_empty(member_theta_parts, np.float64),
               'r': _concat_or_empty(member_r_parts, np.float64),
               'gid': _concat_or_empty(member_gid_parts, np.int64)}
    boundary = {'theta': _concat_or_empty(boundary_theta_parts, np.float64),
                'r': _concat_or_empty(boundary_r_parts, np.float64)}
    return members, boundary, stats


def _concat_or_empty(parts, dtype):
    if not parts:
        return np.array([], dtype=dtype)
    return np.concatenate(parts).astype(dtype, copy=False)


def _filter_voids(voids, dec_lo, dec_hi, rmax_mpc):
    mask = (np.asarray(voids['id']) >= 0)
    mask &= np.isfinite(voids['ra']) & np.isfinite(voids['dec']) & np.isfinite(voids['r'])
    mask &= (voids['dec'] >= dec_lo) & (voids['dec'] <= dec_hi)
    mask &= voids['r'] <= rmax_mpc
    return {key: np.asarray(value)[mask] for key, value in voids.items()}


def _build_void_color_map(void_ids, cmap_name):
    unique = np.unique(np.asarray(void_ids, dtype=np.int64))
    unique = unique[unique >= 0]
    if len(unique) == 0:
        return {}
    cmap = plt.get_cmap(cmap_name)
    denom = max(len(unique) - 1, 1)
    return {int(vid): cmap(i / denom) for i, vid in enumerate(unique.tolist())}


def _colors_for_groups(group_ids, color_map):
    fallback = (1.0, 1.0, 1.0, 0.9)
    return np.array([color_map.get(int(gid), fallback) for gid in group_ids])


def _ellipse_theta_r(theta0, r0, a, b, npts):
    t = np.linspace(0.0, 2.0 * np.pi, npts, dtype=np.float64)
    dr = a * np.cos(t)
    dtan = b * np.sin(t)
    r_safe = r0 if r0 > 1.0e-6 else 1.0e-6
    theta = theta0 + (dtan / r_safe)
    radius = r0 + dr
    return theta, radius


def _radial_label_rotation(ax, theta_deg, r_min, r_max):
    theta = np.deg2rad(theta_deg)
    p0 = ax.transData.transform((theta, float(r_min)))
    p1 = ax.transData.transform((theta, float(r_max)))
    return float(np.degrees(np.arctan2(p1[1] - p0[1], p1[0] - p0[0])))


def _plot_voids(ax, voids, void_theta, color_map, args, point_scale=1.0,
                alpha_boost=0.0, zorder_base=4):
    for i in range(len(voids['id'])):
        vid = int(voids['id'][i])
        color = color_map.get(vid, (1.0, 1.0, 1.0, 0.9))
        theta0 = float(void_theta[i])
        r0 = float(voids['r'][i])
        a = float(voids['a'][i]) if np.isfinite(voids['a'][i]) and voids['a'][i] > 0 else float(voids['reff'][i])
        b = float(voids['b'][i]) if np.isfinite(voids['b'][i]) and voids['b'][i] > 0 else float(voids['reff'][i])
        if not np.isfinite(a) or a <= 0:
            a = float(voids['reff'][i]) if np.isfinite(voids['reff'][i]) and voids['reff'][i] > 0 else 0.0
        if not np.isfinite(b) or b <= 0:
            b = float(voids['reff'][i]) if np.isfinite(voids['reff'][i]) and voids['reff'][i] > 0 else 0.0
        if r0 <= 0 or (a <= 0 and b <= 0):
            continue

        if args.plot_void_ellipses:
            theta_e, r_e = _ellipse_theta_r(theta0, r0, a, b, args.void_ellipse_npts)
            ax.plot(theta_e, r_e, color=color, linewidth=args.void_ellipse_lw,
                    alpha=min(1.0, args.void_ellipse_alpha + alpha_boost),
                    zorder=zorder_base)

        if args.plot_void_centers:
            ax.scatter([theta0], [r0], s=args.void_center_size * point_scale, c=[color],
                       alpha=min(1.0, args.void_alpha + alpha_boost), linewidths=0,
                       zorder=zorder_base + 1)


def _axis_theta_limits(void_theta, members, boundary, ra_center_deg):
    parts = []
    if len(void_theta) > 0:
        parts.append(np.rad2deg(void_theta))
    if len(members['theta']) > 0:
        parts.append(np.rad2deg(members['theta']))
    if len(boundary['theta']) > 0:
        parts.append(np.rad2deg(boundary['theta']))
    if not parts:
        raise RuntimeError('Nothing to plot after DEC and radius filtering.')
    theta_all = np.concatenate(parts)
    theta_min = float(np.nanmin(theta_all))
    theta_max = float(np.nanmax(theta_all))
    if theta_min == theta_max:
        theta_min -= 1.0
        theta_max += 1.0
    pad = min(2.0, 0.04 * (theta_max - theta_min))
    return theta_min - pad, theta_max + pad, theta_min + ra_center_deg, theta_max + ra_center_deg


def _add_inset(fig, ax, voids, void_theta, color_map, members, member_colors,
               boundary, args, theta_min_deg, theta_max_deg):
    axins = fig.add_axes((args.inset_left, args.inset_bottom,
                          args.inset_width, args.inset_height),
                         projection='polar', facecolor=args.bg_color)
    axins.set_theta_zero_location('N')
    axins.set_theta_direction(1)

    if len(boundary['theta']) > 0:
        bdeg = np.rad2deg(boundary['theta'])
        bmask = ((bdeg >= theta_min_deg) & (bdeg <= theta_max_deg)
                 & (boundary['r'] >= args.zoom_r_min) & (boundary['r'] <= args.zoom_r_max))
        if np.any(bmask):
            axins.scatter(boundary['theta'][bmask], boundary['r'][bmask],
                          s=args.boundary_point_size * args.zoom_point_scale,
                          c=args.boundary_color,
                          alpha=min(1.0, args.boundary_alpha + args.zoom_alpha_boost),
                          linewidths=0, rasterized=True, zorder=1)

    if len(members['theta']) > 0:
        mdeg = np.rad2deg(members['theta'])
        mmask = ((mdeg >= theta_min_deg) & (mdeg <= theta_max_deg)
                 & (members['r'] >= args.zoom_r_min) & (members['r'] <= args.zoom_r_max))
        if np.any(mmask):
            axins.scatter(members['theta'][mmask], members['r'][mmask],
                          s=args.member_point_size * args.zoom_point_scale,
                          c=member_colors[mmask],
                          alpha=min(1.0, args.member_alpha + args.zoom_alpha_boost),
                          linewidths=0, rasterized=True, zorder=2)

    vdeg = np.rad2deg(void_theta)
    vmask = ((vdeg >= theta_min_deg) & (vdeg <= theta_max_deg)
             & (voids['r'] >= args.zoom_r_min) & (voids['r'] <= args.zoom_r_max))
    if np.any(vmask):
        zoom_voids = {key: np.asarray(value)[vmask] for key, value in voids.items()}
        _plot_voids(axins, zoom_voids, void_theta[vmask], color_map, args,
                    point_scale=args.zoom_point_scale,
                    alpha_boost=args.zoom_alpha_boost, zorder_base=4)

    axins.set_thetamin(theta_min_deg)
    axins.set_thetamax(theta_max_deg)
    axins.set_ylim(args.zoom_r_min, args.zoom_r_max)
    axins.grid(color='0.50', alpha=0.30, linewidth=0.6)
    axins.spines['polar'].set_color('0.75')
    axins.spines['polar'].set_linewidth(1.0)

    dtheta = theta_max_deg - theta_min_deg
    ra_tick_values = np.arange(np.ceil((theta_min_deg + args.ra_center_deg) / args.zoom_ra_tick_step) * args.zoom_ra_tick_step,
                               np.floor((theta_max_deg + args.ra_center_deg) / args.zoom_ra_tick_step) * args.zoom_ra_tick_step
                               + 0.5 * args.zoom_ra_tick_step,
        args.zoom_ra_tick_step)
    if len(ra_tick_values) > 0:
        axins.set_xticks(np.deg2rad(ra_tick_values - args.ra_center_deg))
        axins.set_xticklabels([rf'${tick:.0f}^\circ$' for tick in ra_tick_values], fontsize=8)

    if args.zoom_r_tick_step > 0:
        rticks = np.arange(np.ceil(args.zoom_r_min / args.zoom_r_tick_step) * args.zoom_r_tick_step,
                           args.zoom_r_max + 0.5 * args.zoom_r_tick_step,
                           args.zoom_r_tick_step)
        if len(rticks) > 0:
            max_r_ticks = 4
            if len(rticks) > max_r_ticks:
                idx = np.linspace(0, len(rticks) - 1, max_r_ticks).round().astype(int)
                rticks = rticks[idx]
            axins.set_yticks(rticks)
            axins.set_yticklabels([rf'${int(tick):d}$' for tick in rticks], fontsize=8)
            axins.set_rlabel_position(theta_min_deg + 0.12 * dtheta)

    axins.tick_params(axis='x', colors='white', labelsize=8, pad=-3)
    axins.tick_params(axis='y', colors='white', labelsize=8, pad=-2)
    axins.text(0.5, 0.85, r'$\alpha\;(\mathrm{RA})$',
               transform=axins.transAxes, ha='center', va='bottom',
               fontsize=12, color='white', rotation=6,
               rotation_mode='anchor', clip_on=False)

    r_axis_theta_deg = theta_min_deg + 0.2 * dtheta
    r_axis_rotation = _radial_label_rotation(axins, r_axis_theta_deg,
                                             args.zoom_r_min, args.zoom_r_max)
    r_label_radius = args.zoom_r_min + 0.3 * (args.zoom_r_max - args.zoom_r_min)
    axins.text(np.deg2rad(r_axis_theta_deg), r_label_radius,
               r'$r\,[\mathrm{Mpc}]$', fontsize=10, color='white',
               rotation=r_axis_rotation, rotation_mode='anchor',
               ha='left', va='center', clip_on=False)

    th1 = np.deg2rad(theta_min_deg)
    th2 = np.deg2rad(theta_max_deg)
    ax.plot([th1, th1], [args.zoom_r_min, args.zoom_r_max], color='white', lw=0.9, alpha=0.9)
    ax.plot([th2, th2], [args.zoom_r_min, args.zoom_r_max], color='white', lw=0.9, alpha=0.9)
    ax.plot(np.linspace(th1, th2, 400), np.full(400, args.zoom_r_max),
            color='white', lw=0.9, alpha=0.9)

    fig.add_artist(ConnectionPatch(xyA=(th1, args.zoom_r_max), coordsA=axins.transData,
                                   xyB=(th1, args.zoom_r_max), coordsB=ax.transData,
                                   color='white', lw=0.8, alpha=0.6))
    fig.add_artist(ConnectionPatch(xyA=(th2, args.zoom_r_max), coordsA=axins.transData,
                                   xyB=(th2, args.zoom_r_max), coordsB=ax.transData,
                                   color='white', lw=0.8, alpha=0.6))
    return axins


def main():
    args = parse_args()
    void_path = Path(args.void_input)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if args.chunk_size <= 0:
        raise ValueError('--chunk-size must be positive.')

    voids_all, h_value, h_source, header_boundary_id, boundary_source, mode = _read_void_catalog(path=void_path,
                                                                                                 ext=args.void_ext,
                                                                                                 requested_distance_mode=args.distance_mode,
                                                                                                 h_override=args.h)
    boundary_id = header_boundary_id if args.boundary_id is None else int(args.boundary_id)
    if args.boundary_id is not None:
        boundary_source = '--boundary-id'

    if args.dec0 is None:
        dec_values = _read_dec_for_auto(void_path, args.members_include_rand, args.chunk_size)
        dec0, top5 = _select_auto_dec0(dec_values, args.slice_width_deg, args.dec_scan_step)
    else:
        dec0 = float(args.dec0)
        top5 = []
    half = 0.5 * args.slice_width_deg
    dec_lo = dec0 - half
    dec_hi = dec0 + half

    voids = _filter_voids(voids_all, dec_lo, dec_hi, args.rmax_mpc)
    plotted_void_ids = voids['id']
    members, boundary, member_stats = _read_membership_points(path=void_path,
                                                              mode=mode,
                                                              h=h_value,
                                                              dec_lo=dec_lo,
                                                              dec_hi=dec_hi,
                                                              rmax_mpc=args.rmax_mpc,
                                                              ra_center_deg=args.ra_center_deg,
                                                              include_rand=args.members_include_rand,
                                                              boundary_id=boundary_id,
                                                              plotted_void_ids=plotted_void_ids,
                                                              chunk_size=args.chunk_size,
                                                              plot_boundary=args.plot_boundary)

    color_ids = plotted_void_ids
    if len(members['gid']) > 0:
        color_ids = np.concatenate([plotted_void_ids, members['gid']])
    color_map = _build_void_color_map(color_ids, args.void_cmap)
    member_colors = _colors_for_groups(members['gid'], color_map) if len(members['gid']) > 0 else np.empty((0, 4))
    void_theta = np.deg2rad(voids['ra'] - args.ra_center_deg)
    theta_min_deg, theta_max_deg, ra_min, ra_max = _axis_theta_limits(void_theta=void_theta,
                                                                      members=members,
                                                                      boundary=boundary,
                                                                      ra_center_deg=args.ra_center_deg)

    fig = plt.figure(figsize=(11.0, 7.2), dpi=args.dpi, facecolor=args.bg_color)
    ax = fig.add_subplot(111, projection='polar', facecolor=args.bg_color)
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(1)

    if len(boundary['theta']) > 0:
        ax.scatter(boundary['theta'], boundary['r'], s=args.boundary_point_size,
                   c=args.boundary_color, alpha=args.boundary_alpha,
                   linewidths=0, rasterized=True, zorder=1)
    if len(members['theta']) > 0:
        ax.scatter(members['theta'], members['r'], s=args.member_point_size,
                   c=member_colors, alpha=args.member_alpha,
                   linewidths=0, rasterized=True, zorder=2)

    _plot_voids(ax, voids, void_theta, color_map, args, zorder_base=4)

    ax.set_thetamin(theta_min_deg)
    ax.set_thetamax(theta_max_deg)
    ax.set_ylim(0.0, args.rmax_mpc)
    ax.grid(color='0.50', alpha=0.30, linewidth=0.7)
    ax.spines['polar'].set_color('0.75')
    ax.spines['polar'].set_linewidth(1.0)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.tick_params(axis='both', which='both', length=0)
    ax.set_title(rf'$\mathrm{{BGS\ NGC\ void\ members}},\ \delta={args.slice_width_deg:.0f}^\circ'
                 rf'\ (\mathrm{{DEC\ center}}={dec0:.2f}^\circ)$',
                 fontsize=17, pad=28, color='white')
    ax.text(0.2, 0.215, rf'$\delta = {args.slice_width_deg:.0f}^\circ$',
            transform=ax.transAxes, ha='left', va='center',
            fontsize=20, color='white', rotation=-7, clip_on=False)

    if args.plot_inset:
        theta_zoom_min_deg = theta_min_deg if args.zoom_ra_min is None else args.zoom_ra_min - args.ra_center_deg
        theta_zoom_max_deg = theta_max_deg if args.zoom_ra_max is None else args.zoom_ra_max - args.ra_center_deg
        if theta_zoom_min_deg >= theta_zoom_max_deg:
            raise ValueError('Inset RA range must satisfy --zoom-ra-min < --zoom-ra-max.')
        _add_inset(fig, ax, voids, void_theta, color_map, members, member_colors,
                   boundary, args, theta_zoom_min_deg, theta_zoom_max_deg)

    if not args.hide_legend:
        handles = [Line2D([0], [0], marker='o', linestyle='', markersize=6,
                   markerfacecolor='none', markeredgecolor='white',
                   label='Void members', alpha=0.9)]
        if args.plot_void_ellipses:
            handles.append(Line2D([0], [0], color='white',
                                  linewidth=args.void_ellipse_lw,
                                  label='Void ellipses'))
        if args.plot_boundary:
            handles.append(Line2D([0], [0], marker='o', linestyle='', markersize=5,
                                  markerfacecolor=args.boundary_color,
                                  markeredgecolor='none',
                                  label='Boundary points',
                                  alpha=max(args.boundary_alpha, 0.6)))
        leg = fig.legend(handles=handles, labels=[h.get_label() for h in handles],
                         loc='upper right', bbox_to_anchor=(0.985, 0.965),
                         framealpha=0.6, ncol=1, fontsize=10)
        for text in leg.get_texts():
            text.set_color('white')

    fig.subplots_adjust(left=0.04, right=0.98, bottom=0.06, top=0.90)
    fig.savefig(out_path, dpi=args.dpi)
    plt.close(fig)

    print('--- BGS NGC Wedge (Void Members + Boundary) Summary ---')
    print(f'Void catalog: {void_path}')
    print(f'Void extension: {args.void_ext}')
    print(f'h: {h_value:.6f} ({h_source})')
    print(f'Distance mode: {mode}')
    print(f'Boundary GROUPID: {boundary_id} ({boundary_source})')
    print(f'Chosen DEC slice: [{dec_lo:.3f}, {dec_hi:.3f}] deg (center={dec0:.3f})')
    if top5:
        print('Top DEC centers by membership population (deg, count):', top5)
    print(f'RA span plotted: [{ra_min:.3f}, {ra_max:.3f}] deg')
    print(f'Void groups plotted: {len(voids["id"]):,} / {len(voids_all["id"]):,}')
    print(f'Member points plotted: {len(members["gid"]):,}')
    print(f'Boundary points plotted: {len(boundary["theta"]):,}')
    print(f'Candidate membership rows in slice/rmax: {member_stats["candidate_rows"]:,}')
    print(f'Unassigned points ignored: {member_stats["unassigned_ignored"]:,}')
    print(f'Output PNG: {out_path}')


if __name__ == '__main__':
    main()