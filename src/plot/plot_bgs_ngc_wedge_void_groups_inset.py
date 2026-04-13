import argparse, os
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

DEFAULT_RAW = '/pscratch/sd/v/vtorresg/cosmic-web/dr2/raw/zone_NGC_BGS.fits.gz'
DEFAULT_PROB = ('/pscratch/sd/v/vtorresg/cosmic-web/dr2/probabilities/bgs/ngc/'
                'zone_NGC_BGS_probability_iterdata.fits.gz')
DEFAULT_CACHE = 'cache/zone_NGC_BGS_target_class_cache.fits.gz'
DEFAULT_VOIDS = '/pscratch/sd/v/vtorresg/astra-voids/v1.0/voids_BGS_ANY_NGC.fits'
DEFAULT_OUTPUT = 'plots/bgs_ngc_wedge_void_groups_inset.png'

CLASS_NAMES = np.array(['Void', 'Sheet', 'Filament', 'Knot'], dtype='U8')
CLASS_COLORS = {'Void': 'none',
                                'Sheet': 'white',
                                'Filament': 'white',
                                'Knot': 'white',}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--raw-input', default=DEFAULT_RAW)
    p.add_argument('--prob-input', default=DEFAULT_PROB)
    p.add_argument('--class-cache', default=DEFAULT_CACHE)
    p.add_argument('--rebuild-cache', action='store_true')
    p.add_argument('--void-input', default=DEFAULT_VOIDS)
    p.add_argument('--output', default=DEFAULT_OUTPUT)
    p.add_argument('--slice-width-deg', type=float, default=6.0)
    p.add_argument('--dec0', type=float, default=None)
    p.add_argument('--dec-scan-step', type=float, default=0.25)
    p.add_argument('--ra-center-deg', type=float, default=180.0)
    p.add_argument('--rmax-mpc', type=float, default=1250.0)

    p.add_argument('--point-size', dest='galaxy_point_size', type=float, default=0.03)
    p.add_argument('--galaxy-point-size', dest='galaxy_point_size', type=float, default=0.03)
    p.add_argument('--alpha', type=float, default=0.45)
    p.add_argument('--void-center-size', type=float, default=3.0)
    p.add_argument('--void-alpha', type=float, default=0.95)
    p.add_argument('--void-ellipse-alpha', type=float, default=0.9)
    p.add_argument('--void-ellipse-lw', type=float, default=0.4)
    p.add_argument('--void-ellipse-npts', type=int, default=200)

    p.add_argument('--member-point-size', type=float, default=0.08)
    p.add_argument('--member-alpha', type=float, default=0.7)
    p.add_argument('--plot-members', dest='plot_members', action='store_true')
    p.add_argument('--no-members', dest='plot_members', action='store_false')
    p.set_defaults(plot_members=True)
    p.add_argument('--members-include-rand', action='store_true')

    p.add_argument('--plot-void-ellipses', dest='plot_void_ellipses', action='store_true')
    p.add_argument('--no-void-ellipses', dest='plot_void_ellipses', action='store_false')
    p.set_defaults(plot_void_ellipses=True)

    p.add_argument('--void-cmap', default='turbo')
    p.add_argument('--plot-galaxies', action='store_true')

    p.add_argument('--void-z-col', default=None)
    p.add_argument('--void-use-xyz', action='store_true')

    p.add_argument('--dpi', type=int, default=500)
    p.add_argument('--ra-tick-step', type=float, default=15.0)
    p.add_argument('--rtick-step', type=float, default=250.0)
    p.add_argument('--data-randiter', type=int, default=-1)
    p.add_argument('--z-grid-size', type=int, default=4096)
    p.add_argument('--bg-color', default='#000000')
    p.add_argument('--hide-legend', action='store_true')
    p.add_argument('--plot-void-class', action='store_true')

    p.add_argument('--zoom-ra-min', type=float, default=214.0)
    p.add_argument('--zoom-ra-max', type=float, default=226.0)
    p.add_argument('--zoom-r-min', type=float, default=0.0)
    p.add_argument('--zoom-r-max', type=float, default=300.0)
    p.add_argument('--zoom-ra-tick-step', type=float, default=5.0)
    p.add_argument('--zoom-r-tick-step', type=float, default=100.0)
    p.add_argument('--zoom-point-scale', type=float, default=2.2)
    p.add_argument('--zoom-galaxy-point-scale', type=float, default=0.3)
    p.add_argument('--zoom-alpha-boost', type=float, default=0.15)
    p.add_argument('--inset-left', type=float, default=0.61)
    p.add_argument('--inset-bottom', type=float, default=0.56)
    p.add_argument('--inset-width', type=float, default=0.31)
    p.add_argument('--inset-height', type=float, default=0.31)

    return p.parse_args()


def find_first_non_data_row(hdu, data_randiter):
    nrows = hdu.get_nrows()
    lo, hi = 0, nrows
    while lo < hi:
        mid = (lo + hi) // 2
        value = int(hdu.read(columns=['RANDITER'], rows=[mid])['RANDITER'][0])
        if value == data_randiter:
            lo = mid + 1
        else:
            hi = mid
    return lo


def select_auto_dec0(dec, width_deg, step_deg):
    half = 0.5 * width_deg
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
    return float(centers[best_idx]), centers, counts


def build_class_cache(prob_path, cache_path):
    with fitsio.FITS(str(prob_path)) as fobj:
        hdu = fobj[1]
        parr = hdu.read(columns=['TARGETID', 'PVOID', 'PSHEET', 'PFILAMENT', 'PKNOT'])

    tid = np.asarray(parr['TARGETID'], dtype=np.int64)
    probs = np.vstack([np.asarray(parr['PVOID'], dtype=np.float32),
                       np.asarray(parr['PSHEET'], dtype=np.float32),
                       np.asarray(parr['PFILAMENT'], dtype=np.float32),
                       np.asarray(parr['PKNOT'], dtype=np.float32),]).T
    class_code = np.argmax(probs, axis=1).astype(np.uint8)
    class_bytes = np.array([b'Void', b'Sheet', b'Filament', b'Knot'], dtype='S8')
    class_str = class_bytes[class_code]

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    out = np.empty(len(tid), dtype=[('TARGETID', 'i8'), ('CLASS_CODE', 'u1'), ('CLASS', 'S8')])
    out['TARGETID'] = tid
    out['CLASS_CODE'] = class_code
    out['CLASS'] = class_str
    fitsio.write(str(cache_path), out, clobber=True)
    return tid, class_code


def load_or_build_class_cache(prob_path, cache_path, rebuild):
    if (not rebuild) and cache_path.exists():
        carr = fitsio.read(str(cache_path), columns=['TARGETID', 'CLASS_CODE'])
        return np.asarray(carr['TARGETID'], dtype=np.int64), np.asarray(carr['CLASS_CODE'], dtype=np.uint8), 'loaded'

    tid, class_code = build_class_cache(prob_path=prob_path, cache_path=cache_path)
    return tid, class_code, 'rebuilt'


def map_classes_to_raw(raw_tid, class_tid, class_code):
    if len(raw_tid) == len(class_tid) and np.array_equal(raw_tid, class_tid):
        return class_code.copy(), 0

    order = np.argsort(class_tid)
    stid = class_tid[order]
    scode = class_code[order]
    idx = np.searchsorted(stid, raw_tid)
    valid = idx < len(stid)
    valid_match = np.zeros_like(valid)
    valid_match[valid] = stid[idx[valid]] == raw_tid[valid]
    valid = valid_match

    out = np.full(len(raw_tid), 255, dtype=np.uint8)
    out[valid] = scode[idx[valid]]
    misses = int((~valid).sum())
    return out, misses


def _parse_h_from_header(header):
    for key in ('H', 'H0'):
        try:
            val = header.get(key)
        except Exception:
            val = None
        if val is None:
            continue
        try:
            h_val = float(val)
            if key == 'H0' and h_val > 5.0:
                h_val = h_val / 100.0
            if np.isfinite(h_val) and h_val > 0.0:
                return h_val, key
        except Exception:
            if isinstance(val, str):
                import re

                match = re.search(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', val)
                if match:
                    try:
                        h_val = float(match.group(0))
                        if key == 'H0' and h_val > 5.0:
                            h_val = h_val / 100.0
                        if np.isfinite(h_val) and h_val > 0.0:
                            return h_val, key
                    except Exception:
                        pass
    return 1.0, ''


def _read_voids_table(voids_path, z_col = None, use_xyz = False):
    with fitsio.FITS(str(voids_path)) as fobj:
        header = fobj[0].read_header()
        h, h_key = _parse_h_from_header(header)
        try:
            hdu = fobj['VOIDS']
        except Exception:
            hdu = fobj[1]
        cols = hdu.get_colnames()
        if use_xyz:
            z_kind = 'xyz'
            z_col_final = 'Z'
        elif z_col is not None:
            z_kind = 'redshift'
            z_col_final = z_col
        elif 'REDSHIFT' in cols:
            z_kind = 'redshift'
            z_col_final = 'REDSHIFT'
        elif 'Z' in cols and not (('X' in cols) and ('Y' in cols)):
            z_kind = 'redshift'
            z_col_final = 'Z'
        elif ('X' in cols) and ('Y' in cols) and ('Z' in cols):
            z_kind = 'xyz'
            z_col_final = 'Z'
        else:
            raise RuntimeError('Missing redshift column. Provide --void-z-col or --void-use-xyz if X,Y,Z exist.')

        use_cols = ['VOID_ID', 'RA', 'DEC', 'R_EFF', 'SEMI_AXIS_A', 'SEMI_AXIS_B', 'SEMI_AXIS_C']
        if z_kind == 'redshift':
            use_cols.append(z_col_final)
        else:
            use_cols.extend(['X', 'Y', 'Z'])

        missing = [c for c in use_cols if c not in cols]
        if missing:
            raise RuntimeError(f'Missing columns in void catalog: {missing}')
        data = hdu.read(columns=use_cols)

    colmap = {'void_id': 'VOID_ID',
              'ra': 'RA',
              'dec': 'DEC',
              'r_eff': 'R_EFF',
              'a': 'SEMI_AXIS_A',
              'b': 'SEMI_AXIS_B',
              'c': 'SEMI_AXIS_C',}
    if z_kind == 'redshift':
        colmap['z'] = z_col_final
    else:
        colmap['x'] = 'X'
        colmap['y'] = 'Y'
        colmap['z_cart'] = 'Z'
    return data, h, h_key, z_kind, colmap


def _build_void_color_map(void_ids, cmap_name):
    unique = np.unique(void_ids.astype(np.int64))
    unique = unique[unique >= 0]
    n = len(unique)
    cmap = plt.get_cmap(cmap_name, max(n, 1))
    color_map = {}
    if n == 0:
        return color_map
    for i, vid in enumerate(unique.tolist()):
        color_map[int(vid)] = cmap(i / max(n - 1, 1))
    return color_map


def _ellipse_theta_r(theta0, r0, a, b, npts):
    t = np.linspace(0.0, 2.0 * np.pi, npts, dtype=np.float64)
    dr = a * np.cos(t)
    dtan = b * np.sin(t)
    r_safe = r0 if r0 > 1e-6 else 1e-6
    theta = theta0 + (dtan / r_safe)
    r = r0 + dr
    return theta, r


def _xyz_to_radec(x, y, z):
    r = np.sqrt(x * x + y * y + z * z)
    ra = np.degrees(np.arctan2(y, x)) % 360.0
    dec = np.degrees(np.arcsin(np.clip(z / np.where(r > 0, r, 1.0), -1.0, 1.0)))
    return ra, dec, r


def _read_point_membership(voids_path):
    with fitsio.FITS(str(voids_path)) as fobj:
        try:
            hdu = fobj['POINT_MEMBERSHIP']
        except Exception:
            return None, None
        cols = hdu.get_colnames()

        use_cols = ['GROUPID']
        if 'IS_DATA' in cols:
            use_cols.append('IS_DATA')
        if 'RA' in cols and 'DEC' in cols:
            use_cols.extend(['RA', 'DEC'])
        if 'Z' in cols:
            use_cols.append('Z')
        if 'X_CART' in cols and 'Y_CART' in cols and 'Z_CART' in cols:
            use_cols.extend(['X_CART', 'Y_CART', 'Z_CART'])

        use_cols = list(dict.fromkeys(use_cols))
        data = hdu.read(columns=use_cols)

    colmap = {'groupid': 'GROUPID'}
    if 'IS_DATA' in cols:
        colmap['is_data'] = 'IS_DATA'
    if 'RA' in cols and 'DEC' in cols:
        colmap['ra'] = 'RA'
        colmap['dec'] = 'DEC'
    if 'Z' in cols:
        colmap['z'] = 'Z'
    if 'X_CART' in cols and 'Y_CART' in cols and 'Z_CART' in cols:
        colmap['x'] = 'X_CART'
        colmap['y'] = 'Y_CART'
        colmap['z_cart'] = 'Z_CART'

    return data, colmap


def add_void_inset(fig, ax, void_theta, void_r, void_ids, void_a, void_b, void_reff,
                   void_colors, ra_center_deg, theta_zoom_min_deg, theta_zoom_max_deg,
                   r_zoom_min, r_zoom_max, inset_rect, point_size, alpha, ellipse_lw,
                   ellipse_alpha, ellipse_npts, ra_tick_step, rtick_step, bg_color,
                   plot_ellipses=True, galaxy_theta=None, galaxy_r=None, galaxy_cls=None,
                   class_names=None, class_colors=None, galaxy_point_size=None,
                   member_theta=None, member_r=None, member_colors=None,
                   member_point_size=None, member_alpha=None):
    axins = fig.add_axes(inset_rect, projection='polar', facecolor=bg_color)
    axins.set_theta_zero_location('N')
    axins.set_theta_direction(1)

    theta_deg = np.rad2deg(void_theta)
    mzoom = ((theta_deg >= theta_zoom_min_deg)
              & (theta_deg <= theta_zoom_max_deg)
              & (void_r >= r_zoom_min)
              & (void_r <= r_zoom_max))

    if galaxy_theta is not None and galaxy_r is not None and galaxy_cls is not None:
        gdeg = np.rad2deg(galaxy_theta)
        gzoom = ((gdeg >= theta_zoom_min_deg)
                  & (gdeg <= theta_zoom_max_deg)
                  & (galaxy_r >= r_zoom_min)
                  & (galaxy_r <= r_zoom_max))
        if class_names is not None and class_colors is not None:
            for code, cname in enumerate(class_names):
                mask = gzoom & (galaxy_cls == code)
                if np.any(mask):
                    axins.scatter(galaxy_theta[mask], galaxy_r[mask],
                                  s=galaxy_point_size if galaxy_point_size is not None else point_size,
                                  c=class_colors[cname], alpha=alpha, linewidths=0,
                                  rasterized=True, zorder=1,)

    if member_theta is not None and member_r is not None and member_colors is not None:
        mdeg = np.rad2deg(member_theta)
        mzoom2 = ((mdeg >= theta_zoom_min_deg)
                   & (mdeg <= theta_zoom_max_deg)
                   & (member_r >= r_zoom_min)
                   & (member_r <= r_zoom_max))
        if np.any(mzoom2):
            axins.scatter(member_theta[mzoom2], member_r[mzoom2], s=member_point_size,
                          c=member_colors[mzoom2], alpha=member_alpha, linewidths=0,
                          rasterized=True, zorder=2,)

    for i in np.flatnonzero(mzoom):
        vid = int(void_ids[i])
        color = void_colors.get(vid, (1.0, 1.0, 1.0, 0.9))
        theta0 = float(void_theta[i])
        r0 = float(void_r[i])
        a = float(void_a[i]) if np.isfinite(void_a[i]) and void_a[i] > 0 else float(void_reff[i])
        b = float(void_b[i]) if np.isfinite(void_b[i]) and void_b[i] > 0 else float(void_reff[i])
        if not np.isfinite(a) or a <= 0:
            a = float(void_reff[i]) if np.isfinite(void_reff[i]) and void_reff[i] > 0 else 0.0
        if not np.isfinite(b) or b <= 0:
            b = float(void_reff[i]) if np.isfinite(void_reff[i]) and void_reff[i] > 0 else 0.0
        if r0 <= 0 or (a <= 0 and b <= 0):
            continue

        axins.scatter([theta0], [r0], s=point_size, c=[color], alpha=alpha, linewidths=0, zorder=5)

        if plot_ellipses:
            theta_e, r_e = _ellipse_theta_r(theta0=theta0, r0=r0, a=a, b=b, npts=ellipse_npts)
            axins.plot(theta_e, r_e, color=color, linewidth=ellipse_lw, alpha=ellipse_alpha, zorder=4)

    axins.set_thetamin(theta_zoom_min_deg)
    axins.set_thetamax(theta_zoom_max_deg)
    axins.set_ylim(r_zoom_min, r_zoom_max)

    axins.grid(color='0.50', alpha=0.30, linewidth=0.6)
    axins.spines['polar'].set_color('0.75')
    axins.spines['polar'].set_linewidth(1.0)

    dtheta = theta_zoom_max_deg - theta_zoom_min_deg

    ra_tick_values = np.arange(np.ceil((theta_zoom_min_deg + ra_center_deg) / ra_tick_step) * ra_tick_step,
                               np.floor((theta_zoom_max_deg + ra_center_deg) / ra_tick_step) * ra_tick_step + 0.5 * ra_tick_step,
                               ra_tick_step,)

    if len(ra_tick_values) > 0:
        max_ra_ticks = 7
        if len(ra_tick_values) > max_ra_ticks:
            idx = np.linspace(0, len(ra_tick_values) - 1, max_ra_ticks).round().astype(int)
            ra_tick_values = ra_tick_values[idx]

        axins.set_xticks(np.deg2rad(ra_tick_values - ra_center_deg))
        axins.set_xticklabels([rf'${t:.0f}^\circ$' for t in ra_tick_values], fontsize=8)

    rticks = np.arange(np.ceil(r_zoom_min / rtick_step) * rtick_step, r_zoom_max + 0.5 * rtick_step, rtick_step,)

    if len(rticks) > 0:
        max_r_ticks = 4
        if len(rticks) > max_r_ticks:
            idx = np.linspace(0, len(rticks) - 1, max_r_ticks).round().astype(int)
            rticks = rticks[idx]

        axins.set_yticks(rticks)
        axins.set_yticklabels([rf'${int(t):d}$' for t in rticks], fontsize=8)
        axins.set_rlabel_position(theta_zoom_min_deg + 0.12 * dtheta)

    axins.tick_params(colors='white', labelsize=8)

    th1 = np.deg2rad(theta_zoom_min_deg)
    th2 = np.deg2rad(theta_zoom_max_deg)

    ax.plot([th1, th1], [r_zoom_min, r_zoom_max], color='white', lw=0.9, alpha=0.9)
    ax.plot([th2, th2], [r_zoom_min, r_zoom_max], color='white', lw=0.9, alpha=0.9)
    ax.plot(np.linspace(th1, th2, 400), np.full(400, r_zoom_max), color='white', lw=0.9, alpha=0.9,)

    con1 = ConnectionPatch(xyA=(th1, r_zoom_max), coordsA=axins.transData, xyB=(th1, r_zoom_max),
                           coordsB=ax.transData, color='white', lw=0.8, alpha=0.6,)
    con2 = ConnectionPatch(xyA=(th2, r_zoom_max), coordsA=axins.transData, xyB=(th2, r_zoom_max),
                           coordsB=ax.transData, color='white', lw=0.8, alpha=0.6,)

    fig.add_artist(con1)
    fig.add_artist(con2)

    return axins


def main():
    args = parse_args()
    raw_path = Path(args.raw_input)
    prob_path = Path(args.prob_input)
    cache_path = Path(args.class_cache)
    void_path = Path(args.void_input)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    class_tid, class_code, cache_status = load_or_build_class_cache(prob_path=prob_path,
                                                                    cache_path=cache_path,
                                                                    rebuild=args.rebuild_cache)

    with fitsio.FITS(str(raw_path)) as fobj:
        hdu = fobj[1]
        n_total = hdu.get_nrows()
        n_data = find_first_non_data_row(hdu, data_randiter=args.data_randiter)
        rows = np.arange(n_data, dtype=np.int64)
        arr = hdu.read(columns=['TARGETID', 'RA', 'DEC', 'Z'], rows=rows)

    raw_tid = np.asarray(arr['TARGETID'], dtype=np.int64)
    ra = np.asarray(arr['RA'], dtype=np.float64)
    dec = np.asarray(arr['DEC'], dtype=np.float64)
    redshift = np.asarray(arr['Z'], dtype=np.float64)

    finite = np.isfinite(ra) & np.isfinite(dec) & np.isfinite(redshift)
    n_nonfinite = int((~finite).sum())
    if n_nonfinite > 0:
        raw_tid = raw_tid[finite]
        ra = ra[finite]
        dec = dec[finite]
        redshift = redshift[finite]

    mapped_class, n_class_miss = map_classes_to_raw(raw_tid=raw_tid, class_tid=class_tid, class_code=class_code)
    valid_class = mapped_class <= 3
    if not np.all(valid_class):
        raw_tid = raw_tid[valid_class]
        ra = ra[valid_class]
        dec = dec[valid_class]
        redshift = redshift[valid_class]
        mapped_class = mapped_class[valid_class]

    half = 0.5 * args.slice_width_deg
    if args.dec0 is None:
        dec0, centers, counts = select_auto_dec0(dec=dec, width_deg=args.slice_width_deg, step_deg=args.dec_scan_step)
        order = np.argsort(counts)[::-1][:5]
        top5 = [(float(centers[i]), int(counts[i])) for i in order]
    else:
        dec0 = float(args.dec0)
        top5 = []

    dec_lo = dec0 - half
    dec_hi = dec0 + half
    in_slice = (dec >= dec_lo) & (dec <= dec_hi)
    if not np.any(in_slice):
        raise RuntimeError('No galaxies left after DEC slice selection.')

    ra_slice = ra[in_slice]
    z_slice = redshift[in_slice]
    cls_slice = mapped_class[in_slice]

    z_max = float(np.max(z_slice))
    z_grid = np.linspace(0.0, z_max + 0.02, args.z_grid_size, dtype=np.float64)
    r_grid_mpc = Planck18.comoving_distance(z_grid).value
    r_slice_mpc = np.interp(z_slice, z_grid, r_grid_mpc)

    n_total_slice = int(in_slice.sum())
    in_r = r_slice_mpc <= args.rmax_mpc
    n_in_rmax = int(in_r.sum())

    theta = np.deg2rad(ra_slice - args.ra_center_deg)

    void_data, h_void, h_key, z_kind, vcols = _read_voids_table(voids_path=void_path,
                                                                z_col=args.void_z_col,
                                                                use_xyz=args.void_use_xyz)
    if (not np.isfinite(h_void)) or h_void <= 0.0:
        h_void = 1.0
        h_key = ''
    void_ra = np.asarray(void_data[vcols['ra']], dtype=np.float64)
    void_dec = np.asarray(void_data[vcols['dec']], dtype=np.float64)
    void_ids = np.asarray(void_data[vcols['void_id']], dtype=np.int64)
    void_a = np.asarray(void_data[vcols['a']], dtype=np.float64) / h_void
    void_b = np.asarray(void_data[vcols['b']], dtype=np.float64) / h_void
    void_c = np.asarray(void_data[vcols['c']], dtype=np.float64) / h_void
    void_reff = np.asarray(void_data[vcols['r_eff']], dtype=np.float64) / h_void
    if z_kind == 'redshift':
        void_z = np.asarray(void_data[vcols['z']], dtype=np.float64)
    else:
        void_x = np.asarray(void_data[vcols['x']], dtype=np.float64)
        void_y = np.asarray(void_data[vcols['y']], dtype=np.float64)
        void_zcart = np.asarray(void_data[vcols['z_cart']], dtype=np.float64)

    n_void_total = len(void_ra)
    void_finite = np.isfinite(void_ra) & np.isfinite(void_dec)
    if z_kind == 'redshift':
        void_finite &= np.isfinite(void_z)
    else:
        void_finite &= np.isfinite(void_x) & np.isfinite(void_y) & np.isfinite(void_zcart)
    if not np.all(void_finite):
        void_ra = void_ra[void_finite]
        void_dec = void_dec[void_finite]
        void_ids = void_ids[void_finite]
        void_a = void_a[void_finite]
        void_b = void_b[void_finite]
        void_c = void_c[void_finite]
        void_reff = void_reff[void_finite]
        if z_kind == 'redshift':
            void_z = void_z[void_finite]
        else:
            void_x = void_x[void_finite]
            void_y = void_y[void_finite]
            void_zcart = void_zcart[void_finite]
    n_void_finite = len(void_ra)

    void_in_slice = (void_dec >= dec_lo) & (void_dec <= dec_hi)
    void_ra = void_ra[void_in_slice]
    void_dec = void_dec[void_in_slice]
    void_ids = void_ids[void_in_slice]
    void_a = void_a[void_in_slice]
    void_b = void_b[void_in_slice]
    void_c = void_c[void_in_slice]
    void_reff = void_reff[void_in_slice]
    if z_kind == 'redshift':
        void_z = void_z[void_in_slice]
    else:
        void_x = void_x[void_in_slice]
        void_y = void_y[void_in_slice]
        void_zcart = void_zcart[void_in_slice]
    n_void_slice = len(void_ra)

    if z_kind == 'redshift':
        void_r_mpc = Planck18.comoving_distance(void_z).value
    else:
        void_r_mpc = np.sqrt(void_x * void_x + void_y * void_y + void_zcart * void_zcart) / h_void

    void_in_r = void_r_mpc <= args.rmax_mpc
    void_ra = void_ra[void_in_r]
    void_dec = void_dec[void_in_r]
    void_ids = void_ids[void_in_r]
    void_a = void_a[void_in_r]
    void_b = void_b[void_in_r]
    void_c = void_c[void_in_r]
    void_reff = void_reff[void_in_r]
    void_r_mpc = void_r_mpc[void_in_r]
    if z_kind == 'redshift':
        void_z = void_z[void_in_r]
    else:
        void_x = void_x[void_in_r]
        void_y = void_y[void_in_r]
        void_zcart = void_zcart[void_in_r]
    n_void_rmax = len(void_ra)

    valid_void = void_ids >= 0
    if not np.all(valid_void):
        void_ra = void_ra[valid_void]
        void_dec = void_dec[valid_void]
        void_ids = void_ids[valid_void]
        void_a = void_a[valid_void]
        void_b = void_b[valid_void]
        void_c = void_c[valid_void]
        void_reff = void_reff[valid_void]
        void_r_mpc = void_r_mpc[valid_void]
        if z_kind == 'redshift':
            void_z = void_z[valid_void]
        else:
            void_x = void_x[valid_void]
            void_y = void_y[valid_void]
            void_zcart = void_zcart[valid_void]

    member_theta = None
    member_r = None
    member_gid = None
    n_members_total = 0
    n_members_plotted = 0
    if args.plot_members:
        member_data, member_cols = _read_point_membership(void_path)
        if member_data is not None and member_cols is not None:
            gid = np.asarray(member_data[member_cols['groupid']], dtype=np.int64)
            n_members_total = len(gid)
            mask = np.isfinite(gid)

            if (not args.members_include_rand) and ('is_data' in member_cols):
                is_data = np.asarray(member_data[member_cols['is_data']], dtype=np.int8)
                mask &= is_data == 1

            mra = None
            mdec = None
            mr = None
            if 'ra' in member_cols and 'dec' in member_cols:
                mra = np.asarray(member_data[member_cols['ra']], dtype=np.float64)
                mdec = np.asarray(member_data[member_cols['dec']], dtype=np.float64)
            if 'z' in member_cols:
                mz = np.asarray(member_data[member_cols['z']], dtype=np.float64)
                mr = Planck18.comoving_distance(mz).value
            if ('x' in member_cols) and ('y' in member_cols) and ('z_cart' in member_cols):
                mx = np.asarray(member_data[member_cols['x']], dtype=np.float64)
                my = np.asarray(member_data[member_cols['y']], dtype=np.float64)
                mzcart = np.asarray(member_data[member_cols['z_cart']], dtype=np.float64)
                if mra is None or mdec is None:
                    mra, mdec, mr_xyz = _xyz_to_radec(mx, my, mzcart)
                    if mr is None:
                        mr = mr_xyz / h_void
                elif mr is None:
                    mr = np.sqrt(mx * mx + my * my + mzcart * mzcart) / h_void

            if mra is not None and mdec is not None and mr is not None:
                mask &= np.isfinite(mra) & np.isfinite(mdec) & np.isfinite(mr)
                mask &= (mdec >= dec_lo) & (mdec <= dec_hi)
                mask &= mr <= args.rmax_mpc
                mask &= gid >= 0

                if np.any(mask):
                    member_theta = np.deg2rad(mra[mask] - args.ra_center_deg)
                    member_r = mr[mask]
                    member_gid = gid[mask]
                    n_members_plotted = len(member_gid)

    ra_min = float(np.min(ra_slice))
    ra_max = float(np.max(ra_slice))
    if len(void_ra) > 0:
        ra_min = min(ra_min, float(np.min(void_ra)))
        ra_max = max(ra_max, float(np.max(void_ra)))

    theta_min_deg = float(np.rad2deg(np.min(theta)))
    theta_max_deg = float(np.rad2deg(np.max(theta)))
    if len(void_ra) > 0:
        theta_void = np.deg2rad(void_ra - args.ra_center_deg)
        theta_min_deg = min(theta_min_deg, float(np.rad2deg(np.min(theta_void))))
        theta_max_deg = max(theta_max_deg, float(np.rad2deg(np.max(theta_void))))

    fig = plt.figure(figsize=(11.0, 7.2), dpi=args.dpi, facecolor=args.bg_color)
    ax = fig.add_subplot(111, projection='polar', facecolor=args.bg_color)
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(1)

    if args.plot_galaxies:
        for code, cname in enumerate(CLASS_NAMES):
            if cname == 'Void' and (not args.plot_void_class):
                continue
            if cname == 'Void' and CLASS_COLORS[cname] == 'none':
                continue
            mask = in_r & (cls_slice == code)
            if np.any(mask):
                ax.scatter(theta[mask], r_slice_mpc[mask], s=args.galaxy_point_size,
                           c=CLASS_COLORS[cname], alpha=args.alpha, linewidths=0,
                           rasterized=True, zorder=1,)

    color_ids = void_ids
    if member_gid is not None and len(member_gid) > 0:
        color_ids = np.concatenate([void_ids, member_gid])
    void_colors = _build_void_color_map(color_ids, args.void_cmap)

    if member_theta is not None and member_gid is not None and len(member_gid) > 0:
        member_colors = np.array([void_colors.get(int(g), (1.0, 1.0, 1.0, 0.9)) for g in member_gid])
        ax.scatter( member_theta, member_r, s=args.member_point_size, c=member_colors,
                   alpha=args.member_alpha, linewidths=0, rasterized=True, zorder=2,)
    else:
        member_colors = None

    void_theta = np.deg2rad(void_ra - args.ra_center_deg)
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

        ax.scatter([theta0], [r0], s=args.void_center_size, c=[color], alpha=args.void_alpha, linewidths=0, zorder=5,)

        if args.plot_void_ellipses:
            theta_e, r_e = _ellipse_theta_r(theta0=theta0, r0=r0, a=a, b=b, npts=args.void_ellipse_npts)
            ax.plot(theta_e, r_e, color=color, linewidth=args.void_ellipse_lw,
                    alpha=args.void_ellipse_alpha, zorder=4,)

    ax.set_thetamin(theta_min_deg)
    ax.set_thetamax(theta_max_deg)
    ax.set_ylim(0.0, args.rmax_mpc)
    ax.grid(color='0.50', alpha=0.30, linewidth=0.7)
    ax.spines['polar'].set_color('0.75')
    ax.spines['polar'].set_linewidth(1.0)

    ax.set_xticks([])
    ax.set_yticks([])
    ax.tick_params(axis='both', which='both', length=0)

    ax.set_title(rf'$\mathrm{{BGS\ NGC\ void\ groups}},\ \delta={args.slice_width_deg:.0f}^\circ'
                 rf'\ (\mathrm{{DEC\ center}}={dec0:.2f}^\circ)$',
                 fontsize=17,
                 pad=28,
                 color='white',)

    theta_plot = theta[in_r]
    r_plot = r_slice_mpc[in_r]
    cls_plot = cls_slice[in_r]

    theta_zoom_min_deg = args.zoom_ra_min - args.ra_center_deg
    theta_zoom_max_deg = args.zoom_ra_max - args.ra_center_deg

    add_void_inset(fig=fig, ax=ax, void_theta=void_theta, void_r=void_r_mpc,
                   void_ids=void_ids, void_a=void_a, void_b=void_b, void_reff=void_reff,
                   void_colors=void_colors, ra_center_deg=args.ra_center_deg,
                   theta_zoom_min_deg=theta_zoom_min_deg, theta_zoom_max_deg=theta_zoom_max_deg,
                   r_zoom_min=args.zoom_r_min, r_zoom_max=args.zoom_r_max,
                   inset_rect=(args.inset_left, args.inset_bottom, args.inset_width, args.inset_height),
                   point_size=args.void_center_size * args.zoom_point_scale,
                   alpha=min(1.0, args.void_alpha + args.zoom_alpha_boost), ellipse_lw=args.void_ellipse_lw,
                   ellipse_alpha=min(1.0, args.void_ellipse_alpha + args.zoom_alpha_boost),
                   ellipse_npts=args.void_ellipse_npts, ra_tick_step=args.zoom_ra_tick_step,
                   rtick_step=args.zoom_r_tick_step, bg_color=args.bg_color, plot_ellipses=args.plot_void_ellipses,
                   galaxy_theta=(theta_plot if args.plot_galaxies else None), galaxy_r=(r_plot if args.plot_galaxies else None),
                   galaxy_cls=(cls_plot if args.plot_galaxies else None), class_names=CLASS_NAMES, class_colors=CLASS_COLORS,
                   galaxy_point_size=(args.galaxy_point_size * args.zoom_galaxy_point_scale), member_theta=member_theta,
                   member_r=member_r, member_colors=member_colors, member_point_size=(args.member_point_size * args.zoom_point_scale),
                   member_alpha=min(1.0, args.member_alpha + args.zoom_alpha_boost),)

    if not args.hide_legend:
        handles = []
        if args.plot_galaxies:
            handles.append(Line2D([0], [0],
                           marker='o',
                           linestyle='',
                           markersize=7,
                           markerfacecolor='white',
                           markeredgecolor='none',
                           label='Sheet/Filament/Knot',
                           alpha=0.9,))
        if member_theta is not None and member_gid is not None and len(member_gid) > 0:
            handles.append(Line2D([0], [0],
                                  marker='o',
                                  linestyle='',
                                  markersize=7,
                                  markerfacecolor='none',
                                  markeredgecolor='white',
                                  label='Void members',
                                  alpha=0.9,))
        handles.append(Line2D([0], [0],
                              marker='o',
                              linestyle='',
                              markersize=7,
                              markerfacecolor='none',
                              markeredgecolor='white',
                              label='Void groups',
                              alpha=0.9,))
        leg = fig.legend(handles=handles,
                         labels=[h.get_label() for h in handles],
                         loc='upper right',
                         bbox_to_anchor=(0.985, 0.965),
                         framealpha=0.6,
                         ncol=1,
                         fontsize=10,)
        for text in leg.get_texts():
            text.set_color('white')

    fig.subplots_adjust(left=0.04, right=0.98, bottom=0.06, top=0.90)
    fig.savefig(out_path, dpi=args.dpi)
    plt.close(fig)

    class_counts_slice = np.bincount(cls_slice[in_r], minlength=4)

    print('--- BGS NGC Wedge (Void Groups + Inset) Summary ---')
    print(f'Raw FITS: {raw_path}')
    print(f'Prob FITS: {prob_path}')
    print(f'Class cache: {cache_path} ({cache_status})')
    print(f'Void catalog: {void_path}')
    if h_key:
        print(f'Void catalog h: {h_void:.6f} (from {h_key})')
    else:
        print(f'Void catalog h: {h_void:.6f} (default, header missing H/H0)')
    print(f'Void distance mode: {z_kind}')
    print(f'Void counts: total={n_void_total:,} | finite={n_void_finite:,} | '
          f'in DEC slice={n_void_slice:,} | in rmax={n_void_rmax:,}')
    if args.plot_members:
        print(f'Void members: total={n_members_total:,} | plotted={n_members_plotted:,}')
    print(f'Output PNG: {out_path}')
    print(f'Total rows in raw FITS: {n_total:,}')
    print(f'Rows kept as real galaxies (RANDITER={args.data_randiter}): {n_data:,}')
    print(f'Rows after finite/class filtering: {len(ra):,}')
    print(f'Cross-match misses TARGETID: {n_class_miss:,}')
    print(f'Non-finite rows removed: {n_nonfinite}')
    if top5:
        print('Top DEC centers by population (deg, count):', top5)
    print(f'Chosen DEC slice: [{dec_lo:.3f}, {dec_hi:.3f}] deg (center={dec0:.3f}) -> '
        f'{n_total_slice:,} galaxies')
    print(f'Within r <= {args.rmax_mpc:.1f} Mpc: {n_in_rmax:,}')
    print('Class counts plotted slice (all): '
          + ', '.join([f'{CLASS_NAMES[i]}={int(class_counts_slice[i]):,}' for i in range(4)]))
    print(f'Void groups plotted: {len(void_ids):,}')


if __name__ == '__main__':
    main()