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

plt.style.use('dark_background')
plt.rcParams.update({'text.usetex': True})

try:
    from plot_bgs_ngc_wedge_void_groups_inset import (CLASS_COLORS,
                                                      CLASS_NAMES,
                                                      DEFAULT_CACHE,
                                                      DEFAULT_CLASS_GLOB,
                                                      DEFAULT_PROB,
                                                      DEFAULT_RANDOM_VOID_CACHE,
                                                      DEFAULT_RAW,
                                                      _build_void_color_map,
                                                      _ellipse_theta_r,
                                                      _read_point_membership,
                                                      _read_voids_table,
                                                      _xyz_to_radec,
                                                      find_first_non_data_row,
                                                      load_or_build_class_cache,
                                                      load_or_build_random_void_cache,
                                                      map_classes_to_raw,
                                                      select_auto_dec0)
except ImportError:
    from .plot_bgs_ngc_wedge_void_groups_inset import (CLASS_COLORS,
                                                       CLASS_NAMES,
                                                       DEFAULT_CACHE,
                                                       DEFAULT_CLASS_GLOB,
                                                       DEFAULT_PROB,
                                                       DEFAULT_RANDOM_VOID_CACHE,
                                                       DEFAULT_RAW,
                                                       _build_void_color_map,
                                                       _ellipse_theta_r,
                                                       _read_point_membership,
                                                       _read_voids_table,
                                                       _xyz_to_radec,
                                                       find_first_non_data_row,
                                                       load_or_build_class_cache,
                                                       load_or_build_random_void_cache,
                                                       map_classes_to_raw,
                                                       select_auto_dec0)


DEFAULT_VOIDS_CARTESIAN = '/pscratch/sd/v/vtorresg/cosmic-web/dr2/void-cat-v2/dr2/voids_BGS_ANY_NGC_v4.fits'
DEFAULT_OUTPUT = 'plots/bgs_ngc_cartesian_void_groups.png'


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--raw-input', default=DEFAULT_RAW)
    p.add_argument('--prob-input', default=DEFAULT_PROB)
    p.add_argument('--class-cache', default=DEFAULT_CACHE)
    p.add_argument('--rebuild-cache', action='store_true')
    p.add_argument('--classification-glob', default=DEFAULT_CLASS_GLOB)
    p.add_argument('--random-void-cache', default=DEFAULT_RANDOM_VOID_CACHE)
    p.add_argument('--rebuild-random-void-cache', action='store_true')
    p.add_argument('--random-iterations', default='all')
    p.add_argument('--random-chunk-size', type=int, default=1_000_000)
    p.add_argument('--random-void-background', dest='random_void_background', action='store_true')
    p.add_argument('--no-random-void-background', dest='random_void_background', action='store_false')
    p.set_defaults(random_void_background=True)
    p.add_argument('--random-void-color', default='#f2f2f2')
    p.add_argument('--random-void-point-size', type=float, default=0.015)
    p.add_argument('--random-void-alpha', type=float, default=0.22)

    p.add_argument('--void-input', default=DEFAULT_VOIDS_CARTESIAN)
    p.add_argument('--output', default=DEFAULT_OUTPUT)
    p.add_argument('--slice-width-deg', type=float, default=2.0)
    p.add_argument('--dec0', type=float, default=None)
    p.add_argument('--dec-scan-step', type=float, default=0.25)
    p.add_argument('--ra-center-deg', type=float, default=180.0)
    p.add_argument('--rmax-mpc', type=float, default=1250.0)
    p.add_argument('--box-size-mpc', type=float, default=500.0)

    p.add_argument('--point-size', dest='galaxy_point_size', type=float, default=0.03)
    p.add_argument('--galaxy-point-size', dest='galaxy_point_size', type=float, default=0.03)
    p.add_argument('--alpha', type=float, default=0.45)
    p.add_argument('--void-center-size', type=float, default=1.0)
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
    p.add_argument('--data-randiter', type=int, default=-1)
    p.add_argument('--z-grid-size', type=int, default=4096)
    p.add_argument('--bg-color', default='#000000')
    p.add_argument('--hide-legend', action='store_true')
    p.add_argument('--plot-void-class', action='store_true')

    return p.parse_args()


def relative_ra_deg(ra_deg, ra_center_deg):
    return (np.asarray(ra_deg, dtype=np.float64) - ra_center_deg + 180.0) % 360.0 - 180.0


def radec_r_to_xyz(ra_deg, dec_deg, r_mpc):
    ra_rad = np.deg2rad(np.asarray(ra_deg, dtype=np.float64))
    dec_rad = np.deg2rad(np.asarray(dec_deg, dtype=np.float64))
    r = np.asarray(r_mpc, dtype=np.float64)
    cos_dec = np.cos(dec_rad)
    x = r * cos_dec * np.cos(ra_rad)
    y = r * cos_dec * np.sin(ra_rad)
    z = r * np.sin(dec_rad)
    return x, y, z


def radec_r_to_xy(ra_deg, dec_deg, r_mpc):
    x, y, _ = radec_r_to_xyz(ra_deg=ra_deg, dec_deg=dec_deg, r_mpc=r_mpc)
    return x, y


def ellipse_xy(ra0_deg, dec0_deg, r0, a, b, npts, ra_center_deg):
    theta0 = np.deg2rad(relative_ra_deg(ra0_deg, ra_center_deg))
    theta_e, r_e = _ellipse_theta_r(theta0=theta0, r0=r0, a=a, b=b, npts=npts)
    ra_e = ra_center_deg + np.rad2deg(theta_e)
    dec_e = np.full_like(ra_e, float(dec0_deg), dtype=np.float64)
    return radec_r_to_xy(ra_e, dec_e, r_e)


def set_centered_square_limits(ax, x, y, side_mpc):
    if side_mpc <= 0.0:
        raise ValueError('--box-size-mpc must be positive.')

    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    finite = np.isfinite(x) & np.isfinite(y)
    if not np.any(finite):
        raise RuntimeError('Cannot center x-y plot: no finite data coordinates.')

    x = x[finite]
    y = y[finite]
    xmin = float(np.min(x))
    xmax = float(np.max(x))
    ymin = float(np.min(y))
    ymax = float(np.max(y))
    xmid = 0.5 * (xmin + xmax)
    ymid = 0.5 * (ymin + ymax)
    half = 0.5 * side_mpc
    ax.set_xlim(xmid - half, xmid + half)
    ax.set_ylim(ymid - half, ymid + half)
    ax.set_aspect('equal', adjustable='box')
    return xmid, ymid


def main():
    args = parse_args()
    raw_path = Path(args.raw_input)
    prob_path = Path(args.prob_input)
    cache_path = Path(args.class_cache)
    random_void_cache_path = Path(args.random_void_cache)
    void_path = Path(args.void_input)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    class_tid, class_code, cache_status = load_or_build_class_cache(
        prob_path=prob_path,
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

    mapped_class, n_class_miss = map_classes_to_raw(
        raw_tid=raw_tid,
        class_tid=class_tid,
        class_code=class_code)
    valid_class = mapped_class <= 3
    if not np.all(valid_class):
        raw_tid = raw_tid[valid_class]
        ra = ra[valid_class]
        dec = dec[valid_class]
        redshift = redshift[valid_class]
        mapped_class = mapped_class[valid_class]

    half = 0.5 * args.slice_width_deg
    if args.dec0 is None:
        dec0, centers, counts = select_auto_dec0(
            dec=dec,
            width_deg=args.slice_width_deg,
            step_deg=args.dec_scan_step)
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
    dec_slice = dec[in_slice]
    z_slice = redshift[in_slice]
    cls_slice = mapped_class[in_slice]

    random_void_ra = np.array([], dtype=np.float64)
    random_void_dec = np.array([], dtype=np.float64)
    random_void_z = np.array([], dtype=np.float64)
    random_void_r_value = np.array([], dtype=np.float32)
    random_void_status = 'disabled'
    n_random_void_total = 0
    n_random_void_finite = 0
    if args.random_void_background:
        random_arr, random_void_status = load_or_build_random_void_cache(
            raw_path=raw_path,
            class_pattern=args.classification_glob,
            iteration_spec=args.random_iterations,
            cache_path=random_void_cache_path,
            data_randiter=args.data_randiter,
            chunk_size=args.random_chunk_size,
            dec_bounds=(dec_lo, dec_hi),
            rebuild=args.rebuild_random_void_cache)
        n_random_void_total = len(random_arr)
        if n_random_void_total > 0:
            random_void_ra = np.asarray(random_arr['RA'], dtype=np.float64)
            random_void_dec = np.asarray(random_arr['DEC'], dtype=np.float64)
            random_void_z = np.asarray(random_arr['Z'], dtype=np.float64)
            random_void_r_value = np.asarray(random_arr['R_VALUE'], dtype=np.float32)
            finite_random = (np.isfinite(random_void_ra)
                             & np.isfinite(random_void_dec)
                             & np.isfinite(random_void_z))
            finite_random &= (random_void_dec >= dec_lo) & (random_void_dec <= dec_hi)
            random_void_ra = random_void_ra[finite_random]
            random_void_dec = random_void_dec[finite_random]
            random_void_z = random_void_z[finite_random]
            random_void_r_value = random_void_r_value[finite_random]
            n_random_void_finite = len(random_void_ra)

    z_max = float(np.max(z_slice))
    if len(random_void_z) > 0:
        z_max = max(z_max, float(np.max(random_void_z)))
    z_grid = np.linspace(0.0, z_max + 0.02, args.z_grid_size, dtype=np.float64)
    r_grid_mpc = Planck18.comoving_distance(z_grid).value
    r_slice_mpc = np.interp(z_slice, z_grid, r_grid_mpc)
    random_void_r_mpc = (np.interp(random_void_z, z_grid, r_grid_mpc)
                         if len(random_void_z) > 0 else np.array([], dtype=np.float64))

    random_void_in_r = random_void_r_mpc <= args.rmax_mpc
    n_random_void_rmax = int(np.count_nonzero(random_void_in_r))
    if n_random_void_rmax > 0:
        random_void_x, random_void_y = radec_r_to_xy(random_void_ra[random_void_in_r],
                                                     random_void_dec[random_void_in_r],
                                                     random_void_r_mpc[random_void_in_r])
    else:
        random_void_x = np.array([], dtype=np.float64)
        random_void_y = np.array([], dtype=np.float64)

    n_total_slice = int(in_slice.sum())
    in_r = r_slice_mpc <= args.rmax_mpc
    n_in_rmax = int(in_r.sum())
    if n_in_rmax == 0:
        raise RuntimeError(f'No galaxies left within r <= {args.rmax_mpc:.1f} Mpc.')

    galaxy_x, galaxy_y = radec_r_to_xy(ra_slice, dec_slice, r_slice_mpc)

    void_data, h_void, h_key, z_kind, vcols = _read_voids_table(
        voids_path=void_path,
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

    member_x = None
    member_y = None
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
                    member_x, member_y = radec_r_to_xy(mra[mask], mdec[mask], mr[mask])
                    member_r = mr[mask]
                    member_gid = gid[mask]
                    n_members_plotted = len(member_gid)

    if z_kind == 'redshift':
        void_plot_x, void_plot_y = radec_r_to_xy(void_ra, void_dec, void_r_mpc)
    else:
        void_plot_x = void_x / h_void
        void_plot_y = void_y / h_void

    fig = plt.figure(figsize=(11.0, 7.2), dpi=args.dpi, facecolor=args.bg_color)
    ax = fig.add_subplot(111, facecolor=args.bg_color)
    ax.grid(color='0.50', alpha=0.25, linewidth=0.7)
    ax.axhline(0.0, color='0.65', alpha=0.35, linewidth=0.8, zorder=-10)
    ax.axvline(0.0, color='0.65', alpha=0.35, linewidth=0.8, zorder=-10)

    if args.random_void_background and n_random_void_rmax > 0:
        ax.scatter(random_void_x,
                   random_void_y,
                   s=args.random_void_point_size,
                   c=args.random_void_color,
                   alpha=args.random_void_alpha,
                   linewidths=0,
                   rasterized=True,
                   zorder=0)

    if args.plot_galaxies:
        for code, cname in enumerate(CLASS_NAMES):
            if cname == 'Void' and (not args.plot_void_class):
                continue
            if cname == 'Void' and CLASS_COLORS[cname] == 'none':
                continue
            mask = in_r & (cls_slice == code)
            if np.any(mask):
                ax.scatter(galaxy_x[mask], galaxy_y[mask], s=args.galaxy_point_size,
                           c=CLASS_COLORS[cname], alpha=args.alpha, linewidths=0,
                           rasterized=True, zorder=1)

    color_ids = void_ids
    if member_gid is not None and len(member_gid) > 0:
        color_ids = np.concatenate([void_ids, member_gid])
    void_colors = _build_void_color_map(color_ids, args.void_cmap)

    if member_x is not None and member_gid is not None and len(member_gid) > 0:
        member_colors = np.array([void_colors.get(int(g), (1.0, 1.0, 1.0, 0.9)) for g in member_gid])
        ax.scatter(member_x, member_y, s=args.member_point_size, c=member_colors,
                   alpha=args.member_alpha, linewidths=0, rasterized=True,
                   zorder=2)
    else:
        member_colors = None

    for i in range(len(void_ids)):
        vid = int(void_ids[i])
        color = void_colors.get(vid, (1.0, 1.0, 1.0, 0.9))
        r0 = float(void_r_mpc[i])
        a = float(void_a[i]) if np.isfinite(void_a[i]) and void_a[i] > 0 else float(void_reff[i])
        b = float(void_b[i]) if np.isfinite(void_b[i]) and void_b[i] > 0 else float(void_reff[i])
        if not np.isfinite(a) or a <= 0:
            a = float(void_reff[i]) if np.isfinite(void_reff[i]) and void_reff[i] > 0 else 0.0
        if not np.isfinite(b) or b <= 0:
            b = float(void_reff[i]) if np.isfinite(void_reff[i]) and void_reff[i] > 0 else 0.0
        if r0 <= 0 or (a <= 0 and b <= 0):
            continue

        x0 = float(void_plot_x[i])
        y0 = float(void_plot_y[i])
        ax.scatter([x0], [y0], s=args.void_center_size, c=[color],
                   alpha=args.void_alpha, linewidths=0, zorder=5)

        if args.plot_void_ellipses:
            xe, ye = ellipse_xy(ra0_deg=float(void_ra[i]),
                                dec0_deg=float(void_dec[i]),
                                r0=r0, a=a, b=b,
                                npts=args.void_ellipse_npts,
                                ra_center_deg=args.ra_center_deg)
            ax.plot(xe, ye, color=color, linewidth=args.void_ellipse_lw,
                    alpha=args.void_ellipse_alpha, zorder=4)

    plot_center_x, plot_center_y = set_centered_square_limits(
        ax, galaxy_x[in_r], galaxy_y[in_r], args.box_size_mpc)
    ax.tick_params(axis='both', colors='white', labelsize=10)
    ax.set_xlabel(r'$x = r\cos(\delta)\cos(\alpha)\ [\mathrm{Mpc}]$',
                  fontsize=13, color='white')
    ax.set_ylabel(r'$y = r\cos(\delta)\sin(\alpha)\ [\mathrm{Mpc}]$',
                  fontsize=13, color='white')
    for spine in ax.spines.values():
        spine.set_color('0.75')
        spine.set_linewidth(1.0)

    ax.set_title(rf'$\mathrm{{BGS\ NGC\ void\ groups\ }}(x,y),\ '
                 rf'\delta={args.slice_width_deg:.0f}^\circ'
                 rf'\ (\mathrm{{DEC\ center}}={dec0:.2f}^\circ)$',
                 fontsize=17,
                 pad=18,
                 color='white')

    ax.text(0.03, 0.04,
            rf'$\mathrm{{DEC}}\in[{dec_lo:.2f}^\circ,{dec_hi:.2f}^\circ]$',
            transform=ax.transAxes, ha='left', va='bottom',
            fontsize=13, color='white', clip_on=False)

    if not args.hide_legend:
        handles = []
        if args.random_void_background and n_random_void_rmax > 0:
            handles.append(Line2D([0], [0],
                                  marker='o',
                                  linestyle='',
                                  markersize=5,
                                  markerfacecolor=args.random_void_color,
                                  markeredgecolor='none',
                                  label='Random voids',
                                  alpha=min(0.8, max(args.random_void_alpha * 3.0, 0.25))))
        if args.plot_galaxies:
            handles.append(Line2D([0], [0],
                                  marker='o',
                                  linestyle='',
                                  markersize=7,
                                  markerfacecolor='white',
                                  markeredgecolor='none',
                                  label='Sheet/Filament/Knot',
                                  alpha=0.9))
        if member_x is not None and member_gid is not None and len(member_gid) > 0:
            handles.append(Line2D([0], [0],
                                  marker='o',
                                  linestyle='',
                                  markersize=7,
                                  markerfacecolor='none',
                                  markeredgecolor='white',
                                  label='Void members',
                                  alpha=0.9))
        handles.append(Line2D([0], [0],
                              marker='o',
                              linestyle='',
                              markersize=7,
                              markerfacecolor='none',
                              markeredgecolor='white',
                              label='Void groups',
                              alpha=0.9))
        leg = fig.legend(handles=handles,
                         labels=[h.get_label() for h in handles],
                         loc='upper right',
                         bbox_to_anchor=(0.985, 0.965),
                         framealpha=0.6,
                         ncol=1,
                         fontsize=10)
        for text in leg.get_texts():
            text.set_color('white')

    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.10, top=0.90)
    fig.savefig(out_path, dpi=args.dpi)
    plt.close(fig)

    class_counts_slice = np.bincount(cls_slice[in_r], minlength=4)

    print('--- BGS NGC Cartesian x-y (Void Groups) Summary ---')
    print(f'Raw FITS: {raw_path}')
    print(f'Prob FITS: {prob_path}')
    print(f'Class cache: {cache_path} ({cache_status})')
    print(f'Classification glob: {args.classification_glob}')
    print(f'Random void cache: {random_void_cache_path} ({random_void_status})')
    print(f'Void catalog: {void_path}')
    if h_key:
        print(f'Void catalog h: {h_void:.6f} (from {h_key})')
    else:
        print(f'Void catalog h: {h_void:.6f} (default, header missing H/H0)')
    print(f'Void distance mode: {z_kind}')
    print('Cartesian coordinates: x=r*cos(DEC)*cos(RA), y=r*cos(DEC)*sin(RA)')
    print(f'Void counts: total={n_void_total:,} | finite={n_void_finite:,} | '
          f'in DEC slice={n_void_slice:,} | in rmax={n_void_rmax:,}')
    if args.plot_members:
        print(f'Void members: total={n_members_total:,} | plotted={n_members_plotted:,}')
    if args.random_void_background:
        print(f'Random void background: total={n_random_void_total:,} | '
              f'finite={n_random_void_finite:,} | in rmax={n_random_void_rmax:,}')
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
    print(f'Plot box: center=({plot_center_x:.3f}, {plot_center_y:.3f}) Mpc | '
          f'side={args.box_size_mpc:.1f} Mpc')
    print('Class counts plotted slice (all): '
          + ', '.join([f'{CLASS_NAMES[i]}={int(class_counts_slice[i]):,}' for i in range(4)]))
    print(f'Void groups plotted: {len(void_ids):,}')


if __name__ == '__main__':
    main()