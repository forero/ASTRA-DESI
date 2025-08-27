import os, sys, re, glob, argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.offsetbox import AnnotationBbox, DrawingArea
from astropy.cosmology import Planck18
from astropy.table import join as at_join
# plt.style.use('dark_background')
LINE_COLOR = 'black'
plt.rcParams.update({'font.family': 'serif'})

from plot_groups import (read_groups, read_raw_min, mask_source, tracer_prefixes,
                              ORDERED_TRACERS, TRACER_ZLIMS)


def _detect_zones(groups_dir, webtype):
    """
    Detects the zones present in the group files for a given webtype.

    Args:
        groups_dir (str): Directory containing the group files.
        webtype (str): Type of web structure (e.g., 'void', 'sheet', 'filament', 'knot').
    Returns:
        list: List of zone numbers present in the group files.
    """
    patt = os.path.join(groups_dir, f'zone_*_groups_fof_{webtype}.fits.gz')
    zones = []
    for path in glob.glob(patt):
        m = re.search(r'zone_(\d+)_groups_fof_', os.path.basename(path))
        if m: zones.append(int(m.group(1)))
    return sorted(set(zones))


def _join_zone(groups_dir, raw_dir, class_dir, zone, webtype, source):
    """
    Joins the groups and raw data for a specific zone and webtype.

    Args:
        groups_dir (str): Directory containing the group files.
        raw_dir (str): Directory containing the raw data files.
        class_dir (str): Directory containing the class data files.
        zone (int): Zone number.
        webtype (str): Type of web structure (e.g., 'void', 'sheet', 'filament', 'knot').
        source (str): Source type, can be 'data', 'rand', or 'both'.
    Returns:
        Table: Astropy Table containing the joined data.
    """
    groups = read_groups(groups_dir, zone, webtype)
    gm = mask_source(np.asarray(groups['RANDITER']), source)
    groups = groups[gm]
    raw = read_raw_min(raw_dir, class_dir, zone)
    rm = mask_source(np.asarray(raw['RANDITER']), source)
    raw = raw[rm]
    joined = at_join(groups, raw, keys=['TARGETID','TRACERTYPE','RANDITER'], join_type='inner')
    return joined


def _pick_tracers_available(joined, requested=None):
    """
    Picks the available tracers from the joined table based on the requested list.

    Args:
        joined (Table): The joined Astropy Table containing the data.
        requested (list, optional): List of requested tracer prefixes. If None, all available tracers are returned.
    Returns:
        list: List of available tracer prefixes.
    """
    available = set(map(str, tracer_prefixes(np.asarray(joined['TRACERTYPE']).astype(str))))
    if requested:
        req = [t.split('_',1)[0].upper() for t in requested]
        return [t for t in ORDERED_TRACERS if (t in req) and (t in available)]
    return [t for t in ORDERED_TRACERS if t in available]


def _to_pizza_angles(ra, ra_min, ra_max, theta0, theta1):
    """
    Converts RA coordinates to pizza slice angles.

    Args:
        ra (ndarray): RA coordinates.
        ra_min (float): Minimum RA value.
        ra_max (float): Maximum RA value.
        theta0 (float): Starting angle.
        theta1 (float): Ending angle.
    Returns:
        ndarray: Pizza slice angles corresponding to the RA coordinates.
    """
    if ra_max <= ra_min:
        return 0.5*(theta0+theta1)*np.ones_like(ra, dtype=float)
    t = (ra - ra_min) / (ra_max - ra_min)
    return theta0 + t * (theta1 - theta0)


def _draw_zone_label(ax, mid_theta, r_edge, text, *, r_pad_frac=0.01, fontsize=55):
    """
    Draw a zone label hugging the outer edge of the wedge, oriented tangentially.

    Args:
        ax (matplotlib.axes.Axes): Polar axes.
        mid_theta (float): Middle angle of the wedge (radians).
        r_edge (float): Local outer radius for this wedge (in axis units for the chosen coord).
        text (str): Text to draw (e.g., 'Zone 05').
        r_pad_frac (float, optional): Fractional inward radial padding relative to r_edge to keep the text just inside the circle boundary.
        fontsize (int, optional): Font size for the label.
    """
    r = r_edge * max(0.0, 1.0 - r_pad_frac)

    dth = 1e-3
    p0 = ax.transData.transform((mid_theta, r))
    p1 = ax.transData.transform((mid_theta + dth, r))
    dx, dy = (p1[0] - p0[0], p1[1] - p0[1])
    rot = np.degrees(np.arctan2(dy, dx))

    if rot < -90.0 or rot > 90.0:
        rot += 180.0

    ax.text(mid_theta, r, text, rotation=rot, rotation_mode='anchor',
            ha='center', va='center', fontsize=fontsize, color=LINE_COLOR,)


def _plot_slice(ax, joined, tracer, theta0, theta1, coord, max_z, bins, connect_lines, smin):
    """
    Plots a slice of the data in the specified coordinate system.
    
    Args:
        ax (matplotlib.axes.Axes): The axes to plot on.
        joined (Table): The joined Astropy Table containing the data.
        tracer (str): The tracer to plot.
        theta0 (float): Starting angle for the plot.
        theta1 (float): Ending angle for the plot.
        coord (str): Coordinate system to use ('z' or 'dc').
        max_z (float, optional): Maximum redshift to consider.
        bins (int): Number of bins for the histogram.
        connect_lines (bool): Whether to connect points with lines.
        smin (int): Minimum marker size.
    Returns:
        float: Maximum radius value plotted.
    """
    tr_types = np.asarray(joined['TRACERTYPE']).astype(str)
    ravec = np.asarray(joined['RA'], float)
    zvec = np.asarray(joined['Z'], float)
    gids = np.asarray(joined['GROUPID'], int)

    tr_pref = tracer_prefixes(tr_types)
    m = (tr_pref == tracer)

    z_cap = TRACER_ZLIMS.get(tracer, None)
    if max_z is not None:
        z_cap = min(z_cap, max_z) if z_cap is not None else max_z
    if z_cap is not None:
        m &= (zvec <= z_cap)

    if not np.any(m):
        return 0.0

    if coord == 'dc':
        r_all = Planck18.comoving_distance(zvec[m]).value
    else:
        r_all = zvec[m]

    ra_min, ra_max = float(np.nanmin(ravec[m])), float(np.nanmax(ravec[m]))
    theta = _to_pizza_angles(ravec[m], ra_min, ra_max, theta0, theta1)

    _, color_idx = np.unique(gids[m], return_inverse=True)
    cmap = plt.get_cmap('tab20')
    palette = np.asarray(cmap.colors)
    idx_mod = color_idx % cmap.N

    base_size = 20
    if tracer == 'BGS':
        size = base_size
    elif tracer in ('LRG', 'ELG'):
        size = base_size + 1
    elif tracer == 'QSO':
        size = base_size + 10
    else:
        size = base_size

    ax.scatter(theta, r_all, s=size, c=palette[idx_mod], alpha=1.0)

    if connect_lines:
        gid_sub = gids[m]
        theta_sub = theta
        r_sub = r_all
        for g in np.unique(gid_sub):
            sel = (gid_sub == g)
            if np.count_nonzero(sel) < 2:
                continue
            order = np.argsort(r_sub[sel])
            th = theta_sub[sel][order]
            rr = r_sub[sel][order]
            ax.plot(th, rr, lw=1.0, alpha=1.0, color=palette[int(idx_mod[sel][0])])

    return float(np.nanmax(r_all))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--raw-dir', default='/pscratch/sd/v/vtorresg/cosmic-web/edr/raw')
    p.add_argument('--class-dir', default='/pscratch/sd/v/vtorresg/cosmic-web/edr/class')
    p.add_argument('--groups-dir',default='/pscratch/sd/v/vtorresg/cosmic-web/edr/groups')
    p.add_argument('--output', default='/pscratch/sd/v/vtorresg/cosmic-web/edr/figs/wedges/pizza')
    p.add_argument('--webtype', choices=['void','sheet','filament','knot'], default='filament')
    p.add_argument('--zones', nargs='+', default=['all'])
    p.add_argument('--tracers', nargs='*', default=['BGS','LRG','ELG','QSO'])
    p.add_argument('--source', choices=['data','rand','both'], default='data')
    p.add_argument('--max-z', type=float, default=None)
    p.add_argument('--coord', choices=['z','dc'], default='z')
    p.add_argument('--bins', type=int, default=10)
    p.add_argument('--connect-lines', action='store_true')
    p.add_argument('--smin', type=int, default=30)
    p.add_argument('--title', default=None)
    return p.parse_args()


def main():
    args = parse_args()

    if len(args.zones)==1 and args.zones[0].lower()=='all':
        zones = _detect_zones(args.groups_dir, args.webtype)
    else:
        zones = sorted(set(int(z) for z in args.zones))
    if not zones:
        raise SystemExit('No zones')

    tracers_final = None
    for z in zones:
        try:
            joined0 = _join_zone(args.groups_dir, args.raw_dir, args.class_dir, z, args.webtype, args.source)
            cand = _pick_tracers_available(joined0, args.tracers)
            if cand:
                tracers_final = cand
                break
        except FileNotFoundError:
            continue
    if not tracers_final:
        raise SystemExit('No tracers for zone')

    os.makedirs(args.output, exist_ok=True)

    for tr in tracers_final:
        fig = plt.figure(figsize=(70, 70))
        ax = plt.subplot(111, projection='polar')

        fig.subplots_adjust(right=0.86)
        ax.set_xticks([])
        ax.set_xticklabels([])
        ax.yaxis.grid(True, color=LINE_COLOR, lw=1.0)
        ax.spines['polar'].set_edgecolor(LINE_COLOR)
        ax.tick_params(colors=LINE_COLOR)
        ax.set_theta_zero_location('N')
        ax.set_theta_direction(-1)
        ax.set_rlabel_position(0)

        if args.title:
            fig.suptitle(args.title, y=0.98, fontsize=54)

        nZ = len(zones)
        dtheta = 2*np.pi / nZ
        rmax_global = 0.0

        for k, zone in enumerate(zones):
            theta0 = k * dtheta
            theta1 = (k+1) * dtheta
            mid = 0.5*(theta0+theta1)

            try:
                joined = _join_zone(args.groups_dir, args.raw_dir, args.class_dir, zone, args.webtype, args.source)
            except FileNotFoundError:
                ax.plot([theta0, theta0], [0, 1], color='gray', lw=1.0, alpha=1.0, transform=ax.transData)
                ax.text(mid, 0.05, f'Zone {zone}', ha='center', va='bottom', fontsize=55)
                continue

            rmax_local = _plot_slice(ax, joined, tr, theta0, theta1,
                                     coord=args.coord, max_z=args.max_z,
                                     bins=args.bins, connect_lines=args.connect_lines, smin=args.smin)
            rmax_global = max(rmax_global, rmax_local)

            ax.plot([theta0, theta0], [0, rmax_local], color=LINE_COLOR, lw=1.0, alpha=1.0)
            _draw_zone_label(ax, mid, rmax_local if rmax_local>0 else 1.0, f'Zone {zone}', r_pad_frac=-0.03, fontsize=55)

        if rmax_global <= 0:
            rmax_global = 1.0
        ax.set_rlim(0, rmax_global*1.0)

        tick_vals = np.linspace(0, rmax_global, 5)[1:]
        if args.coord == 'dc':
            tick_labels = [f'{v:.0f}' for v in tick_vals]
        else:
            tick_labels = [f'{v:.2f}' for v in tick_vals]

        ax.set_yticklabels([])

        theta_label = 0.5*np.pi
        dtheta = 0.10

        label_bar_lw = 80
        label_bar_dtheta = 0.2
        text_offset = 0.01

        rect_w_px = 150
        rect_h_px = 120

        for v, lab in zip(tick_vals, tick_labels):
            da = DrawingArea(rect_w_px, rect_h_px, clip=False)
            da.add_artist(plt.Rectangle((0, 0), rect_w_px, rect_h_px, facecolor='white', edgecolor='none'))

            ab = AnnotationBbox(da, (theta_label + text_offset, v),
                                xycoords='data', frameon=False, box_alignment=(0.15, 0.5),
                                zorder=1000)
            ax.add_artist(ab)

            ax.text(theta_label + text_offset + 0.006, v, lab, ha='left', va='center', fontsize=50,
                    color=LINE_COLOR, rotation=0, clip_on=False, zorder=1001)

        ax.set_axisbelow(False)
        theta_label = 0.5*np.pi

        ax.set_rlabel_position(0)
        ax.yaxis.set_label_position("right")
        ax.yaxis.tick_right()

        if args.coord == 'dc':
            ax.set_ylabel(r'$D_c$ [Mpc]', fontsize=72, labelpad=20, family='serif')
        else:
            ax.set_ylabel(r'z', fontsize=72, labelpad=6, family='serif')

        ax.yaxis.label.set_rotation(0)
        ax.yaxis.set_label_coords(1.06, 0.5)

        ax.set_title(f'{tr} — {args.webtype.capitalize()}s (RA slices mapped to angular wedges)', va='bottom',
                     fontsize=80, pad=54, y=1.05)

        out = os.path.join(args.output, f'pizza_{tr}_{args.webtype}_{args.coord}.png')
        fig.savefig(out, dpi=100)
        plt.close(fig)
        print(out)


if __name__ == '__main__':
    main()