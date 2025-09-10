import os, argparse
import numpy as np
import matplotlib.pyplot as plt
from astropy.table import Table, join
from astropy.cosmology import Planck18
from matplotlib.lines import Line2D

# Support both package and script execution
try:
    # When run as a package: python -m src.plot.plot_groups
    from .plot_extra import get_zone_paths
except Exception:
    # When run directly: python src/plot/plot_groups.py or from src/plot/
    import os as _os, sys as _sys
    _here = _os.path.abspath(_os.path.dirname(__file__))
    _sys.path.append(_os.path.dirname(_here))  # add src/ to sys.path
    from plot.plot_extra import get_zone_paths
# plt.style.use('dark_background')

plt.rcParams.update({'font.family': 'serif', 'font.size': 20, 'axes.labelsize': 20,
                     'xtick.labelsize': 20,'ytick.labelsize': 20, 'legend.fontsize': 10,})

CLASS_COLORS = {'void': 'red', 'sheet': '#9ecae1', 'filament': '#3182bd', 'knot': 'navy'}
CLASS_ZORDER = {'void': 0, 'sheet': 1, 'filament': 2, 'knot': 3}
ORDERED_TRACERS = ['BGS', 'LRG', 'ELG', 'QSO']
TRACER_ZLIMS = {'BGS': 0.45, 'LRG': 1.0, 'ELG': 1.4, 'QSO': 2.2}

RAW_COLS = ['TARGETID','RA','Z','TRACERTYPE','RANDITER']
GROUPS_COLS = ['TARGETID','TRACERTYPE','RANDITER','GROUPID','NPTS']
main_color, sec_color = 'black', 'gray'
# main_color, sec_color = 'white', 'gainsboro' #for dark background


def _zone_tag(zone):
    """
    Convert a zone number to a zero-padded string.

    Args:
        zone (int or str): Zone number (0-99) or label (e.g., 'NGC1').
    Returns:
        str: Zero-padded zone number as a string.
    """
    try:
        return f'{int(zone):02d}'
    except Exception:
        return str(zone)

def read_groups(groups_dir, zone, webtype):
    """
    Reads the groups table for a given zone and webtype.
    
    Args:
        groups_dir (str): Directory where the groups files are stored.
        zone (int): Zone number or label.
        webtype (str): Type of web structure (e.g., 'void', 'sheet', 'filament', 'knot').
    Returns:
        Table: Astropy Table containing the groups data.
    """
    tag = _zone_tag(zone)
    path = os.path.join(groups_dir, f'zone_{tag}_groups_fof_{webtype}.fits.gz')
    try:
        tbl = Table.read(path, memmap=True)
        missing = [c for c in GROUPS_COLS if c not in tbl.colnames]
        if missing:
            raise KeyError(f'Missing columns {missing} in {path}')
        return tbl[GROUPS_COLS]
    except TypeError:
        tbl = Table.read(path)
        missing = [c for c in GROUPS_COLS if c not in tbl.colnames]
        if missing:
            raise KeyError(f'Missing columns {missing} in {path}')
        return tbl[GROUPS_COLS]


def read_raw_min(raw_dir, class_dir, zone):
    """
    Reads the raw data for a given zone, filtering to include only necessary columns.
    
    Args:
        raw_dir (str): Directory where the raw data files are stored.
        class_dir (str): Directory where the class data files are stored.
        zone (int): Zone number.
    Returns:
        Table: Astropy Table containing the raw data with selected columns.
    """
    raw_path, _ = get_zone_paths(raw_dir, class_dir, zone)
    try:
        return Table.read(raw_path, hdu=1, include_names=RAW_COLS, memmap=True)
    except TypeError:
        tbl = Table.read(raw_path, hdu=1, memmap=True)
        missing = [c for c in RAW_COLS if c not in tbl.colnames]
        if missing:
            raise KeyError(f'Missing columns {missing} in {raw_path}')
        return tbl[RAW_COLS]


def mask_source(randiter, source):
    """
    Masks the data based on the source type.
    
    Args:
        randiter (np.ndarray): Array indicating whether the data is from the random sample (-1) or not.
        source (str): Source type, can be 'data', 'rand', or 'both'.
    Returns:
        np.ndarray: Boolean mask indicating which rows to keep based on the source type.
    """
    is_data = (randiter == -1)
    return is_data if source == 'data' else (~is_data if source == 'rand' else np.ones_like(is_data, dtype=bool))


def tracer_prefixes(tr_types):
    """
    Extracts the tracer prefixes from the TRACERTYPE strings.
    
    Args:
        tr_types (np.ndarray): Array of TRACERTYPE strings.
    Returns:
        np.ndarray: Array of tracer prefixes, which are the parts of the TRACERTYPE
    """
    return np.char.partition(tr_types, '_')[:, 0]


def pick_tracers(available, wanted):
    """
    Selects the tracers from the available ones based on the wanted list.
    
    Args:
        available (np.ndarray): Array of available tracer prefixes.
        wanted (list or None): List of wanted tracer prefixes. If None, all available tracers are returned.
    Returns:
        np.ndarray: Array of tracer prefixes that are both available and wanted.
    """
    if not wanted:
        return np.unique(available)
    w = np.asarray(wanted, dtype=str)

    present = set(available.tolist())
    return np.array([t for t in w if t in present], dtype=str)


def subplot_grid(n):
    """
    Computes the number of rows and columns for subplots based on the number of tracers.
    
    Args:
        n (int): Number of tracers.
    Returns:
        tuple: Number of rows and columns for the subplots.
    """
    ncols = min(4, max(1, n))
    nrows = (n + ncols - 1) // ncols
    return nrows, ncols


def _init_ax(ax, title):
    """
    Initializes the axis for the wedge plot with a title and removes spines and ticks.

    Args:
        ax (matplotlib.axes.Axes): The axis to initialize.
        title (str): The title for the plot.
    """
    ax.set_title(title, fontsize=24, y=1.05)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_xticks([]); ax.set_yticks([])


def _compute_zone_params(ravec, zvec, z_lim):
    """
    Computes parameters for the zone plot based on RA and redshift vectors.
    
    Args:
        ravec (np.ndarray): Array of RA values.
        zvec (np.ndarray): Array of redshift values.
        z_lim (float): Redshift limit for the plot.
    Returns:
        tuple: A tuple containing the computed parameters (ra_min, ra_max, ra_ctr, Dc, half_w, zmax).
    """
    zmax_data = float(np.nanmax(zvec)) * 1.02
    zmax = min(zmax_data, z_lim) if z_lim is not None else zmax_data
    ra_min, ra_max = float(np.nanmin(ravec)), float(np.nanmax(ravec))
    ra_ctr = 0.5 * (ra_min + ra_max)
    Dc = Planck18.comoving_distance(zmax).value
    half_w = Dc * np.deg2rad(ra_max - ra_ctr)
    return ra_min, ra_max, ra_ctr, Dc, half_w, zmax


def _draw_grid(ax, ra_min, ra_max, ra_ctr, Dc, half_w, y_max, n_ra, n_z, coord):
    """
    Draws a grid of lines on the wedge plot to indicate the redshift and RA ticks.

    Args:
        ax (matplotlib.axes.Axes): The axis on which to draw the grid.
        ra_min (float): Minimum RA value.
        ra_max (float): Maximum RA value.
        ra_ctr (float): Central RA value.
        Dc (float): Comoving distance at the maximum redshift.
        half_w (float): Half width of the wedge at the maximum redshift.
        y_max (float): Maximum y value for the plot.
        n_ra (int): Number of RA ticks to draw.
        n_z (int): Number of redshift ticks to draw.
        coord (str): Coordinate system being used ('z' or 'dc').
    Returns:
        tuple: A tuple containing the redshift and RA tick values.
    """
    if coord == 'z':
        zs = np.linspace(0.0, y_max, 300)
        z_ticks = np.linspace(0.0, y_max, n_z)
    else:
        zs = np.linspace(0.0, y_max, 300)
        z_ticks = np.linspace(0.0, y_max, n_z)
    ra_ticks = np.linspace(ra_min, ra_max, n_ra)
    for z0 in z_ticks:
        w0 = half_w * (z0 / y_max) if y_max > 0 else 0
        ax.hlines(z0, -w0, w0, color=sec_color, lw=0.5, alpha=0.5)
    step = max(1, n_ra // 4)
    for rt in ra_ticks[::step]:
        dx = Dc * np.deg2rad(rt - ra_ctr)
        ax.plot((dx / y_max) * zs if y_max > 0 else np.zeros_like(zs), zs, color=sec_color, lw=0.5, alpha=0.5)
    return z_ticks, ra_ticks


def _draw_borders(ax, half_w, y_max):
    """
    Draws the borders of the wedge plot.

    Args:
        ax (matplotlib.axes.Axes): The axis on which to draw the borders.
        half_w (float): Half width of the wedge at the maximum redshift.
        y_max (float): Maximum y value for the plot.
    """
    ax.plot([-half_w, 0], [y_max, 0], lw=1.5, c=main_color)
    ax.plot([ half_w, 0], [y_max, 0], lw=1.5, c=main_color)
    ax.plot([-half_w, half_w], [y_max, y_max], lw=1.5, c=main_color)
    ax.set_xlim(-half_w, half_w)
    ax.set_ylim(0, y_max)


def _annotate_ra_top(ax, ra_ticks, ra_ctr, Dc, y_max):
    """
    Annotates the top of the wedge plot with RA tick values.

    Args:
        ax (matplotlib.axes.Axes): The axis on which to draw the annotations.
        ra_ticks (np.ndarray): Array of RA tick values.
        ra_ctr (float): Central RA value.
        Dc (float): Comoving distance at the maximum redshift.
        y_max (float): Maximum y value for the plot.
    """
    top4 = np.linspace(ra_ticks.min(), ra_ticks.max(), 4)
    x_top = Dc * np.deg2rad(top4 - ra_ctr)
    for xt, rt in zip(x_top, top4):
        ax.text(xt, y_max + 0.01*y_max, f'{rt:.0f}', ha='center', va='bottom', fontsize=20)
    ax.text(0, y_max + 0.03*y_max, 'RA (deg)', ha='center', va='bottom', fontsize=20)


def _annotate_y_side(ax, z_ticks, half_w, y_max, idx, ylabel):
    """
    Annotates the side of the wedge plot with redshift tick values.

    Args:
        ax (matplotlib.axes.Axes): The axis on which to draw the annotations.
        z_ticks (np.ndarray): Array of redshift tick values.
        half_w (float): Half width of the wedge at the maximum redshift.
        y_max (float): Maximum y value for the plot.
        idx (int): Index of the current subplot.
        ylabel (str): Label for the y-axis.
    """
    for z0 in z_ticks:
        x0r = half_w * (z0 / y_max) if y_max > 0 else 0
        angle = np.degrees(np.arctan2(-z0, -x0r)) if y_max > 0 else 0
        offset = np.sign(x0r) * half_w * 0.11
        ax.text(x0r + offset, z0, f'{z0:.2f}', ha='left', va='center', rotation=angle + 180, fontsize=20)
    if idx == 0:
        ax.set_ylabel(ylabel, fontsize=25, labelpad=15)


def plot_wedges(joined, tracers, zone, webtype, out_png, smin, max_z, n_ra=15, n_z=10, coord='z',
                connect_lines=False, line_min_npts=2):
    """
    Plots the wedge diagrams for the given tracers and zone.

    Args:
        joined (pd.DataFrame): The joined data containing tracer information.
        tracers (list): List of tracer types to plot.
        zone (int): The zone number to plot.
        webtype (str): The type of web to plot.
        out_png (str): Output file path for the plot.
        smin (float): Minimum marker size.
        max_z (float): Global maximum redshift cap. Per-tracer limits (TRACER_ZLIMS) are applied and this
                       value acts as an additional upper bound.
        n_ra (int): Number of RA ticks to draw.
        n_z (int): Number of redshift ticks to draw.
        coord (str): Coordinate system being used ('z' or 'dc').
        connect_lines (bool): Whether to connect points in the same group with lines.
        line_min_npts (int): Minimum number of points in a group to draw connecting lines.
    """
    tracers = [str(t).split('_', 1)[0].upper() for t in tracers]
    tracers = [t for t in ORDERED_TRACERS if t in tracers]

    tr_types = np.asarray(joined['TRACERTYPE']).astype(str)
    ravec = np.asarray(joined['RA'], float)
    zvec = np.asarray(joined['Z'], float)
    gids = np.asarray(joined['GROUPID'], int)
    npts = np.asarray(joined['NPTS'], int)

    tr_pref = tracer_prefixes(tr_types)

    nrows, ncols = subplot_grid(len(tracers))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 30*nrows), sharex=False, sharey=False)
    axes = np.atleast_1d(axes).ravel()
    plt.suptitle(f'{webtype.capitalize()}s in zone {_zone_tag(zone)}', fontsize=27, y=1.01)

    cmap = plt.get_cmap('tab20')
    palette = np.asarray(cmap.colors)

    for i, tr in enumerate(tracers):
        tr = str(tr)
        ax = axes[i]
        _init_ax(ax, tr)

        m = (tr_pref == tr)
        tr_key = str(tr).split('_', 1)[0].upper()
        z_cap = TRACER_ZLIMS.get(tr_key, None)
        if max_z is not None:
            z_cap = min(z_cap, max_z) if z_cap is not None else max_z
        if z_cap is not None:
            m &= (zvec <= z_cap)

        if not np.any(m):
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes, fontsize=16)
            continue

        ra_min, ra_max, ra_ctr, Dc_maxz, half_w, zmax = _compute_zone_params(ravec[m], zvec[m], z_cap)
        y_max = Planck18.comoving_distance(zmax).value if coord == 'dc' else zmax

        ra = ravec[m]
        yvals = zvec[m] if coord == 'z' else Planck18.comoving_distance(zvec[m]).value
        gid = gids[m]

        z_ticks, ra_ticks = _draw_grid(ax, ra_min, ra_max, ra_ctr, Dc_maxz, half_w, y_max, n_ra, n_z, coord)

        _, color_idx = np.unique(gid, return_inverse=True)
        idx_mod = color_idx % palette.shape[0]

        scale = yvals / y_max
        x = scale * (Dc_maxz * np.deg2rad(ra - ra_ctr))
        w_at_y = half_w * scale
        mclip = np.abs(x) <= w_at_y

        if connect_lines:
            ci_sub = idx_mod[mclip]
            gid_sub = gid[mclip]
            x_sub = x[mclip]
            y_sub = yvals[mclip]

            for g in np.unique(gid_sub):
                selg = (gid_sub == g)
                if np.count_nonzero(selg) < line_min_npts:
                    continue
                order = np.argsort(y_sub[selg])
                xs = x_sub[selg][order]
                ys = y_sub[selg][order]
                ax.plot(xs, ys, lw=0.5, alpha=0.9, color=palette[int(ci_sub[selg][0])], zorder=0)

        ax.scatter(x[mclip], yvals[mclip], s=max(smin, 8), c=palette[idx_mod[mclip]], alpha=0.75)

        _draw_borders(ax, half_w, y_max)
        _annotate_ra_top(ax, ra_ticks, ra_ctr, Dc_maxz, y_max)
        _annotate_y_side(ax, z_ticks, half_w, y_max, i, 'z' if coord == 'z' else r'$D_c$ [Mpc]')

    for j in range(i+1, len(axes)):
        axes[j].axis('off')

    fig.tight_layout()    
    fig.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.close(fig)
    return out_png


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--release', choices=['EDR','DR1'], default='DR1', help='Dataset release: EDR (zones 00..19) or DR1 (NGC1/NGC2)')
    p.add_argument('--raw-dir', default='/pscratch/sd/v/vtorresg/cosmic-web/dr1/raw', help='Raw data dir')
    p.add_argument('--class-dir', default='/pscratch/sd/v/vtorresg/cosmic-web/dr1/class', help='Classification dir')
    p.add_argument('--groups-dir', default='/pscratch/sd/v/vtorresg/cosmic-web/dr1/groups', help='Output groups dir')
    p.add_argument('--output', default='/pscratch/sd/v/vtorresg/cosmic-web/dr1/figs/wedges/filaments')
    p.add_argument('--zone', type=str, default='NGC1', help='EDR: 0..19 or 00..19; DR1: NGC1/NGC2')
    p.add_argument('--webtype', choices=['void','sheet','filament','knot'], default='filament')
    p.add_argument('--tracers', nargs='*', default=None)
    p.add_argument('--source', choices=['data','rand','both'], default='data')
    p.add_argument('--smin', type=int, default=1)
    p.add_argument('--max-z', type=float, default=None)
    p.add_argument('--coord', choices=['z','dc'], default='z', help='Y-axis coordinate: redshift z (default) or comoving distance Dc')
    p.add_argument('--bins', type=int, default=10, help='Number of grid bins for RA and z/Dc')
    p.add_argument('--connect-lines', action='store_true', help='Connect points that belong to the same group with a line')
    p.add_argument('--line-min-npts', type=int, default=2, help='Minimum number of points in a group to draw its connecting line')
    return p.parse_args()


def main():
    args = parse_args()

    # Interpret zone per release
    if args.release.upper() == 'EDR':
        try:
            zone = int(args.zone)
        except Exception:
            # allow strings like '00'
            zone = int(str(args.zone))
    else:
        zone = str(args.zone)

    # Ensure output directory exists
    os.makedirs(args.output, exist_ok=True)

    groups = read_groups(args.groups_dir, zone, args.webtype)
    gm = mask_source(np.asarray(groups['RANDITER']), args.source)
    groups = groups[gm]

    raw = read_raw_min(args.raw_dir, args.class_dir, zone)
    rm = mask_source(np.asarray(raw['RANDITER']), args.source)
    raw = raw[rm]

    joined = join(groups, raw, keys=['TARGETID','TRACERTYPE','RANDITER'], join_type='inner')

    available = tracer_prefixes(np.asarray(joined['TRACERTYPE']).astype(str))
    avail_set = set(map(str, available))

    if args.tracers:
        tokens = []
        for t in args.tracers:
            tokens.extend(str(t).replace(',', ' ').split())
        req_pref = [tok.split('_', 1)[0].upper() for tok in tokens]
        req_set = set(req_pref)
        tracers = [t for t in ORDERED_TRACERS if (t in req_set) and (t in avail_set)]
    else:
        tracers = [t for t in ORDERED_TRACERS if t in avail_set]

    tag = _zone_tag(zone)
    out_png = os.path.join(args.output, f'groups_wedges_zone_{tag}_{args.webtype}_{args.coord}.png')
    path = plot_wedges(joined, tracers, args.zone, args.webtype, out_png, args.smin, args.max_z,
                       n_ra=args.bins, n_z=args.bins, coord=args.coord, connect_lines=args.connect_lines,
                       line_min_npts=args.line_min_npts)


if __name__ == '__main__':
    main()
