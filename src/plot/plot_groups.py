import os, argparse
import numpy as np
import matplotlib.pyplot as plt
from astropy.table import Table, join
from astropy.cosmology import Planck18
from matplotlib.lines import Line2D

from plot.plot_extra import get_zone_paths

plt.rcParams.update({'font.family': 'serif', 'font.size': 12, 'axes.labelsize': 12,
                     'xtick.labelsize': 10,'ytick.labelsize': 10, 'legend.fontsize': 10,})

CLASS_COLORS = {'void': 'red', 'sheet': '#9ecae1', 'filament': '#3182bd', 'knot': 'navy'}
CLASS_ZORDER = {'void': 0, 'sheet': 1, 'filament': 2, 'knot': 3}

RAW_COLS = ['TARGETID','RA','Z','TRACERTYPE','RANDITER']
GROUPS_COLS = ['TARGETID','TRACERTYPE','RANDITER','GROUPID','NPTS']


def read_groups(groups_dir, zone, webtype):
    """
    Reads the groups table for a given zone and webtype.
    
    Args:
        groups_dir (str): Directory where the groups files are stored.
        zone (int): Zone number.
        webtype (str): Type of web structure (e.g., 'void', 'sheet', 'filament', 'knot').
    Returns:
        Table: Astropy Table containing the groups data.
    """
    path = os.path.join(groups_dir, f'zone_{zone:02d}_groups_fof_{webtype}.fits.gz')
    tbl = Table.read(path)
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
        return Table.read(raw_path, hdu=1, include_names=RAW_COLS)
    except TypeError:
        tbl = Table.read(raw_path, hdu=1)
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
    ax.set_title(title, fontsize=14, y=1.05)
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
        ax.hlines(z0, -w0, w0, color='gray', lw=0.5, alpha=0.5)
    step = max(1, n_ra // 4)
    for rt in ra_ticks[::step]:
        dx = Dc * np.deg2rad(rt - ra_ctr)
        ax.plot((dx / y_max) * zs if y_max > 0 else np.zeros_like(zs), zs, color='gray', lw=0.5, alpha=0.5)
    return z_ticks, ra_ticks

def _draw_borders(ax, half_w, y_max):
    """
    Draws the borders of the wedge plot.

    Args:
        ax (matplotlib.axes.Axes): The axis on which to draw the borders.
        half_w (float): Half width of the wedge at the maximum redshift.
        y_max (float): Maximum y value for the plot.
    """
    ax.plot([-half_w, 0], [y_max, 0], 'k-', lw=1.5)
    ax.plot([ half_w, 0], [y_max, 0], 'k-', lw=1.5)
    ax.plot([-half_w, half_w], [y_max, y_max], 'k-', lw=1.5)
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
        ax.text(xt, y_max + 0.01*y_max, f"{rt:.0f}", ha='center', va='bottom', fontsize=10)
    ax.text(0, y_max + 0.03*y_max, 'RA (deg)', ha='center', va='bottom', fontsize=11)


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
        ax.text(x0r + offset, z0, f"{z0:.2f}", ha='left', va='center', rotation=angle + 180, fontsize=10)
    if idx == 0:
        ax.set_ylabel(ylabel, fontsize=20, labelpad=15)
        ax.set_yticks(z_ticks)
        ax.set_yticklabels([f"{zt:.2f}" for zt in z_ticks], fontsize=10)


def plot_wedges(joined, tracers, zone, webtype, out_png, smin, max_z, n_ra=15, n_z=10, coord='z'):
    """
    Plots the wedge diagrams for the given tracers and zone.

    Args:
        joined (pd.DataFrame): The joined data containing tracer information.
        tracers (list): List of tracer types to plot.
        zone (int): The zone number to plot.
        webtype (str): The type of web to plot.
        out_png (str): Output file path for the plot.
        smin (float): Minimum marker size.
        max_z (float): Maximum redshift to consider.
        n_ra (int): Number of RA ticks to draw.
        n_z (int): Number of redshift ticks to draw.
        coord (str): Coordinate system being used ('z' or 'dc').
    """
    tr_types = np.asarray(joined['TRACERTYPE']).astype(str)
    ravec = np.asarray(joined['RA'], float)
    zvec = np.asarray(joined['Z'], float)
    gids = np.asarray(joined['GROUPID'], int)
    npts = np.asarray(joined['NPTS'], int)

    tr_pref = tracer_prefixes(tr_types)

    if coord == 'dc':
        y_all = Planck18.comoving_distance(zvec).value
    else:
        y_all = zvec

    ra_min, ra_max, ra_ctr, Dc_maxz, half_w, zmax = _compute_zone_params(ravec, zvec, max_z)
    y_max = Planck18.comoving_distance(zmax).value if coord == 'dc' else zmax

    nrows, ncols = subplot_grid(tracers.size)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4*ncols, 25*nrows), sharex=False, sharey=False)
    axes = np.atleast_1d(axes).ravel()

    for i, tr in enumerate(tracers):
        ax = axes[i]
        _init_ax(ax, f'{tr} — {webtype} — zone {zone:02d}')

        m = (tr_pref == tr)
        if max_z is not None:
            m &= (zvec <= max_z)

        if not np.any(m):
            ax.set_xlim(-half_w, half_w)
            ax.set_ylim(0, y_max)
            continue

        ra = ravec[m]
        yvals = y_all[m]
        gid = gids[m]

        z_ticks, ra_ticks = _draw_grid(ax, ra_min, ra_max, ra_ctr, Dc_maxz, half_w, y_max, n_ra, n_z, coord)

        _, color_idx = np.unique(gid, return_inverse=True)

        Dc_each = yvals if coord == 'dc' else Planck18.comoving_distance(yvals).value
        x = (Dc_each / y_max) * (Dc_maxz * np.deg2rad(ra - ra_ctr))

        ax.scatter(x, yvals, s=max(smin, 6), c=color_idx, cmap='tab20', alpha=0.65)

        _draw_borders(ax, half_w, y_max)
        _annotate_ra_top(ax, ra_ticks, ra_ctr, Dc_maxz, y_max)
        _annotate_y_side(ax, z_ticks, half_w, y_max, i, 'z' if coord == 'z' else r'$D_c$ [Mpc]')

    for j in range(i+1, len(axes)):
        axes[j].axis('off')

    fig.tight_layout()
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    fig.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.close(fig)
    return out_png


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--raw-dir', default='/pscratch/sd/v/vtorresg/cosmic-web/edr/raw', help='Raw data dir')
    p.add_argument('--class-dir', default='/pscratch/sd/v/vtorresg/cosmic-web/edr/class', help='Classification dir')
    p.add_argument('--groups-dir', default='/pscratch/sd/v/vtorresg/cosmic-web/edr/groups', help='Output groups dir')
    p.add_argument('--output', default='.')
    p.add_argument('--zone', type=int, default=0)
    p.add_argument('--webtype', choices=['void','sheet','filament','knot'], default='filament')
    p.add_argument('--tracers', nargs='*', default=None)
    p.add_argument('--source', choices=['data','rand','both'], default='data')
    p.add_argument('--smin', type=int, default=1)
    p.add_argument('--max-z', type=float, default=None)
    p.add_argument('--coord', choices=['z','dc'], default='z', help='Y-axis coordinate: redshift z (default) or comoving distance Dc')
    p.add_argument('--bins', type=int, default=10, help='Number of grid bins for RA and z/Dc')
    return p.parse_args()


def main():
    args = parse_args()

    groups = read_groups(args.groups_dir, args.zone, args.webtype)
    gm = mask_source(np.asarray(groups['RANDITER']), args.source)
    groups = groups[gm]

    raw = read_raw_min(args.raw_dir, args.class_dir, args.zone)
    rm = mask_source(np.asarray(raw['RANDITER']), args.source)
    raw = raw[rm]

    joined = join(groups, raw, keys=['TARGETID','TRACERTYPE','RANDITER'], join_type='inner')

    available = tracer_prefixes(np.asarray(joined['TRACERTYPE']).astype(str))
    tracers = pick_tracers(available, args.tracers)

    out_png = os.path.join(args.output, f'groups_wedges_zone_{args.zone:02d}_{args.webtype}_{args.coord}.png')
    path = plot_wedges(joined, tracers, args.zone, args.webtype, out_png, args.smin, args.max_z, n_ra=args.bins, n_z=args.bins, coord=args.coord)


if __name__ == '__main__':
    main()