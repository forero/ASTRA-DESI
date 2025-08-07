import os, argparse
import numpy as np, pandas as pd
from astropy.cosmology import Planck18
from matplotlib.lines import Line2D

from plot_extra import get_zone_paths, get_prob_path, infer_zones, make_output_dirs, \
                        load_raw_df, load_class_df, load_prob_df, compute_r

import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams.update({'font.family': 'serif', 'font.size': 12, 'axes.labelsize': 12,
                     'xtick.labelsize': 10,'ytick.labelsize': 10, 'legend.fontsize': 10,})

CLASS_COLORS = {'void': 'red', 'sheet': '#9ecae1', 'filament': '#3182bd', 'knot': 'navy'}
CLASS_ZORDER = {'void': 0, 'sheet': 1, 'filament': 2, 'knot': 3}

RAW_DIR = "/pscratch/sd/v/vtorresg/cosmic-web/edr/raw"
CLASS_DIR = "/pscratch/sd/v/vtorresg/cosmic-web/edr/class"
OUTPUT = "../plots"


def _save_real_data(real, tracer_name, zone, output_dir):
    """
    Saves the real data for a specific tracer and zone to a parquet file.
    The file is stored in a subdirectory named after the tracer within the output directory.

    Args:
        real (DataFrame): The DataFrame containing the real data.
        tracer_name (str): The name of the tracer.
        zone (int): The zone number.
        output_dir (str): The base output directory where the data will be saved.
    """
    data_zone_dir = os.path.join(output_dir, 'data', tracer_name)
    os.makedirs(data_zone_dir, exist_ok=True)
    fname = f"{tracer_name}_zone_{zone:02d}.parquet"
    real.to_parquet(os.path.join(data_zone_dir, fname), index=False)


def _compute_zone_params(real, z_lim):
    """
    Computes parameters for the wedge plot based on the real data for a specific zone.
    
    Args:
        real (DataFrame): The DataFrame containing the real data for the zone.
        z_lim (float): The maximum redshift limit for the plot.
    """
    zmax_data = real['Z'].max() * 1.02
    zmax = min(zmax_data, z_lim) if z_lim is not None else zmax_data

    ra_min, ra_max = real['RA'].min(), real['RA'].max()
    ra_ctr = 0.5 * (ra_min + ra_max)
    dec_ctr = 0.5 * (real['DEC'].min() + real['DEC'].max())

    Dc = Planck18.comoving_distance(zmax).value
    half_w = Dc * np.deg2rad(ra_max - ra_ctr) * np.cos(np.deg2rad(dec_ctr))

    return ra_min, ra_max, ra_ctr, dec_ctr, Dc, half_w, zmax


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


def _draw_grid(ax, ra_min, ra_max, ra_ctr, dec_ctr, Dc, half_w, zmax, n_ra, n_z):
    """
    Draws a grid of horizontal and vertical lines on the wedge plot to represent redshift and RA ticks.
    
    Args:
        ax (matplotlib.axes.Axes): The axis on which to draw the grid.
        ra_min (float): Minimum RA value.
        ra_max (float): Maximum RA value.
        ra_ctr (float): Central RA value.
        dec_ctr (float): Central DEC value.
        Dc (float): Comoving distance at the maximum redshift.
        half_w (float): Half width of the wedge at the maximum redshift.
        zmax (float): Maximum redshift for the plot.
        n_ra (int): Number of RA ticks to draw.
        n_z (int): Number of redshift ticks to draw.
    """
    zs = np.linspace(0, zmax, 300)
    z_ticks = np.linspace(0, zmax, n_z)
    ra_ticks = np.linspace(ra_min, ra_max, n_ra)

    for z0 in z_ticks:
        w0 = half_w * (z0 / zmax)
        ax.hlines(z0, -w0, w0, color='gray', lw=0.5, alpha=0.5)

    step = max(1, n_ra // 4)
    for rt in ra_ticks[::step]:
        dx = Dc * np.deg2rad(rt - ra_ctr) * np.cos(np.deg2rad(dec_ctr))
        ax.plot(dx / zmax * zs, zs, color='gray', lw=0.5, alpha=0.5)

    return z_ticks, ra_ticks


def _plot_classes(ax, real, tracer, ra_ctr, dec_ctr, Dc, half_w, zmax):
    """
    Plots the different classes of objects in the wedge plot based on their RA and redshift values.
    
    Args:
        ax (matplotlib.axes.Axes): The axis on which to plot the classes.
        real (DataFrame): The DataFrame containing the real data for the zone.
        tracer (str): The tracer type being plotted.
        ra_ctr (float): Central RA value.
        dec_ctr (float): Central DEC value.
        Dc (float): Comoving distance at the maximum redshift.
        half_w (float): Half width of the wedge at the maximum redshift.
        zmax (float): Maximum redshift for the plot.
    """
    for cls, color in CLASS_COLORS.items():
        sel = real['CLASS'] == cls
        if not sel.any():
            continue
        sub = real.loc[sel]
        Dc_vals = Planck18.comoving_distance(sub['Z']).value
        x = Dc_vals * np.deg2rad(sub['RA'] - ra_ctr) * np.cos(np.deg2rad(dec_ctr))
        y = sub['Z'].values
        mask = np.abs(x) <= half_w * (y / zmax)
        ax.scatter(x[mask], y[mask], s=(10 if cls == 'void' else 6), c=color, zorder=CLASS_ZORDER[cls],
                   edgecolor='black', lw=0.08)


def _draw_borders(ax, half_w, zmax):
    """
    Draws the borders of the wedge plot, including the top and sides.
    
    Args:
        ax (matplotlib.axes.Axes): The axis on which to draw the borders.
        half_w (float): Half width of the wedge at the maximum redshift.
        zmax (float): Maximum redshift for the plot.
    """
    ax.plot([-half_w, 0], [zmax, 0], 'k-', lw=1.5)
    ax.plot([ half_w, 0], [zmax, 0], 'k-', lw=1.5)
    ax.plot([-half_w, half_w], [zmax, zmax], 'k-', lw=1.5)
    ax.set_xlim(-half_w, half_w)
    ax.set_ylim(0, zmax)


def _annotate_ra_top(ax, ra_ticks, ra_ctr, dec_ctr, Dc, zmax):
    """
    Annotates the top of the wedge plot with RA tick labels.
    
    Args:
        ax (matplotlib.axes.Axes): The axis on which to annotate the RA ticks.
        ra_ticks (array-like): The RA tick values.
        ra_ctr (float): Central RA value.
        dec_ctr (float): Central DEC value.
        Dc (float): Comoving distance at the maximum redshift.
        zmax (float): Maximum redshift for the plot.
    """
    top4 = np.linspace(ra_ticks.min(), ra_ticks.max(), 4)
    x_top = Dc * np.deg2rad(top4 - ra_ctr) * np.cos(np.deg2rad(dec_ctr))
    for xt, rt in zip(x_top, top4):
        ax.text(xt, zmax + 0.01*zmax, f"{rt:.0f}", ha='center', va='bottom', fontsize=10)
    ax.text(0, zmax + 0.03*zmax, 'RA (deg)', ha='center', va='bottom', fontsize=11)


def _annotate_z_side(ax, z_ticks, half_w, zmax, idx):
    """
    Annotates the side of the wedge plot with redshift tick labels.
    
    Args:
        ax (matplotlib.axes.Axes): The axis on which to annotate the redshift ticks.
        z_ticks (array-like): The redshift tick values.
        half_w (float): Half width of the wedge at the maximum redshift.
        zmax (float): Maximum redshift for the plot.
        idx (int): Index of the current subplot, used to determine if the y-axis label should be added.
    """
    for z0 in z_ticks:
        x0r = half_w * (z0 / zmax)
        angle = np.degrees(np.arctan2(-z0, -x0r))
        offset = np.sign(x0r) * half_w * 0.11
        ax.text(x0r + offset, z0, f"{z0:.2f}", ha='left', va='center', rotation=angle + 180, fontsize=10)
    if idx == 0:
        ax.set_ylabel('z', fontsize=20, labelpad=15)
        ax.set_yticks(z_ticks)
        ax.set_yticklabels([f"{zt:.2f}" for zt in z_ticks], fontsize=10)


def plot_tracer_wedges_by_zones(raw_df, prob_df, zones, tracer, output_dir, n_ra=15, n_z=10, z_lim=0.2):
    """
    Plots wedge diagrams for a specific tracer across multiple zones.
    Each zone's data is processed to create a wedge plot showing the distribution of objects
    classified by their RA and redshift values.
    
    Args:
        raw_df (DataFrame): DataFrame containing raw data for the tracer.
        prob_df (DataFrame): DataFrame containing probability data for the tracer.
        zones (list): List of zone numbers to plot.
        tracer (str): The tracer type to plot (e.g., 'BGS_ANY', 'ELG', 'LRG', 'QSO').
        output_dir (str): Directory where the output plots will be saved.
        n_ra (int): Number of RA bins to use in the wedge plot.
        n_z (int): Number of redshift bins to use in the wedge plot.
        z_lim (float): Maximum redshift limit for the plot. If None, uses the maximum redshift in the data.
    """
    tracer_name = tracer.replace('_ANY', '').lower()
    out_dir = os.path.join(output_dir, f"tracer_zone/{tracer_name}")
    os.makedirs(out_dir, exist_ok=True)

    num = len(zones)
    ncols = min(num, 4)
    nrows = (num + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4*ncols, 15*nrows), sharex=False,
                             sharey=False, gridspec_kw={'wspace': 0.4, 'hspace': 0.3})
    axes = np.array(axes).reshape(-1)

    for idx, zone in enumerate(zones):
        ax = axes[idx]
        df_z = raw_df[raw_df['ZONE'] == zone]
        df_p = prob_df[prob_df['TARGETID'].isin(df_z['TARGETID'])]
        df = df_z.merge(df_p, on='TARGETID', how='left')
        real = df[(df['ISDATA']) & (df['BASE'] == tracer)]

        _save_real_data(real, tracer_name, zone, output_dir)
        _init_ax(ax, f"Zone {zone}")

        if real.empty:
            ax.set_xlim(0, 1); ax.set_ylim(0, 1)
            continue

        ra_min, ra_max, ra_ctr, dec_ctr, Dc, half_w, zmax = _compute_zone_params(real, z_lim)

        z_ticks, ra_ticks = _draw_grid(ax, ra_min, ra_max, ra_ctr, dec_ctr, Dc, half_w, zmax, n_ra, n_z)
        _plot_classes(ax, real, tracer, ra_ctr, dec_ctr, Dc, half_w, zmax)
        _draw_borders(ax, half_w, zmax)
        _annotate_ra_top(ax, ra_ticks, ra_ctr, dec_ctr, Dc, zmax)
        _annotate_z_side(ax, z_ticks, half_w, zmax, idx)

    handles = [Line2D([], [], marker='o', color=c, linestyle='', markersize=6, label=k)
               for k, c in CLASS_COLORS.items()]
    fig.legend(handles, CLASS_COLORS.keys(), bbox_to_anchor=(0.5, 0.965), loc='upper center',
               ncol=len(CLASS_COLORS))
    plt.suptitle(tracer.replace('_ANY', ''), fontsize=18)

    fname = f"{tracer_name}_zones_{'_'.join(f'{z:02d}' for z in zones)}.png"
    fig.savefig(os.path.join(out_dir, fname), dpi=360, bbox_inches='tight')
    plt.close(fig)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--raw-dir', default=RAW_DIR)
    p.add_argument('--class-dir', default=CLASS_DIR)
    p.add_argument('--zones', nargs='+', type=int, default=None)
    p.add_argument('--output', default=OUTPUT)
    p.add_argument('--bins', type=int, default=10)
    p.add_argument('--tracers', nargs='+', default=['BGS_ANY', 'ELG', 'LRG', 'QSO'])
    return p.parse_args()


def main():
    args = parse_args()
    zones = infer_zones(args.raw_dir, args.zones)
    make_output_dirs(args.output)

    raw_cache, prob_cache = {}, {}
    zones = [zones[1], zones[4], zones[17], zones[15]]
    for zone in zones:
        raw_p, _ = get_zone_paths(args.raw_dir, args.class_dir, zone)
        prob_p = get_prob_path(args.raw_dir, args.class_dir, zone)
        df_raw = load_raw_df(raw_p);   df_raw['ZONE'] = zone
        df_prob = load_prob_df(prob_p);  df_prob['ZONE'] = zone
        raw_cache[zone] = df_raw
        prob_cache[zone] = df_prob

    raw_all = pd.concat(raw_cache.values(), ignore_index=True)
    prob_all = pd.concat(prob_cache.values(), ignore_index=True)

    data_dir = os.path.join(args.output, 'data', 'full')
    os.makedirs(data_dir, exist_ok=True)
    raw_all.to_parquet(os.path.join(data_dir, 'raw_all.parquet'), index=False)
    prob_all.to_parquet(os.path.join(data_dir, 'prob_all.parquet'), index=False)

    for tracer in args.tracers[:1]:
        print(f"Plotting: {tracer}")
        plot_tracer_wedges_by_zones(raw_all, prob_all, zones, tracer, args.output,
                                    n_ra=args.bins, n_z=args.bins)


if __name__ == "__main__":
    main()