#!/usr/bin/env python3
import os, argparse
import numpy as np, seaborn as sns
import matplotlib.pyplot as plt

from astropy.table import Table
from astropy.cosmology import Planck18
from matplotlib.lines import Line2D
plt.rcParams.update({'font.family': 'serif', 'font.size': 12, 'axes.labelsize': 12,
                     'xtick.labelsize': 10,'ytick.labelsize': 10, 'legend.fontsize': 10,})

CLASS_COLORS = {'void': 'red', 'sheet': '#9ecae1', 'filament': '#3182bd', 'knot': 'navy'}
CLASS_ZORDER = {'void':0, 'sheet':1, 'filament':2, 'knot':3}

#! TODO  - clustering, fof? plots

def get_zone_paths(raw_dir, class_dir, zone):
    """
    Get file paths for a given zone number.

    Args:
        raw_dir (str): Directory containing raw data files.
        class_dir (str): Directory containing classification files.
        zone (int): Zone number.
    Returns:
        tuple: Paths to the raw data file and classification file for the zone.
    """
    z2 = f"{zone:02d}"
    return (os.path.join(raw_dir, f"zone_{z2}.fits.gz"),
            os.path.join(class_dir, f"zone_{z2}_class.fits.gz"),)


def get_prob_path(raw_dir, class_dir, zone):
    """
    Get the probability file path for a given zone number.

    Args:
        raw_dir (str): Directory containing raw data files.
        class_dir (str): Directory containing classification files.
        zone (int): Zone number.
    Returns:
        str: Path to the probability file for the zone.
    """
    z2 = f"{zone:02d}"
    return os.path.join(class_dir, f"zone_{z2}_probability.fits.gz")


def infer_zones(raw_dir, provided):
    """
    Infer available zones from raw directory if not provided.

    Args:
        raw_dir (str): Directory containing raw data files.
        provided (list or None): List of zone numbers if provided.
    Returns:
        list: List of zone numbers.
    """
    if provided: 
        return provided
    files = os.listdir(raw_dir)
    zones = sorted(int(f.split('_')[1].split('.')[0]) for f in files
                   if f.startswith('zone_') and f.endswith('.fits.gz'))
    return zones


def make_output_dirs(base):
    """
    Create output directories for plots.
    
    Args:
        base (str): Base output directory.
    Returns:
        dict: Dictionary with paths for different plot types.
    """
    out = {'z': os.path.join(base, 'z_histograms'),
           'radial': os.path.join(base, 'radial'),
           'cdf': os.path.join(base, 'cdf')}
    for d in out.values():
        os.makedirs(d, exist_ok=True)
    return out


def load_raw_df(path):
    """
    Load raw data from FITS file into a pandas DataFrame.

    Args:
        path (str): Path to the FITS file.
    Returns:
        pd.DataFrame: DataFrame containing the raw data.
    """
    tbl = Table.read(path)
    df = tbl.to_pandas()
    df['TRACERTYPE'] = df['TRACERTYPE'].apply(
                lambda x: x.decode('utf-8') if isinstance(x, (bytes, bytearray)) else x)
    df['BASE'] = df['TRACERTYPE'].str.replace(r'_(DATA|RAND)$','', regex=True)
    df['ISDATA'] = df['TRACERTYPE'].str.endswith('_DATA')
    return df


def load_class_df(path):
    """
    Load classification data from FITS file into a pandas DataFrame.

    Args:
        path (str): Path to the FITS file.
    Returns:
        pd.DataFrame: DataFrame containing the classification data.
    """
    tbl = Table.read(path, memmap=True)
    df = tbl.to_pandas()
    return df[['TARGETID','NDATA','NRAND','ISDATA']]


def load_prob_df(path):
    """
    Load probability data from FITS file into a pandas DataFrame.
    
    Args:
        path (str): Path to the FITS file.
    Returns:
        pd.DataFrame: DataFrame containing the probability data with assigned classes.
    """
    tbl = Table.read(path)
    df = tbl.to_pandas()
    prob_cols = [c for c in ['PVOID','PSHEET','PFILAMENT','PKNOT'] if c in df]
    df['CLASS'] = df[prob_cols].idxmax(axis=1).str[1:].str.lower()
    return df[['TARGETID','CLASS']]


def compute_r(df):
    """
    Compute the r for each entry in the DataFrame.

    Args:
        df (pd.DataFrame): DataFrame containing 'NDATA' and 'NRAND' columns.
    Returns:
        pd.DataFrame: DataFrame with an additional 'r' column.
    """
    df = df.copy()
    denom = (df['NDATA'] + df['NRAND']).replace(0, np.nan)
    df['r'] = (df['NDATA'] - df['NRAND']) / denom
    return df


def plot_z_histogram(df, zone, bins, out_dir):
    """
    Plot histogram of Z values for real and random data.
    
    Args:
        df (pd.DataFrame): DataFrame containing 'Z' and 'ISDATA' columns.
        zone (int): Zone number for title and filename.
        bins (int): Number of bins for the histogram.
        out_dir (str): Output directory to save the plot.
    """
    real, rand = df[df['ISDATA']]['Z'], df[~df['ISDATA']]['Z']

    fig, ax = plt.subplots()
    ax.grid(linewidth=0.7)
    ax.hist([real, rand], bins=bins, label=['Data','Random'], alpha=0.7)
    ax.set(xlabel='Z', ylabel='Count', title=f'Zone {zone}')
    ax.legend()

    path = os.path.join(out_dir, f'z_hist_zone_{zone}.png')
    fig.savefig(path, dpi=360); plt.close(fig)


def plot_radial_distribution(raw_df, zone, tracers, out_dir, bins):
    """
    Plot radial distribution histograms for specified tracers.

    Args:
        raw_df (pd.DataFrame): DataFrame containing raw data with 'XCART', 
                    'YCART', 'ZCART', 'TRACERTYPE', and 'RANDITER' columns.
        zone (int): Zone number for title and filename.
        tracers (list): List of tracer types to plot.
        out_dir (str): Output directory to save the plots.
    """
    fig, axes = plt.subplots(1, len(tracers), figsize=(4*len(tracers),4))

    for ax, tracer in zip(axes, tracers):
        if tracer=='BGS_ANY':
            tracer = 'BGS'
        sub = raw_df[raw_df['TRACERTYPE'].str.startswith(tracer)]
        real = sub[sub['RANDITER']==-1]
        rand = sub[sub['RANDITER']!=-1].sample(len(real), random_state=0)

        r_real = np.linalg.norm(real[['XCART','YCART','ZCART']], axis=1)
        r_rand = np.linalg.norm(rand[['XCART','YCART','ZCART']], axis=1)

        ax.grid(linewidth=0.7)
        ax.hist([r_real, r_rand], bins, label=['Real','Random'], alpha=0.7)
        ax.set(title=tracer, xlabel=r'$r$', ylabel='Count')
        ax.legend(fontsize=6)

    fig.suptitle(f'Radial Zone {zone}')
    fig.tight_layout()
    path = os.path.join(out_dir, f'radial_zone_{zone}.png')
    fig.savefig(path, dpi=360); plt.close(fig)


def plot_cdf(df_r, zone, tracers, out_dir):
    """
    Plot CDF of r values for specified tracers.

    Args:
        df_r (pd.DataFrame): DataFrame containing 'r', 'TRACERTYPE', and 'ISDATA' columns.
        zone (int): Zone number for title and filename.
        tracers (list): List of tracer types to plot.
        out_dir (str): Output directory to save the plot.
    """
    cmap = {'BGS_ANY':'blue','ELG':'red','LRG':'green','QSO':'purple'}
    fig, ax = plt.subplots()
    ax.grid(linewidth=0.7)

    for tr in tracers:
        sub = df_r[df_r['TRACERTYPE'].str.startswith(tr)]
        sns.ecdfplot(sub[sub['ISDATA']]['r'], ax=ax, label=f'{tr} real',
                     color=cmap[tr], linewidth=1)
        sns.ecdfplot(sub[~sub['ISDATA']]['r'], ax=ax, label=f'{tr} rand',
                     color=cmap[tr], linestyle='--', linewidth=1)

    ax.set(xlabel='r', ylabel='CDF', title=f'CDF Zone {zone}')
    ax.legend(fontsize=6)
    path = os.path.join(out_dir, f'cdf_zone_{zone}.png')
    fig.savefig(path, dpi=360); plt.close(fig)


def plot_wedges(raw_df, prob_df, zone, output_dir,
                z_max=0.4, n_ra=15, n_z=10):
    """
    Plot complete wedges for specified tracers.
    
    Args:
        raw_df (pd.DataFrame): DataFrame containing raw data with 'RA', 'DEC', 'Z', 
                    'TRACERTYPE', and 'ISDATA' columns.
        prob_df (pd.DataFrame): DataFrame containing classification data with 'TARGETID'
                                and 'CLASS' columns.
        zone (int): Zone number for title and filename.
        output_dir (str): Output directory to save the plots.
        z_max (float): Maximum z value for the plot.
        n_ra (int): Number of RA ticks.
        n_z (int): Number of Z ticks.
    """
    out_dir = os.path.join(output_dir, "wedges/complete")
    os.makedirs(out_dir, exist_ok=True)

    tracers = ["BGS_ANY", "LRG", "ELG", "QSO"]
    df = raw_df.merge(prob_df, on='TARGETID', how='left')
    real = df[df['ISDATA']]

    zs = np.linspace(0, z_max, 300)

    fig, axes = plt.subplots(1, len(tracers), figsize=(4*len(tracers), 20),
                             sharey=False, gridspec_kw={'wspace': 0.5})
    axes = axes.flatten()

    for i, tracer in enumerate(tracers):
        ax = axes[i]
        sub = real[real['BASE'] == tracer]

        title = tracer.split('_')[0] if tracer=='BGS_ANY' else tracer
        ax.set_title(f"{title}", fontsize=16, y=1.05)

        ax.set_frame_on(False)
        ax.yaxis.tick_right()
        ax.tick_params(left=False, right=True,
                       labelleft=False, labelright=True)
        ax.set_xticks([])

        z_sub_max = max(z_max, sub['Z'].max() * 1.02)
        ra_min, ra_max = sub['RA'].min(), sub['RA'].max()
        ra_ctr = 0.5*(ra_min + ra_max)
        dec_ctr = 0.5*(sub['DEC'].min() + sub['DEC'].max())
        Dc_sub = Planck18.comoving_distance(z_sub_max).value
        half_w = Dc_sub * np.deg2rad(ra_max - ra_ctr) * np.cos(np.deg2rad(dec_ctr))

        z_ticks = np.linspace(0, z_sub_max, n_z)
        ra_ticks = np.linspace(ra_min, ra_max, n_ra)

        for z0 in z_ticks:
            w0 = half_w * (z0 / z_sub_max)
            ax.hlines(z0, -w0, w0, color='gray', lw=0.5, alpha=0.5)
        for rt in ra_ticks[::4]:
            dx = Dc_sub * np.deg2rad(rt - ra_ctr) * np.cos(np.deg2rad(dec_ctr))
            ax.plot(dx / z_sub_max * zs, zs, color='gray', lw=0.5, alpha=0.5)

        for cls in ['void','sheet','filament','knot']:
            df_cls = sub[sub['CLASS'] == cls]
            if df_cls.empty: continue 
            Dc_vals = Planck18.comoving_distance(df_cls['Z']).value
            x_vals = Dc_vals * np.deg2rad(df_cls['RA'] - ra_ctr) * np.cos(np.deg2rad(dec_ctr))
            y_vals = df_cls['Z'].values

            size = 10 if cls in ('void','knot') else 4
            ax.scatter(x_vals, y_vals, s=size, c=CLASS_COLORS[cls], zorder=CLASS_ZORDER[cls])

        ax.plot([-half_w, 0], [z_sub_max, 0], 'k-', lw=1.5, zorder=CLASS_ZORDER['knot']+1)
        ax.plot([half_w, 0], [z_sub_max, 0], 'k-', lw=1.5, zorder=CLASS_ZORDER['knot']+1)

        ax.set_xlim(-half_w, half_w)
        ax.set_ylim(0, z_sub_max)
        ax.set_yticks(z_ticks)
        ax.set_yticklabels([f"{zt:.1f}" for zt in z_ticks], fontsize=13)
        if i == 0:
            ax.set_ylabel('z', fontsize=20)

        def x2ra(xv):
            return ra_ctr + np.rad2deg(xv / (Dc_sub * np.cos(np.deg2rad(dec_ctr))))
        def ra2x(rv):
            return Dc_sub * np.deg2rad(rv - ra_ctr) * np.cos(np.deg2rad(dec_ctr))
        sec = ax.secondary_xaxis('top', functions=(x2ra, ra2x))
        sec.set_xticks(ra_ticks[::4])
        sec.set_xticklabels([f"{rt:.0f}" for rt in ra_ticks[::4]], fontsize=13)
        sec.set_xlabel('RA [deg]', fontsize=14, labelpad=13)

    handles = [Line2D([], [], marker='o', color=CLASS_COLORS[c], linestyle='', markersize=8,
                      label=c) for c in CLASS_COLORS]
    fig.legend(handles, CLASS_COLORS.keys(), bbox_to_anchor=(0.5, 0.965),
               loc='upper center', ncol=len(CLASS_COLORS), fontsize=14)
    plt.suptitle(f"Zone {zone}", fontsize=18)

    path = os.path.join(out_dir, f"wedge_zone_{zone:02d}.png")
    fig.savefig(path, dpi=360)
    plt.close(fig)
    # print(f"Saved {path}")


def plot_wedges_slice(raw_df, prob_df, zone, output_dir,
                        n_ra=15, n_z=10, offset=0.3):
    """
    Plot wedge slices for specified tracers.
    
    Args:
        raw_df (pd.DataFrame): DataFrame containing raw data with 'RA', 'DEC', 'Z', 
                    'TRACERTYPE', and 'ISDATA' columns.
        prob_df (pd.DataFrame): DataFrame containing classification data with 'TARGETID'
                                and 'CLASS' columns.
        zone (int): Zone number for title and filename.
        output_dir (str): Output directory to save the plots.
        n_ra (int): Number of RA ticks.
        n_z (int): Number of Z ticks.
        offset (float): Offset to apply when determining z range.
    """
    out_dir = os.path.join(output_dir, "wedges/slice")
    os.makedirs(out_dir, exist_ok=True)

    tracers = ["BGS_ANY","LRG","ELG","QSO"]
    df = raw_df.merge(prob_df, on='TARGETID', how='left')
    real = df[df['ISDATA']]

    mins, maxs = [], []
    for tr in tracers:
        zs = real.loc[real['BASE'] == tr, 'Z']
        mins.append(zs.min()); maxs.append(zs.max())
    z_lo = max(mins) - offset
    z_hi = min(maxs) + offset
    if z_hi <= z_lo:
        raise RuntimeError(f"Empty z-range [{z_lo}, {z_hi}]")

    ra_min_g, ra_max_g = real['RA'].min(), real['RA'].max()
    ra_ctr_g = 0.5*(ra_min_g + ra_max_g)
    dec_ctr_g = 0.5*(real['DEC'].min() + real['DEC'].max())

    ra_ticks = np.linspace(ra_min_g, ra_max_g, n_ra)
    Dc_hi = Planck18.comoving_distance(z_hi).value
    half_w_g = Dc_hi * np.deg2rad(ra_max_g - ra_ctr_g) * np.cos(np.deg2rad(dec_ctr_g))

    zs = np.linspace(z_lo, z_hi, 300)
    z_ticks = np.linspace(z_lo, z_hi, n_z)

    fig, axes = plt.subplots(1, len(tracers), figsize=(4*len(tracers), 20),
                             sharex=True, sharey=True, gridspec_kw={'wspace': 0.5})
    axes = axes.flatten()

    for i, tr in enumerate(tracers):
        ax = axes[i]
        sub = real[real['BASE'] == tr]

        ax.set_title(f"{tr.replace('_ANY','')}", fontsize=16, y=1.05)
        ax.set_frame_on(False)
        ax.yaxis.tick_right()
        ax.tick_params(left=False, right=True,
                       labelleft=False, labelright=True)
        ax.set_xticks([])

        for z0 in z_ticks:
            w0 = half_w_g * ((z0 - z_lo) / (z_hi - z_lo))
            ax.hlines(z0, -w0, w0, color='gray', lw=0.5, alpha=0.5)
        for rt in ra_ticks[::4]:
            dx = Planck18.comoving_distance(zs).value * \
                 np.deg2rad(rt - ra_ctr_g) * np.cos(np.deg2rad(dec_ctr_g))
            ax.plot(dx, zs, color='gray', lw=0.5, alpha=0.5)

        for cls in ['void','sheet','filament','knot']:
            df_cls = sub[sub['CLASS'] == cls]
            if df_cls.empty: continue

            Dc_vals = Planck18.comoving_distance(df_cls['Z']).value
            x_vals = Dc_vals * np.deg2rad(df_cls['RA'] - ra_ctr_g) * np.cos(np.deg2rad(dec_ctr_g))
            y_vals = df_cls['Z']
            size = 10 if cls in ('void','knot') else 4
            ax.scatter(x_vals, y_vals, s=size, c=CLASS_COLORS[cls],
                       zorder=CLASS_ZORDER[cls])

        for ra_edge in (real['RA'].min(), real['RA'].max()):
            x_edge = Planck18.comoving_distance(zs).value * \
                     np.deg2rad(ra_edge - ra_ctr_g) * np.cos(np.deg2rad(dec_ctr_g))
            ax.plot(x_edge, zs, 'k-', lw=2)

        ax.set_xlim(-half_w_g, half_w_g)
        ax.set_ylim(z_lo, z_hi)
        ax.set_yticks(z_ticks)
        ax.set_yticklabels([f"{zt:.1f}" for zt in z_ticks], fontsize=13)
        if i == 0:
            ax.set_ylabel('z', fontsize=20)

        def x2ra(xv):
            return ra_ctr_g + np.rad2deg(xv / (Dc_hi * np.cos(np.deg2rad(dec_ctr_g))))
        def ra2x(rv):
            return Dc_hi * np.deg2rad(rv - ra_ctr_g) * np.cos(np.deg2rad(dec_ctr_g))
        sec = ax.secondary_xaxis('top', functions=(x2ra, ra2x))
        sec.set_xticks(ra_ticks[::4])
        sec.set_xticklabels([f"{rt:.0f}" for rt in ra_ticks[::4]], fontsize=13)
        sec.set_xlabel('RA (deg)', fontsize=14, labelpad=13)

    handles = [Line2D([], [], marker='o', color=CLASS_COLORS[c], linestyle='',
                      markersize=8, label=c) for c in CLASS_COLORS]
    fig.legend(handles, CLASS_COLORS.keys(), bbox_to_anchor=(0.5, 0.965),
               loc='upper center', ncol=len(CLASS_COLORS), fontsize=14)
    plt.suptitle(f"Zone {zone}", fontsize=18)

    path = os.path.join(out_dir, f"wedge_zone_{zone:02d}.png")
    fig.savefig(path, dpi=360)
    plt.close(fig)
    # print(f"Saved {path}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--raw-dir', required=True)
    p.add_argument('--class-dir', required=True)
    p.add_argument('--zones', nargs='+', type=int, default=None)
    p.add_argument('--output', required=True)
    p.add_argument('--bins', type=int, default=10)
    p.add_argument('--tracers', nargs='+',
                   default=['BGS_ANY','ELG','LRG','QSO'])
    p.add_argument('--plot-z', action='store_true', default=True)
    p.add_argument('--plot-radial', action='store_true', default=True)
    p.add_argument('--plot-cdf', action='store_true', default=True)
    p.add_argument('--plot-wedges', action='store_true', default=True)
    return p.parse_args()


def main():
    args = parse_args()
    zones = infer_zones(args.raw_dir, args.zones)
    outdirs = make_output_dirs(args.output)

    raw_cache, class_cache, prob_cache = {}, {}, {}

    for zone in zones:
        raw_path, cls_path = get_zone_paths(args.raw_dir, args.class_dir, zone)
        prob_path = get_prob_path(args.raw_dir, args.class_dir, zone)

        raw_df = raw_cache.setdefault(zone, load_raw_df(raw_path))
        cls_df = class_cache.setdefault(zone, load_class_df(cls_path))
        cls_df = cls_df[cls_df['ISDATA'] == True]
        # cls_df = cls_df.drop_duplicates(subset='TARGETID')
        merged = raw_df.merge(cls_df[['TARGETID','NDATA','NRAND']], on='TARGETID', how='left')
        r_df = compute_r(merged)

        if args.plot_z:
            plot_z_histogram(merged, zone, args.bins, outdirs['z'])
        if args.plot_cdf:
            plot_cdf(r_df, zone, args.tracers, outdirs['cdf'])
        if args.plot_radial:
            plot_radial_distribution(raw_df, zone, args.tracers, outdirs['radial'], args.bins)
        if args.plot_wedges:
            prob_df = prob_cache.setdefault(zone, load_prob_df(prob_path))
            plot_wedges(raw_df, prob_df, zone, args.output)
            plot_wedges_slice(raw_df, prob_df, zone, args.output)


if __name__ == '__main__':
    main()
#python plot_add.py --raw-dir /pscratch/sd/v/vtorresg/cosmic-web/edr/raw --class-dir /pscratch/sd/v/vtorresg/cosmic-web/edr/class --output ../plots