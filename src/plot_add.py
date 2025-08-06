import os, argparse
from astropy.table import Table, join
import matplotlib.pyplot as plt
import numpy as np
from astropy.cosmology import Planck18

import seaborn as sns
import matplotlib.pyplot as plt
plt.rcParams.update({'font.family': 'serif', 'font.size': 12, 'axes.labelsize': 12,
                     'xtick.labelsize': 10,'ytick.labelsize': 10, 'legend.fontsize': 10,})


def get_zone_paths(raw_dir, class_dir, zone):
    """
    Construct file paths for raw and class tables for a given zone.
    Pads single-digit zones with a leading zero.
    Args:
        raw_dir: Directory with raw zone FITS files.
        class_dir: Directory with class zone FITS files.
        zone: Zone number (int).
    Returns:
        Tuple of (raw_file_path, class_file_path)
    """
    zone_str = f"{zone:02d}"
    raw_file = os.path.join(raw_dir, f"zone_{zone_str}.fits.gz")
    class_file = os.path.join(class_dir, f"zone_{zone_str}_class.fits.gz")
    return raw_file, class_file


def load_zone_tables(raw_path, class_path):
    """
    Load raw and class tables from disk.
    Args:
        raw_path: Path to the raw zone FITS file.
        class_path: Path to the class zone FITS file.
    Returns:
        Tuple of (raw_table, class_table)
    """
    return Table.read(raw_path), Table.read(class_path, memmap=True)


def merge_tables(raw_tbl, class_tbl):
    """
    Join raw and class tables on TARGETID to add ISDATA flag.
    """
    return join(raw_tbl, class_tbl['TARGETID', 'ISDATA'], keys='TARGETID', join_type='left')


def extract_z_values(joined_tbl):
    """
    From joined table, return arrays of Z for data and random.
    Args:
        joined_tbl: Table with ISDATA and Z columns.
    Returns:
        Tuple of (z_data, z_rand)
    """
    is_data = joined_tbl['ISDATA']
    z = joined_tbl['Z']
    return z[is_data], z[~is_data]


def infer_zones(raw_dir, provided_zones):
    """
    Determine zones list from raw_dir if not provided.
    
    Args:
        raw_dir: Directory with raw zone FITS files.
        provided_zones: List of zones to process, if any.
    Returns:
        List of zone numbers to process.
    """
    if provided_zones is not None:
        return provided_zones
    files = [f for f in os.listdir(raw_dir)
             if f.startswith('zone_') and f.endswith('.fits.gz')]
    return sorted(int(f.split('_')[1].split('.')[0]) for f in files)


def make_output_dirs(base_output):
    """
    Create subdirectories for each plot type.
    
    Args:
        base_output: Base output directory for plots.
    Returns:
        Dictionary with directories for z histograms, radial plots, and cdf plots.
    """
    dirs = {'z': os.path.join(base_output, 'z_histograms'),
            'radial': os.path.join(base_output, 'radial'),
            'cdf': os.path.join(base_output, 'cdf')}
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)
    return dirs


def load_zone_dataframe(raw_dir, zone):
    """
    Load a zone FITS file into a pandas DataFrame, decoding bytes.

    Args:
        raw_dir: Directory with raw zone FITS files.
        zone: Zone number to load.
    Returns:
        DataFrame with columns from the FITS file.
    """
    path = os.path.join(raw_dir, f"zone_{zone:02d}.fits.gz")
    df = Table.read(path).to_pandas()
    df['TRACERTYPE'] = df['TRACERTYPE'].apply(lambda x: x.decode('utf-8') if
                                              isinstance(x, (bytes, bytearray)) else x)
    return df


def compute_r_df(raw_dir, class_dir, zone):
    """
    Compute r = (NDATA - NRAND)/(NDATA + NRAND) and attach TRACERTYPE.

    Args:
        raw_dir: Directory with raw zone FITS files.
        class_dir: Directory with class zone FITS files.
        zone: Zone number to process.
    Returns:
        DataFrame with TARGETID, TRACERTYPE, NDATA, NRAND, and
    """
    raw_path, class_path = get_zone_paths(raw_dir, class_dir, zone)
    raw_tbl = Table.read(raw_path).to_pandas()
    class_tbl = Table.read(class_path).to_pandas()
    df = class_tbl.merge(raw_tbl[['TARGETID','TRACERTYPE']], on='TARGETID', how='left')
    
    df['TRACERTYPE'] = df['TRACERTYPE'].apply(lambda x: x.decode('utf-8') if 
                                              isinstance(x, (bytes, bytearray)) else x)
    df['r'] = (df['NDATA'] - df['NRAND']) / (df['NDATA'] + df['NRAND']).replace(0, np.nan)
    return df
    

def plot_histogram(z_data, z_rand, zone, bins, output_dir):
    """
    Plot and save histogram for one zone

    Args:
        z_data: Array of Z values for data.
        z_rand: Array of Z values for random.
        zone: Zone number for the plot title.
        bins: Number of bins for the histogram.
        output_dir: Directory to save the histogram plot.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(z_data, bins=bins, alpha=0.7, label='Data', zorder=10)
    ax.hist(z_rand, bins=bins, alpha=0.7, label='Random')
    ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
    ax.set(xlabel='Z', ylabel='Count', title=f'Z Distribution for Zone {zone}')
    ax.legend()

    out_file = os.path.join(output_dir, f'z_hist_zone_{zone}.png')
    fig.savefig(out_file, dpi=360)
    plt.close(fig)


def plot_z_histograms(raw_dir, class_dir, zones, output_dir, bins=50):
    """
    Plot Z histograms for specified zones.

    Args:
        raw_dir: Directory with raw zone FITS files.
        class_dir: Directory with class zone FITS files.
        zones: List of zone numbers to process.
        output_dir: Directory to save the histogram plots.
        bins: Number of bins for the histogram.
    """
    os.makedirs(output_dir, exist_ok=True)
    for zone in zones:
        raw_path, class_path = get_zone_paths(raw_dir, class_dir, zone)
        raw_tbl, class_tbl = load_zone_tables(raw_path, class_path)
        joined = merge_tables(raw_tbl, class_tbl)
        z_data, z_rand = extract_z_values(joined)
        plot_histogram(z_data, z_rand, zone, bins, output_dir)


def plot_radial_by_zone(raw_dir, zones, tracers, output_dir=None):
    """
    Plot radial distributions for each tracer in each zone.

    Args:
        raw_dir: Directory with raw zone FITS files.
        zones: List of zone numbers to process.
        tracers: List of tracer types to plot.
        output_dir: Directory to save the radial distribution plots.
    """
    for zone in zones:
        df = load_zone_dataframe(raw_dir, zone)
        fig, axes = plt.subplots(1, len(tracers), figsize=(4*len(tracers), 4))

        for ax, tracer in zip(axes, tracers):
            sub = df[df['TRACERTYPE'].str.startswith(tracer)]
            real = sub[sub['RANDITER'] == -1]
            rand = sub[sub['RANDITER'] != -1].sample(n=len(real), random_state=0)
            d_real = np.linalg.norm(real[['XCART','YCART','ZCART']].values, axis=1)
            d_rand = np.linalg.norm(rand[['XCART','YCART','ZCART']].values, axis=1)

            ax.hist(d_real, bins=30, alpha=0.7, label='Real', zorder=10)
            ax.hist(d_rand, bins=30, alpha=0.7, label='Random')
            ax.set_title(tracer)
            ax.set_xlabel(r'$r = \sqrt{x^2 + y^2 + z^2}$')
            ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
            ax.legend(loc='upper left', fontsize=6)
    
        axes[0].set_ylabel('Count')
        fig.suptitle(f'Radial Distribution Zone {zone}')
        plt.tight_layout()

        os.makedirs(output_dir, exist_ok=True)
        fig.savefig(os.path.join(output_dir, f'radial_zone_{zone}.png'), dpi=360)
        plt.close(fig)


def plot_cdf_by_zone(raw_dir, class_dir, zones, tracers, output_dir=None):
    """
    Plot CDF of r values for each tracer in each zone.

    Args:
        raw_dir: Directory with raw zone FITS files.
        class_dir: Directory with class zone FITS files.
        zones: List of zone numbers to process.
        tracers: List of tracer types to plot.
        output_dir: Directory to save the CDF plots.
    """
    color_map = {'BGS_ANY': 'blue', 'ELG': 'red', 'LRG': 'green', 'QSO': 'purple'}

    fig, axes = plt.subplots(1, len(zones), figsize=(5*len(zones), 5), sharey=True)
    if len(zones) == 1:
        axes = [axes]

    for ax, zone in zip(axes, zones):
        df = compute_r_df(raw_dir, class_dir, zone)

        for tracer in tracers:
            sub = df[df['TRACERTYPE'].str.startswith(tracer)]
            is_data = sub['ISDATA']
            color = color_map.get(tracer, 'black')

            sns.ecdfplot(sub.loc[is_data, 'r'], ax=ax, label=f'{tracer} real',
                         color=color, linewidth=1)
            sns.ecdfplot(sub.loc[~is_data, 'r'], ax=ax, label=f'{tracer} rand',
                         color=color, linestyle='--', linewidth=1)
        ax.set_xlabel('r')
        ax.set_title(f'Zone {zone}')
        ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
        ax.legend(loc='upper left', fontsize=6)
    axes[0].set_ylabel('CDF')
    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    fig.savefig(os.path.join(output_dir, 'cdf_zones.png'), dpi=360)
    plt.close(fig)

#! ---------------------------------------------------------
def make_merged(zone, raw_dir, prob_dir):
    """
    Merge raw and classification tables for a given zone into a single DataFrame.
    Adds columns: RA, DEC, Z, XCART, YCART, ZCART from raw, and ISDATA, NDATA, NRAND, r, CLASS from classification.
    Args:
        zone (int): Zone number to merge.
        raw_dir (str): Directory with raw FITS files.
        prob_dir (str): Directory with classification FITS files.
    Returns:
        pandas.DataFrame: Merged data containing all necessary columns.
    """
    # Load raw and classification
    raw_path = os.path.join(raw_dir, f"zone_{zone:02d}.fits.gz")
    class_path = os.path.join(prob_dir, f"zone_{zone:02d}_class.fits.gz")
    raw_df = Table.read(raw_path).to_pandas()
    class_df = Table.read(class_path).to_pandas()
    # Decode tracer types
    raw_df['TRACERTYPE'] = raw_df['TRACERTYPE'].apply(
        lambda x: x.decode('utf-8') if isinstance(x, (bytes, bytearray)) else x
    )
    # Merge on TARGETID
    df = raw_df.merge(
        class_df[['TARGETID', 'ISDATA', 'NDATA', 'NRAND']],
        on='TARGETID', how='left'
    )
    # Compute r and CLASS
    df['r'] = (df['NDATA'] - df['NRAND']) / (df['NDATA'] + df['NRAND']).replace(0, np.nan)
    # CLASS labels based on r
    def classify_type(r):
        if -1.0 <= r <= -0.9: return 'void'
        elif -0.9 < r <= 0.0: return 'sheet'
        elif 0.0 < r <= 0.9: return 'filament'
        elif 0.9 < r <= 1.0: return 'knot'
        else: return 'undefined'
    df['CLASS'] = df['r'].apply(classify_type)
    return df


def plot_colored_wedges(raw_dir, prob_dir, zone, output_dir,
                        tracers=['BGS_ANY','ELG','LRG','QSO'],
                        z_max=0.213, n_ra=20, n_z=20):
    """
    Plot wedge diagrams for a given zone in a 2xN grid (data on top row, random on bottom).
    Args:
        raw_dir (str): Directory with raw FITS files.
        prob_dir (str): Directory with classification FITS files.
        zone (int): Zone number to plot.
        output_dir (str): Directory to save the output plot.
        tracers (list): List of tracer prefixes to include.
        z_max (float): Maximum redshift to plot.
        n_ra (int): Number of RA grid lines.
        n_z (int): Number of Z grid lines.
    """
    os.makedirs(output_dir, exist_ok=True)
    COLORS = {'data': 'red', 'random': 'red'}  # solid and dashed will differentiate
    df = make_merged(zone, raw_dir, prob_dir)
    ra_min, ra_max = df['RA'].min(), df['RA'].max()
    ra_ctr = 0.5 * (ra_min + ra_max)
    dec_ctr = 0.5 * (df['DEC'].min() + df['DEC'].max())
    Dc_max = Planck18.comoving_distance(z_max).value
    half_w = Dc_max * np.deg2rad(ra_max - ra_ctr) * np.cos(np.deg2rad(dec_ctr))
    z_ticks = np.linspace(0, z_max, n_z)
    ra_ticks = np.linspace(ra_min, ra_max, n_ra)
    zs = np.linspace(0, z_max, 200)
    fig, axes = plt.subplots(2, len(tracers),
                             figsize=(4*len(tracers), 8),
                             sharex=True, sharey=True)

    # Draw grid
    for ax_row in axes:
        for ax in ax_row:
            for z0 in z_ticks:
                w0 = half_w * (z0 / z_max)
                ax.hlines(z0, -w0, w0, color='gray', lw=0.5, alpha=0.5)
            for rt in ra_ticks[::max(1, n_ra//4)]:
                dx = Dc_max * np.deg2rad(rt - ra_ctr) * np.cos(np.deg2rad(dec_ctr))
                ax.plot(dx/z_max * zs, zs, color='gray', lw=0.5, alpha=0.5)
            ax.set_frame_on(False)
            ax.set_xlim(-half_w, half_w)
            ax.set_ylim(0, z_max)
            ax.set_xticks([])
            ax.set_yticks([])

    # Compute positions
    Dc = Planck18.comoving_distance(df['Z'].values).value
    x = Dc * np.deg2rad(df['RA'] - ra_ctr) * np.cos(np.deg2rad(dec_ctr))

    # Plot data and random per tracer
    for i, tracer in enumerate(tracers):
        sub = df[df['TRACERTYPE'].str.startswith(tracer)]
        for row, (label, mask) in enumerate([('data', sub['RANDITER']==-1),
                                              ('random', sub['RANDITER']!=-1)]):
            ax = axes[row, i]
            grp = sub[mask]
            Dc_grp = Planck18.comoving_distance(grp['Z'].values).value
            x_grp = Dc_grp * np.deg2rad(grp['RA'] - ra_ctr) * np.cos(np.deg2rad(dec_ctr))
            ax.scatter(x_grp, grp['Z'], s=1,
                       color=COLORS[label],
                       alpha=0.6,
                       label=label,
                       linestyle='-' if label=='data' else '--')
            if row == 0:
                ax.set_title(tracer)

    # Legend
    handles, labels = axes[0,0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower right', fontsize=6)
    plt.tight_layout()
    out_file = os.path.join(output_dir, f'wedges_zone_{zone:02d}.png')
    fig.savefig(out_file, dpi=360)
    plt.close(fig)

#! ---------------------------------------------------------

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw-dir', required=True, help='Directory with raw zone FITS files')
    parser.add_argument('--class-dir', required=True, help='Directory with class zone FITS files')
    parser.add_argument('--zones', nargs='+', type=int, default=None,
                        help='List of zones to process (default: all in raw-dir)')

    parser.add_argument('--output', required=True, help='Base output directory for plots')
    parser.add_argument('--bins', type=int, default=50, help='Number of histogram bins')
    parser.add_argument('--tracers', nargs='+', default=['BGS_ANY','ELG','LRG','QSO'],
                        help='List of tracers for radial and cdf plots')

    parser.add_argument('--plot-z', default=True, action='store_true', help='Plot Z histograms')
    parser.add_argument('--plot-radial', default=True, action='store_true', help='Plot radial distributions')
    parser.add_argument('--plot-cdf', default=True, action='store_true', help='Plot cdf of r values')
    parser.add_argument('--plot-wedges', default=True, action='store_true', help='Plot colored wedges')
    return parser.parse_args()


def main():
    args = parse_arguments()
    zones = infer_zones(args.raw_dir, args.zones)
    out_dirs = make_output_dirs(args.output)

    # if args.plot_z:
    #     plot_z_histograms(args.raw_dir, args.class_dir, zones, out_dirs['z'], args.bins)
    # if args.plot_radial:
    #     plot_radial_by_zone(args.raw_dir, zones, args.tracers, out_dirs['radial'])
    # if args.plot_cdf:
    #     plot_cdf_by_zone(args.raw_dir, args.class_dir, zones, args.tracers, out_dirs['cdf'])
    if args.plot_wedges:
        plot_colored_wedges(args.raw_dir, args.class_dir, zones[0], args.output)

if __name__ == '__main__':
    main()