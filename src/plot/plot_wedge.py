import os, re, argparse
import numpy as np
import pandas as pd
from astropy.table import Table
from astropy.cosmology import Planck18
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

plt.rcParams.update({'font.family': 'serif', 'font.size': 12, 'axes.labelsize': 12,
                     'xtick.labelsize': 10, 'ytick.labelsize': 10,'legend.fontsize': 10,})

CLASS_COLORS = {'void': 'red', 'sheet': '#9ecae1', 'filament': '#3182bd', 'knot': 'navy'}
CLASS_ZORDER = {'void': 0, 'sheet': 1, 'filament': 2, 'knot': 3}


def _tracer_mask(df, tracer):
    """
    Create a boolean mask for the specified tracer in the DataFrame.
    
    Args:
        df (pd.DataFrame): The input DataFrame.
        tracer (str): The tracer name to mask.
    Returns:
        pd.Series: A boolean mask for the specified tracer.
    """
    base = tracer.rsplit('_', 1)[0] if '_' in tracer else tracer
    base = str(base).strip()
    base_col = df.get('BASE', pd.Series(index=df.index, dtype=object)).astype(str)
    core_col = df.get('BASE_CORE', base_col).astype(str)
    starts = base_col.str.startswith(base, na=False)
    exact = (base_col == tracer)
    core = (core_col == base)
    return exact | core | starts


def infer_zones(raw_dir, zones_cli):
    """
    Infer the zones to process from the raw directory or command line input.
    
    Args:
        raw_dir (str): The path to the raw data directory.
        zones_cli (list): A list of zone numbers from the command line.
    Returns:
        list: A list of inferred zone numbers.
    """
    if zones_cli:
        return sorted(set(int(z) for z in zones_cli))
    pat = re.compile(r'^zone_(\d{1,3})\.fits(\.gz)?$')
    found = []
    if os.path.isdir(raw_dir):
        for fn in os.listdir(raw_dir):
            m = pat.match(fn)
            if m:
                found.append(int(m.group(1)))
    if not found:
        raise FileNotFoundError(f'No zones in {raw_dir}')
    return sorted(found)


def make_output_dirs(base_out):
    """
    Create the necessary output directories.
    
    Args:
        base_out (str): The base output directory.
    """
    os.makedirs(base_out, exist_ok=True)
    os.makedirs(os.path.join(base_out, 'data'), exist_ok=True)
    os.makedirs(os.path.join(base_out, 'tracer_zone'), exist_ok=True)


def get_zone_paths(raw_dir, class_dir, zone):
    """
    Get the file paths for the raw and class data for a specific zone.
    
    Args:
        raw_dir (str): The path to the raw data directory.
        class_dir (str): The path to the class data directory.
        zone (int): The zone number to process.
    Returns:
        tuple: A tuple containing the raw and class data file paths.
    """
    raw_gz = os.path.join(raw_dir, f'zone_{zone:02d}.fits.gz')
    raw_fx = os.path.join(raw_dir, f'zone_{zone:02d}.fits')
    cls_gz = os.path.join(class_dir, f'zone_{zone:02d}_class.fits.gz')
    cls_fx = os.path.join(class_dir, f'zone_{zone:02d}_class.fits')

    raw_path = raw_gz if os.path.exists(raw_gz) else raw_fx
    cls_path = cls_gz if os.path.exists(cls_gz) else cls_fx

    if not os.path.exists(raw_path):
        raise FileNotFoundError(f'No RAW: {raw_gz} ni {raw_fx}')
    if not os.path.exists(cls_path):
        raise FileNotFoundError(f'No CLASS: {cls_gz} ni {cls_fx}')
    return raw_path, cls_path


def get_prob_path(class_dir, zone):
    """
    Get the file path for the probability data for a specific zone.

    Args:
        class_dir (str): The path to the class data directory.
        zone (int): The zone number to process.
    Returns:
        str: The file path for the probability data.
    """
    prob_gz = os.path.join(class_dir, f'zone_{zone:02d}_probability.fits.gz')
    prob_fx = os.path.join(class_dir, f'zone_{zone:02d}_probability.fits')
    prob_path = prob_gz if os.path.exists(prob_gz) else prob_fx
    if not os.path.exists(prob_path):
        raise FileNotFoundError(f'No existe PROB: {prob_gz} ni {prob_fx}')
    return prob_path


def load_raw_df(raw_path):
    """
    Load the raw data from a FITS file.

    Args:
        raw_path (str): The path to the raw data file.
    Returns:
        pd.DataFrame: The loaded raw data as a DataFrame.
    """
    t = Table.read(raw_path, memmap=True)
    df = t.to_pandas()

    if 'RANDITER' in df.columns:
        df['ISDATA'] = (df['RANDITER'].to_numpy() == -1)
    else:
        df['ISDATA'] = True

    if 'TRACERTYPE' in df.columns:
        def _norm_tt(x):
            if isinstance(x, (bytes, bytearray)):
                try:
                    x = x.decode('utf-8', errors='ignore')
                except Exception:
                    x = str(x)
            return str(x).strip()
        df['BASE'] = df['TRACERTYPE'].apply(_norm_tt)
        df['BASE_CORE'] = df['BASE'].str.rsplit('_', n=1).str[0]
    else:
        df['BASE'] = 'ALL'
        df['BASE_CORE'] = 'ALL'

    if 'TARGETID' in df.columns:
        df['TARGETID'] = df['TARGETID'].astype(np.int64)

    required = ['RA', 'DEC', 'Z', 'TARGETID']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f'missing col {missing} in RAW: {raw_path}')
    return df


def load_prob_df(prob_path):
    """
    Load the probability data from a FITS file.

    Args:
        prob_path (str): The path to the probability data file.
    Returns:
        pd.DataFrame: The loaded probability data as a DataFrame.
    """
    t = Table.read(prob_path, memmap=True)
    df = t.to_pandas()
    for c in ['TARGETID','PVOID','PSHEET','PFILAMENT','PKNOT']:
        if c not in df.columns:
            df[c] = 0.0 if c != 'TARGETID' else df.get('TARGETID', pd.Series(dtype=np.int64))
    df['TARGETID'] = df['TARGETID'].astype(np.int64, copy=False)
    return df[['TARGETID','PVOID','PSHEET','PFILAMENT','PKNOT']]


def assign_most_likely_class(df_prob):
    """
    Assign the most likely class to each row in the probability DataFrame.
    
    Args:
        df_prob (pd.DataFrame): The probability DataFrame.
    Returns:
        pd.Series: The assigned most likely class for each row.
    """
    cols = ['PVOID', 'PSHEET', 'PFILAMENT', 'PKNOT']
    arr = df_prob[cols].to_numpy(dtype=float)
    idx = np.argmax(arr, axis=1)
    mapping = np.array(['void', 'sheet', 'filament', 'knot'], dtype=object)
    return pd.Series(mapping[idx], index=df_prob.index, dtype=object)


def _labels_for_iteration(tclass, use_isdata, iter_j):
    """
    Get the labels for a specific iteration of the classification.
    """
    df = tclass.to_pandas()
    need = {'TARGETID','RANDITER','ISDATA','NDATA','NRAND'}
    if not need.issubset(df.columns):
        raise ValueError(f'CLASS file missing columns: {sorted(need - set(df.columns))}')

    df = df[(df['ISDATA'] == bool(use_isdata)) & (df['RANDITER'] == int(iter_j))].copy()
    denom = (df['NDATA'].to_numpy(dtype=float) + df['NRAND'].to_numpy(dtype=float))
    num = (df['NDATA'].to_numpy(dtype=float) - df['NRAND'].to_numpy(dtype=float))

    r = np.full_like(denom, np.nan, dtype=float)
    np.divide(num, denom, out=r, where=(np.isfinite(denom) & (denom > 0)))
    rvals = r

    lab = np.empty(rvals.shape[0], dtype=object); lab[:] = None
    lab[np.isfinite(rvals) & (rvals <= -0.9)] = 'void'
    lab[np.isfinite(rvals) & (rvals > -0.9) & (rvals < 0.0)] = 'sheet'
    lab[np.isfinite(rvals) & (rvals >= 0.0) & (rvals < 0.9)] = 'filament'
    lab[np.isfinite(rvals) & (rvals >= 0.9)] = 'knot'

    out = pd.DataFrame({'TARGETID': df['TARGETID'].astype(np.int64), 'CLASS': lab})
    out = out.dropna(subset=['CLASS'])
    return out


def _compute_zone_params(real, z_lim):
    """
    Compute the parameters for a specific zone based on the real data.

    Args:
        real (pd.DataFrame): The real data DataFrame.
        z_lim (float): The redshift limit.
    Returns:
        tuple: The computed parameters for the zone.
    """
    zmax_data = real['Z'].max() * 1.02
    zmax = min(zmax_data, z_lim) if z_lim is not None else zmax_data

    ra_min, ra_max = real['RA'].min(), real['RA'].max()
    ra_ctr = 0.5 * (ra_min + ra_max)
    dec_ctr = 0.5 * (real['DEC'].min() + real['DEC'].max())

    Dc = Planck18.comoving_distance(zmax).value if zmax > 0 else 1.0
    half_w = Dc * np.deg2rad(ra_max - ra_ctr) * np.cos(np.deg2rad(dec_ctr))
    return ra_min, ra_max, ra_ctr, dec_ctr, Dc, half_w, zmax


def _init_ax(ax, title):
    """
    Initialize the axis for the plot.

    Args:
        ax (plt.Axes): The matplotlib axes to initialize.
        title (str): The title of the plot.
    """
    ax.set_title(title, fontsize=14, y=1.05)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_xticks([]); ax.set_yticks([])


def _draw_grid(ax, ra_min, ra_max, ra_ctr, cos_dec, Dc, half_w, zmax, n_ra, n_z):
    """
    Draw the grid lines for the plot.

    Args:
        ax (plt.Axes): The matplotlib axes to draw the grid on.
        ra_min (float): The minimum RA value.
        ra_max (float): The maximum RA value.
        ra_ctr (float): The center RA value.
        cos_dec (float): The cosine of the declination.
        Dc (float): The comoving distance.
        half_w (float): Half the width of the wedge.
        zmax (float): The maximum redshift.
        n_ra (int): The number of RA bins.
        n_z (int): The number of z bins.
    Returns:
        tuple: The z and RA ticks.
    """
    zs = np.linspace(0, zmax, 300)
    z_ticks = np.linspace(0, zmax, n_z)
    ra_ticks = np.linspace(ra_min, ra_max, n_ra)

    for z0 in z_ticks:
        w0 = half_w * (z0 / zmax) if zmax > 0 else 0.0
        ax.hlines(z0, -w0, w0, color='gray', lw=0.5, alpha=0.5)

    step = max(1, n_ra // 4)
    for rt in ra_ticks[::step]:
        dx = Dc * np.deg2rad(rt - ra_ctr) * cos_dec
        ax.plot((dx / zmax) * zs if zmax > 0 else np.zeros_like(zs), zs,
                color='gray', lw=0.5, alpha=0.5)
    return z_ticks, ra_ticks


def _plot_classes(ax, real, ra_ctr, cos_dec, Dc_all, zmax, half_w):
    """
    Plot the different classes of objects in the wedge.

    Args:
        ax (plt.Axes): The matplotlib axes to plot on.
        real (pd.DataFrame): The real data DataFrame.
        ra_ctr (float): The center RA value.
        cos_dec (float): The cosine of the declination.
        Dc_all (np.ndarray): The comoving distance for all objects.
        zmax (float): The maximum redshift.
        half_w (float): Half the width of the wedge.
    """
    for cls, color in CLASS_COLORS.items():
        sel = (real['CLASS'] == cls)

        ra_sel = real.loc[sel, 'RA'].to_numpy()
        z_sel = real.loc[sel, 'Z'].to_numpy()
        Dc_sel = Dc_all[sel.to_numpy()]

        x = Dc_sel * np.deg2rad(ra_sel - ra_ctr) * cos_dec
        y = z_sel

        scale = (y / zmax) if zmax > 0 else np.zeros_like(y)
        mask = np.abs(x) <= (half_w * scale)
        ax.scatter(x[mask], y[mask], s=(10 if cls == 'void' else 6), c=color,
                   zorder=CLASS_ZORDER[cls], edgecolor='black', lw=0.08)


def _draw_borders(ax, half_w, zmax):
    """
    Draw the borders of the wedge.

    Args:
        ax (plt.Axes): The matplotlib axes to draw the borders on.
        half_w (float): Half the width of the wedge.
        zmax (float): The maximum redshift.
    """
    ax.plot([-half_w, 0], [zmax, 0], 'k-', lw=1.5)
    ax.plot([ half_w, 0], [zmax, 0], 'k-', lw=1.5)
    ax.plot([-half_w, half_w], [zmax, zmax], 'k-', lw=1.5)
    ax.set_xlim(-half_w, half_w)
    ax.set_ylim(0, zmax)


def _annotate_ra_top(ax, ra_ticks, ra_ctr, cos_dec, Dc, zmax):
    """
    Annotate the top of the wedge with RA tick labels.

    Args:
        ax (plt.Axes): The matplotlib axes to annotate.
        ra_ticks (np.ndarray): The RA tick positions.
        ra_ctr (float): The center RA value.
        cos_dec (float): The cosine of the declination.
        Dc (float): The comoving distance.
        zmax (float): The maximum redshift.
    """
    top4 = np.linspace(ra_ticks.min(), ra_ticks.max(), 4)
    x_top = Dc * np.deg2rad(top4 - ra_ctr) * cos_dec
    for xt, rt in zip(x_top, top4):
        ax.text(xt, zmax + 0.01*zmax, f'{rt:.0f}', ha='center', va='bottom', fontsize=10)
    ax.text(0, zmax + 0.03*zmax, 'RA (deg)', ha='center', va='bottom', fontsize=11)


def _annotate_z_side(ax, z_ticks, half_w, zmax, idx):
    """
    Annotate the side of the wedge with z tick labels.

    Args:
        ax (plt.Axes): The matplotlib axes to annotate.
        z_ticks (np.ndarray): The z tick positions.
        half_w (float): Half the width of the wedge.
        zmax (float): The maximum redshift.
        idx (int): The index of the current zone.
    """
    for z0 in z_ticks:
        x0r = half_w * (z0 / zmax) if zmax > 0 else 0.0
        angle = np.degrees(np.arctan2(-z0, -x0r)) if (zmax > 0 and x0r != 0) else 0.0
        offset = np.sign(x0r) * abs(half_w) * 0.11
        ax.text(x0r + offset, z0, f'{z0:.2f}', ha='left', va='center', rotation=angle + 180, fontsize=10)
    if idx == 0:
        ax.set_ylabel('z', fontsize=20, labelpad=15)
        ax.set_yticks(z_ticks)
        ax.set_yticklabels([f'{zt:.2f}' for zt in z_ticks], fontsize=10)


def plot_wedges_data(raw_df, prob_df, zones, tracer, output_dir, n_ra=15, n_z=10, z_lim=0.2):
    """
    Plot tracer wedges by zones.

    Args:
        raw_df (pd.DataFrame): The raw data DataFrame.
        prob_df (pd.DataFrame): The probability data DataFrame.
        zones (list): The list of zones to plot.
        tracer (str): The tracer to plot.
        output_dir (str): The output directory for the plots.
        n_ra (int): The number of RA bins.
        n_z (int): The number of z bins.
        z_lim (float): The redshift limit.
    """
    tracer_name = tracer.replace('_ANY', '').lower()
    out_dir = os.path.join(output_dir, f'tracer_zone/{tracer_name}')
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

        df_p = df_p.copy()
        df_p['CLASS'] = assign_most_likely_class(df_p)

        df = df_z.merge(df_p, on='TARGETID', how='left')
        m_tracer = _tracer_mask(df, tracer)
        if not m_tracer.any():
            sample_counts = df.loc[df['ISDATA'], 'BASE'].astype(str).value_counts().head(10)
        real = df[(df['ISDATA']) & m_tracer].copy()

        _init_ax(ax, f'Zone {zone}')

        if real.empty:
            n_zone = len(df_z)
            n_prob = len(df_p)
            n_isdata = int(df['ISDATA'].sum())
            n_tr_base = int(m_tracer.sum())
            ax.set_xlim(0, 1); ax.set_ylim(0, 1)
            continue

        ra_min, ra_max, ra_ctr, dec_ctr, Dc, half_w, zmax = _compute_zone_params(real, z_lim)
        cos_dec = np.cos(np.deg2rad(dec_ctr))
        Dc_all = Planck18.comoving_distance(real['Z']).value if real['Z'].max() > 0 else np.zeros(len(real))

        z_ticks, ra_ticks = _draw_grid(ax, ra_min, ra_max, ra_ctr, cos_dec, Dc, half_w, zmax, n_ra, n_z)
        _plot_classes(ax, real, ra_ctr, cos_dec, Dc_all, zmax, half_w)
        _draw_borders(ax, half_w, zmax)
        _annotate_ra_top(ax, ra_ticks, ra_ctr, cos_dec, Dc, zmax)
        _annotate_z_side(ax, z_ticks, half_w, zmax, idx)

    handles = [Line2D([], [], marker='o', color=c, linestyle='', markersize=6, label=k)
               for k, c in CLASS_COLORS.items()]
    fig.legend(handles, CLASS_COLORS.keys(), bbox_to_anchor=(0.5, 0.965), loc='upper center',
               ncol=len(CLASS_COLORS))
    plt.suptitle(tracer.replace('_ANY', ''), fontsize=18)

    fname = f'{tracer_name}_zones_{"_".join(f"{z:02d}" for z in zones)}.png'
    fig.savefig(os.path.join(out_dir, fname), dpi=360, bbox_inches='tight')
    plt.close(fig)


def plot_wedges_rand(raw_df, labels_df, zones, tracer, output_dir, n_ra=15, n_z=10, z_lim=0.2, iter_j=0):
    """
    Plot tracer wedges by zones for a random iteration.
    
    Args:
        raw_df (pd.DataFrame): The raw data DataFrame.
        labels_df (pd.DataFrame): The labels data DataFrame.
        zones (list): The list of zones to plot.
        tracer (str): The tracer to plot.
        output_dir (str): The output directory for the plots.
        n_ra (int): The number of RA bins.
        n_z (int): The number of z bins.
        z_lim (float): The redshift limit.
        iter_j (int): The random iteration index.
    """
    tracer_name = (tracer.replace('_ANY', '') + f'_iter{iter_j}').lower()
    out_dir = os.path.join(output_dir, f'tracer_zone/{tracer_name}')
    os.makedirs(out_dir, exist_ok=True)

    num = len(zones)
    ncols = min(num, 4)
    nrows = (num + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4*ncols, 15*nrows), sharex=False,
                             sharey=False, gridspec_kw={'wspace': 0.4, 'hspace': 0.3})
    axes = np.array(axes).reshape(-1)

    for idx, zone in enumerate(zones):
        ax = axes[idx]
        df_z = raw_df[(raw_df['ZONE'] == zone) & (raw_df['RANDITER'] == int(iter_j))]
        m_tracer = _tracer_mask(df_z, tracer)
        if not m_tracer.any():
            sample_counts = df_z['BASE'].astype(str).value_counts().head(10)
            print(f'Zone {zone} tracer \'{tracer}\' not found in random iter {iter_j}. BASE:\n{sample_counts}')
        df_z = df_z[m_tracer]

        df_lbl = labels_df[labels_df['TARGETID'].isin(df_z['TARGETID'])]
        df = df_z.merge(df_lbl, on='TARGETID', how='left')
        real = df.copy()

        _init_ax(ax, f'Zone {zone} (iter {iter_j})')

        if real.empty:
            n_zone = len(df_z)
            print(f'Zone {zone} - random empty after tracer filter. zone_rows={n_zone}')
            ax.set_xlim(0, 1); ax.set_ylim(0, 1)
            continue

        ra_min, ra_max, ra_ctr, dec_ctr, Dc, half_w, zmax = _compute_zone_params(real, z_lim)
        cos_dec = np.cos(np.deg2rad(dec_ctr))
        Dc_all = Planck18.comoving_distance(real['Z']).value if real['Z'].max() > 0 else np.zeros(len(real))

        z_ticks, ra_ticks = _draw_grid(ax, ra_min, ra_max, ra_ctr, cos_dec, Dc, half_w, zmax, n_ra, n_z)
        _plot_classes(ax, real, ra_ctr, cos_dec, Dc_all, zmax, half_w)
        _draw_borders(ax, half_w, zmax)
        _annotate_ra_top(ax, ra_ticks, ra_ctr, cos_dec, Dc, zmax)
        _annotate_z_side(ax, z_ticks, half_w, zmax, idx)

    handles = [Line2D([], [], marker='o', color=c, linestyle='', markersize=6, label=k)
               for k, c in CLASS_COLORS.items()]
    fig.legend(handles, CLASS_COLORS.keys(), bbox_to_anchor=(0.5, 0.965), loc='upper center', ncol=len(CLASS_COLORS))
    plt.suptitle(tracer.replace('_ANY', '') + f' (iter {iter_j})', fontsize=18)

    fname = f'{tracer_name}_zones_{{}}.png'.format("_".join(f"{z:02d}" for z in zones))
    fig.savefig(os.path.join(out_dir, fname), dpi=360, bbox_inches='tight')
    plt.close(fig)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--raw-dir', default='/pscratch/sd/v/vtorresg/cosmic-web/edr/raw')
    p.add_argument('--class-dir', default='/pscratch/sd/v/vtorresg/cosmic-web/edr/class')
    p.add_argument('--zones', nargs='+', type=int, default=None)
    p.add_argument('--output', default='/pscratch/sd/v/vtorresg/cosmic-web/edr/figs')
    p.add_argument('--bins', type=int, default=10)
    p.add_argument('--tracers', nargs='+', default=['BGS_ANY','LRG','ELG','QSO'])
    p.add_argument('--zlim', type=float, default=4.0)
    p.add_argument('--catalog', choices=['real','random'], default='real')
    p.add_argument('--randiter', type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()
    zones = infer_zones(args.raw_dir, args.zones)
    make_output_dirs(args.output)

    raw_cache = {}
    if args.catalog == 'real':
        prob_cache = {}
        for zone in zones:
            raw_p, _ = get_zone_paths(args.raw_dir, args.class_dir, zone)
            prob_p = get_prob_path(args.class_dir, zone)

            df_raw = load_raw_df(raw_p); df_raw['ZONE'] = zone
            df_prob = load_prob_df(prob_p); df_prob['ZONE'] = zone

            raw_cache[zone] = df_raw
            prob_cache[zone] = df_prob

        raw_all = pd.concat(raw_cache.values(), ignore_index=True)
        prob_all = pd.concat(prob_cache.values(), ignore_index=True)

        data_dir = os.path.join(args.output, 'data', 'full'); os.makedirs(data_dir, exist_ok=True)

        for tracer in args.tracers:
            plot_wedges_data(raw_all, prob_all, zones, tracer, args.output,
                                        n_ra=args.bins, n_z=args.bins, z_lim=args.zlim)
    else: 
        labels_cache = {}
        for zone in zones:
            raw_p, cls_p = get_zone_paths(args.raw_dir, args.class_dir, zone)
            df_raw = load_raw_df(raw_p); df_raw['ZONE'] = zone
            tc = Table.read(cls_p, memmap=True)
            df_labels = _labels_for_iteration(tc, use_isdata=False, iter_j=args.randiter)
            df_labels['ZONE'] = zone
            raw_cache[zone] = df_raw
            labels_cache[zone] = df_labels

        raw_all = pd.concat(raw_cache.values(), ignore_index=True)
        labels_all = pd.concat(labels_cache.values(), ignore_index=True)

        data_dir = os.path.join(args.output, 'data', 'full'); os.makedirs(data_dir, exist_ok=True)

        for tracer in args.tracers:
            plot_wedges_rand(raw_all, labels_all, zones, tracer, args.output,
                                               n_ra=args.bins, n_z=args.bins, z_lim=args.zlim, iter_j=args.randiter)


if __name__ == "__main__":
    main()