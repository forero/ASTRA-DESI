import os, argparse, re
import numpy as np, pandas as pd
import matplotlib.pyplot as plt

from astropy.table import Table
from astropy.cosmology import Planck18
from matplotlib.lines import Line2D
import matplotlib
matplotlib.rcParams['text.usetex']# is set from CLI (--usetex)
PLOT_DPI = 360
# plt.rcParams.update({'axes.labelsize': 12, 'xtick.labelsize': 10,
#                      'ytick.labelsize': 10, 'legend.fontsize': 10,})

CLASS_COLORS = {'void': 'red', 'sheet': '#9ecae1', 'filament': '#3182bd', 'knot': 'navy'}
CLASS_ZORDER = {'void':0, 'sheet':1, 'filament':2, 'knot':3}


def _zone_tag(zone):
    """
    Convert zone to tag used in filenames. Accepts int or string labels.

    Args:
        zone (int | str): Zone number (EDR) or label like 'NGC1' (DR1).
    Returns:
        str: Zone tag for filenames.
    Raises:
        ValueError: If zone is not a valid int or str.
    """
    if isinstance(zone, int):
        return f"{zone:02d}"
    elif isinstance(zone, str):
        return zone
    else:
        raise ValueError(f"Invalid zone type: {type(zone)}")


def _safe_tag(tag):
    if tag is None:
        return ''
    safe = re.sub(r"[^A-Za-z0-9_\-\.\+]+", "_", str(tag))
    return f'_{safe}' if safe else ''

def get_zone_paths(raw_dir, class_dir, zone, out_tag=None):
    """
    Get file paths for a given zone number or label.

    Args:
        raw_dir (str): Directory containing raw data files.
        class_dir (str): Directory containing classification files.
        zone (int | str): Zone number (EDR) or label like 'NGC1' (DR1).
    Returns:
        tuple: Paths to the raw data file and classification file for the zone.
    """
    ztag = _zone_tag(zone)
    tsuf = _safe_tag(out_tag)
    return (os.path.join(raw_dir, f'zone_{ztag}{tsuf}.fits.gz'),
            os.path.join(class_dir, f'zone_{ztag}{tsuf}_class.fits.gz'),)


def get_prob_path(raw_dir, class_dir, zone, out_tag=None):
    """
    Get the probability file path for a given zone number or label.

    Args:
        raw_dir (str): Directory containing raw data files.
        class_dir (str): Directory containing classification files.
        zone (int | str): Zone number (EDR) or label like 'NGC1' (DR1).
    Returns:
        str: Path to the probability file for the zone.
    """
    ztag = _zone_tag(zone)
    tsuf = _safe_tag(out_tag)
    return os.path.join(class_dir, f'zone_{ztag}{tsuf}_probability.fits.gz')


def list_zone_out_tags(raw_dir, class_dir, zone):
    """
    Discover available out-tags for a zone by scanning the class directory.
    Falls back to [''] (no tag) if only untagged files are present.

    Args:
        raw_dir (str): Raw directory path (used to validate raw files exist).
        class_dir (str): Class directory path to scan.
        zone (int|str): Zone identifier (e.g., 00 or 'NGC1').
    Returns:
        list[str]: Sorted list of out-tags ('' means untagged/legacy single-file).
    """
    ztag = _zone_tag(zone)
    tags = set()
    pat = re.compile(rf'^zone_{re.escape(ztag)}(?:_(?P<tag>.+?))?_class\.fits(?:\.gz)?$')
    try:
        for f in os.listdir(class_dir):
            m = pat.match(f)
            if m:
                tag = m.group('tag') or ''
                raw_path = os.path.join(raw_dir, f'zone_{ztag}{("_"+tag) if tag else ""}.fits.gz')
                if os.path.exists(raw_path):
                    tags.add(tag)
    except FileNotFoundError:
        pass
    if not tags:
        if os.path.exists(os.path.join(class_dir, f'zone_{ztag}_class.fits.gz')) and \
           os.path.exists(os.path.join(raw_dir, f'zone_{ztag}.fits.gz')):
            return ['']
    return sorted(tags)


def infer_zones(raw_dir, provided):
    """
    Infer available zones from raw directory if not provided.

    Supports both numeric zones (e.g., 0, 01, 12) and string labels
    (e.g., 'NGC1'). Returns a list of zone identifiers as strings
    or ints depending on input in `provided`. When inferred from
    filenames, returns the raw tag from filenames without casting.

    Args:
        raw_dir (str): Directory containing raw data files.
        provided (list or None): List of zone identifiers if provided.
    Returns:
        list: List of zone identifiers (strings like 'NGC1' or numeric-like strings).
    Raises:
        ValueError: If provided is not a list or None.
    """
    if provided:
        return list(dict.fromkeys(provided))

    pat = re.compile(r"^zone_([^_.]+)(?:_.+)?\.fits(?:\.gz)?$")
    tags = set()
    for f in os.listdir(raw_dir):
        m = pat.match(f)
        if m:
            tags.add(m.group(1))

    def _key(t):
        try:
            return (0, int(t))
        except Exception:
            return (1, str(t))

    return sorted(tags, key=_key)


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
    cols = ['TARGETID','TRACERTYPE','RANDITER','XCART','YCART','ZCART','RA','DEC','Z']
    try:
        tbl = Table.read(path, hdu=1, include_names=cols, memmap=True)
    except Exception:
        tbl = Table.read(path, memmap=True)
        tbl = tbl[[c for c in cols if c in tbl.colnames]]
    df = tbl.to_pandas()

    if df['TRACERTYPE'].dtype == object:
        df['TRACERTYPE'] = df['TRACERTYPE'].astype(str)
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
    try:
        tbl = Table.read(path, hdu=1, include_names=['TARGETID','NDATA','NRAND','ISDATA'], memmap=True)
    except Exception:
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
    try:
        tbl = Table.read(path, hdu=1, include_names=['TARGETID','PVOID','PSHEET','PFILAMENT','PKNOT'], memmap=True)
    except Exception:
        tbl = Table.read(path, memmap=True)
    df = tbl.to_pandas()
    prob_cols = [c for c in ['PVOID','PSHEET','PFILAMENT','PKNOT'] if c in df]
    df['CLASS'] = df[prob_cols].idxmax(axis=1).str[1:].str.lower()
    return df[['TARGETID','CLASS','PVOID','PSHEET','PFILAMENT','PKNOT']]


def _concat_existing(paths, loader):
    dfs = []
    for p in paths:
        if os.path.exists(p):
            dfs.append(loader(p))
    if not dfs:
        raise FileNotFoundError(f'None of the expected files exist: {paths}')
    if len(dfs) == 1:
        return dfs[0]
    return pd.concat(dfs, ignore_index=True, copy=False)


def compute_r(df):
    """
    Compute the r for each entry in the DataFrame.

    Args:
        df (pd.DataFrame): DataFrame containing 'NDATA' and 'NRAND' columns.
    Returns:
        pd.DataFrame: DataFrame with an additional 'r' column.
    """
    df = df.copy()
    denom = (df['NDATA'] + df['NRAND']).to_numpy()
    num = (df['NDATA'] - df['NRAND']).to_numpy()
    r = np.full_like(num, np.nan, dtype=float)
    np.divide(num, denom, out=r, where=(np.isfinite(denom) & (denom > 0)))
    df['r'] = r
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
    ax.set(xlabel='$z$', ylabel='Count', title=f'Zone {zone}')
    ax.legend()

    path = os.path.join(out_dir, f'z_hist_zone_{zone}.png')
    fig.savefig(path, dpi=PLOT_DPI); plt.close(fig)


def plot_radial_distribution(raw_df, zone, tracers, out_dir, bins):
    """
    Plot radial distribution histograms for specified tracers.

    Args:
        raw_df (pd.DataFrame): DataFrame containing raw data with 'XCART', 
                    'YCART', 'ZCART', 'TRACERTYPE', and 'RANDITER' columns.
        zone (int): Zone number for title and filename.
        tracers (list): List of tracer types to plot.
        out_dir (str): Output directory to save the plots.
        bins (int): Number of bins for the histograms.
    """
    fig, axes = plt.subplots(1, len(tracers), figsize=(4*len(tracers),4))
    if len(tracers) == 1:
        axes = [axes]

    for ax, tracer in zip(axes, tracers):
        tr_prefix = 'BGS' if tracer == 'BGS_BRIGHT' else tracer
        sub = raw_df[raw_df['TRACERTYPE'].str.startswith(tr_prefix)]
        real = sub[sub['RANDITER'] == -1]
        rand_pool = sub[sub['RANDITER'] != -1]
        if len(rand_pool) == 0 or len(real) == 0:
            ax.set(title=tr_prefix, xlabel=r'$r$', ylabel='Count')
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            ax.grid(linewidth=0.7)
            continue
        nsamp = min(len(rand_pool), len(real))
        rand = rand_pool.sample(nsamp, random_state=0, replace=False)

        r_real = np.linalg.norm(real[['XCART','YCART','ZCART']].to_numpy(), axis=1)
        r_rand = np.linalg.norm(rand[['XCART','YCART','ZCART']].to_numpy(), axis=1)

        ax.grid(linewidth=0.7)
        ax.hist([r_real, r_rand], bins, label=['Real','Random'], alpha=0.7)
        ax.set(title=tr_prefix, xlabel=r'$r$', ylabel='Count')
        ax.legend(fontsize=6)

    fig.suptitle(f'Radial Zone {zone}')
    fig.tight_layout()
    path = os.path.join(out_dir, f'radial_zone_{zone}.png')
    fig.savefig(path, dpi=PLOT_DPI)
    plt.close(fig)


def plot_cdf(df_r, zone, tracers, out_dir):
    """
    Plot CDF of r values for specified tracers using NumPy ECDF.
    
    Args:
        df_r (pd.DataFrame): DataFrame containing 'r' values and 'TRACERTYPE' columns.
        zone (int): Zone number for title and filename.
        tracers (list): List of tracer types to plot.
        out_dir (str): Output directory to save the plot.
    """
    cmap = {'BGS_BRIGHT':'blue','ELG':'red','LRG':'green','QSO':'purple'}
    fig, ax = plt.subplots()
    ax.grid(linewidth=0.7)

    def ecdf(arr):
        a = np.asarray(arr, dtype=float)
        a = a[~np.isnan(a)]
        if a.size == 0:
            return None, None
        a = np.sort(a)
        y = np.arange(1, a.size + 1) / a.size
        return a, y

    for tr in tracers:
        tr_key = str(tr).upper()
        tr_prefix = 'BGS' if tr_key.startswith('BGS') else tr_key
        sub = df_r[df_r['TRACERTYPE'].str.startswith(tr_prefix)]
        xr, yr = ecdf(sub[sub['ISDATA']]['r'])
        if xr is not None:
            ax.plot(xr, yr, label=f'{tr_prefix} real', linewidth=1, color=cmap.get(tr_key, 'black'))
        xr, yr = ecdf(sub[~sub['ISDATA']]['r'])
        if xr is not None:
            ax.plot(xr, yr, label=f'{tr_prefix} rand', linewidth=1, linestyle='--', color=cmap.get(tr_key, 'black'))

    ax.set(xlabel='$r$', ylabel='CDF', title=f'CDF Zone {zone}')
    ax.legend(fontsize=6)
    path = os.path.join(out_dir, f'cdf_zone_{zone}.png')
    fig.savefig(path, dpi=PLOT_DPI)
    plt.close(fig)


def plot_cdf_dispersion(raw_dir, class_dir, zones, out_dir, tracers=None, xbins=400, subsample_per_zone=None, progress=False, out_tags=None):
    """
    Plot the dispersion (percentile band) of CDFs over multiple zones in one figure.
    
    Args:
        raw_dir (str): Directory with raw zone files.
        class_dir (str): Directory with class zone files.
        zones (list[int]): Zone numbers to include (e.g., range(20)).
        out_dir (str): Base output directory (the figure is written under `<out_dir>/cdf`).
        tracers (list[str] or None): Tracers to include. Default: ['BGS_BRIGHT','ELG','LRG','QSO'].
        xbins (int): Number of points in the common x-grid for interpolation.
        subsample_per_zone (int or None): Max samples per (zone,tracer,real/rand) for ECDF calculation.
        progress (bool): Print progress logs.
    """
    if tracers is None:
        tracers = ['BGS_BRIGHT','ELG','LRG','QSO']

    def _ecdf_interp(arr, xgrid):
        a = np.asarray(arr, dtype=float)
        a = a[~np.isnan(a)]
        if a.size == 0:
            return np.full_like(xgrid, np.nan, dtype=float)
        a.sort()
        y = np.arange(1, a.size + 1) / a.size
        return np.interp(xgrid, a, y, left=0.0, right=1.0)

    xgrid = np.linspace(-1.0, 1.0, xbins)

    if subsample_per_zone is not None and subsample_per_zone <= 0:
        subsample_per_zone = None

    per_tracer_real = {t: [] for t in tracers}
    per_tracer_rand = {t: [] for t in tracers}

    for i, z in enumerate(zones):
        tags = out_tags if out_tags else list_zone_out_tags(raw_dir, class_dir, z)
        if progress:
            print(f'[cdf-disp] zone {z}: tags={tags if tags else "(legacy single-file)"}')
        ztag = _zone_tag(z)
        tag_list = tags if tags else ['']
        raw_paths = [os.path.join(raw_dir,  f'zone_{ztag}{("_"+t) if t else ""}.fits.gz') for t in tag_list]
        cls_paths = [os.path.join(class_dir, f'zone_{ztag}{("_"+t) if t else ""}_class.fits.gz') for t in tag_list]

        raw_df = _concat_existing(raw_paths, load_raw_df)
        cls_df = _concat_existing(cls_paths, load_class_df)
        merged = raw_df.merge(cls_df[['TARGETID','NDATA','NRAND']], on='TARGETID', how='left')
        r_df = compute_r(merged)

        for tr in tracers:
            tr_prefix = 'BGS' if tr == 'BGS_BRIGHT' else tr
            sub = r_df[r_df['TRACERTYPE'].str.startswith(tr_prefix)]
            real_r = sub[sub['ISDATA']]['r'].to_numpy()
            rand_r = sub[~sub['ISDATA']]['r'].to_numpy()

            if subsample_per_zone is not None:
                if real_r.size > subsample_per_zone:
                    idx = np.random.default_rng(0).choice(real_r.size, subsample_per_zone, replace=False)
                    real_r = real_r[idx]
                if rand_r.size > subsample_per_zone:
                    idx = np.random.default_rng(0).choice(rand_r.size, subsample_per_zone, replace=False)
                    rand_r = rand_r[idx]
            y_real = _ecdf_interp(real_r, xgrid)
            y_rand = _ecdf_interp(rand_r, xgrid)
            if np.isfinite(y_real).any():
                per_tracer_real[tr].append(y_real)
            if np.isfinite(y_rand).any():
                per_tracer_rand[tr].append(y_rand)

    fig, ax = plt.subplots(figsize=(9, 7))
    ax.grid(linewidth=0.7)

    for tr in tracers:
        color = CLASS_COLORS.get('sheet', 'C0')
        tracer_color_map = {'BGS_BRIGHT':'blue','ELG':'red','LRG':'green','QSO':'purple'}
        color = tracer_color_map.get(tr, 'black')

        if len(per_tracer_real[tr]) > 0:
            Y = np.vstack(per_tracer_real[tr])
            p16, p50, p84 = np.nanpercentile(Y, [16,50,84], axis=0)
            ax.fill_between(xgrid, p16, p84, alpha=0.15, label=f'{tr} real ±1σ', color=color)
            ax.plot(xgrid, p50, linewidth=1.5, color=color, label=f'{tr} real median')

        if len(per_tracer_rand[tr]) > 0:
            Y = np.vstack(per_tracer_rand[tr])
            p16, p50, p84 = np.nanpercentile(Y, [16,50,84], axis=0)
            ax.fill_between(xgrid, p16, p84, alpha=0.10, label=f'{tr} rand ±1σ', color=color)
            ax.plot(xgrid, p50, linewidth=1.2, linestyle='--', color=color, label=f'{tr} rand median')

    ax.set_ylabel('CDF')
    ax.set_xlabel(r"$r = \frac{N_{\mathrm{data}} - N_{\mathrm{rand}}}{N_{\mathrm{data}} + N_{\mathrm{rand}}}$")
    ax.set_title('Dispersion of CDF across zones', fontsize=16)
    ax.legend(fontsize=9, ncol=2)

    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, 'cdf/cdf_dispersion_zones.png')
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=PLOT_DPI)
    plt.close(fig)


def _prepare_real(raw_df, prob_df):
    """
    Merge raw and prob DataFrames, filter to real data only.

    Args:
        raw_df (pd.DataFrame): Raw data DataFrame.
        prob_df (pd.DataFrame): Probability data DataFrame.
    Returns:
        pd.DataFrame: Merged DataFrame with only real data.
    """
    df = raw_df.merge(prob_df, on='TARGETID', how='left')
    return df[df['ISDATA']]


def _compute_bounds(sub, z_lo=0.7, z_hi=None, offset=0, use_intersection=False):
    """
    Compute RA/DEC center, z limits, and half-width for wedge plotting.

    Args:
        sub (pd.DataFrame): Subset DataFrame for a specific tracer.
        z_lo (float or None): Lower z limit. If None, computed from data.
        z_hi (float or None): Upper z limit. If None, computed from data.
        offset (float): Symmetric padding applied when inferring limits from data.
        use_intersection (bool): If True, use the overlapping z-range across BASE groups
            before applying the offset (used by wedge slices).
    Returns:
        tuple: (ra_ctr, dec_ctr, z_lo, z_hi, half_w)
    """
    if sub.empty:
        raise ValueError('Cannot compute bounds for empty subset.')

    z_vals = sub['Z'].to_numpy(dtype=float, copy=False)
    z_vals = z_vals[np.isfinite(z_vals)]
    if z_vals.size == 0:
        raise ValueError('No finite redshift values available to compute bounds.')

    data_min = float(z_vals.min())
    data_max = float(z_vals.max())

    if use_intersection:
        grouped = sub.groupby('BASE')['Z']
        gmins = grouped.min().to_numpy(dtype=float, copy=False)
        gmaxs = grouped.max().to_numpy(dtype=float, copy=False)
        lower = np.nanmax(gmins) if gmins.size else data_min
        upper = np.nanmin(gmaxs) if gmaxs.size else data_max
        z_lo_eff = lower - offset
        z_hi_eff = upper + offset
    else:
        if z_lo is None:
            z_lo_eff = data_min
        else:
            z_lo_eff = float(z_lo)
        if z_hi is None:
            z_hi_eff = data_max * 1.02
        else:
            z_hi_eff = float(z_hi)

        if offset:
            z_lo_eff = z_lo_eff - offset
            z_hi_eff = z_hi_eff + offset

    if not np.isfinite(z_lo_eff):
        z_lo_eff = data_min
    if not np.isfinite(z_hi_eff):
        z_hi_eff = data_max

    z_lo_eff = max(0.0, z_lo_eff)

    if z_hi_eff <= z_lo_eff:
        z_hi_eff = z_lo_eff + max(1e-3, 0.02 * max(1.0, abs(z_lo_eff)))

    ra_min = sub['RA'].min(skipna=True)
    ra_max = sub['RA'].max(skipna=True)
    ra_ctr = 0.5 * (ra_min + ra_max)
    dec_min = sub['DEC'].min(skipna=True)
    dec_max = sub['DEC'].max(skipna=True)
    dec_ctr = 0.5 * (dec_min + dec_max)

    Dc = Planck18.comoving_distance(z_hi_eff).value
    half_w = Dc * np.deg2rad(ra_max - ra_ctr) * np.cos(np.deg2rad(dec_ctr))

    return ra_ctr, dec_ctr, z_lo_eff, z_hi_eff, half_w


def _draw_grid(ax, ra_ctr, dec_ctr, z_lo, z_hi, half_w, n_ra, n_z):
    """
    Draw grid lines for constant z and RA on the wedge plot.

    Args:
        ax (matplotlib.axes.Axes): Axes to draw on.
        ra_ctr (float): Central RA in degrees.
        dec_ctr (float): Central DEC in degrees.
        z_lo (float): Lower z limit.
        z_hi (float): Upper z limit.
        half_w (float): Half-width of the wedge at z_hi.
        n_ra (int): Number of RA ticks.
        n_z (int): Number of z ticks.
    Returns:
        tuple: (z_ticks, ra_ticks)
    """
    zs = np.linspace(z_lo, z_hi, 300)
    z_ticks = np.linspace(z_lo, z_hi, n_z) if n_z > 0 else np.asarray([])
    Dc_hi = Planck18.comoving_distance(z_hi).value
    cos_dec = np.cos(np.deg2rad(dec_ctr))
    denom = Dc_hi * cos_dec
    if n_ra > 0 and np.isfinite(denom) and denom != 0:
        half_w_ang = half_w / denom * (180/np.pi)
        ra_ticks = np.linspace(ra_ctr - half_w_ang, ra_ctr + half_w_ang, n_ra)
    else:
        ra_ticks = np.asarray([])

    span = z_hi - z_lo
    if not np.isfinite(span) or span == 0:
        span = np.finfo(float).eps

    for z0 in z_ticks:
        w0 = half_w * ((z0 - z_lo) / span)
        ax.hlines(z0, -w0, w0, color='gray', lw=0.5, alpha=0.5)

    if ra_ticks.size:
        Dc_zs = Planck18.comoving_distance(zs).value
        base = Dc_zs * cos_dec
        for rt in ra_ticks[::4]:
            delta = np.deg2rad(rt - ra_ctr)
            ax.plot(base * delta, zs, color='gray', lw=0.5, alpha=0.5)

    ax.set_xlim(-half_w, half_w)
    ax.set_ylim(z_lo, z_hi)
    return z_ticks, ra_ticks


def _plot_classes(ax, sub, ra_ctr, dec_ctr, z_lo, z_hi, half_w):
    """
    Plot points colored by CLASS on the wedge plot.

    Args:
        ax (matplotlib.axes.Axes): Axes to draw on.
        sub (pd.DataFrame): Subset DataFrame for a specific tracer.
        ra_ctr (float): Central RA in degrees.
        dec_ctr (float): Central DEC in degrees.
        z_lo (float): Lower z limit.
        z_hi (float): Upper z limit.
        half_w (float): Half-width of the wedge at z_hi.
    """
    if sub.empty:
        return

    cos_dec = np.cos(np.deg2rad(dec_ctr))
    z = sub['Z'].to_numpy(dtype=float, copy=False)
    ra = sub['RA'].to_numpy(dtype=float, copy=False)
    classes = sub['CLASS'].to_numpy(copy=False)
    Dc_vals = Planck18.comoving_distance(z).value
    x_all = Dc_vals * np.deg2rad(ra - ra_ctr) * cos_dec
    y_all = z

    span = z_hi - z_lo
    if not np.isfinite(span) or span == 0:
        span = np.finfo(float).eps

    base_mask = np.isfinite(x_all) & np.isfinite(y_all)
    base_mask &= np.abs(x_all) <= half_w * ((y_all - z_lo) / span)

    for cls, color in CLASS_COLORS.items():
        cls_mask = base_mask & (classes == cls)
        if not np.any(cls_mask):
            continue
        size = 5 if cls in ('void', 'knot') else 1
        ax.scatter(x_all[cls_mask], y_all[cls_mask], s=size, c=color,
                   zorder=CLASS_ZORDER[cls])


def _plot_simple(ax, sub, ra_ctr, dec_ctr, z_lo, z_hi, half_w):
    """
    Plot points on the wedge plot without class distinction.

    Args:
        ax (matplotlib.axes.Axes): Axes to draw on.
        sub (pd.DataFrame): Subset DataFrame for a specific tracer.
        ra_ctr (float): Central RA in degrees.
        dec_ctr (float): Central DEC in degrees.
        z_lo (float): Lower z limit.
        z_hi (float): Upper z limit.
        half_w (float): Half-width of the wedge at z_hi.
    """
    if sub.empty:
        return

    cos_dec = np.cos(np.deg2rad(dec_ctr))
    z = sub['Z'].to_numpy(dtype=float, copy=False)
    ra = sub['RA'].to_numpy(dtype=float, copy=False)
    Dc_vals = Planck18.comoving_distance(z).value
    x = Dc_vals * np.deg2rad(ra - ra_ctr) * cos_dec

    span = z_hi - z_lo
    if not np.isfinite(span) or span == 0:
        span = np.finfo(float).eps

    mask = np.isfinite(x) & np.isfinite(z)
    mask &= np.abs(x) <= half_w * ((z - z_lo) / span)
    ax.scatter(x[mask], z[mask], s=0.001, c='black')



def _draw_border(ax, half_w, z_lo, z_hi):
    """
    Draw border lines of the wedge plot.

    Args:
        ax (matplotlib.axes.Axes): Axes to draw on.
        half_w (float): Half-width of the wedge at z_hi.
        z_lo (float): Lower z limit.
        z_hi (float): Upper z limit.
    """
    ax.plot([-half_w, 0], [z_hi, z_lo], 'k-', lw=1.5)
    ax.plot([ half_w, 0], [z_hi, z_lo], 'k-', lw=1.5)


def _annotate_z_side(ax, z_ticks, half_w, z_lo, z_hi, side='right'):
    """
    Write z tick labels along the slanted wedge border.

    Args:
        ax (matplotlib.axes.Axes): Axes to draw on.
        z_ticks (np.ndarray): Array of z tick positions to annotate.
        half_w (float): Half-width of the wedge at z_hi.
        z_lo (float): Lower z limit.
        z_hi (float): Upper z limit.
        side (str): Which slanted side to annotate: 'right' or 'left'.
    """
    if not np.isfinite(half_w) or (z_hi - z_lo) <= 0:
        return

    angle_deg = np.degrees(np.arctan2((z_hi - z_lo), half_w))
    offset = 0.05 * abs(half_w)

    for z0 in np.asarray(z_ticks, dtype=float):
        if not np.isfinite(z0):
            continue
        s = (z0 - z_lo) / (z_hi - z_lo)
        s = np.clip(s, 0.0, 1.0)

        if side == 'right':
            x_border = half_w * s
            x_text = x_border + offset
            rot = angle_deg
            ha = 'left'
        else:
            x_border = -half_w * s
            x_text = x_border - offset
            rot = -angle_deg
            ha = 'right'
        ax.text(x_text, z0, f'{z0:.2f}', rotation=rot, ha=ha, va='center', fontsize=15)


def _configure_axes(ax, side='right'):
    """
    Configure the axes for the wedge plot.

    Args:
        ax (matplotlib.axes.Axes): Axes to configure.
        side (str): (kept for API compatibility; no effect on tick placement)
    """
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_frame_on(False)
    ax.set_yticks([])
    ax.tick_params(left=False, right=False, labelleft=False, labelright=False)
    ax.set_xticks([])


def plot_wedges(raw_df, prob_df, zone, output_dir, n_ra=15, n_z=10, grouped=False):
    """
    Plot and save wedges for all tracers in a given zone.

    Args:
        raw_df (pd.DataFrame): Raw data DataFrame.
        prob_df (pd.DataFrame): Probability data DataFrame.
        zone (int): Zone number for title and filename.
        output_dir (str): Output directory to save the plots.
        n_ra (int): Number of RA ticks.
        n_z (int): Number of z ticks.
    """
    real = _prepare_real(raw_df, prob_df)
    tracers = ['BGS_BRIGHT', 'LRG', 'ELG', 'QSO']
    fig, axes = plt.subplots(1, len(tracers), figsize=(4*len(tracers), 20), gridspec_kw={'wspace': 0.5})
    axes = axes.flatten()

    for ax, tracer in zip(axes, tracers):
        tr_prefix = 'BGS' if tracer == 'BGS_BRIGHT' else tracer
        sub = real[real['BASE'].str.startswith(tr_prefix)]
        _configure_axes(ax, side='right')
        ax.set_title(tracer.replace('_BRIGHT',''), fontsize=16, y=1.05)
        if sub.empty:
            print(f'No data for tracer {tracer} in zone {zone}, skipping plot.')
            continue

        ra_ctr, dec_ctr, z_lo, z_hi, half_w = _compute_bounds(sub)
        z_ticks, ra_ticks = _draw_grid(ax, ra_ctr, dec_ctr, z_lo, z_hi, half_w, n_ra, n_z)
        if grouped:
            _plot_classes(ax, sub, ra_ctr, dec_ctr, z_lo, z_hi, half_w)
        else:
            _plot_simple(ax, sub, ra_ctr, dec_ctr, z_lo, z_hi, half_w)
        _draw_border(ax, half_w, z_lo, z_hi)
        _annotate_z_side(ax, z_ticks, half_w, z_lo, z_hi, side='right')

        Dc_hi = Planck18.comoving_distance(z_hi).value
        cos_dec = np.cos(np.deg2rad(dec_ctr))
        denom = Dc_hi * cos_dec
        sec = ax.secondary_xaxis('top',
            functions=(lambda x, _den=denom: ra_ctr + np.rad2deg(x / _den),
                       lambda r, _den=denom: _den * np.deg2rad(r - ra_ctr)))
        sec.set_xticks(ra_ticks[::4])
        sec.set_xticklabels([f'{rt:.0f}' for rt in ra_ticks[::4]], fontsize=13)
        sec.set_xlabel('$RA$ [deg]', fontsize=14, labelpad=13)

        if ax is axes[0]:
            ax.set_ylabel('$z$', fontsize=20, labelpad=15)
    if grouped:
        handles = [Line2D([], [], marker='o', color=c, linestyle='', markersize=8, label=k)
                for k,c in CLASS_COLORS.items()]
        fig.legend(handles, CLASS_COLORS.keys(), bbox_to_anchor=(0.5,0.965), loc='upper center',
               ncol=len(CLASS_COLORS), fontsize=14)

    plt.suptitle(f'Zone {_zone_tag(zone)}', fontsize=18)
    os.makedirs(os.path.join(output_dir, 'wedges/complete'), exist_ok=True)
    fig.savefig(os.path.join(output_dir, f'wedges/complete/wedge_zone_{_zone_tag(zone)}.png'), dpi=PLOT_DPI)
    plt.close(fig)


def plot_wedges_slice(raw_df, prob_df, zone, output_dir, n_ra=15, n_z=10, offset=0.3, grouped=False):
    """
    Plot and save wedges for all tracers in a given zone, slicing z by BASE type.

    Args:
        raw_df (pd.DataFrame): Raw data DataFrame.
        prob_df (pd.DataFrame): Probability data DataFrame.
        zone (int): Zone number for title and filename.
        output_dir (str): Output directory to save the plots.
        n_ra (int): Number of RA ticks.
        n_z (int): Number of z ticks.
        offset (float): Offset to apply when slicing by BASE.
    """
    real = _prepare_real(raw_df, prob_df)
    tracers = ['BGS_BRIGHT', 'LRG', 'ELG', 'QSO']
    fig, axes = plt.subplots(1, len(tracers), figsize=(4*len(tracers), 20), sharex=True, sharey=True,
                             gridspec_kw={'wspace': 0.5})
    axes = axes.flatten()

    if real.empty:
        print(f'No real data for zone {zone}, skipping slice wedges.')
        plt.close(fig)
        return

    ra_ctr, dec_ctr, z_lo, z_hi, half_w = _compute_bounds(real, z_lo=None, z_hi=None,
                                                          offset=offset, use_intersection=True)

    for ax, tracer in zip(axes, tracers):
        tr_prefix = 'BGS' if tracer == 'BGS_BRIGHT' else tracer
        sub = real[real['BASE'].str.startswith(tr_prefix)]
        _configure_axes(ax, side='left')
        ax.set_title(tracer.replace('_ANY',''), fontsize=16, y=1.05)

        if sub.empty:
            print(f'No data for tracer {tracer} in zone {zone}, skipping plot.')
            continue

        z_ticks, ra_ticks = _draw_grid(ax, ra_ctr, dec_ctr, z_lo, z_hi, half_w, n_ra, n_z)
        if grouped:
            _plot_classes(ax, sub, ra_ctr, dec_ctr, z_lo, z_hi, half_w)
        else:
            _plot_simple(ax, sub, ra_ctr, dec_ctr, z_lo, z_hi, half_w)
        _draw_border(ax, half_w, z_lo, z_hi)
        _annotate_z_side(ax, z_ticks, half_w, z_lo, z_hi, side='left')

        Dc_hi = Planck18.comoving_distance(z_hi).value
        cos_dec = np.cos(np.deg2rad(dec_ctr))
        denom = Dc_hi * cos_dec
        sec = ax.secondary_xaxis('top',
            functions=(lambda x, _den=denom: ra_ctr + np.rad2deg(x / _den),
                       lambda r, _den=denom: _den * np.deg2rad(r - ra_ctr)))
        sec.set_xticks(ra_ticks[::4])
        sec.set_xticklabels([f'{rt:.0f}' for rt in ra_ticks[::4]], fontsize=13)
        sec.set_xlabel('$RA$ [deg]', fontsize=14, labelpad=13)

        if ax is axes[0]:
            ax.set_ylabel('$z$', fontsize=20, labelpad=15)
    if grouped:
        handles = [Line2D([], [], marker='o', color=c, linestyle='', markersize=8, label=k)
                for k,c in CLASS_COLORS.items()]
        fig.legend(handles, CLASS_COLORS.keys(), bbox_to_anchor=(0.5,0.965), loc='upper center',
                ncol=len(CLASS_COLORS), fontsize=14)

    plt.suptitle(f'Zone {_zone_tag(zone)}', fontsize=18)
    os.makedirs(os.path.join(output_dir, 'wedges/slice'), exist_ok=True)
    fig.savefig(os.path.join(output_dir, f'wedges/slice/wedge_slice_zone_{_zone_tag(zone)}.png'), dpi=PLOT_DPI)
    plt.close(fig)


def entropy(df):
    """
    Compute the Shannon entropy for each row in the DataFrame.
    
    Args:
        df (pd.DataFrame): The input DataFrame containing probability columns.
    Returns:
        pd.DataFrame: The input DataFrame with an additional column for entropy.
    """
    cols = ['PVOID', 'PSHEET', 'PFILAMENT', 'PKNOT']
    P = np.column_stack([np.asarray(df[c], dtype=float) for c in cols])
    out = df.copy()

    # H = - sum p_i log2 p_i / log2(4)
    with np.errstate(divide='ignore', invalid='ignore'):
        terms = P * np.log2(P)
    terms[~np.isfinite(terms)] = 0.0
    H = -np.sum(terms, axis=1) / np.log2(4.0)

    out['H'] = H.astype(np.float32)
    return out


def _read_raw_df_min(raw_path):
    """
    Read the raw FITS file and return a minimal DataFrame.
    
    Args:
        raw_path (str): The path to the raw FITS file.
    Returns:
        pd.DataFrame: A minimal DataFrame containing relevant columns.
    """
    tbl = Table.read(raw_path, memmap=True)
    df = tbl.to_pandas()
    df['TRACERTYPE'] = df['TRACERTYPE'].apply(lambda x: x.decode('utf-8')
                                              if isinstance(x, (bytes, bytearray)) else x)
    df['BASE'] = df['TRACERTYPE'].str.replace(r'_(DATA|RAND)$','', regex=True)
    df['ISDATA'] = df['TRACERTYPE'].str.endswith('_DATA')
    return df[['TARGETID','TRACERTYPE','BASE','ISDATA']]


def _targets_of_tracer_real(raw_df, tracer_prefix):
    """
    Get the set of TARGETIDs for a specific tracer prefix from the raw dataframe.
    
    Args:
        raw_df (pd.DataFrame): The raw dataframe containing tracer information.
        tracer_prefix (str): The tracer prefix to filter by.
    Returns:
        set: A set of TARGETIDs matching the tracer prefix.
    """
    if tracer_prefix == 'BGS_BRIGHT':
        tracer_prefix = 'BGS'
    m = raw_df['ISDATA'] & raw_df['BASE'].str.startswith(tracer_prefix)
    return set(raw_df.loc[m, 'TARGETID'].to_numpy(dtype=np.int64))


def plot_pdf_entropy(raw_dir, class_dir, zones, tracers, out_path, bins=25):
    """
    Plot the PDF of the normalized Shannon entropy H for specified tracers across zones.
    
    Args:
        raw_dir (str): Directory containing raw data files.
        class_dir (str): Directory containing class data files.
        zones (list): List of zone identifiers.
        tracers (list): List of tracer identifiers.
        out_path (str): Output path for the plot.
        bins (int): Number of bins for the histogram.
    """
    colors = plt.cm.tab20(np.linspace(0, 1, max(20, len(zones))))
    fig, axes = plt.subplots(2, 2, figsize=(10,10), sharey=True, sharex=True)
    axes = np.ravel(axes)

    for ax, tracer in zip(axes, tracers):
        ax.grid(True, alpha=0.3)
        ax.set_title(tracer.replace('_ANY', ''))

        for iz, z in enumerate(zones):
            raw_path, _ = get_zone_paths(raw_dir, class_dir, z)
            prob_path = get_prob_path(raw_dir, class_dir, z)

            raw_df = _read_raw_df_min(raw_path)
            tids_tr = _targets_of_tracer_real(raw_df, tracer.split('_', 1)[0])

            probs_df = load_prob_df(prob_path).copy()
            probs_df = entropy(probs_df)

            tids = probs_df['TARGETID'].to_numpy(dtype=np.int64, copy=False)
            m = np.isin(tids, list(tids_tr))
            v = probs_df.loc[m, 'H'].to_numpy(dtype=float, copy=False)

            hist, edges = np.histogram(v, bins=bins, range=(0, 0.6), density=True)
            centers = 0.5 * (edges[:-1] + edges[1:])
            label = f'Zone {_zone_tag(z)}' if ax is axes[0] else None
            ax.plot(centers, hist, color=colors[iz], label=label)

        if ax in (axes[0], axes[2]):
            ax.set_ylabel('PDF')
        if ax in (axes[2], axes[3]):
            ax.set_xlabel('$H$')

    handles, labels = [], []
    for ax in axes:
        h, l = ax.get_legend_handles_labels()
        for hh, ll in zip(h, l):
            if ll not in labels and ll is not None:
                handles.append(hh); labels.append(ll)

    if handles:
        fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.01),
                   bbox_transform=fig.transFigure, ncol=min(7, len(labels)), frameon=False)
    fig.subplots_adjust(bottom=0.14, top=0.85)
    plt.suptitle('Normalized Shannon Entropy', y=0.94)

    path = f'{out_path}/entropy'
    os.makedirs(path, exist_ok=True)
    fig.savefig(f'{path}/pdf_entropy.png', dpi=360)
    return fig, axes


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--raw-dir', default='/pscratch/sd/v/vtorresg/cosmic-web/dr1/raw')
    p.add_argument('--class-dir', default='/pscratch/sd/v/vtorresg/cosmic-web/dr1/class')
    p.add_argument('--output', default='/pscratch/sd/v/vtorresg/cosmic-web/dr1/figs')

    p.add_argument('--zones', nargs='+', default=None)
    p.add_argument('--bins', type=int, default=10)
    p.add_argument('--tracers', nargs='+', default=['BGS_BRIGHT','LRG','ELG','QSO'])

    p.add_argument('--plot-z', action='store_true', default=False)
    p.add_argument('--plot-radial', action='store_true', default=False)
    p.add_argument('--plot-cdf', action='store_true', default=False)
    p.add_argument('--plot-cdf-dispersion', action='store_true', default=False)
    p.add_argument('--plot-wedges', action='store_true', default=False)
    p.add_argument('--plot-wedges-slice', action='store_true', default=False)
    p.add_argument('--plot-wedges-grouped', action='store_true', default=False)
    p.add_argument('--plot-entropy-cdf', action='store_true', default=False)
    p.add_argument('--all', action='store_true', help='Enable all plots (careful: heavy)')
    p.add_argument('--usetex', action='store_true', help='Use LaTeX text rendering (heavy)')
    p.add_argument('--dpi', type=int, default=200, help='Figure DPI')
    p.add_argument('--xbins', type=int, default=200, help='Number of x grid points for CDF interpolation (default: 200)')
    p.add_argument('--subsample-per-zone', type=int, default=10000, help='Max samples per (zone,tracer,real/rand) for dispersion plot')
    p.add_argument('--progress', action='store_true', default=False, help='Print simple progress logs')
    p.add_argument('--out-tags', nargs='*', default=None, help='Limit to these out-tags (e.g., LRG ELG). If omitted, auto-discovers tags per zone.')
    return p.parse_args()


def main():
    args = parse_args()
    if args.usetex:
        matplotlib.rcParams['text.usetex'] = True
    if args.all:
        args.plot_z = True
        args.plot_radial = True
        args.plot_cdf = True
        args.plot_cdf_dispersion = True
        args.plot_wedges = True
        args.plot_wedges_slice = True
        args.plot_wedges_grouped = True
        args.plot_entropy_cdf = True
    zones = infer_zones(args.raw_dir, args.zones)
    outdirs = make_output_dirs(args.output)

    raw_cache, class_cache, prob_cache = {}, {}, {}
    for zone in zones:
        tags = args.out_tags if args.out_tags else list_zone_out_tags(args.raw_dir, args.class_dir, zone)
        if args.progress:
            print(f'[plot] zone {zone}: tags={tags if tags else "(legacy single-file)"}')

        ztag = _zone_tag(zone)
        tag_list = tags if tags else ['']
        raw_paths = [os.path.join(args.raw_dir, f'zone_{ztag}{("_"+t) if t else ""}.fits.gz') for t in tag_list]
        cls_paths = [os.path.join(args.class_dir, f'zone_{ztag}{("_"+t) if t else ""}_class.fits.gz') for t in tag_list]
        prob_paths= [os.path.join(args.class_dir, f'zone_{ztag}{("_"+t) if t else ""}_probability.fits.gz') for t in tag_list]

        raw_df = raw_cache.setdefault(zone, _concat_existing(raw_paths, load_raw_df))
        cls_df = class_cache.setdefault(zone, _concat_existing(cls_paths, load_class_df))
        cls_df = cls_df[cls_df['ISDATA'] == True]

        merged = raw_df.merge(cls_df[['TARGETID','NDATA','NRAND']], on='TARGETID', how='left')
        r_df = compute_r(merged)

        if args.plot_z:
            plot_z_histogram(merged, zone, args.bins, outdirs['z'])
        if args.plot_cdf:
            plot_cdf(r_df, zone, args.tracers, outdirs['cdf'])
        if args.plot_radial:
            plot_radial_distribution(raw_df, zone, args.tracers, outdirs['radial'], args.bins)
        if args.plot_wedges or args.plot_wedges_slice:
            prob_df = prob_cache.setdefault(zone, _concat_existing(prob_paths, load_prob_df))
            if args.plot_wedges:
                plot_wedges(raw_df, prob_df, zone, args.output, grouped=args.plot_wedges_grouped)
            if args.plot_wedges_slice:
                plot_wedges_slice(raw_df, prob_df, zone, args.output, grouped=args.plot_wedges_grouped)

    if args.plot_cdf_dispersion:
        plot_cdf_dispersion(args.raw_dir, args.class_dir, zones, args.output, args.tracers,
                            xbins=args.xbins, subsample_per_zone=args.subsample_per_zone,
                            progress=args.progress, out_tags=args.out_tags)
    if args.plot_entropy_cdf:
        plot_pdf_entropy(args.raw_dir, args.class_dir, zones, args.tracers, args.output, args.bins)


if __name__ == '__main__':
    main()