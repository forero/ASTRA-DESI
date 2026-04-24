import argparse, os, re
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.io import fits
from matplotlib.lines import Line2D


P_COLS = ['PVOID', 'PSHEET', 'PFILAMENT', 'PKNOT']
ENV_ORDER = ['Void', 'Sheet', 'Filament', 'Knot']
ENV_ORDER_COSMOS = ['Field', 'Filament', 'Cluster']
ENV_TO_COSMOS = {'Void': 'Field', 'Sheet': 'Field', 'Filament': 'Filament', 'Knot': 'Cluster'}

COSMOS_REF = {(0.1, 0.5): {'y': [0.31, 0.20, -0.10], 'yerr': [0.08, 0.08, 0.09]},
              (0.5, 0.8): {'y': [0.88, 0.86, 0.79], 'yerr': [0.09, 0.09, 0.10]},
              (0.8, 1.2): {'y': [1.14, 1.19, 1.23], 'yerr': [0.10, 0.10, 0.10]}}

SDSS_REF = {'Void': {'x': [9.12, 9.52, 9.95, 10.30, 10.64, 10.88],
                     'y': [-9.55, -9.68, -9.98, -10.28, -11.05, -11.35]},
            'Sheet': {'x': [9.12, 9.50, 9.93, 10.27, 10.63, 10.90, 11.20],
                      'y': [-9.72, -9.82, -10.05, -10.38, -11.02, -11.30, -11.70]},
            'Filament': {'x': [9.15, 9.58, 10.02, 10.36, 10.68, 10.97, 11.35, 11.62, 12.36],
                         'y': [-9.88, -10.00, -10.18, -10.65, -11.22, -11.45, -11.80, -11.78, -12.60]},
            'Knot': {'x': [9.12, 9.60, 9.92, 10.26, 10.57, 10.88, 11.22, 11.60, 12.00],
                     'y': [-10.08, -10.28, -10.55, -10.92, -11.22, -11.38, -11.75, -12.02, -12.48]}}

NEXUS_REF = {'Void': {'x': [9.00, 9.30, 9.60, 9.82, 10.02, 10.18, 10.40, 10.58, 10.80, 11.10, 11.42],
                      'y': [-9.88, -9.87, -9.85, -9.88, -9.95, -10.08, -10.40, -10.92, -11.68, -11.76, -11.88]},
             'Sheet': {'x': [9.00, 9.30, 9.60, 9.82, 10.02, 10.18, 10.40, 10.58, 10.80, 11.08, 11.42],
                       'y': [-9.98, -9.97, -9.95, -9.98, -10.06, -10.22, -10.52, -10.95, -11.73, -11.78, -11.98]},
             'Filament': {'x': [9.00, 9.30, 9.60, 9.82, 10.00, 10.18, 10.38, 10.58, 10.78, 11.18, 11.62],
                          'y': [-10.10, -10.08, -10.08, -10.12, -10.28, -10.48, -10.76, -11.10, -11.72, -12.02, -11.95]},
             'Knot': {'x': [9.00, 9.20, 9.40, 9.62, 9.84, 10.02, 10.20, 10.38, 10.60, 10.88, 11.18, 11.48, 12.02],
                      'y': [-10.45, -10.50, -10.56, -10.60, -10.68, -10.84, -10.90, -11.02, -11.25, -11.55, -11.78, -12.10, -12.25]}}


DATASET_STYLES = {'EDR': {'color': 'royalblue', 'marker_low': '*', 'marker_high': 'v', 'alpha_low': 0.18, 'alpha_high': 0.08},
                  'DR1': {'color': 'darkred', 'marker_low': 'P', 'marker_high': 'X', 'alpha_low': 0.14, 'alpha_high': 0.06}}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', required=True, choices=['edr', 'dr1', 'compare'])
    parser.add_argument('--base-dir', default=None)
    parser.add_argument('--base-edr', default=None)
    parser.add_argument('--base-dr1', default=None)
    parser.add_argument('--out-dir', default=None)
    parser.add_argument('--zones', nargs='+', default=None)
    parser.add_argument('--zones-edr', nargs='+', default=None)
    parser.add_argument('--zones-dr1', nargs='+', default=None)
    parser.add_argument('--max-zones', type=int, default=None)
    parser.add_argument('--max-zones-edr', type=int, default=None)
    parser.add_argument('--max-zones-dr1', type=int, default=None)
    parser.add_argument('--dpi', type=int, default=360)
    parser.add_argument('--no-tex', action='store_true')
    parser.add_argument('--cosmos-zone', default=None)
    parser.add_argument('--cosmos-zone-edr', default='00')
    parser.add_argument('--cosmos-zone-dr1', default=None)
    parser.add_argument('--cosmos-dr1-center-ra', type=float, default=150.10)
    parser.add_argument('--cosmos-dr1-center-dec', type=float, default=2.182)
    parser.add_argument('--cosmos-dr1-radius-deg', type=float, default=1.44)
    parser.add_argument('--nboot', type=int, default=500)
    parser.add_argument('--min-bin-count', type=int, default=20)
    parser.add_argument('--min-zone-count', type=int, default=5)
    parser.add_argument('--low-z-min', type=float, default=0.02)
    parser.add_argument('--low-z-max', type=float, default=0.10)
    parser.add_argument('--high-z-min', type=float, default=0.10)
    parser.add_argument('--high-z-max', type=float, default=0.60)
    return parser.parse_args()


def setup_style(use_tex=True, dpi=360):
    matplotlib.rcParams['figure.dpi'] = dpi
    if use_tex:
        matplotlib.rcParams['text.usetex'] = True
        matplotlib.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'
    else:
        matplotlib.rcParams['text.usetex'] = False

    plt.rcParams.update({'axes.labelsize': 16,
                         'legend.fontsize': 12,
                         'xtick.labelsize': 13,
                         'ytick.labelsize': 13})


def zone_sort_key(value):
    text = str(value)
    if re.fullmatch(r'\d+', text):
        return (0, int(text))
    return (1, text)


def normalize_zone_tag(zone):
    text = str(zone).strip()
    if re.fullmatch(r'\d+', text):
        return f'{int(text):02d}'
    return text


def discover_zones(raw_dir):
    pattern = re.compile(r'^zone_(?P<zone>[^_.]+)\.fits(?:\.gz)?$', re.IGNORECASE)
    zones = set()
    for name in os.listdir(raw_dir):
        match = pattern.match(name)
        if match:
            zones.add(normalize_zone_tag(match.group('zone')))
    if not zones:
        raise RuntimeError(f'No zone files found under {raw_dir}')
    return sorted(zones, key=zone_sort_key)


def resolve_raw_path(raw_dir, zone):
    zone_str = normalize_zone_tag(zone)
    candidates = [os.path.join(raw_dir, f'zone_{zone_str}.fits.gz'),
                  os.path.join(raw_dir, f'zone_{zone_str}.fits')]
    for path in candidates:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(candidates[0])


def resolve_prob_path(prob_dir, zone):
    zone_str = normalize_zone_tag(zone)
    candidates = [os.path.join(prob_dir, f'zone_{zone_str}_probability.fits.gz'),
                  os.path.join(prob_dir, f'zone_{zone_str}_probability.fits')]
    for path in candidates:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(candidates[0])


def decode_text_array(values):
    arr = np.asarray(values)
    if arr.dtype.kind in ('S', 'a'):
        return np.char.decode(arr, 'utf-8', errors='ignore')
    return arr.astype(str)


def as_native_endian(values):
    arr = np.asarray(values)
    if arr.dtype.byteorder in ('=', '|'):
        return arr
    return arr.astype(arr.dtype.newbyteorder('='), copy=False)


def read_raw_data_rows(raw_path, zone):
    required = ['TARGETID', 'TRACERTYPE', 'RANDITER', 'RA', 'DEC', 'Z', 'SED_SFR', 'SED_MASS', 'FLUX_G', 'FLUX_R']
    with fits.open(raw_path, memmap=True) as hdul:
        if len(hdul) < 2 or hdul[1].data is None:
            raise ValueError(f'Raw file has no table HDU 1: {raw_path}')
        available = set(hdul[1].columns.names)
        missing = [col for col in required if col not in available]
        if missing:
            raise KeyError(f'Raw file {raw_path} missing columns: {missing}')

        data = hdul[1].data
        randiter = np.asarray(data['RANDITER'])
        idx = np.flatnonzero(randiter == -1)

        out = {col: as_native_endian(np.asarray(data[col])[idx]) for col in required}

    df = pd.DataFrame(out)
    df['TARGETID'] = df['TARGETID'].astype(np.int64, copy=False)
    df['TRACERTYPE'] = decode_text_array(df['TRACERTYPE'].to_numpy())
    df['ZONE'] = normalize_zone_tag(zone)
    return df


def read_probability_rows(prob_path, zone):
    required = ['TARGETID', 'TRACERTYPE', 'PVOID', 'PSHEET', 'PFILAMENT', 'PKNOT']
    with fits.open(prob_path, memmap=True) as hdul:
        if len(hdul) < 2 or hdul[1].data is None:
            raise ValueError(f'Probability file has no table HDU 1: {prob_path}')
        available = set(hdul[1].columns.names)
        missing = [col for col in required if col not in available]
        if missing:
            raise KeyError(f'Probability file {prob_path} missing columns: {missing}')

        data = hdul[1].data
        out = {col: as_native_endian(np.asarray(data[col])) for col in required}

    df = pd.DataFrame(out)
    df['TARGETID'] = df['TARGETID'].astype(np.int64, copy=False)
    df['TRACERTYPE'] = decode_text_array(df['TRACERTYPE'].to_numpy())
    for col in P_COLS:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df['P_MAX'] = df[P_COLS].max(axis=1)
    df = df.sort_values(['TARGETID', 'TRACERTYPE', 'P_MAX'], ascending=[True, True, False])
    df = df.drop_duplicates(subset=['TARGETID', 'TRACERTYPE'], keep='first')
    df['ZONE'] = normalize_zone_tag(zone)
    return df


def load_release_dataframe(base_dir, zones):
    raw_dir = os.path.join(base_dir, 'raw')
    prob_dir = os.path.join(base_dir, 'probabilities')
    frames = []

    for zone in zones:
        raw_path = resolve_raw_path(raw_dir, zone)
        prob_path = resolve_prob_path(prob_dir, zone)

        print(f'[load] zone={zone} raw={raw_path}')
        raw_df = read_raw_data_rows(raw_path, zone)
        print(f'[load] zone={zone} prob={prob_path}')
        prob_df = read_probability_rows(prob_path, zone)

        merged = raw_df.merge(prob_df[['TARGETID', 'TRACERTYPE', 'PVOID', 'PSHEET', 'PFILAMENT', 'PKNOT']],
                              on=['TARGETID', 'TRACERTYPE'], how='inner')
        print(f'[merge] zone={zone} data_rows={len(raw_df)} merged_rows={len(merged)}')
        frames.append(merged)

    if not frames:
        raise RuntimeError('No data loaded from requested zones')
    return pd.concat(frames, ignore_index=True)


def tracer_core(label):
    text = str(label).upper()
    if text.startswith('BGS'):
        return 'BGS'
    if text.startswith('LRG'):
        return 'LRG'
    if text.startswith('ELG'):
        return 'ELG'
    if text.startswith('QSO'):
        return 'QSO'
    return text


def add_derived_columns(df):
    out = df.copy()
    out['TRACER'] = out['TRACERTYPE'].map(tracer_core)

    for col in ['RA', 'DEC', 'Z', 'SED_SFR', 'SED_MASS', 'FLUX_G', 'FLUX_R'] + P_COLS:
        out[col] = pd.to_numeric(out[col], errors='coerce')

    col_to_class = {'PVOID': 'Void', 'PSHEET': 'Sheet', 'PFILAMENT': 'Filament', 'PKNOT': 'Knot'}
    higher_col = out[P_COLS].idxmax(axis=1)
    out['ENV'] = higher_col.map(col_to_class)

    out['GR'] = np.nan
    valid_flux = (out['FLUX_G'] > 0) & (out['FLUX_R'] > 0)
    out.loc[valid_flux, 'GR'] = -2.5 * np.log10(out.loc[valid_flux, 'FLUX_G'] / out.loc[valid_flux, 'FLUX_R'])

    out['LOGM'] = np.nan
    valid_mass = out['SED_MASS'] > 0
    out.loc[valid_mass, 'LOGM'] = np.log10(out.loc[valid_mass, 'SED_MASS'])

    out['LOGSFR'] = np.nan
    valid_sfr = out['SED_SFR'] > 0
    out.loc[valid_sfr, 'LOGSFR'] = np.log10(out.loc[valid_sfr, 'SED_SFR'])

    out['LOGSSFR'] = out['LOGSFR'] - out['LOGM']
    return out


def build_bgs_sample(df):
    out = df.copy()
    mask = out['TRACER'] == 'BGS'
    mask &= np.isfinite(out['Z'])
    mask &= out['Z'] < 0.6
    mask &= (out['SED_SFR'] > 0) & (out['SED_MASS'] > 0)
    mask &= np.isfinite(out['LOGSFR']) & np.isfinite(out['LOGM']) & np.isfinite(out['LOGSSFR'])
    mask &= (out['LOGSFR'] > -4.0) & (out['LOGSFR'] < 3.0)
    out = out[mask].copy()

    finite = np.isfinite(out['GR']) & np.isfinite(out['LOGM']) & np.isfinite(out['LOGSSFR']) & np.isfinite(out['LOGSFR'])
    out = out[finite].copy()

    if not out.empty:
        unique_tracers = set(out['TRACER'].astype(str).unique().tolist())
        if unique_tracers != {'BGS'}:
            raise RuntimeError(f'Unexpected tracers after BGS selection: {sorted(unique_tracers)}')
    return out


def bootstrap_median_err(values, n_boot=500, seed=12345):
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return np.nan, np.nan, np.nan

    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        sample = rng.choice(arr, size=len(arr), replace=True)
        boots[i] = np.median(sample)

    med = np.median(arr)
    q16, q84 = np.percentile(boots, [16, 84])
    return float(med), float(med - q16), float(q84 - med)


def angular_separation_deg(ra_deg, dec_deg, ra0_deg, dec0_deg):
    ra = np.deg2rad(np.asarray(ra_deg, dtype=float))
    dec = np.deg2rad(np.asarray(dec_deg, dtype=float))
    ra0 = np.deg2rad(float(ra0_deg))
    dec0 = np.deg2rad(float(dec0_deg))
    cosang = np.sin(dec) * np.sin(dec0) + np.cos(dec) * np.cos(dec0) * np.cos(ra - ra0)
    cosang = np.clip(cosang, -1.0, 1.0)
    return np.rad2deg(np.arccos(cosang))


def build_cosmos_summary(df_bgs, z_range=(0.1, 0.5), zone_filter=None, n_boot=500,
                         cone_center=None, cone_radius_deg=None):
    zmin, zmax = z_range
    df = df_bgs[(df_bgs['Z'] >= zmin) & (df_bgs['Z'] < zmax)].copy()
    if zone_filter is not None:
        ztoken = normalize_zone_tag(zone_filter)
        df = df[df['ZONE'] == ztoken].copy()
    if cone_center is not None and cone_radius_deg is not None:
        if 'RA' not in df.columns or 'DEC' not in df.columns:
            raise RuntimeError('RA/DEC columns are required for cone filtering in cosmos summary')
        ang = angular_separation_deg(df['RA'].to_numpy(), df['DEC'].to_numpy(),
                                     cone_center[0], cone_center[1])
        df = df[np.isfinite(ang) & (ang < float(cone_radius_deg))].copy()

    df['ENV_COSMOS'] = df['ENV'].map(ENV_TO_COSMOS)

    rows = []
    for env in ENV_ORDER_COSMOS:
        vals = df.loc[df['ENV_COSMOS'] == env, 'LOGSFR'].to_numpy()
        med, elo, ehi = bootstrap_median_err(vals, n_boot=n_boot)
        rows.append({'ENV': env, 'N': int(np.isfinite(vals).sum()), 'median': med, 'elo': elo, 'ehi': ehi})

    return pd.DataFrame(rows)


def summary_has_finite_signal(summary_df):
    return int(np.isfinite(summary_df['median']).sum()) > 0


def binned_median_zone_scatter(x, y, zone, bins, min_n_bin=20, min_n_zone=5):
    x = np.asarray(x)
    y = np.asarray(y)
    zone = np.asarray(zone).astype(str)

    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    zone = zone[mask]

    xc, ym, elo, ehi = [], [], [], []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (x >= lo) & (x < hi)
        if int(np.sum(m)) < min_n_bin:
            continue

        xb, yb, zb = x[m], y[m], zone[m]
        med_global = float(np.median(yb))

        df_bin = pd.DataFrame({'y': yb, 'zone': zb})
        zone_stats = df_bin.groupby('zone')['y'].agg(['median', 'count']).reset_index()
        zone_stats = zone_stats[zone_stats['count'] >= min_n_zone].copy()
        zone_meds = zone_stats['median'].to_numpy(dtype=float)

        if len(zone_meds) <= 1:
            sigma = 0.0 if len(zone_meds) == 1 else np.nan
        else:
            sigma = float(np.std(zone_meds, ddof=1))

        xc.append(0.5 * (lo + hi))
        ym.append(med_global)
        elo.append(sigma)
        ehi.append(sigma)

    return np.asarray(xc), np.asarray(ym), np.asarray(elo), np.asarray(ehi)


def save_figure(fig, path, dpi):
    fig.savefig(path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    print(f'[saved] {path}')


def plot_cosmos_environment_comparison(summary_map, out_path, dpi, z_range=(0.1, 0.5)):
    x = np.arange(len(ENV_ORDER_COSMOS))
    fig, ax = plt.subplots(figsize=(5, 5))

    order = ['EDR', 'DR1']
    for label in order:
        if label not in summary_map:
            continue
        summary = summary_map[label]
        style = DATASET_STYLES.get(label, {'color': 'royalblue'})
        y = summary['median'].to_numpy(dtype=float)
        yerr = np.vstack([summary['elo'].to_numpy(dtype=float), summary['ehi'].to_numpy(dtype=float)])
        ax.errorbar(x, y, yerr=yerr, fmt='o-', color=style['color'], lw=1.8, ms=6,
                    capsize=3.0, label=rf'{label}: ${z_range[0]:.1f}<z<{z_range[1]:.1f}$')

    ref = COSMOS_REF.get(tuple(z_range))
    if ref is not None:
        yc = np.asarray(ref['y'], dtype=float)
        ec = np.asarray(ref['yerr'], dtype=float)
        ax.errorbar(x, yc, yerr=ec, fmt='s--', color='black', lw=1.8, ms=6, capsize=3.0,
                    alpha=0.9, label=rf'COSMOS: ${z_range[0]:.1f}<z<{z_range[1]:.1f}$')

    ax.set_xticks(x)
    ax.set_xticklabels(ENV_ORDER_COSMOS)
    ax.set_ylabel(r'$\log_{10}(\mathrm{SFR} / M_\odot\,yr^{-1})$', labelpad=10)
    ax.grid(lw=0.5, ls='--', alpha=0.5)
    ax.legend(frameon=False, loc='lower left')

    save_figure(fig, out_path, dpi)


def plot_reference_mass_ssfr(env_data_map, out_path, dpi, low_range=(0.02, 0.1), high_range=(0.1, 0.6),
                             min_bin_count=20, min_zone_count=5):
    fig, axes = plt.subplots(2, 2, figsize=(8, 8), sharex=True, sharey=True,
                             gridspec_kw={'wspace': 0, 'hspace': 0})
    axes = axes.ravel()
    mass_bins = np.arange(9.0, 12.6, 0.25)

    legend_handles = []
    for i, (ax, env) in enumerate(zip(axes, ENV_ORDER)):
        sdss = SDSS_REF[env]
        nexus = NEXUS_REF[env]
        ax.plot(sdss['x'], sdss['y'], 'o--', color='tab:gray', lw=1.3, ms=4.8, alpha=0.9)
        ax.plot(nexus['x'], nexus['y'], '--', color='darkorange', lw=1.4, alpha=0.9)

        for label in ['EDR', 'DR1']:
            if label not in env_data_map:
                continue
            df = env_data_map[label]
            style = DATASET_STYLES.get(label, {'color': 'royalblue', 'marker_low': '*', 'marker_high': 'v',
                                               'alpha_low': 0.18, 'alpha_high': 0.08})
            env_df = df[df['ENV'] == env].copy()

            plot_low = (label == 'EDR')
            low = env_df[(env_df['Z'] > low_range[0]) & (env_df['Z'] < low_range[1])]
            high = env_df[(env_df['Z'] > high_range[0]) & (env_df['Z'] < high_range[1])]

            if plot_low:
                xc, ym, elo, ehi = binned_median_zone_scatter(low['LOGM'], low['LOGSSFR'], low['ZONE'],
                                                          bins=mass_bins, min_n_bin=min_bin_count,
                                                          min_n_zone=min_zone_count)
                if xc.size:
                    ax.fill_between(xc, ym - elo, ym + ehi, color=style['color'], alpha=style['alpha_low'])
                    ax.plot(xc, ym, marker=style['marker_low'], color=style['color'], lw=1.8, ms=6)

            xc2, ym2, elo2, ehi2 = binned_median_zone_scatter(high['LOGM'], high['LOGSSFR'], high['ZONE'],
                                                               bins=mass_bins, min_n_bin=min_bin_count,
                                                               min_n_zone=min_zone_count)
            if xc2.size:
                ax.fill_between(xc2, ym2 - elo2, ym2 + ehi2, color=style['color'], alpha=style['alpha_high'])
                ax.plot(xc2, ym2, marker=style['marker_high'], color=style['color'], lw=1.5, ms=5)

        ax.text(0.97, 0.95, env, transform=ax.transAxes, ha='right', va='top',
                fontsize=16, color='black', fontweight='bold')
        ax.set_xlim(9.0, 12.0)
        ax.set_ylim(-12.8, -9.0)
        ax.set_xticks([10, 11, 12])
        ax.grid(lw=0.5, ls='--', alpha=0.5)

    axes[0].set_ylabel(r'$\log(\mathrm{sSFR}/\mathrm{yr}^{-1})$', labelpad=10)
    axes[2].set_ylabel(r'$\log(\mathrm{sSFR}/\mathrm{yr}^{-1})$', labelpad=10)
    axes[2].set_xlabel(r'$\log(M_\star/M_\odot)$', labelpad=10)
    axes[3].set_xlabel(r'$\log(M_\star/M_\odot)$', labelpad=10)

    legend_handles.extend([Line2D([0], [0], color='tab:gray', lw=1.5, ls='--', marker='o', markersize=5, label='SDSS ref'),
                           Line2D([0], [0], color='darkorange', lw=1.5, ls='-', label='NEXUS+ ref')])

    for label in ['EDR', 'DR1']:
        if label not in env_data_map:
            continue
        style = DATASET_STYLES[label]
        if label == 'EDR':
            legend_handles.append(Line2D([0], [0], color=style['color'], lw=1.8, marker=style['marker_low'], markersize=6,
                                          label=rf'{label}: ${low_range[0]:.2f}<z<{low_range[1]:.2f}$'))
        legend_handles.append(Line2D([0], [0], color=style['color'], lw=1.5, marker=style['marker_high'], markersize=5,
                                      label=rf'{label}: ${high_range[0]:.2f}<z<{high_range[1]:.2f}$'))

    fig.legend(handles=legend_handles, loc='upper center', ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    save_figure(fig, out_path, dpi)


def select_zones(raw_dir, zones_arg=None, max_zones=None):
    zones = [normalize_zone_tag(z) for z in zones_arg] if zones_arg else discover_zones(raw_dir)
    if max_zones is not None and max_zones > 0:
        zones = zones[:max_zones]
    return zones


def load_and_prepare_bgs(base_dir, zones):
    merged = load_release_dataframe(base_dir, zones)
    merged = add_derived_columns(merged)
    return build_bgs_sample(merged)


def run_single_mode(release, args):
    default_base = os.path.join('/pscratch/sd/v/vtorresg/cosmic-web', release)
    base_dir = args.base_dir or default_base
    raw_dir = os.path.join(base_dir, 'raw')
    if not os.path.isdir(raw_dir):
        raise RuntimeError(f'Raw directory not found: {raw_dir}')

    zones = select_zones(raw_dir, zones_arg=args.zones, max_zones=args.max_zones)
    out_dir = args.out_dir or os.path.join(base_dir, 'figs', 'stellar_compare')
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    print(f'[config] mode={release}')
    print(f'[config] base_dir={base_dir}')
    print(f'[config] zones={zones}')
    print(f'[config] out_dir={out_dir}')

    df_bgs = load_and_prepare_bgs(base_dir, zones)
    print(f'[sample] release={release.upper()} bgs_rows={len(df_bgs)}')

    label = release.upper()
    cosmos_summary = build_cosmos_summary(df_bgs, z_range=(0.1, 0.5), zone_filter=args.cosmos_zone, n_boot=args.nboot)
    plot_cosmos_environment_comparison({label: cosmos_summary},
                                       os.path.join(out_dir, f'cosmos_env_{release}.png'),
                                       dpi=args.dpi, z_range=(0.1, 0.5))

    plot_reference_mass_ssfr({label: df_bgs},
                             os.path.join(out_dir, f'ssfr_mass_reference_{release}.png'),
                             dpi=args.dpi,
                             low_range=(args.low_z_min, args.low_z_max),
                             high_range=(args.high_z_min, args.high_z_max),
                             min_bin_count=args.min_bin_count,
                             min_zone_count=args.min_zone_count)


def run_compare_mode(args):
    base_edr = args.base_edr or '/pscratch/sd/v/vtorresg/cosmic-web/edr'
    base_dr1 = args.base_dr1 or '/pscratch/sd/v/vtorresg/cosmic-web/dr1'

    raw_edr = os.path.join(base_edr, 'raw')
    raw_dr1 = os.path.join(base_dr1, 'raw')
    if not os.path.isdir(raw_edr):
        raise RuntimeError(f'Raw directory not found: {raw_edr}')
    if not os.path.isdir(raw_dr1):
        raise RuntimeError(f'Raw directory not found: {raw_dr1}')

    zones_edr = select_zones(raw_edr, zones_arg=args.zones_edr, max_zones=args.max_zones_edr)
    zones_dr1 = select_zones(raw_dr1, zones_arg=args.zones_dr1, max_zones=args.max_zones_dr1)

    out_dir = args.out_dir or os.path.join('/pscratch/sd/v/vtorresg/cosmic-web', 'compare', 'stellar_compare')
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    print('[config] mode=compare')
    print(f'[config] base_edr={base_edr}')
    print(f'[config] base_dr1={base_dr1}')
    print(f'[config] zones_edr={zones_edr}')
    print(f'[config] zones_dr1={zones_dr1}')
    print(f'[config] out_dir={out_dir}')

    df_edr = load_and_prepare_bgs(base_edr, zones_edr)
    df_dr1 = load_and_prepare_bgs(base_dr1, zones_dr1)

    print(f'[sample] EDR bgs_rows={len(df_edr)}')
    print(f'[sample] DR1 bgs_rows={len(df_dr1)}')
    print(f'[compare] tracer=BGS for both releases')
    print(f'[compare] cosmos_zrange=0.10<z<0.50 for both releases')
    print(f'[compare] cosmos_edr_zone={normalize_zone_tag(args.cosmos_zone_edr)} (rosetta 0)')
    print(f'[compare] cosmos_dr1_cone=center({args.cosmos_dr1_center_ra:.3f},{args.cosmos_dr1_center_dec:.3f}) '
          f'radius={args.cosmos_dr1_radius_deg:.3f} deg')
    print(f'[compare] low_zrange={args.low_z_min:.2f}<z<{args.low_z_max:.2f} for both releases')
    print(f'[compare] high_zrange={args.high_z_min:.2f}<z<{args.high_z_max:.2f} for both releases')

    edr_summary = build_cosmos_summary(df_edr, z_range=(0.1, 0.5), zone_filter=args.cosmos_zone_edr, n_boot=args.nboot)
    dr1_summary = build_cosmos_summary(df_dr1, z_range=(0.1, 0.5), zone_filter=args.cosmos_zone_dr1, n_boot=args.nboot,
                                       cone_center=(args.cosmos_dr1_center_ra, args.cosmos_dr1_center_dec),
                                       cone_radius_deg=args.cosmos_dr1_radius_deg)

    if not summary_has_finite_signal(dr1_summary):
        zmin, zmax = 0.1, 0.5
        dr1_base = df_dr1[(df_dr1['Z'] >= zmin) & (df_dr1['Z'] < zmax)].copy()
        if args.cosmos_zone_dr1 is not None:
            dr1_base = dr1_base[dr1_base['ZONE'] == normalize_zone_tag(args.cosmos_zone_dr1)].copy()
        if not dr1_base.empty:
            ang = angular_separation_deg(dr1_base['RA'].to_numpy(), dr1_base['DEC'].to_numpy(),
                                         args.cosmos_dr1_center_ra, args.cosmos_dr1_center_dec)
            min_ang = float(np.nanmin(ang))
            print(f'[warn] DR1 cone selection has 0 rows; minimum angular distance is {min_ang:.3f} deg')
        else:
            print('[warn] DR1 base sample is empty before cone selection')

        fallback_zone = normalize_zone_tag(args.cosmos_zone_dr1) if args.cosmos_zone_dr1 is not None else 'NGC'
        print(f'[warn] Falling back to DR1 zone={fallback_zone} for cosmos comparison')
        dr1_summary = build_cosmos_summary(df_dr1, z_range=(0.1, 0.5), zone_filter=fallback_zone, n_boot=args.nboot)

    summaries = {'EDR': edr_summary, 'DR1': dr1_summary}
    plot_cosmos_environment_comparison(summaries,
                                       os.path.join(out_dir, 'cosmos_env_compare_edr_dr1.png'),
                                       dpi=args.dpi,
                                       z_range=(0.1, 0.5))

    plot_reference_mass_ssfr({'EDR': df_edr, 'DR1': df_dr1},
                             os.path.join(out_dir, 'ssfr_mass_reference_compare_edr_dr1.png'),
                             dpi=args.dpi,
                             low_range=(args.low_z_min, args.low_z_max),
                             high_range=(args.high_z_min, args.high_z_max),
                             min_bin_count=args.min_bin_count,
                             min_zone_count=args.min_zone_count)


def main():
    args = parse_args()
    setup_style(use_tex=not args.no_tex, dpi=args.dpi)

    mode = args.mode.lower()
    if mode in ('edr', 'dr1'):
        run_single_mode(mode, args)
    elif mode == 'compare':
        run_compare_mode(args)
    else:
        raise RuntimeError(f'Unsupported mode: {mode}')

    print('[done] stellar comparison figures processed')


if __name__ == '__main__':
    main()