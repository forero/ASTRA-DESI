import argparse, os, re
import sys
import json
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.io import fits
from matplotlib.lines import Line2D


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
from dr1_plot_io import table_chunks
from src.plot.plot_stellar_env import dr1_input_paths


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
    parser.add_argument('--mode', default='compare', choices=['edr', 'dr1', 'compare'])
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
    tex = parser.add_mutually_exclusive_group()
    tex.add_argument('--no-tex', action='store_true', help='Use Matplotlib text (default)')
    tex.add_argument('--tex', action='store_true', help='Use an installed LaTeX renderer')
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
    parser.add_argument('--tracers', nargs='+', choices=TRACERS, default=list(TRACERS))
    parser.add_argument('--chunk-rows', type=int, default=250_000)
    parser.add_argument('--cache-dir', type=Path, default=PROJECT_ROOT / 'plots/dr1/stellar_compare/cache')
    parser.add_argument('--force', action='store_true', help='Rebuild joined DR1 caches')
    args = parser.parse_args()
    if min(args.dpi, args.nboot, args.chunk_rows, args.min_bin_count, args.min_zone_count) <= 0:
        parser.error('dpi, nboot, chunk-rows and minimum counts must be positive')
    for prefix in ('low', 'high'):
        lo, hi = getattr(args, prefix + '_z_min'), getattr(args, prefix + '_z_max')
        if not np.isfinite([lo, hi]).all() or lo >= hi:
            parser.error('Redshift ranges must be finite and increasing')
    for name in ('max_zones', 'max_zones_edr', 'max_zones_dr1'):
        if getattr(args, name) is not None and getattr(args, name) <= 0:
            parser.error('Maximum zone counts must be positive')
    if not np.isfinite(args.cosmos_dr1_radius_deg) or args.cosmos_dr1_radius_deg <= 0:
        parser.error('COSMOS radius must be positive and finite')
    return args


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
    return text.upper()


def discover_zones(raw_dir):
    pattern = re.compile(r'^zone_(?P<zone>[^_.]+)(?:_(?:BGS_ANY|BGS_BRIGHT|LRG|ELG|QSO))?\.fits(?:\.gz)?$', re.IGNORECASE)
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
    if text.startswith('BGS_BRIGHT'):
        return 'BGS_BRIGHT'
    if text.startswith('BGS'):
        return 'BGS_ANY'
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
    return build_tracer_sample(df, 'BGS_ANY')


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


def plot_cosmos_environment_comparison(summary_map, out_path, dpi, z_range=(0.1, 0.5), title=None):
    x = np.arange(len(ENV_ORDER_COSMOS))
    fig, ax = plt.subplots(figsize=(7, 6))

    empty_labels = []
    order = ['EDR', 'DR1']
    for label in order:
        if label not in summary_map:
            continue
        summary = summary_map[label]
        if not summary_has_finite_signal(summary):
            empty_labels.append(label)
            continue
        style = DATASET_STYLES.get(label, {'color': 'royalblue'})
        y = summary['median'].to_numpy(dtype=float)
        yerr = np.vstack([summary['elo'].to_numpy(dtype=float), summary['ehi'].to_numpy(dtype=float)])
        legend_label = (rf'DESI EDR (This work): ${z_range[0]:.1f}<z<{z_range[1]:.1f}$'
                        if label == 'EDR' else
                        rf'{label}: ${z_range[0]:.1f}<z<{z_range[1]:.1f}$')
        ax.errorbar(x, y, yerr=yerr, fmt='o-', color=style['color'], lw=1.8, ms=6,
                    capsize=3.0, label=legend_label)

    ref = COSMOS_REF.get(tuple(z_range))
    if ref is not None:
        yc = np.asarray(ref['y'], dtype=float)
        ec = np.asarray(ref['yerr'], dtype=float)
        ax.errorbar(x, yc, yerr=ec, fmt='s--', color='black', lw=1.8, ms=6, capsize=3.0,
                    alpha=0.9, label=rf'COSMOS: ${z_range[0]:.1f}<z<{z_range[1]:.1f}$')

    if empty_labels:
        ax.text(.03, .97, 'No selected objects: ' + ', '.join(empty_labels),
                transform=ax.transAxes, va='top', fontsize=10)
    if title:
        ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(['Void+Sheet/Field', 'Filament', 'Knot/Cluster'], fontsize=18)
    ax.set_ylabel(r'$\log_{10}(\mathrm{SFR} / M_\odot\,yr^{-1})$', labelpad=10)
    ax.grid(lw=0.5, ls='--', alpha=0.5)
    ax.legend(frameon=False, loc='lower left', fontsize=16)
    fig.subplots_adjust(left=.15, right=.98, bottom=.18, top=.96)

    save_figure(fig, out_path, dpi)


def plot_reference_mass_ssfr(env_data_map, out_path, dpi, low_range=(0.02, 0.1), high_range=(0.1, 0.6),
                             min_bin_count=20, min_zone_count=5, title=None):
    fig, axes = plt.subplots(2, 2, figsize=(9, 9), sharex=True, sharey=True,
                             gridspec_kw={'wspace': 0, 'hspace': 0})
    axes = axes.ravel()
    mass_bins = np.arange(9.0, 12.6, 0.25)

    environment_styles = {'Void': {'reference': '#0b8dbd', 'data': 'navy'},
                          'Sheet': {'reference': '#ff9a4d', 'data': 'darkorange'},
                          'Filament': {'reference': '#00a83b', 'data': 'darkgreen'},
                          'Knot': {'reference': '#ef172c', 'data': 'darkred'}}
    for i, (ax, env) in enumerate(zip(axes, ENV_ORDER)):
        has_signal = False
        sdss = SDSS_REF[env]
        nexus = NEXUS_REF[env]
        colors = environment_styles[env]
        ax.plot(sdss['x'], sdss['y'], 'o--', color=colors['reference'], lw=1.5,
                ms=5, alpha=0.95, label='SDSS')
        ax.plot(nexus['x'], nexus['y'], '-', color=colors['reference'], lw=1.5,
                alpha=0.95, label='SAM')

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
                    has_signal = True
                    ax.fill_between(xc, ym - elo, ym + ehi, color=colors['data'], alpha=.18)
                    ax.plot(xc, ym, marker='*', color=colors['data'], lw=1.8, ms=7,
                            label=r'This work: $z<0.1$')

            xc2, ym2, elo2, ehi2 = binned_median_zone_scatter(high['LOGM'], high['LOGSSFR'], high['ZONE'],
                                                               bins=mass_bins, min_n_bin=min_bin_count,
                                                               min_n_zone=min_zone_count)
            if xc2.size:
                has_signal = True
                ax.fill_between(xc2, ym2 - elo2, ym2 + ehi2, color=colors['data'], alpha=.08)
                ax.plot(xc2, ym2, marker='v', color=colors['data'], lw=1.5, ms=5.5,
                        label=r'This work: $z>0.1$')

        if not has_signal:
            ax.text(.05, .05, 'No DESI bins meeting selection', transform=ax.transAxes, fontsize=9)
        ax.text(0.97, 0.95, env, transform=ax.transAxes, ha='right', va='top',
                fontsize=18, color='black', fontweight='bold')
        ax.set_xlim(9.0, 12.0)
        ax.set_ylim(-12.8, -9.0)
        ax.set_xticks([10, 11, 12])
        ax.grid(lw=0.5, ls='--', alpha=0.5)
        legend_handles = [
            Line2D([0], [0], color=colors['reference'], lw=1.5, ls='--', marker='o',
                   markersize=5, label='SDSS'),
            Line2D([0], [0], color=colors['reference'], lw=1.5, ls='-', label='SAM'),
            Line2D([0], [0], color=colors['data'], lw=1.8, marker='*', markersize=7,
                   label=r'This work: $z<0.1$'),
            Line2D([0], [0], color=colors['data'], lw=1.5, marker='v', markersize=5.5,
                   label=r'This work: $z>0.1$'),
        ]
        ax.legend(handles=legend_handles, frameon=False, loc='lower left', fontsize=13)

    axes[0].set_ylabel(r'$\log(\mathrm{sSFR}/\mathrm{yr}^{-1})$', labelpad=10)
    axes[2].set_ylabel(r'$\log(\mathrm{sSFR}/\mathrm{yr}^{-1})$', labelpad=10)
    axes[2].set_xlabel(r'$\log(M_\star/M_\odot)$', labelpad=10)
    axes[3].set_xlabel(r'$\log(M_\star/M_\odot)$', labelpad=10)

    missing_releases = [label for label, frame in env_data_map.items() if frame.empty]
    if missing_releases:
        fig.text(.5, -.01, 'No selected objects: ' + ', '.join(missing_releases), ha='center', fontsize=11)
    if title:
        fig.suptitle(title, y=1.08)
    fig.tight_layout()
    save_figure(fig, out_path, dpi)


def select_zones(raw_dir, zones_arg=None, max_zones=None):
    zones = [normalize_zone_tag(z) for z in zones_arg] if zones_arg else discover_zones(raw_dir)
    if max_zones is not None and max_zones > 0:
        zones = zones[:max_zones]
    if len(zones) != len(set(zones)):
        raise ValueError('Repeated zones would double-count targets')
    return zones


def load_and_prepare_bgs(base_dir, zones):
    merged = load_release_dataframe(base_dir, zones)
    merged = add_derived_columns(merged)
    return build_bgs_sample(merged)


TRACERS = ('BGS_ANY', 'BGS_BRIGHT', 'LRG', 'ELG', 'QSO')
FAMILIES = dict(zip(TRACERS, ('bgs', 'bgs', 'lrg', 'elg', 'qso')))


def read_chunk_frame(path, fields, chunk_rows, real_only=False):
    parts = []
    for chunk in table_chunks(path, fields, chunk_rows):
        mask = chunk['RANDITER'] == -1 if real_only else slice(None)
        parts.append(pd.DataFrame({col: as_native_endian(chunk[col][mask]).copy() for col in fields}))
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=fields)


def load_dr1_tracer(base_dir, zones, tracer, cache_dir, chunk_rows=250_000, force=False):
    """Cache compact joined real rows; source metadata invalidates stale caches."""
    frames = []
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    for zone in zones:
        zone = str(zone).upper()
        prop, prob = dr1_input_paths(base_dir, zone, tracer)
        raw = Path(base_dir) / 'raw' / f'zone_{zone}_{tracer}.fits.gz'
        if not raw.is_file():
            raw = raw.with_suffix('')
        sources = (raw, prop, prob)
        metadata = json.dumps({'version': 1, 'tracer': tracer, 'zone': zone,
                               'sources': [(str(p.resolve()), p.stat().st_size, p.stat().st_mtime_ns)
                                           for p in sources]}, sort_keys=True)
        cache = cache_dir / f'{zone}_{tracer}_joined.npz'
        frame = None
        if cache.is_file() and not force:
            try:
                with np.load(cache, allow_pickle=False) as stored:
                    if str(stored['metadata']) == metadata:
                        frame = pd.DataFrame({col: stored[col] for col in ('TARGETID', 'RA', 'DEC', 'Z',
                                                                           'SED_SFR', 'SED_MASS', *P_COLS)})
                        print(f'[cache] {zone} {tracer}', flush=True)
            except (OSError, ValueError, KeyError, EOFError):
                pass
        if frame is None:
            print(f'[load] {zone} {tracer}: real coordinates, properties, probabilities', flush=True)
            properties = read_chunk_frame(prop, ('TARGETID', 'SED_SFR', 'SED_MASS'), chunk_rows)
            if properties['TARGETID'].duplicated().any():
                raise ValueError(f'Duplicate properties TARGETID: {prop}')
            properties = properties[np.isfinite(properties['SED_MASS']) & (properties['SED_MASS'] > 0)
                                    & np.isfinite(properties['SED_SFR']) & (properties['SED_SFR'] > 0)]
            probabilities = read_chunk_frame(prob, ('TARGETID', 'TRACERTYPE', *P_COLS), chunk_rows)
            labels = pd.Series(decode_text_array(probabilities.pop('TRACERTYPE').to_numpy())).map(tracer_core)
            probabilities = probabilities[labels == tracer].copy()
            values = probabilities[P_COLS].to_numpy()
            probabilities = probabilities[np.isfinite(values).all(axis=1) & (values >= 0).all(axis=1)
                                          & (values <= 1).all(axis=1) & (values.sum(axis=1) > 0)].copy()
            probabilities['P_MAX'] = probabilities[P_COLS].max(axis=1)
            probabilities = probabilities.sort_values('P_MAX', ascending=False, kind='stable').drop_duplicates('TARGETID')
            joined = properties.merge(probabilities.drop(columns='P_MAX'), on='TARGETID', validate='one_to_one')
            del properties, probabilities
            # Read the large raw file in blocks, retaining only real matched targets.
            parts = []
            fields = ('TARGETID', 'RANDITER', 'RA', 'DEC', 'Z')
            wanted = joined['TARGETID'].to_numpy()
            for chunk in table_chunks(raw, fields, chunk_rows):
                real = chunk['RANDITER'] == -1
                if not real.any():
                    continue
                block = pd.DataFrame({col: as_native_endian(chunk[col][real]).copy()
                                      for col in fields if col != 'RANDITER'})
                parts.append(block[block['TARGETID'].isin(wanted)])
            coordinates = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=['TARGETID', 'RA', 'DEC', 'Z'])
            if coordinates['TARGETID'].duplicated().any():
                raise ValueError(f'Duplicate real TARGETID in {raw}')
            frame = coordinates.merge(joined, on='TARGETID', validate='one_to_one')
            with tempfile.NamedTemporaryFile(dir=cache_dir, suffix='.npz', delete=False) as stream:
                temporary = stream.name
                np.savez(stream, metadata=metadata, **{col: frame[col].to_numpy() for col in frame})
            os.replace(temporary, cache)
            print(f'[joined] {zone} {tracer}: {len(frame):,} valid mass/SFR targets', flush=True)
        frame['ZONE'] = zone
        frame['TRACERTYPE'] = tracer
        # Colour is not used by either comparison; no colour selection is imposed.
        frame['FLUX_G'] = np.nan
        frame['FLUX_R'] = np.nan
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def build_tracer_sample(df, tracer):
    mask = (df['TRACER'] == tracer) & np.isfinite(df['Z']) & df['ENV'].notna()
    mask &= np.isfinite(df['LOGM']) & np.isfinite(df['LOGSFR']) & np.isfinite(df['LOGSSFR'])
    if tracer.startswith('BGS'):
        mask &= (df['Z'] < .6) & (df['LOGSFR'] > -4) & (df['LOGSFR'] < 3)
    if 'ROSETTE_R' in df:
        radius = pd.to_numeric(df['ROSETTE_R'], errors='coerce')
        mask &= np.isfinite(radius) & (radius <= 1.5)
        if tracer == 'ELG':
            mask &= radius >= 0.3
    return df.loc[mask].copy()


def load_edr_cached(base_dir, zones, cache_dir, chunk_rows=250_000, force=False):
    """Read each EDR region once and cache the compact real-target join."""
    frames = []
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    for zone in zones:
        raw = Path(resolve_raw_path(Path(base_dir) / 'raw', zone))
        prob = Path(resolve_prob_path(Path(base_dir) / 'probabilities', zone))
        metadata = json.dumps({'version': 2, 'zone': str(zone),
                               'sources': [(str(p.resolve()), p.stat().st_size, p.stat().st_mtime_ns)
                                           for p in (raw, prob)]}, sort_keys=True)
        cache = cache_dir / f'EDR_{zone}_joined.npz'
        frame = None
        if cache.is_file() and not force:
            try:
                with np.load(cache, allow_pickle=False) as stored:
                    if str(stored['metadata']) == metadata:
                        frame = pd.DataFrame({col: stored[col] for col in stored.files if col != 'metadata'})
                        print(f'[cache] EDR {zone}', flush=True)
            except (OSError, ValueError, KeyError, EOFError):
                pass
        if frame is None:
            print(f'[load] EDR {zone}', flush=True)
            fields = ('TARGETID', 'TRACERTYPE', 'RANDITER', 'ROSETTE_R', 'RA', 'DEC', 'Z', 'SED_SFR', 'SED_MASS')
            real = read_chunk_frame(raw, fields, chunk_rows, real_only=True).drop(columns='RANDITER')
            real['TRACERTYPE'] = pd.Series(decode_text_array(real['TRACERTYPE'].to_numpy())).map(tracer_core)
            probability = read_chunk_frame(prob, ('TARGETID', 'TRACERTYPE', *P_COLS), chunk_rows)
            probability['TRACERTYPE'] = pd.Series(decode_text_array(probability['TRACERTYPE'].to_numpy())).map(tracer_core)
            values = probability[P_COLS].to_numpy()
            valid = (np.isfinite(values).all(axis=1) & (values >= 0).all(axis=1)
                     & (values <= 1).all(axis=1) & (values.sum(axis=1) > 0))
            probability = probability.loc[valid].copy()
            probability['P_MAX'] = probability[P_COLS].max(axis=1)
            probability = probability.sort_values('P_MAX', ascending=False, kind='stable').drop_duplicates(['TARGETID', 'TRACERTYPE'])
            frame = real.merge(probability.drop(columns='P_MAX'), on=['TARGETID', 'TRACERTYPE'], validate='many_to_one')
            with tempfile.NamedTemporaryFile(dir=cache_dir, suffix='.npz', delete=False) as stream:
                temporary = stream.name
                np.savez(stream, metadata=metadata, **{col: frame[col].to_numpy(dtype=str if col == 'TRACERTYPE' else None)
                                                       for col in frame})
            os.replace(temporary, cache)
        frame['ZONE'] = normalize_zone_tag(zone)
        frame['FLUX_G'] = np.nan
        frame['FLUX_R'] = np.nan
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def prepared_sample(release, base, zones, tracer, args, out):
    if release == 'dr1':
        merged = load_dr1_tracer(base, zones, tracer, args.cache_dir, args.chunk_rows, args.force)
    else:
        merged = load_edr_cached(base, zones, args.cache_dir, args.chunk_rows, args.force)
    return build_tracer_sample(add_derived_columns(merged), tracer)


def plot_samples(samples, tracer, args, out, suffix, compare=False):
    summaries = {}
    for release, frame in samples.items():
        zone = (args.cosmos_zone_edr if release == 'EDR' else args.cosmos_zone_dr1) if compare else args.cosmos_zone
        cone = dict(cone_center=(args.cosmos_dr1_center_ra, args.cosmos_dr1_center_dec),
                    cone_radius_deg=args.cosmos_dr1_radius_deg) if compare and release == 'DR1' else {}
        summaries[release] = build_cosmos_summary(frame, z_range=(.1, .5), zone_filter=zone,
                                                 n_boot=args.nboot, **cone)
        if not summary_has_finite_signal(summaries[release]):
            print(f'[empty] {release} {tracer}: no COSMOS sample in requested selection', flush=True)
    plot_cosmos_environment_comparison(summaries, Path(out) / f'cosmos_env_{suffix}_{tracer}.png',
                                       args.dpi, title=tracer.replace('_', ' '))
    plot_reference_mass_ssfr(samples, Path(out) / f'ssfr_mass_reference_{suffix}_{tracer}.png', args.dpi,
                             low_range=(args.low_z_min, args.low_z_max),
                             high_range=(args.high_z_min, args.high_z_max),
                             min_bin_count=args.min_bin_count, min_zone_count=args.min_zone_count,
                             title=tracer.replace('_', ' '))


def run_single_mode(release, args):
    base = args.base_dir or f'/pscratch/sd/v/vtorresg/cosmic-web/{release}'
    zones = select_zones(Path(base) / 'raw', args.zones, args.max_zones)
    out = Path(args.out_dir or PROJECT_ROOT / f'plots/{release}/stellar_compare')
    out.mkdir(parents=True, exist_ok=True)
    for tracer in args.tracers:
        frame = prepared_sample(release, base, zones, tracer, args, out)
        print(f'[sample] {release} {tracer}: {len(frame):,}', flush=True)
        plot_samples({release.upper(): frame}, tracer, args, out, release)


def run_compare_mode(args):
    bases = {'edr': args.base_edr or '/pscratch/sd/v/vtorresg/cosmic-web/edr',
             'dr1': args.base_dr1 or '/pscratch/sd/v/vtorresg/cosmic-web/dr1'}
    zones = {release: select_zones(Path(base) / 'raw', getattr(args, f'zones_{release}'),
                                   getattr(args, f'max_zones_{release}')) for release, base in bases.items()}
    out = Path(args.out_dir or PROJECT_ROOT / 'plots/compare/stellar_compare')
    out.mkdir(parents=True, exist_ok=True)
    edr = add_derived_columns(load_edr_cached(bases['edr'], zones['edr'], args.cache_dir, args.chunk_rows, args.force))
    available = set(edr['TRACER'].unique())
    print(f'[EDR tracers] {sorted(available)}', flush=True)
    for tracer in args.tracers:
        if tracer not in available:
            print(f'[unavailable] EDR {tracer}: absent from source catalogues', flush=True)
        samples = {'EDR': build_tracer_sample(edr, tracer),
                   'DR1': prepared_sample('dr1', bases['dr1'], zones['dr1'], tracer, args, out)}
        print(f'[sample] {tracer}: EDR={len(samples["EDR"]):,}, DR1={len(samples["DR1"]):,}', flush=True)
        plot_samples(samples, tracer, args, out, 'compare_edr_dr1', compare=True)


def main():
    args = parse_args()
    setup_style(use_tex=args.tex, dpi=args.dpi)

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
