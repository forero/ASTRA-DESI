import argparse, os, re
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.io import fits
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns
from scipy.ndimage import gaussian_filter
from sklearn.metrics import mutual_info_score


ENV_ORDER = ['Void', 'Sheet', 'Filament', 'Knot']
P_COLS = ['PVOID', 'PSHEET', 'PFILAMENT', 'PKNOT']

TRACER_FILTERS = {'BGS': {'z_min': -np.inf, 'z_max': 0.6, 'logsfr_min': -4.0, 'logsfr_max': 3.0},
                  'LRG': {'z_min': 0.4, 'z_max': 1.1, 'logsfr_min': -4.0, 'logsfr_max': 2.0}}

ENV_COLORS = {'Void': 'tab:blue', 'Sheet': 'tab:orange', 'Filament': 'tab:green', 'Knot': 'tab:red'}

ENV_MARKERS = {'Void': 'o', 'Sheet': 's', 'Filament': '^', 'Knot': 'D'}

LABELS_LATEX = {'GR': r'$(g-r)$', 'LOGM': r'$\log_{10}(M_*/M_\odot)$',
                'LOGSFR': r'$\log_{10}(\mathrm{SFR}/M_\odot\,\mathrm{yr}^{-1})$',
                'LOGSSFR': r'$\log_{10}(\mathrm{sSFR}/\mathrm{yr}^{-1})$'}

PAIR_LABELS = {'GR|LOGM': r'$(g-r)$' + '\n' + r'$\mathrm{vs}$' + '\n' + r'$\log_{10}M_*$',
               'GR|LOGSSFR': r'$(g-r)$' + '\n' + r'$\mathrm{vs}$' + '\n' + r'$\log_{10}\mathrm{sSFR}$',
               'LOGM|LOGSSFR': r'$\log_{10}M_*$' + '\n' + r'$\mathrm{vs}$' + '\n' + r'$\log_{10}\mathrm{sSFR}$'}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--release', required=True, choices=['edr', 'dr1', 'EDR', 'DR1'])
    parser.add_argument('--base-dir', default=None)
    parser.add_argument('--out-dir', default=None)
    parser.add_argument('--zones', nargs='+', default=None)
    parser.add_argument('--max-zones', type=int, default=None)
    parser.add_argument('--dpi', type=int, default=360)
    parser.add_argument('--nmi-jackknife', type=int, default=50)
    parser.add_argument( '--no-tex', action='store_true')
    return parser.parse_args()


def setup_style(use_tex=True, dpi=360):
    matplotlib.rcParams['figure.dpi'] = dpi
    if use_tex:
        matplotlib.rcParams['text.usetex'] = True
        matplotlib.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'
    else:
        matplotlib.rcParams['text.usetex'] = False

    plt.rcParams.update({'axes.labelsize': 16,
                         'legend.fontsize': 16,
                         'xtick.labelsize': 16,
                         'ytick.labelsize': 16})
    return sns.color_palette('mako_r', as_cmap=True)


def zone_sort_key(value):
    text = str(value)
    if re.fullmatch(r'\d+', text):
        return (0, int(text))
    return (1, text)


def discover_zones(raw_dir):
    pattern = re.compile(r'^zone_(?P<zone>[^_.]+)\.fits(?:\.gz)?$', re.IGNORECASE)
    zones = set()
    for name in os.listdir(raw_dir):
        match = pattern.match(name)
        if match:
            zones.add(match.group('zone'))
    if not zones:
        raise RuntimeError(f'No zone files found under {raw_dir}')
    return sorted(zones, key=zone_sort_key)


def resolve_raw_path(raw_dir, zone):
    candidates = [os.path.join(raw_dir, f'zone_{zone}.fits.gz'),
                  os.path.join(raw_dir, f'zone_{zone}.fits')]
    for path in candidates:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(candidates[0])


def resolve_prob_path(prob_dir: str, zone: str) -> str:
    candidates = [os.path.join(prob_dir, f'zone_{zone}_probability.fits.gz'),
                  os.path.join(prob_dir, f'zone_{zone}_probability.fits')]
    for path in candidates:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(candidates[0])


def decode_text_array(values):
    if values.dtype.kind in ('S', 'a'):
        return np.char.decode(values, 'utf-8', errors='ignore')
    return values.astype(str)


def as_native_endian(values):
    arr = np.asarray(values)
    if arr.dtype.byteorder in ('=', '|'):
        return arr
    return arr.astype(arr.dtype.newbyteorder('='), copy=False)


def read_raw_data_rows(raw_path, zone):
    required = ['TARGETID', 'TRACERTYPE', 'RANDITER', 'Z', 'SED_SFR', 'SED_MASS',
                'FLUX_G', 'FLUX_R']
    with fits.open(raw_path, memmap=True) as hdul:
        if len(hdul) < 2:
            raise ValueError(f'Raw file has no table HDU 1: {raw_path}')
        data = hdul[1].data
        if data is None:
            raise ValueError(f'Raw file has empty table HDU 1: {raw_path}')
        available = set(hdul[1].columns.names)
        missing = [col for col in required if col not in available]
        if missing:
            raise KeyError(f'Raw file {raw_path} missing columns: {missing}')

        randiter = np.asarray(data['RANDITER'])
        idx = np.flatnonzero(randiter == -1)

        out = {}
        for col in required:
            out[col] = as_native_endian(np.asarray(data[col])[idx])

    df = pd.DataFrame(out)
    df['TARGETID'] = df['TARGETID'].astype(np.int64, copy=False)
    df['TRACERTYPE'] = decode_text_array(df['TRACERTYPE'].to_numpy())
    df['ZONE'] = str(zone)
    return df


def read_probability_rows(prob_path, zone):
    required = ['TARGETID', 'TRACERTYPE', 'PVOID', 'PSHEET', 'PFILAMENT', 'PKNOT']
    with fits.open(prob_path, memmap=True) as hdul:
        if len(hdul) < 2:
            raise ValueError(f'Probability file has no table HDU 1: {prob_path}')
        data = hdul[1].data
        if data is None:
            raise ValueError(f'Probability file has empty table HDU 1: {prob_path}')
        available = set(hdul[1].columns.names)
        missing = [col for col in required if col not in available]
        if missing:
            raise KeyError(f'Probability file {prob_path} missing columns: {missing}')
        out = {col: as_native_endian(np.asarray(data[col])) for col in required}

    df = pd.DataFrame(out)
    df['TARGETID'] = df['TARGETID'].astype(np.int64, copy=False)
    df['TRACERTYPE'] = decode_text_array(df['TRACERTYPE'].to_numpy())
    for col in P_COLS:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df['P_MAX'] = df[P_COLS].max(axis=1)
    df = df.sort_values(['TARGETID', 'TRACERTYPE', 'P_MAX'], ascending=[True, True, False])
    df = df.drop_duplicates(subset=['TARGETID', 'TRACERTYPE'], keep='first')
    df['ZONE'] = str(zone)
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

    for col in ['Z', 'SED_SFR', 'SED_MASS', 'FLUX_G', 'FLUX_R'] + P_COLS:
        out[col] = pd.to_numeric(out[col], errors='coerce')

    higher_col = out[P_COLS].idxmax(axis=1)
    col_to_class = {'PVOID': 'Void',
                    'PSHEET': 'Sheet',
                    'PFILAMENT': 'Filament',
                    'PKNOT': 'Knot'}
    out['ENV'] = higher_col.map(col_to_class)
    out['P_MAX'] = out[P_COLS].max(axis=1)

    out['GR'] = np.nan
    valid_flux = (out['FLUX_G'] > 0) & (out['FLUX_R'] > 0)
    out.loc[valid_flux, 'GR'] = -2.5 * np.log10(out.loc[valid_flux, 'FLUX_G'] / out.loc[valid_flux, 'FLUX_R'])

    out['LOGM'] = np.nan
    valid_mass = out['SED_MASS'] > 0
    out.loc[valid_mass, 'LOGM'] = np.log10(out.loc[valid_mass, 'SED_MASS'])

    clipped_sfr = out['SED_SFR'].clip(lower=1e-5)
    out['LOGSFR'] = np.log10(clipped_sfr)
    out['LOGSSFR'] = out['LOGSFR'] - out['LOGM']

    return out


def build_tracer_samples(df):
    samples = {}
    for tracer, cuts in TRACER_FILTERS.items():
        mask = df['TRACER'] == tracer
        mask &= np.isfinite(df['Z']) & np.isfinite(df['LOGSFR']) & np.isfinite(df['LOGM']) & np.isfinite(df['LOGSSFR'])
        mask &= df['Z'] >= cuts['z_min']
        mask &= df['Z'] < cuts['z_max']
        mask &= df['LOGSFR'] > cuts['logsfr_min']
        mask &= df['LOGSFR'] < cuts['logsfr_max']
        samples[tracer] = df.loc[mask].copy()
    return samples


def save_figure(fig, path, dpi):
    fig.savefig(path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    print(f'[saved] {path}')


def plot_main_sequence_hexbin(df_bgs, out_path, cmap, dpi):
    sub = df_bgs[np.isfinite(df_bgs['LOGM']) & np.isfinite(df_bgs['LOGSFR'])].copy()
    if sub.empty:
        print('[skip] seq_bgs: no finite BGS rows')
        return

    x = sub['LOGM'].to_numpy()
    y = sub['LOGSFR'].to_numpy()

    m = 0.7
    b_ms = -7.00
    b_bcgv = -7.52
    b_gvrs = -8.02

    xq = np.nanquantile(x, [0.01, 0.99])
    yq = np.nanquantile(y, [0.01, 0.99])
    xmin, xmax = float(xq[0] - 0.2), float(xq[1] + 0.2)
    ymin, ymax = float(yq[0] - 0.3), float(yq[1] + 0.3)

    xx = np.linspace(xmin, xmax, 300)
    y_ms = m * xx + b_ms
    y_bcgv = m * xx + b_bcgv
    y_gvrs = m * xx + b_gvrs
    angle = np.degrees(np.arctan(m))

    fig, ax = plt.subplots(figsize=(6, 5), sharex=True, sharey=True)
    ax.grid(lw=0.2)

    ax.fill_between(xx, y_bcgv, 1e9, color='royalblue', alpha=0.2, zorder=0)
    ax.fill_between(xx, y_gvrs, y_bcgv, color='lightgreen', alpha=0.2, zorder=0)
    ax.fill_between(xx, -1e9, y_gvrs, color='lightcoral', alpha=0.2, zorder=0)

    hb = ax.hexbin(x, y, gridsize=120, cmap=cmap, mincnt=1, linewidths=0)

    ax.plot(xx, y_ms, ls='--', lw=1.0, color='black')
    ax.plot(xx, y_bcgv, ls=':', lw=0.9, color='darkgreen')
    ax.plot(xx, y_gvrs, ls=':', lw=0.9, color='firebrick')

    # ax.set_xlim(xmin, xmax)
    # ax.set_ylim(ymin, ymax)
    ax.set_xlim(9.3, 11.7)
    ax.set_ylim(-3.0, 3.0)

    ax.plot([0.04, 0.10], [0.93, 0.93], transform=ax.transAxes, ls='--', lw=1.2,
            color='black', clip_on=False)
    ax.text(0.12, 0.93, 'Main Sequence', transform=ax.transAxes, ha='left',
            va='center', fontsize=15,
            color='black', fontweight='bold')
    ax.text(0.04, 0.86, 'Blue Cloud', transform=ax.transAxes, ha='left', va='top',
            fontsize=15,
            color='royalblue', fontweight='bold')
    ax.text(0.96, 0.05, 'Red Sequence', transform=ax.transAxes, ha='right', va='bottom',
            fontsize=15,
            color='firebrick', fontweight='bold')
    x0, x1 = ax.get_xlim()
    x_text = x0 + 0.02 * (x1 - x0)
    y_text = m * x_text + 0.5 * (b_bcgv + b_gvrs)
    ax.text(x_text, y_text, 'Green Valley',
            rotation=angle, rotation_mode='anchor', transform_rotates_text=True,
            fontsize=15, color='darkgreen', ha='left', va='center')

    ax.set_ylabel(r'$\log_{10}(\mathrm{SFR}/M_\odot\,\mathrm{yr}^{-1})$', fontsize=22, labelpad=10)
    ax.set_xlabel(r'$\log_{10}(M_*/M_\odot)$', fontsize=22, labelpad=10)

    cbar = fig.colorbar(hb, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(r'$N_{\mathrm{Gal}}$', fontsize=22, labelpad=12)

    save_figure(fig, out_path, dpi)


def split_by_env(df):
    return {env: df[df['ENV'] == env].copy() for env in ENV_ORDER}


def plot_pdf_distributions(df_bgs, out_path, dpi):
    envs = split_by_env(df_bgs)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))

    for env in ENV_ORDER:
        x = envs[env]['GR'].to_numpy()
        x = x[np.isfinite(x)]
        if x.size:
            axes[0].hist(x, bins=40, density=True, histtype='step', lw=1.5,
                         color=ENV_COLORS[env], ls='-')

    axes[0].set_xlabel(r'$(g-r)$', labelpad=10, fontsize=20)
    axes[0].set_ylabel('PDF', labelpad=10, fontsize=19)
    axes[0].set_xlim(-0.1, 2.2)
    axes[0].grid(ls='--', lw=0.5, alpha=0.5)

    for env in ENV_ORDER:
        x = envs[env]['LOGSFR'].to_numpy()
        x = x[np.isfinite(x)]
        if x.size:
            axes[1].hist(x, bins=40, density=True, histtype='step', lw=1.5,
                         color=ENV_COLORS[env], ls='-')

    axes[1].set_xlabel(r'$\log_{10}(\mathrm{SFR}/M_\odot\,\mathrm{yr}^{-1})$',
                       labelpad=10, fontsize=18)
    axes[1].set_xlim(-3.7, 3.0)
    axes[1].grid(ls='--', lw=0.5, alpha=0.5)

    for env in ENV_ORDER:
        x = envs[env]['LOGSSFR'].to_numpy()
        x = x[np.isfinite(x)]
        if x.size:
            axes[2].hist(x, bins=40, density=True, histtype='step', lw=1.5,
                         color=ENV_COLORS[env], ls='-')

    axes[2].set_xlabel(r'$\log_{10}(\mathrm{sSFR}/\mathrm{yr}^{-1})$', labelpad=10,
                       fontsize=20)
    axes[2].set_xlim(-14.0, -7.7)
    axes[2].grid(ls='--', lw=0.5, alpha=0.5)

    handles = [Line2D([0], [0], color=ENV_COLORS[e], lw=2, ls='-', label=e) for e in ENV_ORDER]
    fig.legend(handles=handles, loc='upper center', ncol=4, frameon=True, fontsize=19)
    fig.tight_layout(rect=[0, 0, 1, 0.85])

    save_figure(fig, out_path, dpi)


def discretize(series, bins=30, qclip=(0.01, 0.99)):
    arr = np.asarray(series, dtype=float)
    mask = np.isfinite(arr)
    out = np.full(arr.shape, np.nan, dtype=float)
    if mask.sum() < 2:
        return out

    finite = arr[mask]
    lo, hi = np.quantile(finite, qclip)
    if not np.isfinite(lo) or not np.isfinite(hi):
        return out
    if hi <= lo:
        hi = lo + 1e-6

    edges = np.linspace(lo, hi, bins + 1)
    out[mask] = np.digitize(np.clip(finite, lo, hi), edges[1:-1], right=False)
    return out


def nmi_from_binned(xb, yb):
    mask = np.isfinite(xb) & np.isfinite(yb)
    if mask.sum() < 2:
        return np.nan
    x = xb[mask].astype(int)
    y = yb[mask].astype(int)

    info = mutual_info_score(x, y)

    px = np.bincount(x) / len(x)
    py = np.bincount(y) / len(y)
    px = px[px > 0]
    py = py[py > 0]

    hx = -np.sum(px * np.log(px))
    hy = -np.sum(py * np.log(py))
    if hx <= 0 or hy <= 0:
        return np.nan
    return float(info / np.sqrt(hx * hy))


def jackknife_masks(n, n_jack=50, seed=0):
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    chunks = np.array_split(idx, n_jack)
    for chunk in chunks:
        mask = np.ones(n, dtype=bool)
        mask[chunk] = False
        yield mask


def nmi_matrix_with_jackknife(df_sub, cols, bins=30, n_jack=50, seed=0):
    binned = {col: discretize(df_sub[col].to_numpy(), bins=bins) for col in cols}

    p = len(cols)
    mtx = np.zeros((p, p), dtype=float)
    err = np.zeros((p, p), dtype=float)

    for i in range(p):
        for j in range(p):
            if i == j:
                mtx[i, j] = 1.0
            else:
                mtx[i, j] = nmi_from_binned(binned[cols[i]], binned[cols[j]])

    n = len(df_sub)
    n_jack = max(2, min(int(n_jack), max(2, n)))
    jk_vals = np.zeros((n_jack, p, p), dtype=float)

    for k, mask in enumerate(jackknife_masks(n, n_jack=n_jack, seed=seed)):
        for i in range(p):
            for j in range(p):
                if i == j:
                    jk_vals[k, i, j] = 1.0
                else:
                    jk_vals[k, i, j] = nmi_from_binned(binned[cols[i]][mask], binned[cols[j]][mask])

    mean_jk = np.nanmean(jk_vals, axis=0)
    k = n_jack
    err = np.sqrt((k - 1) / k * np.nansum((jk_vals - mean_jk) ** 2, axis=0))
    return mtx, err


def get_pair_results(df, cols, bins=30, n_jack=50, seed=0):
    rows = []
    for env in ENV_ORDER:
        sub = df[df['ENV'] == env].copy()
        if len(sub) < 10:
            continue
        mtx, err = nmi_matrix_with_jackknife(sub, cols, bins=bins, n_jack=n_jack, seed=seed)
        for i in range(len(cols)):
            for j in range(i):
                var1, var2 = cols[j], cols[i]
                pair_id = f'{var1}|{var2}'
                rows.append({'ENV': env, 'var1': var1, 'var2': var2, 'pair_id': pair_id,
                             'NMI': mtx[i, j], 'ERR': err[i, j]})
    return pd.DataFrame(rows)


def plot_nmi_comparison(samples, out_path, dpi, n_jack=50):
    cols = ['GR', 'LOGM', 'LOGSSFR']
    tracers = [tr for tr in ('BGS', 'LRG') if tr in samples and not samples[tr].empty]
    if not tracers:
        print('[skip] nmi: no BGS/LRG samples')
        return

    fig, axes = plt.subplots(1, len(tracers), figsize=(5.5 * len(tracers), 5), squeeze=False)

    legend_handles = []
    for i, tracer in enumerate(tracers):
        ax = axes[0, i]
        df_pairs = get_pair_results(samples[tracer], cols, bins=30, n_jack=n_jack, seed=0)
        if df_pairs.empty:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(tracer)
            continue

        knot_vals = (df_pairs[df_pairs['ENV'] == 'Knot'][['pair_id', 'NMI']]
                     .rename(columns={'NMI': 'NMI_KNOT'}))
        dfp = df_pairs.merge(knot_vals, on='pair_id', how='left')

        pair_order = (dfp[['pair_id', 'NMI_KNOT']].drop_duplicates()
                      .sort_values('NMI_KNOT', ascending=True)['pair_id']
                      .tolist())

        y_base = np.arange(len(pair_order)) * 0.3
        pair_to_y = dict(zip(pair_order, y_base))
        offsets = {'Void': -0.06, 'Sheet': -0.02, 'Filament': +0.02, 'Knot': +0.06}

        for env in ENV_ORDER:
            sub = dfp[dfp['ENV'] == env].copy()
            if sub.empty:
                continue
            sub['y'] = sub['pair_id'].map(pair_to_y) + offsets[env]

            handle = ax.errorbar(sub['NMI'], sub['y'], xerr=sub['ERR'], fmt=ENV_MARKERS[env],
            capsize=2, label=env, color=ENV_COLORS[env], markersize=6.5, lw=1.0)

            if i == 0:
                legend_handles.append((env, handle))

        labels = [PAIR_LABELS.get(pid, pid.replace('|', '\nvs\n')) for pid in pair_order]
        ax.set_yticks(y_base)
        ax.set_yticklabels(labels, fontsize=14, multialignment='center')
        ax.tick_params(axis='y', pad=3)
        ax.set_xlabel('NMI', labelpad=4)
        ax.grid(alpha=0.5, ls='--', lw=0.5)
        ax.set_title(tracer)

    unique = {}
    for env, handle in legend_handles:
        if env not in unique:
            unique[env] = handle
    if unique:
        fig.legend(unique.values(), unique.keys(), loc='upper center', ncol=4,
                   frameon=True, bbox_to_anchor=(0.5, 1.02), markerscale=1.2, fontsize=16)

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    save_figure(fig, out_path, dpi)


def binned_median_bootstrap(values, nboot=200, rng=None):
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    n = len(arr)
    if n == 0:
        return np.nan, np.nan, 0
    median = float(np.median(arr))
    if nboot <= 1:
        return median, np.nan, n

    if rng is None:
        rng = np.random.default_rng(0)

    boots = np.empty(nboot, dtype=float)
    for i in range(nboot):
        sample = rng.choice(arr, size=n, replace=True)
        boots[i] = np.median(sample)
    return median, float(np.std(boots, ddof=1)), n


def build_quantile_ranges(values, n_bins=4):
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size < n_bins + 1:
        return []

    edges = np.quantile(arr, np.linspace(0, 1, n_bins + 1))
    edges = np.asarray(edges, dtype=float)
    for i in range(1, len(edges)):
        if edges[i] <= edges[i - 1]:
            edges[i] = edges[i - 1] + 1e-6
    return [(float(edges[i]), float(edges[i + 1])) for i in range(len(edges) - 1)]


def plot_env_vs_property_by_z(samples, ycol, ylabel, out_path, dpi, zbins=4,
                              nboot=200, min_n=30):
    tracers = [tr for tr in ('BGS', 'LRG') if tr in samples and not samples[tr].empty]
    if not tracers:
        print(f'[skip] {os.path.basename(out_path)}: no BGS/LRG samples')
        return

    fig, axes = plt.subplots(1, len(tracers), figsize=(6 * len(tracers), 5), sharey=False,
                             squeeze=False)
    cmap = plt.get_cmap('tab10')

    for i, tracer in enumerate(tracers):
        ax = axes[0, i]
        df = samples[tracer].copy()
        df = df[np.isfinite(df['Z']) & np.isfinite(df[ycol])]
        df = df[df['ENV'].isin(ENV_ORDER)]

        ranges = build_quantile_ranges(df['Z'].to_numpy(), n_bins=zbins)
        if not ranges:
            ax.text(0.5, 0.5, 'No z-bins', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(tracer)
            continue

        x = np.arange(len(ENV_ORDER))
        for iz, (zlo, zhi) in enumerate(ranges):
            dz = df[(df['Z'] >= zlo) & (df['Z'] < zhi)]
            med_vals = []
            err_vals = []
            rng = np.random.default_rng(iz)

            for env in ENV_ORDER:
                de = dz[dz['ENV'] == env]
                med, err, n = binned_median_bootstrap(de[ycol].to_numpy(), nboot=nboot, rng=rng)
                if n < min_n:
                    med_vals.append(np.nan)
                    err_vals.append(np.nan)
                else:
                    med_vals.append(med)
                    err_vals.append(err)

            ax.errorbar(x, med_vals, yerr=err_vals, marker='o', lw=1.6, capsize=2.5,
                        label=fr'${zlo:.2f} \leq z < {zhi:.2f}$', color=cmap(iz))

        ax.set_xticks(x)
        ax.set_xticklabels(ENV_ORDER)
        ax.grid(lw=0.25, alpha=0.6)
        ax.set_title(tracer)
        ax.legend(loc='lower left', ncol=1, frameon=False)

    axes[0, 0].set_ylabel(ylabel)
    fig.subplots_adjust(top=0.88, wspace=0.15, hspace=0.18)
    save_figure(fig, out_path, dpi)


def plot_env_vs_property_by_mass_bin(samples, ycol, ylabel, out_path, dpi, nbins=4, nboot=200,
                                     min_n=30, mass_cut=10.0):
    tracers = [tr for tr in ('BGS', 'LRG') if tr in samples and not samples[tr].empty]
    if not tracers:
        print(f'[skip] {os.path.basename(out_path)}: no BGS/LRG samples')
        return

    fig, axes = plt.subplots(1, len(tracers), figsize=(6 * len(tracers), 5), sharey=False, squeeze=False)
    cmap = plt.get_cmap('tab10')

    for i, tracer in enumerate(tracers):
        ax = axes[0, i]
        df = samples[tracer].copy()
        df = df[df['LOGM'] >= mass_cut]
        df = df[np.isfinite(df['LOGM']) & np.isfinite(df[ycol])]
        df = df[df['ENV'].isin(ENV_ORDER)]

        ranges = build_quantile_ranges(df['LOGM'].to_numpy(), n_bins=nbins)
        if not ranges:
            ax.text(0.5, 0.5, 'No mass bins', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(tracer)
            continue

        x = np.arange(len(ENV_ORDER))
        for ib, (lo, hi) in enumerate(ranges):
            db = df[(df['LOGM'] >= lo) & (df['LOGM'] < hi)]
            med_vals = []
            err_vals = []
            rng = np.random.default_rng(ib)

            for env in ENV_ORDER:
                de = db[db['ENV'] == env]
                med, err, n = binned_median_bootstrap(de[ycol].to_numpy(), nboot=nboot, rng=rng)
                if n < min_n:
                    med_vals.append(np.nan)
                    err_vals.append(np.nan)
                else:
                    med_vals.append(med)
                    err_vals.append(err)

            ax.errorbar(x, med_vals, yerr=err_vals, marker='o', lw=1.6, capsize=2.5,
                        label=fr'${lo:.2f} \leq \log M_* < {hi:.2f}$', color=cmap(ib))

        ax.set_xticks(x)
        ax.set_xticklabels(ENV_ORDER)
        ax.grid(lw=0.25, alpha=0.6)
        ax.set_title(tracer)
        ax.legend(loc='lower left', ncol=1, frameon=False)

    axes[0, 0].set_ylabel(ylabel)
    fig.subplots_adjust(top=0.88, wspace=0.15)
    save_figure(fig, out_path, dpi)


def plot_sfr_mass_lines(df_bgs, out_path, dpi, nboot=200, min_n=30):
    df = df_bgs[np.isfinite(df_bgs['LOGM']) & np.isfinite(df_bgs['LOGSFR'])].copy()
    if df.empty:
        print('[skip] sfr_mass_lines: no finite BGS rows')
        return

    mbins = np.linspace(9.0, 12.0, 7)
    x_centers = 0.5 * (mbins[:-1] + mbins[1:])

    fig, ax = plt.subplots(1, figsize=(6, 4.5))
    env_palette = {'Void': 'blue', 'Sheet': 'seagreen', 'Filament': 'orange', 'Knot': 'magenta'}

    for env in ENV_ORDER:
        med_vals = []
        err_vals = []
        for lo, hi in zip(mbins[:-1], mbins[1:]):
            sub = df[(df['LOGM'] >= lo) & (df['LOGM'] < hi) & (df['ENV'] == env)]
            med, err, n = binned_median_bootstrap(sub['LOGSFR'].to_numpy(), nboot=nboot)
            if n > min_n:
                med_vals.append(med)
                err_vals.append(err)
            else:
                med_vals.append(np.nan)
                err_vals.append(np.nan)

        med_arr = np.asarray(med_vals, dtype=float)
        err_arr = np.asarray(err_vals, dtype=float)
        ax.plot(x_centers, med_arr, lw=2, label=env, color=env_palette[env])
        ax.fill_between(x_centers, med_arr - err_arr, med_arr + err_arr, alpha=0.25, color=env_palette[env])

    ax.set_xlabel(r'$\log_{10}(M_*/M_\odot)$')
    ax.set_ylabel(r'$\log_{10}(\mathrm{SFR}/M_\odot\,\mathrm{yr}^{-1})$')
    ax.grid(alpha=0.3)
    ax.legend(frameon=False, fontsize=12)

    save_figure(fig, out_path, dpi)


def plot_mass_color_hexbin_env(df_bgs, out_path, cmap, dpi):
    envs = split_by_env(df_bgs)
    fig, axes = plt.subplots(2, 2, figsize=(9, 8), sharex=True, sharey=True,
                             gridspec_kw={'wspace': 0, 'hspace': 0})
    axes = axes.flatten()

    hb_last = None
    xmin, xmax = 0.0, 2.2
    ymin, ymax = 9.0, 12.5

    for j, env in enumerate(ENV_ORDER):
        ax = axes[j]
        x = envs[env]['GR'].to_numpy()
        y = envs[env]['LOGM'].to_numpy()
        mask = np.isfinite(x) & np.isfinite(y)
        x, y = x[mask], y[mask]

        if x.size:
            hb = ax.hexbin(x, y, gridsize=40, mincnt=1, cmap=cmap, extent=(xmin, xmax, ymin, ymax))
            hb_last = hb
        else:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)

        ax.grid(lw=0.5, ls='--', alpha=0.5)
        ax.text(0.97, 0.95, env, transform=ax.transAxes, ha='right', va='top', fontsize=17, color='black',
                fontweight='bold')

        if j in [0, 2]:
            ax.set_ylabel(r'$\log_{10}(M_*/M_\odot)$', labelpad=10)
        if j in [2, 3]:
            ax.set_xlabel(r'$(g-r)$', labelpad=10)

        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_xticks([0, 1, 2])

    if hb_last is not None:
        cbar = fig.colorbar(hb_last, ax=axes, shrink=0.9)
        cbar.set_label(r'$N_{\mathrm{Gal}}$')

    save_figure(fig, out_path, dpi)


def plot_mass_color_hexbin_two_zranges(df_bgs, out_path, dpi, low_range=(0.02, 0.1), high_range=(0.1, 0.6)):
    env_order = ENV_ORDER
    low = df_bgs[(df_bgs['Z'] > low_range[0]) & (df_bgs['Z'] < low_range[1])].copy()
    high = df_bgs[(df_bgs['Z'] > high_range[0]) & (df_bgs['Z'] < high_range[1])].copy()

    env_low = {e: low[low['ENV'] == e].copy() for e in env_order}
    env_high = {e: high[high['ENV'] == e].copy() for e in env_order}

    fig, axes = plt.subplots(2, 2, figsize=(9, 8), sharex=True, sharey=True,
                             gridspec_kw={'wspace': 0, 'hspace': 0})
    axes = axes.flatten()

    xmin, xmax = 0.0, 2.2
    ymin, ymax = 9.0, 12.5

    hb_low = None
    hb_high = None

    for j, env in enumerate(env_order):
        ax = axes[j]

        xh = env_high[env]['GR'].to_numpy()
        yh = env_high[env]['LOGM'].to_numpy()
        mh = np.isfinite(xh) & np.isfinite(yh)
        xh, yh = xh[mh], yh[mh]
        if xh.size:
            hb_high = ax.hexbin( xh, yh, gridsize=40, mincnt=1, extent=(xmin, xmax, ymin, ymax),
                                cmap='Reds', linewidths=0)

        xl = env_low[env]['GR'].to_numpy()
        yl = env_low[env]['LOGM'].to_numpy()
        ml = np.isfinite(xl) & np.isfinite(yl)
        xl, yl = xl[ml], yl[ml]
        if xl.size:
            hb_low = ax.hexbin(xl, yl, gridsize=40, mincnt=1, extent=(xmin, xmax, ymin, ymax),
                               cmap='Blues', linewidths=0)

        ax.grid(lw=0.5, ls='--', alpha=0.5)
        ax.text(0.97, 0.95, env, transform=ax.transAxes, ha='right', va='top', fontsize=17,
                color='black', fontweight='bold')

        if j in [0, 2]:
            ax.set_ylabel(r'$\log_{10}(M_*/M_\odot)$', labelpad=10)
        if j in [2, 3]:
            ax.set_xlabel(r'$(g-r)$', labelpad=10)

        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_xticks([0, 1, 2])

    if hb_low is not None:
        cbar1 = fig.colorbar(hb_low, ax=axes, shrink=0.82, pad=0.02)
        cbar1.set_label(r'$N_{\mathrm{gal}}\ (z<0.1)$')
    if hb_high is not None:
        cbar2 = fig.colorbar(hb_high, ax=axes, shrink=0.82, pad=0.10)
        cbar2.set_label(r'$N_{\mathrm{gal}}\ (z>0.1)$')

    save_figure(fig, out_path, dpi)


def plot_mass_color_contours_two_zranges(df_bgs, out_path, dpi, low_range=(0.02, 0.1), high_range=(0.1, 0.6)):
    env_order = ENV_ORDER
    low = df_bgs[(df_bgs['Z'] > low_range[0]) & (df_bgs['Z'] < low_range[1])].copy()
    high = df_bgs[(df_bgs['Z'] > high_range[0]) & (df_bgs['Z'] < high_range[1])].copy()

    env_low = {e: low[low['ENV'] == e].copy() for e in env_order}
    env_high = {e: high[high['ENV'] == e].copy() for e in env_order}

    fig, axes = plt.subplots(2, 2, figsize=(7.5, 7), sharex=True, sharey=True,
                             gridspec_kw={'wspace': 0, 'hspace': 0})
    axes = axes.flatten()

    xmin, xmax = 0.0, 2.2
    ymin, ymax = 9.0, 12.5

    xbins = np.linspace(xmin, xmax, 80)
    ybins = np.linspace(ymin, ymax, 80)
    xc = 0.5 * (xbins[:-1] + xbins[1:])
    yc = 0.5 * (ybins[:-1] + ybins[1:])
    xx, yy = np.meshgrid(xc, yc, indexing='ij')

    for j, env in enumerate(env_order):
        ax = axes[j]

        x1 = env_low[env]['GR'].to_numpy()
        y1 = env_low[env]['LOGM'].to_numpy()
        m1 = np.isfinite(x1) & np.isfinite(y1)
        x1, y1 = x1[m1], y1[m1]

        if x1.size:
            h1, xedges, yedges = np.histogram2d(x1, y1, bins=[xbins, ybins], range=[[xmin, xmax], [ymin, ymax]])
            h1 = gaussian_filter(h1, sigma=1.2)
            if np.nanmax(h1) > 0:
                levels1 = np.nanmax(h1) * np.array([0.1, 0.3, 0.5, 0.7])
                ax.contour(xx, yy, h1, levels=levels1, cmap='Blues', linewidths=1.8)

        x2 = env_high[env]['GR'].to_numpy()
        y2 = env_high[env]['LOGM'].to_numpy()
        m2 = np.isfinite(x2) & np.isfinite(y2)
        x2, y2 = x2[m2], y2[m2]

        if x2.size:
            h2, _, _ = np.histogram2d(x2, y2, bins=[xbins, ybins], range=[[xmin, xmax], [ymin, ymax]])
            h2 = gaussian_filter(h2, sigma=1.2)
            if np.nanmax(h2) > 0:
                levels2 = np.nanmax(h2) * np.array([0.1, 0.3, 0.5, 0.7])
                ax.contour(xx, yy, h2, levels=levels2, cmap='Reds', linewidths=1.8, linestyles='--')

        ax.grid(lw=0.5, ls='--', alpha=0.5)
        ax.text(0.97, 0.95, env, transform=ax.transAxes, ha='right', va='top', fontsize=17,
                color='black', fontweight='bold')

        if j in [0, 2]:
            ax.set_ylabel(r'$\log_{10}(M_*/M_\odot)$', labelpad=10)
        if j in [2, 3]:
            ax.set_xlabel(r'$(g-r)$', labelpad=10)

        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_xticks([0, 1, 2])

    handles = [Line2D([0], [0], color='tab:blue', lw=2, label=r'$z<0.1$'),
               Line2D([0], [0], color='tab:red', lw=2, ls='--', label=r'$z>0.1$')]
    axes[0].legend(handles=handles, frameon=False, loc='lower right', fontsize=11)

    fig.tight_layout()
    save_figure(fig, out_path, dpi)


def plot_mass_color_contours_all(df_bgs, out_path, cmap, dpi):
    envs = split_by_env(df_bgs)

    fig, axes = plt.subplots(2, 2, figsize=(7.5, 7), sharex=True, sharey=True,
                             gridspec_kw={'wspace': 0, 'hspace': 0})
    axes = axes.flatten()

    xmin, xmax = 0.0, 2.2
    ymin, ymax = 9.0, 12.5

    xbins = np.linspace(xmin, xmax, 80)
    ybins = np.linspace(ymin, ymax, 80)

    for j, env in enumerate(ENV_ORDER):
        ax = axes[j]
        x = envs[env]['GR'].to_numpy()
        y = envs[env]['LOGM'].to_numpy()
        m = np.isfinite(x) & np.isfinite(y)
        x, y = x[m], y[m]

        if x.size:
            h, xedges, yedges = np.histogram2d(x, y, bins=[xbins, ybins], range=[[xmin, xmax], [ymin, ymax]])
            h = gaussian_filter(h, sigma=1.2)

            xc = 0.5 * (xedges[:-1] + xedges[1:])
            yc = 0.5 * (yedges[:-1] + yedges[1:])
            xx, yy = np.meshgrid(xc, yc, indexing='ij')

            if np.nanmax(h) > 0:
                levels = np.nanmax(h) * np.array([0.1, 0.3, 0.5, 0.7])
                ax.contour(xx, yy, h, levels=levels, linewidths=1.8, cmap=cmap, linestyles='--')
        else:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)

        ax.grid(lw=0.5, ls='--', alpha=0.5)
        ax.text(0.97, 0.95, env, transform=ax.transAxes, ha='right', va='top', fontsize=17,
                color='black', fontweight='bold')

        if j in [0, 2]:
            ax.set_ylabel(r'$\log_{10}(M_*/M_\odot)$', labelpad=10)
        if j in [2, 3]:
            ax.set_xlabel(r'$(g-r)$', labelpad=10)

        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_xticks([0, 1, 2])

    fig.tight_layout()
    save_figure(fig, out_path, dpi)


def main() -> None:
    args = parse_args()
    release = args.release.lower()

    default_base = os.path.join('/pscratch/sd/v/vtorresg/cosmic-web', release)
    base_dir = args.base_dir or default_base
    raw_dir = os.path.join(base_dir, 'raw')

    if not os.path.isdir(raw_dir):
        raise RuntimeError(f'Raw directory not found: {raw_dir}')

    zones = [str(z) for z in args.zones] if args.zones else discover_zones(raw_dir)
    if args.max_zones is not None and args.max_zones > 0:
        zones = zones[: args.max_zones]

    out_dir = args.out_dir or os.path.join(base_dir, 'figs', 'stellar_props')
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    print(f'[config] release={release}')
    print(f'[config] base_dir={base_dir}')
    print(f'[config] out_dir={out_dir}')
    print(f'[config] zones={zones}')

    cmap = setup_style(use_tex=not args.no_tex, dpi=args.dpi)

    merged = load_release_dataframe(base_dir, zones)
    merged = add_derived_columns(merged)
    samples = build_tracer_samples(merged)

    for tracer in ('BGS', 'LRG'):
        count = len(samples.get(tracer, pd.DataFrame()))
        print(f'[sample] tracer={tracer} rows={count}')

    df_bgs = samples.get('BGS', pd.DataFrame())

    if not df_bgs.empty:
        plot_main_sequence_hexbin(df_bgs, os.path.join(out_dir, 'seq_bgs.png'), cmap=cmap, dpi=args.dpi)
        plot_pdf_distributions(df_bgs, os.path.join(out_dir, 'pdf.png'), dpi=args.dpi)
        plot_sfr_mass_lines(df_bgs, os.path.join(out_dir, 'sfr_mass_lines_env.png'), dpi=args.dpi)
        plot_mass_color_hexbin_env(df_bgs, os.path.join(out_dir, 'mass_color_hexbin.png'), cmap=cmap, dpi=args.dpi)
        plot_mass_color_hexbin_two_zranges(df_bgs, os.path.join(out_dir, 'mass_color_hexbin_two_zranges.png'), dpi=args.dpi)
        plot_mass_color_contours_two_zranges(df_bgs, os.path.join(out_dir, 'mass_color_contours_two_zranges.png'), dpi=args.dpi)
        plot_mass_color_contours_all(df_bgs, os.path.join(out_dir, 'mass_color.png'), cmap=cmap, dpi=args.dpi)
    else:
        print('[skip] BGS-only figures: empty BGS sample')

    plot_nmi_comparison(samples, os.path.join(out_dir, 'nmi.png'), dpi=args.dpi, n_jack=args.nmi_jackknife)

    plot_env_vs_property_by_z(samples, ycol='LOGSFR',
                              ylabel=r'$\log_{10}(\mathrm{SFR}/M_\odot\,\mathrm{yr}^{-1})$',
                              out_path=os.path.join(out_dir, 'sfr_z_env.png'),
                              dpi=args.dpi)
    plot_env_vs_property_by_z(samples, ycol='LOGSSFR',
                              ylabel=r'$\log_{10}(\mathrm{sSFR}/\mathrm{yr}^{-1})$',
                              out_path=os.path.join(out_dir, 'ssfr_z_env.png'),
                              dpi=args.dpi)

    plot_env_vs_property_by_mass_bin(samples,
                                     ycol='LOGSFR',
                                     ylabel=r'$\log_{10}(\mathrm{SFR}/M_\odot\,\mathrm{yr}^{-1})$',
                                     out_path=os.path.join(out_dir, 'sfr_mass_env.png'),
                                     dpi=args.dpi)
    plot_env_vs_property_by_mass_bin(samples,
                                     ycol='LOGSSFR',
                                     ylabel=r'$\log_{10}(\mathrm{sSFR}/\mathrm{yr}^{-1})$',
                                     out_path=os.path.join(out_dir, 'ssfr_mass_env.png'),
                                     dpi=args.dpi)
    print('[done] all requested plots processed')


if __name__ == '__main__':
    main()
