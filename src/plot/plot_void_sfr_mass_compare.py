import argparse, os, re
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.io import fits
import matplotlib
import matplotlib.pyplot as plt


P_COLS = ['PVOID', 'PSHEET', 'PFILAMENT', 'PKNOT']
STYLES = {'EDR': {'color': 'royalblue', 'marker': 'o'},
          'DR1': {'color': 'darkred', 'marker': 's'}}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--base-edr', default=None)
    p.add_argument('--base-dr1', default=None)
    p.add_argument('--out-dir', default=None)
    p.add_argument('--out-name', default='void_sfr_mass_edr_dr1.png')
    p.add_argument('--zones-edr', nargs='+', default=None)
    p.add_argument('--zones-dr1', nargs='+', default=None)
    p.add_argument('--max-zones-edr', type=int, default=None)
    p.add_argument('--max-zones-dr1', type=int, default=None)
    p.add_argument('--tracer', default='BGS')
    p.add_argument('--z-min', type=float, default=None)
    p.add_argument('--z-max', type=float, default=None)
    p.add_argument('--mass-min', type=float, default=9.0)
    p.add_argument('--mass-max', type=float, default=12.0)
    p.add_argument('--nbins', type=int, default=12)
    p.add_argument('--min-bin-count', type=int, default=30,
                   help='Min total galaxies per bin (all zones combined) to plot.')
    p.add_argument('--min-bin-count-zone', type=int, default=5,
                   help='Min galaxies per zone per bin to compute zone median.')
    p.add_argument('--min-zone-count', type=int, default=2)
    p.add_argument('--center-stat', default='median', choices=['median', 'mean'])
    p.add_argument('--dpi', type=int, default=360)
    p.add_argument('--no-tex', action='store_true')
    return p.parse_args()


def setup_style(use_tex=True, dpi=360):
    matplotlib.rcParams['figure.dpi'] = dpi
    if use_tex:
        matplotlib.rcParams['text.usetex'] = True
        matplotlib.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'
    else:
        matplotlib.rcParams['text.usetex'] = False

    plt.rcParams.update({'axes.labelsize': 16,
                         'legend.fontsize': 13,
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


def resolve_zone_list(raw_dir, zones):
    available = discover_zones(raw_dir)
    available_lower = {z.lower(): z for z in available}
    resolved = []
    missing = []
    for zone in zones:
        token = normalize_zone_tag(zone)
        if token in available:
            resolved.append(token)
            continue
        lower = token.lower()
        if lower in available_lower:
            resolved.append(available_lower[lower])
            continue
        resolved.append(token)
        missing.append(token)
    return resolved, missing


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
    required = ['TARGETID', 'TRACERTYPE', 'RANDITER', 'Z', 'SED_SFR', 'SED_MASS']
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
    for col in ['Z', 'SED_SFR', 'SED_MASS'] + P_COLS:
        out[col] = pd.to_numeric(out[col], errors='coerce')

    out['LOGM'] = np.nan
    valid_mass = out['SED_MASS'] > 0
    out.loc[valid_mass, 'LOGM'] = np.log10(out.loc[valid_mass, 'SED_MASS'])

    out['LOGSFR'] = np.nan
    valid_sfr = out['SED_SFR'] > 0
    out.loc[valid_sfr, 'LOGSFR'] = np.log10(out.loc[valid_sfr, 'SED_SFR'])
    return out


def select_voids(df, tracer, z_min=None, z_max=None):
    out = df.copy()
    out['P_MAX'] = out[P_COLS].max(axis=1)
    higher_col = out[P_COLS].idxmax(axis=1)
    is_void = higher_col == 'PVOID'

    mask = is_void
    if tracer is not None:
        mask &= out['TRACER'] == tracer_core(tracer)
    mask &= np.isfinite(out['LOGM']) & np.isfinite(out['LOGSFR'])

    if z_min is not None:
        mask &= out['Z'] >= float(z_min)
    if z_max is not None:
        mask &= out['Z'] < float(z_max)

    return out[mask].copy()


def binned_median(x, y, bins, min_count=30):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    centers, medians, counts = [], [], []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (x >= lo) & (x < hi)
        n = int(np.sum(m))
        centers.append(0.5 * (lo + hi))
        counts.append(n)
        if n < min_count:
            medians.append(np.nan)
        else:
            medians.append(float(np.median(y[m])))

    return np.asarray(centers), np.asarray(medians), np.asarray(counts)


def binned_zone_curve(df, bins, min_bin_count_total=30, min_bin_count_zone=5,
                      min_zone_count=2, center_stat='median'):
    zones = sorted(df['ZONE'].astype(str).unique().tolist())
    n_bins = len(bins) - 1
    if not zones:
        centers = 0.5 * (bins[:-1] + bins[1:])
        return centers, np.full(n_bins, np.nan), np.full(n_bins, np.nan), np.zeros(n_bins, dtype=int), np.zeros(n_bins, dtype=int)

    med_matrix = []
    for zone in zones:
        sub = df[df['ZONE'] == zone]
        _, meds, _ = binned_median(sub['LOGM'], sub['LOGSFR'], bins=bins, min_count=min_bin_count_zone)
        med_matrix.append(meds)

    med_matrix = np.asarray(med_matrix, dtype=float)
    centers = 0.5 * (bins[:-1] + bins[1:])

    center_vals = np.full(n_bins, np.nan)
    disp_vals = np.full(n_bins, np.nan)
    n_zone = np.zeros(n_bins, dtype=int)
    n_total = np.zeros(n_bins, dtype=int)

    x_all = np.asarray(df['LOGM'], dtype=float)
    y_all = np.asarray(df['LOGSFR'], dtype=float)
    valid_all = np.isfinite(x_all) & np.isfinite(y_all)
    x_all = x_all[valid_all]

    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        n_total[i] = int(np.sum((x_all >= lo) & (x_all < hi)))
        vals = med_matrix[:, i]
        vals = vals[np.isfinite(vals)]
        n = int(vals.size)
        n_zone[i] = n
        if n_total[i] < min_bin_count_total or n < min_zone_count:
            continue
        if center_stat == 'mean':
            center_vals[i] = float(np.mean(vals))
        else:
            center_vals[i] = float(np.median(vals))
        if n >= 2:
            disp_vals[i] = float(np.std(vals, ddof=1))
        else:
            disp_vals[i] = 0.0

    return centers, center_vals, disp_vals, n_zone, n_total


def plot_void_sfr_mass(curves, out_path, dpi, mass_min, mass_max):
    fig, ax = plt.subplots(figsize=(6.2, 4.6))
    ax.grid(lw=0.4, ls='--', alpha=0.5)

    for label, curve in curves.items():
        style = STYLES.get(label, {'color': 'tab:blue', 'marker': 'o'})
        x = curve['x']
        y = curve['y']
        sigma = curve['sigma']
        mask = np.isfinite(y)
        if not np.any(mask):
            print(f'[warn] {label} curve has no finite bins')
            continue
        band = mask & np.isfinite(sigma)
        if np.any(band):
            ax.fill_between(x[band], y[band] - sigma[band], y[band] + sigma[band],
                            color=style['color'], alpha=0.25, lw=0, zorder=1)
        ax.plot(x[mask], y[mask], lw=2.0, marker=style['marker'],
                color=style['color'], label=label, zorder=2)

    ax.set_xlabel(r'$\log_{10}(M_*/M_\odot)$')
    ax.set_ylabel(r'$\log_{10}(\mathrm{SFR}/M_\odot\,\mathrm{yr}^{-1})$')
    ax.set_xlim(mass_min, mass_max)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    print(f'[saved] {out_path}')


def select_zones(raw_dir, zones_arg=None, max_zones=None):
    if zones_arg:
        zones, missing = resolve_zone_list(raw_dir, zones_arg)
        if missing:
            print(f'[warn] requested zones not found in {raw_dir}: {missing}')
    else:
        zones = discover_zones(raw_dir)
    if max_zones is not None and max_zones > 0:
        zones = zones[:max_zones]
    return zones


def main():
    args = parse_args()
    setup_style(use_tex=not args.no_tex, dpi=args.dpi)

    base_edr = args.base_edr or '/pscratch/sd/v/vtorresg/cosmic-web/edr'
    base_dr1 = args.base_dr1 or '/pscratch/sd/v/vtorresg/cosmic-web/dr1'

    raw_edr = os.path.join(base_edr, 'raw')
    raw_dr1 = os.path.join(base_dr1, 'raw')
    if not os.path.isdir(raw_edr):
        raise RuntimeError(f'Raw directory not found: {raw_edr}')
    if not os.path.isdir(raw_dr1):
        raise RuntimeError(f'Raw directory not found: {raw_dr1}')

    zones_edr = select_zones(raw_edr, zones_arg=args.zones_edr, max_zones=args.max_zones_edr)
    if args.zones_dr1 is None:
        default_dr1 = ['NGC', 'SGC']
        resolved, missing = resolve_zone_list(raw_dr1, default_dr1)
        if missing:
            print(f'[warn] default DR1 zones not found, falling back to all zones: {missing}')
            zones_dr1 = select_zones(raw_dr1, zones_arg=None, max_zones=args.max_zones_dr1)
        else:
            zones_dr1 = select_zones(raw_dr1, zones_arg=resolved, max_zones=args.max_zones_dr1)
    else:
        zones_dr1 = select_zones(raw_dr1, zones_arg=args.zones_dr1, max_zones=args.max_zones_dr1)

    out_dir = args.out_dir or os.path.join('/pscratch/sd/v/vtorresg/cosmic-web', 'compare', 'voids')
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    out_path = os.path.join(out_dir, args.out_name)

    print('[config] mode=compare-voids')
    print(f'[config] base_edr={base_edr}')
    print(f'[config] base_dr1={base_dr1}')
    print(f'[config] zones_edr={zones_edr}')
    print(f'[config] zones_dr1={zones_dr1}')
    print(f'[config] tracer={args.tracer}')
    print(f'[config] center_stat={args.center_stat}')
    print(f'[config] min_zone_count={args.min_zone_count}')
    print(f'[config] out_path={out_path}')

    df_edr = add_derived_columns(load_release_dataframe(base_edr, zones_edr))
    df_dr1 = add_derived_columns(load_release_dataframe(base_dr1, zones_dr1))

    void_edr = select_voids(df_edr, args.tracer, z_min=args.z_min, z_max=args.z_max)
    void_dr1 = select_voids(df_dr1, args.tracer, z_min=args.z_min, z_max=args.z_max)

    print(f'[sample] EDR void rows={len(void_edr)}')
    print(f'[sample] DR1 void rows={len(void_dr1)}')

    bins = np.linspace(args.mass_min, args.mass_max, args.nbins + 1)
    x_edr, y_edr, s_edr, n_edr, n_edr_tot = binned_zone_curve(
        void_edr, bins=bins, min_bin_count_total=args.min_bin_count,
        min_bin_count_zone=args.min_bin_count_zone, min_zone_count=args.min_zone_count,
        center_stat=args.center_stat
    )
    x_dr1, y_dr1, s_dr1, n_dr1, n_dr1_tot = binned_zone_curve(
        void_dr1, bins=bins, min_bin_count_total=args.min_bin_count,
        min_bin_count_zone=args.min_bin_count_zone, min_zone_count=args.min_zone_count,
        center_stat=args.center_stat
    )

    print(f'[zones] EDR used zones={len(void_edr["ZONE"].unique())}')
    print(f'[zones] DR1 used zones={len(void_dr1["ZONE"].unique())}')
    print(f'[bins] EDR zones per bin={n_edr.tolist()} total per bin={n_edr_tot.tolist()}')
    print(f'[bins] DR1 zones per bin={n_dr1.tolist()} total per bin={n_dr1_tot.tolist()}')

    curves = {
        'EDR': {'x': x_edr, 'y': y_edr, 'sigma': s_edr, 'nzone': n_edr},
        'DR1': {'x': x_dr1, 'y': y_dr1, 'sigma': s_dr1, 'nzone': n_dr1},
    }
    plot_void_sfr_mass(curves, out_path, args.dpi, args.mass_min, args.mass_max)
    print('[done] void SFR-mass comparison finished')


if __name__ == '__main__':
    main()