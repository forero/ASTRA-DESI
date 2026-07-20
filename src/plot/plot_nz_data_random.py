import argparse
import os
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

plt.style.use('dark_background')

try:
    from .color_theme import load_theme, apply_matplotlib_theme
    from .io_common import (discover_available_zones, discover_raw_catalog,
                            find_col, get_columns, iter_fits_chunks,
                            safe_upper, tracer_mask)
except ImportError:
    from color_theme import load_theme, apply_matplotlib_theme
    from io_common import (discover_available_zones, discover_raw_catalog,
                           find_col, get_columns, iter_fits_chunks,
                           safe_upper, tracer_mask)


DEFAULT_BASE = '/pscratch/sd/v/vtorresg/cosmic-web/dr2'
DEFAULT_TRACER = 'BGS'


def setup_style():
    theme_name, theme = load_theme('PLOT_WEDGE_THEME', default='dark')
    apply_matplotlib_theme(theme)
    plt.rcParams.update({'grid.linewidth': 0.25,
                         'grid.alpha': 0.35,
                         'text.usetex': True})
    return theme_name, theme


def parse_int_list(values):
    if values is None:
        return None
    out = []
    for item in values:
        for token in str(item).split(','):
            token = token.strip()
            if token:
                out.append(int(token))
    return sorted(set(out))


def _isdata_from_chunk(chunk, isdata_col, randiter_col, nrows):
    if isdata_col is not None:
        return np.asarray(chunk[isdata_col]).astype(bool)
    if randiter_col is not None:
        return np.asarray(chunk[randiter_col]) == -1
    return np.ones(nrows, dtype=bool)


def accumulate_nz(raw_paths, tracer=None, random_iters=None, bins=50,
                  zmin=0.0, zmax=3.5, chunk_rows=500_000):
    edges = np.linspace(zmin, zmax, bins + 1, dtype=np.float64)
    data_counts = np.zeros(bins, dtype=np.float64)
    random_counts = np.zeros(bins, dtype=np.float64)

    n_data_total = 0
    n_random_total = 0

    for raw_path in raw_paths:
        columns = get_columns(raw_path)
        z_col = find_col(columns, ('Z', 'z'))
        randiter_col = find_col(columns, ('RANDITER', 'randiter'))
        isdata_col = find_col(columns, ('ISDATA', 'isdata'))
        tracer_col = find_col(columns, ('TRACERTYPE', 'tracertype'))

        if z_col is None:
            raise KeyError(f'{raw_path} must contain a Z column')
        if randiter_col is None and isdata_col is None:
            raise KeyError(f'{raw_path} must contain RANDITER or ISDATA')

        wanted = [z_col]
        for col in (randiter_col, isdata_col, tracer_col):
            if col is not None and col not in wanted:
                wanted.append(col)

        for chunk in iter_fits_chunks(raw_path, wanted, chunk_rows=chunk_rows):
            z = np.asarray(chunk[z_col], dtype=np.float32)
            mask = np.isfinite(z) & (z >= zmin) & (z <= zmax)

            if tracer is not None and tracer_col is not None:
                mask &= tracer_mask(chunk[tracer_col], tracer)

            if not np.any(mask):
                continue

            isdata = _isdata_from_chunk(chunk, isdata_col, randiter_col, len(z))
            data_mask = mask & isdata
            random_mask = mask & (~isdata)

            if random_iters is not None:
                random_mask &= np.isin(np.asarray(chunk[randiter_col]), random_iters)

            if np.any(data_mask):
                vals = z[data_mask]
                data_counts += np.histogram(vals, bins=edges)[0]
                n_data_total += len(vals)

            if np.any(random_mask):
                vals = z[random_mask]
                random_counts += np.histogram(vals, bins=edges)[0]
                n_random_total += len(vals)

    if n_data_total == 0:
        raise RuntimeError('No data rows were found for the requested selection')
    if n_random_total == 0:
        raise RuntimeError('No random rows were found for the requested selection')

    return edges, data_counts, random_counts, n_data_total, n_random_total


def normalize_counts(counts, widths, mode, reference_total=None, own_total=None):
    counts = np.asarray(counts, dtype=np.float64)
    if mode == 'unit-area':
        area = np.sum(counts)
        if area <= 0.0:
            return np.full_like(counts, np.nan)
        return counts / (area * widths)

    if mode == 'counts-density':
        return counts / widths

    if mode == 'random-to-data':
        if reference_total is None or own_total is None or own_total <= 0:
            return np.full_like(counts, np.nan)
        return counts * (reference_total / own_total) / widths

    raise ValueError(f'Unknown normalization: {mode}')


def plot_nz(edges, data_counts, random_counts, n_data_total, n_random_total,
            outpath, theme, normalization='unit-area', title=None):
    widths = np.diff(edges)
    centers = 0.5 * (edges[:-1] + edges[1:])

    if normalization == 'unit-area':
        y_data = normalize_counts(data_counts, widths, mode='unit-area')
        y_random = normalize_counts(random_counts, widths, mode='unit-area')
        ylabel = r'$\hat{n}(z)$'
        data_label = r'$n_{\rm data}(z)$'
        random_label = r'$n_{\rm random}(z)$ reescalado'
    elif normalization == 'random-to-data':
        y_data = normalize_counts(data_counts, widths, mode='counts-density')
        y_random = normalize_counts(random_counts, widths, mode='random-to-data',
                                    reference_total=n_data_total,
                                    own_total=n_random_total)
        ylabel = r'$N(z)/\Delta z$'
        data_label = r'$n_{\rm data}(z)$'
        random_label = r'$n_{\rm random}(z)$ reescalado'
    else:
        y_data = normalize_counts(data_counts, widths, mode=normalization)
        y_random = normalize_counts(random_counts, widths, mode=normalization)
        ylabel = r'$N(z)/\Delta z$'
        data_label = r'$n_{\rm data}(z)$'
        random_label = r'$n_{\rm random}(z)$'

    data_color = theme['primary']
    random_color = theme['center_color']

    fig, ax = plt.subplots(figsize=(6.8, 5.0))
    ax.set_facecolor(theme['axes_facecolor'])
    ax.grid(lw=0.25, alpha=0.32)

    ax.step(centers, y_random, where='mid', color=random_color, lw=2.0,
            ls='--', label=random_label)
    ax.step(centers, y_data, where='mid', color=data_color, lw=2.2,
            label=data_label)

    ax.set_xlim(edges[0], edges[-1])
    ax.set_ylim(bottom=0.0)
    ax.set_xlabel(r'$z$', fontsize=15)
    ax.set_ylabel(ylabel, fontsize=15)
    ax.tick_params(axis='both', labelsize=12)

    if title:
        ax.set_title(title, pad=8)

    leg = ax.legend(loc='best', frameon=True, fontsize=10)
    leg.get_frame().set_alpha(0.82)

    fig.tight_layout()
    fig.savefig(outpath, dpi=360, bbox_inches='tight')
    plt.close(fig)


def resolve_raw_paths(args):
    if args.raw_path:
        return [str(Path(path)) for path in args.raw_path]

    if args.base is None:
        raise ValueError('Use --raw-path or provide --base')

    zones = args.zones if args.zones else discover_available_zones(args.base)
    paths = []
    for zone in zones:
        raw_path = discover_raw_catalog(args.base, args.tracer, zone)
        if raw_path is not None:
            paths.append(raw_path)

    if not paths:
        tracer_label = safe_upper(args.tracer) if args.tracer else 'combined'
        raise FileNotFoundError(f'No raw catalogues found for {tracer_label}')

    return paths


def default_outpath(args):
    if args.output:
        return Path(args.output)
    name = 'nz_data_random'
    if args.tracer:
        name += f'_{safe_upper(args.tracer).lower()}'
    return Path(args.outdir) / f'{name}.png'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw-path', nargs='+', default=None,
                        help='Raw FITS catalogue(s) containing Z and RANDITER/ISDATA.')
    parser.add_argument('--base', default=DEFAULT_BASE,
                        help='Release root used to discover raw catalogues.')
    parser.add_argument('--zones', nargs='*', default=None)
    parser.add_argument('--tracer', default=DEFAULT_TRACER)
    parser.add_argument('--outdir', default='figures')
    parser.add_argument('--output', default=None)
    parser.add_argument('--bins', type=int, default=50)
    parser.add_argument('--zmin', type=float, default=0.0)
    parser.add_argument('--zmax', type=float, default=3.5)
    parser.add_argument('--chunk-rows', type=int, default=500_000)
    parser.add_argument('--random-iters', nargs='*', default=None,
                        help='RANDITER values to include for randoms, e.g. 0 1 2 or 0,1,2.')
    parser.add_argument('--normalization',
                        choices=['unit-area', 'random-to-data', 'counts-density'],
                        default='unit-area')
    parser.add_argument('--title', default=None)
    args = parser.parse_args()

    theme_name, theme = setup_style()
    os.makedirs(args.outdir, exist_ok=True)

    raw_paths = resolve_raw_paths(args)
    random_iters = parse_int_list(args.random_iters)

    print(f'[plot_nz_data_random] theme={theme_name}')
    print(f'[plot_nz_data_random] raw_files={len(raw_paths)}')

    edges, data_counts, random_counts, n_data_total, n_random_total = accumulate_nz(
        raw_paths=raw_paths,
        tracer=args.tracer,
        random_iters=random_iters,
        bins=args.bins,
        zmin=args.zmin,
        zmax=args.zmax,
        chunk_rows=args.chunk_rows)

    outpath = default_outpath(args)
    outpath.parent.mkdir(parents=True, exist_ok=True)
    plot_nz(edges=edges,
            data_counts=data_counts,
            random_counts=random_counts,
            n_data_total=n_data_total,
            n_random_total=n_random_total,
            outpath=outpath,
            theme=theme,
            normalization=args.normalization,
            title=args.title)

    print(f'[plot_nz_data_random] data_rows={n_data_total}')
    print(f'[plot_nz_data_random] random_rows={n_random_total}')
    print(f'[plot_nz_data_random] saved={outpath}')


if __name__ == '__main__':
    main()