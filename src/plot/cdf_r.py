import os
import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
plt.style.use('dark_background')

try:
    from .io_common import (discover_available_zones,
                            discover_classification_realizations, find_col,
                            get_columns, iter_fits_chunks, tracer_mask)
except ImportError:
    from io_common import (discover_available_zones,
                           discover_classification_realizations, find_col,
                           get_columns, iter_fits_chunks, tracer_mask)


def setup_style():
    plt.rcParams.update({'grid.linewidth': 0.3,
                         'text.usetex': True})


def r_from_counts(ndata, nrand):
    ndata = np.asarray(ndata, dtype=np.float32)
    nrand = np.asarray(nrand, dtype=np.float32)
    denom = ndata + nrand
    r = np.full_like(denom, np.nan, dtype=np.float32)
    valid = np.isfinite(denom) & (denom > 0)
    r[valid] = (ndata[valid] - nrand[valid]) / denom[valid]
    return r


def ecdf(values):
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return None, None
    x = np.sort(x)
    y = np.arange(1, x.size + 1, dtype=float) / x.size
    return x, y


def ecdf_on_grid(values, xgrid):
    x, y = ecdf(values)
    if x is None:
        return None
    return np.interp(xgrid, x, y, left=0.0, right=1.0)


def load_r_real_rand(path, tracer, chunk_rows=500_000):
    cols = get_columns(path)

    ndata_col = find_col(cols, ('NDATA', 'ndata'))
    nrand_col = find_col(cols, ('NRAND', 'nrand'))
    isdata_col = find_col(cols, ('ISDATA', 'isdata'))
    tracer_col = find_col(cols, ('TRACERTYPE', 'tracertype'))

    if ndata_col is None or nrand_col is None or isdata_col is None:
        raise ValueError()

    wanted = [ndata_col, nrand_col, isdata_col]
    if tracer_col is not None:
        wanted.append(tracer_col)

    real_chunks = []
    rand_chunks = []

    for chunk in iter_fits_chunks(path, wanted, chunk_rows=chunk_rows):
        ndata = np.asarray(chunk[ndata_col], dtype=np.float32)
        nrand = np.asarray(chunk[nrand_col], dtype=np.float32)
        isdata = np.asarray(chunk[isdata_col]).astype(bool)

        mask = np.ones(len(ndata), dtype=bool)
        if tracer_col is not None:
            mask &= tracer_mask(chunk[tracer_col], tracer)

        if not np.any(mask):
            continue

        ndata = ndata[mask]
        nrand = nrand[mask]
        isdata = isdata[mask]

        r = r_from_counts(ndata, nrand)
        valid = np.isfinite(r)

        if np.any(valid & isdata):
            real_chunks.append(r[valid & isdata])

        if np.any(valid & (~isdata)):
            rand_chunks.append(r[valid & (~isdata)])

    r_real = np.concatenate(real_chunks) if real_chunks else np.array([], dtype=np.float32)
    r_rand = np.concatenate(rand_chunks) if rand_chunks else np.array([], dtype=np.float32)

    return r_real, r_rand


def discover_zone_iter_files(base, tracer, zones=None):
    if zones is None:
        zones = discover_available_zones(base)

    out = {}
    for zone in zones:
        files = discover_classification_realizations(base, tracer, zone)
        if files:
            out[zone] = files
    return out


def build_zone_mean_cdfs(files_for_zone, xgrid, chunk_rows=500_000,
                         tracer=None, iter_min=None, iter_max=None):
    real_curves = []
    rand_curves = []

    for it, path in files_for_zone:
        if iter_min is not None and it is not None and it < iter_min:
            continue
        if iter_max is not None and it is not None and it > iter_max:
            continue

        r_real, r_rand = load_r_real_rand(path, tracer=tracer, chunk_rows=chunk_rows)

        y_real = ecdf_on_grid(r_real, xgrid)
        y_rand = ecdf_on_grid(r_rand, xgrid)

        if y_real is not None:
            real_curves.append(y_real)
        if y_rand is not None:
            rand_curves.append(y_rand)

    mean_real = None
    mean_rand = None

    if len(real_curves) > 0:
        mean_real = np.mean(np.vstack(real_curves), axis=0)

    if len(rand_curves) > 0:
        mean_rand = np.mean(np.vstack(rand_curves), axis=0)

    n_iter_used = max(len(real_curves), len(rand_curves))
    return mean_real, mean_rand, n_iter_used


def plot_cdf_mean100_sigma_zones(base, outdir, chunk_rows=500_000,
                                 xbins=400, zones=None, iter_min=None, iter_max=None):
    colors = {'BGS': 'deepskyblue',
              'LRG': 'green',
              'ELG': 'darkorange',
              'QSO': 'crimson'}

    tracers = ['BGS', 'ELG', 'LRG', 'QSO']
    xgrid = np.linspace(-1.0, 1.0, xbins)

    fig, ax = plt.subplots()
    ax.grid(lw=0.1, alpha=0.5)

    plotted = False

    for tracer in tracers:
        zone_map = discover_zone_iter_files(base, tracer, zones=zones)

        if len(zone_map) == 0:
            continue

        zone_mean_real = []
        zone_mean_rand = []

        for zone, files_for_zone in zone_map.items():
            mean_real, mean_rand, n_iter_used = build_zone_mean_cdfs(
                files_for_zone, xgrid=xgrid, chunk_rows=chunk_rows,
                tracer=tracer,
                iter_min=iter_min, iter_max=iter_max)

            if mean_real is not None:
                zone_mean_real.append(mean_real)
            if mean_rand is not None:
                zone_mean_rand.append(mean_rand)

            print(f'---- {tracer} {zone}: {n_iter_used} realizations used')

        color = colors[tracer]

        if len(zone_mean_real) > 0:
            Y = np.vstack(zone_mean_real)
            mean_zone = np.mean(Y, axis=0)
            std_zone = np.std(Y, axis=0, ddof=1) if Y.shape[0] > 1 else np.zeros_like(mean_zone)

            low = np.clip(mean_zone - std_zone, 0.0, 1.0)
            high = np.clip(mean_zone + std_zone, 0.0, 1.0)

            ax.fill_between(xgrid, low, high, color=color, alpha=0.18, zorder=2)
            ax.plot(xgrid, mean_zone, color=color, lw=1.2, zorder=3,
                    label=rf'{tracer} real')
            plotted = True

        if len(zone_mean_rand) > 0:
            Y = np.vstack(zone_mean_rand)
            mean_zone = np.mean(Y, axis=0)
            std_zone = np.std(Y, axis=0, ddof=1) if Y.shape[0] > 1 else np.zeros_like(mean_zone)

            low = np.clip(mean_zone - std_zone, 0.0, 1.0)
            high = np.clip(mean_zone + std_zone, 0.0, 1.0)

            ax.fill_between(xgrid, low, high, color=color, alpha=0.08, zorder=1)
            ax.plot(xgrid, mean_zone, color=color, lw=1., ls=':', zorder=3,
                    label=rf'{tracer} rand')
            plotted = True

    if not plotted:
        raise RuntimeError()

    ax.set_xlim(-1.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel(r'$r$', fontsize=14)
    ax.set_ylabel(r'CDF', fontsize=14)
    ax.tick_params(axis='both', labelsize=13)
    # ax.set_title(r'CDF of $r$', pad=10)

    leg = ax.legend(loc='upper left', frameon=True)
    fig.tight_layout()

    suffix = 'alliters'
    if iter_min is not None or iter_max is not None:
        suffix = f"iters_{iter_min if iter_min is not None else 'min'}_{iter_max if iter_max is not None else 'max'}"

    outpath = Path(outdir) / f'cdf_r_mean_realizations_sigma_zones_{suffix}.png'
    fig.savefig(outpath, dpi=250, bbox_inches='tight')
    plt.close(fig)
    print(f'{outpath}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base', required=True)
    parser.add_argument('--outdir', required=True)
    parser.add_argument('--chunk-rows', type=int, default=500_000)
    parser.add_argument('--xbins', type=int, default=400)
    parser.add_argument('--zones', nargs='*', default=None)
    parser.add_argument('--iter-min', type=int, default=None)
    parser.add_argument('--iter-max', type=int, default=None)
    args = parser.parse_args()

    setup_style()
    os.makedirs(args.outdir, exist_ok=True)

    plot_cdf_mean100_sigma_zones(base=args.base, outdir=args.outdir,
                                 chunk_rows=args.chunk_rows, xbins=args.xbins, zones=args.zones,
                                 iter_min=args.iter_min, iter_max=args.iter_max)


if __name__ == '__main__':
    main()