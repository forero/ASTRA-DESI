import os, argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
plt.style.use('dark_background')

try:
    from .io_common import (discover_raw_catalog, find_col, get_columns,
                            iter_fits_chunks, tracer_mask)
except ImportError:
    from io_common import (discover_raw_catalog, find_col, get_columns,
                           iter_fits_chunks, tracer_mask)


def setup_style():
    plt.rcParams.update({'text.usetex': True})


def load_redshift_for_tracer(base, tracer, zones, chunk_rows=500_000):
    z_all = []

    for zone in zones:
        raw_path = discover_raw_catalog(base, tracer, zone)
        if raw_path is None:
            continue

        cols = get_columns(raw_path)
        z_col = find_col(cols, ('Z', 'z'))
        isdata_col = find_col(cols, ('ISDATA', 'isdata'))
        randiter_col = find_col(cols, ('RANDITER', 'randiter'))
        tracer_col = find_col(cols, ('TRACERTYPE', 'tracertype'))

        if z_col is None:
            continue

        wanted = [z_col]
        for c in (isdata_col, randiter_col, tracer_col):
            if c is not None:
                wanted.append(c)

        chunks = []
        for chunk in iter_fits_chunks(raw_path, wanted, chunk_rows=chunk_rows):
            z = np.asarray(chunk[z_col], dtype=np.float32)

            mask = np.isfinite(z)
            if tracer_col is not None:
                mask &= tracer_mask(chunk[tracer_col], tracer)

            if isdata_col is not None:
                mask &= np.asarray(chunk[isdata_col]).astype(bool)
            elif randiter_col is not None:
                mask &= (np.asarray(chunk[randiter_col]) == -1)

            if np.any(mask):
                chunks.append(z[mask])

        if chunks:
            z_all.append(np.concatenate(chunks))

    return np.concatenate(z_all) if z_all else np.array([], dtype=np.float32)


def plot_histogram(base, zones, outdir, bins=30, zmin=0.0, zmax=3.5, chunk_rows=500_000):
    colors = {'BGS': 'deepskyblue',
              'LRG': 'green',
              'ELG': 'darkorange',
              'QSO': 'crimson'}

    tracers = ['BGS', 'LRG', 'ELG', 'QSO']

    fig, ax = plt.subplots()
    ax.grid(lw=0.3, alpha=0.5)

    bin_edges = np.linspace(zmin, zmax, bins + 1)
    widths = np.diff(bin_edges)

    plotted = False

    for tracer in tracers:
        z = load_redshift_for_tracer(base, tracer, zones, chunk_rows=chunk_rows)

        if z.size == 0:
            continue

        counts, _ = np.histogram(z, bins=bin_edges)
        y = counts / widths

        ax.bar(bin_edges[:-1], y, width=widths, align='edge',
               color=colors[tracer], alpha=0.85,
               edgecolor=colors[tracer], linewidth=0.3,
               label=rf'{tracer} object')

        plotted = True

    if not plotted:
        raise RuntimeError()
    ax.set_xlim(zmin, zmax)
    ax.set_xlabel(r'$z$', fontsize=13)
    ax.set_ylabel(r'$N_{\mathrm{Gal}}/\Delta z$', fontsize=13)
    ax.tick_params(axis='both', labelsize=13)
    # ax.set_yscale('log')

    leg = ax.legend(loc='upper right')
    fig.tight_layout()

    outpath = Path(outdir) / 'redshift_distribution_by_tracer.png'
    fig.savefig(outpath, dpi=360, bbox_inches='tight')
    plt.close(fig)
    print(f'\n------------ {outpath}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base', required=True)
    parser.add_argument('--zones', nargs='+', required=True)
    parser.add_argument('--outdir', required=True)
    parser.add_argument('--bins', type=int, default=30)
    parser.add_argument('--zmin', type=float, default=0.0)
    parser.add_argument('--zmax', type=float, default=3.5)
    parser.add_argument('--chunk-rows', type=int, default=500_000)
    args = parser.parse_args()

    setup_style()
    os.makedirs(args.outdir, exist_ok=True)

    plot_histogram(base=args.base, zones=args.zones,
                   outdir=args.outdir, bins=args.bins,
                   zmin=args.zmin, zmax=args.zmax,
                   chunk_rows=args.chunk_rows)


if __name__ == '__main__':
    main()