import argparse
import os
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', '/tmp/matplotlib')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from astropy.table import Table


DEFAULT_CATALOG = '/pscratch/sd/v/vtorresg/void_catalog/dr2/voids_LRG_NGC.fits'
DEFAULT_OUTPUT_DIR = 'plots/void_catalog_diagnostics'


def parse_args():
    parser = argparse.ArgumentParser( )
    parser.add_argument('--catalog', default=DEFAULT_CATALOG,
                        help='Input FITS catalogue. Default: %(default)s')
    parser.add_argument('--output-dir', default=DEFAULT_OUTPUT_DIR,
                        help='Directory for PNG outputs. Default: %(default)s')
    parser.add_argument('--prefix', default=None,
                        help='Output filename prefix. Default: input catalogue stem.')
    parser.add_argument('--bins', type=int, default=40,
                        help='Histogram bins. Default: %(default)s')
    parser.add_argument('--dpi', type=int, default=180,
                        help='Output figure DPI. Default: %(default)s')
    parser.add_argument('--min-reff', type=float, default=None,
                        help='Optional minimum R_EFF cut.')
    parser.add_argument('--max-reff', type=float, default=None,
                        help='Optional maximum R_EFF cut.')
    parser.add_argument('--exclude-footprint-edge', action='store_true',
                        help='If FOOTPRINT_EDGE exists, plot only FOOTPRINT_EDGE == False.')
    return parser.parse_args()


def read_void_table(path):
    with fits.open(path, memmap=True) as hdul:
        hdu_name = 'VOIDS' if 'VOIDS' in hdul else 1
        table = Table(hdul[hdu_name].data)
        for card in hdul[0].header.cards:
            if card.keyword not in ('', 'COMMENT', 'HISTORY'):
                table.meta[card.keyword] = card.value

    if 'R_EFF' not in table.colnames:
        raise KeyError(f'{path} missing required column R_EFF')
    return table


def as_float_array(table, name):
    return np.asarray(table[name], dtype=np.float64)


def ellipticity(table):
    if 'ELLIP' in table.colnames:
        return as_float_array(table, 'ELLIP')

    out = np.full(len(table), np.nan, dtype=np.float64)
    if 'LAMBDA_1' in table.colnames and 'LAMBDA_3' in table.colnames:
        lam1 = as_float_array(table, 'LAMBDA_1')
        lam3 = as_float_array(table, 'LAMBDA_3')
        valid = np.isfinite(lam1) & np.isfinite(lam3) & (lam1 > 0.0) & (lam3 > 0.0)
        ratio = np.clip(lam3[valid] / lam1[valid], 0.0, 1.0)
        out[valid] = 1.0 - np.power(ratio, 0.25)
        return out

    if 'SEMI_AXIS_A' in table.colnames and 'SEMI_AXIS_C' in table.colnames:
        semi_a = as_float_array(table, 'SEMI_AXIS_A')
        semi_c = as_float_array(table, 'SEMI_AXIS_C')
        valid = (np.isfinite(semi_a) & np.isfinite(semi_c) &
                 (semi_a > 0.0) & (semi_c > 0.0))
        ratio = np.clip((semi_c[valid] * semi_c[valid]) /
                        (semi_a[valid] * semi_a[valid]), 0.0, 1.0)
        out[valid] = 1.0 - np.power(ratio, 0.25)
        return out

    raise KeyError('Catalogue must contain ELLIP, LAMBDA_1/LAMBDA_3, or SEMI_AXIS_A/SEMI_AXIS_C.')


def finite_percentiles(values):
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 'no finite values'
    p16, p50, p84, p90, p95 = np.percentile(finite, [16, 50, 84, 90, 95])
    return (f'n={finite.size:,}\n'
            f'p16={p16:.3g}\n'
            f'median={p50:.3g}\n'
            f'p84={p84:.3g}\n'
            f'p90={p90:.3g}  p95={p95:.3g}')


def apply_cuts(table, args):
    reff = as_float_array(table, 'R_EFF')
    mask = np.isfinite(reff) & (reff > 0.0)
    if args.min_reff is not None:
        mask &= reff >= float(args.min_reff)
    if args.max_reff is not None:
        mask &= reff <= float(args.max_reff)
    if args.exclude_footprint_edge:
        if 'FOOTPRINT_EDGE' not in table.colnames:
            raise KeyError('--exclude-footprint-edge requested, but catalogue has no FOOTPRINT_EDGE column.')
        mask &= ~np.asarray(table['FOOTPRINT_EDGE'], dtype=bool)
    return table[mask]


def plot_hist(values, output_path, *, bins, dpi, xlabel, title, color, xlim=None):
    values = np.asarray(values, dtype=np.float64)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        raise RuntimeError(f'No finite values available for {title}')

    fig, ax = plt.subplots(figsize=(7.2, 4.8), dpi=dpi)
    ax.hist(finite, bins=bins, color=color, alpha=0.78, edgecolor='white', linewidth=0.35)
    ax.axvline(np.median(finite), color='black', linewidth=1.1, alpha=0.75)
    ax.text(0.98, 0.96, finite_percentiles(finite), transform=ax.transAxes,
            ha='right', va='top', fontsize=9,
            bbox={'facecolor': 'white', 'edgecolor': '0.75', 'alpha': 0.9})
    if xlim is not None:
        ax.set_xlim(xlim)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('count')
    ax.set_title(title)
    ax.grid(alpha=0.25, linewidth=0.6)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def main():
    args = parse_args()
    catalog = Path(args.catalog)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = args.prefix or catalog.stem

    table = apply_cuts(read_void_table(str(catalog)), args)
    if len(table) == 0:
        raise RuntimeError('No voids left after cuts.')

    reff = as_float_array(table, 'R_EFF')
    ellip = ellipticity(table)

    reff_path = output_dir / f'{prefix}_R_EFF.png'
    ellip_path = output_dir / f'{prefix}_ELLIP.png'

    plot_hist(reff, reff_path, bins=args.bins, dpi=args.dpi,
              xlabel=r'$R_{\rm eff}$ [Mpc/h]',
              title=r'$R_{\rm eff}=\sqrt{5\langle r^2\rangle/3}$',
              color='#2563EB')
    plot_hist(ellip, ellip_path, bins=args.bins, dpi=args.dpi,
              xlabel='ELLIP',
              title=r'Ellipticity: $1-(\lambda_3/\lambda_1)^{1/4}$',
              color='#DC2626', xlim=(-0.02, 1.02))

    print(f'Loaded voids: {len(table):,}')
    if 'FOOTPRINT_EDGE' in table.colnames:
        footprint_edge = np.asarray(table['FOOTPRINT_EDGE'], dtype=bool)
        print(f'FOOTPRINT_EDGE=True: {int(np.count_nonzero(footprint_edge)):,}')
    if 'EDGE' in table.colnames:
        edge = np.asarray(table['EDGE'], dtype=bool)
        print(f'EDGE=True: {int(np.count_nonzero(edge)):,}')
    print('R_EFF:', finite_percentiles(reff).replace('\n', '; '))
    print('ELLIP:', finite_percentiles(ellip).replace('\n', '; '))
    print(f'Wrote: {reff_path}')
    print(f'Wrote: {ellip_path}')


if __name__ == '__main__':
    main()
