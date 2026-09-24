import argparse
from pathlib import Path
import shutil
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import numpy as np
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
from src.plot.plot_stellar_env_compare import load_edr_cached, select_zones


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=Path('/pscratch/sd/v/vtorresg/cosmic-web/edr'))
    parser.add_argument('--cache-dir', type=Path, default=PROJECT_ROOT / 'plots/dr1/stellar_compare/cache')
    parser.add_argument('--zones', nargs='+', default=None, help='EDR regions, e.g. 00 01; default: all')
    parser.add_argument('--output', type=Path, default=PROJECT_ROOT / 'plots/edr/stellar_props/main_sequence_bgs_any_edr.png')
    parser.add_argument('--dpi', type=int, default=300)
    parser.add_argument('--gridsize', type=int, default=120)
    parser.add_argument('--chunk-rows', type=int, default=250_000)
    parser.add_argument('--vmax', type=float, help='Upper count colour limit; default is the observed maximum')
    parser.add_argument('--force', action='store_true', help='Rebuild the joined EDR caches')
    parser.add_argument('--no-tex', action='store_true', help='Optional Matplotlib mathtext fallback')
    args = parser.parse_args()
    if min(args.dpi, args.gridsize, args.chunk_rows) <= 0:
        parser.error('dpi, gridsize and chunk-rows must be positive')
    if args.vmax is not None and (not np.isfinite(args.vmax) or args.vmax <= 0):
        parser.error('--vmax must be finite and positive')
    if args.zones and len(args.zones) != len(set(args.zones)):
        parser.error('Zones must not be repeated')
    if not args.no_tex and (not shutil.which('latex') or not shutil.which('dvipng')):
        parser.error('LaTeX rendering requires latex and dvipng on PATH; alternatively use --no-tex')
    return args


def load_mass_sfr(args):
    zones = select_zones(args.data_dir / 'raw', args.zones)
    if any(not zone.isdigit() for zone in zones):
        raise ValueError('Expected numeric EDR regions, e.g. 00 01')
    frame = load_edr_cached(args.data_dir, zones, args.cache_dir, args.chunk_rows, args.force)
    frame = frame.loc[frame['TRACERTYPE'] == 'BGS_ANY']
    print(f'[source] EDR only; {len(zones)} regions; BGS ANY')
    mass = frame['SED_MASS'].to_numpy()
    sfr = frame['SED_SFR'].to_numpy()
    redshift = frame['Z'].to_numpy()
    valid = (np.isfinite(mass) & (mass > 0) & np.isfinite(sfr) & (sfr > 0)
             & np.isfinite(redshift) & (redshift < .6))
    x, y = np.log10(mass[valid]), np.log10(sfr[valid])
    selected = (y > -4) & (y < 3)
    return x[selected], y[selected]


def make_figure(x, y, gridsize=120, vmax=None, use_tex=True):
    if not len(x):
        raise ValueError('No BGS ANY objects pass the selection')
    xmin, xmax, ymin, ymax = 5.75, 12.35, -3.95, 3.0
    style = {'text.usetex': use_tex, 'font.family': 'serif',
             'font.serif': ['Computer Modern Roman', 'DejaVu Serif'],
             'axes.linewidth': 1.1, 'font.size': 16,
             'xtick.labelsize': 19, 'ytick.labelsize': 19}
    with plt.rc_context(style):
        fig, ax = plt.subplots(figsize=(8.6, 7.1))
        fig.subplots_adjust(left=.16, right=.85, bottom=.16, top=.96)
        xx = np.linspace(xmin, xmax, 300)
        slope, main, upper, lower = .7, -7., -7.52, -8.02
        ax.fill_between(xx, slope*xx + upper, ymax, color='royalblue', alpha=.20, zorder=1)
        ax.fill_between(xx, slope*xx + lower, slope*xx + upper, color='lightgreen', alpha=.20, zorder=1)
        ax.fill_between(xx, ymin, slope*xx + lower, color='lightcoral', alpha=.20, zorder=1)
        visible = (x >= xmin) & (x <= xmax) & (y >= ymin) & (y <= ymax)
        hb = ax.hexbin(x[visible], y[visible], gridsize=gridsize,
                       extent=(xmin, xmax, ymin, ymax), mincnt=1,
                       cmap='plasma',#sns.color_palette('mako_r', as_cmap=True),
                       vmin=0, vmax=vmax, linewidths=0, rasterized=True,
                       alpha=0.9,
                       zorder=0)
        ax.plot(xx, slope*xx + main, '--', color='black', lw=1.2, zorder=1)
        ax.plot(xx, slope*xx + upper, ':', color='darkgreen', lw=1.0, zorder=1, alpha=1)
        ax.plot(xx, slope*xx + lower, ':', color='firebrick', lw=1.0, zorder=1, alpha=1)
        ax.set(xlim=(xmin, xmax), ylim=(ymin, ymax))
        ax.xaxis.set_major_locator(MultipleLocator(2))
        ax.set_yticks([-3, -2, -1, 0, 1, 2])
        ax.grid(lw=.5, alpha=.25)
        ax.plot([.04, .10], [.93, .93], transform=ax.transAxes, ls='--', lw=1.3, color='black')
        ax.text(.115, .93, r'\textsf{Main Sequence}' if use_tex else 'Main Sequence',
                transform=ax.transAxes, va='center', fontsize=18)
        ax.text(.035, .865, r'\textsf{Blue Cloud}' if use_tex else 'Blue Cloud',
                transform=ax.transAxes, va='top', color='royalblue', fontsize=19)
        ax.text(.96, .045, r'\textsf{Red Sequence}' if use_tex else 'Red Sequence',
                transform=ax.transAxes, ha='right', va='bottom', color='firebrick', fontsize=18)
        label_x = xmin + .035*(xmax-xmin)
        ax.text(label_x, slope*label_x + .5*(upper+lower),
                r'\textsf{Green Valley}' if use_tex else 'Green Valley', color='darkgreen',
                rotation=np.degrees(np.arctan(slope)), transform_rotates_text=True,
                rotation_mode='anchor', va='center', fontsize=18)
        ax.set_xlabel(r'$\log_{10}(M_*/M_\odot)$', fontsize=26, labelpad=13)
        ax.set_ylabel(r'$\log_{10}(\mathrm{SFR}/M_\odot\,\mathrm{yr}^{-1})$', fontsize=26, labelpad=13)
        colorbar = fig.colorbar(hb, ax=ax, fraction=.046, pad=.04)
        colorbar.set_label(r'$N_{\mathrm{Gal}}$', fontsize=25, labelpad=13)
        colorbar.ax.tick_params(labelsize=18)
        if vmax is not None and np.max(hb.get_array()) > vmax:
            print(f'[colour] Counts above {vmax:g} use the darkest colour')
        print(f'[sample] Selected={len(x):,}; plotted={int(visible.sum()):,}; max hexagon={int(np.max(hb.get_array())):,}')
    return fig


def main():
    args = parse_args()
    x, y = load_mass_sfr(args)
    fig = make_figure(x, y, args.gridsize, args.vmax, not args.no_tex)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        fig.savefig(args.output, dpi=args.dpi, bbox_inches='tight', facecolor='white')
    finally:
        plt.close(fig)
    print(f'[saved] {args.output}')


if __name__ == '__main__':
    main()