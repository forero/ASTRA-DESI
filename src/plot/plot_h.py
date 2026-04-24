import os, re, glob, argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
plt.style.use('dark_background')


def setup_style():
    plt.rcParams.update({'grid.linewidth': 0.3,
                         'text.usetex': True})


def histogram_to_pdf_from_samples(H, bin_edges):
    H = np.asarray(H, dtype=float)
    H = H[np.isfinite(H)]
    counts, _ = np.histogram(H, bins=bin_edges)
    widths = np.diff(bin_edges)
    total = counts.sum()

    if total <= 0:
        pdf = np.full(len(widths), np.nan, dtype=float)
    else:
        pdf = counts / total / widths

    centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    return centers, pdf


def discover_file(summary_dir, tracer, zone):
    summary_dir = Path(summary_dir)
    tracer = str(tracer).lower()
    zone = str(zone).lower()

    patterns = [str(summary_dir / f'{tracer}_{zone}_entropy_from_r_classification.npz'),
                str(summary_dir / f'{tracer}_{zone}_entropy_from_classification.npz'),
                str(summary_dir / f'{tracer}_*{zone}*_entropy_from_r_classification.npz'),
                str(summary_dir / f'{tracer}_*{zone}*_entropy_from_classification.npz')]

    matches = []
    for pat in patterns:
        matches.extend(glob.glob(pat))

    matches = sorted(set(matches))
    return matches[0] if matches else None


def discover_files_for_tracer(summary_dir, tracer, zones=None):
    tracer = str(tracer).lower()
    summary_dir = Path(summary_dir)

    out = {}
    if zones is not None:
        for zone in zones:
            path = discover_file(summary_dir, tracer, zone)
            if path is not None:
                out[str(zone).upper()] = path
        return out

    patterns = [str(summary_dir / f'{tracer}_*_entropy_from_r_classification.npz'),
                str(summary_dir / f'{tracer}_*_entropy_from_classification.npz')]
    files = []
    for pat in patterns:
        files.extend(glob.glob(pat))

    files = sorted(set(files))
    for path in files:
        name = os.path.basename(path)
        m = re.match(rf'{re.escape(tracer)}_(.+?)_entropy_from(?:_r)?_classification\.npz$', name)
        if not m:
            continue
        zone = str(m.group(1)).upper()
        if zone not in out:
            out[zone] = path
    return out


def load_H_obj(path):
    d = np.load(path, allow_pickle=True)
    H = d['H_obj']
    H = H[np.isfinite(H)]
    return H


def plot_joint(summary_dir, outdir, bins=35, xmin=0.0, xmax=0.56, zones=None):
    colors = {'BGS': 'deepskyblue',
              'LRG': 'green',
              'ELG': 'darkorange',
              'QSO': 'crimson'}

    tracers = ['BGS', 'LRG', 'ELG', 'QSO']

    bin_edges = np.linspace(xmin, xmax, bins + 1)

    fig, ax = plt.subplots()
    ax.grid(lw=0.3)

    plotted = False

    for tracer in tracers:
        zone_files = discover_files_for_tracer(summary_dir, tracer, zones=zones)
        if len(zone_files) == 0:
            continue

        color = colors[tracer]
        zone_pdfs = []
        zone_labels = []
        xvals = None

        for zone_label, path in sorted(zone_files.items()):
            H = load_H_obj(path)
            x, pdf = histogram_to_pdf_from_samples(H, bin_edges)
            xvals = x
            zone_pdfs.append(pdf)
            zone_labels.append(zone_label)

        if len(zone_pdfs) > 1:
            Y = np.vstack(zone_pdfs)
            pdf_mean = np.nanmean(Y, axis=0)
            pdf_std = np.nanstd(Y, axis=0, ddof=1) if Y.shape[0] > 1 else np.zeros_like(pdf_mean)

            ax.fill_between(xvals, np.clip(pdf_mean - pdf_std, 0.0, None), pdf_mean + pdf_std,
                            color=color, alpha=0.35, zorder=2,
                            label=rf'{tracer} $\pm 1\sigma_{{\rm zone}}$')

            ax.plot(xvals, pdf_mean, color=color, lw=1.6, zorder=4,
                    label=rf'{tracer} mean')
        else:
            pdf = zone_pdfs[0]
            zone = zone_labels[0]

            ax.plot(xvals, pdf, color=color, lw=2.4, zorder=4,
                    label=rf'{tracer} {zone}')

        plotted = True

    if not plotted:
        raise RuntimeError()

    ax.set_xlim(xmin, xmax)
    ax.set_xlabel(r'$H$', fontsize=13)
    ax.set_ylabel(r'PDF', fontsize=13)
    ax.tick_params(axis='both', labelsize=13)

    leg = ax.legend(# loc='lower center',
                    # bbox_to_anchor=(0.5, -0.12),
                    ncol=1,
                    # frameon=True,
                    # fancybox=True
                    )

    fig.tight_layout()
    outpath = Path(outdir) / 'joint_entropy_pdf_style.png'
    fig.savefig(outpath, dpi=360, bbox_inches='tight')
    plt.close(fig)

    print(f'{outpath}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--summary-dir', required=True)
    parser.add_argument('--outdir', required=True)
    parser.add_argument('--zones', nargs='*', default=None)
    parser.add_argument('--bins', type=int, default=35)
    parser.add_argument('--xmin', type=float, default=0.0)
    parser.add_argument('--xmax', type=float, default=1.0)
    args = parser.parse_args()

    setup_style()
    os.makedirs(args.outdir, exist_ok=True)

    plot_joint(summary_dir=args.summary_dir,
               outdir=args.outdir,
               zones=args.zones,
               bins=args.bins,
               xmin=args.xmin,
               xmax=args.xmax)


if __name__ == '__main__':
    main()