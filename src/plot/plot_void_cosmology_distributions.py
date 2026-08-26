#!/usr/bin/env python3
"""Compare fiducial consensus-void distributions with catalogue dispersion.

NGC and SGC are combined for each tracer.  The fiducial cosmology is drawn as
a line.  A filled one-sigma band is calculated bin by bin from the normalized
densities of the two bracketing cosmologies, fiber-assignment mocks, and
complete-target mocks.
"""

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/astra-desi-matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/astra-desi-cache")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np

DEFAULT_INPUT_ROOT = Path("/pscratch/sd/v/vtorresg/void_catalog")
DEFAULT_OUTPUT_DIR = Path("plots/void_cosmology_distributions")
COSMOLOGIES = {0.301: "DR2_Om_1_Om0p301_h0p6736",
               0.315: "DR2_Om_2_Om0p315_h0p6736",
               0.329: "DR2_Om_3_Om0p329_h0p6736"}
FIDUCIAL_OMEGA_M = 0.315
ALTERNATIVE_CATALOGS = {r"$\Omega_{\mathrm{m}}=0.301$": "DR2_Om_1_Om0p301_h0p6736",
                        r"$\Omega_{\mathrm{m}}=0.329$": "DR2_Om_3_Om0p329_h0p6736",
                        "fiber assignment": "fiber_assignment/altmtl",
                        "complete mocks": "complete_targets/complete"}
TRACERS = ("BGS", "LRG", "ELG", "QSO")
CAPS = ("NGC", "SGC")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--bins", type=int, default=36)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument(
        "--no-tex",
        action="store_true",
        help="Use Matplotlib mathtext instead of an external LaTeX installation.",
    )
    return parser.parse_args(argv)


def load_consensus(input_root, directory, tracer):
    arrays = []
    for cap in CAPS:
        path = (input_root / directory / "consensus" / f"voids_{tracer}_{cap}_n100.npy")
        if not path.is_file():
            raise FileNotFoundError(f"Missing catalogue: {path}")
        data = np.load(path, allow_pickle=False)
        missing = {"ELLIP", "R_EFF"}.difference(data.dtype.names or ())
        if missing:
            raise KeyError(f"{path} is missing columns: {sorted(missing)}")
        arrays.append(data)
    return np.concatenate(arrays)


def load_samples(input_root):
    samples = {}
    fiducial_directory = COSMOLOGIES[FIDUCIAL_OMEGA_M]
    for tracer in TRACERS:
        samples[("fiducial", tracer)] = load_consensus(input_root, fiducial_directory,
                                                       tracer)
        for label, directory in ALTERNATIVE_CATALOGS.items():
            samples[(label, tracer)] = load_consensus(input_root, directory, tracer)
    return samples


def finite_values(sample, column):
    values = np.asarray(sample[column], dtype=np.float64)
    return values[np.isfinite(values)]


def nice_upper(value, step):
    return step * np.ceil(float(value) / step)


def common_edges(samples, tracer, column, bins):
    pooled = np.concatenate([finite_values(samples[(sample_name, tracer)], column)
                             for sample_name in ("fiducial", *ALTERNATIVE_CATALOGS)])
    if column == "ELLIP":
        # Only a handful of extreme objects lie above 0.65; retaining this
        # fixed range makes the four tracer panels directly comparable.
        lower, upper = 0.0, 0.65
    else:
        lower = 0.0
        upper = nice_upper(np.max(pooled), 10.0)
    return np.linspace(lower, upper, int(bins) + 1)


def normalized_pdf(values, edges):
    """Return the normalized density used by ``numpy.histogram``."""
    pdf, _ = np.histogram(values, bins=edges, density=True)
    return pdf


def configure_style(use_tex):
    plt.rcParams.update({"text.usetex": bool(use_tex),
                         "font.family": "serif",
                         "font.size": 11,
                         "axes.labelsize": 12,
                         "axes.titlesize": 13,
                         "legend.fontsize": 10,
                         "axes.facecolor": "white",
                         "figure.facecolor": "white",
                         "savefig.facecolor": "white",
                         "savefig.edgecolor": "white",
                         "axes.edgecolor": "black",
                         "axes.linewidth": 0.8,
                         "xtick.direction": "in",
                         "ytick.direction": "in",
                         "xtick.top": True,
                         "ytick.right": True})


def plot_variable(samples, column, output_stem, bins, dpi):
    line_color = "#1f4e79"
    band_color = "#78a6c8"
    figure, axes = plt.subplots(1,
                                4,
                                figsize=(14.2, 3.45),
                                squeeze=False,
                                sharex=False,
                                sharey=True)
    axes = axes[0]

    for ax, tracer in zip(axes, TRACERS):
        edges = common_edges(samples, tracer, column, bins)
        centers = 0.5 * (edges[1:] + edges[:-1])
        fiducial_pdf = normalized_pdf(
            finite_values(samples[("fiducial", tracer)], column), edges)
        alternatives = np.vstack([
            normalized_pdf(finite_values(samples[(sample_name, tracer)], column), edges)
            for sample_name in ALTERNATIVE_CATALOGS])
        # Sample standard deviation of the four alternative normalized PDFs.
        # The requested uncertainty is displayed around the fiducial curve.
        sigma = np.std(alternatives, axis=0, ddof=1)
        lower = np.clip(fiducial_pdf - sigma, 0.0, None)
        upper = fiducial_pdf + sigma

        if column == "R_EFF":
            positive = np.concatenate(
                (fiducial_pdf[fiducial_pdf > 0.0], upper[upper > 0.0]))
            log_floor = 0.5 * np.min(positive)
            plotted_lower = np.maximum(lower, log_floor)
        else:
            plotted_lower = lower

        ax.fill_between(centers,
                        plotted_lower,
                        upper,
                        step="mid",
                        color=band_color,
                        alpha=0.48,
                        linewidth=0.0,
                        zorder=1,)
        ax.stairs(fiducial_pdf,
                  edges,
                  color=line_color,
                  linewidth=1.8,
                  zorder=2,)
        ax.set_title(rf"$\mathrm{{{tracer}}}$")
        ax.set_xlim(edges[0], edges[-1])
        if column == "R_EFF":
            ax.set_yscale("log")
            ax.set_ylim(bottom=log_floor)
        else:
            ax.set_ylim(bottom=0.0)
        ax.grid(True, color="0.88", linewidth=0.55, alpha=0.75)
        ax.set_axisbelow(True)

    if column == "ELLIP":
        xlabel = r"$\epsilon$"
        output_name = "void_ellipticity_cosmology"
    else:
        xlabel = r"$R_{\mathrm{eff}}\,[h^{-1}\,\mathrm{Mpc}]$"
        output_name = "void_effective_radius_cosmology"

    for ax in axes:
        ax.set_xlabel(xlabel)
    axes[0].set_ylabel(r"$\mathrm{Normalized\ density}$")

    handles = [Line2D([0], [0],
                      color=line_color,
                      linewidth=1.8,
                      label=r"$\Omega_{\mathrm{m}}=0.315$ (fiducial)"),
               Patch(facecolor=band_color,
                     edgecolor="none",
                     alpha=0.48,
                     label=(r"$\pm1\sigma$: $\Omega_{\mathrm{m}}=0.301,0.329$ + "
                            r"fiber/complete mocks")),]
    figure.legend(handles=handles,
                  loc="upper center",
                  ncol=2,
                  frameon=False,
                  bbox_to_anchor=(0.5, 1.02),
                  handlelength=2.8,)
    figure.subplots_adjust(left=0.062, right=0.992, bottom=0.18, top=0.79, wspace=0.10)

    output_stem.mkdir(parents=True, exist_ok=True)
    written = []
    for suffix in ("png", "pdf"):
        output_path = output_stem / f"{output_name}.{suffix}"
        figure.savefig(output_path,
                       dpi=dpi,
                       bbox_inches="tight",
                       facecolor="white",
                       transparent=False)
        written.append(output_path)
    plt.close(figure)
    return written


def main(argv=None):
    args = parse_args(argv)
    if args.bins < 1:
        raise ValueError("--bins must be positive")
    configure_style(use_tex=not args.no_tex)
    samples = load_samples(args.input_root)
    outputs = []
    outputs.extend(plot_variable(samples, "ELLIP", args.output_dir, args.bins,
                                 args.dpi))
    outputs.extend(plot_variable(samples, "R_EFF", args.output_dir, args.bins,
                                 args.dpi))
    for output in outputs:
        print(f"Wrote {output}")


if __name__ == "__main__":
    main()
