import argparse
import csv
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/astra-desi-matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/astra-desi-cache")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
from astropy.io import fits
from scipy.special import gammaln

DEFAULT_INPUT_ROOT = Path("/pscratch/sd/v/vtorresg/void_catalog_dr2_new")
DEFAULT_OUTPUT_DIR = Path("plots/astra_systematics_bayes")
TRACERS = ("BGS", "LRG", "ELG", "QSO")
CAPS = ("NGC", "SGC")
CATALOGUES = {"fiber": "fiber_assignment",
              "complete": "complete_targets",
              "omega_low": "low_omega",
              "omega_high": "high_omega"}
COMPARISONS = (("fiber assignment", "fiber", "complete"), (r"$\Omega_{\mathrm{m}}$",
                                                           "omega_low", "omega_high"))
OBSERVABLES = (
    ("R_EFF", r"$R_{\mathrm{eff}}$", "#1f77b4"),
    ("ELLIP", r"$\epsilon$", "#ff7f0e"),
    ("ALS", "A.L.S.", "#d62728"),
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--bins", type=int, default=10)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=24680)
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args(argv)


def catalogue_path(input_root, catalogue, tracer, cap):
    return (input_root / CATALOGUES[catalogue] / f"voids_{tracer}_{cap}.fits")


def available_common_caps(input_root, first, second, tracer):
    """Return caps for which both catalogues in a comparison exist."""
    return tuple(cap for cap in CAPS
                 if catalogue_path(input_root, first, tracer, cap).is_file()
                 and catalogue_path(input_root, second, tracer, cap).is_file())


def _als_from_table(sample, path):
    required = ("X", "Y", "Z", "EIGVEC_1_X", "EIGVEC_1_Y", "EIGVEC_1_Z")
    missing = [name for name in required if name not in sample.names]
    if missing:
        raise KeyError(f"{path} has no {', '.join(missing)} column(s)")

    center = np.column_stack(
        tuple(np.asarray(sample[name], dtype=np.float64) for name in ("X", "Y", "Z")))
    major_axis = np.column_stack(
        tuple(np.asarray(sample[name], dtype=np.float64)
              for name in ("EIGVEC_1_X", "EIGVEC_1_Y", "EIGVEC_1_Z")))
    denominator = (np.linalg.norm(center, axis=1) * np.linalg.norm(major_axis, axis=1))
    cosine = np.full(len(sample), np.nan, dtype=np.float64)
    valid = denominator > 0.0
    cosine[valid] = np.abs(
        np.einsum("ij,ij->i", center[valid], major_axis[valid]) / denominator[valid])
    return np.arccos(np.clip(cosine, 0.0, 1.0))


def load_observable(input_root, catalogue, tracer, column, caps):
    """Load and combine finite values from the requested sky caps."""
    pieces = []
    for cap in caps:
        path = catalogue_path(input_root, catalogue, tracer, cap)
        if not path.is_file():
            raise FileNotFoundError(f"Missing catalogue: {path}")
        with fits.open(path, memmap=True) as hdul:
            sample = hdul[1].data
            if column == "ALS":
                values = _als_from_table(sample, path)
            else:
                if column not in sample.names:
                    raise KeyError(f"{path} has no {column} column")
                values = np.asarray(sample[column], dtype=np.float64)
        pieces.append(values[np.isfinite(values)])
    if not pieces:
        raise ValueError(f"No common caps for {catalogue}/{tracer}")
    combined = np.concatenate(pieces)
    if combined.size == 0:
        raise ValueError(f"No finite {column} values for {catalogue}/{tracer}")
    return combined


def common_quantile_edges(samples, bins):
    """Return common, approximately equal-occupancy histogram edges."""
    pooled = np.concatenate(tuple(samples))
    edges = np.quantile(pooled, np.linspace(0.0, 1.0, bins + 1))
    edges = np.unique(edges)
    if edges.size != bins + 1:
        raise ValueError(f"Requested {bins} bins but repeated quantiles leave only "
                         f"{edges.size - 1}; reduce --bins")
    # Include values equal to the sample extrema despite floating-point
    # roundoff in histogram's half-open intervals.
    edges[0] = np.nextafter(edges[0], -np.inf)
    edges[-1] = np.nextafter(edges[-1], np.inf)
    return edges


def log_dirichlet_multinomial_evidence(counts, alpha=0.5):
    """Log evidence for an ordered categorical sample with Dirichlet prior."""
    counts = np.asarray(counts, dtype=np.int64)
    if counts.ndim != 1 or np.any(counts < 0):
        raise ValueError("counts must be a one-dimensional non-negative array")
    n_categories = counts.size
    return (gammaln(n_categories * alpha) -
            gammaln(n_categories * alpha + counts.sum()) +
            np.sum(gammaln(alpha + counts) - gammaln(alpha)))


def log_bayes_same_over_different(counts_a, counts_b):
    """Return ln evidence(common distribution / separate distributions)."""
    counts_a = np.asarray(counts_a, dtype=np.int64)
    counts_b = np.asarray(counts_b, dtype=np.int64)
    if counts_a.shape != counts_b.shape:
        raise ValueError("The two histograms must have the same shape")
    return (log_dirichlet_multinomial_evidence(counts_a + counts_b) -
            log_dirichlet_multinomial_evidence(counts_a) -
            log_dirichlet_multinomial_evidence(counts_b))


def bootstrap_log_bayes(counts_a, counts_b, n_bootstrap, rng):
    """Bootstrap a binned Bayes factor using multinomial resampling."""
    counts_a = np.asarray(counts_a, dtype=np.int64)
    counts_b = np.asarray(counts_b, dtype=np.int64)
    probability_a = counts_a / counts_a.sum()
    probability_b = counts_b / counts_b.sum()
    draws = np.empty(n_bootstrap, dtype=np.float64)
    for index in range(n_bootstrap):
        sample_a = rng.multinomial(int(counts_a.sum()), probability_a)
        sample_b = rng.multinomial(int(counts_b.sum()), probability_b)
        draws[index] = log_bayes_same_over_different(sample_a, sample_b)
    lower, center, upper = np.percentile(draws, (16.0, 50.0, 84.0))
    return float(center), float(center - lower), float(upper - center)


def calculate_results(input_root, bins, n_bootstrap, seed):
    rng = np.random.default_rng(seed)
    rows = []
    for tracer in TRACERS:
        for column, quantity_label, _ in OBSERVABLES:
            for comparison_label, first, second in COMPARISONS:
                caps = available_common_caps(input_root, first, second, tracer)
                if not caps:
                    rows.append({"tracer": tracer,
                                 "systematic": comparison_label,
                                 "catalogue_a": first,
                                 "catalogue_b": second,
                                 "quantity": column,
                                 "quantity_label": quantity_label,
                                 "log_bayes": np.nan,
                                 "err_minus": np.nan,
                                 "err_plus": np.nan,
                                 "plugin_log_bayes": np.nan,
                                 "n_a": 0,
                                 "n_b": 0,
                                 "bins": bins,
                                 "caps": "",
                                 "note": "no common sky caps available"})
                    continue

                sample_a = load_observable(input_root, first, tracer, column, caps)
                sample_b = load_observable(input_root, second, tracer, column, caps)
                edges = common_quantile_edges((sample_a, sample_b), bins)
                counts_a, _ = np.histogram(sample_a, bins=edges)
                counts_b, _ = np.histogram(sample_b, bins=edges)
                plugin_value = log_bayes_same_over_different(counts_a, counts_b)
                center, err_minus, err_plus = bootstrap_log_bayes(
                    counts_a, counts_b, n_bootstrap, rng)
                rows.append({"tracer":
                             tracer,
                             "systematic":
                             comparison_label,
                             "catalogue_a":
                             first,
                             "catalogue_b":
                             second,
                             "quantity":
                             column,
                             "quantity_label":
                             quantity_label,
                             "log_bayes":
                             center,
                             "err_minus":
                             err_minus,
                             "err_plus":
                             err_plus,
                             "plugin_log_bayes":
                             float(plugin_value),
                             "n_a":
                             int(counts_a.sum()),
                             "n_b":
                             int(counts_b.sum()),
                             "bins":
                             bins,
                             "caps":
                             "+".join(caps),
                             "note": ("" if len(caps) == len(CAPS) else
                                      f"provisional: {'+'.join(caps)} only"),})
    return rows


def _lookup(rows, tracer, systematic, quantity):
    matches = [row for row in rows if row["tracer"] == tracer
               and row["systematic"] == systematic and row["quantity"] == quantity]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one result for {tracer}/{systematic}/{quantity}")
    return matches[0]


def configure_style():
    plt.rcParams.update({"text.usetex": False,
                         "font.family": "serif",
                         "font.size": 10.5,
                         "axes.labelsize": 12,
                         "axes.titlesize": 14,
                         "legend.fontsize": 10.5,
                         "axes.facecolor": "white",
                         "figure.facecolor": "white",
                         "savefig.facecolor": "white",
                         "axes.edgecolor": "0.2",
                         "axes.linewidth": 0.9,
                         "xtick.direction": "in",
                         "ytick.direction": "in",
                         "ytick.right": True})


def plot_results(rows, output_dir, dpi):
    configure_style()
    figure, axes = plt.subplots(1, 4, figsize=(13.6, 4.25), sharey=True, squeeze=False)
    axes = axes[0]
    x = np.arange(len(COMPARISONS), dtype=np.float64)
    width = 0.24
    offsets = (-width, 0.0, width)

    finite_rows = [row for row in rows if np.isfinite(row["log_bayes"])]
    y_min = min(row["log_bayes"] - row["err_minus"] for row in finite_rows)
    y_max = max(row["log_bayes"] + row["err_plus"] for row in finite_rows)
    span = max(y_max - y_min, 1.0)
    lower_limit = min(0.0, y_min - 0.10 * span)
    upper_limit = max(0.0, y_max + 0.16 * span)

    for axis, tracer in zip(axes, TRACERS):
        for offset, (quantity, _, color) in zip(offsets, OBSERVABLES):
            for group, (systematic, _, _) in enumerate(COMPARISONS):
                row = _lookup(rows, tracer, systematic, quantity)
                location = x[group] + offset
                if np.isfinite(row["log_bayes"]):
                    axis.bar(location,
                             row["log_bayes"],
                             width=width,
                             color=color,
                             edgecolor="black",
                             linewidth=0.75,
                             yerr=np.array([[row["err_minus"]], [row["err_plus"]]]),
                             error_kw={"elinewidth": 0.8,
                                       "capsize": 2.2,
                                       "capthick": 0.8},
                             zorder=3,)
                else:
                    # An outlined marker reserves an unavailable position
                    # without implying a numerical Bayes factor of zero.
                    marker_height = 0.035 * (upper_limit - lower_limit)
                    marker_bottom = -0.5 * marker_height
                    axis.bar(location,
                             marker_height,
                             bottom=marker_bottom,
                             width=width,
                             facecolor="white",
                             edgecolor=color,
                             hatch="///",
                             linewidth=0.9,
                             zorder=3,)
                    axis.annotate("N/D",
                                  (location, marker_bottom + marker_height),
                                  xytext=(0, 3),
                                  textcoords="offset points",
                                  ha="center",
                                  va="bottom",
                                  fontsize=7.5,
                                  color="0.35",
                                  rotation=90,)

        axis.axhline(0.0, color="black", linewidth=0.9, zorder=2)
        used_caps = {row["caps"]
                     for row in rows if row["tracer"] == tracer and row["caps"]}
        title = tracer
        if used_caps == {"SGC"}:
            title += "\n(SGC only; provisional)"
        axis.set_title(title)
        axis.set_xticks(x)
        axis.set_xticklabels(("fiber\nassignment", r"$\Omega_{\mathrm{m}}$"))
        axis.set_xlim(-0.55, len(COMPARISONS) - 0.45)
        axis.set_ylim(lower_limit, upper_limit)
        axis.grid(axis="y", color="0.72", linestyle=":", linewidth=0.8)
        axis.set_axisbelow(True)

    axes[0].set_ylabel('log (Bayes Factor)')
    legend_handles = [
        Patch(facecolor=color, edgecolor="black", linewidth=0.75, label=label)
        for _, label, color in OBSERVABLES[:2]]
    legend_handles.append(Patch(facecolor=OBSERVABLES[2][2],
                                edgecolor="black",
                                linewidth=0.75,
                                label="A.L.S."))
    # legend_handles.append(
    #     Patch(facecolor="white", edgecolor="0.35", hatch="///",
    #           linewidth=0.9, label="N/D (catálogo pendiente)"))
    figure.legend(handles=legend_handles,
                  loc="upper center",
                  ncol=4,
                  frameon=True,
                  bbox_to_anchor=(0.5, 1.015),
                  columnspacing=1.5,
                  handlelength=1.8,)
    figure.subplots_adjust(left=0.065, right=0.992, bottom=0.17, top=0.80, wspace=0.08)

    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = []
    for suffix in ("png", "pdf"):
        output = output_dir / f"astra_systematics_bayes.{suffix}"
        figure.savefig(output, dpi=dpi, bbox_inches="tight")
        outputs.append(output)
    plt.close(figure)
    return outputs


def write_results(rows, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / "astra_systematics_bayes_values.csv"
    fieldnames = ("tracer", "systematic", "catalogue_a", "catalogue_b", "quantity",
                  "log_bayes", "err_minus", "err_plus", "plugin_log_bayes", "n_a",
                  "n_b", "bins", "note", "caps")
    with output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return output


def main(argv=None):
    args = parse_args(argv)
    if args.bins < 2:
        raise ValueError("--bins must be at least 2")
    if args.bootstrap < 2:
        raise ValueError("--bootstrap must be at least 2")
    rows = calculate_results(args.input_root, args.bins, args.bootstrap, args.seed)
    outputs = plot_results(rows, args.output_dir, args.dpi)
    outputs.append(write_results(rows, args.output_dir))
    for output in outputs:
        print(f"Wrote {output}")


if __name__ == "__main__":
    main()
