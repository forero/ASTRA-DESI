import argparse
import csv
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/astra-desi-matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/astra-desi-cache")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits

DEFAULT_INPUT_ROOT = Path("/pscratch/sd/v/vtorresg/void_catalog_dr2_new")
DEFAULT_OUTPUT_DIR = Path("plots/astra_void_count_comparisons")
TRACERS = ("BGS", "LRG", "ELG", "QSO")
CAPS = ("NGC", "SGC")
TRACER_COLORS = {
    "BGS": "#1f77b4",
    "LRG": "#d62728",
    "ELG": "#17becf",
    "QSO": "#9467bd",}
COMPARISONS = ({"key": "fiber_assignment",
                "label": "fiber\nassignment",
                "catalogue_a": "fiber_assignment",
                "catalogue_b": "complete_targets",
                "label_a": "altMTL",
                "label_b": "no fiber"},
               {"key": "omega_m",
                "label": r"$\Omega_{\mathrm{m}}$",
                "catalogue_a": "low_omega",
                "catalogue_b": "high_omega",
                "label_a": r"$\Omega_{\mathrm{m}}=0.301$",
                "label_b": r"$\Omega_{\mathrm{m}}=0.329$"},)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args(argv)


def catalogue_path(input_root, catalogue, tracer, cap):
    return input_root / catalogue / f"voids_{tracer}_{cap}.fits"


def available_common_caps(input_root, comparison, tracer):
    return tuple(
        cap for cap in CAPS
        if catalogue_path(input_root, comparison["catalogue_a"], tracer, cap).is_file()
        and catalogue_path(input_root, comparison["catalogue_b"], tracer,
                           cap).is_file())


def load_iteration_counts(input_root, catalogue, tracer, caps):
    """Return a mapping of realization ID to count, summed over caps."""
    counts = {}
    for cap in caps:
        path = catalogue_path(input_root, catalogue, tracer, cap)
        if not path.is_file():
            raise FileNotFoundError(f"Missing catalogue: {path}")
        with fits.open(path, memmap=True) as hdul:
            names = hdul[1].columns.names
            if "SRC_ITER" not in names:
                raise KeyError(f"{path} has no SRC_ITER column")
            iterations = np.asarray(hdul[1].data["SRC_ITER"], dtype=np.int64)
            unique, frequency = np.unique(iterations, return_counts=True)
        for iteration, count in zip(unique, frequency):
            key = int(iteration)
            counts[key] = counts.get(key, 0) + int(count)
    if not counts:
        raise ValueError(f"No rows found for {catalogue}/{tracer}")
    return counts


def _aligned_counts(counts_a, counts_b):
    iteration_ids = sorted(set(counts_a) | set(counts_b))
    sample_a = np.asarray([counts_a.get(index, 0) for index in iteration_ids],
                          dtype=float)
    sample_b = np.asarray([counts_b.get(index, 0) for index in iteration_ids],
                          dtype=float)
    return iteration_ids, sample_a, sample_b


def total_and_error(counts):
    """Return total count and empirical error on the scaled mean count."""
    sample = np.asarray(counts, dtype=float)
    total = float(sample.sum())
    if sample.size < 2:
        return total, float(np.sqrt(total))
    error = float(np.std(sample, ddof=1) * np.sqrt(sample.size))
    return total, error


def relative_change_and_error(sample_a, sample_b):
    """Return N_a/N_b-1 and its paired realization standard error."""
    total_b = sample_b.sum()
    if total_b <= 0.0:
        return np.nan, np.nan
    relative = float(sample_a.sum() / total_b - 1.0)
    valid = sample_b > 0.0
    realization_values = sample_a[valid] / sample_b[valid] - 1.0
    if realization_values.size < 2:
        return relative, np.nan
    error = float(np.std(realization_values, ddof=1) / np.sqrt(realization_values.size))
    return relative, error


def calculate_results(input_root):
    rows = []
    for tracer in TRACERS:
        for comparison in COMPARISONS:
            caps = available_common_caps(input_root, comparison, tracer)
            base = {"tracer": tracer,
                    "systematic": comparison["key"],
                    "systematic_label": comparison["label"],
                    "catalogue_a": comparison["catalogue_a"],
                    "catalogue_b": comparison["catalogue_b"],
                    "label_a": comparison["label_a"],
                    "label_b": comparison["label_b"],
                    "caps": "+".join(caps)}
            if not caps:
                rows.append({**base,
                             "n_iterations": 0,
                             "count_a": np.nan,
                             "count_a_error": np.nan,
                             "count_b": np.nan,
                             "count_b_error": np.nan,
                             "relative_change": np.nan,
                             "relative_change_error": np.nan,
                             "note": "no common sky caps available",})
                continue

            counts_a = load_iteration_counts(input_root, comparison["catalogue_a"],
                                             tracer, caps)
            counts_b = load_iteration_counts(input_root, comparison["catalogue_b"],
                                             tracer, caps)
            iterations, sample_a, sample_b = _aligned_counts(counts_a, counts_b)
            count_a, error_a = total_and_error(sample_a)
            count_b, error_b = total_and_error(sample_b)
            relative, relative_error = relative_change_and_error(sample_a, sample_b)
            rows.append({**base,
                         "n_iterations":
                         len(iterations),
                         "count_a":
                         int(count_a),
                         "count_a_error":
                         error_a,
                         "count_b":
                         int(count_b),
                         "count_b_error":
                         error_b,
                         "relative_change":
                         relative,
                         "relative_change_error":
                         relative_error,
                         "note": ("" if len(caps) == len(CAPS) else
                                  f"provisional: {'+'.join(caps)} only"),})
    return rows


def configure_style():
    plt.rcParams.update({"text.usetex": False,
                         "font.family": "serif",
                         "font.size": 11,
                         "axes.labelsize": 12,
                         "axes.titlesize": 14,
                         "legend.fontsize": 10.5,
                         "axes.edgecolor": "0.2",
                         "axes.linewidth": 0.9,
                         "xtick.direction": "in",
                         "ytick.direction": "in",
                         "ytick.right": True,
                         "figure.facecolor": "white",
                         "axes.facecolor": "white",
                         "savefig.facecolor": "white"})


def _rows_for_tracer(rows, tracer):
    selected = [row for row in rows if row["tracer"] == tracer]
    if len(selected) != len(COMPARISONS):
        raise RuntimeError(f"Incomplete result set for {tracer}")
    return selected


def save_figure(figure, output_dir, stem, dpi):
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = []
    for suffix in ("png", "pdf"):
        output = output_dir / f"{stem}.{suffix}"
        figure.savefig(output, dpi=dpi, bbox_inches="tight")
        outputs.append(output)
    plt.close(figure)
    return outputs


def plot_all_tracers_counts(rows, output_dir, dpi):
    """Plot one shared-y count panel for each tracer."""
    figure, axes = plt.subplots(1,
                                len(TRACERS),
                                figsize=(14.2, 4.4),
                                sharey=True,
                                squeeze=False)
    axes = axes[0]
    centers = np.arange(len(COMPARISONS), dtype=float)
    width = 0.34
    colors = ("#1f77b4", "#ff7f0e")

    for axis, tracer in zip(axes, TRACERS):
        tracer_rows = _rows_for_tracer(rows, tracer)
        for index, row in enumerate(tracer_rows):
            if np.isfinite(row["count_a"]):
                for offset, key, error_key, color in ((-width / 2, "count_a",
                                                       "count_a_error", colors[0]),
                                                      (width / 2, "count_b",
                                                       "count_b_error", colors[1])):
                    axis.bar(centers[index] + offset,
                             row[key],
                             width=width,
                             yerr=row[error_key],
                             color=color,
                             alpha=0.88,
                             edgecolor="black",
                             linewidth=0.8,
                             capsize=2.5,
                             zorder=3,)
            else:
                axis.text(centers[index],
                          0.5,
                          "N/D",
                          transform=axis.get_xaxis_transform(),
                          ha="center",
                          va="center",
                          color="0.4",
                          fontsize=10,)

        used_caps = {row["caps"] for row in tracer_rows if row["caps"]}
        title = tracer
        if used_caps == {"SGC"}:
            title += "\n(SGC only; provisional)"
        axis.set_title(title)
        bar_locations = []
        bar_labels = []
        for index, row in enumerate(tracer_rows):
            bar_locations.extend(
                (centers[index] - width / 2, centers[index] + width / 2))
            bar_labels.extend((row["label_a"], row["label_b"]))
        axis.set_xticks(bar_locations)
        axis.set_xticklabels(bar_labels,
                             rotation=45,
                             ha="right",
                             rotation_mode="anchor")
        axis.grid(axis="y", linestyle=":", linewidth=0.8, color="0.72")
        axis.set_axisbelow(True)

    axes[0].set_ylabel("total void count")
    axes[0].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    figure.suptitle("Void count comparison across tracers", y=0.99)
    figure.subplots_adjust(left=0.065, right=0.995, bottom=0.30, top=0.82, wspace=0.08)
    return save_figure(figure, output_dir, "all_tracers_void_counts", dpi)


def plot_all_tracers_relative(rows, output_dir, dpi):
    figure, axis = plt.subplots(figsize=(7.0, 4.8))
    centers = np.arange(len(COMPARISONS), dtype=float)
    offsets = np.linspace(-0.24, 0.24, len(TRACERS))
    row_lookup = {(row["tracer"], row["systematic"]): row for row in rows}
    for offset, tracer in zip(offsets, TRACERS):
        values = []
        errors = []
        for comparison in COMPARISONS:
            row = row_lookup[(tracer, comparison["key"])]
            values.append(row["relative_change"])
            errors.append(row["relative_change_error"])
        values = np.asarray(values, dtype=float)
        errors = np.asarray(errors, dtype=float)
        finite = np.isfinite(values)
        axis.errorbar(centers[finite] + offset,
                      values[finite],
                      yerr=errors[finite],
                      fmt="o",
                      color=TRACER_COLORS[tracer],
                      label=tracer,
                      markeredgecolor="white",
                      markeredgewidth=0.5,
                      markersize=7,
                      elinewidth=2.,
                      capsize=5,
                      zorder=3,)
    axis.axhline(0.0, color="black", linestyle=":", linewidth=1.3)
    axis.set_xticks(centers)
    axis.set_xticklabels([item["label"] for item in COMPARISONS])
    axis.set_ylabel(r"relative change in void count")
    axis.set_title("Relative void-count change across tracers")
    axis.legend(frameon=True, ncol=2)
    axis.grid(axis="x", color="0.78", linewidth=0.9)
    axis.set_axisbelow(True)
    figure.tight_layout()
    return save_figure(figure, output_dir, "all_tracers_void_count_relative_change",
                       dpi)


def write_results(rows, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / "void_count_comparison_values.csv"
    fieldnames = ("tracer", "systematic", "catalogue_a", "catalogue_b", "caps",
                  "n_iterations", "count_a", "count_a_error", "count_b",
                  "count_b_error", "relative_change", "relative_change_error", "note")
    with output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return output


def main(argv=None):
    args = parse_args(argv)
    configure_style()
    rows = calculate_results(args.input_root)
    outputs = plot_all_tracers_counts(rows, args.output_dir, args.dpi)
    outputs.extend(plot_all_tracers_relative(rows, args.output_dir, args.dpi))
    outputs.append(write_results(rows, args.output_dir))
    for output in outputs:
        print(f"Wrote {output}")


if __name__ == "__main__":
    main()
