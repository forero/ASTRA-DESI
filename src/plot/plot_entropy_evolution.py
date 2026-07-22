"""Plot the evolution of the classification-entropy PDF.

For every requested checkpoint (5, 10, ..., 100 by default), this script
uses the first N classification realizations to estimate, for each object,
the probabilities of being classified as a void, sheet, filament, or knot.
It then computes the normalized Shannon entropy and the PDF over all objects.

The DR2 classification directory may be passed directly; unlike older plot
helpers, the path does not need to be the parent release directory.
"""

import argparse
import csv
import json
import os
import re
from pathlib import Path

import numpy as np


DEFAULT_CLASSIFICATION_DIR = Path(
    "/pscratch/sd/v/vtorresg/cosmic-web/dr2/classification"
)
ENV_NAMES = ("void", "sheet", "filament", "knot")
ITER_RE = re.compile(r"iter(\d+)", flags=re.IGNORECASE)

import matplotlib.pyplot as plt
plt.style.use('dark_background')
plt.rcParams.update({'grid.linewidth': 0.3,
                         'text.usetex': True})


def parse_iteration(path):
    """Return the zero-based realization number encoded in a file name."""
    match = ITER_RE.search(Path(path).name)
    return int(match.group(1)) if match else None


def discover_iteration_files(classification_dir, tracer, zone, max_iterations=100):
    """Discover and validate the first ``max_iterations`` DR2 realizations."""
    classification_dir = Path(classification_dir)
    tracer = str(tracer).strip().upper()
    zone = str(zone).strip().upper()

    search_dirs = [classification_dir / tracer.lower() / zone.lower()]
    # Also accept a path already pointing at the tracer or zone directory.
    if classification_dir.name.lower() == tracer.lower():
        search_dirs.append(classification_dir / zone.lower())
    if classification_dir.name.lower() == zone.lower():
        search_dirs.append(classification_dir)

    matches = []
    patterns = (
        f"zone_{zone}_{tracer}_iter*.fits",
        f"zone_{zone}_{tracer}_iter*.fits.gz",
    )
    for directory in search_dirs:
        for pattern in patterns:
            matches.extend(directory.glob(pattern))

    by_iteration = {}
    for path in sorted(set(matches)):
        iteration = parse_iteration(path)
        if iteration is None:
            continue
        if iteration in by_iteration:
            raise RuntimeError(
                f"Duplicate realization {iteration} for {tracer} {zone}: "
                f"{by_iteration[iteration]} and {path}"
            )
        by_iteration[iteration] = path

    expected = list(range(int(max_iterations)))
    missing = [iteration for iteration in expected if iteration not in by_iteration]
    if missing:
        preview = ", ".join(str(value) for value in missing[:10])
        suffix = "..." if len(missing) > 10 else ""
        raise RuntimeError(
            f"Missing {len(missing)} realization(s) for {tracer} {zone}: "
            f"{preview}{suffix}. Searched below {classification_dir}."
        )

    return [by_iteration[iteration] for iteration in expected]


def _read_with_fitsio(path, columns, rows=None):
    import fitsio

    kwargs = {"ext": 1, "columns": list(columns)}
    if rows is not None:
        kwargs["rows"] = np.asarray(rows, dtype=np.int64)
    table = fitsio.read(str(path), **kwargs)
    return {name: np.asarray(table[name]) for name in columns}


def _read_with_astropy(path, columns, rows=None):
    from astropy.io import fits

    with fits.open(path, memmap=True) as hdul:
        data = hdul[1].data
        if data is None:
            raise ValueError(f"Empty FITS table: {path}")
        if rows is not None:
            data = data[np.asarray(rows, dtype=np.int64)]
        return {name: np.asarray(data[name]) for name in columns}


def read_fits_columns(path, columns, rows=None):
    """Read selected FITS columns, preferring fitsio for large gzip tables."""
    try:
        import fitsio  # noqa: F401
    except ImportError:
        return _read_with_astropy(path, columns, rows=rows)
    return _read_with_fitsio(path, columns, rows=rows)


def data_layout(first_path):
    """Return real-data row indices and their TARGETIDs from one realization."""
    flags = read_fits_columns(first_path, ("ISDATA",))["ISDATA"]
    rows = np.flatnonzero(np.asarray(flags, dtype=bool))
    block = read_fits_columns(first_path, ("TARGETID",), rows=rows)
    targetids = np.asarray(block["TARGETID"], dtype=np.int64)

    if targetids.size == 0:
        raise RuntimeError(f"No ISDATA rows found in {first_path}")
    if np.any(targetids[1:] <= targetids[:-1]):
        raise RuntimeError(
            f"Real-data TARGETIDs are not strictly increasing in {first_path}"
        )
    return rows, targetids


def read_real_classification(path, data_rows, reference_targetids):
    """Read NDATA/NRAND for real objects and verify the invariant row layout."""
    columns = ("TARGETID", "NDATA", "NRAND")
    block = read_fits_columns(path, columns, rows=data_rows)
    targetids = np.asarray(block["TARGETID"], dtype=np.int64)

    if not np.array_equal(targetids, reference_targetids):
        # A changed table layout is unexpected for DR2. Re-read via ISDATA so
        # that a harmless row reordering does not silently corrupt the result.
        full_columns = ("TARGETID", "ISDATA", "NDATA", "NRAND")
        full = read_fits_columns(path, full_columns)
        mask = np.asarray(full["ISDATA"], dtype=bool)
        targetids = np.asarray(full["TARGETID"][mask], dtype=np.int64)
        order = np.argsort(targetids)
        targetids = targetids[order]
        if not np.array_equal(targetids, reference_targetids):
            raise RuntimeError(
                f"The real-object TARGETID set changed in realization {path}"
            )
        ndata = np.asarray(full["NDATA"][mask])[order]
        nrand = np.asarray(full["NRAND"][mask])[order]
        return ndata, nrand

    return np.asarray(block["NDATA"]), np.asarray(block["NRAND"])


def classify_counts(ndata, nrand):
    """Classify NDATA/NRAND ratios using the project's four web thresholds."""
    ndata = np.asarray(ndata, dtype=np.float32)
    nrand = np.asarray(nrand, dtype=np.float32)
    denominator = ndata + nrand
    ratio = np.full(denominator.shape, np.nan, dtype=np.float32)
    valid = np.isfinite(denominator) & (denominator > 0)
    ratio[valid] = (ndata[valid] - nrand[valid]) / denominator[valid]

    environment = np.full(ratio.shape, -1, dtype=np.int8)
    environment[valid & (ratio >= -1.0) & (ratio <= -0.25)] = 0
    environment[valid & (ratio > -0.25) & (ratio <= 0.25)] = 1
    environment[valid & (ratio > 0.25) & (ratio <= 0.65)] = 2
    environment[valid & (ratio > 0.65) & (ratio <= 1.0)] = 3
    return environment


def entropy_from_class_counts(class_counts):
    """Compute H/log2(4) for the four classification frequencies per object."""
    class_counts = np.asarray(class_counts)
    totals = class_counts.sum(axis=1).astype(np.float32)
    entropy = np.full(len(class_counts), np.nan, dtype=np.float32)
    classified = totals > 0
    entropy[classified] = 0.0

    for environment in range(class_counts.shape[1]):
        nonzero = classified & (class_counts[:, environment] > 0)
        probabilities = (
            class_counts[nonzero, environment].astype(np.float32) / totals[nonzero]
        )
        entropy[nonzero] -= probabilities * np.log2(probabilities) / np.log2(4.0)
    # Protect the closed theoretical interval from float32 round-off at H=1.
    entropy[classified] = np.clip(entropy[classified], 0.0, 1.0)
    return entropy


def summarize_entropy(entropy, bin_edges):
    """Return histogram counts and scalar summary statistics."""
    finite = np.asarray(entropy[np.isfinite(entropy)], dtype=np.float64)
    hist_counts, _ = np.histogram(finite, bins=bin_edges)
    if finite.size == 0:
        return hist_counts, 0, np.nan, np.nan
    mean = float(np.mean(finite))
    std = float(np.std(finite, ddof=1)) if finite.size > 1 else np.nan
    return hist_counts, int(finite.size), mean, std


def process_zone(files, checkpoints, bin_edges, zone):
    """Accumulate classifications and entropy histograms for a single zone."""
    data_rows, reference_targetids = data_layout(files[0])
    class_counts = np.zeros((len(reference_targetids), 4), dtype=np.uint16)
    checkpoint_set = set(int(value) for value in checkpoints)

    histograms = []
    object_counts = []
    mean_entropy = []
    std_entropy = []

    for number, path in enumerate(files, start=1):
        ndata, nrand = read_real_classification(
            path, data_rows=data_rows, reference_targetids=reference_targetids
        )
        environment = classify_counts(ndata, nrand)
        valid = environment >= 0
        row_indices = np.flatnonzero(valid)
        class_counts[row_indices, environment[valid]] += 1

        if number not in checkpoint_set:
            continue

        entropy = entropy_from_class_counts(class_counts)
        hist, n_objects, mean, std = summarize_entropy(entropy, bin_edges)
        histograms.append(hist)
        object_counts.append(n_objects)
        mean_entropy.append(mean)
        std_entropy.append(std)
        print(
            f"[{zone}] N_iter={number:3d}: {n_objects:,} objects, "
            f"mean(H)={mean:.6f}",
            flush=True,
        )

    return {
        "hist_counts": np.asarray(histograms, dtype=np.int64),
        "n_objects": np.asarray(object_counts, dtype=np.int64),
        "mean_entropy": np.asarray(mean_entropy, dtype=np.float64),
        "std_entropy": np.asarray(std_entropy, dtype=np.float64),
    }


def counts_to_pdf(hist_counts, bin_edges):
    """Normalize each entropy histogram so that its integral is one."""
    hist_counts = np.asarray(hist_counts, dtype=np.float64)
    widths = np.diff(np.asarray(bin_edges, dtype=np.float64))
    totals = hist_counts.sum(axis=1)
    pdf = np.full(hist_counts.shape, np.nan, dtype=np.float64)
    valid = totals > 0
    pdf[valid] = hist_counts[valid] / totals[valid, None] / widths[None, :]
    return pdf


def combine_zone_results(zone_results, bin_edges):
    """Pool zone histograms and moments, weighted by their object counts."""
    hist_counts = np.sum(
        np.stack([result["hist_counts"] for result in zone_results]), axis=0
    )
    n_objects = np.sum(
        np.stack([result["n_objects"] for result in zone_results]), axis=0
    )

    weighted_sum = np.sum(
        np.stack(
            [
                result["mean_entropy"] * result["n_objects"]
                for result in zone_results
            ]
        ),
        axis=0,
    )
    mean_entropy = np.divide(
        weighted_sum,
        n_objects,
        out=np.full(weighted_sum.shape, np.nan),
        where=n_objects > 0,
    )

    # Reconstruct the pooled sample variance from each zone's n, mean, and
    # sample variance without retaining millions of object-level entropies.
    pooled_ss = np.sum(
        np.stack(
            [
                (result["n_objects"] - 1)
                * np.nan_to_num(result["std_entropy"], nan=0.0) ** 2
                + result["n_objects"] * result["mean_entropy"] ** 2
                for result in zone_results
            ]
        ),
        axis=0,
    )
    variance = np.divide(
        pooled_ss - n_objects * mean_entropy ** 2,
        n_objects - 1,
        out=np.full(pooled_ss.shape, np.nan),
        where=n_objects > 1,
    )
    std_entropy = np.sqrt(np.clip(variance, 0.0, None))
    return {
        "hist_counts": hist_counts,
        "pdf": counts_to_pdf(hist_counts, bin_edges),
        "n_objects": n_objects,
        "mean_entropy": mean_entropy,
        "std_entropy": std_entropy,
    }


def checkpoint_edges(checkpoints):
    """Construct cell edges centered on possibly non-uniform checkpoints."""
    checkpoints = np.asarray(checkpoints, dtype=np.float64)
    if checkpoints.size == 1:
        return np.array([checkpoints[0] - 0.5, checkpoints[0] + 0.5])
    midpoints = 0.5 * (checkpoints[:-1] + checkpoints[1:])
    first = checkpoints[0] - (midpoints[0] - checkpoints[0])
    last = checkpoints[-1] + (checkpoints[-1] - midpoints[-1])
    return np.concatenate(([first], midpoints, [last]))


def iteration_ticks(checkpoints, tick_step=10):
    """Return regularly spaced iteration ticks within the plotted interval."""
    checkpoints = np.asarray(checkpoints, dtype=np.float64)
    first = np.ceil(np.min(checkpoints) / tick_step) * tick_step
    last = np.floor(np.max(checkpoints) / tick_step) * tick_step
    if first > last:
        return checkpoints
    return np.arange(first, last + tick_step, tick_step)


def plot_entropy_pdf(checkpoints, bin_edges, pdf, mean_entropy, tracer, zones,
                     output_path, color_scale="log", dpi=300):
    """Render the entropy-PDF evolution as a heat map."""
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    values = np.asarray(pdf, dtype=np.float64).T
    cmap = plt.get_cmap("viridis").copy()
    norm = None
    colorbar_label = "PDF"
    if color_scale == "log":
        positive = values[np.isfinite(values) & (values > 0)]
        if positive.size:
            vmax = float(np.max(positive))
            vmin = max(float(np.min(positive)), vmax * 1.0e-4)
            norm = LogNorm(vmin=vmin, vmax=vmax)
            values = np.ma.masked_less_equal(values, 0.0)
            cmap.set_bad("white")
            cmap.set_under("white")
            colorbar_label = "PDF (log scale)"

    fig, axis = plt.subplots(figsize=(8.2, 5.4))
    mesh = axis.pcolormesh(
        checkpoint_edges(checkpoints),
        bin_edges,
        values,
        shading="flat",
        cmap=cmap,
        norm=norm,
        rasterized=True,
    )
    axis.plot(
        checkpoints,
        mean_entropy,
        color="white",
        linewidth=2.2,
        marker="o",
        markersize=3.0,
        markeredgecolor="black",
        markeredgewidth=0.35,
        label=r"Mean $H$",
    )
    axis.set_xlim(checkpoint_edges(checkpoints)[[0, -1]])
    axis.set_ylim(bin_edges[[0, -1]])
    axis.set_xticks(iteration_ticks(checkpoints))
    # axis.tick_params(axis="x", labelrotation=45)
    axis.set_xlabel(r"Number of iterations $N_{\rm iter}$")
    axis.set_ylabel(r"Normalized entropy $H$")
    zone_label = "+".join(zones)
    # axis.set_title(f"{tracer} ({zone_label}): evolution of the entropy PDF")
    axis.legend(loc="upper right", frameon=True)
    colorbar = fig.colorbar(mesh, ax=axis, pad=0.02)
    colorbar.set_label(colorbar_label)
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_mean_entropy(checkpoints, mean_entropy, std_entropy, tracer, zones,
                      output_path, dpi=300):
    """Plot the mean classification entropy as a function of realizations."""
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    checkpoints = np.asarray(checkpoints)
    mean_entropy = np.asarray(mean_entropy, dtype=np.float64)
    std_entropy = np.asarray(std_entropy, dtype=np.float64)
    lower = np.clip(mean_entropy - std_entropy, 0.0, 1.0)
    upper = np.clip(mean_entropy + std_entropy, 0.0, 1.0)

    fig, axis = plt.subplots(figsize=(7,5))
    axis.fill_between(
        checkpoints,
        lower,
        upper,
        # color="royalblue",
        alpha=0.22,
        linewidth=0.0,
        label=r"$\pm 1\sigma$",
    )
    axis.plot(
        checkpoints,
        mean_entropy,
        # color="royalblue",
        linewidth=2.4,
        marker="o",
        markersize=4.5,
        # markerfacecolor="white",
        # markeredgewidth=1.2,
        label=r"Mean $H$",
    )
    axis.axhline(
        0.45,
        color="white",
        linestyle=":",
        linewidth=1.5,
        label=r"$H=0.45$",
    )

    finite_limits = np.concatenate((lower[np.isfinite(lower)], upper[np.isfinite(upper)]))
    if finite_limits.size:
        span = float(np.ptp(finite_limits))
        margin = max(0.025, 0.08 * span)
        axis.set_ylim(
            max(0.0, float(np.min(finite_limits)) - margin),
            min(1.0, float(np.max(finite_limits)) + margin),
        )
    axis.set_xlim(checkpoint_edges(checkpoints)[[0, -1]])
    axis.set_xticks(iteration_ticks(checkpoints))
    # axis.tick_params(axis="x", labelrotation=45)
    axis.set_xlabel(r"Number of iterations $N_{\rm iter}$")
    axis.set_ylabel(r"Mean entropy $\langle H\rangle$")
    zone_label = "+".join(zones)
    # axis.set_title(f"{tracer} ({zone_label}): classification entropy convergence")
    axis.grid(lw=0.3)
    # axis.set_axisbelow(True)
    axis.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def write_summary_csv(path, checkpoints, combined):
    with open(path, "w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            ("n_iterations", "n_objects", "mean_entropy", "std_entropy")
        )
        for number, n_objects, mean, std in zip(
            checkpoints,
            combined["n_objects"],
            combined["mean_entropy"],
            combined["std_entropy"],
        ):
            writer.writerow(
                (int(number), int(n_objects), f"{mean:.10g}", f"{std:.10g}")
            )


def load_cached_results(npz_path, tracer, zones, checkpoints, bin_edges):
    """Load a compatible saved calculation, or return ``None`` with a reason."""
    npz_path = Path(npz_path)
    if not npz_path.exists():
        return None, "cache file does not exist"

    required = {
        "tracer",
        "zones",
        "checkpoints",
        "entropy_bin_edges",
        "hist_counts",
        "pdf",
        "n_objects",
        "mean_entropy",
        "std_entropy",
    }
    try:
        with np.load(npz_path, allow_pickle=False) as saved:
            missing = sorted(required.difference(saved.files))
            if missing:
                return None, "missing arrays: " + ", ".join(missing)

            saved_tracer = str(np.asarray(saved["tracer"]).item()).upper()
            saved_zones = tuple(str(value).upper() for value in saved["zones"].tolist())
            saved_checkpoints = np.asarray(saved["checkpoints"])
            saved_bin_edges = np.asarray(saved["entropy_bin_edges"])

            if saved_tracer != str(tracer).upper():
                return None, f"tracer is {saved_tracer}, not {tracer}"
            if saved_zones != tuple(str(value).upper() for value in zones):
                return None, f"zones are {saved_zones}, not {tuple(zones)}"
            if not np.array_equal(saved_checkpoints, checkpoints):
                return None, "iteration checkpoints changed"
            if not np.array_equal(saved_bin_edges, bin_edges):
                return None, "entropy bins changed"

            combined = {
                "hist_counts": np.asarray(saved["hist_counts"]).copy(),
                "pdf": np.asarray(saved["pdf"]).copy(),
                "n_objects": np.asarray(saved["n_objects"]).copy(),
                "mean_entropy": np.asarray(saved["mean_entropy"]).copy(),
                "std_entropy": np.asarray(saved["std_entropy"]).copy(),
            }
    except (OSError, ValueError) as error:
        return None, f"could not read cache: {error}"

    expected_shape = (len(checkpoints), len(bin_edges) - 1)
    if combined["pdf"].shape != expected_shape:
        return None, (
            f"PDF shape is {combined['pdf'].shape}, expected {expected_shape}"
        )
    return combined, "compatible cache"


def build_parser():
    parser = argparse.ArgumentParser(
        description="Plot the cumulative DR2 classification-entropy PDF."
    )
    parser.add_argument(
        "--classification-dir",
        type=Path,
        default=DEFAULT_CLASSIFICATION_DIR,
        help="Directory containing tracer/zone DR2 classification files.",
    )
    parser.add_argument(
        "--tracer",
        required=True,
        type=str.upper,
        choices=("BGS", "LRG", "ELG", "QSO"),
    )
    parser.add_argument("--zones", nargs="+", default=("NGC", "SGC"))
    parser.add_argument("--step", type=int, default=5)
    parser.add_argument("--max-iterations", type=int, default=100)
    parser.add_argument("--entropy-bins", type=int, default=100)
    parser.add_argument(
        "--color-scale", choices=("linear", "log"), default="log"
    )
    parser.add_argument("--outdir", type=Path, default=Path("plots/entropy_evolution"))
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument(
        "--recompute",
        action="store_true",
        help="Ignore a compatible NPZ cache and reread all classification FITS files.",
    )
    return parser


def main():
    args = build_parser().parse_args()
    if args.step <= 0:
        raise ValueError("--step must be positive")
    if args.max_iterations <= 0:
        raise ValueError("--max-iterations must be positive")
    if args.max_iterations % args.step != 0:
        raise ValueError("--max-iterations must be divisible by --step")
    if args.entropy_bins <= 0:
        raise ValueError("--entropy-bins must be positive")

    tracer = args.tracer.upper()
    zones = tuple(str(zone).strip().upper() for zone in args.zones)
    checkpoints = np.arange(
        args.step, args.max_iterations + 1, args.step, dtype=np.int32
    )
    bin_edges = np.linspace(0.0, 1.0, args.entropy_bins + 1)

    args.outdir.mkdir(parents=True, exist_ok=True)
    zone_tag = "-".join(zone.lower() for zone in zones)
    stem = f"{tracer.lower()}_{zone_tag}_entropy_pdf_evolution"
    plot_path = args.outdir / f"{stem}.png"
    line_plot_path = args.outdir / f"{tracer.lower()}_{zone_tag}_mean_entropy_evolution.png"
    npz_path = args.outdir / f"{stem}.npz"
    csv_path = args.outdir / f"{stem}.csv"
    metadata_path = args.outdir / f"{stem}.meta.json"

    if not args.recompute:
        cached, cache_reason = load_cached_results(
            npz_path,
            tracer=tracer,
            zones=zones,
            checkpoints=checkpoints,
            bin_edges=bin_edges,
        )
        if cached is not None:
            print(f"Reusing cached calculation: {npz_path}", flush=True)
            plot_entropy_pdf(
                checkpoints,
                bin_edges,
                cached["pdf"],
                cached["mean_entropy"],
                tracer=tracer,
                zones=zones,
                output_path=plot_path,
                color_scale=args.color_scale,
                dpi=args.dpi,
            )
            plot_mean_entropy(
                checkpoints,
                cached["mean_entropy"],
                cached["std_entropy"],
                tracer=tracer,
                zones=zones,
                output_path=line_plot_path,
                dpi=args.dpi,
            )
            print("Skipped FITS processing; only plots were updated.")
            print(f"Plot: {plot_path}")
            print(f"Mean-entropy plot: {line_plot_path}")
            return
        print(f"Cache not reused ({cache_reason}); processing FITS files.", flush=True)
    else:
        print("Forced recomputation requested; processing FITS files.", flush=True)

    zone_results = []
    files_by_zone = {}
    for zone in zones:
        files = discover_iteration_files(
            args.classification_dir,
            tracer=tracer,
            zone=zone,
            max_iterations=args.max_iterations,
        )
        files_by_zone[zone] = files
        print(f"[{zone}] Found {len(files)} realizations for {tracer}", flush=True)
        zone_results.append(
            process_zone(files, checkpoints, bin_edges=bin_edges, zone=zone)
        )

    combined = combine_zone_results(zone_results, bin_edges=bin_edges)
    zone_hist_counts = np.stack(
        [result["hist_counts"] for result in zone_results], axis=0
    )
    zone_pdf = np.stack(
        [counts_to_pdf(result["hist_counts"], bin_edges) for result in zone_results],
        axis=0,
    )

    plot_entropy_pdf(
        checkpoints,
        bin_edges,
        combined["pdf"],
        combined["mean_entropy"],
        tracer=tracer,
        zones=zones,
        output_path=plot_path,
        color_scale=args.color_scale,
        dpi=args.dpi,
    )
    plot_mean_entropy(
        checkpoints,
        combined["mean_entropy"],
        combined["std_entropy"],
        tracer=tracer,
        zones=zones,
        output_path=line_plot_path,
        dpi=args.dpi,
    )
    np.savez_compressed(
        npz_path,
        tracer=np.array(tracer),
        zones=np.asarray(zones),
        checkpoints=checkpoints,
        entropy_bin_edges=bin_edges,
        entropy_bin_centers=0.5 * (bin_edges[:-1] + bin_edges[1:]),
        hist_counts=combined["hist_counts"],
        pdf=combined["pdf"],
        n_objects=combined["n_objects"],
        mean_entropy=combined["mean_entropy"],
        std_entropy=combined["std_entropy"],
        zone_hist_counts=zone_hist_counts,
        zone_pdf=zone_pdf,
        zone_n_objects=np.stack(
            [result["n_objects"] for result in zone_results], axis=0
        ),
        zone_mean_entropy=np.stack(
            [result["mean_entropy"] for result in zone_results], axis=0
        ),
    )
    write_summary_csv(csv_path, checkpoints, combined)

    metadata = {
        "tracer": tracer,
        "zones": list(zones),
        "classification_directory": str(args.classification_dir),
        "iteration_files": {
            zone: [str(path) for path in files_by_zone[zone]] for zone in zones
        },
        "checkpoints": checkpoints.tolist(),
        "entropy_bins": int(args.entropy_bins),
        "classification_ratio": "r = (NDATA - NRAND) / (NDATA + NRAND)",
        "classification_thresholds": {
            "void": [-1.0, -0.25],
            "sheet": [-0.25, 0.25],
            "filament": [0.25, 0.65],
            "knot": [0.65, 1.0],
        },
        "entropy_definition": (
            "H_i(N) = -sum_w p_iw(N) log2[p_iw(N)] / log2(4)"
        ),
        "pdf_normalization": "sum_j PDF(N, H_j) * delta_H_j = 1",
        "zone_combination": "Pooled object histograms (object-count weighted).",
        "unclassified_rule": (
            "Object-realization pairs with NDATA + NRAND = 0 are omitted; objects "
            "with no valid classification at a checkpoint are omitted from its PDF."
        ),
        "outputs": {
            "pdf_plot": str(plot_path),
            "mean_entropy_plot": str(line_plot_path),
            "arrays": str(npz_path),
            "summary": str(csv_path),
        },
    }
    with open(metadata_path, "w", encoding="utf-8") as stream:
        json.dump(metadata, stream, indent=2)

    print(f"Plot: {plot_path}")
    print(f"Mean-entropy plot: {line_plot_path}")
    print(f"Arrays: {npz_path}")
    print(f"Summary: {csv_path}")
    print(f"Metadata: {metadata_path}")


if __name__ == "__main__":
    main()
