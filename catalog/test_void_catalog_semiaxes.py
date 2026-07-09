#!/usr/bin/env python3
"""
Consistency tests for the semi-axes stored in a watershed void FITS catalog.

This is the catalog-level version of the uniform-ellipsoid moment-tensor test:
instead of generating synthetic clouds, it checks that the catalog obeys the
definitions written in its header and, when POINT_MEMBERSHIP is present,
recomputes the second-moment tensor directly from the random members.
"""

import argparse
import os
from pathlib import Path

import numpy as np

try:
    from astropy.io import fits
except ImportError as exc:
    raise SystemExit(
        "This script needs astropy. On NERSC, run: module load python/3.12"
    ) from exc


DEFAULT_CATALOG = (
    "/pscratch/sd/v/vtorresg/void_catalog/"
    "DR2_Om_1_Om0p301_h0p6736/voids_LRG_NGC.fits"
)

AXIS_COLUMNS = (
    ("A", "LAMBDA_1", "SEMI_AXIS_A"),
    ("B", "LAMBDA_2", "SEMI_AXIS_B"),
    ("C", "LAMBDA_3", "SEMI_AXIS_C"),
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Test SEMI_AXIS_A/B/C, R_EFF, ELLIP, and optional member-based "
            "second moments in a void catalog."
        )
    )
    parser.add_argument(
        "--catalog",
        default=DEFAULT_CATALOG,
        help="Input FITS catalog. Default: %(default)s",
    )
    parser.add_argument(
        "--skip-membership",
        action="store_true",
        help="Skip direct recomputation from POINT_MEMBERSHIP.",
    )
    parser.add_argument(
        "--core-rtol",
        type=float,
        default=1.0e-10,
        help="Relative tolerance for algebraic VOIDS-table checks.",
    )
    parser.add_argument(
        "--core-atol",
        type=float,
        default=1.0e-8,
        help="Absolute tolerance for algebraic VOIDS-table checks.",
    )
    parser.add_argument(
        "--ellip-atol",
        type=float,
        default=1.0e-6,
        help="Absolute tolerance for ELLIP, which is stored as float32.",
    )
    parser.add_argument(
        "--member-rtol",
        type=float,
        default=1.0e-9,
        help="Relative tolerance for POINT_MEMBERSHIP recomputation checks.",
    )
    parser.add_argument(
        "--member-atol",
        type=float,
        default=1.0e-6,
        help="Absolute tolerance for POINT_MEMBERSHIP recomputation checks.",
    )
    parser.add_argument(
        "--member-axis-atol",
        type=float,
        default=1.0e-4,
        help=(
            "Absolute tolerance for semi-axes recomputed from POINT_MEMBERSHIP. "
            "This is separate because sqrt(lambda) amplifies tiny lambda_3 "
            "differences near zero."
        ),
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=5,
        help="Maximum failing row examples to print per check.",
    )
    return parser.parse_args()


def names_of(data):
    return set(data.names or ())


def require_columns(data, columns, hdu_name):
    missing = [name for name in columns if name not in names_of(data)]
    if missing:
        raise KeyError(f"{hdu_name} missing required columns: {missing}")


def as_float(data, column):
    return np.asarray(data[column], dtype=np.float64)


def as_int(data, column):
    return np.asarray(data[column], dtype=np.int64)


def finite_percentiles(values):
    values = np.asarray(values, dtype=np.float64)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return "no finite values"
    p16, p50, p84 = np.percentile(finite, [16, 50, 84])
    return f"p16={p16:.6g}, median={p50:.6g}, p84={p84:.6g}"


class Audit:
    def __init__(self, void_ids, max_examples=5):
        self.void_ids = np.asarray(void_ids, dtype=np.int64)
        self.max_examples = int(max_examples)
        self.failures = 0

    def _print_examples(self, bad_rows, actual=None, expected=None):
        for row in bad_rows[: self.max_examples]:
            line = f"      row={int(row)} VOID_ID={int(self.void_ids[row])}"
            if actual is not None:
                line += f" actual={np.asarray(actual)[row]!r}"
            if expected is not None:
                line += f" expected={np.asarray(expected)[row]!r}"
            print(line)

    def check_mask(self, name, bad_mask, detail=""):
        bad_rows = np.flatnonzero(np.asarray(bad_mask, dtype=bool))
        n_bad = int(bad_rows.size)
        n_total = int(np.asarray(bad_mask).size)
        status = "PASS" if n_bad == 0 else "FAIL"
        suffix = f" ({detail})" if detail else ""
        print(f"[{status}] {name}: bad={n_bad:,}/{n_total:,}{suffix}")
        if n_bad:
            self.failures += 1
            self._print_examples(bad_rows)
        return n_bad == 0

    def check_equal_int(self, name, actual, expected):
        actual = np.asarray(actual)
        expected = np.asarray(expected)
        bad = actual != expected
        bad_rows = np.flatnonzero(bad)
        status = "PASS" if bad_rows.size == 0 else "FAIL"
        print(f"[{status}] {name}: bad={bad_rows.size:,}/{actual.size:,}")
        if bad_rows.size:
            self.failures += 1
            self._print_examples(bad_rows, actual=actual, expected=expected)
        return bad_rows.size == 0

    def check_close(self, name, actual, expected, mask, rtol, atol):
        actual = np.asarray(actual, dtype=np.float64)
        expected = np.asarray(expected, dtype=np.float64)
        mask = np.asarray(mask, dtype=bool)
        if actual.shape != expected.shape:
            raise ValueError(f"{name}: shape mismatch {actual.shape} vs {expected.shape}")
        if actual.shape != mask.shape:
            raise ValueError(f"{name}: mask shape mismatch {mask.shape} vs {actual.shape}")

        finite = mask & np.isfinite(actual) & np.isfinite(expected)
        n_check = int(np.count_nonzero(finite))
        if n_check == 0:
            self.failures += 1
            print(f"[FAIL] {name}: no finite rows to check")
            return False

        close = np.isclose(actual, expected, rtol=rtol, atol=atol, equal_nan=False)
        bad = finite & ~close
        diff = np.abs(actual[finite] - expected[finite])
        scale = np.maximum(np.abs(expected[finite]), atol)
        rel = diff / scale
        max_abs = float(np.max(diff)) if diff.size else np.nan
        max_rel = float(np.max(rel)) if rel.size else np.nan

        bad_rows = np.flatnonzero(bad)
        status = "PASS" if bad_rows.size == 0 else "FAIL"
        print(
            f"[{status}] {name}: checked={n_check:,}, bad={bad_rows.size:,}, "
            f"max_abs={max_abs:.3e}, max_rel={max_rel:.3e}"
        )
        if bad_rows.size:
            self.failures += 1
            self._print_examples(bad_rows, actual=actual, expected=expected)
        return bad_rows.size == 0


def min_rand_for_shape(header):
    if "MINRSHAP" in header:
        return int(header["MINRSHAP"])
    return 3


def expected_ellipticity(semi_a, semi_b, semi_c):
    expected = np.full(semi_a.shape, np.nan, dtype=np.float64)
    valid = (
        np.isfinite(semi_a)
        & np.isfinite(semi_b)
        & np.isfinite(semi_c)
        & (semi_a > 0.0)
        & (semi_b > 0.0)
        & (semi_c > 0.0)
    )
    ratio = (semi_c[valid] * semi_c[valid] + semi_b[valid] * semi_b[valid])
    ratio /= semi_b[valid] * semi_b[valid] + semi_a[valid] * semi_a[valid]
    expected[valid] = 1.0 - np.power(np.clip(ratio, 0.0, 1.0), 0.25)
    return expected, valid


def run_void_table_checks(voids, header, args):
    required = (
        "VOID_ID",
        "N_RAND_IN_GROUP",
        "LAMBDA_1",
        "LAMBDA_2",
        "LAMBDA_3",
        "SEMI_AXIS_A",
        "SEMI_AXIS_B",
        "SEMI_AXIS_C",
        "R_EFF",
        "ELLIP",
    )
    require_columns(voids, required, "VOIDS")

    void_ids = as_int(voids, "VOID_ID")
    audit = Audit(void_ids, max_examples=args.max_examples)
    min_shape = min_rand_for_shape(header)

    n_rand = as_int(voids, "N_RAND_IN_GROUP")
    lambdas = np.column_stack([as_float(voids, col) for _, col, _ in AXIS_COLUMNS])
    axes = np.column_stack([as_float(voids, col) for _, _, col in AXIS_COLUMNS])
    reff = as_float(voids, "R_EFF")
    ellip = as_float(voids, "ELLIP")

    finite_lambdas = np.all(np.isfinite(lambdas), axis=1)
    finite_axes = np.all(np.isfinite(axes), axis=1)
    finite_reff = np.isfinite(reff)
    valid_shape = n_rand >= min_shape
    finite_shape = finite_lambdas & finite_axes & finite_reff

    print("\nVOIDS-table algebraic checks")
    print(f"  MINRSHAP={min_shape}")
    print(f"  rows={len(voids):,}")
    print(f"  finite shape rows={np.count_nonzero(finite_shape):,}")
    print(f"  rows with N_RAND_IN_GROUP < MINRSHAP={np.count_nonzero(~valid_shape):,}")

    audit.check_mask(
        "finite lambda/axis/R_EFF mask follows N_RAND_IN_GROUP >= MINRSHAP",
        finite_shape != valid_shape,
    )

    order_bad = (
        finite_lambdas
        & ((lambdas[:, 0] + args.core_atol < lambdas[:, 1])
           | (lambdas[:, 1] + args.core_atol < lambdas[:, 2]))
    )
    audit.check_mask("lambda ordering LAMBDA_1 >= LAMBDA_2 >= LAMBDA_3", order_bad)

    axis_order_bad = (
        finite_axes
        & ((axes[:, 0] + args.core_atol < axes[:, 1])
           | (axes[:, 1] + args.core_atol < axes[:, 2]))
    )
    audit.check_mask("semi-axis ordering A >= B >= C", axis_order_bad)

    expected_axes = np.sqrt(5.0 * np.clip(lambdas, 0.0, None))
    for j, (label, lambda_col, axis_col) in enumerate(AXIS_COLUMNS):
        mask = np.isfinite(lambdas[:, j]) & np.isfinite(axes[:, j])
        audit.check_close(
            f"{axis_col} = sqrt(5*{lambda_col})",
            axes[:, j],
            expected_axes[:, j],
            mask,
            rtol=args.core_rtol,
            atol=args.core_atol,
        )

    valid_reff = finite_axes & (axes[:, 0] >= 0.0) & (axes[:, 1] >= 0.0) & (axes[:, 2] >= 0.0)
    expected_reff = np.full(len(voids), np.nan, dtype=np.float64)
    expected_reff[valid_reff] = np.cbrt(
        axes[valid_reff, 0] * axes[valid_reff, 1] * axes[valid_reff, 2]
    )
    audit.check_close(
        "R_EFF = (A*B*C)^(1/3)",
        reff,
        expected_reff,
        valid_reff & np.isfinite(reff),
        rtol=args.core_rtol,
        atol=args.core_atol,
    )

    valid_lambda_reff = finite_lambdas & np.all(lambdas >= 0.0, axis=1)
    expected_reff_lam = np.full(len(voids), np.nan, dtype=np.float64)
    expected_reff_lam[valid_lambda_reff] = (
        np.sqrt(5.0)
        * np.power(
            lambdas[valid_lambda_reff, 0]
            * lambdas[valid_lambda_reff, 1]
            * lambdas[valid_lambda_reff, 2],
            1.0 / 6.0,
        )
    )
    audit.check_close(
        "R_EFF = sqrt(5)*(L1*L2*L3)^(1/6)",
        reff,
        expected_reff_lam,
        valid_lambda_reff & np.isfinite(reff),
        rtol=args.core_rtol,
        atol=args.core_atol,
    )

    expected_ellip, valid_ellip = expected_ellipticity(
        axes[:, 0], axes[:, 1], axes[:, 2]
    )
    audit.check_mask(
        "ELLIP finite mask follows positive finite semi-axes",
        np.isfinite(ellip) != valid_ellip,
    )
    audit.check_close(
        "ELLIP = 1-((C^2+B^2)/(B^2+A^2))^0.25",
        ellip,
        expected_ellip,
        valid_ellip & np.isfinite(ellip),
        rtol=0.0,
        atol=args.ellip_atol,
    )

    if "GEOM_BAD" in names_of(voids):
        geom_bad = np.asarray(voids["GEOM_BAD"], dtype=bool)
        geom_expected = np.zeros(len(voids), dtype=bool)
        valid_geom = finite_axes & (axes[:, 0] > 0.0)
        geom_expected[valid_geom] = (1.0 - axes[valid_geom, 2] / axes[valid_geom, 0]) > 0.9
        audit.check_mask("GEOM_BAD = (1-C/A > 0.9)", geom_bad != geom_expected)

    valid_trace = finite_lambdas & np.all(lambdas >= 0.0, axis=1)
    r_trace = np.full(len(voids), np.nan, dtype=np.float64)
    r_trace[valid_trace] = np.sqrt(5.0 * np.sum(lambdas[valid_trace], axis=1) / 3.0)
    ratio = np.full(len(voids), np.nan, dtype=np.float64)
    valid_ratio = np.isfinite(r_trace) & np.isfinite(reff) & (reff > 0.0)
    ratio[valid_ratio] = r_trace[valid_ratio] / reff[valid_ratio]
    print(
        "  Trace radius R_trace=sqrt(5*sum(lambda)/3) over R_EFF: "
        + finite_percentiles(ratio)
    )

    return audit


def catalog_row_index(group_ids, void_ids):
    group_ids = np.asarray(group_ids, dtype=np.int64)
    void_ids = np.asarray(void_ids, dtype=np.int64)
    sorter = np.argsort(void_ids, kind="stable")
    sorted_ids = void_ids[sorter]
    pos = np.searchsorted(sorted_ids, group_ids)
    in_range = pos < sorted_ids.size
    safe_pos = np.minimum(pos, max(sorted_ids.size - 1, 0))
    matched = in_range & (sorted_ids[safe_pos] == group_ids)
    row_index = np.full(group_ids.shape, -1, dtype=np.int64)
    row_index[matched] = sorter[pos[matched]]
    return row_index, matched


def grouped_weighted_sum(row_index, values, n_rows):
    return np.bincount(row_index, weights=values, minlength=n_rows).astype(np.float64)


def recompute_random_moments(points, void_ids):
    require_columns(
        points,
        ("GROUPID", "IS_DATA", "X_CART", "Y_CART", "Z_CART"),
        "POINT_MEMBERSHIP",
    )

    group_id = as_int(points, "GROUPID")
    is_data = np.asarray(points["IS_DATA"], dtype=bool)
    n_rows = len(void_ids)

    random_candidate = (~is_data) & (group_id >= 0)
    random_rows, random_matched = catalog_row_index(group_id[random_candidate], void_ids)
    random_rows = random_rows[random_matched]

    data_candidate = is_data & (group_id >= 0)
    data_rows, data_matched = catalog_row_index(group_id[data_candidate], void_ids)
    data_rows = data_rows[data_matched]

    n_rand = np.bincount(random_rows, minlength=n_rows).astype(np.int64)
    n_data = np.bincount(data_rows, minlength=n_rows).astype(np.int64)

    x_all = as_float(points, "X_CART")
    y_all = as_float(points, "Y_CART")
    z_all = as_float(points, "Z_CART")
    random_idx = np.flatnonzero(random_candidate)[random_matched]
    x = x_all[random_idx]
    y = y_all[random_idx]
    z = z_all[random_idx]

    sum_x = grouped_weighted_sum(random_rows, x, n_rows)
    sum_y = grouped_weighted_sum(random_rows, y, n_rows)
    sum_z = grouped_weighted_sum(random_rows, z, n_rows)
    sum_x2 = grouped_weighted_sum(random_rows, x * x, n_rows)
    sum_y2 = grouped_weighted_sum(random_rows, y * y, n_rows)
    sum_z2 = grouped_weighted_sum(random_rows, z * z, n_rows)
    sum_xy = grouped_weighted_sum(random_rows, x * y, n_rows)
    sum_xz = grouped_weighted_sum(random_rows, x * z, n_rows)
    sum_yz = grouped_weighted_sum(random_rows, y * z, n_rows)

    center = np.full((n_rows, 3), np.nan, dtype=np.float64)
    has_randoms = n_rand > 0
    n_float = n_rand.astype(np.float64)
    center[has_randoms, 0] = sum_x[has_randoms] / n_float[has_randoms]
    center[has_randoms, 1] = sum_y[has_randoms] / n_float[has_randoms]
    center[has_randoms, 2] = sum_z[has_randoms] / n_float[has_randoms]

    shape = np.full((n_rows, 3, 3), np.nan, dtype=np.float64)
    dx2 = np.clip(sum_x2 - n_float * center[:, 0] * center[:, 0], 0.0, None)
    dy2 = np.clip(sum_y2 - n_float * center[:, 1] * center[:, 1], 0.0, None)
    dz2 = np.clip(sum_z2 - n_float * center[:, 2] * center[:, 2], 0.0, None)
    dxy = sum_xy - n_float * center[:, 0] * center[:, 1]
    dxz = sum_xz - n_float * center[:, 0] * center[:, 2]
    dyz = sum_yz - n_float * center[:, 1] * center[:, 2]

    inv_n = np.zeros(n_rows, dtype=np.float64)
    inv_n[has_randoms] = 1.0 / n_float[has_randoms]
    shape[has_randoms, 0, 0] = dx2[has_randoms] * inv_n[has_randoms]
    shape[has_randoms, 1, 1] = dy2[has_randoms] * inv_n[has_randoms]
    shape[has_randoms, 2, 2] = dz2[has_randoms] * inv_n[has_randoms]
    shape[has_randoms, 0, 1] = shape[has_randoms, 1, 0] = dxy[has_randoms] * inv_n[has_randoms]
    shape[has_randoms, 0, 2] = shape[has_randoms, 2, 0] = dxz[has_randoms] * inv_n[has_randoms]
    shape[has_randoms, 1, 2] = shape[has_randoms, 2, 1] = dyz[has_randoms] * inv_n[has_randoms]

    ignored_random = int(np.count_nonzero(random_candidate) - random_rows.size)
    ignored_data = int(np.count_nonzero(data_candidate) - data_rows.size)
    return {
        "n_rand": n_rand,
        "n_data": n_data,
        "center": center,
        "shape": shape,
        "ignored_random": ignored_random,
        "ignored_data": ignored_data,
    }


def run_membership_checks(voids, points, header, audit, args):
    print("\nPOINT_MEMBERSHIP recomputation checks")
    require_columns(
        voids,
        (
            "VOID_ID",
            "N_DATA_IN_GROUP",
            "N_RAND_IN_GROUP",
            "X",
            "Y",
            "Z",
            "LAMBDA_1",
            "LAMBDA_2",
            "LAMBDA_3",
            "SEMI_AXIS_A",
            "SEMI_AXIS_B",
            "SEMI_AXIS_C",
        ),
        "VOIDS",
    )

    void_ids = as_int(voids, "VOID_ID")
    min_shape = min_rand_for_shape(header)
    moments = recompute_random_moments(points, void_ids)
    print(f"  matched random member rows={int(np.sum(moments['n_rand'])):,}")
    print(f"  matched data member rows={int(np.sum(moments['n_data'])):,}")
    print(f"  ignored random rows with GROUPID>=0 but not in VOIDS={moments['ignored_random']:,}")
    print(f"  ignored data rows with GROUPID>=0 but not in VOIDS={moments['ignored_data']:,}")

    audit.check_equal_int(
        "N_RAND_IN_GROUP matches random POINT_MEMBERSHIP counts",
        as_int(voids, "N_RAND_IN_GROUP"),
        moments["n_rand"],
    )
    audit.check_equal_int(
        "N_DATA_IN_GROUP matches data POINT_MEMBERSHIP counts",
        as_int(voids, "N_DATA_IN_GROUP"),
        moments["n_data"],
    )

    center = moments["center"]
    for j, col in enumerate(("X", "Y", "Z")):
        audit.check_close(
            f"{col} center from random members",
            as_float(voids, col),
            center[:, j],
            moments["n_rand"] > 0,
            rtol=args.member_rtol,
            atol=args.member_atol,
        )

    valid_shape = moments["n_rand"] >= min_shape
    lambdas_calc = np.full((len(voids), 3), np.nan, dtype=np.float64)
    if np.any(valid_shape):
        eigvals = np.linalg.eigvalsh(moments["shape"][valid_shape])[:, ::-1]
        lambdas_calc[valid_shape] = np.clip(eigvals, 0.0, None)
    axes_calc = np.sqrt(5.0 * lambdas_calc)

    lambdas_catalog = np.column_stack([as_float(voids, col) for _, col, _ in AXIS_COLUMNS])
    axes_catalog = np.column_stack([as_float(voids, col) for _, _, col in AXIS_COLUMNS])
    for j, (label, lambda_col, axis_col) in enumerate(AXIS_COLUMNS):
        mask_lam = valid_shape & np.isfinite(lambdas_catalog[:, j]) & np.isfinite(lambdas_calc[:, j])
        audit.check_close(
            f"{lambda_col} from POINT_MEMBERSHIP random second moment",
            lambdas_catalog[:, j],
            lambdas_calc[:, j],
            mask_lam,
            rtol=args.member_rtol,
            atol=args.member_atol,
        )
        mask_axis = valid_shape & np.isfinite(axes_catalog[:, j]) & np.isfinite(axes_calc[:, j])
        audit.check_close(
            f"{axis_col} from POINT_MEMBERSHIP random second moment",
            axes_catalog[:, j],
            axes_calc[:, j],
            mask_axis,
            rtol=args.member_rtol,
            atol=args.member_axis_atol,
        )


def main():
    args = parse_args()
    catalog = Path(args.catalog)
    if not catalog.exists():
        raise FileNotFoundError(str(catalog))

    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

    print(f"Catalog: {catalog}")
    with fits.open(catalog, memmap=True) as hdul:
        if "VOIDS" not in hdul:
            raise KeyError(f"{catalog} has no VOIDS HDU")
        voids = hdul["VOIDS"].data
        header = hdul[0].header

        print(f"HDUs: {[hdu.name for hdu in hdul]}")
        audit = run_void_table_checks(voids, header, args)

        if args.skip_membership:
            print("\nPOINT_MEMBERSHIP recomputation skipped by --skip-membership")
        elif "POINT_MEMBERSHIP" in hdul:
            run_membership_checks(voids, hdul["POINT_MEMBERSHIP"].data, header, audit, args)
        else:
            print("\nPOINT_MEMBERSHIP HDU not present; direct member recomputation skipped")

    if audit.failures:
        raise SystemExit(f"\nFAILED: {audit.failures} consistency check(s) failed")
    print("\nPASS: all requested consistency checks passed")


if __name__ == "__main__":
    main()
