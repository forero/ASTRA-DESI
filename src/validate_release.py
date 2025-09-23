
#!/usr/bin/env python3
"""Validate ASTRA-DESI release products for structural and consistency checks."""

import argparse
import os
import sys
from collections import Counter
from types import SimpleNamespace

import numpy as np
from astropy.table import Table

from desiproc.paths import (
    classification_path,
    probability_path,
    pairs_path,
    safe_tag,
    zone_tag,
    normalize_release_dir,
)

RAW_COLUMNS = ("TARGETID", "TRACERTYPE", "RANDITER")
CLASS_COLUMNS = ("TARGETID", "RANDITER", "ISDATA", "NDATA", "NRAND", "TRACERTYPE")
PAIR_COLUMNS = ("TARGETID1", "TARGETID2", "RANDITER")
PROB_COLUMNS = ("TARGETID", "TRACERTYPE", "PVOID", "PSHEET", "PFILAMENT", "PKNOT")
GROUP_COLUMNS = ("TARGETID", "TRACERTYPE", "RANDITER", "WEBTYPE", "GROUPID")


class ValidationContext:
    """Hold shared configuration for validation runs."""

    def __init__(self, raw_dir, release_root, groups_dir, n_random, webtypes):
        self.raw_dir = raw_dir
        self.release_root = release_root
        self.groups_dir = groups_dir
        self.n_random = n_random
        self.webtypes = tuple(webtypes)
        self.iteration_set = set(range(n_random))


class ZoneEntry(SimpleNamespace):
    """Store bookkeeping for a single zone/tag combination."""

    @property
    def zone_label(self):
        return zone_tag(self.zone_value)

    @property
    def tag_label(self):
        return self.tag if self.tag is not None else "(none)"


def _read_table(path, columns=None):
    """Read a FITS table, restricting to selected columns when requested."""
    read_kwargs = {"hdu": 1}
    if columns is not None:
        read_kwargs["include_names"] = list(columns)
    try:
        return Table.read(path, memmap=True, **read_kwargs)
    except TypeError:
        # Older astropy versions may not understand memmap for gz files
        return Table.read(path, **read_kwargs)


def _parse_zone_tag(filepath):
    """Infer (zone, tag) from a raw filename."""
    name = os.path.basename(filepath)
    if not name.startswith("zone_") or not name.endswith(".fits.gz"):
        raise ValueError(f"Unrecognized raw filename format: {name}")
    body = name[len("zone_") : -len(".fits.gz")]
    if "_" in body:
        zone_token, tag_part = body.split("_", 1)
        tag = tag_part
    else:
        zone_token, tag = body, None
    if zone_token.isdigit():
        zone_value = int(zone_token)
    else:
        zone_value = zone_token
    return zone_value, tag


def _discover_zone_entries(raw_dir, zone_filter=None, tag_filter=None):
    entries = []
    for path in sorted(os.listdir(raw_dir)):
        if not path.startswith("zone_") or not path.endswith(".fits.gz"):
            continue
        full_path = os.path.join(raw_dir, path)
        zone_value, tag = _parse_zone_tag(full_path)
        zone_norm = zone_tag(zone_value)
        if zone_filter is not None and zone_norm not in zone_filter:
            continue
        if tag_filter is not None and tag not in tag_filter:
            continue
        entries.append(ZoneEntry(zone_value=zone_value, tag=tag, raw_path=full_path))
    return entries


def _ensure_columns(table, required, errors, context):
    missing = [col for col in required if col not in table.colnames]
    if missing:
        errors.append(f"{context} missing columns: {', '.join(missing)}")
        return False
    return True


def _split_tracer_label(label, suffix):
    if label.endswith(suffix):
        return label[: -len(suffix)]
    return None


def _validate_raw(entry, ctx):
    errors, warnings = [], []
    try:
        raw_tbl = _read_table(entry.raw_path, RAW_COLUMNS)
    except FileNotFoundError:
        errors.append(f"raw file not found: {entry.raw_path}")
        return None, errors, warnings
    except Exception as exc:  # pragma: no cover - runtime diagnostics
        errors.append(f"failed to read raw file {entry.raw_path}: {exc}")
        return None, errors, warnings

    if not _ensure_columns(raw_tbl, RAW_COLUMNS, errors, "raw table"):
        return None, errors, warnings

    tids = np.asarray(raw_tbl["TARGETID"], dtype=np.int64)
    tracers = np.asarray(raw_tbl["TRACERTYPE"]).astype(str)
    randiters = np.asarray(raw_tbl["RANDITER"], dtype=np.int32)

    data_mask = randiters == -1
    random_mask = randiters >= 0
    neg_random = (~data_mask) & (randiters < 0)
    if np.any(neg_random):
        bad = int(randiters[neg_random][0])
        errors.append(f"raw table has negative RANDITER value {bad} outside data rows")

    info = SimpleNamespace(
        data_ids=set(),
        data_tracer={},
        random_pairs=set(),
        random_tracer={},
        tracer_data_counts={},
        tracer_random_counts={},
        data_row_count=int(data_mask.sum()),
        random_row_count=int(random_mask.sum()),
        total_rows=len(raw_tbl),
        all_ids=set(int(x) for x in tids.tolist()),
    )

    if info.data_row_count == 0:
        warnings.append("raw table contains no data rows")
    else:
        data_tids = tids[data_mask]
        unique_data = set(int(x) for x in np.unique(data_tids))
        if len(unique_data) != data_tids.size:
            errors.append("raw data rows contain duplicate TARGETID values")
        info.data_ids = unique_data
        info.data_tracer = {
            int(tid): tracer
            for tid, tracer in zip(data_tids.tolist(), tracers[data_mask].tolist())
        }

    if info.random_row_count > 0:
        rand_values = randiters[random_mask]
        min_it = int(rand_values.min())
        max_it = int(rand_values.max())
        if min_it < 0 or max_it >= ctx.n_random:
            errors.append(
                f"raw random iterations outside expected range [0, {ctx.n_random - 1}]"
            )
        iter_set = set(int(x) for x in np.unique(rand_values))
        if info.data_row_count > 0:
            missing = sorted(ctx.iteration_set - iter_set)
            if missing:
                errors.append(
                    "raw table missing random iterations: " + ", ".join(str(x) for x in missing)
                )
    elif info.data_row_count > 0:
        errors.append("raw table has data rows but no random catalog entries")

    # Per-tracer checks
    data_labels = tracers[data_mask]
    random_labels = tracers[random_mask]

    for label in np.unique(data_labels):
        base = _split_tracer_label(label, "_DATA")
        if base is None:
            errors.append(f"unexpected data tracer label '{label}' (expected *_DATA)")
            continue
        data_count = int(np.count_nonzero(data_labels == label))
        info.tracer_data_counts[base] = data_count
        random_label = f"{base}_RAND"
        random_count = int(np.count_nonzero(random_labels == random_label))
        info.tracer_random_counts[base] = random_count
        if data_count > 0:
            expected = data_count * ctx.n_random
            if random_count != expected:
                errors.append(
                    f"tracer {base}: random rows {random_count} != data_rows({data_count}) * n_random({ctx.n_random})"
                )
            if random_count > 0:
                per_iter = np.bincount(
                    randiters[random_mask & (tracers == random_label)], minlength=ctx.n_random
                )
                missing_iters = np.where(per_iter == 0)[0]
                if missing_iters.size:
                    miss = ", ".join(str(int(x)) for x in missing_iters.tolist())
                    errors.append(f"tracer {base}: missing random iterations {miss}")
                wrong_counts = np.where((per_iter != 0) & (per_iter != data_count))[0]
                if wrong_counts.size:
                    wrong = ", ".join(str(int(x)) for x in wrong_counts.tolist())
                    errors.append(
                        f"tracer {base}: random counts per iteration differ from data count (iterations {wrong})"
                    )
        elif random_count > 0:
            errors.append(f"tracer {base}: random rows present but no data rows")

    for label in np.unique(random_labels):
        base = _split_tracer_label(label, "_RAND")
        if base is None:
            errors.append(f"unexpected random tracer label '{label}' (expected *_RAND)")
            continue
        if info.tracer_data_counts.get(base, 0) == 0:
            errors.append(f"tracer {base}: random entries without matching data tracer")

    # Map random entries for later cross-checks
    for tid, riter, label in zip(tids[random_mask], randiters[random_mask], random_labels):
        key = (int(tid), int(riter))
        existing = info.random_tracer.get(key)
        if existing is not None and existing != label:
            errors.append(
                f"random entry (TARGETID={tid}, RANDITER={riter}) has inconsistent tracer labels"
            )
        info.random_tracer[key] = label
        info.random_pairs.add(key)

    if info.data_row_count > 0 and info.random_row_count != info.data_row_count * ctx.n_random:
        errors.append(
            "raw table total random count does not match data_count * n_random"
        )

    return info, errors, warnings


def _validate_pairs(entry, ctx, raw_info):
    errors, warnings = [], []
    pairs_pathname = pairs_path(ctx.release_root, entry.zone_value, entry.tag)
    if not os.path.exists(pairs_pathname):
        errors.append(f"pairs file missing: {pairs_pathname}")
        return errors, warnings
    try:
        pairs_tbl = _read_table(pairs_pathname, PAIR_COLUMNS)
    except Exception as exc:  # pragma: no cover - runtime diagnostics
        errors.append(f"failed to read pairs file {pairs_pathname}: {exc}")
        return errors, warnings

    if not _ensure_columns(pairs_tbl, PAIR_COLUMNS, errors, "pairs table"):
        return errors, warnings

    randiters = np.asarray(pairs_tbl["RANDITER"], dtype=np.int32)
    if randiters.size:
        if randiters.min() < 0 or randiters.max() >= ctx.n_random:
            errors.append("pairs table RANDITER values outside expected range")

    if raw_info is not None and len(pairs_tbl) > 0:
        ids = np.concatenate(
            [np.asarray(pairs_tbl["TARGETID1"], dtype=np.int64), np.asarray(pairs_tbl["TARGETID2"], dtype=np.int64)]
        )
        extra = set(int(x) for x in ids.tolist()) - raw_info.all_ids
        if extra:
            errors.append(
                "pairs table contains TARGETID values not present in raw table: "
                + ", ".join(str(x) for x in sorted(extra)[:10])
            )
    return errors, warnings


def _validate_classification(entry, ctx, raw_info):
    errors, warnings = [], []
    class_pathname = classification_path(ctx.release_root, entry.zone_value, entry.tag)
    if not os.path.exists(class_pathname):
        errors.append(f"classification file missing: {class_pathname}")
        return errors, warnings
    try:
        class_tbl = _read_table(class_pathname, CLASS_COLUMNS)
    except Exception as exc:  # pragma: no cover - runtime diagnostics
        errors.append(f"failed to read classification file {class_pathname}: {exc}")
        return errors, warnings

    if not _ensure_columns(class_tbl, CLASS_COLUMNS, errors, "classification table"):
        return errors, warnings

    tids = np.asarray(class_tbl["TARGETID"], dtype=np.int64)
    randiters = np.asarray(class_tbl["RANDITER"], dtype=np.int32)
    isdata = np.asarray(class_tbl["ISDATA"], dtype=bool)
    ndata = np.asarray(class_tbl["NDATA"], dtype=np.int64)
    nrand = np.asarray(class_tbl["NRAND"], dtype=np.int64)
    tracers = np.asarray(class_tbl["TRACERTYPE"]).astype(str)

    if randiters.size:
        if randiters.min() < 0 or randiters.max() >= ctx.n_random:
            errors.append("classification RANDITER values outside expected range")
    if np.any(ndata < 0) or np.any(nrand < 0):
        errors.append("classification counts (NDATA/NRAND) must be non-negative")

    if raw_info is None:
        return errors, warnings

    data_mask = isdata
    random_mask = ~isdata

    data_ids = tids[data_mask]
    unique_data = set(int(x) for x in np.unique(data_ids))
    if raw_info.data_ids:
        missing = raw_info.data_ids - unique_data
        if missing:
            errors.append(
                "classification missing data TARGETIDs: "
                + ", ".join(str(x) for x in sorted(missing)[:10])
            )
        extra = unique_data - raw_info.data_ids
        if extra:
            errors.append(
                "classification has unknown data TARGETIDs: "
                + ", ".join(str(x) for x in sorted(extra)[:10])
            )

        counts = Counter(int(x) for x in data_ids.tolist())
        expected_iters = ctx.iteration_set
        for tid_val in raw_info.data_ids:
            count = counts.get(tid_val, 0)
            if count != ctx.n_random:
                errors.append(
                    f"data TARGETID {tid_val} appears {count} times (expected {ctx.n_random})"
                )
            mask_tid = data_mask & (tids == tid_val)
            if mask_tid.any():
                seen_iters = set(int(x) for x in randiters[mask_tid].tolist())
                missing_iters = expected_iters - seen_iters
                if missing_iters:
                    errors.append(
                        f"data TARGETID {tid_val} missing iterations: "
                        + ", ".join(str(x) for x in sorted(missing_iters))
                    )
                labels = set(tracers[mask_tid].tolist())
                expected_label = raw_info.data_tracer.get(tid_val)
                if expected_label and labels != {expected_label}:
                    errors.append(
                        f"data TARGETID {tid_val} tracer mismatch (found {labels}, expected {expected_label})"
                    )

    random_pairs = set()
    for tid_val, riter, label in zip(tids[random_mask], randiters[random_mask], tracers[random_mask]):
        key = (int(tid_val), int(riter))
        random_pairs.add(key)
        expected = raw_info.random_tracer.get(key)
        if expected is None:
            errors.append(
                f"classification random entry missing from raw table (TARGETID={tid_val}, RANDITER={riter})"
            )
        elif expected != label:
            errors.append(
                f"classification random entry tracer mismatch for TARGETID={tid_val}, RANDITER={riter}"
            )

    missing_pairs = raw_info.random_pairs - random_pairs
    if missing_pairs:
        sample = ", ".join(f"(id={tid},iter={it})" for tid, it in sorted(missing_pairs)[:10])
        errors.append(f"classification missing random combinations: {sample}")

    extra_pairs = random_pairs - raw_info.random_pairs
    if extra_pairs:
        sample = ", ".join(f"(id={tid},iter={it})" for tid, it in sorted(extra_pairs)[:10])
        errors.append(f"classification has extra random combinations: {sample}")

    expected_rows = raw_info.data_row_count * ctx.n_random + raw_info.random_row_count
    if len(class_tbl) != expected_rows:
        errors.append(
            f"classification table length {len(class_tbl)} does not match expected {expected_rows}"
        )

    return errors, warnings


def _validate_probability(entry, ctx, raw_info):
    errors, warnings = [], []
    prob_pathname = probability_path(ctx.release_root, entry.zone_value, entry.tag)
    if not os.path.exists(prob_pathname):
        errors.append(f"probability file missing: {prob_pathname}")
        return errors, warnings
    try:
        prob_tbl = _read_table(prob_pathname, PROB_COLUMNS)
    except Exception as exc:  # pragma: no cover - runtime diagnostics
        errors.append(f"failed to read probability file {prob_pathname}: {exc}")
        return errors, warnings

    if not _ensure_columns(prob_tbl, PROB_COLUMNS, errors, "probability table"):
        return errors, warnings

    tids = np.asarray(prob_tbl["TARGETID"], dtype=np.int64)
    tracers = np.asarray(prob_tbl["TRACERTYPE"]).astype(str)
    pvoid = np.asarray(prob_tbl["PVOID"], dtype=np.float64)
    psheet = np.asarray(prob_tbl["PSHEET"], dtype=np.float64)
    pfil = np.asarray(prob_tbl["PFILAMENT"], dtype=np.float64)
    pknot = np.asarray(prob_tbl["PKNOT"], dtype=np.float64)

    total = pvoid + psheet + pfil + pknot
    if np.any(np.isnan(total)):
        errors.append("probability table contains NaN values")
    diff = np.abs(total - 1.0)
    if np.any(diff > 1e-3):
        errors.append("probability rows do not sum to unity within tolerance (1e-3)")
    if np.any((pvoid < -1e-6) | (psheet < -1e-6) | (pfil < -1e-6) | (pknot < -1e-6)):
        errors.append("probability table has negative probability entries")
    if np.any((pvoid > 1 + 1e-6) | (psheet > 1 + 1e-6) | (pfil > 1 + 1e-6) | (pknot > 1 + 1e-6)):
        errors.append("probability table has probability entries greater than 1")

    if raw_info is None:
        return errors, warnings

    expected_ids = raw_info.data_ids
    ids_set = set(int(x) for x in tids.tolist())
    missing = expected_ids - ids_set
    if missing:
        errors.append(
            "probability table missing TARGETIDs: " + ", ".join(str(x) for x in sorted(missing)[:10])
        )
    extra = ids_set - expected_ids
    if extra:
        errors.append(
            "probability table has unexpected TARGETIDs: "
            + ", ".join(str(x) for x in sorted(extra)[:10])
        )
    for tid_val, label in zip(tids.tolist(), tracers.tolist()):
        expected = raw_info.data_tracer.get(int(tid_val))
        if expected and expected != label:
            errors.append(
                f"probability tracer mismatch for TARGETID={tid_val} (expected {expected}, found {label})"
            )
    if expected_ids and len(prob_tbl) != len(expected_ids):
        errors.append(
            f"probability table length {len(prob_tbl)} != number of data targets {len(expected_ids)}"
        )

    return errors, warnings


def _groups_path(groups_dir, zone_value, tag, webtype):
    zone_str = zone_tag(zone_value)
    tsuf = safe_tag(tag)
    filename = f"zone_{zone_str}{tsuf}_groups_fof_{webtype}.fits.gz"
    return os.path.join(groups_dir, filename)


def _validate_groups(entry, ctx, raw_info):
    if ctx.groups_dir is None or not ctx.webtypes:
        return [], []
    errors, warnings = [], []
    for webtype in ctx.webtypes:
        path = _groups_path(ctx.groups_dir, entry.zone_value, entry.tag, webtype)
        if not os.path.exists(path):
            warnings.append(f"groups file missing for webtype '{webtype}': {path}")
            continue
        try:
            grp_tbl = _read_table(path, GROUP_COLUMNS)
        except Exception as exc:  # pragma: no cover - runtime diagnostics
            errors.append(f"failed to read groups file {path}: {exc}")
            continue
        if not _ensure_columns(grp_tbl, GROUP_COLUMNS, errors, f"groups table ({webtype})"):
            continue
        if len(grp_tbl) == 0:
            warnings.append(f"groups file {path} contains no rows")
            continue
        if "WEBTYPE" in grp_tbl.colnames:
            unique_web = set(str(x) for x in np.unique(grp_tbl["WEBTYPE"]))
            if unique_web != {webtype}:
                errors.append(
                    f"groups file {path} has WEBTYPE values {unique_web} (expected '{webtype}')"
                )
        if raw_info is not None:
            ids = set(int(x) for x in np.asarray(grp_tbl["TARGETID"], dtype=np.int64).tolist())
            extra = ids - raw_info.all_ids
            if extra:
                errors.append(
                    "groups file contains TARGETIDs not present in raw table: "
                    + ", ".join(str(x) for x in sorted(extra)[:10])
                )
    return errors, warnings


def validate_entry(entry, ctx):
    errors, warnings = [], []

    raw_info, raw_errors, raw_warnings = _validate_raw(entry, ctx)
    errors.extend(raw_errors)
    warnings.extend(raw_warnings)

    pair_errors, pair_warnings = _validate_pairs(entry, ctx, raw_info)
    errors.extend(pair_errors)
    warnings.extend(pair_warnings)

    class_errors, class_warnings = _validate_classification(entry, ctx, raw_info)
    errors.extend(class_errors)
    warnings.extend(class_warnings)

    prob_errors, prob_warnings = _validate_probability(entry, ctx, raw_info)
    errors.extend(prob_errors)
    warnings.extend(prob_warnings)

    group_errors, group_warnings = _validate_groups(entry, ctx, raw_info)
    errors.extend(group_errors)
    warnings.extend(group_warnings)

    return errors, warnings


def _normalise_zone_values(values):
    normalised = set()
    for value in values:
        try:
            normalised.add(zone_tag(int(value)))
        except (ValueError, TypeError):
            normalised.add(str(value))
    return normalised


def _normalise_tag_values(values):
    normalised = set()
    for value in values:
        if value is None:
            normalised.add(None)
            continue
        lowered = str(value).lower()
        if lowered in {"none", "null", "combined"}:
            normalised.add(None)
        else:
            normalised.add(str(value))
    return normalised


def build_parser():
    parser = argparse.ArgumentParser(
        description="Validate ASTRA-DESI pipeline outputs for a set of zones",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--raw-dir", required=True, help="Directory containing raw/*.fits.gz files")
    parser.add_argument(
        "--release-root",
        required=True,
        help="Classification root (directory with classification/, probabilities/, pairs/)",
    )
    parser.add_argument(
        "--groups-dir",
        default=None,
        help="Optional groups directory to validate (expects zone_*_groups_fof_*.fits.gz)",
    )
    parser.add_argument("--n-random", type=int, default=100, help="Expected number of random iterations")
    parser.add_argument(
        "--webtypes",
        nargs="*",
        default=("filament",),
        help="Webtypes to validate in groups outputs (ignored when --groups-dir is omitted)",
    )
    parser.add_argument(
        "--zones",
        nargs="*",
        default=None,
        help="Optional subset of zones to validate (accepts integers or labels like NGC1)",
    )
    parser.add_argument(
        "--tags",
        nargs="*",
        default=None,
        help="Optional subset of tags to validate (use 'combined' or 'none' for untagged files)",
    )
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)

    raw_dir = os.path.abspath(args.raw_dir)
    release_root = normalize_release_dir(os.path.abspath(args.release_root))
    groups_dir = os.path.abspath(args.groups_dir) if args.groups_dir else None

    if not os.path.isdir(raw_dir):
        parser.error(f"raw directory not found: {raw_dir}")
    if not os.path.isdir(release_root):
        parser.error(f"release root not found: {release_root}")
    if groups_dir and not os.path.isdir(groups_dir):
        parser.error(f"groups directory not found: {groups_dir}")

    zone_filter = _normalise_zone_values(args.zones) if args.zones else None
    tag_filter = _normalise_tag_values(args.tags) if args.tags else None

    entries = _discover_zone_entries(raw_dir, zone_filter=zone_filter, tag_filter=tag_filter)
    if not entries:
        print("No raw files found matching the provided filters.", file=sys.stderr)
        return 1

    ctx = ValidationContext(raw_dir, release_root, groups_dir, args.n_random, args.webtypes)

    total_errors = 0
    total_warnings = 0

    for entry in entries:
        errors, warnings = validate_entry(entry, ctx)
        status = "OK"
        if errors:
            status = "FAIL"
        elif warnings:
            status = "WARN"
        print(f"[{status}] zone {entry.zone_label} tag {entry.tag_label}")
        for msg in errors:
            print(f"  ERROR: {msg}")
        for msg in warnings:
            print(f"  WARN: {msg}")
        total_errors += len(errors)
        total_warnings += len(warnings)

    if total_errors:
        print(f"Validation completed with {total_errors} error(s) and {total_warnings} warning(s).", file=sys.stderr)
        return 1
    print(f"Validation completed with {total_warnings} warning(s).")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    sys.exit(main())
