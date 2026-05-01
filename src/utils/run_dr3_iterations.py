import argparse, os, sys


SRC_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)

from desiproc import implement_astra as astra
from desiproc.paths import (classification_path, ensure_release_subdirs,
                            normalize_release_dir, pairs_path)
from desiproc.read_data import process_real_dr2
from releases import dr3


def _parse_iterations(tokens, n_random):
    if not tokens:
        return list(range(n_random))
    out = []
    for token in tokens:
        text = str(token).strip()
        if not text:
            continue
        if '-' in text:
            lo, hi = text.split('-', 1)
            out.extend(range(int(lo), int(hi) + 1))
        else:
            out.append(int(text))
    clean = []
    seen = set()
    for val in out:
        if val < 0 or val >= n_random:
            raise ValueError(f'Iteration {val} outside valid range 0..{n_random - 1}')
        if val not in seen:
            clean.append(val)
            seen.add(val)
    return clean


def _normalise_tracer(label):
    text = str(label).strip()
    if text in dr3.TRACERS:
        return text
    upper = text.upper()
    if upper in dr3.TRACERS:
        return upper
    key = text.lower()
    if key in dr3.TRACER_ALIAS:
        return dr3.TRACER_ALIAS[key]
    raise ValueError(f'Unknown DR3 tracer {label!r}; available: {", ".join(dr3.TRACERS)}')


def _save_classification_direct(class_store, output_path, meta):
    old_split = os.environ.get('ASTRA_CLASS_SPLIT_ITER')
    old_skip = os.environ.get('ASTRA_CLASS_SKIP_COMBINED')
    os.environ['ASTRA_CLASS_SPLIT_ITER'] = '0'
    os.environ['ASTRA_CLASS_SKIP_COMBINED'] = '0'
    try:
        astra.save_classification_fits(class_store, output_path, meta=meta)
    finally:
        if old_split is None:
            os.environ.pop('ASTRA_CLASS_SPLIT_ITER', None)
        else:
            os.environ['ASTRA_CLASS_SPLIT_ITER'] = old_split
        if old_skip is None:
            os.environ.pop('ASTRA_CLASS_SKIP_COMBINED', None)
        else:
            os.environ['ASTRA_CLASS_SKIP_COMBINED'] = old_skip


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-dir', required=True)
    parser.add_argument('--raw-out', required=True)
    parser.add_argument('--class-out', required=True)
    parser.add_argument('--zone', required=True, choices=dr3.DEFAULT_ZONES)
    parser.add_argument('--tracer', required=True)
    parser.add_argument('--iterations', nargs='*', default=None)
    parser.add_argument('--n-random', type=int, default=100)
    parser.add_argument('--n-random-files', type=int, default=dr3.N_RANDOM_FILES)
    parser.add_argument('--spill-dir', default=None)
    parser.add_argument('--force', action='store_true')
    parser.add_argument('--progress', action='store_true')
    args = parser.parse_args()

    if args.progress:
        os.environ.setdefault('ASTRA_PROGRESS', '1')

    tracer = _normalise_tracer(args.tracer)
    zone = str(args.zone).upper()
    iterations = _parse_iterations(args.iterations, args.n_random)
    class_root = normalize_release_dir(args.class_out)

    os.makedirs(args.raw_out, exist_ok=True)
    os.makedirs(class_root, exist_ok=True)
    ensure_release_subdirs(class_root)
    if args.spill_dir:
        os.makedirs(args.spill_dir, exist_ok=True)

    dr3.register_tracer_mapping(dr3.TRACER_IDS, dr3.TRACER_FULL_LABELS)

    real_tables, random_tables = dr3.preload_dr3_tables(args.base_dir, [tracer],
                                                        n_random_files=args.n_random_files,
                                                        zones_to_keep=[zone])
    zone_value = dr3.DR2_ZONE_VALUES.get(zone, 3999)
    real_table = process_real_dr2(real_tables, tracer, zone, zone_value=zone_value,
                                  tracer_id=dr3.TRACER_IDS.get(tracer),
                                  include_tracertype=False, downcast=True)
    print(f'dr3 --> tracer={tracer} zone={zone} real rows={len(real_table)}', flush=True)

    pairs_base = pairs_path(class_root, zone, tracer)
    class_base = classification_path(class_root, zone, tracer)

    for iteration in iterations:
        raw_path = dr3.raw_iteration_path(args.raw_out, zone, tracer, iteration)
        pairs_iter_path = astra._split_iter_path(pairs_base, iteration)
        class_iter_path = astra._split_iter_path(class_base, iteration)
        for path in (pairs_iter_path, class_iter_path):
            os.makedirs(os.path.dirname(path), exist_ok=True)

        if (not args.force and os.path.exists(raw_path)
                and os.path.exists(pairs_iter_path)
                and os.path.exists(class_iter_path)):
            print(f'dr3 --> skip tracer={tracer} zone={zone} iter={iteration:03d}: outputs exist',
                  flush=True)
            continue

        raw_tbl = dr3.build_raw_dr3_iteration(zone, tracer, iteration, real_tables,
                                             random_tables, args.raw_out,
                                             zone_value=zone_value,
                                             release_tag='DR3',
                                             real_table=real_table,
                                             force=args.force)
        meta = {'ZONE': zone,
                'RELEASE': 'DR3',
                'ITER': int(iteration)}
        pair_store = class_store = None
        try:
            pair_store, class_store, _ = astra.generate_pairs_for_iterations(
                raw_tbl, [iteration], n_jobs=1, spill_dir=args.spill_dir)
            astra.save_pairs_fits(pair_store, pairs_iter_path, meta=meta)
            _save_classification_direct(class_store, class_iter_path, meta=meta)
            print(f'dr3 --> done tracer={tracer} zone={zone} iter={iteration:03d}', flush=True)
        finally:
            for store in (pair_store, class_store):
                cleanup = getattr(store, 'cleanup', None)
                if callable(cleanup):
                    cleanup()


if __name__ == '__main__':
    main()