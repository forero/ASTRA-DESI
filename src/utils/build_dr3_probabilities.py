import argparse, os, sys
import numpy as np
from astropy.table import Table


SRC_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)

from desiproc import implement_astra as astra
from desiproc.paths import (classification_path, normalize_release_dir,
                            probability_path)
from releases import dr3


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


def _class_array_from_table(tbl):
    mask = np.asarray(tbl['ISDATA'], dtype=bool)
    out = np.empty(int(mask.sum()), dtype=astra.CLASS_ROW_DTYPE)
    for name in out.dtype.names:
        out[name] = np.asarray(tbl[name][mask])
    return out


def _set_env(name, value):
    old = os.environ.get(name)
    os.environ[name] = value
    return old


def _restore_env(name, old):
    if old is None:
        os.environ.pop(name, None)
    else:
        os.environ[name] = old


def _read_table(path, columns):
    try:
        return Table.read(path, memmap=True, include_names=columns)
    except TypeError:
        tbl = Table.read(path, memmap=True)
        return tbl[[col for col in columns if col in tbl.colnames]]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw-out', required=True)
    parser.add_argument('--class-out', required=True)
    parser.add_argument('--zone', required=True, choices=dr3.DEFAULT_ZONES)
    parser.add_argument('--tracer', required=True)
    parser.add_argument('--n-random', type=int, default=100)
    parser.add_argument('--spill-dir', default=None)
    parser.add_argument('--r-lower', type=float, default=-0.25)
    parser.add_argument('--r-med', type=float, default=0.25)
    parser.add_argument('--r-upper', type=float, default=0.65)
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args()

    if args.r_lower >= 0 or args.r_upper <= 0 or not (args.r_lower < args.r_med < args.r_upper):
        raise ValueError('r thresholds must satisfy r-lower < r-med < r-upper with r-lower < 0 < r-upper.')

    tracer = _normalise_tracer(args.tracer)
    zone = str(args.zone).upper()
    class_root = normalize_release_dir(args.class_out)
    if args.spill_dir:
        os.makedirs(args.spill_dir, exist_ok=True)

    dr3.register_tracer_mapping(dr3.TRACER_IDS, dr3.TRACER_FULL_LABELS)

    class_base = classification_path(class_root, zone, tracer)
    prob_base = probability_path(class_root, zone, tracer)
    prob_iterdata = astra._split_iter_path(prob_base, 'data')
    os.makedirs(os.path.dirname(prob_iterdata), exist_ok=True)

    if os.path.exists(prob_iterdata) and not args.force:
        print(f'[dr3-prob] skip existing {prob_iterdata}', flush=True)
        return

    raw_path = None
    for iteration in range(args.n_random):
        candidate = dr3.raw_iteration_path(args.raw_out, zone, tracer, iteration)
        if os.path.exists(candidate):
            raw_path = candidate
            break
    if raw_path is None:
        raise FileNotFoundError(f'No raw iteration file found for {tracer} {zone} in {args.raw_out}')
    raw_table = _read_table(raw_path, ['TARGETID', 'RANDITER', 'TRACER_ID', 'TRACERTYPE'])

    class_store = astra.TempTableStore(astra.CLASS_ROW_DTYPE, prefix='dr3_prob_class',
                                       base_dir=args.spill_dir)
    try:
        missing = []
        for iteration in range(args.n_random):
            class_path = astra._split_iter_path(class_base, iteration)
            if not os.path.exists(class_path):
                missing.append(class_path)
                continue
            tbl = _read_table(class_path, list(astra.CLASS_ROW_DTYPE.names))
            arr = _class_array_from_table(tbl)
            class_store.append(arr)
            print(f'[dr3-prob] loaded tracer={tracer} zone={zone} iter={iteration:03d} '
                  f'data_rows={arr.size}', flush=True)

        if missing:
            preview = '\n'.join(missing[:10])
            extra = '' if len(missing) <= 10 else f'\n... and {len(missing) - 10} more'
            raise FileNotFoundError(f'Missing classification shards:\n{preview}{extra}')

        meta = {'ZONE': zone,
                'RELEASE': 'DR3',
                'RLOWER': float(args.r_lower),
                'RMED': float(args.r_med),
                'RUPPER': float(args.r_upper)}
        old_split = _set_env('ASTRA_PROB_SPLIT_ITER', '1')
        old_skip = _set_env('ASTRA_PROB_SKIP_COMBINED', '1')
        try:
            astra.save_probability_fits(class_store, raw_table=raw_table,
                                        output_path=prob_base,
                                        r_lower=args.r_lower,
                                        r_med=args.r_med,
                                        r_upper=args.r_upper,
                                        meta=meta)
        finally:
            _restore_env('ASTRA_PROB_SPLIT_ITER', old_split)
            _restore_env('ASTRA_PROB_SKIP_COMBINED', old_skip)
        print(f'[dr3-prob] wrote {prob_iterdata}', flush=True)
    finally:
        class_store.cleanup()


if __name__ == '__main__':
    main()