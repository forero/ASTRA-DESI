import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
try:
    from .io_common import (discover_classification_realizations, find_col,
                            get_columns, iter_fits_chunks, safe_upper,
                            tracer_mask, tracer_name_variants)
except ImportError:
    from io_common import (discover_classification_realizations, find_col,
                           get_columns, iter_fits_chunks, safe_upper,
                           tracer_mask, tracer_name_variants)


R_THRESHOLDS = np.array([-0.25, 0.25, 0.65], dtype=np.float32)


def r_from_counts(ndata, nrand):
    ndata = np.asarray(ndata, dtype=np.float32)
    nrand = np.asarray(nrand, dtype=np.float32)
    denom = ndata + nrand
    r = np.full_like(denom, np.nan, dtype=np.float32)
    valid = np.isfinite(denom) & (denom > 0)
    r[valid] = (ndata[valid] - nrand[valid]) / denom[valid]
    return r


def classify_from_r(r):
    out = np.full(r.shape, -1, dtype=np.int8)

    out[np.isfinite(r) & (r >= -1.0) & (r <= -0.25)] = 0 # Void
    out[np.isfinite(r) & (r > -0.25) & (r <= 0.25)] = 1 # Sheet
    out[np.isfinite(r) & (r > 0.25) & (r <= 0.65)] = 2 # Filament
    out[np.isfinite(r) & (r > 0.65) & (r <= 1.0)] = 3 # Knot

    return out


def discover_files(base, tracer, zone):
    return discover_classification_realizations(base, tracer, zone)


def _path_has_tracer_token(path, tracer):
    base = os.path.basename(path)
    for suffix in ('.fits.gz', '.fits'):
        if base.endswith(suffix):
            base = base[:-len(suffix)]
            break
    base = safe_upper(base)
    for token in tracer_name_variants(tracer):
        token = safe_upper(token)
        if f'_{token}_' in base:
            return True
    return False


def _use_tracer_filter(path, tracer, tracer_col, mode):
    if tracer_col is None or mode == 'never':
        return False
    if mode == 'always':
        return True
    return not _path_has_tracer_token(path, tracer)


def _accumulate_counts(ndata, nrand, isdata, counts_obj, counts_rand):
    denom = ndata + nrand
    valid = np.isfinite(ndata) & np.isfinite(nrand) & (denom > 0)
    if not np.any(valid):
        return

    valid_idx = np.flatnonzero(valid)
    r = (ndata[valid] - nrand[valid]) / denom[valid]
    valid_r = np.isfinite(r) & (r >= -1.0) & (r <= 1.0)
    if not np.any(valid_r):
        return

    env = np.searchsorted(R_THRESHOLDS, r[valid_r], side='left')
    data_mask = isdata[valid_idx[valid_r]]

    if np.any(data_mask):
        counts_obj += np.bincount(env[data_mask], minlength=4)[:4].astype(np.int64)
    if np.any(~data_mask):
        counts_rand += np.bincount(env[~data_mask], minlength=4)[:4].astype(np.int64)


def one_iteration_fractions(path, tracer, chunk_rows=500_000, tracer_filter='auto'):
    cols = get_columns(path)

    ndata_col = find_col(cols, ('NDATA', 'ndata'))
    nrand_col = find_col(cols, ('NRAND', 'nrand'))
    isdata_col = find_col(cols, ('ISDATA', 'isdata'))
    tracer_col = find_col(cols, ('TRACERTYPE', 'tracertype'))

    if ndata_col is None or nrand_col is None or isdata_col is None:
        raise ValueError()

    filter_tracer = _use_tracer_filter(path, tracer, tracer_col, tracer_filter)

    wanted = [ndata_col, nrand_col, isdata_col]
    if filter_tracer:
        wanted.append(tracer_col)

    counts_obj = np.zeros(4, dtype=np.int64)
    counts_rand = np.zeros(4, dtype=np.int64)

    for chunk in iter_fits_chunks(path, wanted, chunk_rows=chunk_rows):
        ndata = np.asarray(chunk[ndata_col], dtype=np.float32)
        nrand = np.asarray(chunk[nrand_col], dtype=np.float32)
        isdata = np.asarray(chunk[isdata_col]).astype(bool)

        if filter_tracer:
            mask = tracer_mask(chunk[tracer_col], tracer)
            if not np.any(mask):
                continue
            ndata = ndata[mask]
            nrand = nrand[mask]
            isdata = isdata[mask]

        _accumulate_counts(ndata, nrand, isdata, counts_obj, counts_rand)

    frac_obj = counts_obj / counts_obj.sum() if counts_obj.sum() > 0 else np.full(4, np.nan)
    frac_rand = counts_rand / counts_rand.sum() if counts_rand.sum() > 0 else np.full(4, np.nan)

    return frac_obj, frac_rand


def _one_iteration_task(args):
    path, tracer, chunk_rows, tracer_filter = args
    return one_iteration_fractions(path, tracer=tracer, chunk_rows=chunk_rows,
                                   tracer_filter=tracer_filter)


def _fraction_results(files, tracer, chunk_rows, tracer_filter, workers, progress_label=None):
    tasks = [(path, tracer, chunk_rows, tracer_filter) for _, path in files]

    if workers <= 1 or len(tasks) <= 1:
        for idx, task in enumerate(tasks, 1):
            if progress_label is not None:
                print(f'[count_fraction] {progress_label}: {idx}/{len(tasks)}',
                      file=sys.stderr, flush=True)
            yield _one_iteration_task(task)
        return

    max_workers = min(int(workers), len(tasks))
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        for idx, result in enumerate(executor.map(_one_iteration_task, tasks), 1):
            if progress_label is not None:
                print(f'[count_fraction] {progress_label}: {idx}/{len(tasks)}',
                      file=sys.stderr, flush=True)
            yield result


def zone_mean_fractions(base, tracer, zone, chunk_rows=500_000, iter_min=None,
                        iter_max=None, workers=1, tracer_filter='auto',
                        progress=False):
    files = discover_files(base, tracer, zone)

    if iter_min is not None:
        files = [(it, p) for it, p in files if it is None or it >= iter_min]
    if iter_max is not None:
        files = [(it, p) for it, p in files if it is None or it <= iter_max]

    if len(files) == 0:
        return None

    obj_list = []
    rand_list = []

    progress_label = f'{tracer} {safe_upper(zone)}' if progress else None
    for frac_obj, frac_rand in _fraction_results(files, tracer, chunk_rows,
                                                 tracer_filter, workers,
                                                 progress_label=progress_label):
        obj_list.append(frac_obj)
        rand_list.append(frac_rand)

    return {'zone': safe_upper(zone),
            'n_iter': len(files),
            'object_mean': np.nanmean(np.vstack(obj_list), axis=0),
            'random_mean': np.nanmean(np.vstack(rand_list), axis=0)}


def build_count_fraction_table(base, zones, tracers, chunk_rows=500_000,
                               iter_min=None, iter_max=None, workers=1,
                               tracer_filter='auto', progress=False):
    rows = []

    for tracer in tracers:
        zone_results = []
        for zone in zones:
            zres = zone_mean_fractions(base=base,
                                       tracer=tracer,
                                       zone=zone,
                                       chunk_rows=chunk_rows,
                                       iter_min=iter_min,
                                       iter_max=iter_max,
                                       workers=workers,
                                       tracer_filter=tracer_filter,
                                       progress=progress)
            if zres is not None:
                zone_results.append(zres)

        if len(zone_results) == 0:
            continue

        obj_zone = np.vstack([z['object_mean'] for z in zone_results])
        rand_zone = np.vstack([z['random_mean'] for z in zone_results])

        obj_mean = np.nanmean(obj_zone, axis=0)
        rand_mean = np.nanmean(rand_zone, axis=0)

        obj_std = np.nanstd(obj_zone, axis=0, ddof=1) if obj_zone.shape[0] > 1 else np.zeros(4)
        rand_std = np.nanstd(rand_zone, axis=0, ddof=1) if rand_zone.shape[0] > 1 else np.zeros(4)

        rows.append({'Catalog': 'Object',
                     'Tracer': tracer,
                     'Void': f'{100*obj_mean[0]:.2f} ± {100*obj_std[0]:.2f}',
                     'Sheet': f'{100*obj_mean[1]:.2f} ± {100*obj_std[1]:.2f}',
                     'Filament': f'{100*obj_mean[2]:.2f} ± {100*obj_std[2]:.2f}',
                     'Knot': f'{100*obj_mean[3]:.2f} ± {100*obj_std[3]:.2f}'})

        rows.append({'Catalog': 'Random',
                     'Tracer': tracer,
                     'Void': f'{100*rand_mean[0]:.2f} ± {100*rand_std[0]:.2f}',
                     'Sheet': f'{100*rand_mean[1]:.2f} ± {100*rand_std[1]:.2f}',
                     'Filament': f'{100*rand_mean[2]:.2f} ± {100*rand_std[2]:.2f}',
                     'Knot': f'{100*rand_mean[3]:.2f} ± {100*rand_std[3]:.2f}'})

    return pd.DataFrame(rows, columns=['Catalog', 'Tracer', 'Void', 'Sheet', 'Filament', 'Knot'])


def _default_workers():
    value = os.environ.get('COUNT_FRACTION_WORKERS')
    if not value:
        return 1
    try:
        return max(1, int(value))
    except ValueError:
        return 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base', required=True)
    parser.add_argument('--zones', nargs='+', required=True)
    parser.add_argument('--tracers', nargs='+', default=['BGS', 'LRG', 'ELG', 'QSO'])
    parser.add_argument('--chunk-rows', type=int, default=500_000)
    parser.add_argument('--iter-min', type=int, default=None)
    parser.add_argument('--iter-max', type=int, default=None)
    parser.add_argument('--workers', type=int, default=_default_workers(),
                        help='Number of iteration files to process in parallel.')
    parser.add_argument('--tracer-filter', choices=['auto', 'always', 'never'],
                        default='auto',
                        help='Filter rows by TRACERTYPE. auto skips this for tracer-specific files.')
    parser.add_argument('--progress', action='store_true',
                        help='Print per-tracer/zone progress to stderr.')
    args = parser.parse_args()

    df = build_count_fraction_table(base=args.base,
                                    zones=args.zones,
                                    tracers=args.tracers,
                                    chunk_rows=args.chunk_rows,
                                    iter_min=args.iter_min,
                                    iter_max=args.iter_max,
                                    workers=args.workers,
                                    tracer_filter=args.tracer_filter,
                                    progress=args.progress)

    print('')
    print(df.to_string(index=False))
    print('')


if __name__ == '__main__':
    main()