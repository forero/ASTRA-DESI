import argparse

import numpy as np
import pandas as pd
try:
    from .io_common import (discover_classification_realizations, find_col,
                            get_columns, iter_fits_chunks, safe_upper, tracer_mask)
except ImportError:
    from io_common import (discover_classification_realizations, find_col,
                           get_columns, iter_fits_chunks, safe_upper, tracer_mask)


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


def one_iteration_fractions(path, tracer, chunk_rows=500_000):
    cols = get_columns(path)

    ndata_col = find_col(cols, ('NDATA', 'ndata'))
    nrand_col = find_col(cols, ('NRAND', 'nrand'))
    isdata_col = find_col(cols, ('ISDATA', 'isdata'))
    tracer_col = find_col(cols, ('TRACERTYPE', 'tracertype'))

    if ndata_col is None or nrand_col is None or isdata_col is None:
        raise ValueError()

    wanted = [ndata_col, nrand_col, isdata_col]
    if tracer_col is not None:
        wanted.append(tracer_col)

    counts_obj = np.zeros(4, dtype=np.int64)
    counts_rand = np.zeros(4, dtype=np.int64)

    for chunk in iter_fits_chunks(path, wanted, chunk_rows=chunk_rows):
        ndata = np.asarray(chunk[ndata_col], dtype=np.float32)
        nrand = np.asarray(chunk[nrand_col], dtype=np.float32)
        isdata = np.asarray(chunk[isdata_col]).astype(bool)

        mask = np.ones(len(ndata), dtype=bool)
        if tracer_col is not None:
            mask &= tracer_mask(chunk[tracer_col], tracer)

        if not np.any(mask):
            continue

        ndata = ndata[mask]
        nrand = nrand[mask]
        isdata = isdata[mask]

        r = r_from_counts(ndata, nrand)
        env = classify_from_r(r)

        valid = env >= 0
        if np.any(valid & isdata):
            counts_obj += np.bincount(env[valid & isdata], minlength=4).astype(np.int64)
        if np.any(valid & (~isdata)):
            counts_rand += np.bincount(env[valid & (~isdata)], minlength=4).astype(np.int64)

    frac_obj = counts_obj / counts_obj.sum() if counts_obj.sum() > 0 else np.full(4, np.nan)
    frac_rand = counts_rand / counts_rand.sum() if counts_rand.sum() > 0 else np.full(4, np.nan)

    return frac_obj, frac_rand


def zone_mean_fractions(base, tracer, zone, chunk_rows=500_000, iter_min=None, iter_max=None):
    files = discover_files(base, tracer, zone)

    if iter_min is not None:
        files = [(it, p) for it, p in files if it is None or it >= iter_min]
    if iter_max is not None:
        files = [(it, p) for it, p in files if it is None or it <= iter_max]

    if len(files) == 0:
        return None

    obj_list = []
    rand_list = []

    for it, path in files:
        frac_obj, frac_rand = one_iteration_fractions(path, tracer=tracer, chunk_rows=chunk_rows)
        obj_list.append(frac_obj)
        rand_list.append(frac_rand)

    return {'zone': safe_upper(zone),
            'n_iter': len(files),
            'object_mean': np.nanmean(np.vstack(obj_list), axis=0),
            'random_mean': np.nanmean(np.vstack(rand_list), axis=0)}


def build_count_fraction_table(base, zones, tracers, chunk_rows=500_000, iter_min=None, iter_max=None):
    rows = []

    for tracer in tracers:
        zone_results = []
        for zone in zones:
            zres = zone_mean_fractions(base=base,
                                       tracer=tracer,
                                       zone=zone,
                                       chunk_rows=chunk_rows,
                                       iter_min=iter_min,
                                       iter_max=iter_max)
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base', required=True)
    parser.add_argument('--zones', nargs='+', required=True)
    parser.add_argument('--tracers', nargs='+', default=['BGS', 'LRG', 'ELG', 'QSO'])
    parser.add_argument('--chunk-rows', type=int, default=500_000)
    parser.add_argument('--iter-min', type=int, default=None)
    parser.add_argument('--iter-max', type=int, default=None)
    args = parser.parse_args()

    df = build_count_fraction_table(base=args.base,
                                    zones=args.zones,
                                    tracers=args.tracers,
                                    chunk_rows=args.chunk_rows,
                                    iter_min=args.iter_min,
                                    iter_max=args.iter_max,)

    print('')
    print(df.to_string(index=False))
    print('')


if __name__ == '__main__':
    main()