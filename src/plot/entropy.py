import os, json, argparse
from pathlib import Path

import numpy as np

try:
    from .io_common import (discover_classification_realizations, find_col,
                            get_columns, iter_fits_chunks, parse_iter, safe_upper,
                            tracer_mask)
except ImportError:
    from io_common import (discover_classification_realizations, find_col,
                           get_columns, iter_fits_chunks, parse_iter, safe_upper,
                           tracer_mask)


ENV_NAMES = ('void', 'sheet', 'filament', 'knot')
LOG2_4 = np.log2(4.0)


def normalized_shannon_from_probs(P):
    with np.errstate(divide='ignore', invalid='ignore'):
        terms = P * np.log2(P)
    terms[~np.isfinite(terms)] = 0.0
    H = -np.sum(terms, axis=1) / LOG2_4
    return H.astype(np.float32, copy=False)


def normalized_shannon_from_counts(counts):
    counts = np.asarray(counts, dtype=np.float64)
    total = counts.sum()
    if total <= 0:
        return np.nan
    p = counts / total
    with np.errstate(divide='ignore', invalid='ignore'):
        terms = p * np.log2(p)
    terms[~np.isfinite(terms)] = 0.0
    return float(-terms.sum() / LOG2_4)


def r_from_counts(ndata, nrand):
    ndata = np.asarray(ndata, dtype=np.float32)
    nrand = np.asarray(nrand, dtype=np.float32)
    denom = ndata + nrand
    r = np.full_like(denom, np.nan, dtype=np.float32)
    valid = np.isfinite(denom) & (denom > 0)
    r[valid] = (ndata[valid] - nrand[valid]) / denom[valid]
    return r


def classify_from_r(r):
    r = np.asarray(r, dtype=np.float32)
    out = np.full(r.shape, -1, dtype=np.int8)

    m_void = np.isfinite(r) & (r >= -1.0) & (r <= -0.25)
    m_sheet = np.isfinite(r) & (r > -0.25) & (r <= 0.25)
    m_fil = np.isfinite(r) & (r > 0.25) & (r <= 0.65)
    m_knot = np.isfinite(r) & (r > 0.65) & (r <= 1.0)

    out[m_void] = 0
    out[m_sheet] = 1
    out[m_fil] = 2
    out[m_knot] = 3

    return out


def discover_classification_files(base, tracer, zone):
    realized = discover_classification_realizations(base, tracer, zone)
    return [path for _, path in realized]


def collect_targetids_and_population(class_files, tracer, chunk_rows=500_000):
    targetid_union = set()
    iterations = []
    env_counts_per_iter = []

    for path in class_files:
        cols = get_columns(path)

        tid_col = find_col(cols, ('TARGETID', 'targetid', 'TargetID'))
        ndata_col = find_col(cols, ('NDATA', 'ndata'))
        nrand_col = find_col(cols, ('NRAND', 'nrand'))
        isdata_col = find_col(cols, ('ISDATA', 'isdata'))
        tracer_col = find_col(cols, ('TRACERTYPE', 'tracertype'))

        if tid_col is None or ndata_col is None or nrand_col is None:
            raise ValueError(f'{path}: no encontré TARGETID, NDATA o NRAND')

        wanted = [tid_col, ndata_col, nrand_col]
        if isdata_col is not None:
            wanted.append(isdata_col)
        if tracer_col is not None:
            wanted.append(tracer_col)

        env_counts = np.zeros(4, dtype=np.int64)

        for chunk in iter_fits_chunks(path, wanted, chunk_rows=chunk_rows):
            tids = np.asarray(chunk[tid_col], dtype=np.int64)
            ndata = np.asarray(chunk[ndata_col], dtype=np.float32)
            nrand = np.asarray(chunk[nrand_col], dtype=np.float32)
            mask = np.ones(len(tids), dtype=bool)

            if isdata_col is not None:
                mask &= np.asarray(chunk[isdata_col]).astype(bool)
            if tracer_col is not None:
                mask &= tracer_mask(chunk[tracer_col], tracer)

            tids = tids[mask]
            ndata = ndata[mask]
            nrand = nrand[mask]

            if len(tids) == 0:
                continue

            r = r_from_counts(ndata, nrand)
            env_idx = classify_from_r(r)

            valid = env_idx >= 0
            tids = tids[valid]
            env_idx = env_idx[valid]

            if len(tids) == 0:
                continue

            targetid_union.update(tids.tolist())
            env_counts += np.bincount(env_idx, minlength=4).astype(np.int64)

        it = parse_iter(path)
        iterations.append(-1 if it is None else int(it))
        env_counts_per_iter.append(env_counts)

    iterations = np.asarray(iterations, dtype=np.int32)
    env_counts_per_iter = np.asarray(env_counts_per_iter, dtype=np.int64)

    order = np.argsort(iterations)
    iterations = iterations[order]
    env_counts_per_iter = env_counts_per_iter[order]

    return targetid_union, iterations, env_counts_per_iter


def build_object_class_counts(class_files, targetid_union, tracer, chunk_rows=500_000):
    targetids = np.array(sorted(targetid_union), dtype=np.int64)
    index_map = {tid: i for i, tid in enumerate(targetids)}

    counts = np.zeros((len(targetids), 4), dtype=np.uint16)

    for path in class_files:
        cols = get_columns(path)

        tid_col = find_col(cols, ('TARGETID', 'targetid', 'TargetID'))
        ndata_col = find_col(cols, ('NDATA', 'ndata'))
        nrand_col = find_col(cols, ('NRAND', 'nrand'))
        isdata_col = find_col(cols, ('ISDATA', 'isdata'))
        tracer_col = find_col(cols, ('TRACERTYPE', 'tracertype'))

        wanted = [tid_col, ndata_col, nrand_col]
        if isdata_col is not None:
            wanted.append(isdata_col)
        if tracer_col is not None:
            wanted.append(tracer_col)

        for chunk in iter_fits_chunks(path, wanted, chunk_rows=chunk_rows):
            tids = np.asarray(chunk[tid_col], dtype=np.int64)
            ndata = np.asarray(chunk[ndata_col], dtype=np.float32)
            nrand = np.asarray(chunk[nrand_col], dtype=np.float32)
            mask = np.ones(len(tids), dtype=bool)

            if isdata_col is not None:
                mask &= np.asarray(chunk[isdata_col]).astype(bool)
            if tracer_col is not None:
                mask &= tracer_mask(chunk[tracer_col], tracer)

            tids = tids[mask]
            ndata = ndata[mask]
            nrand = nrand[mask]

            if len(tids) == 0:
                continue

            r = r_from_counts(ndata, nrand)
            env_idx = classify_from_r(r)

            valid = env_idx >= 0
            tids = tids[valid]
            env_idx = env_idx[valid]

            if len(tids) == 0:
                continue

            obj_idx = np.fromiter((index_map[t] for t in tids), dtype=np.int64, count=len(tids))
            np.add.at(counts, (obj_idx, env_idx), 1)

    return targetids, counts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base', required=True)
    parser.add_argument('--tracer', required=True)
    parser.add_argument('--zone', required=True)
    parser.add_argument('--outdir', required=True)
    parser.add_argument('--chunk-rows', type=int, default=500_000)
    parser.add_argument('--hmin', type=float, default=0.0)
    parser.add_argument('--hmax', type=float, default=1.0)
    parser.add_argument('--hbins', type=int, default=100)
    parser.add_argument('--iter-min', type=int, default=None)
    parser.add_argument('--iter-max', type=int, default=None)

    args = parser.parse_args()

    tracer = safe_upper(args.tracer)
    zone = safe_upper(args.zone)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    class_files = discover_classification_files(args.base, tracer, zone)

    if args.iter_min is not None:
        class_files = [f for f in class_files
                       if parse_iter(f) is None or parse_iter(f) >= args.iter_min]
    if args.iter_max is not None:
        class_files = [f for f in class_files
                       if parse_iter(f) is None or parse_iter(f) <= args.iter_max]

    if len(class_files) == 0:
        raise RuntimeError(f'No files for tracer={tracer}, zone={zone}')

    print(f'{len(class_files)} files')

    targetid_union, iterations, env_counts_per_iter = collect_targetids_and_population(
        class_files, tracer=tracer, chunk_rows=args.chunk_rows)

    print(f'N real data: {len(targetid_union)}')

    H_pop = np.array([normalized_shannon_from_counts(c) for c in env_counts_per_iter],
                     dtype=np.float64)

    targetids, obj_counts = build_object_class_counts(
        class_files, targetid_union, tracer=tracer, chunk_rows=args.chunk_rows)

    n_iter_per_obj = obj_counts.sum(axis=1).astype(np.int32)

    P_obj = np.zeros_like(obj_counts, dtype=np.float32)
    valid = n_iter_per_obj > 0
    P_obj[valid] = obj_counts[valid] / n_iter_per_obj[valid, None]

    H_obj = np.full(len(targetids), np.nan, dtype=np.float32)
    H_obj[valid] = normalized_shannon_from_probs(P_obj[valid])

    bin_edges = np.linspace(args.hmin, args.hmax, args.hbins + 1, dtype=np.float64)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    hist_counts, _ = np.histogram(H_obj[np.isfinite(H_obj)], bins=bin_edges)

    Hf = H_obj[np.isfinite(H_obj)].astype(np.float64)
    sum_H = np.sum(Hf) if Hf.size else np.nan
    sum_H2 = np.sum(Hf ** 2) if Hf.size else np.nan
    mean_H = np.mean(Hf) if Hf.size else np.nan
    std_H = np.std(Hf, ddof=1) if Hf.size > 1 else np.nan

    out_npz = outdir / f'{tracer.lower()}_{zone.lower()}_entropy_from_r_classification.npz'
    np.savez_compressed(out_npz, tracer=np.array(tracer), zone=np.array(zone), iterations=iterations,
                        env_names=np.array(ENV_NAMES), env_counts_per_iter=env_counts_per_iter,
                        H_pop=H_pop, targetids=targetids, obj_class_counts=obj_counts,
                        n_iter_per_obj=n_iter_per_obj, H_obj=H_obj, bin_edges=bin_edges,
                        bin_centers=bin_centers, hist_counts=hist_counts, sum_H=np.array(sum_H),
                        sum_H2=np.array(sum_H2), mean_H=np.array(mean_H), std_H=np.array(std_H))

    meta = {'tracer': tracer, 'zone': zone, 'n_iterations': int(len(iterations)),
            'n_objects': int(len(targetids)), 'classification_definition': {'void': [-1.0, -0.25],
                                                                            'sheet': [-0.25, 0.25],
                                                                            'filament': [0.25, 0.65],
                                                                            'knot': [0.65, 1.0]},
        'r_definition': 'r = (NDATA - NRAND) / (NDATA + NRAND)',
        'entropy_object_definition': 'H_i = -sum_w p_iw log2(p_iw) / log2(4), with p_iw = n_iw / N_iter_i',
        'entropy_population_definition': 'H_pop(r) = -sum_w f_w(r) log2(f_w(r)) / log2(4)',
        'object_probabilities_source': 'classification counts across iterations using r thresholds',
        'population_source': 'classification counts per iteration using r thresholds',
        'chunk_rows': int(args.chunk_rows),
        'output_npz': str(out_npz)}

    out_json = outdir / f'{tracer.lower()}_{zone.lower()}_entropy_from_r_classification.meta.json'
    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2)

    print(out_npz)
    print(out_json)


if __name__ == '__main__':
    main()