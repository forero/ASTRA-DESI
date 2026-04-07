import os
import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
plt.style.use('dark_background')

try:
    from .io_common import (discover_classification_realizations, discover_raw_catalog,
                            find_col, get_columns, iter_fits_chunks, safe_upper,
                            tracer_mask)
except ImportError:
    from io_common import (discover_classification_realizations, discover_raw_catalog,
                           find_col, get_columns, iter_fits_chunks, safe_upper,
                           tracer_mask)

ENV_NAMES = ['Void', 'Sheet', 'Filament', 'Knot']


def setup_style():
    plt.rcParams.update({'text.usetex': True})


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
    out[np.isfinite(r) & (r >= -1.0) & (r <= -0.25)] = 0
    out[np.isfinite(r) & (r > -0.25) & (r <= 0.25)] = 1
    out[np.isfinite(r) & (r > 0.25) & (r <= 0.65)] = 2
    out[np.isfinite(r) & (r > 0.65) & (r <= 1.0)] = 3
    return out


def discover_class_files(base, tracer, zone):
    return discover_classification_realizations(base, tracer, zone)


def discover_raw_file(base, tracer, zone):
    return discover_raw_catalog(base, tracer, zone)


def load_z_maps(raw_path, tracer, chunk_rows=500_000):
    cols = get_columns(raw_path)
    tid_col = find_col(cols, ('TARGETID', 'targetid'))
    z_col = find_col(cols, ('Z', 'z'))
    randiter_col = find_col(cols, ('RANDITER', 'randiter'))
    tracer_col = find_col(cols, ('TRACERTYPE', 'tracertype'))

    if tid_col is None or z_col is None or randiter_col is None:
        raise ValueError()

    wanted = [tid_col, z_col, randiter_col]
    if tracer_col is not None:
        wanted.append(tracer_col)

    z_real = {}
    z_rand = {}

    for chunk in iter_fits_chunks(raw_path, wanted, chunk_rows=chunk_rows):
        tid = np.asarray(chunk[tid_col], dtype=np.int64)
        z = np.asarray(chunk[z_col], dtype=np.float32)
        randiter = np.asarray(chunk[randiter_col], dtype=np.int32)

        mask = np.isfinite(z)
        if tracer_col is not None:
            mask &= tracer_mask(chunk[tracer_col], tracer)

        tid = tid[mask]
        z = z[mask]
        randiter = randiter[mask]

        for t, zz, rr in zip(tid, z, randiter):
            if rr == -1:
                if int(t) not in z_real:
                    z_real[int(t)] = float(zz)
            else:
                key = (int(t), int(rr))
                if key not in z_rand:
                    z_rand[key] = float(zz)

    return z_real, z_rand


def one_iteration_fraction_vs_z(class_path, tracer, z_real, z_rand, z_edges, chunk_rows=500_000):
    cols = get_columns(class_path)

    tid_col = find_col(cols, ('TARGETID', 'targetid'))
    ndata_col = find_col(cols, ('NDATA', 'ndata'))
    nrand_col = find_col(cols, ('NRAND', 'nrand'))
    isdata_col = find_col(cols, ('ISDATA', 'isdata'))
    randiter_col = find_col(cols, ('RANDITER', 'randiter'))
    tracer_col = find_col(cols, ('TRACERTYPE', 'tracertype'))

    if tid_col is None or ndata_col is None or nrand_col is None or isdata_col is None or randiter_col is None:
        raise ValueError()

    wanted = [tid_col, ndata_col, nrand_col, isdata_col, randiter_col]
    if tracer_col is not None:
        wanted.append(tracer_col)

    nbin = len(z_edges) - 1
    counts_obj = np.zeros((4, nbin), dtype=np.int64)
    counts_rand = np.zeros((4, nbin), dtype=np.int64)

    for chunk in iter_fits_chunks(class_path, wanted, chunk_rows=chunk_rows):
        tid = np.asarray(chunk[tid_col], dtype=np.int64)
        ndata = np.asarray(chunk[ndata_col], dtype=np.float32)
        nrand = np.asarray(chunk[nrand_col], dtype=np.float32)
        isdata = np.asarray(chunk[isdata_col]).astype(bool)
        randiter = np.asarray(chunk[randiter_col], dtype=np.int32)
        mask = np.ones(len(tid), dtype=bool)
        if tracer_col is not None:
            mask &= tracer_mask(chunk[tracer_col], tracer)

        if not np.any(mask):
            continue

        tid = tid[mask]
        ndata = ndata[mask]
        nrand = nrand[mask]
        isdata = isdata[mask]
        randiter = randiter[mask]

        z = np.full(len(tid), np.nan, dtype=np.float32)

        m_obj = isdata
        if np.any(m_obj):
            z[m_obj] = np.array([z_real.get(int(t), np.nan) for t in tid[m_obj]], dtype=np.float32)

        m_rand = ~isdata
        if np.any(m_rand):
            z[m_rand] = np.array([z_rand.get((int(t), int(r)), np.nan)
                                  for t, r in zip(tid[m_rand], randiter[m_rand])],
                                 dtype=np.float32)

        valid_z = np.isfinite(z)
        if not np.any(valid_z):
            continue

        ndata = ndata[valid_z]
        nrand = nrand[valid_z]
        isdata = isdata[valid_z]
        z = z[valid_z]

        r = r_from_counts(ndata, nrand)
        env = classify_from_r(r)

        valid = env >= 0
        if not np.any(valid):
            continue

        z = z[valid]
        env = env[valid]
        isdata = isdata[valid]

        zbin = np.digitize(z, z_edges) - 1
        inbin = (zbin >= 0) & (zbin < nbin)

        zbin = zbin[inbin]
        env = env[inbin]
        isdata = isdata[inbin]

        if len(zbin) == 0:
            continue

        for e in range(4):
            m_obj = (env == e) & isdata
            if np.any(m_obj):
                counts_obj[e] += np.bincount(zbin[m_obj], minlength=nbin)

            m_rand = (env == e) & (~isdata)
            if np.any(m_rand):
                counts_rand[e] += np.bincount(zbin[m_rand], minlength=nbin)

    frac_obj = np.full((4, nbin), np.nan, dtype=float)
    frac_rand = np.full((4, nbin), np.nan, dtype=float)

    total_obj = counts_obj.sum(axis=0)
    total_rand = counts_rand.sum(axis=0)

    for j in range(nbin):
        if total_obj[j] > 0:
            frac_obj[:, j] = counts_obj[:, j] / total_obj[j]
        if total_rand[j] > 0:
            frac_rand[:, j] = counts_rand[:, j] / total_rand[j]

    return frac_obj, frac_rand


def zone_mean_fraction_vs_z(base, tracer, zone, z_edges, chunk_rows=500_000,
                            iter_min=None, iter_max=None):
    class_files = discover_class_files(base, tracer, zone)
    raw_file = discover_raw_file(base, tracer, zone)

    if raw_file is None or len(class_files) == 0:
        return None

    if iter_min is not None:
        class_files = [(it, p) for it, p in class_files if it is None or it >= iter_min]
    if iter_max is not None:
        class_files = [(it, p) for it, p in class_files if it is None or it <= iter_max]

    if len(class_files) == 0:
        return None

    z_real, z_rand = load_z_maps(raw_file, tracer=tracer, chunk_rows=chunk_rows)

    frac_obj_list = []
    frac_rand_list = []

    for it, path in class_files:
        frac_obj, frac_rand = one_iteration_fraction_vs_z(path,
                                                          tracer,
                                                          z_real,
                                                          z_rand,
                                                          z_edges,
                                                          chunk_rows=chunk_rows)
        frac_obj_list.append(frac_obj)
        frac_rand_list.append(frac_rand)

    frac_obj_mean = np.nanmean(np.stack(frac_obj_list, axis=0), axis=0)
    frac_rand_mean = np.nanmean(np.stack(frac_rand_list, axis=0), axis=0)

    return {'zone': safe_upper(zone),
            'n_iter': len(class_files),
            'object_mean': frac_obj_mean,
            'random_mean': frac_rand_mean}


def plot_count_fraction_vs_z(base, zones, outdir, zmin=0.0, zmax=3.5, zbins=20,
                             chunk_rows=500_000, iter_min=None, iter_max=None):
    colors = {'BGS': 'crimson',
              'LRG': 'green',
              'ELG': 'darkorange',
              'QSO': 'deepskyblue'}

    tracers = ['BGS', 'LRG', 'ELG', 'QSO']

    z_edges = np.linspace(zmin, zmax, zbins + 1)
    z_centers = 0.5 * (z_edges[:-1] + z_edges[1:])

    fig, axes = plt.subplots(2, 4, figsize=(16, 7), sharex=True,
                             gridspec_kw={'hspace': 0.0, 'wspace': 0.08})

    legend_handles = []

    for tracer in tracers:
        zone_results = []
        for zone in zones:
            zres = zone_mean_fraction_vs_z(base=base, tracer=tracer,
                                           zone=zone, z_edges=z_edges,
                                           chunk_rows=chunk_rows,
                                           iter_min=iter_min,
                                           iter_max=iter_max)
            if zres is not None:
                zone_results.append(zres)

        if len(zone_results) == 0:
            continue

        color = colors[tracer]

        obj_zone = np.stack([z['object_mean'] for z in zone_results], axis=0)
        rand_zone = np.stack([z['random_mean'] for z in zone_results], axis=0)

        obj_mean = np.nanmean(obj_zone, axis=0)
        rand_mean = np.nanmean(rand_zone, axis=0)

        obj_std = np.nanstd(obj_zone, axis=0, ddof=1) if obj_zone.shape[0] > 1 else np.zeros_like(obj_mean)
        rand_std = np.nanstd(rand_zone, axis=0, ddof=1) if rand_zone.shape[0] > 1 else np.zeros_like(rand_mean)

        h, = axes[0, 0].plot([], [], color=color, lw=1.7, marker='o', ms=2.2, label=tracer)
        legend_handles.append(h)

        for ienv in range(4):
            ax = axes[0, ienv]
            y = 100.0 * obj_mean[ienv]
            yerr = 100.0 * obj_std[ienv]
            m = np.isfinite(y)

            ax.plot(z_centers[m], y[m], color=color, lw=1.2, marker='o', ms=2.2)
            ax.fill_between(z_centers[m], y[m] - yerr[m], y[m] + yerr[m],
                            color=color, alpha=0.22, linewidth=0)

            ax = axes[1, ienv]
            y = 100.0 * rand_mean[ienv]
            yerr = 100.0 * rand_std[ienv]
            m = np.isfinite(y)

            ax.plot(z_centers[m], y[m], color=color, lw=1.2, marker='o', ms=2.2)
            ax.fill_between(z_centers[m], y[m] - yerr[m], y[m] + yerr[m],
                            color=color, alpha=0.22, linewidth=0)

    for irow in range(2):
        for icol in range(4):
            ax = axes[irow, icol]
            ax.grid(lw=0.3, alpha=0.5)
            ax.set_xlim(zmin, zmax)

    for ienv, env in enumerate(ENV_NAMES):
        axes[0, ienv].set_title(env, pad=10)

    axes[0, 0].text(-0.18, 0.5, 'Object', transform=axes[0, 0].transAxes,
                    rotation=90, va='center', ha='center')
    axes[1, 0].text(-0.18, 0.5, 'Random', transform=axes[1, 0].transAxes,
                    rotation=90, va='center', ha='center')

    fig.text(0.03, 0.5, r'Count Fraction (\%)', va='center', rotation='vertical')
    fig.text(0.5, 0.04, r'Redshift ($z$)', ha='center')

    if legend_handles:
        leg = fig.legend(handles=legend_handles, loc='upper center',
                         bbox_to_anchor=(0.5, 0.98), ncol=4, frameon=True)

    fig.tight_layout(rect=[0.06, 0.08, 0.98, 0.88])

    outpath = Path(outdir) / 'count_fraction_vs_redshift_horizontal.png'
    fig.savefig(outpath, dpi=250, bbox_inches='tight')
    plt.close(fig)
    print(f'\n--------- {outpath}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base', required=True)
    parser.add_argument('--zones', nargs='+', required=True)
    parser.add_argument('--outdir', required=True)
    parser.add_argument('--zmin', type=float, default=0.0)
    parser.add_argument('--zmax', type=float, default=3.5)
    parser.add_argument('--zbins', type=int, default=20)
    parser.add_argument('--chunk-rows', type=int, default=500_000)
    parser.add_argument('--iter-min', type=int, default=None)
    parser.add_argument('--iter-max', type=int, default=None)
    args = parser.parse_args()

    setup_style()
    os.makedirs(args.outdir, exist_ok=True)

    plot_count_fraction_vs_z(base=args.base, zones=args.zones,
                             outdir=args.outdir, zmin=args.zmin,
                             zmax=args.zmax, zbins=args.zbins,
                             chunk_rows=args.chunk_rows,
                             iter_min=args.iter_min,
                             iter_max=args.iter_max)


if __name__ == '__main__':
    main()