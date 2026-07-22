import argparse
import csv
from pathlib import Path
import sys

import fitsio
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

if __package__ is None or __package__ == '':
    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from group_finder.read_data import _ang2pix


DEFAULT_BASE_DIR = '/pscratch/sd/v/vtorresg/cosmic-web/dr2'
DEFAULT_OUT_DIR = 'plots/edge_mask_stats/dr2_lrg_iter000'
DEFAULT_CAPS = ('NGC', 'SGC')


def parse_args():
    parser = argparse.ArgumentParser(
        description='DR2 LRG iter0 edge-mask footprint and N_data/pixel stats.'
    )
    parser.add_argument('--base-dir', default=DEFAULT_BASE_DIR)
    parser.add_argument('--out-dir', default=DEFAULT_OUT_DIR)
    parser.add_argument('--tracer', default='LRG')
    parser.add_argument('--iteration', type=int, default=0)
    parser.add_argument('--caps', nargs='+', default=list(DEFAULT_CAPS),
                        choices=list(DEFAULT_CAPS))
    parser.add_argument('--nside', type=int, default=256)
    parser.add_argument('--min-randoms-per-pix', type=int, default=3)
    parser.add_argument('--min-data-per-pix', type=int, default=10)
    parser.add_argument('--chunk-size', type=int, default=500_000)
    parser.add_argument('--ra-bins', type=int, default=360)
    parser.add_argument('--dec-bins', type=int, default=250)
    parser.add_argument('--dec-min', type=float, default=-40.0)
    parser.add_argument('--dec-max', type=float, default=85.0)
    parser.add_argument('--dpi', type=int, default=220)
    return parser.parse_args()


def class_path(base_dir, tracer, cap, iteration):
    tracer_lower = tracer.lower()
    return (Path(base_dir) / 'classification' / tracer_lower / cap.lower() /
            f'zone_{cap}_{tracer}_iter{iteration:03d}.fits.gz')


def raw_path(base_dir, tracer, cap):
    return Path(base_dir) / 'raw' / f'zone_{cap}_{tracer}.fits.gz'


def iter_bounds(base_dir, tracer, cap, iteration):
    cpath = class_path(base_dir, tracer, cap, iteration)
    if not cpath.exists():
        raise FileNotFoundError(cpath)

    n_class = fitsio.FITS(str(cpath))[1].get_nrows()
    if n_class % 2 != 0:
        raise RuntimeError(f'{cpath}: expected data+one-random rows, got odd nrows={n_class}')

    n_data = n_class // 2
    data_start, data_stop = 0, n_data
    random_start = (iteration + 1) * n_data
    random_stop = (iteration + 2) * n_data
    return data_start, data_stop, random_start, random_stop, n_data, n_class


def valid_ra_dec(ra, dec):
    return (np.isfinite(ra) & np.isfinite(dec) &
            (dec >= -90.0) & (dec <= 90.0))


def healpix_counts(path, start, stop, nside, npix, chunk_size):
    hdu = fitsio.FITS(str(path))[1]
    counts = np.zeros(npix, dtype=np.int64)
    n_valid = 0

    for lo in range(start, stop, chunk_size):
        hi = min(lo + chunk_size, stop)
        arr = hdu.read(columns=['RA', 'DEC'], rows=range(lo, hi))
        ra = np.asarray(arr['RA'], dtype=np.float64)
        dec = np.asarray(arr['DEC'], dtype=np.float64)
        valid = valid_ra_dec(ra, dec)
        if not np.any(valid):
            continue

        theta = np.radians(90.0 - dec[valid])
        phi = np.radians(np.mod(ra[valid], 360.0))
        pix = _ang2pix(nside, theta, phi)
        counts += np.bincount(pix, minlength=npix)
        n_valid += int(pix.size)

    return counts, n_valid


def hist2d_from_rows(path, start, stop, ra_edges, dec_edges, nside,
                     chunk_size, keep_hpx=None):
    hdu = fitsio.FITS(str(path))[1]
    hist = np.zeros((len(dec_edges) - 1, len(ra_edges) - 1), dtype=np.int64)
    n_kept = 0

    for lo in range(start, stop, chunk_size):
        hi = min(lo + chunk_size, stop)
        arr = hdu.read(columns=['RA', 'DEC'], rows=range(lo, hi))
        ra = np.asarray(arr['RA'], dtype=np.float64)
        dec = np.asarray(arr['DEC'], dtype=np.float64)
        valid = valid_ra_dec(ra, dec)
        if not np.any(valid):
            continue

        ra = np.mod(ra[valid], 360.0)
        dec = dec[valid]

        if keep_hpx is not None:
            theta = np.radians(90.0 - dec)
            phi = np.radians(ra)
            pix = _ang2pix(nside, theta, phi)
            keep = keep_hpx[pix]
            if not np.any(keep):
                continue
            ra = ra[keep]
            dec = dec[keep]

        chunk_hist, _, _ = np.histogram2d(dec, ra, bins=[dec_edges, ra_edges])
        hist += chunk_hist.astype(np.int64, copy=False)
        n_kept += int(ra.size)

    return hist, n_kept


def percentile_stats(values):
    vals = np.asarray(values, dtype=np.float64)
    p2, p16, p50, p84, p98 = np.percentile(vals, [2.275, 16, 50, 84, 97.725])
    mean = float(np.mean(vals))
    std = float(np.std(vals, ddof=1))
    return {
        'n_pixels': int(vals.size),
        'min': float(np.min(vals)),
        'max': float(np.max(vals)),
        'mean': mean,
        'std': std,
        'median': float(p50),
        'p16': float(p16),
        'p84': float(p84),
        'p2p275': float(p2),
        'p97p725': float(p98),
        'mean_minus_1std': mean - std,
        'mean_plus_1std': mean + std,
        'mean_minus_2std': mean - 2.0 * std,
        'mean_plus_2std': mean + 2.0 * std,
    }


def write_stats_csv(path, stats_by_name, count_summary_by_name=None):
    fields = [
        'sample', 'n_pixels', 'min', 'max', 'mean', 'std', 'median',
        'p16', 'p84', 'p2p275', 'p97p725',
        'mean_minus_1std', 'mean_plus_1std',
        'mean_minus_2std', 'mean_plus_2std',
        'n_data_valid', 'n_data_ge_mean_minus_2std',
        'frac_data_ge_mean_minus_2std', 'n_hpx_ge_mean_minus_2std',
    ]
    count_summary_by_name = count_summary_by_name or {}
    with open(path, 'w', newline='') as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for name, stats in stats_by_name.items():
            row = {'sample': name}
            row.update(stats)
            row.update(count_summary_by_name.get(name, {}))
            writer.writerow(row)


def print_stats(stats_by_name, count_summary_by_name=None):
    count_summary_by_name = count_summary_by_name or {}
    print('')
    print('N_data/N_pixel distribution stats')
    print('1sigma percentile interval = [p16, p84]')
    print('2sigma percentile interval = [p2.275, p97.725]')
    for name, stats in stats_by_name.items():
        print(
            f'{name}: n={stats["n_pixels"]} '
            f'median={stats["median"]:.6g} '
            f'mean={stats["mean"]:.6g} std={stats["std"]:.6g} '
            f'1sigma_pct=[{stats["p16"]:.6g}, {stats["p84"]:.6g}] '
            f'2sigma_pct=[{stats["p2p275"]:.6g}, {stats["p97p725"]:.6g}] '
            f'mean+/-1std=[{stats["mean_minus_1std"]:.6g}, {stats["mean_plus_1std"]:.6g}] '
            f'mean+/-2std=[{stats["mean_minus_2std"]:.6g}, {stats["mean_plus_2std"]:.6g}]'
        )
        counts = count_summary_by_name.get(name)
        if counts:
            print(
                f'  data_valid={counts["n_data_valid"]} '
                f'data_after_mean_minus_2std={counts["n_data_ge_mean_minus_2std"]} '
                f'frac_after_mean_minus_2std={counts["frac_data_ge_mean_minus_2std"]:.6g} '
                f'hpx_after_mean_minus_2std={counts["n_hpx_ge_mean_minus_2std"]}'
            )


def plot_hist2d_2x2(path, ra_edges, dec_edges, histograms, dpi):
    panels = [
        ('NGC_real', histograms['NGC']['real']),
        ('NGC_ge10', histograms['NGC']['ge10']),
        ('SGC_real', histograms['SGC']['real']),
        ('SGC_ge10', histograms['SGC']['ge10']),
    ]
    vmax = int(max(hist.max() for _, hist in panels))
    fig, axes = plt.subplots(
        2, 2, figsize=(10, 8), sharex=True, sharey=True,
        constrained_layout=True,
    )
    mesh = None
    for ax, (_, hist) in zip(axes.ravel(), panels):
        plot_hist = np.ma.masked_where(hist <= 0, hist)
        mesh = ax.pcolormesh(
            ra_edges, dec_edges, plot_hist, cmap='viridis',
            vmin=1, vmax=vmax, shading='auto',
        )
        ax.grid(True, axis='both', alpha=0.22, linewidth=0.6)
        ax.set_axisbelow(False)

    for ax in axes[1, :]:
        ax.set_xlabel('RA [deg]')
    for ax in axes[:, 0]:
        ax.set_ylabel('DEC [deg]')

    fig.colorbar(mesh, ax=axes, label='Counts', shrink=0.92)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def plot_ndata_hist(path, values_by_cap, stats_by_name, dpi):
    vmax = max(int(np.max(vals)) for vals in values_by_cap.values())
    vmin = min(int(np.min(vals)) for vals in values_by_cap.values())
    bins = np.arange(vmin - 0.5, vmax + 1.5, 1.0)

    fig, axes = plt.subplots(
        1, 2, figsize=(8, 4), sharex=True, sharey=True,
        constrained_layout=True,
    )
    for ax, cap in zip(axes, ['NGC', 'SGC']):
        ax.hist(
            values_by_cap[cap], bins=bins, color='royalblue',
            edgecolor='white', linewidth=0.35,
        )
        ax.axvline(
            stats_by_name[cap]['mean_minus_2std'],
            color='crimson', linestyle='--', linewidth=1.7,
        )
        ax.axvline(
            stats_by_name[cap]['mean_plus_2std'],
            color='crimson', linestyle='--', linewidth=1.7,
        )
        ax.grid(True, axis='both', alpha=0.28, linewidth=0.7)
        ax.set_axisbelow(True)
        ax.set_xlabel('N_data/N_pixel')

    axes[0].set_ylabel('Counts')
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def main():
    args = parse_args()
    base_dir = Path(args.base_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    npix = 12 * args.nside * args.nside
    ra_edges = np.linspace(0.0, 360.0, args.ra_bins + 1)
    dec_edges = np.linspace(args.dec_min, args.dec_max, args.dec_bins + 1)

    histograms = {}
    values_by_cap = {}
    runtime_rows = {}
    stats_by_name = {}
    count_summary_by_name = {}
    data_counts_by_cap = {}
    real_hpx_mask_by_cap = {}

    for cap in args.caps:
        rpath = raw_path(base_dir, args.tracer, cap)
        if not rpath.exists():
            raise FileNotFoundError(rpath)

        data_start, data_stop, rand_start, rand_stop, n_data, n_class = iter_bounds(
            base_dir, args.tracer, cap, args.iteration,
        )

        print(f'[{cap}] reading data HEALPix counts...')
        data_counts, n_data_valid = healpix_counts(
            rpath, data_start, data_stop, args.nside, npix, args.chunk_size,
        )
        print(f'[{cap}] reading random HEALPix counts...')
        rand_counts, n_rand_valid = healpix_counts(
            rpath, rand_start, rand_stop, args.nside, npix, args.chunk_size,
        )

        real_hpx_mask = rand_counts >= int(args.min_randoms_per_pix)
        clean_hpx_mask = real_hpx_mask & (data_counts >= int(args.min_data_per_pix))
        data_counts_by_cap[cap] = data_counts
        real_hpx_mask_by_cap[cap] = real_hpx_mask
        ndata_per_mask_pixel = data_counts[real_hpx_mask].astype(np.int32, copy=False)
        values_by_cap[cap] = ndata_per_mask_pixel
        stats_by_name[cap] = percentile_stats(ndata_per_mask_pixel)
        minus2_cut = stats_by_name[cap]['mean_minus_2std']
        minus2_hpx_mask = real_hpx_mask & (data_counts >= minus2_cut)
        n_data_minus2 = int(data_counts[minus2_hpx_mask].sum())
        n_hpx_minus2 = int(np.count_nonzero(minus2_hpx_mask))
        count_summary_by_name[cap] = {
            'n_data_valid': int(n_data_valid),
            'n_data_ge_mean_minus_2std': n_data_minus2,
            'frac_data_ge_mean_minus_2std': (
                float(n_data_minus2 / n_data_valid) if n_data_valid else np.nan
            ),
            'n_hpx_ge_mean_minus_2std': n_hpx_minus2,
        }

        print(f'[{cap}] building 2D random footprint histograms...')
        real_hist, n_random_real = hist2d_from_rows(
            rpath, rand_start, rand_stop, ra_edges, dec_edges,
            args.nside, args.chunk_size,
        )
        clean_hist, n_random_clean = hist2d_from_rows(
            rpath, rand_start, rand_stop, ra_edges, dec_edges,
            args.nside, args.chunk_size, keep_hpx=clean_hpx_mask,
        )

        histograms[cap] = {'real': real_hist, 'ge10': clean_hist}
        runtime_rows[cap] = {
            'n_class_iter_rows': int(n_class),
            'n_data': int(n_data),
            'n_data_valid': int(n_data_valid),
            'n_random_valid': int(n_rand_valid),
            'n_hpx_real': int(np.count_nonzero(real_hpx_mask)),
            'n_hpx_ge10': int(np.count_nonzero(clean_hpx_mask)),
            'n_hpx_ge_mean_minus_2std': n_hpx_minus2,
            'n_data_ge_mean_minus_2std': n_data_minus2,
            'n_random_real_hist': int(n_random_real),
            'n_random_ge10_hist': int(n_random_clean),
        }

    if set(args.caps) == set(DEFAULT_CAPS):
        stats_by_name['ALL'] = percentile_stats(
            np.concatenate([values_by_cap['NGC'], values_by_cap['SGC']])
        )
        n_data_valid_all = sum(count_summary_by_name[cap]['n_data_valid']
                               for cap in DEFAULT_CAPS)
        minus2_cut_all = stats_by_name['ALL']['mean_minus_2std']
        n_data_minus2_all = 0
        n_hpx_minus2_all = 0
        for cap in DEFAULT_CAPS:
            minus2_hpx_mask = (
                real_hpx_mask_by_cap[cap] &
                (data_counts_by_cap[cap] >= minus2_cut_all)
            )
            n_data_minus2_all += int(data_counts_by_cap[cap][minus2_hpx_mask].sum())
            n_hpx_minus2_all += int(np.count_nonzero(minus2_hpx_mask))
        count_summary_by_name['ALL'] = {
            'n_data_valid': int(n_data_valid_all),
            'n_data_ge_mean_minus_2std': int(n_data_minus2_all),
            'frac_data_ge_mean_minus_2std': (
                float(n_data_minus2_all / n_data_valid_all)
                if n_data_valid_all else np.nan
            ),
            'n_hpx_ge_mean_minus_2std': int(n_hpx_minus2_all),
        }

    suffix = f'{args.tracer.lower()}_iter{args.iteration:03d}'
    hist2d_path = out_dir / f'hist2d_2x2_dr2_{suffix}_real_vs_data_ge{args.min_data_per_pix}_randoms.png'
    ndata_hist_path = out_dir / f'hist_dr2_{suffix}_ndata_per_pixel_by_cap.png'
    hists_npz_path = out_dir / f'dr2_{suffix}_hist2d_real_vs_data_ge{args.min_data_per_pix}_randoms.npz'
    values_npz_path = out_dir / f'dr2_{suffix}_ndata_per_pixel_by_cap.npz'
    stats_csv_path = out_dir / f'dr2_{suffix}_ndata_per_pixel_stats.csv'

    plot_hist2d_2x2(hist2d_path, ra_edges, dec_edges, histograms, args.dpi)
    plot_ndata_hist(ndata_hist_path, values_by_cap, stats_by_name, args.dpi)

    np.savez_compressed(
        hists_npz_path,
        ngc_real=histograms['NGC']['real'],
        ngc_ge10=histograms['NGC']['ge10'],
        sgc_real=histograms['SGC']['real'],
        sgc_ge10=histograms['SGC']['ge10'],
        ra_edges=ra_edges,
        dec_edges=dec_edges,
        nside=np.int32(args.nside),
        min_randoms_per_pix=np.int32(args.min_randoms_per_pix),
        min_data_per_pix=np.int32(args.min_data_per_pix),
    )
    np.savez_compressed(
        values_npz_path,
        ngc=values_by_cap['NGC'],
        sgc=values_by_cap['SGC'],
        ngc_mean_minus_2std=np.float64(stats_by_name['NGC']['mean_minus_2std']),
        sgc_mean_minus_2std=np.float64(stats_by_name['SGC']['mean_minus_2std']),
        ngc_n_data_ge_mean_minus_2std=np.int64(
            count_summary_by_name['NGC']['n_data_ge_mean_minus_2std']
        ),
        sgc_n_data_ge_mean_minus_2std=np.int64(
            count_summary_by_name['SGC']['n_data_ge_mean_minus_2std']
        ),
        ngc_n_hpx_ge_mean_minus_2std=np.int64(
            count_summary_by_name['NGC']['n_hpx_ge_mean_minus_2std']
        ),
        sgc_n_hpx_ge_mean_minus_2std=np.int64(
            count_summary_by_name['SGC']['n_hpx_ge_mean_minus_2std']
        ),
        nside=np.int32(args.nside),
        min_randoms_per_pix=np.int32(args.min_randoms_per_pix),
        min_data_per_pix=np.int32(args.min_data_per_pix),
    )
    write_stats_csv(stats_csv_path, stats_by_name, count_summary_by_name)
    print_stats(stats_by_name, count_summary_by_name)

    print('')
    print('Runtime counts')
    for cap, row in runtime_rows.items():
        print(
            f'{cap}: n_data={row["n_data"]} n_hpx_real={row["n_hpx_real"]} '
            f'n_hpx_ge{args.min_data_per_pix}={row["n_hpx_ge10"]} '
            f'n_data_ge_mean_minus_2std={row["n_data_ge_mean_minus_2std"]} '
            f'n_hpx_ge_mean_minus_2std={row["n_hpx_ge_mean_minus_2std"]} '
            f'random_real_hist={row["n_random_real_hist"]} '
            f'random_ge{args.min_data_per_pix}_hist={row["n_random_ge10_hist"]}'
        )

    print('')
    print(f'2D hist figure: {hist2d_path}')
    print(f'N_data/N_pixel hist figure: {ndata_hist_path}')
    print(f'2D hist arrays: {hists_npz_path}')
    print(f'N_data/N_pixel arrays: {values_npz_path}')
    print(f'Stats CSV: {stats_csv_path}')


if __name__ == '__main__':
    main()
