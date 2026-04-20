import argparse, csv, os
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits

plt.style.use('dark_background')

plt.rcParams.update({'text.usetex': True,})


def _colmap(names):
    return {str(name).strip().upper(): name for name in names}


def _pick_column(names, candidates):
    cmap = _colmap(names)
    for cand in candidates:
        key = str(cand).strip().upper()
        if key in cmap:
            return cmap[key]
    return None


def _resolve_table_hdu(hdul, preferred, required=True):
    if isinstance(preferred, int):
        hdu = hdul[preferred]
        if getattr(hdu, 'data', None) is None or not hasattr(hdu, 'columns'):
            if required:
                raise ValueError(f'HDU {preferred} is not a table HDU')
            return None, None
        return preferred, hdu

    pref_name = str(preferred).strip().upper() if preferred is not None else None
    if pref_name:
        for idx, hdu in enumerate(hdul):
            name = str(getattr(hdu, 'name', '')).strip().upper()
            if name == pref_name and getattr(hdu, 'data', None) is not None and hasattr(hdu, 'columns'):
                return idx, hdu

    for idx, hdu in enumerate(hdul[1:], start=1):
        if getattr(hdu, 'data', None) is not None and hasattr(hdu, 'columns'):
            if required:
                print(f'Warning: HDU "{preferred}" not found. '
                      f'Using table HDU {idx} ("{getattr(hdu, 'name', '')}").')
            return idx, hdu

    if required:
        raise ValueError(f'Could not find a table HDU (requested "{preferred}")')
    return None, None


def _extract_reff(void_data, radius_col=None, volume_col=None):
    names = list(void_data.dtype.names or [])
    if not names:
        raise ValueError('Void table has no named columns')

    radius_candidates = []
    if radius_col:
        radius_candidates.append(radius_col)
    radius_candidates.extend(['R_EFF', 'REFF', 'RADIUS', 'R'])
    picked_r = _pick_column(names, radius_candidates)
    if picked_r is not None:
        r_eff = np.asarray(void_data[picked_r], dtype=np.float64)
        return r_eff, picked_r, None

    volume_candidates = []
    if volume_col:
        volume_candidates.append(volume_col)
    volume_candidates.extend(['VOLUME', 'VOID_VOLUME', 'VOL'])
    picked_v = _pick_column(names, volume_candidates)
    if picked_v is None:
        raise ValueError('Could not find radius column. Tried: '
                         f'{radius_candidates}. Could not find volume column either')

    volume = np.asarray(void_data[picked_v], dtype=np.float64)
    r_eff = np.full_like(volume, np.nan, dtype=np.float64)
    valid = np.isfinite(volume) & (volume > 0.0)
    r_eff[valid] = np.cbrt(3.0 * volume[valid] / (4.0 * np.pi))
    return r_eff, None, picked_v


def _count_randoms(point_data):
    names = list(point_data.dtype.names or [])
    cmap = _colmap(names)

    if 'IS_DATA' in cmap:
        is_data = np.asarray(point_data[cmap['IS_DATA']]).astype(np.int8)
        return int(np.count_nonzero(is_data == 0)), 'IS_DATA==0'

    if 'RANDITER' in cmap:
        randiter = np.asarray(point_data[cmap['RANDITER']], dtype=np.int64)
        return int(np.count_nonzero(randiter >= 0)), 'RANDITER>=0'

    if 'TRACERTYPE' in cmap:
        tracer = np.asarray(point_data[cmap['TRACERTYPE']]).astype(str)
        is_rand = np.char.endswith(np.char.upper(tracer), '_RAND')
        return int(np.count_nonzero(is_rand)), 'TRACERTYPE endswith _RAND'

    raise ValueError('Could not infer N_rand from POINT_MEMBERSHIP (missing IS_DATA/RANDITER/TRACERTYPE)')


def _build_bins(r_eff, rmin=None, rmax=None, n_bins=18, bin_edges=None):
    if bin_edges is not None and len(bin_edges) > 0:
        edges = np.asarray(bin_edges, dtype=np.float64)
    else:
        if n_bins < 1:
            raise ValueError('--n-bins must be >= 1')
        finite = r_eff[np.isfinite(r_eff) & (r_eff > 0.0)]
        if finite.size == 0:
            raise ValueError('No finite positive R_eff values found')
        lo = float(np.nanmin(finite)) if rmin is None else float(rmin)
        hi = float(np.nanmax(finite)) if rmax is None else float(rmax)
        if lo <= 0.0:
            raise ValueError('Logarithmic bins require rmin > 0 (or positive radii in catalog)')
        if hi <= lo:
            raise ValueError('Need rmax > rmin to build logarithmic bins')
        edges = np.logspace(np.log10(lo), np.log10(hi), int(n_bins) + 1, dtype=np.float64)

    if edges.ndim != 1 or edges.size < 2:
        raise ValueError('Need at least two bin edges')
    if not np.all(np.isfinite(edges)):
        raise ValueError('Bin edges contain non-finite values')
    if not np.all(np.diff(edges) > 0):
        raise ValueError('Bin edges must be strictly increasing')
    return edges


def _get_survey_volume(args, nrand):
    if args.survey_volume is not None:
        if args.survey_volume <= 0:
            raise ValueError('--survey-volume must be > 0')
        return float(args.survey_volume), 'user-provided'

    if args.anchor_volume is not None or args.anchor_nrand is not None:
        if args.anchor_volume is None or args.anchor_nrand is None:
            raise ValueError('Use --anchor-volume and --anchor-nrand together')
        if args.anchor_volume <= 0 or args.anchor_nrand <= 0:
            raise ValueError('--anchor-volume and --anchor-nrand must be > 0')
        if nrand is None or nrand <= 0:
            raise ValueError('N_rand is required for anchor-volume scaling')
        survey_volume = float(args.anchor_volume) * (float(nrand) / float(args.anchor_nrand))
        return survey_volume, 'scaled from anchor-volume/anchor-nrand'

    if args.random_density is not None:
        if args.random_density <= 0:
            raise ValueError('--random-density must be > 0')
        if nrand is None or nrand <= 0:
            raise ValueError('N_rand is required when using --random-density')
        survey_volume = float(nrand) / float(args.random_density)
        return survey_volume, 'N_rand / random_density'

    raise ValueError('Physical normalization requested but survey volume is unknown. '
                     'Provide --survey-volume, or (--anchor-volume and --anchor-nrand), '
                     'or --random-density')


def _compute_vsf(counts, widths, normalization, nrand=None, survey_volume=None):
    counts = np.asarray(counts, dtype=np.float64)
    widths = np.asarray(widths, dtype=np.float64)
    errors = np.sqrt(np.clip(counts, 0.0, None))

    if normalization == 'counts':
        denom = np.ones_like(widths)
        ylabel = r'$N_{\rm void}$'
    elif normalization == 'per_dr':
        denom = widths
        ylabel = r'$N_{\rm void}/\Delta R$'
    elif normalization == 'per_nrand_dr':
        if nrand is None or nrand <= 0:
            raise ValueError('Normalization "per_nrand_dr" requires N_rand > 0')
        denom = float(nrand) * widths
        ylabel = r'$N_{\rm void}/(N_{\rm rand}\,\Delta R)$'
    elif normalization == 'physical':
        if survey_volume is None or survey_volume <= 0:
            raise ValueError('Normalization "physical" requires survey volume > 0')
        denom = float(survey_volume) * widths
        ylabel = r'$n(R)=N_{\rm void}/(V_{\rm survey}\,\Delta R)$'
    else:
        raise ValueError(f'Unknown normalization: {normalization}')

    y = counts / denom
    yerr = errors / denom
    return y, yerr, ylabel


def _write_csv(path, left, right, center, width, counts, counts_err, y, yerr):
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with open(path, 'w', newline='', encoding='utf-8') as fh:
        writer = csv.writer(fh)
        writer.writerow(['R_left_Mpc_h',
                         'R_right_Mpc_h',
                         'R_center_Mpc_h',
                         'dR_Mpc_h',
                         'N_void',
                         'N_void_err_poisson',
                         'VSF',
                         'VSF_err_poisson',])
        for vals in zip(left, right, center, width, counts, counts_err, y, yerr):
            writer.writerow([f'{v:.12g}' for v in vals])


def _build_zoom_output_path(output_path, xlim):
    root, ext = os.path.splitext(output_path)
    if not ext:
        ext = '.png'
    x0 = int(xlim[0]) if float(xlim[0]).is_integer() else xlim[0]
    x1 = int(xlim[1]) if float(xlim[1]).is_integer() else xlim[1]
    return f'{root}_xlim_{x0}_{x1}{ext}'


def _save_plot(output_path, center, y, yerr, width, edges, ylabel, title, logy, dpi, xlim=None):
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.0, 5.0))
    ax.errorbar(center, y, yerr=yerr, xerr=0.5 * width,
                fmt='o', ls='none', ms=4, capsize=3,
                lw=1.0, color='cyan', ecolor='cyan',
                markerfacecolor='cyan', markeredgecolor='cyan')
    ax.grid(alpha=0.25, lw=0.5)
    ax.set_xlabel(r'$R_{\rm eff}\ [{\rm Mpc}/h]$')
    ax.set_ylabel(ylabel)
    if xlim is not None:
        ax.set_xlim(float(xlim[0]), float(xlim[1]))
    if logy:
        ax.set_yscale('log')
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--void-catalog', required=True)
    p.add_argument('--void-hdu', default='VOIDS')
    p.add_argument('--point-hdu', default='POINT_MEMBERSHIP')
    p.add_argument('--radius-col', default=None)
    p.add_argument('--volume-col', default=None)

    p.add_argument('--rmin', type=float, default=None)
    p.add_argument('--rmax', type=float, default=None)
    p.add_argument('--n-bins', type=int, default=18)
    p.add_argument('--bin-edges', type=float, nargs='+', default=None)

    p.add_argument('--normalization',
                   choices=['counts', 'per_dr', 'per_nrand_dr', 'physical'])
    p.add_argument('--survey-volume', type=float, default=None)
    p.add_argument('--anchor-volume', type=float, default=None)
    p.add_argument('--anchor-nrand', type=float, default=None)
    p.add_argument('--random-density', type=float, default=None)

    p.add_argument('--output', default='plots/vsf.png')
    p.add_argument('--table-output', default=None)
    p.add_argument('--title', default=None)
    p.add_argument('--logy', action='store_true')
    p.add_argument('--dpi', type=int, default=250)
    p.add_argument('--xlim-zoom', type=float, nargs=2, default=(5.0, 100.0),
                   metavar=('XMIN', 'XMAX'))
    p.add_argument('--zoom-output', default=None)
    return p.parse_args()


def main():
    args = parse_args()

    with fits.open(args.void_catalog, memmap=True) as hdul:
        _, void_hdu = _resolve_table_hdu(hdul, args.void_hdu, required=True)
        void_data = void_hdu.data
        r_eff, used_radius_col, used_volume_col = _extract_reff(void_data,
                                                                radius_col=args.radius_col,
                                                                volume_col=args.volume_col)

        nrand = None
        nrand_method = None
        need_nrand = args.normalization in {'per_nrand_dr', 'physical'} or any(
            v is not None for v in (args.anchor_nrand, args.anchor_volume, args.random_density))
        if need_nrand:
            _, point_hdu = _resolve_table_hdu(hdul, args.point_hdu, required=True)
            nrand, nrand_method = _count_randoms(point_hdu.data)

    valid = np.isfinite(r_eff) & (r_eff > 0.0)
    r_eff = r_eff[valid]
    if r_eff.size == 0:
        raise RuntimeError('No valid R_eff values available after filtering non-finite/non-positive entries')

    edges = _build_bins(r_eff, rmin=args.rmin, rmax=args.rmax, n_bins=args.n_bins, bin_edges=args.bin_edges)
    counts, _ = np.histogram(r_eff, bins=edges)
    left = edges[:-1]
    right = edges[1:]
    width = right - left
    center = 0.5 * (left + right)

    survey_volume = None
    survey_volume_mode = None
    if args.normalization == 'physical':
        survey_volume, survey_volume_mode = _get_survey_volume(args, nrand=nrand)

    y, yerr, ylabel = _compute_vsf(counts=counts, widths=width, normalization=args.normalization, nrand=nrand, survey_volume=survey_volume)

    title = args.title
    if title is None:
        title = rf'Void Size Function ({args.normalization})'

    zoom_xlim = tuple(args.xlim_zoom)
    if zoom_xlim[1] <= zoom_xlim[0]:
        raise ValueError('--xlim-zoom must satisfy XMAX > XMIN')

    zoom_output = args.zoom_output
    if zoom_output is None:
        zoom_output = _build_zoom_output_path(args.output, zoom_xlim)

    _save_plot(output_path=args.output, center=center, y=y, yerr=yerr, width=width, edges=edges, ylabel=ylabel,
               title=title, logy=args.logy, dpi=args.dpi, xlim=None,)
    _save_plot(output_path=zoom_output, center=center, y=y, yerr=yerr, width=width, edges=edges, ylabel=ylabel,
               title=title + rf', $R_{{\rm eff}}\in[{zoom_xlim[0]:g},{zoom_xlim[1]:g}]$',
               logy=args.logy, dpi=args.dpi, xlim=zoom_xlim)

    table_output = args.table_output
    if table_output is None:
        root, _ = os.path.splitext(args.output)
        table_output = f'{root}.csv'
    counts_err = np.sqrt(np.clip(counts.astype(np.float64), 0.0, None))
    _write_csv(table_output, left, right, center, width, counts, counts_err, y, yerr)

    print(f'Input void catalog: {args.void_catalog}')
    print(f'Valid voids used: {r_eff.size}')
    if used_radius_col is not None:
        print(f'Radius column: {used_radius_col}')
    else:
        print(f'Radius from volume: {used_volume_col}')
    print(f'Bins: {len(width)} (dr median = {np.median(width):.3f} Mpc/h)')
    if nrand is not None:
        print(f'N_rand: {nrand} ({nrand_method})')
    if survey_volume is not None:
        print(f'V_survey: {survey_volume:.6g} (Mpc/h)^3 [{survey_volume_mode}]')
    print(f'Normalization: {args.normalization}')
    print(f'Figure saved: {args.output}')
    print(f'Figure saved: {zoom_output}')
    print(f'CSV saved: {table_output}')


if __name__ == '__main__':
    main()