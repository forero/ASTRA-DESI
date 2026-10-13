import argparse, csv, json, os
from pathlib import Path
import sys
import numpy as np
from astropy.io import fits

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import LogLocator, FuncFormatter

sys.dont_write_bytecode = True
BASE = Path('/pscratch/sd/v/vtorresg/cosmic-web/dr1')
TRACERS = ('BGS_BRIGHT', 'BGS_ANY', 'LRG', 'ELG', 'QSO')
COLORS = {'BGS_BRIGHT': '#ff8c26',
          'BGS_ANY': '#6666ff',
          'LRG': '#66ad69',
          'ELG': '#ff6269',
          'QSO': '#ad68ae'}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--root', type=Path, default=BASE)
    p.add_argument('--output-dir', type=Path, default=Path('plots/dr1_consensus_abundance'))
    p.add_argument('--tracers', nargs='+', choices=TRACERS, default=list(TRACERS))
    p.add_argument('--volumes-json', type=Path)
    p.add_argument('--min-randoms-per-pixel', type=int, default=3)
    p.add_argument('--min-randoms-per-radial-bin', type=int, default=3)
    p.add_argument('--bins', type=int, default=24)
    p.add_argument('--rmin', type=float)
    p.add_argument('--rmax', type=float)
    args = p.parse_args()

    out = args.output_dir.expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    for name, sub in [('MPLCONFIGDIR', '.mplconfig'), ('XDG_CACHE_HOME', '.cache'), ('TMPDIR', '.tmp')]:
        path = out/sub
        path.mkdir(exist_ok=True)
        os.environ[name] = str(path)

    supplied = json.loads(args.volumes_json.read_text()) if args.volumes_json else None
    samples = []
    for tracer in dict.fromkeys(args.tracers):
        for cap in ('NGC', 'SGC'):
            key = f'{tracer}_{cap}'
            cat = args.root/'groups_100x/consensus'/f'voids_{key}_n100.fits'
            meta = json.loads(cat.with_suffix('.json').read_text())
            with fits.open(cat, memmap=True) as hdus:
                table = hdus['VOIDS'].data
                hdr = hdus['VOIDS'].header
                cut = float(hdr['VCUT'])
                selected = np.asarray(table['FRAC_V']) > cut
                radius = np.array(table['R_EFF'][selected], dtype=float)
                if hdr['NITER'] != 100 or hdr['TRACER'] != tracer or hdr['CAP'] != cap:
                    raise ValueError(f'Unexpected catalog metadata: {cat}')
            if not len(radius) or not np.all(np.isfinite(radius) & (radius > 0)):
                raise ValueError(f'Empty sample or invalid selected radii: {cat}')
            if len(radius) != meta['n_after_support_cut']:
                raise ValueError(f'Count does not match consensus summary: {cat}')
            info = dict(tracer=tracer, cap=cap, catalog=str(cat), n_voids=len(radius), v_cut=cut,
                        r_min=float(radius.min()), r_max=float(radius.max()))
            if supplied is not None:
                volume = float(supplied[key])
                info['volume_method'] = 'user-supplied'
            else:
                volumes, mask_files = [], []
                for iteration, source in zip(meta['iterations'], meta['inputs'], strict=True):
                    summary = json.loads(Path(source).with_name('summary.json').read_text())
                    h = float(summary['h'])
                    raw = Path(summary['case']['input']).name
                    mask_tracer = summary['case']['tracer']
                    pattern = f'{raw}.*.{mask_tracer}.iter{iteration:03d}.nside*.random-selection-counts.npz'
                    matches = list((args.root/'groups_100x_masks').glob(pattern))
                    if len(matches) != 1:
                        raise ValueError(f'Expected one mask for {key} iteration {iteration}, got {matches}')
                    with np.load(matches[0], allow_pickle=False) as data:
                        m = json.loads(str(data['metadata']))
                        angular = data['random_counts'] >= args.min_randoms_per_pixel
                        radial = data['radial_counts'] >= args.min_randoms_per_radial_bin
                        edges = data['radial_bin_edges'] * h
                        if m['n_random_iterations'] != 1 or len(angular) != 12*m['nside']**2:
                            raise ValueError(f'Unexpected mask: {matches[0]}')
                        omega = 4*np.pi * angular.mean()
                        volumes.append(float(omega*np.diff(edges**3)[radial].sum()/3))
                    mask_files.append(str(matches[0]))
                volume = float(np.mean(volumes))
                info.update(volume_method='mean separable geometric mask volume; assumed count thresholds',
                            min_randoms_per_pixel=args.min_randoms_per_pixel,
                            min_randoms_per_radial_bin=args.min_randoms_per_radial_bin,
                            mask_files=mask_files, h=h)
            if not np.isfinite(volume) or volume <= 0:
                raise ValueError(f'Invalid volume for {key}: {volume}')
            info['volume_Mpc_h_cubed'] = volume
            samples.append((info, np.sort(radius)))
            print(f'{key:15s} N={len(radius):6d} V={volume:.6e} (Mpc/h)^3', flush=True)

    lo = args.rmin if args.rmin is not None else min(r[0] for _, r in samples)*0.85
    hi = args.rmax if args.rmax is not None else max(r[-1] for _, r in samples)*1.05
    if not (0 < lo < hi):
        p.error('Require 0 < rmin < rmax')
    edges = np.geomspace(lo, hi, args.bins+1)
    centers = np.sqrt(edges[:-1]*edges[1:])
    plt.rcParams.update({'font.family': 'serif', 'font.size': 13, 'mathtext.fontset': 'cm',
                         'axes.linewidth': 1., 'xtick.direction': 'in', 'ytick.direction': 'in'})
    fig, ax = plt.subplots(figsize=(10.8, 7.2))
    inset = ax.inset_axes([0.12, 0.13, 0.42, 0.35])
    cumulative, differential = [], []
    for info, radius in samples:
        tracer, cap, volume = info['tracer'], info['cap'], info['volume_Mpc_h_cubed']
        x = np.unique(np.r_[lo, radius[(radius >= lo) & (radius <= hi)], hi])
        counts = len(radius)-np.searchsorted(radius, x, side='right')  # strict > R
        abundance = counts/volume

        ax.step(x, np.where(counts > 0, abundance, np.nan), where='post',
                color=COLORS[tracer], ls='-' if cap == 'NGC' else '--', lw=1.65)
        hist, _ = np.histogram(radius, edges)
        dn = hist/(volume*np.diff(edges))
        inset.plot(centers, np.where(hist > 0, dn, np.nan), color=COLORS[tracer],
                   ls='-' if cap == 'NGC' else '--', lw=1.4)
        cumulative.extend((tracer, cap, xx, int(nn), yy) for xx, nn, yy in zip(x, counts, abundance))
        differential.extend((tracer, cap, a, b, c, int(n), y) for a,b,c,n,y in zip(edges[:-1],edges[1:],centers,hist,dn))
    for a in (ax, inset):
        a.set_xscale('log'); a.set_yscale('log'); a.set_xlim(lo, hi)
        a.tick_params(which='both', top=True, right=True)
        a.xaxis.set_major_locator(LogLocator(base=10, subs=(1, 2, 5)))
        a.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:g}'))

    inset.tick_params(labelsize=9)
    ax.set_xlabel(r'$R_{\rm eff}\ [h^{-1}\,\mathrm{Mpc}]$', fontsize=20)
    ax.set_ylabel(r'$n(>R_{\rm eff})\ [h^3\,\mathrm{Mpc}^{-3}]$', fontsize=20)
    inset.set_xlabel(r'$R_{\rm eff}\ [h^{-1}\,\mathrm{Mpc}]$', fontsize=12)
    inset.set_ylabel(r'$dn/dR_{\rm eff}\ [h^4\,\mathrm{Mpc}^{-4}]$', fontsize=12)
    handles = [Line2D([],[],color=COLORS[t], lw=2, label=t.replace('_',' ')) for t in dict.fromkeys(args.tracers)]
    handles += [Line2D([],[],color='black',ls=s,lw=2,label=c) for c,s in [('NGC','-'),('SGC','--')]]
    ax.legend(handles=handles, loc='upper right', framealpha=1, fontsize=11, ncol=2)
    # ax.set_title('DESI DR1 — consensus voids', fontsize=16)
    # if supplied is None:
    #     fig.text(.98, .015, 'Normalization: estimated geometric mask volume', ha='right', fontsize=9, color='0.35')
    fig.subplots_adjust(left=.12, right=.98, top=.93, bottom=.12)
    for ext in ('png','pdf'):
        fig.savefig(out/f'dr1_consensus_abundance.{ext}', dpi=220)
    plt.close(fig)
    for name, header, rows in [
        ('cumulative', ['tracer','cap','R_Mpc_h','N_gt_R','n_gt_R_h3_Mpc_minus3'], cumulative),
        ('differential', ['tracer','cap','R_left_Mpc_h','R_right_Mpc_h','R_center_Mpc_h','N_bin','dn_dR_h4_Mpc_minus4'], differential)]:
        with (out/f'{name}.csv').open('w', newline='') as f:
            w=csv.writer(f); w.writerow(header); w.writerows(rows)
    (out/'metadata.json').write_text(json.dumps({'samples':[s for s,_ in samples],
        'caveat':'Geometric normalization does not correct radius-dependent boundary losses or consensus completeness.',
        'differential_bin_edges_Mpc_h':edges.tolist()}, indent=2)+'\n')
    print(f'Figures, tables and metadata: {out}')


if __name__ == '__main__':
    main()