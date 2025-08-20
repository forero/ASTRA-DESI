import os, re, glob, argparse
import numpy as np
import matplotlib.pyplot as plt
from astropy.cosmology import Planck18

from plot.plot_groups import (read_groups, read_raw_min, mask_source, tracer_prefixes,
                              ORDERED_TRACERS, TRACER_ZLIMS)

plt.style.use('dark_background')

def _detect_zones(groups_dir, webtype):
    patt = os.path.join(groups_dir, f'zone_*_groups_fof_{webtype}.fits.gz')
    zones = []
    for path in glob.glob(patt):
        m = re.search(r'zone_(\d+)_groups_fof_', os.path.basename(path))
        if m: zones.append(int(m.group(1)))
    return sorted(set(zones))

def _join_zone(groups_dir, raw_dir, class_dir, zone, webtype, source):
    from astropy.table import join as at_join
    groups = read_groups(groups_dir, zone, webtype)
    gm = mask_source(np.asarray(groups['RANDITER']), source)
    groups = groups[gm]
    raw = read_raw_min(raw_dir, class_dir, zone)
    rm = mask_source(np.asarray(raw['RANDITER']), source)
    raw = raw[rm]
    joined = at_join(groups, raw, keys=['TARGETID','TRACERTYPE','RANDITER'], join_type='inner')
    return joined

def _pick_tracers_available(joined, requested=None):
    available = set(map(str, tracer_prefixes(np.asarray(joined['TRACERTYPE']).astype(str))))
    if requested:
        req = [t.split('_',1)[0].upper() for t in requested]
        return [t for t in ORDERED_TRACERS if (t in req) and (t in available)]
    return [t for t in ORDERED_TRACERS if t in available]

def _to_pizza_angles(ra, ra_min, ra_max, theta0, theta1):
    if ra_max <= ra_min:
        return 0.5*(theta0+theta1)*np.ones_like(ra, dtype=float)
    t = (ra - ra_min) / (ra_max - ra_min)
    return theta0 + t * (theta1 - theta0)

def _plot_slice(ax, joined, tracer, theta0, theta1, coord, max_z, bins, connect_lines, smin):
    tr_types = np.asarray(joined['TRACERTYPE']).astype(str)
    ravec = np.asarray(joined['RA'], float)
    zvec = np.asarray(joined['Z'], float)
    gids = np.asarray(joined['GROUPID'], int)

    tr_pref = tracer_prefixes(tr_types)
    m = (tr_pref == tracer)

    z_cap = TRACER_ZLIMS.get(tracer, None)
    if max_z is not None:
        z_cap = min(z_cap, max_z) if z_cap is not None else max_z
    if z_cap is not None:
        m &= (zvec <= z_cap)

    if not np.any(m):
        return 0.0

    if coord == 'dc':
        r_all = Planck18.comoving_distance(zvec[m]).value
    else:
        r_all = zvec[m]

    ra_min, ra_max = float(np.nanmin(ravec[m])), float(np.nanmax(ravec[m]))
    theta = _to_pizza_angles(ravec[m], ra_min, ra_max, theta0, theta1)

    _, color_idx = np.unique(gids[m], return_inverse=True)
    cmap = plt.get_cmap('tab20')
    palette = np.asarray(cmap.colors)
    idx_mod = color_idx % cmap.N

    ax.scatter(theta, r_all, s=max(smin, 7), c=palette[idx_mod], alpha=0.8, linewidths=0)

    if connect_lines:
        gid_sub = gids[m]
        theta_sub = theta
        r_sub = r_all
        for g in np.unique(gid_sub):
            sel = (gid_sub == g)
            if np.count_nonzero(sel) < 2:
                continue
            order = np.argsort(r_sub[sel])
            th = theta_sub[sel][order]
            rr = r_sub[sel][order]
            ax.plot(th, rr, lw=0.6, alpha=0.9, color=palette[int(idx_mod[sel][0])])

    return float(np.nanmax(r_all))

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--raw-dir', default='/pscratch/sd/v/vtorresg/cosmic-web/edr/raw')
    p.add_argument('--class-dir', default='/pscratch/sd/v/vtorresg/cosmic-web/edr/class')
    p.add_argument('--groups-dir',default='/pscratch/sd/v/vtorresg/cosmic-web/edr/groups')
    p.add_argument('--output', default='.')
    p.add_argument('--webtype', choices=['void','sheet','filament','knot'], default='filament')
    p.add_argument('--zones', nargs='+', default=['all'])
    p.add_argument('--tracers', nargs='*', default=None)
    p.add_argument('--source', choices=['data','rand','both'], default='data')
    p.add_argument('--max-z', type=float, default=None)
    p.add_argument('--coord', choices=['z','dc'], default='z')
    p.add_argument('--bins', type=int, default=10)
    p.add_argument('--connect-lines', action='store_true')
    p.add_argument('--smin', type=int, default=1)
    p.add_argument('--title', default=None)
    return p.parse_args()

def main():
    args = parse_args()

    if len(args.zones)==1 and args.zones[0].lower()=='all':
        zones = _detect_zones(args.groups_dir, args.webtype)
    else:
        zones = sorted(set(int(z) for z in args.zones))
    if not zones:
        raise SystemExit('No zones')

    tracers_final = None
    for z in zones:
        try:
            joined0 = _join_zone(args.groups_dir, args.raw_dir, args.class_dir, z, args.webtype, args.source)
            cand = _pick_tracers_available(joined0, args.tracers)
            if cand:
                tracers_final = cand
                break
        except FileNotFoundError:
            continue
    if not tracers_final:
        raise SystemExit('No tracers for zone')

    os.makedirs(args.output, exist_ok=True)

    for tr in tracers_final:
        fig = plt.figure(figsize=(80, 80))
        ax = plt.subplot(111, projection='polar')
        if args.title:
            fig.suptitle(args.title, y=0.98, fontsize=34)

        nZ = len(zones)
        dtheta = 2*np.pi / nZ
        rmax_global = 0.0

        for k, zone in enumerate(zones):
            theta0 = k * dtheta
            theta1 = (k+1) * dtheta
            mid = 0.5*(theta0+theta1)

            try:
                joined = _join_zone(args.groups_dir, args.raw_dir, args.class_dir, zone, args.webtype, args.source)
            except FileNotFoundError:
                ax.plot([theta0, theta0], [0, 1], color='gray', lw=0.5, alpha=0.5, transform=ax.transData)
                ax.text(mid, 1.02, f'Zone {zone:02d}', ha='center', va='bottom', rotation=np.degrees(mid),
                        rotation_mode='anchor', fontsize=35, transform=ax.transData)
                continue

            rmax_local = _plot_slice(ax, joined, tr, theta0, theta1,
                                     coord=args.coord, max_z=args.max_z,
                                     bins=args.bins, connect_lines=args.connect_lines, smin=args.smin)
            rmax_global = max(rmax_global, rmax_local)

            ax.plot([theta0, theta0], [0, rmax_local], color='gray', lw=0.5, alpha=0.5)
            ax.text(mid, rmax_local*1.02 if rmax_local>0 else 1.0, f'Zone {zone:02d}',
                    ha='center', va='bottom', rotation=np.degrees(mid),
                    rotation_mode='anchor', fontsize=35)

        ax.set_theta_zero_location('N')
        ax.set_theta_direction(-1)
        if rmax_global <= 0:
            rmax_global = 1.0
        ax.set_rlim(0, rmax_global*1.05)

        ax.set_rlabel_position(22.5)
        if args.coord == 'dc':
            ax.set_ylabel(r'$D_c$ [Mpc]', fontsize=42, labelpad=12)
        else:
            ax.set_ylabel(r'$z$', fontsize=42, labelpad=12)

        ax.set_title(f'{tr} — {args.webtype.capitalize()}s (RA slices mapped to angular wedges)', va='bottom', fontsize=40, pad=14)

        out = os.path.join(args.output, f'pizza_{tr}_{args.webtype}_{args.coord}.png')
        fig.tight_layout()
        fig.savefig(out, dpi=100)#, bbox_inches='tight')
        plt.close(fig)
        print(out)

if __name__ == '__main__':
    main()