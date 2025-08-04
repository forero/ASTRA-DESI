import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from astropy.table import Table
from astropy.cosmology import Planck18

def load_reals(zone, base_dir):
    path = os.path.join(base_dir, f'ZONE_{zone:02d}.fits.gz')
    df = Table.read(path).to_pandas()
    return df[df['RANDITER'] == -1].reset_index(drop=True)

def plot_polar_wedges_vertical(zones, z_max, n_ra_ticks, n_r_ticks, base_dir):
    z_levels = np.linspace(0, z_max, n_r_ticks)
    r_levels = Planck18.comoving_distance(z_levels).value

    fig, axes = plt.subplots(len(zones), 1, subplot_kw={'projection':'polar'}, figsize=(6, 4*len(zones)), squeeze=False)
    axes = axes.flatten()

    for ax, zone in zip(axes, zones):
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)

        df = load_reals(zone, base_dir)
        ra_min, ra_max = df['RA'].min(), df['RA'].max()
        ra_ctr = 0.5*(ra_min + ra_max)
        dec_ctr = 0.5*(df['DEC'].min() + df['DEC'].max())

        dra = (df['RA'].values - ra_ctr) * np.cos(np.deg2rad(dec_ctr))
        theta = np.deg2rad(dra)
        r = Planck18.comoving_distance(df['Z'].values).value

        ax.scatter(theta, r, s=1, color='k', alpha=0.6)
        ax.set_ylim(0, Planck18.comoving_distance(z_max).value)
        ax.set_rgrids(r_levels, labels=[f'{z:.2f}' for z in z_levels], angle=90, color='gray', fontsize=8)

        ra_ticks = np.linspace(ra_min, ra_max, n_ra_ticks)
        theta_ticks = np.deg2rad((ra_ticks - ra_ctr) * np.cos(np.deg2rad(dec_ctr)))
        ax.set_thetagrids(np.rad2deg(theta_ticks), labels=[f'{rt:.1f}' for rt in ra_ticks], fontsize=8)
        half = np.max(np.abs(theta_ticks))

        ax.set_thetamin(-np.rad2deg(half))
        ax.set_thetamax(np.rad2deg(half))
        ax.patch.set_alpha(0)

        ax.grid(True, color='gray', lw=0.5, alpha=0.5)
        ax.spines['polar'].set_visible(False)

        ax.set_title(f'Zone {zone}', va='bottom', fontsize=12)

    plt.tight_layout()
    plt.savefig('polar_wedges_vertical.png', dpi=360)


def plot_wedges(zones, z_max, n_ra_grid, n_z_grid, base_dir):
    fig, axes = plt.subplots(1, len(zones), figsize=(2*len(zones), 6), sharey=True)

    for ax, zone in zip(axes, zones):
        df = load_reals(zone, base_dir)
        ra_min, ra_max = df['RA'].min(), df['RA'].max()
        ra_ctr = 0.5*(ra_min + ra_max)
        dec_ctr = 0.5*(df['DEC'].min() + df['DEC'].max())

        Dc_max = Planck18.comoving_distance(z_max).value
        half_w = Dc_max * np.deg2rad(ra_max - ra_ctr) * np.cos(np.deg2rad(dec_ctr))

        z_ticks = np.linspace(0, z_max, n_z_grid)
        ra_ticks = np.linspace(ra_min, ra_max, n_ra_grid)

        for z0 in z_ticks:
            w0 = half_w * (z0 / z_max)
            ax.hlines(z0, -w0, w0, color='gray', lw=0.5, alpha=0.5)

        zs = np.linspace(0, z_max, 200)

        for rt in ra_ticks[::4]:
            dx = Dc_max * np.deg2rad(rt - ra_ctr) * np.cos(np.deg2rad(dec_ctr))
            ax.plot(dx/z_max * zs, zs, color='gray', lw=0.5, alpha=0.5)

        Dc = Planck18.comoving_distance(df['Z'].values).value
        x = Dc * np.deg2rad(df['RA'].values - ra_ctr) * np.cos(np.deg2rad(dec_ctr))

        ax.scatter(x, df['Z'], s=0.1, color='k', alpha=0.6)
        ax.plot([-half_w, 0], [z_max, 0], 'k-', lw=1, zorder=10)
        ax.plot([half_w, 0], [z_max, 0], 'k-', lw=1, zorder=10)

        ax.set_frame_on(False)
        ax.set_xlim(-half_w, half_w)
        ax.set_ylim(0, z_max)

        ax.set_yticks(z_ticks[::4])
        ax.yaxis.tick_right()

        ax.tick_params(left=False, right=True, labelleft=False, labelright=True)
        ax.set_xticks([])

        def x2ra(xv):
            return ra_ctr + np.rad2deg(xv / (Dc_max * np.cos(np.deg2rad(dec_ctr))))
        def ra2x(rv):
            return Dc_max * np.deg2rad(rv - ra_ctr) * np.cos(np.deg2rad(dec_ctr))

        sec = ax.secondary_xaxis('top', functions=(x2ra, ra2x))
        sec_ra = ra_ticks[::4]

        sec.set_xticks(sec_ra)
        sec.set_xticklabels([f'{rt:.1f}' for rt in sec_ra])
        sec.set_xlabel('RA (deg)')
        ax.set_title(f'Zone {zone}', pad=10)

    plt.tight_layout()
    plt.savefig('wedges.png', dpi=360)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['polar', 'flat'], default='flat')
    parser.add_argument('--zones', nargs='+', type=int, default=[0,1,2])
    parser.add_argument('--zmax', type=float, default=0.213)
    parser.add_argument('--n_ra', type=int, default=7)
    parser.add_argument('--n_r', type=int, default=8)
    parser.add_argument('--n_ra_grid', type=int, default=20)
    parser.add_argument('--n_z_grid', type=int, default=20)
    parser.add_argument('--base_dir', default='01_CREATE_RAW')
    args = parser.parse_args()
    if args.mode == 'polar':
        plot_polar_wedges_vertical(args.zones, args.zmax, args.n_ra, args.n_r, args.base_dir)
    else:
        plot_wedges(args.zones, args.zmax, args.n_ra_grid, args.n_z_grid, args.base_dir)