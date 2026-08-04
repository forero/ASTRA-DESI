import argparse, os
from pathlib import Path

import healpy as hp
import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.table import Table

import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'
matplotlib.rcParams['text.usetex'] = True

import matplotlib.pyplot as plt
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['savefig.facecolor'] = 'white'

from matplotlib import transforms
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch


DEFAULT_TILES_PATH = '/global/cfs/cdirs/desi/public/dr1/survey/ops/surveyops/tags/1.0/ops/tiles-main.ecsv'
DEFAULT_LSS_BASE = '/global/cfs/cdirs/desi/public/dr1/survey/catalogs/dr1/LSS/iron/LSScats/v1.5pip'
PROGRAM_CONFIG = {'bright': {'tile_program': 'BRIGHT', 'tracers': ('BGS_ANY', 'BGS_BRIGHT')},
                  'dark': {'tile_program': 'DARK', 'tracers': ('LRG', 'ELG_LOPnotqso', 'QSO')},}
CATALOG_ZONES = ('NGC', 'SGC')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--nside', type=int, default=128)
    parser.add_argument('--tile-radius-deg', type=float, default=1.605)
    parser.add_argument('--min-tiles', type=int, default=4)
    parser.add_argument('--min-lss-objects', type=int, default=1)
    parser.add_argument('--smooth-sigma-deg', type=float, default=2.8)
    parser.add_argument('--smooth-threshold', type=float, default=0.41)
    parser.add_argument('--disable-smoothing', action='store_true')
    parser.add_argument('--manual-ra-cut-max', type=float, default=None)
    parser.add_argument('--tiles-path', default=DEFAULT_TILES_PATH)
    parser.add_argument('--lss-base', default=DEFAULT_LSS_BASE)
    parser.add_argument('--out-dir', default=None)
    parser.add_argument('--dpi', type=int, default=300)
    return parser.parse_args()


def resolve_output_dir(user_out_dir):
    if user_out_dir:
        out_dir = Path(user_out_dir).expanduser().resolve()
    else:
        pscratch = os.environ.get('PSCRATCH') or os.environ.get('SCRATCH')
        if not pscratch:
            raise RuntimeError('PSCRATCH/SCRATCH is not defined; pass --out-dir explicitly')
        out_dir = Path(pscratch) / 'cosmic-web' / 'dr1' / 'masks' / 'bright_dark'
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def radec_to_vec(ra_deg, dec_deg):
    theta = np.radians(90.0 - dec_deg)
    phi = np.radians(ra_deg)
    return hp.ang2vec(theta, phi)


def build_tile_count_map(nside, tiles, tile_radius_deg):
    npix = hp.nside2npix(nside)
    tile_count_map = np.zeros(npix, dtype=np.int32)
    tile_radius_rad = np.radians(tile_radius_deg)

    for ra, dec in zip(np.asarray(tiles['RA']), np.asarray(tiles['DEC'])):
        vec = radec_to_vec(ra, dec)
        pix_in_disc = hp.query_disc(nside, vec, tile_radius_rad, inclusive=True)
        tile_count_map[pix_in_disc] += 1
    return tile_count_map


def load_tiles_for_program(tiles_path, program):
    tiles = Table.read(tiles_path, format='ascii.ecsv')
    required = {'IN_DESI', 'PROGRAM', 'STATUS', 'RA', 'DEC'}
    missing = sorted(required.difference(tiles.colnames))
    if missing:
        raise KeyError(f'tiles file missing columns: {missing}')

    in_desi = np.asarray(tiles['IN_DESI']).astype(bool)
    tile_program = np.asarray(tiles['PROGRAM']).astype(str)
    tile_status = np.asarray(tiles['STATUS']).astype(str)
    sel = in_desi & (tile_program == program) & np.isin(tile_status, ['done', 'obsend'])
    return tiles[sel]


def read_lss_radec(path):
    with fits.open(path, memmap=True) as hdul:
        data = hdul[1].data
        ra = np.asarray(data['RA'], dtype=np.float64)
        dec = np.asarray(data['DEC'], dtype=np.float64)

    valid = np.isfinite(ra) & np.isfinite(dec)
    return ra[valid], dec[valid], int(valid.sum())


def build_lss_count_maps_by_zone(nside, lss_base, tracers):
    """Build independent HEALPix count maps from the NGC and SGC catalogues."""
    count_maps = {
        zone: np.zeros(hp.nside2npix(nside), dtype=np.int32)
        for zone in CATALOG_ZONES
    }
    n_total_rows = 0
    per_file_rows = {}

    for tracer in tracers:
        for zone in CATALOG_ZONES:
            path = Path(lss_base) / f'{tracer}_{zone}_clustering.dat.fits'
            if not path.exists():
                raise FileNotFoundError(f'Missing file: {path}')

            ra, dec, n_valid = read_lss_radec(path)
            theta = np.radians(90.0 - dec)
            phi = np.radians(ra)
            pix = hp.ang2pix(nside, theta, phi)
            count_maps[zone] += np.bincount(
                pix, minlength=count_maps[zone].size
            ).astype(np.int32, copy=False)

            per_file_rows[path.name] = n_valid
            n_total_rows += n_valid

    return count_maps, n_total_rows, per_file_rows


def build_zone_plot_map(mask_ngc, mask_sgc):
    zone_map = np.full(mask_ngc.size, hp.UNSEEN, dtype=np.float64)
    zone_map[mask_ngc] = 1.0
    zone_map[mask_sgc] = 2.0
    return zone_map


def smooth_and_threshold_mask(mask, sigma_deg=2.8, threshold=0.338, nest=False):
    mask_float = np.asarray(mask, dtype=float)
    sigma_rad = np.deg2rad(sigma_deg)
    smoothed = hp.smoothing(mask_float, sigma=sigma_rad, nest=nest, verbose=False)
    cleaned_mask = smoothed >= threshold
    return cleaned_mask, smoothed


def apply_manual_ra_cut(mask, ra_max_deg=None, nest=False):
    if ra_max_deg is None:
        return mask

    nside = hp.get_nside(mask)
    theta, phi = hp.pix2ang(nside, np.arange(mask.size), nest=nest)
    ra = np.degrees(phi)
    cleaned = np.asarray(mask, dtype=bool).copy()
    cleaned[ra < float(ra_max_deg)] = False
    return cleaned


def write_program_outputs(out_dir, program_key, nside, tile_count_map, lss_count_map,
                          tile_mask, lss_mask, final_mask, final_ngc, final_sgc,
                          smoothed_map=None, lss_count_map_ngc=None,
                          lss_count_map_sgc=None):
    prefix = out_dir / f'dr1_mask_{program_key}_nside{nside}'
    path_final = Path(f'{prefix}_final.fits')
    path_ngc = Path(f'{prefix}_ngc.fits')
    path_sgc = Path(f'{prefix}_sgc.fits')
    path_zones = Path(f'{prefix}_zones.fits')
    path_npz = Path(f'{prefix}_maps.npz')

    if smoothed_map is None:
        smoothed_map = final_mask.astype(float)

    zone_labels = np.zeros(final_mask.size, dtype=np.int16)
    zone_labels[final_ngc] = 1
    zone_labels[final_sgc] = 2

    hp.write_map(path_final, final_mask.astype(np.int16), overwrite=True, dtype=np.int16, coord='C')
    hp.write_map(path_ngc, final_ngc.astype(np.int16), overwrite=True, dtype=np.int16, coord='C')
    hp.write_map(path_sgc, final_sgc.astype(np.int16), overwrite=True, dtype=np.int16, coord='C')
    hp.write_map(path_zones, zone_labels, overwrite=True, dtype=np.int16, coord='C')

    maps = {
        'nside': np.int32(nside),
        'program': np.array(program_key),
        'zone_source': np.array('catalog_filename'),
        'tile_count_map': tile_count_map,
        'lss_count_map': lss_count_map,
        'tile_mask': tile_mask.astype(np.uint8),
        'lss_mask': lss_mask.astype(np.uint8),
        'final_mask': final_mask.astype(np.uint8),
        'final_mask_ngc': final_ngc.astype(np.uint8),
        'final_mask_sgc': final_sgc.astype(np.uint8),
        'final_mask_zones': zone_labels,
        'smoothed_map': np.asarray(smoothed_map, dtype=np.float32),
    }
    if lss_count_map_ngc is not None:
        maps['lss_count_map_ngc'] = np.asarray(lss_count_map_ngc, dtype=np.int32)
    if lss_count_map_sgc is not None:
        maps['lss_count_map_sgc'] = np.asarray(lss_count_map_sgc, dtype=np.int32)
    np.savez_compressed(path_npz, **maps)

    return path_final, path_ngc, path_sgc, path_zones, path_npz


def save_mollweide_two_programs(bright_zone_plot, dark_zone_plot, output_path, dpi):
    panel_bg = 'white'
    tick_fontsize = 13
    axis_label_fontsize = 16
    title_fontsize = 17
    legend_fontsize = 17
    graticule_style = dict(dpar=30, dmer=60,
                           linestyle='-', linewidth=1.0,
                           color='#c7c7c7', alpha=0.85)

    def apply_panel_bg(ax):
        if hasattr(ax, 'set_facecolor'):
            ax.set_facecolor(panel_bg)
        if hasattr(ax, 'patch') and ax.patch is not None:
            ax.patch.set_facecolor(panel_bg)
            ax.patch.set_alpha(1.0)

    def draw_galactic_plane_line():
        lon = np.linspace(0.0, 360.0, 2048)
        lat = np.zeros_like(lon)
        icrs = SkyCoord(l=lon * u.deg, b=lat * u.deg, frame='galactic').icrs
        ra = icrs.ra.wrap_at(180 * u.deg).degree
        dec = icrs.dec.degree

        jumps = np.where(np.abs(np.diff(ra)) > 180.0)[0]
        start = 0
        for stop in jumps:
            hp.projplot(ra[start:stop + 1], dec[start:stop + 1], lonlat=True,
                        color='#8a8a8a', lw=3.6, alpha=0.55)
            start = stop + 1
        hp.projplot(ra[start:], dec[start:], lonlat=True,
                    color='#8a8a8a', lw=3.6, alpha=0.55)

    def draw_radec_ticks(show_dec_axis_label=False):
        ra_ticks = np.arange(0.0, 360.0, 30.0, dtype=np.float64)
        dec_ticks = np.array([-75, -60, -45, -30, -15, 0, 15, 30, 45, 60, 75], dtype=np.float64)
        ra_tick_dec = 4.0
        dec_label_ra = 299.0
        dec_tick_offset = transforms.ScaledTranslation(-10.0 / 72.0, 0.0, fig.dpi_scale_trans)
        dec_tick_extreme_offset = transforms.ScaledTranslation(-14.0 / 72.0, 0.0, fig.dpi_scale_trans)
        dec_axis_offset = transforms.ScaledTranslation(-44.0 / 72.0, 0.0, fig.dpi_scale_trans)
        ra_axis_offset = transforms.ScaledTranslation(0.0, -8.0 / 72.0, fig.dpi_scale_trans)

        for ra_tick in ra_ticks:
            if int(ra_tick) == 300:
                continue
            hp.projtext(ra_tick, ra_tick_dec, rf'${int(ra_tick)}^\circ$',
                        lonlat=True, fontsize=tick_fontsize, ha='center', va='center', color='black')

        for dec_tick in dec_ticks:
            dec_label = f'{int(dec_tick)}'
            dec_tick_text = hp.projtext(dec_label_ra, dec_tick, rf'${dec_label}^\circ$',
                                        lonlat=True, fontsize=tick_fontsize, ha='right', va='center',
                                        color='black', clip_on=False)
            tick_offset = dec_tick_extreme_offset if abs(int(dec_tick)) == 75 else dec_tick_offset
            dec_tick_text.set_transform(dec_tick_text.get_transform() + tick_offset)
            if (int(dec_tick) == 75):
                dec_tick_text.set_y(dec_tick_text.get_position()[1] + 0.025)
            elif (int(dec_tick) == -75) or (int(dec_tick) == -60) or (int(dec_tick) == -45):
                dec_tick_text.set_y(dec_tick_text.get_position()[1] - 0.05)

        ra_axis_text = hp.projtext(180.0, -89.0, r'$\mathrm{RA}\,[\mathrm{deg}]$', lonlat=True,
                                   fontsize=axis_label_fontsize, ha='center', va='top', color='black')
        ra_axis_text.set_transform(ra_axis_text.get_transform() + ra_axis_offset)
        if show_dec_axis_label:
            dec_axis_text = hp.projtext(dec_label_ra, 0.0, r'$\mathrm{DEC}\,[\mathrm{deg}]$', lonlat=True,
                                        fontsize=axis_label_fontsize, ha='center', va='center',
                                        color='black', clip_on=False, rotation=90)
            dec_axis_text.set_transform(dec_axis_text.get_transform() + dec_axis_offset)

    def sanitize_plot_values(zone_plot):
        values = np.asarray(zone_plot, dtype=np.float64).copy()
        valid = (values == 1.0) | (values == 2.0)
        values[~valid] = hp.UNSEEN
        return values

    cmap = ListedColormap(['#3b82f6', '#ef4444'])
    cmap.set_bad(color=panel_bg)
    cmap.set_under(color=panel_bg)
    cmap.set_over(color=panel_bg)

    fig = plt.figure(figsize=(14, 6), facecolor=panel_bg)
    bright_plot = sanitize_plot_values(bright_zone_plot)
    dark_plot = sanitize_plot_values(dark_zone_plot)

    hp.mollview(bright_plot, fig=fig.number, sub=(1, 2, 1),
                title='', margins=(0.06, 0.12, 0.02, 0.08),
                rot=(120, 0, 0),
                cmap=cmap, min=0.5, max=2.5, cbar=False,
                badcolor=panel_bg, bgcolor=panel_bg)
    bright_ax = plt.gca()
    apply_panel_bg(bright_ax)
    hp.graticule(**graticule_style)
    draw_radec_ticks(show_dec_axis_label=True)
    draw_galactic_plane_line()
    bright_ax.set_title('BRIGHT final mask (BGS)', fontsize=title_fontsize,y=1.2)

    hp.mollview(dark_plot, fig=fig.number, sub=(1, 2, 2),
                rot=(120, 0, 0), margins=(0.06, 0.12, 0.02, 0.08),
                title='',
                cmap=cmap, min=0.5, max=2.5, cbar=False,
                badcolor=panel_bg, bgcolor=panel_bg)
    dark_ax = plt.gca()
    apply_panel_bg(dark_ax)
    hp.graticule(**graticule_style)
    draw_radec_ticks(show_dec_axis_label=True)
    draw_galactic_plane_line()
    dark_ax.set_title('DARK final mask (LRG+ELG+QSO)', fontsize=title_fontsize,y=1.2)

    handles = [Patch(facecolor='#3b82f6', edgecolor='none', label=r'NGC'),
               Patch(facecolor='#ef4444', edgecolor='none', label=r'SGC'),]
    fig.legend(handles=handles, loc='lower center', ncol=2,
               prop={'size': legend_fontsize})
    fig.patch.set_facecolor(panel_bg)
    plt.savefig(output_path, dpi=dpi, facecolor=panel_bg)
    plt.close(fig)


def process_program(program_key, program_cfg, args, out_dir):
    tile_program = program_cfg['tile_program']
    tracers = program_cfg['tracers']

    tiles = load_tiles_for_program(args.tiles_path, tile_program)
    tile_count_map = build_tile_count_map(args.nside, tiles, args.tile_radius_deg)
    tile_mask = tile_count_map >= int(args.min_tiles)

    lss_count_maps, n_lss, per_file_rows = build_lss_count_maps_by_zone(
        args.nside, args.lss_base, tracers
    )
    lss_masks = {
        zone: lss_count_maps[zone] >= int(args.min_lss_objects)
        for zone in CATALOG_ZONES
    }

    final_by_zone = {}
    smoothed_by_zone = {}
    for zone in CATALOG_ZONES:
        raw_zone_mask = tile_mask & lss_masks[zone]
        if args.disable_smoothing:
            final_zone = raw_zone_mask.copy()
            smoothed_zone = raw_zone_mask.astype(float)
        else:
            final_zone, smoothed_zone = smooth_and_threshold_mask(
                raw_zone_mask,
                sigma_deg=args.smooth_sigma_deg,
                threshold=args.smooth_threshold,
                nest=False
            )
        final_by_zone[zone] = apply_manual_ra_cut(
            final_zone, ra_max_deg=args.manual_ra_cut_max, nest=False
        )
        smoothed_by_zone[zone] = smoothed_zone

    final_ngc = final_by_zone['NGC']
    final_sgc = final_by_zone['SGC']
    final_mask = final_ngc | final_sgc
    lss_mask = lss_masks['NGC'] | lss_masks['SGC']
    lss_count_map = lss_count_maps['NGC'] + lss_count_maps['SGC']
    smoothed_map = np.maximum(smoothed_by_zone['NGC'], smoothed_by_zone['SGC'])

    print(f'[{program_key}] tile program={tile_program} tracers={",".join(tracers)}')
    print(f'[{program_key}] selected tiles={len(tiles)}')
    print(f'[{program_key}] LSS rows used={n_lss}')
    for name, count in sorted(per_file_rows.items()):
        print(f'[{program_key}] {name}: {count}')
    print(f'[{program_key}] tile_mask pixels={int(tile_mask.sum())} ({tile_mask.mean():.4f})')
    print(f'[{program_key}] lss_mask pixels={int(lss_mask.sum())} ({lss_mask.mean():.4f})')
    print(f'[{program_key}] combined final_mask pixels={int(final_mask.sum())} ({final_mask.mean():.4f})')
    print(f'[{program_key}] final NGC pixels={int(final_ngc.sum())} ({final_ngc.mean():.4f})')
    print(f'[{program_key}] final SGC pixels={int(final_sgc.sum())} ({final_sgc.mean():.4f})')

    out_paths = write_program_outputs(out_dir, program_key, args.nside,
                                      tile_count_map, lss_count_map,
                                      tile_mask, lss_mask,
                                      final_mask, final_ngc, final_sgc,
                                      smoothed_map=smoothed_map,
                                      lss_count_map_ngc=lss_count_maps['NGC'],
                                      lss_count_map_sgc=lss_count_maps['SGC'])
    return final_ngc, final_sgc, out_paths


def main():
    args = parse_args()
    out_dir = resolve_output_dir(args.out_dir)
    if not Path(args.tiles_path).exists():
        raise FileNotFoundError(f'tiles file not found: {args.tiles_path}')

    print(f'-- nside={args.nside} npix={hp.nside2npix(args.nside)}')
    print(f'-- tiles_path={args.tiles_path}')
    print(f'-- lss_base={args.lss_base}')
    print(f'-- out_dir={out_dir}')
    print(f'-- min_tiles={args.min_tiles}')
    print(f'-- min_lss_objects={args.min_lss_objects}')
    print(f'-- disable_smoothing={args.disable_smoothing}')
    print(f'-- smooth_sigma_deg={args.smooth_sigma_deg}')
    print(f'-- smooth_threshold={args.smooth_threshold}')
    print(f'-- manual_ra_cut_max={args.manual_ra_cut_max}')

    bright_ngc, bright_sgc, bright_paths = process_program('bright',
                                                           PROGRAM_CONFIG['bright'],
                                                           args,
                                                           out_dir)
    dark_ngc, dark_sgc, dark_paths = process_program('dark',
                                                     PROGRAM_CONFIG['dark'],
                                                     args,
                                                     out_dir)

    bright_zone_plot = build_zone_plot_map(bright_ngc, bright_sgc)
    dark_zone_plot = build_zone_plot_map(dark_ngc, dark_sgc)

    fig_path = out_dir / f'dr1_masks_bright_dark_ngc_sgc_mollweide_nside{args.nside}.png'
    save_mollweide_two_programs(bright_zone_plot, dark_zone_plot, fig_path, args.dpi)

    print('-- output files')
    for label, paths in [('bright', bright_paths), ('dark', dark_paths)]:
        for p in paths:
            print(f'[{label}] wrote {p}')
    print(f'[plot] wrote {fig_path}')


if __name__ == '__main__':
    main()
