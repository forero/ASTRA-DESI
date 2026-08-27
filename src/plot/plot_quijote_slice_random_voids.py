from __future__ import annotations

import argparse
from pathlib import Path

import fitsio
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from plot_quijote_cube import (DEFAULT_ASTRA_ROOT, DEFAULT_INPUT_ROOT, SNAPSHOT,
                               _configure_matplotlib, _load_positions)
from plot_quijote_slice_xy import _style_axis


DEFAULT_OUTPUT = Path('figs/quijote_fiducial_0_slice_xy_real_random_voids.png')


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-root', type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument('--astra-root', type=Path, default=DEFAULT_ASTRA_ROOT)
    parser.add_argument('--parameter', default='fiducial')
    parser.add_argument('--realization', type=int, default=0)
    parser.add_argument('--snapshot', type=int, default=SNAPSHOT)
    parser.add_argument('--box-size', type=float, default=1000.0)
    parser.add_argument('--xy-size', type=float, default=500.0,
                        help='Central x-y window side in Mpc/h (default: 500).')
    parser.add_argument('--slice-thickness', type=float, default=100.0,
                        help='Full z-slice thickness in Mpc/h (default: 100).')
    parser.add_argument('--slice-center-z', type=float, default=None,
                        help='Slice centre in original coordinates; default: box centre.')
    parser.add_argument('--real-marker-size', type=float, default=4.0)
    parser.add_argument('--random-marker-size', type=float, default=0.45)
    parser.add_argument('--random-alpha', type=float, default=0.18)
    parser.add_argument('--chunk-rows', type=int, default=500_000,
                        help='Random-void FITS rows read per block.')
    parser.add_argument('--readfof-path', type=Path, default=None)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--dpi', type=int, default=360)
    parser.add_argument('--no-tex', action='store_true')
    return parser


def _header_value(header, key):
    value = header.get(key)
    if isinstance(value, bytes):
        value = value.decode('ascii', errors='replace').strip()
    return value


def _load_random_void_xy(path: Path, parameter: str, realization: int,
                         snapshot: int, box_size: float, xy_size: float,
                         slice_center_z: float, slice_thickness: float,
                         chunk_rows: int) -> tuple[np.ndarray, int]:
    if not path.is_file():
        raise FileNotFoundError(f'ASTRA random-void file not found: {path}')

    centre_xy = np.float32(0.5 * box_size)
    xy_half = np.float32(0.5 * xy_size)
    z_half = np.float32(0.5 * slice_thickness)
    selected_chunks = []

    with fitsio.FITS(str(path), mode='r') as hdus:
        if len(hdus) < 2:
            raise ValueError(f'Missing table extension in {path}')
        table = hdus[1]
        header = table.read_header()
        expected = {'PARAM': parameter, 'REALIZ': realization,
                    'SNAPNUM': snapshot, 'PRODUCT': 'RANDVOID'}
        for key, wanted in expected.items():
            found = _header_value(header, key)
            if found != wanted:
                raise ValueError(
                    f'Random-void header mismatch for {key}: expected '
                    f'{wanted!r}, found {found!r}.')

        total_rows = int(table.get_nrows())
        for start in range(0, total_rows, chunk_rows):
            stop = min(start + chunk_rows, total_rows)
            rows = table[start:stop]
            x = np.asarray(rows['X'], dtype=np.float32)
            y = np.asarray(rows['Y'], dtype=np.float32)
            z = np.asarray(rows['Z'], dtype=np.float32)
            inside = ((np.abs(x - centre_xy) <= xy_half)
                      & (np.abs(y - centre_xy) <= xy_half)
                      & (np.abs(z - slice_center_z) <= z_half))
            if np.any(inside):
                xy = np.empty((int(np.count_nonzero(inside)), 2), dtype=np.float32)
                xy[:, 0] = x[inside] - centre_xy
                xy[:, 1] = y[inside] - centre_xy
                selected_chunks.append(xy)

    if not selected_chunks:
        raise ValueError('The requested slice contains no ASTRA random voids.')
    return np.concatenate(selected_chunks, axis=0), int(header.get('NITER', -1))


def _select_real_xy(positions: np.ndarray, box_size: float, xy_size: float,
                    slice_center_z: float, slice_thickness: float) -> np.ndarray:
    centre_xy = np.float32(0.5 * box_size)
    xy_half = np.float32(0.5 * xy_size)
    z_half = np.float32(0.5 * slice_thickness)
    centred_xy = positions[:, :2] - centre_xy
    inside = ((np.abs(centred_xy[:, 0]) <= xy_half)
              & (np.abs(centred_xy[:, 1]) <= xy_half)
              & (np.abs(positions[:, 2] - slice_center_z) <= z_half))
    if not np.any(inside):
        raise ValueError('The requested slice contains no real halos.')
    return centred_xy[inside]


def _make_figure(real_xy: np.ndarray, random_xy: np.ndarray,
                 n_iterations: int, args) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7.2, 7.2))
    fig.subplots_adjust(left=0.15, right=0.96, bottom=0.12, top=0.84)

    # Random voids are drawn first; real halos remain readable on top.
    ax.scatter(random_xy[:, 0], random_xy[:, 1],
               s=args.random_marker_size, c='#D62728', alpha=args.random_alpha,
               linewidths=0, rasterized=True, zorder=1)
    ax.scatter(real_xy[:, 0], real_xy[:, 1],
               s=args.real_marker_size, c='#202020', alpha=0.88,
               linewidths=0, rasterized=True, zorder=10)

    random_label = ('Random voids' if n_iterations < 0
                    else rf'Random voids ({n_iterations} iter.)')
    handles = [
        Line2D([0], [0], marker='o', linestyle='none', markerfacecolor='#D62728',
               markeredgecolor='none', markersize=6, label=random_label),
        Line2D([0], [0], marker='o', linestyle='none', markerfacecolor='#202020',
               markeredgecolor='none', markersize=6, label='Real halos'),
    ]
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, 0.965),
               ncol=2, frameon=False, handletextpad=0.5, columnspacing=2.0)
    _style_axis(ax, args.xy_size)
    return fig


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    if args.realization < 0:
        parser.error('--realization must be non-negative.')
    if args.box_size <= 0:
        parser.error('--box-size must be positive.')
    if args.xy_size <= 0 or args.xy_size > args.box_size:
        parser.error('--xy-size must be positive and no larger than --box-size.')
    if args.slice_thickness <= 0 or args.slice_thickness > args.box_size:
        parser.error('--slice-thickness must be positive and no larger than --box-size.')
    if args.real_marker_size <= 0 or args.random_marker_size <= 0:
        parser.error('Marker sizes must be positive.')
    if not 0 < args.random_alpha <= 1:
        parser.error('--random-alpha must lie in (0, 1].')
    if args.chunk_rows <= 0:
        parser.error('--chunk-rows must be positive.')

    parameter = str(args.parameter).strip()
    relative = Path(parameter) / str(args.realization)
    catalogue_root = args.input_root.expanduser().resolve() / relative
    random_path = (args.astra_root.expanduser().resolve() / relative
                   / f'group_{args.snapshot:03d}_random_voids.fits.gz')
    slice_center_z = (0.5 * args.box_size if args.slice_center_z is None
                      else float(args.slice_center_z))
    z_lower = slice_center_z - 0.5 * args.slice_thickness
    z_upper = slice_center_z + 0.5 * args.slice_thickness
    if z_lower < 0 or z_upper > args.box_size:
        parser.error('The requested z slice lies outside the simulation box.')

    print(f'Loading real GroupPos from {catalogue_root} ...', flush=True)
    positions = _load_positions(catalogue_root, args.snapshot, args.readfof_path)
    real_xy = _select_real_xy(
        positions, args.box_size, args.xy_size, slice_center_z,
        args.slice_thickness)
    del positions
    print(f'Loading ASTRA random voids from {random_path} ...', flush=True)
    random_xy, n_iterations = _load_random_void_xy(
        random_path, parameter, args.realization, args.snapshot,
        args.box_size, args.xy_size, slice_center_z, args.slice_thickness,
        args.chunk_rows)

    print(f'Central x-y window: {args.xy_size:g} Mpc/h; z range: '
          f'[{z_lower:g}, {z_upper:g}] Mpc/h; real halos={len(real_xy):,}; '
          f'random voids={len(random_xy):,}; ASTRA iterations={n_iterations}',
          flush=True)

    _configure_matplotlib(use_tex=not args.no_tex)
    fig = _make_figure(real_xy, random_xy, n_iterations, args)
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=args.dpi, bbox_inches=None)
    plt.close(fig)
    print(f'Wrote {output}', flush=True)


if __name__ == '__main__':
    main()
