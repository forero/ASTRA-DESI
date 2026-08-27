from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from plot_quijote_cube import (CLASS_COLORS, CLASS_LABELS, DEFAULT_ASTRA_ROOT,
                               DEFAULT_INPUT_ROOT, SNAPSHOT,
                               _configure_matplotlib, _load_classes,
                               _load_positions)


DEFAULT_OUTPUT = Path('figs/quijote_fiducial_0_slice_xy_z20.png')


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-root', type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument('--astra-root', type=Path, default=DEFAULT_ASTRA_ROOT)
    parser.add_argument('--parameter', default='fiducial')
    parser.add_argument('--realization', type=int, default=0)
    parser.add_argument('--snapshot', type=int, default=SNAPSHOT)
    parser.add_argument('--box-size', type=float, default=1000.0,
                        help='Full simulation-box side in Mpc/h (default: 1000).')
    parser.add_argument('--xy-size', type=float, default=500.0,
                        help='Central x-y window side in Mpc/h (default: 500).')
    parser.add_argument('--slice-thickness', type=float, default=100.0,
                        help='Full z-slice thickness in Mpc/h (default: 20).')
    parser.add_argument('--slice-center-z', type=float, default=None,
                        help='Slice centre in original coordinates; default: box centre.')
    parser.add_argument('--marker-size', type=float, default=4.0,
                        help='Scatter marker area in points squared (default: 4).')
    parser.add_argument('--readfof-path', type=Path, default=None)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--dpi', type=int, default=360)
    parser.add_argument('--no-tex', action='store_true')
    return parser


def _select_slice(positions: np.ndarray, classes: np.ndarray,
                  confidence: np.ndarray, box_size: float, xy_size: float,
                  slice_center_z: float,
                  slice_thickness: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    centre_xy = np.float32(0.5 * box_size)
    centred_xy = positions[:, :2] - centre_xy
    xy_half = np.float32(0.5 * xy_size)
    z_half = np.float32(0.5 * slice_thickness)
    inside = ((np.abs(centred_xy[:, 0]) <= xy_half)
              & (np.abs(centred_xy[:, 1]) <= xy_half)
              & (np.abs(positions[:, 2] - slice_center_z) <= z_half))
    if not np.any(inside):
        raise ValueError('The requested x-y slice contains no halos.')
    return centred_xy[inside], classes[inside], confidence[inside]


def _style_axis(ax, xy_size: float) -> None:
    half_side = 0.5 * xy_size
    unit = r'h^{-1}\,\mathrm{Mpc}'
    ax.set_xlabel(rf'$x\;[{unit}]$')
    ax.set_ylabel(rf'$y\;[{unit}]$')
    ax.set_xlim(-half_side, half_side)
    ax.set_ylim(-half_side, half_side)
    ticks = np.linspace(-half_side, half_side, 5)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_aspect('equal', adjustable='box')
    ax.set_facecolor('white')
    ax.grid(True, color='#d9d9d9', linewidth=0.45, alpha=0.7, zorder=-10)
    for spine in ax.spines.values():
        spine.set_color('#555555')
        spine.set_linewidth(0.8)


def _make_figure(xy: np.ndarray, classes: np.ndarray, args) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(13.6, 6.2))
    fig.subplots_adjust(left=0.07, right=0.97, bottom=0.12, top=0.80, wspace=0.16)

    axes[0].scatter(xy[:, 0], xy[:, 1], s=args.marker_size,
                    c='#303030', alpha=0.72, linewidths=0,
                    rasterized=True, zorder=1)

    full_counts = np.bincount(classes, minlength=len(CLASS_LABELS))
    legend_handles = []
    zorders = (1, 10, 100, 1000)
    alphas = (0.70, 0.80, 0.90, 1.00)
    for class_id, (label, color) in enumerate(zip(CLASS_LABELS, CLASS_COLORS)):
        selected = classes == class_id
        point_size = args.marker_size * (1.8 if class_id == 3 else 1.0)
        axes[1].scatter(xy[selected, 0], xy[selected, 1],
                        s=point_size, c=color, alpha=alphas[class_id],
                        linewidths=0, rasterized=True, zorder=zorders[class_id])
        fraction = 100.0 * full_counts[class_id] / len(classes)
        legend_handles.append(Line2D(
            [0], [0], marker='o', linestyle='none', markerfacecolor=color,
            markeredgecolor='none', markersize=5.5,
            label=rf'{label}: ${fraction:.1f}\%$'))

    # Centre the legend in figure coordinates instead of anchoring it to the
    # right panel; this keeps the full four-column legend inside the PNG canvas.
    fig.legend(handles=legend_handles, loc='upper center',
               bbox_to_anchor=(0.5, 0.965), ncol=4, frameon=False,
               facecolor='white', edgecolor='#bdbdbd', framealpha=0.94,
               borderpad=0.6, handletextpad=0.4, columnspacing=1.8)
    for ax in axes:
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
    if args.marker_size <= 0:
        parser.error('--marker-size must be positive.')

    parameter = str(args.parameter).strip()
    relative = Path(parameter) / str(args.realization)
    catalogue_root = args.input_root.expanduser().resolve() / relative
    probability_path = (args.astra_root.expanduser().resolve() / relative
                        / f'group_{args.snapshot:03d}_probability.fits.gz')
    slice_center_z = (0.5 * args.box_size if args.slice_center_z is None
                      else float(args.slice_center_z))
    z_lower = slice_center_z - 0.5 * args.slice_thickness
    z_upper = slice_center_z + 0.5 * args.slice_thickness
    if z_lower < 0 or z_upper > args.box_size:
        parser.error('The requested z slice lies outside the simulation box.')

    print(f'Loading GroupPos from {catalogue_root} ...', flush=True)
    positions = _load_positions(catalogue_root, args.snapshot, args.readfof_path)
    print(f'Loading ASTRA probabilities from {probability_path} ...', flush=True)
    classes, confidence = _load_classes(
        probability_path, len(positions), parameter, args.realization, args.snapshot)
    xy, classes, confidence = _select_slice(
        positions, classes, confidence, args.box_size, args.xy_size,
        slice_center_z, args.slice_thickness)

    class_counts = np.bincount(classes, minlength=len(CLASS_LABELS))
    summary = ', '.join(f'{name}={int(count):,}' for name, count in
                        zip(('void', 'sheet', 'filament', 'knot'), class_counts))
    print(f'Central x-y window: {args.xy_size:g} Mpc/h; z range: '
          f'[{z_lower:g}, {z_upper:g}] Mpc/h; plotted: {len(xy):,} halos '
          f'(all); {summary}; mean Pmax={np.mean(confidence):.3f}', flush=True)

    _configure_matplotlib(use_tex=not args.no_tex)
    fig = _make_figure(xy, classes, args)
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=args.dpi, bbox_inches=None)
    plt.close(fig)
    print(f'Wrote {output}', flush=True)


if __name__ == '__main__':
    main()
