#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import importlib
import importlib.util
import os
import shutil
from pathlib import Path

import fitsio
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


DEFAULT_INPUT_ROOT = Path('/pscratch/sd/v/vtorresg/quijotes/Halos/FoF')
DEFAULT_ASTRA_ROOT = Path('/pscratch/sd/v/vtorresg/quijotes/ASTRA/FoF')
DEFAULT_OUTPUT = Path('figs/quijote_fiducial_0_cube.png')
DEFAULT_READFOF = Path(
    '/global/homes/v/vtorresg/venvs/pylians/lib64/python3.6/site-packages/readfof.py')
SNAPSHOT = 3

PROBABILITY_COLUMNS = ('PVOID', 'PSHEET', 'PFILAMENT', 'PKNOT')
CLASS_LABELS = (r"Void", r"Sheet", 'Filament', 'Knot')
# CLASS_COLORS = ('#0072B2', '#E69F00', '#009E73', '#CC79A7')
CLASS_COLORS = ('#7B6FD0', '#3BA99C', '#E9A23B', '#C94C5C')


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=('Two-panel 3D view of a Quijote halo cube: spatial '
                     'distribution and ASTRA hard classification.'))
    parser.add_argument('--input-root', type=Path, default=DEFAULT_INPUT_ROOT,
                        help='Quijote FoF root containing PARAMETER/REALIZATION.')
    parser.add_argument('--astra-root', type=Path, default=DEFAULT_ASTRA_ROOT,
                        help='ASTRA root containing PARAMETER/REALIZATION.')
    parser.add_argument('--parameter', default='fiducial',
                        help='Quijote parameter directory (default: fiducial).')
    parser.add_argument('--realization', type=int, default=0,
                        help='Quijote realization (default: 0).')
    parser.add_argument('--snapshot', type=int, default=SNAPSHOT,
                        help=f'FoF snapshot (default: {SNAPSHOT}, z=0.5).')
    parser.add_argument('--redshift', type=float, default=0.5,
                        help='Redshift shown in the figure title (default: 0.5).')
    parser.add_argument('--box-size', type=float, default=1000.0,
                        help='Full simulation-box side in Mpc/h (default: 1000).')
    parser.add_argument('--subcube-size', type=float, default=500.0,
                        help='Central plotted-cube side in Mpc/h (default: 500).')
    parser.add_argument('--marker-size', type=float, default=1.25,
                        help='Scatter marker area in points squared (default: 1.25).')
    parser.add_argument('--elev', type=float, default=22.0,
                        help='Camera elevation in degrees.')
    parser.add_argument('--azim', type=float, default=-55.0,
                        help='Camera azimuth in degrees.')
    parser.add_argument('--readfof-path', type=Path, default=None,
                        help='Path to Pylians readfof.py if it is not importable.')
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT,
                        help=f'Output figure (default: {DEFAULT_OUTPUT}).')
    parser.add_argument('--dpi', type=int, default=360,
                        help='Resolution for raster output (default: 360).')
    parser.add_argument('--no-tex', action='store_true',
                        help='Disable LaTeX rendering (useful only as a fallback).')
    return parser


def _import_readfof(explicit_path: Path | None):
    if explicit_path is None:
        try:
            return importlib.import_module('readfof')
        except ModuleNotFoundError:
            pass

    candidates = []
    if explicit_path is not None:
        candidates.append(explicit_path)
    env_path = os.environ.get('PYLIANS_READFOF', '').strip()
    if env_path:
        candidates.append(Path(env_path))
    candidates.append(DEFAULT_READFOF)

    checked = []
    for supplied in candidates:
        candidate = supplied / 'readfof.py' if supplied.is_dir() else supplied
        candidate = candidate.expanduser().resolve()
        checked.append(str(candidate))
        if not candidate.is_file():
            continue
        spec = importlib.util.spec_from_file_location('_plot_quijote_readfof', candidate)
        if spec is None or spec.loader is None:
            continue
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    raise ModuleNotFoundError(
        'Could not import Pylians readfof. Use --readfof-path. Checked: '
        + ', '.join(checked))


def _load_positions(catalogue_root: Path, snapshot: int,
                    readfof_path: Path | None) -> np.ndarray:
    readfof = _import_readfof(readfof_path)
    catalogue = readfof.FoF_catalog(
        str(catalogue_root), snapshot, long_ids=False, swap=False,
        SFR=False, read_IDs=False)
    # Quijote GroupPos is stored in kpc/h; plot in Mpc/h.
    positions = np.asarray(catalogue.GroupPos, dtype=np.float32) / np.float32(1000.0)
    positions = np.array(positions, dtype=np.float32, order='C', copy=True)
    del catalogue
    gc.collect()
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError(f'Unexpected GroupPos shape: {positions.shape}')
    if not np.all(np.isfinite(positions)):
        raise ValueError('GroupPos contains non-finite coordinates.')
    return positions


def _load_classes(probability_path: Path, expected_rows: int, parameter: str,
                  realization: int, snapshot: int) -> tuple[np.ndarray, np.ndarray]:
    if not probability_path.is_file():
        raise FileNotFoundError(f'ASTRA probability file not found: {probability_path}')

    with fitsio.FITS(str(probability_path), mode='r') as hdus:
        if len(hdus) < 2:
            raise ValueError(f'Missing table extension in {probability_path}')
        header = hdus[1].read_header()
        rows = hdus[1].read(columns=list(PROBABILITY_COLUMNS))

    expected_header = {'PARAM': parameter, 'REALIZ': realization,
                       'SNAPNUM': snapshot, 'ROWALIGN': 'GROUPPOS'}
    for key, expected in expected_header.items():
        actual = header.get(key)
        if isinstance(actual, bytes):
            actual = actual.decode('ascii', errors='replace').strip()
        if actual != expected:
            raise ValueError(
                f'ASTRA header mismatch for {key}: expected {expected!r}, '
                f'found {actual!r}.')
    if len(rows) != expected_rows:
        raise ValueError(
            f'Row-alignment failure: {len(rows):,} probability rows but '
            f'{expected_rows:,} GroupPos rows.')

    probabilities = np.column_stack([rows[name] for name in PROBABILITY_COLUMNS])
    del rows
    if not np.all(np.isfinite(probabilities)):
        raise ValueError('ASTRA probabilities contain non-finite values.')
    if not np.allclose(probabilities.sum(axis=1), 1.0, rtol=0.0, atol=2e-5):
        raise ValueError('ASTRA probabilities do not sum to one.')

    classes = np.argmax(probabilities, axis=1).astype(np.uint8)
    confidence = np.max(probabilities, axis=1)
    return classes, confidence


def _select_central_subcube(positions: np.ndarray, classes: np.ndarray,
                            confidence: np.ndarray, box_size: float,
                            subcube_size: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Select every halo in the central subcube and place its origin at the centre."""
    centre = np.full(3, 0.5 * box_size, dtype=np.float32)
    half_side = np.float32(0.5 * subcube_size)
    centred_positions = positions - centre
    inside = np.all(np.abs(centred_positions) <= half_side, axis=1)
    if not np.any(inside):
        raise ValueError('The central subcube contains no halos.')
    return centred_positions[inside], classes[inside], confidence[inside]


def _configure_matplotlib(use_tex: bool) -> None:
    if use_tex and shutil.which('latex') is None:
        raise RuntimeError('LaTeX was requested but no latex executable was found. '
                           'Install/load LaTeX or pass --no-tex.')
    mpl.rcParams.update({
        'text.usetex': use_tex,
        # 'font.family': 'serif',
        # 'font.serif': ['Computer Modern Roman', 'DejaVu Serif'],
        'font.size': 13,
        'axes.labelsize': 14,
        'axes.titlesize': 14,
        'legend.fontsize': 14.5,
        'axes.facecolor': 'white',
        'figure.facecolor': 'white',
        'savefig.facecolor': 'white',
    })
    if use_tex:
        mpl.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'


def _draw_box(ax, subcube_size: float) -> None:
    half_side = 0.5 * subcube_size
    corners = np.array([[x, y, z]
                        for x in (-half_side, half_side)
                        for y in (-half_side, half_side)
                        for z in (-half_side, half_side)])
    for i, first in enumerate(corners):
        for second in corners[i + 1:]:
            if np.count_nonzero(first != second) == 1:
                ax.plot(*zip(first, second), color='#454545', lw=1.,
                        alpha=1, zorder=100000)


def _style_axis(ax, subcube_size: float, elev: float, azim: float) -> None:
    unit = r'h^{-1}\,\mathrm{Mpc}'
    ax.set_xlabel(rf'$x\;[{unit}]$', labelpad=8)
    ax.set_ylabel(rf'$y\;[{unit}]$', labelpad=8)
    ax.set_zlabel(rf'$z\;[{unit}]$', labelpad=8)
    half_side = 0.5 * subcube_size
    limits = (-half_side, half_side)
    ax.set(xlim=limits, ylim=limits, zlim=limits)
    ticks = np.linspace(-half_side, half_side, 5)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_zticks(ticks)
    ax.tick_params(axis='x', pad=1)
    ax.tick_params(axis='y', pad=12)
    ax.tick_params(axis='z', pad=3)
    ax.view_init(elev=elev, azim=azim)
    ax.set_box_aspect((1, 1, 1))
    ax.grid(True, color='#d9d9d9', linewidth=0.35, alpha=0.7)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_facecolor((1.0, 1.0, 1.0, 1.0))
        axis.pane.set_edgecolor('#bdbdbd')
        axis.pane.set_alpha(0.16)
    _draw_box(ax, subcube_size)


def _make_figure(positions: np.ndarray, classes: np.ndarray,
                 confidence: np.ndarray, args) -> tuple[plt.Figure, np.ndarray]:
    fig = plt.figure(figsize=(14.2, 6.8), constrained_layout=False)
    # Keep the explicit scatter zorders below. Without this, mplot3d replaces
    # them with its own depth-based ordering and can hide the rare knots.
    axes = np.array([
        fig.add_subplot(1, 2, 1, projection='3d', computed_zorder=False),
        fig.add_subplot(1, 2, 2, projection='3d', computed_zorder=False),
    ])
    # Reserve a band above the cubes for the classification legend.
    fig.subplots_adjust(left=0.025, right=0.94, bottom=0.08, top=0.82, wspace=0.01)

    axes[0].scatter(positions[:, 0], positions[:, 1], positions[:, 2],
                    s=args.marker_size, c='#303030', alpha=0.5,)
    # axes[0].set_title(
    #     r'$(a)\quad \mathrm{Distribuci\acute{o}n\ de\ halos\ (subcubo\ central)}$',
    #     pad=14)

    legend_handles = []
    full_counts = np.bincount(classes, minlength=len(CLASS_LABELS))
    for class_id, (label, color) in enumerate(zip(CLASS_LABELS, CLASS_COLORS)):
        selected = classes == class_id
        points = positions[selected]
        # print(class_id, (label, color))
        if class_id == 0:
            # Make voids slightly larger to be more visible.
            axes[1].scatter(points[:, 0], points[:, 1], points[:, 2],
                            s=args.marker_size , c=color, alpha=0.7,
                            zorder=0)
        elif class_id == 1:
            axes[1].scatter(points[:, 0], points[:, 1], points[:, 2],
                            s=args.marker_size, c=color, alpha=0.8,
                            zorder=10)
        elif class_id == 2:
            axes[1].scatter(points[:, 0], points[:, 1], points[:, 2],
                            s=args.marker_size, c=color, alpha=0.9,
                            zorder=100)
        elif class_id == 3:
            axes[1].scatter(points[:, 0], points[:, 1], points[:, 2],
                            s=args.marker_size * 1.5, c=color, alpha=1.0,
                            zorder=100000)

        fraction = 100.0 * full_counts[class_id] / len(classes)
        legend_handles.append(Line2D(
            [0], [0], marker='o', linestyle='none', markerfacecolor=color,
            markeredgecolor='none', markersize=5.5,
            label=rf'{label}: ${fraction:.1f}\%$'))
    # axes[1].set_title(
    #     r'$(b)\quad \mathrm{Clasificaci\acute{o}n\ ASTRA}\ '
    #     r'[\arg\max(P_i)]$', pad=14)
    # Match the 2D figure: centre the complete legend in figure coordinates so
    # it remains outside both cubes and fully inside the PNG canvas.
    fig.legend(handles=legend_handles, loc='upper center',
               bbox_to_anchor=(0.5, 0.965), ncol=4, frameon=False,
               facecolor='white', edgecolor='#bdbdbd', framealpha=0.94,
               borderpad=0.6, handletextpad=0.4, columnspacing=1.8)

    for ax in axes:
        _style_axis(ax, args.subcube_size, args.elev, args.azim)

    mean_confidence = float(np.mean(confidence))
    parameter_label = str(args.parameter).replace('_', r'\_')
    # fig.suptitle(
    #     rf'$\mathrm{{Quijote\ {parameter_label}}},\ '
    #     rf'\mathrm{{realizaci\acute{{o}}n}}\ {args.realization},\ '
    #     rf'z={args.redshift:g}\qquad '
    #     rf'L={args.subcube_size:g}\,h^{{-1}}\,\mathrm{{Mpc}}\qquad '
    #     rf'\langle P_{{\max}}\rangle={mean_confidence:.3f}$',
    #     y=0.975, fontsize=14)
    return fig, axes


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    if args.realization < 0:
        parser.error('--realization must be non-negative.')
    if args.box_size <= 0:
        parser.error('--box-size must be positive.')
    if args.subcube_size <= 0 or args.subcube_size > args.box_size:
        parser.error('--subcube-size must be positive and no larger than --box-size.')
    if args.marker_size <= 0:
        parser.error('--marker-size must be positive.')

    parameter = str(args.parameter).strip()
    relative = Path(parameter) / str(args.realization)
    catalogue_root = args.input_root.expanduser().resolve() / relative
    probability_path = (args.astra_root.expanduser().resolve() / relative
                        / f'group_{args.snapshot:03d}_probability.fits.gz')

    print(f'Loading GroupPos from {catalogue_root} ...', flush=True)
    positions = _load_positions(catalogue_root, args.snapshot, args.readfof_path)
    print(f'Loading ASTRA probabilities from {probability_path} ...', flush=True)
    classes, confidence = _load_classes(
        probability_path, len(positions), parameter, args.realization, args.snapshot)
    total_halos = len(positions)
    positions, classes, confidence = _select_central_subcube(
        positions, classes, confidence, args.box_size, args.subcube_size)

    class_counts = np.bincount(classes, minlength=len(CLASS_LABELS))
    summary = ', '.join(f'{name}={int(count):,}' for name, count in
                        zip(('void', 'sheet', 'filament', 'knot'), class_counts))
    lower = 0.5 * (args.box_size - args.subcube_size)
    upper = lower + args.subcube_size
    print(f'Full box: {total_halos:,} halos; central range: '
          f'[{lower:g}, {upper:g}] Mpc/h on each axis; plotted: '
          f'{len(positions):,} halos (all); {summary}', flush=True)

    _configure_matplotlib(use_tex=not args.no_tex)
    fig, _ = _make_figure(positions, classes, confidence, args)
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(output, dpi=args.dpi, bbox_inches=None)
    plt.close(fig)
    print(f'Wrote {output}', flush=True)


if __name__ == '__main__':
    main()
