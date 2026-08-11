import argparse
import os

from desiproc.hod import (DEFAULT_BOX_ORIGIN,
                          DEFAULT_BOX_SIZE,
                          DEFAULT_COORD_COLUMNS,
                          DEFAULT_ID_COLUMN,
                          DEFAULT_INPUT_ROOT,
                          DEFAULT_N_ITERATIONS,
                          HODRunConfig,
                          normalize_cosmology,
                          normalize_hod,
                          read_manifest_entry,
                          run_hod_pipeline)


def _selection(args, parser):
    if args.manifest:
        if args.cosmology or args.hod:
            parser.error('--manifest cannot be combined with --cosmology/--hod')
        task_index = args.task_index
        if task_index is None:
            value = os.environ.get('SLURM_ARRAY_TASK_ID')
            if value is None:
                parser.error('--task-index is required outside a SLURM array')
            task_index = int(value)
        cosmology, hod = read_manifest_entry(args.manifest, task_index)
        return cosmology, hod, int(args.phase)
    if not args.cosmology or not args.hod:
        parser.error('provide both --cosmology and --hod, or use --manifest')
    cosmology, embedded_phase = normalize_cosmology(args.cosmology)
    phase = int(args.phase if embedded_phase is None else embedded_phase)
    return cosmology, normalize_hod(args.hod), phase


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-root', default=DEFAULT_INPUT_ROOT)
    parser.add_argument('--output-root', required=True)
    parser.add_argument('--cosmology', help='Cosmology, e.g. c000')
    parser.add_argument('--hod', help='HOD realization, e.g. hod000')
    parser.add_argument('--manifest', help='Two-column cosmology/HOD task manifest')
    parser.add_argument('--task-index', type=int, default=None, help='defaults to SLURM_ARRAY_TASK_ID')
    parser.add_argument('--phase', type=int, default=0)
    parser.add_argument('--simulation-seed', type=int, default=0, help='(default: seed0)')
    parser.add_argument('--n-iterations', '--iterations', dest='n_iterations', type=int, default=DEFAULT_N_ITERATIONS, help=f'Independent ASTRA random iterations (default: {DEFAULT_N_ITERATIONS})')
    parser.add_argument('--random-seed', type=int, default=0, help='Base seed; cosmology, HOD and iteration are mixed into it')
    parser.add_argument('--coordinate-columns', nargs=3, metavar=('X', 'Y', 'Z'), default=DEFAULT_COORD_COLUMNS)
    parser.add_argument('--id-column', default=DEFAULT_ID_COLUMN)
    parser.add_argument('--box-origin', nargs=3, type=float, metavar=('X0', 'Y0', 'Z0'), default=DEFAULT_BOX_ORIGIN)
    parser.add_argument('--box-size', type=float, default=DEFAULT_BOX_SIZE, help='(default: 2000)')
    parser.add_argument('--r-lower', type=float, default=-0.25)
    parser.add_argument('--r-med', type=float, default=0.25)
    parser.add_argument('--r-upper', type=float, default=0.65)
    parser.add_argument('--iteration-workers', type=int, default=None, help='Concurrent iterations. Default: 1 interactively, up to CPUs under SLURM.')
    parser.add_argument('--count-chunk-vertices', type=int, default=250_000, help='Vertices per bounded CSR-neighbour counting chunk')
    parser.add_argument('--io-chunk-rows', type=int, default=500_000, help='Rows per FITS write chunk')
    parser.add_argument('--qhull-options', default=None, help='Advanced options forwarded to scipy.spatial.Delaunay')
    parser.add_argument('--no-save-classification', action='store_true', help='Only save the final probability file (classification directory remains empty)')
    parser.add_argument('--force', action='store_true', help='Replace even when compatible outputs already exist')
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    cosmology, hod, phase = _selection(args, parser)
    config = HODRunConfig(input_root=args.input_root, output_root=args.output_root,
                          cosmology=cosmology, hod=hod, phase=phase,
                          simulation_seed=args.simulation_seed,
                          n_iterations=args.n_iterations, random_seed=args.random_seed,
                          coordinate_columns=tuple(args.coordinate_columns),
                          id_column=args.id_column,
                          box_origin=tuple(args.box_origin), box_size=args.box_size,
                          r_lower=args.r_lower, r_med=args.r_med, r_upper=args.r_upper,
                          iteration_workers=args.iteration_workers,
                          count_chunk_vertices=args.count_chunk_vertices,
                          io_chunk_rows=args.io_chunk_rows,
                          qhull_options=args.qhull_options,
                          save_classification=not args.no_save_classification,
                          force=args.force)
    run_hod_pipeline(config)


if __name__ == '__main__':
    main()