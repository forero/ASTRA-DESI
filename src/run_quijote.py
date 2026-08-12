import argparse
import os

from desiproc.quijote import (DEFAULT_BOX_ORIGIN,
                              DEFAULT_BOX_SIZE,
                              DEFAULT_INPUT_ROOT,
                              DEFAULT_N_ITERATIONS,
                              DEFAULT_OUTPUT_ROOT,
                              QuijoteRunConfig,
                              read_manifest_entry,
                              run_quijote_pipeline)


def _selection(args, parser):
    if args.manifest:
        if args.parameter is not None or args.realization is not None:
            parser.error('--manifest cannot be combined with --parameter/--realization')
        task_index = args.task_index
        if task_index is None:
            value = os.environ.get('SLURM_ARRAY_TASK_ID')
            if value is None:
                parser.error('--task-index is required outside a SLURM array')
            task_index = int(value)
        return read_manifest_entry(args.manifest, task_index)
    if args.parameter is None or args.realization is None:
        parser.error('provide both --parameter and --realization, or use --manifest')
    return args.parameter, int(args.realization)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-root', default=DEFAULT_INPUT_ROOT,
                        help='FoF root containing PARAM/REALIZATION '
                             f'(default: {DEFAULT_INPUT_ROOT})')
    parser.add_argument('--output-root', default=DEFAULT_OUTPUT_ROOT,
                        help=f'Compact ASTRA output root (default: {DEFAULT_OUTPUT_ROOT})')
    parser.add_argument('--parameter', help='Quijote parameter directory, e.g. fiducial or s8_p')
    parser.add_argument('--realization', type=int, help='Quijote realization number')
    parser.add_argument('--manifest', help='Two-column parameter/realization task manifest')
    parser.add_argument('--task-index', type=int, default=None,
                        help='Zero-based manifest row; defaults to SLURM_ARRAY_TASK_ID')
    parser.add_argument('--n-iterations', '--iterations', dest='n_iterations', type=int,
                        default=DEFAULT_N_ITERATIONS,
                        help='Independent ASTRA random iterations '
                             f'(default: {DEFAULT_N_ITERATIONS})')
    parser.add_argument('--random-seed', type=int, default=0,
                        help='Base seed; parameter, realization and iteration are mixed into it')
    parser.add_argument('--box-origin', nargs=3, type=float,
                        metavar=('X0', 'Y0', 'Z0'), default=DEFAULT_BOX_ORIGIN)
    parser.add_argument('--box-size', type=float, default=DEFAULT_BOX_SIZE,
                        help=f'Quijote box side in Mpc/h (default: {DEFAULT_BOX_SIZE:g})')
    parser.add_argument('--r-lower', type=float, default=-0.25)
    parser.add_argument('--r-med', type=float, default=0.25)
    parser.add_argument('--r-upper', type=float, default=0.65)
    parser.add_argument('--iteration-workers', type=int, default=None,
                        help='Concurrent iterations; default: 1 interactively, '
                             'up to allocated CPUs under SLURM')
    parser.add_argument('--count-chunk-vertices', type=int, default=250_000,
                        help='Vertices per bounded CSR-neighbour counting chunk')
    parser.add_argument('--io-chunk-rows', type=int, default=500_000,
                        help='Rows per compact FITS output chunk')
    parser.add_argument('--qhull-options', default=None,
                        help='Advanced options forwarded to scipy.spatial.Delaunay')
    parser.add_argument('--readfof-path', default=None,
                        help='Path to Pylians readfof.py; normally auto-detected '
                             'from ~/venvs/pylians')
    parser.add_argument('--force', action='store_true',
                        help='Replace an existing output even when it is incompatible')
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    parameter, realization = _selection(args, parser)
    config = QuijoteRunConfig(
        input_root=args.input_root,
        output_root=args.output_root,
        parameter=parameter,
        realization=realization,
        n_iterations=args.n_iterations,
        random_seed=args.random_seed,
        box_origin=tuple(args.box_origin),
        box_size=args.box_size,
        r_lower=args.r_lower,
        r_med=args.r_med,
        r_upper=args.r_upper,
        iteration_workers=args.iteration_workers,
        count_chunk_vertices=args.count_chunk_vertices,
        io_chunk_rows=args.io_chunk_rows,
        qhull_options=args.qhull_options,
        readfof_path=args.readfof_path,
        force=args.force)
    run_quijote_pipeline(config)


if __name__ == '__main__':
    main()
