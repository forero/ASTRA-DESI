import argparse
import os
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path

DEFAULT_INPUT_ROOT = '/pscratch/sd/v/vtorresg/quijotes/Halos/FoF'
DEFAULT_ASTRA_OUTPUT_ROOT = '/pscratch/sd/v/vtorresg/quijotes/ASTRA/FoF'
SNAPSHOT = 3
PROBABILITY_COLUMNS = ('PVOID', 'PSHEET', 'PFILAMENT', 'PKNOT')
RANDOM_VOID_COLUMNS = ('RANDITER', 'RANDINDEX', 'X', 'Y', 'Z')


def discover(input_root, parameters=None):
    root = Path(input_root).expanduser().resolve()
    selected = set(parameters) if parameters else None
    entries = []
    pattern = f'*/*/groups_{SNAPSHOT:03d}/group_tab_{SNAPSHOT:03d}.0'
    for first_file in root.glob(pattern):
        relative = first_file.relative_to(root)
        if len(relative.parts) != 4:
            continue
        parameter, realization_text, _, _ = relative.parts
        if selected is not None and parameter not in selected:
            continue
        try:
            realization = int(realization_text)
        except ValueError:
            continue
        if realization < 0:
            continue
        entries.append((parameter, realization))
    return sorted(set(entries), key=lambda item: (item[0], item[1]))


def _complete_astra_products(output_root, parameter, realization,
                             expected_iterations, validate_headers=False):
    root = (Path(output_root).expanduser().resolve()
            / parameter / str(int(realization)))
    products = ((root / f'group_{SNAPSHOT:03d}_probability.fits.gz', PROBABILITY_COLUMNS, None),
                (root / f'group_{SNAPSHOT:03d}_random_voids.fits.gz', RANDOM_VOID_COLUMNS, 'RANDVOID'))
    if not all(path.is_file() for path, _, _ in products):
        return False
    if not validate_headers:
        return True

    import fitsio

    expected = {'PARAM': parameter,
                'REALIZ': int(realization),
                'SNAPNUM': SNAPSHOT,
                'NITER': int(expected_iterations)}
    try:
        for path, columns, product in products:
            with fitsio.FITS(str(path), mode='r') as hdus:
                if len(hdus) < 2:
                    return False
                table = hdus[1]
                if tuple(table.get_colnames()) != columns:
                    return False
                header = table.read_header()
                if any(header.get(key) != value for key, value in expected.items()):
                    return False
                if product is not None and header.get('PRODUCT') != product:
                    return False
        return True
    except Exception:
        return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-root', default=DEFAULT_INPUT_ROOT)
    parser.add_argument('--parameter', action='append', default=None, help='Only include this parameter directory; repeatable')
    parser.add_argument('--skip-complete', action='store_true', help='Omit simulations with both compatible ASTRA products')
    parser.add_argument('--astra-output-root', default=DEFAULT_ASTRA_OUTPUT_ROOT)
    parser.add_argument('--expected-iterations', type=int, default=100, help='NITER required by --skip-complete (default: 100)')
    parser.add_argument('--validation-workers', type=int, default=16, help='Concurrent FITS-header checks for --skip-complete')
    parser.add_argument('--validate-complete', action='store_true', help='Open and validate completed FITS headers')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()

    entries = discover(args.input_root, args.parameter)
    if not entries:
        raise RuntimeError(
            f'No groups_{SNAPSHOT:03d} FoF catalogues found under {args.input_root}')
    discovered = len(entries)
    if args.skip_complete:
        if args.expected_iterations <= 0 or args.validation_workers <= 0:
            parser.error('--expected-iterations and --validation-workers must be positive')
        check = partial(_complete_astra_products, args.astra_output_root,
                        expected_iterations=args.expected_iterations,
                        validate_headers=args.validate_complete)
        with ThreadPoolExecutor(max_workers=args.validation_workers) as pool:
            complete = pool.map(lambda entry: check(entry[0], entry[1]), entries)
            entries = [entry for entry, is_complete in zip(entries, complete)
                       if not is_complete]

    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f'.{output.name}.tmp.{os.getpid()}')
    try:
        with open(temporary, 'w', encoding='utf-8') as stream:
            stream.write('# zero-based task rows: parameter realization\n')
            for parameter, realization in entries:
                stream.write(f'{parameter} {realization}\n')
        os.replace(temporary, output)
    finally:
        if temporary.exists():
            temporary.unlink()
    print(f'wrote {len(entries)} Quijote tasks to {output} '
          f'(discovered={discovered}, skipped_complete={discovered-len(entries)})')


if __name__ == '__main__':
    main()