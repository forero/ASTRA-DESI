import argparse
import os
from pathlib import Path

DEFAULT_INPUT_ROOT = '/pscratch/sd/v/vtorresg/quijotes/Halos/FoF'
SNAPSHOT = 3


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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-root', default=DEFAULT_INPUT_ROOT)
    parser.add_argument('--parameter', action='append', default=None,
                        help='Only include this parameter directory; repeatable')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()

    entries = discover(args.input_root, args.parameter)
    if not entries:
        raise RuntimeError(
            f'No groups_{SNAPSHOT:03d} FoF catalogues found under {args.input_root}')

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
    print(f'wrote {len(entries)} Quijote tasks to {output}')


if __name__ == '__main__':
    main()
