import argparse
import os
import re
from pathlib import Path


COSMOLOGY_PATTERN = re.compile(r'^c(?P<number>\d+)_ph(?P<phase>\d+)$', re.IGNORECASE)
HOD_PATTERN = re.compile(r'^hod(?P<number>\d+)\.fits$', re.IGNORECASE)
DEFAULT_INPUT_ROOT = '/pscratch/sd/n/ntbfin/emulator/hods/z0.5/yuan23_prior'


def discover(input_root, phase=0, simulation_seed=0):
    root = Path(input_root).expanduser().resolve()
    entries = []
    for cosmology_dir in root.glob('c*_ph*'):
        match = COSMOLOGY_PATTERN.fullmatch(cosmology_dir.name)
        if match is None or int(match.group('phase')) != int(phase):
            continue
        cosmology_number = int(match.group('number'))
        seed_dir = cosmology_dir / f'seed{int(simulation_seed)}'
        for hod_path in seed_dir.glob('hod*.fits'):
            hod_match = HOD_PATTERN.fullmatch(hod_path.name)
            if hod_match is None:
                continue
            hod_number = int(hod_match.group('number'))
            entries.append((cosmology_number, hod_number))
    entries.sort()
    return [(f'c{cosmology:03d}', f'hod{hod:03d}') for cosmology, hod in entries]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-root', default=DEFAULT_INPUT_ROOT)
    parser.add_argument('--output', required=True)
    parser.add_argument('--phase', type=int, default=0)
    parser.add_argument('--simulation-seed', type=int, default=0)
    args = parser.parse_args()

    entries = discover(args.input_root, args.phase, args.simulation_seed)
    if not entries:
        raise RuntimeError(f'No HOD FITS catalogues found under {args.input_root}')

    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f'.{output.name}.tmp.{os.getpid()}')
    try:
        with open(temporary, 'w', encoding='utf-8') as stream:
            stream.write('# zero-based task rows: cosmology hod\n')
            for cosmology, hod in entries:
                stream.write(f'{cosmology} {hod}\n')
        os.replace(temporary, output)
    finally:
        if temporary.exists():
            temporary.unlink()
    print(f'wrote {len(entries)} HOD tasks to {output}')


if __name__ == '__main__':
    main()