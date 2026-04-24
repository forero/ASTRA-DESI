import argparse, re
from pathlib import Path
import numpy as np
from astropy.table import Table

ZONE_RE = re.compile(r'^zone_(?P<zone>[^_.]+)', re.IGNORECASE)


def _existing(base, stem):
    for suffix in ('.fits.gz',):
        path = base / f'{stem}{suffix}'
        if path.exists():
            return path
    raise FileNotFoundError(f'No file found for stem {stem!r} in {base}')


def _discover_zones(raw_dir):
    found = set()
    for path in raw_dir.glob('zone_*' + '*.fits*'):
        match = ZONE_RE.search(path.name)
        if match:
            found.add(str(match.group('zone')).upper())

    def _zone_sort_key(value):
        text = str(value)
        return (0, int(text)) if text.isdigit() else (1, text)

    return sorted(found, key=_zone_sort_key)


def _load_table(path, include=None):
    if include is None:
        return Table.read(path, memmap=True)
    return Table.read(path, memmap=True)


def check_zone(zone, raw_dir, class_dir, prob_dir, expected_random):
    tag = zone
    raw_path = f'{raw_dir}/zone_{tag}.fits.gz'
    class_path = f'{class_dir}/zone_{tag}_classified.fits.gz'
    prob_path = f'{prob_dir}/zone_{tag}_probability.fits.gz'

    raw_tbl = _load_table(raw_path, include=('TARGETID', 'TRACERTYPE', 'RANDITER'))
    class_tbl = _load_table(class_path, include=('TARGETID', 'RANDITER', 'ISDATA', 'NDATA', 'NRAND', 'TRACERTYPE'))
    prob_tbl = _load_table(prob_path, include=('TARGETID',))

    randiter_raw = np.asarray(raw_tbl['RANDITER'], dtype=int)
    is_data_mask = randiter_raw == -1
    n_data = int(is_data_mask.sum())
    n_raw = len(raw_tbl)
    n_rand_rows = n_raw - n_data

    n_class = len(class_tbl)
    n_prob = len(prob_tbl)

    uniq_class_iters = np.unique(np.asarray(class_tbl['RANDITER'], dtype=int))
    eff_random = np.nan
    if n_data > 0:
        eff_random = (n_class - n_rand_rows) / float(n_data)

    result = {'zone': zone, 'raw': n_raw, 'prob': n_prob, 'class': n_class,
              'ndata': n_data, 'rand_rows': n_rand_rows,
              'unique_iters': uniq_class_iters.size,
              'min_iter': int(uniq_class_iters.min()) if uniq_class_iters.size else 0,
              'max_iter': int(uniq_class_iters.max()) if uniq_class_iters.size else -1,
              'effective_random': eff_random,}

    if expected_random is not None and np.isfinite(eff_random):
        result['expected_random'] = expected_random
        result['random_delta'] = eff_random - expected_random

    return result


def format_result(info):
    pieces = [f'zone {info["zone"]}',
              f'raw={info["raw"]}',
              f'prob={info["prob"]}',
              f'class={info["class"]}',
              f'ndata={info["ndata"]}',
              f'rand_rows={info["rand_rows"]}',
              f'iters={info["unique_iters"]} ({info["min_iter"]}..{info["max_iter"]})',]
    eff = info.get('effective_random')
    if eff is not None and np.isfinite(eff):
        pieces.append(f'n_random≈{eff:.2f}')
    if 'random_delta' in info:
        pieces.append(f'Δn={info["random_delta"]:+.2f}')
    return ' | '.join(pieces)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--raw_dir', type=Path, default='/pscratch/sd/v/vtorresg/cosmic-web/dr1/raw')
    p.add_argument('--class_dir', type=Path, default='/pscratch/sd/v/vtorresg/cosmic-web/dr1/classification')
    p.add_argument('--prob_dir', type=Path, default='/pscratch/sd/v/vtorresg/cosmic-web/dr1/probabilities')
    p.add_argument('--zones', nargs='*', type=str, default=['NGC', 'SGC'])
    p.add_argument('--expected-random', type=int, default=100)
    return p.parse_args()


def main():
    args = parse_args()
    raw_dir = args.raw_dir
    class_dir = args.class_dir
    prob_dir = args.prob_dir

    if args.zones:
        zones = sorted({str(z).upper() for z in args.zones},
                       key=lambda value: (0, int(value)) if str(value).isdigit() else (1, str(value)))
    else:
        zones = _discover_zones(raw_dir)
        if not zones:
            raise SystemExit(f'No zone files found in {raw_dir}')

    for zone in zones:
        try:
            info = check_zone(zone, raw_dir, class_dir, prob_dir, args.expected_random)
            print(format_result(info))
        except FileNotFoundError as exc:
            print(f'zone {zone} | missing: {exc}')


if __name__ == '__main__':
    main()