from pathlib import Path
import fitsio
import numpy as np


ZONES = ('NGC', 'SGC')
TRACER_LABELS = {'BGS': 'BGS_BRIGHT',
                 'LRG': 'LRG',
                 'ELG': 'ELG_LOPnotqso',
                 'QSO': 'QSO'}
TRACER_DISPLAY = {'BGS_BRIGHT': 'BGS',
                  'LRG': 'LRG',
                  'ELG_LOPnotqso': 'ELG',
                  'QSO': 'QSO'}
TRACER_CODES = {'BGS_BRIGHT': 1,
                'LRG': 2,
                'ELG_LOPnotqso': 3,
                'QSO': 4}
RAW_COLUMNS = ('TARGETID', 'RA', 'DEC', 'Z', 'XCART', 'YCART', 'ZCART')


def _decode_text(value):
    if isinstance(value, (bytes, np.bytes_)):
        return value.decode('utf-8').strip()
    return str(value).strip()


def normalize_zone(value):
    zone = str(value).strip().upper()
    if zone not in ZONES:
        raise ValueError(f'zone must be one of {ZONES}; got {value!r}.')
    return zone


def normalize_tracer(value):
    label = str(value).strip()
    upper = label.upper()
    aliases = {'BGS': 'BGS_BRIGHT',
               'BGS_BRIGHT': 'BGS_BRIGHT',
               'LRG': 'LRG',
               'ELG': 'ELG_LOPnotqso',
               'ELG_LOPNOTQSO': 'ELG_LOPnotqso',
               'QSO': 'QSO',
               'QSOS': 'QSO'}
    if upper not in aliases:
        raise ValueError(
            f'Unknown tracer {value!r}; expected BGS, LRG, ELG, or QSO.')
    return aliases[upper]


def raw_zone_path(raw_dir, zone):
    return Path(raw_dir) / f'zone_{normalize_zone(zone)}.fits.gz'


def _row_key(hdu, row):
    record = hdu.read(columns=['TRACERTYPE', 'RANDITER'], rows=[int(row)])[0]
    return _decode_text(record['TRACERTYPE']), int(record['RANDITER'])


def _lower_bound(hdu, target):
    lower = 0
    upper = int(hdu.get_nrows())
    while lower < upper:
        middle = (lower + upper) // 2
        if _row_key(hdu, middle) < target:
            lower = middle + 1
        else:
            upper = middle
    return lower


def read_raw_realization(path, tracer, iteration):
    path = Path(path)
    tracer = normalize_tracer(tracer)
    if not path.is_file():
        raise FileNotFoundError(f'Raw FITS does not exist: {path}.')
    if isinstance(iteration, (bool, np.bool_)) or not isinstance(
            iteration, (int, np.integer)):
        raise TypeError('iteration must be a non-negative integer.')
    iteration = int(iteration)
    if iteration < 0:
        raise ValueError('iteration must be a non-negative integer.')

    with fitsio.FITS(str(path)) as raw:
        hdu = raw[1]
        data_start = _lower_bound(hdu, (tracer, -1))
        data_stop = _lower_bound(hdu, (tracer, 0))
        random_start = _lower_bound(hdu, (tracer, iteration))
        random_stop = _lower_bound(hdu, (tracer, iteration + 1))
        if data_start == data_stop:
            raise ValueError(f'No object rows found for tracer={tracer}.')
        if random_start == random_stop:
            raise ValueError(f'No random rows found for tracer={tracer}, RANDITER={iteration}.')

        objects = hdu.read(columns=list(RAW_COLUMNS), rows=np.arange(data_start, data_stop, dtype=np.int64))
        randoms = hdu.read(columns=list(RAW_COLUMNS), rows=np.arange(random_start, random_stop, dtype=np.int64))
        table_header = hdu.read_header()
        primary_header = raw[0].read_header()

    zone = _decode_text(
        table_header.get('ZONE', primary_header.get('ZONE', 'UNKNOWN')))
    release = _decode_text(
        table_header.get('RELEASE', primary_header.get('RELEASE', 'UNKNOWN')))
    return objects, randoms, {'input': str(path),
                              'tracer': tracer,
                              'zone': zone,
                              'release': release,
                              'iteration': iteration,
                              'data_start': int(data_start),
                              'data_stop': int(data_stop),
                              'random_start': int(random_start),
                              'random_stop': int(random_stop)}


def cartesian_positions(records):
    positions = np.column_stack((np.asarray(records['XCART'], dtype=np.float64),
                                 np.asarray(records['YCART'], dtype=np.float64),
                                 np.asarray(records['ZCART'], dtype=np.float64)))
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError('Raw Cartesian columns must form an (N, 3) array.')
    if not np.all(np.isfinite(positions)):
        raise ValueError('Raw Cartesian columns contain non-finite values.')
    return positions


__all__ = ['RAW_COLUMNS',
           'TRACER_CODES',
           'TRACER_DISPLAY',
           'TRACER_LABELS',
           'ZONES',
           'cartesian_positions',
           'normalize_tracer',
           'normalize_zone',
           'raw_zone_path',
           'read_raw_realization']