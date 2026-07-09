import glob, os, re
from pathlib import Path
import numpy as np
from astropy.io import fits


def safe_upper(value):
    return str(value).strip().upper()


def normalize_zone_tag(zone):
    text = str(zone).strip()
    try:
        return f'{int(text):02d}'
    except Exception:
        return text.upper()


def zone_name_variants(zone):
    raw = str(zone).strip()
    norm = normalize_zone_tag(zone)
    vals = [norm, safe_upper(norm), safe_upper(raw), raw]
    out = []
    seen = set()
    for val in vals:
        if not val:
            continue
        if val in seen:
            continue
        seen.add(val)
        out.append(val)
    return out


def tracer_aliases(tracer):
    t = safe_upper(tracer)
    mapping = {'BGS': ('BGS', 'BGS_ANY', 'BGS_BRIGHT'),
               'BGS_ANY': ('BGS_ANY', 'BGS', 'BGS_BRIGHT'),
               'BGS_BRIGHT': ('BGS_BRIGHT', 'BGS', 'BGS_ANY'),
               'BGS_BRIGHT-21.35': ('BGS_BRIGHT-21.35', 'BGS_BRIGHT_21.35',
                                     'BGS_BRIGHT_M21.35', 'BGS-21.35',
                                     'BGS_21.35', 'BGS_M21.35'),
               'BGS_BRIGHT_21.35': ('BGS_BRIGHT_21.35', 'BGS_BRIGHT-21.35',
                                     'BGS_BRIGHT_M21.35', 'BGS_21.35',
                                     'BGS-21.35', 'BGS_M21.35'),
               'BGS_BRIGHT_M21.35': ('BGS_BRIGHT_M21.35', 'BGS_BRIGHT-21.35',
                                     'BGS_BRIGHT_21.35', 'BGS_M21.35',
                                     'BGS-21.35', 'BGS_21.35'),
               'ELG': ('ELG', 'ELG_LOPNOTQSO', 'ELG_LOPnotqso'),
               'ELG_LOPNOTQSO': ('ELG_LOPNOTQSO', 'ELG', 'ELG_LOPnotqso'),
               'LRG': ('LRG',),
               'QSO': ('QSO',)}
    return mapping.get(t, (t,))


def tracer_name_variants(tracer):
    vals = []
    for alias in tracer_aliases(tracer):
        text = str(alias).strip()
        vals.extend([text, text.upper(), text.lower()])

    out = []
    seen = set()
    for val in vals:
        if not val:
            continue
        if val in seen:
            continue
        seen.add(val)
        out.append(val)
    return out


def _glob_unique(patterns):
    matches = []
    for pat in patterns:
        matches.extend(glob.glob(pat, recursive=True))
    return sorted(set(matches))


def parse_iter(path):
    base = os.path.basename(path)
    m = re.search(r'iter(\d+)', base, flags=re.IGNORECASE)
    return int(m.group(1)) if m else None


def _sort_iter_paths(paths):
    return sorted(paths, key=lambda p: (parse_iter(p) is None,
                                        parse_iter(p) if parse_iter(p) is not None else 10**9,
                                        p))


def discover_classification_realizations(base, tracer, zone):
    root = Path(base) / 'classification'
    zone_tokens = zone_name_variants(zone)
    tracer_tokens = tracer_name_variants(tracer)

    tracer_iter_patterns = []
    tracer_class_patterns = []
    combined_iter_patterns = []
    combined_class_patterns = []

    for ztok in zone_tokens:
        for ttok in tracer_tokens:
            tracer_iter_patterns.extend([str(root / '**' / f'zone_{ztok}_{ttok}_iter*.fits'),
                                         str(root / '**' / f'zone_{ztok}_{ttok}_iter*.fits.gz')])
            tracer_class_patterns.extend([str(root / '**' / f'zone_{ztok}_{ttok}_classified.fits'),
                                          str(root / '**' / f'zone_{ztok}_{ttok}_classified.fits.gz')])

        combined_iter_patterns.extend([str(root / '**' / f'zone_{ztok}_iter*.fits'),
                                       str(root / '**' / f'zone_{ztok}_iter*.fits.gz')])
        combined_class_patterns.extend([str(root / '**' / f'zone_{ztok}_classified.fits'),
                                        str(root / '**' / f'zone_{ztok}_classified.fits.gz')])

    tracer_iter = _sort_iter_paths(_glob_unique(tracer_iter_patterns))
    if tracer_iter:
        return [(parse_iter(path), path) for path in tracer_iter]

    combined_iter = _sort_iter_paths(_glob_unique(combined_iter_patterns))
    if combined_iter:
        return [(parse_iter(path), path) for path in combined_iter]

    tracer_class = _glob_unique(tracer_class_patterns)
    if tracer_class:
        return [(parse_iter(path), path) for path in tracer_class]

    combined_class = _glob_unique(combined_class_patterns)
    return [(parse_iter(path), path) for path in combined_class]


def discover_raw_catalog(base, tracer, zone):
    root = Path(base) / 'raw'
    zone_tokens = zone_name_variants(zone)
    tracer_tokens = tracer_name_variants(tracer)

    tracer_patterns = []
    combined_patterns = []
    for ztok in zone_tokens:
        for ttok in tracer_tokens:
            tracer_patterns.extend([str(root / f'zone_{ztok}_{ttok}.fits'),
                                    str(root / f'zone_{ztok}_{ttok}.fits.gz')])

        combined_patterns.extend([str(root / f'zone_{ztok}.fits'),
                                  str(root / f'zone_{ztok}.fits.gz')])

    tracer_matches = _glob_unique(tracer_patterns)
    if tracer_matches:
        return tracer_matches[0]

    combined_matches = _glob_unique(combined_patterns)
    return combined_matches[0] if combined_matches else None


def discover_available_zones(base):
    root = Path(base) / 'classification'
    patterns = [str(root / '**' / 'zone_*_classified.fits'),
                str(root / '**' / 'zone_*_classified.fits.gz'),
                str(root / '**' / 'zone_*_iter*.fits'),
                str(root / '**' / 'zone_*_iter*.fits.gz')]

    zones = set()
    for path in _glob_unique(patterns):
        base_name = os.path.basename(path)
        if base_name.endswith('.fits.gz'):
            stem = base_name[:-8]
        elif base_name.endswith('.fits'):
            stem = base_name[:-5]
        else:
            stem = base_name
        m = re.match(r'zone_([^_]+)_', stem, flags=re.IGNORECASE)
        if not m:
            continue
        zones.add(normalize_zone_tag(m.group(1)))

    def _zone_key(token):
        if token.isdigit():
            return (0, int(token))
        return (1, token)

    return sorted(zones, key=_zone_key)


def _to_text(value):
    if isinstance(value, (bytes, bytearray)):
        return value.decode('ascii', errors='ignore').strip()
    return str(value).strip()


def normalize_tracer_label(value):
    text = safe_upper(_to_text(value))
    head, sep, tail = text.rpartition('_')
    if sep and tail in {'DATA', 'RAND'}:
        return head
    return text


def tracer_mask(values, tracer):
    aliases = {safe_upper(alias) for alias in tracer_aliases(tracer)}
    arr = np.asarray(values)
    if arr.ndim == 0:
        arr = arr.reshape(1)
    return np.asarray([normalize_tracer_label(v) in aliases for v in arr], dtype=bool)


def get_columns(path):
    with fits.open(path, memmap=True) as hdul:
        return list(hdul[1].columns.names)


def find_col(columns, candidates):
    for c in candidates:
        if c in columns:
            return c
    return None


def iter_fits_chunks(path, columns, chunk_rows=500_000):
    with fits.open(path, memmap=True) as hdul:
        hdu = hdul[1]
        data = hdu.data
        if data is None:
            return

        nrows = int(hdu.header.get('NAXIS2', 0))
        if nrows == 0:
            return

        use_cols = [c for c in columns if c in hdu.columns.names]
        for start in range(0, nrows, chunk_rows):
            stop = min(start + chunk_rows, nrows)
            block = data[start:stop]
            yield {col: np.asarray(block[col]) for col in use_cols}