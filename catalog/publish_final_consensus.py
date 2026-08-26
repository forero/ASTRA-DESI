import json
import os
from pathlib import Path
import shutil
import tempfile

from astropy.io import fits
import numpy as np

from group_finder.consensus import CONSENSUS_DTYPE
from group_finder.read_data import normalize_zone

try:
    from .add_sky_coordinates_to_consensus import (SKY_COLUMNS, augment_fits_catalog)
except ImportError:
    from add_sky_coordinates_to_consensus import (SKY_COLUMNS, augment_fits_catalog)

DEFAULT_FINAL_CATALOG_ROOT = Path('/pscratch/sd/v/vtorresg/void_catalog_dr2_new')
FINAL_DATASET_NAMES = {'DR2_Om_1_Om0p301_h0p6736': 'low_omega',
                       'DR2_Om_2_Om0p315_h0p6736': 'default',
                       'DR2_Om_3_Om0p329_h0p6736': 'high_omega',
                       'complete': 'complete_targets',
                       'altmtl': 'fiber_assignment'}


def normalize_final_tracer(value):
    tracer = str(value).strip().upper()
    aliases = {'BGS_ANY': 'BGS',
               'BGS_BRIGHT': 'BGS',
               'ELGNOTQSO': 'ELG',
               'ELG_LOPNOTQSO': 'ELG'}
    tracer = aliases.get(tracer, tracer)
    if tracer not in {'BGS', 'LRG', 'ELG', 'QSO'}:
        raise ValueError(f'Unsupported final-catalog tracer: {value!r}.')
    return tracer


def final_product_paths(output_root, dataset, tracer, zone):
    """Return the final FITS and JSON paths with the established names."""
    dataset = str(dataset).strip()
    if not dataset or Path(dataset).name != dataset:
        raise ValueError('dataset must be one directory name.')
    tracer = normalize_final_tracer(tracer)
    zone = normalize_zone(zone)
    stem = f'voids_{tracer}_{zone}'
    root = Path(output_root).expanduser() / dataset
    return {'fits': root / f'{stem}.fits', 'summary': root / 'logs' / f'{stem}.json'}


def _fits_has_final_schema(path, omega_m):
    path = Path(path)
    if not path.is_file() or path.stat().st_size <= 0:
        return False
    try:
        with fits.open(path, memmap=True) as catalog:
            names = set(catalog[1].columns.names or ())
            stored_omega_m = float(catalog[1].header['OMEGA_M'])
    except (OSError, IndexError, KeyError, TypeError, ValueError):
        return False
    required = set(CONSENSUS_DTYPE.names) | set(SKY_COLUMNS)
    return (required.issubset(names)
            and np.isclose(stored_omega_m, float(omega_m), rtol=0.0, atol=1.0e-12))


def _atomic_copy(source, destination):
    source = Path(source)
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f'.{destination.name}.',
                                                  suffix='.tmp',
                                                  dir=destination.parent)
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        shutil.copy2(source, temporary)
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def _inspect_source_catalog(path, omega_m):
    with fits.open(path, memmap=True) as catalog:
        names = set(catalog[1].columns.names or ())
        header = catalog[1].header
    missing = set(CONSENSUS_DTYPE.names) - names
    if missing:
        raise ValueError(f'{path} is missing current consensus columns: ' +
                         ', '.join(sorted(missing)))
    present = set(SKY_COLUMNS) & names
    if present and present != set(SKY_COLUMNS):
        raise ValueError(f'{path} has only a subset of the sky columns: ' +
                         ', '.join(sorted(present)))
    has_sky = present == set(SKY_COLUMNS)
    if has_sky:
        try:
            stored_omega_m = float(header['OMEGA_M'])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f'{path} has sky columns but no valid OMEGA_M '
                             'header.') from exc
        if not np.isclose(stored_omega_m, float(omega_m), rtol=0.0, atol=1.0e-12):
            raise ValueError(f'{path} uses OMEGA_M={stored_omega_m:g}, expected '
                             f'{float(omega_m):g}.')
    return has_sky


def publish_consensus_products(consensus_paths,
                               output_root,
                               dataset,
                               tracer,
                               zone,
                               omega_m,
                               resume=False,
                               overwrite=False):
    """Publish one consensus FITS plus its JSON in the compact final tree."""
    sources = {name: Path(path) for name, path in consensus_paths.items()}
    missing_keys = {'fits', 'summary'} - set(sources)
    if missing_keys:
        raise ValueError('Missing consensus source paths: ' +
                         ', '.join(sorted(missing_keys)))
    for name in ('fits', 'summary'):
        if not sources[name].is_file():
            raise FileNotFoundError(sources[name])
    try:
        json.loads(sources['summary'].read_text(encoding='utf-8'))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(f'Invalid consensus JSON {sources["summary"]}: {exc}') from exc

    source_has_sky = _inspect_source_catalog(sources['fits'], omega_m)
    destinations = final_product_paths(output_root, dataset, tracer, zone)
    summary_matches = (destinations['summary'].is_file()
                       and destinations['summary'].read_bytes()
                       == sources['summary'].read_bytes())
    complete = (_fits_has_final_schema(destinations['fits'], omega_m)
                and summary_matches)
    if complete and resume and not overwrite:
        return destinations

    existing = [path for path in destinations.values() if path.exists()]
    if existing and not (resume or overwrite):
        raise FileExistsError('Final consensus products already exist: ' +
                              ', '.join(str(path) for path in existing) +
                              '. Use --resume or --overwrite.')

    destinations['fits'].parent.mkdir(parents=True, exist_ok=True)
    destinations['summary'].parent.mkdir(parents=True, exist_ok=True)
    if source_has_sky:
        _atomic_copy(sources['fits'], destinations['fits'])
    else:
        augment_fits_catalog(sources['fits'],
                             destinations['fits'],
                             omega_m=omega_m,
                             overwrite=bool(resume or overwrite))
    _atomic_copy(sources['summary'], destinations['summary'])

    if not _fits_has_final_schema(destinations['fits'], omega_m):
        raise RuntimeError(f'Published FITS failed schema validation: '
                           f'{destinations["fits"]}')
    return destinations


__all__ = ['DEFAULT_FINAL_CATALOG_ROOT',
           'FINAL_DATASET_NAMES',
           'final_product_paths',
           'normalize_final_tracer',
           'publish_consensus_products']
