import gc
import os
from typing import Dict, List

import numpy as np
from argparse import Namespace
from astropy.table import Column, Table

from desiproc.implement_astra import TempTableStore, register_tracer_mapping
from desiproc.read_data import generate_randoms_dr2, process_real_dr2
from desiproc.paths import safe_tag, zone_tag

from .base import ReleaseConfig


TRACERS: List[str] = ['BGS_ANY', 'ELG', 'LRG', 'QSO']
N_RANDOM_FILES = 18
REAL_COLUMNS = ['TARGETID', 'RA', 'DEC', 'Z']
RANDOM_COLUMNS = ['TARGETID', 'RA', 'DEC', 'Z']
DR2_RA_MIN = 90.0
DR2_RA_MAX = 300.0
DR2_ZONE_VALUES = {'NGC': 2001, 'SGC': 2002}
TRACER_ALIAS = {'bgs': 'BGS_ANY', 'elg': 'ELG', 'lrg': 'LRG', 'qso': 'QSO'}
DEFAULT_ZONES = ['NGC', 'SGC']
TRACER_IDS = {name: idx for idx, name in enumerate(TRACERS)}
TRACER_FULL_LABELS = {}
for tracer_name, tracer_idx in TRACER_IDS.items():
    TRACER_FULL_LABELS[(tracer_idx, True)] = f'{tracer_name}_DATA'.encode('ascii')
    TRACER_FULL_LABELS[(tracer_idx, False)] = f'{tracer_name}_RAND'.encode('ascii')


def _progress(message: str) -> None:
    """
    Print a progress message when verbose progress is enabled.
    """
    if os.environ.get('ASTRA_PROGRESS'):
        print(f'[progress] {message}', flush=True)


def build_raw_dr2_zone(zone_label, tracers, real_tables, random_tables, output_raw,
                       n_random, zone_value, out_tag, release_tag):
    """
    Build and persist the DR2 raw table for ``zone_label``.
    
    Args:
        zone_label: Label for the zone being processed.
        tracers: List of tracers to process.
        real_tables: Dictionary with real tables per tracer.
        random_tables: Dictionary with random tables per tracer.
        output_raw: Path to the output raw directory.
        n_random: Number of randoms per data object.
        zone_value: Integer value to assign to the ZONE column.
        out_tag: Optional tag to append to the output file name.
        release_tag: Release tag string or None.
    Returns:
        The combined table written to disk.
    """
    tag_suffix = safe_tag(out_tag)
    out_path = os.path.join(output_raw, f'zone_{zone_label}{tag_suffix}.fits.gz')
    existing_path = None
    if os.path.exists(out_path):
        existing_path = out_path
    else:
        if out_path.endswith('.gz'):
            alt_path = out_path[:-3]
            if os.path.exists(alt_path):
                existing_path = alt_path
    if existing_path is not None:
        try:
            cached = Table.read(existing_path, memmap=True)
            print(f'[dr2] reuse existing raw {existing_path}', flush=True)
        except Exception as exc:
            print(f'[dr2] warning: cannot read existing raw {existing_path} ({exc}); rebuilding', flush=True)
        else:
            _progress(f'zone {zone_label}: reusing cached raw table {existing_path}')
            tbl_cached = cached.copy()
            if 'RANDITER' in tbl_cached.colnames:
                tbl_cached['RANDITER'] = np.asarray(tbl_cached['RANDITER'], dtype=np.int32)
            if 'ZONE' not in tbl_cached.colnames:
                tbl_cached.add_column(Column(np.full(len(tbl_cached), int(zone_value), dtype=np.int32), name='ZONE'))
            return tbl_cached

    skipped: List[str] = []

    raw_store = None
    store_cols: List[str] = []
    total_rows = 0
    tmp_base = (os.environ.get('ASTRA_RAW_SPILL_DIR')
                or os.environ.get('ASTRA_TMPDIR')
                or os.environ.get('PSCRATCH')
                or os.environ.get('TMPDIR'))

    def _append_chunk(chunk: Table):
        """
        Append a chunk of data to the raw store.
        """
        nonlocal raw_store, store_cols, total_rows
        if chunk is None or len(chunk) == 0:
            return
        if not store_cols:
            store_cols = list(chunk.colnames)
        elif list(chunk.colnames) != store_cols:
            missing = sorted(set(store_cols) - set(chunk.colnames))
            if missing:
                raise RuntimeError(f'Chunk missing expected columns {missing}')
            chunk = chunk[store_cols]
        arr = np.asarray(chunk.as_array())
        if raw_store is None:
            raw_store = TempTableStore(arr.dtype, prefix=f'dr2_raw_{zone_label}', base_dir=tmp_base)
        raw_store.append(arr)
        total_rows += len(chunk)
        _progress(f'zone {zone_label}: appended {len(chunk)} rows (cumulative {total_rows})')

    for tr in tracers:
        tracer_id = TRACER_IDS.get(tr)
        try:
            rt = process_real_dr2(real_tables, tr, zone_label, zone_value=zone_value,
                                  tracer_id=tracer_id, include_tracertype=False, downcast=True)
        except ValueError as exc:
            print(f'[warn] {tr} empty in DR2 zone {zone_label}: {exc}')
            skipped.append(tr)
            continue
        _progress(f'zone {zone_label}: processing tracer {tr} real sample ({len(rt)} rows)')
        _append_chunk(rt)
        rpt = generate_randoms_dr2(random_tables, tr, zone_label, n_random, rt,
                                   zone_value=zone_value, tracer_id=tracer_id,
                                   include_tracertype=False, downcast=True)
        _progress(f'zone {zone_label}: tracer {tr} random catalogue ({len(rpt)} rows) ready')
        _append_chunk(rpt)
        del rpt
        del rt
        gc.collect()

    if raw_store is None:
        raise ValueError(f'No data in DR2 zone {zone_label} (tracers tried: {tracers})')

    combined_memmap = raw_store.as_array()
    combined = np.array(combined_memmap, copy=True)
    del combined_memmap
    tbl = Table(combined, copy=False)
    _progress(f'zone {zone_label}: combined raw table assembled ({len(tbl)} rows)')
    if 'RANDITER' in tbl.colnames:
        tbl['RANDITER'] = np.asarray(tbl['RANDITER'], dtype=np.int16)
    if 'ZONE' in tbl.colnames and tbl['ZONE'].dtype != np.int16:
        tbl['ZONE'] = np.asarray(tbl['ZONE'], dtype=np.int16)

    tmp_path = out_path + '.tmp'
    tbl.meta['ZONE'] = zone_tag(zone_label)
    tbl.meta['RELEASE'] = str(release_tag) if release_tag is not None else 'UNKNOWN'

    removed_zone = None
    if 'ZONE' in tbl.colnames:
        removed_zone = tbl['ZONE'].copy()
        tbl.remove_column('ZONE')

    removed_trtype = None
    if 'TRACERTYPE' not in tbl.colnames:
        tracer_ids = np.asarray(tbl['TRACER_ID'], dtype=np.uint8) if 'TRACER_ID' in tbl.colnames else np.full(len(tbl), 255, dtype=np.uint8)
        randiters = np.asarray(tbl['RANDITER'], dtype=np.int32) if 'RANDITER' in tbl.colnames else np.full(len(tbl), -1, dtype=np.int32)
        labels = np.full(len(tbl), b'UNKNOWN', dtype='S24')
        for tid in np.unique(tracer_ids):
            mask = tracer_ids == tid
            if not mask.any():
                continue
            data_mask = mask & (randiters == -1)
            rand_mask = mask & (randiters != -1)
            if data_mask.any():
                labels[data_mask] = TRACER_FULL_LABELS.get((int(tid), True), b'UNKNOWN')
            if rand_mask.any():
                labels[rand_mask] = TRACER_FULL_LABELS.get((int(tid), False), b'UNKNOWN')
        removed_trtype = Column(labels.astype('U24'), name='TRACERTYPE')
        tbl.add_column(removed_trtype)
        del labels

    try:
        _progress(f'zone {zone_label}: writing raw FITS to {out_path}')
        tbl.write(tmp_path, format='fits', overwrite=True)
    finally:
        if removed_zone is not None and 'ZONE' not in tbl.colnames:
            tbl.add_column(removed_zone, name='ZONE')
        if removed_trtype is not None:
            tbl.remove_column('TRACERTYPE')
            removed_trtype = None
        if raw_store is not None:
            raw_store.cleanup()

    os.replace(tmp_path, out_path)
    _progress(f'zone {zone_label}: completed raw FITS {out_path}')

    if skipped:
        print(f'[info] In DR2 {zone_label} skipped tracers (empty): {", ".join(skipped)}')
    return tbl


def create_config(args: Namespace) -> ReleaseConfig:
    """
    Create the release configuration from command line arguments.
    
    Args:        
        args: Parsed command line arguments.
    Returns:        
        The release configuration object.
    Raises:        
        RuntimeError: If --config is provided or if zones are missing cuts.
    """
    if args.config:
        raise RuntimeError('--config not supported for DR2. RA/DEC cuts are fixed.')

    if args.zones is not None:
        zones = [str(z).upper() for z in args.zones]
    else:
        zones = DEFAULT_ZONES.copy()

    register_tracer_mapping(TRACER_IDS, TRACER_FULL_LABELS)

    def _build(zone, real_tables, random_tables, sel_tracers, parsed_args, release_tag):
        label = str(zone).upper()
        zone_value = DR2_ZONE_VALUES.get(label, 2999)
        return build_raw_dr2_zone(label, sel_tracers, real_tables, random_tables,
                                  parsed_args.raw_out, parsed_args.n_random, zone_value,
                                  out_tag=parsed_args.out_tag, release_tag=release_tag)

    preload_kwargs = {'ra_min': DR2_RA_MIN,
                      'ra_max': DR2_RA_MAX}

    return ReleaseConfig(name='DR2', release_tag='DR2', tracers=TRACERS,
                         tracer_alias=TRACER_ALIAS, real_suffix=None,
                         random_suffix=None, n_random_files=N_RANDOM_FILES,
                         real_columns=REAL_COLUMNS, random_columns=RANDOM_COLUMNS,
                         use_dr2_preload=True, preload_kwargs=preload_kwargs,
                         zones=zones, build_raw=_build, combine_outputs=False)