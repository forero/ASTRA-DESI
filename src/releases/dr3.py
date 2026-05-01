import os

import numpy as np
from astropy.table import Column, Table, vstack

from desiproc.implement_astra import register_tracer_mapping
from desiproc.paths import safe_tag, zone_tag
from desiproc.read_data import (generate_randoms_dr2_iteration,
                                preload_dr2_tables,
                                process_real_dr2)

from .base import ReleaseConfig
from .dr2 import (DR2_RA_MAX, DR2_RA_MIN, DR2_ZONE_VALUES,
                  build_raw_dr2_zone)


TRACERS = ['QSO', 'ELG_LOPnotqso', 'LRG', 'BGS_BRIGHT']
N_RANDOM_FILES = 18
REAL_COLUMNS = ['TARGETID', 'RA', 'DEC', 'Z']
RANDOM_COLUMNS = ['TARGETID', 'RA', 'DEC', 'Z']
REAL_TEMPLATE = '{tracer}_{zone}_clustering.dat.fits'
RANDOM_TEMPLATE = '{tracer}_{zone}_{idx}_clustering.ran.fits'
TRACER_ALIAS = {'qso': 'QSO',
                'elg': 'ELG_LOPnotqso',
                'elg_lopnotqso': 'ELG_LOPnotqso',
                'lrg': 'LRG',
                'bgs': 'BGS_BRIGHT',
                'bgs_bright': 'BGS_BRIGHT'}
DEFAULT_ZONES = ['NGC', 'SGC']
TRACER_IDS = {name: idx for idx, name in enumerate(TRACERS)}
TRACER_FULL_LABELS = {}
for tracer_name, tracer_idx in TRACER_IDS.items():
    TRACER_FULL_LABELS[(tracer_idx, True)] = f'{tracer_name}'.encode('ascii')
    TRACER_FULL_LABELS[(tracer_idx, False)] = f'{tracer_name}'.encode('ascii')


def preload_dr3_tables(base_dir, tracers, n_random_files=N_RANDOM_FILES, zones_to_keep=None):
    """
    Preload DR3 zone-specific clustering catalogues.
    """
    return preload_dr2_tables(base_dir, tracers, REAL_COLUMNS, RANDOM_COLUMNS,
                              n_random_files,
                              ra_min=DR2_RA_MIN, ra_max=DR2_RA_MAX,
                              real_template=REAL_TEMPLATE,
                              random_template=RANDOM_TEMPLATE,
                              log_label='dr3',
                              zones_to_keep=zones_to_keep)


def raw_iteration_path(output_raw, zone_label, tracer, iteration):
    """
    Return the DR3 raw path for one tracer/zone/random iteration shard.
    """
    return os.path.join(output_raw,
                        f'zone_{zone_tag(zone_label)}{safe_tag(tracer)}_iter{int(iteration):03d}.fits.gz')


def build_raw_dr3_iteration(zone_label, tracer, iteration, real_tables, random_tables,
                            output_raw, zone_value=None, release_tag='DR3',
                            real_table=None, force=False):
    """
    Build one DR3 raw shard containing real data plus a single random iteration.

    The random sampling delegates to the same DR2 full-sky sampler used by the
    normal multi-iteration pipeline, restricted to ``iteration``.
    """
    label = str(zone_label).upper()
    if zone_value is None:
        zone_value = DR2_ZONE_VALUES.get(label, 3999)
    iteration = int(iteration)
    tracer_id = TRACER_IDS.get(tracer)
    out_path = raw_iteration_path(output_raw, label, tracer, iteration)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    existing_path = None
    if os.path.exists(out_path):
        existing_path = out_path
    elif out_path.endswith('.gz') and os.path.exists(out_path[:-3]):
        existing_path = out_path[:-3]
    if existing_path is not None and not force:
        cached = Table.read(existing_path, memmap=True)
        print(f'[dr3] reuse existing raw iteration {existing_path}', flush=True)
        return cached.copy()

    if real_table is None:
        real_tbl = process_real_dr2(real_tables, tracer, label, zone_value=zone_value,
                                    tracer_id=tracer_id, include_tracertype=False,
                                    downcast=True)
    else:
        real_tbl = real_table
    rand_tbl = generate_randoms_dr2_iteration(random_tables, tracer, label, iteration,
                                              real_tbl, zone_value=zone_value,
                                              tracer_id=tracer_id,
                                              include_tracertype=False,
                                              downcast=True,
                                              log_label='dr3')
    tbl = vstack([real_tbl, rand_tbl], metadata_conflicts='silent')
    if 'RANDITER' in tbl.colnames:
        tbl['RANDITER'] = np.asarray(tbl['RANDITER'], dtype=np.int16)
    if 'ZONE' in tbl.colnames:
        tbl['ZONE'] = np.asarray(tbl['ZONE'], dtype=np.int16)

    labels = np.full(len(tbl), str(tracer), dtype='U24')
    tbl.add_column(Column(labels, name='TRACERTYPE'))
    tbl.meta['ZONE'] = zone_tag(label)
    tbl.meta['RELEASE'] = str(release_tag)
    tbl.meta['ITER'] = iteration

    tmp_path = out_path + '.tmp'
    try:
        tbl.write(tmp_path, format='fits', overwrite=True)
        os.replace(tmp_path, out_path)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
    print(f'[dr3] wrote raw iteration {out_path} rows={len(tbl)}', flush=True)
    return tbl


def create_config(args):
    """
    Create the DR3 release configuration for the standard full-zone pipeline.

    DR3 production at current catalogue sizes should use the per-iteration
    runner in ``src/utils/run_dr3_iterations.py``.  This config exists so the
    regular CLI understands the DR3 catalogue layout for small tests.
    """
    if args.config:
        raise RuntimeError('--config not supported for DR3. RA/DEC cuts are fixed to match DR2.')

    if args.zones is not None:
        zones = [str(z).upper() for z in args.zones]
    else:
        zones = DEFAULT_ZONES.copy()

    register_tracer_mapping(TRACER_IDS, TRACER_FULL_LABELS)

    def _build(zone, real_tables, random_tables, sel_tracers, parsed_args, release_tag):
        label = str(zone).upper()
        zone_value = DR2_ZONE_VALUES.get(label, 3999)
        return build_raw_dr2_zone(label, sel_tracers, real_tables, random_tables,
                                  parsed_args.raw_out, parsed_args.n_random, zone_value,
                                  out_tag=parsed_args.out_tag, release_tag=release_tag,
                                  tracer_ids=TRACER_IDS,
                                  tracer_full_labels=TRACER_FULL_LABELS,
                                  log_label='dr3')

    preload_kwargs = {'ra_min': DR2_RA_MIN,
                      'ra_max': DR2_RA_MAX,
                      'real_template': REAL_TEMPLATE,
                      'random_template': RANDOM_TEMPLATE,
                      'log_label': 'dr3',
                      'zones_to_keep': zones}

    return ReleaseConfig(name='DR3', release_tag='DR3', tracers=TRACERS,
                         tracer_alias=TRACER_ALIAS, real_suffix=None,
                         random_suffix=None, n_random_files=N_RANDOM_FILES,
                         real_columns=REAL_COLUMNS, random_columns=RANDOM_COLUMNS,
                         use_dr2_preload=True, preload_kwargs=preload_kwargs,
                         zones=zones, build_raw=_build, combine_outputs=False)