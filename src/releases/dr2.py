import os
from typing import Dict, List

import numpy as np
from argparse import Namespace
from astropy.table import Table, vstack

from desiproc.read_data import generate_randoms_dr2, process_real_dr2
from desiproc.paths import safe_tag, zone_tag

from .base import ReleaseConfig


TRACERS: List[str] = ['BGS_ANY', 'ELG', 'LRG', 'QSO']
N_RANDOM_FILES = 18
REAL_COLUMNS = ['TARGETID', 'RA', 'DEC', 'Z']
RANDOM_COLUMNS = ['TARGETID', 'RA', 'DEC']
DR2_RA_MIN = 90.0
DR2_RA_MAX = 300.0
DR2_ZONE_VALUES = {'NGC': 2001, 'SGC': 2002}
DR2_REDSHIFT_OVERRIDES = {'BGS_ANY': 'Z_not4clus', 'ELG': 'Z_not4clus', 'LRG': 'Z_not4clus', 'QSO': 'Z_RR'}
TRACER_ALIAS = {'bgs': 'BGS_ANY', 'elg': 'ELG', 'lrg': 'LRG', 'qso': 'QSO'}
DEFAULT_ZONES = ['NGC', 'SGC']


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
    parts: List[Table] = []
    skipped: List[str] = []
    for tr in tracers:
        try:
            rt = process_real_dr2(real_tables, tr, zone_label, zone_value=zone_value)
        except ValueError as exc:
            print(f'[warn] {tr} empty in DR2 zone {zone_label}: {exc}')
            skipped.append(tr)
            continue
        parts.append(rt)
        rpt = generate_randoms_dr2(random_tables, tr, zone_label, n_random, rt, zone_value=zone_value)
        parts.append(rpt)

    if not parts:
        raise ValueError(f'No data in DR2 zone {zone_label} (tracers tried: {tracers})')

    tbl = vstack(parts)
    if 'RANDITER' in tbl.colnames:
        tbl['RANDITER'] = np.asarray(tbl['RANDITER'], dtype=np.int32)

    tag_suffix = safe_tag(out_tag)
    out_path = os.path.join(output_raw, f'zone_{zone_label}{tag_suffix}.fits.gz')
    tmp_path = out_path + '.tmp'

    tbl_out = tbl.copy()
    if 'ZONE' in tbl_out.colnames:
        tbl_out.remove_column('ZONE')

    tbl_out.meta['ZONE'] = zone_tag(zone_label)
    tbl_out.meta['RELEASE'] = str(release_tag) if release_tag is not None else 'UNKNOWN'

    tbl_out.write(tmp_path, format='fits', overwrite=True)
    os.replace(tmp_path, out_path)

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

    def _build(zone, real_tables, random_tables, sel_tracers, parsed_args, release_tag):
        label = str(zone).upper()
        zone_value = DR2_ZONE_VALUES.get(label, 2999)
        return build_raw_dr2_zone(label, sel_tracers, real_tables, random_tables,
                                  parsed_args.raw_out, parsed_args.n_random, zone_value,
                                  out_tag=parsed_args.out_tag, release_tag=release_tag)

    preload_kwargs = {'ra_min': DR2_RA_MIN,
                      'ra_max': DR2_RA_MAX,
                      'redshift_overrides': DR2_REDSHIFT_OVERRIDES}

    return ReleaseConfig(name='DR2', release_tag='DR2', tracers=TRACERS,
                         tracer_alias=TRACER_ALIAS, real_suffix=None,
                         random_suffix=None, n_random_files=N_RANDOM_FILES,
                         real_columns=REAL_COLUMNS, random_columns=RANDOM_COLUMNS,
                         use_dr2_preload=True, preload_kwargs=preload_kwargs,
                         zones=zones, build_raw=_build)