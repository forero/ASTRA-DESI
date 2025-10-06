import os
from typing import Dict, List, Set

import numpy as np
from argparse import Namespace
from astropy.table import Table, vstack

from desiproc.read_data import generate_randoms, process_real
from desiproc.paths import safe_tag, zone_tag

from .base import ReleaseConfig


TRACERS = ['BGS_ANY', 'ELG', 'LRG', 'QSO']
REAL_SUFFIX = {'N': '_N_clustering.dat.fits', 'S': '_S_clustering.dat.fits'}
RANDOM_SUFFIX = {'N': '_N_{i}_clustering.ran.fits', 'S': '_S_{i}_clustering.ran.fits'}
N_RANDOM_FILES = 18
REAL_COLUMNS = ['TARGETID', 'ROSETTE_NUMBER', 'RA', 'DEC', 'Z']
RANDOM_COLUMNS = REAL_COLUMNS
N_ZONES = 20
NORTH_ROSETTES = {3, 6, 7, 11, 12, 13, 14, 15, 18, 19}
TRACER_ALIAS = {'bgs': 'BGS_ANY', 'elg': 'ELG', 'lrg': 'LRG', 'qso': 'QSO'}


def build_raw_table(zone, real_tables, random_tables, output_raw, n_random, tracers,
                    north_rosettes, out_tag, release_tag):
    """
    Build and persist the EDR raw table for ``zone``.
    
    Args:
        zone: Zone number (0-19).
        real_tables: Preloaded real tables.
        random_tables: Preloaded random tables.
        output_raw: Output directory for raw tables.
        n_random: Number of randoms to generate per real.
        tracers: List of tracers to include.
        north_rosettes: Set of rosette numbers in the North.
        out_tag: Optional tag to append to output filename.
        release_tag: Optional release tag to include in metadata.
    Returns:
        The combined raw table for the specified zone.
    """
    parts: List[Table] = []
    for tr in tracers:
        rt = process_real(real_tables, tr, zone, north_rosettes)
        parts.append(rt)
        count = len(rt)
        rpt = generate_randoms(random_tables, tr, zone, north_rosettes, n_random, count)
        parts.append(rpt)

    tbl = vstack(parts)
    if 'RANDITER' in tbl.colnames:
        tbl['RANDITER'] = np.asarray(tbl['RANDITER'], dtype=np.int32)

    tag_suffix = safe_tag(out_tag)
    out_path = os.path.join(output_raw, f'zone_{zone:02d}{tag_suffix}.fits.gz')
    tmp_path = out_path + '.tmp'

    tbl_out = tbl.copy()
    if 'ZONE' in tbl_out.colnames:
        tbl_out.remove_column('ZONE')

    tbl_out.meta['ZONE'] = zone_tag(zone)
    tbl_out.meta['RELEASE'] = str(release_tag) if release_tag is not None else 'UNKNOWN'

    tbl_out.write(tmp_path, format='fits', overwrite=True)
    os.replace(tmp_path, out_path)
    return tbl


def create_config(args: Namespace) -> ReleaseConfig:
    """
    Create the EDR release configuration.
    
    Args:
        args: Parsed command-line arguments.
    Returns:
        The EDR release configuration.
    Raises:
        RuntimeError: If a specified zone is out of range.
    """
    if args.zone is not None:
        if not 0 <= int(args.zone) < N_ZONES:
            raise RuntimeError(f"Zone {args.zone} out of range (0-{N_ZONES-1})")
        zones = [int(args.zone)]
    else:
        zones = list(range(N_ZONES))

    def _build(zone, real_tables, random_tables, sel_tracers, parsed_args, release_tag):
        return build_raw_table(int(zone), real_tables, random_tables, parsed_args.raw_out,
                               parsed_args.n_random, sel_tracers, NORTH_ROSETTES,
                               out_tag=parsed_args.out_tag, release_tag=release_tag)

    return ReleaseConfig(name='EDR', release_tag='EDR', tracers=TRACERS,
                         tracer_alias=TRACER_ALIAS, real_suffix=REAL_SUFFIX,
                         random_suffix=RANDOM_SUFFIX, n_random_files=N_RANDOM_FILES,
                         real_columns=REAL_COLUMNS, random_columns=RANDOM_COLUMNS,
                         use_dr2_preload=False, preload_kwargs={}, zones=zones,
                         build_raw=_build)