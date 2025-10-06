from typing import Mapping, Optional, Union

import numpy as np
from astropy.table import Column, Table, vstack

__all__ = ["stack_zone_randoms", "assign_random_redshift_column"]


def stack_zone_randoms(zone_tables, zone_value):
    """
    Return a stacked table containing all random entries for one zone.

    Args:
        zone_tables: Mapping of file index -> astropy table for a DR2 zone.
        zone_value: Value used to populate the ``ZONE`` column when missing.
    Returns:
        A single table containing the concatenated rows, or ``None`` when no
        entries are available.
    """
    stacked = []
    for tbl in zone_tables.values():
        if tbl is None or len(tbl) == 0:
            continue
        copy = tbl.copy()
        if 'ZONE' not in copy.colnames:
            copy.add_column(Column(np.full(len(copy), int(zone_value), dtype=int), name='ZONE'))
        stacked.append(copy)
    if not stacked:
        return None
    return vstack(stacked)


def assign_random_redshift_column(sample, real_redshifts, rng):
    """
    Attach a ``Z`` column sampled from ``real_redshifts`` to ``sample``.

    Args:
        sample: Table whose rows require a synthetic ``Z`` column.
        real_redshifts: Array of redshift values sourced from the real catalogue.
        rng: Random number generator used for reproducible draws.
    Returns:
        The input table with a ``Z`` column assigned (the same instance).
    """
    if real_redshifts.size == 0:
        raise ValueError('real_redshifts cannot be empty when sampling random Z values')

    draws = rng.choice(real_redshifts, size=len(sample), replace=True)
    if 'Z' in sample.colnames:
        sample['Z'] = draws.astype(float)
    else:
        sample.add_column(Column(draws.astype(float), name='Z'))
    return sample