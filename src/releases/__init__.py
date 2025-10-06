from typing import Callable, Dict

from argparse import Namespace

from .base import ReleaseConfig
from . import dr1, dr2, edr

RELEASE_FACTORIES: Dict[str, Callable[[Namespace], ReleaseConfig]] = {
    'EDR': edr.create_config,
    'DR1': dr1.create_config,
    'DR2': dr2.create_config,
}

__all__ = ['ReleaseConfig', 'RELEASE_FACTORIES']
