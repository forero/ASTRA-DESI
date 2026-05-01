from typing import Callable, Dict

from argparse import Namespace
from importlib import import_module

from .base import ReleaseConfig


def _lazy_factory(module_name: str) -> Callable[[Namespace], ReleaseConfig]:
    def _create(args: Namespace) -> ReleaseConfig:
        module = import_module(f'.{module_name}', __name__)
        return module.create_config(args)
    return _create

RELEASE_FACTORIES: Dict[str, Callable[[Namespace], ReleaseConfig]] = {'EDR': _lazy_factory('edr'),
                                                                      'DR1': _lazy_factory('dr1'),
                                                                      'DR2': _lazy_factory('dr2'),
                                                                      'DR3': _lazy_factory('dr3'),}

__all__ = ['ReleaseConfig', 'RELEASE_FACTORIES']
