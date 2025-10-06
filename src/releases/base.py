from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Sequence

from argparse import Namespace
from astropy.table import Table


@dataclass(frozen=True)
class ReleaseConfig:
    """
    Container with release-specific configuration needed by the pipeline.
    """
    name: str
    release_tag: str
    tracers: List[str]
    tracer_alias: Dict[str, str]
    real_suffix: Dict[str, str] | None
    random_suffix: Dict[str, str] | None
    n_random_files: int
    real_columns: Sequence[str]
    random_columns: Sequence[str]
    use_dr2_preload: bool
    preload_kwargs: Dict[str, Any]
    zones: List[Any]
    build_raw: Callable[[Any, Dict[str, Any], Dict[str, Any], List[str], Namespace, str], Table]