import os, re

__all__ = ["zone_tag",
           "safe_tag",
           "tracer_tag",
           "zone_prefix",
           "classification_filename",
           "probability_filename",
           "pairs_filename",
           "classification_path",
           "probability_path",
           "pairs_path",
           "ensure_release_subdirs",
           "locate_classification_file",
           "locate_probability_file",
           "locate_pairs_file",
           "normalize_release_dir"]


_SAFE_PATTERN = re.compile(r"[^A-Za-z0-9_\\-\\.\\+]+")


def zone_tag(zone):
    """
    Return a normalized zone tag.

    Args:
        zone (object): Zone identifier as string or integer.
    Returns:
        str: Zero-padded zone tag.
    """
    try:
        return f"{int(zone):02d}"
    except Exception:
        return str(zone)


def safe_tag(tag):
    """
    Return a sanitized filename suffix for a tag.

    Args:
        tag (object | None): Tag to be sanitized.
    Returns:
        str: Sanitized suffix including a leading underscore when non-empty.
    """
    if tag is None:
        return ''
    safe = _SAFE_PATTERN.sub('_', str(tag))
    return f'_{safe}' if safe else ''


def tracer_tag(tag):
    """
    Return the tracer token used in classification filenames.

    Args:
        tag (object | None): Tracer tag to normalize.
    Returns:
        str: Tracer token or ``'combined'`` when the tag is empty.
    """
    if tag is None:
        return 'combined'
    safe = _SAFE_PATTERN.sub('_', str(tag)).strip('_')
    return safe if safe else 'combined'


def zone_prefix(zone, tag=None):
    """
    Return the base name used for zone-specific outputs.

    Args:
        zone (object): Zone identifier.
        tag (object | None): Optional additional tag.
    Returns:
        str: Filename prefix ``zone_{zone}{_tag}``.
    """
    return f"zone_{zone_tag(zone)}{safe_tag(tag)}"


def classification_filename(zone, tag=None):
    """
    Return the filename for classification products.

    Args:
        zone (object): Zone identifier.
        tag (object | None): Optional tracer tag.
    Returns:
        str: Classification filename.
    """
    tracer = tracer_tag(tag)
    suffix = '' if tracer == 'combined' else f'_{tracer}'
    return f"zone_{zone_tag(zone)}{suffix}_classified.fits.gz"


def probability_filename(zone, tag=None):
    """
    Return the filename for probability products.

    Args:
        zone (object): Zone identifier.
        tag (object | None): Optional tracer tag.
    Returns:
        str: Probability filename.
    """
    return f"{zone_prefix(zone, tag)}_probability.fits.gz"


def pairs_filename(zone, tag=None):
    """
    Return the filename for pairs products.

    Args:
        zone (object): Zone identifier.
        tag (object | None): Optional tracer tag.
    Returns:
        str: Pairs filename.
    """
    return f"{zone_prefix(zone, tag)}_pairs.fits.gz"


def _subdir(base_dir, name):
    """
    Return the path to a named subdirectory within ``base_dir``.

    Args:
        base_dir (str): Parent directory path.
        name (str): Subdirectory name.
    Returns:
        str: Combined path.
    """
    return os.path.join(base_dir, name)


def classification_path(base_dir, zone, tag=None):
    """
    Return the full path for a classification file.

    Args:
        base_dir (str): Release root directory.
        zone (object): Zone identifier.
        tag (object | None): Optional tracer tag.
    Returns:
        str: Classification file path.
    """
    return os.path.join(_subdir(base_dir, 'classification'), classification_filename(zone, tag))


def probability_path(base_dir, zone, tag=None):
    """
    Return the full path for a probability file.

    Args:
        base_dir (str): Release root directory.
        zone (object): Zone identifier.
        tag (object | None): Optional tracer tag.
    Returns:
        str: Probability file path.
    """
    return os.path.join(_subdir(base_dir, 'probabilities'), probability_filename(zone, tag))


def pairs_path(base_dir, zone, tag=None):
    """
    Return the full path for a pairs file.

    Args:
        base_dir (str): Release root directory.
        zone (object): Zone identifier.
        tag (object | None): Optional tracer tag.
    Returns:
        str: Pairs file path.
    """
    return os.path.join(_subdir(base_dir, 'pairs'), pairs_filename(zone, tag))


def ensure_release_subdirs(base_dir):
    """
    Create the standard output subdirectories under ``base_dir`` if needed.

    Args:
        base_dir (str): Release root directory.
    """
    for sub in ('classification', 'probabilities', 'pairs'):
        os.makedirs(_subdir(base_dir, sub), exist_ok=True)


def normalize_release_dir(path):
    """
    Return the release root, stripping legacy subdirectory suffixes.

    Args:
        path (str): User-specified release directory or legacy subdirectory.
    Returns:
        str: Normalized release root path.
    """
    norm = os.path.normpath(path)
    tail = os.path.basename(norm).lower()
    if tail in {'class', 'classification', 'probabilities', 'pairs'}:
        parent = os.path.dirname(norm)
        return parent if parent else norm
    return norm


def locate_classification_file(base_dir, zone, tag=None):
    """
    Return the path to an existing classification file.

    Args:
        base_dir (str): Release root directory.
        zone (object): Zone identifier.
        tag (object | None): Optional tracer tag.
    Returns:
        str: Path to the classification file.
    Raises:
        FileNotFoundError: If the file does not exist.
    """
    path = classification_path(base_dir, zone, tag)
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return path


def locate_probability_file(base_dir, zone, tag=None):
    """
    Return the path to an existing probability file.

    Args:
        base_dir (str): Release root directory.
        zone (object): Zone identifier.
        tag (object | None): Optional tracer tag.
    Returns:
        str: Path to the probability file.
    Raises:
        FileNotFoundError: If the file does not exist.
    """
    path = probability_path(base_dir, zone, tag)
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return path


def locate_pairs_file(base_dir, zone, tag=None):
    """
    Return the path to an existing pairs file.

    Args:
        base_dir (str): Release root directory.
        zone (object): Zone identifier.
        tag (object | None): Optional tracer tag.
    Returns:
        str: Path to the pairs file.
    Raises:
        FileNotFoundError: If the file does not exist.
    """
    path = pairs_path(base_dir, zone, tag)
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return path