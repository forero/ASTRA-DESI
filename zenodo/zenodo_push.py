import os, json, tarfile
import argparse, time
from pathlib import Path
from typing import List, Dict, Optional

from zenodo_upl import (ensure_pscratch_copy, push_to_zenodo, slugify)

DEFAULT_RELEASE_SUBDIRS = ('raw', 'classification', 'probabilities', 'pairs', 'groups')


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--paths', nargs='+', help='Absolute path for files or directories to upload')
    p.add_argument('--release-root', default=None,
                   help='Base directory containing the standard EDR layout (raw/, classification/, probabilities/, pairs/, groups/)')
    p.add_argument('--include-figs', action='store_true',
                   help='Include the figs/ subdirectory when --release-root is provided and the folder exists')
    p.add_argument('--pscratch-dir', required=True, help='Nersc base at /pscratch to create temp folder')
    p.add_argument('--keep-tree', action='store_true', help='Preserve directory structure (e.g., raw/, classification/, probabilities/).')
    p.add_argument('--title', required=True, help='Zenodo record title')
    p.add_argument('--description', default=None, help='Record description (plain text or HTML)')
    p.add_argument('--description-file', default=None, help='Path to a file containing the record description (overrides --description)')
    p.add_argument('--creators-json', required=True, help='Json of members info or path')

    p.add_argument('--keywords', nargs='*', default=None)
    p.add_argument('--communities', nargs='*', default=None)
    p.add_argument('--access-right', default='open', choices=['open','embargoed','restricted','closed'])
    p.add_argument('--license', default='cc-by-4.0')
    p.add_argument('--version', default=None)
    p.add_argument('--related-identifiers-json', default=None, help='Json for other identifiers')

    p.add_argument('--publish', action='store_true', help='Publish record after uploading')
    p.add_argument('--sandbox', action='store_true', help='Use https://sandbox.zenodo.org')
    p.add_argument('--dry-run', action='store_true', help='Only create staging and records, DONT upload')

    p.add_argument('--token-env', default='ZENODO_TOKEN', help='Env variable for token')
    p.add_argument('--token-file', default=None, help='Plain text file with token')

    args = p.parse_args()
    if not args.paths and not args.release_root:
        p.error('specify --paths and/or --release-root to select content to upload')
    if not args.description and not args.description_file:
        p.error('provide --description and/or --description-file for the Zenodo record')
    return args


def _make_folder_tarballs(staging_dir: str) -> List[str]:
    """
    Creates tar.gz files for each subdirectory in the staging directory.

    Args:
        staging_dir (str): The path to the staging directory.
    Returns:
        List[str]: A list of paths to the created .tar.gz files.
    """
    s = Path(staging_dir)
    tar_paths: List[str] = []
    subdirs = [p for p in s.iterdir() if p.is_dir()]
    if not subdirs:
        tar_path = s / "dataset.tar.gz"
        with tarfile.open(tar_path, mode="w:gz") as tf:
            for p in s.rglob("*"):
                if p.is_file() and not p.name.endswith(".tar.gz"):
                    tf.add(p, arcname=p.relative_to(s))
        tar_paths.append(str(tar_path))
        return tar_paths

    for d in subdirs:
        tar_path = s / f"{d.name}.tar.gz"
        with tarfile.open(tar_path, mode="w:gz") as tf:
            tf.add(d, arcname=d.name)
        tar_paths.append(str(tar_path))

    return tar_paths


def _load_json_or_string(value: Optional[str]):
    """
    Load JSON from a string or a file path.
    If the value is None or empty, return None.
    If the value is a valid file path, read and parse the JSON from the file.
    Otherwise, parse the value as a JSON string.
    
    Args:
        value (Optional[str]): The JSON string or file path.
    Returns:
        Optional[dict or list]: The parsed JSON object, or None if input is None/
    """
    if not value:
        return None
    if os.path.exists(value) and os.path.isfile(value):
        with open(value, 'r') as f:
            return json.load(f)
    return json.loads(value)


def _get_token(env_name: str, token_file: Optional[str]) -> str:
    """
    Retrieve the Zenodo token from an environment variable or a file.
    
    Args:
        env_name (str): The name of the environment variable to check.
        token_file (Optional[str]): The path to a file containing the token.
    Returns:
        str: The Zenodo token.
    Raises:
        SystemExit: If neither the environment variable nor the file provides a valid token.
    """
    tok = os.getenv(env_name)
    if tok:
        return tok.strip()
    if token_file:
        with open(token_file, 'r') as f:
            return f.read().strip()
    raise SystemExit(f'ERROR: not a valid token. Define venv {env_name} or --token-file PATH.')


def _resolve_description(text: Optional[str], path: Optional[str]) -> str:
    """
    Resolve the description text from a string or a file path.
    
    Args:
        text (Optional[str]): The description text.
        path (Optional[str]): The path to a file containing the description.
    Returns:
        str: The resolved description text.
    Raises:
        SystemExit: If neither the text nor the file provides a valid description.
    """
    if path:
        file_path = Path(path).expanduser().resolve()
        if not file_path.exists():
            raise SystemExit(f'ERROR: description file not found: {file_path}')
        return file_path.read_text(encoding='utf-8')
    if text:
        return text
    raise SystemExit('ERROR: description text not provided. Use --description or --description-file.')


def _collect_release_paths(release_root: str, include_figs: bool) -> List[str]:
    """
    Collect the paths to the release subdirectories to include in the upload.
    
    Args:
        release_root (str): The root directory of the release.
        include_figs (bool): Whether to include the 'figs' subdirectory if it exists.
    Returns:
        List[str]: A list of paths to the selected subdirectories.
    Raises:
        FileNotFoundError: If the release root directory does not exist.
    """
    root = Path(release_root).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f'Release root not found: {root}')

    selected: List[str] = []
    missing = []
    for sub in DEFAULT_RELEASE_SUBDIRS:
        candidate = root / sub
        if candidate.exists():
            selected.append(str(candidate))
        else:
            missing.append(candidate)

    if missing:
        print('WARNING: missing release subdirectories:')
        for m in missing:
            print(f' - {m}')

    if include_figs:
        figs_dir = root / 'figs'
        if figs_dir.exists():
            selected.append(str(figs_dir))
        else:
            print(f'WARNING: figs directory requested but not found: {figs_dir}')

    return selected


def main():
    args = parse_args()

    token = _get_token(args.token_env, args.token_file)
    base_url = 'https://sandbox.zenodo.org' if args.sandbox else 'https://zenodo.org'
    init_t = time.time()

    staging_name = slugify(args.title)
    description = _resolve_description(args.description, args.description_file)
    release_paths = _collect_release_paths(args.release_root, args.include_figs) if args.release_root else []
    manual_paths = args.paths or []
    source_paths = list(dict.fromkeys(release_paths + manual_paths))
    if not source_paths:
        raise SystemExit('ERROR: no valid source paths to stage. Check --paths/--release-root inputs.')

    print('- Source paths selected:')
    for src in source_paths:
        print(f'  - {src}')

    staging_dir, copied_paths = ensure_pscratch_copy(source_paths=source_paths,
                                                     pscratch_base_dir=args.pscratch_dir,
                                                     staging_name=staging_name,
                                                     keep_tree=args.keep_tree)

    print(f'--- Staging folder:\n  {staging_dir}')
    print(f'- Files copied: {len(copied_paths)}')
    if len(copied_paths) <= 20:
        for p in copied_paths:
            print(f' - {p}')
    else:
        for p in copied_paths[:10]:
            print(f' - {p}')
        print(' - ...')
        for p in copied_paths[-5:]:
            print(f' - {p}')

    tar_paths = _make_folder_tarballs(staging_dir)
    print(f'- Tarballs to upload: {len(tar_paths)}')
    for t in tar_paths:
        print(f' - {t}')

    if args.dry_run:
        print('---- SKIPPED upload: omitted due to --dry-run.')
        return

    creators: List[Dict] = _load_json_or_string(args.creators_json) or []
    related = _load_json_or_string(args.related_identifiers_json)

    dep = push_to_zenodo(token=token, base_url=base_url, files_on_disk=tar_paths,
                         title=args.title, description=description, creators=creators,
                         keywords=args.keywords, access_right=args.access_right, license_id=args.license,
                         communities=args.communities, publish=args.publish, version=args.version,
                         related_identifiers=related)

    out = {'staging_dir': staging_dir,
           'deposition_id': dep.get('id'),
           'state': dep.get('state'),
           'links': dep.get('links', {}),
           'title': dep.get('title') or dep.get('metadata', {}).get('title'),
           'published': bool(dep.get('doi')),
           'doi': dep.get('doi'),
           'record_url': dep.get('links', {}).get('record_html') or dep.get('links', {}).get('html'),
           'elapsed t': f'{(time.time() - init_t)/60} min'}
    print(json.dumps(out, indent=2))

if __name__ == '__main__':
    main()