import os, json, tarfile
import argparse, time
from pathlib import Path
from typing import List, Dict, Optional

from zenodo_upl import (ensure_pscratch_copy, push_to_zenodo, slugify)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--paths', nargs='+', required=True,help='Absolute path for files')
    p.add_argument('--pscratch-dir', required=True, help='Nersc base at /pscratch to create temp folder')
    p.add_argument('--keep-tree', action='store_true', help='Preserve nersc structure (raw/, class/, ...).')
    p.add_argument('--title', required=True, help='Zenodo record title')
    p.add_argument('--description', required=True, help='Record description')
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

    return p.parse_args()


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


def main():
    args = parse_args()

    token = _get_token(args.token_env, args.token_file)
    base_url = 'https://sandbox.zenodo.org' if args.sandbox else 'https://zenodo.org'
    init_t = time.time()

    staging_name = slugify(args.title)
    staging_dir, copied_paths = ensure_pscratch_copy(source_paths=args.paths,
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
                         title=args.title, description=args.description, creators=creators,
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