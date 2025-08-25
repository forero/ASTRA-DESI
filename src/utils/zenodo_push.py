import os
import json
import argparse
from typing import List, Dict

from zenodo_upl import (ensure_pscratch_copy, push_to_zenodo,)

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--paths', nargs='+', required=True,
                   help='Paths of files or folders to upload (they will be copied first to /pscratch).')
    p.add_argument('--pscratch-dir', required=True,
                   help='Base directory in /pscratch where the files will be copied.')
    p.add_argument('--keep-tree', action='store_true',
                   help='Preserve relative tree from repo root when copying to /pscratch.')
    p.add_argument('--title', required=True)
    p.add_argument('--description', required=True)
    p.add_argument('--creators-json', required=True,
                   help="JSON with list of creators, e.g. \'[{'name':'author1','orcid':'0000-0000-...'}]\'")

    p.add_argument('--keywords', nargs='*', default=None)
    p.add_argument('--communities', nargs='*', default=None)
    p.add_argument('--access-right', default='open', choices=['open', 'embargoed', 'restricted', 'closed'])
    p.add_argument('--license', default='cc-by-4.0')
    p.add_argument('--version', default=None)
    p.add_argument('--related-identifiers-json', default=None,
                   help="JSON with list of objects {'identifier': '...', 'relation': '...', 'scheme': 'doi|url|...'}")

    p.add_argument('--publish', action='store_true', help='Publish the deposition after uploading files.')
    p.add_argument('--sandbox', action='store_true', help='Use https://sandbox.zenodo.org')
    p.add_argument('--dry-run', action='store_true', help='Only copy to /pscratch; do not upload to Zenodo.')
    p.add_argument('--token-env', default='ZENODO_TOKEN',
                   help='Name of the environment variable where the token is stored. Default is ZENODO_TOKEN.')
    return p.parse_args()


def main():
    args = parse_args()

    token = os.getenv(args.token_env)
    if not args.dry_run and not token:
        raise SystemExit(f'ERROR: env variable {args.token_env} is not set')

    base_url = 'https://sandbox.zenodo.org' if args.sandbox else 'https://zenodo.org'

    # 1. copy to /pscratch
    copied_paths = ensure_pscratch_copy(args.paths, args.pscratch_dir, keep_tree=args.keep_tree)

    # 2. upload
    if args.dry_run:
        print('[DRY-RUN] Copied to /pscratch, without uploading to Zenodo:')
        for p in copied_paths:
            print(f' - {p}')
        return

    creators: List[Dict] = json.loads(args.creators_json)
    related = json.loads(args.related_identifiers_json) if args.related_identifiers_json else None

    dep = push_to_zenodo(token=token, base_url=base_url, files_on_disk=copied_paths, title=args.title,
                         description=args.description, creators=creators, keywords=args.keywords,
                         access_right=args.access_right, license_id=args.license, communities=args.communities,
                         publish=args.publish, version=args.version, related_identifiers=related)

    #outputs for logs/CI
    print(json.dumps({'deposition_id': dep.get('id'), 'state': dep.get('state'), 'links': dep.get('links', {}),
                      'title': dep.get('title') or dep.get('metadata', {}).get('title'), 'published': bool(dep.get('doi')),
                      'doi': dep.get('doi'), 'record_url': dep.get('links', {}).get('record_html') or dep.get('links', {}).get('html'),},
                     indent=2))


if __name__ == '__main__':
    main()
#export ZENODO_TOKEN=LuJ15tBnQfwEOvUQW6WKULA1iBhNruzyg2VZORLIFHRTJna1mod3zxzYG0Z5