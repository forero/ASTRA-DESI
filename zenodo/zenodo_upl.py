import os, re, json, time, pathlib, copy
import shutil, pathlib, mimetypes
from os.path import commonprefix, relpath
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Tuple, Union
import requests


def slugify(text: str) -> str:
    """
    Simplified slugify: lowercase, spaces to '-', remove special chars.
    
    Args:
        text (str): The input text to slugify.
    Returns:
        str: The slugified text.
    """
    text = (text or '').strip().lower()
    text = re.sub(r'\s+', '-', text)
    text = re.sub(r'[^a-z0-9_.-]+', '-', text)
    text = re.sub(r'-{2,}', '-', text).strip('-')
    return text or 'dataset'


def timestamp_compact() -> str:
    """
    Returns a compact timestamp string: YYYYMMDDTHHMM
   
    Returns:
        str: The compact timestamp.
    """
    return time.strftime('%Y%m%dT%H%M')

def make_unique_dir(parent: pathlib.Path, base_name: str) -> pathlib.Path:
    """
    Create a unique directory under 'parent' with the given 'base_name'. If a directory
    with 'base_name' exists, appends '_copyN' suffix to make it unique.
    
    Args:
        parent (pathlib.Path): The parent directory where to create the new directory.
        base_name (str): The desired base name for the new directory.
    Returns:
        pathlib.Path: The path to the newly created unique directory.
    """
    parent.mkdir(parents=True, exist_ok=True)
    candidate = parent / base_name
    if not candidate.exists():
        candidate.mkdir(parents=True, exist_ok=False)
        return candidate

    n = 1
    while True:
        cand = parent / f'{base_name}_copy{n}'
        if not cand.exists():
            cand.mkdir(parents=True, exist_ok=False)
            return cand
        n += 1


def iter_files_recursive(root: pathlib.Path) -> List[pathlib.Path]:
    """
    Recursively list all files under the given root directory.
    
    Args:
        root (pathlib.Path): The root directory to search.
    Returns:
        List[pathlib.Path]: A list of all file paths found.
    """
    files: List[pathlib.Path] = []
    for dirpath, _, filenames in os.walk(root):
        for fname in filenames:
            files.append(pathlib.Path(dirpath) / fname)
    return files


def safe_copy2(src: pathlib.Path, dst: pathlib.Path) -> None:
    """
    Copy a file from src to dst, creating parent directories if needed.
    
    Args:
        src (pathlib.Path): Source file path.
        dst (pathlib.Path): Destination file path.
    """
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(str(src), str(dst))


def safe_copytree(src_dir: pathlib.Path, dst_dir: pathlib.Path) -> None:
    """
    Copy a directory tree from src_dir to dst_dir. If dst_dir exists, appends _copyN to make it unique.
    
    Args:
        src_dir (pathlib.Path): Source directory path.
        dst_dir (pathlib.Path): Destination directory path.
    """
    if not dst_dir.exists():
        shutil.copytree(str(src_dir), str(dst_dir))
        return

    n = 1
    while True:
        new_dst = dst_dir.parent / f'{dst_dir.name}_copy{n}'
        if not new_dst.exists():
            shutil.copytree(str(src_dir), str(new_dst))
            return
        n += 1


def ensure_pscratch_copy(source_paths: List[str], pscratch_base_dir: str, staging_name: str,
                         keep_tree: bool = False,) -> Tuple[str, List[str]]:
    """
    Copy the given source paths (files or directories) into a unique staging directory
    under the specified pscratch base directory. Optionally keep the directory tree structure.
    
    Args:
        source_paths (List[str]): List of file or directory paths to copy.
        pscratch_base_dir (str): The base directory under which to create the staging area.
        staging_name (str): Base name for the staging directory.
        keep_tree (bool): If True, keep the directory structure; otherwise, flatten it.
    Returns:
        Tuple[str, List[str]]: The path to the staging directory and a list of all copied file paths.
    Raises:
        FileNotFoundError: If any of the source paths do not exist.
    """
    pscratch_base = pathlib.Path(pscratch_base_dir).expanduser().resolve()

    staging_root = pscratch_base / 'zenodo_staging'
    base_tag = f'{slugify(staging_name)}_{timestamp_compact()}'
    staging_dir = make_unique_dir(staging_root, base_tag)

    copied_files: List[str] = []

    for src in source_paths:
        srcp = pathlib.Path(src).expanduser().resolve()
        if not srcp.exists():
            raise FileNotFoundError(f'Source not found: {srcp}')

        if srcp.is_dir():
            if keep_tree:
                dst_dir = staging_dir / srcp.name
                safe_copytree(srcp, dst_dir)
                for f in iter_files_recursive(dst_dir):
                    copied_files.append(str(f))
            else:
                for dirpath, _, filenames in os.walk(srcp):
                    for fname in filenames:
                        srcf = pathlib.Path(dirpath) / fname
                        dstf = staging_dir / fname

                        if dstf.exists():
                            stem = dstf.stem
                            suf = dstf.suffix
                            k = 1
                            while dstf.exists():
                                dstf = staging_dir / f'{stem}_copy{k}{suf}'
                                k += 1
                        safe_copy2(srcf, dstf)
                        copied_files.append(str(dstf))
        else:
            dstf = staging_dir / srcp.name
            if dstf.exists():
                stem = dstf.stem
                suf = dstf.suffix
                k = 1
                while dstf.exists():
                    dstf = staging_dir / f'{stem}_copy{k}{suf}'
                    k += 1
            safe_copy2(srcp, dstf)
            copied_files.append(str(dstf))

    return (str(staging_dir), copied_files)


#? ------ classes for API
@dataclass
class ZenodoConfig:
    token: str
    base_url: str = 'https://zenodo.org'
    timeout: int = 60
    retries: int = 3
    retry_backoff_s: float = 2.0

    @property
    def api(self) -> str:
        return f'{self.base_url}/api'

@dataclass
class Creator:
    name: str
    affiliation: Optional[str] = None
    orcid: Optional[str] = None

    def to_zenodo(self) -> Dict[str, Any]:
        out = {'name': self.name}
        if self.affiliation:
            out['affiliation'] = self.affiliation
        if self.orcid:
            out['orcid'] = self.orcid
        return out

@dataclass
class DepositionMeta:
    title: str
    upload_type: str = 'dataset'
    description: str = 'Data products generated by ASTRA-DESI pipeline.'
    creators: List[Creator] = field(default_factory=list)
    keywords: List[str] = field(default_factory=lambda: ['ASTRA', 'DESI', 'cosmic web'])
    access_right: str = 'open'
    license: str = 'cc-by-4.0'
    communities: Optional[List[str]] = None
    version: Optional[str] = None
    related_identifiers: Optional[List[Dict[str, str]]] = None

    def to_zenodo(self) -> Dict[str, Any]:
        md: Dict[str, Any] = {'metadata': {'title': self.title,
                                           'upload_type': self.upload_type,
                                           'description': self.description,
                                           'creators': [c.to_zenodo() for c in self.creators] or [{'name': 'Unknown'}],
                                           'keywords': self.keywords,
                                           'access_right': self.access_right,
                                           'license': self.license,}}
        if self.communities:
            md['metadata']['communities'] = [{'identifier': c} for c in self.communities]
        if self.version:
            md['metadata']['version'] = self.version
        if self.related_identifiers:
            md['metadata']['related_identifiers'] = self.related_identifiers
        return md


class ZenodoUploader:
    def __init__(self, cfg: ZenodoConfig):
        self.cfg = cfg
        self.session = requests.Session()
        self.session.headers.update({'Authorization': f'Bearer {self.cfg.token}'})

    def _request(self, method: str, url: str, **kwargs) -> requests.Response:
        last_exc = None
        for attempt in range(1, self.cfg.retries + 1):
            try:
                resp = self.session.request(method, url, timeout=self.cfg.timeout, **kwargs)

                if resp.status_code >= 400:
                    if resp.status_code in (502, 503, 504) and attempt < self.cfg.retries:
                        time.sleep(self.cfg.retry_backoff_s * attempt)
                        continue
                return resp
            except requests.RequestException as exc:
                last_exc = exc
                if attempt < self.cfg.retries:
                    time.sleep(self.cfg.retry_backoff_s * attempt)
                else:
                    raise
        raise last_exc

    def create_deposition(self, meta: DepositionMeta) -> Dict[str, Any]:
        """
        Create a new deposition with the given metadata.
        
        Args:
            meta (DepositionMeta): The metadata for the deposition.
        Returns:
            Dict[str, Any]: The JSON response from Zenodo with deposition details.
        """
        url = f'{self.cfg.api}/deposit/depositions'
        resp = self._request('POST', url, json=meta.to_zenodo())
        resp.raise_for_status()
        return resp.json()

    def create_new_version(self, deposition_id: int) -> Dict[str, Any]:
        """
        Create a new draft version for an existing deposition.
        
        Args:
            deposition_id (int): The ID of the existing deposition.
        Returns:
            Dict[str, Any]: The JSON response from Zenodo with the new draft version details.
        """
        url = f'{self.cfg.api}/deposit/depositions/{deposition_id}/actions/newversion'
        resp = self._request('POST', url)
        resp.raise_for_status()
        data = resp.json()
        latest_draft_url = data.get('links', {}).get('latest_draft')
        if latest_draft_url:
            latest_resp = self._request('GET', latest_draft_url)
            latest_resp.raise_for_status()
            return latest_resp.json()
        return data

    def update_deposition_metadata(self, deposition_id: int, meta: Union[DepositionMeta, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Update metadata for an existing deposition draft.
        
        Args:
            deposition_id (int): The ID of the deposition.
            meta (Union[DepositionMeta, Dict[str, Any]]): The new metadata to set.
        Returns:
            Dict[str, Any]: The JSON response from Zenodo with updated deposition details.
        """
        url = f'{self.cfg.api}/deposit/depositions/{deposition_id}'
        payload = meta.to_zenodo() if isinstance(meta, DepositionMeta) else meta
        resp = self._request('PUT', url, json=payload)
        resp.raise_for_status()
        return resp.json()

    def delete_file(self, deposition_id: int, file_id: int) -> None:
        """
        Remove a file from a deposition draft.
        
        Args:
            deposition_id (int): The ID of the deposition.
            file_id (int): The ID of the file to delete.
        Raises:
            requests.HTTPError: If the deletion fails.
        """
        url = f'{self.cfg.api}/deposit/depositions/{deposition_id}/files/{file_id}'
        resp = self._request('DELETE', url)
        resp.raise_for_status()

    def clear_deposition_files(self, deposition_id: int, files: Optional[List[Dict[str, Any]]]) -> None:
        """
        Delete all files listed from a deposition draft.
        
        Args:
            deposition_id (int): The ID of the deposition.
            files (Optional[List[Dict[str, Any]]]): List of file info dicts from Zenodo.
        Raises:
            requests.HTTPError: If any deletion fails.
        """
        for info in files or []:
            file_id = info.get('id')
            if file_id is None:
                continue
            self.delete_file(deposition_id, file_id)

    def get_deposition(self, deposition_id: int) -> Dict[str, Any]:
        """
        Retrieve details of an existing deposition by its ID.
        
        Args:
            deposition_id (int): The ID of the deposition.
        Returns:
            Dict[str, Any]: The JSON response from Zenodo with deposition details.
        """
        url = f'{self.cfg.api}/deposit/depositions/{deposition_id}'
        resp = self._request('GET', url)
        resp.raise_for_status()
        return resp.json()

    def upload_file_via_bucket(self, bucket_url: str, filepath: str, dest_name: Optional[str] = None):
        """
        Upload a file (e.g., .tar.gz created from staging folders) to the deposition's bucket.
        
        Args:
            bucket_url (str): The bucket URL from the deposition.
            filepath (str): The local file path to upload.
            dest_name (Optional[str]): The destination filename in Zenodo. If None, uses the local filename.
        Raises:
            requests.HTTPError: If the upload fails.
        """
        filename = dest_name or os.path.basename(filepath)
        upload_url = f"{bucket_url}/{filename}"

        with open(filepath, "rb") as f:
            resp = self._request("PUT", upload_url, data=f,
                                 headers={"Content-Type": "application/octet-stream"})
        resp.raise_for_status()

    def publish(self, deposition_id: int) -> Dict[str, Any]:
        """
        Publish the deposition with the given ID.
        
        Args:
            deposition_id (int): The ID of the deposition to publish.
        Returns:
            Dict[str, Any]: The JSON response from Zenodo after publishing.
        """
        url = f'{self.cfg.api}/deposit/depositions/{deposition_id}/actions/publish'
        resp = self._request('POST', url)
        resp.raise_for_status()
        return resp.json()


def push_to_zenodo(token: str, base_url: str, files_on_disk: List[str], title: str, description: str,
                   creators: List[Dict[str, str]], keywords: Optional[List[str]] = None,
                   access_right: str = 'open', license_id: str = 'cc-by-4.0',
                   communities: Optional[List[str]] = None, publish: bool = False,
                   version: Optional[str] = None,
                   related_identifiers: Optional[List[Dict[str, str]]] = None,
                   existing_deposition_id: Optional[int] = None,
                   keep_existing_files: bool = False,
                   reuse_metadata: bool = False,) -> Dict[str, Any]:
    """
    Push files to Zenodo with the given metadata.

    Args:
        token (str): Zenodo API token.
        base_url (str): Base URL for Zenodo API (e.g., 'https://zenodo.org' or sandbox).
        files_on_disk (List[str]): List of file paths to upload.
        title (str): Title of the deposition.
        description (str): Description of the deposition.
        creators (List[Dict[str, str]]): List of creators with 'name', optional 'affiliation' and 'orcid'.
        keywords (Optional[List[str]]): List of keywords.
        access_right (str): Access right ('open', 'embargoed', 'restricted', 'closed').
        license_id (str): License ID (e.g., 'cc-by-4.0').
        communities (Optional[List[str]]): List of community identifiers.
        publish (bool): Whether to publish the deposition after upload.
        version (Optional[str]): Version string for the deposition.
        related_identifiers (Optional[List[Dict[str, str]]]): Related identifiers.
        existing_deposition_id (Optional[int]): Existing deposition to create a new Zenodo version from.
        keep_existing_files (bool): Keep files copied from the previous version.
        reuse_metadata (bool): Reuse metadata already stored in the deposition.
    Returns:
        Dict[str, Any]: The JSON response from Zenodo with deposition details.
    """
    cfg = ZenodoConfig(token=token, base_url=base_url)
    up = ZenodoUploader(cfg)

    meta = DepositionMeta(title=title, description=description, creators=[Creator(**c) for c in creators],
                          keywords=keywords or ['ASTRA', 'DESI', 'cosmic web'], access_right=access_right,
                          license=license_id, communities=communities, version=version,
                          related_identifiers=related_identifiers)

    if existing_deposition_id:
        dep = up.create_new_version(existing_deposition_id)
        dep_id = dep.get('id')
        if dep_id is None:
            raise RuntimeError('Failed to create new Zenodo draft version.')
        if reuse_metadata:
            metadata = copy.deepcopy(dep.get('metadata') or {})
            if not metadata:
                raise RuntimeError('Existing deposition metadata missing; cannot reuse.')
            metadata.pop('prereserve_doi', None)
            metadata.pop('doi', None)
            if version is not None:
                metadata['version'] = version
            meta_payload = {'metadata': metadata}
        else:
            meta_payload = meta
        dep = up.update_deposition_metadata(dep_id, meta_payload)
        if not keep_existing_files:
            up.clear_deposition_files(dep_id, dep.get('files'))
            dep = up.get_deposition(dep_id)
    else:
        dep = up.create_deposition(meta)
        dep_id = dep.get('id')
        if dep_id is None:
            raise RuntimeError('Failed to create Zenodo deposition.')

    bucket = dep.get('links', {}).get('bucket')
    if not bucket:
        raise RuntimeError('Zenodo response missing bucket upload URL.')

    for fpath in files_on_disk:
        up.upload_file_via_bucket(bucket, fpath)

    if publish:
        dep = up.publish(dep_id)

    return dep