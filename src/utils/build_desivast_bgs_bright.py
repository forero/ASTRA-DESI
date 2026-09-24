import argparse
from pathlib import Path
import fitsio
import healpy as hp
import numpy as np
from astropy.table import Table


DEFAULT_DESIVAST = Path('/global/cfs/cdirs/desi/public/dr1/vac/dr1/desivast/v1.0')
DEFAULT_FASTSPEC = Path('/global/cfs/cdirs/desi/public/dr1/vac/dr1/fastspecfit/iron/v2.1/catalogs/fastspec-iron.fits')
DEFAULT_LSS = Path('/global/cfs/cdirs/desi/public/dr1/survey/catalogs/dr1/LSS/iron/LSScats/v1.5pip')
CAPS = ('NGC', 'SGC')
DEFAULT_MASK_NSIDE = 256
DEFAULT_RANDOM_FILES = 18
DEFAULT_CHUNK_ROWS = 1_000_000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--desivast-dir', type=Path, default=DEFAULT_DESIVAST)
    parser.add_argument('--fastspec', type=Path, default=DEFAULT_FASTSPEC)
    parser.add_argument('--output-dir', type=Path, default=Path('desivast_bgs_bright_astra'))
    parser.add_argument('--lss-dir', type=Path, default=DEFAULT_LSS)
    parser.add_argument('--random-mode', choices=('build', 'none'), default='build')
    parser.add_argument('--random-seed', type=int, default=20260924)
    parser.add_argument('--mask-nside', type=int, default=DEFAULT_MASK_NSIDE)
    parser.add_argument('--chunk-rows', type=int, default=DEFAULT_CHUNK_ROWS)
    parser.add_argument('--n-random-files', type=int, default=DEFAULT_RANDOM_FILES)
    parser.add_argument('--link-randoms', action='store_true')
    parser.add_argument('--overwrite', action='store_true')
    return parser.parse_args()


def require_file(path):
    if not path.is_file():
        raise FileNotFoundError(f'Required file not found: {path}')


def read_fastspec(path):
    '''Read aligned FastSpecFit columns and apply the E correction in place.'''
    require_file(path)
    print(f'Reading FastSpecFit: {path}', flush=True)
    with fitsio.FITS(path) as hdus:
        meta = hdus[2].read(columns=['TARGETID', 'RA', 'DEC', 'Z', 'ZWARN', 'DELTACHI2'])
        fastspec = hdus[1].read(columns=['ABSMAG01_SDSS_R', 'ABSMAG01_IVAR_SDSS_R'])

    if len(meta) != len(fastspec):
        raise RuntimeError(f'FASTSPEC/METADATA row mismatch: {len(fastspec):,} != {len(meta):,}')

    # Match DESIVAST's iron.py operation exactly: preserve the catalogue dtype
    # and modify the FastSpecFit absolute magnitude array in place.
    fastspec['ABSMAG01_SDSS_R'] += 0.97 * (meta['Z'] - 0.1)

    targetid = np.asarray(meta['TARGETID'])
    order = np.argsort(targetid)
    sorted_targetid = targetid[order]
    # FastSpecFit contains repeat TARGETIDs from different reductions. Keep
    # the first sorted occurrence, exactly as DESIVAST's np.argsort followed by
    # np.searchsorted does; rejecting duplicates would not reproduce iron.py.
    ndup = int(np.count_nonzero(sorted_targetid[1:] == sorted_targetid[:-1]))

    print(f'Z dtype: {meta['Z'].dtype}', flush=True)
    print(f'K+E magnitude dtype: {fastspec['ABSMAG01_SDSS_R'].dtype}', flush=True)
    print(f'Repeated FastSpecFit TARGETIDs: {ndup:,}', flush=True)
    return meta, fastspec, order, sorted_targetid


def match_targetids(requested, targetid, order, sorted_targetid, cap):
    '''Return safe FastSpecFit row indices for requested TARGETIDs.'''
    positions = np.searchsorted(sorted_targetid, requested, side='left')
    in_range = positions < len(sorted_targetid)
    matched = np.zeros(len(requested), dtype=bool)
    matched[in_range] = (sorted_targetid[positions[in_range]] == requested[in_range])
    if not np.all(matched):
        missing = requested[~matched]
        preview = ', '.join(map(str, missing[:10]))
        raise RuntimeError(f'{cap}: {len(missing):,} GALZONE TARGETIDs are absent from '
                           f'FastSpecFit; first values: {preview}')
    indices = order[positions]
    if not np.all(targetid[indices] == requested):
        raise AssertionError(f'{cap}ernal TARGETID matching failure')
    return indices


def write_real_catalogue(cap, desivast_dir, output_dir, meta, fastspec, order, sorted_targetid, overwrite):
    source = desivast_dir / f'DESIVAST_BGS_VOLLIM_V2_REVOLVER_{cap}.fits'
    require_file(source)

    with fitsio.FITS(source) as hdus:
        gal_targetid = hdus['GALZONE'].read(columns=['TARGET'])['TARGET']
        expected = int(hdus[0].read_header()['MSKGAL'])

    targetid = np.asarray(meta['TARGETID'])
    idx = match_targetids(gal_targetid, targetid, order, sorted_targetid, cap)
    z = meta['Z'][idx]
    mr_ke = fastspec['ABSMAG01_SDSS_R'][idx]

    # These strict inequalities reproduce the released DESIVAST construction.
    # GALZONE already comes from the quality-selected BGS Bright parent sample,
    # so do not reapply ZWARN/DELTACHI2 to the arbitrary first FastSpecFit row
    # of duplicate TARGETIDs. Doing that incorrectly rejects valid objects.
    first_row_bad_quality = int(np.count_nonzero((meta['ZWARN'][idx] != 0) | (meta['DELTACHI2'][idx] <= 45.0)))
    select = (np.isfinite(z)
              & np.isfinite(mr_ke)
              & (z > 0.0)
              & (z < 0.24)
              & (mr_ke < -20.0))

    selected_idx = idx[select]
    output = Table()
    output['TARGETID'] = np.asarray(meta['TARGETID'][selected_idx], dtype=np.int64)
    output['RA'] = np.asarray(meta['RA'][selected_idx], dtype=np.float64)
    output['DEC'] = np.asarray(meta['DEC'][selected_idx], dtype=np.float64)
    output['Z'] = np.asarray(meta['Z'][selected_idx], dtype=np.float64)
    output.meta['SAMPLE'] = 'DESIVAST DR1 BGS Bright volume limited'
    output.meta['CAP'] = cap
    output.meta['ZMIN'] = 0.0
    output.meta['ZMAX'] = 0.24
    output.meta['MRMAX'] = -20.0
    output.meta['ECORR'] = 'MrK + 0.97*(z-0.1)'
    output.meta['ZWARN'] = 0
    output.meta['DCHI2MIN'] = 45.0
    output.meta['MSKGAL'] = expected

    output_path = output_dir / f'BGS_BRIGHT_{cap}_clustering.dat.fits'
    if output_path.exists() and not overwrite:
        raise FileExistsError(f'Output exists; use --overwrite: {output_path}')
    output.write(output_path, format='fits', overwrite=overwrite)

    actual = len(output)
    print(f'\n{cap}', flush=True)
    print(f'  GALZONE total : {len(gal_targetid):,}', flush=True)
    print(f'  selected      : {actual:,}', flush=True)
    print(f'  MSKGAL        : {expected:,}', flush=True)
    print(f'  first-row quality mismatch (diagnostic): {first_row_bad_quality:,}', flush=True)
    print(f'  difference    : {expected - actual:+,}', flush=True)
    print(f'  exact match   : {actual == expected}', flush=True)
    print(f'  wrote         : {output_path}', flush=True)
    return output_path, actual



def build_desivast_angular_mask(cap, desivast_dir, nside):
    '''Reconstruct the DESIVAST angular support from its dense GALZONE parent.'''
    if not hp.isnsideok(nside):
        raise ValueError(f'Invalid HEALPix NSIDE: {nside}')
    source = desivast_dir / f'DESIVAST_BGS_VOLLIM_V2_REVOLVER_{cap}.fits'
    require_file(source)
    xyz = fitsio.read(source, ext='GALZONE', columns=['X', 'Y', 'Z'])
    radius_xy = np.hypot(xyz['X'], xyz['Y'])
    valid = np.isfinite(radius_xy) & np.isfinite(xyz['Z']) & (radius_xy > 0)
    ra = np.degrees(np.arctan2(xyz['Y'][valid], xyz['X'][valid])) % 360.0
    dec = np.degrees(np.arctan2(xyz['Z'][valid], radius_xy[valid]))
    pixels = hp.ang2pix(nside, ra, dec, lonlat=True, nest=True)
    mask = np.zeros(hp.nside2npix(nside), dtype=bool)
    mask[np.unique(pixels)] = True
    area = np.count_nonzero(mask) * hp.nside2pixarea(nside, degrees=True)
    print(f'{cap} DESIVAST mask: NSIDE={nside} NESTED, '
          f'pixels={np.count_nonzero(mask):,}, area={area:,.3f} deg2',
          flush=True)
    return mask


def sample_masked_random_positions(source, angular_mask, nside, sample_size, rng, chunk_rows):
    '''Uniformly select masked random positions with a streaming priority sample.'''
    require_file(source)
    if chunk_rows <= 0:
        raise ValueError('chunk_rows must be greater than zero')

    reservoir = None
    reservoir_keys = np.empty(0, dtype=np.float64)
    inside_count = 0
    columns = ['TARGETID', 'RA', 'DEC']
    with fitsio.FITS(source) as hdus:
        hdu = hdus[1]
        missing = sorted(set(columns).difference(hdu.get_colnames()))
        if missing:
            raise KeyError(f'{source} is missing columns: {missing}')
        nrows = hdu.get_nrows()
        for start in range(0, nrows, chunk_rows):
            stop = min(start + chunk_rows, nrows)
            rows = np.arange(start, stop, dtype=np.int64)
            chunk = hdu.read(columns=columns, rows=rows)
            ra = np.asarray(chunk['RA'], dtype=np.float64)
            dec = np.asarray(chunk['DEC'], dtype=np.float64)
            valid = (np.isfinite(ra)
                     & np.isfinite(dec)
                     & (dec >= -90.0)
                     & (dec <= 90.0))
            keep = np.zeros(len(chunk), dtype=bool)
            if np.any(valid):
                pix = hp.ang2pix(nside, np.mod(ra[valid], 360.0), dec[valid],
                                 lonlat=True, nest=True)
                keep[valid] = angular_mask[pix]
            candidates = chunk[keep]
            inside_count += len(candidates)
            if not len(candidates):
                continue

            keys = rng.random(len(candidates))
            if len(candidates) > sample_size:
                local = np.argpartition(keys, sample_size - 1)[:sample_size]
                candidates = candidates[local]
                keys = keys[local]
            if reservoir is None:
                reservoir = candidates.copy()
                reservoir_keys = keys
            else:
                reservoir = np.concatenate((reservoir, candidates))
                reservoir_keys = np.concatenate((reservoir_keys, keys))
            if len(reservoir) > sample_size:
                chosen = np.argpartition(reservoir_keys, sample_size - 1)[:sample_size]
                reservoir = reservoir[chosen]
                reservoir_keys = reservoir_keys[chosen]

    if reservoir is None or len(reservoir) < sample_size:
        available = 0 if reservoir is None else len(reservoir)
        raise RuntimeError(f'{source.name}: only {available:,} masked random positions for '
                           f'{sample_size:,} galaxies')
    order = rng.permutation(sample_size)
    return reservoir[order], inside_count


def prepare_output_path(path, overwrite):
    '''Safely remove an old output, including links made by earlier versions.'''
    if path.is_symlink():
        if not overwrite:
            raise FileExistsError(f'Symlink exists; use --overwrite: {path}')
        path.unlink()
    elif path.exists() and not overwrite:
        raise FileExistsError(f'Output exists; use --overwrite: {path}')


def build_random_catalogues(output_dir, lss_dir, desivast_dir, nside,
                            n_random_files, seed, chunk_rows, overwrite):
    '''Write ASTRA randoms with matched angular support and exact per-cap n(z).'''
    if n_random_files <= 0:
        raise ValueError('n_random_files must be greater than zero')
    for cap_index, cap in enumerate(CAPS):
        real_path = output_dir / f'BGS_BRIGHT_{cap}_clustering.dat.fits'
        require_file(real_path)
        real = fitsio.read(real_path, ext=1, columns=['Z'])
        real_z = np.asarray(real['Z'], dtype=np.float64)
        angular_mask = build_desivast_angular_mask(cap, desivast_dir, nside)

        for index in range(n_random_files):
            name = f'BGS_BRIGHT_{cap}_{index}_clustering.ran.fits'
            source = lss_dir / name
            destination = output_dir / name
            rng = np.random.default_rng(np.random.SeedSequence([int(seed), cap_index, index]))
            positions, inside_count = sample_masked_random_positions(source=source,
                                                                     angular_mask=angular_mask,
                                                                     nside=nside,
                                                                     sample_size=len(real_z),
                                                                     rng=rng,
                                                                     chunk_rows=chunk_rows)
            shuffled_z = real_z[rng.permutation(len(real_z))]
            random_table = Table()
            random_table['TARGETID'] = np.asarray(positions['TARGETID'], dtype=np.int64)
            random_table['RA'] = np.asarray(positions['RA'], dtype=np.float64)
            random_table['DEC'] = np.asarray(positions['DEC'], dtype=np.float64)
            random_table['Z'] = shuffled_z
            random_table.meta['SAMPLE'] = 'DESIVAST DR1 BGS Bright random'
            random_table.meta['CAP'] = cap
            random_table.meta['RANINDEX'] = index
            random_table.meta['SEED'] = int(seed)
            random_table.meta['NSIDE'] = int(nside)
            random_table.meta['ORDERING'] = 'NESTED'
            random_table.meta['NZMATCH'] = 'exact permutation of real Z'

            prepare_output_path(destination, overwrite)
            random_table.write(destination, format='fits', overwrite=True)
            print(f'  wrote random {cap} {index:02d}: rows={len(random_table):,}, '
                  f'masked candidates={inside_count:,} -> {destination}',
                  flush=True)


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    meta, fastspec, order, sorted_targetid = read_fastspec(args.fastspec)

    counts = {}
    for cap in CAPS:
        _, counts[cap] = write_real_catalogue(cap=cap, desivast_dir=args.desivast_dir,
                                              output_dir=args.output_dir, meta=meta,
                                              fastspec=fastspec, order=order,
                                              sorted_targetid=sorted_targetid,
                                              overwrite=args.overwrite)

    if args.link_randoms:
        print('WARNING: --link-randoms is deprecated; building matched randoms.', flush=True)
    random_mode = 'build' if args.link_randoms else args.random_mode
    if random_mode == 'build':
        build_random_catalogues(output_dir=args.output_dir,
                                lss_dir=args.lss_dir,
                                desivast_dir=args.desivast_dir,
                                nside=args.mask_nside,
                                n_random_files=args.n_random_files,
                                seed=args.random_seed,
                                chunk_rows=args.chunk_rows,
                                overwrite=args.overwrite)

    print(f'\nTOTAL selected: {counts['NGC'] + counts['SGC']:,}', flush=True)
    print(f'ASTRA input directory: {args.output_dir.resolve()}', flush=True)


if __name__ == '__main__':
    main()