"""Raw_data_script

Original file is located at
    https://colab.research.google.com/drive/1Rn7tw2LayYu6Ta9SCUc2t27aHYF1Q_Xo
"""

import os
import pandas as pd
import numpy as np
from astropy.io import fits
from astropy.coordinates import SkyCoord
from astropy.cosmology import Planck18
import astropy.units as u
from astropy.table import Table
from tqdm import tqdm
import time
import random 

base_url = "https://data.desi.lbl.gov/public/edr/vac/edr/lss/v2.0/LSScats/clustering/"
tracers = ['BGS_ANY', 'ELG', 'LRG', 'QSO']
real_suffix = {'N': '_N_clustering.dat.fits', 'S': '_S_clustering.dat.fits'}
random_suffix = {'N': '_N_{}_clustering.ran.fits', 'S': '_S_{}_clustering.ran.fits'}
n_random_files = 5 # Number of random files available per hemisphere at the url
n_random = 10
n_zones = 4
output_dir = "01_create_raw"
os.makedirs(output_dir, exist_ok=True)

# Rosettas per hemisphere
north_rosettas = {3, 6, 7, 11, 12, 13, 14, 15, 18, 19}
south_rosettas = {0, 1, 2, 4, 5, 8, 9, 10, 16, 17}

real_cache = {}    
random_cache = {}  

def load_fits_file_from_url(url, columns):

    while True:
        try:
            table = Table.read(url)
            break
        except Exception as e:
            wait = random.uniform(5, 10)
            print(f"❌ Error downloading {url}: {e}\nRetrying...\n")
            time.sleep(wait)

    cols_to_select = [col for col in columns if col in table.colnames]
    table = table[cols_to_select]
    if 'ROSETTE_NUMBER' in table.colnames:
        table.rename_column('ROSETTE_NUMBER', 'ZONE')
    return table.to_pandas()


def compute_cartesian(df):
    df = df.copy()
    comoving_distance = Planck18.comoving_distance(df['Z'].values)
    coords = SkyCoord(ra=df['RA'].values * u.deg, dec=df['DEC'].values * u.deg, distance=comoving_distance)
    df['XCART'] = coords.cartesian.x.value
    df['YCART'] = coords.cartesian.y.value
    df['ZCART'] = coords.cartesian.z.value
    return df

def get_hemisphere(zone):
    return 'N' if zone in north_rosettas else 'S'

def process_real(tracer, zone):
    hemi = get_hemisphere(zone)
    df = real_cache[tracer][hemi]
    df = df[df['ZONE'] == zone].copy()
    df = compute_cartesian(df)
    df['TRACERTYPE'] = np.full(len(df), tracer + "_DATA", dtype='S14')
    df['RANDITER'] = -1
    return df


def get_real_count(tracer, zone):
    hemi = get_hemisphere(zone)
    df = real_cache[tracer][hemi]
    return len(df[df['ZONE'] == zone])

def generate_randoms_for_zone(tracer, zone):
    hemi = get_hemisphere(zone)
    
    # Access the 18 preloaded random files for that tracer and hemisphere
    zone_random_dfs = [df[df['ZONE'] == zone] for df in random_cache[tracer][hemi].values()]

    # Amount of real points for this rosetta (zone)
    real_count = get_real_count(tracer, zone)
    all_randoms = []

    used_ran = set()
    ran_file_ids = list(range(len(zone_random_dfs)))

    for j in range(n_random):
        if len(used_ran) == len(ran_file_ids):
            used_ran = set()

        candidates = [i for i in ran_file_ids if i not in used_ran]
        ran_file = random.Random(j).choice(candidates)
        used_ran.add(ran_file)

        df = zone_random_dfs[ran_file]
        sample_df = df.sample(n=real_count, replace=False, random_state=j).reset_index(drop=True)
        sample_df = compute_cartesian(sample_df)
        sample_df['TRACERTYPE'] = np.full(len(sample_df), tracer + "_RAND", dtype='S14')
        sample_df['RANDITER'] = j
        all_randoms.append(sample_df)

        print(f"→ [Zone {zone} | Tracer {tracer}] random #{j+1:02d} uses file #{ran_file}")

    return pd.concat(all_randoms, ignore_index=True)

# --- MAIN EXECUTION ---

# --- PRELOAD: Cache all data and random files first ---

print("Preloading all real and random files...")

# Preload real data
for tracer in tracers:
    real_cache[tracer] = {}
    for hemi in ['N', 'S']:
        url = base_url + tracer + real_suffix[hemi]
        print(f"Downloading real data: {url}")
        df = load_fits_file_from_url(url, ['TARGETID', 'ROSETTE_NUMBER', 'RA', 'DEC', 'Z'])
        real_cache[tracer][hemi] = df

# Preload random files
for tracer in tracers:
    random_cache[tracer] = {}
    for hemi in ['N', 'S']:
        random_cache[tracer][hemi] = {}
        for i in range(n_random_files):
            url = base_url + tracer + f"_{hemi}_{i}_clustering.ran.fits"
            print(f"Downloading random file: {url}")
            df = load_fits_file_from_url(url, ['TARGETID', 'ROSETTE_NUMBER', 'RA', 'DEC', 'Z'])
            random_cache[tracer][hemi][i] = df


for zone in tqdm(range(n_zones), desc="Zones"):
    all_dfs = []
    for tracer in tracers:
        print(f"\nProcessing tracer: {tracer} | Zone: {zone}")
        real_df = process_real(tracer, zone)
        rand_df = generate_randoms_for_zone(tracer, zone)
        all_dfs.extend([real_df, rand_df])

    combined = pd.concat(all_dfs, ignore_index=True)
    combined = combined[['TARGETID', 'TRACERTYPE', 'RANDITER', 'RA', 'DEC', 'Z', 'XCART', 'YCART', 'ZCART']]
    combined['TRACERTYPE'] = combined['TRACERTYPE'].astype(str).values.astype('S14')

    output_path = os.path.join(output_dir, f"zone_{zone:02d}.fits.gz")
    fits.writeto(output_path, combined.to_records(index=False), overwrite=True)
    print(f"✅ Saved: {output_path}")




