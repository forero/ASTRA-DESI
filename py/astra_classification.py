import numpy as np
import pandas as pd
from scipy.spatial import Delaunay
from itertools import combinations
from astropy.io import fits
import os
from collections import defaultdict

input_dir = "./01_create_raw/"                           # Folder with zone_*.fits.gz files
output_dir = "./02_astra_classification/"                # Folder where the files are saved
n_random = 10                                            # Number of random iterations to use


def load_dataframe(path):
    with fits.open(path) as hdul:
        data = hdul[1].data
        df = pd.DataFrame({
            col: np.array(data[col]).astype(data[col].dtype.newbyteorder('='))
            for col in data.columns.names
        })
    return df

def generate_pairs_classification_probability(df, n_random):
    pair_rows = []
    class_rows = []
    r_by_tid = {}

    tracers = df['TRACERTYPE'].apply(lambda x: x.split('_')[0]).unique()

    for tracer in tracers:
        df_tracer = df[df['TRACERTYPE'].str.startswith(tracer)]

        for rand_iter in range(n_random):
            mask = (df_tracer['RANDITER'] == -1) | (df_tracer['RANDITER'] == rand_iter)
            df_iter = df_tracer[mask]

            coords = df_iter[['XCART', 'YCART', 'ZCART']].values
            targetids = df_iter['TARGETID'].values
            is_data = (df_iter['RANDITER'] == -1).values

            tri = Delaunay(coords)

            # 1. For the pairs file
            neighbors = {i: set() for i in range(len(coords))}
            for simplex in tri.simplices:
                for i, j in combinations(simplex, 2):
                    # We save connections by index
                    neighbors[i].add(j)
                    neighbors[j].add(i)

                    # We save connections by TARGETID
                    tid1, tid2 = targetids[i], targetids[j]
                    pair_rows.append((tid1, tid2, rand_iter))

            # 2. For the class file
            for i, nbrs in neighbors.items():
                tid = targetids[i]
                ndata = np.sum(is_data[list(nbrs)])
                nrand = len(nbrs) - ndata

                is_data_flag = bool(is_data[i])  # True si es real, False si es random
                class_rows.append((tid, rand_iter, is_data_flag, ndata, nrand))

                # 3. For the probability file
                if is_data_flag and (ndata + nrand) > 0:
                    r = (ndata - nrand) / (ndata + nrand)
                    if tid not in r_by_tid:
                        r_by_tid[tid] = []
                    r_by_tid[tid].append(r)


    return pair_rows, class_rows, r_by_tid

def save_pairs_fits(rows, output_path):

    array = np.array(rows, dtype=[
        ('TARGETID1', 'i8'),
        ('TARGETID2', 'i8'),
        ('RANDITER', 'i4')
    ])
    hdu = fits.BinTableHDU(data=array)
    hdu.writeto(output_path, overwrite=True)
    print(f"Saved: {output_path}")

def save_classification_fits(rows, output_path):
    array = np.array(rows, dtype=[
        ('TARGETID', 'i8'),
        ('RANDITER', 'i4'),
        ('ISDATA', 'bool'),        
        ('NDATA', 'i4'),
        ('NRAND', 'i4')
    ])
    hdu = fits.BinTableHDU(data=array)
    hdu.writeto(output_path, overwrite=True)
    print(f"Saved: {output_path}")

def classify_type(r):  
    if -1.0 <= r <= -0.9:
        return 'void'
    elif -0.9 < r <= 0.0:
        return 'sheet'
    elif 0.0 < r <= 0.9:
        return 'filament'
    elif 0.9 < r <= 1.0:
        return 'knot'

def save_probability_fits(r_by_tid, output_path):
    rows = []
    for tid, r_list in r_by_tid.items():
        total = len(r_list)
        counts = {'void': 0, 'sheet': 0, 'filament': 0, 'knot': 0}
        for r in r_list:
            counts[classify_type(r)] += 1
        rows.append((
            tid,
            counts['void'] / total,
            counts['sheet'] / total,
            counts['filament'] / total,
            counts['knot'] / total
        ))

    array = np.array(rows, dtype=[
        ('TARGETID', 'i8'),
        ('PVOID', 'f4'),
        ('PSHEET', 'f4'),
        ('PFILAMENT', 'f4'),
        ('PKNOT', 'f4'),
    ])
    hdu = fits.BinTableHDU(data=array)
    hdu.writeto(output_path, overwrite=True)
    print(f"Saved: {output_path}")


def main():
    os.makedirs(output_dir, exist_ok=True)

    for fname in sorted(os.listdir(input_dir)):

        input_path = os.path.join(input_dir, fname)
        zone_name = fname.replace(".fits.gz", "")
        output_pairs = os.path.join(output_dir, f"{zone_name}_pairs.fits.gz")
        output_class = os.path.join(output_dir, f"{zone_name}_class.fits.gz")
        output_prob = os.path.join(output_dir, f"{zone_name}_probability.fits.gz")

        print(f"\nProcessing {zone_name}")
        df = load_dataframe(input_path)
        pair_rows, class_rows, r_by_tid = generate_pairs_classification_probability(df, n_random)
        save_pairs_fits(pair_rows, output_pairs)
        save_classification_fits(class_rows, output_class)
        save_probability_fits(r_by_tid, output_prob)

if __name__ == "__main__":
    main()
