import numpy as np
import pandas as pd
from scipy.spatial import Delaunay
from itertools import combinations
from astropy.io import fits
import os

input_dir = "./01_create_raw/"                           # Carpeta con los archivos zone_*.fits.gz
output_dir = "./02_astra_classification/"                # Carpeta donde se guarda el zone_*.pairs.fits.gz
n_random = 10                                            # Número de iteraciones random a usar

def load_dataframe(path):
    with fits.open(path) as hdul:
        data = hdul[1].data
        df = pd.DataFrame({
            col: np.array(data[col]).astype(data[col].dtype.newbyteorder('='))
            for col in data.columns.names
        })
    return df

def generate_pairs_and_classification(df, n_random):
    pair_rows = []
    class_rows = []

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

            neighbors = {i: set() for i in range(len(coords))}
            for simplex in tri.simplices:
                for i, j in combinations(simplex, 2):
                    # Vecinos de cada punto (por índice)
                    neighbors[i].add(j)
                    neighbors[j].add(i)

                    # IDs de los puntos conectados (para guardar como par)
                    tid1, tid2 = targetids[i], targetids[j]
                    pair_rows.append((tid1, tid2, rand_iter))

            # Clasificación por punto real
            for i, nbrs in neighbors.items():
                tid = targetids[i]
                ndata = np.sum(is_data[list(nbrs)])
                nrand = len(nbrs) - ndata

                is_data_flag = bool(is_data[i])  # True si es real, False si es random
                class_rows.append((tid, rand_iter, is_data_flag, ndata, nrand))


    return pair_rows, class_rows

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

def main():
    os.makedirs(output_dir, exist_ok=True)

    for fname in sorted(os.listdir(input_dir)):

        input_path = os.path.join(input_dir, fname)
        zone_name = fname.replace(".fits.gz", "")
        output_pairs = os.path.join(output_dir, f"{zone_name}.pairs.fits.gz")
        output_class = os.path.join(output_dir, f"{zone_name}.class.fits.gz")

        print(f"\nProcessing {zone_name}")
        df = load_dataframe(input_path)
        pair_rows, class_rows = generate_pairs_and_classification(df, n_random)
        save_pairs_fits(pair_rows, output_pairs)
        save_classification_fits(class_rows, output_class)

if __name__ == "__main__":
    main()
