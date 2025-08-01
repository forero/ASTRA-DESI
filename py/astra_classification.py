import numpy as np
import pandas as pd
from scipy.spatial import Delaunay
from itertools import combinations
from astropy.io import fits
import os

input_dir = "../01_create_raw/"        # Carpeta con los archivos zone_*.fits.gz
output_dir = "./"                    # Carpeta donde se guarda el zone_*.pairs.fits.gz
n_random = 10                       # Número de iteraciones random a usar

def load_dataframe_fix_endianness(path):
    with fits.open(path) as hdul:
        data = hdul[1].data
        df = pd.DataFrame({
            col: np.array(data[col]).astype(data[col].dtype.newbyteorder('='))
            for col in data.columns.names
        })
    return df

def generate_pairs_for_zone(df, n_random):
    output_rows = []

    tracers = df['TRACERTYPE'].apply(lambda x: x.split('_')[0]).unique()

    for tracer in tracers:
        df_tracer = df[df['TRACERTYPE'].str.startswith(tracer)]
        df_data = df_tracer[df_tracer['RANDITER'] == -1]

        for rand_iter in range(n_random):
            df_rand = df_tracer[df_tracer['RANDITER'] == rand_iter]

            df_iter = pd.concat([df_data, df_rand], ignore_index=True)
            coords = df_iter[['XCART', 'YCART', 'ZCART']].values
            targetids = df_iter['TARGETID'].values

            tri = Delaunay(coords)

            connected = set()
            for simplex in tri.simplices:
                for i, j in combinations(simplex, 2):
                    tid1, tid2 = int(targetids[i]), int(targetids[j])
                    connected.add(tuple(sorted((tid1, tid2))))

            for tid1, tid2 in connected:
                output_rows.append((tid1, tid2, rand_iter))

    return output_rows

def save_pairs_fits(rows, output_path):

    array = np.array(rows, dtype=[
        ('TARGETID1', 'i8'),
        ('TARGETID2', 'i8'),
        ('RANDITER', 'i4')
    ])
    hdu = fits.BinTableHDU(data=array)
    hdu.writeto(output_path, overwrite=True)
    print(f"Saved: {output_path}")

def main():
    for fname in sorted(os.listdir(input_dir)):
        if not fname.startswith("zone_") or not fname.endswith(".fits.gz"):
            continue

        input_path = os.path.join(input_dir, fname)
        zone_name = fname.replace(".fits.gz", "")
        output_path = os.path.join(output_dir, f"{zone_name}.pairs.fits.gz")

        print(f"\n[→] Processing {zone_name}")
        df = load_dataframe_fix_endianness(input_path)
        rows = generate_pairs_for_zone(df, n_random)
        save_pairs_fits(rows, output_path)

if __name__ == "__main__":
    main()
