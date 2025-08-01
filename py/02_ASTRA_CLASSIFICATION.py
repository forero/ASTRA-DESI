import numpy as np
import pandas as pd
from scipy.spatial import Delaunay
from itertools import combinations
from astropy.io import fits
import os
from collections import defaultdict

def load_data_from_fits(path):
    with fits.open(path) as hdul:
        data = hdul[1].data
        return pd.DataFrame({col: data[col] for col in data.names})

def classify_type(r):
    if -1.0 <= r <= -0.9:
        return 'void'
    elif -0.9 < r <= 0.0:
        return 'sheet'
    elif 0.0 < r <= 0.9:
        return 'filament'
    elif 0.9 < r <= 1.0:
        return 'knot'
    else:
        return 'error'

type_to_index = {'void': 0, 'sheet': 1, 'filament': 2, 'knot': 3}

def process_file(input_path, output_pairs_path, output_class_path, output_prob_path):
    df = load_data_from_fits(input_path)
    tracers = df['TRACERTYPE'].apply(lambda x: x.split('_')[0]).unique()

    pair_rows = []
    class_rows = []
    prob_counts = defaultdict(lambda: np.zeros(4, dtype=int))  # key: TARGETID, value: [void, sheet, filament, knot]
    real_targetids_set = set()

    for tracer in tracers:
        df_tracer = df[df['TRACERTYPE'].str.startswith(tracer)]
        df_data = df_tracer[df_tracer['RANDITER'] == -1]

        for rand_iter in range(100):
            df_rand = df_tracer[df_tracer['RANDITER'] == rand_iter]
            if df_rand.empty:
                continue

            df_iter = pd.concat([df_data, df_rand], ignore_index=True)
            coords = df_iter[['XCART', 'YCART', 'ZCART']].values
            targetids = df_iter['TARGETID'].values
            is_data = (df_iter['RANDITER'] == -1).values

            try:
                tri = Delaunay(coords)
            except Exception as e:
                print(f"[!] Error triangulating {os.path.basename(input_path)} | {tracer} | Iter {rand_iter}")
                print("    ", str(e))
                continue

            # PARES
            neighbors = {i: set() for i in range(len(coords))}
            for simplex in tri.simplices:
                for i, j in combinations(simplex, 2):
                    neighbors[i].add(j)
                    neighbors[j].add(i)
                    tid1, tid2 = int(targetids[i]), int(targetids[j])
                    if tid1 != tid2:
                        pair_rows.append((tid1, tid2, rand_iter))

            # CLASIFICACIÓN
            for i, nbrs in neighbors.items():
                tid = int(targetids[i])
                ndata = int(np.sum(is_data[list(nbrs)]))
                nrand = len(nbrs) - ndata
                class_rows.append((tid, rand_iter, ndata, nrand))

                if is_data[i]:  # punto real → contar tipo para PROB
                    total_neighbors = ndata + nrand
                    if total_neighbors == 0:
                        continue
                    r_value = (ndata - nrand) / total_neighbors
                    t = classify_type(r_value)
                    if t in type_to_index:
                        prob_counts[tid][type_to_index[t]] += 1
                        real_targetids_set.add(tid)

    # GUARDAR .PAIRS
    if pair_rows:
        array_pairs = np.array(pair_rows, dtype=[
            ('TARGETID1', 'i8'),
            ('TARGETID2', 'i8'),
            ('RANDITER', 'i4')
        ])
        hdu_pairs = fits.BinTableHDU(data=array_pairs)
        hdu_pairs.writeto(output_pairs_path, overwrite=True)
        print(f"[✓] Pairs saved: {output_pairs_path}")
    else:
        print(f"[!] No pairs for {input_path}")

    # GUARDAR .CLASS
    if class_rows:
        array_class = np.array(class_rows, dtype=[
            ('TARGETID', 'i8'),
            ('RANDITER', 'i4'),
            ('NDATA', 'i4'),
            ('NRAND', 'i4')
        ])
        hdu_class = fits.BinTableHDU(data=array_class)
        hdu_class.writeto(output_class_path, overwrite=True)
        print(f"[✓] Class saved: {output_class_path}")
    else:
        print(f"[!] No class data for {input_path}")

    # GUARDAR .PROB
    prob_rows = []
    for tid in sorted(real_targetids_set):
        counts = prob_counts[tid]
        total = counts.sum()
        if total == 0:
            probs = [0.0, 0.0, 0.0, 0.0]
        else:
            probs = counts / total
        prob_rows.append((tid, *probs))

    if prob_rows:
        array_prob = np.array(prob_rows, dtype=[
            ('TARGETID', 'i8'),
            ('PVOID', 'f4'),
            ('PSHEET', 'f4'),
            ('PFILAMENT', 'f4'),
            ('PKNOT', 'f4')
        ])
        hdu_prob = fits.BinTableHDU(data=array_prob)
        hdu_prob.writeto(output_prob_path, overwrite=True)
        print(f"[✓] Prob saved: {output_prob_path}")
    else:
        print(f"[!] No probability data for {input_path}")

def main():
    input_dir = "./zones"
    output_pairs_dir = "./pairs"
    output_class_dir = "./class"
    output_prob_dir = "./prob"

    os.makedirs(output_pairs_dir, exist_ok=True)
    os.makedirs(output_class_dir, exist_ok=True)
    os.makedirs(output_prob_dir, exist_ok=True)

    for fname in sorted(os.listdir(input_dir)):
        if fname.endswith(".fits.gz") and fname.startswith("zone_"):
            base = fname.replace(".fits.gz", "")
            zone_path = os.path.join(input_dir, fname)
            output_pairs = os.path.join(output_pairs_dir, f"{base}.PAIRS.fits.gz")
            output_class = os.path.join(output_class_dir, f"{base}.CLASS.fits.gz")
            output_prob = os.path.join(output_prob_dir, f"{base}.PROB.fits.gz")
            print(f"\n[→] Processing {fname}")
            process_file(zone_path, output_pairs, output_class, output_prob)

if __name__ == "__main__":
    main()
