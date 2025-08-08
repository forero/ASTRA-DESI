import numpy as np
from astropy.table import Table
import pandas as pd
from scipy.spatial import cKDTree
from astropy.io import fits
import time
import os
import gzip
import shutil

start_total = time.time()

base_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(base_dir, "03_groups")
os.makedirs(output_dir, exist_ok=True)

def open_raw(zone=1, randiter=1, tracer_type='filament'):
    columns = ['TRACERTYPE', 'RANDITER', 'TARGETID', 'XCART', 'YCART', 'ZCART']
    filename = f'01_create_raw/zone_{zone:02d}.fits.gz'

    with fits.open(filename, memmap=True) as hdul:
        data_fits = hdul[1].data

    df = pd.DataFrame({
            name: np.array(data_fits[name]).astype(data_fits[name].dtype.newbyteorder('='))
            if data_fits[name].dtype.byteorder not in ('=', '|') else data_fits[name]
            for name in columns
        })

    mask = (
        ((df['RANDITER'] == randiter) | (df['RANDITER'] == -1)) &
        ((df['TRACERTYPE'] == f'{tracer_type}_DATA') | (df['TRACERTYPE'] == f'{tracer_type}_RAND'))
    )
    data_zone = df[mask]

    return data_zone


def open_webtype(data_zone, zone=1, randiter=1):
    ids = data_zone['TARGETID'].tolist()

    columns = ['TARGETID', 'RANDITER', 'NDATA', 'NRAND','ISDATA']
    filename = f'02_astra_classification/zone_{zone:02d}_class.fits.gz'

    with fits.open(filename, memmap=True) as hdul:
        data_fits = hdul[1].data

    data_class = pd.DataFrame({
            name: np.array(data_fits[name]).astype(data_fits[name].dtype.newbyteorder('='))
            if data_fits[name].dtype.byteorder not in ('=', '|') else data_fits[name]
            for name in columns
        })

    data_class = data_class[data_class['RANDITER'] == randiter].reset_index(drop=True)

    data_class = data_class[['TARGETID','RANDITER','NDATA', 'NRAND','ISDATA']]

    return data_class

def ids_webtype(data_class,limit, webtype = 'filament'):

    data_webtype = parameter_r(data_class,limit)
    data_webtype_filtered = data_webtype[['TARGETID','WEBTYPE']]
    data_webtype_filtered = data_webtype_filtered[data_webtype_filtered['WEBTYPE'] == webtype]

    ids_webtype = list(data_webtype_filtered['TARGETID'])

    return ids_webtype

def parameter_r(data_class,limit):

    n_data = data_class['NDATA'].values
    n_rand = data_class['NRAND'].values
    data_class['r'] = (n_data - n_rand) / (n_data + n_rand)
    data_webtype = classification(data_class,limit)

    return data_webtype

def classification(data_class,limit):

    data_class.loc[(data_class['r'] >= -1.0) & (data_class['r'] <= -(limit)), 'WEBTYPE'] = 'void'
    data_class.loc[(data_class['r'] >  -(limit)) & (data_class['r'] <=  0.0), 'WEBTYPE'] = 'sheet'
    data_class.loc[(data_class['r'] >   0.0) & (data_class['r'] <=  (limit)), 'WEBTYPE'] = 'filament'
    data_class.loc[(data_class['r'] >   (limit)) & (data_class['r'] <=  1.0), 'WEBTYPE'] = 'knot'

    return data_class

def identify_fof_groups(df_raw, ids_webtype, type_data, tracer_type, webtype, linking_length=50,randiter=1):
    
    df_webtype = df_raw[df_raw['TARGETID'].isin(ids_webtype)].copy()

    if type_data == 'data':
        df_fof = df_webtype[df_webtype['TRACERTYPE']==f'{tracer_type}_DATA']
    if type_data == 'rand':
        df_fof = df_webtype[df_webtype['TRACERTYPE']==f'{tracer_type}_RAND']
    

    #Aplicar Friends-of-Friends (FoF) con KDTree
    positions = df_fof[['XCART', 'YCART', 'ZCART']].values
    tree = cKDTree(positions)
    pairs = tree.query_pairs(r=linking_length)

    #Construir grupos (componentes conexas)
    parent = list(range(len(positions)))

    #Representante del conjunto
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    #Union de grupos
    def union(i, j):
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj

    for i, j in pairs:
        union(i, j)

    group_ids = np.array([find(i) for i in range(len(positions))])
    _, group_ids = np.unique(group_ids, return_inverse=True)
    df_fof['GROUPID'] = group_ids

    df_fof = df_fof[['TRACERTYPE','TARGETID','GROUPID','RANDITER','XCART', 'YCART', 'ZCART']]

    df_fof['WEBTYPE'] = webtype
    df_fof['RANDITER'] = randiter

    return df_fof

def inertia_tensor(df_fof):

    df_fof = df_fof.copy()

    x, y , z = df_fof['XCART'].values, df_fof['YCART'].values, df_fof['ZCART'].values
    xcm, ycm, zcm = np.mean(x),np.mean(y),np.mean(z)
    df_fof['XCM'], df_fof['YCM'], df_fof['ZCM'] = xcm, ycm, zcm

    x = x - xcm
    y = y - ycm
    z = z - zcm

    if len(df_fof) < 2:
        df_fof[['A', 'B', 'C']] = np.nan
        return df_fof


    r = np.sqrt(x**2 + y**2 + z**2)
    I = np.ones((3,3))
    
    I[0,0] = np.sum(r**2 - x*x)
    I[1,1] = np.sum(r**2 - y*y)
    I[2,2] = np.sum(r**2 - z*z)
    
    I[0,1] = -np.sum(x*y)
    I[1,0] = I[0,1]
    
    I[0,2] = -np.sum(x*z)
    I[2,0] = I[0,2]
    
    I[1,2] = -np.sum(y*z)
    I[2,1] = I[1,2]
    
    values, vectors = np.linalg.eig(I)
    ii = np.argsort(-values)
    
    sqrt_values = values[ii]
    #vectors_final = vectors[:,ii]

    values[values < 0] = 0  # O usa np.clip

    a, b, c = np.sqrt(values[ii])

    df_fof['A'] = a
    df_fof['B'] = b
    df_fof['C'] = c
    
    return df_fof 

n_zones = 2
n_rand = 10
web_type = ['filament']
data_type = ['data']
tracer_type = ['BGS_ANY', 'ELG', 'LRG', 'QSO']
linking_length = [10,30,50,70]
limit_classification_r = 0.9

for length in linking_length:
    for wtype, dtype in zip(web_type, data_type):
        for i in range(n_zones):

            start_zone = time.time()
            print(f'Webtype: {wtype}, Zone: {i}')
            
            dfs_groups = []

            for tracer in tracer_type:
                print(f'Tracer: {tracer}')
                for j in range(n_rand):
                    data_zone_raw = open_raw(zone=i, randiter=j, tracer_type=tracer)
                    #print(data_zone_raw)
                    data_webtype = open_webtype(data_zone_raw, zone=i, randiter=j)
                    #print(data_webtype)
                    ids_classified = ids_webtype(data_webtype, limit=limit_classification_r, webtype=wtype)
                    #print(ids_classified)
                    groups = identify_fof_groups(data_zone_raw, ids_classified, dtype,
                                                tracer, webtype=wtype, linking_length=length, randiter=j)

                    df_result = (
                        groups.groupby('GROUPID', group_keys=False)
                        .apply(inertia_tensor)
                        .reset_index(drop=True)
                    )

                    if df_result.empty:
                        print("DataFrame vacío. Continuando...")
                    else: 
                        df_inertia = df_result[['TRACERTYPE', 'TARGETID', 'GROUPID', 'RANDITER', 'WEBTYPE',
                                            'XCM', 'YCM', 'ZCM', 'A', 'B', 'C']]

                    dfs_groups.append(df_inertia)
            if dfs_groups:
                df_final = pd.concat(dfs_groups, ignore_index=True)
                table_connection = Table.from_pandas(df_final)

                uncompressed_file = f"03_groups/zone_{i:02d}_groups_fof_{wtype}_{length}.fits"
                compressed_file = uncompressed_file + ".gz"
                table_connection.write(uncompressed_file, format='fits', overwrite=True)

                with open(uncompressed_file, 'rb') as f_in:
                    with gzip.open(compressed_file, 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)

                os.remove(uncompressed_file)

            elapsed_zone = (time.time() - start_zone)/60
            print(f'Zona {i}: {elapsed_zone:.2f} minutes')

    total_elapsed = (time.time() - start_total)/60
    print(f'Total time:{total_elapsed:.2f} minutes')
