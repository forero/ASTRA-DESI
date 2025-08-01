import numpy as np
from astropy.table import Table
from astropy.io import ascii
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
import pandas as pd
from itertools import combinations
import networkx as nx
from sklearn.decomposition import PCA
import matplotlib.cm as cm
import itertools
from scipy.spatial import cKDTree
from astropy.io import fits




def open_raw(zone, rand, tracer_type):
    filename = f'01_create_raw/zone_{zone}.fits.gz'

    with fits.open(filename) as hdul:
        data_fits = hdul[1].data
        columns = ['TRACERTYPE', 'RANDITER', 'TARGETID', 'X_CART', 'Y_CART', 'Z_CART']
        data_selected = {col: data_fits[col] for col in columns}

    data_zone = pd.DataFrame(data_selected)

    data_zone = data_zone[
    ((data_zone['RANDITER'] == rand) | (data_zone['RANDITER'] == -1)) &
    ((data_zone['TRACERTYPE'] == f'{tracer_type}_DATA') | (data_zone['TRACERTYPE'] == f'{tracer_type}_RAND') )
    ].reset_index(drop=True)

    data_zone = data_zone[['TRACERTYPE','TARGETID','RANDITER','X_CART', 'Y_CART', 'Z_CART']]

    return data_zone

def open_pairs(zone,data):

    ids = list(data['TARGETID'].values)

    filename = f'02_astra_classification/pairs/zone_{zone}_pairs.fits.gz'

    with fits.open(filename) as hdul:
        data_fits = hdul[1].data

    data_pairs = pd.DataFrame(data_fits)

    data_pairs = data_pairs[
        (data_pairs['TARGETID'].isin(ids))
    ].reset_index(drop=True)

    data_pairs = data_pairs[['TARGETID1','TARGETID2']]

    return data_pairs

def ids_webtype(zone,rand,data_raw, web_type):

    ids = set(data_raw['TARGETID'].tolist())

    filename = f'02_astra_classification/classified/zone_{zone}_classified.fits.gz'

    with fits.open(filename) as hdul:
        data_fits = hdul[1].data

    columns = ['TARGETID', 'RANDITER', 'NDATA', 'NRAND']
    data_selected = {col: data_fits[col] for col in columns}
    data_class = pd.DataFrame(data_selected)

    data_class = data_class[
    (data_class['TARGETID'].isin(ids)) & 
    (data_class['RANDITER'] == rand)
    ].reset_index(drop=True)

    data_class = data_class[['TARGETID','NDATA', 'NRAND']]
    data_webtype = parameter_r(data_class)
    data_webtype_filtered = data_webtype[['TARGETID','WEBTYPE']]

    data_webtype_filtered = data_webtype_filtered[data_webtype_filtered['WEBTYPE'] == web_type]

    ids_webtype = set(data_webtype_filtered['TARGETID'])

    return ids_webtype

def parameter_r(data):

    n_data = data['NDATA'].values
    n_rand = data['NRAND'].values
    data['r'] = (n_data - n_rand) / (n_data + n_rand)
    data_webtype = classification(data)

    return data_webtype

def classification(data):

    data.loc[(data['r'] >= -1.0) & (data['r'] <= -0.9), 'WEBTYPE'] = 'void'
    data.loc[(data['r'] >  -0.9) & (data['r'] <=  0.0), 'WEBTYPE'] = 'sheet'
    data.loc[(data['r'] >   0.0) & (data['r'] <=  0.9), 'WEBTYPE'] = 'filament'
    data.loc[(data['r'] >   0.9) & (data['r'] <=  1.0), 'WEBTYPE'] = 'knot'

    return data

def identify_fof_groups(df_raw, ids_webtype, df_pairs,type_data, linking_length=50):

    df_webtype = df_raw[df_raw['TARGETID'].isin(ids_webtype)].copy()

    connected_ids = set(df_pairs['TARGETID1']).union(set(df_pairs['TARGETID2']))

    df_connected_by_webtype = df_webtype[df_webtype['TARGETID'].isin(connected_ids)].copy()
    
    if type_data == 'data':
        df_fof = df_connected_by_webtype[df_connected_by_webtype['TRACERTYPE']==f'{tracer_type}_DATA']
    if type_data == 'rand':
        df_fof = df_connected_by_webtype[df_connected_by_webtype['TRACERTYPE']==f'{tracer_type}_RAND']

    #Aplicar Friends-of-Friends (FoF) con KDTree
    positions = df_fof[['X_CART', 'Y_CART', 'Z_CART']].values
    tree = cKDTree(positions)
    pairs = tree.query_pairs(r=linking_length)

    #Construir grupos (componentes conexas)
    parent = list(range(len(positions)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj

    for i, j in pairs:
        union(i, j)

    group_ids = np.array([find(i) for i in range(len(positions))])
    _, group_ids = np.unique(group_ids, return_inverse=True)
    df_connected_by_webtype['GROUPID'] = group_ids

    df_connected_by_webtype = df_connected_by_webtype[['TRACERTYPE','TARGETID','GROUPID','RANDITER']]

    df_connected_by_webtype['WEBTYPE'] = webtype

    return df_connected_by_webtype

n_zones = 20
n_rand = 100
webtype = 'filament'
data_type = 'data'
tracer_type = 'LRG'

for i in range(n_zones):
    for j in range(n_rand):

        #data_zone = data_zone[['TRACERTYPE','TARGETID','RANDITER','X_CART', 'Y_CART', 'Z_CART']]
        data_zone = open_raw(i,j,tracer_type)
        #data_pairs = data_pairs[['TARGETID1','TARGETID2']]
        data_pairs = open_pairs(i,data_zone)
        #List of id according to the webtype
        ids_classified = ids_webtype(i, j, data_zone, webtype)
        #df_connected_by_webtype = df_connected_by_webtype[['TRACERTYPE','TARGETID','GROUPID','RANDITER']]
        groups = identify_fof_groups(data_zone,ids_classified,data_pairs,data_type,linking_length=20)
        #Save file
        table = Table.from_pandas(groups)
        filename = f'03_groups/groups_fof/zone_{i}.fits.gz'
        table.write(filename, overwrite=True)

        del data_zone,data_pairs,ids_classified,groups

