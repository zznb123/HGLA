import pickle

import pandas as pd
import numpy as np
import scipy.sparse as sp
import csv
#
# # d=pd.read_csv('dtifold1.emb', skiprows=1, header=None, sep=' ', ).sort_values(by=[0]).set_index([0])
# # print(d.head())
#
def normalize(mx):
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx
#
#
df_data = pd.read_csv('../DTI/fold1/train.csv')
df_drug_list = pd.read_csv('entity_list.csv')

idx = df_drug_list['Entity_ID'].tolist()
idx = np.array(idx)
idx_map = {j:i for i, j in enumerate(idx)}
# # print(idx_map)
# #
# # df_data_t = df_data[df_data.label == 1]
# # edges_unordered = df_data_t[['Drug_ID', 'Protein_ID']].values
# emb = pd.read_csv('dtifold1.csv', skiprows=1, header=None, sep=' ', error_bad_lines=False)
# for idx in emb[0]:
#     print(idx)
#     break
# new_index = [idx_map[idx] for idx in emb[0]]
# # print(idx_map.keys())
# #
# #
# # for id in emb[0]:
# #     for key in idx_map:
# #         if(id == key):
# #             print(idx_map[key])
#
# # print(idx_map)
# # print(emb.index.values)
# # new_index = [idx_map[idx] for idx in emb.index]
# # for idx in emb.index:
# #     print(idx)
# #     break
# # emb = emb.reindex(new_index)
# # print(emb.head())
# dtifold1 = pd.read_csv('../DTI/fold1/train.csv')
# print(dtifold1.head())
# print(idx_map[dtifold1.iloc[0].Drug_ID])
# # for i in np.setdiff1d(np.arange(7343), emb.index.values):
#     emb.loc[i] = (np.sum(emb.values, axis=0) / emb.values.shape[0])
# features = emb
#
# features = normalize(features)
# print(features)

# 2
# emb = pd.read_csv('dtifold1.csv', skiprows=1,header=None, sep=' ', error_bad_lines=False)#.sort_values(by=[0]).set_index([0])
# dtifold1 = []
# for i in range(7284):
#     dtifold1.append(emb[0][i])
#
# print(dtifold1)
# entity = pd.read_csv('entity_list.csv')
# print(entity['Entity_ID'])
# for id in dtifold1:
#     for idx in entity['Entity_ID']:
#         if id == idx:
#             id = idx.index
#             dtifold1.append(id)
# print(dtifold1)


# with open('entity_list.csv', 'r') as csvfile:
#     reader = csv.DictReader(csvfile)
#     column = [row['Entity_ID'] for row in reader]
#     nodes = tuple(column)
#     print(nodes)
# new_index = [idx_map[idx] for idx in emb.index]
# emb = emb.reindex(new_index)
#
# for i in np.setdiff1d(np.arange(7343), emb.index.values):
#     emb.loc[i] = (np.sum(emb.values, axis=0) / emb.values.shape[0])
# features = emb.sort_index().values
#
# features = normalize(features)

# emb = pd.read_csv('dti.emb', skiprows=1, header=None, sep=' ', error_bad_lines=False).sort_values(by=[0]).set_index([0])
# new_index = [idx_map[idx] for idx in emb.index]
# print(emb.head())

emb = pd.read_csv('dti.emb', skiprows=1, header=None, sep=' ', error_bad_lines=False).sort_values(by=[0]).set_index([0])
print(emb.head())
new_index = [idx_map[idx] for idx in emb.index]
print(new_index)
emb.index = new_index
print(emb.head())
print(emb.index.values)

for i in np.setdiff1d(np.arange(7343), emb.index.values):
    emb.loc[i] = (np.sum(emb.values, axis=0) / emb.values.shape[0])
features = emb.sort_index().values
# for i in range(7343):
#     for j in range(64):
#         print(features[i][j])
features = normalize(features)
print(features)
