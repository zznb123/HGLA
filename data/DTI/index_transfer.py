import pandas as pd
import numpy as np
fold1= pd.read_csv('../DTI/fold1/train.csv')

df_data = pd.read_csv('../DTI/fold1/train.csv')
df_data_t = df_data[df_data.label==1]
# print(df_data_t.head())
df_drug_list = pd.read_csv('entity_list.csv')
edges_unordered = df_data_t[['Drug_ID', 'Protein_ID']].values
print(edges_unordered)
idx = df_drug_list['Entity_ID'].tolist()
idx = np.array(idx)
idx_map = {j:i for i, j in enumerate(idx)}
edges = np.array(list(map(idx_map.get, edges_unordered.flatten()))).reshape(edges_unordered.shape)
print(edges)

# idx1 = idx_map[fold1.iloc[0].Drug_ID]
# idx2 = idx_map[fold1.iloc[0].Protein_ID]
# print(idx1)
# print(idx2)