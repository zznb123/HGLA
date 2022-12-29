import numpy as np
import scipy.sparse as sp
import torch
from torch.utils import data
import pandas as pd
from texttable import Texttable
import networkx as nx
import matplotlib.pyplot as plt
import random
from param_parser import parameter_parser


def draw(A, edge):
    length = len(A)
    G = nx.Graph()
    Edge = []
    node = range(length)
    for i in edge:
        Edge.append(tuple(i))
    G.add_nodes_from(node)
    G.add_edges_from(Edge)
    nx.draw_spring(G, font_size=1, width=1, node_size=50, node_color='#A0CBE2')
    plt.show()


def tab_printer(args):
    args = vars(args)
    keys = sorted(args.keys())
    t = Texttable()
    t.add_rows([["Parameter", "Value"]])
    t.add_rows([[k.replace("_", " ").capitalize(), args[k]] for k in keys])
    print(t.draw())


def normalize(mx):
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def normalize_adjacency_matrix(A, I):
    A_tilde = A + 2 * I
    degrees = A_tilde.sum(axis=0)[0].tolist()
    D = sp.diags(degrees, [0])

    D = D.power(-0.5)
    A_tilde_hat = D.dot(A_tilde).dot(D)
    return A_tilde_hat


def create_propagator_matrix(A, device):
    I = sp.eye(A.shape[0])
    A_tilde_hat = normalize_adjacency_matrix(A, I)
    propagator = dict()
    A_tilde_hat = sp.coo_matrix(A_tilde_hat)
    ind = np.concatenate([A_tilde_hat.row.reshape(-1, 1), A_tilde_hat.col.reshape(-1, 1)], axis=1)
    propagator["indices"] = torch.LongTensor(ind.T).to(device)
    propagator["values"] = torch.FloatTensor(A_tilde_hat.data).to(device)
    return propagator


def features_to_sparse(features, device):
    index_1, index_2 = features.nonzero()
    values = [1.0]*len(index_1)
    node_count = features.shape[0]
    feature_count = features.shape[1]
    features = sp.coo_matrix((values, (index_1, index_2)),
                                 shape=(node_count, feature_count),
                                 dtype=np.float32)
    out_features = dict()
    ind = np.concatenate([features.row.reshape(-1, 1), features.col.reshape(-1, 1)], axis=1)
    out_features["indices"] = torch.LongTensor(ind.T).to(device)
    out_features["values"] = torch.FloatTensor(features.data).to(device)
    out_features["dimensions"] = features.shape
    return out_features


class Data_DDI(data.Dataset):
    def __init__(self, idx_map, labels, df):
        self.labels = labels
        self.idx_map = idx_map
        self.df = df

    def __len__(self):
        return len(self.df)  # 训练集总数67919

    def __getitem__(self, index):
        # p = random.random()
        idx1 = self.idx_map[self.df.iloc[index].Drug1_ID]
        idx2 = self.idx_map[self.df.iloc[index].Drug2_ID]
        y = self.labels[index]
        '''
        if p > 0.5:
            k = idx1
            idx1 = idx2
            idx2 = k
        '''
        return y, (idx1, idx2)  


class Data_PPI(data.Dataset):
    # df : a list of data, which includes an index for the pair, an index for entity1 and entity2, from a list that combines all the entities. we want the
    def __init__(self, idx_map, labels, df):
        'Initialization'
        self.labels = labels
        self.idx_map = idx_map
        self.df = df

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.df)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        # Load data and get label
        idx1 = self.idx_map[self.df.iloc[index].Protein1_ID]
        idx2 = self.idx_map[self.df.iloc[index].Protein2_ID]
        y = self.labels[index]
        return y, (idx1, idx2)


class Data_DTI(data.Dataset):
    # df : a list of data, which includes an index for the pair, an index for entity1 and entity2, from a list that combines all the entities. we want the
    def __init__(self, idx_map, labels, df):
        'Initialization'
        self.labels = labels
        self.idx_map = idx_map
        self.df = df

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.df)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        # Load data and get label
        idx1 = self.idx_map[self.df.iloc[index].Drug_ID]
        idx2 = self.idx_map[self.df.iloc[index].Protein_ID]
        y = self.labels[index]
        return y, (idx1, idx2)


class Data_GDI(data.Dataset):
    # df : a list of data, which includes an index for the pair, an index for entity1 and entity2, from a list that combines all the entities. we want the
    def __init__(self, idx_map, labels, df):
        'Initialization'
        self.labels = labels
        self.idx_map = idx_map
        self.df = df

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.df)

    def __getitem__(self, index):
        idx1 = self.idx_map[str(self.df.iloc[index].Gene_ID)]
        idx2 = self.idx_map[self.df.iloc[index].Disease_ID]
        y = self.labels[index]
        return y, (idx1, idx2)


def load_data_link_prediction_DDI(path, network_type, inp, device):
    print('Loading DDI dataset...')
    path_up = f'./data/{network_type}'
    df_data = pd.read_csv(path + '/train.csv')
    df_drug_list = pd.read_csv(path_up + '/ddi_unique_smiles.csv')

    idx = df_drug_list['Drug1_ID'].tolist()
    idx = np.array(idx)
    idx_map = {j: i for i, j in enumerate(idx)}  

    df_data_t = df_data[df_data.label == 1]
    edges_unordered = df_data_t[['Drug1_ID', 'Drug2_ID']].values 

    if inp == 'node2vec':
        emb = pd.read_csv(path + '/ddi.emb', skiprows=1, header=None, sep=' ').sort_values(by=[0]).set_index([0])
        new_index = [idx_map[idx] for idx in emb.index]
        emb.index = new_index

        for i in np.setdiff1d(np.arange(1514), emb.index.values):
            emb.loc[i] = (np.sum(emb.values, axis=0) / emb.values.shape[0])

        features = emb.sort_index().values
        features = normalize(features)
    elif inp == 'one_hot':
        features = np.eye(1514)

    features = features_to_sparse(features, device)  
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten()))).reshape(edges_unordered.shape)  
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(len(idx), len(idx)),
                        dtype=np.float32) 
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)  
    adj = create_propagator_matrix(adj, device)  
    return adj, features, idx_map, edges  

def load_data_link_prediction_PPI(path, network_type, inp, device):
    print('Loading PPI dataset...')
    path_up = f'./data/{network_type}'
    df_data = pd.read_csv(path + '/train.csv')
    df_drug_list = pd.read_csv(path_up + '/protein_list.csv')

    idx = df_drug_list['Protein1_ID'].tolist()
    idx = np.array(idx)
    idx_map = {j: i for i, j in enumerate(idx)}

    df_data_t = df_data[df_data.label == 1]
    edges_unordered = df_data_t[['Protein1_ID', 'Protein2_ID']].values

    if inp == 'node2vec':
        emb = pd.read_csv(path + '/ppi.emb', skiprows=1, header=None, sep=' ').sort_values(by=[0]).set_index([0])
        new_index = [idx_map[idx] for idx in emb.index]
        emb.index = new_index

        for i in np.setdiff1d(np.arange(5604), emb.index.values):
            emb.loc[i] = (np.sum(emb.values, axis=0) / emb.values.shape[0])
        features = emb.sort_index().values
        features = normalize(features)
    elif inp == 'one_hot':
        features = np.eye(5604)

    features = features_to_sparse(features, device)
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten()))).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(len(idx), len(idx)),
                        dtype=np.float32)

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj = create_propagator_matrix(adj, device)

    return adj, features, idx_map, edges


def load_data_link_prediction_DTI(path, network_type, inp, device):
    print('Loading DTI dataset...')
    path_up = f'./data/{network_type}'
    df_data = pd.read_csv(path + '/train.csv')
    df_drug_list = pd.read_csv(path_up + '/entity_list.csv')

    idx = df_drug_list['Entity_ID'].tolist()
    idx = np.array(idx)
    idx_map = {j: i for i, j in enumerate(idx)}

    df_data_t = df_data[df_data.label == 1]
    edges_unordered = df_data_t[['Drug_ID', 'Protein_ID']].values

    if inp == 'node2vec':
        emb = pd.read_csv(path_up + '/dti.emb', skiprows=1, header=None, sep=' ').sort_values(by=[0]).set_index([0])
        new_index = [idx_map[idx] for idx in emb.index]
        emb.index = new_index

        for i in np.setdiff1d(np.arange(7343), emb.index.values):
            emb.loc[i] = (np.sum(emb.values, axis=0) / emb.values.shape[0])
        features = emb.sort_index().values

        features = normalize(features)
    elif inp == 'one_hot':
        features = np.eye(7343)

    features = features_to_sparse(features, device)

    edges = np.array(list(map(idx_map.get, edges_unordered.flatten()))).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(len(idx), len(idx)),
                        dtype=np.float32)

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj = create_propagator_matrix(adj, device)
    return adj, features, idx_map, edges


def load_data_link_prediction_GDI(path, network_type, inp, device):
    print('Loading GDI dataset...')
    path_up = f'./data/{network_type}'
    df_data = pd.read_csv(path + '/train.csv')
    df_drug_list = pd.read_csv(path_up + '/entity_list.csv')
    idx = df_drug_list['Entity_ID'].tolist()
    idx = np.array(idx)
    idx_map = {j: i for i, j in enumerate(idx)}

    df_data_t = df_data[df_data.label == 1]
    df_data_t['Gene_ID'] = df_data_t['Gene_ID'].apply(str)
    edges_unordered = df_data_t[['Gene_ID', 'Disease_ID']].values

    if inp == 'node2vec':
        emb = pd.read_csv(path + '/gdi.emb', skiprows=1, header=None, sep=' ').sort_values(by=[0]).set_index([0])
        new_index = [idx_map[idx] for idx in emb.index]
        emb.index = new_index

        for i in np.setdiff1d(np.arange(19783), emb.index.values):
            emb.loc[i] = (np.sum(emb.values, axis=0) / emb.values.shape[0])
        features = emb.sort_index().values
        features = normalize(features)
    elif inp == 'one_hot':
        features = np.eye(19783)

    features = features_to_sparse(features, device)

    edges = np.array(list(map(idx_map.get, edges_unordered.flatten()))).reshape(edges_unordered.shape)

    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(len(idx), len(idx)),
                        dtype=np.float32)

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj = create_propagator_matrix(adj, device)
    return adj, features, idx_map, edges


def load_data(args):
    data_path = f"./data/{args.network_type}/fold{args.fold_id}"
    if args.ratio:
        data_path = f"./data/{args.network_type}/{args.train_percent}/fold{args.fold_id}"

    if args.network_type == 'DDI':
        adj, features, idx_map, edges = load_data_link_prediction_DDI(data_path, args.network_type, args.input_type, args.device)
        Data_class = Data_DDI
    elif args.network_type == 'PPI':
        adj, features, idx_map, edges = load_data_link_prediction_PPI(data_path, args.network_type, args.input_type, args.device)
        Data_class = Data_PPI
    elif args.network_type == 'DTI':
        adj, features, idx_map, edges = load_data_link_prediction_DTI(data_path, args.network_type, args.input_type, args.device)
        Data_class = Data_DTI
    elif args.network_type == 'GDI':
        adj, features, idx_map, edges = load_data_link_prediction_GDI(data_path, args.network_type, args.input_type, args.device)
        Data_class = Data_GDI

    return adj, features, idx_map, Data_class

