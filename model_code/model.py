import torch
import torch.nn.functional as F
from layers_new import SparseNGCNLayer, DenseNGCNLayer, Depth
from torch_sparse import spmm


class HGLA(torch.nn.Module):
    def __init__(self, args, feature_number, class_number):
        super(HGLA, self).__init__()
        self.args = args
        self.feature_number = feature_number
        self.class_number = class_number
        self.dropout = self.args.dropout
        self.calculate_layer_sizes()
        self.setup_layer_structure()
        self.weight_matrix = torch.nn.Parameter(torch.ones(feature_number, 1), requires_grad=False).to(self.args.device)

    def calculate_layer_sizes(self):
        self.abstract_feature_number_1 = sum(self.args.layers_1)
        self.abstract_feature_number_2 = sum(self.args.layers_2)
        self.order_1 = len(self.args.layers_1)
        self.order_2 = len(self.args.layers_2)

    def setup_layer_structure(self):

        self.linear = torch.nn.Linear(32, 128)

        self.base_gcn = SparseNGCNLayer(self.feature_number, self.args.layers_1[0], 1, self.args.dropout, self.args.device)

        self.breaths_1 = torch.nn.ModuleList([SparseNGCNLayer(self.feature_number, self.args.layers_1[i - 1], i, self.args.dropout, self.args.device) for i in range(1, self.order_1 + 1)])

        self.breaths_2 = torch.nn.ModuleList([DenseNGCNLayer(self.abstract_feature_number_1, self.args.layers_2[i - 1], i, self.args.dropout, self.args.device) for i in range(1, self.order_2 + 1)])

       

        self.bilinear = torch.nn.Bilinear(128, 128, 64)

        self.decoder = torch.nn.Sequential(torch.nn.Linear(self.args.hidden1, self.args.hidden2),
                                           torch.nn.ELU(),
                                           torch.nn.Linear(self.args.hidden2, 1))

    def embed(self, normalized_adjacency_matrix, features):
        abstract_features_1 = torch.cat([self.breaths_1[i](normalized_adjacency_matrix, features) for i in range(self.order_1)], dim=1)
        abstract_features_1 = F.dropout(abstract_features_1, self.dropout, training=self.training)

        h_tmps = []
        for i, l in enumerate(self.breaths_2):
            h_tmps.append(self.breaths_2[i](normalized_adjacency_matrix, abstract_features_1))

        return h_tmps

    def forward(self, normalized_adjacency_matrix, features, idx):

        h_tmps = self.embed(normalized_adjacency_matrix, features)

        base_features = self.base_gcn(normalized_adjacency_matrix, features)
        h = torch.zeros(1, base_features.shape[0], self.args.lstm_hidden).to(self.args.device)
        c = torch.zeros(1, base_features.shape[0], self.args.lstm_hidden).to(self.args.device)
        base = base_features[None, :]
        en_features = []
        for i, l in enumerate(self.depths):
            in_cat = torch.cat((h_tmps[i][None, :], base), -1) #[* ,64]
            en_feature, (h, c) = self.depths[i](in_cat, h, c) # base [1,xx,32]
            en_features.append(en_feature)

        abstract_features_2 = torch.cat([en_features[i] for i in range(4)], dim=-1)
        abstract_features_2 = F.dropout(abstract_features_2, self.dropout, training=self.training)  # 多尺度连接特征聚合


        latent_features = abstract_features_2[0]
        feat_p1 = latent_features[idx[0]]
        feat_p2 = latent_features[idx[1]]
        feat = F.elu(self.bilinear(feat_p1, feat_p2))
        feat = F.dropout(feat, self.dropout, training=self.training)
        predictions = self.decoder(feat)

        return predictions, latent_features
