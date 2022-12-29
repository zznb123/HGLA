from torch_sparse import spmm
import torch.nn as nn
import torch
import torch.fft
import torch.nn.functional as F




class SparseNGCNLayer(nn.Module):
    def __init__(self, in_channels, out_channels, iterations, dropout_rate, device):
        super(SparseNGCNLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.iterations = iterations
        self.dropout_rate = dropout_rate
        self.device = device

        self.bias = nn.Parameter(torch.Tensor(1, self.out_channels)).to(self.device)
        self.weight_matrix = nn.Parameter(torch.Tensor(self.in_channels, self.out_channels)).to(self.device)
        self.norm = nn.LayerNorm(out_channels)

        self.init_parameters()

    def init_parameters(self):
        nn.init.xavier_uniform_(self.weight_matrix)
        nn.init.xavier_uniform_(self.bias)

    def forward(self, normalized_adjacency_matrix, features):
        feature_count, _ = torch.max(features["indices"], dim=1)
        feature_count = feature_count + 1
        base_features = spmm(features["indices"], features["values"], feature_count[0], feature_count[1], self.weight_matrix)
        base_features = base_features + self.bias
        base_features = self.norm(base_features)
        base_features = F.dropout(base_features, p=self.dropout_rate, training=self.training)
        base_features = F.relu(base_features)

        for _ in range(self.iterations - 1):
            base_features = spmm(normalized_adjacency_matrix["indices"],
                                 normalized_adjacency_matrix["values"],
                                 base_features.shape[0],
                                 base_features.shape[0],
                                 base_features)
        return base_features

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_channels) + ' -> ' \
               + str(self.out_channels) + ')'


class DenseNGCNLayer(nn.Module):
    def __init__(self, in_channels, out_channels, iterations, dropout_rate, device):
        super(DenseNGCNLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.iterations = iterations
        self.dropout_rate = dropout_rate
        self.device = device

        self.bias = nn.Parameter(torch.Tensor(1, self.out_channels)).to(self.device)
        self.weight_matrix = nn.Parameter(torch.Tensor(self.in_channels, self.out_channels)).to(self.device)
        self.norm = nn.LayerNorm(out_channels)

        self.init_parameters()

    def init_parameters(self):
        nn.init.xavier_uniform_(self.weight_matrix)
        nn.init.xavier_uniform_(self.bias)

    def forward(self, normalized_adjacency_matrix, features):
        base_features = torch.mm(features, self.weight_matrix)
        base_features = F.dropout(base_features, p=self.dropout_rate, training=self.training)
        for _ in range(self.iterations - 1):
            base_features = spmm(normalized_adjacency_matrix["indices"],
                                 normalized_adjacency_matrix["values"],
                                 base_features.shape[0],
                                 base_features.shape[0],
                                 base_features)
        base_features = base_features + self.bias
        base_features = self.norm(base_features)
        return base_features

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_channels) + ' -> ' \
               + str(self.out_channels) + ')'


class Attention(nn.Module):
    def __init__(self, channel, reduction=4):
        super(Attention, self).__init__()
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.conv = nn.Sequential(nn.Linear(in_features=channel, out_features=channel // reduction, bias=False),
                                  nn.PReLU(),
                                  nn.Linear(in_features=channel // reduction, out_features=channel, bias=False),
                                  nn.Sigmoid())

    def forward(self, x):
        y = self.pool(x).squeeze(dim=-1)
        y = self.conv(y).unsqueeze(dim=-1)
        return x * y


class Depth(torch.nn.Module):
    def __init__(self, in_dim, hidden, channel):
        super(Depth, self).__init__()
        self.norm = nn.LayerNorm(in_dim)
        self.lstm = torch.nn.LSTM(in_dim, hidden, 1, bias=False)
        self.attention = Attention(channel)

    def forward(self, x, h, c):
        x = self.norm(x)
        x = self.attention(x)
        x, (h, c) = self.lstm(x, (h, c))
        return x, (h, c)

