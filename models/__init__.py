import numpy as np
import torch
import torch.nn as nn
from typing import Optional


Tensor = torch.FloatTensor


class MergeLayer(nn.Module):
    def __init__(self, dim1, dim2, dim3, dim4):
        super().__init__()
        self.fc1 = nn.Linear(dim1 + dim2, dim3)
        self.fc2 = nn.Linear(dim3, dim4)
        self.activation = nn.ReLU()
        self.initialise_weights()

    def initialise_weights(self):
        nn.init.xavier_normal_(self.fc1.weight)
        nn.init.xavier_normal_(self.fc2.weight)

    def forward(self, x1, x2):
        x = torch.cat([x1, x2], dim=1)
        h = self.activation(self.fc1(x))
        return self.fc2(h)


class TemporalAttentionLayer(nn.Module):
    """
    Layer to compute the temporal embedding of a node given the node, its immediate neighbours and edge timestamps.
    """

    def __init__(
            self,
            n_node_feat: int,
            n_neighbors_feat: int,
            n_edge_feat: int,
            time_dim: int,
            output_dim: int,
            attention_heads: Optional[int] = 2,
            dropout: Optional[float] = 0.1
    ):
        super().__init__()

        self.attention_heads = attention_heads

        self.feat_dim = n_node_feat
        self.time_dim = time_dim
        self.query_dim = n_node_feat + time_dim
        self.key_dim = n_neighbors_feat + time_dim + n_edge_feat

        self.merger = MergeLayer(
            dim1=self.query_dim,
            dim2=n_node_feat,
            dim3=n_node_feat,
            dim4=output_dim,
        )

        self.multi_head_target = nn.MultiheadAttention(
            embed_dim=self.query_dim,
            kdim=self.key_dim,
            vdim=self.key_dim,
            num_heads=attention_heads,
            dropout=dropout,
        )

    def forward(
            self,
            src_node_feat: Tensor,
            src_time_feat: Tensor,
            neighbors_feat: Tensor,
            neighbors_time_feat: Tensor,
            edge_feat: Tensor,
            neighbors_padding: Tensor,
    ):
        """
        :param src_node_feat: Tensor of shape (batch_size, n_node_feat)
        :param src_time_feat: Tensor of shape (batch_size, 1, time_dim)
        :param neighbors_feat: Tensor of shape (batch_size, n_neighbors, n_node_feat)
        :param neighbors_time_feat: Tensor of shape (batch_size, n_neighbors, time_dim)
        :param edge_feat: Tensor of shape (batch_size, n_neighbors, n_edge_feat)
        :param neighbors_padding: Tensor of shape (batch_size, n_neighbors)
        :return:
        """

        unsqueezed_feat = torch.unsqueeze(src_node_feat, dim=1)
        query = torch.cat([unsqueezed_feat, src_time_feat], dim=2)
        key = torch.cat([neighbors_feat, edge_feat, neighbors_time_feat], dim=2)

        # Reshape tensors to the expected shape by the multihead attention layer
        permutation = [1, 0, 2]
        query = query.permute(permutation)  # New shape -> (1, batch_size, n_features)
        key = key.permute(permutation)  # New shape -> (n_neighbors, batch_size, n_features)

        # Compute mask of nodes with all invalid neighbors
        invalid_neighborhood = neighbors_padding.all(dim=1, keepdim=True)
        # For those source nodes, force its first neighbor to be valid
        neighbors_padding[invalid_neighborhood.squeeze(), 0] = False

        out, weights = self.multi_head_target(
            query=query,
            key=key,
            value=key,
            key_padding_mask=neighbors_padding
        )

        out = out.squeeze()
        weights = weights.squeeze()

        # Source nodes with no neighbors have an all zero output
        out = out.masked_fill(invalid_neighborhood, 0)
        weights = weights.masked_fill(invalid_neighborhood, 0)
        out = self.merger(out, src_node_feat)

        return out, weights


class TimeEncoder(nn.Module):
    """
    Time encoding method proposed by TGAT.
    https://doi.org/10.48550/arXiv.2002.07962
    """
    def __init__(self, dim):
        super().__init__()

        self.dim = dim
        self.w = nn.Linear(1, dim)
        w_param = torch.from_numpy(1 / 10 ** np.linspace(0, 9, dim))
        self.w.weight = nn.Parameter(w_param.float().reshape(dim, -1))
        self.w.bias = nn.Parameter(torch.zeros(dim).float())

    def forward(self, t):
        """
        :param t: Tensor of shape (batch_size, seq_length)
        :return: Output tensor of shape (batch_size, seq_length, dim)
        """
        t = t.unsqueeze(dim=2)  # New shape -> (batch_size, seq_length, 1) to apply linear transformation
        return torch.cos(self.w(t))
