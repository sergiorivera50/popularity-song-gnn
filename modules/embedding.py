from typing import Optional
import numpy as np
import torch
import torch.nn as nn
from abc import ABC, abstractmethod

from models import TemporalAttentionLayer


class EmbeddingModule(nn.Module, ABC):
    """
    Abstract module used to create node embeddings considering temporal dynamics.
    """

    def __init__(
            self,
            node_feat,
            edge_feat,
            neighbor_finder,
            time_encoder,
            n_layers: int,
            n_node_feat: int,
            n_edge_feat: int,
            n_time_feat: int,
            embedding_dim: int,
            dropout: float,
            device: torch.cuda.device,
    ):
        super().__init__()

        self.node_feat = node_feat
        self.edge_feat = edge_feat
        self.neighbor_finder = neighbor_finder
        self.time_encoder = time_encoder
        self.n_layers = n_layers
        self.n_node_feat = n_node_feat
        self.n_edge_feat = n_edge_feat
        self.n_time_feat = n_time_feat
        self.dropout = dropout
        self.embedding_dim = embedding_dim
        self.device = device

    @abstractmethod
    def compute_embedding(self, memory, src_nodes, timestamps, n_layers, n_neighbors=20):
        pass


class AttentionEmbedding(EmbeddingModule):
    """
    Implementation of the temporal attention mechanism over neighbors to generate embeddings.
    """

    def __init__(
            self,
            node_feat,
            edge_feat,
            neighbor_finder,
            time_encoder,
            n_layers,
            n_node_feat,
            n_edge_feat,
            n_time_feat,
            embedding_dim,
            device,
            dropout: Optional[float] = 0.1,
            attention_heads: Optional[int] = 2,
            use_memory: Optional[bool] = True
    ):
        super().__init__(
            node_feat=node_feat,
            edge_feat=edge_feat,
            neighbor_finder=neighbor_finder,
            time_encoder=time_encoder,
            n_layers=n_layers,
            n_node_feat=n_node_feat,
            n_edge_feat=n_edge_feat,
            n_time_feat=n_time_feat,
            embedding_dim=embedding_dim,
            dropout=dropout,
            device=device,
        )

        self.use_memory = use_memory
        self.attention_models = nn.ModuleList([
            TemporalAttentionLayer(
                n_node_feat=n_node_feat,
                n_neighbors_feat=n_node_feat,
                n_edge_feat=n_edge_feat,
                time_dim=n_time_feat,
                attention_heads=attention_heads,
                dropout=dropout,
                output_dim=n_node_feat
            ) for _ in range(n_layers)
        ])

    def compute_embedding(
            self,
            memory,
            src_nodes,
            timestamps,
            n_layers,
            n_neighbors: Optional[int] = 20,
    ):
        assert n_layers >= 0  # ensure number of layers is non-negative

        src_nodes_torch = torch.from_numpy(src_nodes).long().to(self.device)
        timestamps_torch = torch.from_numpy(timestamps).float().to(self.device)
        timestamps_torch = torch.unsqueeze(timestamps_torch, dim=1)

        src_nodes_time_embedding = self.time_encoder(torch.zeros_like(timestamps_torch))
        src_node_feat = self.node_feat[src_nodes_torch, :]

        if self.use_memory:
            # Update source node features with external memory if used
            src_node_feat = memory[src_nodes, :] + src_node_feat

        if n_layers == 0:
            # Base case: return node features if no layers are left to process
            return src_node_feat
        else:
            # Recursive call to compute embeddings for the next layer
            src_node_conv_embeddings = self.compute_embedding(
                memory=memory,
                src_nodes=src_nodes,
                timestamps=timestamps,
                n_layers=n_layers - 1,
                n_neighbors=n_neighbors,
            )

            # Find temporal neighbors
            neighbors, edge_ix, edge_times = self.neighbor_finder.get_temporal_neighbor(
                src_nodes,
                timestamps,
                n_neighbors=n_neighbors,
            )

            # Prepare data for attention computation
            neighbors_torch = torch.from_numpy(neighbors).long().to(self.device)
            edge_ix = torch.from_numpy(edge_ix).long().to(self.device)
            edge_deltas = timestamps[:, np.newaxis] - edge_times
            edge_deltas_torch = torch.from_numpy(edge_deltas).float().to(self.device)

            # Compute embeddings for neighbors to use in the attention mechanism
            neighbors = neighbors.flatten()
            neighbor_embeddings = self.compute_embedding(
                memory=memory,
                src_nodes=neighbors,
                timestamps=np.repeat(timestamps, n_neighbors),
                n_layers=n_layers - 1,
                n_neighbors=n_neighbors,
            )

            valid_n_neighbors = n_neighbors if n_neighbors > 0 else 1
            neighbor_embeddings = neighbor_embeddings.view(len(src_nodes), valid_n_neighbors, -1)
            edge_time_embeddings = self.time_encoder(edge_deltas_torch)

            edge_feat = self.edge_feat[edge_ix, :]
            mask = neighbors_torch == 0

            # Aggregate information from neighbors using the attention mechanism
            src_embedding = self.aggregate(
                n_layer=n_layers,
                src_node_feat=src_node_conv_embeddings,
                src_node_time_embedding=src_nodes_time_embedding,
                neighbor_embeddings=neighbor_embeddings,
                edge_time_embeddings=edge_time_embeddings,
                edge_feat=edge_feat,
                mask=mask,
            )

            return src_embedding

    def aggregate(
            self,
            n_layer: int,
            src_node_feat,
            src_node_time_embedding,
            neighbor_embeddings,
            edge_time_embeddings,
            edge_feat,
            mask
    ):
        attn_model = self.attention_models[n_layer - 1]
        src_embedding, _ = attn_model(
            src_node_feat=src_node_feat,
            src_time_feat=src_node_time_embedding,
            neighbors_feat=neighbor_embeddings,
            neighbors_time_feat=edge_time_embeddings,
            edge_feat=edge_feat,
            neighbors_padding=mask,
        )
        return src_embedding


def get_embedding_module(
        module_type: str,
        node_feat,
        edge_feat,
        neighbor_finder,
        time_encoder,
        n_layers,
        n_node_feat,
        n_edge_feat,
        n_time_feat,
        embedding_dim,
        device,
        attention_heads: Optional[int] = 2,
        dropout: Optional[float] = 0.1,
        use_memory: Optional[bool] = True,
):
    match module_type:
        case "attention":
            return AttentionEmbedding(
                node_feat=node_feat,
                edge_feat=edge_feat,
                neighbor_finder=neighbor_finder,
                time_encoder=time_encoder,
                n_layers=n_layers,
                n_node_feat=n_node_feat,
                n_edge_feat=n_edge_feat,
                n_time_feat=n_time_feat,
                embedding_dim=embedding_dim,
                device=device,
                attention_heads=attention_heads,
                dropout=dropout,
                use_memory=use_memory,
            )
        case _:
            raise NotImplemented(f"Embedding module '{module_type}' not implemented.")
