import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from sklearn.metrics import roc_auc_score, average_precision_score
import pandas as pd
import numpy as np

class GCN(torch.nn.Module):
    """
    A Graph Convolutional Network (GCN) model for link prediction tasks.

    This model computes node embeddings via two stacked GCN layers and
    uses a dot-product decoder to predict the existence of edges between
    pairs of nodes. The design follows the common link prediction pipeline
    in PyTorch Geometric, where `forward` produces embeddings and the
    `decode` function maps candidate edges to scalar scores.

    Parameters
    ----------
    in_channels : int
        Dimensionality of the input node features.
    hidden_channels : int
        Number of hidden units in the first GCN layer.
    out_channels : int
        Dimensionality of the output embeddings (typically used for decoding).

    Methods
    -------
    encode(x, edge_index)
        Computes node embeddings by applying two GCN layers with ReLU
        activation after the first layer.
    decode(z, edge_label_index)
        Computes edge scores for given pairs of nodes using a dot product
        decoder. Returns a tensor of shape [num_edges].
    forward(data)
        Runs the encoder on the input graph data and returns node embeddings.
        The decoder must be called separately on the embeddings and candidate
        edge indices.
    """
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def encode(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x   # Knoteneinbettungen z

    def decode(self, z, edge_label_index):
        # Dot Product Decoder
        z = z[edge_label_index[0]] * z[edge_label_index[1]]
        return z.sum(dim=-1)

    def forward(self, data):
        # liefert nur z (Embeddings), Decoder separat aufrufen
        return self.encode(data.x, data.edge_index)
