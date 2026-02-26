import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class GCN(nn.Module):
    """
    GCN for link prediction with an edge-aware decoder.

    Encoder: 2x GCNConv -> node embeddings z
    Decoder: MLP on pair features + edge_attr (e.g., distance)
    """

    def __init__(self, in_channels, hidden_channels, out_channels, edge_dim=1, decoder_hidden=128):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

        # Decoder: uses pairwise embedding features + edge_attr
        in_dim = 4 * out_channels + edge_dim  # zu, zv, |diff|, prod, edge_attr
        self.edge_mlp = nn.Sequential(
            nn.Linear(in_dim, decoder_hidden),
            nn.ReLU(),
            nn.Linear(decoder_hidden, 1),
        )

    def encode(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x

    def decode(self, z, edge_label_index, edge_attr=None):
        """
        edge_attr: Tensor [E, edge_dim], e.g. distance features for each candidate edge.
        If edge_attr is None, it defaults to zeros (so you can still run without it).
        """
        u, v = edge_label_index
        zu, zv = z[u], z[v]

        # Optional: keep normalization like your dot-product version (often helps)
        zu = F.normalize(zu, p=2, dim=-1)
        zv = F.normalize(zv, p=2, dim=-1)

        pair_feat = torch.cat([zu, zv, (zu - zv).abs(), zu * zv], dim=1)  # [E, 4*out_channels]

        if edge_attr is None:
            edge_attr = torch.zeros((pair_feat.size(0), 1), device=pair_feat.device, dtype=pair_feat.dtype)

        feat = torch.cat([pair_feat, edge_attr], dim=1)  # [E, 4*out + edge_dim]
        logits = self.edge_mlp(feat).squeeze(-1)         # [E]
        return logits

    def forward(self, data):
        return self.encode(data.x, data.edge_index)