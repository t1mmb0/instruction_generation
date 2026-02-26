import torch

def edge_distance(batch, edge_label_index):
    u, v = edge_label_index
    dpos = batch.pos[u] - batch.pos[v]
    dist = torch.norm(dpos, dim=1, keepdim=True)
    dist = torch.log1p(dist)
    
    return dist