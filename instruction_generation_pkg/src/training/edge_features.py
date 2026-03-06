import torch

def edge_distance(batch, edge_label_index):
    u, v = edge_label_index
    dpos = batch.pos[u] - batch.pos[v]

    r = torch.norm(dpos, dim=1, keepdim=True)
    log_r = torch.log1p(r)
    log_r2 = torch.log1p(r*r)
    edge_attr = torch.cat([log_r, log_r2], dim=1)   # statt r und r2 roh

    return edge_attr
