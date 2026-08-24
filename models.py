"""
GNN classifiers for the benchmark.
  GCN           – 2-layer PyG GCN (Kipf & Welling, 2017)
  GAT           – 2-layer PyG GAT
  GIN           – 2-layer PyG GIN
  GCN_inductive – sparse-adj GCN used by Bonsai on SAINT datasets
                  and by GDEM-style evaluation (identical to original models.py)
  H2GCN         – Beyond Homophily (Zhu et al., NeurIPS 2020)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_sparse
import torch_geometric.nn as gnn


# ── exact copies from official models.py ────────────────────────────────────

class GCN(nn.Module):
    def __init__(self, indim, outdim, *, hidim=128):
        super().__init__()
        self.l1 = gnn.GCNConv(indim, hidim)
        self.l2 = gnn.GCNConv(hidim, outdim)

    def forward(self, x, edge_index, edge_weight=None):
        h = F.relu(self.l1(x, edge_index, edge_weight))
        return self.l2(h, edge_index, edge_weight)


class GAT(nn.Module):
    def __init__(self, indim, outdim, *, hidim=128):
        super().__init__()
        self.l1 = gnn.GATConv(indim, hidim)
        self.l2 = gnn.GATConv(hidim, outdim)

    def forward(self, x, edge_index, edge_weight=None):
        h = F.relu(self.l1(x, edge_index))
        return self.l2(h, edge_index)


class GIN(nn.Module):
    def __init__(self, indim, outdim, *, hidim=128):
        super().__init__()
        self.l1 = gnn.GINConv(nn.Linear(indim, hidim))
        self.l2 = gnn.GINConv(nn.Linear(hidim, outdim))

    def forward(self, x, edge_index, edge_weight=None):
        h = F.relu(self.l1(x, edge_index))
        return self.l2(h, edge_index)


class _GraphConvolution(nn.Module):
    """Simple sparse GCN layer (tkipf/pygcn) – from official model.py."""
    def __init__(self, in_features, out_features, with_bias=True):
        super().__init__()
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        self.bias   = nn.Parameter(torch.FloatTensor(out_features)) if with_bias else None
        nn.init.xavier_uniform_(self.weight.data.T)
        if self.bias is not None:
            self.bias.data.zero_()

    def forward(self, x, adj):
        support = torch.spmm(x, self.weight) if x.data.is_sparse else torch.mm(x, self.weight)
        out = torch_sparse.matmul(adj, support) if isinstance(adj, torch_sparse.SparseTensor) \
              else torch.spmm(adj, support)
        return out + self.bias if self.bias is not None else out


class GCN_inductive(nn.Module):
    """Sparse GCN used on SAINT datasets and GDEM evaluation – from official model.py."""
    def __init__(self, indim, outdim, hidim=256, nlayers=2,
                 with_relu=True, with_bias=True, with_bn=False):
        super().__init__()
        self.with_relu = with_relu
        self.layers = nn.ModuleList()
        dims = [indim] + [hidim] * (nlayers - 1) + [outdim]
        for i in range(nlayers):
            self.layers.append(_GraphConvolution(dims[i], dims[i+1], with_bias))

    def forward(self, x, adj, edge_index=None, edge_weight=None):
        for i, layer in enumerate(self.layers):
            x = layer(x, adj)
            if i < len(self.layers) - 1 and self.with_relu:
                x = F.relu(x)
        return F.log_softmax(x, dim=1)


# ── H2GCN (Zhu et al., NeurIPS 2020) ────────────────────────────────────────
# "Beyond Homophily in Graph Neural Networks: Current Limitations and
#  Effective Designs" – https://arxiv.org/abs/2006.11468
#
# Key design:
#   1. Ego / neighbour embedding separation
#   2. Higher-order neighbourhoods (1-hop + 2-hop aggregation)
#   3. Concatenation of embeddings from each round before classification

class H2GCN(nn.Module):
    """
    H2GCN – designed for heterophilic graphs, but works on homophilic ones too.

    Two rounds of:
        r_ego  = W_ego  @ x
        r_1hop = W_nbr  @ mean_aggregate(x, 1-hop)
        r_2hop = W_nbr2 @ mean_aggregate(x, 2-hop)
    Final representation = concat([r_ego_r1, r_1hop_r1, r_2hop_r1,
                                    r_ego_r2, r_1hop_r2, r_2hop_r2])
    """
    def __init__(self, indim, outdim, *, hidim=64, dropout=0.5):
        super().__init__()
        self.dropout = dropout
        # separate projections for ego vs neighbour channels
        self.W_ego  = nn.Linear(indim,  hidim, bias=False)
        self.W_nbr1 = nn.Linear(indim,  hidim, bias=False)
        self.W_nbr2 = nn.Linear(hidim,  hidim, bias=False)   # applied to 1-hop result -> 2-hop
        # classifier over 2 rounds × 3 channels = 6 × hidim
        self.classifier = nn.Linear(hidim * 6, outdim)
        for m in [self.W_ego, self.W_nbr1, self.W_nbr2, self.classifier]:
            nn.init.xavier_uniform_(m.weight)

    @staticmethod
    def _mean_agg(x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Degree-normalised mean aggregation (no self-loop)."""
        from torch_geometric.utils import degree
        src, dst = edge_index
        n   = x.size(0)
        out = torch.zeros_like(x)
        out.index_add_(0, dst, x[src])
        deg = degree(dst, n, dtype=x.dtype).clamp(min=1).unsqueeze(1)
        return out / deg

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                edge_weight=None) -> torch.Tensor:
        # ---------- round 1 ----------
        ego_1  = F.relu(self.W_ego(x))
        nbr1_1 = F.relu(self.W_nbr1(self._mean_agg(x, edge_index)))
        nbr2_1 = F.relu(self.W_nbr2(nbr1_1))                          # 2-hop via composition

        r1 = F.dropout(torch.cat([ego_1, nbr1_1, nbr2_1], dim=1),
                        p=self.dropout, training=self.training)        # (N, 3H)

        # ---------- round 2 (re-aggregate r1 channels) ----------
        ego_2  = ego_1                                                  # ego unchanged
        nbr1_2 = F.relu(self.W_nbr2(self._mean_agg(ego_1, edge_index)))
        nbr2_2 = F.relu(self.W_nbr2(nbr1_2))

        r2 = F.dropout(torch.cat([ego_2, nbr1_2, nbr2_2], dim=1),
                        p=self.dropout, training=self.training)        # (N, 3H)

        return self.classifier(torch.cat([r1, r2], dim=1))            # (N, outdim)


# ── registry ─────────────────────────────────────────────────────────────────

MODELS = {
    "GCN":          GCN,
    "GAT":          GAT,
    "GIN":          GIN,
    "GCN_inductive":GCN_inductive,
    "H2GCN":        H2GCN,
}


def build_model(name: str, indim: int, outdim: int, hidim: int = 128) -> nn.Module:
    if name not in MODELS:
        raise ValueError(f"Unknown model '{name}'. Choose from {list(MODELS)}")
    return MODELS[name](indim, outdim, hidim=hidim)
