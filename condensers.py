"""
condensers.py  –  Graph condensation methods.

Condensers
----------
  bonsai   – Gradient-free Rev-k-NN + PPR (Gupta et al., ICLR 2025)
             Exact port of official main.py.
  herald   – HERALD: High-fidelity Exemplar Retrieval with Adaptive Landmark
               Distillation. Budget-compatible with Bonsai (same WL+DT feature
               count, same cost formula). Nodes scored by prototype + boundary +
               LID with adaptive weights; same BFS / PPR / rebalance pipeline.
  random   – Induced subgraph of uniformly random training nodes (baseline)
  herding  – Class-balanced coreset via Herding (Welling, 2009; baseline)

"""

from __future__ import annotations
import gc
import heapq
import time
import typing as t
import warnings
from collections import defaultdict
from pathlib import Path

import networkx as nx
import numpy as np
import scipy.sparse as sp
import torch
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances
from torch_geometric.data import Data
from torch_geometric.utils import from_networkx, to_undirected
from tqdm import tqdm

warnings.filterwarnings("ignore")


# ═══════════════════════════════════════════════════════════════════════════════
#  SHARED UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════

def _size_bytes(nnodes, nedges, nfeats, dtype) -> float:
    mx = 1 if dtype == "int" else 2
    return (nnodes * nfeats * mx + nedges * 2) * 2


def _build_sparse_adj_normed(edge_index: torch.Tensor, n: int):
    """D^{-1/2}(A+I)D^{-1/2} as SparseTensor."""
    from torch_sparse import SparseTensor
    d   = np.ones(edge_index.shape[1])
    r   = edge_index[0].cpu().numpy()
    c   = edge_index[1].cpu().numpy()
    adj = sp.csr_matrix((d, (r, c)), shape=(n, n)).tolil()
    adj = adj + sp.eye(n)
    rs  = np.array(adj.sum(1))
    ri  = np.power(rs, -0.5).flatten(); ri[np.isinf(ri)] = 0.0
    rm  = sp.diags(ri)
    adj = rm.dot(adj).dot(rm).tocoo().astype(np.float32)
    return SparseTensor(row=torch.LongTensor(adj.row),
                        col=torch.LongTensor(adj.col),
                        value=torch.FloatTensor(adj.data),
                        sparse_sizes=(n, n))


# ── WL representations (verbatim WL_Distance2.py) ────────────────────────────

def _compute_wl_representations(x: torch.Tensor, adj) -> torch.Tensor:
    from torch_sparse import SparseTensor, matmul as sp_mm
    adj = adj.tolil()
    adj = adj + sp.eye(adj.shape[0])
    rowsum = np.array(adj.sum(1))
    mask   = rowsum == 0
    rowsum[mask] = 1
    r_inv  = (1 / rowsum).flatten()
    r_mat  = sp.diags(r_inv)
    adj    = r_mat.dot(adj).dot(r_mat)
    idx    = np.nonzero(mask)[0]
    adj[idx, idx] = 1
    # Sort COO indices to guarantee deterministic floating-point summation order
    # across runs (unsorted COO can produce different fp results via different
    # scatter-add orderings).
    adj    = adj.tocsr().sorted_indices().tocoo().astype(np.float32)
    sr = torch.LongTensor(adj.row).unsqueeze(1)
    sc = torch.LongTensor(adj.col).unsqueeze(1)
    sv = torch.FloatTensor(adj.data)
    at = torch.sparse_coo_tensor(torch.cat((sr, sc), 1).t(), sv,
                                  torch.Size(adj.shape))
    st = SparseTensor(row=at._indices()[0], col=at._indices()[1],
                      value=at._values(), sparse_sizes=at.size())
    return sp_mm(st, x)


# ── decision-tree feature selection (verbatim utils.py) ──────────────────────

def _transform_features_with_tree(data, ego_graph_vmed, train):
    from sklearn.tree import DecisionTreeClassifier
    features = [t_.numpy() for t_ in ego_graph_vmed]
    labels   = data.y[train].numpy()
    clf      = DecisionTreeClassifier(max_depth=50, random_state=42)
    clf.fit(features, labels)
    used: t.List[int] = []

    def traverse(nid):
        nd = clf.tree_
        if nd.feature[nid] >= 0:
            used.append(nd.feature[nid])
            traverse(nd.children_left[nid])
            traverse(nd.children_right[nid])
    traverse(0)
    used = sorted(set(used))
    return [t[used] for t in ego_graph_vmed], used


# ── HERALD: Feature Selection ─────────────────────────────────────────────────

def _herald_select_features(
    x: torch.Tensor,
    adj: "sp.csr_matrix",
    train: np.ndarray,
    labels: torch.Tensor,
    h: float,
    top_k: int,
) -> t.List[int]:
    """
    HERALD joint feature selection: discriminativeness × activation density.

    score(j) = fisher_n(j) × density_n(j)

    fisher_n : multi-hop weighted Fisher discriminant
               fisher(j) = Σ_{k=0}^{2}  (1-h)^k · fisher_k(j)
               fisher_k(j) = between/within class variance of (A^k X)[:,j]
               On heterophilic graphs (high h): raw-space Fisher dominates,
               avoiding WL smoothing corruption of class signal.

    density_n: mean activation of feature j over train nodes,
               = mean|x[train, j]|.
               Ensures selected features are commonly active so that
               feat_len[v] (= sum of selected features for node v on binary
               datasets) stays comparable to Bonsai's DT selection, keeping
               the budget formula well-behaved.

    Product ensures both criteria hold simultaneously — a rare but discriminative
    feature scores 0, as does a common but uninformative one.  This mirrors what
    Bonsai's DT implicitly achieves (splits require features active on many nodes).

    top_k is set to len(bonsai_feats) so the budget is identical to Bonsai.
    The novelty is *which* features are selected, not how many.
    """
    X_np   = x.cpu().numpy().astype(np.float64)
    y_np   = labels.cpu().numpy().astype(np.int64)
    N, F   = X_np.shape
    ytrain = y_np[train]
    classes = np.unique(ytrain)
    top_k  = max(1, min(int(top_k), F))

    # activation density (normalised to [0,1])
    density   = np.abs(X_np[train]).mean(0)
    density_n = density / (density.max() + 1e-8)

    # multi-hop weighted Fisher
    deg    = np.asarray(adj.sum(1), dtype=np.float64).flatten().clip(min=1)
    A_norm = sp.diags(1.0 / deg) @ adj.astype(np.float64)

    fisher_acc = np.zeros(F, dtype=np.float64)
    Xhop = X_np.copy()
    for k in range(3):
        w_k = (1.0 - h) ** k
        Xtk  = Xhop[train]
        mu   = Xtk.mean(0)
        wv   = np.zeros(F, dtype=np.float64)
        bv   = np.zeros(F, dtype=np.float64)
        for c in classes:
            mask = ytrain == c; nc = int(mask.sum())
            if nc < 2: continue
            Xc = Xtk[mask]; mc = Xc.mean(0)
            wv += Xc.var(0) * nc
            bv += nc * (mc - mu) ** 2
        fk = bv / (wv + 1e-8)
        fm = fk.max()
        if fm > 1e-8: fk /= fm
        fisher_acc += w_k * fk
        if k < 2:
            Xhop = np.asarray(A_norm @ Xhop, dtype=np.float64)

    fisher_n = fisher_acc / (fisher_acc.max() + 1e-8)

    score    = fisher_n * density_n
    selected = np.argsort(-score)[:top_k]
    return sorted(selected.tolist())


# ── Rev-k-NN + CELF (verbatim utils.py) ──────────────────────────────────────

def _wl2rknn(WL_dist: np.ndarray, *, sampled_nodes: np.ndarray, k: int) -> dict:
    # Use argsort (not argpartition) so that ties in WL distance are broken
    # deterministically by node index, giving identical kNN sets across runs.
    knn = []
    for node in tqdm(range(WL_dist.shape[0]), desc="Eval KNN", ascii=True, ncols=120):
        knn.append(np.argsort(WL_dist[node])[:k])
    rknn = defaultdict(set)
    for node, knn_node in tqdm(enumerate(knn), desc="Eval rKNN", ascii=True, ncols=120):
        [rknn[q].add(sampled_nodes[node]) for q in knn_node]
    return {"rknn": dict(rknn), "knn": np.asarray(knn)}


def _celf(rknn_dict: dict) -> t.List[int]:
    """Verbatim from official utils.py."""
    covered, selected = set(), []
    pq = [(-len(v), nd) for nd, v in rknn_dict.items()]
    heapq.heapify(pq)
    if not pq:
        return []
    neg, nd = heapq.heappop(pq)
    covered.update(rknn_dict[nd]); selected.append(nd)
    while pq:
        neg, nd = heapq.heappop(pq)
        if neg == 0:
            selected.extend(n for _, n in pq); selected.append(nd); break
        new_gain = len(set(rknn_dict[nd]) - covered)
        max_gain, max_node = new_gain, nd
        if pq:
            stale, top = pq[0]
            temp = [(new_gain, nd)]; midx = 0; idx = 0
            while new_gain < -stale:
                new_gain = len(set(rknn_dict[top]) - covered)
                temp.append((new_gain, top)); idx += 1
                if new_gain > max_gain: midx = idx; max_gain = new_gain; max_node = top
                heapq.heappop(pq)
                if not pq: break
                stale, top = pq[0]
            temp.pop(midx)
            [heapq.heappush(pq, (-ng, n)) for ng, n in temp]
        selected.append(max_node)
        covered.update(rknn_dict[max_node])
        del rknn_dict[max_node]
    return selected


# ── module-level globals (mirror main.py) ─────────────────────────────────────

_ADJ: sp.csr_matrix | None = None
_GLOBAL_NEIGHBORS_DICT: dict = {}
_GLOBAL_FEATS  = None
_FEAT_LEN      = None
_FEAT_MULTIPLIER = 1


def apply_pca_compression(x_np: np.ndarray, variance_ratio: float = 0.90):
    """
    Fit a PCA on x_np and reduce to the minimum number of components that
    retain at least `variance_ratio` of the total variance.

    Returns
    -------
    x_reduced  : np.ndarray  shape (N, n_components)
    pca        : fitted sklearn.decomposition.PCA object
    n_components : int — number of principal components kept
    """
    max_components = min(x_np.shape[0], x_np.shape[1])
    pca = PCA(n_components=max_components, svd_solver="full")
    pca.fit(x_np)
    cumvar = np.cumsum(pca.explained_variance_ratio_)
    n_components = int(np.searchsorted(cumvar, variance_ratio) + 1)
    n_components = max(1, min(n_components, max_components))
    pca = PCA(n_components=n_components, svd_solver="full")
    x_reduced = pca.fit_transform(x_np)
    return x_reduced.astype(np.float32), pca, n_components


def _build_neighborhood_dict_sparse() -> dict:
    """_ADJ must be set before calling.
    Stores neighbours as sets (for O(1) lookup) but builds from sorted COO
    so the set contents are always identical across scipy versions.
    """
    neighbors: dict = {}
    for node in tqdm(range(_ADJ.shape[0]), ascii=True, ncols=120,
                     desc="build neighbor"):
        # sorted_indices() guarantees deterministic col ordering across runs
        col = _ADJ[[node]].tocsr().sorted_indices().tocoo().col
        nbrs = set(col.tolist()); nbrs.add(node)
        neighbors[node] = nbrs
    return neighbors


def _rknn_sorted2budget_select_merged(sorted_nodes, train, target_size) -> list:
    """Verbatim from main.py."""
    size_sel, sel_nds, sel_es = [], set(), set()
    ints_till = 0; no_sel = 0
    for _, node in tqdm(enumerate(sorted_nodes), ascii=True, ncols=120,
                        total=len(sorted_nodes),
                        desc="rknn_sorted2budget_select_merged"):
        cand_nds, cand_es = set(), set()
        tn  = train[node]
        if tn not in sel_nds: cand_nds.add(tn)
        nadj = _GLOBAL_NEIGHBORS_DICT[tn]
        for nbr in nadj:
            if nbr not in sel_nds: cand_nds.add(nbr)
            e = tuple(sorted([nbr, tn]))
            if e not in sel_es: cand_es.add(e)
            for nn2 in nadj.intersection(_GLOBAL_NEIGHBORS_DICT[nbr]):
                e2 = tuple(sorted([nbr, nn2]))
                if e2 not in sel_es: cand_es.add(e2)
        ints_nodes = sum(_FEAT_LEN[nd] for nd in cand_nds) * _FEAT_MULTIPLIER + ints_till
        size_now   = (ints_nodes + (len(sel_es) + len(cand_es)) * 2) * 2
        if size_now < target_size:
            size_sel.append(node); sel_nds.update(cand_nds)
            sel_es.update(cand_es); ints_till = ints_nodes; no_sel = 0
        else:
            no_sel += 1
            if no_sel > 100: break
    return size_sel


def _match_distribution_bonsai(merged_graph, data, train, rknn_ranked_nodes):
    
    num_nodes = sum(1 for _, v in merged_graph.nodes(data=True) if v["target"])
    cc = torch.bincount(data.y[train]).float()
    scaled = (cc / len(train) * num_nodes).long()
    cd = {i: [] for i in range(len(scaled))}
    mapped = [train[n] for n in rknn_ranked_nodes]
    for nd in mapped: cd[data.y[nd].item()].append(nd)
    added = set()
    for cl, sc in enumerate(scaled):
        cls_nds = [n for n, a in merged_graph.nodes(data=True)
                   if data.y[n].item() == cl and a["target"]]
        lo, hi = 0.99 * sc.item(), 1.01 * sc.item(); cur = len(cls_nds)
        if cur < lo:
            cnt = 0
            for nd in cd[cl]:
                if nd not in merged_graph: merged_graph.add_node(nd); added.add(nd); cnt += 1
                if cnt >= lo - cur: break
        elif cur > hi:
            def sk(nd, _m=mapped):
                try: return _m.index(nd)
                except: return float('inf')
            for nd in sorted(cls_nds, key=sk)[-(cur - int(hi)):]:
                if nd in merged_graph: merged_graph.remove_node(nd)
    gnds = set(merged_graph.nodes)
    for i in range(data.edge_index.shape[1]):
        u, v = data.edge_index[0, i].item(), data.edge_index[1, i].item()
        if (u in added or v in added) and u in gnds and v in gnds:
            if not merged_graph.has_edge(u, v): merged_graph.add_edge(u, v)
    return merged_graph


def _attach_and_convert(G, x_np, y_np, train_set, dataset_name, scaler) -> Data:
    """Attach node attrs to nx.Graph and convert to Data."""
    SAINT = ["flickr", "ogbn-arxiv", "reddit"]
    for nd in list(G.nodes):
        G.nodes[nd]["target"] = (nd in train_set)
        G.nodes[nd]["x"]      = x_np[nd].tolist()
        G.nodes[nd]["y"]      = int(y_np[nd])
    d = from_networkx(G, group_node_attrs=["x"])
    if dataset_name in SAINT:
        d.x  = torch.tensor(scaler(d.x.cpu().numpy()).astype(np.float32))
        d.adj = _build_sparse_adj_normed(d.edge_index, d.x.shape[0])
    return d


# ── Budget / graph statistics printer ────────────────────────────────────────

def _print_budget_stats(
    label: str,
    data_syn,
    features_used: t.List[int],
    size_full: float,
    budget: float,
    target_size_frac: float,
    N_full: int,
    E_full: int,
    F_full: int,
    dtype: str = "float",
) -> None:
    """
    Print a compact table of graph statistics for the condensed dataset.
    """
    ns = data_syn.x.shape[0]
    es = data_syn.edge_index.shape[1]
    fs = data_syn.x.shape[1]

    # use the same mx as _size_bytes so condensed size is comparable
    mx      = 1 if dtype == "int" else 2
    sz_cond = (ns * fs * mx + es * 2) * 2
    actual_frac = sz_cond / max(size_full, 1)

    sep = "─" * 62
    print(f"\n  ┌{sep}┐")
    print(f"  │  [{label}]  Budget & Graph Statistics")
    print(f"  ├{sep}┤")
    print(f"  │  Full graph :  {N_full:>7,} nodes  {E_full:>9,} edges  {F_full:>5} feats")
    print(f"  │  Full size  :  {size_full:>14,.0f} bytes  [dense formula, = size_full]")
    print(f"  │  Budget     :  {budget:>14,.0f} bytes  ({target_size_frac*100:.2f}%)")
    print(f"  │              :  (budget enforced with sparse FEAT_LEN during selection)")
    print(f"  ├{sep}┤")
    print(f"  │  Condensed  :  {ns:>7,} nodes  {es:>9,} edges  {fs:>5} feats")
    print(f"  │  Sr(%)      :  {sz_cond:>14,.0f} bytes  ({actual_frac*100:.3f}% of full)")
    print(f"  │              :  [dense formula — matches paper Table 4 Sr(%) definition]")
    print(f"  ├{sep}┤")
    print(f"  │  Reduction  :  nodes {N_full/max(ns,1):>6.1f}×   "
          f"edges {E_full/max(es,1):>6.1f}×   "
          f"feats {F_full/max(fs,1):>5.1f}×")
    if len(features_used) < F_full:
        print(f"  │  Feat. idx  :  {features_used[:8]}"
              f"{'…' if len(features_used)>8 else ''}")
    print(f"  └{sep}┘")


# ═══════════════════════════════════════════════════════════════════════════════
#  BONSAI  (verbatim port of official main.py)
# ═══════════════════════════════════════════════════════════════════════════════

def condense_bonsai(data: Data, train: np.ndarray, splits: dict,
                    scaler, dataset_name: str, dtype: str,
                    target_size_frac: float, k: int = 5,
                    frac_to_sample: float = 1.0) -> dict:
    """Exact port of official main.py. data.x is NEVER mutated in-place."""
    global _ADJ, _GLOBAL_NEIGHBORS_DICT, _GLOBAL_FEATS, _FEAT_LEN, _FEAT_MULTIPLIER
    t0 = time.perf_counter()
    SAINT = ["flickr", "ogbn-arxiv", "reddit"]

    N = data.x.shape[0]; F_orig = data.x.shape[1]
    nedges = data.edge_index.shape[1]
    rei, cei = data.edge_index.numpy()
    _ADJ = sp.csr_matrix((np.ones(len(rei)), (rei, cei)), shape=(N, N))

    size_full   = _size_bytes(N, nedges, F_orig, dtype)
    target_size = float(f"{target_size_frac * size_full:.2f}")

    x_in = data.x_normed if hasattr(data, "x_normed") else data.x
    wl   = _compute_wl_representations(x_in, _ADJ)[train]
    _GLOBAL_NEIGHBORS_DICT = _build_neighborhood_dict_sparse()
    _GLOBAL_FEATS = x_in

    tmp = Data(x=x_in, y=data.y, edge_index=data.edge_index)
    wl, features_used = _transform_features_with_tree(tmp, wl, train)

    # LOCAL copy — caller's data.x is NEVER modified
    x_sl = x_in[:, features_used]
    _GLOBAL_FEATS = x_sl; nfeats = len(features_used)

    if dataset_name not in ["ogbn-arxiv", "reddit", "flickr", "pubmed"]:
        _FEAT_MULTIPLIER = 1
        # Mirrors official main.py line 602: FEAT_LEN.append(data.x[x].sum().item())
        # where data.x has already been sliced to features_used (line 593).
        _FEAT_LEN = [x_sl[i].sum().item() for i in range(N)]
    elif dataset_name in ["pubmed", "flickr"]:
        _FEAT_MULTIPLIER = 2 if dataset_name == "flickr" else 3
        _FEAT_LEN = [torch.where(x_sl[i] == 0, 0, 1).sum().item() for i in range(N)]
    else:
        _FEAT_MULTIPLIER = 2; _FEAT_LEN = defaultdict(lambda: nfeats)

    # Stack WL representations into a uniform 2D array.
    # After _transform_features_with_tree, `wl` is a list of tensors that may
    # have been sliced to different lengths; torch.stack enforces a homogeneous
    # shape and avoids the "inhomogeneous part" ValueError from np.array().
    if isinstance(wl[0], torch.Tensor):
        wl_mat = torch.stack(wl).numpy()
    else:
        wl_mat = np.vstack(wl)

    if frac_to_sample == 1.0:
        WL_dist = pairwise_distances(wl_mat, n_jobs=1)
        sampled = np.arange(len(wl_mat))
    else:
        n = len(wl_mat); m = min(max(int(frac_to_sample * n), 1), n)
        sampled = np.random.choice(n, m, replace=False)
        WL_dist = pairwise_distances(wl_mat[sampled], wl_mat, n_jobs=1)

    sorted_nodes = _celf(_wl2rknn(WL_dist, sampled_nodes=sampled, k=k)["rknn"])
    del WL_dist; gc.collect()

    og_sel = _rknn_sorted2budget_select_merged(sorted_nodes, train, target_size)
    og_nds: set = set()
    for idx in og_sel:
        tn = train[idx]; og_nds.update(_GLOBAL_NEIGHBORS_DICT[tn]); og_nds.add(tn)
    ogsize = len(og_nds)

    m = 0.9; upscale = 1 + m / (1 - m)
    sel = _rknn_sorted2budget_select_merged(sorted_nodes, train, target_size * upscale)
    if not sel: sel = sorted_nodes[:1] if sorted_nodes else []

    x_np = x_sl.numpy() if isinstance(x_sl, torch.Tensor) else x_sl
    y_np = data.y.numpy(); ts = set(train.tolist())

    G = nx.Graph()
    for oi in sel:
        tn = train[oi]; nbrs = _GLOBAL_NEIGHBORS_DICT[tn]
        G.add_node(tn); rc = _ADJ[[tn]]
        for nbr in nbrs:
            G.add_node(nbr)
            for nn2 in rc.multiply(_ADJ[[nbr]]).tocoo().col:
                G.add_edge(*sorted([nbr, nn2]))
            G.add_edge(*sorted([tn, nbr]))

    for nd in G.nodes:
        G.nodes[nd]["target"] = (nd in ts)
        G.nodes[nd]["x"]      = x_np[nd].tolist()
        G.nodes[nd]["y"]      = int(y_np[nd])

    pers = {nd: 0 for nd in G.nodes()}
    l = 1.0 / max(len(sel), 1)
    for nd in sel: pers[train[nd]] = l
    ppr   = sorted(nx.pagerank(G, personalization=pers).items(), key=lambda x: x[1])
    todel = max(0, len(ppr) - ogsize)
    for nd, _ in ppr[:todel]: G.remove_node(nd)

    data_sl = Data(x=x_sl, y=data.y, edge_index=data.edge_index)
    G = _match_distribution_bonsai(G, data_sl, train, sorted_nodes)
    merged = _attach_and_convert(G, x_np, y_np, ts, dataset_name, scaler)

    _print_budget_stats("BONSAI", merged, features_used,
                        size_full, target_size, target_size_frac,
                        N, nedges, F_orig, dtype)

    return {"data_syn": merged, "features_used": features_used,
            "condense_time": time.perf_counter() - t0}

# ═══════════════════════════════════════════════════════════════════════════════
#  HERALD 
# ═══════════════════════════════════════════════════════════════════════════════


def _herald_detect_heterophily(data, splits) -> float:
    """Verbatim from herald_condenser.py::detect_heterophily()."""
    y   = data.y.cpu().numpy().astype(np.int64)
    src = data.edge_index[0].cpu().numpy().astype(np.int64)
    dst = data.edge_index[1].cpu().numpy().astype(np.int64)
    ts  = set(splits["train"].tolist())
    cross = sum(1 for s, d in zip(src, dst) if s in ts and d in ts and y[s] != y[d])
    total = sum(1 for s, d in zip(src, dst) if s in ts and d in ts)
    return cross / max(total, 1)


def _herald_adaptive_weights(h,
                              alpha_base=0.4, beta_base=0.4, gamma_base=0.2):
    """Verbatim from herald_condenser.py::_adaptive_weights()."""
    t     = 1.0 / (1.0 + np.exp(-8.0 * (h - 0.4)))
    alpha = alpha_base + (1.0 - alpha_base - beta_base - gamma_base) * (1.0 - t)
    beta  = beta_base * t
    gamma = gamma_base * (0.5 + 0.5 * t)
    total = alpha + beta + gamma
    return alpha / total, beta / total, gamma / total


def _herald_prototype_score(X: np.ndarray, y: np.ndarray,
                             tm: np.ndarray, N: int) -> np.ndarray:
    """Verbatim from herald_condenser.py::_prototype_score()."""
    X_norm = X / np.linalg.norm(X, axis=1, keepdims=True).clip(min=1e-8)
    n_cls  = int(y[tm].max()) + 1
    cents  = np.zeros((n_cls, X.shape[1]), dtype=np.float32)
    for c in range(n_cls):
        mask = tm & (y == c)
        if mask.any(): cents[c] = X_norm[mask].mean(0)
    cents /= np.linalg.norm(cents, axis=1, keepdims=True).clip(min=1e-8)
    scores = np.zeros(N, dtype=np.float32)
    for v in range(N):
        c = int(y[v])
        if c < n_cls: scores[v] = float(np.dot(X_norm[v], cents[c]))
    scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)
    return scores


def _herald_boundary_score(src: np.ndarray, dst: np.ndarray,
                            y: np.ndarray, N: int) -> np.ndarray:
    """Verbatim from herald_condenser.py::_boundary_score()."""
    cross = np.zeros(N, dtype=np.float32)
    total = np.zeros(N, dtype=np.float32)
    for s, d in zip(src, dst):
        total[s] += 1.0
        if y[s] != y[d]: cross[s] += 1.0
    return cross / total.clip(min=1)


# def _herald_lid_score(X: np.ndarray, k: int = 10,
#                        batch_size: int = 512) -> np.ndarray:
#     """Verbatim from herald_condenser.py::_lid_score()."""
#     N  = X.shape[0]
#     Xn = X / np.linalg.norm(X, axis=1, keepdims=True).clip(min=1e-8)
#     k_ = min(k, N - 1)
#     lid = np.zeros(N, dtype=np.float32)
#     for s in range(0, N, batch_size):
#         e    = min(s + batch_size, N)
#         sim  = Xn[s:e] @ Xn.T
#         dist = 1.0 - sim
#         for ii in range(e - s): dist[ii, s + ii] = 1e9
#         knn_d    = np.partition(dist, k_, axis=1)[:, :k_]
#         r_max    = knn_d.max(axis=1).clip(min=1e-8)
#         log_rat  = np.log(knn_d.clip(min=1e-8) / r_max[:, None])
#         mean_log = log_rat.mean(axis=1).clip(max=-1e-8)
#         lid[s:e] = -1.0 / mean_log
#     lid = (lid - lid.min()) / (lid.max() - lid.min() + 1e-8)
#     return lid

def _herald_lid_score(
    X: np.ndarray,
    k: int = 10,
    batch_size: int = 512,
    sample_size: int = 512,
    seed: int = 42
) -> np.ndarray:
    """
    Scalable sampled-LID approximation using a fixed global
    reference set.

    Complexity:
        O(N * sample_size * F)

    instead of:
        O(N^2 * F)
    """

    N = X.shape[0]

    # Normalize for cosine distance
    Xn = X / np.linalg.norm(
        X, axis=1, keepdims=True
    ).clip(min=1e-8)

    # Fixed global reference set
    rng = np.random.default_rng(seed)

    m = min(sample_size, N)

    # Need at least k candidates excluding self
    if m <= k:
        m = min(k + 1, N)

    reference_idx = rng.choice(
        N,
        size=m,
        replace=False
    )

    Xref = Xn[reference_idx]

    k_ = min(k, m - 1)

    lid = np.zeros(N, dtype=np.float32)

    for s in range(0, N, batch_size):

        e = min(s + batch_size, N)

        # Compare only against m reference nodes
        # instead of all N nodes
        sim = Xn[s:e] @ Xref.T

        # Cosine distance
        dist = 1.0 - sim

        # Remove self-distance whenever the query is
        # itself contained in the reference set
        for ii in range(e - s):
            node_idx = s + ii

            pos = np.where(
                reference_idx == node_idx
            )[0]

            if len(pos) > 0:
                dist[ii, pos[0]] = 1e9

        # k nearest reference points
        knn_d = np.partition(
            dist,
            k_,
            axis=1
        )[:, :k_]

        # Distance of the kth nearest neighbor
        r_max = knn_d.max(axis=1).clip(min=1e-8)

        # Same LID-MLE calculation as your original code
        log_rat = np.log(
            knn_d.clip(min=1e-8)
            / r_max[:, None]
        )

        mean_log = (
            log_rat.mean(axis=1)
            .clip(max=-1e-8)
        )

        lid[s:e] = -1.0 / mean_log

    # Normalize LID scores exactly as before
    lid = (
        (lid - lid.min())
        / (lid.max() - lid.min() + 1e-8)
    )

    return lid


def _herald_sparsify_enrich(score, ranked, src, dst, adj_list,
                              feat_len, feat_multiplier, budget, L):
    """
    Score-ordered BFS expand with tight budget enforcement.

    Improvements over original herald_condenser.py:
      1. Roots are processed in HERALD score order (highest first).
      2. Budget is enforced strictly: a root's tree is added only if it fits
         within the remaining budget.  Early-exit after 100 consecutive misses
         (same heuristic as Bonsai's rknn_sorted2budget_select_merged).
      3. Edge cost uses incremental counting (only new edges), not full rescan.
         This reduces the O(N·E) ogsize loop to O(N·deg).

    Returns
    -------
    vs_set   : set of all node indices in condensed graph
    rs_set   : set of root (training) node indices
    """
    N  = len(adj_list)
    fl = (np.array([feat_len[i] for i in range(N)], dtype=np.float64)
          if isinstance(feat_len, dict)
          else np.asarray(feat_len, dtype=np.float64))

    # adjacency as set-of-neighbours for fast incremental edge counting
    nbr_sets = [set(adj_list[v]) for v in range(N)]

    def bfs(root):
        visited  = {int(root)}
        frontier = {int(root)}
        for _ in range(L):
            nf = set()
            for v in frontier:
                for nb in nbr_sets[v]:
                    if nb not in visited: visited.add(nb); nf.add(nb)
            frontier = nf
        return visited

    def incremental_cost(new_nodes, vs_set, current_cost):
        """Cost of adding new_nodes to existing vs_set."""
        feat_cost = sum(fl[v] for v in new_nodes) * feat_multiplier
        # Count new directed edges (each undirected edge appears twice in edge_index):
        # (a) edges from new node to any node already in combined set
        # (b) edges from existing node to new node (already counted in (a) via symmetry)
        # To avoid double-counting between two new nodes, count each directed edge once:
        combined = vs_set | new_nodes
        new_edges = 0
        for v in new_nodes:
            for nb in nbr_sets[v]:
                if nb in combined:   # nb already in graph (existing or new)
                    new_edges += 1   # one directed edge v→nb
        # new_edges is the total number of NEW directed edges; matches Bonsai's nedges*2
        return current_cost + (feat_cost * 2) + (new_edges * 2)

    vs_set: set  = set()
    rs_set: set  = set()
    used  = 0.0
    no_fit = 0

    for root in tqdm(ranked, desc="sparsify_enrich", ascii=True, ncols=120):
        ri = int(root)
        if ri in rs_set:
            continue
        if used >= budget:
            break
        tree     = bfs(ri)
        new_nds  = tree - vs_set
        if not new_nds:
            rs_set.add(ri)
            continue
        cb = incremental_cost(new_nds, vs_set, used)
        if cb <= budget:
            vs_set.update(new_nds); rs_set.add(ri); used = cb; no_fit = 0
        else:
            rs_set.add(ri); no_fit += 1
            if no_fit > 100:
                break

    return vs_set, rs_set


def _herald_compute_ogsize(ranked, adj_list, fl, feat_multiplier, budget, L, nbr_sets=None):
    """
    Compute ogsize: number of nodes in the condensed graph at the EXACT budget.

    Uses the same incremental cost logic as _herald_sparsify_enrich (no upscaling).
    This mirrors Bonsai's _rknn_sorted2budget_select_merged for ogsize.
    """
    N = len(adj_list)
    if nbr_sets is None:
        nbr_sets = [set(adj_list[v]) for v in range(N)]

    vs_og: set = set()
    used  = 0.0
    no_fit = 0

    for root in ranked:
        ri = int(root)
        if used >= budget:
            break
        # BFS tree
        tree = {ri}; frontier = {ri}
        for _ in range(L):
            nf = set()
            for v in frontier:
                for nb in nbr_sets[v]:
                    if nb not in tree: tree.add(nb); nf.add(nb)
            frontier = nf
        new_nds = tree - vs_og
        if not new_nds:
            continue
        feat_cost = sum(fl[v] for v in new_nds) * feat_multiplier
        combined  = vs_og | new_nds
        # count directed edges from new nodes into combined set (no nb==v guard needed
        # since self-loops are not present in typical graph datasets)
        new_edges = sum(1 for v in new_nds for nb in nbr_sets[v] if nb in combined)
        cb = used + (feat_cost * 2) + (new_edges * 2)
        if cb <= budget:
            vs_og.update(new_nds); used = cb; no_fit = 0
        else:
            no_fit += 1
            if no_fit > 100:
                break

    return max(len(vs_og), 1)


def _herald_pagerank_prune(vs_set, root_set, src, dst, ogsize):
    
    if len(vs_set) <= ogsize: return vs_set
    vs_arr = np.array(sorted(vs_set), dtype=np.int64); Nk = len(vs_arr)
    remap  = -np.ones(int(vs_arr.max()) + 1, dtype=np.int64)
    remap[vs_arr] = np.arange(Nk)
    valid  = np.isin(src, vs_arr) & np.isin(dst, vs_arr)
    ss = remap[src[valid]]; dd = remap[dst[valid]]
    A  = sp.csr_matrix((np.ones(len(ss)), (ss, dd)), shape=(Nk, Nk))
    A  = A + A.T; deg = np.asarray(A.sum(1)).flatten().clip(min=1)
    An = sp.diags(1.0 / deg) @ A
    rs_arr = np.array([remap[v] for v in root_set
                       if v < len(remap) and remap[v] >= 0])
    p0 = np.zeros(Nk)
    if len(rs_arr): p0[rs_arr] = 1.0 / len(rs_arr)
    alpha_pr = 0.85; pi = np.full(Nk, 1.0 / Nk)
    for _ in range(100):
        pi_new = (1 - alpha_pr) * p0 + alpha_pr * (An.T @ pi)
        s = pi_new.sum()
        if s > 0: pi_new /= s
        if np.abs(pi_new - pi).sum() < 1e-6: pi = pi_new; break
        pi = pi_new
    root_local   = set(rs_arr.tolist())
    non_root     = sorted([i for i in range(Nk) if i not in root_local],
                           key=lambda i: pi[i])
    remove_local = set(non_root[:max(0, Nk - ogsize)])
    return {vs_arr[i] for i in range(Nk) if i not in remove_local}


def _herald_match_distribution(vs_set, root_set, src, dst, y, tm,
                                 train_idx, power, train_rank):
    
    target_nodes = [v for v in vs_set if v in root_set and tm[v]]
    n_target = len(target_nodes)
    if n_target == 0: return vs_set, root_set
    y_train = y[train_idx]; n_cls = int(y.max()) + 1
    cc      = np.bincount(y_train, minlength=n_cls).astype(float)
    desired = np.round(cc / cc.sum() * n_target).astype(int)
    cands   = {c: [] for c in range(n_cls)}
    for v in train_rank: cands[int(y[v])].append(v)
    new_vs = set(vs_set); new_rs = set(root_set)
    for c in range(n_cls):
        cur = [v for v in target_nodes if int(y[v]) == c]
        lo  = int(0.99 * desired[c]); hi = int(1.01 * desired[c]) + 1; cnt = len(cur)
        if cnt < lo:
            short = lo - cnt
            for v in cands[c]:
                if short <= 0: break
                if v not in new_vs: new_vs.add(v); new_rs.add(v); short -= 1
        elif cnt > hi:
            removable = sorted([v for v in cur if v not in root_set],
                                key=lambda v: float(power[v]))
            for v in removable[:cnt - hi]: new_vs.discard(v)
    return new_vs, new_rs


def _herald_assemble(vs_set, root_set, src, dst, X, y, tm, N) -> Data:
    
    vs_arr = np.array(sorted(vs_set), dtype=np.int64); Nk = len(vs_arr)
    remap  = -np.ones(N, dtype=np.int64); remap[vs_arr] = np.arange(Nk)
    em     = np.array([s in vs_set and d in vs_set for s, d in zip(src, dst)])
    ss = remap[src[em]]; dd = remap[dst[em]]
    if ss.size > 0:
        cond_ei = torch.tensor(np.stack([ss, dd]), dtype=torch.long)
        cond_ei = to_undirected(cond_ei, num_nodes=Nk)
    else:
        cond_ei = torch.stack([torch.arange(Nk), torch.arange(Nk)])
    Xk      = torch.tensor(X[vs_arr],  dtype=torch.float32)
    yk      = torch.tensor(y[vs_arr],  dtype=torch.long)
    is_root = np.array([v in root_set for v in vs_arr], dtype=bool)
    target  = torch.tensor(tm[vs_arr] & is_root, dtype=torch.bool)
    out = Data(x=Xk, edge_index=cond_ei, y=yk)
    out.target = target
    return out


def condense_herald(data: Data, train: np.ndarray, splits: dict,
                    scaler, dataset_name: str, dtype: str,
                    target_size_frac: float, k: int = 5,
                    alpha: float | None = None,
                    beta:  float | None = None,
                    gamma: float | None = None,
                    lid_k: int = 10, wl_layers: int = 2,
                    verbose: bool = True,
                    pca_variance_ratio: float | None = None) -> dict:
    """
    HERALD — High-fidelity Exemplar Retrieval with Adaptive Landmark Distillation.

    Feature reduction:
      Runs Bonsai's WL+DT to get the feature count k*, then uses HERALD's
      joint selector (discriminativeness × density) to choose which k* features.
      This keeps feat_len and budget identical to Bonsai while selecting
      features that are both class-discriminative AND commonly active.

    Node scoring (the core novelty vs Bonsai):
      score = α·proto + β·boundary + γ·LID  (weights adapt from h)
      - proto    : how prototypical the node is of its class (exemplar)
      - boundary : proximity to class decision boundaries (landmark)
      - LID      : local intrinsic dimensionality — penalises redundant nodes
      Bonsai selects nodes by topological Rev-k-NN coverage in WL space.
      HERALD selects by information-theoretic criteria in feature space.

    Everything else (BFS expand, PPR prune, class rebalance, assemble)
    is identical to the Bonsai pipeline.
    """
    global _ADJ, _GLOBAL_NEIGHBORS_DICT, _GLOBAL_FEATS, _FEAT_LEN, _FEAT_MULTIPLIER
    SAINT = ["flickr", "ogbn-arxiv", "reddit"]
    t0 = time.perf_counter()

    # ── raw features ─────────────────────────────────────────────────
    x_in = (data.x_normed if hasattr(data, "x_normed") else data.x)
    y  = data.y.cpu().numpy().astype(np.int64)
    N, F = x_in.shape
    src = data.edge_index[0].cpu().numpy().astype(np.int64)
    dst = data.edge_index[1].cpu().numpy().astype(np.int64)
    nedges = data.edge_index.shape[1]

    train_idx = np.asarray(train)
    tm = np.zeros(N, dtype=bool); tm[train_idx] = True

    # ── Step 1: sparse adj ───────────────────────────────────────────
    rei, cei = data.edge_index.numpy()
    _ADJ = sp.csr_matrix((np.ones(len(rei)), (rei, cei)), shape=(N, N))
    _GLOBAL_NEIGHBORS_DICT = _build_neighborhood_dict_sparse()

    # ── Step 2: heterophily (used for scoring weights) ────────────────
    h = _herald_detect_heterophily(data, splits)

    # ── Step 3: feature selection ─────────────────────────────────────
    # Run WL+DT to get Bonsai's feature count k*.
    # Then use HERALD's joint scorer to select a different (better) set of
    # k* features — same count → same feat_len → same budget → fair compare.
    if verbose:
        print(f"  [HERALD] h={h:.3f} — joint feature selection (discriminativeness×density) …")
    wl = _compute_wl_representations(x_in, _ADJ)[train_idx]
    tmp_data = Data(x=x_in, y=data.y, edge_index=data.edge_index)
    _, bonsai_feats = _transform_features_with_tree(tmp_data, wl, train_idx)
    n_select = len(bonsai_feats)

    features_used = _herald_select_features(
        x_in, _ADJ, train_idx, data.y, h, top_k=n_select)

    if verbose:
        print(f"  [HERALD] {F} → {len(features_used)} features "
              f"(Bonsai DT count={n_select}, HERALD indices, h={h:.2f})")

    nfeats = len(features_used)

    # ── Step 4: reduced feature matrix (data.x never mutated) ────────
    x_sl = x_in[:, features_used]   # shape (N, nfeats)
    X    = x_sl.cpu().numpy().astype(np.float32)

    # ── Step 5: feat_len + budget — IDENTICAL to Bonsai ──────────────
    # size_full uses original F (same formula as Bonsai line 306-307).
    # FEAT_LEN uses x_sl (reduced), matching Bonsai lines 320-330.
    size_full = _size_bytes(N, nedges, F, dtype)
    budget    = float(f"{target_size_frac * size_full:.2f}")

    if dataset_name not in ["ogbn-arxiv", "reddit", "flickr", "pubmed"]:
        feat_multiplier = 1
        feat_len_arr    = np.array([float(x_sl[i].sum().item()) for i in range(N)],
                                    dtype=np.float64)
        feat_len        = feat_len_arr.tolist()
    elif dataset_name in ["pubmed", "flickr"]:
        feat_multiplier = 2 if dataset_name == "flickr" else 3
        feat_len_arr    = np.array([float((x_sl[i] != 0).sum().item()) for i in range(N)],
                                    dtype=np.float64)
        feat_len        = feat_len_arr.tolist()
    else:
        feat_multiplier = 2
        feat_len        = defaultdict(lambda: nfeats)
        feat_len_arr    = np.full(N, nfeats, dtype=np.float64)

    fl = feat_len_arr   # shape (N,) numpy array for fast budget checks

    # update module globals for consistency
    _FEAT_LEN        = feat_len
    _FEAT_MULTIPLIER = feat_multiplier
    _GLOBAL_FEATS    = x_sl

    # adj_list for BFS
    adj_list: list = [[] for _ in range(N)]
    for s, d in zip(src, dst): adj_list[s].append(int(d))

    # ── heterophily already computed above; derive adaptive weights ───
    if alpha is None or beta is None or gamma is None:
        ae, be, ge = _herald_adaptive_weights(h)
    else:
        tot = alpha + beta + gamma; ae, be, ge = alpha/tot, beta/tot, gamma/tot

    if verbose:
        print(f"  [HERALD] h={h:.3f}  α={ae:.3f}(proto) "
              f"β={be:.3f}(boundary) γ={ge:.3f}(LID)")

    # ── scoring ───────────────────────────────────────────────────────
    proto    = _herald_prototype_score(X, y, tm, N)
    boundary = _herald_boundary_score(src, dst, y, N)
    # lid      = _herald_lid_score(X, k=lid_k)
    lid = _herald_lid_score(X, k=lid_k, sample_size=512)
    score    = ae * proto + be * boundary + ge * lid
    score    = (score - score.min()) / (score.max() - score.min() + 1e-8)

    # ── rank train nodes by HERALD score (descending) ─────────────────
    ranked = train_idx[np.argsort(-score[train_idx])]

    # ── precompute neighbour sets once (shared by ogsize + BFS) ──────
    nbr_sets = [set(adj_list[v]) for v in range(N)]

    # ── ogsize: nodes at EXACT budget, score-ordered ──────────────────
    # Uses _herald_compute_ogsize which respects budget tightly and
    # early-exits (same heuristic as Bonsai), giving a meaningful
    # pruning target rather than N when budget is large.
    ogsize = _herald_compute_ogsize(
        ranked, adj_list, fl, feat_multiplier, budget, wl_layers, nbr_sets)
    if verbose:
        print(f"  [HERALD] ogsize={ogsize} (nodes at exact budget)")

    # ── BFS expand at upscaled budget (m=0.9 as in Bonsai) ────────────
    m         = 0.9
    budget_up = budget * (1 + m / (1 - m))

    # Always use internal implementation — our incremental cost model
    # must be consistent between ogsize and BFS expand.
    vs_set, root_set = _herald_sparsify_enrich(
        score, ranked, src, dst, adj_list,
        feat_len, feat_multiplier, budget_up, wl_layers)

    if not vs_set:
        top = int(ranked[0]) if len(ranked) else 0
        vs_set = {top}; root_set = {top}

    if verbose:
        print(f"  [HERALD] After BFS: {len(vs_set)}n / {len(root_set)} roots")

    # ── PageRank prune ─────────────────────────────────────────────────
    vs_set = _herald_pagerank_prune(vs_set, root_set, src, dst, ogsize)
    if verbose:
        print(f"  [HERALD] After PPR prune: {len(vs_set)}n  (target={ogsize})")

    # ── class rebalance ────────────────────────────────────────────────
    vs_set, root_set = _herald_match_distribution(
        vs_set, root_set, src, dst, y, tm, train_idx, score, list(ranked))
    if verbose:
        tgt = sum(1 for v in vs_set if v in root_set and tm[v])
        print(f"  [HERALD] After rebalance: {len(vs_set)}n / {tgt} target")

    # ── assemble — uses _herald_assemble(), NOT from_networkx ─────────
    cond_data = _herald_assemble(vs_set, root_set, src, dst, X, y, tm, N)

    if dataset_name in SAINT:
        cond_data.x = torch.tensor(
            scaler(cond_data.x.cpu().numpy()).astype(np.float32))
        cond_data.adj = _build_sparse_adj_normed(
            cond_data.edge_index, cond_data.x.shape[0])

    # ── optional PCA feature compression on the condensed graph ───────
    pca_obj      = None
    n_components = None
    if pca_variance_ratio is not None:
        x_syn_np = cond_data.x.cpu().numpy()
        x_reduced, pca_obj, n_components = apply_pca_compression(
            x_syn_np, variance_ratio=pca_variance_ratio)
        cond_data.x = torch.tensor(x_reduced, dtype=torch.float32)
        if verbose:
            print(f"  [HERALD] Extra PCA on adaptive-reduced feats: "
                  f"{nfeats} → {n_components} "
                  f"({pca_variance_ratio*100:.1f}% variance retained)")

    _print_budget_stats("HERALD", cond_data, features_used,
                        size_full, budget, target_size_frac,
                        N, nedges, F, dtype)

    return {"data_syn":       cond_data,
            "features_used":  features_used,
            "condense_time":  time.perf_counter() - t0,
            "homophily":      h,
            "pca":            pca_obj,
            "n_pca_components": n_components}


# ═══════════════════════════════════════════════════════════════════════════════
#  RANDOM baseline
# ═══════════════════════════════════════════════════════════════════════════════

def condense_random(data: Data, train: np.ndarray, splits: dict,
                    scaler, dataset_name: str, dtype: str,
                    target_size_frac: float, seed: int = 42) -> dict:
    global _ADJ, _GLOBAL_NEIGHBORS_DICT, _GLOBAL_FEATS, _FEAT_LEN, _FEAT_MULTIPLIER
    t0 = time.perf_counter()
    x  = data.x_normed if hasattr(data, "x_normed") else data.x
    xn = x.numpy().astype(np.float32) if isinstance(x, torch.Tensor) else x
    yn = data.y.numpy(); N, F = xn.shape
    nedges = data.edge_index.shape[1]; ts = set(train.tolist())
    rei, cei = data.edge_index.numpy()
    _ADJ = sp.csr_matrix((np.ones(len(rei)), (rei, cei)), shape=(N, N))
    _GLOBAL_NEIGHBORS_DICT = _build_neighborhood_dict_sparse()
    _FEAT_MULTIPLIER = 1 if dtype == "int" else 2
    _FEAT_LEN        = defaultdict(lambda: F)
    target   = target_size_frac * _size_bytes(N, nedges, F, dtype)
    shuffled = list(np.random.RandomState(seed).permutation(len(train)))
    sel = _rknn_sorted2budget_select_merged(shuffled, train, target)
    if not sel: sel = shuffled[:1] if shuffled else []
    G = nx.Graph()
    for oi in sel:
        tn = train[oi]; rc = _ADJ[[tn]]
        for nbr in _GLOBAL_NEIGHBORS_DICT[tn]:
            G.add_node(nbr)
            for nn2 in rc.multiply(_ADJ[[nbr]]).tocoo().col:
                G.add_edge(*sorted([nbr, nn2]))
            G.add_edge(*sorted([tn, nbr]))
        G.add_node(tn)
    d = _attach_and_convert(G, xn, yn, ts, dataset_name, scaler)
    return {"data_syn": d, "features_used": list(range(F)),
            "condense_time": time.perf_counter() - t0}


# ═══════════════════════════════════════════════════════════════════════════════
#  HERDING baseline  (Welling, 2009)
# ═══════════════════════════════════════════════════════════════════════════════

def condense_herding(data: Data, train: np.ndarray, splits: dict,
                     scaler, dataset_name: str, dtype: str,
                     target_size_frac: float) -> dict:
    global _ADJ, _GLOBAL_NEIGHBORS_DICT, _FEAT_LEN, _FEAT_MULTIPLIER
    t0  = time.perf_counter()
    x   = data.x_normed if hasattr(data, "x_normed") else data.x
    xn  = x.numpy().astype(np.float64) if isinstance(x, torch.Tensor) else x.astype(np.float64)
    yn  = data.y.numpy(); N, F = xn.shape
    nedges = data.edge_index.shape[1]; ts = set(train.tolist())
    rei, cei = data.edge_index.numpy()
    _ADJ = sp.csr_matrix((np.ones(len(rei)), (rei, cei)), shape=(N, N))
    _GLOBAL_NEIGHBORS_DICT = _build_neighborhood_dict_sparse()
    _FEAT_MULTIPLIER = 1 if dtype == "int" else 2
    _FEAT_LEN        = defaultdict(lambda: F)
    target   = target_size_frac * _size_bytes(N, nedges, F, dtype)
    n_budget = max(1, int(target / (F * _FEAT_MULTIPLIER * 2)))
    classes  = np.unique(yn[train]); per_cls = max(1, n_budget // len(classes))
    sel_nds: list[int] = []
    for cls in classes:
        mask = np.where(yn[train] == cls)[0]; cnds = train[mask]
        cf = xn[cnds]; cent = cf.mean(0); cum = np.zeros(F)
        rem = list(range(len(cnds))); ch = []
        for _ in range(min(per_cls, len(cnds))):
            if not rem: break
            tv = cent * (len(ch) + 1) - cum
            sim = cf[rem] @ tv; bi = rem.pop(int(np.argmax(sim)))
            ch.append(bi); cum += cf[bi]
        sel_nds.extend(int(cnds[i]) for i in ch)
    G = nx.Graph()
    for nd in sel_nds:
        rc = _ADJ[[nd]]
        for nbr in _GLOBAL_NEIGHBORS_DICT[nd]:
            G.add_node(nbr)
            for nn2 in rc.multiply(_ADJ[[nbr]]).tocoo().col:
                G.add_edge(*sorted([nbr, nn2]))
            G.add_edge(*sorted([nd, nbr]))
        G.add_node(nd)
    if G.number_of_nodes() == 0 and sel_nds: G.add_node(sel_nds[0])
    d = _attach_and_convert(G, xn.astype(np.float32), yn, ts, dataset_name, scaler)
    return {"data_syn": d, "features_used": list(range(F)),
            "condense_time": time.perf_counter() - t0}


# ═══════════════════════════════════════════════════════════════════════════════
#  EXTERNAL LOADERS  (GCond / GDEM / GCSR)
# ═══════════════════════════════════════════════════════════════════════════════

def load_gcond_syn(dataset_name, target_frac, synthetic_root, scaler) -> dict:
    path  = Path(synthetic_root) / f"{dataset_name}-{target_frac}"
    adj   = torch.load(path / "adj_1.pt",    map_location="cpu")
    feat  = torch.load(path / "feat_1.pt",   map_location="cpu")
    label = torch.load(path / "labels_1.pt", map_location="cpu")
    feat  = torch.tensor(scaler(feat.cpu().numpy()).astype(np.float32))
    ajc   = sp.csr_matrix(adj.numpy()).tocoo(); r, c = ajc.row, ajc.col
    ew    = torch.tensor(np.array(ajc.tocsr()[r, c]).flatten())[:, None]
    ei    = torch.stack([torch.tensor(r), torch.tensor(c)]).long()
    d     = Data(x=feat, edge_index=ei, edge_attr=ew, y=label,
                 adj=_build_sparse_adj_normed(ei, feat.shape[0]))
    return {"data_syn": d, "features_used": list(range(feat.shape[1]))}


def load_gdem_syn(dataset_name, target_frac, synthetic_root, scaler) -> dict:
    exp  = 0; path = Path(synthetic_root) / f"{dataset_name}-{target_frac}"
    ev   = torch.load(path / f"eigenvals_syn_{exp}.pt", map_location="cpu")
    evc  = torch.load(path / f"eigenvecs_syn_{exp}.pt", map_location="cpu")
    feat = torch.load(path / f"feat_{exp}.pt",          map_location="cpu")
    lbl  = torch.load(path / f"label_{exp}.pt",         map_location="cpu")
    feat = torch.tensor(scaler(feat.cpu().numpy()).astype(np.float32))
    n    = feat.shape[0]
    adj  = sp.csr_matrix((torch.eye(n) - evc @ torch.diag(ev) @ evc.T).numpy()).tocoo()
    r, c = adj.row, adj.col
    ew   = torch.tensor(np.array(adj.tocsr()[r, c]).flatten())[:, None]
    ei   = torch.stack([torch.tensor(r), torch.tensor(c)]).long()
    d    = Data(x=feat, edge_index=ei, edge_attr=ew, y=lbl,
                adj=_build_sparse_adj_normed(ei, n))
    return {"data_syn": d, "features_used": list(range(feat.shape[1]))}


def load_gcsr_syn(dataset_name, target_frac, synthetic_root, scaler) -> dict:
    path  = Path(synthetic_root) / f"{dataset_name}-{target_frac}"
    adj   = torch.load(path / "adj.pt",   map_location="cpu")
    feat  = torch.load(path / "feat.pt",  map_location="cpu")
    label = torch.load(path / "label.pt", map_location="cpu")
    feat  = torch.tensor(scaler(feat.cpu().numpy()).astype(np.float32))
    ajc   = sp.csr_matrix(adj.numpy()).tocoo(); r, c = ajc.row, ajc.col
    ew    = torch.tensor(np.array(ajc.tocsr()[r, c]).flatten())[:, None]
    ei    = torch.stack([torch.tensor(r), torch.tensor(c)]).long()
    d     = Data(x=feat, edge_index=ei, edge_attr=ew, y=label,
                 adj=_build_sparse_adj_normed(ei, feat.shape[0]))
    return {"data_syn": d, "features_used": list(range(feat.shape[1]))}


# ── dispatcher ────────────────────────────────────────────────────────────────

CONDENSER_FN = {
    "bonsai":  condense_bonsai,
    "herald":  condense_herald,
    "random":  condense_random,
    "herding": condense_herding,
}

EXTERNAL_LOADER = {
    "gcond": load_gcond_syn,
    "gdem":  load_gdem_syn,
    "gcsr":  load_gcsr_syn,
}

ALL_CONDENSERS = list(CONDENSER_FN) + list(EXTERNAL_LOADER)

# ── GDEM as an inline (budget-aware) condenser ────────────────────────
try:
    from gdem_condenser import condense_gdem
    CONDENSER_FN["gdem_inline"] = condense_gdem
    ALL_CONDENSERS.append("gdem_inline")
except ImportError:
    pass   # gdem_condenser.py not on path; skip
