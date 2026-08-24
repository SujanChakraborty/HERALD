#!/usr/bin/env python3
"""
benchmark.py  –  End-to-end graph condensation benchmark.

Covers:
  Datasets    : Cora, CiteSeer, PubMed, Flickr, ogbn-arxiv, Reddit (homophilic)
                Roman-empire, Amazon-ratings, Chameleon, Squirrel (heterophilic)
  Condensers  : bonsai, herald, random, herding  (+ gcond/gdem/gcsr if pre-saved)
  Classifiers : GCN, GAT, GIN, GCN_inductive, H2GCN

Usage
-----
  # full run
  python benchmark.py --datasets cora roman-empire --condensers bonsai herald \
      --gnns GCN H2GCN --fracs 0.005 0.01 --seeds 0 1 2 3 4 --nepochs 200

  # load previously saved condensed data instead of condensing
  python benchmark.py --load_condensed --save_dir saved_condensed ...

  # external condensers (need pre-run official scripts)
  python benchmark.py --condensers gcond --gcond_dir saved_gcond ...
"""

from __future__ import annotations
import argparse
import copy
import gc
import json
import os
import sys
import time
import traceback
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import Data
from tqdm import tqdm

from models import build_model, MODELS
from condensers import (
    CONDENSER_FN, EXTERNAL_LOADER, ALL_CONDENSERS,
    _build_sparse_adj_normed, _size_bytes,
)


# ─────────────────────────────────────────────────────────────
# DATASETS
# ─────────────────────────────────────────────────────────────

PLANETOID = ["cora", "citeseer", "pubmed"]
OGB_DATASETS = ["ogbn-arxiv"]
PYG_EXTRA = ["flickr", "reddit"]
HETERO_DATASETS = ["roman-empire", "amazon-ratings", "chameleon", "squirrel"]
SAINT_DATASETS = ["flickr", "reddit", "ogbn-arxiv"]

ALL_DATASETS = PLANETOID + OGB_DATASETS + PYG_EXTRA + HETERO_DATASETS

DATASET_DTYPE = {
    "cora": "int", "citeseer": "int", "pubmed": "float",
    "flickr": "float", "reddit": "float", "ogbn-arxiv": "float",
    "roman-empire": "float", "amazon-ratings": "float",
    "chameleon": "float", "squirrel": "float",
}


# ─────────────────────────────────────────────────────────────
# SPLITS
# ─────────────────────────────────────────────────────────────

def make_splits(nnodes: int, seed: int = 42) -> dict:
    """
    Fixed 60/20/20 train/val/test split.
    seed controls the train/test split (random_state=seed).
    The train/val sub-split uses the same seed for consistency.
    Splits are fixed across all training seeds — only model weights vary.
    This matches the Bonsai paper's experimental setup (Appendix B).
    """
    train_, test = train_test_split(range(nnodes), test_size=0.2, random_state=seed)
    rng = np.random.RandomState(seed=seed)   # use same seed, not hardcoded 0
    idx_train = rng.choice(train_, size=int(0.7 * len(train_)), replace=False)
    idx_val = list(set(range(nnodes)) - set(idx_train).union(set(test)))
    return {
        "train": np.array(idx_train),
        "val":   np.array(idx_val),
        "test":  np.array(test)
    }


# ─────────────────────────────────────────────────────────────
# LOADER
# ─────────────────────────────────────────────────────────────

def load_dataset(name: str, root: str = "datasets") -> dict:
    name = name.lower()

    # ── PLANETOID ─────────────────────────────────────────────
    if name in PLANETOID:
        from torch_geometric.datasets import Planetoid

        data = Planetoid(root=root, name=name)[0]
        splits = make_splits(data.x.shape[0])

        return {
            "data": data,
            "scaler": lambda x: x,
            "splits": splits,
            "dtype": DATASET_DTYPE[name]
        }

    # ── OGB (ARXIV) ───────────────────────────────────────────
    if name in OGB_DATASETS:
        from ogb.nodeproppred import PygNodePropPredDataset
        from torch_geometric.utils import to_undirected

        dataset = PygNodePropPredDataset(name=name, root=root)
        data = dataset[0]

        # make undirected
        data.edge_index = to_undirected(data.edge_index)

        # normalize features
        feats = data.x.numpy().astype(np.float32)
        sc = StandardScaler().fit(feats)
        data.x = torch.tensor(sc.transform(feats))

        split_idx = dataset.get_idx_split()

        splits = {
            "train": split_idx["train"].numpy(),
            "val": split_idx["valid"].numpy(),
            "test": split_idx["test"].numpy()
        }

        return {
            "data": data,
            "scaler": sc.transform,
            "splits": splits,
            "dtype": DATASET_DTYPE[name]
        }

    # ── PYG EXTRA (FLICKR / REDDIT) ───────────────────────────
    if name in PYG_EXTRA:
        from torch_geometric.datasets import Flickr, Reddit

        dataset_cls = Flickr if name == "flickr" else Reddit
        data = dataset_cls(root=root)[0]

        feats = data.x.numpy().astype(np.float32)
        sc = StandardScaler().fit(feats)
        data.x = torch.tensor(sc.transform(feats))

        splits = make_splits(data.x.shape[0])

        return {
            "data": data,
            "scaler": sc.transform,
            "splits": splits,
            "dtype": DATASET_DTYPE[name]
        }

    # ── HETEROPHILIC ──────────────────────────────────────────
    if name in HETERO_DATASETS:
        from torch_geometric.datasets import HeterophilousGraphDataset, WikipediaNetwork

        canon = {
            "roman-empire":   ("hetero", "Roman-empire"),
            "amazon-ratings": ("hetero", "Amazon-ratings"),
            "chameleon":      ("wiki", "chameleon"),
            "squirrel":       ("wiki", "squirrel"),
        }[name]

        if canon[0] == "hetero":
            ds = HeterophilousGraphDataset(root=root, name=canon[1])
        else:
            ds = WikipediaNetwork(root=root, name=canon[1], geom_gcn_preprocess=True)

        data = ds[0]

        if data.y.dim() > 1:
            data.y = data.y.squeeze(-1)

        feats = data.x.numpy().astype(np.float32)
        sc = StandardScaler().fit(feats)
        data.x = torch.tensor(sc.transform(feats))

        splits = make_splits(data.x.shape[0])

        return {
            "data": data,
            "scaler": sc.transform,
            "splits": splits,
            "dtype": DATASET_DTYPE[name]
        }

    raise ValueError(f"Unknown dataset '{name}'. Choose from {ALL_DATASETS}")


# ═══════════════════════════════════════════════════════════════════════════════
#  TRAINING  (mirrors official train_bonsai.py / train_gdem.py exactly)
# ═══════════════════════════════════════════════════════════════════════════════

INDUCTIVE_MODELS = {"GCN_inductive"}


def train_and_eval(model: nn.Module, model_name: str,
                   data_full: Data, data_syn: Data,
                   splits: dict, nepochs: int,
                   lr: float, weight_decay: float,
                   device: torch.device) -> float:
    """
    Train on data_syn, validate + test on data_full.
    Uses nll_loss for GCN_inductive (sparse adj), CrossEntropy for PyG models.
    Mirrors the training backends in the official train_bonsai.py.
    """
    inductive = model_name in INDUCTIVE_MODELS
    model     = model.to(device)
    opt       = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn   = (lambda o, y: F.nll_loss(o, y)) if inductive else nn.CrossEntropyLoss()

    val, test = splits["val"], splits["test"]

    # ── move data to device ────────────────────────────────────────────
    x_syn  = data_syn.x.to(device)
    y_syn  = data_syn.y.to(device)
    x_full = (data_full.x_normed if hasattr(data_full, "x_normed")
              else data_full.x).to(device)
    y_full = data_full.y.to(device)

    if inductive:
        # build normalised sparse adj for both graphs
        adj_syn = (data_syn.adj.to(device) if hasattr(data_syn, "adj")
                   else _build_sparse_adj_normed(data_syn.edge_index,
                                                  x_syn.shape[0]).to(device))
        adj_full = (data_full.adj.to(device) if hasattr(data_full, "adj")
                    else _build_sparse_adj_normed(data_full.edge_index,
                                                   x_full.shape[0]).to(device))
    else:
        ei_syn  = data_syn.edge_index.to(device)
        ei_full = data_full.edge_index.to(device)

    # target mask
    if hasattr(data_syn, "target") and data_syn.target is not None:
        target = data_syn.target.to(device)
    else:
        target = torch.ones(x_syn.shape[0], dtype=torch.bool, device=device)

    best_val, best_w = 0.0, copy.deepcopy(model.state_dict())

    for epoch in range(nepochs):
        model.train()
        out = (model(x_syn, adj_syn) if inductive
               else model(x_syn, ei_syn))
        if target.dtype == torch.bool:
            loss = loss_fn(out[target], y_syn[target])
        else:
            loss = loss_fn(out[target.long()], y_syn[target.long()])
        opt.zero_grad(); loss.backward(); opt.step()

        model.eval()
        with torch.no_grad():
            out_f = (model(x_full, adj_full) if inductive
                     else model(x_full, ei_full))
            preds   = out_f[val].argmax(1).cpu().numpy()
            val_acc = accuracy_score(y_full[val].cpu().numpy(), preds)
            if val_acc > best_val:
                best_val = val_acc
                best_w   = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_w)
    model.eval()
    with torch.no_grad():
        out_f = (model(x_full, adj_full) if inductive
                 else model(x_full, ei_full))
        preds = out_f[test].argmax(1).cpu().numpy()
    return accuracy_score(y_full[test].cpu().numpy(), preds)


# ═══════════════════════════════════════════════════════════════════════════════
#  RESULTS  I/O
# ═══════════════════════════════════════════════════════════════════════════════

def save_results(rows: list, results_dir: str) -> pd.DataFrame:
    Path(results_dir).mkdir(parents=True, exist_ok=True)
    ts  = time.strftime("%Y%m%d_%H%M%S")
    df  = pd.DataFrame(rows)
    csv = Path(results_dir) / f"results_{ts}.csv"
    jsf = Path(results_dir) / f"results_{ts}.json"
    df.to_csv(csv, index=False)
    with open(jsf, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"\n✓ {csv}\n✓ {jsf}")
    return df


def print_table(df: pd.DataFrame):
    cols = ["acc_mean", "acc_std", "condense_time", "sr_percent",
            "cond_nodes", "cond_edges", "cond_feats"]
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # accuracy pivot (main table)
    pivot = df.pivot_table(
        index=["dataset", "frac", "condenser"],
        columns="gnn", values="acc_mean", aggfunc="first",
    )
    print("\n" + "=" * 90)
    print("RESULTS  (mean test accuracy %)")
    print("=" * 90)
    print(pivot.to_string())
    print("=" * 90)

    # graph size summary (one row per dataset×frac×condenser, GNN-independent)
    size_cols = ["dataset", "frac", "condenser",
                 "sr_percent", "cond_nodes", "cond_edges", "cond_feats",
                 "condense_time"]
    size_cols = [c for c in size_cols if c in df.columns]
    size_df = df[size_cols].drop_duplicates(
        subset=["dataset", "frac", "condenser"]).sort_values(
        ["dataset", "frac", "condenser"])
    print("\n" + "=" * 90)
    print("GRAPH SIZE STATS  (Sr% = dense condensed size / dense full size × 100)")
    print("=" * 90)
    print(size_df.to_string(index=False))
    print("=" * 90)


# ═══════════════════════════════════════════════════════════════════════════════
#  ARGUMENT PARSING
# ═══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    p.add_argument("--datasets",   nargs="+", default=["cora", "citeseer", "pubmed", "roman-empire", "amazon-ratings", "chameleon", "squirrel"],
                   choices=ALL_DATASETS)
    p.add_argument("--condensers", nargs="+",
                   default=["herald", "bonsai", "random", "herding"],
                   choices=ALL_CONDENSERS)
    p.add_argument("--gnns", nargs="+",
                   default=["GCN", "GAT", "GIN", "H2GCN"],
                   choices=list(MODELS))
    p.add_argument("--fracs",  nargs="+", type=float, default=[0.0001, 0.005, 0.01, 0.03])
    p.add_argument("--seeds",  nargs="+", type=int,   default=[0, 1, 2, 3, 4])
    p.add_argument("--nepochs",type=int,  default=200)
    p.add_argument("--lr",     type=float,default=1e-3)
    p.add_argument("--wd",     type=float,default=5e-4,  dest="weight_decay")
    p.add_argument("--hidim",  type=int,  default=128)
    p.add_argument("--data_root",  default="datasets")
    p.add_argument("--results_dir",default="results")

    # condense I/O
    p.add_argument("--save_condensed", action="store_true")
    p.add_argument("--save_dir",       default="saved_condensed")
    p.add_argument("--load_condensed", action="store_true",
                   help="Skip condensation; load from --save_dir")

    # external condenser dirs
    p.add_argument("--gcond_dir", default="saved_gcond")
    p.add_argument("--gdem_dir",  default="saved_gdem")
    p.add_argument("--gcsr_dir",  default="saved_gcsr")

    # Bonsai params
    p.add_argument("--k",           type=int,   default=5)
    p.add_argument("--frac_sample", type=float, default=1.0,
                   help="Sampling fraction for Rev-k-NN (1.0=exact)")

    # Herald params
    p.add_argument("--herald_alpha", type=float, default=None,
                   help="Override adaptive scoring weight α (None=auto from h)")
    p.add_argument("--pca_variance", type=float, default=1.0,
                   help="Extra PCA compression after HERALD feature selection "
                        "(e.g. 0.90 = keep 90%% variance). Default 1.0 = disabled.")
    

    # misc
    p.add_argument("--device",     default=None)
    p.add_argument("--skip_full",  action="store_true")
    p.add_argument("--verbose",    action="store_true")

    return p.parse_args()


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN LOOP
# ═══════════════════════════════════════════════════════════════════════════════

def run_seeds(model_name, indim, nclass, hidim,
              data_full, data_syn, splits,
              nepochs, lr, wd, device, seeds) -> tuple[float, float]:
    accs = []
    for seed in seeds:
        # Fully re-seed for each training run
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        model = build_model(model_name, indim, nclass, hidim)
        try:
            acc = train_and_eval(model, model_name, data_full, data_syn,
                                  splits, nepochs, lr, wd, device)
            accs.append(acc)
            print(f"{acc*100:.2f}", end=" ", flush=True)
        except RuntimeError as e:
            if "nondeterministic" in str(e).lower():
                # Specific PyG ops (e.g. scatter_add on CUDA) raise under
                # deterministic mode. Retry with it disabled — training
                # reproducibility is still controlled by the per-seed seeding.
                torch.use_deterministic_algorithms(False)
                try:
                    torch.manual_seed(seed)
                    np.random.seed(seed)
                    model = build_model(model_name, indim, nclass, hidim)
                    acc = train_and_eval(model, model_name, data_full, data_syn,
                                         splits, nepochs, lr, wd, device)
                    accs.append(acc)
                    print(f"{acc*100:.2f}", end=" ", flush=True)
                except Exception as e2:
                    print(f"[err:{seed}:{e2}]", end=" ", flush=True)
                    accs.append(float("nan"))
            else:
                print(f"[err:{seed}:{e}]", end=" ", flush=True)
                accs.append(float("nan"))
        except Exception as e:
            print(f"[err:{seed}:{e}]", end=" ", flush=True)
            accs.append(float("nan"))
    valid = [a for a in accs if not np.isnan(a)]
    if not valid:
        return float("nan"), float("nan")
    arr = np.array(valid)
    return float(arr.mean()), float(arr.std())


def main():
    args   = parse_args()
    device = torch.device(args.device if args.device
                          else ("cuda" if torch.cuda.is_available() else "cpu"))
    rows   = []

    # ── global determinism ────────────────────────────────────────────────
    # Seed everything before any computation so condensation is reproducible.
    # The condensed graph must be identical across runs (it is computed once,
    # before the per-seed training loop).
    GLOBAL_SEED = 42
    torch.manual_seed(GLOBAL_SEED)
    torch.cuda.manual_seed_all(GLOBAL_SEED)
    np.random.seed(GLOBAL_SEED)
    # Make PyTorch ops deterministic where possible (fixes scatter_add_ in GCN).
    # Note: this may slightly slow down training on GPU.
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass   # older PyTorch versions don't support this
    import os
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    Path(args.results_dir).mkdir(parents=True, exist_ok=True)
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)
    Path("logs").mkdir(exist_ok=True)

    print(f"Device: {device} | datasets={args.datasets} | "
          f"condensers={args.condensers} | gnns={args.gnns}")

    for ds_name in args.datasets:
        print(f"\n{'━'*70}\n  Dataset: {ds_name}\n{'━'*70}")

        try:
            dataset = load_dataset(ds_name, root=args.data_root)
        except Exception as e:
            print(f"  [SKIP] {e}"); continue

        data   = dataset["data"]
        scaler = dataset["scaler"]
        splits = dataset["splits"]
        dtype  = dataset["dtype"]
        train  = splits["train"]
        nnodes = data.x.shape[0]
        nclass = int(data.y.max().item()) + 1
        nedges_full = data.edge_index.shape[1]
        nfeats_full = data.x.shape[1]
        print(f"  Full graph: {nnodes:,} nodes | {nedges_full:,} edges | "
              f"{nfeats_full} feats | {nclass} classes | "
              f"train={len(train):,} val={len(splits['val']):,} "
              f"test={len(splits['test']):,}")

        # ── snapshot original features ONCE, before any condenser call ─
        # condensers must NOT mutate data.x (fixed in condensers.py), but
        # we keep a defensive snapshot here so that x_src[:, features_used]
        # always indexes into the original full-dimension tensor.
        _x_orig        = (data.x_normed if hasattr(data, "x_normed") else data.x).clone()
        _x_normed_orig = data.x_normed.clone() if hasattr(data, "x_normed") else None
        _nfeats_full   = _x_orig.shape[1]

        # ── full-dataset baseline ──────────────────────────────────────
        if not args.skip_full:
            print("  [Full baseline]")
            nfeats  = _nfeats_full
            hidim_f = 1024 if ds_name in SAINT_DATASETS else args.hidim
            # build a syn-compatible Data with a target mask over train nodes
            tmask = torch.zeros(nnodes, dtype=torch.bool)
            tmask[train] = True
            data_syn_full = Data(x=_x_orig, y=data.y,
                                  edge_index=data.edge_index, target=tmask)
            if hasattr(data, "adj"):
                data_syn_full.adj = data.adj
            for gnn_name in args.gnns:
                m, s = run_seeds(gnn_name, nfeats, nclass, hidim_f,
                                  data, data_syn_full, splits,
                                  args.nepochs, args.lr, args.weight_decay,
                                  device, args.seeds)
                rows.append(dict(dataset=ds_name, condenser="full", frac="full",
                                 gnn=gnn_name, acc_mean=round(m*100,2),
                                 acc_std=round(s*100,2),
                                 condense_time=0.0, train_time=0.0,
                                 cond_nodes=nnodes, cond_edges=nedges_full,
                                 cond_feats=_nfeats_full, sr_percent=100.0))
                print(f"    full/{gnn_name}: {m*100:.2f}±{s*100:.2f}")

        # ── per condenser × frac ───────────────────────────────────────
        for condenser_name in args.condensers:
            for frac in args.fracs:
                print(f"\n  Condenser={condenser_name}  frac={frac}")

                # ── condense or load ───────────────────────────────────
                save_path = (Path(args.save_dir) /
                             f"{ds_name}-{condenser_name}-{frac}.pt")
                t_cond = 0.0
                dsyn   = None   # set after condensation or load; used in rows.append

                if args.load_condensed and save_path.exists():
                    try:
                        saved    = torch.load(save_path, map_location="cpu")
                        result   = {"data_syn":        saved["data_syn"],
                                    "features_used":   saved["features_used"],
                                    "pca":             saved.get("pca", None),
                                    "n_pca_components": saved.get("n_pca_components", None),
                                    "condense_time":   0.0}
                        t_cond   = 0.0
                        print(f"    Loaded from {save_path}")
                    except Exception as e:
                        print(f"    [load error] {e}"); continue

                elif condenser_name in EXTERNAL_LOADER:
                    ext_dirs = {"gcond": args.gcond_dir,
                                "gdem":  args.gdem_dir,
                                "gcsr":  args.gcsr_dir}
                    ext_dir = ext_dirs[condenser_name]
                    try:
                        result = EXTERNAL_LOADER[condenser_name](
                            ds_name, frac, ext_dir, scaler)
                        result["condense_time"] = 0.0
                        print(f"    Loaded {condenser_name} from {ext_dir}")
                    except FileNotFoundError:
                        print(f"    [SKIP] {condenser_name} pre-saved data not "
                              f"found at {ext_dir}. Run the official script first.")
                        continue
                    except Exception as e:
                        print(f"    [{condenser_name} load error] {e}"); continue

                else:
                    # Re-seed before condensation so the condensed graph is
                    # identical regardless of what ran before (other condensers,
                    # other datasets). Condensation is deterministic given a
                    # fixed numpy/torch state.
                    torch.manual_seed(GLOBAL_SEED)
                    torch.cuda.manual_seed_all(GLOBAL_SEED)
                    np.random.seed(GLOBAL_SEED)

                    # run condenser
                    cond_fn = CONDENSER_FN[condenser_name]
                    kwargs  = dict(data=data, train=train, splits=splits,
                                   scaler=scaler, dataset_name=ds_name,
                                   dtype=dtype, target_size_frac=frac)
                    if condenser_name == "bonsai":
                        kwargs.update(k=args.k, frac_to_sample=args.frac_sample)
                    elif condenser_name == "herald":
                        kwargs.update(k=args.k, alpha=args.herald_alpha,
                                      pca_variance_ratio=(
                                          None if args.pca_variance >= 1.0
                                          else args.pca_variance))
                    try:
                        result = cond_fn(**kwargs)
                        t_cond = result["condense_time"]
                        n_pca_info = (f"  pca_feats={result['n_pca_components']}"
                                      if result.get("n_pca_components") is not None
                                      else "")
                        dsyn = result["data_syn"]
                        print(f"    Condensed in {t_cond:.1f}s  "
                              f"nodes={dsyn.x.shape[0]}  "
                              f"edges={dsyn.edge_index.shape[1]}  "
                              f"feats={dsyn.x.shape[1]}"
                              f"{n_pca_info}")
                    except Exception as e:
                        print(f"    [condenser error] {e}")
                        traceback.print_exc()
                        for gnn_name in args.gnns:
                            rows.append(dict(dataset=ds_name,
                                             condenser=condenser_name, frac=frac,
                                             gnn=gnn_name, acc_mean="ERR",
                                             acc_std="ERR",
                                             condense_time="ERR", train_time="ERR"))
                        continue

                if args.save_condensed and not args.load_condensed \
                        and condenser_name not in EXTERNAL_LOADER:
                    torch.save({"data_syn":        result["data_syn"],
                                "features_used":   result["features_used"],
                                "pca":             result.get("pca", None),
                                "n_pca_components": result.get("n_pca_components", None)},
                                save_path)

                data_syn      = result["data_syn"]
                dsyn          = data_syn   # alias used in rows.append for size stats
                features_used = result["features_used"]
                pca_obj       = result.get("pca", None)
                n_pca_comp    = result.get("n_pca_components", None)

                # ── build full-graph eval data ─────────────────────────
                if pca_obj is not None:
                    # PCA was fitted on x[features_used] inside the condenser,
                    # so we must slice to features_used FIRST, then transform.
                    x_full_np  = _x_orig[:, features_used].numpy().astype(np.float32)
                    x_eval_np  = pca_obj.transform(x_full_np).astype(np.float32)
                    x_eval     = torch.tensor(x_eval_np)
                    print(f"    [PCA] {_nfeats_full} → {len(features_used)} (DT) "
                          f"→ {n_pca_comp} (PCA) features")
                else:
                    x_eval = _x_orig[:, features_used]

                data_eval = Data(x=x_eval, y=data.y,
                                  edge_index=data.edge_index)
                if hasattr(data, "adj"):
                    data_eval.adj = data.adj
                if _x_normed_orig is not None:
                    if pca_obj is not None:
                        xn_full_np = _x_normed_orig[:, features_used].numpy().astype(np.float32)
                        data_eval.x_normed = torch.tensor(
                            pca_obj.transform(xn_full_np).astype(np.float32))
                    else:
                        data_eval.x_normed = _x_normed_orig[:, features_used]

                nfeats = x_eval.shape[1]
                hidim  = 1024 if ds_name in SAINT_DATASETS else args.hidim

                # ── train each GNN ─────────────────────────────────────
                for gnn_name in args.gnns:
                    print(f"    {gnn_name} ... ", end="", flush=True)
                    t_train_start = time.perf_counter()
                    m, s = run_seeds(gnn_name, nfeats, nclass, hidim,
                                      data_eval, data_syn, splits,
                                      args.nepochs, args.lr,
                                      args.weight_decay, device, args.seeds)
                    t_train = time.perf_counter() - t_train_start
                    print(f"→ {m*100:.2f}±{s*100:.2f}  ({t_train:.0f}s)")

                    rows.append(dict(
                        dataset=ds_name, condenser=condenser_name, frac=frac,
                        gnn=gnn_name,
                        acc_mean=round(m*100, 2), acc_std=round(s*100, 2),
                        condense_time=round(t_cond, 2),
                        train_time=round(t_train/len(args.seeds), 2),
                        n_pca_components=n_pca_comp,
                        # graph size stats (for paper Table 4)
                        cond_nodes=dsyn.x.shape[0] if dsyn is not None else None,
                        cond_edges=dsyn.edge_index.shape[1] if dsyn is not None else None,
                        cond_feats=dsyn.x.shape[1] if dsyn is not None else None,
                        sr_percent=round(
                            _size_bytes(dsyn.x.shape[0],
                                        dsyn.edge_index.shape[1],
                                        dsyn.x.shape[1], dtype) /
                            max(_size_bytes(nnodes,
                                            nedges_full,
                                            _nfeats_full, dtype), 1) * 100, 3)
                                        if dsyn is not None else None,
                    ))

                # interim save after each (condenser × frac)
                _interim = Path(args.results_dir) / f"interim_{ds_name}.json"
                with open(_interim, "w") as f:
                    json.dump(rows, f, indent=2)

                gc.collect()

    df = save_results(rows, args.results_dir)
    print_table(df)


if __name__ == "__main__":
    main()
