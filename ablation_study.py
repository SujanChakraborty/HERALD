#!/usr/bin/env python3
"""
ablation_study.py — HERALD ablation study driver.

Runs the compact, deliberately contrastive ablation matrix used in the
paper's ablation section:

    datasets : cora           (h ~ 0.00,  strongly homophilic)
               amazon-ratings (h ~ 0.38,  moderately heterophilic)
               roman-empire   (h ~ 0.97,  strongly heterophilic)
    ratio    : r = 0.01   (single mid-range storage budget)
    gnns     : GCN    (homophily-biased sanity check)
               H2GCN  (heterophily-aware -- the architecture HERALD should
                       help the most)
    seeds    : 0, 1, 2, 3, 4
    variants : full HERALD + 7 ablation arms (defined in condensers.py,
               section "HERALD ABLATION VARIANTS"):

      herald              full method (reference)
      herald_fixedw       static (alpha0,beta0,gamma0); no heterophily-
                          adaptive weighting
      herald_proto_only   prototype score only
      herald_bound_only   boundary score only
      herald_lid_only     LID (diversity) score only
      herald_no_lid       prototype + boundary, no diversity term
      herald_bonsai_feats HERALD scoring + Bonsai's WL+DT feature selector
                          (isolates the feature-selection axis)
      herald_random_rank  HERALD's full pipeline (feature selection, BFS
                          expand, PPR prune, class rebalance) but nodes
                          ranked uniformly at random (isolates scoring
                          from the shared assembly pipeline)

This script deliberately bypasses benchmark.py's CLI-driven condenser
dispatch (which only special-cases kwargs for the literal names "bonsai"
and "herald") and instead calls condensers directly, so the ablation
variants registered in condensers.HERALD_ABLATION_FN can be driven with
full control over their kwargs. It reuses benchmark.py's dataset loading,
training loop (run_seeds/train_and_eval), and global-seeding convention
verbatim, so results are directly comparable to the main HERALD-vs-
baselines tables produced by benchmark.py itself.

Usage
-----
  # default compact ablation matrix: 3 datasets x 1 ratio x 2 GNNs x
  # 5 seeds x 8 variants (240 GNN-training runs; 24 condensations)
  python ablation_study.py

  # override any axis
  python ablation_study.py --datasets cora roman-empire \\
      --gnns GCN H2GCN --seeds 0 1 2 --frac 0.01

  # also include Bonsai + Random as external anchors for context
  python ablation_study.py --include_anchors

  # sensitivity sweep over LID neighbourhood size k (minor, single dataset)
  python ablation_study.py --sensitivity lid_k \\
      --sensitivity_dataset amazon-ratings --sensitivity_values 5 10 15 20 25

  # sensitivity sweep over the adaptive-weight sigmoid steepness
  python ablation_study.py --sensitivity steepness \\
      --sensitivity_dataset amazon-ratings --sensitivity_values 2 4 8 16 32
"""

from __future__ import annotations
import argparse
import gc
import json
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data

# ── reuse the benchmark's own dataset loading / training-eval code ───────
from benchmark import load_dataset, run_seeds, SAINT_DATASETS
from condensers_ablation import CONDENSER_FN, _size_bytes

GLOBAL_SEED = 42  # matches benchmark.py's main()

# ── the 8 variants at the heart of the ablation study ─────────────────────
DEFAULT_VARIANTS = [
    "herald",
    "herald_fixedw",
    "herald_proto_only",
    "herald_bound_only",
    "herald_lid_only",
    "herald_no_lid",
    "herald_bonsai_feats",
    "herald_random_rank",
]

ANCHOR_VARIANTS = ["bonsai", "random", "herding"]

VARIANT_LABELS = {
    "herald":              "HERALD (full)",
    "herald_fixedw":       "A1  fixed weights",
    "herald_proto_only":   "A2a prototype only",
    "herald_bound_only":   "A2b boundary only",
    "herald_lid_only":     "A2c LID only",
    "herald_no_lid":       "A2d proto+boundary (no LID)",
    "herald_bonsai_feats": "A3  Bonsai feature selector",
    "herald_random_rank":  "A4  random ranking (same pipeline)",
    "bonsai":              "Bonsai (anchor)",
    "random":              "Random (anchor)",
    "herding":             "Herding (anchor)",
}


# ═══════════════════════════════════════════════════════════════════════
#  CONDENSATION DISPATCH
# ═══════════════════════════════════════════════════════════════════════

def run_condenser(variant: str, *, data, train, splits, scaler, ds_name,
                   dtype, frac, k: int, lid_k: int, seed_override: dict | None = None):
    """Call the requested condenser variant with a fixed, reproducible
    global RNG state (mirrors benchmark.py's re-seeding before every
    condensation call), so the condensed graph is identical regardless of
    what ran before it.
    """
    torch.manual_seed(GLOBAL_SEED)
    torch.cuda.manual_seed_all(GLOBAL_SEED)
    np.random.seed(GLOBAL_SEED)

    cond_fn = CONDENSER_FN[variant]
    kwargs = dict(data=data, train=train, splits=splits, scaler=scaler,
                  dataset_name=ds_name, dtype=dtype, target_size_frac=frac)

    if variant == "bonsai":
        kwargs.update(k=k, frac_to_sample=1.0)
    elif variant.startswith("herald"):
        kwargs.update(k=k, lid_k=lid_k)
        if seed_override is not None:
            kwargs.update(seed_override)

    return cond_fn(**kwargs)


# ═══════════════════════════════════════════════════════════════════════
#  MAIN DRIVER
# ═══════════════════════════════════════════════════════════════════════

def evaluate_variant(variant, dataset_bundle, frac, gnns, seeds, k, lid_k,
                      nepochs, lr, weight_decay, device, results_dir,
                      seed_override=None):
    """Condense once with `variant`, then train+evaluate each requested GNN
    over all seeds on the resulting condensed graph. Returns a list of
    per-(gnn) result dicts, or a single ERR row on failure.
    """
    (ds_name, data, scaler, splits, dtype, train, _x_orig, _x_normed_orig,
     nnodes, nclass, nedges_full, nfeats_full) = dataset_bundle

    rows = []
    t_cond = 0.0
    try:
        t0 = time.perf_counter()
        result = run_condenser(variant, data=data, train=train, splits=splits,
                                scaler=scaler, ds_name=ds_name, dtype=dtype,
                                frac=frac, k=k, lid_k=lid_k,
                                seed_override=seed_override)
        t_cond = time.perf_counter() - t0
    except Exception as e:
        print(f"    [{variant}] condenser error: {e}")
        traceback.print_exc()
        for gnn_name in gnns:
            rows.append(dict(dataset=ds_name, variant=variant, frac=frac,
                              gnn=gnn_name, acc_mean="ERR", acc_std="ERR",
                              condense_time="ERR"))
        return rows

    data_syn      = result["data_syn"]
    features_used = result["features_used"]
    pca_obj       = result.get("pca", None)

    if pca_obj is not None:
        x_full_np = _x_orig[:, features_used].numpy().astype(np.float32)
        x_eval    = torch.tensor(pca_obj.transform(x_full_np).astype(np.float32))
    else:
        x_eval = _x_orig[:, features_used]

    data_eval = Data(x=x_eval, y=data.y, edge_index=data.edge_index)
    if hasattr(data, "adj"):
        data_eval.adj = data.adj
    if _x_normed_orig is not None:
        if pca_obj is not None:
            xn_full_np = _x_normed_orig[:, features_used].numpy().astype(np.float32)
            data_eval.x_normed = torch.tensor(pca_obj.transform(xn_full_np).astype(np.float32))
        else:
            data_eval.x_normed = _x_normed_orig[:, features_used]

    nfeats = x_eval.shape[1]
    hidim  = 1024 if ds_name in SAINT_DATASETS else 128

    cond_nodes = data_syn.x.shape[0]
    cond_edges = data_syn.edge_index.shape[1]
    cond_feats = data_syn.x.shape[1]
    sr_percent = round(
        _size_bytes(cond_nodes, cond_edges, cond_feats, dtype) /
        max(_size_bytes(nnodes, nedges_full, nfeats_full, dtype), 1) * 100, 3)

    for gnn_name in gnns:
        print(f"    [{variant}] {gnn_name} ... ", end="", flush=True)
        t_train_start = time.perf_counter()
        m, s = run_seeds(gnn_name, nfeats, nclass, hidim, data_eval, data_syn,
                          splits, nepochs, lr, weight_decay, device, seeds)
        t_train = time.perf_counter() - t_train_start
        print(f"-> {m*100:.2f}+-{s*100:.2f}  ({t_train:.0f}s)")

        rows.append(dict(
            dataset=ds_name, variant=variant, frac=frac, gnn=gnn_name,
            acc_mean=round(m*100, 2), acc_std=round(s*100, 2),
            condense_time=round(t_cond, 2),
            train_time=round(t_train/len(seeds), 2),
            cond_nodes=cond_nodes, cond_edges=cond_edges, cond_feats=cond_feats,
            sr_percent=sr_percent,
        ))

    # interim save after each variant so partial progress is never lost
    interim = Path(results_dir) / f"interim_{ds_name}.json"
    with open(interim, "w") as f:
        json.dump(rows, f, indent=2)

    gc.collect()
    return rows


def load_dataset_bundle(ds_name, data_root):
    """Load a dataset once and snapshot everything needed by every variant,
    mirroring benchmark.py::main()'s per-dataset setup exactly."""
    dataset = load_dataset(ds_name, root=data_root)
    data   = dataset["data"]
    scaler = dataset["scaler"]
    splits = dataset["splits"]
    dtype  = dataset["dtype"]
    train  = splits["train"]

    nnodes = data.x.shape[0]
    nclass = int(data.y.max().item()) + 1
    nedges_full = data.edge_index.shape[1]
    nfeats_full = data.x.shape[1]

    _x_orig        = (data.x_normed if hasattr(data, "x_normed") else data.x).clone()
    _x_normed_orig = data.x_normed.clone() if hasattr(data, "x_normed") else None

    print(f"  Full graph [{ds_name}]: {nnodes:,} nodes | {nedges_full:,} edges | "
          f"{nfeats_full} feats | {nclass} classes | train={len(train):,}")

    return (ds_name, data, scaler, splits, dtype, train, _x_orig,
            _x_normed_orig, nnodes, nclass, nedges_full, nfeats_full)


def main_ablation(args):
    device = torch.device(args.device if args.device
                          else ("cuda" if torch.cuda.is_available() else "cpu"))

    torch.manual_seed(GLOBAL_SEED)
    torch.cuda.manual_seed_all(GLOBAL_SEED)
    np.random.seed(GLOBAL_SEED)
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass

    Path(args.results_dir).mkdir(parents=True, exist_ok=True)

    variants = list(args.variants)
    if args.include_anchors:
        variants += [v for v in ANCHOR_VARIANTS if v not in variants]

    print(f"Device: {device}")
    print(f"Datasets : {args.datasets}")
    print(f"Ratio    : {args.frac}")
    print(f"GNNs     : {args.gnns}")
    print(f"Seeds    : {args.seeds}")
    print(f"Variants : {variants}")

    all_rows = []
    for ds_name in args.datasets:
        print(f"\n{'='*70}\nDataset: {ds_name}\n{'='*70}")
        try:
            bundle = load_dataset_bundle(ds_name, args.data_root)
        except Exception as e:
            print(f"  [SKIP] {e}"); continue

        for variant in variants:
            print(f"\n  -- variant: {variant} ({VARIANT_LABELS.get(variant, variant)}) --")
            rows = evaluate_variant(
                variant, bundle, args.frac, args.gnns, args.seeds,
                args.k, args.lid_k, args.nepochs, args.lr, args.weight_decay,
                device, args.results_dir)
            all_rows.extend(rows)

    df = save_results(all_rows, args.results_dir)
    print_ablation_table(df)


# ═══════════════════════════════════════════════════════════════════════
#  SENSITIVITY SWEEPS (secondary, single-dataset line-plot style)
# ═══════════════════════════════════════════════════════════════════════

def main_sensitivity(args):
    """Single-axis sensitivity sweep on one dataset/GNN, holding everything
    else fixed at the compact-matrix default (r=0.01). Produces one row per
    swept value rather than a full ablation table."""
    device = torch.device(args.device if args.device
                          else ("cuda" if torch.cuda.is_available() else "cpu"))
    torch.manual_seed(GLOBAL_SEED); np.random.seed(GLOBAL_SEED)

    Path(args.results_dir).mkdir(parents=True, exist_ok=True)
    bundle = load_dataset_bundle(args.sensitivity_dataset, args.data_root)

    rows = []
    for val in args.sensitivity_values:
        print(f"\n  -- {args.sensitivity} = {val} --")
        if args.sensitivity == "lid_k":
            r = evaluate_variant("herald", bundle, args.frac, args.gnns,
                                  args.seeds, args.k, int(val),
                                  args.nepochs, args.lr, args.weight_decay,
                                  device, args.results_dir)
        elif args.sensitivity == "steepness":
            r = evaluate_variant("herald", bundle, args.frac, args.gnns,
                                  args.seeds, args.k, args.lid_k,
                                  args.nepochs, args.lr, args.weight_decay,
                                  device, args.results_dir,
                                  seed_override={"weight_steepness": float(val)})
        elif args.sensitivity == "bfs_depth":
            r = evaluate_variant("herald", bundle, args.frac, args.gnns,
                                  args.seeds, args.k, args.lid_k,
                                  args.nepochs, args.lr, args.weight_decay,
                                  device, args.results_dir,
                                  seed_override={"wl_layers": int(val)})
        else:
            raise ValueError(f"Unknown sensitivity axis '{args.sensitivity}'")
        for row in r:
            row[args.sensitivity] = val
        rows.extend(r)

    df = save_results(rows, args.results_dir,
                       prefix=f"sensitivity_{args.sensitivity}")
    print("\n" + "="*70)
    print(f"SENSITIVITY: {args.sensitivity}  (dataset={args.sensitivity_dataset})")
    print("="*70)
    pivot = df.pivot_table(index=args.sensitivity, columns="gnn",
                            values="acc_mean", aggfunc="first")
    print(pivot.to_string())


# ═══════════════════════════════════════════════════════════════════════
#  RESULTS I/O
# ═══════════════════════════════════════════════════════════════════════

def save_results(rows, results_dir, prefix="ablation_results"):
    Path(results_dir).mkdir(parents=True, exist_ok=True)
    ts  = time.strftime("%Y%m%d_%H%M%S")
    df  = pd.DataFrame(rows)
    csv = Path(results_dir) / f"{prefix}_{ts}_comp2.csv"
    jsf = Path(results_dir) / f"{prefix}_{ts}_comp2.json"
    df.to_csv(csv, index=False)
    with open(jsf, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"\n[saved] {csv}\n[saved] {jsf}")
    return df


def print_ablation_table(df: pd.DataFrame):
    for c in ["acc_mean", "acc_std", "condense_time", "sr_percent"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df["label"] = df["variant"].map(lambda v: VARIANT_LABELS.get(v, v))

    pivot = df.pivot_table(
        index=["dataset", "label"], columns="gnn",
        values="acc_mean", aggfunc="first",
    )
    # keep a stable, paper-friendly ordering of variants within each dataset
    order = [VARIANT_LABELS[v] for v in DEFAULT_VARIANTS + ANCHOR_VARIANTS
             if VARIANT_LABELS[v] in pivot.index.get_level_values("label")]
    pivot = pivot.reindex(order, level="label")

    print("\n" + "="*90)
    print("HERALD ABLATION RESULTS  (mean test accuracy %)")
    print("="*90)
    print(pivot.to_string())
    print("="*90)


# ═══════════════════════════════════════════════════════════════════════
#  ARGUMENT PARSING
# ═══════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="HERALD ablation study driver",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # ── compact ablation matrix (paper defaults) ──────────────────────
    p.add_argument("--datasets", nargs="+",
                    # default=["cora", "amazon-ratings", "roman-empire"],
                    default=["amazon-ratings", "squirrel"]
                    # help="cora (h~0.00), amazon-ratings (h~0.38), "
                         # "roman-empire (h~0.97) by default")
                  )
    p.add_argument("--frac", type=float, default=0.005,
                    help="single mid-range compression ratio for the "
                         "ablation headline table")
    p.add_argument("--gnns", nargs="+", default=["GCN", "H2GCN"],
                    choices=["GCN", "GAT", "GIN", "GCN_inductive", "H2GCN"])
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    p.add_argument("--variants", nargs="+", default=DEFAULT_VARIANTS,
                    choices=list(VARIANT_LABELS))
    p.add_argument("--include_anchors", action="store_true",
                    help="also run Bonsai/Random/Herding as external "
                         "reference points alongside the ablation arms")

    # ── training / condensation hyperparameters ───────────────────────
    p.add_argument("--nepochs", type=int, default=200)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--wd", type=float, default=5e-4, dest="weight_decay")
    p.add_argument("--k", type=int, default=5,
                    help="k for Bonsai/HERALD budget-anchoring step")
    p.add_argument("--lid_k", type=int, default=10,
                    help="neighbourhood size for HERALD's LID score")

    # ── misc ───────────────────────────────────────────────────────────
    p.add_argument("--data_root", default="datasets")
    p.add_argument("--results_dir", default="results/ablation")
    p.add_argument("--device", default=None)

    # ── sensitivity-sweep mode (optional, separate from the main table) ─
    p.add_argument("--sensitivity", choices=["lid_k", "steepness", "bfs_depth"],
                    default=None,
                    help="if set, run a single-axis sensitivity sweep "
                         "instead of the full ablation matrix")
    p.add_argument("--sensitivity_dataset", default="amazon-ratings",
                    help="dataset for the sensitivity sweep (default sits "
                         "near the adaptive-weight sigmoid's transition "
                         "point h=0.4)")
    p.add_argument("--sensitivity_values", nargs="+", type=float,
                    default=[5, 10, 15, 20, 25],
                    help="values to sweep for the chosen --sensitivity axis")

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.sensitivity is not None:
        main_sensitivity(args)
    else:
        main_ablation(args)
