#!/usr/bin/env python3
"""
sensitivity_analysis.py — Hyperparameter sensitivity analysis for HERALD.

Sweeps HERALD's four core hyperparameters one-at-a-time (OAT), holding all
others at their paper-default value, and reports downstream node-classification
accuracy at a fixed compression ratio. This is the experiment flagged in the
paper's "To do" list ("Sensitivity analysis") and is a natural complement to
the discrete-variant ablation study already in Appendix C: instead of turning
score components on/off, this sweeps each *continuous* knob HERALD exposes and
checks how much of the paper's headline result depends on the exact value
chosen.

Hyperparameters swept (all defined in Algorithm 1 / Eq. 3-4, 13, 15):
  lid_k       – k, the LID neighbourhood size (Eq. 13)
  wl_layers   – L, the BFS expansion depth (Eq. 15) [also reused as the ogsize
                BFS-depth anchor inside condense_herald]
  steepness   – the sigmoid steepness coefficient in Eq. (3) (paper uses 8)
  alpha0      – the prototype base weight α0 in Eq. (4) (paper uses 0.4;
                β0, γ0 are held at their paper defaults 0.4 / 0.2 while α0
                varies, then all three are renormalised as in Eq. 5)

Datasets: chosen to span the heterophily spectrum used in Fig./Table 3 of the
paper — Cora (h≈0.00, strongly homophilic), Amazon-ratings (h≈0.38-0.62,
straddles the sigmoid's transition point h=0.4), and Roman-empire (h≈0.97,
strongly heterophilic). This is the regime where the adaptive-weight
mechanism (steepness, alpha0) is expected to matter most, per the paper's own
discussion in Appendix C.3.4.

GNNs: GCN (homophily-biased reference) and H2GCN (heterophily-aware target
architecture), matching the model pair already used in the Appendix C
ablation for consistency.

Usage
-----
  python sensitivity_analysis.py \
      --datasets cora amazon-ratings roman-empire \
      --gnns GCN H2GCN --frac 0.005 --seeds 0 1 2 3 4 --nepochs 200

  # quick smoke test (few epochs, one seed, one dataset)
  python sensitivity_analysis.py --datasets cora --gnns GCN \
      --seeds 0 --nepochs 20 --quick
"""

from __future__ import annotations
import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data

from benchmark import load_dataset, run_seeds, SAINT_DATASETS
from condensers_sensitivity import condense_herald, _size_bytes

# ─────────────────────────────────────────────────────────────────────────
#  Sweep configuration
# ─────────────────────────────────────────────────────────────────────────

# Paper-default HERALD hyperparameters — sweeping "none" of these axes must
# reproduce the numbers in Tables 4-6 exactly.
HERALD_DEFAULTS = dict(
    k=5,                 # unused inside condense_herald itself (kept for
                         # signature compatibility with condense_bonsai)
    alpha=None, beta=None, gamma=None,   # None => use adaptive weights below
    lid_k=10,
    wl_layers=2,
    alpha0=0.4, beta0=0.4, gamma0=0.2,
    steepness=8.0, midpoint=0.4,
    pca_variance_ratio=None,
    verbose=False,
)

# One-at-a-time sweep grid. Each entry sweeps ONE key of HERALD_DEFAULTS
# while every other key stays at its default value.
SWEEP_GRID = {
    "lid_k":     [5, 10, 15, 20, 30],
    "wl_layers": [1, 2, 3, 4],
    "steepness": [2.0, 4.0, 8.0, 12.0, 16.0, 24.0],
    "alpha0":    [0.2, 0.3, 0.4, 0.5, 0.6],
}


# ─────────────────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────────────────

def build_eval_data(data, features_used, pca_obj=None) -> Data:
    """
    Mirrors the eval-graph construction in benchmark.py's main loop:
    slice the ORIGINAL full-dimensional feature matrix down to the same
    `features_used` columns HERALD selected on the condensed graph, so the
    downstream GNN sees a consistent feature space at train and eval time.
    """
    x_src = data.x_normed if hasattr(data, "x_normed") else data.x
    if pca_obj is not None:
        x_np = x_src[:, features_used].numpy().astype(np.float32)
        x_eval = torch.tensor(pca_obj.transform(x_np).astype(np.float32))
    else:
        x_eval = x_src[:, features_used]

    data_eval = Data(x=x_eval, y=data.y, edge_index=data.edge_index)
    if hasattr(data, "adj"):
        data_eval.adj = data.adj
    return data_eval


def run_one_condensation(data, train, splits, scaler, ds_name, dtype,
                          frac, herald_kwargs, seed):
    """Condense once with a fixed global seed (condensation is deterministic
    given a fixed torch/numpy state — same convention as benchmark.py)."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    result = condense_herald(
        data=data, train=train, splits=splits, scaler=scaler,
        dataset_name=ds_name, dtype=dtype, target_size_frac=frac,
        **herald_kwargs,
    )
    return result


def evaluate_condensed(result, data, ds_name, dtype, nnodes, nedges_full,
                        nfeats_full, nclass, splits, gnns, seeds, nepochs,
                        lr, weight_decay, hidim, device):
    """Train/eval each requested GNN on one condensed graph, return a list
    of result rows (one per GNN)."""
    data_syn = result["data_syn"]
    features_used = result["features_used"]
    pca_obj = result.get("pca", None)

    data_eval = build_eval_data(data, features_used, pca_obj)
    nfeats = data_eval.x.shape[1]
    eff_hidim = 1024 if ds_name in SAINT_DATASETS else hidim

    rows = []
    for gnn_name in gnns:
        m, s = run_seeds(gnn_name, nfeats, nclass, eff_hidim,
                          data_eval, data_syn, splits,
                          nepochs, lr, weight_decay, device, seeds)
        rows.append(dict(
            gnn=gnn_name,
            acc_mean=round(m * 100, 2),
            acc_std=round(s * 100, 2),
            cond_nodes=data_syn.x.shape[0],
            cond_edges=data_syn.edge_index.shape[1],
            cond_feats=data_syn.x.shape[1],
            sr_percent=round(
                _size_bytes(data_syn.x.shape[0], data_syn.edge_index.shape[1],
                            data_syn.x.shape[1], dtype) /
                max(_size_bytes(nnodes, nedges_full, nfeats_full, dtype), 1)
                * 100, 3),
        ))
    return rows


# ─────────────────────────────────────────────────────────────────────────
#  Main sweep
# ─────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--datasets", nargs="+",
                   default=["cora", "amazon-ratings", "roman-empire"])
    p.add_argument("--gnns", nargs="+", default=["GCN", "H2GCN"])
    p.add_argument("--frac", type=float, default=0.005,
                   help="Storage compression fraction, r (fixed across the sweep)")
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    p.add_argument("--nepochs", type=int, default=200)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--wd", type=float, default=5e-4, dest="weight_decay")
    p.add_argument("--hidim", type=int, default=128)
    p.add_argument("--data_root", default="datasets")
    p.add_argument("--results_dir", default="results_sensitivity")
    p.add_argument("--condense_seed", type=int, default=42,
                   help="Global seed used for condensation (paper convention)")
    p.add_argument("--device", default=None)
    p.add_argument("--quick", action="store_true",
                   help="Debug mode: only run the 3 middle grid points per axis")
    args = p.parse_args()

    device = torch.device(args.device if args.device
                          else ("cuda" if torch.cuda.is_available() else "cpu"))
    Path(args.results_dir).mkdir(parents=True, exist_ok=True)

    grid = SWEEP_GRID
    if args.quick:
        grid = {k: v[len(v) // 2 - 1: len(v) // 2 + 2] for k, v in grid.items()}

    rows = []
    t_start = time.time()

    for ds_name in args.datasets:
        print(f"\n{'='*70}\nDataset: {ds_name}\n{'='*70}")
        dataset = load_dataset(ds_name, root=args.data_root)
        data, scaler, splits, dtype = (dataset["data"], dataset["scaler"],
                                        dataset["splits"], dataset["dtype"])
        train = splits["train"]
        nnodes = data.x.shape[0]
        nclass = int(data.y.max().item()) + 1
        nedges_full = data.edge_index.shape[1]
        nfeats_full = data.x.shape[1]

        # ── baseline (all defaults) — one row per axis for a common x=default point
        print("  [baseline: all HERALD hyperparameters at paper defaults]")
        result = run_one_condensation(data, train, splits, scaler, ds_name,
                                       dtype, args.frac, HERALD_DEFAULTS,
                                       args.condense_seed)
        base_rows = evaluate_condensed(result, data, ds_name, dtype, nnodes,
                                        nedges_full, nfeats_full, nclass,
                                        splits, args.gnns, args.seeds,
                                        args.nepochs, args.lr,
                                        args.weight_decay, args.hidim, device)
        for r in base_rows:
            rows.append(dict(dataset=ds_name, axis="baseline", value="default", **r))

        # ── one-at-a-time sweeps ─────────────────────────────────────────
        for axis, values in grid.items():
            for value in values:
                kwargs = copy.deepcopy(HERALD_DEFAULTS)
                kwargs[axis] = value
                print(f"  [{axis}={value}] condensing …")
                try:
                    result = run_one_condensation(
                        data, train, splits, scaler, ds_name, dtype,
                        args.frac, kwargs, args.condense_seed)
                except Exception as e:
                    print(f"    [condense error] {e}")
                    for gnn_name in args.gnns:
                        rows.append(dict(dataset=ds_name, axis=axis,
                                          value=value, gnn=gnn_name,
                                          acc_mean="ERR", acc_std="ERR"))
                    continue

                eval_rows = evaluate_condensed(
                    result, data, ds_name, dtype, nnodes, nedges_full,
                    nfeats_full, nclass, splits, args.gnns, args.seeds,
                    args.nepochs, args.lr, args.weight_decay, args.hidim,
                    device)
                for r in eval_rows:
                    print(f"    {r['gnn']}: {r['acc_mean']:.2f}±{r['acc_std']:.2f}  "
                          f"(|Vc|={r['cond_nodes']}, SR={r['sr_percent']:.2f}%)")
                    rows.append(dict(dataset=ds_name, axis=axis, value=value, **r))

                # interim save so a long sweep can be inspected/resumed
                pd.DataFrame(rows).to_csv(
                    Path(args.results_dir) / "interim_sensitivity.csv", index=False)

    df = pd.DataFrame(rows)
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_csv = Path(args.results_dir) / f"sensitivity_{ts}.csv"
    df.to_csv(out_csv, index=False)
    print(f"\n✓ Wrote {out_csv}  ({time.time() - t_start:.0f}s total)")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
