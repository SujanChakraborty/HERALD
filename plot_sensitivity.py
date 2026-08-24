#!/usr/bin/env python3
"""
plot_sensitivity.py — Turn sensitivity_analysis.py's output CSV into figures
and a summary table for the paper.

Usage
-----
  python plot_sensitivity.py --csv results_sensitivity/sensitivity_20260101_120000.csv \
      --out_dir figures_sensitivity
"""
from __future__ import annotations
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

AXIS_LABEL = {
    "lid_k":     "LID neighbourhood size  $k$",
    "wl_layers": "BFS expansion depth  $L$",
    "steepness": "Sigmoid steepness  (Eq. 3)",
    "alpha0":    r"Prototype base weight  $\alpha_0$",
}
AXIS_DEFAULT = {"lid_k": 10, "wl_layers": 2, "steepness": 8.0, "alpha0": 0.4}


def make_axis_plot(df: pd.DataFrame, axis: str, out_dir: Path):
    sub = df[(df["axis"] == axis) & (df["acc_mean"] != "ERR")].copy()
    if sub.empty:
        print(f"  [skip] no data for axis={axis}")
        return
    sub["value"] = pd.to_numeric(sub["value"])
    sub["acc_mean"] = pd.to_numeric(sub["acc_mean"])
    sub["acc_std"] = pd.to_numeric(sub["acc_std"])

    gnns = sorted(sub["gnn"].unique())
    datasets = sorted(sub["dataset"].unique())

    fig, axes = plt.subplots(1, len(gnns), figsize=(5.5 * len(gnns), 4.2),
                              sharex=True, squeeze=False)
    axes = axes[0]

    for ax, gnn in zip(axes, gnns):
        for ds in datasets:
            d = sub[(sub["gnn"] == gnn) & (sub["dataset"] == ds)].sort_values("value")
            if d.empty:
                continue
            ax.errorbar(d["value"], d["acc_mean"], yerr=d["acc_std"],
                        marker="o", capsize=3, label=ds)
        default_x = AXIS_DEFAULT[axis]
        ax.axvline(default_x, color="gray", linestyle="--", linewidth=1,
                   label="paper default" if ax is axes[0] else None)
        ax.set_title(gnn)
        ax.set_xlabel(AXIS_LABEL[axis])
        if ax is axes[0]:
            ax.set_ylabel("Test accuracy (%)")
        ax.grid(alpha=0.3)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(datasets) + 1,
              bbox_to_anchor=(0.5, 1.08))
    fig.suptitle(f"HERALD sensitivity: {AXIS_LABEL[axis]}", y=1.15)
    fig.tight_layout()

    out_path = out_dir / f"sensitivity_{axis}.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {out_path}")


def make_summary_table(df: pd.DataFrame, out_dir: Path):
    """Max absolute accuracy swing (max - min across grid values) per
    dataset x GNN x axis — the sensitivity-analysis analogue of Table C14's
    ablation summary."""
    sub = df[(df["axis"] != "baseline") & (df["acc_mean"] != "ERR")].copy()
    sub["acc_mean"] = pd.to_numeric(sub["acc_mean"])

    summary = (sub.groupby(["axis", "dataset", "gnn"])["acc_mean"]
                  .agg(["max", "min"]).reset_index())
    summary["swing"] = (summary["max"] - summary["min"]).round(2)
    summary = summary.drop(columns=["max", "min"])
    pivot = summary.pivot_table(index=["axis", "dataset"], columns="gnn",
                                 values="swing")

    out_path = out_dir / "sensitivity_summary.csv"
    pivot.to_csv(out_path)
    print(f"  ✓ {out_path}")
    print("\nMax accuracy swing (%) across each hyperparameter's grid:")
    print(pivot.to_string())

    latex_path = out_dir / "sensitivity_summary.tex"
    with open(latex_path, "w") as f:
        f.write(pivot.to_latex(float_format="%.2f"))
    print(f"  ✓ {latex_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True)
    p.add_argument("--out_dir", default="figures_sensitivity")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.csv)
    for axis in ["lid_k", "wl_layers", "steepness", "alpha0"]:
        make_axis_plot(df, axis, out_dir)
    make_summary_table(df, out_dir)


if __name__ == "__main__":
    main()
