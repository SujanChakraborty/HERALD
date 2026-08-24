# HERALD: High-Fidelity Exemplar Retrieval with Adaptive Landmark Distillation

**Heterophily-aware, gradient-free graph condensation for node classification.**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)
[![PyTorch Geometric](https://img.shields.io/badge/PyG-2.x-orange.svg)](https://pytorch-geometric.readthedocs.io/)

HERALD produces a small surrogate graph that preserves the downstream
node-classification performance of a much larger original graph — without
ever training a GNN during condensation. It extends
[BONSAI](https://openreview.net/forum?id=BONSAI) (Gupta et al., ICLR 2025)
with a **heterophily-adaptive** node-scoring and feature-selection criterion,
so that condensation quality no longer degrades on graphs where neighboring
nodes frequently belong to different classes (Roman-Empire, Amazon-Ratings,
Chameleon, Squirrel, etc.).

> 📄 Paper: *HERALD: High-Fidelity Exemplar Retrieval with Adaptive Landmark
> Distillation for Heterophily-Aware Graph Condensation* (Chakraborty, Saha,
> Bej — IISER Thiruvananthapuram)

---

## Table of Contents

- [Why HERALD?](#why-herald)
- [Method Overview](#method-overview)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Datasets](#datasets)
- [Reproducing the Paper's Results](#reproducing-the-papers-results)
- [Headline Results](#headline-results)
- [Complexity](#complexity)
- [Citation](#citation)
- [Acknowledgements](#acknowledgements)
- [License](#license)

---

## Why HERALD?

Most graph condensation methods build node representations via
Weisfeiler–Lehman (WL) neighborhood aggregation, which implicitly assumes
that adjacent nodes share the same label. Under **heterophily**, this
assumption breaks: aggregation mixes class information across boundaries,
so WL-based selection criteria (including BONSAI's Rev-*k*-NN coverage
objective) systematically under-represent exactly the boundary nodes that
carry the most discriminative signal on heterophilic graphs.

HERALD keeps BONSAI's proven, architecture-agnostic, gradient-free pipeline
(BFS expansion → Personalized PageRank pruning → class rebalancing, all
under an identical storage budget) and replaces only the two components
that inherit the homophily bias:

| | BONSAI | HERALD |
|---|---|---|
| Feature selection | WL + Decision Tree | Joint Fisher-discriminability × activation-density, heterophily-decayed |
| Node scoring | Rev-*k*-NN coverage over WL-smoothed trees | Adaptive blend of prototype, decision-boundary, and Local Intrinsic Dimensionality (LID) scores |
| Assembly (BFS / PPR / rebalance) | ✓ | ✓ (unchanged, for a controlled comparison) |
| Gradient-free | ✓ | ✓ |
| Heterophily-adaptive | ✗ | ✓ |

Because everything downstream of node scoring is held fixed, differences
in accuracy between BONSAI and HERALD isolate the effect of *which* nodes
and features are selected — not the assembly pipeline.

---

## Method Overview

HERALD runs in seven stages (Algorithm 1 + Algorithm 2 in the paper):

**Block 1 — measure heterophily, derive scoring weights**
1. **Heterophily ratio** `h` — fraction of cross-class edges among training nodes.
2. **Adaptive weights** `(α, β, γ)` — a sigmoid function of `h` that shifts
   weight from prototype-representativeness (low `h`) toward
   boundary-proximity (high `h`), with the LID weight bounded below.

**Block 2 — prepare and score candidate nodes**
3. **Heterophily-aware feature selection** — multi-hop Fisher discriminant
   score, decayed by `(1-h)^k` per hop so raw-space (unsmoothed) signal
   dominates on heterophilic graphs, multiplied by an activation-density
   term. Selects exactly as many features as BONSAI's WL+DT baseline
   (`k*`), so the two methods are compared at an identical feature budget.
4. **Node scoring** — combines
   - a **prototype score** (cosine similarity to the ℓ2-normalized class centroid),
   - a **boundary score** (fraction of neighbors with a different label),
   - a **LID score** (local intrinsic dimensionality over *k*-NN cosine distances, for diversity),

   weighted by `(α, β, γ)` from Stage 2.

**Block 3 — assemble the condensed graph under budget**
5. **BFS expansion** — grow an *L*-hop neighborhood tree around each
   top-ranked root, in score order, accepting a root only if the
   *incremental* storage cost fits within an upscaled budget (`1.9×B`),
   with a 100-consecutive-rejection early-exit.
6. **PPR pruning** — Personalized PageRank on the induced subgraph, then
   remove the lowest-`π` non-root nodes down to the exact budget.
7. **Class rebalancing** — top up under-represented classes with
   highest-scored candidates; trim over-represented classes by removing
   lowest-scored non-root nodes.

Storage cost follows BONSAI's convention: `C(G_c) = 2·m_f·Σ f_v + 2·|E_c|`,
where `m_f` is a dataset-dependent feature-storage multiplier and `f_v` is
the effective (post-selection) feature length of node `v`.

---

## Repository Structure

```
.
├── condensers_pca_v1.py       # Core condensers: HERALD, BONSAI, Random, Herding
│                               # (+ loaders for external GCond/GDEM/GCSR outputs)
├── benchmark_pca_v1.py         # End-to-end benchmark driver: dataset loop,
│                               # per-condenser evaluation, GNN training/eval
├── models/                     # GCN / GAT / GraphSAGE / GIN / H2GCN backbones
├── data/                       # Dataset loading & split generation utilities
├── configs/                    # Per-dataset / per-condenser hyperparameter configs
├── results/                    # Saved condensed graphs, logs, accuracy tables
├── notebooks/                  # Ablation, sensitivity-sweep, and structural-stats analysis
└── README.md
```

> Adjust this tree to match your actual layout — the two files above
> (`condensers_pca_v1.py`, `benchmark_pca_v1.py`) are the ones referenced
> throughout this README and the paper's implementation notes.

---

## Installation

```bash
git clone https://github.com/<your-username>/herald.git
cd herald

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install torch torchvision torchaudio
pip install torch_geometric
pip install numpy scipy scikit-learn networkx tqdm
```

> Install the PyTorch / PyTorch Geometric build that matches your CUDA
> version — see the [PyG installation guide](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html).

Optional, for a significant speed-up on large graphs (e.g. Reddit):

```bash
pip install numba
```

If `herald_condenser.py`'s numba-accelerated core is importable, the BFS
budget-accounting step (`_herald_sparsify_enrich`) will use it
automatically; otherwise it falls back to a pure-Python implementation with
incremental (not full-graph-rescan) cost tracking.

---

## Quick Start

Run a single condenser on a single dataset at one compression ratio:

```bash
python benchmark_pca_v1.py \
    --datasets cora \
    --condensers herald \
    --fractions 0.01 \
    --gnns gcn gat gin h2gcn \
    --seeds 5
```

Run the full benchmark sweep (all datasets × all condensers × all budgets):

```bash
python benchmark_pca_v1.py \
    --datasets cora citeseer pubmed reddit roman-empire amazon-ratings chameleon squirrel \
    --condensers random herding bonsai herald \
    --fractions 0.0001 0.005 0.01 0.03 \
    --gnns gcn gat gin h2gcn \
    --seeds 5
```

> Flag names above follow the convention used in this repo's scripts —
> run `python benchmark_pca_v1.py --help` to confirm the exact CLI surface
> in your checkout, and adjust dataset keys to match your `data/` loaders.

For Reddit specifically, note that **BONSAI must run before HERALD** for a
given dataset if you want HERALD's feature budget to match BONSAI's
WL+DT-selected count (`k*`) exactly, since HERALD's feature selector reuses
that count as its own top-*k* anchor.

---

## Datasets

| Dataset | \|V\| | \|E\| | Features | Classes | Homophily (1−h) | Type |
|---|---:|---:|---:|---:|---:|---|
| Cora | 2,708 | 10,556 | 1,433 | 7 | 0.998 | homophilic |
| CiteSeer | 3,327 | 9,228 | 3,703 | 6 | 0.993 | homophilic |
| PubMed | 19,717 | 88,651 | 500 | 3 | 0.988 | homophilic |
| Reddit | 232,965 | 57.3M | 602 | 41 | 0.75 | homophilic |
| Roman-Empire | 22,662 | 65,854 | 300 | 18 | 0.032 | heterophilic |
| Amazon-Ratings | 24,492 | 186,100 | 300 | 5 | 0.380 | heterophilic |
| Chameleon | 2,277 | 36,101 | 2,325 | 5 | 0.230 | heterophilic |
| Squirrel | 5,201 | 217,073 | 2,089 | 5 | 0.224 | heterophilic |

Splits: 56% train / 24% validation / 20% test, generated once with a fixed
seed and held constant across all runs (only model initialization varies
across the 5 seeds used for evaluation).

---

## Reproducing the Paper's Results

- **Compression ratios evaluated:** `r ∈ {0.0001, 0.005, 0.01, 0.03}`,
  applied identically to every condenser via the shared storage-budget
  formula, for a fair comparison.
- **GNN backbones:** GCN, GAT, GIN, H2GCN — 200 epochs, Adam, lr=1e-3,
  weight decay=5e-4, 128 hidden dims (1024 for Reddit).
- **Metric:** mean test accuracy ± std over 5 seeds, at fixed data splits.
- **Baselines:** Random, Herding, BONSAI, GDEM.

See the paper's Appendix C (ablations), Appendix D (hyperparameter
sensitivity sweep over LID neighborhood size, BFS depth, sigmoid
steepness, and prototype base weight), and Appendix E (condensed-graph
structural statistics: storage ratio, feature-variance retention, class
entropy) for the full experimental protocol.

---

## Headline Results

**Averaged across seven medium-scale datasets** (Reddit reported
separately), HERALD is the best condensed method on every GNN backbone at
`r ∈ {0.005, 0.01, 0.03}`, e.g. at `r=0.03`: 57.6% (GCN), 56.2% (GAT), 54.8%
(GIN), 64.2% (H2GCN) — a 1.0–2.7 point improvement over BONSAI, the
strongest baseline.

**Heterophilic datasets** (Roman-Empire, Amazon-Ratings, Chameleon,
Squirrel): HERALD is best in **15 of 16** condenser/GNN/budget cells, with
the largest margins on H2GCN (e.g. +2.7 points over BONSAI at `r=0.03`).

**Homophilic citation networks** (Cora, CiteSeer, PubMed): HERALD and
BONSAI remain close throughout, with HERALD ahead at the largest budget
(`r=0.03`) on all four backbones and BONSAI ahead at `r=0.005`. HERALD's
feature-selection headroom gives it a large lead at the most extreme
budget (`r=0.0001`) on GCN/GAT/GIN, at the cost of a weaker H2GCN result
in that single regime — see the paper's Limitations (§6.6) for discussion.

**Reddit (large-scale, 232,965 nodes / 57.3M edges):** HERALD wins 13 of 16
cells against the strongest baseline per cell, sweeping GAT and GIN
outright at every budget and reaching 47.3% GCN / 45.8% H2GCN accuracy at
`r=0.03` (full-graph accuracy: 52.5% / 50.9%).

Full per-dataset tables are in the paper's Appendix G.

---

## Complexity

| Stage | Operation | Complexity |
|---|---|---|
| 1 | Heterophily ratio | `O(\|E_tr\|)` |
| 2 | Adaptive weights | `O(1)` |
| 3 | Feature selection (3-hop Fisher) | `O(NF + \|E\|F)` |
| 4 | Prototype + boundary scores | `O(NF + E)` |
| 4 | LID (exact k-NN) | `O(N²F)` |
| 5 | BFS expansion | `O(N(\|V_c\| + \|E_c\|))` |
| 6 | PPR pruning (100 iters) | `O(\|V_c\| + \|E_c\|)` |
| 7 | Class rebalancing | `O(\|V_c\|·C)` |

The dominant cost at scale is the exact LID computation (`O(N²F)`,
pairwise cosine similarity, batched to bound peak memory but not
asymptotic time). For very large graphs, swapping in an approximate
nearest-neighbor index (e.g. HNSW) for the LID step is a natural
scalability improvement — see the paper's Conclusion for discussion of
this and other planned extensions (approximate NN search, dynamic/
heterogeneous graphs, graph-level tasks, locally adaptive heterophily
estimation).

---

## Citation

If you use HERALD in your research, please cite:

```bibtex
@article{chakraborty2026herald,
  title   = {HERALD: High-Fidelity Exemplar Retrieval with Adaptive
             Landmark Distillation for Heterophily-Aware Graph Condensation},
  author  = {Chakraborty, Sujan and Saha, Priyanka and Bej, Saptarshi},
  journal = {Under review},
  year    = {2026}
}
```

Please also consider citing BONSAI, which HERALD builds on:

```bibtex
@inproceedings{gupta2025bonsai,
  title     = {Bonsai: Gradient-free Graph Condensation for Node Classification},
  author    = {Gupta, Mridul and Jain, Sahil and Ramani, Vaibhav and
               Kodamana, Hariprasad and Ranu, Sayan},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2025}
}
```

---

## Acknowledgements

This work builds directly on the official
[BONSAI](https://openreview.net/forum?id=BONSAI) implementation for the
BFS-expansion, PPR-pruning, and class-rebalancing assembly pipeline, and
compares against [GDEM](https://arxiv.org/abs/2405.06938) as a spectral
condensation baseline. Datasets are drawn from the standard citation-network
benchmarks (Cora, CiteSeer, PubMed), [Reddit](https://arxiv.org/abs/1706.02216),
and the heterophily benchmark suite of
[Platonov et al., 2023](https://arxiv.org/abs/2302.11640) (Roman-Empire,
Amazon-Ratings) and [Rozemberczki et al., 2021](https://arxiv.org/abs/1909.13021)
(Chameleon, Squirrel).

---

## License

This project is licensed under the MIT License — see [`LICENSE`](LICENSE)
for details.
