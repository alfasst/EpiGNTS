# GNTS: Graph-based Network Testing Strategies

This repository implements a **network-based epidemic simulation and testing allocation framework**, with a focus on **Graph Neural Network–based Testing Strategies (GNTS)**.
The system supports **SBM synthetic networks** and **real-world SNAP networks**, and produces **reproducible, analysis-ready outputs** for comparative evaluation.

---

## 1. Project Overview

The pipeline consists of four cleanly separated stages:

```
Network generation  →  Simulation & testing  →  Data logging  →  Analysis & plotting
```

Key features:

* Network-aware epidemic simulation (SEAIRQ)
* Multiple testing strategies:

  * Heuristics
  * Multi-armed bandits
  * GNTS (Local & Global, PyG-based)
* Deterministic preprocessing of SNAP networks (≤10k nodes)
* Per-network–per-strategy daily outputs
* Fully reproducible analysis pipeline

---

## 2. Repository Structure

```
.
├── config.py                # Static experiment configuration
├── netgen.py                # Network generation (SBM + SNAP)
├── simulation.py            # Epidemic dynamics + testing process
├── gnts.py                  # GNTS core (PyTorch Geometric)
├── strategies.py            # Strategy wrappers & baselines
├── main.py                  # Experiment runner (writes daily CSVs)
├── analyze.py               # Analysis & plotting
├── requirements.txt         # Python dependencies
│
├── snap_data/               # Raw SNAP datasets (not included)
│   ├── com-orkut.ungraph.txt
│   ├── com-orkut.all.cmty.txt
│   └── ...
│
└── results/
    ├── gpickle/             # Cached networks
    ├── models/              # Trained GNTS models
    ├── experiments/         # Daily CSV outputs
    └── plots/               # SVG figures
```

---

## 3. Networks

### 3.1 SBM Networks (Synthetic)

Nine SBM networks are defined in `config.py`, spanning:

* Different sizes (1k–5k)
* Different modularity regimes (Zero → Max)

These are generated once and cached as `.gpickle` files.

---

### 3.2 SNAP Networks (Real-world)

Supported SNAP networks:

* Orkut
* LiveJournal
* YouTube

Processing steps (handled in `netgen.py`):

1. Load raw edge list and community files
2. Convert overlapping communities → single block per node
3. Remove singleton communities
4. Sample communities proportionally (BFS-based) to **≤10,000 nodes**
5. Relabel nodes to `0 … N−1`
6. Assign `block_id`
7. Cache as `.gpickle`

All downstream code works only with cached graphs.

---

## 4. Testing Strategies

Implemented strategies include:

### Baselines

* `Uniform`
* `Random`
* `Proportional`

### Bandits

* `Beta`
* `Gamma`

### Learning-based

* `LocalGNTS`
* `GlobalGNTS`

GNTS uses **PyTorch Geometric** for graph representation learning.

---

## 5. Installation

### 5.1 Create environment

```bash
python -m venv venv
source venv/bin/activate   # Linux / macOS
venv\\Scripts\\activate    # Windows
```

### 5.2 Install dependencies

```bash
pip install -r requirements.txt
```

> ⚠️ **Important:** PyTorch Geometric requires extra wheels.
> See the official PyG installation guide:
> [https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html)

CPU-only example:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv \
  -f https://data.pyg.org/whl/torch-2.0.0+cpu.html
pip install torch-geometric
```

---

## 6. Execution Workflow

### Step 1 — Generate networks

```bash
python netgen.py
```

Outputs:

* `results/gpickle/*.gpickle`
* `results/network_metrics.csv`

---

### Step 2 — Run experiments

Run **all networks**:

```bash
python main.py
```

Run a **single network only**:

```bash
python main.py --network SBM-5k-High
```

Behavior:

* GNTS models are trained once and cached
* Daily data are averaged over test runs
* One CSV per `(Network, Strategy)` is written

Example output files:

```
results/experiments/SBM-5k-High__LocalGNTS_daily.csv
results/experiments/Orkut__Uniform_daily.csv
```

---

### Step 3 — Analyze results

```bash
python analyze.py
```

This script:

* Discovers existing `(Network, Strategy)` CSVs
* Skips missing combinations
* Computes summary metrics
* Generates SVG plots

Outputs:

```
results/experiments/summary_metrics.csv
results/plots/*.svg
```

---

## 7. Output Data Format

### Daily CSV schema

Each CSV represents **one network–strategy pair**, averaged over test runs:

```
Day
S
E
I
A
Q
R
total_tests_administered
total_positive_tests
total_wasted_tests
```

These are the **only persisted experimental outputs**.

---

## 8. Metrics Computed in Analysis

Derived in `analyze.py` (not stored during simulation):

* Quarantine efficiency:
  `Q / (E + I + A + Q)`
* Peak infections:
  `max(I + A)`
* Time to peak
* Positive test rate

---

## 9. Design Principles

* **No runtime config mutation**
* **One responsibility per file**
* **Raw data → derived metrics**
* **Safe partial runs**
* **No overwriting of experiment results**
* **Analysis is fully reproducible**

---

## 10. Notes

* SNAP data files must be downloaded separately from SNAP.
* The repository assumes **Python ≥ 3.10**.
* All plots are saved in **SVG format** for publication use.

---

## 11. Citation

If you use this codebase in academic work, please cite the associated paper or contact the author for citation details.

---

If you want, I can next:

* add a **Quick Start** section
* provide a **reproducibility checklist**
* or tailor this README for a **journal supplementary material**
