# Astar Island — Solution

## Quick Start

### 1. Fetch historical data
Before running the solver, download the initial maps and ground truths from all completed rounds using `fetch_data.py`:

```bash
AINM_TOKEN=<jwt> python fetch_data.py
```

This populates `data/round_N/seed_K_initial.json` and `seed_K_ground_truth.json` for every completed round. These files are required for the historical prior in the solver.

To fetch initial states only (skip ground truth):
```bash
AINM_TOKEN=<jwt> python fetch_data.py --no-ground-truth
```

### 2. Run the solver
```bash
AINM_TOKEN=<jwt> python astar_island_solve.py
```

Common options:
```bash
# Spend all queries on one seed
AINM_TOKEN=<jwt> python astar_island_solve.py --concentrate-seed 0

# Split queries across seeds (e.g. 3 full passes on seed 0, 2 on seed 1)
AINM_TOKEN=<jwt> python astar_island_solve.py --query-seeds 0:3p,1:2p

# Use historical prior only, no API queries
AINM_TOKEN=<jwt> python astar_island_solve.py --skip-query

# Test without submitting
AINM_TOKEN=<jwt> python astar_island_solve.py --dry-run
```

---

## Solution Description

The solver predicts the final-state probability distribution across all cells of a 40×40 map using a three-layer approach.

**Layer 1 — Historical Prior**
Built from all past rounds' initial maps and ground truths. Each cell is assigned to a feature bucket defined by terrain type, distance to the nearest settlement/port, and coastal adjacency. Per-bucket mean probability vectors are accumulated across all available ground truth maps, providing a stable baseline that captures structural patterns (e.g. mountains almost never become settlements, ocean stays ocean).

**Layer 2 — Observation Calibration**
The active round has unknown hidden parameters (expansion rate, winter severity, etc.) that shift the true distribution away from the historical baseline. Up to 50 viewport queries are issued against the active round's simulation API. The observed cell states are used to update the bucket-level prior via Bayesian averaging:

```
posterior = (hist_weight × historical + obs_counts) / (hist_weight + obs_total)
```

Buckets with many observations shift toward the current round's empirical behaviour; sparse buckets stay close to the historical baseline.

**Layer 3 — Per-cell Blending**
For seeds where per-cell observations are available, each cell's prediction is further refined by blending the bucket posterior with the cell's own empirical distribution:

```
pred = (alpha × bucket_pred + n × cell_empirical) / (alpha + n)
```

All predictions are probability-floor normalised before submission.
