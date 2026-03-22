"""
feature_prior_script_v3.py
--------------------------
Astar Island solver v3: Historical prior + observation calibration.

Architecture (3-layer prediction):
  Layer 1 — HISTORICAL PRIOR (from past rounds' ground truth)
    Built from ALL past round data (initial maps + ground truths).
    Learns structural patterns: how terrain type, distance to settlement,
    and coastal adjacency predict final-state probabilities.
    With 80+ ground truth maps, each bucket has thousands of samples.

  Layer 2 — OBSERVATION CALIBRATION (from current round's 50 queries)
    The current round has different hidden parameters (expansion rate,
    winter severity, etc). We use observations to calibrate the historical
    prior via Bayesian updating at the bucket level:
      posterior = (hist_weight * historical + obs_count * empirical) / (hist_weight + obs_count)
    Buckets with many observations shift toward the current round's behavior;
    sparse buckets stay close to the historical baseline.

  Layer 3 — PER-CELL BLENDING (for observed seeds only)
    For seeds where we have per-cell observations, blend the bucket
    prediction with the cell's own empirical distribution:
      pred = (alpha * bucket_pred + n * cell_empirical) / (alpha + n)


Usage:
  AINM_TOKEN=<jwt> python feature_prior_script_v3.py --concentrate-seed 0
  AINM_TOKEN=<jwt> python feature_prior_script_v3.py --query-seeds 0:3p,1:2p

Historical data setup:
  Place past round data in data/ directory:
    data/round_1/seed_0_initial.json, seed_0_ground_truth.json, ...
    data/round_2/seed_0_initial.json, seed_0_ground_truth.json, ...
    ...
  The script auto-discovers all round_*/seed_*_*.json files.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from collections import defaultdict

import numpy as np

try:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parents[2] / ".env")
except ImportError:
    pass

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BASE_URL = "https://api.ainm.no/astar-island"
MAP_H = 40
MAP_W = 40
MAX_VIEWPORT = 15
NUM_CLASSES = 6
NUM_SEEDS = 5
TOTAL_BUDGET = 50
PROB_FLOOR = 0.001
PERCELL_ALPHA = 150
HIST_WEIGHT = 200  # Bayesian weight of historical prior (in equivalent obs)

INT_CODE_MAP = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 10: 0, 11: 0}
CLASS_NAMES = ["Empty", "Settlement", "Port", "Ruin", "Forest", "Mountain"]

DATA_DIR = Path(__file__).parent / "data"

# Tiled viewport pattern — 9 queries cover all 1600 cells of a 40×40 map.
_FULL_PASS: list[tuple[int, int, int, int]] = [
    (0, 0, 15, 15),
    (15, 0, 15, 15),
    (25, 0, 15, 15),
    (0, 15, 15, 15),
    (15, 15, 15, 15),
    (25, 15, 15, 15),
    (0, 25, 15, 15),
    (15, 25, 15, 15),
    (25, 25, 15, 15),
]

# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------


def _make_session(token: str) -> "requests.Session":
    import requests

    try:
        import urllib3

        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    except Exception:
        pass
    s = requests.Session()
    s.headers["Authorization"] = f"Bearer {token}"
    s.verify = False
    return s


def _get(session, url: str, max_retries: int = 3) -> dict:
    for _ in range(max_retries):
        r = session.get(url)
        if r.status_code == 429:
            print("  Rate-limited, waiting 2 s ...", flush=True)
            time.sleep(2.0)
            continue
        r.raise_for_status()
        return r.json()
    raise RuntimeError(f"GET {url} failed after {max_retries} retries")


def _post(session, url: str, payload: dict, max_retries: int = 3) -> dict:
    for _ in range(max_retries):
        r = session.post(url, json=payload)
        if r.status_code == 429:
            print("  Rate-limited, waiting 2 s ...", flush=True)
            time.sleep(2.0)
            continue
        r.raise_for_status()
        return r.json()
    raise RuntimeError(f"POST {url} failed after {max_retries} retries")


# ---------------------------------------------------------------------------
# API wrappers
# ---------------------------------------------------------------------------


def get_rounds(session) -> list[dict]:
    return _get(session, f"{BASE_URL}/rounds")


def get_round(session, round_id: str) -> dict:
    return _get(session, f"{BASE_URL}/rounds/{round_id}")


def get_budget(session) -> dict:
    return _get(session, f"{BASE_URL}/budget")


def simulate(
    session, round_id: str, seed_index: int, vp_x: int, vp_y: int, vp_w: int, vp_h: int
) -> dict:
    return _post(
        session,
        f"{BASE_URL}/simulate",
        {
            "round_id": round_id,
            "seed_index": seed_index,
            "viewport_x": vp_x,
            "viewport_y": vp_y,
            "viewport_w": vp_w,
            "viewport_h": vp_h,
        },
    )


def submit_prediction(
    session, round_id: str, seed_index: int, prediction: np.ndarray
) -> dict:
    return _post(
        session,
        f"{BASE_URL}/submit",
        {
            "round_id": round_id,
            "seed_index": seed_index,
            "prediction": prediction.tolist(),
        },
    )


# ---------------------------------------------------------------------------
# Grid helpers
# ---------------------------------------------------------------------------


def parse_raw_grid(seed_state: dict | list) -> np.ndarray:
    """Parse initial grid keeping raw values (10=ocean, 11=plains, etc.)."""
    raw = (
        seed_state.get("grid", seed_state)
        if isinstance(seed_state, dict)
        else seed_state
    )
    grid = np.zeros((MAP_H, MAP_W), dtype=np.int8)
    for y, row in enumerate(raw[:MAP_H]):
        for x, cell in enumerate(row[:MAP_W]):
            grid[y, x] = int(cell)
    return grid


def floor_norm(vol: np.ndarray, eps: float = PROB_FLOOR) -> np.ndarray:
    """Clip to eps floor and renormalise so each cell sums to 1."""
    p = np.maximum(vol, eps)
    return p / p.sum(-1, keepdims=True)


# ---------------------------------------------------------------------------
# Feature computation
# ---------------------------------------------------------------------------


def compute_features(raw_grid: np.ndarray) -> np.ndarray:
    """
    Compute features: (manhattan_dist, is_coastal, is_ocean).
    Returns (H, W, 3) float array.
    """
    h, w = raw_grid.shape
    sett_ys, sett_xs = np.where((raw_grid == 1) | (raw_grid == 2))

    dist = np.full((h, w), 999.0, dtype=np.float32)
    for sy, sx in zip(sett_ys, sett_xs):
        for y in range(h):
            for x in range(w):
                d = abs(y - sy) + abs(x - sx)
                if d < dist[y, x]:
                    dist[y, x] = d

    feats = np.zeros((h, w, 3), dtype=np.float32)
    for y in range(h):
        for x in range(w):
            feats[y, x, 0] = dist[y, x]
            coastal = False
            if raw_grid[y, x] != 10:
                for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < h and 0 <= nx < w and raw_grid[ny, nx] == 10:
                        coastal = True
                        break
            feats[y, x, 1] = float(coastal)
            feats[y, x, 2] = float(raw_grid[y, x] == 10)
    return feats


def dist_bucket(d: float) -> int:
    if d <= 1:
        return 0
    elif d <= 3:
        return 1
    elif d <= 6:
        return 2
    else:
        return 3


DIST_LABELS = {0: "d0-1", 1: "d2-3", 2: "d4-6", 3: "d7+"}


def get_bucket_keys(raw_terrain: int, dist: float, coastal: int) -> list[str]:
    """Hierarchical bucket keys (most specific → least specific)."""
    db = DIST_LABELS[dist_bucket(dist)]

    if raw_terrain == 10:
        return ["ocean"]
    if raw_terrain == 5:
        return ["mountain"]
    if raw_terrain == 11:
        return [f"plains_{db}_c{coastal}", f"plains_{db}", "plains", "land"]
    if raw_terrain == 4:
        return [f"forest_{db}_c{coastal}", f"forest_{db}", "forest", "land"]
    if raw_terrain == 1:
        return ["settlement", "land"]
    if raw_terrain == 2:
        return ["port", "settlement", "land"]
    if raw_terrain == 3:
        return ["ruin", "settlement", "land"]
    return ["land"]


# ---------------------------------------------------------------------------
# Querying
# ---------------------------------------------------------------------------


def _issue_viewports(
    session,
    round_id: str,
    seed_idx: int,
    viewports: list[tuple[int, int, int, int]],
    label: str = "",
    offset: int = 0,
) -> list[dict]:
    """Issue a list of viewports for one seed; return list of responses."""
    results = []
    for i, (vp_x, vp_y, vp_w, vp_h) in enumerate(viewports):
        try:
            print(
                f"    [{offset+i+1:2d}]  {label}  x={vp_x:2d} y={vp_y:2d} ... ",
                end="",
                flush=True,
            )
            resp = simulate(session, round_id, seed_idx, vp_x, vp_y, vp_w, vp_h)
            results.append(resp)
            used = resp.get("queries_used", "?")
            bmax = resp.get("queries_max", "?")
            print(f"ok  (budget {used}/{bmax})")
            time.sleep(0.25)
        except Exception as exc:
            print(f"ERROR: {exc}")
            time.sleep(2.0)
    return results


def query_multiseed(
    session, round_id: str, seed_budgets: dict[int, int]
) -> dict[int, list[dict]]:
    """Distribute queries across one or more seeds."""
    responses: dict[int, list[dict]] = {}
    global_offset = 0

    for seed_idx, budget in sorted(seed_budgets.items()):
        n_passes = budget // len(_FULL_PASS)
        leftover = budget % len(_FULL_PASS)
        all_vps = _FULL_PASS * n_passes + list(_FULL_PASS[:leftover])

        print(
            f"\n  Seed {seed_idx}: {budget} queries "
            f"({n_passes} full passes + {leftover} extra = {len(all_vps)} viewports)"
        )

        responses[seed_idx] = []
        results = _issue_viewports(
            session,
            round_id,
            seed_idx,
            all_vps,
            label=f"seed{seed_idx}",
            offset=global_offset,
        )
        responses[seed_idx].extend(results)
        global_offset += len(results)

    # Summary
    print()
    for seed_idx, resps in sorted(responses.items()):
        total_obs = sum(len(r["grid"]) * len(r["grid"][0]) for r in resps)
        unique_cells: set[tuple[int, int]] = set()
        for resp in resps:
            vp = resp["viewport"]
            for dy in range(len(resp["grid"])):
                for dx in range(len(resp["grid"][dy])):
                    unique_cells.add((vp["y"] + dy, vp["x"] + dx))
        obs_per_cell = total_obs // max(len(unique_cells), 1)
        print(
            f"  Seed {seed_idx}: {len(resps)} queries, "
            f"{len(unique_cells)} unique cells, "
            f"{total_obs} obs (~{obs_per_cell}/cell)"
        )

    return responses


def parse_query_seeds(spec: str, total_budget: int, num_seeds: int) -> dict[int, int]:
    """
    Parse --query-seeds specification into {seed: num_queries}.

    Formats:
      "0"           → all budget on seed 0
      "0,1"         → split evenly across seeds 0 and 1
      "0:27,1:18"   → 27 queries on seed 0, 18 on seed 1
      "0:3p,1:2p"   → 3 passes on seed 0, 2 passes on seed 1
      "all"         → split evenly across all 5 seeds
    """
    queries_per_pass = len(_FULL_PASS)

    if spec.lower() == "all":
        per_seed = total_budget // num_seeds
        return {i: per_seed for i in range(num_seeds)}

    parts = [p.strip() for p in spec.split(",")]
    has_explicit = any(":" in p for p in parts)

    if not has_explicit:
        seeds = [int(p) for p in parts]
        per_seed = total_budget // len(seeds)
        return {s: per_seed for s in seeds}

    result: dict[int, int] = {}
    for part in parts:
        seed_str, budget_str = part.split(":")
        seed = int(seed_str)
        if budget_str.endswith("p"):
            result[seed] = int(budget_str[:-1]) * queries_per_pass
        else:
            result[seed] = int(budget_str)

    used = sum(result.values())
    if used > total_budget:
        print(
            f"  WARNING: requested {used} queries but budget is {total_budget}. Scaling down."
        )
        scale = total_budget / used
        result = {k: max(1, int(v * scale)) for k, v in result.items()}

    return result


# ---------------------------------------------------------------------------
# Build historical prior (Layer 1)
# ---------------------------------------------------------------------------


def build_historical_prior() -> (
    tuple[dict[str, np.ndarray], dict[str, np.ndarray], int]
):
    """
    Build feature-based prior from ALL past round ground truths.

    Returns:
      hist_prior:   bucket_key → (6,) mean probability array
      hist_counts:  bucket_key → (6,) total weighted counts (for Bayesian update)
      n_samples:    number of (grid, ground_truth) pairs used
    """
    bucket_sums = defaultdict(lambda: np.zeros(NUM_CLASSES, dtype=np.float64))
    bucket_n = defaultdict(int)
    n_samples = 0

    if not DATA_DIR.exists():
        print(f"  WARNING: data directory {DATA_DIR} not found")
        return {}, {}, 0

    for round_dir in sorted(DATA_DIR.iterdir()):
        if not round_dir.is_dir() or not round_dir.name.startswith("round_"):
            continue
        for seed_idx in range(NUM_SEEDS):
            init_path = round_dir / f"seed_{seed_idx}_initial.json"
            gt_path = round_dir / f"seed_{seed_idx}_ground_truth.json"
            if not init_path.exists() or not gt_path.exists():
                continue

            with open(init_path) as f:
                init_data = json.load(f)
            with open(gt_path) as f:
                gt_data = json.load(f)

            raw_grid = parse_raw_grid(init_data)
            gt_vol = np.array(gt_data["ground_truth"], dtype=np.float64)
            feats = compute_features(raw_grid)
            n_samples += 1

            h, w = raw_grid.shape
            for y in range(h):
                for x in range(w):
                    raw_t = int(raw_grid[y, x])
                    dist = float(feats[y, x, 0])
                    coastal = int(feats[y, x, 1])
                    keys = get_bucket_keys(raw_t, dist, coastal)
                    for key in keys:
                        bucket_sums[key] += gt_vol[y, x]
                        bucket_n[key] += 1

    hist_prior: dict[str, np.ndarray] = {}
    hist_counts: dict[str, np.ndarray] = {}
    for bucket in bucket_sums:
        hist_prior[bucket] = (bucket_sums[bucket] / bucket_n[bucket]).astype(np.float32)
        hist_counts[bucket] = bucket_sums[bucket].astype(np.float32)

    if n_samples > 0:
        print(
            f"  Built historical prior from {n_samples} ground truth maps "
            f"({len(hist_prior)} buckets)"
        )
    else:
        print("  WARNING: no historical data found")

    return hist_prior, hist_counts, n_samples


# ---------------------------------------------------------------------------
# Build observation prior + per-cell data (Layer 2 & 3)
# ---------------------------------------------------------------------------


def build_observation_data(
    responses: dict[int, list[dict]],
    raw_grids: list[np.ndarray],
    feature_arrays: list[np.ndarray],
    map_h: int = MAP_H,
    map_w: int = MAP_W,
) -> tuple[dict[str, np.ndarray], dict[int, tuple[np.ndarray, np.ndarray]]]:
    """
    Build bucket-level observation counts and per-cell data from queries.

    Returns:
      obs_bucket_counts: bucket_key → (6,) raw count array
      percell_data:      seed_idx → (obs_counts[H,W,6], obs_total[H,W])
    """
    bucket_counts = defaultdict(lambda: np.zeros(NUM_CLASSES, dtype=np.float64))
    percell_data: dict[int, tuple[np.ndarray, np.ndarray]] = {}

    for seed_idx, resp_list in responses.items():
        raw_grid = raw_grids[seed_idx]
        feats = feature_arrays[seed_idx]
        obs_counts = np.zeros((map_h, map_w, NUM_CLASSES), dtype=np.float64)
        obs_total = np.zeros((map_h, map_w), dtype=np.float64)

        for resp in resp_list:
            vp = resp["viewport"]
            qgrid = resp["grid"]
            for dy, row in enumerate(qgrid):
                for dx, val in enumerate(row):
                    gy, gx = vp["y"] + dy, vp["x"] + dx
                    if 0 <= gy < map_h and 0 <= gx < map_w:
                        final_cls = INT_CODE_MAP.get(int(val), 0)
                        raw_t = int(raw_grid[gy, gx])
                        dist = float(feats[gy, gx, 0])
                        coastal = int(feats[gy, gx, 1])

                        keys = get_bucket_keys(raw_t, dist, coastal)
                        for key in keys:
                            bucket_counts[key][final_cls] += 1

                        obs_counts[gy, gx, final_cls] += 1
                        obs_total[gy, gx] += 1

        percell_data[seed_idx] = (obs_counts, obs_total)

    return dict(bucket_counts), percell_data


# ---------------------------------------------------------------------------
# Merge historical + observations (Layer 2)
# ---------------------------------------------------------------------------


def merge_priors(
    hist_prior: dict[str, np.ndarray],
    obs_bucket_counts: dict[str, np.ndarray],
    hist_weight: float = HIST_WEIGHT,
) -> dict[str, np.ndarray]:
    """
    Bayesian merge of historical prior with current-round observations.

    For each bucket:
      posterior = (hist_weight * hist_prior + obs_counts) / (hist_weight + obs_total)

    hist_weight controls how much we trust the historical prior vs observations.
    Higher = more conservative (stay close to historical).
    Lower = more adaptive (trust current round observations more).

    Recommended: 100-300 (observations typically total ~200-1000 per bucket).
    """
    merged: dict[str, np.ndarray] = {}
    all_keys = set(hist_prior.keys()) | set(obs_bucket_counts.keys())

    for key in all_keys:
        h_val = hist_prior.get(key)
        o_cts = obs_bucket_counts.get(key)

        if h_val is not None and o_cts is not None:
            o_total = o_cts.sum()
            merged[key] = (
                (hist_weight * h_val + o_cts) / (hist_weight + o_total)
            ).astype(np.float32)
        elif h_val is not None:
            merged[key] = h_val
        elif o_cts is not None:
            o_total = o_cts.sum()
            if o_total > 0:
                merged[key] = (o_cts / o_total).astype(np.float32)

    return merged


# ---------------------------------------------------------------------------
# Predict (all 3 layers)
# ---------------------------------------------------------------------------


def predict_seed(
    prior_table: dict[str, np.ndarray],
    raw_grid: np.ndarray,
    feats: np.ndarray,
    percell: tuple[np.ndarray, np.ndarray] | None = None,
    alpha: float = PERCELL_ALPHA,
    floor: float = PROB_FLOOR,
    map_h: int = MAP_H,
    map_w: int = MAP_W,
) -> np.ndarray:
    """
    Generate prediction for one seed.

    Layer 1+2: Use merged prior table (historical + obs calibration).
    Layer 3: Blend with per-cell observations if available.
    """
    pred = np.zeros((map_h, map_w, NUM_CLASSES), dtype=np.float32)
    uniform = np.full(NUM_CLASSES, 1.0 / NUM_CLASSES, dtype=np.float32)

    for y in range(map_h):
        for x in range(map_w):
            raw_t = int(raw_grid[y, x])
            dist = float(feats[y, x, 0])
            coastal = int(feats[y, x, 1])

            keys = get_bucket_keys(raw_t, dist, coastal)
            found = False
            for key in keys:
                if key in prior_table:
                    pred[y, x] = prior_table[key]
                    found = True
                    break
            if not found:
                pred[y, x] = uniform

    # Layer 3: per-cell Bayesian blending
    if percell is not None:
        obs_counts, obs_total = percell
        for y in range(map_h):
            for x in range(map_w):
                n = obs_total[y, x]
                raw_t = int(raw_grid[y, x])
                if n > 0 and raw_t not in (10, 5):
                    empirical = obs_counts[y, x] / n
                    pred[y, x] = (alpha * pred[y, x] + n * empirical) / (alpha + n)

    return floor_norm(pred, eps=floor)


def print_prior(prior_table: dict[str, np.ndarray], title: str) -> None:
    print(f"\n  {title}:")
    for bucket in sorted(prior_table.keys()):
        dist = prior_table[bucket]
        parts = "  ".join(f"{CLASS_NAMES[c]}:{dist[c]:.3f}" for c in range(NUM_CLASSES))
        print(f"    {bucket:35s}  {parts}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Astar Island — historical prior + obs calibration solver (v3)"
    )
    parser.add_argument("--round-id", help="Round UUID (auto-detect if omitted)")
    parser.add_argument("--token", help="JWT token (fallback: AINM_TOKEN env var)")
    parser.add_argument("--dry-run", action="store_true", help="Skip all API calls")
    parser.add_argument(
        "--skip-query",
        action="store_true",
        help="Submit using historical prior only (no queries)",
    )
    parser.add_argument(
        "--concentrate-seed",
        type=int,
        default=None,
        metavar="SEED",
        help="Spend ALL queries on one seed (shortcut for --query-seeds N)",
    )
    parser.add_argument(
        "--query-seeds",
        type=str,
        default=None,
        help="Query strategy: '0', '0,1', '0:3p,1:2p', 'all'",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=PERCELL_ALPHA,
        help=f"Per-cell blending weight (default: {PERCELL_ALPHA})",
    )
    parser.add_argument(
        "--hist-weight",
        type=float,
        default=HIST_WEIGHT,
        help=f"Historical prior weight for Bayesian merge (default: {HIST_WEIGHT})",
    )
    parser.add_argument(
        "--floor",
        type=float,
        default=PROB_FLOOR,
        help=f"Probability floor (default: {PROB_FLOOR})",
    )
    args = parser.parse_args()

    token = os.getenv("AINM_TOKEN") or args.token
    if not token and not args.dry_run:
        print("ERROR: set AINM_TOKEN env var or pass --token <jwt>")
        sys.exit(1)
    token = token or "dummy"

    import requests  # deferred import

    # ==================================================================
    # Step 1: Historical prior (from past rounds)
    # ==================================================================
    print("=" * 60)
    print("STEP 1: Build historical prior from past rounds")
    print("=" * 60)
    hist_prior, hist_counts, n_hist = build_historical_prior()
    if n_hist > 0:
        print_prior(hist_prior, f"Historical prior ({n_hist} maps)")
    else:
        print("  No historical data — will rely on observations only")

    # ==================================================================
    # Step 2: Find active round
    # ==================================================================
    print("\n" + "=" * 60)
    print("STEP 2: Find active round")
    print("=" * 60)

    session = _make_session(token)
    round_id = args.round_id

    if round_id is None:
        rounds = get_rounds(session)
        active = [r for r in rounds if r.get("status") == "active"]
        if not active:
            print("ERROR: no active round found")
            sys.exit(1)
        round_id = active[0]["id"]
        print(f"  Active round #{active[0].get('round_number')}  id={round_id}")

    round_data = get_round(session, round_id)
    map_h = round_data.get("map_height", MAP_H)
    map_w = round_data.get("map_width", MAP_W)
    num_seeds = round_data.get("seeds_count", NUM_SEEDS)
    init_states = round_data.get("initial_states", [])
    print(f"  Map: {map_w}x{map_h},  seeds: {num_seeds}")

    raw_grids = [parse_raw_grid(s) for s in init_states]

    print("  Precomputing spatial features for all seeds ...")
    feature_arrays = [compute_features(rg) for rg in raw_grids]

    # ==================================================================
    # Step 3: Query and build observation data
    # ==================================================================
    print("\n" + "=" * 60)
    print("STEP 3: Query active round")
    print("=" * 60)

    obs_bucket_counts: dict[str, np.ndarray] = {}
    percell_data: dict[int, tuple[np.ndarray, np.ndarray]] = {}

    if args.skip_query or args.dry_run:
        reason = "--skip-query" if args.skip_query else "--dry-run"
        print(f"  {reason}: using historical prior only.")
    else:
        try:
            budget_info = get_budget(session)
            used = budget_info.get("queries_used", 0)
            total = budget_info.get("queries_max", TOTAL_BUDGET)
            remaining = total - used
            print(f"  Budget: {used}/{total} used,  {remaining} remaining")
        except Exception as exc:
            print(f"  [warn] Could not fetch budget: {exc}")
            remaining = TOTAL_BUDGET

        if remaining <= 0:
            print("  No budget remaining.")
        else:
            # Determine query strategy
            if args.concentrate_seed is not None:
                seed_budgets = {args.concentrate_seed: remaining}
            elif args.query_seeds is not None:
                seed_budgets = parse_query_seeds(args.query_seeds, remaining, num_seeds)
            else:
                seed_budgets = {0: remaining}

            seed_budgets = {min(s, num_seeds - 1): b for s, b in seed_budgets.items()}
            desc = ", ".join(f"seed {s}: {b}q" for s, b in sorted(seed_budgets.items()))
            print(f"  Strategy: {desc}")

            responses = query_multiseed(session, round_id, seed_budgets)

            print("\n  Building observation data ...")
            obs_bucket_counts, percell_data = build_observation_data(
                responses,
                raw_grids,
                feature_arrays,
                map_h,
                map_w,
            )
            print(f"  Observation buckets: {len(obs_bucket_counts)}")
            print(f"  Seeds with per-cell data: {sorted(percell_data.keys())}")

    # ==================================================================
    # Step 4: Merge priors
    # ==================================================================
    print("\n" + "=" * 60)
    print("STEP 4: Merge historical prior with observations")
    print("=" * 60)

    if n_hist > 0 and obs_bucket_counts:
        print(f"  Bayesian merge: hist_weight={args.hist_weight}")
        final_prior = merge_priors(hist_prior, obs_bucket_counts, args.hist_weight)
        print_prior(final_prior, "Merged prior (historical + obs)")
    elif n_hist > 0:
        print("  Using historical prior only (no observations)")
        final_prior = hist_prior
    elif obs_bucket_counts:
        print("  Using observations only (no historical data)")
        final_prior = {
            b: cts / cts.sum() for b, cts in obs_bucket_counts.items() if cts.sum() > 0
        }
        print_prior(final_prior, "Observation-only prior")
    else:
        print("  WARNING: no data at all — using uniform prior")
        final_prior = {
            "land": np.full(NUM_CLASSES, 1.0 / NUM_CLASSES, dtype=np.float32)
        }

    # ==================================================================
    # Step 5: Predict and submit
    # ==================================================================
    print("\n" + "=" * 60)
    print(f"STEP 5: Predict and submit  (floor={args.floor}, alpha={args.alpha})")
    print("=" * 60)

    for seed_idx in range(num_seeds):
        raw_grid = raw_grids[seed_idx]
        feats = feature_arrays[seed_idx]
        percell = percell_data.get(seed_idx, None)

        pred = predict_seed(
            final_prior,
            raw_grid,
            feats,
            percell=percell,
            alpha=args.alpha,
            floor=args.floor,
            map_h=map_h,
            map_w=map_w,
        )
        max_err = float(abs(pred.sum(-1) - 1.0).max())
        classes = np.unique(pred.argmax(-1)).tolist()
        blended = "per-cell blended" if percell is not None else "bucket-only"
        src = (
            "hist+obs"
            if (n_hist > 0 and obs_bucket_counts)
            else ("hist" if n_hist > 0 else "obs")
        )
        print(
            f"  Seed {seed_idx}:  max-err={max_err:.2e}  "
            f"classes={classes}  ({blended}, {src})"
        )

        if args.dry_run:
            print(f"    [dry-run] skipping submit  shape={pred.shape}")
            continue

        print(f"    Submitting ...", end=" ", flush=True)
        try:
            result = submit_prediction(session, round_id, seed_idx, pred)
            status = result.get("status", result.get("score", "?"))
            print(f"ok  ({status})")
        except Exception as exc:
            print(f"SUBMIT ERROR: {exc}")
        time.sleep(0.6)

    print("\nDone!")


if __name__ == "__main__":
    main()
