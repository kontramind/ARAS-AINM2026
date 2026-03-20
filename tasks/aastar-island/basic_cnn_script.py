"""
tasks/aastar-island/basic_cnn_script.py
---------------------------------------
Minimal CNN solver for Astar Island.

Steps:
  1. Train a ResNet-style CNN on historical rounds (initial map -> ground truth).
  2. Query the active round (random viewports per seed) and log responses.
  3. Predict using only the trained CNN and submit.

Queries are logged for later analysis but NOT used to modify predictions.

Usage:
  AINM_TOKEN=<jwt> python tasks/aastar-island/basic_cnn_script.py
  AINM_TOKEN=<jwt> python tasks/aastar-island/basic_cnn_script.py --round-id <uuid>
  AINM_TOKEN=<jwt> python tasks/aastar-island/basic_cnn_script.py --dry-run
  AINM_TOKEN=<jwt> python tasks/aastar-island/basic_cnn_script.py --skip-query
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np

try:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parents[2] / ".env")
except ImportError:
    pass

import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Re-use API helpers from solve.py
# ---------------------------------------------------------------------------
_solve_path = Path(__file__).parent / "solve.py"
_spec = importlib.util.spec_from_file_location("solve", _solve_path)
_solve = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_solve)

BASE_URL: str = _solve.BASE_URL
MAP_H: int = _solve.MAP_H
MAP_W: int = _solve.MAP_W
MAX_VIEWPORT: int = _solve.MAX_VIEWPORT
NUM_CLASSES: int = _solve.NUM_CLASSES
NUM_SEEDS: int = _solve.NUM_SEEDS
PROB_FLOOR: float = _solve.PROB_FLOOR
TOTAL_BUDGET: int = _solve.TOTAL_BUDGET
QUERIES_PER_SEED: int = TOTAL_BUDGET // NUM_SEEDS  # 10

_session = _solve._session
get_budget = _solve.get_budget
get_round = _solve.get_round
get_rounds = _solve.get_rounds
simulate = _solve.simulate
submit = _solve.submit

INT_CODE_MAP = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 10: 0, 11: 0}

DATA_DIR = Path(__file__).parent / "data"
QUERY_LOG_DIR = Path(__file__).resolve().parents[2] / "data" / "query_logs"
MODEL_DIR = Path(__file__).resolve().parents[2] / "models"

# ---------------------------------------------------------------------------
# CNN Architecture
# ---------------------------------------------------------------------------


class ResBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.block(x) + x)


class TinyCNN(nn.Module):
    def __init__(self, num_res_blocks: int = 3):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(NUM_CLASSES, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.res_blocks = nn.Sequential(*[ResBlock(64) for _ in range(num_res_blocks)])
        self.head = nn.Conv2d(64, NUM_CLASSES, 1)

    def forward(self, x):
        x = self.stem(x)
        x = self.res_blocks(x)
        return self.head(x)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def parse_initial_grid(seed_state) -> np.ndarray:
    if isinstance(seed_state, dict):
        raw = seed_state.get("grid", seed_state)
    else:
        raw = seed_state
    grid = np.zeros((MAP_H, MAP_W), dtype=np.int8)
    for y, row in enumerate(raw[:MAP_H]):
        for x, cell in enumerate(row[:MAP_W]):
            grid[y, x] = INT_CODE_MAP.get(int(cell), 0)
    return grid


def grid_to_tensor(grid: np.ndarray) -> torch.Tensor:
    H, W = grid.shape
    oh = np.zeros((1, NUM_CLASSES, H, W), dtype=np.float32)
    for c in range(NUM_CLASSES):
        oh[0, c] = (grid == c).astype(np.float32)
    return torch.tensor(oh)


def floor_norm(vol: np.ndarray, eps: float = PROB_FLOOR) -> np.ndarray:
    p = np.maximum(vol, eps)
    return p / p.sum(-1, keepdims=True)


def load_local_training_data() -> list[tuple[np.ndarray, np.ndarray]]:
    samples = []
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
            grid = parse_initial_grid(init_data)
            gt_vol = np.array(gt_data["ground_truth"], dtype=np.float32)
            samples.append((grid, gt_vol))
    return samples


def ainm_score(
    gt_vol: np.ndarray, pred_vol: np.ndarray, eps: float = PROB_FLOOR
) -> float:
    p = np.maximum(pred_vol, eps)
    p = p / p.sum(-1, keepdims=True)
    gt = np.clip(gt_vol, 1e-10, 1.0)
    entropy = -(gt * np.log(gt)).sum(-1)
    kl = (gt * np.log(gt / p)).sum(-1)
    w = entropy.sum()
    if w < 1e-10:
        return 100.0
    wkl = (entropy * kl).sum() / w
    return float(max(0.0, min(100.0, 100.0 * np.exp(-3.0 * wkl))))


# ---------------------------------------------------------------------------
# Query logging
# ---------------------------------------------------------------------------


class QueryLogger:
    def __init__(self, round_id: str):
        self.round_id = round_id
        self.log_dir = QUERY_LOG_DIR / round_id
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def log(self, seed_index: int, viewport: tuple, response: dict, phase: str = ""):
        vp_x, vp_y, vp_w, vp_h = viewport
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "round_id": self.round_id,
            "seed_index": seed_index,
            "viewport": {"x": vp_x, "y": vp_y, "w": vp_w, "h": vp_h},
            "phase": phase,
            "response": response,
        }
        log_file = self.log_dir / f"seed_{seed_index}.jsonl"
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")


# ---------------------------------------------------------------------------
# Train CNN
# ---------------------------------------------------------------------------


def train_cnn(
    samples: list[tuple[np.ndarray, np.ndarray]],
    n_epochs: int = 250,
    lr: float = 1e-3,
    device: str = "cpu",
) -> TinyCNN:
    model = TinyCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    kl_loss = nn.KLDivLoss(reduction="batchmean")

    print(f"\n--- Training CNN on {len(samples)} maps [{device}] ---")
    model.train()
    for epoch in range(n_epochs):
        total_loss = 0.0
        for grid, gt in samples:
            x = grid_to_tensor(grid).to(device)
            target = torch.tensor(gt[None], dtype=torch.float32).to(device)
            target_hwc = target.reshape(-1, NUM_CLASSES)

            logits = model(x)
            log_p = nn.functional.log_softmax(logits, dim=1)
            log_p_hwc = log_p.permute(0, 2, 3, 1).reshape(-1, NUM_CLASSES)

            loss = kl_loss(log_p_hwc, target_hwc)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if (epoch + 1) % 50 == 0:
            avg = total_loss / len(samples)
            print(f"  epoch {epoch + 1:3d}/{n_epochs}  loss={avg:.4f}")

    model.eval()
    return model


# ---------------------------------------------------------------------------
# Query round (random viewports)
# ---------------------------------------------------------------------------


def generate_random_viewports(
    n: int, map_h: int = MAP_H, map_w: int = MAP_W, vp_size: int = MAX_VIEWPORT
) -> list[tuple[int, int, int, int]]:
    tile_positions = []
    for vy in range(0, map_h, vp_size):
        for vx in range(0, map_w, vp_size):
            x0 = min(vx, map_w - vp_size)
            y0 = min(vy, map_h - vp_size)
            w = min(vp_size, map_w - x0)
            h = min(vp_size, map_h - y0)
            tile_positions.append((x0, y0, max(w, 5), max(h, 5)))

    tile_positions = list(dict.fromkeys(tile_positions))

    rng = np.random.default_rng(42)
    if len(tile_positions) >= n:
        indices = rng.choice(len(tile_positions), size=n, replace=False)
        viewports = [tile_positions[i] for i in indices]
    else:
        viewports = list(tile_positions)
        while len(viewports) < n:
            vx = int(rng.integers(0, max(1, map_w - vp_size + 1)))
            vy = int(rng.integers(0, max(1, map_h - vp_size + 1)))
            w = min(vp_size, map_w - vx)
            h = min(vp_size, map_h - vy)
            viewports.append((vx, vy, max(w, 5), max(h, 5)))

    return viewports[:n]


def query_round(
    session,
    round_id: str,
    num_seeds: int,
    queries_per_seed: int,
    logger: QueryLogger,
) -> None:
    """Query the active round and log responses. Returns nothing."""
    for seed_idx in range(num_seeds):
        viewports = generate_random_viewports(queries_per_seed)
        print(f"\n  Seed {seed_idx}: querying {len(viewports)} viewports")

        for i, (vp_x, vp_y, vp_w, vp_h) in enumerate(viewports):
            try:
                print(
                    f"    query {i + 1}/{len(viewports)}: "
                    f"x={vp_x} y={vp_y} w={vp_w} h={vp_h} ... ",
                    end="",
                    flush=True,
                )
                resp = simulate(session, round_id, seed_idx, vp_x, vp_y, vp_w, vp_h)
                logger.log(
                    seed_idx, (vp_x, vp_y, vp_w, vp_h), resp, phase="basic_explore"
                )
                print("ok")
                time.sleep(0.25)
            except Exception as e:
                print(f"ERROR: {e}")
                time.sleep(2)


# ---------------------------------------------------------------------------
# Predict and submit (pure CNN)
# ---------------------------------------------------------------------------


def predict_and_submit(
    session,
    model: TinyCNN,
    round_id: str,
    initial_grids: list[np.ndarray],
    num_seeds: int,
    device: str = "cpu",
    dry_run: bool = False,
):
    print("\n--- Predicting and submitting (pure CNN) ---")

    for seed_idx in range(num_seeds):
        grid = initial_grids[seed_idx]
        x = grid_to_tensor(grid).to(device)

        with torch.no_grad():
            logits = model(x)
            proba = nn.functional.softmax(logits, dim=1)
            pred_np = proba[0].permute(1, 2, 0).cpu().numpy()

        pred_np = floor_norm(pred_np)

        cell_sums = pred_np.sum(axis=-1)
        max_err = float(abs(cell_sums - 1.0).max())
        print(f"  Seed {seed_idx}: max prob sum error = {max_err:.2e}")

        if dry_run:
            print(f"  [dry-run] Would submit prediction shape={pred_np.shape}")
            continue

        print(f"  Submitting seed {seed_idx} ...", end=" ", flush=True)
        try:
            result = submit(session, round_id, seed_idx, pred_np)
            score = result.get("score", result.get("seed_score", "?"))
            print(f"ok  score={score}")
        except Exception as e:
            print(f"SUBMIT ERROR: {e}")

        time.sleep(0.6)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Astar Island basic CNN solver")
    parser.add_argument("--round-id", help="Round UUID (auto-detect if omitted)")
    parser.add_argument("--token", help="JWT token (fallback if AINM_TOKEN not set)")
    parser.add_argument(
        "--dry-run", action="store_true", help="Don't call API for queries/submissions"
    )
    parser.add_argument(
        "--skip-query",
        action="store_true",
        help="Skip querying (queries are logged only, not used for predictions)",
    )
    parser.add_argument(
        "--epochs", type=int, default=250, help="Training epochs (default: 250)"
    )
    parser.add_argument(
        "--save-model", action="store_true", help="Save trained model to models/"
    )
    args = parser.parse_args()

    token = os.getenv("AINM_TOKEN") or args.token
    if not token and not args.dry_run:
        print("Set AINM_TOKEN env var or pass --token")
        sys.exit(1)
    if not token:
        token = "dummy"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # ==================================================================
    # Step 1: Train CNN on historical data
    # ==================================================================
    print("\n" + "=" * 60)
    print("STEP 1: Load historical data and train CNN")
    print("=" * 60)

    samples = load_local_training_data()
    if not samples:
        print("ERROR: No training data found in", DATA_DIR)
        print("Run fetch_data.py first to download historical rounds.")
        sys.exit(1)
    print(f"Loaded {len(samples)} (grid, ground_truth) pairs from local data")

    model = train_cnn(samples, n_epochs=args.epochs, device=device)

    # Quick sanity check on training data
    model.eval()
    scores = []
    with torch.no_grad():
        for grid, gt in samples[-5:]:
            x = grid_to_tensor(grid).to(device)
            proba = nn.functional.softmax(model(x), dim=1)
            pred_np = floor_norm(proba[0].permute(1, 2, 0).cpu().numpy())
            scores.append(ainm_score(gt, pred_np))
    print(f"  Training scores (last 5): {[f'{s:.1f}' for s in scores]}")

    if args.save_model:
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        model_path = MODEL_DIR / "cnn_astar.pt"
        torch.save(model.state_dict(), model_path)
        print(f"  Model saved to {model_path}")

    # ==================================================================
    # Step 2: Find active round and query (log only)
    # ==================================================================
    print("\n" + "=" * 60)
    print("STEP 2: Find active round and query (log only)")
    print("=" * 60)

    session = _session(token)

    round_id = args.round_id
    if round_id is None:
        rounds = get_rounds(session)
        active = [r for r in rounds if r.get("status") == "active"]
        if not active:
            print("No active round found. Available rounds:")
            for r in rounds:
                print(
                    f"  {r['id']}  status={r.get('status')}  name={r.get('name', '?')}"
                )
            sys.exit(1)
        round_id = active[0]["id"]
    print(f"Round: {round_id}")

    round_data = get_round(session, round_id)
    num_seeds = round_data.get("num_seeds", round_data.get("seeds_count", NUM_SEEDS))
    print(f"  Seeds: {num_seeds}")

    initial_states = round_data.get(
        "initial_states", round_data.get("seeds", [None] * num_seeds)
    )

    initial_grids = []
    for seed_idx in range(num_seeds):
        if seed_idx < len(initial_states) and initial_states[seed_idx] is not None:
            s = initial_states[seed_idx]
            raw = s.get("grid", s) if isinstance(s, dict) else s
            grid = parse_initial_grid({"grid": raw} if isinstance(raw, list) else raw)
        else:
            grid = np.zeros((MAP_H, MAP_W), dtype=np.int8)
        initial_grids.append(grid)

    # Query and log (does not affect predictions)
    if not args.skip_query and not args.dry_run:
        try:
            budget_info = get_budget(session)
            queries_used = budget_info.get("queries_used", 0)
            queries_max = budget_info.get("queries_max", TOTAL_BUDGET)
            remaining = queries_max - queries_used
            print(f"  Budget: {queries_used}/{queries_max} used, {remaining} remaining")
        except Exception as e:
            print(f"  [warn] Could not fetch budget: {e}")
            remaining = TOTAL_BUDGET

        actual_qps = min(QUERIES_PER_SEED, remaining // num_seeds)
        if actual_qps <= 0:
            print("  No query budget remaining.")
        else:
            print(
                f"  Querying {actual_qps} viewports per seed "
                f"({actual_qps * num_seeds} total) — logged for future use"
            )
            logger = QueryLogger(round_id)
            query_round(session, round_id, num_seeds, actual_qps, logger)
    else:
        print(
            "  Skipping queries."
            if args.skip_query
            else "  [dry-run] Skipping queries."
        )

    # ==================================================================
    # Step 3: Predict (pure CNN) and submit
    # ==================================================================
    print("\n" + "=" * 60)
    print("STEP 3: Predict (pure CNN) and submit")
    print("=" * 60)

    predict_and_submit(
        session,
        model,
        round_id,
        initial_grids,
        num_seeds,
        device=device,
        dry_run=args.dry_run,
    )

    print("\nDone!")


if __name__ == "__main__":
    main()
