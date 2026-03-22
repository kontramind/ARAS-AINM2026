"""
tasks/aastar-island/data/fetch_data.py
--------------------------------------
Fetch and store all data needed by basic_script.py from the AINM API.

Output layout (relative to this directory):
  round_1/
    seed_0_initial.json
    seed_0_ground_truth.json
    ...
  round_2/ ... round_N/
    seed_{i}_initial.json
    seed_{i}_ground_truth.json   ← only for completed rounds

Initial JSON format:
  { "grid": [[10, 11, ...], ...],
    "settlements": [{"x": 5, "y": 12, "has_port": true, "alive": true}, ...] }

Ground truth JSON format (H x W x 6 probability tensor):
  { "ground_truth": [[[0.9, 0.0, ...], ...], ...] }

Usage:
  # Full fetch: initial states + ground truth for all completed rounds
  AINM_TOKEN=<jwt> python tasks/astar-island/data/fetch_data.py

  # Skip ground truth (initial states only)
  AINM_TOKEN=<jwt> python tasks/astar-island/data/fetch_data.py --no-ground-truth
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import requests

try:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).parent.parent.parent / ".env")  # project root .env
except ImportError:
    pass  # python-dotenv not installed; rely on environment variable

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
THIS_DIR = Path(__file__).parent.resolve() / "data"  # tasks/astar-island/data/

BASE_URL = "https://api.ainm.no/astar-island"
NUM_SEEDS = 5
RETRY_SLEEP = 1.0  # seconds between retries on rate-limit (429)
MAX_RETRIES = 3


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------


def _make_session(token: str | None) -> requests.Session:
    import urllib3

    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    s = requests.Session()
    if token:
        s.headers["Authorization"] = f"Bearer {token}"
    s.verify = False  # Corporate proxy does SSL inspection
    return s


def _get(session: requests.Session, url: str) -> dict:
    for attempt in range(MAX_RETRIES):
        r = session.get(url)
        if r.status_code == 429:
            print(f"  Rate-limited, waiting {RETRY_SLEEP}s ...", flush=True)
            time.sleep(RETRY_SLEEP)
            continue
        r.raise_for_status()
        return r.json()
    raise RuntimeError(f"GET {url} failed after {MAX_RETRIES} retries")


# ---------------------------------------------------------------------------
# API fetchers
# ---------------------------------------------------------------------------


def fetch_rounds(session: requests.Session) -> list[dict]:
    """Return all rounds, sorted by round_number ascending."""
    rounds = _get(session, f"{BASE_URL}/rounds")
    return sorted(rounds, key=lambda r: r.get("round_number", 0))


def fetch_round_detail(session: requests.Session, round_id: str) -> dict:
    return _get(session, f"{BASE_URL}/rounds/{round_id}")


def fetch_ground_truth(
    session: requests.Session, round_id: str, seed_index: int
) -> dict | None:
    """
    Returns the analysis payload for one seed, or None if not yet available.
    The analysis endpoint is only available after a round is completed/scoring.
    """
    url = f"{BASE_URL}/analysis/{round_id}/{seed_index}"
    try:
        data = _get(session, url)
        return data
    except requests.HTTPError as e:
        if e.response is not None and e.response.status_code in (400, 404):
            return None  # Round not yet scored or no prediction submitted
        raise


# ---------------------------------------------------------------------------
# Save helpers
# ---------------------------------------------------------------------------


def save_json(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f)
    print(f"  Saved {path.relative_to(THIS_DIR)}", flush=True)


# ---------------------------------------------------------------------------
# Core logic: initial states
# ---------------------------------------------------------------------------


def fetch_and_save_initial_states(
    session: requests.Session,
    round_number: int,
    round_id: str,
) -> None:
    print(f"Fetching round {round_number} ({round_id}) initial states ...", flush=True)
    detail = fetch_round_detail(session, round_id)
    initial_states: list[dict] = detail.get("initial_states", [])

    if not initial_states:
        print(f"  WARNING: no initial_states in response for round {round_number}")
        return

    for seed_idx, state in enumerate(initial_states):
        path = THIS_DIR / f"round_{round_number}" / f"seed_{seed_idx}_initial.json"
        # Keep only the fields basic_script.py reads: grid + settlements
        save_obj = {
            "grid": state["grid"],
            "settlements": state.get("settlements", []),
        }
        save_json(path, save_obj)


# ---------------------------------------------------------------------------
# Core logic: ground truth
# ---------------------------------------------------------------------------


def fetch_and_save_ground_truth(
    session: requests.Session,
    round_number: int,
    round_id: str,
    num_seeds: int = NUM_SEEDS,
) -> None:
    print(f"Fetching round {round_number} ground truth ...", flush=True)
    for seed_idx in range(num_seeds):
        data = fetch_ground_truth(session, round_id, seed_idx)
        if data is None:
            print(f"  seed {seed_idx}: not available yet, skipping")
            continue
        if "ground_truth" not in data:
            print(
                f"  seed {seed_idx}: 'ground_truth' key missing in response, skipping"
            )
            continue
        path = THIS_DIR / f"round_{round_number}" / f"seed_{seed_idx}_ground_truth.json"
        save_json(path, {"ground_truth": data["ground_truth"]})


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch AINM Astar Island data and store it locally.",
    )
    parser.add_argument(
        "--token",
        default=os.environ.get("AINM_TOKEN"),
        help="JWT token (defaults to AINM_TOKEN env var)",
    )
    parser.add_argument(
        "--rounds",
        nargs="*",
        type=int,
        metavar="N",
        help="Round numbers to fetch (default: all rounds returned by the API)",
    )
    parser.add_argument(
        "--no-ground-truth",
        action="store_true",
        help="Skip fetching ground truth — only fetch initial states",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.token:
        print(
            "ERROR: AINM_TOKEN not set. "
            "Pass --token <jwt> or set the AINM_TOKEN environment variable.\n"
            "Run with --logs-only to skip API calls.",
            file=sys.stderr,
        )
        sys.exit(1)

    session = _make_session(args.token)

    rounds = fetch_rounds(session)
    if not rounds:
        print("ERROR: no rounds returned from API", file=sys.stderr)
        sys.exit(1)

    # Filter to requested round numbers
    if args.rounds:
        requested = set(args.rounds)
        rounds = [r for r in rounds if r.get("round_number") in requested]
        if not rounds:
            print(
                f"ERROR: none of the requested rounds {sorted(requested)} found in API",
                file=sys.stderr,
            )
            sys.exit(1)

    for rnd in rounds:
        round_id: str = rnd["id"]
        round_number: int = rnd["round_number"]
        status: str = rnd.get("status", "")

        # Initial states
        fetch_and_save_initial_states(session, round_number, round_id)

        # Ground truth — only available for completed / scoring rounds
        if not args.no_ground_truth:
            if status in ("completed", "scoring"):
                fetch_and_save_ground_truth(session, round_number, round_id)
            else:
                print(
                    f"  Round {round_number} status='{status}' — "
                    "ground truth not yet available, skipping"
                )

    print("\nAll done.", flush=True)


if __name__ == "__main__":
    main()
