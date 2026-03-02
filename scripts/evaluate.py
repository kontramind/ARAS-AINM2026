"""
scripts/evaluate.py
--------------------
Local scoring harness for all three competition task types.

Purpose:
  - Mimic the competition judge's evaluation BEFORE submission
  - Measure both ACCURACY and LATENCY (both matter in competition)
  - Identify which task needs the most work

Usage:
  # Evaluate all three tasks against local test data
  python scripts/evaluate.py --task all --data data/

  # Evaluate a specific task
  python scripts/evaluate.py --task 1 --data data/task1_test.csv
  python scripts/evaluate.py --task 2 --data data/task2_test.csv
  python scripts/evaluate.py --task 3 --data data/task3_images/

  # Against a running API (use when testing end-to-end)
  python scripts/evaluate.py --task all --mode api --api-url http://localhost:8000

  # Load a saved model artifact and evaluate locally
  python scripts/evaluate.py --task 1 --model models/tabular_v1.pkl
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import requests


# ===========================================================================
# Metrics helpers
# ===========================================================================

def classification_metrics(y_true, y_pred) -> dict:
    from sklearn.metrics import accuracy_score, f1_score, classification_report
    acc = accuracy_score(y_true, y_pred)
    f1  = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    print(classification_report(y_true, y_pred, zero_division=0))
    return {"accuracy": round(acc, 4), "f1_weighted": round(f1, 4)}


def regression_metrics(y_true, y_pred) -> dict:
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return {"mae": round(mae, 4), "rmse": round(rmse, 4)}


def segmentation_metrics(preds: list, targets: list) -> dict:
    from tasks.vision.segmentation import SegmentationPipeline
    pipeline = SegmentationPipeline()
    return pipeline.evaluate(preds, targets)


def measure_latency(func, n_repeats: int = 10) -> float:
    """Returns median latency in milliseconds over n_repeats calls."""
    times = []
    for _ in range(n_repeats):
        t0 = time.perf_counter()
        func()
        times.append((time.perf_counter() - t0) * 1000)
    return float(np.median(times))


# ===========================================================================
# API mode: call live FastAPI server
# ===========================================================================

def evaluate_via_api(task: int, data_path: str, api_url: str) -> dict:
    """
    Send requests to the running API and score responses against ground truth.
    """
    endpoint = f"{api_url}/task{task}/predict"
    print(f"\n🌐 Evaluating Task {task} via API: {endpoint}")

    # Load test data — expects CSV with 'id', feature columns, and 'label' column
    df = pd.read_csv(data_path)
    if "label" not in df.columns and "target" not in df.columns:
        print("❌ CSV must contain a 'label' or 'target' column for ground truth.")
        return {}

    label_col = "label" if "label" in df.columns else "target"
    y_true = df[label_col].tolist()
    feature_cols = [c for c in df.columns if c not in ("id", label_col)]

    y_pred = []
    latencies = []

    for _, row in df.iterrows():
        payload = {"id": str(row.get("id", "x")), "features": row[feature_cols].tolist()}
        t0 = time.perf_counter()
        try:
            resp = requests.post(endpoint, json=payload, timeout=15)
            resp.raise_for_status()
            result = resp.json()
            y_pred.append(result.get("prediction", result.get("label", "unknown")))
            latencies.append((time.perf_counter() - t0) * 1000)
        except Exception as e:
            print(f"  ⚠️  Request failed for id={row.get('id')}: {e}")
            y_pred.append(None)

    valid = [(yt, yp) for yt, yp in zip(y_true, y_pred) if yp is not None]
    if not valid:
        print("❌ No valid predictions received.")
        return {}

    yt_v, yp_v = zip(*valid)
    metrics = classification_metrics(list(yt_v), list(yp_v))
    metrics["median_latency_ms"] = round(float(np.median(latencies)), 2)
    metrics["p95_latency_ms"]    = round(float(np.percentile(latencies, 95)), 2)
    print(f"⏱  Latency: median={metrics['median_latency_ms']}ms  p95={metrics['p95_latency_ms']}ms")
    return metrics


# ===========================================================================
# Local model mode
# ===========================================================================

def evaluate_task1_local(data_path: str, model_path: Optional[str]) -> dict:
    """Tabular classification — local artifact evaluation."""
    print(f"\n🔢 Task 1 (Tabular) — local evaluation")
    df = pd.read_csv(data_path)
    label_col = "label" if "label" in df.columns else "target"
    y_true = df[label_col]
    X = df.drop(columns=[label_col, "id"], errors="ignore")

    if model_path:
        from tasks.machine_learning import TabularPipeline
        pipeline = TabularPipeline.load(model_path)
    else:
        print("  ℹ️  No model path supplied — fitting a quick baseline on first 80%...")
        from tasks.machine_learning import TabularPipeline
        split = int(0.8 * len(X))
        pipeline = TabularPipeline(task="classification")
        pipeline.fit(X.iloc[:split], y_true.iloc[:split])
        X, y_true = X.iloc[split:], y_true.iloc[split:]

    t0 = time.perf_counter()
    preds = pipeline.predict(X)
    latency = (time.perf_counter() - t0) * 1000 / len(X)

    metrics = classification_metrics(y_true, preds)
    metrics["avg_latency_ms_per_sample"] = round(latency, 4)
    return metrics


def evaluate_task2_local(data_path: str, model_path: Optional[str] = None) -> dict:
    """Text classification — local evaluation."""
    print(f"\n🔤 Task 2 (Language) — local evaluation")
    df = pd.read_csv(data_path)
    label_col = "label" if "label" in df.columns else "target"
    texts  = df["text"].tolist()
    y_true = df[label_col].tolist()

    if model_path:
        from tasks.language import TextClassifier
        clf = TextClassifier.load(model_path)
    else:
        print("  ℹ️  No model path supplied — fitting TF-IDF baseline on first 80%...")
        from tasks.language import TextClassifier
        split = int(0.8 * len(texts))
        clf = TextClassifier(strategy="tfidf")
        clf.fit(texts[:split], y_true[:split])
        texts, y_true = texts[split:], y_true[split:]

    t0 = time.perf_counter()
    preds = clf.predict(texts)
    latency = (time.perf_counter() - t0) * 1000 / max(len(texts), 1)

    metrics = classification_metrics(y_true, preds)
    metrics["avg_latency_ms_per_sample"] = round(latency, 4)
    return metrics


def evaluate_task3_local(data_path: str, model_path: Optional[str] = None) -> dict:
    """Image segmentation — local evaluation using Otsu baseline."""
    print(f"\n🖼  Task 3 (Vision) — local evaluation")
    from tasks.vision import SegmentationPipeline, ImagePreprocessor

    image_dir = Path(data_path)
    image_files = list(image_dir.glob("*.png")) + list(image_dir.glob("*.jpg"))

    if not image_files:
        print(f"  ⚠️  No PNG/JPG files found in {data_path}. Skipping.")
        return {}

    backend = "otsu"
    if model_path:
        backend = "torchvision" if model_path.endswith(".pth") else "otsu"

    pipeline = SegmentationPipeline(backend=backend)
    prep     = ImagePreprocessor(img_size=256, normalize=False)

    preds, targets = [], []
    for f in image_files[:50]:  # Cap at 50 for speed
        arr = prep.load_and_transform(str(f))
        mask_pred = pipeline.predict_single(arr)

        # Look for a matching ground-truth mask (convention: same name in masks/ subfolder)
        mask_path = image_dir / "masks" / f.name
        if mask_path.exists():
            mask_true = (prep.load_and_transform(str(mask_path)) > 0.5).astype(np.uint8)
            preds.append(mask_pred)
            targets.append(mask_true[:, :, 0] if mask_true.ndim == 3 else mask_true)

    if not targets:
        print("  ℹ️  No ground-truth masks found in masks/ subfolder. Reporting prediction shapes only.")
        print(f"  Generated {len(preds)} masks, shape={preds[0].shape if preds else 'N/A'}")
        return {"n_predictions": len(preds)}

    return segmentation_metrics(preds, targets)


# ===========================================================================
# Entry point
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(description="AINM2026 Local Evaluation Harness")
    parser.add_argument("--task",   choices=["1", "2", "3", "all"], default="all",
                        help="Which task to evaluate (default: all)")
    parser.add_argument("--data",   default="data/",
                        help="Path to test data CSV or image directory")
    parser.add_argument("--model",  default=None,
                        help="Path to saved model artifact (.pkl or .pth)")
    parser.add_argument("--mode",   choices=["local", "api"], default="local",
                        help="Evaluate using local model or against a live API")
    parser.add_argument("--api-url", default="http://localhost:8000",
                        help="API base URL (used when --mode=api)")
    args = parser.parse_args()

    results = {}
    tasks = ["1", "2", "3"] if args.task == "all" else [args.task]

    for task in tasks:
        print(f"\n{'='*55}")
        if args.mode == "api":
            results[f"task{task}"] = evaluate_via_api(int(task), args.data, args.api_url)
        else:
            if task == "1":
                data = args.data if not Path(args.data).is_dir() else str(Path(args.data)/"task1_test.csv")
                results["task1"] = evaluate_task1_local(data, args.model) if Path(data).exists() else {}
            elif task == "2":
                data = args.data if not Path(args.data).is_dir() else str(Path(args.data)/"task2_test.csv")
                results["task2"] = evaluate_task2_local(data, args.model) if Path(data).exists() else {}
            elif task == "3":
                results["task3"] = evaluate_task3_local(args.data, args.model)

    print(f"\n{'='*55}")
    print("📊 FINAL RESULTS:")
    print(json.dumps(results, indent=2))
    return results


if __name__ == "__main__":
    main()
