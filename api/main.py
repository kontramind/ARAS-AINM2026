"""
api/main.py
------------
FastAPI application — AINM 2026 competition server.

Structure:
  GET  /           → health check (alias)
  GET  /health     → health check
  POST /task1/predict  → Tabular / classification inference
  POST /task2/predict  → Language / RAG inference
  POST /task3/predict  → Vision / segmentation inference

Design principles:
  - Every endpoint returns { "id": ..., ...predictions } — required by organisers
  - Latency is tracked and returned in meta.latency_ms for self-monitoring
  - Models are loaded ONCE at startup (lifespan) — not per-request
  - All heavy imports are deferred to startup so uvicorn starts fast on reload

Usage:
  uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload  (dev)
  uvicorn api.main:app --host 0.0.0.0 --port 8000            (prod)
"""

import base64
import os
import time
from contextlib import asynccontextmanager
from io import BytesIO
from typing import Optional

import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware

from api.schemas import (
    HealthResponse,
    PredictionMetadata,
    Task1Input, Task1Output,
    Task2Input, Task2Output,
    Task3Input, Task3Output,
)

# ===========================================================================
# MODEL REGISTRY
# ---------------------------------------------------------------------------
# Models are loaded once at startup and stored in this dict.
# On competition day, replace the placeholder values with real model instances.
# Example:
#   models["task1"] = TabularPipeline.load("models/tabular_v2.pkl")
#   models["task2_rag"] = RAGPipeline()
#   models["task3"] = SegmentationPipeline(backend="torchvision")
# ===========================================================================

models: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan: load models at startup, release at shutdown.
    This runs ONCE, not per-request — critical for keeping latency low.
    """
    print("🚀 Starting up AINM2026 API...")

    # ── Task 1 ──────────────────────────────────────────────────────────────
    # from tasks.machine_learning import TabularPipeline
    # models["task1"] = TabularPipeline.load(os.getenv("MODELS_DIR", "models") + "/task1.pkl")

    # ── Task 2 ──────────────────────────────────────────────────────────────
    # from tasks.language import RAGPipeline, TextClassifier
    # models["task2_rag"] = RAGPipeline(persist_directory=os.getenv("CHROMA_PERSIST_DIR", "./data/chroma_db"))
    # models["task2_clf"] = TextClassifier.load(os.getenv("MODELS_DIR", "models") + "/task2_clf.pkl")

    # ── Task 3 ──────────────────────────────────────────────────────────────
    # from tasks.vision import SegmentationPipeline, ImagePreprocessor
    # models["task3_seg"] = SegmentationPipeline(backend="torchvision")
    # models["task3_seg"].load_pretrained(os.getenv("MODELS_DIR", "models") + "/task3_seg.pth")
    # models["task3_prep"] = ImagePreprocessor(img_size=224, normalize=True)

    print(f"✅ API ready. Loaded models: {list(models.keys()) or ['(none loaded yet)']}")
    yield

    # Cleanup on shutdown (free GPU memory, close connections, etc.)
    models.clear()
    print("🛑 API shut down cleanly.")


# ===========================================================================
# APP INIT
# ===========================================================================

app = FastAPI(
    title="AINM 2026 — DNV Competition API",
    description=(
        "Norwegian AI Championship 2026. "
        "Serves predictions for all competition tasks via a unified REST API."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# Allow the competition testbed to call this API from any origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ===========================================================================
# UTILITY
# ===========================================================================

def _decode_image_b64(b64_string: str) -> np.ndarray:
    """Decode a base64-encoded image to a numpy array."""
    try:
        from PIL import Image
        raw = base64.b64decode(b64_string)
        img = Image.open(BytesIO(raw)).convert("RGB")
        return np.array(img)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Could not decode image: {e}")


# ===========================================================================
# HEALTH
# ===========================================================================

@app.get("/", response_model=HealthResponse, tags=["Health"])
@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """
    Health endpoint — the competition testbed pings this to verify the server is alive.
    Must return HTTP 200 with status="ok" at all times.
    """
    return HealthResponse(
        status="ok",
        message="API is live.",
        version="1.0.0",
    )


# ===========================================================================
# TASK 1 — Tabular / Classification
# ===========================================================================

@app.post("/task1/predict", response_model=Task1Output, tags=["Task 1 — Tabular"])
async def predict_task1(payload: Task1Input):
    """
    Task 1 inference endpoint.

    Replace the dummy logic below with your actual model call:
        model = models["task1"]
        pred = model.predict(pd.DataFrame([payload.features]))
    """
    t0 = time.perf_counter()
    try:
        # ── REPLACE THIS BLOCK ON COMPETITION DAY ──────────────────────────
        import random
        dummy_pred = random.choice([0, 1])
        dummy_conf = round(random.uniform(0.5, 0.99), 4)
        # ───────────────────────────────────────────────────────────────────

        latency = round((time.perf_counter() - t0) * 1000, 2)
        return Task1Output(
            id=payload.id,
            prediction=dummy_pred,
            confidence=dummy_conf,
            meta=PredictionMetadata(model_name="dummy", latency_ms=latency),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ===========================================================================
# TASK 2 — Language / RAG / Classification
# ===========================================================================

@app.post("/task2/predict", response_model=Task2Output, tags=["Task 2 — Language"])
async def predict_task2(payload: Task2Input):
    """
    Task 2 inference endpoint.

    Replace the dummy logic below with your actual pipeline call:
        # For RAG:
        chain = models["task2_rag"].build_chain()
        answer = chain.invoke(payload.text)

        # For classification:
        label = models["task2_clf"].predict([payload.text])[0]
    """
    t0 = time.perf_counter()
    try:
        # ── REPLACE THIS BLOCK ON COMPETITION DAY ──────────────────────────
        is_urgent = any(kw in payload.text.lower() for kw in ["urgent", "severe", "critical", "emergency"])
        dummy_label = "urgent" if is_urgent else "non-urgent"
        # ───────────────────────────────────────────────────────────────────

        latency = round((time.perf_counter() - t0) * 1000, 2)
        return Task2Output(
            id=payload.id,
            label=dummy_label,
            is_correct=None,
            answer=None,
            meta=PredictionMetadata(model_name="keyword_heuristic", latency_ms=latency),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ===========================================================================
# TASK 3 — Vision / Segmentation
# ===========================================================================

@app.post("/task3/predict", response_model=Task3Output, tags=["Task 3 — Vision"])
async def predict_task3(payload: Task3Input):
    """
    Task 3 inference endpoint.

    Replace the dummy logic below with your actual pipeline call:
        arr = _decode_image_b64(payload.image_b64)
        arr_preprocessed = models["task3_prep"].load_and_transform(arr)
        mask = models["task3_seg"].predict_single(arr_preprocessed)
        mask_list = mask.tolist()
    """
    t0 = time.perf_counter()
    try:
        # ── REPLACE THIS BLOCK ON COMPETITION DAY ──────────────────────────
        if payload.image_array is not None:
            h = len(payload.image_array)
            w = len(payload.image_array[0]) if h > 0 else 0
        else:
            h, w = 64, 64  # Default dummy mask size

        # Return an all-zeros mask as a safe "do nothing" baseline
        dummy_mask = [[0] * w for _ in range(h)]
        # ───────────────────────────────────────────────────────────────────

        latency = round((time.perf_counter() - t0) * 1000, 2)
        return Task3Output(
            id=payload.id,
            label=None,
            mask=dummy_mask,
            meta=PredictionMetadata(model_name="dummy_zeros", latency_ms=latency),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ===========================================================================
# DEV ENTRYPOINT
# ===========================================================================

if __name__ == "__main__":
    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", "8000"))
    reload = bool(int(os.getenv("API_RELOAD", "1")))
    uvicorn.run("api.main:app", host=host, port=port, reload=reload)
