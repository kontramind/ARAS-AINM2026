# ============================================================
# AINM2026 - Norwegian AI Championship 2026
# Multi-stage Dockerfile: supports both CPU and GPU execution
# ============================================================

# ---- Stage 1: Builder ----
# Installs dependencies in an isolated layer for layer caching efficiency.
FROM python:3.11-slim AS builder

WORKDIR /app

# System-level build tools needed for some Python packages (e.g., lightgbm, opencv)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Use uv (modern, fast pip alternative) for dependency installation
RUN pip install --no-cache-dir uv

# Copy only dependency manifest first to exploit Docker layer cache.
# Changes to code won't invalidate the dependency layer.
COPY pyproject.toml .
RUN uv pip install --system --no-cache -e ".[vision]" 2>/dev/null || \
    uv pip install --system --no-cache -e . 


# ---- Stage 2: Runtime ----
# Lean final image with only what's needed to run.
FROM python:3.11-slim AS runtime

WORKDIR /app

# Runtime system dependencies
# libgl1 + libglib2.0 are required by OpenCV
# curl is useful for health checks in orchestration setups
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder stage
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application source code
COPY . .

# Ensure data and model mount points exist so Docker volumes bind cleanly
RUN mkdir -p data/chroma_db models tmp

# The API port — must match competition requirements (typically 8000)
EXPOSE 8000

# Healthcheck so orchestrators know when the app is ready
HEALTHCHECK --interval=15s --timeout=5s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default: run the FastAPI server
# Override CMD in docker-compose for dev (--reload) vs prod
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]


# ============================================================
# GPU VARIANT (uncomment base image below and comment above if you have NVIDIA GPU)
# Replace the first FROM line with:
#   FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04 AS builder
# Then install Python 3.11 manually:
#   RUN apt-get update && apt-get install -y python3.11 python3.11-dev python3-pip
# ============================================================
