# ============================================================
# AINM2026 — Tripletex Agent for Cloud Run
# ============================================================

FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir uv

# Copy everything and install
COPY . .

RUN uv pip install --system --no-cache .

RUN mkdir -p logs tmp

ENV PORT=8000
ENV PYTHONUNBUFFERED=1
EXPOSE 8000

CMD uvicorn api.main:app --host 0.0.0.0 --port ${PORT} --workers 1
