#!/usr/bin/env bash
# =============================================================================
# scripts/setup.sh
# One-command bootstrap for the AINM2026 competition environment.
#
# Usage:
#   chmod +x scripts/setup.sh
#   ./scripts/setup.sh           # standard setup (no vision/torch)
#   ./scripts/setup.sh --vision  # include PyTorch + OpenCV
#   ./scripts/setup.sh --dev     # include test + dev tools
#   ./scripts/setup.sh --all     # everything
# =============================================================================

set -euo pipefail

VISION=false
DEV=false

# ---- Parse flags ----
for arg in "$@"; do
  case $arg in
    --vision) VISION=true ;;
    --dev)    DEV=true ;;
    --all)    VISION=true; DEV=true ;;
  esac
done

echo ""
echo "╔══════════════════════════════════════════════════╗"
echo "║        AINM 2026 — Environment Bootstrap         ║"
echo "╚══════════════════════════════════════════════════╝"
echo ""

# ---- Verify Python ----
PYTHON_BIN=$(command -v python3 || command -v python)
PYTHON_VERSION=$($PYTHON_BIN --version 2>&1 | awk '{print $2}')
REQUIRED_MAJOR=3
REQUIRED_MINOR=11

IFS='.' read -r major minor patch <<< "$PYTHON_VERSION"
if [[ "$major" -lt "$REQUIRED_MAJOR" ]] || \
   [[ "$major" -eq "$REQUIRED_MAJOR" && "$minor" -lt "$REQUIRED_MINOR" ]]; then
  echo "❌ Python $REQUIRED_MAJOR.$REQUIRED_MINOR+ required. Found: $PYTHON_VERSION"
  exit 1
fi
echo "✅ Python $PYTHON_VERSION found."

# ---- Verify uv is available (fast pip alternative) ----
if ! command -v uv &>/dev/null; then
  echo "⚙️  Installing uv (fast package installer)..."
  pip install --quiet uv
fi

# ---- .env setup ----
if [[ ! -f ".env" ]]; then
  echo "📄 .env not found — creating from .env.example..."
  cp .env.example .env
  echo "⚠️  Please edit .env and fill in your API keys before running the server."
fi

# ---- Install core dependencies ----
echo ""
echo "📦 Installing core dependencies..."
uv pip install -e .

# ---- Optional: vision ----
if $VISION; then
  echo ""
  echo "📦 Installing vision extras (Pillow, OpenCV, PyTorch)..."
  uv pip install -e ".[vision]"
fi

# ---- Optional: dev ----
if $DEV; then
  echo ""
  echo "📦 Installing dev/test extras..."
  uv pip install -e ".[dev]"
fi

# ---- Create required directories ----
mkdir -p data/chroma_db models tmp notebooks/reports
echo ""
echo "📁 Required directories ensured: data/chroma_db, models, tmp, notebooks/reports"

# ---- Verify API can be imported ----
echo ""
echo "🔍 Verifying imports..."
$PYTHON_BIN -c "from api.main import app; print('  ✅ api.main OK')"
$PYTHON_BIN -c "from tasks.machine_learning import TabularPipeline; print('  ✅ tasks.machine_learning OK')"
$PYTHON_BIN -c "from tasks.language import RAGPipeline, TextClassifier; print('  ✅ tasks.language OK')"
$PYTHON_BIN -c "from tasks.vision import ImagePreprocessor, SegmentationPipeline; print('  ✅ tasks.vision OK')"

echo ""
echo "╔══════════════════════════════════════════════════╗"
echo "║              Setup Complete! 🎉                  ║"
echo "╠══════════════════════════════════════════════════╣"
echo "║  Start API:  uvicorn api.main:app --reload       ║"
echo "║  Run tests:  pytest tests/                       ║"
echo "║  EDA report: python scripts/eda_report.py <csv>  ║"
echo "║  Expose API: bash scripts/start_ngrok.sh         ║"
echo "╚══════════════════════════════════════════════════╝"
echo ""
