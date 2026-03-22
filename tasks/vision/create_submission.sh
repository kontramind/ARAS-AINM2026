#!/usr/bin/env bash
# create_submission.sh
# --------------------
# Packs submission.zip correctly:
#   - run.py at zip ROOT (not inside a subfolder)
#   - best.onnx  (exported by export_onnx.py)
#   - best.json  (metadata sidecar — imgsz, dtype etc.)
#
# Usage:
#   bash create_submission.sh
#
# Prerequisites:
#   python train.py        → produces best.pt
#   python export_onnx.py  → produces best.onnx + best.json

set -euo pipefail

# ─── Paths ────────────────────────────────────────────────────────────────────

WEIGHTS_DIR="runs/detect/norgesgruppen/weights"
BEST_ONNX="${WEIGHTS_DIR}/best.onnx"
BEST_JSON="${WEIGHTS_DIR}/best.json"
STAGING="_submission_staging"
ZIPFILE="submission.zip"

# ─── Pre-flight checks ────────────────────────────────────────────────────────

echo "─── Pre-flight checks ───────────────────────────────────────────────"

if [ ! -f "run.py" ]; then
  echo "ERROR: run.py not found in current directory."
  echo "       Run this script from the norgesgruppen/ project root."
  exit 1
fi

if [ ! -f "${BEST_ONNX}" ]; then
  echo "ERROR: ${BEST_ONNX} not found."
  echo ""
  echo "  Run:  python export_onnx.py"
  echo "  (This requires best.pt to exist — run train.py first.)"
  exit 1
fi

ONNX_MB=$(du -m "${BEST_ONNX}" | cut -f1)
echo "  best.onnx : ${ONNX_MB} MB"

if [ "${ONNX_MB}" -gt 420 ]; then
  echo "  ERROR: best.onnx exceeds 420 MB sandbox weight limit!"
  echo "         Re-run: python export_onnx.py --imgsz 1024"
  exit 1
fi
echo "  ✓ Size within 420 MB limit"

if [ ! -f "${BEST_JSON}" ]; then
  echo "  WARN: best.json not found — run.py will use default imgsz=1280"
fi

# ─── Build staging directory ──────────────────────────────────────────────────

echo ""
echo "─── Building zip ────────────────────────────────────────────────────"

rm -rf "${STAGING}"
mkdir -p "${STAGING}"

# run.py at root
cp run.py "${STAGING}/run.py"

# Model weights
cp "${BEST_ONNX}" "${STAGING}/best.onnx"

# Metadata sidecar (tells run.py the imgsz + dtype)
if [ -f "${BEST_JSON}" ]; then
  cp "${BEST_JSON}" "${STAGING}/best.json"
fi

# ─── Create zip ───────────────────────────────────────────────────────────────

cd "${STAGING}"
zip -r "../${ZIPFILE}" . -x ".*" "__MACOSX/*"
cd ..
rm -rf "${STAGING}"

# ─── Verify ───────────────────────────────────────────────────────────────────

echo ""
echo "─── Zip contents ────────────────────────────────────────────────────"
unzip -l "${ZIPFILE}"

echo ""
ZIP_MB=$(du -m "${ZIPFILE}" | cut -f1)
echo "  Output        : ${ZIPFILE}  (${ZIP_MB} MB compressed)"

# Verify run.py is at root
if unzip -l "${ZIPFILE}" | grep -qE "^\s+[0-9]+\s+.*\s+run\.py$"; then
  echo "  ✓ run.py is at zip root"
else
  echo "  ✗ WARNING: run.py not at zip root — re-check the zip!"
fi

# Count files by type
N_PY=$(unzip -l "${ZIPFILE}" | grep -c "\.py$" || true)
N_ONNX=$(unzip -l "${ZIPFILE}" | grep -c "\.onnx$" || true)
N_JSON=$(unzip -l "${ZIPFILE}" | grep -c "\.json$" || true)
echo "  Files: ${N_PY} .py  |  ${N_ONNX} .onnx  |  ${N_JSON} .json"

if [ "${N_PY}" -gt 10 ]; then
  echo "  ✗ WARNING: More than 10 .py files (limit is 10)"
fi
if [ "${N_ONNX}" -gt 3 ]; then
  echo "  ✗ WARNING: More than 3 weight files (limit is 3)"
fi

echo ""
echo "Done!  Upload ${ZIPFILE} at the competition submit page."

