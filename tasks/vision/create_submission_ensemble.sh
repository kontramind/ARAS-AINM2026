#!/usr/bin/env bash
# create_submission_ensemble.sh
# ------------------------------
# Packs a two-model ensemble submission zip.
# run.py = exp4_ensemble.py (adapted as entry point)
# Weights: best.onnx (primary) + best_l.onnx (secondary)
#
# Usage:
#   bash create_submission_ensemble.sh \
#       --model-a runs/detect/norgesgruppen/weights/best.onnx \
#       --model-b runs/detect/exp4_model_l/weights/best_l.onnx

set -euo pipefail

MODEL_A="runs/detect/exp2_staged_lr/weights/best.onnx"
MODEL_B="runs/detect/exp4_model_l/weights/best_l.onnx"
STAGING="_ensemble_staging"
ZIPFILE="submission_ensemble.zip"

for arg in "$@"; do
  case "$arg" in
    --model-a=*) MODEL_A="${arg#*=}" ;;
    --model-b=*) MODEL_B="${arg#*=}" ;;
  esac
done

echo "─── Ensemble submission packer ──────────────────────────────────────"
echo "  Model A : $MODEL_A"
echo "  Model B : $MODEL_B"

for f in "$MODEL_A" "$MODEL_B"; do
  if [ ! -f "$f" ]; then
    echo "ERROR: Not found: $f"
    exit 1
  fi
done

# Check combined size
A_MB=$(du -m "$MODEL_A" | cut -f1)
B_MB=$(du -m "$MODEL_B" | cut -f1)
TOTAL=$((A_MB + B_MB))
echo "  Sizes   : ${A_MB} MB + ${B_MB} MB = ${TOTAL} MB (limit: 420 MB)"
if [ "$TOTAL" -gt 420 ]; then
  echo "ERROR: Combined model size exceeds 420 MB limit!"
  exit 1
fi

rm -rf "$STAGING"
mkdir -p "$STAGING"

# Use run_ensemble_tta.py as run.py (two-model ensemble + TTA)
cp run_ensemble_tta.py "$STAGING/run.py"

cp "$MODEL_A" "$STAGING/best.onnx"
cp "$MODEL_B" "$STAGING/best_l.onnx"

# Copy metadata sidecars if they exist
MA_JSON="${MODEL_A%.onnx}.json"
MB_JSON="${MODEL_B%.onnx}.json"
[ -f "$MA_JSON" ] && cp "$MA_JSON" "$STAGING/best.json"
[ -f "$MB_JSON" ] && cp "$MB_JSON" "$STAGING/best_l.json"

cd "$STAGING"
zip -r "../$ZIPFILE" . -x ".*" "__MACOSX/*"
cd ..
rm -rf "$STAGING"

echo ""
echo "─── Zip contents ────────────────────────────────────────────────────"
unzip -l "$ZIPFILE"
echo ""
echo "Done! Upload $ZIPFILE at the competition submit page."
