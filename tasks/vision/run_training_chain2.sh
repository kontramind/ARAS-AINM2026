#!/usr/bin/env bash
# Sequential training chain: exp8 (YOLOv8x@960) → exp9 (YOLOv8l@960) → exp10 (YOLOv8m@960)
# Trains on improved dataset with 500 synthetic images (rare+common mix, bg labels carried forward)
# Uses ultralytics' internal DDP (--device) rather than external torchrun
set -euo pipefail

PYTHON=".venv/bin/python"
GPUS="0,1,2"
DEVICES="0,1,2"

echo "========================================================"
echo "[chain] Starting exp8_x960: YOLOv8x @ imgsz=960 on GPUs ${GPUS}"
echo "========================================================"
CUDA_VISIBLE_DEVICES=${GPUS} ${PYTHON} train.py \
    --model    yolov8x \
    --device   ${DEVICES} \
    --epochs   150 \
    --patience 30 \
    --batch    12 \
    --imgsz    960 \
    --name     exp8_x960 \
    --warmup-epochs   5 \
    --label-smoothing 0.05 \
    --close-mosaic    15 \
    --copy-paste      0.3 \
    --workers  6 \
    --no-export

echo ""
echo "[chain] exp8_x960 training done. Exporting ONNX (FP16) ..."
CUDA_VISIBLE_DEVICES=0 ${PYTHON} - << 'PYEOF'
from ultralytics import YOLO
import json
from pathlib import Path

pt   = Path("runs/detect/exp8_x960/weights/best.pt")
m    = YOLO(str(pt))
m.export(format="onnx", imgsz=960, opset=17, dynamic=False, simplify=True, half=True, device=0)
onnx = pt.with_suffix(".onnx")
sz   = onnx.stat().st_size / 1e6
print(f"[export] {onnx}  {sz:.1f} MB")
Path("runs/detect/exp8_x960/weights/best.json").write_text(
    json.dumps({"imgsz": 960, "nc": 356, "fp16": True}))
print("[export] best.json written")
PYEOF

echo ""
echo "========================================================"
echo "[chain] Starting exp9_l960: YOLOv8l @ imgsz=960 on GPUs ${GPUS}"
echo "========================================================"
CUDA_VISIBLE_DEVICES=${GPUS} ${PYTHON} train.py \
    --model    yolov8l \
    --device   ${DEVICES} \
    --epochs   150 \
    --patience 30 \
    --batch    12 \
    --imgsz    960 \
    --name     exp9_l960 \
    --warmup-epochs   5 \
    --label-smoothing 0.05 \
    --close-mosaic    15 \
    --copy-paste      0.3 \
    --workers  6 \
    --no-export

echo ""
echo "[chain] exp9_l960 training done. Exporting ONNX (FP16) ..."
CUDA_VISIBLE_DEVICES=0 ${PYTHON} - << 'PYEOF'
from ultralytics import YOLO
import json
from pathlib import Path

pt   = Path("runs/detect/exp9_l960/weights/best.pt")
m    = YOLO(str(pt))
m.export(format="onnx", imgsz=960, opset=17, dynamic=False, simplify=True, half=True, device=0)
onnx = pt.with_suffix(".onnx")
sz   = onnx.stat().st_size / 1e6
print(f"[export] {onnx}  {sz:.1f} MB")
Path("runs/detect/exp9_l960/weights/best.json").write_text(
    json.dumps({"imgsz": 960, "nc": 356, "fp16": True}))
print("[export] best.json written")
PYEOF

echo ""
echo "========================================================"
echo "[chain] Starting exp10_m960: YOLOv8m @ imgsz=960 on GPUs ${GPUS}"
echo "========================================================"
CUDA_VISIBLE_DEVICES=${GPUS} ${PYTHON} train.py \
    --model    yolov8m \
    --device   ${DEVICES} \
    --epochs   150 \
    --patience 30 \
    --batch    15 \
    --imgsz    960 \
    --name     exp10_m960 \
    --warmup-epochs   5 \
    --label-smoothing 0.05 \
    --close-mosaic    15 \
    --copy-paste      0.3 \
    --workers  6 \
    --no-export

echo ""
echo "[chain] exp10_m960 training done. Exporting ONNX (FP16) ..."
CUDA_VISIBLE_DEVICES=0 ${PYTHON} - << 'PYEOF'
from ultralytics import YOLO
import json
from pathlib import Path

pt   = Path("runs/detect/exp10_m960/weights/best.pt")
m    = YOLO(str(pt))
m.export(format="onnx", imgsz=960, opset=17, dynamic=False, simplify=True, half=True, device=0)
onnx = pt.with_suffix(".onnx")
sz   = onnx.stat().st_size / 1e6
print(f"[export] {onnx}  {sz:.1f} MB")
Path("runs/detect/exp10_m960/weights/best.json").write_text(
    json.dumps({"imgsz": 960, "nc": 356, "fp16": True}))
print("[export] best.json written")
PYEOF

echo ""
echo "========================================================"
echo "[chain] ALL TRAINING COMPLETE"
echo "  exp8_x960  : runs/detect/exp8_x960/weights/best.onnx"
echo "  exp9_l960  : runs/detect/exp9_l960/weights/best.onnx"
echo "  exp10_m960 : runs/detect/exp10_m960/weights/best.onnx"
echo "========================================================"
