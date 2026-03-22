#!/usr/bin/env bash
# Sequential training chain: exp6 (YOLOv8l@960) → export → exp7 (YOLOv8m@960) → export
# Uses ultralytics' internal DDP (--device) rather than external torchrun
set -euo pipefail

PYTHON=".venv/bin/python"
GPUS="0,1,2"       # physical GPU IDs for CUDA_VISIBLE_DEVICES
DEVICES="0,1,2"    # device indices passed to ultralytics (matches CUDA_VISIBLE_DEVICES positions)

echo "========================================================"
echo "[chain] Starting exp6_l960: YOLOv8l @ imgsz=960 on GPUs ${GPUS}"
echo "========================================================"
CUDA_VISIBLE_DEVICES=${GPUS} ${PYTHON} train.py \
    --model    yolov8l \
    --device   ${DEVICES} \
    --epochs   150 \
    --patience 30 \
    --batch    12 \
    --imgsz    960 \
    --name     exp6_l960 \
    --warmup-epochs   5 \
    --label-smoothing 0.05 \
    --close-mosaic    15 \
    --copy-paste      0.3 \
    --workers  6 \
    --no-export

echo ""
echo "[chain] exp6_l960 training done. Exporting ONNX (FP16) ..."
CUDA_VISIBLE_DEVICES=0 ${PYTHON} - << 'PYEOF'
from ultralytics import YOLO
import json
from pathlib import Path

pt   = Path("runs/detect/exp6_l960/weights/best.pt")
m    = YOLO(str(pt))
m.export(format="onnx", imgsz=960, opset=17, dynamic=False, simplify=True, half=True, device=0)
onnx = pt.with_suffix(".onnx")
sz   = onnx.stat().st_size / 1e6
print(f"[export] {onnx}  {sz:.1f} MB")
Path("runs/detect/exp6_l960/weights/best.json").write_text(
    json.dumps({"imgsz": 960, "nc": 356, "fp16": True}))
print("[export] best.json written")
PYEOF

echo ""
echo "========================================================"
echo "[chain] Starting exp7_m960: YOLOv8m @ imgsz=960 on GPUs ${GPUS}"
echo "========================================================"
CUDA_VISIBLE_DEVICES=${GPUS} ${PYTHON} train.py \
    --model    yolov8m \
    --device   ${DEVICES} \
    --epochs   150 \
    --patience 30 \
    --batch    15 \
    --imgsz    960 \
    --name     exp7_m960 \
    --warmup-epochs   5 \
    --label-smoothing 0.05 \
    --close-mosaic    15 \
    --copy-paste      0.3 \
    --workers  6 \
    --no-export

echo ""
echo "[chain] exp7_m960 training done. Exporting ONNX (FP16) ..."
CUDA_VISIBLE_DEVICES=0 ${PYTHON} - << 'PYEOF'
from ultralytics import YOLO
import json
from pathlib import Path

pt   = Path("runs/detect/exp7_m960/weights/best.pt")
m    = YOLO(str(pt))
m.export(format="onnx", imgsz=960, opset=17, dynamic=False, simplify=True, half=True, device=0)
onnx = pt.with_suffix(".onnx")
sz   = onnx.stat().st_size / 1e6
print(f"[export] {onnx}  {sz:.1f} MB")
Path("runs/detect/exp7_m960/weights/best.json").write_text(
    json.dumps({"imgsz": 960, "nc": 356, "fp16": True}))
print("[export] best.json written")
PYEOF

echo ""
echo "========================================================"
echo "[chain] ALL TRAINING COMPLETE"
echo "  exp6_l960 : runs/detect/exp6_l960/weights/best.onnx"
echo "  exp7_m960 : runs/detect/exp7_m960/weights/best.onnx"
echo "========================================================"
