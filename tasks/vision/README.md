# NorgesGruppen Grocery Bot — End-to-End Solution

**Model:** YOLOv8x  
**Competition:** NM i AI 2026 — NorgesGruppen Data Challenge  
**Task:** Detect and classify 356 grocery products on store shelves  
**Score breakdown:** 70 % detection mAP + 30 % classification mAP  

---

## Repository Structure

```
norgesgruppen/
├── prepare_data.py                   # Step 1: COCO → YOLO format
├── augment_with_reference_images.py  # Step 2 (optional): synthetic data
├── train.py                          # Step 3: train YOLOv8x
├── validate.py                       # Step 4: local validation
├── run.py                            # Submission entry point
├── run_tta.py                        # Alternative: run.py with TTA
├── create_submission.sh              # Step 5: pack the zip
├── requirements.txt
└── README.md
```

---

## Prerequisites

- Python 3.11
- CUDA-capable GPU (8+ GB VRAM recommended; 24 GB for batch=-1 auto)
- Downloaded competition data (login required at competition website):
  - `NM_NGD_coco_dataset.zip`  (~864 MB)
  - `NM_NGD_product_images.zip` (~60 MB, optional)

---

## Step 0 — Environment Setup

```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate           # Windows: venv\Scripts\activate

# Install dependencies
# GPU (recommended — uses CUDA 12.4 wheels matching the sandbox):
pip install -r requirements.txt --index-url https://download.pytorch.org/whl/cu124

# CPU-only fallback:
pip install -r requirements.txt
```

Verify GPU is available:
```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

---

## Step 1 — Prepare Data

### 1a. Extract competition data

```bash
# Create data directory
mkdir -p data

# Extract COCO dataset (creates data/train/images/ and data/train/annotations.json)
unzip NM_NGD_coco_dataset.zip -d data/
# Resulting layout:
#   data/train/images/img_00001.jpg …
#   data/train/annotations.json

# Optional: extract product reference images
unzip NM_NGD_product_images.zip -d data/product_images/
# Resulting layout:
#   data/product_images/metadata.json
#   data/product_images/7040913336691/main.jpg
#   data/product_images/7040913336691/front.jpg …
```

> **Check:** `ls data/train/images/ | head` should show `img_00001.jpg` etc.

### 1b. Convert to YOLO format

```bash
python prepare_data.py
```

**What it does:**
- Reads `data/train/annotations.json`
- Converts COCO bounding boxes to YOLO format (normalised `cx cy w h`)
- Performs 90/10 train/val split
- Creates `data/yolo/images/train|val/` and `data/yolo/labels/train|val/`
- Writes `data/dataset.yaml`

**Expected output:**
```
Loading annotations from data/train/annotations.json …
  357 categories, 248 images, 22700 annotations
  Train: 224 images  |  Val: 24 images
  Written 20400 label entries (0 images skipped, 0 with no annotations)
Dataset YAML written to: data/dataset.yaml
```

---

## Step 2 (Optional) — Augment with Product Reference Images

This generates synthetic training images by pasting product reference photos
onto real shelf backgrounds. Helps with rare products that have few annotations.

```bash
python augment_with_reference_images.py \
    --product-dir data/product_images \
    --yolo-dir    data/yolo \
    --metadata    data/product_images/metadata.json \
    --annotations data/train/annotations.json \
    --n-synth     300 \
    --min-anno    20        # only boost products with < 20 real annotations
```

> Skip this step for a first run — base training alone should score well.

---

## Step 3 — Train

```bash
# Full training — recommended (uses YOLOv8x, takes 4-8 hours on a single GPU)
python train.py

# Faster smoke-test (smaller model, fewer epochs)
python train.py --model yolov8m --epochs 30

# Resume interrupted training
python train.py --resume runs/detect/norgesgruppen/weights/last.pt

# Custom batch size if auto-detection fails
python train.py --batch 4
```

**Key training settings (in `train.py`):**

| Setting | Value | Reason |
|---|---|---|
| Model | YOLOv8x | Best mAP in the YOLOv8 family |
| Image size | 1280 | Shelf images have many small products |
| Epochs | 150 (patience=30) | Early stopping prevents over-fitting |
| Mosaic | 1.0 | Combines 4 images — helps with variety |
| Copy-paste | 0.1 | Synthetic product placement |
| Label smoothing | 0.1 | Regularises the 357-class softmax |
| AMP | True | Mixed precision — faster + less VRAM |
| Cosine LR | True | Smooth decay to final LR |

**Expected output location:**
```
runs/detect/norgesgruppen/
  weights/
    best.pt     ← use this for submission
    last.pt     ← last epoch (for resuming)
  results.csv
  val_batch*.jpg
  confusion_matrix.png
  PR_curve.png
```

**Weight file size check:**
```bash
ls -lh runs/detect/norgesgruppen/weights/best.pt
# Should be ~136 MB — well within the 420 MB sandbox limit
```

---

## Step 4 — Validate Locally

```bash
python validate.py
# or with explicit path:
python validate.py --weights runs/detect/norgesgruppen/weights/best.pt
```

**Expected metrics (rough targets for a good submission):**
```
mAP@0.5      : 0.65+   → ~70 % score for detection component
mAP@0.5:0.95 : 0.40+
```

Save predictions in competition format for manual inspection:
```bash
python validate.py --save-predictions val_predictions.json
```

---

## Step 5 — Create Submission Zip

### Standard submission (recommended first attempt):
```bash
bash create_submission.sh
```

### With Test-Time Augmentation (slightly higher mAP, uses more of the 300 s budget):
```bash
bash create_submission.sh --tta
```

The script creates `submission.zip` and verifies:
- `run.py` is at the root (not inside a subfolder)
- `best.pt` is present
- Total size is within limits

**Verify manually:**
```bash
unzip -l submission.zip
# Must show:
#   run.py
#   best.pt
# NOT: norgesgruppen/run.py
```

---

## Step 6 — Submit

Upload `submission.zip` at the competition submit page.

---

## Troubleshooting

### `ModuleNotFoundError: ultralytics`
```bash
pip install ultralytics==8.1.0
```

### CUDA out of memory during training
```bash
# Reduce image size:
python train.py --imgsz 960

# Or fix batch size:
python train.py --batch 4
```

### Weight file too large for sandbox (> 420 MB)
The YOLOv8x best.pt is ~136 MB and fits comfortably.
If you trained a custom larger model, export to FP16 ONNX:
```bash
python -c "
from ultralytics import YOLO
m = YOLO('runs/detect/norgesgruppen/weights/best.pt')
m.export(format='onnx', imgsz=1280, opset=17, half=True, simplify=True)
"
# Then submit with:
bash create_submission.sh --onnx
```

### Submission times out (300 s budget)
- Switch from `run_tta.py` to plain `run.py` (no TTA)
- Reduce `IMGSZ` to 1024 in `run.py`
- Reduce `BATCH_SIZE` from 8 to 4

### prepare_data.py: "No images found"
Check that images are in `data/train/images/`, not `data/train/` directly:
```bash
ls data/train/images/ | head -5
# Should show: img_00001.jpg img_00002.jpg ...
```

---

## Tips for Maximum Score

1. **Classification is 30 % of the score** — do not set all `category_id=0`.
   YOLOv8x learns classification well from the 248 training images.

2. **Use augmentation step 2** if some products have < 10 annotations in your
   training split — the reference images are gold.

3. **Confidence threshold matters** — `CONF_THRES=0.15` in `run.py` is a
   deliberate trade-off: lower recall threshold catches more products
   (important for the 70 % detection score) while WBF in `run_tta.py`
   cleans up duplicates.

4. **High image resolution** — do not drop to 640. Shelf images are dense
   with small products; 1280 makes a measurable difference (~3-5 % mAP).

5. **Check the PR curve** at `runs/detect/norgesgruppen/PR_curve.png` after
   training — it shows the precision/recall trade-off for your model.
