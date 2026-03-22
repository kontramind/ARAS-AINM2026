"""
experiments/exp1_corrected_only.py
-----------------------------------
Experiment 1: Train on CORRECTED annotations only.

What changes vs baseline
------------------------
The COCO annotations.json has a "corrected" field per annotation.
Baseline (prepare_data.py) uses ALL annotations including unverified ones.
This experiment keeps only annotations where corrected=True.

Hypothesis
----------
Unverified annotations introduce label noise, especially for the 30%
classification score. Removing them may improve class mAP at the cost
of fewer training samples overall.

How to run
----------
# Step 1 — generate corrected-only YOLO dataset
python experiments/exp1_corrected_only.py

# Step 2 — train on it (writes to runs/detect/exp1_corrected)
python train.py --data data/yolo_corrected/dataset.yaml \
                --name exp1_corrected --epochs 150

# Step 3 — export
python export_onnx.py \
    --weights runs/detect/exp1_corrected/weights/best.pt \
    --out     runs/detect/exp1_corrected/weights/best.onnx

# Step 4 — validate and compare
python validate.py --weights runs/detect/exp1_corrected/weights/best.pt \
                   --data    data/yolo_corrected/dataset.yaml

Compare mAP50 vs baseline run (runs/detect/norgesgruppen).

How to build on this
--------------------
If exp1 improves over baseline, use data/yolo_corrected as the input
to exp3 (SAHI tiling) instead of data/yolo.
"""

import json
import random
import shutil
from pathlib import Path
from collections import defaultdict

# ─── Config ───────────────────────────────────────────────────────────────────

DATA_ROOT    = Path("data")
TRAIN_DIR    = DATA_ROOT / "train"
YOLO_DIR     = DATA_ROOT / "yolo_corrected"   # separate from baseline
YAML_PATH    = DATA_ROOT / "yolo_corrected" / "dataset.yaml"
VAL_SPLIT    = 0.10
SEED         = 42


# ─── Helpers (same as prepare_data.py) ───────────────────────────────────────

def coco_bbox_to_yolo(bbox, img_w, img_h):
    x, y, w, h = bbox
    cx = max(0., min(1., (x + w / 2) / img_w))
    cy = max(0., min(1., (y + h / 2) / img_h))
    nw = max(0.001, min(1., w / img_w))
    nh = max(0.001, min(1., h / img_h))
    return cx, cy, nw, nh


def main():
    random.seed(SEED)

    ann_path = TRAIN_DIR / "annotations.json"
    img_dir  = TRAIN_DIR / "images"

    with open(ann_path) as f:
        coco = json.load(f)

    # ── Report annotation counts ──────────────────────────────────────────────
    total      = len(coco["annotations"])
    corrected  = [a for a in coco["annotations"] if a.get("corrected", False)]
    print(f"[exp1] Total annotations   : {total}")
    print(f"[exp1] Corrected only      : {len(corrected)}  "
          f"({100*len(corrected)/total:.1f}%)")
    print(f"[exp1] Dropped (unverified): {total - len(corrected)}")

    # Build structures
    cats      = sorted(coco["categories"], key=lambda c: c["id"])
    names     = [c["name"] for c in cats]
    nc        = max(c["id"] for c in cats) + 1
    img_info  = {img["id"]: img for img in coco["images"]}

    # Only corrected annotations
    ann_by_img = defaultdict(list)
    for ann in corrected:
        ann_by_img[ann["image_id"]].append(ann)

    # Images that have at least one corrected annotation
    valid_ids = list(ann_by_img.keys())
    random.shuffle(valid_ids)
    n_val     = max(1, int(len(valid_ids) * VAL_SPLIT))
    val_ids   = set(valid_ids[:n_val])
    train_ids = set(valid_ids[n_val:])
    print(f"[exp1] Train images: {len(train_ids)}  |  Val images: {len(val_ids)}")

    for split in ("train", "val"):
        (YOLO_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
        (YOLO_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)

    total_labels = 0
    for img_id in valid_ids:
        info  = img_info[img_id]
        split = "val" if img_id in val_ids else "train"
        fname = info["file_name"]
        src   = img_dir / fname
        dst   = YOLO_DIR / "images" / split / fname
        lbl   = YOLO_DIR / "labels" / split / (Path(fname).stem + ".txt")

        if not src.exists():
            continue
        if not dst.exists():
            try:
                dst.symlink_to(src.resolve())
            except (OSError, NotImplementedError):
                shutil.copy2(src, dst)

        iw, ih = info["width"], info["height"]
        lines = []
        for ann in ann_by_img[img_id]:
            cx, cy, nw, nh = coco_bbox_to_yolo(ann["bbox"], iw, ih)
            lines.append(f"{ann['category_id']} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
        lbl.write_text("\n".join(lines))
        total_labels += len(lines)

    print(f"[exp1] Written {total_labels} label entries")

    # Write dataset.yaml
    yaml_lines = [
        f"path: {YOLO_DIR.resolve()}",
        "train: images/train",
        "val:   images/val",
        f"nc: {nc}",
        "names:",
    ]
    for i, name in enumerate(names):
        yaml_lines.append(f"  {i}: '{name.replace(chr(39), chr(39)*2)}'")
    (YOLO_DIR / "dataset.yaml").write_text("\n".join(yaml_lines) + "\n")
    YAML_PATH.parent.mkdir(parents=True, exist_ok=True)

    print(f"[exp1] Dataset YAML → {YOLO_DIR / 'dataset.yaml'}")
    print("\n[exp1] Next:")
    print(f"  python train.py --data {YOLO_DIR}/dataset.yaml --name exp1_corrected --epochs 150")


if __name__ == "__main__":
    main()
