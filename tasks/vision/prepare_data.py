"""
prepare_data.py
---------------
Converts the NorgesGruppen COCO dataset to YOLO format and creates the
dataset.yaml needed by Ultralytics YOLOv8.

Directory layout expected:
  <data_root>/
    train/
      images/         ← shelf images (img_XXXXX.jpg)
      annotations.json

Outputs (created by this script):
  <data_root>/
    yolo/
      images/
        train/        ← symlinked / copied training images
        val/          ← symlinked / copied validation images
      labels/
        train/        ← YOLO .txt label files
        val/
    dataset.yaml      ← Ultralytics dataset config
"""

import json
import random
import shutil
from pathlib import Path
from collections import defaultdict

# ─── Configuration ────────────────────────────────────────────────────────────

DATA_ROOT   = Path("data")           # Change to your data root if needed
TRAIN_DIR   = DATA_ROOT / "train"
YOLO_DIR    = DATA_ROOT / "yolo"
YAML_PATH   = DATA_ROOT / "dataset.yaml"

VAL_SPLIT   = 0.10    # 10% validation
SEED        = 42

# ─── Helpers ──────────────────────────────────────────────────────────────────

def coco_bbox_to_yolo(bbox, img_w, img_h):
    """Convert COCO [x, y, w, h] → YOLO [cx, cy, w, h] (normalised 0-1)."""
    x, y, w, h = bbox
    cx = (x + w / 2) / img_w
    cy = (y + h / 2) / img_h
    nw = w / img_w
    nh = h / img_h
    # Clamp to [0, 1]
    cx = max(0.0, min(1.0, cx))
    cy = max(0.0, min(1.0, cy))
    nw = max(0.001, min(1.0, nw))
    nh = max(0.001, min(1.0, nh))
    return cx, cy, nw, nh


def load_annotations(ann_path: Path):
    with open(ann_path) as f:
        coco = json.load(f)
    return coco


def build_category_map(coco):
    """Return sorted list of category names and id→yolo_idx mapping."""
    # Category IDs in this dataset start at 0 and go to 356.
    # We keep them as-is; YOLO class index == category_id.
    cats = sorted(coco["categories"], key=lambda c: c["id"])
    id_to_idx = {c["id"]: c["id"] for c in cats}   # identity mapping
    names = [c["name"] for c in cats]
    return names, id_to_idx


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    random.seed(SEED)

    ann_path = TRAIN_DIR / "annotations.json"
    img_dir  = TRAIN_DIR / "images"

    print(f"Loading annotations from {ann_path} …")
    coco = load_annotations(ann_path)

    names, id_to_idx = build_category_map(coco)
    nc = max(id_to_idx.values()) + 1
    print(f"  {nc} categories, {len(coco['images'])} images, "
          f"{len(coco['annotations'])} annotations")

    # Group annotations by image_id
    ann_by_img = defaultdict(list)
    for ann in coco["annotations"]:
        if ann.get("iscrowd", 0):
            continue
        ann_by_img[ann["image_id"]].append(ann)

    # Build image info dict
    img_info = {img["id"]: img for img in coco["images"]}

    # Train / val split on image level
    all_ids = list(img_info.keys())
    random.shuffle(all_ids)
    n_val    = max(1, int(len(all_ids) * VAL_SPLIT))
    val_ids  = set(all_ids[:n_val])
    train_ids = set(all_ids[n_val:])
    print(f"  Train: {len(train_ids)} images  |  Val: {len(val_ids)} images")

    # Create YOLO directory tree
    for split in ("train", "val"):
        (YOLO_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
        (YOLO_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)

    skipped_imgs  = 0
    total_labels  = 0
    no_ann_images = 0

    for img_id, info in img_info.items():
        split  = "val" if img_id in val_ids else "train"
        fname  = info["file_name"]
        src    = img_dir / fname
        dst    = YOLO_DIR / "images" / split / fname
        lbl    = YOLO_DIR / "labels" / split / (Path(fname).stem + ".txt")

        if not src.exists():
            print(f"  [WARN] Missing image: {src}")
            skipped_imgs += 1
            continue

        # Copy image (use symlink for speed if on same filesystem)
        if not dst.exists():
            try:
                dst.symlink_to(src.resolve())
            except (OSError, NotImplementedError):
                shutil.copy2(src, dst)

        # Write YOLO label file
        anns = ann_by_img.get(img_id, [])
        if not anns:
            no_ann_images += 1
            # Still write empty label file so YOLO doesn't crash
            lbl.write_text("")
            continue

        iw, ih = info["width"], info["height"]
        lines = []
        for ann in anns:
            cat_id = ann["category_id"]
            if cat_id not in id_to_idx:
                continue
            yolo_cls = id_to_idx[cat_id]
            cx, cy, nw, nh = coco_bbox_to_yolo(ann["bbox"], iw, ih)
            lines.append(f"{yolo_cls} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

        lbl.write_text("\n".join(lines))
        total_labels += len(lines)

    print(f"  Written {total_labels} label entries "
          f"({skipped_imgs} images skipped, {no_ann_images} with no annotations)")

    # Write dataset.yaml  (no yaml module — write manually)
    yaml_lines = [
        f"path: {YOLO_DIR.resolve()}",
        f"train: images/train",
        f"val: images/val",
        f"nc: {nc}",
        "names:",
    ]
    for i, name in enumerate(names):
        # Escape special characters for YAML safety
        safe_name = name.replace("'", "''")
        yaml_lines.append(f"  {i}: '{safe_name}'")

    YAML_PATH.write_text("\n".join(yaml_lines) + "\n")
    print(f"\nDataset YAML written to: {YAML_PATH.resolve()}")
    print("\nDone! Run train.py next.")


if __name__ == "__main__":
    main()
