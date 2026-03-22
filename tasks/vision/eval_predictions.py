"""
eval_predictions.py
-------------------
Evaluates a competition-format predictions JSON against the val split
ground truth using pycocotools COCO mAP.

Usage:
    python eval_predictions.py /tmp/preds_standard.json
    python eval_predictions.py /tmp/preds_tta.json
"""

import argparse
import json
from pathlib import Path
from collections import defaultdict

# ─── Config ───────────────────────────────────────────────────────────────────

ANN_FILE  = Path("data/train/annotations.json")
VAL_IMGS  = Path("data/yolo/images/val")


def build_val_gt(ann_file: Path, val_img_dir: Path):
    """Build a COCO-format GT dict restricted to val images."""
    with open(ann_file) as f:
        coco = json.load(f)

    # Find val image filenames
    val_fnames = {p.name for p in val_img_dir.iterdir() if p.suffix in (".jpg", ".jpeg")}

    val_img_ids = {img["id"] for img in coco["images"] if img["file_name"] in val_fnames}

    gt = {
        "images":      [img for img in coco["images"] if img["id"] in val_img_ids],
        "annotations": [ann for ann in coco["annotations"] if ann["image_id"] in val_img_ids],
        "categories":  coco["categories"],
    }
    return gt, val_img_ids


def evaluate(pred_file: Path, label: str):
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    import io, contextlib

    print(f"\n{'='*60}")
    print(f"  Evaluating: {label}")
    print(f"  Predictions: {pred_file}")
    print(f"{'='*60}")

    # Build GT
    gt_dict, val_img_ids = build_val_gt(ANN_FILE, VAL_IMGS)
    print(f"  Val images : {len(gt_dict['images'])}")
    print(f"  Val GT anns: {len(gt_dict['annotations'])}")

    # Load predictions
    with open(pred_file) as f:
        preds = json.load(f)

    # Filter predictions to val images only (in case run.py saw extra images)
    preds = [p for p in preds if p["image_id"] in val_img_ids]
    print(f"  Predictions: {len(preds)}")

    if not preds:
        print("  ERROR: No predictions found for val images!")
        return

    # Write GT to temp file for COCO loader
    import tempfile, os
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(gt_dict, f)
        gt_path = f.name

    try:
        # Suppress COCO loader prints
        with contextlib.redirect_stdout(io.StringIO()):
            coco_gt = COCO(gt_path)
            coco_dt = coco_gt.loadRes(preds)

        # ── Detection mAP (category-agnostic, matches 70% of score) ────────────
        print("\n  [ Detection mAP — category-agnostic IoU≥0.5 ]")
        eval_det = COCOeval(coco_gt, coco_dt, "bbox")
        eval_det.params.useCats = 0   # ignore category for detection score
        eval_det.evaluate()
        eval_det.accumulate()
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            eval_det.summarize()
        for line in buf.getvalue().strip().split("\n"):
            print(f"    {line}")
        det_map50   = eval_det.stats[1]   # AP @ IoU=0.50
        det_map5095 = eval_det.stats[0]   # AP @ IoU=0.50:0.95

        # ── Classification mAP (category-aware, matches 30% of score) ──────────
        print("\n  [ Classification mAP — category-aware ]")
        eval_cls = COCOeval(coco_gt, coco_dt, "bbox")
        eval_cls.params.useCats = 1
        eval_cls.evaluate()
        eval_cls.accumulate()
        buf2 = io.StringIO()
        with contextlib.redirect_stdout(buf2):
            eval_cls.summarize()
        for line in buf2.getvalue().strip().split("\n"):
            print(f"    {line}")
        cls_map50   = eval_cls.stats[1]
        cls_map5095 = eval_cls.stats[0]

        # ── Estimated competition score ─────────────────────────────────────────
        est = 0.70 * det_map50 + 0.30 * cls_map50
        print(f"\n  {'─'*50}")
        print(f"  Detection  mAP@0.5     : {det_map50:.4f}")
        print(f"  Detection  mAP@0.5:0.95: {det_map5095:.4f}")
        print(f"  Classif.   mAP@0.5     : {cls_map50:.4f}")
        print(f"  Classif.   mAP@0.5:0.95: {cls_map5095:.4f}")
        print(f"  {'─'*50}")
        print(f"  Est. score (0.7×det + 0.3×cls) @ IoU=0.5 : {est:.4f}")
        print(f"  {'─'*50}\n")

        return dict(det_map50=det_map50, det_map5095=det_map5095,
                    cls_map50=cls_map50, cls_map5095=cls_map5095, est=est)
    finally:
        os.unlink(gt_path)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("pred_file", help="Path to predictions JSON")
    p.add_argument("--label",   default=None)
    args = p.parse_args()
    label = args.label or Path(args.pred_file).stem
    evaluate(Path(args.pred_file), label)
