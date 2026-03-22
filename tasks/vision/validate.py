"""
validate.py
-----------
Runs local validation against the held-out val split and prints COCO mAP metrics.
Useful for checking model quality before submitting.

Usage
-----
    python validate.py
    python validate.py --weights runs/detect/norgesgruppen/weights/best.pt
    python validate.py --save-predictions preds.json   # also writes COCO predictions JSON
"""

import argparse
import json
from pathlib import Path

from ultralytics import YOLO

# ─── Defaults ─────────────────────────────────────────────────────────────────

DEFAULT_WEIGHTS = Path("runs/detect/norgesgruppen/weights/best.pt")
DEFAULT_DATA    = Path("data/dataset.yaml")
IMGSZ           = 1280


# ─── Main ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Validate YOLOv8 on NorgesGruppen val set")
    p.add_argument("--weights", default=str(DEFAULT_WEIGHTS))
    p.add_argument("--data",    default=str(DEFAULT_DATA))
    p.add_argument("--imgsz",   type=int, default=IMGSZ)
    p.add_argument("--conf",    type=float, default=0.001)
    p.add_argument("--iou",     type=float, default=0.6)
    p.add_argument("--save-predictions", default=None,
                   help="Optional path to save predictions as competition JSON")
    return p.parse_args()


def main():
    args = parse_args()

    weights = Path(args.weights)
    if not weights.exists():
        raise FileNotFoundError(f"Weights not found: {weights}\nRun train.py first.")

    print(f"[val] Loading model: {weights}")
    model = YOLO(str(weights))

    print(f"[val] Running validation on val split …")
    metrics = model.val(
        data    = args.data,
        imgsz   = args.imgsz,
        conf    = args.conf,
        iou     = args.iou,
        verbose = True,
        plots   = True,      # saves confusion matrix, PR curve, etc.
    )

    print("\n─── Validation Results ───────────────────────────────────────────")
    print(f"  mAP@0.5       : {metrics.box.map50:.4f}")
    print(f"  mAP@0.5:0.95  : {metrics.box.map:.4f}")
    print(f"  Precision     : {metrics.box.mp:.4f}")
    print(f"  Recall        : {metrics.box.mr:.4f}")
    print()

    # Estimate competition score:
    # 70% detection mAP (category-agnostic IoU≥0.5) + 30% classification mAP
    # As a proxy: detection ≈ mAP50 with high recall, classification ≈ mAP50 with correct class
    detection_map   = metrics.box.map50      # approximate
    classification_map = metrics.box.map50   # conservative — actual may be lower
    est_score = 0.70 * detection_map + 0.30 * classification_map
    print(f"  Est. competition score : {est_score:.4f}")
    print("  (proxy — actual scoring uses category-agnostic detection separately)")

    # Save competition-format predictions if requested
    if args.save_predictions:
        print(f"\n[val] Writing predictions to {args.save_predictions} …")
        val_img_dir = Path("data/yolo/images/val")
        img_paths   = sorted(val_img_dir.glob("*.jpg"))

        predictions = []
        results = model.predict(
            source  = [str(p) for p in img_paths],
            imgsz   = args.imgsz,
            conf    = 0.15,
            iou     = 0.45,
            max_det = 500,
            verbose = False,
        )
        for img_path, result in zip(img_paths, results):
            img_id = int(img_path.stem.split("_")[-1])
            if result.boxes is None:
                continue
            for xyxy, conf, cls in zip(
                    result.boxes.xyxy.cpu().tolist(),
                    result.boxes.conf.cpu().tolist(),
                    result.boxes.cls.cpu().tolist()):
                x1, y1, x2, y2 = xyxy
                predictions.append({
                    "image_id"   : img_id,
                    "category_id": int(cls),
                    "bbox"       : [round(x1,2), round(y1,2),
                                    round(x2-x1,2), round(y2-y1,2)],
                    "score"      : round(float(conf), 4),
                })

        Path(args.save_predictions).write_text(json.dumps(predictions, indent=2))
        print(f"[val] {len(predictions)} predictions saved.")


if __name__ == "__main__":
    main()
