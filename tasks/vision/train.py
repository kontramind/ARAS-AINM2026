"""
train.py
--------
Fine-tunes YOLOv8x on the NorgesGruppen shelf dataset.

Strategy
--------
* Model    : YOLOv8x (best accuracy; ~136 MB weights, fits in 420 MB limit)
* Img size : 1280  (shelf images have many small products — high res matters)
* Epochs   : 150 with patience=30 early stopping
* Augments : full mosaic + mixup + colour jitter + affine; see hyp_ dict below
* Exports  : best.pt kept; also exported to best.onnx for portability

Usage
-----
    python train.py                              # defaults
    python train.py --model yolov8l --epochs 100
    python train.py --resume runs/detect/norgesgruppen/weights/last.pt
"""

import argparse
import json
from pathlib import Path

# Ultralytics must be pinned to 8.1.0 for sandbox compatibility
from ultralytics import YOLO

# ─── Defaults ─────────────────────────────────────────────────────────────────

YAML_PATH   = Path("data/dataset.yaml")
MODELS      = ["yolov8n", "yolov8s", "yolov8m", "yolov8l", "yolov8x"]
DEFAULT_MODEL = "yolov8x"
PROJECT     = "runs/detect"
NAME        = "norgesgruppen"
IMGSZ       = 960


# ─── Hyperparameters ──────────────────────────────────────────────────────────
# These override Ultralytics defaults.  Tuned for dense shelf detection.

HYPERPARAMS = dict(
    # ── Learning rate / optimiser ─────────────────────────────────────────────
    optimizer  = "AdamW",
    lr0        = 0.001,      # initial LR
    lrf        = 0.01,       # final LR = lr0 * lrf
    momentum   = 0.937,
    weight_decay = 0.0005,
    warmup_epochs = 3.0,
    warmup_momentum = 0.8,
    warmup_bias_lr  = 0.1,

    # ── Loss weights ─────────────────────────────────────────────────────────
    box  = 7.5,    # box regression loss gain
    cls  = 0.5,    # classification loss gain — boosted for 357 classes
    dfl  = 1.5,    # distribution focal loss gain

    # ── Augmentation ─────────────────────────────────────────────────────────
    # Geometric
    degrees    = 5.0,    # rotation ±5°; small — shelves are mostly upright
    translate  = 0.15,   # translation fraction
    scale      = 0.6,    # zoom scale ±60 %
    shear      = 2.0,    # shear ±2°
    perspective = 0.0,   # disabled — shelf images are planar
    flipud     = 0.0,    # vertical flip off — shelves have gravity
    fliplr     = 0.5,    # horizontal flip 50 %
    # Colour
    hsv_h      = 0.015,  # hue shift
    hsv_s      = 0.7,    # saturation shift — vary store lighting
    hsv_v      = 0.4,    # value/brightness shift
    # Composite
    mosaic     = 1.0,    # mosaic probability (4-image collage)
    mixup      = 0.15,   # mixup probability
    copy_paste = 0.1,    # copy-paste augmentation (great for dense objects)
    erasing    = 0.4,    # random erasing
)


# ─── Main ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Train YOLOv8 on NorgesGruppen data")
    p.add_argument("--model",   default=DEFAULT_MODEL, choices=MODELS,
                   help="YOLOv8 variant to use (default: yolov8x)")
    p.add_argument("--epochs",  type=int, default=150,
                   help="Max training epochs (default: 150)")
    p.add_argument("--patience",type=int, default=30,
                   help="Early stopping patience (default: 30)")
    p.add_argument("--batch",   type=int, default=-1,
                   help="Batch size (-1 = auto; default: -1)")
    p.add_argument("--imgsz",   type=int, default=IMGSZ,
                   help=f"Input image size (default: {IMGSZ})")
    p.add_argument("--data",    default=str(YAML_PATH),
                   help="Path to dataset.yaml (default: data/dataset.yaml)")
    p.add_argument("--resume",  default=None,
                   help="Resume from this weights file (e.g. last.pt)")
    p.add_argument("--workers", type=int, default=4,
                   help="DataLoader workers (default: 4)")
    p.add_argument("--no-export", action="store_true",
                   help="Skip ONNX export after training")
    p.add_argument("--device",  default=None,
                   help="Device(s) to use, e.g. '0' or '0,1,2' (default: auto)")
    p.add_argument("--name",    default=NAME,
                   help=f"Run name under project dir (default: {NAME})")
    p.add_argument("--label-smoothing", type=float, default=0.1,
                   help="Label smoothing (default: 0.1)")
    p.add_argument("--close-mosaic", type=int, default=10,
                   help="Disable mosaic in last N epochs (default: 10)")
    p.add_argument("--warmup-epochs", type=float, default=3.0,
                   help="Warmup epochs (default: 3.0)")
    p.add_argument("--copy-paste", type=float, default=0.1,
                   help="Copy-paste augmentation probability (default: 0.1)")
    return p.parse_args()


def export_onnx(weights_path: Path, imgsz: int):
    """Export best.pt → best.onnx with opset 17 (compatible with onnxruntime 1.20)."""
    print(f"\n[export] Loading {weights_path} for ONNX export …")
    model = YOLO(str(weights_path))
    model.export(
        format     = "onnx",
        imgsz      = imgsz,
        opset      = 17,
        dynamic    = False,
        simplify   = True,
        half       = True,     # FP16 — smaller + faster on L4 GPU
    )
    onnx_path = weights_path.with_suffix(".onnx")
    print(f"[export] Saved: {onnx_path}")
    return onnx_path


def main():
    args = parse_args()

    # ── Load model ────────────────────────────────────────────────────────────
    if args.resume:
        print(f"[train] Resuming from {args.resume}")
        model = YOLO(args.resume)
    else:
        weights = f"{args.model}.pt"   # downloads pretrained weights
        print(f"[train] Starting from pretrained {weights}")
        model = YOLO(weights)

    # ── Verify dataset YAML ───────────────────────────────────────────────────
    data_yaml = Path(args.data)
    if not data_yaml.exists():
        raise FileNotFoundError(
            f"Dataset YAML not found: {data_yaml}\n"
            "Run prepare_data.py first."
        )

    # ── Training ──────────────────────────────────────────────────────────────
    print(f"\n[train] Training {args.model} | imgsz={args.imgsz} | "
          f"epochs={args.epochs} | data={data_yaml}\n")

    train_kwargs = {}
    if args.device is not None:
        train_kwargs["device"] = args.device

    # CLI overrides for hyperparams (remove from base dict to avoid duplicates)
    hyp = {k: v for k, v in HYPERPARAMS.items()
           if k not in ("warmup_epochs", "copy_paste")}
    hyp["warmup_epochs"] = args.warmup_epochs
    hyp["copy_paste"]    = args.copy_paste

    results = model.train(
        data        = str(data_yaml),
        epochs      = args.epochs,
        patience    = args.patience,
        imgsz       = args.imgsz,
        batch       = args.batch,         # -1 = auto-detect for GPU VRAM
        workers     = args.workers,
        project     = PROJECT,
        name        = args.name,
        exist_ok    = True,               # overwrite if re-running
        pretrained  = True,
        cos_lr      = True,               # cosine LR schedule
        amp         = True,               # mixed precision (FP16)
        cache       = "disk",             # cache images to disk (faster epochs)
        close_mosaic = args.close_mosaic,
        label_smoothing = args.label_smoothing,
        nms         = True,
        conf        = 0.001,              # low conf threshold during training NMS
        iou         = 0.7,
        **hyp,
        **train_kwargs,
    )

    # ── Report ────────────────────────────────────────────────────────────────
    save_dir = Path(PROJECT) / args.name
    best_pt  = save_dir / "weights" / "best.pt"
    print(f"\n[train] Training complete.")
    print(f"  Best weights : {best_pt}")
    print(f"  Results dir  : {save_dir}")

    if results is not None:
        metrics = results.results_dict
        print(f"\n[metrics] Final validation metrics:")
        for k in ["metrics/precision(B)", "metrics/recall(B)",
                   "metrics/mAP50(B)", "metrics/mAP50-95(B)"]:
            if k in metrics:
                print(f"  {k:<30} {metrics[k]:.4f}")

    # ── ONNX export ───────────────────────────────────────────────────────────
    if not args.no_export and best_pt.exists():
        onnx_path = export_onnx(best_pt, args.imgsz)
        size_mb = onnx_path.stat().st_size / 1e6
        print(f"[export] ONNX size: {size_mb:.1f} MB")
        if size_mb > 420:
            print("[warn] ONNX file exceeds 420 MB sandbox limit — use best.pt instead")

    # ── Submission hint ───────────────────────────────────────────────────────
    print("\n─── Next steps ───────────────────────────────────────────────────")
    print(f"  Copy weights for submission:")
    print(f"    cp {best_pt} submission/best.pt")
    print(f"  Then run:  bash create_submission.sh")


if __name__ == "__main__":
    main()
