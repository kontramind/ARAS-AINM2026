"""
experiments/exp2_staged_lr.py
------------------------------
Experiment 2: Staged LR restart (fine-tuning pass).

What changes vs baseline
------------------------
After the initial training run converges, resume from best.pt with:
  - Lower initial LR (lr0 = 0.0001 instead of 0.001)
  - Shorter warmup
  - Mosaic disabled (close_mosaic=0) — model has seen enough composites
  - More conservative augmentation — tighten scale/translate
  - Longer patience

This is a "refinement" pass that squeezes out 1-3% mAP after the
cosine schedule has already decayed near zero.

Hypothesis
----------
The model stalls at a local minimum near the end of training. A fresh
LR cycle with conservative augmentation helps escape it, especially
for the classification head which benefits from cleaner crops.

How to run
----------
# Prerequisite: baseline training must be done first
# (runs/detect/norgesgruppen/weights/best.pt must exist)

# Run the staged LR restart
python experiments/exp2_staged_lr.py

# Or point at exp1's best model if that was better:
python experiments/exp2_staged_lr.py \
    --weights runs/detect/exp1_corrected/weights/best.pt \
    --data    data/yolo_corrected/dataset.yaml

# Validate after
python validate.py --weights runs/detect/exp2_staged_lr/weights/best.pt

How to build on this
--------------------
exp2 produces runs/detect/exp2_staged_lr/weights/best.pt.
Feed this into exp3 (SAHI) or exp4 (ensemble) as the stronger base model.
"""

import argparse
from pathlib import Path
from ultralytics import YOLO

# ─── Staged hyperparameters ───────────────────────────────────────────────────
# These are INTENTIONALLY more conservative than train.py to avoid unlearning.

STAGED_HYP = dict(
    optimizer    = "AdamW",
    lr0          = 0.0001,    # 10x lower than initial training
    lrf          = 0.01,
    momentum     = 0.937,
    weight_decay = 0.0005,
    warmup_epochs = 1.0,      # almost no warmup — model is already warm
    warmup_momentum = 0.8,
    warmup_bias_lr  = 0.01,

    # Loss
    box  = 7.5,
    cls  = 0.5,
    dfl  = 1.5,

    # Augmentation — pulled back significantly
    degrees    = 2.0,
    translate  = 0.05,
    scale      = 0.3,
    shear      = 0.0,
    flipud     = 0.0,
    fliplr     = 0.5,
    hsv_h      = 0.010,
    hsv_s      = 0.4,
    hsv_v      = 0.2,
    mosaic     = 0.0,    # OFF — kills composites in refinement stage
    mixup      = 0.0,    # OFF
    copy_paste = 0.0,    # OFF
    erasing    = 0.2,
)

PROJECT = "runs/detect"
NAME    = "exp2_staged_lr"


def parse_args():
    p = argparse.ArgumentParser(description="Staged LR restart fine-tuning")
    p.add_argument("--weights", default="runs/detect/norgesgruppen/weights/best.pt",
                   help="Starting weights (default: baseline best.pt)")
    p.add_argument("--data",    default="data/dataset.yaml",
                   help="Dataset YAML (default: data/dataset.yaml)")
    p.add_argument("--epochs",  type=int, default=50,
                   help="Fine-tuning epochs (default: 50)")
    p.add_argument("--patience",type=int, default=20)
    p.add_argument("--imgsz",   type=int, default=960)
    p.add_argument("--batch",   type=int, default=-1)
    p.add_argument("--device",  default=None)
    p.add_argument("--no-export", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    weights = Path(args.weights)

    if not weights.exists():
        raise FileNotFoundError(
            f"Weights not found: {weights}\n"
            "Run train.py (baseline) first, or pass --weights path/to/best.pt"
        )

    print(f"[exp2] Staged LR restart from: {weights}")
    print(f"[exp2] lr0=0.0001  mosaic=OFF  epochs={args.epochs}")

    model = YOLO(str(weights))

    train_kwargs = {}
    if args.device is not None:
        train_kwargs["device"] = args.device

    results = model.train(
        data           = args.data,
        epochs         = args.epochs,
        patience       = args.patience,
        imgsz          = args.imgsz,
        batch          = args.batch,
        workers        = 4,
        project        = PROJECT,
        name           = NAME,
        exist_ok       = True,
        pretrained     = False,   # don't reset to imagenet — continue from checkpoint
        cos_lr         = True,
        amp            = True,
        cache          = "disk",
        close_mosaic   = 0,       # mosaic already off in STAGED_HYP
        label_smoothing = 0.05,   # less smoothing — model is more confident now
        nms            = True,
        conf           = 0.001,
        iou            = 0.7,
        **STAGED_HYP,
        **train_kwargs,
    )

    best_pt = Path(PROJECT) / NAME / "weights" / "best.pt"
    print(f"\n[exp2] Done. Best weights: {best_pt}")

    if results is not None:
        m = results.results_dict
        for k in ["metrics/mAP50(B)", "metrics/mAP50-95(B)", "metrics/recall(B)"]:
            if k in m:
                print(f"  {k:<30} {m[k]:.4f}")

    if not args.no_export and best_pt.exists():
        print(f"\n[exp2] Exporting to ONNX …")
        onnx = model.export(format="onnx", imgsz=args.imgsz, opset=17,
                            dynamic=False, simplify=True, half=True, nms=False)
        print(f"[exp2] ONNX → {onnx}")

    print("\n[exp2] Next: validate and compare vs baseline")
    print(f"  python validate.py --weights {best_pt}")


if __name__ == "__main__":
    main()
