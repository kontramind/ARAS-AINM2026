"""
export_onnx.py
--------------
Exports a trained YOLOv8 .pt checkpoint → .onnx for sandbox submission.

Why ONNX?
  - Eliminates ultralytics version mismatch errors in the sandbox
  - onnxruntime-gpu 1.20.0 is pre-installed and rock-solid
  - FP16 ONNX is ~2-3x faster than FP32 PyTorch on an L4 GPU
  - Self-contained: no ultralytics needed at inference time

Usage
-----
    python export_onnx.py
    python export_onnx.py --weights runs/detect/norgesgruppen/weights/best.pt
    python export_onnx.py --imgsz 1024   # if 1280 export is too large

Output
------
    best.onnx   (next to best.pt, or wherever --out points)
    Prints input/output shapes so you can verify before packing the zip.
"""

import argparse
import json
from pathlib import Path

# ─── Args ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Export YOLOv8 .pt → .onnx")
    p.add_argument("--weights", default="runs/detect/norgesgruppen/weights/best.pt",
                   help="Path to trained .pt file")
    p.add_argument("--imgsz",   type=int, default=960,
                   help="Export image size (must match training; default: 1280)")
    p.add_argument("--out",     default=None,
                   help="Output .onnx path (default: same dir as weights, best.onnx)")
    p.add_argument("--fp32",    action="store_true",
                   help="Export as FP32 instead of FP16 (larger, slower, no GPU half-precision)")
    p.add_argument("--opset",   type=int, default=17,
                   help="ONNX opset version (default: 17; sandbox supports up to 20)")
    return p.parse_args()


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    args    = parse_args()
    pt_path = Path(args.weights)

    if not pt_path.exists():
        raise FileNotFoundError(
            f"Weights not found: {pt_path}\n"
            "Run train.py first."
        )

    out_path = Path(args.out) if args.out else pt_path.with_suffix(".onnx")
    half     = not args.fp32

    print(f"[export] Input  : {pt_path}  ({pt_path.stat().st_size / 1e6:.1f} MB)")
    print(f"[export] Output : {out_path}")
    print(f"[export] imgsz  : {args.imgsz}")
    print(f"[export] opset  : {args.opset}")
    print(f"[export] FP16   : {half}")
    print()

    # ── Load model ────────────────────────────────────────────────────────────
    from ultralytics import YOLO
    model = YOLO(str(pt_path))

    # ── Export ────────────────────────────────────────────────────────────────
    # dynamic=False  → fixed input shape; required for best performance with
    #                  onnxruntime CUDAExecutionProvider
    # simplify=True  → runs onnx-simplifier; reduces graph ops, faster inference
    # nms=False      → we handle NMS ourselves in run.py (gives more control)
    exported = model.export(
        format   = "onnx",
        imgsz    = args.imgsz,
        opset    = args.opset,
        dynamic  = False,
        simplify = True,
        half     = half,
        nms      = False,       # raw box output — we apply NMS in run.py
    )

    # Ultralytics writes the .onnx next to the .pt by default
    default_out = pt_path.with_suffix(".onnx")
    if out_path != default_out and default_out.exists():
        default_out.rename(out_path)

    print(f"\n[export] Written: {out_path}")
    size_mb = out_path.stat().st_size / 1e6
    print(f"[export] Size   : {size_mb:.1f} MB  (limit: 420 MB)")

    if size_mb > 420:
        print("[WARN]  File exceeds 420 MB sandbox limit!")
        print("        Re-run with --imgsz 1024 or --fp32 to reduce size.")
    else:
        print(f"[export] ✓ Size OK  ({420 - size_mb:.0f} MB headroom)")

    # ── Verify with onnxruntime ───────────────────────────────────────────────
    print("\n[verify] Checking ONNX model with onnxruntime …")
    try:
        import onnxruntime as ort
        import numpy as np

        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        sess      = ort.InferenceSession(str(out_path), providers=providers)

        inp  = sess.get_inputs()[0]
        outs = sess.get_outputs()

        print(f"[verify] Provider  : {sess.get_providers()[0]}")
        print(f"[verify] Input     : name={inp.name!r}  shape={inp.shape}  dtype={inp.type}")
        for o in outs:
            print(f"[verify] Output    : name={o.name!r}  shape={o.shape}  dtype={o.type}")

        # Determine expected dtype from model
        dtype   = np.float16 if "float16" in inp.type else np.float32
        n, c, h, w = inp.shape
        dummy   = np.random.rand(1, c, h, w).astype(dtype)
        outputs = sess.run(None, {inp.name: dummy})

        print(f"[verify] Test run output shapes:")
        for i, o in enumerate(outputs):
            print(f"           output[{i}]: {o.shape}  dtype={o.dtype}")

        # Save metadata for run.py to read
        meta = {
            "imgsz"      : args.imgsz,
            "input_name" : inp.name,
            "input_dtype": inp.type,
            "output_names": [o.name for o in outs],
            "nc"         : int(outputs[0].shape[1]) - 4,  # 4+nc columns
            "fp16"       : half,
        }
        meta_path = out_path.with_suffix(".json")
        meta_path.write_text(json.dumps(meta, indent=2))
        print(f"\n[verify] Metadata saved → {meta_path}")
        print("\n[export] ✓ All checks passed — ready for submission!")
        print(f"\n  Next: bash create_submission.sh")

    except ImportError:
        print("[verify] onnxruntime not installed locally — skipping runtime check.")
        print("         The sandbox has onnxruntime-gpu 1.20.0 pre-installed.")
    except Exception as e:
        print(f"[verify] ✗ onnxruntime check failed: {e}")
        raise


if __name__ == "__main__":
    main()
