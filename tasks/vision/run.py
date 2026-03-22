"""
run.py  —  NorgesGruppen Grocery Bot Submission
------------------------------------------------
Executed by the competition sandbox as:

    python run.py --input /data/images --output /output/predictions.json

Design
------
* Uses onnxruntime-gpu directly (pre-installed in sandbox, version 1.20.0)
* NO ultralytics import — eliminates version-mismatch failures
* CUDAExecutionProvider first, CPUExecutionProvider fallback
* Handles both FP16 and FP32 ONNX exports automatically
* Full letterbox pre-processing + coordinate rescaling post-processing
* Vectorised numpy NMS (no scipy / torch needed)

Security constraints honoured
------------------------------
* pathlib instead of os/sys
* json instead of yaml
* No eval/exec/compile/__import__
* No subprocess/threading/multiprocessing/socket
"""

import argparse
import json
from pathlib import Path

import numpy as np
import cv2                        # opencv-python-headless — pre-installed

# ─── Configuration ────────────────────────────────────────────────────────────

MODEL_FILE  = Path(__file__).parent / "best.onnx"
META_FILE   = Path(__file__).parent / "best.json"   # written by export_onnx.py

IMGSZ       = 960         # must match export; overridden by meta file if present
CONF_THRES  = 0.15        # keep low — maximises recall for the 70% detection score
IOU_THRES   = 0.45        # NMS IoU threshold
MAX_DET     = 500         # max detections per image

# Sandbox always has an L4 GPU — use it
PROVIDERS   = ["CUDAExecutionProvider", "CPUExecutionProvider"]


# ─── Pre-processing ───────────────────────────────────────────────────────────

def letterbox(img: np.ndarray, target: int, color=(114, 114, 114)):
    """
    Resize image to target×target with letterboxing (no distortion).
    Returns (resized_image, scale, (pad_left, pad_top)).
    """
    h, w = img.shape[:2]
    scale = min(target / h, target / w)
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    img   = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # Pad to square
    pad_w = target - new_w
    pad_h = target - new_h
    top   = pad_h // 2
    left  = pad_w // 2
    img   = cv2.copyMakeBorder(img, top, pad_h - top, left, pad_w - left,
                                cv2.BORDER_CONSTANT, value=color)
    return img, scale, (left, top)


def preprocess(img_bgr: np.ndarray, target: int, dtype: np.dtype):
    """BGR image → model input tensor [1, 3, H, W] + (scale, pad)."""
    lb_img, scale, pad = letterbox(img_bgr, target)
    img = lb_img[:, :, ::-1]             # BGR → RGB
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))   # HWC → CHW
    img = np.expand_dims(img, 0)          # → [1, 3, H, W]
    return img.astype(dtype), scale, pad


# ─── Post-processing ──────────────────────────────────────────────────────────

def xywh2xyxy(boxes: np.ndarray) -> np.ndarray:
    """Convert [cx, cy, w, h] → [x1, y1, x2, y2]."""
    out = np.empty_like(boxes)
    out[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
    out[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
    out[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
    out[:, 3] = boxes[:, 1] + boxes[:, 3] / 2
    return out


def nms(boxes_xyxy: np.ndarray, scores: np.ndarray, iou_thr: float) -> np.ndarray:
    """Vectorised CPU NMS. Returns indices of kept boxes."""
    x1, y1, x2, y2 = boxes_xyxy[:, 0], boxes_xyxy[:, 1], \
                      boxes_xyxy[:, 2], boxes_xyxy[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep  = []

    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        iou   = inter / (areas[i] + areas[order[1:]] - inter + 1e-7)
        order = order[1:][iou <= iou_thr]

    return np.array(keep, dtype=np.int64)


def postprocess(raw: np.ndarray, orig_hw: tuple, scale: float, pad: tuple,
                conf_thr: float, iou_thr: float, max_det: int) -> list:
    """
    Parse raw YOLOv8 ONNX output → list of (x, y, w, h, score, class_id).

    YOLOv8 ONNX raw output shape: [1, 4+nc, num_anchors]
    Coordinates are in letterboxed image space (pixels, 0–imgsz).
    """
    pred = raw[0].T.astype(np.float32)    # [A, 4+nc]

    boxes_xywh = pred[:, :4]              # [A, 4]
    cls_scores  = pred[:, 4:]             # [A, nc]

    class_ids = cls_scores.argmax(axis=1)
    conf      = cls_scores[np.arange(len(cls_scores)), class_ids]

    mask = conf >= conf_thr
    if not mask.any():
        return []

    boxes_xywh = boxes_xywh[mask]
    conf       = conf[mask]
    class_ids  = class_ids[mask]
    boxes_xyxy = xywh2xyxy(boxes_xywh)

    pad_l, pad_t = pad
    orig_h, orig_w = orig_hw
    results = []

    for cls in np.unique(class_ids):
        m    = class_ids == cls
        bx   = boxes_xyxy[m]
        sc   = conf[m]
        keep = nms(bx, sc, iou_thr)

        for idx in keep[:max_det]:
            x1, y1, x2, y2 = bx[idx]

            # Remove letterbox padding, undo resize scale
            x1 = (float(x1) - pad_l) / scale
            y1 = (float(y1) - pad_t) / scale
            x2 = (float(x2) - pad_l) / scale
            y2 = (float(y2) - pad_t) / scale

            # Clamp to original image
            x1 = max(0.0, min(x1, orig_w))
            y1 = max(0.0, min(y1, orig_h))
            x2 = max(0.0, min(x2, orig_w))
            y2 = max(0.0, min(y2, orig_h))

            w = x2 - x1
            h = y2 - y1
            if w < 1 or h < 1:
                continue

            results.append((x1, y1, w, h, float(sc[idx]), int(cls)))

    results.sort(key=lambda r: -r[4])
    return results[:max_det]


# ─── Argument parsing ─────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="NorgesGruppen shelf product detector")
    p.add_argument("--input",  required=True)
    p.add_argument("--output", required=True)
    return p.parse_args()


def extract_image_id(p: str) -> int:
    """img_00042.jpg → 42"""
    return int(Path(p).stem.split("_")[-1])


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    args     = parse_args()
    img_dir  = Path(args.input)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(img_dir.glob("*.jpg")) + sorted(img_dir.glob("*.jpeg"))
    if not image_paths:
        out_path.write_text("[]")
        return

    if not MODEL_FILE.exists():
        raise FileNotFoundError(
            f"ONNX model not found: {MODEL_FILE}\n"
            "Run: python export_onnx.py  then rebuild the zip."
        )

    # ── Load ONNX session ─────────────────────────────────────────────────────
    import onnxruntime as ort

    sess_opts = ort.SessionOptions()
    sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    sess     = ort.InferenceSession(str(MODEL_FILE), sess_opts, providers=PROVIDERS)
    inp_meta = sess.get_inputs()[0]
    inp_name = inp_meta.name
    inp_dtype = np.float16 if "float16" in inp_meta.type else np.float32
    out_names = [o.name for o in sess.get_outputs()]

    # Read imgsz from metadata sidecar if present
    imgsz = IMGSZ
    if META_FILE.exists():
        meta  = json.loads(META_FILE.read_text())
        imgsz = meta.get("imgsz", IMGSZ)

    print(f"[run] Provider : {sess.get_providers()[0]}")
    print(f"[run] Model    : {MODEL_FILE.name}")
    print(f"[run] Input    : {inp_meta.shape}  dtype={inp_dtype.__name__}")
    print(f"[run] imgsz    : {imgsz}  conf={CONF_THRES}  iou={IOU_THRES}")
    print(f"[run] Images   : {len(image_paths)}")

    # ── Inference ─────────────────────────────────────────────────────────────
    all_predictions = []
    n = len(image_paths)

    for i, img_path in enumerate(image_paths):
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            print(f"[warn] Cannot read {img_path.name} — skipping")
            continue

        orig_h, orig_w = img_bgr.shape[:2]
        inp_tensor, scale, pad = preprocess(img_bgr, imgsz, inp_dtype)

        raw  = sess.run(out_names, {inp_name: inp_tensor})
        dets = postprocess(raw[0], (orig_h, orig_w), scale, pad,
                           CONF_THRES, IOU_THRES, MAX_DET)

        img_id = extract_image_id(str(img_path))
        for (x, y, w, h, score, cls_id) in dets:
            all_predictions.append({
                "image_id"   : img_id,
                "category_id": cls_id,
                "bbox"       : [round(x, 2), round(y, 2), round(w, 2), round(h, 2)],
                "score"      : round(score, 4),
            })

        if (i + 1) % 10 == 0 or (i + 1) == n:
            print(f"[run] {i+1}/{n} images done …")

    print(f"[run] Total predictions : {len(all_predictions)}")
    out_path.write_text(json.dumps(all_predictions, indent=2))
    print(f"[run] Saved → {out_path}")


if __name__ == "__main__":
    main()
