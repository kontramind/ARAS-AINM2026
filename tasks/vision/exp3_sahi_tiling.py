"""
experiments/exp3_sahi_tiling.py
---------------------------------
Experiment 3: Sliced inference (SAHI-style) at prediction time.
              + optional tile-augmented training data generation.

What changes vs baseline
------------------------
Shelf images are 2000×1500 px. Many products occupy only 60–120 px.
At imgsz=960, a 100px product becomes ~50px in the letterboxed tensor —
right at the edge of reliable detection.

SAHI (Sliced Inference): divide each image into overlapping tiles at
inference time, run the model on each tile, then merge detections with WBF.
No retraining needed — this is a pure post-processing improvement.

Expected gain: +3–8% detection mAP for small objects.

Two modes
---------
A) --mode infer   Run sliced inference on a folder of images (for submission)
B) --mode tiles   Generate tiled training images and add to yolo dataset

How to run (inference mode — no retraining)
-------------------------------------------
# Generate predictions with tiling on your val set:
python experiments/exp3_sahi_tiling.py \
    --mode   infer \
    --input  data/yolo/images/val \
    --output tiled_preds.json \
    --model  runs/detect/norgesgruppen/weights/best.onnx

# Score locally:
python validate.py --save-predictions tiled_preds.json

How to use as submission run.py
--------------------------------
Copy the predict() function from this file into a new run.py:
    bash create_submission.sh --script experiments/exp3_sahi_tiling.py

How to build on this
--------------------
Chain with exp2: use exp2's best.onnx as --model here.
Chain with exp4: pass tiled predictions into the two-model ensemble.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import cv2
from ensemble_boxes import weighted_boxes_fusion

# ─── Config ───────────────────────────────────────────────────────────────────

MODEL_FILE = Path(__file__).parent.parent / "best.onnx"
META_FILE  = Path(__file__).parent.parent / "best.json"

IMGSZ       = 960         # tile size (matches model export size)
TILE_SIZES  = [960, 640]  # tile sizes to slice at
TILE_OVERLAP = 0.20       # 20% overlap between tiles
CONF_THRES  = 0.10
WBF_IOU     = 0.50
WBF_SKIP    = 0.001
MAX_DET     = 500
PROVIDERS   = ["CUDAExecutionProvider", "CPUExecutionProvider"]


# ─── Pre / post-processing (identical to run.py) ─────────────────────────────

def letterbox(img, target, color=(114, 114, 114)):
    h, w   = img.shape[:2]
    scale  = min(target / h, target / w)
    new_w  = int(round(w * scale))
    new_h  = int(round(h * scale))
    img    = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    pad_w  = target - new_w
    pad_h  = target - new_h
    top    = pad_h // 2
    left   = pad_w // 2
    img    = cv2.copyMakeBorder(img, top, pad_h - top, left, pad_w - left,
                                 cv2.BORDER_CONSTANT, value=color)
    return img, scale, (left, top)


def preprocess(img_bgr, target, dtype):
    lb, scale, pad = letterbox(img_bgr, target)
    t = np.expand_dims(np.transpose(
        lb[:, :, ::-1].astype(np.float32) / 255.0, (2, 0, 1)), 0)
    return t.astype(dtype), scale, pad


def xywh2xyxy(b):
    o = np.empty_like(b)
    o[:, 0] = b[:, 0] - b[:, 2] / 2
    o[:, 1] = b[:, 1] - b[:, 3] / 2
    o[:, 2] = b[:, 0] + b[:, 2] / 2
    o[:, 3] = b[:, 1] + b[:, 3] / 2
    return o


def decode_raw(raw, orig_w, orig_h, scale, pad, conf_thr,
               offset_x=0, offset_y=0):
    """
    Decode one ONNX output, optionally offsetting coordinates by (offset_x, offset_y)
    to convert from tile-space back to full-image space.
    Returns normalised [0,1] boxes relative to orig_w × orig_h.
    """
    pred    = raw[0].T.astype(np.float32)
    bxs     = pred[:, :4]
    cls_sc  = pred[:, 4:]
    cls_ids = cls_sc.argmax(1)
    conf    = cls_sc[np.arange(len(cls_sc)), cls_ids]

    mask = conf >= conf_thr
    if not mask.any():
        return [], [], []

    bxs     = xywh2xyxy(bxs[mask])
    conf    = conf[mask]
    cls_ids = cls_ids[mask]
    pad_l, pad_t = pad

    boxes_out, scores_out, labels_out = [], [], []
    for (x1, y1, x2, y2), c, l in zip(bxs, conf, cls_ids):
        x1 = (float(x1) - pad_l) / scale + offset_x
        y1 = (float(y1) - pad_t) / scale + offset_y
        x2 = (float(x2) - pad_l) / scale + offset_x
        y2 = (float(y2) - pad_t) / scale + offset_y
        x1n = max(0., min(x1 / orig_w, 1.))
        y1n = max(0., min(y1 / orig_h, 1.))
        x2n = max(0., min(x2 / orig_w, 1.))
        y2n = max(0., min(y2 / orig_h, 1.))
        if x2n > x1n and y2n > y1n:
            boxes_out.append([x1n, y1n, x2n, y2n])
            scores_out.append(float(c))
            labels_out.append(int(l))
    return boxes_out, scores_out, labels_out


# ─── Tile generation ──────────────────────────────────────────────────────────

def get_tiles(img_h: int, img_w: int, tile_size: int, overlap: float):
    """
    Generate (x1, y1, x2, y2) tile coordinates for an image.
    Tiles are at most tile_size × tile_size, with `overlap` fraction of overlap.
    """
    stride = int(tile_size * (1 - overlap))
    tiles  = []
    y = 0
    while y < img_h:
        x = 0
        while x < img_w:
            x2 = min(x + tile_size, img_w)
            y2 = min(y + tile_size, img_h)
            tiles.append((x, y, x2, y2))
            if x2 == img_w:
                break
            x += stride
        if y2 == img_h:
            break
        y += stride
    return tiles


# ─── Sliced inference for one image ──────────────────────────────────────────

def sliced_predict(sess, inp_name, inp_dtype, out_names,
                   img_bgr: np.ndarray, tile_sizes, overlap, conf_thr, wbf_iou):
    """
    Run sliced inference over multiple tile sizes.
    Returns WBF-fused (boxes_norm, scores, labels).
    """
    orig_h, orig_w = img_bgr.shape[:2]
    all_boxes, all_scores, all_labels = [], [], []

    # Full-image pass first (catches large objects)
    inp, scale, pad = preprocess(img_bgr, tile_sizes[0], inp_dtype)
    raw = sess.run(out_names, {inp_name: inp})
    b, s, l = decode_raw(raw[0], orig_w, orig_h, scale, pad, conf_thr)
    if b:
        all_boxes.append(b); all_scores.append(s); all_labels.append(l)

    # Tiled passes
    for tile_size in tile_sizes:
        tiles = get_tiles(orig_h, orig_w, tile_size, overlap)
        for (tx1, ty1, tx2, ty2) in tiles:
            tile = img_bgr[ty1:ty2, tx1:tx2]
            inp, scale, pad = preprocess(tile, tile_size, inp_dtype)
            raw  = sess.run(out_names, {inp_name: inp})
            th   = ty2 - ty1
            tw   = tx2 - tx1
            b, s, l = decode_raw(raw[0], orig_w, orig_h, scale, pad, conf_thr,
                                   offset_x=tx1, offset_y=ty1)
            if b:
                all_boxes.append(b); all_scores.append(s); all_labels.append(l)

    if not all_boxes:
        return [], [], []

    return weighted_boxes_fusion(
        all_boxes, all_scores, all_labels,
        iou_thr=wbf_iou, skip_box_thr=WBF_SKIP, conf_type="avg"
    )


# ─── CLI ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="SAHI-style sliced inference")
    p.add_argument("--mode",    choices=["infer"], default="infer",
                   help="infer: run sliced prediction on --input folder")
    p.add_argument("--input",   required=True,
                   help="Folder of images to predict on")
    p.add_argument("--output",  required=True,
                   help="Output JSON path")
    p.add_argument("--model",   default=str(MODEL_FILE),
                   help=f"ONNX model path (default: {MODEL_FILE})")
    p.add_argument("--tile-sizes", nargs="+", type=int, default=TILE_SIZES,
                   help="Tile sizes to use (default: 960 640)")
    p.add_argument("--overlap", type=float, default=TILE_OVERLAP,
                   help=f"Tile overlap fraction (default: {TILE_OVERLAP})")
    return p.parse_args()


def extract_image_id(p: str) -> int:
    return int(Path(p).stem.split("_")[-1])


def main():
    args       = parse_args()
    img_dir    = Path(args.input)
    out_path   = Path(args.output)
    model_path = Path(args.model)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(img_dir.glob("*.jpg")) + sorted(img_dir.glob("*.jpeg"))
    if not image_paths:
        out_path.write_text("[]")
        return

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    import onnxruntime as ort
    sess_opts = ort.SessionOptions()
    sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess      = ort.InferenceSession(str(model_path), sess_opts,
                                      providers=PROVIDERS)
    inp_meta  = sess.get_inputs()[0]
    inp_name  = inp_meta.name
    inp_dtype = np.float16 if "float16" in inp_meta.type else np.float32
    out_names = [o.name for o in sess.get_outputs()]

    meta_file = model_path.with_suffix(".json")
    imgsz = IMGSZ
    if meta_file.exists():
        imgsz = json.loads(meta_file.read_text()).get("imgsz", IMGSZ)

    tile_sizes = args.tile_sizes
    print(f"[exp3] Provider   : {sess.get_providers()[0]}")
    print(f"[exp3] Model      : {model_path.name}")
    print(f"[exp3] Tile sizes : {tile_sizes}  overlap={args.overlap}")
    print(f"[exp3] Images     : {len(image_paths)}")

    all_predictions = []
    n = len(image_paths)

    for i, img_path in enumerate(image_paths):
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            continue
        orig_h, orig_w = img_bgr.shape[:2]
        img_id = extract_image_id(str(img_path))

        fused_boxes, fused_scores, fused_labels = sliced_predict(
            sess, inp_name, inp_dtype, out_names,
            img_bgr, tile_sizes, args.overlap, CONF_THRES, WBF_IOU
        )

        for (x1n, y1n, x2n, y2n), score, label in zip(
                fused_boxes, fused_scores, fused_labels):
            x1 = x1n * orig_w; y1 = y1n * orig_h
            x2 = x2n * orig_w; y2 = y2n * orig_h
            wb = x2 - x1;      hb = y2 - y1
            if wb < 1 or hb < 1:
                continue
            all_predictions.append({
                "image_id"   : img_id,
                "category_id": int(label),
                "bbox"       : [round(x1,2), round(y1,2), round(wb,2), round(hb,2)],
                "score"      : round(float(score), 4),
            })

        if (i + 1) % 10 == 0 or (i + 1) == n:
            print(f"[exp3] {i+1}/{n} done …")

    print(f"[exp3] Total predictions : {len(all_predictions)}")
    out_path.write_text(json.dumps(all_predictions, indent=2))
    print(f"[exp3] Saved → {out_path}")
    print("\n[exp3] To use as submission run.py:")
    print("  cp experiments/exp3_sahi_tiling.py run_sahi.py")
    print("  # Then update MODEL_FILE at the top to point to your best.onnx")
    print("  # and add the '--input / --output' wiring from run.py")


if __name__ == "__main__":
    main()
