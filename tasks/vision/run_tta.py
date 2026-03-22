"""
run_tta.py  —  NorgesGruppen Submission with Test-Time Augmentation
--------------------------------------------------------------------
Drop-in replacement for run.py with TTA (horizontal flip + WBF fusion).
Uses pure onnxruntime — NO ultralytics dependency.

Usage in submission: copy this file as run.py in your zip.

    python run_tta.py --input /data/images --output /output/predictions.json

Expected mAP gain vs run.py: +1–3 %
Timing on L4 GPU @ imgsz=960: ~60–120 s for ~250 images (safe under 300 s)

Sandbox security constraints honoured:
  pathlib not os/sys | json not yaml | no eval/exec/subprocess/threading
"""

import argparse
import json
from pathlib import Path

import numpy as np
import cv2
from ensemble_boxes import weighted_boxes_fusion   # pre-installed in sandbox

# ─── Configuration ────────────────────────────────────────────────────────────

MODEL_FILE  = Path(__file__).parent / "best.onnx"
META_FILE   = Path(__file__).parent / "best.json"

# ONNX models are exported at a fixed spatial size — cannot resize at runtime.
# Override via best.json (written by export_onnx.py).
IMGSZ       = 960
CONF_THRES  = 0.10        # lower than run.py — more candidates for WBF to fuse
IOU_THRES   = 0.50
MAX_DET     = 500

WBF_IOU     = 0.50
WBF_SKIP    = 0.001
TTA_FLIPLR  = True

PROVIDERS   = ["CUDAExecutionProvider", "CPUExecutionProvider"]


# ─── Pre / post-processing ────────────────────────────────────────────────────

def letterbox(img: np.ndarray, target: int, color=(114, 114, 114)):
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


def preprocess(img_bgr: np.ndarray, target: int, dtype: np.dtype):
    lb_img, scale, pad = letterbox(img_bgr, target)
    img = lb_img[:, :, ::-1].astype(np.float32) / 255.0
    img = np.expand_dims(np.transpose(img, (2, 0, 1)), 0)
    return img.astype(dtype), scale, pad


def xywh2xyxy(b: np.ndarray) -> np.ndarray:
    o = np.empty_like(b)
    o[:, 0] = b[:, 0] - b[:, 2] / 2
    o[:, 1] = b[:, 1] - b[:, 3] / 2
    o[:, 2] = b[:, 0] + b[:, 2] / 2
    o[:, 3] = b[:, 1] + b[:, 3] / 2
    return o


# ─── Single forward pass ──────────────────────────────────────────────────────

def infer(sess, inp_name: str, inp_dtype: np.dtype, out_names: list,
          img_bgr: np.ndarray, imgsz: int, fliplr: bool):
    """
    Run one forward pass, optionally with horizontal flip.
    The flip is applied to the image here; coordinate un-mirroring happens
    inside onnx_to_wbf_format so it stays close to the normalisation step.
    Returns: raw_output, orig_w, orig_h, scale, pad
    """
    if fliplr:
        img_bgr = img_bgr[:, ::-1, :]        # flip a copy — doesn't mutate caller
    inp, scale, pad = preprocess(img_bgr, imgsz, inp_dtype)
    raw = sess.run(out_names, {inp_name: inp})
    h, w = img_bgr.shape[:2]
    return raw[0], w, h, scale, pad


def onnx_to_wbf_format(raw: np.ndarray, orig_w: int, orig_h: int,
                        scale: float, pad: tuple,
                        conf_thr: float, fliplr: bool):
    """
    Decode raw ONNX [1, 4+nc, A] → WBF-ready normalised [0,1] boxes.
    Un-mirrors x-coordinates if fliplr=True.
    """
    pred    = raw[0].T.astype(np.float32)     # [A, 4+nc]
    bxs     = pred[:, :4]                     # cx, cy, w, h  (letterbox pixels)
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
        # Remove letterbox
        x1 = (float(x1) - pad_l) / scale
        y1 = (float(y1) - pad_t) / scale
        x2 = (float(x2) - pad_l) / scale
        y2 = (float(y2) - pad_t) / scale

        if fliplr:
            # Mirror x-coords back to original (un-flipped) image space
            x1, x2 = orig_w - x2, orig_w - x1

        # Normalise and clamp to [0, 1]
        x1n = max(0., min(x1 / orig_w, 1.))
        y1n = max(0., min(y1 / orig_h, 1.))
        x2n = max(0., min(x2 / orig_w, 1.))
        y2n = max(0., min(y2 / orig_h, 1.))
        if x2n <= x1n or y2n <= y1n:
            continue

        boxes_out.append([x1n, y1n, x2n, y2n])
        scores_out.append(float(c))
        labels_out.append(int(l))

    return boxes_out, scores_out, labels_out


# ─── CLI ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input",  required=True)
    p.add_argument("--output", required=True)
    return p.parse_args()


def extract_image_id(p: str) -> int:
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
            "Run: python export_onnx.py"
        )

    import onnxruntime as ort
    sess_opts = ort.SessionOptions()
    sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess      = ort.InferenceSession(str(MODEL_FILE), sess_opts, providers=PROVIDERS)
    inp_meta  = sess.get_inputs()[0]
    inp_name  = inp_meta.name
    inp_dtype = np.float16 if "float16" in inp_meta.type else np.float32
    out_names = [o.name for o in sess.get_outputs()]

    imgsz = IMGSZ
    if META_FILE.exists():
        meta  = json.loads(META_FILE.read_text())
        imgsz = meta.get("imgsz", IMGSZ)

    aug_configs = [False, True] if TTA_FLIPLR else [False]

    print(f"[run_tta] Provider : {sess.get_providers()[0]}")
    print(f"[run_tta] Model    : {MODEL_FILE.name}  imgsz={imgsz}")
    print(f"[run_tta] TTA      : fliplr={TTA_FLIPLR} ({len(aug_configs)} pass/image)")
    print(f"[run_tta] Images   : {len(image_paths)}")

    all_predictions = []
    n = len(image_paths)

    for i, img_path in enumerate(image_paths):
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            continue
        orig_h, orig_w = img_bgr.shape[:2]
        img_id = extract_image_id(str(img_path))

        per_aug_boxes, per_aug_scores, per_aug_labels = [], [], []

        for fliplr in aug_configs:
            raw, w, h, scale, pad = infer(
                sess, inp_name, inp_dtype, out_names,
                img_bgr, imgsz, fliplr
            )
            boxes, scores, labels = onnx_to_wbf_format(
                raw, orig_w, orig_h, scale, pad, CONF_THRES, fliplr
            )
            if boxes:
                per_aug_boxes.append(boxes)
                per_aug_scores.append(scores)
                per_aug_labels.append(labels)

        if not per_aug_boxes:
            continue

        fused_boxes, fused_scores, fused_labels = weighted_boxes_fusion(
            per_aug_boxes, per_aug_scores, per_aug_labels,
            iou_thr      = WBF_IOU,
            skip_box_thr = WBF_SKIP,
            conf_type    = "avg",
        )

        for (x1n, y1n, x2n, y2n), score, label in zip(
                fused_boxes, fused_scores, fused_labels):
            x1 = x1n * orig_w;  y1 = y1n * orig_h
            x2 = x2n * orig_w;  y2 = y2n * orig_h
            wb = x2 - x1;       hb = y2 - y1
            if wb < 1 or hb < 1:
                continue
            all_predictions.append({
                "image_id"   : img_id,
                "category_id": int(label),
                "bbox"       : [round(x1,2), round(y1,2), round(wb,2), round(hb,2)],
                "score"      : round(float(score), 4),
            })

        if (i + 1) % 10 == 0 or (i + 1) == n:
            print(f"[run_tta] {i+1}/{n} done …")

    print(f"[run_tta] Total predictions : {len(all_predictions)}")
    out_path.write_text(json.dumps(all_predictions, indent=2))
    print(f"[run_tta] Saved → {out_path}")


if __name__ == "__main__":
    main()
