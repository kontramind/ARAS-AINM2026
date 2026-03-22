"""
run_ensemble3_tta.py  —  NorgesGruppen Submission: Three-Model Ensemble + TTA
------------------------------------------------------------------------------
Runs three ONNX models (YOLOv8x@960 + YOLOv8l@960 + YOLOv8m@960) on each image.
Optionally applies horizontal-flip TTA per model.
All predictions (3 or 6 passes per image) are fused with Weighted Boxes Fusion.

Usage:
    python run_ensemble3_tta.py --input /data/images --output /output/predictions.json

Sandbox compatibility:
    - Pure onnxruntime — no ultralytics
    - Only: pathlib, json, numpy, cv2, ensemble_boxes, onnxruntime
    - No eval/exec/subprocess/threading/socket

Files expected in same directory as this script:
    best.onnx    — model A: YOLOv8x@960, FP16  (131 MB)
    best_l.onnx  — model B: YOLOv8l@960, FP16  ( 84 MB)
    best_m.onnx  — model C: YOLOv8m@960, FP16  ( 50 MB)
    best.json    — metadata sidecar for A (optional)
    best_l.json  — metadata sidecar for B (optional)
    best_m.json  — metadata sidecar for C (optional)
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from ensemble_boxes import weighted_boxes_fusion

# ─── Model file locations ─────────────────────────────────────────────────────
_HERE   = Path(__file__).parent
MODEL_A = _HERE / "best.onnx"    # YOLOv8x@960
MODEL_B = _HERE / "best_l.onnx"  # YOLOv8l@960
MODEL_C = _HERE / "best_m.onnx"  # YOLOv8m@960

PROVIDERS = ["CUDAExecutionProvider", "CPUExecutionProvider"]

# ─── Inference config ─────────────────────────────────────────────────────────
CONF_THRES  = 0.05
WBF_IOU     = 0.55
WBF_SKIP    = 0.001
TTA_FLIPLR  = True    # horizontal-flip TTA (doubles passes per model)


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


def decode(raw: np.ndarray, orig_w: int, orig_h: int,
           scale: float, pad: tuple, conf_thr: float,
           fliplr: bool = False):
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
        x1 = (float(x1) - pad_l) / scale
        y1 = (float(y1) - pad_t) / scale
        x2 = (float(x2) - pad_l) / scale
        y2 = (float(y2) - pad_t) / scale

        if fliplr:
            x1, x2 = orig_w - x2, orig_w - x1

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


# ─── Session loader ───────────────────────────────────────────────────────────

def load_session(model_path: Path):
    import onnxruntime as ort
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess      = ort.InferenceSession(str(model_path), opts, providers=PROVIDERS)
    inp_meta  = sess.get_inputs()[0]
    inp_dtype = np.float16 if "float16" in inp_meta.type else np.float32
    out_names = [o.name for o in sess.get_outputs()]
    meta_file = model_path.with_suffix(".json")
    imgsz     = 960
    if meta_file.exists():
        imgsz = json.loads(meta_file.read_text()).get("imgsz", 960)
    return sess, inp_meta.name, inp_dtype, out_names, imgsz


# ─── CLI ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Three-model ensemble + TTA inference")
    p.add_argument("--input",   required=True,
                   help="Directory of input images")
    p.add_argument("--output",  required=True,
                   help="Output predictions JSON path")
    p.add_argument("--model-a", default=str(MODEL_A))
    p.add_argument("--model-b", default=str(MODEL_B))
    p.add_argument("--model-c", default=str(MODEL_C))
    p.add_argument("--no-tta",  action="store_true",
                   help="Disable horizontal-flip TTA")
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

    model_paths = [Path(args.model_a), Path(args.model_b), Path(args.model_c)]
    for mp in model_paths:
        if not mp.exists():
            raise FileNotFoundError(
                f"Model not found: {mp}\n"
                "Ensure best.onnx, best_l.onnx and best_m.onnx are next to run.py."
            )

    sessions = [load_session(mp) for mp in model_paths]

    use_tta  = TTA_FLIPLR and not args.no_tta
    flips    = [False, True] if use_tta else [False]
    n_passes = len(flips) * len(sessions)

    print(f"[ensemble3] Provider  : {sessions[0][0].get_providers()[0]}")
    for i, (mp, (sess, _, _, _, imgsz)) in enumerate(zip(model_paths, sessions)):
        print(f"[ensemble3] Model {chr(65+i)}   : {mp.name}  imgsz={imgsz}")
    print(f"[ensemble3] TTA fliplr: {use_tta}  ({n_passes} passes/image)")
    print(f"[ensemble3] Images    : {len(image_paths)}")

    all_predictions = []
    n = len(image_paths)

    for i, img_path in enumerate(image_paths):
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            continue
        orig_h, orig_w = img_bgr.shape[:2]
        img_id = extract_image_id(str(img_path))

        all_boxes, all_scores, all_labels = [], [], []

        for (sess, inp_name, inp_dtype, out_names, imgsz) in sessions:
            for fliplr in flips:
                src = img_bgr[:, ::-1, :] if fliplr else img_bgr
                inp, scale, pad = preprocess(src, imgsz, inp_dtype)
                raw = sess.run(out_names, {inp_name: inp})
                b, s, l = decode(raw[0], orig_w, orig_h, scale, pad,
                                 CONF_THRES, fliplr=fliplr)
                if b:
                    all_boxes.append(b)
                    all_scores.append(s)
                    all_labels.append(l)

        if not all_boxes:
            continue

        fused_boxes, fused_scores, fused_labels = weighted_boxes_fusion(
            all_boxes, all_scores, all_labels,
            iou_thr      = WBF_IOU,
            skip_box_thr = WBF_SKIP,
            conf_type    = "box_and_model_avg",
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
                "bbox"       : [round(x1, 2), round(y1, 2),
                                round(wb, 2), round(hb, 2)],
                "score"      : round(float(score), 4),
            })

        if (i + 1) % 10 == 0 or (i + 1) == n:
            print(f"[ensemble3] {i+1}/{n} done ...")

    print(f"[ensemble3] Total predictions : {len(all_predictions)}")
    out_path.write_text(json.dumps(all_predictions, indent=2))
    print(f"[ensemble3] Saved → {out_path}")


if __name__ == "__main__":
    main()
