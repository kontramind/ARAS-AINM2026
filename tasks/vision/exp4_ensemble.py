"""
experiments/exp4_ensemble.py
-----------------------------
Experiment 4: Two-model ensemble with Weighted Boxes Fusion.

What changes vs baseline
------------------------
Train a second model (YOLOv8l at imgsz=640 — faster, different receptive field)
alongside the primary YOLOv8x@960. At inference, run both models on every image
and fuse predictions with WBF.

Hypothesis
----------
Models trained at different scales see products differently. YOLOv8x@960 is
strong on medium-to-large products; YOLOv8l@640 sees more context per product.
WBF rewards predictions both models agree on and filters noise unique to one model.
Expected gain: +2–5% over the single best model.

Both models must fit inside 420 MB total:
  YOLOv8x best.onnx (FP16 @ 960)  ≈ 130 MB
  YOLOv8l best_l.onnx (FP16 @ 640) ≈  85 MB
  Total                             ≈ 215 MB  ✓

How to run
----------
# Step 1 — train second model (smaller/faster)
python train.py \
    --model  yolov8l \
    --imgsz  640 \
    --epochs 150 \
    --name   exp4_model_l

# Step 2 — export second model
python export_onnx.py \
    --weights runs/detect/exp4_model_l/weights/best.pt \
    --imgsz   640 \
    --out     runs/detect/exp4_model_l/weights/best_l.onnx

# Step 3 — test ensemble locally
python experiments/exp4_ensemble.py \
    --model-a  runs/detect/norgesgruppen/weights/best.onnx \
    --model-b  runs/detect/exp4_model_l/weights/best_l.onnx \
    --input    data/yolo/images/val \
    --output   ensemble_preds.json

# Step 4 — to submit the ensemble, use create_submission_ensemble.sh
#          (which packs both .onnx files + this script as run.py)

How to build on this
--------------------
Stack with exp3: wrap sliced_predict() around each model call in run_models().
Stack with exp2: use exp2's refined weights as model-a.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import cv2
from ensemble_boxes import weighted_boxes_fusion

# ─── Default model paths ──────────────────────────────────────────────────────
# In submission zip, rename so both are present:
#   best.onnx   → primary model (YOLOv8x)
#   best_l.onnx → secondary model (YOLOv8l)

MODEL_A = Path(__file__).parent.parent / "best.onnx"
MODEL_B = Path(__file__).parent.parent / "best_l.onnx"

CONF_THRES = 0.10
WBF_IOU    = 0.50
WBF_SKIP   = 0.001
MAX_DET    = 500
PROVIDERS  = ["CUDAExecutionProvider", "CPUExecutionProvider"]


# ─── Pre / post-processing ────────────────────────────────────────────────────

def letterbox(img, target, color=(114, 114, 114)):
    h, w  = img.shape[:2]
    scale = min(target / h, target / w)
    nw    = int(round(w * scale))
    nh    = int(round(h * scale))
    img   = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    ph    = target - nh; pw = target - nw
    t     = ph // 2;     l = pw // 2
    img   = cv2.copyMakeBorder(img, t, ph-t, l, pw-l,
                                cv2.BORDER_CONSTANT, value=color)
    return img, scale, (l, t)


def preprocess(img_bgr, target, dtype):
    lb, scale, pad = letterbox(img_bgr, target)
    t = np.expand_dims(
        np.transpose(lb[:, :, ::-1].astype(np.float32) / 255.0, (2, 0, 1)), 0)
    return t.astype(dtype), scale, pad


def xywh2xyxy(b):
    o = np.empty_like(b)
    o[:, 0] = b[:, 0] - b[:, 2] / 2;  o[:, 1] = b[:, 1] - b[:, 3] / 2
    o[:, 2] = b[:, 0] + b[:, 2] / 2;  o[:, 3] = b[:, 1] + b[:, 3] / 2
    return o


def decode(raw, orig_w, orig_h, scale, pad, conf_thr):
    pred    = raw[0].T.astype(np.float32)
    bxs     = xywh2xyxy(pred[:, :4])
    cls_sc  = pred[:, 4:]
    cls_ids = cls_sc.argmax(1)
    conf    = cls_sc[np.arange(len(cls_sc)), cls_ids]
    mask    = conf >= conf_thr
    if not mask.any():
        return [], [], []
    bxs     = bxs[mask]; conf = conf[mask]; cls_ids = cls_ids[mask]
    pad_l, pad_t = pad
    boxes_out, scores_out, labels_out = [], [], []
    for (x1, y1, x2, y2), c, l in zip(bxs, conf, cls_ids):
        x1 = max(0., min((float(x1) - pad_l) / scale / orig_w, 1.))
        y1 = max(0., min((float(y1) - pad_t) / scale / orig_h, 1.))
        x2 = max(0., min((float(x2) - pad_l) / scale / orig_w, 1.))
        y2 = max(0., min((float(y2) - pad_t) / scale / orig_h, 1.))
        if x2 > x1 and y2 > y1:
            boxes_out.append([x1, y1, x2, y2])
            scores_out.append(float(c))
            labels_out.append(int(l))
    return boxes_out, scores_out, labels_out


# ─── Load session helper ──────────────────────────────────────────────────────

def load_session(model_path: Path):
    import onnxruntime as ort
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess      = ort.InferenceSession(str(model_path), opts, providers=PROVIDERS)
    inp_meta  = sess.get_inputs()[0]
    inp_dtype = np.float16 if "float16" in inp_meta.type else np.float32
    meta_file = model_path.with_suffix(".json")
    imgsz     = 960
    if meta_file.exists():
        imgsz = json.loads(meta_file.read_text()).get("imgsz", 960)
    return sess, inp_meta.name, inp_dtype, [o.name for o in sess.get_outputs()], imgsz


# ─── CLI ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Two-model WBF ensemble")
    p.add_argument("--model-a", default=str(MODEL_A), help="Primary ONNX model")
    p.add_argument("--model-b", default=str(MODEL_B), help="Secondary ONNX model")
    p.add_argument("--input",   required=True)
    p.add_argument("--output",  required=True)
    p.add_argument("--wbf-iou", type=float, default=WBF_IOU)
    return p.parse_args()


def extract_image_id(p: str) -> int:
    return int(Path(p).stem.split("_")[-1])


def main():
    args     = parse_args()
    img_dir  = Path(args.input)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(img_dir.glob("*.jpg")) + sorted(img_dir.glob("*.jpeg"))
    if not image_paths:
        out_path.write_text("[]")
        return

    model_a_path = Path(args.model_a)
    model_b_path = Path(args.model_b)
    for mp in (model_a_path, model_b_path):
        if not mp.exists():
            raise FileNotFoundError(
                f"Model not found: {mp}\n"
                "See the How-to-run section in this file's docstring."
            )

    sess_a, name_a, dtype_a, outs_a, imgsz_a = load_session(model_a_path)
    sess_b, name_b, dtype_b, outs_b, imgsz_b = load_session(model_b_path)

    print(f"[exp4] Model A: {model_a_path.name}  imgsz={imgsz_a}")
    print(f"[exp4] Model B: {model_b_path.name}  imgsz={imgsz_b}")
    print(f"[exp4] Provider: {sess_a.get_providers()[0]}")
    print(f"[exp4] Images: {len(image_paths)}")

    all_predictions = []
    n = len(image_paths)

    for i, img_path in enumerate(image_paths):
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            continue
        orig_h, orig_w = img_bgr.shape[:2]
        img_id = extract_image_id(str(img_path))

        per_model_boxes, per_model_scores, per_model_labels = [], [], []

        for sess, inp_name, inp_dtype, out_names, imgsz in [
            (sess_a, name_a, dtype_a, outs_a, imgsz_a),
            (sess_b, name_b, dtype_b, outs_b, imgsz_b),
        ]:
            inp, scale, pad = preprocess(img_bgr, imgsz, inp_dtype)
            raw = sess.run(out_names, {inp_name: inp})
            b, s, l = decode(raw[0], orig_w, orig_h, scale, pad, CONF_THRES)
            if b:
                per_model_boxes.append(b)
                per_model_scores.append(s)
                per_model_labels.append(l)

        if not per_model_boxes:
            continue

        fused_boxes, fused_scores, fused_labels = weighted_boxes_fusion(
            per_model_boxes, per_model_scores, per_model_labels,
            iou_thr=args.wbf_iou, skip_box_thr=WBF_SKIP, conf_type="avg"
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
            print(f"[exp4] {i+1}/{n} done …")

    print(f"[exp4] Total predictions : {len(all_predictions)}")
    out_path.write_text(json.dumps(all_predictions, indent=2))
    print(f"[exp4] Saved → {out_path}")


if __name__ == "__main__":
    main()
