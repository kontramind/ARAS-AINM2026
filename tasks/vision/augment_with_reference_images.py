"""
augment_with_reference_images.py
---------------------------------
Slot-based product substitution augmentation.

For each synthetic image:
  1. Pick a real shelf image as background (img_*.jpg only)
  2. Load its YOLO annotations
  3. Find slots (bboxes) whose class is a COMMON product (>= min_anno annotations)
  4. For each rare product to inject, find a common slot with close aspect ratio (<=1.5x mismatch)
  5. Scale rare reference image to fill that slot (105% oversize for clean coverage)
  6. Apply simple color transfer so lighting matches the slot region
  7. Hard-paste onto background
  8. Replace the common slot's annotation with the rare product's class ID
  9. Write updated label file (all original labels, with substituted slots updated)

Run BEFORE train.py. Images saved to disk so you can inspect them before training.

Usage
-----
    python augment_with_reference_images.py --n-synth 500 --min-anno 20
    python augment_with_reference_images.py --n-synth 200 --preview-dir /tmp/preview

Arguments
---------
--product-dir   Path to extracted NM_NGD_product_images folder (default: data/product_images)
--yolo-dir      Path to YOLO dataset root (default: data/yolo)
--metadata      Path to metadata.json (default: data/product_images/metadata.json)
--annotations   Path to original annotations.json (default: data/train/annotations.json)
--n-synth       Number of synthetic images to generate (default: 500)
--min-anno      Slots with >= N annotations are considered common (default: 20)
--max-replace   Max slots to replace per image (default: 9)
--ar-tol        Max aspect-ratio mismatch factor (default: 1.5)
--oversize      Scale factor beyond slot size for coverage (default: 1.06)
--seed          Random seed (default: 42)
--preview-dir   If set, also saves bbox-annotated preview images here
"""

import json
import random
import argparse
from pathlib import Path
from collections import defaultdict

try:
    from PIL import Image, ImageDraw
    import numpy as np
except ImportError:
    raise ImportError("pip install Pillow numpy")


# ─── Config ───────────────────────────────────────────────────────────────────

PRODUCT_DIR  = Path("data/product_images")
YOLO_DIR     = Path("data/yolo")
METADATA     = Path("data/product_images/metadata.json")
ANNOTATIONS  = Path("data/train/annotations.json")
PREFERRED_VIEWS = ["front", "main", "left", "right"]


# ─── Helpers ──────────────────────────────────────────────────────────────────

def load_json(path: Path):
    with open(path) as f:
        return json.load(f)


def yolo_to_pixels(cx, cy, nw, nh, img_w, img_h):
    """Convert normalised YOLO box → pixel (x1, y1, x2, y2)."""
    x1 = int((cx - nw / 2) * img_w)
    y1 = int((cy - nh / 2) * img_h)
    x2 = int((cx + nw / 2) * img_w)
    y2 = int((cy + nh / 2) * img_h)
    x1, x2 = max(0, x1), min(img_w, x2)
    y1, y2 = max(0, y1), min(img_h, y2)
    return x1, y1, x2, y2


def color_transfer(src: np.ndarray, ref_region: np.ndarray) -> np.ndarray:
    """
    Match mean/std of src (H×W×3 uint8) to ref_region (H'×W'×3 uint8).
    Returns uint8 array same shape as src.
    """
    src_f   = src.astype(np.float32)
    ref_f   = ref_region.astype(np.float32)
    src_mean = src_f.mean(axis=(0, 1))
    src_std  = src_f.std(axis=(0, 1)) + 1e-6
    ref_mean = ref_f.mean(axis=(0, 1))
    ref_std  = ref_f.std(axis=(0, 1)) + 1e-6
    out = (src_f - src_mean) / src_std * ref_std + ref_mean
    return np.clip(out, 0, 255).astype(np.uint8)


def find_reference_image(product_dir: Path, prod: dict):
    """Return Path to best available reference image, or None."""
    pc   = prod["product_code"]
    view = next(
        (v for v in PREFERRED_VIEWS if v in prod["image_types"]),
        prod["image_types"][0] if prod["image_types"] else None,
    )
    if not view:
        return None
    for ext in ("jpg", "png", "jpeg"):
        p = product_dir / pc / f"{view}.{ext}"
        if p.exists():
            return p
    return None


def get_image_ar(path: Path):
    """Return (width/height) aspect ratio without fully decoding the image."""
    try:
        with Image.open(path) as im:
            w, h = im.size
        return w / max(h, 1)
    except Exception:
        return None


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--product-dir",  default=str(PRODUCT_DIR))
    parser.add_argument("--yolo-dir",     default=str(YOLO_DIR))
    parser.add_argument("--metadata",     default=str(METADATA))
    parser.add_argument("--annotations",  default=str(ANNOTATIONS))
    parser.add_argument("--n-synth",      type=int,   default=500)
    parser.add_argument("--min-anno",     type=int,   default=20)
    parser.add_argument("--max-replace",  type=int,   default=21)
    parser.add_argument("--ar-tol",       type=float, default=1.5)
    parser.add_argument("--oversize",     type=float, default=1.06)
    parser.add_argument("--seed",         type=int,   default=42)
    parser.add_argument("--preview-dir",  default=None,
                        help="Also write bbox-annotated previews here")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    product_dir  = Path(args.product_dir)
    yolo_dir     = Path(args.yolo_dir)
    out_img_dir  = yolo_dir / "images" / "train"
    out_lbl_dir  = yolo_dir / "labels" / "train"
    preview_dir  = Path(args.preview_dir) if args.preview_dir else None
    if preview_dir:
        preview_dir.mkdir(parents=True, exist_ok=True)

    # ── Load metadata and annotations ─────────────────────────────────────────
    metadata = load_json(Path(args.metadata))
    coco     = load_json(Path(args.annotations))

    ann_by_cat = defaultdict(int)
    for ann in coco["annotations"]:
        ann_by_cat[ann["category_id"]] += 1

    catid_by_name = {c["name"].upper(): c["id"] for c in coco["categories"]}
    code_to_catid = {}
    for p in metadata["products"]:
        pname = p["product_name"].upper()
        if pname in catid_by_name:
            code_to_catid[p["product_code"]] = catid_by_name[pname]

    common_catids = {cat_id for cat_id, cnt in ann_by_cat.items()
                     if cnt >= args.min_anno}

    # ── Build rare products list (has reference images, few annotations) ───────
    rare_products = []
    for p in metadata["products"]:
        if not p["has_images"]:
            continue
        pc = p["product_code"]
        if pc not in code_to_catid:
            continue
        cat_id = code_to_catid[pc]
        if ann_by_cat[cat_id] < args.min_anno:
            img_path = find_reference_image(product_dir, p)
            if img_path is None:
                continue
            ar = get_image_ar(img_path)
            if ar is None:
                continue
            rare_products.append({
                "product": p,
                "cat_id":  cat_id,
                "img_path": img_path,
                "ar":       ar,
            })

    if not rare_products:
        print("[aug] No rare products with reference images found.")
        return

    print(f"[aug] {len(rare_products)} rare products with reference images")
    print(f"[aug] {len(common_catids)} common category IDs (>= {args.min_anno} annotations)")

    # ── Background images (original shelf images only) ────────────────────────
    bg_paths = sorted(out_img_dir.glob("img_*.jpg"))
    if not bg_paths:
        raise RuntimeError("No img_*.jpg found — run prepare_data.py first")
    print(f"[aug] {len(bg_paths)} background images available")

    generated = 0
    attempt   = 0
    max_attempts = args.n_synth * 5

    while generated < args.n_synth and attempt < max_attempts:
        attempt += 1

        bg_path = random.choice(bg_paths)
        lbl_path = out_lbl_dir / (bg_path.stem + ".txt")
        if not lbl_path.exists():
            continue

        # Load background image
        try:
            bg = Image.open(bg_path).convert("RGB")
        except Exception:
            continue
        bg_w, bg_h = bg.size
        bg_arr = np.array(bg)

        # Parse YOLO labels
        raw_lines = lbl_path.read_text().strip().splitlines()
        if not raw_lines:
            continue

        parsed = []
        for line in raw_lines:
            parts = line.split()
            if len(parts) != 5:
                continue
            cls_id = int(parts[0])
            cx, cy, nw, nh = map(float, parts[1:])
            parsed.append({"cls_id": cls_id, "cx": cx, "cy": cy,
                            "nw": nw, "nh": nh, "original_line": line})

        # Find common slots (candidates for replacement)
        common_slots = [i for i, p in enumerate(parsed)
                        if p["cls_id"] in common_catids]
        if not common_slots:
            continue

        # Shuffle slots and rare products for this image
        random.shuffle(common_slots)
        rare_shuffled = random.sample(rare_products, len(rare_products))

        replaced = {}   # slot_idx → rare product entry
        used_rare = set()

        for slot_idx in common_slots[:args.max_replace]:
            slot = parsed[slot_idx]
            slot_w_px = slot["nw"] * bg_w
            slot_h_px = slot["nh"] * bg_h
            if slot_h_px < 5 or slot_w_px < 5:
                continue
            slot_ar = slot_w_px / slot_h_px

            # Find best-matching rare product by AR
            best = None
            best_ratio = float("inf")
            for ri, rp in enumerate(rare_shuffled):
                if ri in used_rare:
                    continue
                ratio = max(rp["ar"] / slot_ar, slot_ar / rp["ar"])
                if ratio < best_ratio and ratio <= args.ar_tol:
                    best_ratio = ratio
                    best = (ri, rp)

            if best is None:
                continue

            ri, rp = best
            used_rare.add(ri)
            replaced[slot_idx] = rp

        if not replaced:
            continue

        # Apply substitutions
        bg_out = bg.copy()
        out_parsed = list(parsed)  # copy, will update replaced slots

        for slot_idx, rp in replaced.items():
            slot = parsed[slot_idx]
            x1, y1, x2, y2 = yolo_to_pixels(
                slot["cx"], slot["cy"], slot["nw"], slot["nh"], bg_w, bg_h)
            sw = x2 - x1
            sh = y2 - y1
            if sw < 2 or sh < 2:
                continue

            # Load reference image
            try:
                ref_img = Image.open(rp["img_path"])
            except Exception:
                continue

            has_alpha = ref_img.mode == "RGBA"
            ref_rgba  = ref_img.convert("RGBA")

            # Target size with oversize factor
            tw = int(sw * args.oversize)
            th = int(sh * args.oversize)
            ref_rgba = ref_rgba.resize((tw, th), Image.LANCZOS)

            # Color transfer on RGB channels
            ref_rgb_arr  = np.array(ref_rgba)[:, :, :3]
            slot_region  = bg_arr[y1:y2, x1:x2]
            if slot_region.size == 0:
                continue
            ref_rgb_arr  = color_transfer(ref_rgb_arr, slot_region)

            # Rebuild RGBA with color-transferred RGB
            ref_final = Image.fromarray(
                np.concatenate([ref_rgb_arr,
                                np.array(ref_rgba)[:, :, 3:4]], axis=2),
                mode="RGBA"
            )

            # Paste centered on slot (oversize bleeds slightly outside)
            ox = x1 - (tw - sw) // 2
            oy = y1 - (th - sh) // 2

            if has_alpha:
                bg_out.paste(ref_final, (ox, oy), ref_final)
            else:
                bg_out.paste(ref_final.convert("RGB"), (ox, oy))

            # Update annotation for this slot
            out_parsed[slot_idx] = {
                **slot,
                "cls_id": rp["cat_id"],
                "original_line": (
                    f"{rp['cat_id']} {slot['cx']:.6f} {slot['cy']:.6f} "
                    f"{slot['nw']:.6f} {slot['nh']:.6f}"
                ),
            }

        # Write synthetic image + labels
        syn_name = f"synth_{generated:05d}.jpg"
        bg_out.save(out_img_dir / syn_name, quality=92)

        label_text = "\n".join(p["original_line"] for p in out_parsed)
        (out_lbl_dir / f"synth_{generated:05d}.txt").write_text(label_text)

        # Optional preview with bboxes drawn
        if preview_dir:
            draw = ImageDraw.Draw(bg_out)
            for i, p in enumerate(out_parsed):
                x1, y1, x2, y2 = yolo_to_pixels(
                    p["cx"], p["cy"], p["nw"], p["nh"], bg_w, bg_h)
                color = (255, 0, 0) if i in replaced else (0, 200, 0)
                draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
            bg_out.save(preview_dir / syn_name)

        generated += 1
        if generated % 50 == 0 or generated == args.n_synth:
            print(f"[aug]   {generated}/{args.n_synth} synthetic images generated …")

    print(f"[aug] Done — {generated} synthetic images added to {out_img_dir}")
    if preview_dir:
        print(f"[aug] Previews (red=injected, green=original) → {preview_dir}")


if __name__ == "__main__":
    main()
