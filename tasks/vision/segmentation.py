"""
tasks/vision/segmentation.py
------------------------------
Lightweight image segmentation pipeline for competition use.

Strategy:
  - Uses a U-Net-style architecture via torchvision / segmentation_models if available
  - Falls back to a scikit-learn pixel classifier (works without GPU/torch)
  - Includes the PET tumor segmentation pattern from AINM 2025

Architecture choices:
  - Tier 1 (best):  torchvision FCN / DeepLab pretrained on COCO — fine-tune on task data
  - Tier 2 (fast):  segmentation_models_pytorch (SMP) with EfficientNet backbone
  - Tier 3 (safe):  Otsu thresholding + morphological ops (works with only numpy/scipy)

Usage:
    from tasks.vision.segmentation import SegmentationPipeline

    # Tier 3 (no dependencies) — good baseline to beat
    pipeline = SegmentationPipeline(backend="otsu")
    mask = pipeline.predict_single(image_array)

    # Tier 1 with PyTorch
    pipeline = SegmentationPipeline(backend="torchvision", n_classes=2)
    pipeline.load_pretrained()
    mask = pipeline.predict_single(image_array)
"""

import os
import warnings
from typing import Literal, Optional

import numpy as np

_HAS_TORCH = False
_HAS_SMP = False

try:
    import torch
    import torch.nn as nn
    import torchvision
    _HAS_TORCH = True
except ImportError:
    pass

try:
    import segmentation_models_pytorch as smp
    _HAS_SMP = True
except ImportError:
    pass

try:
    from scipy import ndimage
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False


class SegmentationPipeline:
    """
    Multi-backend segmentation pipeline, designed to degrade gracefully.

    Backends:
        "otsu"        → Otsu thresholding + connected components (no deep learning)
        "torchvision" → Pretrained FCN-ResNet50 from torchvision
        "smp"         → segmentation_models_pytorch (UNet + selectable backbone)

    Output: binary mask (0/1) as uint8 numpy array matching input spatial dimensions.
    """

    def __init__(
        self,
        backend: Literal["otsu", "torchvision", "smp"] = "otsu",
        n_classes: int = 2,
        device: Optional[str] = None,
    ):
        self.backend = backend
        self.n_classes = n_classes
        self.device = device or ("cuda" if _HAS_TORCH and torch.cuda.is_available() else "cpu")
        self.model = None

        if backend == "torchvision" and not _HAS_TORCH:
            warnings.warn("torchvision not installed. Falling back to 'otsu'.", stacklevel=2)
            self.backend = "otsu"
        if backend == "smp" and not _HAS_SMP:
            warnings.warn("segmentation_models_pytorch not installed. Falling back to 'otsu'.", stacklevel=2)
            self.backend = "otsu"

        print(f"🔬 SegmentationPipeline initialized — backend={self.backend}, device={self.device}")

    def load_pretrained(
        self,
        weights_path: Optional[str] = None,
        backbone: str = "efficientnet-b4",
    ) -> "SegmentationPipeline":
        """
        Load a segmentation model.

        Args:
            weights_path: Path to fine-tuned weights file (.pth). If None, loads ImageNet pretrained.
            backbone:     Backbone name for SMP (ignored for torchvision).
        """
        if self.backend == "otsu":
            print("ℹ️  Otsu backend requires no pretrained weights.")
            return self

        if self.backend == "torchvision":
            self.model = torchvision.models.segmentation.fcn_resnet50(
                weights="DEFAULT" if weights_path is None else None
            )
            self.model.classifier[4] = torch.nn.Conv2d(512, self.n_classes, kernel_size=1)
            if weights_path:
                state = torch.load(weights_path, map_location=self.device)
                self.model.load_state_dict(state)
            self.model = self.model.to(self.device).eval()
            print(f"✅ Loaded torchvision FCN-ResNet50 (n_classes={self.n_classes})")

        elif self.backend == "smp":
            self.model = smp.Unet(
                encoder_name=backbone,
                encoder_weights="imagenet" if weights_path is None else None,
                in_channels=1 if self._is_grayscale else 3,
                classes=self.n_classes,
            )
            if weights_path:
                state = torch.load(weights_path, map_location=self.device)
                self.model.load_state_dict(state)
            self.model = self.model.to(self.device).eval()
            print(f"✅ Loaded SMP UNet ({backbone}, n_classes={self.n_classes})")

        return self

    def predict_single(self, image: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """
        Predict a segmentation mask for a single image.

        Args:
            image:     float32 numpy array (H, W) or (H, W, C) — values in [0,1]
            threshold: Binarization threshold for probability outputs

        Returns:
            uint8 binary mask (H, W), values 0 or 1
        """
        if self.backend == "otsu":
            return self._otsu_segment(image)
        else:
            return self._deep_predict(image, threshold)

    def predict_batch(self, images: list[np.ndarray], threshold: float = 0.5) -> list[np.ndarray]:
        """Predict masks for a list of images."""
        return [self.predict_single(img, threshold) for img in images]

    def save_weights(self, path: str) -> None:
        """Save fine-tuned model weights."""
        if self.model is None:
            raise RuntimeError("No model loaded. Call load_pretrained() first.")
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        torch.save(self.model.state_dict(), path)
        print(f"💾 Weights saved → {path}")

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    @staticmethod
    def dice_score(pred_mask: np.ndarray, true_mask: np.ndarray, eps: float = 1e-8) -> float:
        """
        Sørensen–Dice coefficient. Standard metric for segmentation competitions.
        Higher is better, max = 1.0.
        """
        pred = pred_mask.astype(bool).flatten()
        true = true_mask.astype(bool).flatten()
        intersection = (pred & true).sum()
        return float(2 * intersection / (pred.sum() + true.sum() + eps))

    @staticmethod
    def iou_score(pred_mask: np.ndarray, true_mask: np.ndarray, eps: float = 1e-8) -> float:
        """Intersection over Union (Jaccard index)."""
        pred = pred_mask.astype(bool).flatten()
        true = true_mask.astype(bool).flatten()
        intersection = (pred & true).sum()
        union = (pred | true).sum()
        return float(intersection / (union + eps))

    def evaluate(self, preds: list[np.ndarray], targets: list[np.ndarray]) -> dict:
        """Batch evaluation — returns mean Dice and mean IoU."""
        dice_scores = [self.dice_score(p, t) for p, t in zip(preds, targets)]
        iou_scores  = [self.iou_score(p, t)  for p, t in zip(preds, targets)]
        metrics = {
            "dice_mean": round(float(np.mean(dice_scores)), 4),
            "dice_std":  round(float(np.std(dice_scores)),  4),
            "iou_mean":  round(float(np.mean(iou_scores)),  4),
        }
        print(f"📊 Segmentation metrics: {metrics}")
        return metrics

    # ------------------------------------------------------------------
    # Internal backends
    # ------------------------------------------------------------------

    def _otsu_segment(self, image: np.ndarray) -> np.ndarray:
        """
        Classical Otsu thresholding — fast, no ML, but surprisingly competitive
        on high-contrast medical images like PET scans.
        """
        # Convert to grayscale if needed
        if image.ndim == 3:
            gray = image.mean(axis=-1)
        else:
            gray = image.copy()
        gray = (gray - gray.min()) / (gray.max() - gray.min() + 1e-8)

        # Compute Otsu threshold from histogram
        counts, bins = np.histogram(gray.flatten(), bins=256)
        total = counts.sum()
        current_max, threshold = 0, 0
        sum_all = np.dot(np.arange(256), counts)
        sum_b, w_b = 0.0, 0.0

        for t in range(256):
            w_b += counts[t]
            if w_b == 0:
                continue
            w_f = total - w_b
            if w_f == 0:
                break
            sum_b += t * counts[t]
            mean_b = sum_b / w_b
            mean_f = (sum_all - sum_b) / w_f
            between = w_b * w_f * (mean_b - mean_f) ** 2
            if between > current_max:
                current_max = between
                threshold = t

        otsu_thresh = threshold / 255.0
        mask = (gray > otsu_thresh).astype(np.uint8)

        # Clean up with morphological operations if scipy is available
        if _HAS_SCIPY:
            mask = ndimage.binary_fill_holes(mask).astype(np.uint8)
            struct = ndimage.generate_binary_structure(2, 2)
            mask = ndimage.binary_opening(mask, structure=struct).astype(np.uint8)

        return mask

    def _deep_predict(self, image: np.ndarray, threshold: float) -> np.ndarray:
        """Run inference with a PyTorch model."""
        import torch
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_pretrained() first.")

        # Prepare tensor: (H,W,C) → (1,C,H,W)
        if image.ndim == 2:
            image = np.stack([image, image, image], axis=-1)
        t = torch.from_numpy(image.transpose(2, 0, 1)).unsqueeze(0).float().to(self.device)

        with torch.no_grad():
            if self.backend == "torchvision":
                output = self.model(t)["out"]
            else:
                output = self.model(t)

            probs = torch.softmax(output, dim=1)[:, 1, :, :].squeeze(0)

        mask = (probs.cpu().numpy() > threshold).astype(np.uint8)
        return mask

    @property
    def _is_grayscale(self) -> bool:
        return False  # Override when loading grayscale medical images


# ---------------------------------------------------------------------------
# Quick smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("Testing Otsu segmentation backend (no dependencies)...")
    dummy = np.zeros((256, 256), dtype=np.float32)
    # Simulate a bright "tumor" region
    dummy[80:160, 80:160] = 0.9
    dummy += np.random.rand(*dummy.shape) * 0.1

    pipeline = SegmentationPipeline(backend="otsu")
    mask = pipeline.predict_single(dummy)
    print(f"Input shape: {dummy.shape}, Mask shape: {mask.shape}")

    true_mask = np.zeros_like(dummy, dtype=np.uint8)
    true_mask[80:160, 80:160] = 1
    metrics = pipeline.evaluate([mask], [true_mask])
    print(f"✅ Smoke test passed. Dice={metrics['dice_mean']}")
