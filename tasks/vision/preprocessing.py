"""
tasks/vision/preprocessing.py
-------------------------------
Image preprocessing utilities for competition vision tasks.

Covers two common tracks:
  - Standard images (JPEG/PNG): classification, object detection baselines
  - Medical images (DICOM, NIfTI, numpy arrays): PET/CT segmentation tasks

All transforms are stateless functional helpers + a stateful ImagePreprocessor
class with a fit/transform interface for consistent application across train/test.

Usage:
    from tasks.vision.preprocessing import ImagePreprocessor

    prep = ImagePreprocessor(img_size=224, normalize=True)

    # From file paths:
    tensor = prep.load_and_transform("path/to/image.jpg")

    # Batch:
    batch = prep.batch_transform(["img1.jpg", "img2.jpg"])
"""

import os
import warnings
from pathlib import Path
from typing import Optional

import numpy as np

# Gracefully handle missing optional dependencies
try:
    from PIL import Image, ImageOps
    _HAS_PIL = True
except ImportError:
    _HAS_PIL = False
    warnings.warn("Pillow not installed. Install with: pip install Pillow", stacklevel=2)

try:
    import cv2
    _HAS_CV2 = True
except ImportError:
    _HAS_CV2 = False

IMG_SIZE = int(os.getenv("IMG_SIZE", "224"))
USE_MEDICAL_FORMAT = bool(int(os.getenv("USE_MEDICAL_FORMAT", "0")))

# ImageNet-style normalization constants (sensible default for transfer learning)
_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class ImagePreprocessor:
    """
    A consistent image preprocessing pipeline.

    After fitting on a list of image paths, computes dataset-level
    mean/std for normalization. Falls back to ImageNet constants if
    fit() is not called.

    Output format: float32 numpy array of shape (H, W, C) in [0, 1] range,
    optionally normalized. Use `to_tensor()` to convert to (C, H, W) for PyTorch.
    """

    def __init__(
        self,
        img_size: int = IMG_SIZE,
        normalize: bool = True,
        use_imagenet_stats: bool = True,
        grayscale: bool = False,
        augment: bool = False,
    ):
        """
        Args:
            img_size:             Target square size in pixels
            normalize:            Subtract mean and divide by std
            use_imagenet_stats:   Use ImageNet mean/std (True) or fit from data (False)
            grayscale:            Load as single-channel (useful for medical MIP images)
            augment:              Apply random train-time augmentations (flip, rotate)
        """
        self.img_size = img_size
        self.normalize = normalize
        self.use_imagenet_stats = use_imagenet_stats
        self.grayscale = grayscale
        self.augment = augment

        self._mean = _IMAGENET_MEAN if not grayscale else np.array([0.5])
        self._std  = _IMAGENET_STD  if not grayscale else np.array([0.5])

    def fit(self, image_paths: list[str], n_samples: int = 500) -> "ImagePreprocessor":
        """
        Compute dataset-level mean and std from a sample of images.
        Only used when use_imagenet_stats=False.
        """
        if self.use_imagenet_stats:
            return self

        sample = image_paths[:n_samples]
        pixels = []
        for p in sample:
            try:
                arr = self._load_array(p)
                pixels.append(arr.reshape(-1, arr.shape[-1]))
            except Exception:
                continue

        if pixels:
            all_px = np.concatenate(pixels, axis=0).astype(np.float32) / 255.0
            self._mean = all_px.mean(axis=0)
            self._std  = all_px.std(axis=0) + 1e-8
            print(f"📊 Fitted normalization: mean={self._mean.round(4)} std={self._std.round(4)}")

        return self

    def load_and_transform(self, source) -> np.ndarray:
        """
        Load an image from a file path, numpy array, or PIL Image and apply transforms.

        Args:
            source: file path (str/Path), numpy array (H,W[,C]), or PIL.Image

        Returns:
            float32 numpy array (H, W, C) — ready for model input
        """
        arr = self._to_array(source)
        arr = self._resize(arr)

        if self.augment:
            arr = self._apply_augmentation(arr)

        arr = arr.astype(np.float32) / 255.0

        if self.normalize:
            # Broadcast over spatial dims
            arr = (arr - self._mean) / (self._std + 1e-8)

        return arr

    def batch_transform(self, sources: list) -> np.ndarray:
        """
        Transform a list of image sources into a stacked batch.

        Returns:
            np.ndarray of shape (N, H, W, C)
        """
        arrays = [self.load_and_transform(s) for s in sources]
        return np.stack(arrays, axis=0)

    def to_tensor(self, arr: np.ndarray) -> np.ndarray:
        """
        Convert (H, W, C) → (C, H, W) format expected by PyTorch.
        Works on batches too: (N, H, W, C) → (N, C, H, W).
        """
        if arr.ndim == 3:
            return arr.transpose(2, 0, 1)
        elif arr.ndim == 4:
            return arr.transpose(0, 3, 1, 2)
        raise ValueError(f"Expected 3 or 4 dims, got {arr.ndim}")

    # ------------------------------------------------------------------
    # Medical imaging helpers
    # ------------------------------------------------------------------

    @staticmethod
    def load_dicom(path: str) -> np.ndarray:
        """
        Load a DICOM file and return a normalized float32 array.
        Requires: pip install pydicom
        """
        try:
            import pydicom
        except ImportError:
            raise ImportError("Install pydicom: pip install pydicom")
        ds = pydicom.dcmread(path)
        arr = ds.pixel_array.astype(np.float32)
        # Normalize to [0, 1] using the image's own range
        arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)
        return arr

    @staticmethod
    def load_nifti(path: str) -> np.ndarray:
        """
        Load a NIfTI file and return a float32 volume.
        Requires: pip install nibabel
        """
        try:
            import nibabel as nib
        except ImportError:
            raise ImportError("Install nibabel: pip install nibabel")
        img = nib.load(path)
        arr = img.get_fdata().astype(np.float32)
        arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)
        return arr

    @staticmethod
    def mip_projection(volume: np.ndarray, axis: int = 2) -> np.ndarray:
        """
        Maximum Intensity Projection — used in PET imaging segmentation tasks.
        Collapses a 3D volume to a 2D image along the given axis.

        Args:
            volume: 3D float array (X, Y, Z)
            axis:   Axis to project along (default=2, axial)

        Returns:
            2D float array
        """
        return volume.max(axis=axis)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _to_array(self, source) -> np.ndarray:
        if isinstance(source, np.ndarray):
            return self._ensure_rgb(source)
        if _HAS_PIL and isinstance(source, Image.Image):
            return np.array(source.convert("L" if self.grayscale else "RGB"))
        # Treat as file path
        return self._load_array(str(source))

    def _load_array(self, path: str) -> np.ndarray:
        if not _HAS_PIL and not _HAS_CV2:
            raise RuntimeError("Install Pillow or OpenCV to load images.")
        if _HAS_PIL:
            mode = "L" if self.grayscale else "RGB"
            with Image.open(path) as img:
                arr = np.array(img.convert(mode))
        else:
            arr = cv2.imread(path, cv2.IMREAD_GRAYSCALE if self.grayscale else cv2.IMREAD_COLOR)
            if not self.grayscale:
                arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
        return arr

    def _resize(self, arr: np.ndarray) -> np.ndarray:
        h, w = self.img_size, self.img_size
        if _HAS_CV2:
            return cv2.resize(arr, (w, h), interpolation=cv2.INTER_AREA)
        if _HAS_PIL:
            pil = Image.fromarray(arr)
            pil = pil.resize((w, h), Image.LANCZOS)
            return np.array(pil)
        # Fallback: basic numpy resize (low quality, last resort)
        from scipy.ndimage import zoom
        factors = (h / arr.shape[0], w / arr.shape[1])
        if arr.ndim == 3:
            factors = (*factors, 1)
        return zoom(arr, factors).astype(arr.dtype)

    def _ensure_rgb(self, arr: np.ndarray) -> np.ndarray:
        if self.grayscale:
            if arr.ndim == 3:
                return arr[:, :, 0]
            return arr
        if arr.ndim == 2:
            return np.stack([arr, arr, arr], axis=-1)
        if arr.shape[-1] == 4:  # RGBA → RGB
            return arr[:, :, :3]
        return arr

    def _apply_augmentation(self, arr: np.ndarray) -> np.ndarray:
        """Lightweight stochastic augmentations (applied randomly at training time)."""
        rng = np.random.default_rng()
        # Horizontal flip
        if rng.random() > 0.5:
            arr = np.fliplr(arr)
        # Vertical flip
        if rng.random() > 0.5:
            arr = np.flipud(arr)
        # Brightness jitter ±10%
        factor = rng.uniform(0.9, 1.1)
        arr = np.clip(arr * factor, 0, 255).astype(arr.dtype)
        return arr


# ---------------------------------------------------------------------------
# Quick smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import tempfile

    print("Testing ImagePreprocessor with a synthetic image array...")
    dummy = (np.random.rand(300, 400, 3) * 255).astype(np.uint8)

    prep = ImagePreprocessor(img_size=224, normalize=True)
    out = prep.load_and_transform(dummy)
    print(f"Input shape:  {dummy.shape}")
    print(f"Output shape: {out.shape}   dtype={out.dtype}")
    print(f"Value range:  [{out.min():.3f}, {out.max():.3f}]")

    tensor = prep.to_tensor(out)
    print(f"Tensor shape: {tensor.shape}  (C, H, W)")

    print("Testing MIP projection...")
    volume = np.random.rand(64, 64, 64).astype(np.float32)
    mip = ImagePreprocessor.mip_projection(volume, axis=2)
    print(f"Volume shape: {volume.shape} → MIP shape: {mip.shape}")
    print("✅ Preprocessing smoke test passed.")
