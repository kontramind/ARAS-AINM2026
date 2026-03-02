"""
tests/test_vision.py
---------------------
Unit tests for the vision task module (preprocessing + segmentation).

These tests use only numpy arrays so they run without PIL, OpenCV, or torch.
Strategy: test the logic (shape, dtype, value range) rather than visual quality.

Run:
    pytest tests/test_vision.py -v
"""

import numpy as np
import pytest

from tasks.vision.preprocessing import ImagePreprocessor
from tasks.vision.segmentation import SegmentationPipeline


# ===========================================================================
# Fixtures
# ===========================================================================

@pytest.fixture
def rgb_image():
    """Synthetic 300x400 RGB uint8 image."""
    rng = np.random.default_rng(0)
    return (rng.random((300, 400, 3)) * 255).astype(np.uint8)


@pytest.fixture
def gray_image():
    """Synthetic 256x256 grayscale 2D array."""
    rng = np.random.default_rng(1)
    return (rng.random((256, 256)) * 255).astype(np.uint8)


@pytest.fixture
def pet_scan():
    """Synthetic PET-like image: mostly dark with a bright region."""
    scan = np.zeros((256, 256), dtype=np.float32)
    scan[80:160, 80:160] = 0.8 + np.random.default_rng(2).random((80, 80)) * 0.2
    return scan


@pytest.fixture
def tumor_mask():
    """Ground-truth binary mask matching the bright region in pet_scan."""
    mask = np.zeros((256, 256), dtype=np.uint8)
    mask[80:160, 80:160] = 1
    return mask


# ===========================================================================
# ImagePreprocessor
# ===========================================================================

def test_output_shape_unchanged(rgb_image):
    """ImagePreprocessor should output (img_size, img_size, channels)."""
    prep = ImagePreprocessor(img_size=224, normalize=False)
    out = prep.load_and_transform(rgb_image)
    assert out.shape == (224, 224, 3)


def test_output_dtype_float32(rgb_image):
    prep = ImagePreprocessor(img_size=128)
    out = prep.load_and_transform(rgb_image)
    assert out.dtype == np.float32


def test_output_range_0_1_without_normalization(rgb_image):
    prep = ImagePreprocessor(img_size=64, normalize=False)
    out = prep.load_and_transform(rgb_image)
    assert out.min() >= 0.0
    assert out.max() <= 1.0 + 1e-6


def test_normalization_shifts_mean(rgb_image):
    """After ImageNet normalization, the mean should be close to 0."""
    prep = ImagePreprocessor(img_size=224, normalize=True, use_imagenet_stats=True)
    out = prep.load_and_transform(rgb_image)
    # Not rigorous, just check the range is wider than [0,1] (normalization happened)
    assert out.min() < 0.0 or out.max() > 1.0


def test_to_tensor_converts_hwc_to_chw(rgb_image):
    prep = ImagePreprocessor(img_size=64, normalize=False)
    arr = prep.load_and_transform(rgb_image)   # (64, 64, 3)
    tensor = prep.to_tensor(arr)               # (3, 64, 64)
    assert tensor.shape == (3, 64, 64)


def test_to_tensor_batch(rgb_image):
    prep = ImagePreprocessor(img_size=32, normalize=False)
    batch = prep.batch_transform([rgb_image, rgb_image])  # (2, 32, 32, 3)
    assert batch.shape == (2, 32, 32, 3)
    tensors = prep.to_tensor(batch)                        # (2, 3, 32, 32)
    assert tensors.shape == (2, 3, 32, 32)


def test_grayscale_output_has_single_channel(gray_image):
    prep = ImagePreprocessor(img_size=64, grayscale=True, normalize=False)
    out = prep.load_and_transform(gray_image)
    # Grayscale: either (64, 64) or (64, 64, 1) is acceptable
    assert out.ndim in (2, 3)
    if out.ndim == 3:
        assert out.shape[-1] == 1


def test_mip_projection_reduces_dimension():
    volume = np.random.rand(64, 64, 64).astype(np.float32)
    mip = ImagePreprocessor.mip_projection(volume, axis=2)
    assert mip.shape == (64, 64)
    # MIP should take the max along the axis
    assert np.allclose(mip, volume.max(axis=2))


def test_augmentation_does_not_change_shape(rgb_image):
    prep = ImagePreprocessor(img_size=64, augment=True, normalize=False)
    out = prep.load_and_transform(rgb_image)
    assert out.shape == (64, 64, 3)


# ===========================================================================
# SegmentationPipeline — Otsu backend (no dependencies)
# ===========================================================================

def test_otsu_output_is_binary(pet_scan):
    pipeline = SegmentationPipeline(backend="otsu")
    mask = pipeline.predict_single(pet_scan)
    unique_values = np.unique(mask)
    assert set(unique_values).issubset({0, 1}), f"Expected binary mask, got {unique_values}"


def test_otsu_output_shape_matches_input(pet_scan):
    pipeline = SegmentationPipeline(backend="otsu")
    mask = pipeline.predict_single(pet_scan)
    assert mask.shape == pet_scan.shape


def test_otsu_detects_bright_region(pet_scan, tumor_mask):
    """Otsu should correctly identify the bright tumor region."""
    pipeline = SegmentationPipeline(backend="otsu")
    mask = pipeline.predict_single(pet_scan)
    dice = SegmentationPipeline.dice_score(mask, tumor_mask)
    # Otsu on a clean synthetic image should achieve very high Dice
    assert dice > 0.80, f"Dice score too low: {dice:.4f}"


def test_otsu_on_rgb_image(rgb_image):
    """Otsu should handle RGB input gracefully."""
    pipeline = SegmentationPipeline(backend="otsu")
    arr = rgb_image.astype(np.float32) / 255.0
    mask = pipeline.predict_single(arr)
    assert mask.shape == (arr.shape[0], arr.shape[1])


def test_batch_predict_returns_list(pet_scan):
    pipeline = SegmentationPipeline(backend="otsu")
    masks = pipeline.predict_batch([pet_scan, pet_scan, pet_scan])
    assert len(masks) == 3
    for m in masks:
        assert m.shape == pet_scan.shape


# ===========================================================================
# Metrics
# ===========================================================================

def test_dice_score_perfect(tumor_mask):
    score = SegmentationPipeline.dice_score(tumor_mask, tumor_mask)
    assert abs(score - 1.0) < 1e-5


def test_dice_score_zero():
    """Non-overlapping masks should have Dice ≈ 0."""
    a = np.zeros((64, 64), dtype=np.uint8)
    b = np.zeros((64, 64), dtype=np.uint8)
    a[:32, :32] = 1
    b[32:, 32:] = 1
    score = SegmentationPipeline.dice_score(a, b)
    assert score < 0.01


def test_iou_score_perfect(tumor_mask):
    score = SegmentationPipeline.iou_score(tumor_mask, tumor_mask)
    assert abs(score - 1.0) < 1e-5


def test_evaluate_batch_metrics(pet_scan, tumor_mask):
    pipeline = SegmentationPipeline(backend="otsu")
    mask = pipeline.predict_single(pet_scan)
    metrics = pipeline.evaluate([mask], [tumor_mask])
    assert "dice_mean" in metrics
    assert "iou_mean" in metrics
    assert 0.0 <= metrics["dice_mean"] <= 1.0


# ===========================================================================
# Graceful degradation — unsupported backends
# ===========================================================================

def test_torchvision_backend_falls_back_when_torch_missing(monkeypatch):
    import tasks.vision.segmentation as seg_module
    monkeypatch.setattr(seg_module, "_HAS_TORCH", False)
    with pytest.warns(UserWarning, match="torchvision"):
        pipeline = SegmentationPipeline(backend="torchvision")
    assert pipeline.backend == "otsu"


def test_smp_backend_falls_back_when_smp_missing(monkeypatch):
    import tasks.vision.segmentation as seg_module
    monkeypatch.setattr(seg_module, "_HAS_SMP", False)
    with pytest.warns(UserWarning, match="segmentation_models_pytorch"):
        pipeline = SegmentationPipeline(backend="smp")
    assert pipeline.backend == "otsu"
