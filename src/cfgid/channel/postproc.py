"""Controlled post-processing channel operators: resize, crop-zoom, JPEG re-compression.

The blur sweep (``psf_blur``) varies a Gaussian PSF; these operators add other everyday
post-processing an image passes through before the analyst observes it -- a resolution
round-trip (downscale then upscale) and a center crop-and-zoom. They are the resize and crop rows
of the post-processing evaluation (Section IV-E6, Table XII); the JPEG rows there use the
single-pass ``generate.jpeg_compress``. ``jpeg_recompress`` (double JPEG) is available but is
not part of the paper's evaluation. Each is a deterministic, device-agnostic operator on the
same float32 [0, 1] (H, W, 3) image format as ``psf_blur.gaussian_psf_blur`` /
``generate.jpeg_compress`` / ``_save_png``, so they slot into the channel
``decode -> [JPEG] -> blur -> [post-processing op] -> invert`` (``sweep.run_p0``) and give
identical results on Mac MPS and Linux CUDA. The operators preserve the image SIZE (the
inversion backend expects a fixed resolution): resize and crop both round-trip back to the
original (H, W).

Resolution/crop round-trips go through an 8-bit uint8 stage (PIL) using the SAME round-to-nearest
quantization as ``_save_png`` (``np.rint``) -- the analyst sees 8-bit images, so the quantization is
part of the channel, not an artifact. (``jpeg_compress`` quantizes by truncation internally and is
reused as-is for the JPEG path rather than reimplementing its byte semantics.)
"""

from __future__ import annotations

import numpy as np
from PIL import Image

from cfgid.channel.generate import jpeg_compress
from cfgid.numerics import as_finite_float32


def _validate_image(image: np.ndarray) -> np.ndarray:
    img = as_finite_float32("image", image)
    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError(f"image must be (H, W, 3); got {img.shape}")
    if float(img.min()) < -1e-6 or float(img.max()) > 1.0 + 1e-6:
        raise ValueError("image must be in [0, 1]")
    return img


def _to_u8(img: np.ndarray) -> np.ndarray:
    return np.clip(np.rint(img * 255.0), 0.0, 255.0).astype(np.uint8)


def _resize_u8(u8: np.ndarray, size_wh: tuple[int, int], resample: Image.Resampling) -> np.ndarray:
    """PIL resize of an (H, W, 3) uint8 image to ``size_wh`` (W, H); deterministic resampling."""
    return np.asarray(Image.fromarray(u8).resize(size_wh, resample=resample), dtype=np.uint8)


def resize_roundtrip(
    image: np.ndarray, scale: float, *, resample: Image.Resampling = Image.Resampling.BICUBIC
) -> np.ndarray:
    """Downscale the image to ``scale`` of its size then upscale back to the original (H, W).

    The information lost at the smaller resolution is the channel effect (an image saved at a
    lower resolution then viewed full size). ``scale == 1.0`` is the identity. ``scale`` must be
    in (0, 1]; the downscaled side is at least 1px. Deterministic (fixed BICUBIC resampling), 8-bit.
    """
    img = _validate_image(image)
    if not (np.isfinite(scale) and 0.0 < scale <= 1.0):
        raise ValueError(f"scale must be in (0, 1], got {scale}")
    h, w = img.shape[0], img.shape[1]
    if scale == 1.0:
        return img.astype(np.float32, copy=True)
    small_w = max(1, int(round(w * scale)))
    small_h = max(1, int(round(h * scale)))
    u8 = _to_u8(img)
    small = _resize_u8(u8, (small_w, small_h), resample)
    back = _resize_u8(small, (w, h), resample)
    return as_finite_float32("resized", back.astype(np.float32) / 255.0)


def center_crop_zoom(
    image: np.ndarray, crop_frac: float, *, resample: Image.Resampling = Image.Resampling.BICUBIC
) -> np.ndarray:
    """Center-crop to ``crop_frac`` of each side, then resize the crop back to the original (H, W).

    Simulates the analyst seeing a zoomed-in central region (a crop is then displayed at full size).
    ``crop_frac == 1.0`` is the identity. ``crop_frac`` must be in (0, 1]; the crop is at least 1px.
    Deterministic, 8-bit.
    """
    img = _validate_image(image)
    if not (np.isfinite(crop_frac) and 0.0 < crop_frac <= 1.0):
        raise ValueError(f"crop_frac must be in (0, 1], got {crop_frac}")
    h, w = img.shape[0], img.shape[1]
    if crop_frac == 1.0:
        return img.astype(np.float32, copy=True)
    crop_h = max(1, int(round(h * crop_frac)))
    crop_w = max(1, int(round(w * crop_frac)))
    top = (h - crop_h) // 2
    left = (w - crop_w) // 2
    u8 = _to_u8(img)
    crop = u8[top : top + crop_h, left : left + crop_w]
    zoomed = _resize_u8(crop, (w, h), resample)
    return as_finite_float32("cropped", zoomed.astype(np.float32) / 255.0)


def jpeg_recompress(image: np.ndarray, qf1: int, qf2: int) -> np.ndarray:
    """Double JPEG compression: compress at ``qf1``, decode, compress again at ``qf2``, decode.

    Real images are often re-encoded multiple times (upload pipelines, re-saves); double-compress
    leaves a distinct trace and is a standard forensic-channel perturbation. Reuses the single-pass
    ``generate.jpeg_compress`` twice; returns the twice-decoded float32 [0, 1] image (bytes drop).
    """
    img = _validate_image(image)
    once, _ = jpeg_compress(img, qf1)
    twice, _ = jpeg_compress(once, qf2)
    return as_finite_float32("recompressed", twice)
