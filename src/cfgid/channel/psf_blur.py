"""Controlled post-decode blur: a known-width Gaussian PSF applied to the forensic channel.

The main grid applies a Gaussian of known width ``sigma_px`` = sigma_blur in {0, 8, 16, 32}
decoded-image pixels (``configs/p0.yaml`` ``grid.sigma_blur``; Section IV-A) to the decoded
image as part of the channel the forensic analyst observes (decode -> [JPEG] -> blur ->
[post-processing op] -> invert; the order used by ``sweep.run_p0``). The blur widens the
channel's effective PSF, so sweeping ``sigma_px`` sweeps the effective width on a SINGLE decoder
without changing the model; cells are summarized by
sigma_eff = sqrt(sigma_core^2 + sigma_blur^2) (``channel.sigma_calib.sigma_blur_sweep``).
The blur response is reported in Section IV-E5 (Fig. 5) and Table V.

Deterministic (fixed Gaussian kernel, no randomness) and device-agnostic (pure numpy/scipy
post-processing), so the same code produces identical results on Mac MPS and Linux CUDA hosts.
Operates on the same float32 [0, 1] (H, W, 3) image format as ``channel.generate.jpeg_compress`` and
``sweep.run_p0._save_png``.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter

from cfgid.numerics import as_finite_float32


def gaussian_psf_blur(image: np.ndarray, sigma_px: float) -> np.ndarray:
    """Apply an isotropic Gaussian PSF of width ``sigma_px`` (pixels) to an RGB image in [0, 1].

    ``sigma_px == 0`` is the identity (returns a float32 copy). Blurring is spatial-only and
    per-channel (no mixing across RGB). The result is clipped to [0, 1] and finite-guarded — a
    normalized Gaussian of a [0, 1] image already stays in range, the clip is defensive.
    """
    img = as_finite_float32("image", image)
    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError(f"image must be (H, W, 3); got {img.shape}")
    if float(img.min()) < -1e-6 or float(img.max()) > 1.0 + 1e-6:
        raise ValueError("image must be in [0, 1]")
    if not (np.isfinite(sigma_px) and sigma_px >= 0.0):
        raise ValueError(f"sigma_px must be finite and non-negative, got {sigma_px}")
    if sigma_px == 0.0:
        return img.astype(np.float32, copy=True)
    # sigma=(sigma_px, sigma_px, 0): blur the two spatial axes, never across the channel axis.
    blurred = gaussian_filter(
        img.astype(np.float64), sigma=(float(sigma_px), float(sigma_px), 0.0), mode="reflect"
    )
    blurred = np.clip(blurred, 0.0, 1.0)
    return as_finite_float32("blurred", blurred)
