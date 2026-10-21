"""Per-patch texture energy E_tex,p on the latent patch grid (Section III-C, Eq. (3)).

``E_tex,p`` is the energy of a 3x3 zero-sum spatial-rich-model (SRM) high-pass response on the
inversion-source latent z_0, averaged over latent channels and the patch's pixels. It shares the
patch grid with rho and drives the low-texture mask M(x) of Eq. (3) (``channel.rho``); its
retained-patch H0 cell mean is the E_tex of the Gaussian-location shape proxy (Section III-D).
The method is pinned by ``configs/p0.yaml`` (``e_tex.method: srm_highpass``); ``laplacian`` is
an alternative kernel that the paper does not use.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import convolve

from cfgid.channel.inversion.base import infer_patch_grid
from cfgid.numerics import as_finite_float32

# 3x3 SRM high-pass (residual) kernel, normalized by 4; it sums to zero, so flat regions -> 0.
_SRM_HIGHPASS = (
    np.array([[-1.0, 2.0, -1.0], [2.0, -4.0, 2.0], [-1.0, 2.0, -1.0]], dtype=np.float32) / 4.0
)


def _highpass(latent: np.ndarray, method: str) -> np.ndarray:
    """(C, H, W) -> (C, H, W) high-frequency response."""
    if method == "srm_highpass":
        out = np.empty_like(latent)
        for c in range(latent.shape[0]):
            out[c] = convolve(latent[c], _SRM_HIGHPASS, mode="reflect")
        return out
    if method == "laplacian":
        kern = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)
        out = np.empty_like(latent)
        for c in range(latent.shape[0]):
            out[c] = convolve(latent[c], kern, mode="reflect")
        return out
    raise ValueError(f"unknown E_tex method '{method}' (expected srm_highpass|laplacian)")


def _patch_mean_energy(energy: np.ndarray, grid: tuple[int, int]) -> np.ndarray:
    """(C, H, W) per-pixel energy -> (P,) mean energy per patch (row-major)."""
    c, h, w = energy.shape
    rows, cols = grid
    if h % rows != 0 or w % cols != 0:
        raise ValueError(f"latent {h}x{w} not divisible by grid {grid}")
    ph, pw = h // rows, w // cols
    blocks = energy.reshape(c, rows, ph, cols, pw)
    return blocks.mean(axis=(0, 2, 4)).reshape(rows * cols).astype(np.float32)


def estimate_e_tex(
    latent: np.ndarray, patch_size: int, *, method: str = "srm_highpass"
) -> np.ndarray:
    """(C, H, W) latent -> (P,) per-patch high-frequency energy, P = (H//patch)*(W//patch)."""
    x = as_finite_float32("latent", latent)
    if x.ndim != 3:
        raise ValueError(f"latent must be (C, H, W); got {x.shape}")
    grid = infer_patch_grid(x.shape[1], x.shape[2], patch_size)
    # Energy = squared high-pass response, then the mean over channels and patch pixels.
    energy = np.square(_highpass(x, method))
    return _patch_mean_energy(energy, grid)
