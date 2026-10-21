"""Image-only nuisance baselines (negative controls for the gate).

Each maps an RGB image in [0, 1] (H, W, 3) to a scalar; the gate computes each one's AUC over H0/H1
and checks that rho is not weaker than the best of them (after texture/semantic matching). If a
cheap nuisance separates H0/H1 at least as well as rho, the guidance evidence cannot be credited
to the rho channel, so the gate returns inconclusive. CLIP-text-alignment is added by the driver
(needs a model). The per-baseline AUCs are listed under "Negative controls" in
``outputs/p0/decision_memo.md`` (Table VIII, rows 1-2). AUCs are orientation-agnostic
(max(AUC, 1 - AUC)) because a nuisance may move in either direction with guidance.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from cfgid.detector.empirical import auc
from cfgid.numerics import as_finite_float32

_SRM = np.array([[-1.0, 2.0, -1.0], [2.0, -4.0, 2.0], [-1.0, 2.0, -1.0]], dtype=np.float64) / 4.0


def _gray(img: np.ndarray) -> np.ndarray:
    return img.mean(axis=2).astype(np.float64)


def _radial_fraction(img: np.ndarray, *, lo: float, hi: float) -> float:
    # Fraction of 2-D FFT power whose radius lies in [lo, hi) x the maximum radius.
    g = _gray(img)
    power = np.square(np.abs(np.fft.fftshift(np.fft.fft2(g))))
    h, w = g.shape
    yy, xx = np.ogrid[:h, :w]
    r = np.sqrt((yy - h / 2.0) ** 2 + (xx - w / 2.0) ** 2)
    rmax = float(r.max()) or 1.0
    band = power[(r >= lo * rmax) & (r < hi * rmax)].sum()
    return float(band / (power.sum() + 1e-12))


def fft_high_band(img: np.ndarray) -> float:
    return _radial_fraction(img, lo=0.5, hi=1.0001)


def low_frequency(img: np.ndarray) -> float:
    return _radial_fraction(img, lo=0.0, hi=0.25)


def saturation_mean(img: np.ndarray) -> float:
    mx = img.max(axis=2)
    mn = img.min(axis=2)
    return float(np.mean((mx - mn) / (mx + 1e-6)))


def color_dispersion(img: np.ndarray) -> float:
    return float(np.std(img.reshape(-1, 3).mean(axis=0)))


def srm_energy(img: np.ndarray) -> float:
    from scipy.ndimage import convolve

    return float(np.mean(np.square(convolve(_gray(img), _SRM, mode="reflect"))))


NUISANCES: dict[str, Callable[[np.ndarray], float]] = {
    "fft_high": fft_high_band,
    "low_freq": low_frequency,
    "saturation": saturation_mean,
    "color": color_dispersion,
    "srm": srm_energy,
}


def nuisance_features(image: np.ndarray) -> dict[str, float]:
    """RGB image (H, W, 3) in [0, 1] -> one scalar per nuisance baseline."""
    img = as_finite_float32("image", image)
    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError(f"image must be (H, W, 3); got {img.shape}")
    return {name: fn(img) for name, fn in NUISANCES.items()}


def best_nuisance_auc(
    features_by_name: dict[str, np.ndarray], labels: np.ndarray
) -> tuple[str, float]:
    """Return (name, AUC) of the strongest nuisance baseline (orientation-agnostic AUC)."""
    if not features_by_name:
        raise ValueError("features_by_name is empty")
    best_name, best_auc = "", 0.0
    for name, scores in features_by_name.items():
        a = auc(scores, labels)
        a = max(a, 1.0 - a)
        if a > best_auc:
            best_name, best_auc = name, a
    return best_name, best_auc
