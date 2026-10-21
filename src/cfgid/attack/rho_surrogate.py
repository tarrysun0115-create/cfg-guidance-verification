"""Differentiable linear surrogate of the forensic channel rho (for the purification attack).

The channel rho is computed by dual-branch inversion and is NOT differentiable here (no
backpropagation through the diffusion/inversion chain). To move an attacked image's rho toward the
cell's H0 (null) mean with minimal L2 distortion, the attack of Section IV-E7 needs a cheap
DIFFERENTIABLE proxy ``s(image) ~ rho`` to optimize.

This module provides exact TORCH twins of the six physical image statistics of
``scripts/analysis/rho_mechanism.py`` (saturation, contrast, high-frequency energy, colorfulness,
blockiness, mean luminance: the physical battery of Section IV-E1) and fits the least-squares
LINEAR combination of them to the measured raw rho on a fitting pool (the attack script uses each
cell's TEST rows). The attack raises this proxy, then a SEPARATE forward inversion measures the
real rho effect.

The torch twins match ``rho_mechanism.image_features`` numerically (ddof=0 std/var, the same luma
weights, saturation/colorfulness/blockiness formulas) so the proxy is faithful to those features.
Two feature sets: ``global_v1`` (the six global statistics; the stored Section IV-E7 run) and
``patch_v2`` (those plus patch-aligned multi-scale band energies). No diffusion/inversion code is
imported here; these are plain differentiable image statistics.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

# Order is fixed and matches rho_mechanism.FEATURE_NAMES so a fitted surrogate lines up with it.
FEATURE_NAMES: tuple[str, ...] = (
    "saturation",
    "contrast",
    "highfreq",
    "colorfulness",
    "blockiness",
    "mean_luma",
)

_LUMA = (0.299, 0.587, 0.114)


def _validate_bchw(x: torch.Tensor) -> torch.Tensor:
    if x.ndim != 4 or x.shape[1] != 3:
        raise ValueError(f"x must be (B,3,H,W) RGB in [0,1], got shape {tuple(x.shape)}")
    if not torch.is_floating_point(x):
        raise ValueError("x must be a floating point tensor")
    if min(int(x.shape[2]), int(x.shape[3])) < 1:
        raise ValueError(f"x must have non-empty spatial dims, got shape {tuple(x.shape)}")
    return x


def _luma(x: torch.Tensor) -> torch.Tensor:
    """(B,3,H,W) -> (B,H,W) ITU-R BT.601 luminance, matching rho_mechanism._luma."""
    return _LUMA[0] * x[:, 0] + _LUMA[1] * x[:, 1] + _LUMA[2] * x[:, 2]


def physical_features_torch(x: torch.Tensor) -> dict[str, torch.Tensor]:
    """Differentiable torch twins of rho_mechanism.image_features; (B,3,H,W) in [0,1] -> {name:(B,)}

    Every value uses population (ddof=0) std/var to match numpy's defaults, so each twin agrees with
    ``image_features`` to floating-point tolerance on the same image.
    """
    x = _validate_bchw(x)
    b = x.shape[0]
    h, w = int(x.shape[2]), int(x.shape[3])
    y = _luma(x)  # (B,H,W)
    y_flat = y.reshape(b, -1)

    # saturation: mean over pixels of (max-min)/max across channels (0 where max ~ 0)
    mx = x.amax(dim=1)  # (B,H,W)
    mn = x.amin(dim=1)
    sat = torch.where(mx > 1e-8, (mx - mn) / mx.clamp_min(1e-8), torch.zeros_like(mx))
    saturation = sat.reshape(b, -1).mean(dim=1)

    # contrast: population std of luma
    contrast = y_flat.std(dim=1, unbiased=False)

    # highfreq: population variance of the interior discrete Laplacian (roll wraparound sliced off)
    if min(h, w) > 2:
        lap = (
            -4.0 * y
            + torch.roll(y, 1, 1)
            + torch.roll(y, -1, 1)
            + torch.roll(y, 1, 2)
            + torch.roll(y, -1, 2)
        )
        highfreq = lap[:, 1:-1, 1:-1].reshape(b, -1).var(dim=1, unbiased=False)
    else:
        highfreq = x.new_zeros(b)

    # colorfulness: Hasler-Susstrunk on opponent channels rg = R-G, yb = 0.5(R+G)-B.
    # A tiny eps inside the sqrt bounds the gradient at a degenerate (perfectly flat / grey) image
    # where both std and mean are 0 (sqrt has an infinite slope at 0); real images never hit this,
    # and eps=1e-12 perturbs the VALUE by <=1e-6, far below the proxy's standardized scale.
    _eps = 1.0e-12
    rg = (x[:, 0] - x[:, 1]).reshape(b, -1)
    yb = (0.5 * (x[:, 0] + x[:, 1]) - x[:, 2]).reshape(b, -1)
    colorfulness = torch.sqrt(
        rg.std(dim=1, unbiased=False) ** 2 + yb.std(dim=1, unbiased=False) ** 2 + _eps
    ) + 0.3 * torch.sqrt(rg.mean(dim=1) ** 2 + yb.mean(dim=1) ** 2 + _eps)

    # blockiness: 8-px-grid edge energy minus mean edge energy (JPEG block signature)
    if w > 1:
        bh = (y[:, :, 1:] - y[:, :, :-1]).abs()  # (B,H,W-1)
        bh_mean = bh.reshape(b, -1).mean(dim=1)
        bh_block = bh[:, :, 7::8].reshape(b, -1).mean(dim=1) if bh.shape[2] > 7 else x.new_zeros(b)
    else:
        bh_mean = x.new_zeros(b)
        bh_block = x.new_zeros(b)
    if h > 1:
        bv = (y[:, 1:, :] - y[:, :-1, :]).abs()  # (B,H-1,W)
        bv_mean = bv.reshape(b, -1).mean(dim=1)
        bv_block = bv[:, 7::8, :].reshape(b, -1).mean(dim=1) if bv.shape[1] > 7 else x.new_zeros(b)
    else:
        bv_mean = x.new_zeros(b)
        bv_block = x.new_zeros(b)
    blockiness = (bh_block - bh_mean) + (bv_block - bv_mean)

    # mean_luma
    mean_luma = y_flat.mean(dim=1)

    return {
        "saturation": saturation,
        "contrast": contrast,
        "highfreq": highfreq,
        "colorfulness": colorfulness,
        "blockiness": blockiness,
        "mean_luma": mean_luma,
    }


def feature_stack_torch(x: torch.Tensor, names: tuple[str, ...] = FEATURE_NAMES) -> torch.Tensor:
    """Stack the named torch features into (B, P) in ``names`` order."""
    feats = physical_features_torch(x)
    return torch.stack([feats[n] for n in names], dim=1)


# ``patch_v2`` surrogate: PATCH-ALIGNED MULTI-SCALE band-pass energies.
#
# rho keeps guidance signal after residualizing out the six GLOBAL physical features
# (Section IV-E1), so a global linear proxy of rho is weak by construction. rho is a dual-branch
# inversion-RESIDUAL ratio, a winsorized mean over HIGH-E_tex PATCHES, so this surrogate mirrors
# that structure: split the image into a patch grid, measure band-pass energy at several scales
# per patch, and aggregate each band over the high-E_tex patches (soft-weighted by the finest-band
# / high-pass energy, the same texture concentration rho uses). Still a FIXED differentiable
# extractor (Gaussian-pyramid difference filters + a linear fit) -- no learned filters, no model
# training.

_BAND_SIGMAS: tuple[float, ...] = (0.8, 1.6, 3.2, 6.4)
_PATCH_SIZE = 8
_BAND_NAMES: tuple[str, ...] = tuple(f"band{i}" for i in range(len(_BAND_SIGMAS)))
FEATURE_NAMES_V2: tuple[str, ...] = FEATURE_NAMES + _BAND_NAMES


def _gaussian_blur(y: torch.Tensor, sigma: float) -> torch.Tensor:
    """Separable Gaussian blur of (B,H,W) luma with a FIXED kernel (reflect pad), differentiable."""
    h, w = int(y.shape[1]), int(y.shape[2])
    radius = max(1, int(math.ceil(3.0 * sigma)))
    radius = min(radius, max(0, h - 1), max(0, w - 1))  # keep reflect pad valid on small images
    if radius < 1:
        return y
    xs = torch.arange(-radius, radius + 1, device=y.device, dtype=y.dtype)
    k = torch.exp(-0.5 * (xs / sigma) ** 2)
    k = k / k.sum()
    yb = y.unsqueeze(1)  # (B,1,H,W)
    yb = F.pad(yb, (radius, radius, 0, 0), mode="reflect")
    yb = F.conv2d(yb, k.view(1, 1, 1, -1))
    yb = F.pad(yb, (0, 0, radius, radius), mode="reflect")
    yb = F.conv2d(yb, k.view(1, 1, -1, 1))
    return yb.squeeze(1)


def patch_band_features_torch(
    x: torch.Tensor,
    *,
    patch_size: int = _PATCH_SIZE,
    sigmas: tuple[float, ...] = _BAND_SIGMAS,
    etex_floor: float = 1e-8,
) -> torch.Tensor:
    """(B,3,H,W) in [0,1] -> (B, len(sigmas)) high-E_tex-weighted multi-scale band-pass energies."""
    x = _validate_bchw(x)
    b = x.shape[0]
    y = _luma(x)  # (B,H,W)
    h, w = int(y.shape[1]), int(y.shape[2])
    ps = max(1, min(patch_size, h, w))
    n_ph, n_pw = h // ps, w // ps
    y = y[:, : n_ph * ps, : n_pw * ps]  # crop to a whole patch grid
    blurs = [_gaussian_blur(y, s) for s in sigmas]
    # Gaussian-pyramid DIFFERENCE bands: finest = y - blur(smallest sigma), then successive DoGs.
    bands = [y - blurs[0]] + [blurs[i] - blurs[i + 1] for i in range(len(sigmas) - 1)]

    def _patch_energy(band: torch.Tensor) -> torch.Tensor:  # (B,Hc,Wc) -> (B, n_patches)
        bp = band.reshape(b, n_ph, ps, n_pw, ps)
        return bp.pow(2).mean(dim=(2, 4)).reshape(b, n_ph * n_pw)

    energies = [_patch_energy(bd) for bd in bands]  # each (B, P)
    etex = energies[0]  # finest band == high-pass energy, the rho E_tex concentration
    weight = etex / (etex.sum(dim=1, keepdim=True) + etex_floor)  # (B,P) soft high-E_tex weights
    return torch.stack([(e * weight).sum(dim=1) for e in energies], dim=1)  # (B, n_bands)


def feature_stack_torch_v2(x: torch.Tensor) -> torch.Tensor:
    """``patch_v2`` features: the six global twins ++ the band energies, (B, 6 + n_bands)."""
    return torch.cat([feature_stack_torch(x, FEATURE_NAMES), patch_band_features_torch(x)], dim=1)


_FEATURE_SETS = ("global_v1", "patch_v2")


def feature_set_names(feature_set: str) -> tuple[str, ...]:
    if feature_set == "global_v1":
        return FEATURE_NAMES
    if feature_set == "patch_v2":
        return FEATURE_NAMES_V2
    raise ValueError(f"unknown feature_set '{feature_set}', expected one of {_FEATURE_SETS}")


def compute_feature_stack(x: torch.Tensor, feature_set: str) -> torch.Tensor:
    """Differentiable (B,P) feature stack for ``feature_set`` (the space fit + attack share)."""
    if feature_set == "global_v1":
        return feature_stack_torch(x, FEATURE_NAMES)
    if feature_set == "patch_v2":
        return feature_stack_torch_v2(x)
    raise ValueError(f"unknown feature_set '{feature_set}', expected one of {_FEATURE_SETS}")


@dataclass(frozen=True)
class RhoSurrogate:
    """A fitted linear differentiable approximation ``s(x) ~ rho`` over standardized features.

    ``s(x) = bias + sum_p weight_p * (feature_p(x) - mean_p) / std_p`` -- higher s = higher rho;
    the attack raises s toward the cell's H0 mean (the attacked images lie below it).
    Holds float tensors so the surrogate is differentiable and device-agnostic.
    """

    names: tuple[str, ...]
    weight: torch.Tensor  # (P,) standardized-feature coefficients
    bias: float
    feat_mean: torch.Tensor  # (P,)
    feat_std: torch.Tensor  # (P,) > 0
    r2: float  # in-sample fit quality on its fit pool (a diagnostic, not used in __call__)
    feature_set: str = "global_v1"  # which differentiable feature stack the weights index

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        feats = compute_feature_stack(x, self.feature_set)  # (B,P)
        w = self.weight.to(device=feats.device, dtype=feats.dtype)
        mean = self.feat_mean.to(device=feats.device, dtype=feats.dtype)
        std = self.feat_std.to(device=feats.device, dtype=feats.dtype)
        z = (feats - mean) / std
        return self.bias + (z * w).sum(dim=1)


def fit_rho_surrogate(
    feature_matrix: np.ndarray,
    rho: np.ndarray,
    *,
    feature_set: str = "global_v1",
) -> RhoSurrogate:
    """Least-squares fit of raw rho on the standardized ``feature_set`` features (the linear proxy).

    ``feature_matrix`` is (N, P) in the ``feature_set`` column order (``global_v1`` = the six global
    twins; ``patch_v2`` = those ++ the patch-aligned band energies); ``rho`` is the (N,) measured
    rho on the same pool (NOT sign-aligned -- the proxy approximates rho itself so the attack can
    raise it). Constant columns are kept but get a zero coefficient (their standardized column is
    set to 0), so they never contribute. Standardization uses the fitting pool's mean and
    population std (ddof=0); the fit is ordinary least squares with an intercept.
    """
    names = feature_set_names(feature_set)
    f = np.asarray(feature_matrix, dtype=np.float64)
    r = np.asarray(rho, dtype=np.float64).reshape(-1)
    if f.ndim != 2 or f.shape[1] != len(names):
        raise ValueError(
            f"feature_matrix must be (N,{len(names)}) in {feature_set} order, got {f.shape}"
        )
    if f.shape[0] != r.size:
        raise ValueError(f"feature_matrix rows ({f.shape[0]}) must match rho ({r.size})")
    if f.shape[0] < 2:
        raise ValueError("need at least 2 samples to standardize and fit")
    if not (np.all(np.isfinite(f)) and np.all(np.isfinite(r))):
        raise ValueError("fit inputs must be finite")
    mean = f.mean(axis=0)
    std = f.std(axis=0, ddof=0)
    # A genuinely-constant column can carry ~1e-16 float dust (mean rounding), so threshold rather
    # than test > 0 exactly; a column with std below this has no usable signal and is dropped.
    nonconstant = std > 1e-12
    safe_std = np.where(nonconstant, std, 1.0)
    z = np.where(nonconstant, (f - mean) / safe_std, 0.0)  # constant cols -> 0 (zero contribution)
    design = np.column_stack([np.ones(r.size), z])  # [1 | z]
    coef, *_ = np.linalg.lstsq(design, r, rcond=None)
    coef[1:][~nonconstant] = 0.0  # constant columns are zeroed in z; pin their weight to exactly 0
    fitted = design @ coef
    if not np.all(np.isfinite(coef)) or not np.all(np.isfinite(fitted)):
        raise ValueError("rho-surrogate fit produced non-finite coefficients")
    ss_res = float(np.sum((r - fitted) ** 2))
    ss_tot = float(np.sum((r - r.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0.0 else 0.0
    return RhoSurrogate(
        names=tuple(names),
        weight=torch.tensor(coef[1:], dtype=torch.float32),
        bias=float(coef[0]),
        feat_mean=torch.tensor(mean, dtype=torch.float32),
        feat_std=torch.tensor(safe_std, dtype=torch.float32),
        r2=r2,
        feature_set=feature_set,
    )
