"""The rho statistic: per-patch guarded log ratio, texture mask, winsorized image summaries.

Implements Section III-C. Per latent patch p (Eq. (2)),

    q_p   = ||r_cond,p|| / max(||r_uncond,p||, eps_f)        (numerator unfloored)
    rho_p = log(max(q_p, eta_tiny))

with eps_f = ``rho.denom_floor`` (1e-3 in ``configs/p0.yaml``) bounding the denominator and
eta_tiny the smallest positive normal float64, which only guards log(0). The image summaries
keep the patches whose E_tex,p is at or above the within-image q-quantile (mask M(x), Eq. (3);
q = ``rho.low_etex_mask_quantile`` = 0.2), winsorize the retained values within the image at
``rho.winsor_quantiles`` = [0.01, 0.99] (W_kappa), and average them:
``image_rho_summary`` gives rho(x) of Eq. (4) and ``image_m_summary`` gives m(x) of Eq. (8),
the mean winsorized branch-difference magnitude delta_p = ||r_cond,p - r_uncond,p|| (Eq. (6)).
Together they form the deployed summary y(x) = (rho(x), m(x)). Pure numpy / finite-guarded.
"""

from __future__ import annotations

import numpy as np

from cfgid.numerics import as_finite_float32


def rho_patch(
    r_cond: np.ndarray, r_uncond: np.ndarray, *, floor: float, log_ratio: bool
) -> np.ndarray:
    """Per-patch rho_p of Eq. (2) from (P, D) patch-flattened cond/uncond noise predictions.

    ``log_ratio=True`` (the p0 setting) returns rho_p; ``False`` returns the raw ratio q_p.
    Finite-guarded; floor > 0 required.
    """
    if floor <= 0.0 or not np.isfinite(floor):
        raise ValueError(f"floor must be positive and finite, got {floor}")
    rc = as_finite_float32("r_cond", r_cond)
    ru = as_finite_float32("r_uncond", r_uncond)
    if rc.ndim != 2 or ru.ndim != 2:
        raise ValueError(f"residuals must be (P, D); got {rc.shape} and {ru.shape}")
    if rc.shape != ru.shape:
        raise ValueError(f"r_cond shape {rc.shape} != r_uncond shape {ru.shape}")
    # Accumulate the norm in float64: a high-CFG-divergent residual can be finite in float32 yet
    # carry a component above sqrt(float32_max) ~= 1.84e19, whose square overflows float32's 3.4e38
    # ceiling to +inf -- so num/den would silently become inf even though every element passed the
    # finite guard, aborting the whole decoder batch later at pack time. float64 accumulation avoids
    # the overflow; for normal O(1) residuals the float32 cast below is unchanged.
    num = np.linalg.norm(rc.astype(np.float64), axis=1)
    den = np.linalg.norm(ru.astype(np.float64), axis=1)
    ratio = num / np.maximum(den, floor)
    if log_ratio:
        # Guard log(0) with a tiny epsilon, NOT the denominator `floor`: `floor` is in residual-norm
        # units (the denominator clamp from configs/p0.yaml rho.denom_floor), so reusing it as a
        # clamp on the dimensionless ratio would lift every weak-signal patch (ratio < floor) up to
        # log(floor), biasing the low tail of the statistic the channel is designed to measure. The
        # only value that must be excluded is ratio == 0 (||r_cond|| == 0) -> log(0) = -inf.
        ratio = np.log(np.maximum(ratio, np.finfo(np.float64).tiny))
    if not np.all(np.isfinite(ratio)):
        raise ValueError("rho_patch produced non-finite values")
    f32_max = np.finfo(np.float32).max
    if np.any(np.abs(ratio) > f32_max):
        raise ValueError("rho_patch output exceeds finite float32 range")
    return ratio.astype(np.float32)


def winsorize(values: np.ndarray, lo_q: float, hi_q: float) -> np.ndarray:
    """Clip to the [lo_q, hi_q] empirical quantiles of ``values`` (W_kappa within one image)."""
    v = as_finite_float32("values", values)
    if not (0.0 <= lo_q < hi_q <= 1.0):
        raise ValueError(f"winsor quantiles must satisfy 0<=lo<hi<=1, got ({lo_q}, {hi_q})")
    if v.size == 0:
        return v
    lo, hi = np.quantile(v, [lo_q, hi_q])
    return np.clip(v, lo, hi).astype(np.float32)


def etex_keep_mask(e_tex: np.ndarray, quantile: float) -> np.ndarray:
    """Mask M(x) of Eq. (3): patches with E_tex >= the within-image ``quantile`` tau_q."""
    e = as_finite_float32("e_tex", e_tex)
    if not (0.0 <= quantile < 1.0):
        raise ValueError(f"mask quantile must be in [0, 1), got {quantile}")
    if e.size == 0:
        return np.zeros(0, dtype=bool)
    # Within-image threshold tau_q; ">=" keeps ties, so the mask is never empty for P > 0.
    thr = np.quantile(e, quantile)
    return e >= thr


def image_rho_summary(
    rho_p: np.ndarray,
    e_tex: np.ndarray,
    *,
    mask_quantile: float,
    winsor_quantiles: tuple[float, float],
) -> float:
    """Image-level scalar rho(x) of Eq. (4): mean of winsorized rho_p over the mask M(x).

    Masking comes first, then winsorization quantiles are taken over the retained patches only.
    Falls back to all patches if the mask would empty the set.
    """
    rho_arr = as_finite_float32("rho_p", rho_p)
    keep = etex_keep_mask(e_tex, mask_quantile)
    kept = rho_arr[keep] if keep.any() else rho_arr
    if kept.size == 0:
        raise ValueError("image_rho_summary received no patches")
    w = winsorize(kept, winsor_quantiles[0], winsor_quantiles[1])
    return float(np.mean(w))


def image_m_summary(
    residual_patch: np.ndarray,
    e_tex: np.ndarray,
    *,
    mask_quantile: float,
    winsor_quantiles: tuple[float, float],
) -> float:
    """Image-level scalar m(x) of Eq. (8): mean of winsorized delta_p over the mask M(x).

    The magnitude twin of ``image_rho_summary``: the same mask + winsor + mean recipe applied to
    the dual-branch residual difference norms. With the p0 mask/winsor parameters this is the
    deployed pair coordinate ``m`` (masked winsorized diff_energy); the float32 norm/winsor
    arithmetic matches the store-frozen recipe used by the residual-store reanalyses.
    """
    residual = as_finite_float32("residual_patch", residual_patch)
    if residual.ndim != 2:
        raise ValueError(f"residual_patch must be (P, D), got {residual.shape}")
    if residual.shape[0] == 0:
        raise ValueError("image_m_summary received no patches")
    e_tex_arr = as_finite_float32("e_tex", e_tex)
    if e_tex_arr.ndim != 1 or e_tex_arr.shape[0] != residual.shape[0]:
        raise ValueError(
            "e_tex must be (P,) aligned with residual_patch; "
            f"got {e_tex_arr.shape} vs {residual.shape}"
        )
    # delta_p of Eq. (6): Euclidean norm of the per-patch branch difference r_cond,p - r_uncond,p.
    delta = np.linalg.norm(residual, axis=1)
    keep = etex_keep_mask(e_tex_arr, mask_quantile)
    kept = delta[keep] if keep.any() else delta
    if kept.size == 0:
        raise ValueError("image_m_summary received no patches after masking")
    w = winsorize(kept.astype(np.float32), winsor_quantiles[0], winsor_quantiles[1])
    return float(np.mean(w))
