"""Knee-vs-w* alignment regression: OLS slope + Pearson correlation + bootstrap slope CI.

The PREREGISTERED pass criterion is a positive, significant slope (CI excludes 0) of the empirical
knee against the analytic w* across cells — i.e. the deniable region moves where the theory says.
The bootstrap resamples cells with replacement (``numpy.random.default_rng(seed)``, seed 0 by
default); degenerate resamples are skipped, and a degenerate design yields a CI spanning 0.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from cfgid.numerics import as_finite_float32


@dataclass(frozen=True)
class RegressionResult:
    slope: float
    intercept: float
    corr: float
    slope_ci_lo: float
    slope_ci_hi: float
    n: int


def _ols_slope_intercept(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    xm, ym = x.mean(), y.mean()
    var = float(np.sum((x - xm) ** 2))
    if var == 0.0:
        raise ValueError("w* values are constant; cannot regress knee on w*")
    slope = float(np.sum((x - xm) * (y - ym)) / var)
    return slope, float(ym - slope * xm)


def _wls_slope_intercept(x: np.ndarray, y: np.ndarray, w: np.ndarray) -> tuple[float, float]:
    """Weighted least-squares slope + intercept (w are non-negative point weights)."""
    sw = float(np.sum(w))
    if sw <= 0.0:
        raise ValueError("weights sum to zero; cannot regress knee on w*")
    xm = float(np.sum(w * x) / sw)
    ym = float(np.sum(w * y) / sw)
    var = float(np.sum(w * (x - xm) ** 2))
    if var == 0.0:
        raise ValueError("w* values are constant (weighted); cannot regress knee on w*")
    slope = float(np.sum(w * (x - xm) * (y - ym)) / var)
    return slope, float(ym - slope * xm)


def _weighted_corr(x: np.ndarray, y: np.ndarray, w: np.ndarray) -> float:
    sw = float(np.sum(w))
    xm = float(np.sum(w * x) / sw)
    ym = float(np.sum(w * y) / sw)
    cov = float(np.sum(w * (x - xm) * (y - ym)))
    vx = float(np.sum(w * (x - xm) ** 2))
    vy = float(np.sum(w * (y - ym) ** 2))
    if vx <= 0.0 or vy <= 0.0:
        return 0.0
    return cov / float(np.sqrt(vx * vy))


def fit_alignment(
    knees: np.ndarray,
    wstars: np.ndarray,
    *,
    n_boot: int,
    ci_level: float,
    seed: int = 0,
    weights: np.ndarray | None = None,
) -> RegressionResult:
    """Regress empirical knee on analytic w* (x=w*, y=knee). Bootstrap CI for the slope.

    ``weights`` (optional, e.g. ``1 / knee_std**2``) down-weights cells with an unstable knee so a
    single noisy cell can't dominate; ``None`` reproduces the unweighted (OLS) fit exactly. The
    bootstrap resamples cells uniformly and recomputes the WLS slope with the resampled weights.
    """
    y = as_finite_float32("knees", knees).astype(np.float64).reshape(-1)
    x = as_finite_float32("wstars", wstars).astype(np.float64).reshape(-1)
    if x.shape != y.shape or x.size < 3:
        raise ValueError("knees and w* must be equal-length arrays of length >= 3")
    if not (0.0 < ci_level < 1.0):
        raise ValueError(f"ci_level must be in (0, 1), got {ci_level}")
    unweighted = weights is None
    if unweighted:
        w = np.ones_like(x)
        slope, intercept = _ols_slope_intercept(x, y)
        sx, sy = np.std(x), np.std(y)
        corr = float(np.corrcoef(x, y)[0, 1]) if sx > 0 and sy > 0 else 0.0
    else:
        w = as_finite_float32("weights", weights).astype(np.float64).reshape(-1)
        if w.shape != x.shape:
            raise ValueError("weights must align with knees/w*")
        if np.any(w < 0.0):
            raise ValueError("weights must be non-negative")
        slope, intercept = _wls_slope_intercept(x, y, w)
        corr = _weighted_corr(x, y, w)

    rng = np.random.default_rng(seed)
    n = x.size
    slopes = np.empty(n_boot, dtype=np.float64)
    fills = 0
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)  # resampling unit: cells (with replacement)
        xb, yb = x[idx], y[idx]
        if unweighted:
            if np.sum((xb - xb.mean()) ** 2) == 0.0:
                continue  # degenerate resample; skip
            slopes[fills] = float(
                np.sum((xb - xb.mean()) * (yb - yb.mean())) / np.sum((xb - xb.mean()) ** 2)
            )
        else:
            wb = w[idx]
            swb = float(np.sum(wb))
            if swb <= 0.0:
                continue
            xmb = float(np.sum(wb * xb) / swb)
            varb = float(np.sum(wb * (xb - xmb) ** 2))
            if varb == 0.0:
                continue  # degenerate resample; skip
            ymb = float(np.sum(wb * yb) / swb)
            slopes[fills] = float(np.sum(wb * (xb - xmb) * (yb - ymb)) / varb)
        fills += 1
    if fills == 0:
        # Every resample was degenerate -> the slope is unbounded by the bootstrap; report a CI that
        # INCLUDES 0 so the gate never reads a collapsed [slope, slope] CI as "significant".
        return RegressionResult(slope, intercept, corr, min(slope, 0.0), max(slope, 0.0), n)
    alpha = (1.0 - ci_level) / 2.0
    lo, hi = np.quantile(slopes[:fills], [alpha, 1.0 - alpha])
    if hi <= lo:
        # A zero-width bootstrap CI (lo == hi) is NOT a precise estimate: it means the x (w*) design
        # is degenerate -- a plateau of identical w* plus one distinct point, so every resample lies
        # on the same 2-point line and yields the identical slope. Span 0 like the fills==0 branch
        # so the gate (slope_ci_lo > 0) cannot read a collapsed [slope, slope] CI as significant (it
        # would also see corr=+/-1 from the same 2-distinct-point design).
        return RegressionResult(slope, intercept, corr, min(slope, 0.0), max(slope, 0.0), n)
    return RegressionResult(slope, intercept, corr, float(lo), float(hi), n)
