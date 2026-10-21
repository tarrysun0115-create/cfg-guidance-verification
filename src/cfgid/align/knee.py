"""Knee extraction: the interpolated w where AUC first rises a set fraction above the baseline.

The raw crossing (``extract_knee``) is sensitive to non-monotone AUC wiggles from small per-w
samples (one noisy w level can produce a spurious early crossing). ``robust_knee`` first
isotonic-smooths the AUC curve (separability is non-decreasing in guidance in expectation), and
``knee_bootstrap_ci`` resamples the per-w scores to attach an uncertainty (CI + std) that the
alignment regression uses to DOWN-WEIGHT unstable cells, so that a single noisy cell does not
dominate the fit. Used by ``cfgid.sweep.analyze`` for the knee-vs-w* alignment (Section IV-E7).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from cfgid.align.roc import auc_curve


def _finite_vector(name: str, values: np.ndarray) -> np.ndarray:
    try:
        out = np.asarray(values, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if not np.all(np.isfinite(out)):
        raise ValueError(f"{name} contains non-finite values (NaN/Inf)")
    return out


def extract_knee(
    w_values: np.ndarray,
    auc_values: np.ndarray,
    *,
    baseline: float = 0.5,
    rise_frac: float = 0.5,
) -> float:
    """Smallest w (linearly interpolated) where AUC crosses ``baseline + rise_frac*(max-baseline)``.

    If AUC never rises above the baseline, returns the max w (knee beyond the swept range).
    """
    w = _finite_vector("w_values", w_values)
    a = _finite_vector("auc_values", auc_values)
    if w.shape != a.shape or w.size < 2:
        raise ValueError("w_values and auc_values must be equal-length arrays of length >= 2")
    if not (0.0 <= rise_frac <= 1.0):
        raise ValueError(f"rise_frac must be in [0, 1], got {rise_frac}")
    order = np.argsort(w)
    w, a = w[order], a[order]
    if a.max() <= baseline:
        return float(w[-1])
    target = baseline + rise_frac * (a.max() - baseline)
    # Already at/above target at the lowest swept w -> knee is at (or below) w[0].
    if a[0] >= target:
        return float(w[0])
    # Accept only a genuine UPWARD crossing a[i-1] < target <= a[i]; this keeps frac in [0, 1] so
    # the interpolated knee stays in [w[i-1], w[i]] (no off-grid value, no downward crossings).
    for i in range(1, w.size):
        if a[i - 1] < target <= a[i]:
            frac = (target - a[i - 1]) / (a[i] - a[i - 1])
            return float(w[i - 1] + frac * (w[i] - w[i - 1]))
    return float(w[-1])


def isotonic_increasing(y: np.ndarray) -> np.ndarray:
    """Pool-adjacent-violators non-decreasing fit (least-squares, unit weights)."""
    v = _finite_vector("y", y).copy()
    n = v.size
    if n == 0:
        raise ValueError("isotonic_increasing requires a non-empty array")
    # Each block tracks (value, weight, length); merge while the previous block exceeds the next.
    vals: list[float] = []
    wts: list[float] = []
    lens: list[int] = []
    for x in v:
        vals.append(float(x))
        wts.append(1.0)
        lens.append(1)
        while len(vals) > 1 and vals[-2] > vals[-1]:
            w2, w1 = wts[-2], wts[-1]
            merged = (vals[-2] * w2 + vals[-1] * w1) / (w2 + w1)
            vals[-2], wts[-2], lens[-2] = merged, w2 + w1, lens[-2] + lens[-1]
            vals.pop()
            wts.pop()
            lens.pop()
    out = np.empty(n, dtype=np.float64)
    pos = 0
    for value, length in zip(vals, lens, strict=True):
        out[pos : pos + length] = value
        pos += length
    return out


def robust_knee(
    w_values: np.ndarray,
    auc_values: np.ndarray,
    *,
    baseline: float = 0.5,
    rise_frac: float = 0.5,
) -> float:
    """``extract_knee`` on an isotonic-smoothed AUC curve (monotone in w in expectation)."""
    w = _finite_vector("w_values", w_values)
    a = _finite_vector("auc_values", auc_values)
    if w.shape != a.shape or w.size < 2:
        raise ValueError("w_values and auc_values must be equal-length arrays of length >= 2")
    order = np.argsort(w)
    smoothed = isotonic_increasing(a[order])
    return extract_knee(w[order], smoothed, baseline=baseline, rise_frac=rise_frac)


@dataclass(frozen=True)
class KneeCI:
    """Robust knee point estimate with a bootstrap CI and std (the regression's WLS weight)."""

    knee: float
    ci_lo: float
    ci_hi: float
    std: float

    def __post_init__(self) -> None:
        for name in ("knee", "ci_lo", "ci_hi", "std"):
            value = getattr(self, name)
            if not np.isfinite(value):
                raise ValueError(f"{name} must be finite, got {value}")
        if self.std < 0.0:
            raise ValueError(f"std must be non-negative, got {self.std}")
        if not self.ci_lo <= self.knee <= self.ci_hi:
            raise ValueError(
                f"CI must bracket knee: ci_lo={self.ci_lo}, knee={self.knee}, ci_hi={self.ci_hi}"
            )


def knee_bootstrap_ci(
    scores_by_w: dict[int, np.ndarray],
    *,
    h0_w: int = 1,
    n_boot: int = 1000,
    ci_level: float = 0.95,
    seed: int = 0,
    baseline: float = 0.5,
    rise_frac: float = 0.5,
) -> KneeCI:
    """Robust knee + percentile CI by resampling the per-w score arrays (H0 included).

    The point estimate is ``robust_knee`` on the full sample; each bootstrap replicate resamples
    every w-group with replacement (propagating H0 and H1 sampling noise), recomputes the AUC curve
    and robust knee, and the spread of those knees gives ``(ci_lo, ci_hi, std)``. The CI endpoints
    are clamped to bracket the point estimate. A cell with an unstable knee gets a large ``std`` ->
    small WLS weight downstream.
    """
    if not (0.0 < ci_level < 1.0):
        raise ValueError(f"ci_level must be in (0, 1), got {ci_level}")
    if n_boot < 1:
        raise ValueError(f"n_boot must be >= 1, got {n_boot}")

    def _knee(curve: dict[int, float]) -> float:
        w_axis = np.asarray(sorted(curve), dtype=np.float64)
        auc_axis = np.asarray([curve[int(w)] for w in w_axis], dtype=np.float64)
        return robust_knee(w_axis, auc_axis, baseline=baseline, rise_frac=rise_frac)

    point = _knee(auc_curve(scores_by_w, h0_w=h0_w))
    rng = np.random.default_rng(seed)
    knees = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        # Consume the RNG in a CANONICAL (sorted-w) order, not dict-insertion order: the point
        # estimate is order-invariant (auc_curve/_knee sort internally) but a per-w resample in
        # iteration order would make ci_lo/ci_hi/std a function of dict key order too, so identical
        # data + seed could yield different WLS precision weights. Keep it pure in (data, seed).
        resampled: dict[int, np.ndarray] = {}
        for w in sorted(scores_by_w):
            s = scores_by_w[w]
            resampled[w] = s[rng.integers(0, s.size, s.size)] if s.size > 0 else s
        knees[b] = _knee(auc_curve(resampled, h0_w=h0_w))
    alpha = (1.0 - ci_level) / 2.0
    lo, hi = np.quantile(knees, [alpha, 1.0 - alpha])
    return KneeCI(
        knee=point,
        ci_lo=min(float(lo), point),
        ci_hi=max(float(hi), point),
        std=float(np.std(knees)),
    )
