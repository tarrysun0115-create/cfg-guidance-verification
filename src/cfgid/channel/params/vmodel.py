"""Analytic v(QF) model: a smooth alternative to the noisy per-cell empirical null variance.

A per-cell empirical null variance ``v_hat`` estimated from few H0 samples has high variance and
no enforced QF structure. This module models the H0 rho-summary variance as

    v(QF) = v_floor + kappa * jpeg_transfer(QF),

where ``jpeg_transfer(QF) = mean(step^2)/12`` of the STANDARD JPEG luma table at quality QF
(``channel.qf.dqt_for_quality`` -> ``jpeg_quant_noise_var``) is the controlled, NOISE-FREE
QF-dependence (uncompressed -> 0), and ``(v_floor, kappa)`` are two per-decoder scalars fitted by
OLS on the H0 (w=1) rho variances across QF. Only H0 samples are touched (predictive track), and the
two-parameter fit smooths the four noisy per-QF variances onto the analytic monotone shape:
``v(none) < v(90) < v(75) < v(50)``. The w* cross-QF ORDERING depends on this analytic shape, not on
the absolute scale of ``kappa``.

Provenance is ``analytic_no_calibration`` when the fitted slope is positive (the QF dependence is
the controlled DQT model). If the H0 fit cannot identify a positive QF slope, the flat fallback is
kept only as ``unlabeled_null_adapted`` and flagged in ``notes``. Used by ``sweep.evidence`` for
the v of the w* sensitivity calculation; on the uncompressed main grid only one QF level exists,
so the fit is unidentifiable there and the caller falls back to the empirical H0 variance.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from cfgid.bounds.wstar_p0 import V_PROVENANCE
from cfgid.channel.qf import dqt_for_quality, jpeg_quant_noise_var
from cfgid.numerics import as_finite_float32

_MIN_PER_QF = 3  # minimum H0 samples to trust a per-QF variance estimate


class VModelUnidentifiableError(ValueError):
    """Raised when the H0 data are valid but insufficient to identify a QF slope."""


def jpeg_transfer(qf: int | None) -> float:
    """Analytic JPEG quantization-noise regressor for the rho channel (uncompressed -> 0).

    ``qf is None`` or ``qf < 1`` (the uncompressed sentinel, stored as -1 in p0 tables) returns 0.0;
    otherwise ``mean(step^2)/12`` of the standard luma table at that quality.
    """
    if qf is None or int(qf) < 1:
        return 0.0
    return jpeg_quant_noise_var(dqt_for_quality(int(qf)))


@dataclass(frozen=True)
class VModel:
    """Per-decoder analytic v(QF) = v_floor + kappa * jpeg_transfer(QF)."""

    v_floor: float
    kappa: float
    r2: float
    n_levels: int
    provenance: str
    notes: str

    def __post_init__(self) -> None:
        for name in ("v_floor", "kappa", "r2"):
            x = getattr(self, name)
            if not np.isfinite(x):
                raise ValueError(f"{name} must be finite, got {x}")
        if self.v_floor <= 0.0:
            raise ValueError(f"v_floor must be positive, got {self.v_floor}")
        if self.kappa < 0.0:
            raise ValueError(f"kappa must be non-negative, got {self.kappa}")
        if self.provenance not in V_PROVENANCE:
            raise ValueError(f"unknown v provenance '{self.provenance}'")

    def v_at(self, qf: int | None) -> float:
        """Analytic null variance at quality ``qf`` (always positive: v_floor>0, kappa>=0)."""
        value = self.v_floor + self.kappa * jpeg_transfer(qf)
        if not (np.isfinite(value) and value > 0.0):
            raise ValueError(f"v_at({qf}) must be positive and finite, got {value}")
        return float(value)


def fit_v_model(qf_to_h0rho: dict[int | None, np.ndarray]) -> VModel:
    """OLS fit of H0 rho variance on ``jpeg_transfer(QF)`` -> (v_floor, kappa) with flagged clamps.

    ``qf_to_h0rho`` maps each QF level (None / -1 = uncompressed) to its H0 (w=1) rho samples. QF
    levels with fewer than ``_MIN_PER_QF`` samples or zero variance are dropped; at least two
    DISTINCT transfer values are required to identify the slope. A non-physical fit (slope < 0, i.e.
    more compression giving LESS variance — a small-sample artifact) is clamped to ``kappa=0`` (flat
    v) and flagged; a non-positive intercept is clamped to a fraction of the smallest observed
    variance. If the fitted slope is non-positive, the QF shape is not identified; the flat model is
    retained as an H0-only scale estimate but labeled ``unlabeled_null_adapted`` rather than
    ``analytic_no_calibration``. Raises ``ValueError`` when the model is otherwise unidentifiable
    (caller uses empirical v).
    """
    xs: list[float] = []
    ys: list[float] = []
    dropped: list[str] = []
    for qf, rho in qf_to_h0rho.items():
        r = as_finite_float32("h0_rho", rho).astype(np.float64).ravel()
        if r.size < _MIN_PER_QF:
            dropped.append(f"qf={qf}: n={r.size}<{_MIN_PER_QF}")
            continue
        var = float(np.var(r, ddof=1))
        if var <= 0.0:
            dropped.append(f"qf={qf}: zero variance")
            continue
        xs.append(jpeg_transfer(qf))
        ys.append(var)
    x = np.asarray(xs, dtype=np.float64)
    y = np.asarray(ys, dtype=np.float64)
    if np.unique(x).size < 2:
        raise VModelUnidentifiableError(
            f"v model needs >=2 distinct QF transfer levels, got {np.unique(x).size} "
            f"(usable points: {x.size}; dropped: {dropped})"
        )

    a_mat = np.stack([np.ones_like(x), x], axis=1)
    (intercept, slope), *_ = np.linalg.lstsq(a_mat, y, rcond=None)
    flags: list[str] = []
    kappa = float(slope)
    provenance = "analytic_no_calibration"
    if kappa <= 0.0:
        flags.append(
            f"JPEG-noise slope {kappa:.4g}<=0 (small-sample/non-physical); "
            "clamped to 0 -> flat v at mean(H0 var); QF shape unidentifiable"
        )
        kappa = 0.0
        provenance = "unlabeled_null_adapted"
        # Flat MLE for a kappa-fixed-at-0 model is the mean of the observed variances, NOT the
        # joint OLS intercept (which is the x=0 extrapolation of the discarded negative slope and
        # is biased).
        v_floor = float(np.mean(y))
    else:
        v_floor = float(intercept)
    if v_floor <= 0.0:
        floor_clamp = 0.5 * float(np.min(y))
        flags.append(f"intercept {v_floor:.4g}<=0; clamped to 0.5*min_var={floor_clamp:.4g}")
        v_floor = floor_clamp

    y_hat = v_floor + kappa * x
    ss_res = float(np.sum((y - y_hat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0.0 else 0.0
    shape_note = (
        "QF dependence analytic (DQT)"
        if provenance == "analytic_no_calibration"
        else "QF dependence flat (DQT slope unidentifiable)"
    )
    notes = (
        f"OLS v(QF)=v_floor+kappa*jpeg_transfer over {x.size} QF levels "
        f"(v_floor={v_floor:.4g}, kappa={kappa:.4g}, r2={r2:.3f}); "
        f"{shape_note}, (v_floor,kappa) unlabeled H0 scale"
    )
    if provenance != "analytic_no_calibration":
        notes += "; flat model provenance is unlabeled_null_adapted because no DQT ordering remains"
    if flags:
        notes += "; " + "; ".join(flags)
    if dropped:
        notes += f"; dropped {dropped}"
    return VModel(
        v_floor=v_floor,
        kappa=kappa,
        r2=r2,
        n_levels=int(x.size),
        provenance=provenance,
        notes=notes,
    )
