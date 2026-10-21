"""Covariance-weighted delta-deniability threshold for a multi-coordinate Gaussian channel.

Generalizes the scalar channel (``cfgid.bounds.wstar_p0``) to a P-patch Gaussian vector:

    rho_vec | w ~ N((w - 1) * d, Sigma)

H0 is w=1 (mean 0). The Bayes-optimal statistic is the likelihood ratio; for two equal-covariance
Gaussians the total variation depends only on the Mahalanobis separation

    Delta(w) = sqrt(N_eff) * |w - 1| * sqrt(d^T Sigma^{-1} d),
    TV(N_eff) = 2 * Phi(Delta / 2) - 1,

so the effective per-(w-1) signal is the Mahalanobis norm ``S_eff = sqrt(d^T Sigma^{-1} d)``
(Lemma 2 / Lemma C.2; Eq. (10) is its scalar case). The caller supplies ``d`` under an explicit
texture convention: generation-side channels may use ``d_p = g_sigma * E_tex_gen,p``;
measured-E_tex channels should use ``d_p = E_tex_meas,p`` because the post-decode/post-blur
latent energy already carries the PSF attenuation. The within-channel patch covariance carried by
Sigma, taken relative to the signal direction ``d``, determines the separation ``d^T Sigma^{-1} d``;
depending on that direction, correlation can raise or lower it relative to the independent-patch
sum. Only under additional conditions (equal per-patch signals and positive equicorrelation) does
correlation act as a reduction of the effective number of patches.
The scalar generation-side channel is the P=1 special case
``d^T Sigma^{-1} d = (g E_tex_gen)^2 / v``, recovering ``wstar_p0.wstar`` exactly. Imposing TV <=
delta:

    w*_cov = 1 + c(delta, N_eff)/S_eff,   c(delta, N_eff) = 2*Phi^{-1}((1+delta)/2)/sqrt(N_eff).

Pure numpy/scipy. Sigma is conditioned by Ledoit-Wolf shrinkage at estimation time (the raw sample
covariance is ill-posed at main-grid sample sizes); this module validates positive-definiteness and
solves via Cholesky (never forms an explicit inverse). Parameter provenance is carried by the
caller, as in the scalar module.

``ledoit_wolf_cov`` is also the covariance estimator of the deployed T_pair-LDA head (Eq. (11):
per-cell FIT-H0 covariance of the standardized pair, ``scripts/analysis/_readout_common.py``) and
of ``cfgid.sweep.postproc_pair``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import LinAlgError, cho_factor, cho_solve
from scipy.stats import norm

from cfgid.bounds.wstar_p0 import deniability_constant


def _as_finite_float64(name: str, values: np.ndarray) -> np.ndarray:
    """Cast to float64 and reject non-finite values without quantizing analytic math to float32."""
    try:
        out = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if not np.all(np.isfinite(out)):
        raise ValueError(f"{name} contains non-finite values (NaN/Inf)")
    return out


def _cholesky_ok(cov: np.ndarray) -> bool:
    try:
        np.linalg.cholesky(cov)
    except np.linalg.LinAlgError:
        return False
    return True


def ledoit_wolf_cov(samples: np.ndarray) -> tuple[np.ndarray, float]:
    """Ledoit-Wolf (2004) shrinkage covariance toward the scaled-identity target ``mu * I``.

    ``samples`` is ``(n_samples, n_features)``. Returns ``(cov, shrinkage)`` with ``shrinkage`` in
    [0, 1]: ``cov = shrinkage * mu * I + (1 - shrinkage) * S`` where ``S`` is the MLE sample
    covariance (divided by ``n``) and ``mu = trace(S) / p``. Shrinkage guarantees a well-conditioned
    (positive-definite for shrinkage > 0) estimate even when ``n_samples < n_features``.
    """
    x = _as_finite_float64("samples", samples)
    if x.ndim != 2:
        raise ValueError(f"samples must be 2D (n, p); got shape {x.shape}")
    n, p = x.shape
    if n < 2:
        raise ValueError(f"need at least 2 samples for a covariance, got {n}")
    if p < 1:
        raise ValueError("samples must have at least one feature")
    xc = x - x.mean(axis=0, keepdims=True)
    # The samples are provably finite (validated above); some BLAS backends (e.g. macOS
    # Accelerate) raise SPURIOUS FP flags (divide/invalid/overflow) on the matmuls below even at
    # realistic main-grid dims. Suppress those known-false flags locally, like
    # bounds.montecarlo.simulate_tv_cov / detector.sufficiency, then assert finiteness so a REAL
    # overflow raises loudly instead of being masked (e.g. by max(0.0, nan) -> shrinkage 0).
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        s = (xc.T @ xc) / n  # MLE sample covariance (divide by n, matching the LW derivation)
        s = (s + s.T) / 2.0
        mu = float(np.trace(s) / p)
        target = mu * np.eye(p)
        d2 = float(np.sum((s - target) ** 2))  # ||S - F||_F^2
        if d2 <= 0.0:
            if mu <= 0.0:
                raise ValueError(
                    "sample covariance has zero total variance; covariance is degenerate"
                )
            # S already equals the target (e.g. p==1 with positive variance) -> no shrinkage needed.
            return target, 0.0
        sq_norms = np.sum(xc**2, axis=1)  # ||x_i||^2
        s_frob_sq = float(np.sum(s**2))
        quad = float(np.sum(xc * (xc @ s)))  # sum_i x_i^T S x_i
        b_bar2 = max(0.0, float(np.sum(sq_norms**2) - 2.0 * quad + n * s_frob_sq) / (n**2))
        b2 = min(b_bar2, d2)  # b2 <= d2 by construction; clamp for numerical safety
        shrinkage = float(b2 / d2)
        cov = shrinkage * target + (1.0 - shrinkage) * s
        cov = (cov + cov.T) / 2.0
        if not _cholesky_ok(cov):
            if mu <= 0.0:
                raise ValueError(
                    "sample covariance has zero total variance; covariance is degenerate"
                )
            # Some perfectly rank-deficient samples have beta=0, hence LW's raw shrinkage is 0 even
            # though S is singular (common when n < p). A tiny target shrink keeps the estimator in
            # the same LW family while honoring the public contract that this helper returns SPD
            # covariances.
            shrinkage = max(shrinkage, 1.0e-6)
            cov = shrinkage * target + (1.0 - shrinkage) * s
            cov = (cov + cov.T) / 2.0
            if not _cholesky_ok(cov):
                shrinkage = 1.0
                cov = target
    if not (np.isfinite(shrinkage) and bool(np.all(np.isfinite(cov)))):
        raise ValueError("ledoit_wolf_cov produced a non-finite covariance/shrinkage")
    return cov, shrinkage


def mahalanobis_sq(d: np.ndarray, cov: np.ndarray) -> float:
    """``d^T cov^{-1} d`` via Cholesky solve. Raises if ``cov`` is not symmetric PD."""
    dv = _as_finite_float64("d", d).reshape(-1)
    c = _as_finite_float64("cov", cov)
    if c.ndim != 2 or c.shape[0] != c.shape[1]:
        raise ValueError(f"cov must be square 2D; got shape {c.shape}")
    if dv.size != c.shape[0]:
        raise ValueError(f"d ({dv.size}) and cov ({c.shape}) dimensions disagree")
    if not np.allclose(c, c.T, atol=1e-8, rtol=0.0):
        raise ValueError("cov must be symmetric")
    try:
        factor = cho_factor(c, lower=True)
    except LinAlgError as exc:
        raise ValueError("cov is not positive-definite (Cholesky failed)") from exc
    sol = cho_solve(factor, dv)
    value = float(dv @ sol)
    if not np.isfinite(value) or value < 0.0:
        raise ValueError(f"d^T cov^-1 d must be finite and non-negative, got {value}")
    return value


@dataclass(frozen=True)
class CovChannel:
    """Per-patch unit-(w-1) signal slope ``d`` and patch covariance ``Sigma``."""

    d: np.ndarray  # (P,) per-patch mean signal at unit (w-1)
    cov: np.ndarray  # (P, P) symmetric positive-definite patch covariance (shrunk)

    def __post_init__(self) -> None:
        dv = _as_finite_float64("d", self.d).reshape(-1)
        if dv.size < 1:
            raise ValueError("d must have at least one patch")
        if np.all(dv == 0.0):
            raise ValueError("d is all-zero; the guidance signal is unidentified")
        # mahalanobis_sq validates squareness / symmetry / PD and dimension agreement.
        if mahalanobis_sq(dv, self.cov) <= 0.0:
            raise ValueError("d^T Sigma^-1 d <= 0; channel carries no signal")

    @property
    def s_eff(self) -> float:
        """Mahalanobis signal norm ``sqrt(d^T Sigma^{-1} d)`` (per unit (w-1))."""
        return float(np.sqrt(mahalanobis_sq(self.d, self.cov)))


def wstar_cov(channel: CovChannel, *, delta: float, n_eff: float) -> float:
    """Covariance-weighted w* closed form ``1 + c(delta, N_eff) / S_eff``."""
    c = deniability_constant(delta, n_eff)
    d = _as_finite_float64("d", channel.d).reshape(-1)
    cov = _as_finite_float64("cov", channel.cov)
    if d.size == 1 and cov.shape == (1, 1):
        # Preserve the scalar-channel identity bit-for-bit for P=1:
        # w* = 1 + c * sqrt(v) / |d|. For generation-side d = g_sigma * E_tex_gen; for
        # measured-Etex d = E_tex_meas.
        return float(1.0 + c * np.sqrt(cov[0, 0]) / abs(d[0]))
    return 1.0 + c / channel.s_eff


def channel_tv_cov(w: float, channel: CovChannel, *, n_eff: float = 1.0) -> float:
    """Analytic TV between H0 (w=1) and H1 (w) for the N_eff-aggregated vector channel."""
    if not np.isfinite(w):
        raise ValueError(f"w must be finite, got {w}")
    if not (np.isfinite(n_eff) and n_eff > 0):
        raise ValueError(f"n_eff must be positive, got {n_eff}")
    delta_sep = float(np.sqrt(n_eff) * abs(w - 1.0) * channel.s_eff)
    return float(2.0 * norm.cdf(delta_sep / 2.0) - 1.0)
