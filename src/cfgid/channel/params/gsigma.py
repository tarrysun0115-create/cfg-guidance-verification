"""Predictive-track g_sigma estimation with per-parameter provenance protocols.

g_sigma is the sigma-determined gain of the generation-side scalar channel
``rho | w ~ N(g_sigma (w-1) E_tex_gen, v)`` (see ``cfgid.bounds.wstar_p0``), i.e. the heuristic
gain g_sigma = 1 / (1 + sigma^2) in the shape proxy |Delta_rho| ~ (w_0 - 1) g_sigma E_tex of
Section III-D (Eqs. (9)-(10)). The measured-E_tex analysis path (``sweep.analyze``, the w*
sensitivity of Section III-E) does not multiply this plugin into scalar w*, because the
re-encoded E_tex already carries decoder/blur attenuation (Section III-D); it still uses the
g_sigma provenance to enforce predictive-track gating.
Provenance is carried PER PARAMETER (never one global flag) and must be one of
``G_SIGMA_PROVENANCE``:

  * ``unlabeled_null_adapted`` — analytic plugin ``g(sigma_hat)`` from the decoder PSF probe plus
    H0-only (w=1) null statistics for ``v``. No H1 labels touched. Predictive track.
  * ``synthetic_fit`` — least-squares slope of rho vs ``(w-1) E_tex`` on SELF-GENERATED sweeps
    where w is known by construction (not a label on disputed images). Predictive track.
  * ``temperature_calibrated`` — the same fit on H1-LABELED data. Diagnostic track only: the
    preregistered p0 gate must NEVER pass on it. The CALLER owns this distinction (this
    module cannot know where ``rho_means`` came from); see ``synthetic_fit_gsigma``.

Scale caveat: ``plugin_gain`` is a PROVISIONAL first-order Gaussian-MTF low-pass plugin,
``g(sigma) = 1 / (1 + sigma^2)``. Without labels its ABSOLUTE scale is not identified (the paper
treats the slope proxy as uncalibrated in scale, Section III-E). On the measured-E_tex gate path
that scale is not part of scalar w*; on a generation-side ``E_tex_gen`` path, a uniform rescale
would uniformly rescale ``w* - 1``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from cfgid.bounds.wstar_p0 import G_SIGMA_PROVENANCE
from cfgid.channel.qf import dqt_for_quality, jpeg_mtf_attenuation
from cfgid.numerics import as_finite_float32

# Predictive track = estimated WITHOUT H1 labels; must match WStarParams.is_predictive.
PREDICTIVE_PROVENANCE = ("unlabeled_null_adapted", "synthetic_fit")

_N_BOOT_SYNTH = 1000  # internal bootstrap size for synthetic_fit_gsigma
_MIN_VALID_BOOT = 50  # below this many non-degenerate replicates the CI is meaningless


@dataclass(frozen=True)
class GSigmaEstimate:
    """A g_sigma point estimate with CI and explicit per-parameter provenance."""

    value: float
    ci_lo: float
    ci_hi: float
    provenance: str
    notes: str

    def __post_init__(self) -> None:
        for name in ("value", "ci_lo", "ci_hi"):
            x = getattr(self, name)
            if not np.isfinite(x):
                raise ValueError(f"{name} must be finite, got {x}")
        if self.value <= 0:
            raise ValueError(f"value must be positive, got {self.value}")
        if not (self.ci_lo <= self.value <= self.ci_hi):
            raise ValueError(
                f"CI must bracket value: ci_lo={self.ci_lo}, value={self.value}, ci_hi={self.ci_hi}"
            )
        if self.provenance not in G_SIGMA_PROVENANCE:
            raise ValueError(
                f"unknown g_sigma provenance '{self.provenance}' (valid: {G_SIGMA_PROVENANCE})"
            )

    @property
    def is_predictive(self) -> bool:
        """True iff estimated without H1 labels (gate may pass only on this track)."""
        return self.provenance in PREDICTIVE_PROVENANCE


def estimate_v_null(
    h0_rho: np.ndarray, *, n_boot: int = 1000, seed: int = 0
) -> tuple[float, float, float]:
    """H0 (w=1) null variance of rho summaries with a bootstrap percentile 95% CI.

    Returns ``(v_hat, ci_lo, ci_hi)`` with ``ci_lo <= v_hat <= ci_hi`` (percentile endpoints are
    clamped to bracket the full-sample estimate). Uses ONLY null samples — no H1 labels.
    """
    rho = as_finite_float32("h0_rho", h0_rho).ravel().astype(np.float64)
    if rho.size < 3:
        raise ValueError(f"need at least 3 finite H0 rho samples, got {rho.size}")
    if n_boot < 1:
        raise ValueError(f"n_boot must be >= 1, got {n_boot}")
    v_hat = float(np.var(rho, ddof=1))
    if v_hat <= 0.0:
        raise ValueError("h0_rho has zero variance (constant samples); null v is degenerate")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, rho.size, size=(n_boot, rho.size))
    reps = np.var(rho[idx], axis=1, ddof=1)
    ci_lo = min(float(np.percentile(reps, 2.5)), v_hat)
    ci_hi = max(float(np.percentile(reps, 97.5)), v_hat)
    return v_hat, ci_lo, ci_hi


def plugin_gain(sigma_hat: float) -> float:
    """PROVISIONAL analytic low-pass plugin ``g(sigma) = 1 / (1 + sigma^2)``.

    First-order Gaussian-MTF attenuation of the high-band rho statistic by a decoder PSF of width
    sigma (in decoded-image pixels — the unit of ``sigma_calib.sigma_hat`` and of the half-max-core
    sigma_eff = sqrt(sigma_core^2 + sigma_blur^2) fed here for blurred cells). The absolute scale
    of this plugin is NOT identified without labels. The measured-E_tex gate path logs/gates this
    value but does not multiply it into scalar w*; a generation-side ``E_tex_gen`` path would
    treat uniform rescaling as a uniform scaling of ``w* - 1``.
    """
    if not (np.isfinite(sigma_hat) and sigma_hat > 0):
        raise ValueError(f"sigma_hat must be positive and finite, got {sigma_hat}")
    return float(1.0 / (1.0 + sigma_hat**2))


def jpeg_mtf_gain(qf: int | None) -> float:
    """JPEG-MTF high-band attenuation multiplier in (0, 1] (uncompressed -> 1.0).

    Unlike a uniform rescale of ``plugin_gain`` (which leaves the knee-vs-w* ORDERING unchanged),
    this factor is QF-dependent and therefore NON-uniform across a decoder's cells: harsher JPEG
    (lower QF) attenuates the high-band texture the rho statistic uses, lowering g_sigma. That
    cell-to-cell variation is exactly what gives the alignment x-axis QF resolution. PROVISIONAL
    proxy (see ``channel.qf.jpeg_mtf_attenuation``).
    """
    if qf is None or int(qf) < 1:
        return 1.0
    return jpeg_mtf_attenuation(dqt_for_quality(int(qf)))


def null_adapted_gsigma(
    sigma_hat: float,
    sigma_ci: tuple[float, float],
    h0_rho: np.ndarray,
    *,
    qf: int | None = None,
    n_boot: int = 1000,
    seed: int = 0,
) -> GSigmaEstimate:
    """Unlabeled null-adapted g_sigma: plugin gain at sigma_hat x JPEG-MTF(QF), CI from sigma's CI.

    ``plugin_gain`` is monotone DECREASING in sigma, so the sigma CI endpoints swap when mapped:
    ``ci_lo = g(sigma_hi)``, ``ci_hi = g(sigma_lo)``. The optional ``qf`` multiplies value and CI by
    the (deterministic, positive) ``jpeg_mtf_gain(qf)`` -> a QF-dependent, ordering-changing gain
    (``qf=None`` recovers the sigma-only estimate exactly). The H0 null variance ``v_hat`` (from
    ``estimate_v_null``) is recorded in ``notes`` for transparency; no H1 labels are used.
    """
    s_lo, s_hi = float(sigma_ci[0]), float(sigma_ci[1])
    for name, s in (("sigma_ci[0]", s_lo), ("sigma_ci[1]", s_hi)):
        if not (np.isfinite(s) and s > 0):
            raise ValueError(f"{name} must be positive and finite, got {s}")
    if not s_lo <= sigma_hat <= s_hi:
        raise ValueError(f"sigma_ci ({s_lo}, {s_hi}) must bracket sigma_hat={sigma_hat}")
    mtf = jpeg_mtf_gain(qf)
    value = plugin_gain(sigma_hat) * mtf
    ci_lo = (
        plugin_gain(s_hi) * mtf
    )  # monotone decreasing -> ends swap; positive mtf preserves order
    ci_hi = plugin_gain(s_lo) * mtf
    v_hat, v_lo, v_hi = estimate_v_null(h0_rho, n_boot=n_boot, seed=seed)
    notes = (
        f"plugin g(sigma)=1/(1+sigma^2) at sigma_hat={sigma_hat:.6g} x jpeg_mtf(qf={qf})={mtf:.4g} "
        f"(sigma CI [{s_lo:.6g}, {s_hi:.6g}] propagated end-swapped); "
        f"H0 null v_hat={v_hat:.6g} (95% CI [{v_lo:.6g}, {v_hi:.6g}]); "
        "absolute scale provisional; QF dependence changes ordering (gate tests direction)"
    )
    return GSigmaEstimate(
        value=value, ci_lo=ci_lo, ci_hi=ci_hi, provenance="unlabeled_null_adapted", notes=notes
    )


def synthetic_fit_gsigma(
    w_grid: np.ndarray,
    rho_means: np.ndarray,
    e_tex_mean: float,
    *,
    seed: int = 0,
    labeled: bool = False,
) -> GSigmaEstimate:
    """g_sigma as |least-squares slope| of rho_mean vs ``(w-1) * e_tex_mean``.

    Bootstrap-percentile 95% CI over the (w, rho_mean) points; endpoints clamped to bracket the
    full-sample estimate. PROVENANCE IS OWNED BY THE CALLER: with ``labeled=False`` the points
    are declared SELF-GENERATED (w known by construction) -> ``synthetic_fit`` (predictive
    track); ``labeled=True`` declares the rho_means came from H1-LABELED data ->
    ``temperature_calibrated`` (diagnostic track; the p0 gate must never pass on it). This
    function cannot verify the declaration — it only records it.
    """
    w = as_finite_float32("w_grid", w_grid).ravel().astype(np.float64)
    r = as_finite_float32("rho_means", rho_means).ravel().astype(np.float64)
    if w.shape != r.shape:
        raise ValueError(f"w_grid and rho_means must align, got {w.shape} vs {r.shape}")
    if w.size < 3:
        raise ValueError(f"need at least 3 (w, rho_mean) points, got {w.size}")
    if not (np.isfinite(e_tex_mean) and e_tex_mean > 0):
        raise ValueError(f"e_tex_mean must be positive and finite, got {e_tex_mean}")
    x = (w - 1.0) * float(e_tex_mean)
    if np.unique(x).size < 2:
        raise ValueError("w_grid is constant; slope is unidentified")

    def _abs_slope(xv: np.ndarray, yv: np.ndarray) -> float:
        # OLS slope with intercept (robust to a null offset at w=1).
        xc = xv - xv.mean()
        return float(abs(np.dot(xc, yv - yv.mean()) / np.dot(xc, xc)))

    value = _abs_slope(x, r)
    if value <= 0.0:
        raise ValueError("fitted slope is zero; g_sigma is unidentified from these points")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, x.size, size=(_N_BOOT_SYNTH, x.size))
    xb, yb = x[idx], r[idx]
    xc = xb - xb.mean(axis=1, keepdims=True)
    vx = np.einsum("ij,ij->i", xc, xc)
    valid = vx > 0.0  # degenerate resamples (all-identical w) carry no slope information
    if int(valid.sum()) < _MIN_VALID_BOOT:
        raise ValueError("too few non-degenerate bootstrap replicates for a CI")
    cov = np.einsum("ij,ij->i", xc, yb - yb.mean(axis=1, keepdims=True))
    slopes = np.abs(cov[valid] / vx[valid])
    ci_lo = min(float(np.percentile(slopes, 2.5)), value)
    ci_hi = max(float(np.percentile(slopes, 97.5)), value)
    provenance = "temperature_calibrated" if labeled else "synthetic_fit"
    notes = (
        f"|OLS slope| of rho_mean vs (w-1)*E_tex over {w.size} points "
        f"(e_tex_mean={float(e_tex_mean):.6g}, bootstrap n={_N_BOOT_SYNTH}); "
        f"caller declared labeled={labeled} -> provenance '{provenance}'"
    )
    return GSigmaEstimate(value=value, ci_lo=ci_lo, ci_hi=ci_hi, provenance=provenance, notes=notes)
