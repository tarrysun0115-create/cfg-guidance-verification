"""Scalar-channel closed forms: TV, the tolerance c(delta, N_eff) and the threshold w*.

Channel (Eq. (9) with the linear shape proxy of Section III-D): ``rho | w ~ N(g_sigma (w-1) E_tex,
v)``. H0: w=1 (mean 0). H1: w=w0 (mean mu = g_sigma (w0-1) E_tex). Aggregating N_eff effective iid
samples, the total-variation between H0 and H1 for the sufficient statistic is (Eqs. (13)-(14))

    TV(N_eff) = 2 * Phi( sqrt(N_eff) * |mu| / (2 sqrt(v)) ) - 1.

Imposing the finite-sample deniability tolerance TV <= delta (Eq. (15)) and solving for w under the
linear model gives the closed form (Eq. (38))

    w*(sigma, E_tex, QF, N_eff, delta) = 1 + c(delta, N_eff) * sqrt(v) / (g_sigma * E_tex),
    c(delta, N_eff) = 2 * Phi^{-1}((1+delta)/2) / sqrt(N_eff).

At or below w*, the Gaussian rho-channel TV is at most delta (Corollary 1). This is a statement
about the rho channel only: by the data-processing inequality (Eq. (21)) the rho-channel TV is a
lower bound on the image-level TV, so extending it to an arbitrary image detector would need an
extra assumption such as sufficiency of rho. The paper uses w* as a sensitivity quantity (Section
III-E). All pure numpy/scipy. Parameter provenance is carried explicitly (per-parameter, never one
global flag).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import norm

# Provenance protocols (must match configs/p0.yaml provenance.* and detector naming).
# v: analytic_no_calibration = controlled JPEG/DQT noise model; unlabeled_null_adapted = v
# estimated from H0 samples without labels (used on real features: the paper's v comes from
# NULL-split H0 rows, Section III-C); temperature_calibrated = labeled.
# g_sigma: only the label-free protocols (unlabeled_null_adapted, synthetic_fit) are "predictive".
V_PROVENANCE = ("analytic_no_calibration", "unlabeled_null_adapted", "temperature_calibrated")
G_SIGMA_PROVENANCE = ("unlabeled_null_adapted", "synthetic_fit", "temperature_calibrated")


@dataclass(frozen=True)
class WStarParams:
    """Channel parameters + their provenance for a w* evaluation."""

    g_sigma: float  # sigma-determined gain
    v: float  # noise variance (VAE + JPEG/DQT)
    v_provenance: str = "analytic_no_calibration"
    g_sigma_provenance: str = "unlabeled_null_adapted"

    def __post_init__(self) -> None:
        if self.g_sigma <= 0 or not np.isfinite(self.g_sigma):
            raise ValueError(f"g_sigma must be positive and finite, got {self.g_sigma}")
        if self.v <= 0 or not np.isfinite(self.v):
            raise ValueError(f"v must be positive and finite, got {self.v}")
        if self.v_provenance not in V_PROVENANCE:
            raise ValueError(f"unknown v provenance '{self.v_provenance}'")
        if self.g_sigma_provenance not in G_SIGMA_PROVENANCE:
            raise ValueError(f"unknown g_sigma provenance '{self.g_sigma_provenance}'")

    @property
    def is_predictive(self) -> bool:
        """Predictive track = g_sigma estimated WITHOUT H1 labels (gate may pass only on this)."""
        return self.g_sigma_provenance in ("unlabeled_null_adapted", "synthetic_fit")


def deniability_constant(delta: float, n_eff: float) -> float:
    """c(delta, N_eff) = 2 * Phi^{-1}((1+delta)/2) / sqrt(N_eff): the tolerance of Eq. (15)."""
    if not (np.isfinite(delta) and 0.0 < delta < 1.0):
        raise ValueError(f"delta must be in (0, 1), got {delta}")
    if not (np.isfinite(n_eff) and n_eff > 0):
        raise ValueError(f"n_eff must be positive, got {n_eff}")
    return float(2.0 * norm.ppf((1.0 + delta) / 2.0) / np.sqrt(n_eff))


def wstar(e_tex: float, params: WStarParams, *, delta: float, n_eff: float) -> float:
    """w*(.) closed form (Eq. (38)) for a GENERATION-side ``e_tex`` (slope ``g_sigma * e_tex``).

    ``e_tex`` > 0 required (low-texture patches masked upstream). Correct ONLY when ``e_tex`` is
    un-attenuated generation texture ``E_tex_gen``. For the MEASURED post-decode/post-blur latent
    energy ``E_tex_meas`` (what the main-grid pipeline persists) the decoder/blur PSF attenuation
    is ALREADY inside it -- use :func:`wstar_from_measured_etex`, which does NOT re-multiply
    ``g_sigma`` (that would count the PSF attenuation twice).
    """
    if e_tex <= 0 or not np.isfinite(e_tex):
        raise ValueError(f"e_tex must be positive and finite, got {e_tex}")
    c = deniability_constant(delta, n_eff)
    return 1.0 + c * np.sqrt(params.v) / (params.g_sigma * e_tex)


def wstar_from_measured_etex(e_tex_meas: float, v: float, *, delta: float, n_eff: float) -> float:
    """w*(.) when the texture input is the MEASURED latent energy ``E_tex_meas``.

    ``E_tex_meas`` (post-decode/post-blur inversion-source latent) ALREADY carries the decoder/blur
    high-band attenuation that ``g_sigma`` models, so the channel signal slope IS ``E_tex_meas``:
    ``w* = 1 + c(delta,N_eff) * sqrt(v) / E_tex_meas``. A separate plugin ``g_sigma`` here would
    double-count the PSF. ``g_sigma`` is still estimated and its provenance still gates the
    predictive track in the caller; it just must not re-enter the bound value. This is the slope
    proxy of Section III-D (the re-encoded E_tex already includes the degradation attenuation).

    (Note: ``E_tex_meas`` is a SQUARED high-pass energy, so an amplitude-linear response would
    scale with ``sqrt(E_tex_meas)``; the proxy is uncalibrated in scale, which is why the paper
    treats w* as a sensitivity quantity.)
    """
    if e_tex_meas <= 0 or not np.isfinite(e_tex_meas):
        raise ValueError(f"e_tex_meas must be positive and finite, got {e_tex_meas}")
    if v <= 0 or not np.isfinite(v):
        raise ValueError(f"v must be positive and finite, got {v}")
    c = deniability_constant(delta, n_eff)
    return 1.0 + c * np.sqrt(v) / e_tex_meas


def channel_tv(w: float, e_tex: float, params: WStarParams, *, n_eff: float = 1.0) -> float:
    """Analytic TV (Eqs. (13)-(14)) of H0 (w=1) vs H1 (w), GENERATION-side ``e_tex``.

    Signal slope ``g_sigma * e_tex``. For measured ``E_tex_meas`` use
    :func:`channel_tv_from_measured_etex` (slope = E_tex_meas, no separate g_sigma).
    """
    if not np.isfinite(w):
        raise ValueError(f"w must be finite, got {w}")
    if e_tex <= 0 or not np.isfinite(e_tex):
        raise ValueError(f"e_tex must be positive and finite, got {e_tex}")
    if not (np.isfinite(n_eff) and n_eff > 0):
        raise ValueError(f"n_eff must be positive, got {n_eff}")
    mu = params.g_sigma * (w - 1.0) * e_tex
    z = np.sqrt(n_eff) * abs(mu) / (2.0 * np.sqrt(params.v))
    return float(2.0 * norm.cdf(z) - 1.0)


def channel_tv_from_measured_etex(
    w: float, e_tex_meas: float, v: float, *, n_eff: float = 1.0
) -> float:
    """Analytic TV between H0 (w=1) and H1 (w) with the MEASURED-energy signal slope ``E_tex_meas``.

    Signal mean ``mu(w) = E_tex_meas * (w-1)`` (no separate ``g_sigma``: it is already inside
    ``E_tex_meas``). At ``w = wstar_from_measured_etex(...)`` this returns exactly ``delta``.
    """
    if not np.isfinite(w):
        raise ValueError(f"w must be finite, got {w}")
    if e_tex_meas <= 0 or not np.isfinite(e_tex_meas):
        raise ValueError(f"e_tex_meas must be positive and finite, got {e_tex_meas}")
    if v <= 0 or not np.isfinite(v):
        raise ValueError(f"v must be positive and finite, got {v}")
    if not (np.isfinite(n_eff) and n_eff > 0):
        raise ValueError(f"n_eff must be positive, got {n_eff}")
    mu = e_tex_meas * (w - 1.0)
    z = np.sqrt(n_eff) * abs(mu) / (2.0 * np.sqrt(v))
    return float(2.0 * norm.cdf(z) - 1.0)
