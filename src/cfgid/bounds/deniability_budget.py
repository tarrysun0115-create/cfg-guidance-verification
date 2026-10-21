"""Attack-to-w* bridge: the model-implied rho mean-shift gap between guidance w0 and w*.

The deniability bound (``wstar_p0``) gives ``w*(sigma, E_tex, QF, N_eff, delta)``: below it the
Gaussian rho-channel ``TV <= delta`` (Corollary 1). Under the Gaussian-location model with a
variance ``v`` shared by H0 and H1, guidance ``w0 > w*`` implies a channel mean shift
``mu(w0) = g_sigma*(w0-1)*E_tex`` away from H0. The model-implied mean-shift gap to the frontier
``w*`` is

    required_rho_shift = max(0, signal_shift(w0) - mu(w*)) = max(0, g_sigma*(w0 - w*)*E_tex),

and at the frontier ``mu(w*) = c(delta, N_eff) * sqrt(v)`` exactly (the bound's closed form). The
gap concerns the model's rho MEAN, not individual images: moving one image's rho by this amount
gives no per-image undetectability guarantee, and the bound's TV statement covers the rho channel
only. ``scripts/analysis/run_deniability_attack.py`` can record the gap next to the empirical
target (``--bound-params``) and compare the MEASURED rho shift + L2 distortion against it; the
stored Section IV-E7 run used the empirical target only. Pure numpy/scipy; the channel is
sign-agnostic here (it returns the signed mean shift; |.| is what the bound's TV uses).
"""

from __future__ import annotations

import numpy as np

from cfgid.bounds.wstar_p0 import WStarParams, deniability_constant, wstar, wstar_from_measured_etex


def signal_shift(w: float, e_tex: float, params: WStarParams) -> float:
    """Channel mean shift ``mu(w) = g_sigma*(w-1)*E_tex`` for a GENERATION-side ``e_tex``.

    For the MEASURED latent energy ``E_tex_meas`` (signal slope already attenuated) use
    :func:`signal_shift_from_measured_etex`, which does not re-multiply ``g_sigma``.
    """
    if not np.isfinite(w):
        raise ValueError(f"w must be finite, got {w}")
    if e_tex <= 0 or not np.isfinite(e_tex):
        raise ValueError(f"e_tex must be positive and finite, got {e_tex}")
    return float(params.g_sigma * (w - 1.0) * e_tex)


def signal_shift_from_measured_etex(w: float, e_tex_meas: float) -> float:
    """Channel mean shift ``mu(w) = E_tex_meas*(w-1)`` (measured slope; no separate g_sigma)."""
    if not np.isfinite(w):
        raise ValueError(f"w must be finite, got {w}")
    if e_tex_meas <= 0 or not np.isfinite(e_tex_meas):
        raise ValueError(f"e_tex_meas must be positive and finite, got {e_tex_meas}")
    return float(e_tex_meas * (w - 1.0))


def deniable_frontier_signal(*, delta: float, n_eff: float, params: WStarParams) -> float:
    """``mu(w*) = c(delta, N_eff) * sqrt(v)``: the rho-mean shift AT the deniability frontier."""
    return float(deniability_constant(delta, n_eff) * np.sqrt(params.v))


def required_rho_shift(
    w0: float, e_tex: float, params: WStarParams, *, delta: float, n_eff: float
) -> float:
    """Model-implied rho mean-shift gap between guidance ``w0`` and the frontier ``w*``.

    Returns ``max(0, signal_shift(w0) - mu(w*))``: ``g_sigma*(w0 - w*)*E_tex`` for ``w0 > w*`` and
    ``0.0`` when ``w0 <= w*``. The gap is computed under the Gaussian-location model with a variance
    shared by H0 and H1. It is a distribution-level quantity: a zero gap means the model-implied
    rho mean already lies within the ``TV <= delta`` frontier of the rho channel, and a positive
    gap closed for one image gives no per-image undetectability guarantee.
    """
    if not np.isfinite(w0):
        raise ValueError(f"w0 must be finite, got {w0}")
    w_star = wstar(e_tex, params, delta=delta, n_eff=n_eff)
    if w0 <= w_star:
        return 0.0
    return float(params.g_sigma * (w0 - w_star) * e_tex)


def required_rho_shift_from_measured_etex(
    w0: float, e_tex_meas: float, v: float, *, delta: float, n_eff: float
) -> float:
    """``required_rho_shift`` for a MEASURED latent energy ``E_tex_meas`` (slope = E_tex_meas).

    Returns ``E_tex_meas*(w0 - w*)`` for ``w0 > w*`` (``w*`` from :func:`wstar_from_measured_etex`),
    else ``0.0``. No separate ``g_sigma`` -- it is already inside ``E_tex_meas`` (avoids the PSF
    double-count). The frontier signal ``mu(w*) = c*sqrt(v)`` is g_sigma-free, so unchanged. The
    same model and the same limits apply: a model-implied mean-shift gap, with no per-image
    undetectability guarantee.
    """
    if not np.isfinite(w0):
        raise ValueError(f"w0 must be finite, got {w0}")
    if e_tex_meas <= 0 or not np.isfinite(e_tex_meas):
        raise ValueError(f"e_tex_meas must be positive and finite, got {e_tex_meas}")
    if v <= 0 or not np.isfinite(v):
        raise ValueError(f"v must be positive and finite, got {v}")
    w_star = wstar_from_measured_etex(e_tex_meas, v, delta=delta, n_eff=n_eff)
    if w0 <= w_star:
        return 0.0
    return float(e_tex_meas * (w0 - w_star))
