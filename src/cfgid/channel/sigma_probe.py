"""Nominal per-decoder sigma placeholders and the knee-vs-sigma direction check.

These values are a MODEL-LEVEL probe, not a causal or calibrated PSF constant: decoders differ in
many ways at once (denoiser, training distribution, latent scale, prompt adherence, g_sigma), so a
same-direction knee shift across decoders does not by itself attribute to the decoder PSF width.
The calibrated width comes from ``channel.sigma_calib`` (``outputs/sigma_calib/<decoder>.json``),
which supersedes these values wherever it exists. The nominal values only carry a qualitative
ordering (sd15 wider than sdxl, matching the calibration order); they are written as the per-row
``sigma`` metadata by ``sweep.run_p0`` and used by ``sweep.evidence`` only as an ordering-only
fallback when no calibration artifact exists. ``sigma_direction_consistent`` is one condition of
the preregistered knee/w* alignment gate in ``sweep.analyze`` (inconclusive in Section IV-E7).
"""

from __future__ import annotations

PROBE_LABEL = "model-level probe (not a calibrated decoder PSF; see channel.sigma_calib)"

# Ordering-only placeholders; calibrated artifacts supersede them on the gate path. Only the
# sd15 > sdxl ordering was aligned with the calibration; the other entries are provisional
# (pixart_sigma and hunyuandit share the SDXL VAE, so they sit near sdxl). Keys are the decoder
# names in configs/models.yaml.
_NOMINAL_SIGMA: dict[str, float] = {
    "sdxl": 1.0,
    "pixart_sigma": 1.05,
    "hunyuandit": 1.08,
    "playground25": 1.1,
    "sd21": 1.2,
    "sd15": 1.3,
    "flux1-dev": 1.6,
}


def nominal_sigma(decoder: str) -> float:
    """Ordering-only nominal sigma for ``decoder`` (KeyError if the decoder has no entry)."""
    if decoder not in _NOMINAL_SIGMA:
        raise KeyError(f"no nominal sigma for decoder '{decoder}' (have {sorted(_NOMINAL_SIGMA)})")
    return _NOMINAL_SIGMA[decoder]


def sigma_direction_consistent(
    knee_by_decoder: dict[str, float], sigma_by_decoder: dict[str, float]
) -> bool:
    """Larger sigma -> larger w* -> knee at higher w. Require pairwise concordance of knee vs sigma.

    Returns False if fewer than two decoders are comparable, and also False if NO pair makes a
    directional prediction (every comparable pair has tied sigma). Tied sigma across ALL pairs is
    ABSENCE of cross-decoder evidence, symmetric to a tied knee, and must NOT satisfy the gate's
    same-direction condition by vacuous truth.
    """
    decoders = sorted(set(knee_by_decoder) & set(sigma_by_decoder))
    if len(decoders) < 2:
        return False
    made_prediction = False
    for i in range(len(decoders)):
        for j in range(i + 1, len(decoders)):
            di, dj = decoders[i], decoders[j]
            ds = sigma_by_decoder[di] - sigma_by_decoder[dj]
            dk = knee_by_decoder[di] - knee_by_decoder[dj]
            if ds == 0.0:
                continue  # sigmas tied -> this pair makes no directional prediction
            made_prediction = True
            # A tied knee (dk == 0) is ABSENCE of evidence, not agreement: a flat sigma probe must
            # NOT satisfy the gate's same-direction condition. Require strict concordance.
            if dk == 0.0 or (ds > 0) != (dk > 0):
                return False
    return made_prediction
