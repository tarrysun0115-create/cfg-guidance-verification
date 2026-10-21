"""Channel-parameter estimators (g_sigma, v) with per-parameter provenance.

``gsigma`` estimates the heuristic gain g_sigma = 1 / (1 + sigma^2) of the Gaussian-location
shape proxy (Section III-D) and the H0 null variance v; ``vmodel`` fits an analytic v(QF).
Valid g_sigma provenances (must match ``cfgid.bounds.wstar_p0.G_SIGMA_PROVENANCE``):

  * unlabeled_null_adapted — plugin g(sigma_hat) + H0-only null stats (predictive track)
  * synthetic_fit          — slope on self-generated sweeps, w known by construction (predictive)
  * temperature_calibrated — slope on H1-labeled data (diagnostic only; gate must not pass on it)
"""

from cfgid.channel.params.gsigma import (
    PREDICTIVE_PROVENANCE,
    GSigmaEstimate,
    estimate_v_null,
    null_adapted_gsigma,
    plugin_gain,
    synthetic_fit_gsigma,
)

__all__ = [
    "PREDICTIVE_PROVENANCE",
    "GSigmaEstimate",
    "estimate_v_null",
    "null_adapted_gsigma",
    "plugin_gain",
    "synthetic_fit_gsigma",
]
