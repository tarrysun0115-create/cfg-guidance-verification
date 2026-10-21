"""The rho forensic channel: dual-branch inversion -> per-patch rho, E_tex, sigma and QF helpers.

Implements the channel of Section III (Fig. 2): the DDIM inversion backends (``inversion``,
Eq. (1)), the per-patch statistic and image summaries (``rho``, ``e_tex``, ``features``;
Eqs. (2)-(8)), the controlled degradations (JPEG in ``generate``, ``psf_blur``, ``postproc``),
the decoder PSF calibration (``sigma_calib``) and the g_sigma / v parameter estimators
(``params``). The FLUX flow backend is kept for completeness but is not used in the paper.
"""
