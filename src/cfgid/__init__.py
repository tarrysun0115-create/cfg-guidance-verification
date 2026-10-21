"""cfgid: dual-inversion forensics of classifier-free guidance (CFG).

No model training: frozen diffusion backbones are used only for generation and for the
dual-branch (prompt / null-text) DDIM inversion of Section III-B, whose residuals give the scalar
channel rho and the pair summary (rho, m) of Section III-C. Subpackages:

  channel/       inversion backends, rho / E_tex extraction, decoder PSF (sigma) calibration
  featurestore/  on-disk per-image feature store (compact scalars + full residuals)
  sweep/         main-grid (p0) generation/inversion driver, evidence collection and analysis
  bounds/        Gaussian-location formulas of Section III-E (TV, w*) and the Monte-Carlo TV check
  detector/      AUC/ROC primitives, white-box likelihood-ratio detector, sufficiency probe
  align/         knee-vs-w* alignment regression and the preregistered three-state gate
  baselines/     nuisance and CLIP prompt-adherence negative controls
  attack/        surrogate purification attack (Section IV-E7)

See README.md for installation and the main commands.
"""

__version__ = "0.0.0"
