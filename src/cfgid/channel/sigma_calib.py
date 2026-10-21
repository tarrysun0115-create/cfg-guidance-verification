"""Decoder PSF sigma calibration: impulse + checkerboard latents through the VAE decoder.

Produces: each decoder's intrinsic PSF width sigma_core, which sets the effective blur width
sigma_eff = sqrt(sigma_core^2 + sigma_blur^2) used to order cells (Section IV-A) and the
heuristic gain g_sigma = 1 / (1 + sigma^2) (Section III-D). The cross-decoder PSF ordering is
read on the three distinct-VAE decoders (SD1.5, SD2.1, SDXL); PixArt-Sigma and HunyuanDiT share
the SDXL VAE.
Inputs: the decoder's local VAE weights (``configs/models.yaml`` + ``configs/paths.yaml``) and
``configs/p0.yaml`` (``evidence.sigma_calib`` gate, ``grid.resolution``, ``grid.sigma_blur``);
runs on CUDA, MPS or CPU. No images and no feature store are needed.
Outputs: ``outputs/sigma_calib/<decoder>.json`` with ``sigma_hat`` (source-averaged width),
bootstrap CI, ``scalar_ok`` + ``reasons`` (scalarization gate), ``per_source`` fits and
``per_sigma_blur`` (sigma_eff per blur level from the impulse core width; key "0" is the bare
core). ``cfgid.sweep.evidence`` reads this file.
Usage: ``python -m cfgid.channel.sigma_calib --decoder sd15`` (likewise sd21, sdxl,
pixart_sigma).

Pushes synthetic latents (single-pixel impulses, +-1 block checkerboards) through the frozen
decoder VAE, measures the effective point-spread width ``sigma_hat`` (in DECODED-IMAGE pixels)
with pure-numpy half-max/moment/erf fitters, and applies the PSF scalarization gate from
``configs/p0.yaml`` ``evidence.sigma_calib``. A failed gate is a DOWNGRADE
(``scalar_ok=False`` -> report "effective sigma" only), never an exception.

Latent scaling semantics (verified against the installed diffusers source,
``diffusers/models/autoencoders/autoencoder_kl.py``): ``AutoencoderKL.decode`` applies NO
latent scaling itself — SD pipelines divide latents by ``vae.config.scaling_factor`` before
calling ``decode``. The encode side in ``sd_ddim._encode_image_to_latent`` multiplies the
posterior mean by ``scaling_factor`` to produce the UNet-space latent; this module therefore
builds its calibration latents in that same UNet (scaled) latent space and decodes via
``vae.decode(latent / scaling_factor)`` — the exact inverse.

Note: the VAE decoder is nonlinear, so "decode(impulse) - decode(zeros)" is an
effective (linearized) PSF probe, not an exact convolution kernel. Non-Gaussian responses are
expected to fail the r2 gate and downgrade — that is the intended behavior, not a bug.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from cfgid.channel.inversion.base import BackendUnavailableError, torch_dtype
from cfgid.config import ModelSpec
from cfgid.numerics import as_finite_float32
from cfgid.p0config import SigmaCalibConfig

if TYPE_CHECKING:  # heavy import only for type-checkers, never at runtime
    import torch

# Known calibration sources (configs/p0.yaml evidence.sigma_calib.sources must subset this).
KNOWN_SOURCES: tuple[str, ...] = ("impulse", "checkerboard")

# Checkerboard block edge in latent pixels (fixed at 4).
_CHECKER_BLOCK = 4

# Moment estimation ignores mass below this fraction of the peak (kills far-field noise).
# The ~2.4% downward bias of this floor holds ONLY when the measurement window covers >=3 sigma;
# when sigma approaches the window half-width, WINDOW truncation dominates (50-70% bias) — see
# the censoring guard below.
_MOMENT_FLOOR_FRAC = 0.01

# A checkerboard edge-spread measurement is structurally right-censored by its +-half-window
# profile: a true sigma at/above ~0.45 * half-window saturates toward a sigma-independent
# constant. Such a source is a LOWER BOUND only — it stays in the report and forces the
# scalarization downgrade, but is EXCLUDED from the sigma_hat estimate.
_CENSOR_FRAC = 0.45

# Lower clamp so a single-pixel response cannot produce sigma=0 -> division by zero in the fit.
_SIGMA_FLOOR = 1e-3

# Bootstrap resamples for the CI over per-measurement sigmas.
_BOOTSTRAP_RESAMPLES = 256

# Impulse amplitude in UNet (scaled) latent units — a few latent-std, large enough to clear
# the decoder noise floor without leaving the roughly linear regime.
_IMPULSE_AMPLITUDE = 3.0

# The moment window is sized to the PSF CORE as a multiple of its half-max radius (FWHM/2), so the
# far-field skirt -- which survives the 1% moment floor and otherwise dominates the r^2-weighted
# second moment on a frame-sized window -- is excluded. r_halfmax ~ 1.18*sigma, so mult=5 gives a
# window of ~+-6 sigma (captures the core, drops the skirt) and adapts per decoder (sd15 ~8px core
# -> ~48px half; sdxl ~4px core -> ~24px half), matching the window where each fits cleanly.
_CORE_WINDOW_HALFMAX_MULT = 5
_MIN_CORE_HALF = 8  # floor so a near-delta core still gives the moment fit a >=5px window side

# Half-max radius -> Gaussian sigma: a Gaussian falls to half its peak at r = sigma * sqrt(2 ln 2).
_FWHM_TO_SIGMA = 1.0 / math.sqrt(2.0 * math.log(2.0))

# SD/SDXL VAE spatial upscale (3 downsample blocks = 8x). The calibration window is sized so its
# decoded resolution MATCHES the production channel (grid.resolution), so the bare-PSF core is
# measured at the same geometry run_p0 blurs at (sigma_eff at higher sigma_blur is then analytic).
_VAE_SPATIAL_UPSCALE = 8

_erf = np.frompyfunc(math.erf, 1, 1)  # vectorized stdlib erf (pure numpy + stdlib, no scipy)


@dataclass(frozen=True)
class SourceFit:
    """Per-source calibration fit: sigma in decoded-image pixels + Gaussian/erf fit quality."""

    source: str
    sigma: float
    r2: float


@dataclass(frozen=True)
class SigmaCalibResult:
    """Decoder sigma calibration outcome incl. the scalarization-gate decision."""

    decoder: str
    sigma_hat: float
    ci_lo: float
    ci_hi: float
    scalar_ok: bool
    reasons: list[str]
    per_source: dict[str, SourceFit]
    # Effective sigma per controlled-blur level = sqrt(core^2 + sigma_blur^2), keyed by
    # sigma_blur (0.0 = bare PSF core). The sigma_eff axis; empty when impulse is not probed.
    per_sigma_blur: dict[float, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name, value in (
            ("sigma_hat", self.sigma_hat),
            ("ci_lo", self.ci_lo),
            ("ci_hi", self.ci_hi),
        ):
            if not (math.isfinite(float(value)) and float(value) > 0.0):
                raise ValueError(f"{name} must be positive and finite, got {value}")
        if not self.ci_lo <= self.sigma_hat <= self.ci_hi:
            raise ValueError(
                f"sigma CI [{self.ci_lo}, {self.ci_hi}] must bracket sigma_hat={self.sigma_hat}"
            )
        for name, fit in self.per_source.items():
            if fit.source != name:
                raise ValueError(
                    f"per_source key {name!r} does not match fit.source={fit.source!r}"
                )
            if not (math.isfinite(fit.sigma) and fit.sigma > 0.0):
                raise ValueError(
                    f"source '{name}' sigma must be positive and finite, got {fit.sigma}"
                )
            if not math.isfinite(fit.r2):
                raise ValueError(f"source '{name}' r2 must be finite, got {fit.r2}")
        for sb, eff in self.per_sigma_blur.items():
            if not (math.isfinite(float(sb)) and float(sb) >= 0.0):
                raise ValueError(f"per_sigma_blur key must be finite and >= 0, got {sb}")
            # to_dict serializes the key as f"{sb:g}" and evidence.sigma_eff looks it back up by
            # exact float; a key not %g-round-trip-stable would collide/drop SILENTLY on save. The
            # config validator _sigma_blur_levels enforces this at load too -- guard both ends.
            if float(f"{float(sb):g}") != float(sb):
                raise ValueError(
                    f"per_sigma_blur key {sb!r} is not %g-stable (round-trips to "
                    f"{float(f'{float(sb):g}')!r}); it would collide/drop on JSON serialization"
                )
            if not (math.isfinite(float(eff)) and float(eff) > 0.0):
                raise ValueError(f"per_sigma_blur[{sb}] must be positive and finite, got {eff}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "decoder": self.decoder,
            "sigma_hat": float(self.sigma_hat),
            "ci_lo": float(self.ci_lo),
            "ci_hi": float(self.ci_hi),
            "scalar_ok": bool(self.scalar_ok),
            "reasons": list(self.reasons),
            "per_source": {
                name: {"source": fit.source, "sigma": float(fit.sigma), "r2": float(fit.r2)}
                for name, fit in sorted(self.per_source.items())
            },
            # JSON keys must be strings; serialize sigma_blur with a stable %g format.
            "per_sigma_blur": {
                f"{float(sb):g}": float(eff) for sb, eff in sorted(self.per_sigma_blur.items())
            },
        }

    def save(self, path: Path | str) -> None:
        """Write the result as JSON. ``allow_nan=False``: non-finite values are invalid JSON
        (RFC 8259) and must never serialize silently (no silent NaN)."""
        out = Path(path)
        text = json.dumps(
            self.to_dict(), sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False
        )
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text + "\n", encoding="utf-8")


# --------------------------------------------------------------------------- #
# pure-numpy fitters (no torch, no scipy) — unit-testable without weights
# --------------------------------------------------------------------------- #
def fit_gaussian_sigma(response: np.ndarray) -> tuple[float, float]:
    """Estimate isotropic Gaussian sigma of a 2D impulse response + fit r2.

    DORMANT: superseded on the PSF path by ``core_sigma_fit`` (half-max width, skirt-robust). The
    second moment here is inflated 2-10x by a non-Gaussian VAE PSF's far-field skirt. Retained as a
    validated alternative estimator (and unit-tested), not used by ``calibrate_decoder_sigma``.

    ``response`` is a background-subtracted, (approximately) non-negative 2D impulse response;
    small negative residuals from background subtraction are clipped to zero. Sigma comes from
    second moments around the intensity-weighted centroid (mass below 1% of the peak is
    ignored); r2 compares a least-squares-amplitude Gaussian surface at that (centroid, sigma)
    against the response, so non-Gaussian shapes (e.g. rings) score low.
    """
    resp = as_finite_float32("response", response).astype(np.float64)
    if resp.ndim != 2:
        raise ValueError(f"response must be 2D, got shape {resp.shape}")
    if min(resp.shape) < 5:
        raise ValueError(f"response too small for a moment fit, got shape {resp.shape}")
    resp = np.clip(resp, 0.0, None)
    peak = float(resp.max())
    if peak <= 0.0:
        raise ValueError("response has no positive mass (peak <= 0 after clipping)")

    weights = np.where(resp >= _MOMENT_FLOOR_FRAC * peak, resp, 0.0)
    total = float(weights.sum())
    yy, xx = np.indices(resp.shape, dtype=np.float64)
    cy = float((weights * yy).sum() / total)
    cx = float((weights * xx).sum() / total)
    var_y = float((weights * (yy - cy) ** 2).sum() / total)
    var_x = float((weights * (xx - cx) ** 2).sum() / total)
    sigma = max(math.sqrt(max(0.5 * (var_x + var_y), 0.0)), _SIGMA_FLOOR)

    gauss = np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2.0 * sigma * sigma))
    amp = float((gauss * resp).sum() / (gauss * gauss).sum())
    ss_res = float(((resp - amp * gauss) ** 2).sum())
    ss_tot = float(((resp - resp.mean()) ** 2).sum())
    if ss_tot <= 0.0:
        raise ValueError("response is constant; r2 undefined")
    r2 = 1.0 - ss_res / ss_tot

    if not (math.isfinite(sigma) and math.isfinite(r2)):
        raise ValueError(f"gaussian fit produced non-finite estimates (sigma={sigma}, r2={r2})")
    return float(sigma), float(r2)


def core_sigma_fit(response: np.ndarray) -> tuple[float, float]:
    """Effective Gaussian sigma of a 2D PSF response from its HALF-MAX core radius + Gaussian r2.

    ``sigma = r_halfmax / sqrt(2 ln 2)``, where ``r_halfmax`` is the (sub-pixel) radius at which the
    radial profile around the peak first crosses half the peak. The half-max crossing is IMMUNE to
    the low-amplitude far-field skirt of a non-Gaussian VAE decoder PSF, which inflates a
    second-moment OR an MTF-slope estimate ~2-10x and makes it window-edge-dependent (the skirt sits
    far below half-peak, so it never moves the crossing). r2 scores a peak-centered Gaussian of that
    sigma against the response over a core window (low r2 => strongly non-Gaussian PSF -> the
    scalarization gate downgrades). This is the sigma_core estimator behind the sigma_eff axis.
    """
    resp = as_finite_float32("response", response).astype(np.float64)
    if resp.ndim != 2 or min(resp.shape) < 5:
        raise ValueError(f"response must be 2D with side >= 5, got shape {resp.shape}")
    peak = float(resp.max())
    if not (math.isfinite(peak) and peak > 0.0):
        raise ValueError("response has no positive peak for a half-max estimate")

    py, px = np.unravel_index(int(np.argmax(resp)), resp.shape)
    edge_radius = float(min(py, px, resp.shape[0] - 1 - py, resp.shape[1] - 1 - px))
    if edge_radius < 2.0:
        raise ValueError(
            "response peak is too close to the image edge for a radial half-max estimate"
        )
    yy, xx = np.indices(resp.shape)
    r = np.sqrt((yy - int(py)) ** 2 + (xx - int(px)) ** 2)
    # Only complete rings are valid for the radial mean. If the half-max crossing would require a
    # clipped ring, raise instead of returning a border-biased width.
    rmax = int(math.floor(edge_radius))
    half = 0.5 * peak
    prof = np.full(rmax, np.nan)
    rmean = np.full(rmax, np.nan)  # AREA-WEIGHTED mean radius of each annulus (NOT k+0.5)
    for k in range(rmax):
        mask = (r >= k) & (r < k + 1)
        if mask.any():
            prof[k] = float(resp[mask].mean())
            rmean[k] = float(r[mask].mean())
    below = np.where(np.isfinite(prof) & (prof < half))[0]
    if below.size == 0 or int(below[0]) == 0:
        raise ValueError("response has no resolvable half-max core (peak too sharp or too wide)")
    k = int(below[0])
    if float(k + 1) > edge_radius:
        raise ValueError("response half-max crossing is clipped by the image edge")
    p_hi, p_lo = float(prof[k - 1]), float(prof[k])
    frac = (p_hi - half) / (p_hi - p_lo) if p_hi > p_lo else 0.0
    # Interpolate on the TRUE per-ring mean radii, not index midpoints (k-0.5, k+0.5): on the
    # integer-pixel grid the area-weighted mean radius of annulus [j, j+1) is ~j+0.21..j+0.33, not
    # j+0.5, and prof[k] is the value AT that mean radius -- using k+/-0.5 abscissae biased sigma
    # high ~7-8% at small cores (sdxl ~3.6px), distorting the sigma-dependent cross-decoder ratio.
    r_halfmax = float(rmean[k - 1] + (rmean[k] - rmean[k - 1]) * frac)
    sigma = max(r_halfmax * _FWHM_TO_SIGMA, _SIGMA_FLOOR)

    core_half = max(_MIN_CORE_HALF, int(round(_CORE_WINDOW_HALFMAX_MULT * max(r_halfmax, 1.0))))
    y0, y1 = max(0, int(py) - core_half), min(resp.shape[0], int(py) + core_half + 1)
    x0, x1 = max(0, int(px) - core_half), min(resp.shape[1], int(px) + core_half + 1)
    win = resp[y0:y1, x0:x1]
    wy, wx = np.indices(win.shape, dtype=np.float64)
    gcy, gcx = float(int(py) - y0), float(int(px) - x0)
    gauss = np.exp(-((wy - gcy) ** 2 + (wx - gcx) ** 2) / (2.0 * sigma * sigma))
    amp = float((gauss * win).sum() / (gauss * gauss).sum())
    ss_res = float(((win - amp * gauss) ** 2).sum())
    ss_tot = float(((win - win.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0.0 else 0.0

    if not (math.isfinite(sigma) and sigma > 0.0 and math.isfinite(r2)):
        raise ValueError(f"core sigma fit produced invalid estimates (sigma={sigma}, r2={r2})")
    return float(sigma), float(r2)


def edge_spread_sigma(edge_profile: np.ndarray) -> tuple[float, float]:
    """Estimate sigma from a 1D edge-spread (erf-transition) profile + erf-fit r2.

    The line-spread function (discrete derivative of the profile) of an erf edge is a
    box-integrated Gaussian; sigma comes from its second moment (with the 1/12 box-integration
    variance correction), orientation-agnostic via ``|diff|``. r2 then scores the least-squares
    affine erf model ``a + b * Phi((x - mu) / sigma)`` against the profile itself.
    """
    prof = as_finite_float32("edge_profile", edge_profile).astype(np.float64)
    if prof.ndim != 1:
        raise ValueError(f"edge_profile must be 1D, got shape {prof.shape}")
    if prof.size < 8:
        raise ValueError(f"edge_profile too short for an erf fit, got {prof.size} samples")

    lsf = np.abs(np.diff(prof))
    peak = float(lsf.max())
    if peak <= 0.0:
        raise ValueError("edge_profile is flat (no transition)")
    weights = np.where(lsf >= _MOMENT_FLOOR_FRAC * peak, lsf, 0.0)
    total = float(weights.sum())
    x_mid = np.arange(lsf.size, dtype=np.float64) + 0.5  # diff samples live at midpoints
    mu = float((weights * x_mid).sum() / total)
    var = float((weights * (x_mid - mu) ** 2).sum() / total) - 1.0 / 12.0
    sigma = max(math.sqrt(max(var, 0.0)), _SIGMA_FLOOR)

    xs = np.arange(prof.size, dtype=np.float64)
    z = (xs - mu) / (sigma * math.sqrt(2.0))
    phi = 0.5 * (1.0 + np.asarray(_erf(z), dtype=np.float64))
    basis = np.stack([np.ones_like(phi), phi], axis=1)
    coef, _, _, _ = np.linalg.lstsq(basis, prof, rcond=None)
    fitted = basis @ coef
    ss_res = float(((prof - fitted) ** 2).sum())
    ss_tot = float(((prof - prof.mean()) ** 2).sum())
    if ss_tot <= 0.0:
        raise ValueError("edge_profile is constant; r2 undefined")
    r2 = 1.0 - ss_res / ss_tot

    if not (math.isfinite(sigma) and math.isfinite(r2)):
        raise ValueError(f"erf fit produced non-finite estimates (sigma={sigma}, r2={r2})")
    return float(sigma), float(r2)


# MTF radial band used for the Gaussian-MTF fit: avoid the DC bins (few pixels, noisy) and the
# high-frequency noise floor. The fit is over MTF in (_MTF_BAND_LO, _MTF_BAND_HI).
_MTF_BAND_LO = 0.15
_MTF_BAND_HI = 0.9


def mtf_effective_sigma(response: np.ndarray) -> tuple[float, float]:
    """Defensible effective Gaussian sigma from the radial MTF of a 2D PSF response + Gaussian r2.

    DORMANT: superseded on the PSF path by ``core_sigma_fit`` (half-max width). The MTF slope is
    also skirt-sensitive (a far-field skirt flattens the low-frequency MTF and inflates sigma), and
    core-cropping to suppress it trades away the frequency resolution the band fit needs. Retained
    as a validated alternative estimator (and unit-tested), not used by ``calibrate_decoder_sigma``.

    The MTF is the DC-normalized magnitude of the 2D FFT of the (non-negative) impulse response,
    radially averaged to MTF(f) over spatial frequency ``f`` (cycles/pixel). For a Gaussian PSF of
    width sigma, ``MTF(f) = exp(-2 pi^2 sigma^2 f^2)``, so ``log MTF`` is linear in ``f^2`` through
    the origin. We fit that slope by weighted least squares (weight MTF^2 to downweight the noisy
    low-MTF tail) over the band MTF in (0.15, 0.9), giving a principled effective sigma EVEN for
    non-Gaussian PSFs (where a direct spatial Gaussian fit fails). The returned r2 is the Gaussian
    MTF-model fit quality over that band (low r2 flags a strongly non-Gaussian PSF).

    Finite-window guard: a window of side N only resolves frequencies in steps of 1/N. If fewer
    than three independent radial bands land in the MTF fit range, the PSF is too wide for the
    window and the function raises instead of fabricating a number. Within the measurable range,
    finite-window bias is smooth and monotone in the true sigma, so the effective-sigma AXIS (the
    sigma_eff alignment x-axis) keeps the correct ordering.
    """
    resp = as_finite_float32("response", response).astype(np.float64)
    if resp.ndim != 2 or min(resp.shape) < 8:
        raise ValueError(f"response must be 2D with side >= 8, got shape {resp.shape}")
    if float(resp.sum()) <= 0.0:
        raise ValueError("response has no positive mass for an MTF estimate")

    mtf2d = np.abs(np.fft.fftshift(np.fft.fft2(resp)))
    dc = float(mtf2d[resp.shape[0] // 2, resp.shape[1] // 2])
    if dc <= 0.0 or not math.isfinite(dc):
        raise ValueError("degenerate spectrum (non-positive DC)")
    mtf2d /= dc  # explicit DC normalization -> MTF(0)=1 for the PSF response

    yy, xx = np.indices(resp.shape)
    cy, cx = resp.shape[0] // 2, resp.shape[1] // 2
    pixel_radius = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    fy = np.fft.fftshift(np.fft.fftfreq(resp.shape[0]))
    fx = np.fft.fftshift(np.fft.fftfreq(resp.shape[1]))
    freq_y, freq_x = np.meshgrid(fy, fx, indexing="ij")
    freq_radius_sq = freq_y**2 + freq_x**2

    n_bins = int(min(resp.shape)) // 2
    radial = np.empty(n_bins, dtype=np.float64)
    freq_sq = np.empty(n_bins, dtype=np.float64)
    for k in range(n_bins):
        mask = (pixel_radius >= k) & (pixel_radius < k + 1)
        radial[k] = float(mtf2d[mask].mean()) if mask.any() else np.nan
        freq_sq[k] = float(freq_radius_sq[mask].mean()) if mask.any() else np.nan

    band = (
        np.isfinite(radial)
        & np.isfinite(freq_sq)
        & (radial > _MTF_BAND_LO)
        & (radial < _MTF_BAND_HI)
        & (freq_sq > 0.0)
    )
    if int(band.sum()) < 3:
        raise ValueError(
            f"too few MTF band samples ({int(band.sum())}); window too small for this PSF"
        )
    f2 = freq_sq[band]
    log_mtf = np.log(radial[band])
    weights = radial[band] ** 2  # downweight the noisy low-MTF tail
    # Weighted slope of log_mtf vs f^2 through the origin: slope = -2 pi^2 sigma^2.
    denom = float(np.sum(weights * f2 * f2))
    if denom <= 0.0 or not math.isfinite(denom):
        raise ValueError("degenerate MTF fit denominator")
    slope = float(np.sum(weights * f2 * log_mtf) / denom)
    if not (slope < 0.0 and np.isfinite(slope)):
        raise ValueError(f"non-physical MTF slope {slope} (MTF should decay with frequency)")
    sigma = math.sqrt(-slope / (2.0 * math.pi**2))

    model = np.exp(slope * f2)  # exp(-2 pi^2 sigma^2 f^2) at the fitted sigma
    ss_res = float(np.sum((radial[band] - model) ** 2))
    ss_tot = float(np.sum((radial[band] - radial[band].mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0.0 else 0.0
    if not (math.isfinite(sigma) and sigma > 0.0 and math.isfinite(r2)):
        raise ValueError(f"MTF fit produced invalid estimates (sigma={sigma}, r2={r2})")
    return float(sigma), float(r2)


def sigma_blur_sweep(sigma_core: float, sigma_blur_levels: tuple[float, ...]) -> dict[float, float]:
    """sigma_eff(sigma_blur) = sqrt(sigma_core^2 + sigma_blur^2): the decoder's measured bare-PSF
    core width combined with the KNOWN applied Gaussian blur (Gaussian convolution adds variances).

    The forensic channel applies an EXACT Gaussian ``sigma_blur`` (``channel.psf_blur``) on top of
    the decoder PSF, so the effective width at each level is ANALYTIC given the measured core --
    monotone and robust by construction. Re-measuring the blurred impulse response instead is
    fragile: the non-Gaussian VAE skirt corrupts a half-max / MTF / moment re-fit once blur lowers
    the core peak (spurious non-monotone jumps), and the bare core itself is what distinguishes the
    decoders. ``sigma_blur=0`` returns the bare core. Levels must be finite and non-negative.
    """
    if not (math.isfinite(sigma_core) and sigma_core > 0.0):
        raise ValueError(f"sigma_core must be positive and finite, got {sigma_core}")
    out: dict[float, float] = {}
    for sb in sorted(float(s) for s in sigma_blur_levels):
        if not (math.isfinite(sb) and sb >= 0.0):
            raise ValueError(f"sigma_blur level must be finite and non-negative, got {sb}")
        out[float(sb)] = math.sqrt(sigma_core**2 + sb**2)
    return out


def apply_window_censoring(
    per_source: dict[str, SourceFit],
    sigmas_by_source: dict[str, list[float]],
    half_window: float,
) -> tuple[dict[str, SourceFit], list[float], list[str]]:
    """Exclude window-censored checkerboard fits from the sigma_hat estimate (report-only).

    The checkerboard edge-spread profile only spans +-half_window decoded pixels, so a true
    sigma >= ~0.45 * half_window saturates toward a sigma-independent constant (right-censored;
    a LOWER bound, not a measurement). Censored sources stay in ``per_source`` (report + gate)
    but are dropped from the estimation pool. If everything is censored, the full pool is kept
    (with reasons) so sigma_hat stays defined — the scalarization downgrade fires regardless.
    """
    estimate_fits: dict[str, SourceFit] = {}
    pool: list[float] = []
    reasons: list[str] = []
    for source, fit in per_source.items():
        if source == "checkerboard" and fit.sigma >= _CENSOR_FRAC * half_window:
            reasons.append(
                f"source 'checkerboard' window-censored at +-{half_window:.0f}px "
                f"(sigma={fit.sigma:.3f} >= {_CENSOR_FRAC} * half-window): lower bound only; "
                "EXCLUDED from sigma_hat"
            )
            continue
        estimate_fits[source] = fit
        pool.extend(sigmas_by_source.get(source, []))
    if not estimate_fits:  # all sources censored: keep everything, flag loudly
        reasons.append("ALL sources window-censored; sigma_hat is a lower bound only")
        estimate_fits = dict(per_source)
        pool = [s for vals in sigmas_by_source.values() for s in vals]
    return estimate_fits, pool, reasons


def aggregate_sources(
    fits: dict[str, SourceFit],
    sigma_samples: np.ndarray,
    cfg: SigmaCalibConfig,
) -> tuple[float, float, float, bool, list[str]]:
    """Combine per-source fits into (sigma_hat, ci_lo, ci_hi, scalar_ok, reasons).

    ``sigma_hat`` is the mean of per-source sigmas; the CI is the 2.5/97.5 percentile of the
    caller-provided bootstrap samples. Gate failures DOWNGRADE (``scalar_ok=False`` => only an
    "effective sigma" claim survives) — they are recorded as reasons, never raised.
    """
    if not fits:
        raise ValueError("aggregate_sources requires at least one SourceFit")
    for name, fit in fits.items():
        if not (math.isfinite(fit.sigma) and fit.sigma > 0.0):
            raise ValueError(f"source '{name}' has invalid sigma {fit.sigma}")
        if not math.isfinite(fit.r2):
            raise ValueError(f"source '{name}' has non-finite r2 {fit.r2}")
    samples = as_finite_float32("sigma_samples", sigma_samples).astype(np.float64)
    if samples.ndim != 1 or samples.size == 0:
        raise ValueError(f"sigma_samples must be a non-empty 1D array, got shape {samples.shape}")

    sigma_hat = float(np.mean([fit.sigma for fit in fits.values()]))
    ci_lo = float(np.percentile(samples, 2.5))
    ci_hi = float(np.percentile(samples, 97.5))

    reasons: list[str] = []
    for name in sorted(fits):
        fit = fits[name]
        if fit.r2 < cfg.min_r2:
            reasons.append(
                f"source '{name}': fit r2={fit.r2:.3f} < min_r2={cfg.min_r2:.3f} "
                "(non-Gaussian PSF; effective sigma only)"
            )
    rel_width = (ci_hi - ci_lo) / sigma_hat
    if rel_width > cfg.max_rel_ci_width:
        reasons.append(
            f"CI too wide: (ci_hi-ci_lo)/sigma_hat={rel_width:.3f} > "
            f"max_rel_ci_width={cfg.max_rel_ci_width:.3f}"
        )
    names = sorted(fits)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            sig_a, sig_b = fits[names[i]].sigma, fits[names[j]].sigma
            gap = abs(sig_a - sig_b) / (0.5 * (sig_a + sig_b))
            if gap > cfg.max_source_rel_gap:
                reasons.append(
                    f"sources '{names[i]}' vs '{names[j]}' disagree: rel gap={gap:.3f} > "
                    f"max_source_rel_gap={cfg.max_source_rel_gap:.3f}"
                )
    # sigma_hat is an EQUAL-weight mean over sources; the CI is a SAMPLE-COUNT-weighted bootstrap of
    # the flat per-measurement pool. When sources disagree at unequal sample counts the two centers
    # can diverge enough that sigma_hat falls outside [ci_lo, ci_hi] -> __post_init__ would RAISE,
    # violating the module's downgrade contract (a scalarization failure must never throw).
    # That escape is itself a scalarization-quality failure, so DOWNGRADE: widen the CI to bracket
    # the point estimate and record a reason (this sets scalar_ok=False below), leaving the
    # scientific sigma_hat value untouched. (At the production grid both source counts are equal, so
    # mean-of-means == flat-pool mean and this branch never fires.)
    if not (ci_lo <= sigma_hat <= ci_hi):
        reasons.append(
            f"sigma_hat={sigma_hat:.3f} escaped bootstrap CI [{ci_lo:.3f}, {ci_hi:.3f}] "
            "(equal-weight point estimate vs count-weighted interval under unequal source counts); "
            "CI widened to bracket it"
        )
        ci_lo = min(ci_lo, sigma_hat)
        ci_hi = max(ci_hi, sigma_hat)
    return sigma_hat, ci_lo, ci_hi, not reasons, reasons


# --------------------------------------------------------------------------- #
# decoder glue (lazy torch/diffusers) — raises BackendUnavailableError without weights
# --------------------------------------------------------------------------- #
def _load_vae(spec: ModelSpec, device: torch.device) -> Any:
    vae_dir = spec.resolved_local_path() / "vae"
    if not (vae_dir / "config.json").is_file():
        raise BackendUnavailableError(
            f"{spec.name}: VAE weights missing at {vae_dir} (no config.json)"
        )
    try:
        from diffusers import AutoencoderKL
    except ImportError as exc:  # diffusers not installed
        raise BackendUnavailableError(f"diffusers unavailable: {exc}") from exc

    kwargs: dict[str, Any] = {
        "torch_dtype": torch_dtype(spec.dtype),
        "local_files_only": True,
    }
    if spec.variant:
        kwargs["variant"] = spec.variant
    try:
        # Typed as Any: diffusers' ModelMixin.to() overloads confuse pyright, and the
        # caller treats the VAE as opaque model glue anyway.
        vae: Any = AutoencoderKL.from_pretrained(str(vae_dir), **kwargs)
    except (OSError, ValueError) as exc:  # missing/partial snapshot, bad variant, etc.
        raise BackendUnavailableError(
            f"{spec.name}: cannot load VAE from {vae_dir}: {exc}"
        ) from exc
    return vae.to(device).eval()


def _decode_to_gray(vae: Any, latent: torch.Tensor, *, tag: str) -> np.ndarray:
    """Decode a UNet-space (scaled) latent -> float32 grayscale (mean over RGB) numpy image.

    Mirrors (inverts) ``sd_ddim._encode_image_to_latent``: the encode side multiplies the
    posterior mean by ``scaling_factor``, so the decode side divides before ``vae.decode``
    (``AutoencoderKL.decode`` itself applies no scaling — see module docstring).
    """
    import torch

    scaling = float(vae.config.scaling_factor)
    if not (math.isfinite(scaling) and scaling > 0.0):
        raise ValueError(f"vae.config.scaling_factor must be positive/finite, got {scaling}")
    z = (latent / scaling).to(device=next(vae.parameters()).device, dtype=vae.dtype)
    with torch.no_grad():
        decoded = vae.decode(z).sample  # (1, 3, H, W), nominally in [-1, 1]
    gray = decoded[0].float().mean(dim=0).cpu().numpy()
    return as_finite_float32(f"decoded_{tag}", gray)


def _impulse_positions(
    rng: np.random.Generator, latent_size: int, positions: int
) -> list[tuple[int, int]]:
    """Scattered deterministic impulse positions inside the central region (away from borders)."""
    margin = max(2, latent_size // 4)
    coords = np.arange(margin, latent_size - margin)
    if coords.size == 0:
        raise ValueError(f"latent_size={latent_size} too small for impulse placement")
    cells = [(int(y), int(x)) for y in coords for x in coords]
    take = min(positions, len(cells))
    picked = rng.choice(len(cells), size=take, replace=False)
    return [cells[int(k)] for k in picked]


def _measure_impulse(
    vae: Any,
    background: np.ndarray,
    *,
    vae_channels: int,
    latent_size: int,
    positions: list[tuple[int, int]],
    upscale: int,
) -> tuple[list[float], list[float]]:
    import torch

    sigmas: list[float] = []
    r2s: list[float] = []
    for i, (py, px) in enumerate(positions):
        latent = torch.zeros((1, vae_channels, latent_size, latent_size), dtype=torch.float32)
        latent[0, i % vae_channels, py, px] = _IMPULSE_AMPLITUDE  # cycle channels across spikes
        gray = _decode_to_gray(vae, latent, tag=f"impulse_{i}")
        # decode(impulse) - decode(zeros) is NOT a clean compact PSF: the VAE is nonlinear, so an
        # impulse shifts the WHOLE frame's mean (a global DC pedestal) and leaves a low-amplitude
        # far-field skirt. Remove the DC pedestal on the SIGNED response BEFORE rectifying -- abs()
        # first turns the uniform offset into a frame-wide positive FLOOR that inflates any width
        # estimate. After signed de-pedestalling, |.| leaves the localized core with a ~0 far field.
        signed = gray - background
        signed = signed - float(np.median(signed))  # kill the global DC pedestal (signed)
        response = np.abs(signed).astype(np.float64)
        sigma, r2 = core_sigma_fit(response)  # skirt-robust half-max core width (not 2nd moment)
        sigmas.append(sigma)
        r2s.append(r2)
    return sigmas, r2s


def _checkerboard_profiles(
    gray: np.ndarray, *, latent_size: int, upscale: int, rng: np.random.Generator, count: int
) -> list[np.ndarray]:
    """Extract 1D edge-spread profiles across vertical checkerboard edges.

    Each profile averages rows WITHIN one block row (trimmed by ``upscale`` px at each end so
    the horizontal edges do not bleed in), keeping the edge polarity uniform along the band.
    """
    n_blocks = latent_size // _CHECKER_BLOCK
    block_px = _CHECKER_BLOCK * upscale
    half = block_px // 2
    candidates: list[np.ndarray] = []
    for k in range(1, n_blocks):  # interior vertical edges
        edge_x = k * block_px
        for j in range(n_blocks):  # one band per block row
            r0 = j * block_px + upscale
            r1 = (j + 1) * block_px - upscale
            profile = gray[r0:r1, edge_x - half : edge_x + half].mean(axis=0)
            candidates.append(profile.astype(np.float64))
    if not candidates:
        raise ValueError(f"latent_size={latent_size} yields no interior checkerboard edges")
    take = min(max(count, 2), len(candidates))
    picked = rng.choice(len(candidates), size=take, replace=False)
    return [candidates[int(k)] for k in picked]


def _measure_checkerboard(
    vae: Any,
    *,
    vae_channels: int,
    latent_size: int,
    upscale: int,
    rng: np.random.Generator,
    count: int,
) -> tuple[list[float], list[float]]:
    import torch

    bi, bj = np.indices((latent_size, latent_size)) // _CHECKER_BLOCK
    pattern = np.where((bi + bj) % 2 == 0, 1.0, -1.0).astype(np.float32)  # +-1, block 4
    latent = torch.from_numpy(np.broadcast_to(pattern, (1, vae_channels, *pattern.shape)).copy())
    gray = _decode_to_gray(vae, latent, tag="checkerboard")
    sigmas: list[float] = []
    r2s: list[float] = []
    for profile in _checkerboard_profiles(
        gray, latent_size=latent_size, upscale=upscale, rng=rng, count=count
    ):
        sigma, r2 = edge_spread_sigma(profile)
        sigmas.append(sigma)
        r2s.append(r2)
    return sigmas, r2s


def calibrate_decoder_sigma(
    spec: ModelSpec,
    device: torch.device,
    cfg: SigmaCalibConfig,
    *,
    latent_size: int = 32,
    positions: int = 4,
    seed: int = 0,
    sigma_blur_levels: tuple[float, ...] = (0.0,),
) -> SigmaCalibResult:
    """Calibrate the decoder's effective PSF sigma (decoded-image pixels) + scalarization gate.

    Decodes a zeros-latent background, then per-source probes (impulse spikes at scattered
    positions / a +-1 block-4 checkerboard) selected by ``cfg.sources``. Per-measurement
    sigmas feed a bootstrap CI; ``aggregate_sources`` applies the downgrade gate. When ``impulse``
    is probed, also derives ``sigma_eff(sigma_blur) = sqrt(core^2 + sigma_blur^2)`` over
    ``sigma_blur_levels`` (the controlled blur sweep) into ``per_sigma_blur``. Raises
    :class:`BackendUnavailableError` when the VAE weights / diffusers are missing.
    """
    if latent_size < 2 * _CHECKER_BLOCK or latent_size % _CHECKER_BLOCK != 0:
        raise ValueError(
            f"latent_size must be a multiple of {_CHECKER_BLOCK} and >= {2 * _CHECKER_BLOCK}, "
            f"got {latent_size}"
        )
    if positions < 1:
        raise ValueError(f"positions must be >= 1, got {positions}")
    if not cfg.sources:
        raise ValueError("cfg.sources is empty; nothing to calibrate")
    unknown = sorted(set(cfg.sources) - set(KNOWN_SOURCES))
    if unknown:
        raise ValueError(f"unknown sigma_calib sources {unknown} (known: {list(KNOWN_SOURCES)})")

    vae = _load_vae(spec, device)
    import torch

    # One seeded RNG (default seed 0) drives impulse placement, checkerboard profile selection and
    # the bootstrap below, so a calibration run is reproducible.
    rng = np.random.default_rng(seed)
    original_dtype = vae.dtype
    # SDXL's fp16 VAE is numerically unstable; honor force_upcast exactly like sd_ddim does.
    needs_upcast = vae.dtype == torch.float16 and bool(getattr(vae.config, "force_upcast", False))
    try:
        if needs_upcast:
            vae.to(torch.float32)
        zeros = torch.zeros((1, spec.vae_channels, latent_size, latent_size), dtype=torch.float32)
        background = _decode_to_gray(vae, zeros, tag="background")
        if background.shape[0] % latent_size != 0:
            raise ValueError(
                f"decoded size {background.shape} not a multiple of latent_size={latent_size}"
            )
        upscale = background.shape[0] // latent_size

        per_source: dict[str, SourceFit] = {}
        sigmas_by_source: dict[str, list[float]] = {}
        for source in cfg.sources:
            if source == "impulse":
                spots = _impulse_positions(rng, latent_size, positions)
                sigmas, r2s = _measure_impulse(
                    vae,
                    background,
                    vae_channels=spec.vae_channels,
                    latent_size=latent_size,
                    positions=spots,
                    upscale=upscale,
                )
            else:  # "checkerboard" (sources validated above)
                sigmas, r2s = _measure_checkerboard(
                    vae,
                    vae_channels=spec.vae_channels,
                    latent_size=latent_size,
                    upscale=upscale,
                    rng=rng,
                    count=positions,
                )
            # Mean sigma per source; min r2 keeps the gate conservative (one bad position
            # is enough to question the Gaussian-PSF scalarization).
            per_source[source] = SourceFit(
                source=source, sigma=float(np.mean(sigmas)), r2=float(np.min(r2s))
            )
            sigmas_by_source[source] = [float(s) for s in sigmas]
    finally:
        if needs_upcast:
            vae.to(original_dtype)

    half_window = _CHECKER_BLOCK * upscale // 2
    estimate_fits, pool, censor_reasons = apply_window_censoring(
        per_source, sigmas_by_source, half_window
    )
    raw = as_finite_float32("per_measurement_sigmas", np.asarray(pool)).astype(np.float64)
    # Bootstrap the mean of the pooled per-measurement sigmas (_BOOTSTRAP_RESAMPLES replicates).
    boot = rng.choice(raw, size=(_BOOTSTRAP_RESAMPLES, raw.size), replace=True).mean(axis=1)
    sigma_hat, ci_lo, ci_hi, scalar_ok, reasons = aggregate_sources(estimate_fits, boot, cfg)
    if censor_reasons:
        scalar_ok = False  # a censored source means the scalarization claim is not supportable
    # Blur sweep: sigma_eff(sigma_blur) = sqrt(sigma_core^2 + sigma_blur^2), analytic from the
    # measured bare impulse-PSF core. Empty if impulse was not a configured source.
    per_sigma_blur = (
        sigma_blur_sweep(per_source["impulse"].sigma, sigma_blur_levels)
        if "impulse" in per_source
        else {}
    )
    return SigmaCalibResult(
        decoder=spec.name,
        sigma_hat=sigma_hat,
        ci_lo=ci_lo,
        ci_hi=ci_hi,
        scalar_ok=scalar_ok,
        reasons=censor_reasons + reasons,
        per_source=per_source,
        per_sigma_blur=per_sigma_blur,
    )


def main(argv: list[str] | None = None) -> int:
    """CLI: calibrate one decoder's sigma and persist outputs/sigma_calib/<decoder>.json."""
    import argparse

    from cfgid.config import load_models_config
    from cfgid.device import get_device
    from cfgid.p0config import load_p0_config
    from cfgid.paths import ensure_dir, outputs_dir

    parser = argparse.ArgumentParser(prog="cfgid.channel.sigma_calib")
    parser.add_argument("--decoder", required=True)
    # Default 0 -> derive the latent size so the decoded calibration window MATCHES the production
    # grid.resolution (so sigma_eff geometry is consistent and the largest sigma_blur stays
    # measurable). Pass an explicit --latent-size only to override.
    parser.add_argument("--latent-size", type=int, default=0)
    parser.add_argument("--positions", type=int, default=4)
    args = parser.parse_args(argv)

    models = load_models_config()
    if args.decoder not in models:
        print(f"[error] unknown decoder '{args.decoder}' (have {sorted(models)})")
        return 2
    spec = models[args.decoder]
    p0cfg = load_p0_config()
    cfg = p0cfg.evidence.sigma_calib
    latent_size = (
        args.latent_size if args.latent_size > 0 else p0cfg.grid.resolution // _VAE_SPATIAL_UPSCALE
    )
    # Blur levels come from the preregistered grid (so sigma_calib measures sigma_eff at exactly
    # the sigma_blur levels run_p0 applies). Always include 0.0 (the bare PSF).
    sigma_blur_levels = tuple(sorted({0.0, *(float(s) for s in p0cfg.grid.sigma_blur)}))
    try:
        result = calibrate_decoder_sigma(
            spec,
            get_device(),
            cfg,
            latent_size=latent_size,
            positions=args.positions,
            sigma_blur_levels=sigma_blur_levels,
        )
    except BackendUnavailableError as exc:
        print(f"[skip] {exc}")
        return 0
    out = ensure_dir(outputs_dir() / "sigma_calib") / f"{spec.name}.json"
    result.save(out)
    flag = "scalar OK" if result.scalar_ok else "EFFECTIVE-sigma only"
    sweep = ", ".join(f"{sb:g}:{eff:.2f}" for sb, eff in sorted(result.per_sigma_blur.items()))
    print(
        f"[ok] {spec.name}: sigma_hat={result.sigma_hat:.3f} "
        f"CI[{result.ci_lo:.3f}, {result.ci_hi:.3f}] ({flag}); sigma_eff(blur) [{sweep}] -> {out}"
    )
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
