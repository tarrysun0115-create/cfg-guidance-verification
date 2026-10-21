"""Gate-evidence assembly: turn stored main-grid (p0) artifacts into split-isolated gate inputs.

Role in the pipeline: ``cfgid.sweep.analyze`` calls these functions to build its decision memo.
They supply the sigma-calibration evidence (sigma_eff per blur level), the null-adapted v and
g_sigma and the predictive E_tex that enter w* (Eqs. (10) and (15)), the covariance-channel
diagnostic S_eff = sqrt(d^T Sigma^-1 d) / w*_cov, the nuisance-baseline negative controls (Table
VIII rows 1-2, Section IV-E1), the rho-sufficiency probe, the oracle-trace consistency check and the
wrong-prompt null. Several analysis scripts also import the row/store alignment guard and the
per-patch energy features from here.

Every function consumes only persisted artifacts (p0_table.json rows + images/ + traces/ + the
compact/residual feature store + outputs/sigma_calib/*.json) -- no model weights -- so the whole
module is unit-testable on synthetic fixtures. Leakage rules: parameters are fitted on the
``null_adapt`` (NULL) split from H0 rows only (predictive track), evaluation (AUC / knees /
negative controls / sufficiency) uses the ``test`` (TEST) split, and the sufficiency probe is fit
on ``train`` (FIT); the same prompt never spans splits (grouped split assigned by the driver).
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from cfgid.baselines.clip_align import ClipUnavailableError, clip_alignment_scores, clip_available
from cfgid.baselines.nuisance import nuisance_features
from cfgid.bounds.wstar_cov import CovChannel, ledoit_wolf_cov
from cfgid.channel.inversion.base import patchify
from cfgid.channel.oracle import rho_trace_consistency
from cfgid.channel.params.gsigma import GSigmaEstimate, estimate_v_null, null_adapted_gsigma
from cfgid.channel.params.vmodel import VModel, VModelUnidentifiableError, fit_v_model
from cfgid.channel.rho import etex_keep_mask, rho_patch
from cfgid.channel.sigma_probe import nominal_sigma
from cfgid.detector.empirical import auc
from cfgid.detector.sufficiency import fisher_lda_direction, project_lda
from cfgid.featurestore import read_compact, read_residual_v1
from cfgid.numerics import as_finite_float32, loads_strict_json
from cfgid.p0config import P0Config

# Table rows are JSON-loaded dicts; values are accessed with int()/float()/str() coercion at use
# sites (Any here is the honest type of json.loads output, not a typing bypass).
Row = dict[str, Any]
_VALID_LABELS = ("h0", "h1")
_VALID_SIGMA_SOURCES = ("calibrated", "nominal_fallback")


class EvidenceUnavailableError(RuntimeError):
    """Valid artifacts, but this optional evidence item is not estimable from them."""


def _folded_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Orientation-agnostic AUC (a detector may use either sign of the score)."""
    a = auc(scores, labels)
    return float(max(a, 1.0 - a))


def _row_label(row: Row) -> str:
    """Validated table label; corrupted artifacts must not silently become H0 rows."""
    label = str(row.get("label", ""))
    if label not in _VALID_LABELS:
        raise ValueError(f"row label must be one of {_VALID_LABELS}, got {label!r}")
    return label


def _row_finite_float(row: Row, key: str) -> float:
    """Validated numeric table field; NaN/Inf in artifacts must fail loudly."""
    try:
        value = float(row[key])
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"row field '{key}' must be finite numeric") from exc
    if not np.isfinite(value):
        raise ValueError(f"row field '{key}' contains non-finite value {value}")
    return value


def _row_finite_int(row: Row, key: str) -> int:
    """Validated integer-valued table field; rejects NaN/Inf and lossy coercions."""
    value = _row_finite_float(row, key)
    if not value.is_integer():
        raise ValueError(f"row field '{key}' must be integer-valued, got {value}")
    return int(value)


def _rows_labels(rows: list[Row]) -> np.ndarray:
    return np.asarray([1 if _row_label(r) == "h1" else 0 for r in rows], dtype=np.int64)


def validate_table_rows(rows: list[Row]) -> None:
    """Validate core table labels and numeric fields before any verdict can be written."""
    for row in rows:
        _row_label(row)
        # w/qf/sigma_blur define the core grid cell and can otherwise be silently skipped on
        # non-test rows; sigma_blur is the alignment-cell axis so a NaN must fail loud.
        _row_finite_int(row, "w")
        _row_finite_int(row, "qf")
        _row_finite_float(row, "sigma_blur")
        for key in ("prompt", "seed", "gen_seed"):
            if key in row:
                _row_finite_int(row, key)
        _row_finite_float(row, "rho_summary")
        if "rho_summary_wrong_prompt" in row:
            _row_finite_float(row, "rho_summary_wrong_prompt")


def split_rows(rows: list[Row], split: str) -> list[Row]:
    return [r for r in rows if r.get("split") == split]


# --------------------------------------------------------------------------- #
# sigma calibration consumption
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class SigmaEvidence:
    decoder: str
    sigma_hat: float
    ci: tuple[float, float]
    scalar_ok: bool
    source: str  # "calibrated" | "nominal_fallback"
    reasons: list[str] = field(default_factory=list)
    # Effective sigma per controlled-blur level (MTF measure), keyed by sigma_blur (0.0 = bare
    # PSF). This is the per-cell sigma the alignment x-axis uses under the sigma-sweep.
    per_sigma_blur: dict[float, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.source not in _VALID_SIGMA_SOURCES:
            raise ValueError(f"unknown sigma evidence source '{self.source}'")
        if len(self.ci) != 2:
            raise ValueError(f"sigma CI must have two endpoints, got {self.ci}")
        lo, hi = float(self.ci[0]), float(self.ci[1])
        for name, value in (("sigma_hat", self.sigma_hat), ("ci_lo", lo), ("ci_hi", hi)):
            if not (np.isfinite(value) and value > 0.0):
                raise ValueError(f"{name} must be positive and finite, got {value}")
        if not lo <= self.sigma_hat <= hi:
            raise ValueError(f"sigma CI [{lo}, {hi}] must bracket sigma_hat={self.sigma_hat}")
        for sb, eff in self.per_sigma_blur.items():
            if not (np.isfinite(sb) and sb >= 0.0 and np.isfinite(eff) and eff > 0.0):
                raise ValueError(f"per_sigma_blur[{sb}]={eff} must be finite, sb>=0, eff>0")

    def sigma_eff(self, sigma_blur: float) -> float | None:
        """Effective sigma at this blur level.

        Calibrated track: the calibrated value from ``per_sigma_blur`` (None if that blur level
        is absent). Nominal fallback (no calibration artifact): the Gaussian-composition
        approximation ``sqrt(sigma_hat^2 + sigma_blur^2)`` so analyze can still produce a
        DIAGNOSTIC memo — but ``source != "calibrated"`` forces the gate to INCONCLUSIVE
        regardless.
        """
        measured = self.per_sigma_blur.get(float(sigma_blur))
        if measured is not None:
            return measured
        if self.source == "nominal_fallback":
            return float(np.sqrt(self.sigma_hat**2 + float(sigma_blur) ** 2))
        return None


def _parse_per_sigma_blur(raw: object, *, path: Path) -> dict[float, float]:
    """Parse JSON string keys into a collision-free, %g-stable sigma_blur map."""
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: per_sigma_blur must be a JSON object")
    out: dict[float, float] = {}
    for raw_key, raw_value in raw.items():
        try:
            sb = float(raw_key)
            eff = float(raw_value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"{path}: per_sigma_blur key/value must be numeric") from exc
        if not (np.isfinite(sb) and sb >= 0.0):
            raise ValueError(f"{path}: per_sigma_blur key must be finite and >=0, got {raw_key!r}")
        if float(f"{sb:g}") != sb:
            raise ValueError(
                f"{path}: per_sigma_blur key {raw_key!r} is not %g-stable "
                f"(round-trips to {float(f'{sb:g}')!r})"
            )
        if sb in out:
            raise ValueError(
                f"{path}: duplicate per_sigma_blur key after float normalization: {raw_key!r} "
                f"collides with an existing sigma_blur={sb:g}"
            )
        if not (np.isfinite(eff) and eff > 0.0):
            raise ValueError(f"{path}: per_sigma_blur[{raw_key!r}] must be finite and >0")
        out[sb] = eff
    return out


def load_sigma_evidence(decoder: str, calib_root: Path) -> SigmaEvidence:
    """Calibrated sigma_hat from outputs/sigma_calib/<decoder>.json, nominal fallback otherwise."""
    path = calib_root / f"{decoder}.json"
    if path.is_file():
        raw_obj = loads_strict_json(path.read_text(encoding="utf-8"))
        if not isinstance(raw_obj, dict):
            raise ValueError(f"{path}: sigma calibration artifact must be a JSON object")
        raw = raw_obj
        raw_decoder = str(raw.get("decoder", ""))
        if raw_decoder != decoder:
            raise ValueError(
                f"{path}: sigma calibration decoder mismatch: file is for {raw_decoder!r}, "
                f"expected {decoder!r}"
            )
        per_sigma_blur = _parse_per_sigma_blur(raw.get("per_sigma_blur", {}), path=path)
        return SigmaEvidence(
            decoder=decoder,
            sigma_hat=float(raw["sigma_hat"]),
            ci=(float(raw["ci_lo"]), float(raw["ci_hi"])),
            scalar_ok=bool(raw["scalar_ok"]),
            source="calibrated",
            reasons=[str(r) for r in raw.get("reasons", [])],
            per_sigma_blur=per_sigma_blur,
        )
    nominal = nominal_sigma(decoder)
    return SigmaEvidence(
        decoder=decoder,
        sigma_hat=nominal,
        ci=(nominal, nominal),
        scalar_ok=False,
        source="nominal_fallback",
        reasons=["no calibration artifact; ordering-only nominal sigma"],
    )


# --------------------------------------------------------------------------- #
# null-adapted v / g_sigma (predictive track: H0 rows only, null_adapt split)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class NullParams:
    v_hat: float
    v_level: str  # "cell"/"decoder"/"decoder+calibration" or "analytic[...]" or "unavailable"
    g_sigma: GSigmaEstimate | None
    v_provenance: str = "unlabeled_null_adapted"
    # Fallback level of the H0 null sample that sets the g_sigma null SCALE ("cell"/"decoder"/
    # "decoder+calibration"/"unavailable"). The absolute scale is rescale-invariant for the gate
    # (ordering only), so a decoder-level scale is informative, not disqualifying.
    gsigma_level: str = "unavailable"


def _h0_rho(rows: list[Row], splits: tuple[str, ...]) -> np.ndarray:
    vals = [
        _row_finite_float(r, "rho_summary")
        for r in rows
        if _row_label(r) == "h0" and r.get("split") in splits
    ]
    return np.asarray(vals, dtype=np.float64)


def _h0_rho_by_qf(rows: list[Row], splits: tuple[str, ...]) -> dict[int | None, np.ndarray]:
    """H0 rho summaries grouped by QF (int; -1 == uncompressed), restricted to ``splits``."""
    out: dict[int, list[float]] = {}
    for r in rows:
        if _row_label(r) == "h0" and r.get("split") in splits:
            out.setdefault(int(r["qf"]), []).append(_row_finite_float(r, "rho_summary"))
    return {qf: np.asarray(vals, dtype=np.float64) for qf, vals in out.items()}


def fit_decoder_v_model(decoder_rows: list[Row]) -> VModel | None:
    """Analytic v(QF) model from H0 rho across QF (null_adapt, widening to +calibration).

    Predictive track: only H0 (w=1) samples outside train/test feed the fit. Returns ``None`` when
    the model is unidentifiable (e.g. <2 distinct QF), so the caller falls back to empirical v.
    """
    for splits in (("null_adapt",), ("null_adapt", "calibration")):
        # Read OUTSIDE the try so a corrupt-label ValueError (the artifact contract from _row_label)
        # propagates loudly instead of being swallowed as a benign "model unidentifiable" case.
        by_qf = _h0_rho_by_qf(decoder_rows, splits)
        try:
            return fit_v_model(by_qf)
        except VModelUnidentifiableError:
            continue  # genuinely unidentifiable (<2 distinct QF) -> widen split / empirical
    return None


def null_params_for_cell(
    cell_rows: list[Row],
    decoder_rows: list[Row],
    sigma_ev: SigmaEvidence,
    *,
    sigma_blur: float,
    seed: int = 0,
) -> NullParams:
    """g_sigma(sigma_eff(sigma_blur)) + empirical v for one (decoder, sigma_blur) cell.

    Sigma-sweep: g_sigma is the plugin gain at the calibrated effective sigma for this blur level
    (``sigma_ev.sigma_eff(sigma_blur)`` from the calibration sweep). It gates/logs the predictive
    track, but scalar measured-Etex w* does not multiply it because measured E_tex already carries
    the decoder/blur attenuation. ``v`` is the empirical H0 null variance (cell -> decoder ->
    +calibration fallback). Every level uses ONLY w=1 (H0) samples outside the test split, so the
    predictive-track claim holds. Returns ``unavailable`` if the effective sigma was not calibrated
    for this blur level.
    """
    sigma_eff = sigma_ev.sigma_eff(sigma_blur)
    if sigma_eff is None or not (np.isfinite(sigma_eff) and sigma_eff > 0.0):
        return NullParams(
            v_hat=float("nan"),
            v_level="unavailable",
            g_sigma=None,
            v_provenance="unavailable",
            gsigma_level="unavailable",
        )
    for level, rho in (
        ("cell", _h0_rho(cell_rows, ("null_adapt",))),
        ("decoder", _h0_rho(decoder_rows, ("null_adapt",))),
        ("decoder+calibration", _h0_rho(decoder_rows, ("null_adapt", "calibration"))),
    ):
        if rho.size >= 3 and np.std(rho) > 0:
            # Degenerate sigma CI (point sigma_eff per blur level; no per-level CI from the MTF
            # sweep). qf=None -> no JPEG-MTF factor (QF collapsed to none);
            # g = plugin_gain(sigma_eff).
            g = null_adapted_gsigma(sigma_eff, (sigma_eff, sigma_eff), rho, qf=None, seed=seed)
            v_hat, _, _ = estimate_v_null(rho, seed=seed)
            return NullParams(
                v_hat=v_hat,
                v_level=level,
                g_sigma=g,
                v_provenance="unlabeled_null_adapted",
                gsigma_level=level,
            )
    return NullParams(
        v_hat=float("nan"),
        v_level="unavailable",
        g_sigma=None,
        v_provenance="unavailable",
        gsigma_level="unavailable",
    )


def cov_channel_for_cell(
    decoder_rows: list[Row],
    decoder_dir: Path,
    *,
    sigma_blur: float,
) -> CovChannel | None:
    """Per-cell vector channel (S_eff = sqrt(d^T Sigma^-1 d)) from H0 per-patch rho — DIAGNOSTIC.

    Sigma is the Ledoit-Wolf-shrunk covariance of the H0 (null_adapt + calibration) per-patch rho
    vectors for this (decoder, sigma_blur) cell; ``d_p = mean H0 per-patch E_tex_meas`` is the
    predictive per-patch MEASURED signal slope (no H1 labels). ``E_tex_meas`` already carries the
    decoder/blur attenuation, so multiplying by plugin ``g_sigma`` here would double-count the PSF.
    Returns ``None`` when too few H0 images / patches exist or the channel is degenerate; this feeds
    a reported w*_cov column, NOT the gate (the covariance value is bound correctness, not the
    alignment x-axis lever).
    """
    batch, meta = read_compact(decoder_dir)
    _require_row_store_alignment(decoder_rows, meta)
    idx = _h0_indices(decoder_rows, ("null_adapt", "calibration"), sigma_blur=sigma_blur)
    if len(idx) < 4:  # too few H0 images for even a shrunk covariance
        return None
    rho_vecs = np.stack(
        [as_finite_float32("rho_patch", batch.rho[i]).astype(np.float64).ravel() for i in idx]
    )
    etex_vecs = np.stack(
        [as_finite_float32("etex_patch", batch.e_tex[i]).astype(np.float64).ravel() for i in idx]
    )
    if rho_vecs.shape[1] < 2:  # need >=2 patches for a vector channel
        return None
    try:
        cov, _ = ledoit_wolf_cov(rho_vecs)
        d = etex_vecs.mean(axis=0)
        return CovChannel(d=d, cov=cov)
    except ValueError:
        return None


# --------------------------------------------------------------------------- #
# predictive E_tex for w* (H0-only, no test split)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ETexEvidence:
    value: float
    level: str  # "cell" | "decoder" | "decoder+calibration"
    n: int


def _image_etex_summary(e_tex_patch: np.ndarray, mask_quantile: float) -> float:
    """Image-level E_tex paired to image_rho_summary's low-texture mask."""
    e = as_finite_float32("e_tex_patch", e_tex_patch).reshape(-1)
    keep = etex_keep_mask(e, mask_quantile)
    kept = e[keep] if keep.any() else e
    if kept.size == 0:
        raise EvidenceUnavailableError("E_tex summary received no patches")
    value = float(np.mean(kept))
    if not (np.isfinite(value) and value > 0.0):
        raise EvidenceUnavailableError(f"E_tex summary must be positive, got {value}")
    return value


def _h0_indices(
    decoder_rows: list[Row],
    splits: tuple[str, ...],
    *,
    sigma_blur: float | None,
) -> list[int]:
    """H0 store indices in ``splits``; if ``sigma_blur`` is given, restricted to that blur cell."""
    return [
        i
        for i, r in enumerate(decoder_rows)
        if _row_label(r) == "h0"
        and r.get("split") in splits
        and (sigma_blur is None or float(r["sigma_blur"]) == sigma_blur)
    ]


def predictive_etex_for_cell(
    cell_rows: list[Row],
    decoder_rows: list[Row],
    decoder_dir: Path,
    cfg: P0Config,
) -> ETexEvidence:
    """Predictive E_tex scale for w*: H0 null_adapt only, with no H1/test leakage.

    The scalar main-grid channel is over image-level rho summaries, so this returns an image-level
    E_tex summary using the same low-texture mask quantile as rho aggregation. The fallback
    chain mirrors v: cell -> decoder -> decoder+calibration, always H0-only and never test.
    """
    batch, meta = read_compact(decoder_dir)
    _require_row_store_alignment(decoder_rows, meta)
    sbs = {float(r["sigma_blur"]) for r in cell_rows}
    if len(sbs) != 1:
        raise ValueError(f"cell_rows must contain exactly one sigma_blur, got {sorted(sbs)}")
    sigma_blur = next(iter(sbs))

    for level, idx in (
        ("cell", _h0_indices(decoder_rows, ("null_adapt",), sigma_blur=sigma_blur)),
        ("decoder", _h0_indices(decoder_rows, ("null_adapt",), sigma_blur=None)),
        (
            "decoder+calibration",
            _h0_indices(decoder_rows, ("null_adapt", "calibration"), sigma_blur=None),
        ),
    ):
        if not idx:
            continue
        vals = [_image_etex_summary(batch.e_tex[i], cfg.rho.low_etex_mask_quantile) for i in idx]
        value = float(np.mean(vals))
        if np.isfinite(value) and value > 0.0:
            return ETexEvidence(value=value, level=level, n=len(vals))
    raise EvidenceUnavailableError("no positive H0 E_tex samples available outside the test split")


# --------------------------------------------------------------------------- #
# negative controls (test split only)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class NegativeControlCell:
    cell: str
    rho_auc: float
    baseline_aucs: dict[str, float]
    n_h0: int
    n_h1: int

    @property
    def best_baseline_auc(self) -> float:
        return max(self.baseline_aucs.values()) if self.baseline_aucs else 0.5

    @property
    def margin(self) -> float:
        return self.rho_auc - self.best_baseline_auc


@dataclass(frozen=True)
class NegativeControls:
    rho_auc: float
    baseline_aucs: dict[str, float]
    clip_status: str  # "ok" | "unavailable" | "disabled"
    n_h0: int
    n_h1: int
    cells: tuple[NegativeControlCell, ...] = ()
    # CLIP confound diagnostic (report-only; does NOT change rho_beats_nuisance): minimum rho AUC
    # within CLIP-score strata. If rho still separates H0/H1 within semantically-matched strata,
    # it adds signal beyond prompt adherence. None when CLIP is unavailable / too few per stratum.
    clip_stratified_rho_auc: float | None = None
    clip_n_strata: int = 0

    @property
    def best_baseline_auc(self) -> float:
        return max(self.baseline_aucs.values()) if self.baseline_aucs else 0.5

    @property
    def weakest_cell(self) -> NegativeControlCell | None:
        if not self.cells:
            return None
        return min(self.cells, key=lambda c: c.margin)

    def rho_beats_nuisance(self, margin: float) -> bool:
        # FAIL-SAFE: the preregistered contract is PER-CELL. With no comparable cell there is
        # no per-cell evidence, and silently degrading to the pooled comparison would re-open
        # the hole the per-cell gate closed — so no cells means False, never a pooled pass.
        if not self.cells:
            return False
        return all(cell.rho_auc >= cell.best_baseline_auc - margin for cell in self.cells)


_PROGRESS_EVERY = 500  # emit a nuisance-loop progress line every N images


def _progress(msg: str, *, start: bool = False) -> None:
    """Live progress to stderr (flushed) so a multi-minute image loop is visibly alive, not hung."""
    print(("[analyze] " if start else "[analyze]   ") + msg, file=sys.stderr, flush=True)


def negative_controls(
    test_rows: list[Row],
    features_root: Path,
    cfg: P0Config,
    *,
    clip_dir: Path | None,
) -> NegativeControls | None:
    """Real nuisance AUCs on the stored inversion-input images (test split, H0 vs H1).

    Pooled over all w>1 against w=1 and per (decoder, sigma_blur) cell, rho_summary and each raw
    nuisance feature are scored with the orientation-agnostic AUC max(AUC, 1-AUC), i.e. the sign
    is chosen on the evaluated pool; nothing is fitted. These are the distinct-VAE rows of Table
    VIII (Section IV-E1); the weakest cell is the per-cell negative control.
    """
    from PIL import Image

    missing_images = [
        features_root / str(r.get("decoder", "")) / str(r.get("image", ""))
        for r in test_rows
        if not r.get("image")
        or not (features_root / str(r.get("decoder", "")) / str(r.get("image", ""))).is_file()
    ]
    if missing_images:
        return None

    rows = list(test_rows)
    if not rows:
        return None
    labels = _rows_labels(rows)
    if labels.sum() == 0 or labels.sum() == len(rows):
        return None

    image_paths = [features_root / str(r["decoder"]) / str(r["image"]) for r in rows]

    # Stream: load -> compute nuisance features -> free, so peak memory is ONE image, not the whole
    # test split held in a list (that list was ~25GB on a 512-prompt run and starved the box). CLIP,
    # if enabled below, takes the paths and loads its own batches lazily -- no all-images-in-RAM.
    # A live progress line with an ETA prints every _PROGRESS_EVERY images: this loop is the single
    # slowest step of analyze (FFT + SRM per image over the whole test split), and a silent multi-
    # minute grind is indistinguishable from a hang.
    n_imgs = len(image_paths)
    _progress(f"nuisance baselines: {n_imgs} test images", start=True)
    t0 = time.monotonic()
    per_feature_lists: dict[str, list[float]] = {}
    for i, path in enumerate(image_paths, start=1):
        img = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
        for name, value in nuisance_features(img).items():
            per_feature_lists.setdefault(name, []).append(value)
        if i % _PROGRESS_EVERY == 0 or i == n_imgs:
            elapsed = time.monotonic() - t0
            eta = elapsed / i * (n_imgs - i)
            _progress(
                f"nuisance {i}/{n_imgs} ({100 * i // n_imgs}%) "
                f"elapsed {elapsed:.0f}s eta {eta:.0f}s"
            )
    per_feature = {
        name: np.asarray(vals, dtype=np.float64) for name, vals in per_feature_lists.items()
    }
    baseline_aucs = {name: _folded_auc(vals, labels) for name, vals in per_feature.items()}

    clip_status = "disabled"
    if cfg.evidence.clip_baseline_enabled:
        clip_status = "unavailable"
        if clip_dir is not None and clip_available(clip_dir):
            try:
                # Use the negotiated device (cuda>mps>cpu) so the CLIP baseline runs on the GPU when
                # analyze is on a GPU node; falls back to CPU on a CPU-only host (no regression).
                # The import stays lazy (only when the CLIP baseline is actually enabled +
                # available).
                from cfgid.device import get_device

                scores = clip_alignment_scores(
                    image_paths,
                    [str(r["prompt_text"]) for r in rows],
                    clip_dir=clip_dir,
                    device=get_device(),
                )
                per_feature["clip_text_alignment"] = scores.astype(np.float64)
                baseline_aucs["clip_text_alignment"] = _folded_auc(
                    per_feature["clip_text_alignment"], labels
                )
                clip_status = "ok"
            except ClipUnavailableError:
                clip_status = "unavailable"

    rho_scores = np.asarray([_row_finite_float(r, "rho_summary") for r in rows], dtype=np.float64)

    # CLIP confound diagnostic: rho AUC WITHIN CLIP-score strata (terciles). Within a stratum the
    # semantic prompt-adherence is roughly matched, so residual rho separability is not the CLIP
    # confound. Report-only; never feeds rho_beats_nuisance.
    clip_stratified_rho_auc: float | None = None
    clip_n_strata = 0
    if "clip_text_alignment" in per_feature:
        clip_vals = per_feature["clip_text_alignment"]
        edges = np.quantile(clip_vals, [1.0 / 3.0, 2.0 / 3.0])
        # Collapsed quantile edges (clustered/tied CLIP scores so the 1/3 and 2/3 quantiles
        # coincide) make the MIDDLE tercile unreachable: np.digitize then yields only bins {0, 2},
        # the bulk middle cluster pools into the high bin, and the "terciles" degrade to a 2-bin
        # median-ish split with no within-stratum semantic matching -- yet len(strat) >= 2 would
        # still pass and report it as if tercile-controlled. Only stratify when the edges are
        # genuinely distinct; otherwise leave clip_stratified_rho_auc = None (the >=2-strata guard
        # alone does not enforce this).
        if edges[0] < edges[1]:
            bins = np.digitize(clip_vals, edges)
            strat: list[float] = []
            for b in (0, 1, 2):
                mask = bins == b
                lab = labels[mask]
                if lab.size >= 2 and 0 < int(lab.sum()) < lab.size:
                    strat.append(_folded_auc(rho_scores[mask], lab))
            if len(strat) >= 2:
                clip_stratified_rho_auc = float(min(strat))
                clip_n_strata = len(strat)

    groups: dict[tuple[str, float], list[int]] = {}
    for idx, r in enumerate(rows):
        groups.setdefault((str(r["decoder"]), float(r["sigma_blur"])), []).append(idx)
    cells: list[NegativeControlCell] = []
    for (decoder, sigma_blur), idx_list in sorted(groups.items()):
        idx = np.asarray(idx_list, dtype=np.int64)
        cell_labels = labels[idx]
        if cell_labels.sum() == 0 or cell_labels.sum() == len(cell_labels):
            continue
        cell_baselines = {
            name: _folded_auc(vals[idx], cell_labels) for name, vals in per_feature.items()
        }
        cells.append(
            NegativeControlCell(
                cell=f"{decoder}/sb={sigma_blur:g}",
                rho_auc=_folded_auc(rho_scores[idx], cell_labels),
                baseline_aucs=cell_baselines,
                n_h0=int((cell_labels == 0).sum()),
                n_h1=int((cell_labels == 1).sum()),
            )
        )
    return NegativeControls(
        rho_auc=_folded_auc(rho_scores, labels),
        baseline_aucs=baseline_aucs,
        clip_status=clip_status,
        n_h0=int((labels == 0).sum()),
        n_h1=int((labels == 1).sum()),
        cells=tuple(cells),
        clip_stratified_rho_auc=clip_stratified_rho_auc,
        clip_n_strata=clip_n_strata,
    )


# --------------------------------------------------------------------------- #
# rho sufficiency (fit on train, evaluate on test; per-patch energy shallow probe)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class SufficiencyEvidence:
    rho_auc: float
    full_residual_auc: float
    n_train: int
    n_test: int

    @property
    def auc_gap(self) -> float:
        return self.full_residual_auc - self.rho_auc

    def rho_sufficient(self, max_gap: float) -> bool:
        return self.auc_gap <= max_gap


def _residual_energy_features(residual: np.ndarray) -> np.ndarray:
    """(N, P, D) full residual -> (N, P) per-patch L2 energies (the shallow sufficiency probe).

    Full-dimensional LDA on P*D dims is ill-posed at main-grid sample sizes; the per-patch energy
    map keeps the spatial pattern beyond the scalar rho while staying linearly fittable. Deeper
    probes are separate analyses.
    """
    r = as_finite_float32("residual", residual)
    if r.ndim != 3:
        raise ValueError(f"residual must be (N, P, D); got {r.shape}")
    return np.linalg.norm(r, axis=2)


def _require_row_store_alignment(decoder_rows: list[Row], meta: object) -> None:
    """Rows and store records are paired POSITIONALLY; require the image_id sequences to match.

    A re-sorted table / merged resume / regenerated artifact would otherwise silently misassign
    split/label to residual rows — exactly the silent failure that could put test prompts into
    a fitted parameter (leakage) or mispair oracle traces.
    """
    n_images = int(getattr(meta, "n_images", -1))
    if len(decoder_rows) != n_images:
        raise ValueError(
            f"table rows ({len(decoder_rows)}) != stored features ({n_images}); "
            "the table and the feature store must come from the same run"
        )
    store_ids = list(getattr(meta, "image_ids", []) or [])
    if store_ids:
        table_ids = [str(r.get("image_id", "")) for r in decoder_rows]
        if table_ids != store_ids:
            raise ValueError(
                "table image_id order != feature-store image_ids; refusing positional pairing"
            )


def sufficiency_evidence(
    decoder_rows: list[Row],
    decoder_dir: Path,
) -> SufficiencyEvidence | None:
    """Split-isolated sufficiency: LDA fitted on train rows, AUC evaluated on test rows."""
    batch, meta = read_compact(decoder_dir)
    residual, _ = read_residual_v1(decoder_dir)
    _require_row_store_alignment(decoder_rows, meta)
    feats = _residual_energy_features(residual)
    labels = _rows_labels(decoder_rows)
    idx_train = [i for i, r in enumerate(decoder_rows) if r.get("split") == "train"]
    idx_test = [i for i, r in enumerate(decoder_rows) if r.get("split") == "test"]
    y_train, y_test = labels[idx_train], labels[idx_test]
    if (
        len(idx_train) < 4
        or len(idx_test) < 2
        or int((y_train == 0).sum()) < 2
        or int((y_train == 1).sum()) < 2
        or len(set(y_test.tolist())) < 2
    ):
        return None
    direction = fisher_lda_direction(feats[idx_train], y_train)
    full_scores = project_lda(feats[idx_test], direction)
    rho_scores = np.asarray(
        [_row_finite_float(decoder_rows[i], "rho_summary") for i in idx_test], dtype=np.float64
    )
    return SufficiencyEvidence(
        rho_auc=_folded_auc(rho_scores, y_test),
        full_residual_auc=_folded_auc(full_scores, y_test),
        n_train=len(idx_train),
        n_test=len(idx_test),
    )


# --------------------------------------------------------------------------- #
# oracle-trace consistency (inversion-artifact exclusion; report-only)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class OracleEvidence:
    n_checked: int
    mean_pearson_r: float
    max_abs_err: float


def oracle_evidence(
    decoder_rows: list[Row],
    decoder_dir: Path,
    cfg: P0Config,
) -> OracleEvidence | None:
    """Compare inversion rho to generation-trace oracle rho (qf=none, sigma_blur=0, H1)."""
    batch, meta = read_compact(decoder_dir)
    _require_row_store_alignment(decoder_rows, meta)
    rs: list[float] = []
    errs: list[float] = []
    for i, row in enumerate(decoder_rows):
        if (
            _row_label(row) != "h1"
            or row.get("qf") != -1
            or _row_finite_float(row, "sigma_blur") != 0.0
            or not row.get("trace")
        ):
            continue
        trace_path = decoder_dir / str(row["trace"])
        if not trace_path.is_file():
            continue
        with np.load(trace_path) as data:
            if "eps_cond" not in data or "eps_uncond" not in data:
                continue
            pc = patchify(np.asarray(data["eps_cond"], dtype=np.float32), _grid_for(meta))
            pu = patchify(np.asarray(data["eps_uncond"], dtype=np.float32), _grid_for(meta))
        oracle_rho = rho_patch(pc, pu, floor=cfg.rho.denom_floor, log_ratio=cfg.rho.log_ratio)
        cons = rho_trace_consistency(batch.rho[i], oracle_rho)
        rs.append(cons.pearson_r)
        errs.append(cons.max_abs_err)
    if not rs:
        return None
    return OracleEvidence(
        n_checked=len(rs),
        mean_pearson_r=float(np.mean(rs)),
        max_abs_err=float(np.max(errs)),
    )


def _grid_for(meta: object) -> tuple[int, int]:
    grid = getattr(meta, "patch_grid", None)
    if not grid:
        raise ValueError("feature metadata lacks patch_grid")
    return int(grid[0]), int(grid[1])


# --------------------------------------------------------------------------- #
# wrong-prompt null (report-only)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class WrongPromptEvidence:
    true_prompt_auc: float
    wrong_prompt_auc: float
    n: int


def wrong_prompt_evidence(rows: list[Row]) -> WrongPromptEvidence | None:
    """AUC with true vs permuted prompts on the subset tier (rows carrying the extra field)."""
    sub = [r for r in rows if "rho_summary_wrong_prompt" in r]
    labels = _rows_labels(sub)
    if len(sub) < 4 or labels.sum() == 0 or labels.sum() == len(sub):
        return None
    true_scores = np.asarray([_row_finite_float(r, "rho_summary") for r in sub], dtype=np.float64)
    wrong_scores = np.asarray(
        [_row_finite_float(r, "rho_summary_wrong_prompt") for r in sub], dtype=np.float64
    )
    return WrongPromptEvidence(
        true_prompt_auc=_folded_auc(true_scores, labels),
        wrong_prompt_auc=_folded_auc(wrong_scores, labels),
        n=len(sub),
    )
