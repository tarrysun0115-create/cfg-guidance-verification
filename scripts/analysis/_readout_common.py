"""Shared, fail-loud contracts for the deployed (rho, m) pair-readout reanalyses.

Role in the pipeline: every reanalysis of the three deployed readouts (rho alone,
T_pair-zsum and T_pair-LDA) loads its per-image coordinates through ``load_residual_rows``
and fits/scores the per-cell heads through ``fit_deployed`` / ``score_deployed``.

* Pair coordinates (Section III-C): ``rho_summary`` is rho(x) of Eq. (4) and ``m`` is m(x)
  of Eq. (8), the E_tex-masked (q = 0.2), winsorized ([0.01, 0.99]) patch mean of the
  branch-difference norm delta_p of Eq. (6).
* Heads (Section III-D, Remark): FIT-H0 standardization, then either the FIT sign of rho,
  the signed z-sum (T_pair-zsum), or the Ledoit-Wolf Fisher direction of Eq. (11)
  (T_pair-LDA, without its additive constant).

Coordinate source. ``load_residual_rows`` recomputes both coordinates from the server-side
feature store (``residual_v1.dat`` + ``compact_*.npy``) written by the main-grid run;
decoders without the store are skipped.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from cfgid.bounds.wstar_cov import ledoit_wolf_cov
from cfgid.channel.rho import etex_keep_mask, image_rho_summary, winsorize
from cfgid.featurestore import read_compact, read_residual_v1
from cfgid.numerics import finite_float
from cfgid.sweep.evidence import _require_row_store_alignment
from cfgid.sweep.neff_scaling import _load_p0_rows, _row_prompt_index

MASK_Q = 0.2  # E_tex keep-mask quantile q of Eq. (3)
WINSOR = (0.01, 0.99)  # within-image winsorization level kappa
RHO_REPRODUCTION_TOL = 1e-3  # max |recomputed rho - p0_table rho_summary| before aborting
DEPLOYED_READOUTS = ("rho", "zsum_dep", "lda_dep")
# Code split names: train = FIT, null_adapt = NULL, calibration = CAL, test = TEST.
VALID_SPLITS = frozenset({"train", "null_adapt", "calibration", "test"})


def write_json_strict(path: Path, payload: object) -> None:
    """Write standard JSON and reject every NaN/Inf instead of emitting Python extensions."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")


def m_from_residual(residual_i: np.ndarray, e_tex_i: np.ndarray) -> float:
    """m(x) of Eq. (8) from one image's dequantized (P, D) residual difference and E_tex."""
    residual = np.asarray(residual_i, dtype=np.float32)
    e_tex = np.asarray(e_tex_i, dtype=np.float32)
    if residual.ndim != 2:
        raise ValueError(f"residual_i must be (P, D), got {residual.shape}")
    if e_tex.ndim != 1 or e_tex.shape[0] != residual.shape[0]:
        raise ValueError(
            f"e_tex_i must be (P,) aligned with residual_i; got {e_tex.shape} vs {residual.shape}"
        )
    if residual.shape[0] == 0:
        raise ValueError("m cannot be computed from an empty patch set")
    if not np.all(np.isfinite(residual)):
        raise ValueError("residual_i contains non-finite values")
    # delta_p = ||r_cond,p - r_uncond,p|| per patch, in float32.
    delta = np.linalg.norm(residual, axis=1)
    # Keep patches with E_tex >= the within-image q-quantile (all patches if none survive),
    # winsorize within the image, and average; every step stays in float32.
    keep = etex_keep_mask(e_tex, MASK_Q)
    kept = delta[keep] if keep.any() else delta
    value = float(np.mean(winsorize(kept.astype(np.float32), WINSOR[0], WINSOR[1])))
    return finite_float("masked winsorized diff_energy", value)


def rho_from_compact(rho_i: np.ndarray, e_tex_i: np.ndarray) -> float:
    """rho(x) of Eq. (4) from one image's per-patch rho_p and E_tex (compact store)."""
    value = image_rho_summary(
        np.asarray(rho_i, dtype=np.float64),
        np.asarray(e_tex_i, dtype=np.float64),
        mask_quantile=MASK_Q,
        winsor_quantiles=WINSOR,
    )
    return finite_float("masked winsorized rho", value)


def normalize_p0_rows(rows: list[dict], expected_decoder: str) -> list[dict]:
    """Validate and normalize, in place, the p0_table fields used by the deployed readouts.

    Checks the decoder stamp, integer guidance w >= 1, the label/w pairing (h0 iff w = 1),
    the split name, sigma_blur >= 0 and prompt-disjoint splits; coerces the numeric fields to
    Python floats and the prompt index to int.
    """
    prompt_splits: dict[int, str] = {}
    for r in rows:
        decoder = str(r.get("decoder", ""))
        if decoder != expected_decoder:
            raise ValueError(
                f"{expected_decoder}: row decoder {decoder!r} does not match directory name"
            )
        w = finite_float("w", r.get("w"))
        if not w.is_integer() or w < 1:
            raise ValueError(f"w must be a positive integer-valued number, got {w}")
        label = str(r.get("label", ""))
        if label not in {"h0", "h1"} or (label == "h0") != (w == 1.0):
            raise ValueError(f"inconsistent label/w pair: label={label!r}, w={w}")
        split = str(r.get("split", ""))
        if split not in VALID_SPLITS:
            raise ValueError(f"unknown split {split!r}")
        sigma_blur = finite_float("sigma_blur", r.get("sigma_blur"))
        if sigma_blur < 0.0:
            raise ValueError(f"sigma_blur must be non-negative, got {sigma_blur}")
        prompt = _row_prompt_index(r)
        # Splits are assigned per prompt: one prompt may never appear in two splits.
        previous_split = prompt_splits.setdefault(prompt, split)
        if previous_split != split:
            raise ValueError(
                f"{expected_decoder}: prompt {prompt} spans splits {previous_split!r}/{split!r}"
            )
        r["decoder"] = decoder
        r["prompt"] = prompt
        r["rho_summary"] = finite_float("rho_summary", r.get("rho_summary"))
        r["sigma_blur"] = sigma_blur
        r["w"] = w
    return rows


def load_normalized_p0_table(table_path: Path, expected_decoder: str) -> list[dict]:
    """Load one decoder's p0_table.json and apply ``normalize_p0_rows``."""
    return normalize_p0_rows(_load_p0_rows(table_path), expected_decoder)


def load_residual_rows(root: Path) -> list[dict]:
    """Load every aligned p0 table/store pair and attach the frozen per-image m coordinate.

    Per decoder directory under ``root`` (sorted by name), rows keep p0_table.json order;
    ``rho_summary`` is replaced by rho(x) recomputed from the compact store (aborting if it
    differs from the table by more than RHO_REPRODUCTION_TOL) and ``m`` is added. Decoders
    without ``residual_v1.dat`` are skipped.
    """
    if not root.is_dir():
        raise SystemExit(f"features root not found: {root}")
    all_rows: list[dict] = []
    for dec_dir in sorted(p for p in root.iterdir() if (p / "p0_table.json").is_file()):
        rows = _load_p0_rows(dec_dir / "p0_table.json")
        if not (dec_dir / "residual_v1.dat").is_file():
            print(f"[skip] {dec_dir.name}: no residual_v1.dat (m needs the feature store)")
            continue
        residual, residual_meta = read_residual_v1(dec_dir)
        batch, compact_meta = read_compact(dec_dir)
        # Store records pair with table rows by position; require identical image_id order.
        _require_row_store_alignment(rows, residual_meta)
        _require_row_store_alignment(rows, compact_meta)
        normalize_p0_rows(rows, dec_dir.name)
        worst_rho_error = 0.0
        for i, r in enumerate(rows):
            rho = rho_from_compact(batch.rho[i], batch.e_tex[i])
            worst_rho_error = max(worst_rho_error, abs(rho - r["rho_summary"]))
            r["rho_summary"] = rho
            r["m"] = m_from_residual(residual[i], batch.e_tex[i])
        if worst_rho_error > RHO_REPRODUCTION_TOL:
            raise SystemExit(
                f"[abort] {dec_dir.name}: recomputed rho deviates from stored rho_summary by "
                f"{worst_rho_error:.4g} (> {RHO_REPRODUCTION_TOL:g}); refusing numbers"
            )
        all_rows.extend(rows)
        print(f"[load] {dec_dir.name}: {len(rows)} rows, rho/m recomputed from store")
    if not all_rows:
        raise SystemExit("no decoder with both p0_table.json and residual_v1.dat under --root")
    return all_rows


def _side_matrix(cell_rows: list[dict], w_set: set, split: str, side: str) -> np.ndarray:
    """(n, 2) float64 matrix of (rho_summary, m) for one split and hypothesis side of a cell.

    H0 side: label-h0 rows when 1 is in ``w_set`` plus label-h1 rows whose w is in ``w_set``
    (so a low-guidance H0 set such as {3} selects w = 3 rows). H1 side: label-h1 rows with w
    in ``w_set``. Row order follows ``cell_rows``.
    """
    if side not in {"h0", "h1"}:
        raise ValueError(f"side must be h0/h1, got {side!r}")

    def selected(r: dict) -> bool:
        if r["split"] != split:
            return False
        label = r["label"]
        w = r["w"]
        if side == "h0":
            # Exact pair_readout_battery._fit_stats_2d side rule.
            return (label == "h0" and 1.0 in w_set) or (label == "h1" and w in w_set)
        return label == "h1" and w in w_set

    return np.asarray(
        [(r["rho_summary"], r["m"]) for r in cell_rows if selected(r)], dtype=np.float64
    ).reshape(-1, 2)


@dataclass(frozen=True)
class DeployedFit:
    """One fitted head: weights, FIT-H0 mean/std, and the standardized FIT scores."""

    wvec: np.ndarray
    m0: np.ndarray
    s0: np.ndarray
    fit_h0: np.ndarray
    fit_h1: np.ndarray


def fit_deployed(
    cell_rows: list[dict], h1_set: set, h0_set: set, readout: str
) -> DeployedFit | None:
    """Mirror pair_readout_battery._fit_stats_2d exactly for one cell and contrast.

    Uses only FIT (``train``) rows. Returns None when either side has fewer than 8 rows or a
    FIT-H0 coordinate has zero spread.
    """
    if readout not in DEPLOYED_READOUTS:
        raise ValueError(f"unknown readout {readout!r}")
    f0 = _side_matrix(cell_rows, h0_set, "train", "h0")
    f1 = _side_matrix(cell_rows, h1_set, "train", "h1")
    if f0.shape[0] < 8 or f1.shape[0] < 8:
        return None
    if not (np.all(np.isfinite(f0)) and np.all(np.isfinite(f1))):
        raise ValueError("FIT rho/m pool contains non-finite values")
    # Standardize both coordinates by the FIT-H0 mean and sample std (ddof=1).
    m0 = f0.mean(axis=0)
    s0 = f0.std(axis=0, ddof=1)
    if not np.all(np.isfinite(m0)):
        raise ValueError("FIT-H0 coordinate mean is non-finite")
    if not np.all(np.isfinite(s0)):
        raise ValueError("FIT-H0 coordinate std is non-finite")
    if not np.all(s0 > 0):
        return None
    z0 = (f0 - m0) / s0
    z1 = (f1 - m0) / s0
    # Standardized FIT-H1 mean shift (mu_y of Eq. (11)).
    shift = z1.mean(axis=0)
    if not (np.all(np.isfinite(z0)) and np.all(np.isfinite(z1)) and np.all(np.isfinite(shift))):
        raise ValueError("FIT standardization/shift produced non-finite values")
    if readout == "rho":
        # rho alone, oriented by the sign of its FIT shift (ties -> +1).
        wvec = np.array([1.0 if shift[0] >= 0 else -1.0, 0.0])
    elif readout == "zsum_dep":
        # T_pair-zsum: sum of the standardized coordinates with FIT signs (ties -> +1).
        wvec = np.array([np.sign(shift[0]) or 1.0, np.sign(shift[1]) or 1.0])
    else:
        # T_pair-LDA: Ledoit-Wolf FIT-H0 covariance of the standardized pair, w = Sigma^-1 mu.
        cov, _ = ledoit_wolf_cov(z0)
        try:
            wvec = np.linalg.solve(cov, shift)
        except np.linalg.LinAlgError as exc:
            raise ValueError("Ledoit-Wolf covariance solve failed") from exc
    if not np.all(np.isfinite(wvec)):
        raise ValueError("deployed readout weights are non-finite")
    return DeployedFit(wvec=wvec, m0=m0, s0=s0, fit_h0=z0 @ wvec, fit_h1=z1 @ wvec)


def score_deployed(
    cell_rows: list[dict],
    w_set: set,
    split: str,
    side: str,
    fit: DeployedFit,
) -> np.ndarray:
    """Mirror pair_readout_battery._apply_fusion with frozen per-cell FIT parameters."""
    y = _side_matrix(cell_rows, w_set, split, side)
    values = ((y - fit.m0) / fit.s0) @ fit.wvec
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{split}-{side} scoring produced non-finite values")
    return values
