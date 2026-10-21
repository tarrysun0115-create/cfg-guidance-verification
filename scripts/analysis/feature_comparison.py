"""Candidate-channel comparison over the stored residual field (scalar rows of Table X).

Scores every candidate per-patch reduction of the stored inversion residuals on the same
selection-free pool as rho (every cell, H1 fixed at w=20 vs w=1, per-cell FIT-H0 standardization
and FIT sign, pooled TEST scores) and with the IDENTICAL texture mask (keep patches with
E_tex >= its 0.2 quantile), winsorization ([0.01, 0.99]) and patch mean as rho (Section III-C,
Eqs. (3)-(4)). No re-inversion: every candidate is a function of what the store keeps per patch
-- rho_p = log(||r_cond,p|| / ||r_uncond,p||) (``compact_rho``), diff_p = ||r_cond,p - r_uncond,p||
(``residual_v1``) and the texture energy e_tex. The two separate branch norms are NOT stored, so
single-branch readouts (and DIRE/AEROBLADE-style baselines) would need a re-inversion.

Axes per candidate (one row of the scorecard):
* detection        -- pooled FIT-oriented AUC1 (``det_auc1``) and the K=16 point of the
  with-replacement cross-cell group curve (``det_auc_neff16``; ``fixed_h1_allcell._summarize``
  with K in {1, 2, 4, 8, 16}, 2000 groups, seed 0);
* sign             -- whether it carries rho's decoder-specific sign (``has_decoder_sign``) and
  the leave-one-cell-out decoder-majority sign-rule accuracy (``sign_loocv_acc``; Section IV-E4);
* Gaussian fit     -- the sqrt(K) exponent, mean |AUC_emp(K) - Phi(sqrt(K) d / sqrt 2)| and the
  pooled-H1 normality (skew, excess kurtosis, D'Agostino p);
* PSF ordering     -- per decoder, the power-law decay exponent p of the deflection
  d = 2*Phi^-1(cell AUC) along sigma_eff (psf_grounding_features), and the Pearson correlation of
  p with the decoder's measured sigma_core over the distinct-VAE decoders and over all decoders.

The per-candidate ``psf_pearson_*`` fields are each fitted over that candidate's OWN above-chance
cells. Table X's PSF-order column instead uses ``psf_locked_vs_rho``: rho and each candidate
re-fitted over their COMMON above-chance (decoder, sigma_blur) cells, distinct-VAE slice (n=3
decoders), field ``cand_pearson_distinct_vae``.

Produces:
    Table X (Section IV-E3), scalar rows: pooled AUC1 / cross-cell AUC16
    (``feature_comparison.json``: ``scorecard/<candidate>/{det_auc1,det_auc_neff16}``) for
    rho, signed_diff (sign(rho_p)*||dr||), rho_x_diff (rho_p*||dr||), diff_energy (||dr||),
    abs_rho (|rho_p|), tex_norm_diff (TN ||dr||) and lowtex_diff (low-E_tex ||dr||); and the
    PSF-order column plus the omitted signed transforms (signed_sqrt_diff, signed_log1p_diff,
    signed_sq_diff) from ``psf_locked_vs_rho/<candidate>/cand_pearson_distinct_vae``. The
    paper's ``outputs/p0/feature_comparison.json`` predates the ``psf_locked_vs_rho`` key; the
    column can be recomputed on CPU from its stored scorecard by loading this module and calling
    ``_psf_locked_vs_rho(d["scorecard"], d["main_candidates"] + d["alt_candidates"])`` on the
    loaded JSON ``d``.

Inputs: ``outputs/p0/<decoder>/p0_table.json``, the server-side residual feature store
(``residual_v1.dat``, ``compact_rho.npy``), ``outputs/p0/master_table.json`` (sigma_eff per
cell) and ``outputs/sigma_calib/<decoder>.json`` (sigma_core). CPU only. Self-checks: the
recomputed rho must reproduce the stored ``rho_summary`` to 1e-3, and rho's ``det_auc1`` must
match ``--expect-rho-auc`` (0.662) within ``--gate-tol`` (0.03).

Outputs: ``outputs/p0/feature_comparison.json`` (or ``--out``) plus ``.md`` and ``.tex``
renderings next to it.

Usage:
    uv run --no-sync python scripts/analysis/feature_comparison.py [--features-root outputs/p0]
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
from scipy import stats

from cfgid.channel.rho import etex_keep_mask, image_rho_summary, winsorize
from cfgid.featurestore import read_compact, read_residual_v1
from cfgid.featurestore.schema import PATCH_ORDER_ROW_MAJOR
from cfgid.numerics import as_finite_float32, auc_gate_passed, finite_float, finite_or_none
from cfgid.paths import outputs_dir
from cfgid.sweep.evidence import _require_row_store_alignment, _residual_energy_features
from cfgid.sweep.neff_scaling import _load_p0_rows

MASK_Q = 0.2
WINSOR = (0.01, 0.99)
ETEX_EPS = (
    1e-6  # texture-energy floor for the *_tex_norm_* candidates (kept patches are high-E_tex)
)
_SIGNED_CHANNELS = frozenset(
    {
        "rho",
        "signed_diff",
        "signed_sqrt_diff",
        "signed_log1p_diff",
        "signed_sq_diff",
        "rho_x_diff",
        "signed_tex_norm_diff",
    }
)


def _load_sibling(name: str):
    path = Path(__file__).with_name(f"{name}.py")
    spec = importlib.util.spec_from_file_location(f"{name}_reuse", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_PGF = _load_sibling(
    "psf_grounding_features"
)  # _group_cells_with_feature, _feature_psf, _sigma_eff…
_FH = _PGF._FH  # the SAME selection-free fixed-H1 pooler used everywhere


def _candidates() -> dict:
    """name -> (fn(rho_p, diff_p, e_tex) -> per-patch, rationale). All store-only; NO re-inversion.

    Grouped: baseline / magnitude / signed-magnitude compression ladder / ablations / texture-norm.
    """
    return {
        # baseline -- the paper channel
        "rho": (lambda r, d, e: r, "signed log-ratio (paper channel, baseline)"),
        # pure magnitude -- stronger raw detector, but no sign
        "diff_energy": (lambda r, d, e: d, "unsigned ||r_cond - r_uncond|| magnitude"),
        # signed-magnitude compression ladder: identity / sqrt / log1p (sign x compressed magnitude)
        "signed_diff": (lambda r, d, e: np.sign(r) * d, "rho SIGN x diff magnitude"),
        "signed_sqrt_diff": (lambda r, d, e: np.sign(r) * np.sqrt(d), "rho SIGN x sqrt(magnitude)"),
        "signed_log1p_diff": (
            lambda r, d, e: np.sign(r) * np.log1p(d),
            "rho SIGN x log1p(magnitude)  [gentlest compression]",
        ),
        "signed_sq_diff": (
            lambda r, d, e: np.sign(r) * d * d,
            "rho SIGN x magnitude^2  [tail-emphasis end of the compression ladder]",
        ),
        # ablations
        "abs_rho": (lambda r, d, e: np.abs(r), "unsigned log-ratio (sign removed)"),
        "rho_x_diff": (lambda r, d, e: r * d, "signed, magnitude-weighted log-ratio"),
        # texture-normalized magnitude -- probes whether diff_energy rides the texture confound
        "tex_norm_diff": (
            lambda r, d, e: d / (e + ETEX_EPS),
            "unsigned magnitude / E_tex  [texture-confound probe]",
        ),
        "signed_tex_norm_diff": (
            lambda r, d, e: np.sign(r) * d / (e + ETEX_EPS),
            "rho SIGN x magnitude / E_tex",
        ),
    }


def _image_summary(per_patch: np.ndarray, e_tex: np.ndarray) -> float:
    return image_rho_summary(per_patch, e_tex, mask_quantile=MASK_Q, winsor_quantiles=WINSOR)


# --- alternative-aggregation candidates ----------------------------------------------------------
# These do NOT share the per-patch -> winsor-mean template, so they are reported in a separate
# "alt-aggregation" section and EXCLUDED from the apples-to-apples PSF cell-set audit; each still
# gets the full detection / sign / framework / PSF scorecard. Signature: fn(rho_p, diff_p, e_tex,
# patch_grid) -> per-image scalar. patch_grid = (rows, cols) of the latent patch grid (from meta).


def _agg_sign_coherence(
    rp: np.ndarray, dp: np.ndarray, et: np.ndarray, grid: tuple[int, int]
) -> float:
    """|mean sign(rho_p)| over high-E_tex patches: how unanimously SIGNED the field is, in [0, 1].

    Directly probes the "the sign is the signal" thesis -- a coherent decoder-specific sign field
    should separate H0/H1, while a magnitude with no sign structure should not.
    """
    rho = as_finite_float32("rho_p", rp)
    keep = etex_keep_mask(et, MASK_Q)
    kept = rho[keep] if keep.any() else rho
    if kept.size == 0:
        raise ValueError("patch_sign_coherence: no patches")
    return float(abs(np.mean(np.sign(kept.astype(np.float64)))))


def _agg_lowtex_diff(
    rp: np.ndarray, dp: np.ndarray, et: np.ndarray, grid: tuple[int, int]
) -> float:
    """Winsor-mean diff over LOW-E_tex patches (the complement of rho's high-E_tex mask).

    Localization probe: is the CFG trace in the textured regions rho keeps, or in the flat regions
    rho discards? If lowtex_diff still separates, the trace is not texture-localized.
    """
    diff = as_finite_float32("diff_p", dp)
    keep_lo = ~etex_keep_mask(et, MASK_Q)
    kept = diff[keep_lo] if keep_lo.any() else diff
    if kept.size == 0:
        raise ValueError("lowtex_diff: no patches")
    return float(np.mean(winsorize(kept, WINSOR[0], WINSOR[1])))


def _agg_spectral_slope(
    rp: np.ndarray, dp: np.ndarray, et: np.ndarray, grid: tuple[int, int]
) -> float:
    """Radial power-spectrum slope of the per-patch diff magnitude on the latent patch grid.

    Characterizes the SPATIAL-FREQUENCY structure of the residual field (a different axis than any
    per-patch reduction). Raises ValueError if the grid is structurally unusable (caller drops the
    candidate); returns 0.0 for a degenerate (flat) field with too few positive radial bins.
    """
    diff = as_finite_float32("diff_p", dp).astype(np.float64)
    rows, cols = int(grid[0]), int(grid[1])
    if rows * cols != diff.size:
        raise ValueError(f"spectral_slope_diff: grid {grid} != n_patches {diff.size}")
    if min(rows, cols) < 4:
        raise ValueError(f"spectral_slope_diff: grid {grid} too small for a radial spectrum")
    field = diff.reshape(rows, cols)
    field = field - float(field.mean())
    power = np.fft.fftshift(np.abs(np.fft.fft2(field)) ** 2)
    cy, cx = rows // 2, cols // 2
    yy, xx = np.indices((rows, cols))
    rbin = np.hypot(yy - cy, xx - cx).astype(int)
    freqs: list[float] = []
    pae: list[float] = []
    for k in range(1, int(rbin.max()) + 1):  # skip the DC bin (k=0)
        m = rbin == k
        if m.any():
            pk = float(power[m].mean())
            if pk > 0.0:
                freqs.append(float(k))
                pae.append(pk)
    if len(freqs) < 3:
        return 0.0
    try:
        slope = float(np.polyfit(np.log(freqs), np.log(pae), 1)[0])
    except np.linalg.LinAlgError as exc:
        raise ValueError("spectral_slope_diff: radial log-power polyfit failed") from exc
    if not np.isfinite(slope):
        raise ValueError("spectral_slope_diff: radial log-power polyfit was non-finite")
    return slope


def _alt_candidates() -> dict:
    """name -> (image-level aggregator fn(rho_p, diff_p, e_tex, grid) -> scalar, rationale)."""
    return {
        "patch_sign_coherence": (
            _agg_sign_coherence,
            "|mean sign(rho_p)| over high-E_tex patches  [sign-coherence, alt-agg]",
        ),
        "lowtex_diff": (
            _agg_lowtex_diff,
            "winsor-mean diff over LOW-E_tex patches  [texture-localization, alt-agg]",
        ),
        "spectral_slope_diff": (
            _agg_spectral_slope,
            "radial power-spectrum slope of the diff field  [spatial structure, alt-agg]",
        ),
    }


def _is_row_major_patch_store(meta: object) -> bool:
    """Feature-store patches are spatially reconstructable only under row-major patch order."""
    return str(getattr(meta, "patch_order", PATCH_ORDER_ROW_MAJOR)) == PATCH_ORDER_ROW_MAJOR


def _validate_e_tex(image_id: str, e_tex: np.ndarray) -> None:
    """Reject corrupted E_tex stores before texture-normalized candidates divide by it."""
    try:
        arr = np.asarray(e_tex, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise SystemExit(f"image {image_id}: E_tex must be numeric.") from exc
    if not np.all(np.isfinite(arr)):
        raise SystemExit(f"image {image_id}: E_tex contains non-finite values.")
    if np.any(arr < 0.0):
        raise SystemExit(f"image {image_id}: E_tex contains negative energy values.")


def _majority_sign(signs: list[float]) -> float | None:
    n_pos = sum(1 for s in signs if s > 0)
    n_neg = sum(1 for s in signs if s < 0)
    if n_pos == n_neg:
        return None
    return 1.0 if n_pos > n_neg else -1.0


def _sign_loocv(recs: list) -> tuple[float | None, int]:
    """Leave-one-cell-out decoder-majority sign-rule accuracy (the Section IV-E4 decoder rule).

    Only held-out cells WITH same-decoder support are scored, and ties are skipped -- so a pure
    magnitude with no decoder-local sign structure does not borrow a global majority.
    Returns (accuracy, n_scored).
    """
    if len(recs) < 3:
        return None, 0
    signs = [r.sign for r in recs]
    decs = [r.decoder for r in recs]
    correct = 0
    total = 0
    for i in range(len(recs)):
        same = [signs[j] for j in range(len(recs)) if j != i and decs[j] == decs[i]]
        if not same:
            continue
        pred = _majority_sign(same)
        if pred is None:
            continue
        correct += int(pred == signs[i])
        total += 1
    return (correct / total if total else None), total


def _has_decoder_sign(
    n_pos: int, n_neg: int, sign_loocv_acc: float | None, *, is_signed_channel: bool
) -> bool:
    """Whether the candidate carries rho's semantic sign, not merely a fitted orientation."""
    return bool(is_signed_channel and n_pos > 0 and n_neg > 0 and sign_loocv_acc is not None)


def _scorecard(
    root: Path,
    rows_all: list[dict],
    fmap: dict[str, float],
    se_by: dict,
    axis_modes: dict[str, str],
    h1_w: int,
    *,
    is_signed_channel: bool,
    n_groups: int,
    n_perm: int,
) -> dict:
    """Detection / sign / Gaussian-fit / PSF scorecard of one per-image feature map ``fmap``.

    Cells are the stored (decoder, sigma_blur) cells; H1 is fixed at ``h1_w`` vs w=1, each cell
    standardized on FIT-H0 and oriented by its FIT sign; detection numbers are on pooled TEST.
    """
    cells = _PGF._group_cells_with_feature(rows_all, fmap)
    recs = _FH.collect_fixed_h1_records(cells, (h1_w,), min_n=8)
    if not recs:
        return {"n_cells": 0}
    summary = _FH._summarize(
        recs, k_grid=(1, 2, 4, 8, 16), n_groups=n_groups, n_perm=n_perm, seed=0
    )
    _zf0, _zf1, _zt0, zt1 = _FH._aligned_pools(recs)
    emp = np.asarray(summary["auc_emp"], float)
    pred = np.asarray(summary["auc_pred"], float)
    psf = _PGF._feature_psf(root, cells, h1_w, se_by, axis_modes)  # SAME sigma_eff axis per decoder
    n_pos = int(sum(1 for r in recs if r.sign > 0))
    n_neg = int(sum(1 for r in recs if r.sign < 0))
    sign_loocv_acc, sign_loocv_n = _sign_loocv(recs)
    normal = stats.normaltest(zt1) if zt1.size >= 20 else None
    per_decoder = [
        {
            "decoder": d["decoder"],
            "sigma_core": d["sigma_core"],
            "axis": d["axis"],
            "decay_p": d["decay_exponent_p"],
            "fit_r2": d["fit_r2"],
            "n_points": d["n_points"],
            "cells": d.get("cells", []),
        }
        for d in psf["per_decoder"]
    ]
    return {
        "det_auc1": finite_float("det_auc1", summary["auc1_pooled_fitsign"]),
        "det_auc_neff16": finite_or_none(emp[-1]) if emp.size else None,
        "n_cells": summary["n_cells"],
        "n_sign_pos": n_pos,
        "n_sign_neg": n_neg,
        "is_signed_channel": bool(is_signed_channel),
        "has_decoder_sign": _has_decoder_sign(
            n_pos, n_neg, sign_loocv_acc, is_signed_channel=is_signed_channel
        ),
        "sign_loocv_acc": sign_loocv_acc,
        "sign_loocv_n": sign_loocv_n,
        "sqrtk_exponent": summary["exponent"],
        "gauss_pred_mismatch": finite_float("gauss_pred_mismatch", np.mean(np.abs(emp - pred))),
        "h1_skew": finite_or_none(stats.skew(zt1)),
        "h1_excess_kurtosis": finite_or_none(stats.kurtosis(zt1)),
        "h1_normaltest_p": finite_or_none(normal[1]) if normal is not None else None,
        "psf_pearson_distinct_vae": psf["pearson_distinct_vae"],
        "psf_pearson_all": psf["pearson_all"],
        "psf_n_distinct_vae": psf["n_distinct_vae"],
        "psf_per_decoder": per_decoder,
    }


def _fmt(v: object, spec: str, na: str = "n/a") -> str:
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return na
    return format(v, spec)


_MD_HEADER = (
    "| feature | det@N1 | det@N16 | sign? | n+/n- | signLOO | sqrtK-exp | gMiss | "
    "normal-p(H1) | PSF-r(dist) | PSF-r(all) |"
)
_MD_SEP = "|---|---|---|---|---|---|---|---|---|---|---|"


def _md_row(name: str, s: dict) -> str:
    if s.get("n_cells", 0) == 0:
        return f"| `{name}` | (no cells) | | | | | | | | | |"
    pm = f"{s['n_sign_pos']}/{s['n_sign_neg']}"
    return (
        f"| `{name}` | {_fmt(s['det_auc1'], '.3f')} | {_fmt(s['det_auc_neff16'], '.3f')} | "
        f"{'yes' if s['has_decoder_sign'] else 'NO'} | {pm} | "
        f"{_fmt(s['sign_loocv_acc'], '.2f')} | {_fmt(s['sqrtk_exponent'], '+.3f')} | "
        f"{_fmt(s['gauss_pred_mismatch'], '.4f')} | {_fmt(s['h1_normaltest_p'], '.1e')} | "
        f"{_fmt(s['psf_pearson_distinct_vae'], '+.2f')} | "
        f"{_fmt(s['psf_pearson_all'], '+.2f')} |"
    )


def _markdown(
    scores: dict, order: list[str], alt_order: list[str], defs: dict, psf_locked: dict
) -> str:
    lines = [
        "### Candidate-channel comparison (selection-free fixed-H1 all-cell; rho self-check ok)",
        "",
        "Main table -- per-patch -> winsor-mean aggregation (apples-to-apples with rho).",
        "",
        _MD_HEADER,
        _MD_SEP,
    ]
    lines += [_md_row(name, scores[name]) for name in order]
    if alt_order:
        lines += [
            "",
            "Alt-aggregation candidates -- DIFFERENT per-image aggregation (NOT apples-to-apples; "
            "excluded from the PSF cell-set audit).",
            "",
            _MD_HEADER,
            _MD_SEP,
        ]
        lines += [_md_row(name, scores[name]) for name in alt_order]
    lines += [
        "",
        "### Locked PSF-r -- QUOTABLE (rho vs candidate over their COMMON above-chance cells)",
        "",
        "The per-feature PSF-r columns above are each over the feature's OWN surviving cells (the "
        "deflection drop is feature-specific). These re-fit rho and the candidate over the SAME "
        "common (decoder, sigma_blur) cells -- the apples-to-apples number to quote.",
        "",
        "| feature | rho r(dist) | cand r(dist) | rho r(all) | cand r(all) | n_dec |",
        "|---|---|---|---|---|---|",
    ]
    for name in order + alt_order:
        lk = psf_locked.get(name, {})
        lines.append(
            f"| `{name}` | {_fmt(lk.get('rho_pearson_distinct_vae'), '+.2f')} | "
            f"{_fmt(lk.get('cand_pearson_distinct_vae'), '+.2f')} | "
            f"{_fmt(lk.get('rho_pearson_all'), '+.2f')} | "
            f"{_fmt(lk.get('cand_pearson_all'), '+.2f')} | "
            f"{lk.get('n_decoders_distinct_vae', 0)} |"
        )
    lines += ["", "### Per-decoder PSF-ordering (decay exponent p @ measured sigma_core)", ""]
    lines.append("| feature | decoder | sigma_core | decay_p | fit_r2 | n_pts |")
    lines.append("|---|---|---|---|---|---|")
    for name in order + alt_order:
        for d in scores[name].get("psf_per_decoder", []):
            lines.append(
                f"| `{name}` | {d['decoder']} | {_fmt(d['sigma_core'], '.1f')} | "
                f"{_fmt(d['decay_p'], '+.2f')} | {_fmt(d['fit_r2'], '.2f')} | {d['n_points']} |"
            )
    lines += ["", "Candidate definitions (all store-only, no re-inversion):", ""]
    for name in order + alt_order:
        lines.append(f"- `{name}` -- {defs[name]}")
    return "\n".join(lines) + "\n"


def _latex(scores: dict, order: list[str]) -> str:
    rows = []
    for name in order:
        s = scores[name]
        if s.get("n_cells", 0) == 0:
            continue
        rows.append(
            f"{name.replace('_', chr(92) + '_')} & {_fmt(s['det_auc1'], '.3f', '--')} & "
            f"{_fmt(s['det_auc_neff16'], '.3f', '--')} & "
            f"{'yes' if s['has_decoder_sign'] else 'no'} & "
            f"{_fmt(s['sign_loocv_acc'], '.2f', '--')} & "
            f"{_fmt(s['sqrtk_exponent'], '.2f', '--')} & "
            f"{_fmt(s['gauss_pred_mismatch'], '.4f', '--')} & "
            f"{_fmt(s['psf_pearson_distinct_vae'], '+.2f', '--')} \\\\"
        )
    body = "\n".join(rows)
    return (
        "\\begin{table}[t]\n\\centering\n"
        "\\caption{Candidate forensic channels (selection-free fixed-H1 all-cell). "
        "$\\mathrm{AUC}_1$/$\\mathrm{AUC}_{16}$ at $N_\\mathrm{eff}=1$/$16$; "
        "PSF-$r$=Pearson(decay exponent, $\\sigma_\\mathrm{core}$), distinct-VAE decoders.}\n"
        "\\label{tab:feature-comparison}\n"
        "\\begin{tabular}{lcccccc c}\n\\toprule\n"
        "channel & $\\mathrm{AUC}_1$ & $\\mathrm{AUC}_{16}$ & sign & signLOO & "
        "$\\sqrt{K}$ exp & gMiss & PSF-$r$ \\\\\n\\midrule\n"
        f"{body}\n"
        "\\bottomrule\n\\end{tabular}\n\\end{table}\n"
    )


def _cellset_from_psf_decoder(record: dict) -> tuple[float, ...]:
    cells = record.get("cells")
    if not isinstance(cells, list):
        return ()
    return tuple(sorted(finite_float("psf cell sigma_blur", c["sigma_blur"]) for c in cells))


def _psf_cellset_audit(scores: dict, order: list[str]) -> dict:
    """Cross-feature apples-to-apples check for the PSF-ordering comparison.

    The PSF deflection drop (AUC<=0.5 -> None -> cell skipped) in _feature_psf is FEATURE-specific,
    so two candidates can silently fit each decoder's power-law over a different number of
    (decoder, sigma_blur) points -- or over the same count but DIFFERENT sigma_blur cells -- which
    would make the headline rho-vs-candidate PSF-r comparison apples-to-oranges. Identical
    per-decoder sigma_blur cell sets across all candidates (and all candidates present for every
    decoder) is the auditable signal that the comparison is over the SAME cells.
    """
    n_by_decoder: dict[str, dict[str, int]] = {}
    cells_by_decoder: dict[str, dict[str, tuple[float, ...]]] = {}
    for name in order:
        for d in scores.get(name, {}).get("psf_per_decoder", []):
            dec = str(d["decoder"])
            n_by_decoder.setdefault(dec, {})[name] = int(d["n_points"])
            cells_by_decoder.setdefault(dec, {})[name] = _cellset_from_psf_decoder(d)
    per_decoder = {}
    all_consistent = bool(cells_by_decoder)
    for dec, fmap in sorted(cells_by_decoder.items()):
        nmap = n_by_decoder.get(dec, {})
        present = set(fmap) == set(order)
        n_identical = present and len({nmap[name] for name in order}) == 1
        cellsets_match_counts = present and all(len(fmap[name]) == nmap[name] for name in order)
        cellsets_identical = cellsets_match_counts and len({fmap[name] for name in order}) == 1
        all_consistent = all_consistent and present and cellsets_identical
        per_decoder[dec] = {
            "n_points_by_feature": nmap,
            "sigma_blur_cells_by_feature": {name: list(fmap[name]) for name in sorted(fmap)},
            "all_features_present": present,
            "n_points_identical": n_identical,
            "cellsets_match_n_points": cellsets_match_counts,
            "cellsets_identical": cellsets_identical,
        }
    return {"all_consistent": all_consistent, "by_decoder": per_decoder}


def _cells_by_decoder(scorecard: dict) -> dict[str, dict]:
    """{decoder: {"sigma_core": float|None, "cells": {sigma_blur: (axis_value, deflection_d)}}}.

    Pulls the actual per-decoder fitted cells (sigma_blur / axis_value / deflection) that
    _feature_psf records, so a restricted re-fit over a common cell subset is possible WITHOUT
    re-running the fixed-H1 pooler.
    """
    out: dict[str, dict] = {}
    for d in scorecard.get("psf_per_decoder", []):
        cells = {
            finite_float("sigma_blur", c["sigma_blur"]): (
                finite_float("axis_value", c["axis_value"], positive=True),
                finite_float("deflection_d", c["deflection_d"], positive=True),
            )
            for c in d.get("cells", [])
        }
        out[str(d["decoder"])] = {"sigma_core": d.get("sigma_core"), "cells": cells}
    return out


def _locked_decay_p(
    cells: dict[float, tuple[float, float]], common_sb: list[float]
) -> float | None:
    """Power-law decay exponent re-fit over the common (sigma_blur) cells only (>=2 points)."""
    xs = [cells[sb][0] for sb in common_sb]
    ys = [cells[sb][1] for sb in common_sb]
    return finite_or_none(_PGF._PSF._powerlaw_fit(xs, ys)["p"])


def _pearson_xy(pts: list[tuple[float, float]]) -> float | None:
    if len(pts) < 3:
        return None
    c = np.corrcoef([p[0] for p in pts], [p[1] for p in pts])[0, 1]
    return float(c) if np.isfinite(c) else None


def _psf_locked_vs_rho(scores: dict, order: list[str]) -> dict:
    """Strictly apples-to-apples PSF-ordering: rho vs each candidate over their COMMON above-chance
    cells.

    The per-feature psf_pearson_* are fit over each feature's OWN surviving cells (the deflection
    drop at AUC<=0.5 is feature-specific), which the cell-set audit flags as divergent. This re-fits
    rho's and the candidate's per-decoder decay exponent over only the (decoder, sigma_blur) cells
    where BOTH have an above-chance deflection, then Pearson(decay_p, sigma_core). Each row reports
    rho's and the candidate's Pearson over the IDENTICAL common cells -- the number safe to quote.
    """
    shared = _PGF._PSF._SHARED_VAE_DECODERS
    rho_by_dec = _cells_by_decoder(scores.get("rho", {}))
    out: dict[str, dict] = {}
    for name in order:
        cand_by_dec = _cells_by_decoder(scores.get(name, {}))
        rho_all: list[tuple[float, float]] = []
        cand_all: list[tuple[float, float]] = []
        rho_dist: list[tuple[float, float]] = []
        cand_dist: list[tuple[float, float]] = []
        common_counts: dict[str, int] = {}
        for dec in sorted(set(rho_by_dec) & set(cand_by_dec)):
            score = rho_by_dec[dec]["sigma_core"]
            common_sb = sorted(set(rho_by_dec[dec]["cells"]) & set(cand_by_dec[dec]["cells"]))
            common_counts[dec] = len(common_sb)
            if score is None or len(common_sb) < 2:
                continue
            rp = _locked_decay_p(rho_by_dec[dec]["cells"], common_sb)
            cp = _locked_decay_p(cand_by_dec[dec]["cells"], common_sb)
            if rp is None or cp is None:
                continue
            rho_all.append((float(score), rp))
            cand_all.append((float(score), cp))
            if dec not in shared:
                rho_dist.append((float(score), rp))
                cand_dist.append((float(score), cp))
        out[name] = {
            "n_common_cells_by_decoder": common_counts,
            "n_decoders_distinct_vae": len(cand_dist),
            "rho_pearson_distinct_vae": _pearson_xy(rho_dist),
            "cand_pearson_distinct_vae": _pearson_xy(cand_dist),
            "rho_pearson_all": _pearson_xy(rho_all),
            "cand_pearson_all": _pearson_xy(cand_all),
        }
    return out


def run(
    features_root: Path,
    out_path: Path,
    *,
    h1_w: int = 20,
    expect_rho_auc: float = 0.662,
    gate_tol: float = 0.03,
    n_groups: int = 2000,
    n_perm: int = 100,
) -> int:
    decoder_dirs = sorted(d for d in features_root.glob("*/") if (d / "p0_table.json").exists())
    if not decoder_dirs:
        print(f"[skip] no <decoder>/p0_table.json under {features_root}")
        return 0
    cand = _candidates()
    alt = _alt_candidates()
    feat_maps: dict[str, dict[str, float]] = {name: {} for name in cand}
    alt_feat_maps: dict[str, dict[str, float]] = {name: {} for name in alt}
    alt_disabled: set[str] = set()
    rho_stored: dict[str, float] = {}
    rows_all: list[dict] = []
    for dec_dir in decoder_dirs:
        rows = _load_p0_rows(dec_dir / "p0_table.json")
        rows_all.extend(rows)
        for r in rows:
            rho_stored[str(r["image_id"])] = finite_float("rho_summary", r["rho_summary"])
        if not (dec_dir / "residual_v1.dat").exists() or not (dec_dir / "compact_rho.npy").exists():
            print(f"[skip] {dec_dir.name}: no feature store (needs residual_v1 + compact_rho).")
            return 0
        batch, meta = read_compact(dec_dir)
        residual, _ = read_residual_v1(dec_dir)
        _require_row_store_alignment(rows, meta)
        if "spectral_slope_diff" not in alt_disabled and not _is_row_major_patch_store(meta):
            print(
                "[note] alt candidate 'spectral_slope_diff' unavailable "
                f"(patch_order={getattr(meta, 'patch_order', None)!r}); dropping it."
            )
            alt_disabled.add("spectral_slope_diff")
        ids = list(meta.image_ids) or [str(r["image_id"]) for r in rows]
        diff_pp = _residual_energy_features(residual)  # (N, P)
        rho_pp = batch.rho  # (N, P)
        for i, iid in enumerate(ids):
            rp, dp, et = rho_pp[i], diff_pp[i], batch.e_tex[i]
            _validate_e_tex(iid, et)
            for name, (fn, _doc) in cand.items():
                vals = np.asarray(fn(rp, dp, et), dtype=np.float64)
                if not np.all(np.isfinite(vals)):
                    raise SystemExit(f"candidate {name!r} produced non-finite per-patch values.")
                feat_maps[name][iid] = _image_summary(vals, et)
            for name, (afn, _adoc) in alt.items():
                if name in alt_disabled:
                    continue
                try:  # structural failures (e.g. unusable grid) drop the whole alt candidate
                    alt_feat_maps[name][iid] = finite_float(name, afn(rp, dp, et, meta.patch_grid))
                except ValueError as exc:
                    print(f"[note] alt candidate {name!r} unavailable ({exc}); dropping it.")
                    alt_disabled.add(name)

    for name in alt_disabled:
        alt_feat_maps.pop(name, None)
    alt_order = [name for name in alt if name not in alt_disabled]

    err = max(
        finite_float("rho reproduction error", abs(feat_maps["rho"][i] - rho_stored[i]))
        for i in rho_stored
        if i in feat_maps["rho"]
    )
    if err > 1e-3:
        raise SystemExit(f"rho reproduction off by {err:.4g}; refusing to emit numbers.")
    se_by = _PGF._sigma_eff_by_cell(features_root)
    axis_modes = _PGF._sigma_axis_modes(
        features_root, rows_all, se_by
    )  # one shared axis per decoder
    scores = {
        name: _scorecard(
            features_root,
            rows_all,
            fmap,
            se_by,
            axis_modes,
            h1_w,
            is_signed_channel=name in _SIGNED_CHANNELS,
            n_groups=n_groups,
            n_perm=n_perm,
        )
        for name, fmap in {**feat_maps, **alt_feat_maps}.items()
    }
    rho_auc = scores["rho"].get("det_auc1")
    if not auc_gate_passed(rho_auc, expected=expect_rho_auc, tol=gate_tol):
        raise SystemExit(
            f"SELF-CHECK FAILED: rho det_auc1 {rho_auc} != {expect_rho_auc}+/-{gate_tol}."
        )

    order = list(cand)  # apples-to-apples per-patch main table
    psf_cellset = _psf_cellset_audit(scores, order)  # audit only the same-template main candidates
    # The quotable PSF-ordering: re-fit rho vs each candidate over their COMMON above-chance cells,
    # so the rho-vs-candidate Pearson is strictly apples-to-apples even when the audit flags drift.
    psf_locked = _psf_locked_vs_rho(scores, order + alt_order)
    defs = {name: cand[name][1] for name in order}
    defs.update({name: alt[name][1] for name in alt_order})
    result = {
        "analysis": "feature_comparison",
        "h1_w": h1_w,
        "self_check": {"rho_det_auc1": rho_auc, "expected": expect_rho_auc, "passed": True},
        "decoders": sorted({str(r["decoder"]) for r in rows_all}),
        "main_candidates": order,
        "alt_candidates": alt_order,
        "scorecard": scores,
        "psf_cellset_audit": psf_cellset,
        "psf_locked_vs_rho": psf_locked,
        "candidate_definitions": defs,
        "note": (
            "Paper-ready candidate-channel comparison. A strictly better channel beats rho on "
            "det_auc1 AND keeps has_decoder_sign=true with a high sign_loocv_acc (Section "
            "IV-E4) AND a comparable psf_pearson_distinct_vae / gauss_pred_mismatch (Table X + "
            "the Gaussian-location framework). The sign flag is only true for candidates that "
            "carry rho's semantic sign by construction; unsigned features still report "
            "fit-orientation "
            "counts and signLOO diagnostically. The signed-magnitude ladder (signed_diff / "
            "signed_sqrt_diff / signed_log1p_diff / signed_sq_diff) tests how much magnitude "
            "compression keeps both detection and framework; the *_tex_norm_* candidates probe "
            "whether diff_energy rides the texture confound. The alt_candidates use a DIFFERENT "
            "aggregation (sign-coherence / low-E_tex / spatial spectrum) so they are reported "
            "separately and EXCLUDED from the per-patch apples-to-apples PSF cell-set audit. "
            "NO re-inversion."
        ),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    md_path = out_path.with_suffix(".md")
    md_path.write_text(_markdown(scores, order, alt_order, defs, psf_locked), encoding="utf-8")
    tex_path = out_path.with_suffix(".tex")
    tex_path.write_text(_latex(scores, order), encoding="utf-8")

    def _print_row(name: str) -> None:
        s = scores[name]
        if s.get("n_cells", 0) == 0:
            print(f"  {name:<22} (no cells)")
            return
        pm = f"{s['n_sign_pos']}/{s['n_sign_neg']}"
        print(
            f"  {name:<22} {_fmt(s['det_auc1'], '.3f'):>5} {_fmt(s['det_auc_neff16'], '.3f'):>5} "
            f"{'yes' if s['has_decoder_sign'] else 'NO':>4} {pm:>6} "
            f"{_fmt(s['sign_loocv_acc'], '.2f'):>5} {_fmt(s['sqrtk_exponent'], '.2f'):>5} "
            f"{_fmt(s['gauss_pred_mismatch'], '.4f'):>6} "
            f"{_fmt(s['psf_pearson_distinct_vae'], '+.2f'):>6} "
            f"{_fmt(s['psf_pearson_all'], '+.2f'):>6}"
        )

    print("\nComprehensive feature comparison (fixed-H1 all-cell, self-check PASSED):")
    print(
        f"  {'feature':<22} {'N1':>5} {'N16':>5} {'sign':>4} {'+/-':>6} {'sLOO':>5} "
        f"{'exp':>5} {'gMiss':>6} {'PSF-d':>6} {'PSF-a':>6}"
    )
    for name in order:
        _print_row(name)
    if alt_order:
        print("  -- alt-aggregation (NOT apples-to-apples; excluded from PSF cell-set audit) --")
        for name in alt_order:
            _print_row(name)
    if psf_cellset["all_consistent"]:
        print("  PSF apples-to-apples: OK (all candidates fit each decoder over the same cells)")
    else:
        bad = [
            d
            for d, v in psf_cellset["by_decoder"].items()
            if not v["all_features_present"] or not v["cellsets_identical"]
        ]
        print(
            f"  PSF apples-to-apples: DIVERGENT cell sets at "
            f"{bad or 'missing-feature decoders'} -- per-feature PSF-r is over different cells; "
            "use the LOCKED PSF-r below (or psf_cellset_audit) before quoting"
        )
    print(
        "\nLocked PSF-r (QUOTABLE: rho vs candidate re-fit over their COMMON above-chance cells):"
    )
    print(f"  {'feature':<22} {'rho_p':>6} {'cand_p':>6} {'rho_pa':>6} {'cand_pa':>7} {'n_dec':>5}")
    for name in order + alt_order:
        lk = psf_locked.get(name, {})
        print(
            f"  {name:<22} {_fmt(lk.get('rho_pearson_distinct_vae'), '+.2f'):>6} "
            f"{_fmt(lk.get('cand_pearson_distinct_vae'), '+.2f'):>6} "
            f"{_fmt(lk.get('rho_pearson_all'), '+.2f'):>6} "
            f"{_fmt(lk.get('cand_pearson_all'), '+.2f'):>7} "
            f"{lk.get('n_decoders_distinct_vae', 0):>5}"
        )
    print(
        "  (rho_p/cand_p = distinct-VAE Pearson over the SAME common cells; this is the headline)"
    )
    print(f"-> {out_path}\n-> {md_path}\n-> {tex_path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="feature_comparison")
    ap.add_argument(
        "--features-root", default=None, help="dir of <decoder>/ stores (def: outputs/p0)"
    )
    ap.add_argument("--out", default=None, help="def: outputs/p0/feature_comparison.json")
    ap.add_argument("--expect-rho-auc", type=float, default=0.662)
    ap.add_argument("--gate-tol", type=float, default=0.03)
    ap.add_argument("--n-groups", type=int, default=2000)
    ap.add_argument("--n-perm", type=int, default=100)
    args = ap.parse_args(argv)
    features_root = Path(args.features_root) if args.features_root else (outputs_dir() / "p0")
    out_path = Path(args.out) if args.out else (outputs_dir() / "p0" / "feature_comparison.json")
    return run(
        features_root,
        out_path,
        expect_rho_auc=args.expect_rho_auc,
        gate_tol=args.gate_tol,
        n_groups=args.n_groups,
        n_perm=args.n_perm,
    )


if __name__ == "__main__":
    raise SystemExit(main())
