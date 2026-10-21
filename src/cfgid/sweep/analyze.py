"""Main-grid analysis: stored p0 artifacts -> gate evidence -> three-state decision memo + figure.

Runs the preregistered knee-vs-w* alignment gate of configs/p0.yaml on the gate decoders
(``grid.decoders`` = SD1.5, SD2.1, SDXL, the distinct-VAE slice); other decoders found under the
same root (PixArt-Sigma, HunyuanDiT) are dropped with a notice. Every gate input is computed from
persisted artifacts under the prompt-grouped splits; no placeholder may drive the decision.
Predictive track: sigma_eff per blur level from the calibration artifact (cfgid.channel.sigma_calib;
a nominal fallback is flagged and forces INCONCLUSIVE), g_sigma / v null-adapted on the
``null_adapt`` (NULL) split from H0 rows only, and knees, negative controls and the sufficiency
probe evaluated on the ``test`` (TEST) split. Per (decoder, sigma_blur) cell: robust knee of the
TEST AUC-vs-w curve with a bootstrap CI, and the analytic w* of Eq. (15) at N_eff=1 from the
measured H0 E_tex and the NULL-split v. Cells with w* > max(grid.w) are right-censored and excluded
from the knee-vs-w* WLS regression.

Produces: Table VIII rows 1-2 (Section IV-E1) from the memo's "Negative controls" section (pooled
rho AUC on the TEST split vs each nuisance baseline, and the weakest cell with its rho and best
nuisance AUC), and the knee-vs-w* alignment verdict (INCONCLUSIVE) cited in Section IV-E7.
Inputs: outputs/p0/<decoder>/p0_table.json together with the stored PNGs (nuisance baselines),
oracle traces and the compact/residual feature store of each gate decoder (server-side
artifacts), plus outputs/sigma_calib/<decoder>.json; CLIP weights only if the CLIP
baseline is enabled. CPU (CLIP uses the negotiated device).
Outputs: outputs/p0/decision_memo.md, outputs/p0/alignment_cells.json and
outputs/p0/alignment_scatter.png (when matplotlib is installed). Data-gated: skips with a notice
if no main-grid tables exist.
Usage:
  uv run --no-sync python -m cfgid.sweep.analyze [--features outputs/p0] [--out outputs/p0]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import cast

import numpy as np

from cfgid.align.gate import GateInputs, GateResult, evaluate_gate
from cfgid.align.knee import KneeCI, knee_bootstrap_ci
from cfgid.align.regression import RegressionResult, fit_alignment
from cfgid.align.report import build_decision_memo, save_alignment_scatter
from cfgid.bounds.wstar_cov import wstar_cov
from cfgid.bounds.wstar_p0 import WStarParams, wstar, wstar_from_measured_etex
from cfgid.channel.params.gsigma import synthetic_fit_gsigma
from cfgid.channel.sigma_probe import sigma_direction_consistent
from cfgid.config import load_models_config
from cfgid.numerics import loads_strict_json
from cfgid.p0config import P0Config, load_p0_config
from cfgid.paths import checkpoints_dir, outputs_dir
from cfgid.sweep.evidence import (
    EvidenceUnavailableError,
    NegativeControls,
    OracleEvidence,
    Row,
    SigmaEvidence,
    SufficiencyEvidence,
    WrongPromptEvidence,
    _row_finite_float,
    cov_channel_for_cell,
    load_sigma_evidence,
    negative_controls,
    null_params_for_cell,
    oracle_evidence,
    predictive_etex_for_cell,
    split_rows,
    sufficiency_evidence,
    validate_table_rows,
    wrong_prompt_evidence,
)

_KNEE_STD_RIDGE_FLOOR = 0.05


def _load_tables(features_root: Path) -> list[Row]:
    rows: list[Row] = []
    for table_path in sorted(features_root.glob("*/p0_table.json")):
        raw = loads_strict_json(table_path.read_text(encoding="utf-8"))
        if not isinstance(raw, list):
            raise ValueError(f"{table_path}: p0_table.json must contain a list of rows")
        if any(not isinstance(row, dict) for row in raw):
            raise ValueError(f"{table_path}: p0_table.json rows must be objects")
        rows.extend(cast(list[Row], raw))
    return rows


def _effective_branch_mode(row: Row, model_branch_modes: dict[str, str]) -> str:
    """Row branch mode, with a safe legacy fallback for older tables without the field.

    Old SD-only main-grid artifacts did not stamp ``branch_mode`` in every row, while
    metadata/config still identify the decoder. Missing row fields therefore fall back to
    ``models.yaml``. Unknown decoders fail closed as non-core instead of silently entering the
    sigma-w* regression.
    """
    raw = row.get("branch_mode")
    if raw is not None:
        return str(raw)
    return model_branch_modes.get(str(row.get("decoder", "")), "unknown_missing_branch_mode")


def _require_qf_collapsed_to_none(rows: list[Row], cfg: P0Config) -> None:
    """Alignment cells are keyed by (decoder, sigma_blur); reject any live QF axis.

    configs/p0.yaml deliberately collapses QF to none-only. If stale or ad-hoc artifacts
    contain compressed QF rows, grouping only by sigma_blur would silently pool distinct channels
    into one alignment cell. Fail loud instead of letting residual QF variation drive the sigma
    regression.
    """
    expected_qfs = {-1 if qf is None else int(qf) for qf in cfg.grid.qf}
    if expected_qfs != {-1}:
        raise ValueError(
            "Alignment requires configs/p0.yaml grid.qf to be [none]; "
            f"got normalized QF sentinels {sorted(expected_qfs)}"
        )
    observed_qfs = {int(r["qf"]) for r in rows}
    extra = sorted(observed_qfs - {-1})
    if extra:
        raise ValueError(
            "Alignment cells are (decoder, sigma_blur) with QF collapsed to none; "
            f"found non-none qf row value(s) {extra}. Refusing to pool QF tiers."
        )


def _restrict_to_grid_decoders(
    rows: list[Row], cfg: P0Config, *, known_decoders: set[str] | None = None
) -> list[Row]:
    """Drop known co-located decoders not in ``grid.decoders`` before the gate check.

    A breadth decoder (e.g. ``pixart_sigma`` or ``hunyuandit``) is written under the SAME
    ``outputs/p0/`` root so the cross-decoder analyses can pool it, but the FROZEN gate runs ONLY on
    its preregistered ``grid.decoders``. Without this, one co-located dir makes
    ``_require_rows_match_config_grid`` raise and poisons the entire multi-decoder gate run.
    Filtering (not raising) keeps the gate on its 3 decoders while leaving the breadth decoders to
    the cross-decoder analyses; add a decoder to ``grid.decoders`` to opt it INTO the gate
    regression.

    Unknown extras are still rejected: a stale/typo decoder directory must not be silently treated
    as an intentional breadth track.
    """
    expected = set(cfg.grid.decoders)
    known = known_decoders if known_decoders is not None else set(load_models_config())
    extra = sorted({str(r["decoder"]) for r in rows} - expected)
    if not extra:
        return rows
    unknown_extra = sorted(set(extra) - known)
    if unknown_extra:
        raise ValueError(
            "Main-grid artifacts contain decoder(s) outside configs/p0.yaml grid.decoders and "
            "models.yaml: "
            f"{unknown_extra}. Refusing to ignore unknown/stale decoder artifacts."
        )
    print(
        f"[notice] analyze: ignoring co-located decoder(s) {extra} not in grid.decoders "
        f"{list(cfg.grid.decoders)} -- the frozen gate runs on grid.decoders only (they remain in "
        "the cross-decoder breadth analyses). Add to grid.decoders to include them in the gate."
    )
    return [r for r in rows if str(r["decoder"]) in expected]


def _require_rows_match_config_grid(rows: list[Row], cfg: P0Config) -> None:
    """Reject stale/ad-hoc main-grid artifacts whose observed grid is not the current config grid.

    The censor guard is defined against ``max(config.grid.w)``. If old artifacts with a lower
    observed w ceiling were analyzed under the extended config, right-censored knees could enter
    the regression as if the high-w cells had been measured. Fail before any evidence assembly.
    """
    expected_decoders = set(cfg.grid.decoders)
    observed_decoders = sorted({str(r["decoder"]) for r in rows})
    missing_decoders = sorted(expected_decoders - set(observed_decoders))
    if missing_decoders:
        raise ValueError(
            "Main-grid artifacts are missing decoder(s) required by configs/p0.yaml grid.decoders: "
            f"{missing_decoders}. Regenerate the missing gate decoder artifacts before running "
            "the frozen gate."
        )
    unknown_decoders = sorted(set(observed_decoders) - expected_decoders)
    if unknown_decoders:
        raise ValueError(
            "Main-grid artifacts contain decoder(s) outside configs/p0.yaml grid.decoders: "
            f"{unknown_decoders}. Regenerate with the current configs/p0.yaml."
        )

    expected_w = tuple(sorted(int(w) for w in cfg.grid.w))
    expected_sb = tuple(sorted(float(sb) for sb in cfg.grid.sigma_blur))
    for decoder in observed_decoders:
        drows = [r for r in rows if str(r["decoder"]) == decoder]
        observed_w = tuple(sorted({int(r["w"]) for r in drows}))
        if observed_w != expected_w:
            raise ValueError(
                f"{decoder}: artifact grid.w levels {observed_w} do not match "
                f"configs/p0.yaml grid.w {expected_w}. Regenerate features with the current grid."
            )
        observed_sb = tuple(sorted({float(r["sigma_blur"]) for r in drows}))
        if observed_sb != expected_sb:
            raise ValueError(
                f"{decoder}: artifact grid.sigma_blur levels {observed_sb} do not match "
                f"configs/p0.yaml grid.sigma_blur {expected_sb}. Regenerate features with the "
                "current grid."
            )
        for sb in expected_sb:
            cell_w = tuple(sorted({int(r["w"]) for r in drows if float(r["sigma_blur"]) == sb}))
            if cell_w != expected_w:
                raise ValueError(
                    f"{decoder}/sb={sb:g}: artifact grid.w levels {cell_w} do not match "
                    f"configs/p0.yaml grid.w {expected_w}. Regenerate features with the current "
                    "grid."
                )


def _missing_calibrated_sigma_eff(
    rows_by_decoder: dict[str, list[Row]], sigma_ev: dict[str, SigmaEvidence]
) -> list[str]:
    """Configured artifacts may be calibrated but still miss generated blur levels."""
    missing: list[str] = []
    for decoder, drows in rows_by_decoder.items():
        ev = sigma_ev[decoder]
        if ev.source != "calibrated":
            continue
        for sb in sorted({float(r["sigma_blur"]) for r in drows}):
            if ev.sigma_eff(sb) is None:
                missing.append(f"{decoder}/sb={sb:g}")
    return missing


def _knee_ci_for_cell(cell_rows: list[Row], cfg: P0Config) -> KneeCI | None:
    """Robust knee + bootstrap CI from the TEST split of one (decoder, sigma_blur) cell.

    None if not computable (no H0 or <2 distinct w). The knee std becomes the regression's WLS
    precision weight, so a cell with an unstable knee cannot dominate the alignment fit.
    """
    test = split_rows(cell_rows, "test")
    scores: dict[int, np.ndarray] = {}
    for w in sorted({int(r["w"]) for r in test}):
        vals = [_row_finite_float(r, "rho_summary") for r in test if int(r["w"]) == w]
        if vals:
            scores[w] = np.asarray(vals, dtype=np.float64)
    if 1 not in scores or len(scores) < 2:
        return None
    return knee_bootstrap_ci(
        scores,
        h0_w=1,
        n_boot=cfg.gate.bootstrap_resamples,
        ci_level=cfg.gate.slope_ci_level,
        seed=cfg.split.seed,
    )


def _evidence_sections(
    sigma_ev: dict[str, SigmaEvidence],
    controls: NegativeControls | None,
    suff: dict[str, SufficiencyEvidence],
    oracle: dict[str, OracleEvidence],
    wrong: WrongPromptEvidence | None,
    v_levels: list[str],
    etex_levels: list[str],
) -> list[tuple[str, list[str]]]:
    sections: list[tuple[str, list[str]]] = []

    lines = []
    for name, ev in sigma_ev.items():
        flag = "scalar OK" if ev.scalar_ok else "EFFECTIVE-sigma only (scalarization downgraded)"
        lines.append(
            f"- {name}: sigma_hat={ev.sigma_hat:.3f} CI[{ev.ci[0]:.3f}, {ev.ci[1]:.3f}] "
            f"({ev.source}; {flag})"
        )
        for r in ev.reasons:
            lines.append(f"  - {r}")
    sections.append(("Sigma calibration", lines))

    if controls is not None:
        lines = [
            f"- rho AUC (test split): {controls.rho_auc:.3f} "
            f"(n_h0={controls.n_h0}, n_h1={controls.n_h1})"
        ]
        for name, value in sorted(controls.baseline_aucs.items()):
            lines.append(f"- {name}: {value:.3f}")
        weakest = controls.weakest_cell
        if weakest is not None:
            lines.append(
                f"- cell-level gate: {len(controls.cells)} comparable cells; weakest "
                f"{weakest.cell} margin rho-best={weakest.margin:+.3f} "
                f"(rho {weakest.rho_auc:.3f}, best nuisance {weakest.best_baseline_auc:.3f})"
            )
        lines.append(f"- CLIP baseline: {controls.clip_status}")
        if controls.clip_stratified_rho_auc is not None:
            lines.append(
                f"- CLIP-stratified rho AUC (diagnostic, report-only): min over "
                f"{controls.clip_n_strata} CLIP strata = {controls.clip_stratified_rho_auc:.3f} "
                "(rho separability with semantics ~matched; >0.5 => rho adds signal beyond CLIP)"
            )
        sections.append(("Negative controls (nuisance baselines)", lines))
    else:
        sections.append(
            ("Negative controls (nuisance baselines)", ["- not computable (test split too small)"])
        )

    if suff:
        lines = [
            f"- {name}: full-residual AUC {ev.full_residual_auc:.3f} vs rho AUC "
            f"{ev.rho_auc:.3f} -> gap {ev.auc_gap:+.3f} (train n={ev.n_train}, test n={ev.n_test})"
            for name, ev in suff.items()
        ]
    else:
        lines = ["- not computable (splits too small for the LDA probe)"]
    sections.append(("Rho sufficiency (per-patch energy probe, fit train / eval test)", lines))

    if oracle:
        lines = [
            f"- {name}: mean pearson r {ev.mean_pearson_r:.3f}, max |err| {ev.max_abs_err:.3f} "
            f"over {ev.n_checked} H1 uncompressed rows"
            for name, ev in oracle.items()
        ]
        sections.append(("Oracle-trace consistency (inversion-artifact exclusion)", lines))

    if wrong is not None:
        sections.append(
            (
                "Wrong-prompt null (report-only)",
                [
                    f"- true-prompt AUC {wrong.true_prompt_auc:.3f} vs wrong-prompt AUC "
                    f"{wrong.wrong_prompt_auc:.3f} (n={wrong.n})"
                ],
            )
        )

    sections.append(("Null-adapted v levels per cell", [f"- {lvl}" for lvl in v_levels]))
    sections.append(("Predictive E_tex levels per cell", [f"- {lvl}" for lvl in etex_levels]))
    return sections


def _gsigma_level_warnings(level_records: list[tuple[str, str]]) -> list[str]:
    """Memo warnings when the g_sigma null SCALE (H0 sample) falls back below cell level.

    The controlled sigma-sweep resolves the x-axis per cell via sigma_eff(sigma_blur) (the
    calibration sweep), so the only reduced resolution left is the H0 null sample that sets the
    g_sigma absolute scale. That scale
    is rescale-invariant for the gate (ordering only), so this is a transparency note, not blocking.
    """
    if not level_records:
        return []
    n = len(level_records)
    counts: dict[str, int] = {}
    for _cell, level in level_records:
        counts[level] = counts.get(level, 0) + 1
    warnings: list[str] = []
    fallback = n - counts.get("cell", 0)
    if fallback * 2 >= n:
        warnings.append(
            f"{fallback}/{n} usable cells used a below-cell-level g_sigma null scale (decoder or "
            "decoder+calibration H0 pool); the g_sigma absolute scale is coarser (rescale-"
            "invariant for the gate, which tests ordering)."
        )
    shared_level, shared_count = max(counts.items(), key=lambda kv: kv[1])
    if shared_level != "cell" and shared_count * 2 >= n:
        warnings.append(
            f"{shared_count}/{n} usable cells share g_sigma null level '{shared_level}'; sigma_eff "
            "still resolves the x-axis per cell, but more null_adapt H0 samples would sharpen the "
            "g_sigma absolute scale."
        )
    return warnings


def _knee_precision_weights(
    knee_stds: np.ndarray, *, ridge_floor: float = _KNEE_STD_RIDGE_FLOOR
) -> np.ndarray:
    """Finite WLS weights for knee stds, with a ridge so edge-censored std=0 cannot dominate."""
    std = np.asarray(knee_stds, dtype=np.float64).reshape(-1)
    if std.size == 0:
        raise ValueError("knee_stds must be non-empty")
    if not np.all(np.isfinite(std)):
        raise ValueError("knee_stds contains non-finite values")
    if np.any(std < 0.0):
        raise ValueError("knee_stds must be non-negative")
    if not (np.isfinite(ridge_floor) and ridge_floor > 0.0):
        raise ValueError(f"ridge_floor must be positive and finite, got {ridge_floor}")
    positive = std[std > 0.0]
    median_positive = float(np.median(positive)) if positive.size else ridge_floor
    s0 = max(median_positive, ridge_floor)
    weights = 1.0 / (std**2 + s0**2)
    if not np.all(np.isfinite(weights)):
        raise ValueError("knee precision weights are non-finite")
    return weights


def _censored_cells_note(censored_cells: list[str], max_w: float) -> str | None:
    if not censored_cells:
        return None
    return (
        f"{len(censored_cells)} cell(s) had w* > max(grid.w)={max_w:g} and were EXCLUDED from "
        "the knee-vs-w* regression as right-censored (the knee cannot appear past the "
        f"w-grid ceiling): {censored_cells}. If many high-sigma cells are censored, extend "
        "grid.w to cover the analytic w* range (it is the deniability threshold, not a bug)."
    )


def _write_insufficient_cells_memo(
    out_dir: Path,
    *,
    usable_cells: int,
    sigma_ev: dict[str, SigmaEvidence],
    missing_sigma_eff: list[str],
    downgraded_sigma: list[str],
    v_levels: list[str],
    etex_levels: list[str],
    cell_records: list[dict[str, object]],
    censored_cells: list[str],
    max_w: float,
    reason_override: str | None = None,
) -> None:
    """Persist an explicit INCONCLUSIVE verdict when real rows exist but regression cannot run."""
    reason = reason_override or (
        f"only {usable_cells} usable cells; need >= 3 for a knee-vs-w* regression"
    )
    notes = [
        "Real main-grid rows were present, but the current artifacts do not support the "
        "preregistered "
        "alignment regression. This run writes an explicit INCONCLUSIVE memo so stale decision "
        "artifacts from earlier runs cannot be mistaken for the current gate verdict.",
    ]
    missing_sigma = [name for name, ev in sigma_ev.items() if ev.source != "calibrated"]
    if missing_sigma:
        notes.append(
            "Sigma calibration artifact missing for "
            f"{missing_sigma}; nominal fallback is diagnostic only and cannot support a gate "
            "decision. Run `python -m cfgid.channel.sigma_calib` for each core decoder first."
        )
    if missing_sigma_eff:
        notes.append(
            "Sigma calibration artifacts are incomplete for generated blur cells "
            f"{missing_sigma_eff}; those cells are diagnostic-only skipped and cannot support a "
            "gate decision. Re-run `python -m cfgid.channel.sigma_calib` with the full "
            "grid.sigma_blur sweep."
        )
    if downgraded_sigma:
        notes.append(
            "Sigma calibration scalarization was downgraded (effective-sigma only) for "
            f"{downgraded_sigma}; the gate uses the calibrated effective-sigma sweep "
            "(rescale-invariant: it tests knee-vs-w* direction/monotonicity), but any PASS/FAIL is "
            "CAVEATED -- do not claim a true Gaussian scalar PSF or an absolute calibrated "
            "sigma/w* magnitude from these artifacts (inspect outputs/sigma_calib/<decoder>.json)."
        )
    censor_note = _censored_cells_note(censored_cells, max_w)
    if censor_note is not None:
        notes.append(censor_note)
    decision = GateResult("inconclusive", [reason])
    regression = RegressionResult(
        slope=0.0,
        intercept=0.0,
        corr=0.0,
        slope_ci_lo=0.0,
        slope_ci_hi=0.0,
        n=usable_cells,
    )
    provenance = {
        "v": "unavailable (insufficient usable cells for gate regression)",
        "g_sigma": "unavailable (insufficient usable cells for gate regression)",
        "sigma": ", ".join(f"{d}:{e.source}" for d, e in sigma_ev.items()),
    }
    sections = _evidence_sections(
        sigma_ev,
        controls=None,
        suff={},
        oracle={},
        wrong=None,
        v_levels=v_levels,
        etex_levels=etex_levels,
    )
    memo = build_decision_memo(
        decision=decision,
        regression=regression,
        provenance=provenance,
        notes=notes,
        sections=sections,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "decision_memo.md").write_text(memo, encoding="utf-8")
    (out_dir / "alignment_cells.json").write_text(
        json.dumps(cell_records, indent=2, allow_nan=False), encoding="utf-8"
    )


def analyze(features_root: Path, out_dir: Path, *, cfg: P0Config | None = None) -> int:
    rows = _load_tables(features_root)
    if not rows:
        print(
            f"[skip] no main-grid features under {features_root}; run `cfgid.sweep.run_p0` first."
        )
        return 0
    cfg = cfg if cfg is not None else load_p0_config()

    # Invariant: only true_cfg_core rows may enter the sigma-w* core regression. Rows from a
    # guidance-distilled non-core decoder (FLUX) are excluded loudly, never silently pooled.
    model_branch_modes = {name: spec.branch_mode for name, spec in load_models_config().items()}
    non_core = [r for r in rows if _effective_branch_mode(r, model_branch_modes) != "true_cfg_core"]
    if non_core:
        dropped = sorted({str(r["decoder"]) for r in non_core})
        print(
            f"[warn] excluded {len(non_core)} non-true_cfg_core rows from decoders {dropped} "
            "(non-core branch mode; never pooled into the core regression)"
        )
        rows = [r for r in rows if _effective_branch_mode(r, model_branch_modes) == "true_cfg_core"]
        if not rows:
            print("[skip] no true_cfg_core rows left to analyze.")
            return 0
    rows = _restrict_to_grid_decoders(rows, cfg, known_decoders=set(model_branch_modes))
    if not rows:
        print("[skip] no grid.decoders rows left to analyze.")
        return 0
    validate_table_rows(rows)
    _require_qf_collapsed_to_none(rows, cfg)
    _require_rows_match_config_grid(rows, cfg)

    decoders = sorted({str(r["decoder"]) for r in rows})
    rows_by_decoder = {d: [r for r in rows if r["decoder"] == d] for d in decoders}
    calib_root = outputs_dir() / "sigma_calib"
    sigma_ev = {d: load_sigma_evidence(d, calib_root) for d in decoders}
    missing_sigma_eff = _missing_calibrated_sigma_eff(rows_by_decoder, sigma_ev)
    downgraded_sigma = [
        name for name, ev in sigma_ev.items() if ev.source == "calibrated" and not ev.scalar_ok
    ]

    # --- per-cell knees + w* over the controlled sigma-sweep (predictive track) ---
    # Cells are (decoder, sigma_blur). Scalar w* uses measured E_tex from the compact store
    # directly; that E_tex already carries decoder/blur attenuation. g_sigma is still estimated from
    # sigma_eff and gates the predictive track, but it must not re-enter the measured-E_tex bound.
    knees: list[float] = []
    knee_stds: list[float] = []
    wstars: list[float] = []
    knee_by_decoder_sb: dict[str, dict[float, float]] = {}
    censored_cells: list[str] = []  # w* > max(grid.w): the empirical knee is right-censored
    max_w = float(max(cfg.grid.w))
    v_levels: list[str] = []
    etex_levels: list[str] = []
    v_records: list[tuple[str, str]] = []
    cell_records: list[dict[str, object]] = []  # per-cell diagnostic table (alignment_cells.json)
    predictive = True
    for decoder in decoders:
        drows = rows_by_decoder[decoder]
        decoder_dir = features_root / decoder
        for sb in sorted({float(r["sigma_blur"]) for r in drows}):
            cell = f"{decoder}/sb={sb:g}"
            cell_rows = [r for r in drows if float(r["sigma_blur"]) == sb]
            # y-axis: isotonic-smoothed TEST-split AUC(w) knee with a bootstrap CI (seed = split
            # seed, gate.bootstrap_resamples replicates).
            knee_ci = _knee_ci_for_cell(cell_rows, cfg)
            if knee_ci is None:
                continue
            # v (empirical H0 variance) and g_sigma from H0 rows of the NULL split (widening to
            # decoder level, then NULL+CAL, when the cell is too small); never H1 or TEST rows.
            params = null_params_for_cell(cell_rows, drows, sigma_ev[decoder], sigma_blur=sb)
            if params.g_sigma is None or not np.isfinite(params.v_hat) or params.v_hat <= 0:
                v_levels.append(f"{cell}: unavailable -> cell skipped")
                continue
            try:
                etex = predictive_etex_for_cell(cell_rows, drows, decoder_dir, cfg)
            except (FileNotFoundError, EvidenceUnavailableError) as exc:
                etex_levels.append(f"{cell}: unavailable -> cell skipped ({exc})")
                continue
            sigma_eff = sigma_ev[decoder].sigma_eff(sb)
            v_levels.append(
                f"{cell}: sigma_eff={sigma_eff:.3f} g_sigma={params.g_sigma.value:.4g} "
                f"v={params.v_hat:.6g} ({params.v_level})"
            )
            etex_levels.append(f"{cell}: E_tex={etex.value:.6g} ({etex.level}, H0 n={etex.n})")
            v_records.append((cell, params.gsigma_level))
            predictive = predictive and params.g_sigma.is_predictive
            # etex.value is the MEASURED post-decode/post-blur latent energy (compact_e_tex.npy),
            # which ALREADY carries the decoder/blur PSF attenuation g_sigma models -- so the signal
            # slope IS etex.value; multiplying g_sigma again double-counts the PSF. g_sigma is still
            # estimated, gates the predictive track above, and is logged below; it just must not
            # re-enter the bound value (Section III-D: no g_sigma multiplier on measured E_tex).
            # w* = 1 + c(delta, N_eff) * sqrt(v) / E_tex_meas with N_eff = 1: the closed form under
            # the linear sensitivity model of Section III-E (Eq. (15) and Eq. (38)).
            w_star = wstar_from_measured_etex(
                e_tex_meas=etex.value, v=params.v_hat, delta=cfg.bounds.delta, n_eff=1.0
            )
            knee = knee_ci.knee
            # The analytic deniability threshold may exceed the w-grid ceiling: then the empirical
            # knee is RIGHT-CENSORED (a knee at w*>max(w) cannot appear within the grid), so the
            # cell is uninformative for the knee-vs-w* alignment and must NOT enter the fit (else it
            # is pinned at the ceiling, biasing the slope). It is still recorded + flagged.
            cell_censored = bool(w_star > max_w)  # bool(): json-serializable, not numpy bool_
            if cell_censored:
                censored_cells.append(f"{cell}: w*={w_star:.3g} > max(w)={max_w:g}")
            else:
                knees.append(knee)
                knee_stds.append(knee_ci.std)
                wstars.append(w_star)
                knee_by_decoder_sb.setdefault(decoder, {})[sb] = knee
            # Covariance-channel diagnostic (S_eff / w*_cov); None when not estimable (report-only).
            cov_ch = cov_channel_for_cell(drows, decoder_dir, sigma_blur=sb)
            s_eff = float(cov_ch.s_eff) if cov_ch is not None else None
            w_star_cov = (
                float(wstar_cov(cov_ch, delta=cfg.bounds.delta, n_eff=1.0))
                if cov_ch is not None
                else None
            )
            # Calibrated w*: the analytic plugin/measured-E_tex slope's ABSOLUTE scale is
            # unidentified (gsigma plugin docstring), which collapses w* to ~1 (the model slope
            # ~E_tex_meas is orders of magnitude above the measured drho/dw). synthetic_fit fits the
            # signal slope on the SELF-GENERATED sweep (w known by construction -> predictive, not
            # an H1 label on a disputed image), on the TRAIN split (disjoint from the test knee).
            # Then
            # |slope| = g_fit * E_tex = measured drho/dw, so w*_cal = 1 + c*sqrt(v)/|slope| -- the
            # non-trivial, empirically-scaled deniability threshold. Report-only (not a gate input).
            w_star_calibrated: float | None = None
            train_rows = split_rows(cell_rows, "train")
            ws_cal = sorted({int(r["w"]) for r in train_rows})
            if len(ws_cal) >= 3:
                rho_means_cal = [
                    float(
                        np.mean(
                            [
                                _row_finite_float(r, "rho_summary")
                                for r in train_rows
                                if int(r["w"]) == w
                            ]
                        )
                    )
                    for w in ws_cal
                ]
                try:
                    g_fit = synthetic_fit_gsigma(
                        np.asarray(ws_cal, dtype=np.float64),
                        np.asarray(rho_means_cal, dtype=np.float64),
                        etex.value,
                        labeled=False,
                    )
                    wp_cal = WStarParams(
                        g_sigma=g_fit.value,
                        v=params.v_hat,
                        g_sigma_provenance=g_fit.provenance,
                        v_provenance=params.v_provenance,
                    )
                    w_star_calibrated = float(
                        wstar(e_tex=etex.value, params=wp_cal, delta=cfg.bounds.delta, n_eff=1.0)
                    )
                except ValueError:
                    w_star_calibrated = None
            cell_records.append(
                {
                    "decoder": decoder,
                    "sigma_blur": float(sb),
                    "sigma_eff": float(sigma_eff) if sigma_eff is not None else None,
                    "knee": float(knee),
                    "knee_ci_lo": float(knee_ci.ci_lo),
                    "knee_ci_hi": float(knee_ci.ci_hi),
                    "knee_std": float(knee_ci.std),
                    "wstar": float(w_star),
                    "wstar_calibrated": w_star_calibrated,
                    "wstar_cov": w_star_cov,
                    "s_eff": s_eff,
                    "e_tex": float(etex.value),
                    "v_hat": float(params.v_hat),
                    "v_level": params.v_level,
                    "v_provenance": params.v_provenance,
                    "g_sigma": float(params.g_sigma.value),
                    "g_sigma_provenance": params.g_sigma.provenance,
                    "censored": cell_censored,
                }
            )

    if len(knees) < 3:
        _write_insufficient_cells_memo(
            out_dir,
            usable_cells=len(knees),
            sigma_ev=sigma_ev,
            missing_sigma_eff=missing_sigma_eff,
            downgraded_sigma=downgraded_sigma,
            v_levels=v_levels,
            etex_levels=etex_levels,
            cell_records=cell_records,
            censored_cells=censored_cells,
            max_w=max_w,
        )
        print(
            f"[ok] decision=inconclusive; only {len(knees)} usable cells; "
            f"memo -> {out_dir / 'decision_memo.md'}; figure skipped"
        )
        return 0

    # WLS precision weights from the bootstrap knee std, ridge-floored at the median std so neither
    # an unstable cell (large std -> small weight) nor an ultra-precise / edge-censored knee
    # (std~0) dominates the fit. None of this touches the preregistered slope/corr thresholds.
    weights = _knee_precision_weights(np.asarray(knee_stds, dtype=np.float64))
    try:
        reg = fit_alignment(
            np.asarray(knees),
            np.asarray(wstars),
            n_boot=cfg.gate.bootstrap_resamples,
            ci_level=cfg.gate.slope_ci_level,
            seed=cfg.split.seed,
            weights=weights,
        )
    except ValueError as exc:
        if "w* values are constant" not in str(exc):
            raise
        # All non-censored w* are identical (zero weighted x-variance) -> the WLS regression is
        # undefined and fit_alignment raises. Degrade to an explicit INCONCLUSIVE memo like every
        # other unfittable case in this function (the <3-usable / missing-sigma paths), instead of
        # crashing with an uncaught traceback that would also leave a stale decision_memo in place.
        _write_insufficient_cells_memo(
            out_dir,
            usable_cells=len(knees),
            sigma_ev=sigma_ev,
            missing_sigma_eff=missing_sigma_eff,
            downgraded_sigma=downgraded_sigma,
            v_levels=v_levels,
            etex_levels=etex_levels,
            cell_records=cell_records,
            censored_cells=censored_cells,
            max_w=max_w,
            reason_override=f"alignment regression undefined ({len(knees)} usable cells): {exc}",
        )
        print(
            f"[ok] decision=inconclusive; alignment regression undefined ({exc}); "
            f"memo -> {out_dir / 'decision_memo.md'}; figure skipped"
        )
        return 0
    # Sigma direction probe on COMMON sigma_blur cells: each decoder's mean knee averages the same
    # blur composition, so the cross-decoder knee ordering can be compared against the cross-decoder
    # effective-sigma (at sigma_blur=0) ordering apples-to-apples.
    common_sbs: set[float] = (
        set.intersection(*(set(m) for m in knee_by_decoder_sb.values()))
        if len(knee_by_decoder_sb) >= 2
        else set()
    )
    sigma_eff0 = {d: sigma_ev[d].sigma_eff(0.0) for d in knee_by_decoder_sb}
    if common_sbs and all(v is not None for v in sigma_eff0.values()):
        mean_knee = {
            d: float(np.mean([m[s] for s in sorted(common_sbs)]))
            for d, m in knee_by_decoder_sb.items()
        }
        sigma_for_direction = {d: float(sigma_eff0[d]) for d in mean_knee}  # type: ignore[arg-type]
        sigma_ok = sigma_direction_consistent(mean_knee, sigma_for_direction)
        sigma_note = f"common sigma_blur cells {sorted(common_sbs)} vs sigma_eff(0)"
    else:
        sigma_ok = False  # <2 decoders / no shared blur / missing sigma_eff -> not comparable
        sigma_note = "no common sigma_blur cells (or missing sigma_eff) -> direction not comparable"

    # --- real negative controls / sufficiency / oracle / wrong-prompt ---
    clip_dir = checkpoints_dir() / cfg.evidence.clip_local_dir
    controls = negative_controls(split_rows(rows, "test"), features_root, cfg, clip_dir=clip_dir)
    suff: dict[str, SufficiencyEvidence] = {}
    oracle: dict[str, OracleEvidence] = {}
    for decoder in decoders:
        decoder_dir = features_root / decoder
        try:
            s = sufficiency_evidence(rows_by_decoder[decoder], decoder_dir)
            o = oracle_evidence(rows_by_decoder[decoder], decoder_dir, cfg)
        except FileNotFoundError as exc:
            print(f"[warn] {decoder}: store-backed evidence unavailable ({exc})")
            s, o = None, None
        if s is not None:
            suff[decoder] = s
        if o is not None:
            oracle[decoder] = o
    wrong = wrong_prompt_evidence(rows)

    # FAIL-SAFE: the preregistered baseline SET includes CLIP-text-alignment when enabled. If it
    # is enabled but the model is unavailable, the strongest known confound is missing from the
    # comparison — an incomplete baseline set may never support a pass.
    clip_complete = controls is None or controls.clip_status in ("ok", "disabled")
    rho_beats = (
        controls is not None
        and clip_complete
        and controls.rho_beats_nuisance(cfg.gate.negative_control_margin)
    )
    # Sufficiency must be computable for EVERY decoder in the regression; a decoder whose
    # evidence is missing cannot count as "sufficient" (fail-safe, matches the memo note).
    rho_sufficient = set(suff) == set(decoders) and all(
        ev.rho_sufficient(cfg.gate.rho_sufficiency_max_auc_gap) for ev in suff.values()
    )

    inputs = GateInputs(
        slope=reg.slope,
        slope_ci_lo=reg.slope_ci_lo,
        slope_ci_hi=reg.slope_ci_hi,
        corr=reg.corr,
        predictive_track=predictive,
        rho_beats_nuisance=rho_beats,
        rho_sufficient=rho_sufficient,
        sigma_direction_ok=sigma_ok,
        sigma_evidence_real=all(ev.source == "calibrated" for ev in sigma_ev.values())
        and not missing_sigma_eff,
    )
    decision = evaluate_gate(inputs, cfg.gate)

    provenance = {
        "v": "unlabeled_null_adapted (empirical H0 null variance per (decoder, sigma_blur) cell)",
        "g_sigma": "unlabeled_null_adapted (plugin g(sigma_eff) at the calibrated effective "
        "sigma per blur level; predictive-track gate/logging only for scalar measured-E_tex w*)",
        "sigma": ", ".join(f"{d}:{e.source}" for d, e in sigma_ev.items()),
    }
    notes = [
        "All gate inputs are computed from persisted artifacts under grouped splits "
        "(fit: null_adapt; eval: test); no placeholder values.",
        "Controlled sigma-sweep: cells are (decoder, sigma_blur); the scalar x-axis w* uses the "
        "measured H0 E_tex from the post-decode/post-blur inversion-source latent. That measured "
        "E_tex already carries decoder/blur attenuation, so plugin g_sigma is logged and gates "
        "provenance but is not multiplied into the scalar bound. QF collapsed to none (a "
        "preliminary QF sweep showed the knee ignores QF). y-axis: robust isotonic knee + "
        "bootstrap CI; alignment WLS-weighted by knee precision. Gate thresholds are unchanged "
        "from the preregistration (configs/p0.yaml gate).",
        "Both the reported slope AND correlation are precision-weighted (consistent with the WLS "
        "fit); the frozen |corr|>=0.30 criterion is applied to the knee-precision-weighted "
        "correlation (not the unweighted Pearson r of the original preregistration; threshold "
        "value unchanged).",
        "w*_cov / S_eff (covariance channel, n_eff=1) are reported per cell as a DIAGNOSTIC; the "
        "gate regression uses the scalar w* (covariance value is bound correctness, not lever).",
        "negative-controls / sufficiency not computable => the corresponding gate input is "
        "False (fail-safe), surfacing as inconclusive rather than pass.",
        f"sigma direction probe: {sigma_note}.",
    ]
    if controls is not None and not clip_complete:
        notes.append(
            "CLIP baseline is ENABLED but unavailable -> negative controls are an incomplete "
            "baseline set and cannot support a pass (download CLIP per README.md)."
        )
    missing_sigma = [name for name, ev in sigma_ev.items() if ev.source != "calibrated"]
    if missing_sigma:
        notes.append(
            "Sigma calibration artifact missing for "
            f"{missing_sigma}; nominal fallback is diagnostic only and cannot support a gate "
            "decision. Run `python -m cfgid.channel.sigma_calib` for each core decoder first."
        )
    if missing_sigma_eff:
        notes.append(
            "Sigma calibration artifacts are incomplete for generated blur cells "
            f"{missing_sigma_eff}; those cells are diagnostic-only skipped and cannot support a "
            "gate decision. Re-run `python -m cfgid.channel.sigma_calib` with the full "
            "grid.sigma_blur sweep."
        )
    if downgraded_sigma:
        notes.append(
            "Sigma calibration scalarization was downgraded (effective-sigma only) for "
            f"{downgraded_sigma}; the gate uses the calibrated effective-sigma sweep "
            "(rescale-invariant: it tests knee-vs-w* direction/monotonicity), but any PASS/FAIL is "
            "CAVEATED -- do not claim a true Gaussian scalar PSF or an absolute calibrated "
            "sigma/w* magnitude from these artifacts (inspect outputs/sigma_calib/<decoder>.json)."
        )
    censor_note = _censored_cells_note(censored_cells, max_w)
    if censor_note is not None:
        notes.append(censor_note)
    notes.extend(_gsigma_level_warnings(v_records))
    sections = _evidence_sections(sigma_ev, controls, suff, oracle, wrong, v_levels, etex_levels)
    cov_lines: list[str] = []
    for rec in cell_records:
        cov = format(rec["wstar_cov"], ".4f") if isinstance(rec["wstar_cov"], float) else "n/a"
        seff = format(rec["s_eff"], ".4g") if isinstance(rec["s_eff"], float) else "n/a"
        cov_lines.append(
            f"- {rec['decoder']}/sb={rec['sigma_blur']:g}: knee={rec['knee']:.3f} "  # type: ignore[call-overload]
            f"CI[{rec['knee_ci_lo']:.3f}, {rec['knee_ci_hi']:.3f}] std={rec['knee_std']:.3f}; "
            f"w*_cov={cov} (S_eff={seff})"
        )
    sections.append(("Covariance channel + knee robustness (diagnostic, report-only)", cov_lines))
    memo = build_decision_memo(
        decision=decision,
        regression=reg,
        provenance=provenance,
        notes=notes,
        sections=sections,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "decision_memo.md").write_text(memo, encoding="utf-8")
    # Per-cell diagnostic table: the knee / w* points behind the scatter, labelled by (decoder,
    # sigma_blur) with sigma_eff and the inputs (E_tex, empirical v, g_sigma), the bootstrap knee
    # std/CI, and the covariance-channel diagnostic (S_eff, w*_cov). allow_nan=False: None becomes
    # JSON null, but a stray NaN must never serialize silently (no silent NaN).
    (out_dir / "alignment_cells.json").write_text(
        json.dumps(cell_records, indent=2, allow_nan=False), encoding="utf-8"
    )

    def _fmt(value: object, spec: str) -> str:
        return format(value, spec) if isinstance(value, (int, float)) else "    n/a"

    print(
        "[cells] decoder    sb  sig_eff |   knee knee_sd     w*  w*_cov  s_eff    E_tex   g_sigma"
    )
    for rec in cell_records:
        print(
            f"[cells] {str(rec['decoder']):>6} {_fmt(rec['sigma_blur'], '5.1f')} "  # type: ignore[call-overload]
            f"{_fmt(rec['sigma_eff'], '7.2f')} | "
            f"{_fmt(rec['knee'], '6.3f')} {_fmt(rec['knee_std'], '6.3f')} "
            f"{_fmt(rec['wstar'], '6.3f')} {_fmt(rec['wstar_cov'], '6.3f')} "
            f"{_fmt(rec['s_eff'], '6.3f')} {_fmt(rec['e_tex'], '8.4f')} "
            f"{_fmt(rec['g_sigma'], '9.4g')}"
        )
    drew = save_alignment_scatter(
        np.asarray(wstars), np.asarray(knees), str(out_dir / "alignment_scatter.png")
    )
    fig_state = "written" if drew else "skipped (matplotlib not installed)"
    print(
        f"[ok] decision={decision.decision}; memo -> {out_dir / 'decision_memo.md'}; "
        f"figure {fig_state}"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="cfgid.sweep.analyze",
        description="Main-grid knee-vs-w* alignment -> preregistered gate -> decision memo",
    )
    parser.add_argument("--features", default=None)
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)
    root = Path(args.features) if args.features else outputs_dir() / "p0"
    out = Path(args.out) if args.out else outputs_dir() / "p0"
    return analyze(root, out)


if __name__ == "__main__":
    sys.exit(main())
