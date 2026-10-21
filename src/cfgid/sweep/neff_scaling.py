"""Per-cell N_eff (sqrt(K)) aggregation test and the shared main-grid row loaders / cell groupers.

The deniability bound's finite-sample content is the deflection law (Eq. (14)): for a FIXED
(decoder, sigma_blur) cell at a FIXED guidance w, a detector that aggregates ``N_eff = K``
independent images has deflection ``d(K) = sqrt(K) * d(1)``, so the group-K separability is

    AUC_pred(K, w) = Phi( sqrt(K) * d(1, w) / sqrt(2) ),     d(1, w) = sqrt(2) * Phi^{-1}(AUC_1(w)),

equivalently the deniability threshold scales as ``w*(K) - 1 ∝ 1/sqrt(K)``. On the measured-Etex
analysis path, ``w* - 1 = c(delta, N_eff) sqrt(v)/E_tex_meas`` and ``c ∝ 1/sqrt(N_eff)``.

This is the ONE knob that spreads the analytic threshold WITHOUT collapsing the per-image channel
(the measured signal slope stays fixed; we only vary how many independent images enter the
decision), so it avoids the structural trap where reliable knees and spread w* never co-occur.

NON-CIRCULARITY (the whole point): the single-image anchor ``d(1, w)`` is measured on the FIT split
(``train``); the group-K AUC is measured on the DISJOINT ``test`` split; the ``sqrt(K)`` factor is
the THEOREM'S prediction, never fitted. A plain "knee rises with N_eff" regression would be a
tautology -- this instead tests (a) the absolute predicted-vs-empirical AUC match across K and
(b) the pre-registered exponent ``d_emp(K) ∝ K^{+1/2}`` (log-log slope 0.5).

Pure numpy/scipy, consumes the persisted ``p0_table.json`` (per-image ``rho_summary`` + label +
split). ZERO re-inversion. Cells are selected as "alive" on the fit split (single-image AUC above a
floor) so the held-out test split remains reserved for measuring group-K AUC. The pooled,
better-powered version of this test is ``cfgid.sweep.pooled_neff_scaling``.

Role in the pipeline: besides the per-cell test, this module is the single closure point for
reading main-grid tables (``_load_p0_rows``) and grouping them into (decoder, sigma_blur) cells
with FIT (``train``) / TEST (``test``) score lists (``_group_rows_by_cell``,
``filter_core_none_qf_rows``, the per-condition groupers used by the post-processing analysis);
the Gaussian AUC/deflection maps (``deflection_from_auc``, ``predicted_group_auc``) and the
mean-of-K group AUC (``empirical_group_auc``) are reused by the pooled test and by the AUC_K
analysis scripts (e.g. scripts/analysis/weak_contrast_rows.py).

Produces: outputs/p0/neff_scaling.json (per-cell diagnostic; no printed paper number is read from
it).
Inputs: the outputs/p0/<decoder>/p0_table.json files only; CPU.
Outputs: outputs/p0/neff_scaling.json.
Usage: uv run --no-sync python -m cfgid.sweep.neff_scaling [--features-root outputs/p0]
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from operator import index
from pathlib import Path

import numpy as np
from scipy.stats import norm

from cfgid.detector.empirical import auc
from cfgid.numerics import loads_strict_json

# AUC is clamped before the probit so a saturated cell (AUC -> 0 or 1) gives a finite deflection.
_AUC_CLAMP = 1.0e-4
DEFAULT_TARGET_AUC1 = 0.68


def _positive_index(name: str, value: int) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    try:
        out = index(value)
    except TypeError as exc:
        raise ValueError(f"{name} must be a positive integer, got {value!r}") from exc
    if out < 1:
        raise ValueError(f"{name} must be >= 1, got {out}")
    return out


def _row_finite_float(row: dict, key: str) -> float:
    """Validated numeric p0_table field; NaN/Inf in artifacts must fail loudly."""
    try:
        raw = row[key]
    except KeyError as exc:
        raise ValueError(f"row field '{key}' must be finite numeric") from exc
    if isinstance(raw, (bool, np.bool_)):
        raise ValueError(f"row field '{key}' must be finite numeric, got boolean {raw!r}")
    try:
        value = float(raw)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"row field '{key}' must be finite numeric") from exc
    if not np.isfinite(value):
        raise ValueError(f"row field '{key}' contains non-finite value {value}")
    return value


def _row_finite_int(row: dict, key: str) -> int:
    """Validated integer-valued p0_table field; rejects NaN/Inf and lossy coercions."""
    value = _row_finite_float(row, key)
    if not value.is_integer():
        raise ValueError(f"row field '{key}' must be integer-valued, got {value}")
    return int(value)


def _qf_sentinel(row: dict) -> int:
    """QF sentinel for a p0_table row: missing/None -> -1 (none), else the int QF; loud on a bad
    value (run_p0 writes ``qf if not None else -1``; -1 is the none sentinel)."""
    q = row.get("qf", -1)
    if q is None:
        return -1
    if isinstance(q, (bool, np.bool_)):
        raise ValueError(f"row 'qf' must be an int sentinel or None, got bool {q!r}")
    if not isinstance(q, (int, float, np.integer, np.floating)):
        raise ValueError(f"row 'qf' must be an int sentinel or None, got {q!r}")
    value = float(q)
    if not np.isfinite(value) or not value.is_integer():
        raise ValueError(f"row 'qf' must be an int sentinel or None, got {q!r}")
    try:
        return int(value)
    except (OverflowError, ValueError) as exc:
        raise ValueError(f"row 'qf' must be an int sentinel or None, got {q!r}") from exc


def filter_core_none_qf_rows(rows: list[dict]) -> list[dict]:
    """Keep only true_cfg_core rows (drop explicitly-stamped non-core, e.g. non-core FLUX) and FAIL
    LOUD on any compressed-QF sentinel (cells are (decoder, sigma_blur), QF collapsed to none).
    The single closure point every (decoder, sigma_blur) cell grouper must route through so the
    favorable-subset alive-set, the robustness diagnostics, and the descriptive analyses pool the
    SAME images.

    branch_mode contract: MISSING or explicit-``None`` branch_mode -> core (old unstamped SD
    artifacts are kept); only an explicitly-stamped non-core string is dropped. Sound because run_p0
    always stamps a non-None branch_mode, so no unstamped non-core (FLUX) table exists; for such a
    hypothetical table this intentionally differs from sweep/analyze._effective_branch_mode's
    models.yaml fallback (unreachable with run_p0 tables)."""
    core: list[dict] = []
    for row in rows:
        bm = row.get("branch_mode")
        if bm is None or str(bm) == "true_cfg_core":
            core.append(row)
            continue
        if str(bm) == "flux_specific_appendix":
            continue
        raise ValueError(
            "row 'branch_mode' must be missing/None, 'true_cfg_core', or "
            f"'flux_specific_appendix'; got {bm!r}"
        )
    extra_qf = sorted({_qf_sentinel(r) for r in core} - {-1})
    if extra_qf:
        raise ValueError(
            "neff_scaling cells are (decoder, sigma_blur) with QF collapsed to none; got "
            f"compressed-QF sentinels {extra_qf} -- rebuild the table with grid.qf=[none]"
        )
    return core


_PROMPT_INDEX_RE = re.compile(r"(?:^|_)p(\d+)_s")


def _row_prompt_index(row: dict) -> int:
    """Prompt index from a p0_table row; prefer the structured field, fallback to image_id.

    Current run_p0 artifacts carry a numeric ``prompt`` field. Legacy rows can still be subsampled
    from the stamped image_id, but the regex is anchored to the run_p0 ``_p<idx>_s`` segment so an
    unrelated decoder/token containing ``_p123_`` cannot masquerade as the prompt index.
    """
    if "prompt" in row and row.get("prompt") is not None:
        prompt = _row_finite_int(row, "prompt")
        if prompt < 0:
            raise ValueError(f"row field 'prompt' must be non-negative, got {prompt}")
        return prompt
    m = _PROMPT_INDEX_RE.search(str(row.get("image_id", "")))
    if m is None:
        raise ValueError(
            f"row image_id {row.get('image_id')!r} has no _p<idx>_ prompt index (cannot subsample)"
        )
    return int(m.group(1))


def filter_to_first_n_prompts(rows: list[dict], n: int | None) -> list[dict]:
    """Keep rows whose prompt index (from image_id) is < ``n``; ``n`` None/<=0 disables (all rows).

    run_p0's placeholder prompts are deterministic by index, so the first-``n`` subset is NESTED
    (first 128 prompts subset first 256 subset first 512). Subsampling a single large inversion to
    {128, 256, 512} therefore yields an apples-to-apples prompt-sample-size convergence curve (and
    keeps cross-decoder comparisons at a common N) WITHOUT re-inverting at each N.
    """
    if n is None or n <= 0:
        return rows
    return [r for r in rows if _row_prompt_index(r) < n]


def deflection_from_auc(auc_value: float) -> float:
    """Equal-variance-Gaussian deflection d = sqrt(2) * Phi^{-1}(AUC). AUC clamped off {0,1}."""
    if not (np.isfinite(auc_value) and 0.0 <= auc_value <= 1.0):
        raise ValueError(f"auc_value must be finite and in [0, 1], got {auc_value}")
    a = float(np.clip(auc_value, _AUC_CLAMP, 1.0 - _AUC_CLAMP))
    return float(np.sqrt(2.0) * norm.ppf(a))


def predicted_group_auc(d1: float, k: int) -> float:
    """AUC of the K-aggregated detector under the deflection law: Phi( sqrt(K) * d1 / sqrt(2) )."""
    if not np.isfinite(d1):
        raise ValueError(f"d1 must be finite, got {d1}")
    k = _positive_index("k", k)
    return float(norm.cdf(np.sqrt(float(k)) * d1 / np.sqrt(2.0)))


def empirical_group_auc(
    h0: np.ndarray, h1: np.ndarray, k: int, *, n_groups: int, rng: np.random.Generator
) -> float:
    """Empirical AUC of the mean-of-K detector: each 'sample' is the mean of K independent images.

    Groups are drawn with replacement (bootstrap) since per-cell image counts are small; this
    estimates the K-aggregated separability, not a finite-population statistic.
    """
    k = _positive_index("k", k)
    n_groups = _positive_index("n_groups", n_groups)
    h0 = np.asarray(h0, dtype=np.float64).reshape(-1)
    h1 = np.asarray(h1, dtype=np.float64).reshape(-1)
    if h0.size == 0 or h1.size == 0:
        raise ValueError("empirical_group_auc needs non-empty H0 and H1 scores")
    g0 = h0[rng.integers(0, h0.size, size=(n_groups, k))].mean(axis=1)
    g1 = h1[rng.integers(0, h1.size, size=(n_groups, k))].mean(axis=1)
    scores = np.concatenate([g1, g0])
    labels = np.concatenate([np.ones(n_groups, dtype=int), np.zeros(n_groups, dtype=int)])
    return auc(scores, labels)


@dataclass(frozen=True)
class NeffScalingResult:
    """Per-(cell, w) N_eff scaling test result."""

    cell: str  # "<decoder>/sb=<sigma_blur>"
    w: int
    d1_fit: float  # single-image deflection on the FIT split (the anchor)
    auc1_test: float  # single-image AUC on the TEST split (sanity)
    k_grid: tuple[int, ...]
    auc_pred: tuple[float, ...]  # predicted AUC(K) from d1_fit (sqrt(K) law)
    auc_emp: tuple[float, ...]  # empirical AUC(K) on the TEST split
    exponent: float | None  # slope of log(d_emp) vs log(K); theorem predicts +0.5
    exponent_ok: bool  # |exponent - 0.5| <= exponent_tol (pre-registered)
    mean_abs_auc_err: float  # mean |auc_emp - auc_pred| over the K grid


def neff_scaling_for_cell(
    *,
    cell: str,
    w: int,
    h0_fit: np.ndarray,
    h1_fit: np.ndarray,
    h0_test: np.ndarray,
    h1_test: np.ndarray,
    k_grid: tuple[int, ...] = (1, 2, 4, 8, 16),
    n_groups: int = 4000,
    exponent_tol: float = 0.15,
    seed: int = 0,
) -> NeffScalingResult:
    """Test the sqrt(K) deflection law for one (cell, w): anchor on FIT, predict+test on TEST.

    The anchor d1 is the single-image deflection on the FIT split; predictions for K>1 use only d1
    and the theorem's sqrt(K). The empirical AUC(K) is measured on the disjoint TEST split. The
    exponent is fit on the UNSATURATED part of the curve (AUC strictly inside (0.5+eps, 1-eps)).
    """
    if not k_grid:
        raise ValueError("k_grid must be non-empty")
    k_grid = tuple(_positive_index("k", k) for k in k_grid)
    n_groups = _positive_index("n_groups", n_groups)
    rng = np.random.default_rng(seed)
    d1_fit = deflection_from_auc(
        auc(
            np.concatenate([h1_fit, h0_fit]),
            np.concatenate([np.ones(h1_fit.size, int), np.zeros(h0_fit.size, int)]),
        )
    )
    auc1_test = auc(
        np.concatenate([h1_test, h0_test]),
        np.concatenate([np.ones(h1_test.size, int), np.zeros(h0_test.size, int)]),
    )
    auc_pred = tuple(predicted_group_auc(d1_fit, k) for k in k_grid)
    auc_emp = tuple(
        empirical_group_auc(h0_test, h1_test, k, n_groups=n_groups, rng=rng) for k in k_grid
    )
    # Exponent fit: d_emp(K) should be sqrt(K)*d(1) -> log d_emp = 0.5 log K + const. Use only the
    # unsaturated K (deflection finite + AUC not pinned at the clamp) so saturation doesn't bias it.
    d_emp = np.array([deflection_from_auc(a) for a in auc_emp], dtype=np.float64)
    ks = np.array(k_grid, dtype=np.float64)
    unsat = np.array(
        [
            (_AUC_CLAMP < a < 1.0 - _AUC_CLAMP) and d > 0.0
            for a, d in zip(auc_emp, d_emp, strict=True)
        ]
    )
    if unsat.sum() >= 2:
        lx, ly = np.log(ks[unsat]), np.log(d_emp[unsat])
        exponent: float | None = float(np.polyfit(lx, ly, 1)[0])
    else:
        exponent = None
    exponent_ok = bool(exponent is not None and abs(exponent - 0.5) <= exponent_tol)
    mean_abs_auc_err = float(np.mean(np.abs(np.array(auc_emp) - np.array(auc_pred))))
    return NeffScalingResult(
        cell=cell,
        w=w,
        d1_fit=d1_fit,
        auc1_test=float(auc1_test),
        k_grid=tuple(k_grid),
        auc_pred=auc_pred,
        auc_emp=auc_emp,
        exponent=exponent,
        exponent_ok=exponent_ok,
        mean_abs_auc_err=mean_abs_auc_err,
    )


@dataclass
class _Cell:
    h0: dict[str, list[float]] = field(default_factory=lambda: {"fit": [], "test": []})
    h1: dict[int, dict[str, list[float]]] = field(default_factory=dict)
    decoder: str = ""
    sigma_blur: float = 0.0


def _load_p0_rows(table_path: Path) -> list[dict]:
    """Load a p0_table artifact with the same fail-loud JSON contract as the core gate."""
    raw = loads_strict_json(table_path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError(f"{table_path}: p0_table.json must contain a list of rows")
    rows: list[dict] = []
    for row in raw:
        if not isinstance(row, dict):
            raise ValueError(f"{table_path}: p0_table.json rows must be objects")
        rows.append(dict(row))
    return rows


def _group_rows_by_cell(rows: list[dict]) -> dict[str, _Cell]:
    """Group p0_table rows into per-cell H0/H1-by-w fit/test score lists.

    fit = the ``train`` split (anchor); test = the ``test`` split (held-out). H0 is label 'h0'
    (w=1); H1 is label 'h1' keyed by w. Cells are (decoder, sigma_blur).

    Two exclusions make this the shared main-grid closure point (backing BOTH run_neff_scaling and
    run_pooled_neff_scaling):

    * branch_mode: a row whose stamped ``branch_mode`` is non-core (e.g. the non-core FLUX decoder
      with ``branch_mode='flux_specific_appendix'``) is FILTERED OUT before anything else, so a
      non-core FLUX run cannot enter the favorable-subset pool. A missing ``branch_mode`` defaults
      to ``true_cfg_core`` (old unstamped SD artifacts are KEPT); only explicitly-stamped non-core
      rows are dropped. This mirrors sweep/analyze's ``_effective_branch_mode`` core filter. The
      filter runs BEFORE the QF guard so a non-core FLUX run's own QF cannot trip the core QF check.
    * QF: cells are (decoder, sigma_blur) with QF collapsed to none, so a live QF axis is
      rejected: compressed-QF rows (run_p0 writes the int QF, e.g. 75 for a config with
      grid.qf=[none,75]) must NOT be silently pooled with qf=none rows into one cell. sweep/analyze
      enforces the same via _require_qf_collapsed_to_none.

    Both exclusions live in the shared ``filter_core_none_qf_rows`` so every bespoke
    (decoder, sigma_blur) grouper routes through the SAME filter and cannot drift from this one.
    """
    rows = filter_core_none_qf_rows(rows)
    cells: dict[str, _Cell] = {}
    for r in rows:
        rho = _row_finite_float(r, "rho_summary")
        sigma_blur = _row_finite_float(r, "sigma_blur")
        w = _row_finite_int(r, "w")
        if "decoder" not in r:
            raise ValueError("row field 'decoder' is required")
        split = r.get("split")
        bucket = "fit" if split == "train" else ("test" if split == "test" else None)
        if bucket is None:
            continue
        key = f"{r['decoder']}/sb={sigma_blur:g}"
        cell = cells.setdefault(key, _Cell())
        cell.decoder = str(r["decoder"])
        cell.sigma_blur = sigma_blur
        label = str(r.get("label"))
        if label == "h0":
            cell.h0[bucket].append(rho)
        elif label == "h1":
            cell.h1.setdefault(w, {"fit": [], "test": []})[bucket].append(float(rho))
        else:
            raise ValueError("row label must be one of ('h0', 'h1')")
    return cells


def _core_branch_mode_rows(rows: list[dict]) -> list[dict]:
    """The core branch_mode filter from ``filter_core_none_qf_rows``, WITHOUT its QF-collapse guard.

    Same branch_mode contract (keep missing/None/``true_cfg_core``, drop ``flux_specific_appendix``,
    fail loud on anything else), but no compressed-QF rejection, so a per-QF robustness grouper can
    admit the very rows the main-grid path rejects. ``filter_core_none_qf_rows`` itself stays
    byte-identical (this is a separate helper, not a refactor of it), so the main-grid guard never
    drifts.
    """
    core: list[dict] = []
    for row in rows:
        bm = row.get("branch_mode")
        if bm is None or str(bm) == "true_cfg_core":
            core.append(row)
            continue
        if str(bm) == "flux_specific_appendix":
            continue
        raise ValueError(
            "row 'branch_mode' must be missing/None, 'true_cfg_core', or "
            f"'flux_specific_appendix'; got {bm!r}"
        )
    return core


def _row_sigma_blur_default0(row: dict) -> float:
    """``sigma_blur`` for a row, defaulting a MISSING/None field to 0.0.

    Older QF-only tables predate the sigma_blur axis, so their rows carry no ``sigma_blur``
    field; treat them as the unblurred sb=0 condition. A PRESENT field must still be finite (no
    silent NaN) -- reuse the same fail-loud guard the main-grid grouper uses.
    """
    if row.get("sigma_blur") is None:
        return 0.0
    return _row_finite_float(row, "sigma_blur")


def group_rows_by_cell_per_condition(rows: list[dict], key: str, value: object) -> dict[str, _Cell]:
    """Group p0_table rows for ONE post-processing CONDITION (``row[key] == value``) into per-cell
    H0/H1 fit/test score lists.

    Robustness-analysis counterpart of ``_group_rows_by_cell``: identical cell/bucket/label
    semantics (fit = ``train`` split, test = ``test`` split, H0 = label ``'h0'``, H1 = label
    ``'h1'`` keyed by ``w``, cell key ``f"{decoder}/sb={sigma_blur:g}"``) EXCEPT it filters to a
    single condition BEFORE grouping instead of failing loud on a compressed QF. ``key='qf'``
    matches the int QF sentinel (so ``value=-1`` means uncompressed/none); any other key (e.g.
    ``'postproc'``) matches ``str(row[key]) == str(value)``. This is intentionally a SEPARATE entry
    point: ``filter_core_none_qf_rows`` and ``_group_rows_by_cell`` stay byte-identical and keep
    guarding the main-grid (decoder, sigma_blur) QF=none pool, so a per-condition robustness pool
    can never silently contaminate the favorable-subset alive-set. ``sigma_blur`` defaults to 0.0
    for the older QF-only tables that predate that axis.
    """
    qf_mode = key == "qf"
    want_qf = int(value) if qf_mode else None  # type: ignore[arg-type]
    want_str = str(value)
    core = _core_branch_mode_rows(rows)
    cells: dict[str, _Cell] = {}
    for r in core:
        if qf_mode:
            if _qf_sentinel(r) != want_qf:
                continue
        elif str(r.get(key)) != want_str:
            continue
        rho = _row_finite_float(r, "rho_summary")
        sigma_blur = _row_sigma_blur_default0(r)
        w = _row_finite_int(r, "w")
        if "decoder" not in r:
            raise ValueError("row field 'decoder' is required")
        split = r.get("split")
        bucket = "fit" if split == "train" else ("test" if split == "test" else None)
        if bucket is None:
            continue
        cell_key = f"{r['decoder']}/sb={sigma_blur:g}"
        cell = cells.setdefault(cell_key, _Cell())
        cell.decoder = str(r["decoder"])
        cell.sigma_blur = sigma_blur
        label = str(r.get("label"))
        if label == "h0":
            cell.h0[bucket].append(rho)
        elif label == "h1":
            cell.h1.setdefault(w, {"fit": [], "test": []})[bucket].append(float(rho))
        else:
            raise ValueError("row label must be one of ('h0', 'h1')")
    return cells


def group_rows_by_cell_per_qf(rows: list[dict], qf: int) -> dict[str, _Cell]:
    """Per-QF grouper: ``group_rows_by_cell_per_condition`` with key='qf' (-1 == uncompressed/none).

    A separate entry point from the main-grid ``_group_rows_by_cell`` /
    ``filter_core_none_qf_rows``, which stay byte-identical and keep failing loud on a compressed
    QF so the main-grid pool is pure.
    """
    return group_rows_by_cell_per_condition(rows, "qf", qf)


def _alive_w_for_cell(
    cell: _Cell, *, alive_floor: float, target_auc1: float, sign_aware: bool = False
) -> int | None:
    """Pick the w whose single-image FIT AUC is alive and closest to target (unsaturated regime).

    The sqrt(K) sweep is most informative where the single-image AUC is moderate (so AUC(K) rises
    across K instead of saturating at K=1). Selection uses only the fit split; the test split is
    reserved for the empirical group-K AUC measurement. Returns None if no w clears the alive floor.

    ``sign_aware``: the rho channel is sign-unstable (CFG LOWERS rho in many cells, AUC<0.5).
    When True, "alive" and "closeness to target" are measured on the signal MAGNITUDE |AUC-0.5| (so
    an inverted-but-strong cell is kept), with ``alive_floor`` / ``target_auc1`` read as their
    distance above 0.5. Default False preserves the one-sided (high-AUC only) per-cell selection.
    """
    best_w: int | None = None
    best_gap = float("inf")
    floor_mag = abs(alive_floor - 0.5)
    target_mag = abs(target_auc1 - 0.5)
    for w, sub in cell.h1.items():
        h0_f, h1_f = cell.h0["fit"], sub["fit"]
        h0_t, h1_t = cell.h0["test"], sub["test"]
        if min(len(h0_f), len(h1_f), len(h0_t), len(h1_t)) < 4:
            continue
        a1 = auc(
            np.concatenate([np.asarray(h1_f), np.asarray(h0_f)]),
            np.concatenate([np.ones(len(h1_f), int), np.zeros(len(h0_f), int)]),
        )
        if sign_aware:
            strength = abs(a1 - 0.5)
            if strength < floor_mag:
                continue
            gap = abs(strength - target_mag)
        else:
            if a1 < alive_floor:
                continue
            gap = abs(a1 - target_auc1)
        if gap < best_gap:
            best_gap, best_w = gap, w
    return best_w


def run_neff_scaling(
    features_root,
    out_dir,
    *,
    k_grid: tuple[int, ...] = (1, 2, 4, 8, 16),
    alive_floor: float = 0.60,
    target_auc1: float = DEFAULT_TARGET_AUC1,
    n_groups: int = 4000,
    seed: int = 0,
) -> int:
    """CLI body: run the N_eff scaling test on every alive cell and persist a summary + memo.

    Consumes the persisted ``p0_table.json`` (no re-inversion). For each alive cell it anchors d(1)
    on the train split, predicts AUC(K) via the sqrt(K) law, and tests against the held-out test
    split + the pre-registered 0.5 exponent.
    """
    import json

    features_root = Path(features_root)
    out_dir = Path(out_dir)
    rows: list[dict] = []
    for table_path in sorted(features_root.glob("*/p0_table.json")):
        rows.extend(_load_p0_rows(table_path))
    if not rows:
        print(f"[skip] no p0_table.json under {features_root}")
        return 0
    cells = _group_rows_by_cell(rows)
    results: list[NeffScalingResult] = []
    for key in sorted(cells):
        cell = cells[key]
        w = _alive_w_for_cell(cell, alive_floor=alive_floor, target_auc1=target_auc1)
        if w is None:
            print(f"[skip] {key}: no alive w (single-image fit AUC < {alive_floor})")
            continue
        sub = cell.h1[w]
        if len(cell.h0["fit"]) < 4 or len(sub["fit"]) < 4:
            print(f"[skip] {key}/w={w}: too few fit-split images to anchor d(1)")
            continue
        results.append(
            neff_scaling_for_cell(
                cell=key,
                w=w,
                h0_fit=np.asarray(cell.h0["fit"]),
                h1_fit=np.asarray(sub["fit"]),
                h0_test=np.asarray(cell.h0["test"]),
                h1_test=np.asarray(sub["test"]),
                k_grid=k_grid,
                n_groups=n_groups,
                seed=seed,
            )
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = [
        {
            "cell": r.cell,
            "w": r.w,
            "d1_fit": r.d1_fit,
            "auc1_test": r.auc1_test,
            "k_grid": list(r.k_grid),
            "auc_pred": list(r.auc_pred),
            "auc_emp": list(r.auc_emp),
            "exponent": r.exponent,
            "exponent_ok": r.exponent_ok,
            "mean_abs_auc_err": r.mean_abs_auc_err,
        }
        for r in results
    ]
    (out_dir / "neff_scaling.json").write_text(
        json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8"
    )
    n_ok = sum(r.exponent_ok for r in results)
    print(f"\nN_eff scaling test: {len(results)} alive cells, {n_ok} pass the 0.5-exponent law")
    for r in results:
        flag = "OK " if r.exponent_ok else "no "
        exp = f"{r.exponent:+.3f}" if r.exponent is not None else "n/a"
        print(
            f"  [{flag}] {r.cell:16s} w={r.w:2d}  d1={r.d1_fit:.3f}  "
            f"exponent={exp} (want 0.5)  AUC-err={r.mean_abs_auc_err:.3f}"
        )
    print(f"-> {out_dir / 'neff_scaling.json'}")
    return 0


def main(argv: list[str] | None = None) -> int:
    import argparse

    from cfgid.paths import outputs_dir

    ap = argparse.ArgumentParser(prog="cfgid.sweep.neff_scaling")
    ap.add_argument(
        "--features-root",
        default=None,
        help="dir of <decoder>/p0_table.json (default: outputs/p0, matching cfgid.sweep.analyze)",
    )
    ap.add_argument("--out-dir", default=None, help="default: outputs/p0")
    ap.add_argument("--alive-floor", type=float, default=0.60)
    ap.add_argument("--n-groups", type=int, default=4000)
    args = ap.parse_args(argv)
    # run_p0 writes outputs/p0/<decoder>/p0_table.json and analyze reads outputs/p0; match it.
    features_root = args.features_root or (outputs_dir() / "p0")
    out_dir = args.out_dir or (outputs_dir() / "p0")
    return run_neff_scaling(
        features_root, out_dir, alive_floor=args.alive_floor, n_groups=args.n_groups
    )


if __name__ == "__main__":
    raise SystemExit(main())
