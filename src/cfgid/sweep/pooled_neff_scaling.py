"""Favorable-subset selection and the pooled N_eff sqrt(K) test (Section IV-E, Eq. (14)).

The per-cell test of ``cfgid.sweep.neff_scaling`` is underpowered: each cell has a small TEST
split, so the single-image deflection is noisy and the exponent test cannot distinguish sqrt(K)
from neighbouring laws, and an alive-cell SELECTION made on the fit split alone is noise-dominated
(pure-noise cells can pass and train-selected cells can sign-flip on test). This module fixes both:

1. STANDARDIZE each (decoder, sigma_blur) cell to z-scores using the FIT-split H0 mean/std, so cells
   share a common scale (z: H0 ~ N(0,1), H1 ~ N(d_cell, 1)).
2. HELD-OUT, SIGN-AWARE ALIVE GATE: split the persisted TEST rows deterministically into gate/eval
   halves. Keep a cell only if the TEST-gate H1-H0 z-mean shift is significant (|t|>=alive_t) AND
   the SAME SIGN as the fit shift. The rho channel is sign-unstable (CFG LOWERS per-patch rho in
   many cells), so the gate is two-sided; the fit/test sign-consistency kills the train-only /
   sign-flip false positives that a fit-only floor admits. The TEST-eval half is reserved for the
   AUC and permutation null, so gate labels and test labels are disjoint. This is the paper's
   prespecified favorable-subset gate (Welch |t| >= 1.5 on the gate half, shift sign agreeing with
   the FIT sign, each cell at its FIT-best w against w=1).
3. SIGN-ALIGN + POOL: flip the inverted cells to a common positive shift, then pool -> a large
   common-scale, coherent sample (H0 ~ N(0,1), H1 ~ N(d_pool, ~1)), reducing the per-cell sampling
   variance. The sqrt(K) test then runs on |signal| regardless of the per-cell sign.
4. SQRT(K) TEST on the pool: aggregate K independent images (mean-of-K), measure the group AUC vs
   the analytic Phi(sqrt(K)*d_pool/sqrt(2)) and fit the log-log exponent.
5. LABEL-PERMUTATION NULL: shuffle H0/H1 labels in the pool and recompute the pooled single-image
   AUC + the K-aggregation rise; the one-sided p-value (1 + #{perm >= obs}) / (n_perm + 1), with
   n_perm = 1000 by default, tells whether the pooled signal AND its sqrt(K) growth could arise
   from pure noise.

Heterogeneity caveat: pooling cells with different d_cell makes the ABSOLUTE Phi(sqrt(K)*d/sqrt2)
prediction approximate (the pooled H1 is a mixture); the EXPONENT (1/K variance scaling) and the
permutation null are robust to it. Both are reported.

Anchor (d for the prediction) is estimated on the pooled FIT z; the empirical group AUC + the
permutation null are measured on the disjoint pooled TEST-eval z. NO re-inversion: consumes
p0_table.json per-image rho_summary. CLIP-confound regression is a separate step (per-image CLIP is
not in the table; it is recomputed from the stored PNGs by the nuisance baseline).

Role in the pipeline: ``collect_alive_cell_records`` / ``pooled_eval_auc_perm`` define the
favorable subset and its pooled single-image test; the scripts behind Section IV-E (e.g.
scripts/analysis/clip_vs_rho.py, sign_provenance.py, pooled_robustness.py, rho_ablation.py,
postproc_robustness.py) import this selection, so every analysis that selects favorable cells
uses the same rule.

Produces: the favorable subset of Section IV-E and Appendix B (``n_alive_cells``, ``alive_cells``,
``auc1_pool_test`` with ``auc1_perm_p``, ``n_pool_h0`` / ``n_pool_h1`` evaluation images) and its
pooled AUC_K curve (``auc_emp`` vs ``auc_pred``, ``exponent``).
Inputs: the outputs/p0/<decoder>/p0_table.json files only; CPU.
Outputs: outputs/p0/pooled_neff_scaling.json.
Usage: uv run --no-sync python -m cfgid.sweep.pooled_neff_scaling [--alive-t 1.5]
"""

from __future__ import annotations

from dataclasses import dataclass
from operator import index

import numpy as np

from cfgid.detector.empirical import auc
from cfgid.sweep.neff_scaling import (
    _group_rows_by_cell,
    _load_p0_rows,
    deflection_from_auc,
    empirical_group_auc,
    predicted_group_auc,
)


def _positive_int(name: str, value: int) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    try:
        out = index(value)
    except TypeError as exc:
        raise ValueError(f"{name} must be a positive integer, got {value!r}") from exc
    if out < 1:
        raise ValueError(f"{name} must be >= 1, got {out}")
    return out


def _standardize(values: np.ndarray, m0: float, s0: float) -> np.ndarray:
    if not (np.isfinite(m0) and np.isfinite(s0) and s0 > 0.0):
        raise ValueError(f"standardization needs finite s0 > 0, got m0={m0}, s0={s0}")
    return (np.asarray(values, dtype=np.float64).reshape(-1) - m0) / s0


def _group_auc(
    h0: np.ndarray, h1: np.ndarray, k: int, *, n_groups: int, rng: np.random.Generator
) -> float:
    """AUC of the mean-of-K detector on pooled z-scores (bootstrap groups)."""
    return empirical_group_auc(h0, h1, k, n_groups=n_groups, rng=rng)


def _split_gate_eval(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Deterministically split a held-out test vector into gate and eval halves.

    No shuffling: the first floor(n/2) values (persisted p0_table row order) form the gate half and
    the rest the eval half, so the split is reproducible from the table alone.
    """
    v = np.asarray(values, dtype=np.float64).reshape(-1)
    mid = v.size // 2
    return v[:mid], v[mid:]


def _two_sample_shift_t(h0: np.ndarray, h1: np.ndarray) -> tuple[float, float]:
    """Return H1-H0 mean shift and Welch t-stat on already-standardized test-gate z-scores."""
    z0 = np.asarray(h0, dtype=np.float64).reshape(-1)
    z1 = np.asarray(h1, dtype=np.float64).reshape(-1)
    if min(z0.size, z1.size) < 2:
        return float("nan"), float("nan")
    shift = float(np.mean(z1) - np.mean(z0))
    se = float(np.sqrt(np.var(z0, ddof=1) / z0.size + np.var(z1, ddof=1) / z1.size))
    if not (np.isfinite(se) and se > 0.0):
        return shift, float("nan")
    return shift, float(shift / se)


@dataclass(frozen=True)
class PooledResult:
    n_alive_cells: int
    alive_cells: tuple[str, ...]
    n_pool_h0: int
    n_pool_h1: int
    d_pool_fit: float  # anchor deflection from the pooled FIT z
    auc1_pool_test: float  # pooled single-image AUC on TEST
    auc1_perm_p: float  # permutation p-value for the pooled single-image AUC
    k_grid: tuple[int, ...]
    auc_pred: tuple[float, ...]  # Phi(sqrt(K)*d_pool_fit/sqrt(2))
    auc_emp: tuple[float, ...]  # pooled TEST group-K AUC
    rise_emp: float  # auc_emp[-1] - auc_emp[0] (the K-aggregation rise)
    rise_perm_p: float  # permutation p-value for the rise
    exponent: float | None  # log-log slope of d_emp vs K on the unsaturated points
    exponent_ok: bool


def pooled_sqrtk(
    z_h0_fit: np.ndarray,
    z_h1_fit: np.ndarray,
    z_h0_test: np.ndarray,
    z_h1_test: np.ndarray,
    *,
    alive_cells: tuple[str, ...],
    k_grid: tuple[int, ...] = (1, 2, 4, 8, 16, 32),
    n_groups: int = 8000,
    n_perm: int = 1000,
    exponent_tol: float = 0.15,
    seed: int = 0,
) -> PooledResult:
    """Run the sqrt(K) test + label-permutation null on the pooled standardized z-scores."""
    if not k_grid:
        raise ValueError("k_grid must be non-empty")
    k_grid = tuple(_positive_int("k", k) for k in k_grid)
    n_groups = _positive_int("n_groups", n_groups)
    n_perm = _positive_int("n_perm", n_perm)
    rng = np.random.default_rng(seed)
    z_h0_fit = np.asarray(z_h0_fit, dtype=np.float64).reshape(-1)
    z_h1_fit = np.asarray(z_h1_fit, dtype=np.float64).reshape(-1)
    z_h0_test = np.asarray(z_h0_test, dtype=np.float64).reshape(-1)
    z_h1_test = np.asarray(z_h1_test, dtype=np.float64).reshape(-1)
    for name, a in (
        ("z_h0_fit", z_h0_fit),
        ("z_h1_fit", z_h1_fit),
        ("z_h0_test", z_h0_test),
        ("z_h1_test", z_h1_test),
    ):
        if a.size == 0 or not np.all(np.isfinite(a)):
            raise ValueError(f"{name} must be non-empty and finite")

    d_pool_fit = deflection_from_auc(
        auc(
            np.concatenate([z_h1_fit, z_h0_fit]),
            np.concatenate([np.ones(z_h1_fit.size, int), np.zeros(z_h0_fit.size, int)]),
        )
    )
    test_scores = np.concatenate([z_h1_test, z_h0_test])
    test_labels = np.concatenate([np.ones(z_h1_test.size, int), np.zeros(z_h0_test.size, int)])
    auc1_test = float(auc(test_scores, test_labels))

    auc_pred = tuple(predicted_group_auc(d_pool_fit, k) for k in k_grid)
    auc_emp = tuple(_group_auc(z_h0_test, z_h1_test, k, n_groups=n_groups, rng=rng) for k in k_grid)
    rise_emp = float(auc_emp[-1] - auc_emp[0])

    # Label-permutation null: shuffle test labels, recompute the single-image AUC and the rise.
    perm_auc1 = np.empty(n_perm)
    perm_rise = np.empty(n_perm)
    pool = test_scores
    n1 = z_h1_test.size
    for p in range(n_perm):
        perm = rng.permutation(pool.size)
        zh1 = pool[perm[:n1]]
        zh0 = pool[perm[n1:]]
        perm_auc1[p] = auc(
            np.concatenate([zh1, zh0]),
            np.concatenate([np.ones(zh1.size, int), np.zeros(zh0.size, int)]),
        )
        # The null replicate of the rise MUST use the same Monte-Carlo resolution (n_groups) as the
        # observed rise_emp. A coarser null estimator (the old hard-coded 2000 vs the 8000-default
        # observed) over-disperses perm_rise and inflates rise_perm_p -- conservative, but a
        # permutation test is only valid when the observed statistic and its null replicates are
        # computed by the identical procedure.
        a_lo = _group_auc(zh0, zh1, k_grid[0], n_groups=n_groups, rng=rng)
        a_hi = _group_auc(zh0, zh1, k_grid[-1], n_groups=n_groups, rng=rng)
        perm_rise[p] = a_hi - a_lo
    auc1_perm_p = float((1.0 + np.sum(perm_auc1 >= auc1_test)) / (n_perm + 1.0))
    if len(k_grid) == 1:
        # A single-K grid has no K-aggregation rise to test: rise_emp = auc_emp[0]-auc_emp[0] = 0
        # exactly, while each perm_rise = a_hi - a_lo draws TWO independent group-AUCs at the SAME
        # k, so the null is centred (noisy) about 0 and the permutation p would land near 0.5 -- a
        # misleadingly "borderline" value for a structurally vacuous test. Report 1.0 (no evidence
        # of a rise) honestly. The default k_grid has 6 elements, so this never fires in production.
        rise_perm_p = 1.0
    else:
        rise_perm_p = float((1.0 + np.sum(perm_rise >= rise_emp)) / (n_perm + 1.0))

    d_emp = np.array([deflection_from_auc(a) for a in auc_emp])
    ks = np.array(k_grid, dtype=np.float64)
    unsat = np.array([1e-4 < a < 1 - 1e-4 and d > 0 for a, d in zip(auc_emp, d_emp, strict=True)])
    if unsat.sum() >= 2:
        exponent: float | None = float(np.polyfit(np.log(ks[unsat]), np.log(d_emp[unsat]), 1)[0])
    else:
        exponent = None
    exponent_ok = bool(exponent is not None and abs(exponent - 0.5) <= exponent_tol)

    return PooledResult(
        n_alive_cells=len(alive_cells),
        alive_cells=alive_cells,
        n_pool_h0=int(z_h0_test.size),
        n_pool_h1=int(z_h1_test.size),
        d_pool_fit=d_pool_fit,
        auc1_pool_test=auc1_test,
        auc1_perm_p=auc1_perm_p,
        k_grid=tuple(k_grid),
        auc_pred=auc_pred,
        auc_emp=auc_emp,
        rise_emp=rise_emp,
        rise_perm_p=rise_perm_p,
        exponent=exponent,
        exponent_ok=exponent_ok,
    )


def _best_abs_signal_w(cell, *, min_n: int) -> int | None:
    """The w with the strongest |fit single-image AUC - 0.5| (signal MAGNITUDE, either sign).

    The rho channel is sign-UNSTABLE (CFG LOWERS per-patch rho in many cells, AUC<0.5), so we
    select on |AUC-0.5| rather than the highest AUC; an inverted-but-real cell is kept.
    """
    best_w, best_mag = None, -1.0
    for w, sub in cell.h1.items():
        h0_f, h1_f = cell.h0["fit"], sub["fit"]
        h0_t, h1_t = cell.h0["test"], sub["test"]
        if min(len(h0_f), len(h1_f), len(h0_t), len(h1_t)) < min_n:
            continue
        a = auc(
            np.concatenate([np.asarray(h1_f), np.asarray(h0_f)]),
            np.concatenate([np.ones(len(h1_f), int), np.zeros(len(h0_f), int)]),
        )
        mag = abs(a - 0.5)
        if mag > best_mag:
            best_mag, best_w = mag, w
    return best_w


@dataclass(frozen=True)
class CellRecord:
    """One alive cell's UNSIGNED standardized z-scores + its FIT calibrated sign.

    The z-scores are stored UNSIGNED (sign-alignment is applied by the consumer); ``sign`` is the
    FIT shift sign (+1/-1) that ``_pool_alive_cells`` uses to flip the cell into a positive-shift
    channel. Keeping sign separate from the z-scores lets the sign-provenance analysis re-pool the
    SAME cells under a PREDICTED (rather than calibrated) sign without re-deriving the selection.
    """

    key: str
    decoder: str
    sigma_blur: float
    w: int
    sign: float
    zf0: np.ndarray  # standardized FIT H0 (unsigned)
    zf1: np.ndarray  # standardized FIT H1 (unsigned)
    zt0_eval: np.ndarray  # standardized TEST-eval H0 (unsigned)
    zt1_eval: np.ndarray  # standardized TEST-eval H1 (unsigned)

    def label(self) -> str:
        return f"{self.key}/w={self.w}/sign={'+' if self.sign > 0 else '-'}"


def collect_alive_cell_records(cells: dict, *, alive_t: float) -> list[CellRecord]:
    """Standardize each cell on fit-H0, gate on a SIGN-CONSISTENT held-out shift; keep survivors.

    The persisted TEST rows are split into deterministic gate/eval halves. A cell is kept only if
    its TEST-gate H1-H0 shift is significant (|t|>=alive_t) AND the same sign as its FIT shift
    (kills train-only / sign-flip cells). The TEST-eval half is reserved for the pooled AUC and
    permutation null, so selection labels and evaluation labels stay disjoint. The returned records
    hold UNSIGNED z-scores plus the FIT sign, so the pooled test (sign-aligned) and the
    sign-provenance analysis (predicted sign) share one selection without drift.
    """
    if not (np.isfinite(alive_t) and alive_t >= 0.0):
        raise ValueError(f"alive_t must be finite and non-negative, got {alive_t}")
    records: list[CellRecord] = []
    for key in sorted(cells):
        cell = cells[key]
        w = _best_abs_signal_w(cell, min_n=8)
        if w is None:
            continue
        h0_fit = np.asarray(cell.h0["fit"], dtype=np.float64)
        h1_fit = np.asarray(cell.h1[w]["fit"], dtype=np.float64)
        h0_test = np.asarray(cell.h0["test"], dtype=np.float64)
        h1_test = np.asarray(cell.h1[w]["test"], dtype=np.float64)
        h0_gate, h0_eval = _split_gate_eval(h0_test)
        h1_gate, h1_eval = _split_gate_eval(h1_test)
        if min(h0_gate.size, h0_eval.size, h1_gate.size, h1_eval.size) < 4:
            continue
        # FIT-H0 moments (ddof=1) standardize this cell's FIT, gate and eval scores alike.
        m0, s0 = float(np.mean(h0_fit)), float(np.std(h0_fit, ddof=1))
        if not (np.isfinite(s0) and s0 > 0):
            continue
        zf1c = _standardize(h1_fit, m0, s0)
        s = float(np.sign(np.mean(zf1c)))  # FIT shift sign
        if s == 0.0:
            continue
        zt0_gate = _standardize(h0_gate, m0, s0)
        zt1_gate = _standardize(h1_gate, m0, s0)
        test_shift, t = _two_sample_shift_t(zt0_gate, zt1_gate)
        test_sign = float(np.sign(test_shift))
        if (
            not np.isfinite(t) or abs(t) < alive_t or test_sign != s
        ):  # significant on gate AND sign-consistent with fit
            continue
        records.append(
            CellRecord(
                key=key,
                decoder=cell.decoder,
                sigma_blur=cell.sigma_blur,
                w=w,
                sign=s,
                zf0=_standardize(h0_fit, m0, s0),
                zf1=zf1c,
                zt0_eval=_standardize(h0_eval, m0, s0),
                zt1_eval=_standardize(h1_eval, m0, s0),
            )
        )
    return records


def _aligned_eval_pool(records: list[CellRecord], signs) -> tuple[np.ndarray, np.ndarray]:
    """Concatenate each cell's UNSIGNED eval z aligned by ``signs[i]`` -> (scores, labels)."""
    signs = list(signs)
    if len(signs) != len(records):
        raise ValueError(f"signs ({len(signs)}) must match records ({len(records)})")
    if not records:
        raise ValueError("aligned eval pool needs at least one cell record")
    z0 = np.concatenate([s * r.zt0_eval for s, r in zip(signs, records, strict=True)])
    z1 = np.concatenate([s * r.zt1_eval for s, r in zip(signs, records, strict=True)])
    scores = np.concatenate([z1, z0])
    labels = np.concatenate([np.ones(z1.size, int), np.zeros(z0.size, int)])
    return scores, labels


def pooled_eval_auc(records: list[CellRecord], signs) -> float:
    """Pooled single-image TEST-eval AUC under a per-cell sign vector (no permutation)."""
    scores, labels = _aligned_eval_pool(records, signs)
    return float(auc(scores, labels))


def pooled_eval_auc_perm(
    records: list[CellRecord], signs, *, n_perm: int, seed: int
) -> tuple[float, float]:
    """Pooled single-image TEST-eval AUC under a per-cell sign vector + one-sided permutation p.

    Aligns each cell's UNSIGNED eval z by ``signs[i]`` (so a correct sign sends H1 to a positive
    shift), concatenates, and returns the single-image AUC plus the one-sided label-permutation
    p-value (1+#{perm>=obs})/(n_perm+1). The diagnostic scripts (sign provenance, robustness) share
    this one implementation so "pooled AUC under signs" is computed identically everywhere. The
    signs are derived without the eval labels, so shuffling those labels is a valid null.
    """
    n_perm = _positive_int("n_perm", n_perm)
    scores, labels = _aligned_eval_pool(records, signs)
    obs = float(auc(scores, labels))
    rng = np.random.default_rng(seed)
    perm = np.array([auc(scores, rng.permutation(labels)) for _ in range(n_perm)])
    p = float((1.0 + np.sum(perm >= obs)) / (n_perm + 1.0))
    return obs, p


def _pool_alive_cells(
    cells: dict, *, alive_t: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, tuple[str, ...]]:
    """Standardize each cell on fit-H0, gate on a SIGN-CONSISTENT held-out shift, sign-align, pool.

    Thin wrapper over ``collect_alive_cell_records``: it sign-ALIGNS each survivor (flips the
    inverted cells to a positive shift) and concatenates them into one coherent pool, so the
    sqrt(K) test runs on |signal| regardless of the per-cell sign.
    """
    records = collect_alive_cell_records(cells, alive_t=alive_t)
    zf0 = [r.sign * r.zf0 for r in records]
    zf1 = [r.sign * r.zf1 for r in records]
    zt0 = [r.sign * r.zt0_eval for r in records]
    zt1 = [r.sign * r.zt1_eval for r in records]
    alive = tuple(r.label() for r in records)
    cat = lambda xs: np.concatenate(xs) if xs else np.empty(0)  # noqa: E731
    return cat(zf0), cat(zf1), cat(zt0), cat(zt1), alive


def run_pooled_neff_scaling(
    features_root,
    out_dir,
    *,
    alive_t: float = 1.5,
    n_perm: int = 1000,
    n_groups: int = 8000,
    seed: int = 0,
) -> int:
    import json
    from pathlib import Path

    features_root = Path(features_root)
    out_dir = Path(out_dir)
    rows: list[dict] = []
    for table_path in sorted(features_root.glob("*/p0_table.json")):
        rows.extend(_load_p0_rows(table_path))
    if not rows:
        print(f"[skip] no p0_table.json under {features_root}")
        return 0
    cells = _group_rows_by_cell(rows)
    zf0, zf1, zt0, zt1, alive = _pool_alive_cells(cells, alive_t=alive_t)
    if not alive or zf1.size < 4 or zt1.size < 4:
        print(f"[skip] no cells passed the held-out alive gate (alive_t={alive_t}); pool empty.")
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "pooled_neff_scaling.json").write_text(
            json.dumps({"n_alive_cells": 0, "alive_cells": list(alive)}, indent=2, allow_nan=False),
            encoding="utf-8",
        )
        return 0
    res = pooled_sqrtk(
        zf0, zf1, zt0, zt1, alive_cells=alive, n_perm=n_perm, n_groups=n_groups, seed=seed
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "pooled_neff_scaling.json").write_text(
        json.dumps(
            {
                "n_alive_cells": res.n_alive_cells,
                "alive_cells": list(res.alive_cells),
                "n_pool_h0": res.n_pool_h0,
                "n_pool_h1": res.n_pool_h1,
                "d_pool_fit": res.d_pool_fit,
                "auc1_pool_test": res.auc1_pool_test,
                "auc1_perm_p": res.auc1_perm_p,
                "k_grid": list(res.k_grid),
                "auc_pred": list(res.auc_pred),
                "auc_emp": list(res.auc_emp),
                "rise_emp": res.rise_emp,
                "rise_perm_p": res.rise_perm_p,
                "exponent": res.exponent,
                "exponent_ok": res.exponent_ok,
            },
            indent=2,
            allow_nan=False,
        ),
        encoding="utf-8",
    )
    print(f"\nPooled N_eff sqrt(K) test: {res.n_alive_cells} cells passed the held-out gate")
    print(f"  alive cells: {list(res.alive_cells)}")
    print(f"  pool: H0={res.n_pool_h0} H1={res.n_pool_h1} images (standardized z)")
    print(
        f"  single-image: AUC={res.auc1_pool_test:.3f}  permutation p={res.auc1_perm_p:.4f}  "
        f"(<0.05 => the pooled channel is REAL, not noise)"
    )
    print(
        f"  K-aggregation rise (AUC[K={res.k_grid[-1]}]-AUC[1])={res.rise_emp:+.3f}  "
        f"permutation p={res.rise_perm_p:.4f}"
    )
    emp = "  ".join(f"{a:.3f}" for a in res.auc_emp)
    prd = "  ".join(f"{a:.3f}" for a in res.auc_pred)
    print(f"  K={list(res.k_grid)}")
    print(f"    AUC_emp [{emp}]")
    print(f"    AUC_pred[{prd}]")
    exp = "n/a" if res.exponent is None else f"{res.exponent:+.3f}"
    print(f"  sqrt(K) exponent={exp} (want 0.5){'  OK' if res.exponent_ok else ''}")
    print(f"-> {out_dir / 'pooled_neff_scaling.json'}")
    return 0


def main(argv: list[str] | None = None) -> int:
    import argparse

    from cfgid.paths import outputs_dir

    ap = argparse.ArgumentParser(prog="cfgid.sweep.pooled_neff_scaling")
    ap.add_argument(
        "--features-root", default=None, help="dir of <decoder>/p0_table.json (default: outputs/p0)"
    )
    ap.add_argument(
        "--alive-t", type=float, default=1.5, help="held-out-test alive gate (one-sided t)"
    )
    ap.add_argument("--n-perm", type=int, default=1000)
    ap.add_argument("--n-groups", type=int, default=8000)
    args = ap.parse_args(argv)
    from pathlib import Path

    features_root = Path(args.features_root) if args.features_root else (outputs_dir() / "p0")
    return run_pooled_neff_scaling(
        features_root,
        outputs_dir() / "p0",
        alive_t=args.alive_t,
        n_perm=args.n_perm,
        n_groups=args.n_groups,
    )


if __name__ == "__main__":
    raise SystemExit(main())
