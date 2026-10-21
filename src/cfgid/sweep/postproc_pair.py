"""Core grouping and FIT-only pair readout for post-processing robustness analysis.

Role in the pipeline: library core of the post-processing pair analysis (its driver script is not
included in this repository), which produces the pair column of the refit arm of Table XII(a)
(Section IV-E6) from the post-processing pool written by
``run_p0 --config configs/robustness.yaml --lean``. For each condition (``postproc`` value) and
(decoder, sigma_blur) cell, the (rho, m) rows are grouped into FIT (``train``) / TEST (``test``)
lists; a Pair-LDA head of the form of Eq. (11) is fitted on FIT only (FIT-H0 standardization of
each coordinate, Ledoit-Wolf covariance of the standardized FIT-H0 rows, weights
solve(cov, mean FIT-H1 shift)), and the best w against w=1 is the one with the largest
|FIT AUC - 0.5|. TEST is split deterministically into a gate half (admits the cell when Welch
|t| >= alive_t with a positive shift) and an eval half that feeds the pooled AUC_1 and its
permutation test. ``_project_rho_cells`` rebuilds the rho-only cells from the same rows for the
rho self-check against postproc_robustness.json.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from cfgid.bounds.wstar_cov import ledoit_wolf_cov
from cfgid.detector.empirical import auc
from cfgid.sweep.neff_scaling import (
    _Cell,
    _core_branch_mode_rows,
    _row_finite_float,
    _row_finite_int,
    _row_sigma_blur_default0,
)
from cfgid.sweep.pooled_neff_scaling import (
    CellRecord,
    _split_gate_eval,
    _two_sample_shift_t,
)

_MIN_N: int = 8  # mirrors _best_abs_signal_w's min_n on the rho path
_MIN_HALF: int = 4  # mirrors collect_alive_cell_records' gate/eval half floor


@dataclass
class _PairCell:
    """Per-cell (rho, m) score lists with the same bucket semantics as neff_scaling._Cell."""

    h0: dict[str, list[tuple[float, float]]] = field(
        default_factory=lambda: {"fit": [], "test": []}
    )
    h1: dict[int, dict[str, list[tuple[float, float]]]] = field(default_factory=dict)
    decoder: str = ""
    sigma_blur: float = 0.0


def group_pair_rows_by_cell_per_condition(
    rows: list[dict], key: str, value: object
) -> dict[str, _PairCell]:
    """Group one condition into per-cell H0/H1 fit/test (rho, m) lists.

    This mirrors ``neff_scaling.group_rows_by_cell_per_condition`` for its non-QF branch. Rows
    without ``m_summary`` fail loud because dropping them would desynchronize the deterministic
    gate/eval halves from the rho projection.
    """
    want_str = str(value)
    core = _core_branch_mode_rows(rows)
    cells: dict[str, _PairCell] = {}
    for row in core:
        if str(row.get(key)) != want_str:
            continue
        rho = _row_finite_float(row, "rho_summary")
        if "m_summary" not in row:
            raise ValueError(
                "row lacks 'm_summary' -- these tables predate the pair field; rerun the "
                "robustness sweep with the m_summary-writing run_p0"
            )
        m = _row_finite_float(row, "m_summary")
        sigma_blur = _row_sigma_blur_default0(row)
        w = _row_finite_int(row, "w")
        if "decoder" not in row:
            raise ValueError("row field 'decoder' is required")
        split = row.get("split")
        bucket = "fit" if split == "train" else ("test" if split == "test" else None)
        if bucket is None:
            continue
        cell_key = f"{row['decoder']}/sb={sigma_blur:g}"
        cell = cells.setdefault(cell_key, _PairCell())
        cell.decoder = str(row["decoder"])
        cell.sigma_blur = sigma_blur
        label = str(row.get("label"))
        if label == "h0":
            cell.h0[bucket].append((rho, m))
        elif label == "h1":
            cell.h1.setdefault(w, {"fit": [], "test": []})[bucket].append((rho, m))
        else:
            raise ValueError("row label must be one of ('h0', 'h1')")
    return cells


def _project_rho_cells(pair_cells: dict[str, _PairCell]) -> dict[str, _Cell]:
    """Project rho back into neff_scaling._Cell without changing row or w insertion order."""
    out: dict[str, _Cell] = {}
    for key, pair_cell in pair_cells.items():
        cell = _Cell()
        cell.decoder, cell.sigma_blur = pair_cell.decoder, pair_cell.sigma_blur
        cell.h0 = {bucket: [rm[0] for rm in pair_cell.h0[bucket]] for bucket in ("fit", "test")}
        cell.h1 = {
            w: {bucket: [rm[0] for rm in sub[bucket]] for bucket in ("fit", "test")}
            for w, sub in pair_cell.h1.items()
        }
        out[key] = cell
    return out


def _fit_pair_head(
    f0: np.ndarray, f1: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """FIT-H0 standardization + FIT-H1 shift + FIT-H0 Ledoit-Wolf Fisher weights."""
    m0 = f0.mean(axis=0)
    s0 = f0.std(axis=0, ddof=1)
    if not (np.all(np.isfinite(s0)) and np.all(s0 > 0.0)):
        return None
    z0f = (f0 - m0) / s0
    z1f = (f1 - m0) / s0
    shift = z1f.mean(axis=0)
    if not np.all(np.isfinite(shift)):
        return None
    cov, _ = ledoit_wolf_cov(z0f)
    try:
        wvec = np.linalg.solve(cov, shift)
    except np.linalg.LinAlgError:
        return None
    if not np.all(np.isfinite(wvec)):
        return None
    return m0, s0, wvec


def collect_pair_cell_records(
    cells: dict[str, _PairCell], *, alive_t: float
) -> tuple[list[CellRecord], list[dict]]:
    """Fit and gate the pair analogue of ``collect_alive_cell_records``.

    The head and best-w selection use FIT only. TEST is split deterministically into gate/eval;
    gate only decides survival, while the returned ``CellRecord`` contains eval scores for AUC and
    permutation testing. Scores are standardized by FIT-H0 score statistics before pooling.
    """
    if not (np.isfinite(alive_t) and alive_t >= 0.0):
        raise ValueError(f"alive_t must be finite and non-negative, got {alive_t}")
    records: list[CellRecord] = []
    audit: list[dict] = []
    for key in sorted(cells):
        cell = cells[key]
        f0 = np.asarray(cell.h0["fit"], dtype=np.float64)
        t0 = np.asarray(cell.h0["test"], dtype=np.float64)
        best: tuple[float, int, np.ndarray, np.ndarray, np.ndarray] | None = None
        # Preserve the grouper's insertion order, matching _best_abs_signal_w's tie semantics.
        for w, sub in cell.h1.items():
            f1 = np.asarray(sub["fit"], dtype=np.float64)
            t1 = np.asarray(sub["test"], dtype=np.float64)
            if min(len(f0), len(f1), len(t0), len(t1)) < _MIN_N:
                continue
            head = _fit_pair_head(f0, f1)
            if head is None:
                continue
            m0, s0, wvec = head
            sc0f = ((f0 - m0) / s0) @ wvec
            sc1f = ((f1 - m0) / s0) @ wvec
            a = auc(
                np.concatenate([sc1f, sc0f]),
                np.concatenate([np.ones(sc1f.size, int), np.zeros(sc0f.size, int)]),
            )
            mag = abs(float(a) - 0.5)
            if best is None or mag > best[0]:
                best = (mag, w, m0, s0, wvec)
        if best is None:
            audit.append({"cell": key, "kept": False, "reason": "no fittable w"})
            continue
        _, w, m0, s0, wvec = best
        sub = cell.h1[w]
        f1 = np.asarray(sub["fit"], dtype=np.float64)
        t1 = np.asarray(sub["test"], dtype=np.float64)
        sc0f = ((f0 - m0) / s0) @ wvec
        sc1f = ((f1 - m0) / s0) @ wvec
        sc_m, sc_s = float(np.mean(sc0f)), float(np.std(sc0f, ddof=1))
        if not (np.isfinite(sc_s) and sc_s > 0.0):
            audit.append({"cell": key, "kept": False, "reason": "degenerate FIT score scale"})
            continue

        def z_of(
            x: np.ndarray,
            m0: np.ndarray = m0,
            s0: np.ndarray = s0,
            wvec: np.ndarray = wvec,
            sc_m: float = sc_m,
            sc_s: float = sc_s,
        ) -> np.ndarray:
            return (((x - m0) / s0) @ wvec - sc_m) / sc_s

        zt0_gate, zt0_eval = _split_gate_eval(z_of(t0))
        zt1_gate, zt1_eval = _split_gate_eval(z_of(t1))
        if min(zt0_gate.size, zt0_eval.size, zt1_gate.size, zt1_eval.size) < _MIN_HALF:
            audit.append({"cell": key, "kept": False, "reason": "gate/eval half too small"})
            continue
        shift_g, t = _two_sample_shift_t(zt0_gate, zt1_gate)
        if not np.isfinite(t) or abs(t) < alive_t or shift_g <= 0.0:
            audit.append(
                {
                    "cell": key,
                    "kept": False,
                    "reason": "gate not sign-consistent",
                    "gate_shift": float(shift_g),
                    "gate_t": float(t) if np.isfinite(t) else None,
                }
            )
            continue
        records.append(
            CellRecord(
                key=key,
                decoder=cell.decoder,
                sigma_blur=cell.sigma_blur,
                w=int(w),
                sign=1.0,
                zf0=(sc0f - sc_m) / sc_s,
                zf1=(sc1f - sc_m) / sc_s,
                zt0_eval=zt0_eval,
                zt1_eval=zt1_eval,
            )
        )
        audit.append(
            {
                "cell": key,
                "kept": True,
                "w": int(w),
                "n_fit_h0": int(f0.shape[0]),
                "n_fit_h1": int(f1.shape[0]),
                "n_eval_h0": int(zt0_eval.size),
                "n_eval_h1": int(zt1_eval.size),
                "gate_shift": float(shift_g),
                "gate_t": float(t),
                "w_rho": float(wvec[0]),
                "w_m": float(wvec[1]),
            }
        )
    return records, audit
