"""Finite-sample one-sided FPR bounds on the main grid from the reserved calibration (CAL) split.

FIT-null quantile thresholds carry no one-sided guarantee; this script uses the CAL split
instead. Per (decoder, sigma_blur) cell on the fixed w=20-vs-1 contrast, it scores the CAL-split
H0 rows with each deployed readout (rho, T_pair-zsum, T_pair-LDA; heads fitted on FIT) and sets
the conformal order-statistic threshold

    tau_alpha = k-th smallest CAL-H0 score with k = ceil((n_cal + 1) * (1 - alpha)),

which under exchangeability guarantees P(new H0 score > tau_alpha) <= alpha in finite
samples (distribution-free one-sided bound; infeasible when k > n_cal, i.e. n_cal < 19 at
alpha=0.05 and n_cal < 99 at alpha=0.01), then reports the realized FPR on the disjoint held-out
TEST H0 rows with an exact one-sided 95% Clopper--Pearson upper bound, and the TPR on TEST H1.
Pooled rows aggregate per-cell thresholds at row level (blur siblings span cells, so pooled rows
stay descriptive).

Readout recipes mirror deployed_within_contrasts.py / pair_readout_battery (fusion_lda_full:
H0-only FIT standardization + Ledoit--Wolf covariance; the magnitude coordinate m of Eq. (8) is
the store's masked winsorized diff_energy). Pure reanalysis, no re-inversion.

Produces: the single-image (K=1) main-grid conformal results of Section IV-D (CAL-null rows per
cell, cells feasible per alpha, pooled realized FPR per readout, Pair-LDA cellwise FPR and CP95).
Its ``_conformal_threshold`` and ``_cp_upper`` are reused by cal_complexity.py,
group_conformal_k.py and freeze_prospective_cal.py.
Inputs: main-grid outputs/p0/<dec>/p0_table.json + the server-side feature
store; CPU.
Outputs: outputs/p0/cal_split_fpr.json.
Usage:
  uv run --no-sync python scripts/analysis/cal_split_fpr.py
"""

from __future__ import annotations

import argparse
import math
from numbers import Integral
from pathlib import Path

import numpy as np
from _readout_common import (
    DEPLOYED_READOUTS,
    fit_deployed,
    load_residual_rows,
    score_deployed,
    write_json_strict,
)
from scipy.stats import beta

from cfgid.numerics import finite_float
from cfgid.paths import outputs_dir

ALPHAS = (0.05, 0.01)
H1_SET = {20.0}
H0_SET = {1.0}
READOUTS = DEPLOYED_READOUTS
RLABEL = {"rho": "rho", "zsum_dep": "deployed z-sum", "lda_dep": "deployed LW-LDA"}


def _fit_weight(
    cell_rows: list[dict], readout: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Deployed FIT recipe -> (wvec, m0, s0); None when the cell is too thin."""
    fit = fit_deployed(cell_rows, H1_SET, H0_SET, readout)
    if fit is None:
        return None
    return fit.wvec, fit.m0, fit.s0


def _conformal_threshold(cal_h0: np.ndarray, alpha: float) -> float | None:
    """k-th smallest CAL-H0 value, k = ceil((n + 1)(1 - alpha)); None when infeasible (k > n)."""
    values = np.asarray(cal_h0, dtype=np.float64).reshape(-1)
    if not np.all(np.isfinite(values)):
        raise ValueError("CAL-H0 scores contain non-finite values")
    if not (0.0 < alpha < 1.0) or not math.isfinite(alpha):
        raise ValueError(f"alpha must be finite and in (0, 1), got {alpha}")
    n = len(values)
    k = math.ceil((n + 1) * (1.0 - alpha))
    if k > n:
        return None
    return float(np.sort(values)[k - 1])


def _cp_upper(successes: int, n: int, conf: float = 0.95) -> float:
    """Exact one-sided Clopper--Pearson upper bound beta.ppf(conf, s + 1, n - s); 1.0 if s = n."""
    if (
        isinstance(successes, bool)
        or isinstance(n, bool)
        or not isinstance(successes, Integral)
        or not isinstance(n, Integral)
    ):
        raise ValueError("successes and n must be integers")
    s, total = int(successes), int(n)
    if not (0 <= s <= total):
        raise ValueError(f"require 0 <= successes <= n, got {s}/{total}")
    if not (0.0 < conf < 1.0) or not math.isfinite(conf):
        raise ValueError(f"conf must be finite and in (0, 1), got {conf}")
    if total == 0:
        return 1.0
    if s >= total:
        return 1.0
    return finite_float("Clopper-Pearson upper bound", beta.ppf(conf, s + 1, total - s))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default=None, help="dir of <decoder>/ p0 outputs (def outputs/p0)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    root = Path(args.root) if args.root else outputs_dir() / "p0"
    all_rows = load_residual_rows(root)

    cells: dict[tuple[str, float], list[dict]] = {}
    for r in all_rows:
        cells.setdefault((r["decoder"], r["sigma_blur"]), []).append(r)

    result: dict[str, dict] = {}
    for readout in READOUTS:
        per_cell: dict[str, dict] = {}
        pooled = {
            a: {
                "fp": 0,
                "n0": 0,
                "tp": 0,
                "n1": 0,
                "feasible": 0,
                "cells": 0,
                "raw_units": set(),
            }
            for a in ALPHAS
        }
        for (dec, sb), cr in sorted(cells.items()):
            fit = fit_deployed(cr, H1_SET, H0_SET, readout)
            if fit is None:
                continue
            cal_h0 = score_deployed(cr, H0_SET, "calibration", "h0", fit)
            t0 = score_deployed(cr, H0_SET, "test", "h0", fit)
            t1 = score_deployed(cr, H1_SET, "test", "h1", fit)
            if len(cal_h0) < 8 or len(t0) < 8 or len(t1) < 8:
                continue
            entry: dict = {
                "n_cal": int(len(cal_h0)),
                "n_test_h0": int(len(t0)),
                "n_test_h1": int(len(t1)),
            }
            for a in ALPHAS:
                thr = _conformal_threshold(cal_h0, a)
                pooled[a]["cells"] += 1
                if thr is None:
                    entry[f"alpha{a}"] = {"feasible": False}
                    continue
                # Realized rates on the disjoint TEST rows; a score strictly above tau fires.
                fp = int(np.sum(t0 > thr))
                tp = int(np.sum(t1 > thr))
                entry[f"alpha{a}"] = {
                    "feasible": True,
                    "threshold": thr,
                    "fpr_test": fp / len(t0),
                    "fpr_cp95_upper": _cp_upper(fp, len(t0)),
                    "tpr_test": tp / len(t1),
                }
                pooled[a]["fp"] += fp
                pooled[a]["n0"] += len(t0)
                pooled[a]["tp"] += tp
                pooled[a]["n1"] += len(t1)
                pooled[a]["feasible"] += 1
                # Count distinct (decoder, prompt) raw generations behind the pooled TEST-H0 rows.
                pooled[a]["raw_units"].update(
                    (dec, r["prompt"]) for r in cr if r["w"] in H0_SET and r["split"] == "test"
                )
            per_cell[f"{dec}_sb{sb:g}"] = entry
        rows_out: dict = {"per_cell": per_cell, "pooled_rowlevel": {}}
        for a in ALPHAS:
            p = pooled[a]
            rows_out["pooled_rowlevel"][str(a)] = {
                "cells_feasible": f"{p['feasible']}/{p['cells']}",
                "fpr": (p["fp"] / p["n0"]) if p["n0"] else None,
                "fpr_cp95_upper": _cp_upper(p["fp"], p["n0"]) if p["n0"] else None,
                "tpr": (p["tp"] / p["n1"]) if p["n1"] else None,
                "n0": p["n0"],
                "n0_raw_units": len(p["raw_units"]),
            }
        result[readout] = rows_out

    if not any(r["per_cell"] for r in result.values()):
        raise SystemExit("no cell had usable FIT/CAL/TEST pools; refusing empty output")

    out_path = Path(args.out) if args.out else outputs_dir() / "p0" / "cal_split_fpr.json"
    write_json_strict(out_path, {"alphas": list(ALPHAS), "results": result})

    print("\n=== CAL-split conformal thresholds -> held-out TEST FPR (w20 vs 1) ===")
    for readout, r in result.items():
        for a in ALPHAS:
            p = r["pooled_rowlevel"][str(a)]
            if p["fpr"] is None:
                continue
            print(
                f"{RLABEL[readout]:>16} alpha={a:<5} feasible {p['cells_feasible']:>6}  "
                f"pooled FPR {p['fpr']:.4f}  CP95-upper {p['fpr_cp95_upper']:.4f}  "
                f"TPR {p['tpr']:.3f}  (n0={p['n0']})"
            )
    print(
        "\n[cal_split_fpr] conformal guarantee is per-cell/per-draw under exchangeability; "
        "pooled rows are row-level (blur siblings span cells) and stay descriptive."
    )
    print(f"[cal_split_fpr] wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
