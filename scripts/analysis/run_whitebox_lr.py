"""Per-cell controllable-FPR white-box likelihood-ratio detector on the main-grid rho features.

Produces: no printed paper number directly (shared helper plus a standalone diagnostic).
``pair_readout_battery.py`` calls ``run_whitebox_lr`` once per candidate readout
(``outputs/p0/pair_readouts/<cand>/whitebox/``); the ``whitebox_lr.json`` written here is a
diagnostic and is not printed in the paper. The detector is ``cfgid.detector.whitebox_lr``
(Gaussian-location LLR of Eq. (23), N_eff-image mean of Eq. (14)).

Inputs: only ``outputs/p0/<decoder>/p0_table.json`` (from run_p0). Pure numpy, CPU only; no
re-inversion.

Outputs: ``outputs/p0/whitebox_lr.json`` (or ``--out-dir``).

Usage: python scripts/analysis/run_whitebox_lr.py --features-root outputs/p0

For each cell it picks, on FIT only, the guidance w whose single-image |AUC - 0.5| clears
``--alive-floor`` - 0.5 (0.10 by default) and lies closest to the target AUC; it then anchors the
signal shift
``mu = mean(rho|H1) - mean(rho|H0)`` on the FIT (``train``) split, sets the FPR threshold from the
FIT H0 distribution (no H1 labels in the threshold), and on the held-out TEST split measures (a)
the realised FPR against the target ``alpha`` and (b) the power against the analytic prediction
Phi(sqrt(N)*mu/sqrt(v) - z_alpha), across alpha in {0.01, 0.05} and N_eff in {1, 4, 16}
(``--n-groups`` 8,000 Monte-Carlo groups, seed 0).

mu is MEASURED on FIT H1 (labeled), so this checks the detector's FPR control and power prediction;
it does not supply a label-free mu.

Selection: the w is chosen with ``sign_aware=True`` (on the signal MAGNITUDE |AUC-0.5|), so cells
where guidance lowers rho (AUC<0.5) are kept, unlike the one-sided selection in
``cfgid.sweep.neff_scaling``; the detector direction ``sign`` is anchored on the FIT mean shift and
the test runs on the aligned statistic sign*rho. Cells whose rank (AUC) direction disagrees with the
mean-shift sign are skipped, since the mean-LR detector is ill-defined there.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from cfgid.detector.empirical import auc
from cfgid.detector.whitebox_lr import evaluate_cell
from cfgid.paths import outputs_dir
from cfgid.sweep.neff_scaling import (
    DEFAULT_TARGET_AUC1,
    _alive_w_for_cell,
    _group_rows_by_cell,
    _load_p0_rows,
)


def _sample_var_ddof1(scores: np.ndarray) -> float:
    """Sample variance with the same ddof as the LR threshold fit."""
    return float(np.var(np.asarray(scores, dtype=np.float64).reshape(-1), ddof=1))


def _check_nonnegative_finite(name: str, value: float) -> None:
    if not (np.isfinite(value) and value >= 0.0):
        raise ValueError(f"{name} must be non-negative and finite, got {value}")


def run_whitebox_lr(
    features_root: Path,
    out_dir: Path,
    *,
    alphas: tuple[float, ...] = (0.01, 0.05),
    n_eff_grid: tuple[int, ...] = (1, 4, 16),
    alive_floor: float = 0.60,
    target_auc1: float = DEFAULT_TARGET_AUC1,
    n_groups: int = 8000,
    fpr_tol: float = 0.01,
    power_tol: float = 0.05,
    seed: int = 0,
) -> int:
    _check_nonnegative_finite("fpr_tol", fpr_tol)
    _check_nonnegative_finite("power_tol", power_tol)
    rows: list[dict] = []
    for table_path in sorted(features_root.glob("*/p0_table.json")):
        # strict loader (fail-loud on NaN/Inf/bareword), matching neff_scaling / analyze / etex --
        # this script must not parse a corrupted artifact at a weaker trust level than its siblings.
        rows.extend(_load_p0_rows(table_path))
    if not rows:
        print(f"[skip] no p0_table.json under {features_root}")
        return 0
    cells = _group_rows_by_cell(rows)
    records: list[dict] = []
    for key in sorted(cells):
        cell = cells[key]
        # sign_aware: keep inverted-but-strong cells (|AUC-0.5|), since guidance LOWERS rho in some
        # cells; the direction is recovered below and the detector aligns to it.
        w = _alive_w_for_cell(
            cell, alive_floor=alive_floor, target_auc1=target_auc1, sign_aware=True
        )
        if w is None:
            print(
                f"[skip] {key}: no alive w (single-image |fit AUC-0.5| < {abs(alive_floor - 0.5)})"
            )
            continue
        h0_fit = np.asarray(cell.h0["fit"], dtype=np.float64)
        h1_fit = np.asarray(cell.h1[w]["fit"], dtype=np.float64)
        h0_test = np.asarray(cell.h0["test"], dtype=np.float64)
        h1_test = np.asarray(cell.h1[w]["test"], dtype=np.float64)
        if h0_fit.size < 2 or h1_fit.size < 2 or h0_test.size < 4 or h1_test.size < 4:
            print(f"[skip] {key}/w={w}: too few fit/test images")
            continue
        h0_fit_var = _sample_var_ddof1(h0_fit)
        if not (np.isfinite(h0_fit_var) and h0_fit_var > 0.0):
            print(f"[skip] {key}/w={w}: non-positive H0 fit variance {h0_fit_var:.4g}")
            continue
        raw_mu = float(np.mean(h1_fit) - np.mean(h0_fit))  # train anchor (labeled), signed
        if not np.isfinite(raw_mu) or raw_mu == 0.0:
            # No usable channel direction on the fit split -> can't aim the one-sided test.
            print(f"[skip] {key}/w={w}: train signal shift mu={raw_mu:.4g} is zero/non-finite")
            continue
        # SIGN-AWARE: the response sign is decoder-specific (CFG LOWERS rho in some cells). Anchor
        # the direction on the fit split and run the detector on the aligned statistic sign*rho, so
        # a CFG-lowers-rho cell is tested in its own tail instead of being skipped.
        sign = float(np.sign(raw_mu))
        # The cell was SELECTED on the rank |AUC-0.5|, but the mean-LR detector aligns on the mean
        # shift. rho is a heavy-tailed log-ratio, so a few outliers can flip the mean-shift sign vs
        # the rank/AUC direction; the mean detector would then point at the WRONG tail and read
        # power on a cell selected as strong. Skip such cells (the mean-LR is ill-defined there)
        # instead of silently mis-aiming the test.
        auc_fit = float(
            auc(
                np.concatenate([h1_fit, h0_fit]),
                np.concatenate([np.ones(h1_fit.size, int), np.zeros(h0_fit.size, int)]),
            )
        )
        if float(np.sign(auc_fit - 0.5)) != sign:
            print(
                f"[skip] {key}/w={w}: AUC-direction (AUC={auc_fit:.3f}) disagrees with the "
                f"mean-shift sign ({sign:+.0f}) -- heavy-tailed cell, mean-LR detector ill-defined"
            )
            continue
        mu = abs(raw_mu)  # |aligned shift| >= 0
        mu_test = float(np.mean(h1_test) - np.mean(h0_test))  # signed test shift
        test_shift_sign_flip = bool(np.isfinite(mu_test) and np.sign(mu_test) != sign)
        for alpha in alphas:
            for n_eff in n_eff_grid:
                res = evaluate_cell(
                    cell=key,
                    w=w,
                    h0_fit=h0_fit,
                    h0_test=h0_test,
                    h1_test=h1_test,
                    mu_predictive=mu,
                    sign=sign,
                    alpha=alpha,
                    n_eff=n_eff,
                    n_groups=n_groups,
                    fpr_tol=fpr_tol,
                    power_tol=power_tol,
                    seed=seed,
                )
                fpr_abs_err = abs(res.fpr_emp - res.alpha)
                power_abs_err = abs(res.tpr_emp - res.power_analytic)
                records.append(
                    {
                        "cell": res.cell,
                        "w": res.w,
                        "alpha": res.alpha,
                        "n_eff": res.n_eff,
                        "n_groups": n_groups,
                        "n_h0_fit": int(h0_fit.size),
                        "n_h0_test": int(h0_test.size),
                        "n_h1_fit": int(h1_fit.size),
                        "n_h1_test": int(h1_test.size),
                        "sign": res.sign,
                        "mu_train_signed": raw_mu,
                        "mu_aligned": res.mu_predictive,
                        "mu_test": mu_test,
                        "test_shift_sign_flip": test_shift_sign_flip,
                        "threshold": res.threshold,
                        "fpr_emp": res.fpr_emp,
                        "fpr_abs_err": fpr_abs_err,
                        "fpr_tol": fpr_tol,
                        "tpr_emp": res.tpr_emp,
                        "power_analytic": res.power_analytic,
                        "power_abs_err": power_abs_err,
                        "power_tol": power_tol,
                        "fpr_resolved": res.fpr_resolved,
                        "fpr_controlled": res.fpr_controlled,
                        "power_matches": res.power_matches,
                    }
                )
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "whitebox_lr.json").write_text(
        json.dumps(records, indent=2, allow_nan=False), encoding="utf-8"
    )
    n_fpr_ok = sum(r["fpr_controlled"] for r in records)
    n_unresolved = sum(not r["fpr_resolved"] for r in records)
    n_pow_ok = sum(r["power_matches"] for r in records)
    n = len(records)
    print(
        f"\nwhite-box LR: {n} (cell,alpha,N_eff) evals; "
        f"FPR-controlled {n_fpr_ok}/{n} ({n_unresolved} unresolved at this pool size), "
        f"power matches analytic {n_pow_ok}/{n}"
    )
    for r in records:
        fp = (
            "FPR-?? "
            if not r["fpr_resolved"]
            else ("FPR-ok " if r["fpr_controlled"] else "FPR-no ")
        )
        pw = "pow-ok" if r["power_matches"] else "pow-no"
        sg = "+" if r["sign"] > 0 else "-"
        print(
            f"  [{fp}{pw}] {r['cell']:14s} w={r['w']:2d} sign={sg} a={r['alpha']:.2f} "
            f"N={r['n_eff']:2d}  FPR={r['fpr_emp']:.3f} (err={r['fpr_abs_err']:.3f}, "
            f"tol={r['fpr_tol']:.3f})  pow={r['tpr_emp']:.3f} vs {r['power_analytic']:.3f}"
            f"{'  TEST-SIGN-FLIP' if r['test_shift_sign_flip'] else ''}"
        )
    print(f"-> {out_dir / 'whitebox_lr.json'}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="run_whitebox_lr")
    ap.add_argument(
        "--features-root",
        default=None,
        help="dir of <decoder>/p0_table.json (default: outputs/p0, matching cfgid.sweep.analyze)",
    )
    ap.add_argument("--out-dir", default=None, help="default: outputs/p0")
    ap.add_argument("--alive-floor", type=float, default=0.60)
    ap.add_argument("--n-groups", type=int, default=8000)
    ap.add_argument("--fpr-tol", type=float, default=0.01)
    ap.add_argument("--power-tol", type=float, default=0.05)
    args = ap.parse_args(argv)
    # run_p0 writes outputs/p0/<decoder>/p0_table.json and analyze reads outputs/p0; match it.
    features_root = Path(args.features_root) if args.features_root else (outputs_dir() / "p0")
    out_dir = Path(args.out_dir) if args.out_dir else (outputs_dir() / "p0")
    return run_whitebox_lr(
        features_root,
        out_dir,
        alive_floor=args.alive_floor,
        n_groups=args.n_groups,
        fpr_tol=args.fpr_tol,
        power_tol=args.power_tol,
    )


if __name__ == "__main__":
    raise SystemExit(main())
