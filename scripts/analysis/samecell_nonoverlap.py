"""Recompute the Table I aggregation audit, with three aggregation constructions side by side.

For several guidance contrasts and three readouts (rho, audit pair z-sum, Pair-Ridge-Audit) on
the selection-free all-cell main-grid pool, this reports AUC_1, macro-cell AUC, the sqrt-K
exponent, K=16 group AUC, and TPR@{1,5}%. The cross-cell exponent and the with-replacement K=16
group AUC mix decoder x blur cells and repeat images, so they are neither same-configuration nor
independent; the script therefore prints THREE aggregation constructions:
  * within_cell   -- non-overlapping groups inside each cell, macro over cells (same-config)
  * pool_nonoverlap -- non-overlapping groups over the concatenated all-cell pool (distinct images)
  * pool_withrepl -- the cross-cell with-replacement construction (for the delta)
Non-overlapping points average --n-boot random partitions; with-replacement points average
--n-boot // 4 draws of 1,000 groups; all draws share one generator (--seed). The within-cell K=16
value also gets a 95% cell-cluster bootstrap CI (4,000 resamples of the per-cell values, rng
seed 1) and its worst cell. The rho single-image quantities reproduce
the paper directly; audit pair quantities need not equal the deployed-pipeline values.

rho is read from p0_table.json (rho_summary). The pair coordinate m is recomputed from the residual
feature store (residual_v1.dat + compact_e_tex) with the frozen p0 mask/winsor; pair z-sum is an
audit reconstruction and pair ridge-LDA uses FIT-H0 sample covariance plus ``1e-6 I``. It is not the
deployed Ledoit--Wolf ``T_pair-LDA``. Pure reanalysis, no re-inversion. Best-w (alive-gate) rows
are not reproduced here.

Produces: Table I, rho same-cell AUC_16 [95% cell-cluster CI] for w in {10,14,20} vs 1 and w=20
vs 1 (``rho|hi_vs_1`` and ``rho|w20_vs_1``: ``k16_within_cell``, ``k16_within_ci``); the other
rows are audit cross-checks (the pair same-cell column of Table I comes from
deployed_within_contrasts.py). Monte-Carlo partition averages can move in the third decimal
between independent runs.
Inputs: main-grid p0_table.json and the server-only residual feature store (decoders without
residual_v1.dat are skipped).
Outputs: outputs/samecell_nonoverlap/samecell_nonoverlap.json.
Usage:
  uv run --no-sync python scripts/analysis/samecell_nonoverlap.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import norm

from cfgid.channel.rho import etex_keep_mask, winsorize
from cfgid.detector.empirical import auc
from cfgid.featurestore import read_compact, read_residual_v1
from cfgid.paths import outputs_dir
from cfgid.sweep.neff_scaling import _load_p0_rows

MASK_Q = 0.2
WINSOR = (0.01, 0.99)
GROUP_K = [1, 2, 4, 8, 16]
RIDGE = 1e-6

# Table I fixed-H1 contrasts: (name, H1 guidance set, H0 guidance set).
CONTRASTS = [
    ("w5_vs_3", {5.0}, {3.0}),
    ("w10_vs_3", {10.0}, {3.0}),
    ("w20_vs_1", {20.0}, {1.0}),
    ("hi_vs_1", {10.0, 14.0, 20.0}, {1.0}),
    ("hi_vs_lo", {10.0, 14.0, 20.0}, {1.0, 2.0, 3.0}),
]
READOUTS = ["rho", "zsum", "lda"]
RLABEL = {"rho": "rho", "zsum": "pair z-sum audit", "lda": "pair ridge-LDA audit"}


def _m_from_residual(residual_i: np.ndarray, e_tex_i: np.ndarray) -> float:
    delta = np.linalg.norm(residual_i, axis=1)
    keep = etex_keep_mask(e_tex_i, MASK_Q)
    kept = delta[keep] if keep.any() else delta
    return float(np.mean(winsorize(kept.astype(np.float32), WINSOR[0], WINSOR[1])))


def _two_sample_auc(h0: np.ndarray, h1: np.ndarray) -> float:
    scores = np.concatenate([h0, h1])
    labels = np.concatenate([np.zeros(len(h0)), np.ones(len(h1))])
    return float(auc(scores, labels))


def _deflection(a: float) -> float:
    return float(np.sqrt(2.0) * norm.ppf(min(max(a, 0.5001), 0.9999)))


def _fit_exponent(ks: list[int], aucs: list[float]) -> float | None:
    xs = [k for k, a in zip(ks, aucs, strict=True) if a is not None and a > 0.5]
    ds = [_deflection(a) for k, a in zip(ks, aucs, strict=True) if a is not None and a > 0.5]
    if len(xs) < 2:
        return None
    return float(np.polyfit(np.log(xs), np.log(ds), 1)[0])


def _tpr_at_fpr(h0: np.ndarray, h1: np.ndarray, alpha: float) -> float:
    thr = np.quantile(h0, 1.0 - alpha)
    return float(np.mean(h1 > thr))


def _cell_test_scores(
    cell_rows: list[dict], h1_set: set, h0_set: set, readout: str
) -> tuple | None:
    """FIT-orient/standardize (+ LDA weight) on FIT, return (test_h0, test_h1) score arrays."""

    def pop(w_set: set, split: str, key: str) -> np.ndarray:
        return np.array(
            [r[key] for r in cell_rows if r["w"] in w_set and r["split"] == split], dtype=np.float64
        )

    # Per coordinate: FIT sign (sign of the FIT H1-H0 mean shift) and FIT-H0 mean/std.
    fit = {}
    for key in ("rho_summary", "m"):
        h0 = pop(h0_set, "train", key)
        h1 = pop(h1_set, "train", key)
        if len(h0) < 4 or len(h1) < 2:
            return None
        sign = 1.0 if h1.mean() >= h0.mean() else -1.0
        sd = float(h0.std())
        fit[key] = {"sign": sign, "mu": float(h0.mean()), "sd": sd if sd > 0 else 1.0}

    def z(w_set: set, split: str, key: str) -> np.ndarray:
        f = fit[key]
        return f["sign"] * (pop(w_set, split, key) - f["mu"]) / f["sd"]

    if readout == "rho":
        return z(h0_set, "test", "rho_summary"), z(h1_set, "test", "rho_summary")
    if readout == "zsum":
        return (
            z(h0_set, "test", "rho_summary") + z(h0_set, "test", "m"),
            z(h1_set, "test", "rho_summary") + z(h1_set, "test", "m"),
        )
    # Audit ridge-LDA on FIT-standardized (rho, m); this is not deployed Ledoit--Wolf.
    x0 = np.vstack([z(h0_set, "train", "rho_summary"), z(h0_set, "train", "m")]).T
    x1 = np.vstack([z(h1_set, "train", "rho_summary"), z(h1_set, "train", "m")]).T
    if len(x0) < 4 or len(x1) < 2:
        return None
    w_lda = np.linalg.solve(np.cov(x0.T) + RIDGE * np.eye(2), x1.mean(axis=0) - x0.mean(axis=0))
    t0 = np.vstack([z(h0_set, "test", "rho_summary"), z(h0_set, "test", "m")]).T @ w_lda
    t1 = np.vstack([z(h1_set, "test", "rho_summary"), z(h1_set, "test", "m")]).T @ w_lda
    return t0, t1


def _group_auc_nonoverlap(
    h0: np.ndarray, h1: np.ndarray, k: int, n_boot: int, rng: np.random.Generator
) -> float | None:
    g0, g1 = len(h0) // k, len(h1) // k
    if min(g0, g1) < 2:
        return None
    vals = []
    for _ in range(n_boot):
        p0 = rng.permutation(len(h0))[: g0 * k].reshape(g0, k)
        p1 = rng.permutation(len(h1))[: g1 * k].reshape(g1, k)
        vals.append(_two_sample_auc(h0[p0].mean(1), h1[p1].mean(1)))
    return float(np.mean(vals))


def _group_auc_withrepl(
    h0: np.ndarray,
    h1: np.ndarray,
    k: int,
    n_boot: int,
    rng: np.random.Generator,
    n_groups: int = 1000,
) -> float:
    vals = []
    for _ in range(n_boot // 4):
        i0 = rng.integers(0, len(h0), size=(n_groups, k))
        i1 = rng.integers(0, len(h1), size=(n_groups, k))
        vals.append(_two_sample_auc(h0[i0].mean(1), h1[i1].mean(1)))
    return float(np.mean(vals))


def _row(cells_scores: list[tuple], n_boot: int, rng: np.random.Generator) -> dict:
    """Given each cell's (test_h0, test_h1), compute one audit row (all columns, all modes)."""
    pooled_h0 = np.concatenate([c[0] for c in cells_scores])
    pooled_h1 = np.concatenate([c[1] for c in cells_scores])
    auc1 = _two_sample_auc(pooled_h0, pooled_h1)
    macro = float(np.mean([_two_sample_auc(h0, h1) for h0, h1 in cells_scores]))

    within, poolno, poolwr = {}, {}, {}
    per16: list[float] = []
    for k in GROUP_K:
        per = [_group_auc_nonoverlap(h0, h1, k, n_boot, rng) for h0, h1 in cells_scores]
        per = [v for v in per if v is not None]
        within[k] = float(np.mean(per)) if per else None
        if k == 16:
            per16 = per
        poolno[k] = _group_auc_nonoverlap(pooled_h0, pooled_h1, k, n_boot, rng)
        poolwr[k] = _group_auc_withrepl(pooled_h0, pooled_h1, k, n_boot, rng)

    # cell-cluster bootstrap 95% CI + worst-cell on the K=16 within-cell per-cell values
    # (resample the per-cell AUCs with replacement -- the standard CI for a macro-over-cells
    # estimand).
    ci16: list[float] | None = None
    worst16: float | None = None
    if per16:
        arr = np.array(per16, dtype=np.float64)
        rb = np.random.default_rng(1)
        boot = np.array([rb.choice(arr, size=arr.size, replace=True).mean() for _ in range(4000)])
        ci16 = [float(x) for x in np.percentile(boot, [2.5, 97.5])]
        worst16 = float(arr.min())

    return {
        "auc1": auc1,
        "macro_cell": macro,
        "tpr@1%": _tpr_at_fpr(pooled_h0, pooled_h1, 0.01),
        "tpr@5%": _tpr_at_fpr(pooled_h0, pooled_h1, 0.05),
        "k16_within_cell": within[16],
        "k16_within_ci": ci16,
        "k16_within_worst": worst16,
        "k16_pool_nonoverlap": poolno[16],
        "k16_pool_withrepl": poolwr[16],
        "exp_within_cell": _fit_exponent(GROUP_K, [within[k] for k in GROUP_K]),
        "exp_pool_withrepl": _fit_exponent(GROUP_K, [poolwr[k] for k in GROUP_K]),
        "curve_within_cell": within,
        "curve_pool_withrepl": poolwr,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default=None, help="dir of <decoder>/ p0 outputs (def outputs/p0)")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    rng = np.random.default_rng(args.seed)
    root = Path(args.root) if args.root else outputs_dir() / "p0"

    all_rows: list[dict] = []
    for dec_dir in sorted(p for p in root.iterdir() if (p / "p0_table.json").is_file()):
        rows = _load_p0_rows(dec_dir / "p0_table.json")
        if not (dec_dir / "residual_v1.dat").is_file():
            print(f"[skip] {dec_dir.name}: no residual_v1.dat (m needs the feature store)")
            continue
        residual, _ = read_residual_v1(dec_dir)
        batch, _ = read_compact(dec_dir)
        if residual.shape[0] != len(rows) or batch.e_tex.shape[0] != len(rows):
            raise SystemExit(f"{dec_dir.name}: store/table length mismatch")
        for i, r in enumerate(rows):
            r["m"] = _m_from_residual(residual[i], batch.e_tex[i])
            r["decoder"] = dec_dir.name
            r["w"] = float(r["w"])
        all_rows.extend(rows)
        print(f"[load] {dec_dir.name}: {len(rows)} rows, m computed")

    cells: dict[tuple[str, int], list[dict]] = {}
    for r in all_rows:
        cells.setdefault((r["decoder"], int(r["sigma_blur"])), []).append(r)

    table = {}
    for readout in READOUTS:
        for cname, h1_set, h0_set in CONTRASTS:
            cs = [_cell_test_scores(cr, h1_set, h0_set, readout) for cr in cells.values()]
            cs = [c for c in cs if c is not None and len(c[0]) and len(c[1])]
            if len(cs) < 2:
                continue
            table[f"{readout}|{cname}"] = {"n_cells": len(cs), **_row(cs, args.n_boot, rng)}

    out_dir = Path(args.out) if args.out else outputs_dir() / "samecell_nonoverlap"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "samecell_nonoverlap.json").write_text(json.dumps(table, indent=2))

    print("\n=== Table I audit recomputed (within-cell vs cross-cell with-replacement) ===")
    hdr = (
        f"{'readout':>12} {'contrast':>9} | {'AUC1':>6} {'macro':>6} | "
        f"{'K16 within':>10} {'K16 poolNO':>10} {'K16 poolWR':>10} | "
        f"{'exp within':>10} {'exp WR':>8} | {'TPR1%':>6} {'TPR5%':>6}"
    )
    print(hdr)
    print("-" * len(hdr))
    for key, r in table.items():
        ro, cn = key.split("|")

        def s(x, w=6):
            return f"{x:.3f}".rjust(w) if isinstance(x, float) else "  -  ".rjust(w)

        print(
            f"{RLABEL[ro]:>12} {cn:>9} | {s(r['auc1'])} {s(r['macro_cell'])} | "
            f"{s(r['k16_within_cell'], 10)} {s(r['k16_pool_nonoverlap'], 10)} "
            f"{s(r['k16_pool_withrepl'], 10)} | {s(r['exp_within_cell'], 10)} "
            f"{s(r['exp_pool_withrepl'], 8)} | {s(r['tpr@1%'])} {s(r['tpr@5%'])}"
        )
    print("\n=== within-cell K=16: macro [cell-cluster 95% CI]  (worst-cell) ===")
    for key, r in table.items():
        ro, cn = key.split("|")
        ci = r.get("k16_within_ci")
        ci_s = f"[{ci[0]:.3f}, {ci[1]:.3f}]" if ci else "[--, --]"
        wc = r.get("k16_within_worst")
        wc_s = f"{wc:.3f}" if isinstance(wc, float) else "--"
        mac = r.get("k16_within_cell")
        mac_s = f"{mac:.3f}" if isinstance(mac, float) else "--"
        print(f"{RLABEL[ro]:>12} {cn:>9} | {mac_s:>6} {ci_s:>16}   worst-cell {wc_s}")

    print(
        f"\n[samecell] {len(table)} rows / {len(cells)} cells -> "
        f"{out_dir / 'samecell_nonoverlap.json'}"
    )
    print(
        "[samecell] poolWR = cross-cell with-replacement estimand; "
        "within-cell = same-cell non-overlapping estimand."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
