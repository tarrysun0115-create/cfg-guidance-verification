"""Selection-free fixed-H1 all-cell rho detection and its cross-cell sqrt(K) curve.

Produces: the rho rows of Table I (Section IV-B) -- pooled (standardized) AUC_1
(``variants.<v>.all_cell.auc1_pooled_fitsign``), macro AUC_1 (``macro_cell_auc``) and descriptive
TPR at 1%/5% pooled TEST-H0 quantiles (``tpr_at_1pct_fpr_k1``, ``tpr_at_5pct_fpr_k1``) for
``fixed_w20`` (w=20 vs 1), ``fixed_w_ge_10`` ({10,14,20} vs 1) and ``low3_vs_high10``
({10,14,20} vs {1,2,3}) -- and the all-cell label-permutation p of Section IV-B
(``auc1_perm_p``, 500 permutations). Its ``fixed_w20.all_cell`` curve (``auc_emp``/``auc_pred``)
is the data of Fig. 2(D). The w=10 vs 3 row comes from ``run`` with another variant.

Inputs: only ``outputs/p0/<decoder>/p0_table.json`` (from run_p0). Pure numpy, CPU only; no
re-inversion.

Outputs: ``outputs/p0/fixed_h1_allcell.json`` (or ``--out``) and the diagnostic figure
``outputs/figures/f5_detector_roc.png`` (not a paper figure).

Usage: python scripts/analysis/fixed_h1_allcell.py [--n-groups 4000] [--n-perm 500] [--min-n 8]

Role as a helper: ``candidate_readouts.py``, ``pair_readout_battery.py`` and several other
scripts reuse ``run``/``collect_fixed_h1_records``/``_summarize``; ``candidate_readouts.py``
checks these signatures with ``inspect``, so parameter names must stay as they are.

Protocol (main detection protocol, Section IV-A/IV-B), per (decoder, sigma_blur) cell:

* pin H1 to a FIXED guidance set (and H0 to w=1 or a low set) instead of a per-cell best w;
* keep EVERY cell (no gate) -> a selection-free, all-cell pool;
* standardize on the FIT-split (``train``) H0 moments and orient each cell by its FIT-split sign
  (``sign(mean(z_h1_fit))``), a 1-bit parameter fit on FIT and evaluated on the disjoint TEST
  split; TEST labels never choose the sign or the w.

It then reports, for all-cell and per-decoder, the pooled single-image TEST AUC (FIT-oriented,
standardized pooling), the macro-cell AUC (mean of per-cell AUCs), a TEST label-permutation p, and
the cross-cell aggregation curve: with-replacement mean-of-K group AUC on TEST (``n_groups``
groups per K, seed 0) against the Gaussian prediction from the FIT separation, plus the log-log
exponent. The best-w favorable-subset pooled AUC is stored alongside as an upper-envelope
reference (``best_w_upper_envelope_reference``); the selection-free AUC is expected to be below it.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from cfgid.detector.empirical import auc, roc_curve
from cfgid.paths import outputs_dir
from cfgid.sweep.neff_scaling import _Cell, _group_rows_by_cell, _load_p0_rows
from cfgid.sweep.pooled_neff_scaling import (
    _standardize,
    collect_alive_cell_records,
    pooled_eval_auc,
    pooled_sqrtk,
)

# Each variant is (H0 guidance set, H1 guidance set). H0 defaults to the canonical w=1 reference;
# low3_vs_high10 uses a realistic low-guidance set (w<=3) vs a high set (w>=10) to show detection
# reads the guidance LEVEL, not merely the presence of amplification (not an artifact of w=1).
DEFAULT_VARIANTS: dict[str, tuple[tuple[int, ...], tuple[int, ...]]] = {
    "fixed_w20": ((1,), (20,)),
    "fixed_w_ge_10": ((1,), (10, 14, 20)),
    "low3_vs_high10": ((1, 2, 3), (10, 14, 20)),
}


def _pool_h1(cell: _Cell, h1_ws: tuple[int, ...], bucket: str) -> np.ndarray:
    """Concatenate the cell's H1 rho_summary over the H1 guidance set, for one split bucket."""
    parts = [np.asarray(cell.h1[w][bucket], dtype=np.float64) for w in h1_ws if w in cell.h1]
    return np.concatenate(parts) if parts else np.empty(0, dtype=np.float64)


def _pool_h0(cell: _Cell, h0_ws: tuple[int, ...], bucket: str) -> np.ndarray:
    """Concatenate the cell's H0 rho_summary over the H0 guidance set for one split bucket.

    w=1 (label 'h0') lives in ``cell.h0``; any low reference w>1 (label 'h1' by w) lives in
    ``cell.h1[w]``. Pooling both lets H0 be a realistic low-guidance set, not only w=1.
    """
    parts: list[np.ndarray] = []
    for w in h0_ws:
        if w == 1:
            parts.append(np.asarray(cell.h0[bucket], dtype=np.float64))
        elif w in cell.h1:
            parts.append(np.asarray(cell.h1[w][bucket], dtype=np.float64))
    return np.concatenate(parts) if parts else np.empty(0, dtype=np.float64)


class FixedH1Record:
    """One cell's FIT-oriented standardized z-scores for a fixed-H1 set (no alive gate)."""

    __slots__ = ("key", "decoder", "sign", "zf0", "zf1", "zt0", "zt1")

    def __init__(
        self,
        key: str,
        decoder: str,
        sign: float,
        zf0: np.ndarray,
        zf1: np.ndarray,
        zt0: np.ndarray,
        zt1: np.ndarray,
    ) -> None:
        self.key = key
        self.decoder = decoder
        self.sign = sign
        self.zf0 = zf0
        self.zf1 = zf1
        self.zt0 = zt0
        self.zt1 = zt1

    def cell_test_auc(self) -> float:
        """Single-image TEST AUC for this cell, oriented by its FIT sign."""
        scores = self.sign * np.concatenate([self.zt1, self.zt0])
        labels = np.concatenate([np.ones(self.zt1.size, int), np.zeros(self.zt0.size, int)])
        return float(auc(scores, labels))


def collect_fixed_h1_records(
    cells: dict[str, _Cell], h1_ws: tuple[int, ...], *, min_n: int = 8,
    h0_ws: tuple[int, ...] = (1,),
) -> list[FixedH1Record]:
    """Standardize each cell on FIT-H0, orient by the FIT sign, keep ALL cells (no alive gate).

    A cell is included iff it has >= ``min_n`` fit-H0, fit-H1, test-H0, test-H1 samples for the
    requested H0/H1 sets and a finite FIT-H0 std. The sign is the FIT shift sign; the TEST split is
    held out for evaluation. No test-split selection and no best-w: this is the selection-free pool.
    ``h0_ws`` defaults to the canonical (1,) reference; a wider set makes H0 a low-guidance band
    pooled the same way as H1.
    """
    records: list[FixedH1Record] = []
    for key in sorted(cells):
        cell = cells[key]
        h0_fit = _pool_h0(cell, h0_ws, "fit")
        h0_test = _pool_h0(cell, h0_ws, "test")
        h1_fit = _pool_h1(cell, h1_ws, "fit")
        h1_test = _pool_h1(cell, h1_ws, "test")
        if min(h0_fit.size, h0_test.size, h1_fit.size, h1_test.size) < min_n:
            continue
        m0, s0 = float(np.mean(h0_fit)), float(np.std(h0_fit, ddof=1))
        if not (np.isfinite(s0) and s0 > 0.0):
            continue
        zf1 = _standardize(h1_fit, m0, s0)
        sign = float(np.sign(np.mean(zf1)))  # FIT-split sign (label-free w.r.t. EVAL)
        if sign == 0.0:
            continue
        records.append(
            FixedH1Record(
                key=key,
                decoder=cell.decoder,
                sign=sign,
                zf0=_standardize(h0_fit, m0, s0),
                zf1=zf1,
                zt0=_standardize(h0_test, m0, s0),
                zt1=_standardize(h1_test, m0, s0),
            )
        )
    return records


def _aligned_pools(
    records: list[FixedH1Record],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Sign-align each cell's fit/test z to a common positive shift and concatenate."""
    zf0 = np.concatenate([r.sign * r.zf0 for r in records])
    zf1 = np.concatenate([r.sign * r.zf1 for r in records])
    zt0 = np.concatenate([r.sign * r.zt0 for r in records])
    zt1 = np.concatenate([r.sign * r.zt1 for r in records])
    return zf0, zf1, zt0, zt1


def _tpr_at_fpr(h0: np.ndarray, h1: np.ndarray, alpha: float) -> float:
    """One-sided TPR at target FPR alpha: thr = (1-alpha) quantile of H0, TPR = P(H1>thr)."""
    if h0.size == 0 or h1.size == 0:
        return float("nan")
    thr = float(np.quantile(h0, 1.0 - alpha))
    return float(np.mean(h1 > thr))


def _summarize(
    records: list[FixedH1Record], *, k_grid: tuple[int, ...], n_groups: int, n_perm: int, seed: int
) -> dict:
    """Pooled single-image AUC + macro-cell AUC + sqrt(K) law for a set of cell records."""
    if not records:
        return {"n_cells": 0}
    zf0, zf1, zt0, zt1 = _aligned_pools(records)
    res = pooled_sqrtk(
        zf0,
        zf1,
        zt0,
        zt1,
        alive_cells=tuple(r.key for r in records),
        k_grid=k_grid,
        n_groups=n_groups,
        n_perm=n_perm,
        seed=seed,
    )
    macro = float(np.mean([r.cell_test_auc() for r in records]))
    return {
        "n_cells": len(records),
        "cells": [r.key for r in records],
        "n_pool_h0": int(zt0.size),
        "n_pool_h1": int(zt1.size),
        "auc1_pooled_fitsign": res.auc1_pool_test,
        "auc1_perm_p": res.auc1_perm_p,
        "macro_cell_auc": macro,
        "d_pool_fit": res.d_pool_fit,
        "k_grid": list(res.k_grid),
        "auc_emp": list(res.auc_emp),
        "auc_pred": list(res.auc_pred),
        "exponent": res.exponent,
        "rise_emp": res.rise_emp,
        "rise_perm_p": res.rise_perm_p,
        "tpr_at_1pct_fpr_k1": _tpr_at_fpr(zt0, zt1, 0.01),
        "tpr_at_5pct_fpr_k1": _tpr_at_fpr(zt0, zt1, 0.05),
    }


def _auc_roc(h0: np.ndarray, h1: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """AUC + ROC (fpr, tpr) for one H1-vs-H0 score pair (labels 1 for H1, 0 for H0)."""
    scores = np.concatenate([h1, h0])
    labels = np.concatenate([np.ones(h1.size, dtype=int), np.zeros(h0.size, dtype=int)])
    fpr, tpr = roc_curve(scores, labels)
    return float(auc(scores, labels)), fpr, tpr


def _bootstrap_group_scores(
    h0: np.ndarray, h1: np.ndarray, k: int, *, n_groups: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Mean-of-K bootstrap scores using the same draw order as ``pooled_sqrtk``."""
    g0 = h0[rng.integers(0, h0.size, size=(n_groups, k))].mean(axis=1)
    g1 = h1[rng.integers(0, h1.size, size=(n_groups, k))].mean(axis=1)
    return g0, g1


def _draw_roc_figure(
    records: list[FixedH1Record],
    *,
    k_grid: tuple[int, ...] = (1, 2, 4, 8, 16, 32),
    k_group: int = 16,
    n_groups: int = 4000,
    seed: int = 0,
) -> None:
    """Selection-free fixed-H1 all-cell detector ROC figure (diagnostic; not a paper figure).

    Single-image TEST ROC and the K-image group ROC (the cross-cell sqrt(K) aggregate), both on
    the sign-aligned all-cell pool. The group ROC averages random K-image groups of the held-out
    TEST scores, matching the sqrt(K) curve's grouping; reproducible (seeded RNG, no
    re-inversion).
    """
    from cfgid import viz

    plt = viz.pyplot()
    if plt is None or not records:
        return
    _zf0, _zf1, zt0, zt1 = _aligned_pools(records)
    a1, f1, t1 = _auc_roc(zt0, zt1)
    rng = np.random.default_rng(seed)
    g0 = np.empty(0, dtype=np.float64)
    g1 = np.empty(0, dtype=np.float64)
    for k in k_grid:
        g0, g1 = _bootstrap_group_scores(zt0, zt1, k, n_groups=n_groups, rng=rng)
        if k == k_group:
            break
    else:
        raise ValueError(f"k_group={k_group} is not present in k_grid={k_grid!r}")
    a16, f16, t16 = _auc_roc(g0, g1)
    print(
        f"[fixed_h1_allcell] fixed-H1 ROC: single-image AUC={a1:.4f}, "
        f"K={k_group} group AUC={a16:.4f}"
    )
    fig, ax = plt.subplots(figsize=(5.2, 5.2))
    ax.plot([0.0, 1.0], [0.0, 1.0], color="#bbbbbb", lw=1.0, ls="--", label="chance")
    ax.plot(f1, t1, color="#555555", lw=2.0, label=f"single image (AUC={a1:.3f})")
    ax.plot(
        f16,
        t16,
        color="#2a9d8f",
        lw=2.8,
        solid_capstyle="round",
        label=f"$K={k_group}$ group bootstrap (AUC={a16:.3f})",
    )
    ax.set_xlabel("false positive rate")
    ax.set_ylabel("true positive rate")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_title("Selection-free all-cell detector ROC")
    ax.legend(loc="lower right", fontsize=9)
    viz.save_figure(fig, "f5_detector_roc")


def run(
    features_root: Path,
    out_path: Path,
    *,
    variants: dict[str, tuple[int, ...]] | None = None,
    k_grid: tuple[int, ...] = (1, 2, 4, 8, 16, 32),
    n_groups: int = 4000,
    n_perm: int = 500,
    min_n: int = 8,
    seed: int = 0,
) -> int:
    variants = variants or DEFAULT_VARIANTS
    rows: list[dict] = []
    for table_path in sorted(features_root.glob("*/p0_table.json")):
        rows.extend(_load_p0_rows(table_path))
    if not rows:
        print(f"[skip] no <decoder>/p0_table.json under {features_root}")
        return 0
    cells = _group_rows_by_cell(rows)
    decoders = sorted({c.decoder for c in cells.values()})

    out: dict = {
        "analysis": "fixed_h1_allcell",
        "decoders": decoders,
        "n_cells_total": len(cells),
        "k_grid": list(k_grid),
        "variants": {},
        "note": (
            "Selection-free: H1 pinned to fixed guidance (no best-w), no alive gate, sign fit on "
            "the FIT (train) split, evaluated on the disjoint TEST split. auc1_pooled_fitsign is "
            "the pooled single-image TEST AUC under the FIT sign; macro_cell_auc is the mean of "
            "per-cell oriented AUCs. Below the best-w upper-envelope 0.758 by construction."
        ),
    }
    for name, (h0_ws, h1_ws) in variants.items():
        recs = collect_fixed_h1_records(cells, h1_ws, min_n=min_n, h0_ws=h0_ws)
        if name == "fixed_w20":
            _draw_roc_figure(recs, n_groups=n_groups, seed=seed)
        per_dec = {}
        for dec in decoders:
            drecs = [r for r in recs if r.decoder == dec]
            per_dec[dec] = _summarize(
                drecs, k_grid=k_grid, n_groups=n_groups, n_perm=n_perm, seed=seed
            )
        out["variants"][name] = {
            "h0_ws": list(h0_ws),
            "h1_ws": list(h1_ws),
            "all_cell": _summarize(
                recs, k_grid=k_grid, n_groups=n_groups, n_perm=n_perm, seed=seed
            ),
            "per_decoder": per_dec,
        }

    # Upper-envelope reference: best-w favorable-subset pooled AUC (the 0.758 construction).
    alive = collect_alive_cell_records(cells, alive_t=1.5)
    if alive:
        ue_auc = pooled_eval_auc(alive, [r.sign for r in alive])
        out["best_w_upper_envelope_reference"] = {
            "n_alive_cells": len(alive),
            "auc1_pooled_bestw_alive": ue_auc,
            "note": "per-cell best-w over the alive pool (the upper-envelope 0.758 construction)",
        }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, allow_nan=False), encoding="utf-8")

    print(
        f"\nfixed-H1 all-cell selection-free detection ({len(cells)} cells, {len(decoders)} dec):"
    )
    for name in out["variants"]:
        ac = out["variants"][name]["all_cell"]
        if ac.get("n_cells", 0) == 0:
            print(f"  [{name}] no cells")
            continue
        exp = "n/a" if ac["exponent"] is None else f"{ac['exponent']:+.3f}"
        print(
            f"  [{name}] all-cell ({ac['n_cells']} cells): "
            f"AUC1={ac['auc1_pooled_fitsign']:.3f} (perm p={ac['auc1_perm_p']:.4f})  "
            f"macro={ac['macro_cell_auc']:.3f}  "
            f"K=[{' '.join(f'{a:.3f}' for a in ac['auc_emp'])}]  exp={exp}"
        )
    if "best_w_upper_envelope_reference" in out:
        ue = out["best_w_upper_envelope_reference"]
        print(
            f"  [reference] best-w alive upper-envelope: "
            f"AUC1={ue['auc1_pooled_bestw_alive']:.3f} ({ue['n_alive_cells']} alive cells)"
        )
    print(f"-> {out_path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="fixed_h1_allcell")
    ap.add_argument(
        "--features-root", default=None, help="dir of <decoder>/p0_table.json (def: outputs/p0)"
    )
    ap.add_argument(
        "--out", default=None, help="output JSON (def: outputs/p0/fixed_h1_allcell.json)"
    )
    ap.add_argument("--n-groups", type=int, default=4000)
    ap.add_argument("--n-perm", type=int, default=500)
    ap.add_argument("--min-n", type=int, default=8)
    args = ap.parse_args(argv)
    features_root = Path(args.features_root) if args.features_root else (outputs_dir() / "p0")
    out_path = Path(args.out) if args.out else (outputs_dir() / "p0" / "fixed_h1_allcell.json")
    return run(
        features_root, out_path, n_groups=args.n_groups, n_perm=args.n_perm, min_n=args.min_n
    )


if __name__ == "__main__":
    raise SystemExit(main())
