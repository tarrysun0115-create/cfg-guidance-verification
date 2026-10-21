"""Selection-free Table I rows for w=5 vs 3 and w=20 vs 1 (rho, T_pair-zsum, T_pair-LDA).

Computes, for each readout in READOUTS and each contrast in CONTRASTS, the Table I columns on the
production selection-free pools: per cell, rows are standardized on the FIT-H0 rows and oriented
by the FIT sign (``fixed_h1_allcell.collect_fixed_h1_records``), then TEST scores are pooled
over all twenty cells. Per (readout, contrast) it reports pooled AUC1, macro AUC1 (mean of
per-cell TEST AUCs), the cross-cell sqrt(K) exponent point on the unsaturated K points (group
AUC in (1e-4, 1 - 1e-4), K in {1, 2, 4, 8, 16, 32}, 4000 with-replacement groups per K, seed 0),
its matched 95% bootstrap CI (400 class-wise row resamples, the same unsaturated-mask estimator;
``pair_variant_ci_matched._unsat_exponent``), and TPR at 1% and 5% FPR at pooled TEST-H0
quantile thresholds. Per-image readout values come from the virtual tables of
``pair_readout_battery._compute_all_features`` (feature-store read). The same-cell AUC16 column
of these rows comes from deployed_within_contrasts.py (the deployed w=20 vs 1 head; ``*_mainfit``
entries for the weaker contrasts).

Fail-loud gates (before any w=5 vs 3 number is emitted): for every readout, the w=20 contrast
recomputed from the same virtual tables must reproduce the reference
``pair_readouts/fixed_h1_<readout>.json`` pooled AUC1 to 1e-6 and its exponent point to 0.01.

Produces:
    Table I, rows rho w=5 vs 3, T_pair-zsum w=20 vs 1, T_pair-LDA w=5 vs 3 and T_pair-LDA w=20
    vs 1: pooled AUC1, macro AUC1, exponent 95% CI, TPR@1%/5%
    (``weak_contrast_rows.json``: ``readouts/<readout>/<contrast>/{auc1_pooled_fitsign,
    macro_cell_auc,exp_ci95_matched,tpr_at_1pct_fpr_k1,tpr_at_5pct_fpr_k1}``); the
    T_pair-LDA w=20 vs 1 pooled AUC1 is the headline pair value of the Abstract, Section I and
    Section IV-B.

Inputs: ``outputs/p0/<decoder>/p0_table.json``, the server-side residual feature store
(``residual_v1.dat``, ``compact_rho.npy``) and the reference ``outputs/p0/pair_readouts/
fixed_h1_{rho,fusion_zsum,fusion_lda_full}.json`` (gate references). CPU only.

Outputs: ``outputs/p0/weak_contrast_rows.json``.

Usage:
    uv run --no-sync python scripts/analysis/weak_contrast_rows.py
"""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent.parent / "src"))


def _load_sibling(name: str):
    spec = importlib.util.spec_from_file_location(name, _HERE / f"{name}.py")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_PRB = _load_sibling("pair_readout_battery")
_FH = _load_sibling("fixed_h1_allcell")
_PVCM = _load_sibling("pair_variant_ci_matched")

from cfgid.detector.empirical import auc  # noqa: E402
from cfgid.paths import outputs_dir  # noqa: E402
from cfgid.sweep.neff_scaling import (  # noqa: E402
    _group_rows_by_cell,
    _load_p0_rows,
    empirical_group_auc,
)

K_GRID = (1, 2, 4, 8, 16, 32)
N_GROUPS = 4000  # production-matched (pair_readout_battery -> fixed_h1_allcell)
N_BOOT = 400
READOUTS = ("rho", "fusion_zsum", "fusion_lda_full")
# (h1_ws, h0_ws) ordering, matching pair_variant_ci_matched.VARIANTS
CONTRASTS = {"w20": ((20,), (1,)), "w5_vs_w3": ((5,), (3,))}


def _log(msg: str) -> None:
    print(f"[weak_contrast] {msg}", flush=True)


def _pools(cells, h1_ws, h0_ws):
    """Sign-aligned pooled TEST scores (h0, h1) over all cells, plus the macro per-cell AUC."""
    recs = _FH.collect_fixed_h1_records(cells, h1_ws, min_n=8, h0_ws=h0_ws)
    h0 = np.concatenate([r.sign * r.zt0 for r in recs])
    h1 = np.concatenate([r.sign * r.zt1 for r in recs])
    macro = float(np.mean([r.cell_test_auc() for r in recs]))
    return recs, h0, h1, macro


def main() -> int:
    p0 = outputs_dir() / "p0"
    refs: dict[str, dict] = {}
    for cand in READOUTS:
        ref_path = p0 / "pair_readouts" / f"fixed_h1_{cand}.json"
        if not ref_path.exists():
            print(f"[abort] shipped reference missing: {ref_path}")
            return 1
        refs[cand] = json.loads(ref_path.read_text())["variants"]

    work = Path(tempfile.mkdtemp(prefix="weak_contrast_rows_work_"))
    _log(f"stage-1 rebuild under {work} (feature store read)")
    roots, _diag, _counts = _PRB._compute_all_features(p0, work)

    out: dict = {
        "analysis": "weak_contrast_rows",
        "protocol": (
            "production fixed_h1 pools (per-cell FIT-H0 z-score + FIT sign, pooled TEST); "
            "unsaturated-mask exponent point + matched 400x bootstrap CI "
            "(pair_variant_ci_matched estimator); TPR at pooled FIT-H0 quantile thresholds; "
            "w20 self-check gates against pair_readouts shipped refs"
        ),
        "k_grid": list(K_GRID),
        "readouts": {},
    }

    for cand in READOUTS:
        rows: list[dict] = []
        for tp in sorted(roots[cand].glob("*/p0_table.json")):
            rows.extend(_load_p0_rows(tp))
        if not rows:
            print(f"[abort] no virtual tables for {cand}")
            return 1
        cells = _group_rows_by_cell(rows)
        entry: dict = {}
        for name, (h1_ws, h0_ws) in CONTRASTS.items():
            recs, h0, h1, macro = _pools(cells, h1_ws, h0_ws)
            lab = np.r_[np.zeros(h0.size), np.ones(h1.size)]
            a1 = float(auc(np.concatenate([h0, h1]), lab))

            # point estimate: cross-cell group-AUC curve on the full pools (seed 0)
            rng_pt = np.random.default_rng(0)
            auc_full = np.array(
                [empirical_group_auc(h0, h1, k, n_groups=N_GROUPS, rng=rng_pt) for k in K_GRID]
            )
            exp_pt, n_unsat = _PVCM._unsat_exponent(auc_full)

            if name == "w20":
                ref_all = refs[cand].get("w20", {}).get("all_cell", {})
                ref_a1 = ref_all.get("auc1_pooled_fitsign")
                ref_exp = ref_all.get("exponent")
                if ref_a1 is None or abs(a1 - float(ref_a1)) > 1e-6:
                    print(f"[abort] gate FAILED {cand}/w20: AUC1 {a1:.6f} vs shipped {ref_a1}")
                    return 1
                if ref_exp is not None and exp_pt is not None and abs(exp_pt - ref_exp) > 0.01:
                    print(f"[abort] gate FAILED {cand}/w20: exp {exp_pt:.4f} vs {ref_exp:.4f}")
                    return 1
                _log(f"gate OK {cand}/w20: AUC1 {a1:.6f}, exp {exp_pt:.4f}")

            # bootstrap: resample each class's pooled rows, rebuild the curve, refit the
            # unsaturated exponent (one rng threads through all replicates)
            rng = np.random.default_rng(0)
            b_matched: list[float] = []
            for _ in range(N_BOOT):
                hb0 = h0[rng.integers(0, h0.size, h0.size)]
                hb1 = h1[rng.integers(0, h1.size, h1.size)]
                emp = np.array(
                    [empirical_group_auc(hb0, hb1, k, n_groups=N_GROUPS, rng=rng) for k in K_GRID]
                )
                e_m, _ = _PVCM._unsat_exponent(emp)
                if e_m is not None:
                    b_matched.append(e_m)

            entry[name] = {
                "n_cells": len(recs),
                "auc1_pooled_fitsign": a1,
                "macro_cell_auc": macro,
                "exp_point_unsat": None if exp_pt is None else round(float(exp_pt), 4),
                "n_unsat_point": n_unsat,
                "exp_ci95_matched": (
                    [
                        round(float(np.percentile(b_matched, 2.5)), 3),
                        round(float(np.percentile(b_matched, 97.5)), 3),
                    ]
                    if b_matched
                    else None
                ),
                "tpr_at_1pct_fpr_k1": _FH._tpr_at_fpr(h0, h1, 0.01),
                "tpr_at_5pct_fpr_k1": _FH._tpr_at_fpr(h0, h1, 0.05),
                "auc_emp_k": [round(float(x), 4) for x in auc_full],
            }
            _log(
                f"{cand:16s} {name:8s} AUC1={a1:.4f} macro={macro:.4f} "
                f"exp={entry[name]['exp_point_unsat']} CI={entry[name]['exp_ci95_matched']} "
                f"TPR1%={entry[name]['tpr_at_1pct_fpr_k1']:.3f}"
            )
        out["readouts"][cand] = entry

    out_path = p0 / "weak_contrast_rows.json"
    out_path.write_text(json.dumps(out, indent=1, allow_nan=False), encoding="utf-8")
    _log(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
