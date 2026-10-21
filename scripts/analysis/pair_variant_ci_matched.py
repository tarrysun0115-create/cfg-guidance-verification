"""Matched-estimator bootstrap CIs for the pair readouts' cross-cell sqrt(K) exponent (Table I).

The exponent point estimate (``fixed_h1_allcell`` / ``pooled_neff_scaling``) fits the log-log
slope of the deflection inferred from the cross-cell group AUC against K ONLY over unsaturated K,
i.e. K with 1e-4 < groupAUC < 1 - 1e-4 and positive deflection. For the pair readout at w=20 vs
w=1 this typically drops K=32, whose group AUC is within 1e-4 of 1. A bootstrap that instead
refits the slope over ALL K in {1, 2, 4, 8, 16, 32}, with ``deflection_from_auc`` clamping the
saturated points, keeps the clamped top point and lowers the slope, so its interval belongs to a
different statistic than the point. This script recomputes the CI with the SAME unsaturated mask
as the point (``_unsat_exponent``), so point and CI use one estimator; the all-K CI is reported
next to it (``exp_ci95_allk_legacy``) to show the difference, together with how many K points
survive the mask in each replicate (``n_unsat_hist``).

Pools, FIT-H0 standardization and FIT sign follow ``fixed_h1_allcell`` on the virtual tables of
``pair_readout_battery._compute_all_features``. Recipe: point curve on the full pools with seed 0
and 4000 groups per K; 400 bootstrap replicates resampling each class's pooled rows, one rng
(seed 0) threaded through all replicates, 4000 groups per K. Self-checks abort unless pooled AUC1
reproduces the reference ``pair_readouts/fixed_h1_<readout>.json`` value to 1e-6 and the
recomputed AUC(K) curve matches the reference one to 5e-3; the recomputed exponent point is
compared with the reference one (warning above 0.01).

Produces:
    Table I, exponent 95% CIs of the T_pair-LDA rows w=10 vs 3 and {10,14,20} vs {1,2,3}
    (``pair_variant_ci_matched.json``: ``fusion_lda_full/{w3_vs_w10,low3_vs_high10}/
    exp_ci95_matched``). The w=20 vs 1 entries use the same recipe as weak_contrast_rows.py,
    which supplies the T_pair-zsum and T_pair-LDA w=20 vs 1 CIs printed in Table I.

Inputs: ``outputs/p0/<decoder>/p0_table.json``, the server-side residual feature store
(``residual_v1.dat``, ``compact_rho.npy``) and the reference ``outputs/p0/pair_readouts/
fixed_h1_{fusion_lda_full,fusion_zsum}.json`` (self-check references). CPU only.

Outputs: ``outputs/p0/pair_variant_ci_matched.json``. For each readout and contrast:
``exp_point_production`` is the reference point estimate (read from the reference),
``exp_ci95_matched`` the matched CI, ``exp_ci95_matched_covers_0.5`` whether that CI covers the
theoretical 1/2 of Eq. (14), and ``exp_point_recomputed_4000_unsat`` a self-check that should
equal the reference point.

Usage:
    uv run --no-sync python scripts/analysis/pair_variant_ci_matched.py
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

from cfgid.paths import outputs_dir  # noqa: E402
from cfgid.sweep.neff_scaling import (  # noqa: E402
    _group_rows_by_cell,
    _load_p0_rows,
    deflection_from_auc,
    empirical_group_auc,
)

VARIANTS = {  # (h1_ws, h0_ws) for the paper's Table I contrasts
    "w20": ((20,), (1,)),
    "low3_vs_high10": ((10, 14, 20), (1, 2, 3)),
    "w3_vs_w10": ((10,), (3,)),
}
K_GRID = (1, 2, 4, 8, 16, 32)
KS = np.asarray(K_GRID, dtype=float)
LOGK = np.log(KS)
N_BOOT = 400
N_GROUPS = 4000  # MUST match production (pair_readout_battery -> fixed_h1_allcell n_groups=4000)
# Production unsaturated mask (the ``unsat`` mask in src/cfgid/sweep/pooled_neff_scaling.py).
_LO, _HI = 1e-4, 1.0 - 1e-4


def _unsat_exponent(auc_by_k: np.ndarray) -> tuple[float | None, int]:
    """Production-matched slope: log-log fit over UNSATURATED K only.

    Returns (exponent_or_None, n_unsaturated_points). Mirrors pooled_neff_scaling.py exactly:
    keep K where `_LO < groupAUC < _HI and deflection > 0`, need >= 2 to fit.
    """
    d = np.array([deflection_from_auc(a) for a in auc_by_k])
    unsat = np.array([_LO < a < _HI and dd > 0 for a, dd in zip(auc_by_k, d, strict=True)])
    if unsat.sum() >= 2:
        return float(np.polyfit(np.log(KS[unsat]), np.log(d[unsat]), 1)[0]), int(unsat.sum())
    return None, int(unsat.sum())


def _all_k_exponent(auc_by_k: np.ndarray) -> float:
    """All-K slope: log-log fit over ALL 6 K with clamped deflection (the mismatched estimator)."""
    defl = np.log([deflection_from_auc(a) for a in auc_by_k])
    return float(np.polyfit(LOGK, defl, 1)[0])


def main() -> int:
    p0 = outputs_dir() / "p0"
    pools = {"fusion_lda_full": dict(VARIANTS), "fusion_zsum": {"w20": VARIANTS["w20"]}}
    refs: dict[str, dict] = {}
    for cand in pools:
        ref_path = p0 / "pair_readouts" / f"fixed_h1_{cand}.json"
        if not ref_path.exists():
            print(f"[abort] shipped reference missing: {ref_path}")
            return 1
        refs[cand] = json.loads(ref_path.read_text())["variants"]

    work = Path(tempfile.mkdtemp(prefix="pair_variant_ci_matched_work_"))
    print(f"[stage1] rebuilding per-image candidate scores under {work} (feature store read)")
    roots, _diag, _counts = _PRB._compute_all_features(p0, work)

    out: dict = {
        "analysis": "pair_variant_ci_matched",
        "protocol": (
            "fixed_h1_allcell pools; production UNSATURATED-mask exponent (1e-4<AUC<1-1e-4, "
            "defl>0, >=2 pts) matched between point and 400x bootstrap CI; legacy all-K CI shown "
            "for comparison; seed 0, n_groups 4000 (production-matched)"
        ),
        "k_grid": list(K_GRID),
    }
    for cand, variants in pools.items():
        out[cand] = {}
        rows: list[dict] = []
        for tp in sorted(roots[cand].glob("*/p0_table.json")):
            rows.extend(_load_p0_rows(tp))
        if not rows:
            print(f"[abort] no virtual tables for {cand}")
            return 1
        cells = _group_rows_by_cell(rows)
        for name, (h1_ws, h0_ws) in variants.items():
            recs = _FH.collect_fixed_h1_records(cells, h1_ws, h0_ws=h0_ws)
            h0 = np.concatenate([r.sign * r.zt0 for r in recs])
            h1 = np.concatenate([r.sign * r.zt1 for r in recs])

            # --- self-check gate: pooled AUC1 must reproduce the reference value ---
            scores = np.concatenate([h0, h1])
            order = scores.argsort(kind="mergesort")
            ranks = np.empty(order.size, dtype=float)
            ranks[order] = np.arange(1, scores.size + 1)
            sv = scores[order]
            i = 0
            while i < sv.size:
                j = i
                while j + 1 < sv.size and sv[j + 1] == sv[i]:
                    j += 1
                if j > i:
                    ranks[order[i : j + 1]] = (i + 1 + j + 1) / 2.0
                i = j + 1
            auc1 = float(
                (ranks[h0.size :].sum() - h1.size * (h1.size + 1) / 2.0) / (h0.size * h1.size)
            )
            ref_auc1 = refs[cand].get(name, {}).get("all_cell", {}).get("auc1_pooled_fitsign")
            if ref_auc1 is None or abs(auc1 - float(ref_auc1)) > 1e-6:
                print(f"[abort] self-check FAILED {cand}/{name}: {auc1:.6f} vs shipped {ref_auc1}")
                return 1
            print(f"[selfcheck] {cand}/{name}: AUC1 {auc1:.6f} == shipped {float(ref_auc1):.6f}")

            # --- production point + curve come FROM the reference (4000-group UNSAT), so the
            #     reported POINT is exactly production; we only recompute to self-check them ---
            ref_all = refs[cand].get(name, {}).get("all_cell", {})
            ref_exp = ref_all.get("exponent")  # production 4000-group UNSAT exponent
            ref_auc_emp = ref_all.get("auc_emp")  # production 4000-group K-curve
            ref_kgrid = ref_all.get("k_grid")

            # --- POINT estimate on full data (both estimators), fixed seed; must reproduce ref ---
            rng_pt = np.random.default_rng(0)
            auc_full = np.array(
                [empirical_group_auc(h0, h1, k, n_groups=N_GROUPS, rng=rng_pt) for k in K_GRID]
            )
            exp_pt_matched, n_unsat_pt = _unsat_exponent(auc_full)
            exp_pt_allk = _all_k_exponent(auc_full)

            # curve self-check: recomputed 4000-group AUC(K) must reproduce the reference auc_emp
            if ref_auc_emp is not None and list(ref_kgrid or ()) == list(K_GRID):
                dmax = float(np.max(np.abs(auc_full - np.asarray(ref_auc_emp, dtype=float))))
                if dmax > 5e-3:
                    print(f"[abort] curve self-check FAILED {cand}/{name}: max diff {dmax:.4g}")
                    return 1
                print(f"[selfcheck] {cand}/{name}: AUC(K) reproduces shipped (max {dmax:.4g})")
            # point self-check: recomputed UNSAT exponent must reproduce the reference point
            if ref_exp is not None and exp_pt_matched is not None:
                dexp = abs(exp_pt_matched - float(ref_exp))
                flag = "" if dexp <= 0.01 else "  [WARN >0.01: report the ref exp]"
                re = float(ref_exp)
                print(f"[selfcheck] {cand}/{name}: exp {exp_pt_matched:.4f} vs ref {re:.4f}{flag}")

            # --- bootstrap: SAME unsaturated mask per replicate (matched) + legacy all-K ---
            rng = np.random.default_rng(0)
            b_matched, b_allk, b_k16, n_unsat_hist = [], [], [], []
            for _ in range(N_BOOT):
                hb0 = h0[rng.integers(0, h0.size, h0.size)]
                hb1 = h1[rng.integers(0, h1.size, h1.size)]
                emp = np.array(
                    [empirical_group_auc(hb0, hb1, k, n_groups=N_GROUPS, rng=rng) for k in K_GRID]
                )
                b_k16.append(float(emp[4]))
                e_m, n_u = _unsat_exponent(emp)
                n_unsat_hist.append(n_u)
                if e_m is not None:
                    b_matched.append(e_m)
                b_allk.append(_all_k_exponent(emp))

            def _ci(v: list[float]) -> list[float]:
                return [
                    round(float(np.percentile(v, 2.5)), 3),
                    round(float(np.percentile(v, 97.5)), 3),
                ]

            hist = {
                int(k): int((np.asarray(n_unsat_hist) == k).sum())
                for k in sorted(set(n_unsat_hist))
            }
            rec = {
                # REPORT as the Table I point: reference production 4000-group UNSAT exponent.
                "exp_point_production": None if ref_exp is None else round(float(ref_exp), 4),
                "exp_ci95_matched": _ci(b_matched)
                if b_matched
                else None,  # 4000-group UNSAT bootstrap
                "exp_ci95_matched_covers_0.5": (
                    bool(b_matched) and _ci(b_matched)[0] <= 0.5 <= _ci(b_matched)[1]
                ),
                "exp_ci95_matched_n_replicates": len(b_matched),
                # cross-checks (must be ~equal to exp_point_production; not for the paper):
                "exp_point_recomputed_4000_unsat": None
                if exp_pt_matched is None
                else round(exp_pt_matched, 4),
                "exp_point_allk_legacy": round(exp_pt_allk, 4),
                "exp_ci95_allk_legacy": _ci(b_allk),  # all-K estimator CI, for comparison
                "auc_full_by_k": [round(float(a), 6) for a in auc_full],
                "k16_emp": round(float(np.mean(b_k16)), 4),
                "k16_ci95": _ci(b_k16),
                "n_unsat_hist": hist,
                "n_h0": int(h0.size),
                "n_h1": int(h1.size),
                "n_cells": len(recs),
            }
            out[cand][name] = rec
            print(
                f"[done] {cand}/{name}: PRODUCTION point={rec['exp_point_production']} "
                f"CI_matched(4000,unsat)={rec['exp_ci95_matched']} "
                f"(covers 0.5: {rec['exp_ci95_matched_covers_0.5']}) | "
                f"old legacy CI={rec['exp_ci95_allk_legacy']} | n_unsat_hist={hist}"
            )

    out_path = p0 / "pair_variant_ci_matched.json"
    out_path.write_text(json.dumps(out, indent=1))
    print(f"[saved] {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
