"""Group-level (mean-of-K) conformal operating points from the reserved CAL split (Section IV-D).

MOTIVATION
----------
The single-image conformal route (cal_split_fpr.py) controls FPR per image, while the
aggregation results concern means of K same-cell images. A mean-of-K conformal operating point
needs enough calibration groups: the order-statistic rule needs ceil((G+1)(1-alpha)) <= G
non-overlapping K-group means, i.e. G >= 19 at alpha=0.05 (G >= 99 at alpha=0.01). The reserved
CAL-H0 split (51--103 rows/cell) supports G = floor(n_cal/K) >= 19 for K=2 in EVERY cell and for
K=4 in the larger cells. This script evaluates the deployed pipeline's mean-of-K conformal
operating point exactly where the data permit, and records the infeasibility frontier (K>=8)
explicitly.

Construction (mirrors cal_split_fpr.py, the single-image conformal construction):
per (decoder, sigma_blur) cell at the canonical w=20-vs-1 contrast, deployed FIT recipe
(_readout_common.fit_deployed) -> CAL-H0 / TEST-H0 / TEST-H1 scores (score_deployed) ->
non-overlapping mean-of-K groups, formed within each split and class separately (G = floor(n/K)
groups, remainder dropped; each split is permuted on its own, so TEST-H0 and TEST-H1 groups are
drawn independently and may share prompts; the conformal threshold needs only the within-class
CAL-H0 groups) -> CAL-group order-statistic threshold
at alpha (k-th smallest group mean, k = ceil((G+1)(1-alpha)); infeasible when k > G) -> realized
group FPR/TPR on disjoint TEST groups.

Two readings are reported per (readout, cell, K, alpha):
  * fixed partition: each split is permuted once with its own seed (CAL 0, TEST-H0 1, TEST-H1 2;
    K=1 uses the rows as-is): ONE draw of the procedure, with an exact one-sided
    Clopper--Pearson 95% upper bound on the group FPR;
  * partition-averaged (N_PART=200 random re-partitions from one generator seeded 1000+K): the
    expected realized rate of the procedure (descriptive; partitions reuse rows, so no fresh
    guarantee is implied).
The summary macro-averages FPR/TPR over the feasible cells for both readings.

FAIL-LOUD GATE: the K=1 fixed-partition path must reproduce the reference
cal_split_fpr.json per-cell realized FPR/TPR exactly (1e-9) for every readout/alpha before any
K>1 number is emitted.

Produces: the group-conformal statements of Section IV-D (K=2 and K=4 macro FPR on a fixed
partition and partition-averaged, TPR against K=1, groups per cell, infeasible K) and the
group-level conformal construction of Appendix B.
Inputs: main-grid outputs/p0/<dec>/p0_table.json + the server-side feature
store, and outputs/p0/cal_split_fpr.json (from cal_split_fpr.py) for the K=1 gate; CPU.
Outputs: outputs/p0/group_conformal_k.json. An infeasible (cell, K, alpha) entry stores
"needed_groups" = ceil(1/alpha) (20 at alpha=0.05, 100 at alpha=0.01), a conservative sufficient
count; the exact minimum number of CAL groups for a finite threshold is 19 and 99 (as in the paper).
Usage:
  uv run --no-sync python scripts/analysis/group_conformal_k.py
"""

from __future__ import annotations

import importlib.util
import json
import math
import sys
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


_CSF = _load_sibling("cal_split_fpr")

from _readout_common import (  # noqa: E402
    fit_deployed,
    load_residual_rows,
    score_deployed,
    write_json_strict,
)

from cfgid.paths import outputs_dir  # noqa: E402

H1_SET, H0_SET = {20.0}, {1.0}
READOUTS = ("rho", "zsum_dep", "lda_dep")
ALPHAS = (0.05, 0.01)
K_LIST = (1, 2, 4, 8)
N_PART = 200
GATE_TOL = 1e-9


def _log(msg: str) -> None:
    print(f"[groupconf] {msg}", flush=True)


def _group_means(x: np.ndarray, k: int, rng: np.random.Generator | None) -> np.ndarray:
    """Non-overlapping mean-of-k groups; rng=None keeps the original row order (K=1 gate)."""
    v = np.asarray(x, dtype=np.float64).reshape(-1)
    if k == 1:
        return v
    if rng is not None:
        v = rng.permutation(v)
    g = v.size // k
    if g == 0:
        return np.empty(0)
    return v[: g * k].reshape(g, k).mean(axis=1)


def _find_cell_entry(shipped: dict, readout: str, cellkey: str) -> dict | None:
    """Locate the reference cal_split_fpr per-cell record INSIDE the readout's subtree.

    The reference artifact may key readouts by code name ("zsum_dep") or display name
    (cal_split_fpr.RLABEL, e.g. "deployed z-sum"). NO whole-dict fallback: a scoped miss
    returns None so the caller can fail loudly -- a whole-dict search could match rho's
    cells against the pair readouts.
    """

    def walk(o):
        if isinstance(o, dict):
            if cellkey in o and isinstance(o[cellkey], dict) and "n_cal" in o[cellkey]:
                return o[cellkey]
            for v in o.values():
                r = walk(v)
                if r is not None:
                    return r
        return None

    def find_subtree(o, names):
        """Depth-first: the first dict stored under any of `names`, at any nesting depth."""
        if isinstance(o, dict):
            for k, v in o.items():
                if k in names and isinstance(v, dict):
                    return v
            for v in o.values():
                r = find_subtree(v, names)
                if r is not None:
                    return r
        return None

    rlabel = getattr(_CSF, "RLABEL", {})
    names = {n for n in (readout, rlabel.get(readout)) if n}
    sub = find_subtree(shipped, names)
    return walk(sub) if isinstance(sub, dict) else None


def main() -> int:
    p0 = outputs_dir() / "p0"
    shipped_path = p0 / "cal_split_fpr.json"
    if not shipped_path.exists():
        print(f"[abort] shipped reference missing: {shipped_path}")
        return 1
    shipped = json.loads(shipped_path.read_text())

    rows = load_residual_rows(p0)
    cells: dict[tuple[str, float], list[dict]] = {}
    for r in rows:
        cells.setdefault((r["decoder"], r["sigma_blur"]), []).append(r)

    out: dict = {
        "analysis": "group_conformal_k",
        "contrast": {"H1_w": [20], "H0_w": [1]},
        "k_list": list(K_LIST),
        "alphas": list(ALPHAS),
        "n_part": N_PART,
        "protocol": (
            "deployed FIT recipe -> CAL-H0 / TEST scores (cal_split_fpr construction) -> "
            "non-overlapping mean-of-K groups -> CAL-group order-statistic threshold -> "
            "TEST-group realized FPR/TPR; fixed-partition draw with CP95 upper bound plus "
            "partition-averaged descriptive rates; K=1 gated against shipped cal_split_fpr"
        ),
        "readouts": {},
    }

    gate_checked = 0
    gate_by_readout: dict[str, int] = {}
    for readout in READOUTS:
        per_cell: dict[str, dict] = {}
        for (dec, sb), cr in sorted(cells.items()):
            fit = fit_deployed(cr, H1_SET, H0_SET, readout)
            if fit is None:
                continue
            cal = score_deployed(cr, H0_SET, "calibration", "h0", fit)
            t0 = score_deployed(cr, H0_SET, "test", "h0", fit)
            t1 = score_deployed(cr, H1_SET, "test", "h1", fit)
            if len(cal) < 8 or len(t0) < 8 or len(t1) < 8:
                continue
            cellkey = f"{dec}_sb{sb:g}"
            entry: dict = {
                "n_cal": int(len(cal)),
                "n_test_h0": int(len(t0)),
                "n_test_h1": int(len(t1)),
            }
            for k in K_LIST:
                g_cal = len(cal) // k
                kentry: dict = {
                    "cal_groups": int(g_cal),
                    "test_h0_groups": int(len(t0) // k),
                    "test_h1_groups": int(len(t1) // k),
                }
                # fixed partition: CAL seed 0, TEST-H0 seed 1, TEST-H1 seed 2; K=1 keeps the
                # stored row order (needed for the exact K=1 gate)
                rng_fixed = None if k == 1 else np.random.default_rng(0)
                gc = _group_means(cal, k, rng_fixed)
                g0 = _group_means(t0, k, None if k == 1 else np.random.default_rng(1))
                g1 = _group_means(t1, k, None if k == 1 else np.random.default_rng(2))
                for a in ALPHAS:
                    thr = _CSF._conformal_threshold(gc, a)
                    akey = f"alpha{a}"
                    if thr is None:
                        # "needed_groups" records ceil(1/alpha) (20 at alpha=0.05, 100 at
                        # alpha=0.01): a conservative sufficient count, kept as stored. The exact
                        # minimum number of CAL groups for a finite threshold is (1 - alpha)/alpha,
                        # i.e. 19 at alpha=0.05 and 99 at alpha=0.01, as in the paper
                        # (cal_complexity.analytic_min_ncal).
                        kentry[akey] = {"feasible": False, "needed_groups": int(math.ceil(1.0 / a))}
                        continue
                    fp, tp = int(np.sum(g0 > thr)), int(np.sum(g1 > thr))
                    rec = {
                        "feasible": True,
                        "threshold_fixed": float(thr),
                        "fpr_fixed": fp / len(g0),
                        "fpr_cp95_upper_fixed": _CSF._cp_upper(fp, len(g0)),
                        "tpr_fixed": tp / len(g1),
                    }
                    # partition-averaged (descriptive): N_PART re-partitions of all three
                    # splits drawn from one generator seeded 1000 + K
                    if k > 1:
                        rng = np.random.default_rng(1000 + k)
                        fprs, tprs = [], []
                        for _ in range(N_PART):
                            gcp = _group_means(cal, k, rng)
                            thr_p = _CSF._conformal_threshold(gcp, a)
                            if thr_p is None:
                                continue
                            g0p = _group_means(t0, k, rng)
                            g1p = _group_means(t1, k, rng)
                            fprs.append(float(np.mean(g0p > thr_p)))
                            tprs.append(float(np.mean(g1p > thr_p)))
                        if fprs:
                            rec["fpr_partavg"] = float(np.mean(fprs))
                            rec["tpr_partavg"] = float(np.mean(tprs))
                            rec["n_partitions"] = len(fprs)
                    kentry[akey] = rec
                entry[f"k{k}"] = kentry
            # -------- K=1 gate against the reference single-image cal_split_fpr.json --------
            ref = _find_cell_entry(shipped, readout, cellkey)
            if ref is not None:
                for a in ALPHAS:
                    r_ref = ref.get(f"alpha{a}")
                    r_new = entry["k1"].get(f"alpha{a}")
                    if not r_ref or not r_new:
                        continue
                    if bool(r_ref.get("feasible")) != bool(r_new.get("feasible")):
                        print(f"[abort] gate FAILED {readout}/{cellkey}/a={a}: feasibility differs")
                        return 1
                    if r_ref.get("feasible"):
                        # Gate on the scale-invariant sufficient statistics of the procedure.
                        # The reference pair thresholds carry an older weight-scaling convention
                        # (monotone rescale: ranks, hence FPR/TPR, are unchanged); the raw
                        # threshold is therefore recorded as a diagnostic ratio, not gated.
                        for mine, theirs in (("fpr_fixed", "fpr_test"), ("tpr_fixed", "tpr_test")):
                            if abs(float(r_new[mine]) - float(r_ref[theirs])) > GATE_TOL:
                                print(
                                    f"[abort] gate FAILED {readout}/{cellkey}/a={a}: "
                                    f"{mine} {r_new[mine]} vs shipped {r_ref[theirs]}"
                                )
                                return 1
                        thr_ref = float(r_ref.get("threshold", float("nan")))
                        thr_new = float(r_new["threshold_fixed"])
                        r_new["threshold_ratio_vs_shipped"] = (
                            thr_ref / thr_new if thr_new not in (0.0,) else None
                        )
                        gate_checked += 1
                        gate_by_readout[readout] = gate_by_readout.get(readout, 0) + 1
            per_cell[cellkey] = entry
        # -------- summary --------
        summary: dict = {}
        for k in K_LIST:
            for a in ALPHAS:
                feas = [
                    c
                    for c in per_cell.values()
                    if c.get(f"k{k}", {}).get(f"alpha{a}", {}).get("feasible")
                ]
                key = f"k{k}_alpha{a}"
                if not feas:
                    summary[key] = {"feasible_cells": 0}
                    continue
                summary[key] = {
                    "feasible_cells": len(feas),
                    "cells_total": len(per_cell),
                    "macro_fpr_fixed": float(
                        np.mean([c[f"k{k}"][f"alpha{a}"]["fpr_fixed"] for c in feas])
                    ),
                    "macro_tpr_fixed": float(
                        np.mean([c[f"k{k}"][f"alpha{a}"]["tpr_fixed"] for c in feas])
                    ),
                    "macro_fpr_partavg": (
                        float(
                            np.mean(
                                [
                                    c[f"k{k}"][f"alpha{a}"]["fpr_partavg"]
                                    for c in feas
                                    if "fpr_partavg" in c[f"k{k}"][f"alpha{a}"]
                                ]
                            )
                        )
                        if k > 1
                        else None
                    ),
                    "macro_tpr_partavg": (
                        float(
                            np.mean(
                                [
                                    c[f"k{k}"][f"alpha{a}"]["tpr_partavg"]
                                    for c in feas
                                    if "tpr_partavg" in c[f"k{k}"][f"alpha{a}"]
                                ]
                            )
                        )
                        if k > 1
                        else None
                    ),
                }
        out["readouts"][readout] = {"per_cell": per_cell, "summary": summary}
        for k in K_LIST:
            for a in ALPHAS:
                sm = out["readouts"][readout]["summary"][f"k{k}_alpha{a}"]
                if sm.get("feasible_cells"):
                    msg = (
                        f"{readout:9s} K={k:2d} a={a:4.2f}"
                        f" feas={sm['feasible_cells']:2d}/{sm['cells_total']}"
                        f" FPRfix={sm['macro_fpr_fixed']:.4f} TPRfix={sm['macro_tpr_fixed']:.3f}"
                    )
                    if sm.get("macro_fpr_partavg") is not None:
                        msg += (
                            f" FPRavg={sm['macro_fpr_partavg']:.4f}"
                            f" TPRavg={sm['macro_tpr_partavg']:.3f}"
                        )
                    _log(msg)
                else:
                    _log(f"{readout:9s} K={k:2d} a={a:4.2f} INFEASIBLE in all cells")

    if gate_checked == 0:
        print(
            "[abort] K=1 gate never engaged (shipped per-cell entries not found); refusing numbers"
        )
        return 1
    for readout in READOUTS:
        if gate_by_readout.get(readout, 0) == 0:
            print(f"[abort] K=1 gate never engaged for readout {readout!r}; refusing numbers")
            return 1
    _log(f"K=1 gate: {gate_checked} (readout, cell, alpha) records reproduced shipped exactly")
    out["k1_gate_records"] = gate_checked

    write_json_strict(p0 / "group_conformal_k.json", out)
    _log(f"wrote {p0 / 'group_conformal_k.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
