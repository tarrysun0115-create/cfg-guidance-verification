"""Favorable-subset sensitivity: leave-one-cell-out and a sweep of the gate threshold.

Produces: the subset-selection sensitivity of Section IV-E7: the lowest leave-one-cell-out pooled
AUC_1 (``leave_one_cell_out.min_loo_auc``, with the removed cell in ``per_cell``) and the pooled
AUC_1 and permutation p at each gate threshold (``alive_t_sweep[*].auc``, ``.perm_p``; 2,000
label permutations). ``pair_readout_battery.py`` reuses ``run`` for each candidate readout.

Inputs: only ``outputs/p0/<decoder>/p0_table.json`` (from run_p0). Pure numpy, CPU only; no
re-inversion.

Outputs: ``outputs/p0/pooled_robustness.json``.

Usage: python scripts/analysis/pooled_robustness.py [--alive-t 1.5] [--n-perm 2000]

The favorable subset pools the cells admitted by a single gate threshold (Welch |t| >= 1.5 on the
gate half of TEST, Section IV-E). Two selection threats are checked, reusing the SAME selection and
per-cell FIT-sign alignment as the pooled module (``cfgid.sweep.pooled_neff_scaling``):

  - LEAVE-ONE-CELL-OUT: drop each admitted cell, re-pool the rest with each cell's own FIT sign,
    recompute the AUC. Report the per-cell influence (full - drop-one), the min jackknife AUC, and
    a permutation p-value for the pool with the single MOST-influential cell removed.
  - GATE-THRESHOLD SWEEP: re-run the whole cell selection + pooled AUC + permutation null across
    alive_t in {0.5, 1.0, 1.5, 2.0, 2.5} (plus the primary ``--alive-t``), with seed 100 + index.

Verdict: ROBUST if (a) the most-influential-cell-removed pool stays significant (perm p<0.05) AND
its AUC stays above ``auc_floor``, AND (b) the sweep keeps >=2 thresholds significant with AUC
above the floor. FRAGILE if dropping one cell or changing the threshold collapses it. INCONCLUSIVE
if too few cells survive to jackknife. The subset is outcome-selected, so these runs bound the
magnitude on that subset only.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from cfgid.paths import outputs_dir
from cfgid.sweep.neff_scaling import _group_rows_by_cell, _load_p0_rows
from cfgid.sweep.pooled_neff_scaling import (
    CellRecord,
    collect_alive_cell_records,
    pooled_eval_auc,
    pooled_eval_auc_perm,
)

DEFAULT_ALIVE_T_GRID = (0.5, 1.0, 1.5, 2.0, 2.5)


def _signs(records: list[CellRecord]) -> list[float]:
    return [r.sign for r in records]


def leave_one_cell_out(records: list[CellRecord], *, n_perm: int, seed: int) -> dict:
    """Drop each cell, re-pool the rest under oracle sign; report influence + the worst drop."""
    full_auc = pooled_eval_auc(records, _signs(records))
    per_cell = []
    for i, r in enumerate(records):
        kept = [c for j, c in enumerate(records) if j != i]
        loo_auc = pooled_eval_auc(kept, _signs(kept))
        per_cell.append(
            {"cell": r.label(), "loo_auc": loo_auc, "influence": float(full_auc - loo_auc)}
        )
    # The pool that drops the single cell whose presence helps the AUC most (largest influence):
    worst = max(range(len(records)), key=lambda i: per_cell[i]["influence"])
    kept = [c for j, c in enumerate(records) if j != worst]
    drop_auc, drop_p = pooled_eval_auc_perm(kept, _signs(kept), n_perm=n_perm, seed=seed)
    return {
        "full_auc": full_auc,
        "per_cell": per_cell,
        "min_loo_auc": min(pc["loo_auc"] for pc in per_cell),
        "max_influence_cell": per_cell[worst]["cell"],
        "drop_most_influential_auc": drop_auc,
        "drop_most_influential_perm_p": drop_p,
    }


def alive_t_sweep(cells: dict, grid: tuple[float, ...], *, n_perm: int, seed: int) -> list[dict]:
    """Re-select + re-pool under oracle sign across a grid of alive_t thresholds."""
    out = []
    for i, t in enumerate(grid):
        recs = collect_alive_cell_records(cells, alive_t=t)
        if len(recs) < 1:
            out.append({"alive_t": t, "n_cells": 0, "auc": None, "perm_p": None})
            continue
        auc_t, p_t = pooled_eval_auc_perm(recs, _signs(recs), n_perm=n_perm, seed=seed + i)
        out.append({"alive_t": t, "n_cells": len(recs), "auc": auc_t, "perm_p": p_t})
    return out


def assess(
    cells: dict,
    *,
    alive_t: float,
    grid: tuple[float, ...],
    n_perm: int,
    seed: int,
    auc_floor: float = 0.70,
) -> dict:
    # Always sweep through the operating point: the leave-one-cell-out jackknife is run at the
    # primary alive_t, so the threshold-robustness sweep must re-pool there too, else the ROBUST/
    # FRAGILE verdict mixes two selections (e.g. CLI --alive-t 3.0 vs a fixed 0.5..2.5 grid).
    grid = tuple(sorted(set(grid) | {float(alive_t)}))
    records = collect_alive_cell_records(cells, alive_t=alive_t)
    n = len(records)
    if n < 3:
        return {
            "n_alive_cells": n,
            "alive_cells": [r.label() for r in records],
            "verdict": "INCONCLUSIVE",
            "reason": f"only {n} alive cells at alive_t={alive_t}; too few to jackknife.",
        }
    loo = leave_one_cell_out(records, n_perm=n_perm, seed=seed)
    sweep = alive_t_sweep(cells, grid, n_perm=n_perm, seed=seed + 100)

    cell_robust = bool(
        loo["drop_most_influential_perm_p"] < 0.05 and loo["drop_most_influential_auc"] >= auc_floor
    )
    sig_thresholds = [
        s for s in sweep if s["perm_p"] is not None and s["perm_p"] < 0.05 and s["auc"] >= auc_floor
    ]
    threshold_robust = bool(len(sig_thresholds) >= 2)

    if cell_robust and threshold_robust:
        verdict = "ROBUST"
        drop_auc = loo["drop_most_influential_auc"]
        drop_p = loo["drop_most_influential_perm_p"]
        reason = (
            f"dropping the most-influential cell ({loo['max_influence_cell']}) still gives "
            f"AUC={drop_auc:.3f} (perm p={drop_p:.4f}), and {len(sig_thresholds)}/{len(grid)} "
            f"alive_t thresholds stay significant above {auc_floor}: not a one-cell/knob artifact."
        )
    else:
        verdict = "FRAGILE"
        bits = []
        if not cell_robust:
            d_auc = loo["drop_most_influential_auc"]
            d_p = loo["drop_most_influential_perm_p"]
            bits.append(
                f"removing the most-influential cell collapses it (AUC={d_auc:.3f}, p={d_p:.4f})"
            )
        if not threshold_robust:
            bits.append(
                f"only {len(sig_thresholds)}/{len(grid)} alive_t thresholds stay significant above "
                f"{auc_floor}"
            )
        reason = "; ".join(bits) + " -> treat the pooled AUC as a small-sample point estimate."

    return {
        "n_alive_cells": n,
        "alive_cells": [r.label() for r in records],
        "auc_floor": auc_floor,
        "leave_one_cell_out": loo,
        "alive_t_sweep": sweep,
        "cell_robust": cell_robust,
        "threshold_robust": threshold_robust,
        "verdict": verdict,
        "reason": reason,
    }


def run(
    features_root: Path,
    out_dir: Path,
    *,
    alive_t: float,
    grid: tuple[float, ...],
    n_perm: int,
    seed: int,
) -> int:
    rows: list[dict] = []
    for table_path in sorted(features_root.glob("*/p0_table.json")):
        rows.extend(_load_p0_rows(table_path))
    if not rows:
        print(f"[skip] no <decoder>/p0_table.json under {features_root}")
        return 0
    res = assess(_group_rows_by_cell(rows), alive_t=alive_t, grid=grid, n_perm=n_perm, seed=seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "pooled_robustness.json").write_text(
        json.dumps(res, indent=2, allow_nan=False), encoding="utf-8"
    )
    print(f"\nPooled robustness ({res['n_alive_cells']} alive cells, alive_t={alive_t}):")
    if "leave_one_cell_out" in res:
        loo = res["leave_one_cell_out"]
        print(
            f"  full pooled AUC={loo['full_auc']:.3f}  "
            f"min leave-one-cell-out AUC={loo['min_loo_auc']:.3f}"
        )
        print(
            f"  drop most-influential ({loo['max_influence_cell']}): "
            f"AUC={loo['drop_most_influential_auc']:.3f} "
            f"perm p={loo['drop_most_influential_perm_p']:.4f}"
        )
        print("  alive_t sweep:")
        for s in res["alive_t_sweep"]:
            auc_s = "n/a" if s["auc"] is None else f"{s['auc']:.3f}"
            p_s = "n/a" if s["perm_p"] is None else f"{s['perm_p']:.4f}"
            print(f"    alive_t={s['alive_t']:.1f}  n_cells={s['n_cells']}  AUC={auc_s}  p={p_s}")
    print(f"  => {res['verdict']}: {res['reason']}")
    print(f"-> {out_dir / 'pooled_robustness.json'}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="pooled_robustness")
    ap.add_argument(
        "--features-root", default=None, help="dir of <decoder>/p0_table.json (default: outputs/p0)"
    )
    ap.add_argument(
        "--alive-t", type=float, default=1.5, help="primary alive gate (pooled default)"
    )
    ap.add_argument("--n-perm", type=int, default=2000)
    args = ap.parse_args(argv)
    features_root = Path(args.features_root) if args.features_root else (outputs_dir() / "p0")
    out_dir = outputs_dir() / "p0"
    return run(
        features_root,
        out_dir,
        alive_t=args.alive_t,
        grid=DEFAULT_ALIVE_T_GRID,
        n_perm=args.n_perm,
        seed=0,
    )


if __name__ == "__main__":
    raise SystemExit(main())
