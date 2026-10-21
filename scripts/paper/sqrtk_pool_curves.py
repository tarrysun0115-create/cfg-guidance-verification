"""Selection-free cross-cell AUC_K curves of rho on several pools (leave-one-decoder-out, low w).

Produces: the rho rows of Table III (Section IV-B): the leave-one-decoder-out cross-cell K=1 AUC
(``leave_decoder_out.<decoder>.auc1``) and the low-guidance H1 rows w=3 and w=5
(``low_guidance.w3.auc1``, ``low_guidance.w5.auc1``). ``pair_readout_battery.py`` calls ``main``
for each candidate readout to obtain the corresponding pair rows.

Inputs: only ``outputs/p0/<decoder>/p0_table.json`` (from run_p0). CPU only (numpy + matplotlib); no
re-inversion.

Outputs: ``<out-dir>/sqrtk_pool_curves.json`` and ``<out-dir>/sqrtk_pool_curves.png``. The paper's
``outputs/p0/sqrtk_pool_curves.json`` was written with ``--out-dir outputs/p0``.

Usage: python scripts/paper/sqrtk_pool_curves.py --features-root outputs/p0 --out-dir outputs/p0

Pools, all from the stored per-image rho(x):
  (1) the all-cell AUC_K-vs-K curve at w=20 vs w=1;
  (2) the same with one decoder left out (is the curve driven by one strong decoder?);
  (3) low-guidance curves (H1 = w in {3, 5} vs w=1);
  (4) single-image AUC_1 for each fixed w in {2,3,5,7,10,14,20} vs w=1;
  (5) the K=16 group AUC with a bootstrap 95% CI (``--n-boot`` 400 resamples, 1,000 groups each),
      next to a PROMPT-DISJOINT K=16 point (non-overlapping distinct-prompt groups), so the
      with-replacement estimate can be checked against an independent-image estimate.

All pooling is selection-free (no gate): each cell is sign-aligned by its FIT (``train``) mean
shift and z-scored on its FIT-H0 moments, then the standardized TEST scores are pooled. AUC_K is
the with-replacement cross-cell group AUC (``--n-groups`` 8,000 groups per K) with K in
{1,2,4,8,16,32}; the K=1 point is the "cross-cell K=1" estimand of Table III. One RNG (seed
``--seed``) is shared across all pools in the order above.
"""
from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path


def _log(msg: str) -> None:
    print(f"[sqrtk] {msg}", flush=True)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--features-root", default="outputs/p0")
    ap.add_argument("--out-dir", default="outputs/p0/sqrtk_curves")
    ap.add_argument("--min-per-cell", type=int, default=4)
    ap.add_argument("--n-groups", type=int, default=8000)
    ap.add_argument("--n-boot", type=int, default=400, help="bootstrap reps for the K=16 CI")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    os.environ.setdefault("MPLCONFIGDIR", os.path.abspath(".cache/mpl"))
    Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    from cfgid.detector.empirical import auc
    from cfgid.sweep.neff_scaling import (
        _load_p0_rows,
        _row_prompt_index,
        deflection_from_auc,
        empirical_group_auc,
        filter_core_none_qf_rows,
        predicted_group_auc,
    )

    rng = np.random.default_rng(args.seed)
    root = Path(args.features_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    k_grid = (1, 2, 4, 8, 16, 32)

    # ---- load rows, group into per-(decoder, sigma_blur) cells --------------------------
    rows: list[dict] = []
    for tp in sorted(root.glob("*/p0_table.json")):
        rows.extend(filter_core_none_qf_rows(_load_p0_rows(tp)))
    if not rows:
        _log(f"no core p0_table rows under {root}")
        return 0
    cells: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        cells[f"{r['decoder']}/sb={float(r['sigma_blur']):g}"].append(r)
    _log(f"{len(rows)} core rows over {len(cells)} cells")

    def _vals(cell_rows, w, split):
        return np.array(
            [float(r["rho_summary"]) for r in cell_rows
             if int(r["w"]) == w and r.get("split") == split
             and np.isfinite(float(r["rho_summary"]))],
            dtype=np.float64,
        )

    def _vals_prompt(cell_rows, w, split):
        out = []
        for r in cell_rows:
            if (int(r["w"]) == w and r.get("split") == split
                    and np.isfinite(float(r["rho_summary"]))):
                out.append((_row_prompt_index(r), float(r["rho_summary"])))
        return out

    def build_pool(w_h1, exclude=None, with_prompts=False):
        """Selection-free pool: per-cell fit-sign-aligned + fit-H0 z-scored TEST scores."""
        h0_all, h1_all, h1_prompts = [], [], []
        for key, crows in cells.items():
            if exclude is not None and key.split("/")[0] == exclude:
                continue
            h0f, h1f = _vals(crows, 1, "train"), _vals(crows, w_h1, "train")
            h0t, h1t = _vals(crows, 1, "test"), _vals(crows, w_h1, "test")
            if min(h0f.size, h1f.size, h0t.size, h1t.size) < args.min_per_cell:
                continue
            sign = 1.0 if (h1f.mean() - h0f.mean()) >= 0 else -1.0
            m0, s0 = float((sign * h0f).mean()), float((sign * h0f).std(ddof=1))
            if not (s0 > 0):
                continue
            h0_all.extend(((sign * h0t - m0) / s0).tolist())
            h1_all.extend(((sign * h1t - m0) / s0).tolist())
            if with_prompts:
                for pid, v in _vals_prompt(crows, w_h1, "test"):
                    h1_prompts.append((pid, (sign * v - m0) / s0))
        return np.array(h0_all), np.array(h1_all), h1_prompts

    def kcurve(h0, h1):
        if h0.size < args.min_per_cell or h1.size < args.min_per_cell:
            return None
        d1 = deflection_from_auc(auc(np.concatenate([h0, h1]),
                                     np.r_[np.zeros(h0.size), np.ones(h1.size)]))
        emp = [empirical_group_auc(h0, h1, k, n_groups=args.n_groups, rng=rng) for k in k_grid]
        pred = [predicted_group_auc(d1, k) for k in k_grid]
        return {"k_grid": list(k_grid), "auc_emp": emp, "auc_pred": pred, "auc1": emp[0]}

    diag: dict = {"protocol": "selection-free; per-cell fit-sign-aligned + fit-H0 z-scored"}

    # (1) all-cell + (2) leave-decoder-out ------------------------------------------------
    decoders = sorted({k.split("/")[0] for k in cells})
    h0, h1, _ = build_pool(20)
    diag["all_cell_w20"] = kcurve(h0, h1)
    _log(f"all-cell w20: AUC1={diag['all_cell_w20']['auc1']:.3f} "
         f"K16={diag['all_cell_w20']['auc_emp'][4]:.3f}")
    diag["leave_decoder_out"] = {}
    for dec in decoders:
        h0e, h1e, _ = build_pool(20, exclude=dec)
        r = kcurve(h0e, h1e)
        if r:
            diag["leave_decoder_out"][dec] = {"auc1": r["auc1"], "auc_k16": r["auc_emp"][4]}
    _log("leave-decoder-out K16: "
         + ", ".join(f"-{d}:{v['auc_k16']:.3f}" for d, v in diag["leave_decoder_out"].items()))

    # (3) low-guidance curves -------------------------------------------------------------
    diag["low_guidance"] = {}
    for w in (3, 5):
        h0w, h1w, _ = build_pool(w)
        r = kcurve(h0w, h1w)
        if r:
            diag["low_guidance"][f"w{w}"] = r

    # (4) per-fixed-w single-image AUC1 ---------------------------------------------------
    diag["per_w_auc1"] = {}
    for w in (2, 3, 5, 7, 10, 14, 20):
        h0w, h1w, _ = build_pool(w)
        if h0w.size >= args.min_per_cell and h1w.size >= args.min_per_cell:
            lab = np.r_[np.zeros(h0w.size), np.ones(h1w.size)]
            diag["per_w_auc1"][f"w{w}"] = float(auc(np.concatenate([h0w, h1w]), lab))

    # (5) K=16 bootstrap CI + prompt-disjoint independent estimate ------------------------
    h0, h1, h1p = build_pool(20, with_prompts=True)
    boots = []
    for _ in range(args.n_boot):
        hb0 = h0[rng.integers(0, h0.size, h0.size)]
        hb1 = h1[rng.integers(0, h1.size, h1.size)]
        boots.append(empirical_group_auc(hb0, hb1, 16, n_groups=1000, rng=rng))
    lo, hi = float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))
    # prompt-disjoint: groups of 16 with DISTINCT prompt ids, non-overlapping
    by_pid = defaultdict(list)
    for pid, v in h1p:
        by_pid[pid].append(v)
    pid_means = np.array([np.mean(vs) for vs in by_pid.values()])  # one score per prompt
    dj = None
    if pid_means.size >= 32:
        n_dj = pid_means.size // 16
        idx = rng.permutation(pid_means.size)[: n_dj * 16].reshape(n_dj, 16)
        h1_groups = pid_means[idx].mean(1)
        h0_groups = np.array([h0[rng.integers(0, h0.size, 16)].mean() for _ in range(n_dj)])
        dj = float(auc(np.r_[h0_groups, h1_groups], np.r_[np.zeros(n_dj), np.ones(n_dj)]))
    diag["k16_all_cell"] = {
        "group_auc": diag["all_cell_w20"]["auc_emp"][4],
        "bootstrap_ci95": [lo, hi],
        "note": "group AUC is a with-replacement bootstrap over overlapping K=16 groups",
        "prompt_disjoint_auc": dj,
        "prompt_disjoint_groups": None if dj is None else int(pid_means.size // 16),
    }
    _log(f"K=16 all-cell: {diag['all_cell_w20']['auc_emp'][4]:.3f} "
         f"CI[{lo:.3f},{hi:.3f}] prompt-disjoint={dj}")

    # ---- figure -------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    ax = axes[0]
    ac = diag["all_cell_w20"]
    ax.plot(k_grid, ac["auc_emp"], "o-", label="all-cell (w=20)")
    ax.plot(k_grid, ac["auc_pred"], "k--", lw=1, label="sqrt(K) prediction")
    for w, r in diag["low_guidance"].items():
        ax.plot(k_grid, r["auc_emp"], "s--", ms=4, label=f"all-cell ({w})")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("pooled images K")
    ax.set_ylabel("group AUC")
    ax.set_title("Selection-free group AUC vs K")
    ax.legend(fontsize=8)
    ax = axes[1]
    ws = [2, 3, 5, 7, 10, 14, 20]
    ax.plot(ws, [diag["per_w_auc1"].get(f"w{w}", np.nan) for w in ws], "o-")
    ax.axhline(0.5, color="#D1495B", ls="--", lw=1)
    ax.set_xlabel("guidance w (H1 vs w=1)")
    ax.set_ylabel("single-image AUC")
    ax.set_title("Per-fixed-w single-image AUC (selection-free)")
    fig.tight_layout()
    fig.savefig(out_dir / "sqrtk_pool_curves.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    (out_dir / "sqrtk_pool_curves.json").write_text(json.dumps(diag, indent=2), encoding="utf-8")
    _log(f"wrote {out_dir/'sqrtk_pool_curves.json'} + .png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
