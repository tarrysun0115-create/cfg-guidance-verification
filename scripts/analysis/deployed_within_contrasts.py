"""Deployed-readout within-cell K curves: same-cell aggregation of the deployed heads.

Evaluates the DEPLOYED estimator under the theorem-matched fixed-configuration, non-overlapping
grouping of Lemma 2 (Eq. (14)). It rebuilds the deployed readout per cell -- H0-only
per-coordinate standardization on FIT, Ledoit--Wolf-regularized 2x2 H0 covariance, plug-in
Fisher weights (Eq. (11); mirroring ``pair_readout_battery._fit_stats_2d`` / ``_apply_fusion``
with ``w_full``, the ``fusion_lda_full`` estimator; the magnitude coordinate is the store's
masked winsorized diff_energy, m of Eq. (8)) -- scores held-out TEST rows, then forms
non-overlapping mean-of-K groups strictly inside each (decoder, sigma_blur) cell (K=1,2,4,8,16;
averaged over --n-part random partitions), and reports the macro over cells, a cell-cluster
bootstrap 95% CI, the worst cell, the per-cell exponent fitted over unsaturated K, and
single-image FPR/TPR at FIT-H0 quantile thresholds.

Each cell also gets an OUTER PROMPT-BLOCK bootstrap: prompts are the independent units inside a
cell (one blur level -- no raw-generation siblings), so each replicate resamples FIT prompts
(refitting standardization AND Ledoit--Wolf weights) and TEST prompts, then recomputes the K=16
non-overlap AUC. This propagates FIT-parameter and prompt-sampling uncertainty that partition
averaging alone cannot. A two-level bootstrap (cells, then one prompt-block replicate per drawn
cell) gives the interval that combines both. Tail resolution: TEST has only 51/102 images per
class per cell, i.e. 3-6 independent K=16 groups; the FPR/TPR columns are diagnostic and no
strict-alpha claim follows.

Contrasts: ``w20_vs_1`` and ``w5_vs_3`` are fitted and scored on their own contrast; the
``*_mainfit`` entries score weaker contrasts with the deployed head fitted at w=20 vs 1.
Readouts: rho, deployed z-sum (``w_zsum``), deployed Ledoit--Wolf LDA (``w_full``). This is the
deployed estimator, NOT the audit ridge-LDA (+1e-6 I) of Fig. 3 / samecell_nonoverlap; compare
against samecell_nonoverlap.json for the audit-vs-deployed delta. Pure reanalysis.

Produces: Table I same-cell AUC_16 [cell-cluster CI] entries of T_pair-zsum and T_pair-LDA
(w=20 vs 1) and of the weaker-contrast rows on the deployed head (``*_mainfit``); Table V
per-cell values; the Section IV-C worst cell, median
per-cell exponent and two-level interval; the pair AUC_1 values quoted in Section IV-E
("Blur response").
Inputs: main-grid outputs/p0/<dec>/p0_table.json + the server-side feature
store; CPU.
Outputs: outputs/p0/deployed_within_contrasts.json.
Usage:
  uv run --no-sync python scripts/analysis/deployed_within_contrasts.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from _readout_common import (
    DEPLOYED_READOUTS,
    fit_deployed,
    load_residual_rows,
    score_deployed,
    write_json_strict,
)
from scipy.stats import norm

from cfgid.detector.empirical import auc
from cfgid.paths import outputs_dir

GROUP_K = [1, 2, 4, 8, 16]
ALPHAS = (0.05, 0.01)

# (name, scored H1 set, scored H0 set, FIT contrast or None).
# FIT contrast None == fit on the scored contrast itself. All contrasts draw from ONE shared
# RNG stream in list order, so entries are only ever appended: inserting one earlier would
# shift every later draw.
MAIN_FIT = ({20.0}, {1.0})
CONTRASTS = [
    ("w20_vs_1", {20.0}, {1.0}, None),
    ("w5_vs_3", {5.0}, {3.0}, None),
    # Table I is the main-protocol table, so its weaker-contrast same-cell rows are scored
    # with the deployed head, i.e. the one fitted at w=20 vs 1, not a head refitted on the
    # weaker contrast. These stay after w5_vs_3 so the first two contrasts keep their exact
    # RNG draws.
    ("w5_vs_3_mainfit", {5.0}, {3.0}, MAIN_FIT),
    ("w10_vs_3_mainfit", {10.0}, {3.0}, MAIN_FIT),
    ("hi_vs_low_mainfit", {10.0, 14.0, 20.0}, {1.0, 2.0, 3.0}, MAIN_FIT),
]
N_BLOCKBOOT = 200  # outer prompt-block bootstrap replicates (refit LW each time)
READOUTS = DEPLOYED_READOUTS
RLABEL = {"rho": "rho", "zsum_dep": "deployed z-sum", "lda_dep": "deployed LW-LDA"}


def _two_sample_auc(h0: np.ndarray, h1: np.ndarray) -> float:
    scores = np.concatenate([h0, h1])
    labels = np.concatenate([np.zeros(len(h0)), np.ones(len(h1))])
    return float(auc(scores, labels))


def _deflection(a: float) -> float:
    """Effective separation sqrt(2) * Phi^{-1}(AUC) (Eq. (12) inverted); AUC clipped first."""
    return float(np.sqrt(2.0) * norm.ppf(min(max(a, 0.5001), 0.9999)))


def _fit_exponent(ks: list[int], aucs: list[float | None]) -> float | None:
    """Log-log slope of separation vs K over unsaturated K (0.5 < AUC < 0.995); Eq. (14): 0.5."""
    xs = [k for k, a in zip(ks, aucs, strict=True) if a is not None and 0.5 < a < 0.995]
    ds = [
        _deflection(a) for k, a in zip(ks, aucs, strict=True) if a is not None and 0.5 < a < 0.995
    ]
    if len(xs) < 2:
        return None
    return float(np.polyfit(np.log(xs), np.log(ds), 1)[0])


def _cell_deployed_scores(
    cell_rows: list[dict],
    h1_set: set,
    h0_set: set,
    readout: str,
    fit_contrast: tuple[set, set] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    """Deployed-recipe scores: (fit_h0, test_h0, test_h1, fit_h1).

    Mirrors pair_readout_battery._fit_stats_2d/_apply_fusion (fusion_lda_full): H0-only
    per-coordinate mean/std (ddof=1) on FIT, standardized H1 shift, Ledoit--Wolf H0
    covariance, weights w_full = Sigma_LW^{-1} shift (LDA) or sign(shift) (z-sum). No
    per-coordinate sign flip is applied for the pair readouts; rho uses the FIT sign as in
    the deployed scalar map.
    """
    f1_set, f0_set = fit_contrast if fit_contrast is not None else (h1_set, h0_set)
    fit = fit_deployed(cell_rows, f1_set, f0_set, readout)
    if fit is None:
        return None
    return (
        fit.fit_h0,
        score_deployed(cell_rows, h0_set, "test", "h0", fit),
        score_deployed(cell_rows, h1_set, "test", "h1", fit),
        fit.fit_h1,
    )


def _group_auc_nonoverlap(
    h0: np.ndarray, h1: np.ndarray, k: int, n_part: int, rng: np.random.Generator
) -> float | None:
    """Same-cell AUC_K: per class, permute, cut floor(n/K) non-overlapping groups, compare means.

    Averaged over n_part random partitions; None when either class has fewer than two groups.
    """
    if k < 1 or n_part < 1:
        raise ValueError(f"k and n_part must be positive, got k={k}, n_part={n_part}")
    g0, g1 = len(h0) // k, len(h1) // k
    if min(g0, g1) < 2:
        return None
    vals = []
    for _ in range(n_part):
        p0 = rng.permutation(len(h0))[: g0 * k].reshape(g0, k)
        p1 = rng.permutation(len(h1))[: g1 * k].reshape(g1, k)
        vals.append(_two_sample_auc(h0[p0].mean(1), h1[p1].mean(1)))
    return float(np.mean(vals))


def _cell_fpr_tpr(fit_h0: np.ndarray, test_h0: np.ndarray, test_h1: np.ndarray) -> dict:
    """Descriptive K=1 FPR/TPR on TEST at the FIT-null (1 - alpha) quantile threshold."""
    out = {}
    for a in ALPHAS:
        thr = float(np.quantile(fit_h0, 1.0 - a))
        out[f"fpr@{a}"] = float(np.mean(test_h0 > thr))
        out[f"tpr@{a}"] = float(np.mean(test_h1 > thr))
        out[f"n_h0@{a}"] = int(len(test_h0))
    return out


def _cell_blockboot_k16(
    cell_rows: list[dict],
    h1_set: set,
    h0_set: set,
    readout: str,
    n_boot: int,
    rng: np.random.Generator,
    fit_contrast: tuple[set, set] | None = None,
) -> list[float] | None:
    """Outer prompt-block bootstrap of the K=16 non-overlap AUC.

    Prompts are the independent units within a cell (single blur level, no raw-generation
    siblings; splits are prompt-disjoint). Each replicate resamples prompts with replacement
    *within the FIT and TEST strata*. Every selected prompt carries all of its w rows, so H0/H1
    prompt pairing is preserved while the deployed recipe is refit and the K=16 non-overlap AUC
    is recomputed. This propagates FIT-parameter and prompt-sampling uncertainty that within-pool
    partition averaging cannot.
    """
    if n_boot < 1:
        raise ValueError(f"n_boot must be positive, got {n_boot}")
    by_split_prompt: dict[str, dict[int, list[dict]]] = {"train": {}, "test": {}}
    prompt_split: dict[int, str] = {}
    for r in cell_rows:
        split = r["split"]
        if split not in by_split_prompt:
            continue
        prompt = r["prompt"]
        previous = prompt_split.setdefault(prompt, split)
        if previous != split:
            raise ValueError(f"prompt {prompt} spans splits {previous!r}/{split!r}")
        by_split_prompt[split].setdefault(prompt, []).append(r)
    if not all(by_split_prompt.values()):
        return None
    vals: list[float] = []
    for _ in range(n_boot):
        rows_b: list[dict] = []
        for split in ("train", "test"):
            blocks = by_split_prompt[split]
            prompts = sorted(blocks)
            draw = rng.integers(0, len(prompts), size=len(prompts))
            rows_b.extend(r for j in draw for r in blocks[prompts[j]])
        sc = _cell_deployed_scores(rows_b, h1_set, h0_set, readout, fit_contrast)
        if sc is None:
            continue
        _, t0, t1, _ = sc
        a = _group_auc_nonoverlap(t0, t1, 16, 8, rng)
        if a is not None:
            vals.append(a)
    return vals if len(vals) >= max(20, n_boot // 4) else None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default=None, help="dir of <decoder>/ p0 outputs (def outputs/p0)")
    ap.add_argument("--n-part", type=int, default=2000, help="non-overlap partitions per K")
    ap.add_argument("--n-cellboot", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    if args.n_part < 1 or args.n_cellboot < 1:
        raise SystemExit("--n-part and --n-cellboot must both be positive")
    # One shared generator for every partition and prompt-block draw, consumed in
    # CONTRASTS x READOUTS x sorted-cell order.
    rng = np.random.default_rng(args.seed)
    root = Path(args.root) if args.root else outputs_dir() / "p0"
    all_rows = load_residual_rows(root)

    cells: dict[tuple[str, float], list[dict]] = {}
    for r in all_rows:
        cells.setdefault((r["decoder"], r["sigma_blur"]), []).append(r)

    result: dict[str, dict] = {}
    bb_lists: dict[str, list[list[float]]] = {}
    for cname, h1_set, h0_set, fit_contrast in CONTRASTS:
        for readout in READOUTS:
            per_cell: dict[str, dict] = {}
            for (dec, sb), cr in sorted(cells.items()):
                sc = _cell_deployed_scores(cr, h1_set, h0_set, readout, fit_contrast)
                if sc is None:
                    continue
                fit_h0, t0, t1, _ = sc
                if len(t0) < 2 * max(GROUP_K) or len(t1) < 2 * max(GROUP_K):
                    continue
                curve = {k: _group_auc_nonoverlap(t0, t1, k, args.n_part, rng) for k in GROUP_K}
                bb = _cell_blockboot_k16(
                    cr, h1_set, h0_set, readout, N_BLOCKBOOT, rng, fit_contrast
                )
                if bb:
                    bb_lists.setdefault(f"{readout}|{cname}", []).append(bb)
                per_cell[f"{dec}_sb{sb:g}"] = {
                    "auc1": _two_sample_auc(t0, t1),
                    "curve": curve,
                    "exponent": _fit_exponent(GROUP_K, [curve[k] for k in GROUP_K]),
                    "k16_blockboot_ci95": (
                        [float(np.percentile(bb, 2.5)), float(np.percentile(bb, 97.5))]
                        if bb
                        else None
                    ),
                    "k16_blockboot_n": len(bb) if bb else 0,
                    **_cell_fpr_tpr(fit_h0, t0, t1),
                }
            if len(per_cell) < 2:
                continue
            k16 = np.array(
                [c["curve"][16] for c in per_cell.values() if c["curve"][16] is not None]
            )
            # Cell-cluster bootstrap of the K=16 macro: resample per-cell values (seed 1).
            rb = np.random.default_rng(1)
            boot = np.array(
                [rb.choice(k16, size=k16.size, replace=True).mean() for _ in range(args.n_cellboot)]
            )
            exps = [c["exponent"] for c in per_cell.values() if c["exponent"] is not None]
            result[f"{readout}|{cname}"] = {
                "n_cells": len(per_cell),
                "macro_auc1": float(np.mean([c["auc1"] for c in per_cell.values()])),
                "k16_within_macro": float(k16.mean()),
                "k16_within_ci95_cellboot": [float(x) for x in np.percentile(boot, [2.5, 97.5])],
                "k16_within_worst": float(k16.min()),
                "exponent_median": float(np.median(exps)) if exps else None,
                "fpr@0.05_macro": float(np.mean([c["fpr@0.05"] for c in per_cell.values()])),
                "fpr@0.01_macro": float(np.mean([c["fpr@0.01"] for c in per_cell.values()])),
                "tpr@0.05_macro": float(np.mean([c["tpr@0.05"] for c in per_cell.values()])),
                "tpr@0.01_macro": float(np.mean([c["tpr@0.01"] for c in per_cell.values()])),
                "per_cell": per_cell,
            }
            bbs = bb_lists.get(f"{readout}|{cname}")
            if bbs and len(bbs) >= 2:
                # Two-level bootstrap: resample cells with replacement, then draw one outer
                # prompt-block replicate (LW refit) within each drawn cell. A separate generator
                # (fixed seed 20260821) leaves the per-cell and cell-bootstrap streams unchanged.
                rt = np.random.default_rng(20260821)
                mac = []
                for _ in range(args.n_cellboot):
                    idx = rt.integers(0, len(bbs), len(bbs))
                    mac.append(
                        float(np.mean([bbs[i][int(rt.integers(0, len(bbs[i])))] for i in idx]))
                    )
                result[f"{readout}|{cname}"]["k16_within_ci95_twolevel"] = [
                    float(x) for x in np.percentile(mac, [2.5, 97.5])
                ]
                result[f"{readout}|{cname}"]["twolevel_n_cells"] = len(bbs)
                result[f"{readout}|{cname}"]["twolevel_note"] = (
                    "cell resample x per-cell prompt-block replicate (LW refit each): "
                    "propagates cell, prompt-sampling, and FIT-parameter uncertainty "
                    "jointly; descriptive"
                )

    if not result:
        raise SystemExit("no contrast/readout had at least two valid cells; refusing empty output")

    out_path = (
        Path(args.out) if args.out else outputs_dir() / "p0" / "deployed_within_contrasts.json"
    )
    write_json_strict(
        out_path,
        {"n_part": args.n_part, "group_k": GROUP_K, "alphas": list(ALPHAS), "results": result},
    )

    print("\n=== DEPLOYED readout, within-cell non-overlap (theorem-matched) ===")
    hdr = (
        f"{'readout':>16} {'contrast':>9} | {'cells':>5} {'AUC1':>6} | "
        f"{'K16 within':>10} {'cell-boot 95% CI':>18} {'worst':>6} {'exp med':>7} | "
        f"{'FPR@5%':>6} {'FPR@1%':>6} {'TPR@1%':>6}"
    )
    print(hdr)
    print("-" * len(hdr))
    for key, r in result.items():
        ro, cn = key.split("|")
        ci = r["k16_within_ci95_cellboot"]
        em = r["exponent_median"]
        em_s = f"{em:.3f}" if em is not None else "  -  "
        print(
            f"{RLABEL[ro]:>16} {cn:>9} | {r['n_cells']:>5} {r['macro_auc1']:>6.3f} | "
            f"{r['k16_within_macro']:>10.3f} [{ci[0]:.3f}, {ci[1]:.3f}]{'':>2} "
            f"{r['k16_within_worst']:>6.3f} {em_s:>7} | "
            f"{r['fpr@0.05_macro']:>6.3f} {r['fpr@0.01_macro']:>6.3f} {r['tpr@0.01_macro']:>6.3f}"
        )
    print(
        "\n[deployed_within] FPR/TPR are single-image, per-cell FIT-H0 quantile thresholds "
        "scored on held-out TEST (macro over cells; diagnostic, no one-sided guarantee)."
    )
    print(
        "[deployed_within] per-cell k16_blockboot_ci95 (JSON) = outer prompt-block bootstrap "
        f"({N_BLOCKBOOT} replicates, LW refit per draw); K=16 has only 3-6 independent "
        "groups/class/cell, so tails stay unresolved by design."
    )
    print(f"[deployed_within] wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
