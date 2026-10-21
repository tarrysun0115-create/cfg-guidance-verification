"""Same-configuration aggregation curves and raw-block pooled statistics (Fig. 3, Section IV-C).

Produces: outputs/p0/within_config_curves.json for the default contrast w=20 vs 1, the source of
Fig. 3 and of the Section IV-C exponent values for rho, the audit z-sum and Pair-Ridge-Audit
(readout keys ``rho`` / ``zsum`` / ``lda``): per-cell median exponents (``per_cell_exponent``:
``median_informative`` over cells with AUC1 >= 0.55, ``median_all`` with its cell-resampled 95%
interval), the single exponent fitted to the cell-averaged curve (``macro_exp_obs``), the same
fit applied to synthetic curves built with exponent 0.5 (``macro_exp_pred_same_estimator``), and
the Pair-Ridge-Audit cell-averaged K=16 AUC (``readouts.lda.macro_obs["16"]``, the
aligned-coordinate reconstruction of the deployed head). Fig. 3 is drawn from this file.

Part 1 (same-configuration sqrt-K law, Eq. (14)): for every (decoder, sigma_blur) cell, at the FIXED
contrast, build NON-OVERLAPPING mean-of-K groups strictly inside the cell for K = 1,2,4,8,16
(partition-averaged), together with the FIT-PREDICTED curve AUC_K = Phi(sqrt(K) * d_fit / sqrt(2))
from the FIT-split deflection. Reports per-cell observed curves, per-cell exponents (unsaturated K),
their median/IQR (all cells and the >=0.55-AUC1 "informative" subset), and the macro curve with BOTH
observed and FIT-predicted macro exponents -- the predicted macro exponent shows how much the
macro-of-AUCs estimator itself deflates 0.5, so a sub-0.5 macro number is not evidence against the
per-cell law.

Part 2 (sampling units): the four blur cells of a decoder are post-decode siblings of ONE raw
generation, so pooled constructions must cluster by raw generation. This script re-reports the
pooled fixed-H1 statistics with the correct units: a raw-block PAIRED permutation p (blocks =
(decoder, prompt); a block's H0 and H1 rows, including all blur siblings, swap together) and a
decoder-stratified raw-block OUTER bootstrap CI for the pooled AUC1, K=16 group AUC, and sqrt-K
exponent (fresh with-replacement groups inside each replicate).

Readouts: rho uses the p0_table ``rho_summary`` only; the audit z-sum and audit ridge-LDA
(Pair-Ridge-Audit) also need m. m is recomputed from the server residual store when
``residual_v1.dat`` is present; decoders without it are skipped for the pair readouts.
Ridge-LDA uses
FIT-H0 sample covariance plus ``1e-6 I`` and is not the deployed Ledoit--Wolf readout. Pure
reanalysis, no re-inversion.

Inputs: outputs/p0/<decoder>/p0_table.json plus the server-side feature store for the pair
readouts; CPU only.
Outputs: <root>/within_config_curves.json (root defaults to outputs/p0).
Usage:
    uv run --no-sync python scripts/analysis/within_config_curves.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import norm

from cfgid.detector.empirical import auc
from cfgid.featurestore import read_compact, read_residual_v1
from cfgid.paths import outputs_dir
from cfgid.sweep.neff_scaling import _load_p0_rows

K_GRID = [1, 2, 4, 8, 16]
RIDGE = 1e-6
MASK_Q = 0.2
WINSOR = (0.01, 0.99)
CONTRASTS = {
    "w20_vs_1": ({20.0}, {1.0}),
    "w10_vs_3": ({10.0}, {3.0}),
    "hi_vs_1": ({10.0, 14.0, 20.0}, {1.0}),
}


def _m_from_residual(residual_i: np.ndarray, e_tex_i: np.ndarray) -> float:
    """m(x) of Eq. (8): E_tex-masked, winsorized mean of the per-patch norms (float32 steps)."""
    from cfgid.channel.rho import etex_keep_mask, winsorize

    delta = np.linalg.norm(residual_i, axis=1)
    keep = etex_keep_mask(e_tex_i, MASK_Q)
    kept = delta[keep] if keep.any() else delta
    return float(np.mean(winsorize(kept.astype(np.float32), WINSOR[0], WINSOR[1])))


def _two_auc(h0: np.ndarray, h1: np.ndarray) -> float:
    """Empirical ROC AUC with H1 as the positive class."""
    scores = np.concatenate([h0, h1])
    labels = np.concatenate([np.zeros(len(h0)), np.ones(len(h1))])
    return float(auc(scores, labels))


def _dA(a: float) -> float:
    """Deflection d_A = sqrt(2) * Phi^-1(AUC), with AUC clipped to [0.5001, 0.9999]."""
    return float(np.sqrt(2.0) * norm.ppf(min(max(a, 0.5001), 0.9999)))


def _cell_scores_full(cell_rows: list[dict], h1_set: set, h0_set: set, readout: str):
    """FIT-orient/standardize (samecell_nonoverlap recipe); return fit/test z + test prompts."""

    def pop(w_set, split, key):
        return np.array(
            [r[key] for r in cell_rows if r["w"] in w_set and r["split"] == split],
            dtype=np.float64,
        )

    def prompts(w_set, split):
        return [r["prompt"] for r in cell_rows if r["w"] in w_set and r["split"] == split]

    keys = ("rho_summary",) if readout == "rho" else ("rho_summary", "m")
    # Per coordinate, from FIT (train) rows only: orientation = sign of the FIT H1-H0 mean
    # difference, center/scale = FIT-H0 mean and population std (ddof=0; 1.0 if zero).
    fit = {}
    for key in keys:
        h0, h1 = pop(h0_set, "train", key), pop(h1_set, "train", key)
        if len(h0) < 4 or len(h1) < 2:
            return None
        sign = 1.0 if h1.mean() >= h0.mean() else -1.0
        sd = float(h0.std())
        fit[key] = {"sign": sign, "mu": float(h0.mean()), "sd": sd if sd > 0 else 1.0}

    def z(w_set, split, key):
        f = fit[key]
        return f["sign"] * (pop(w_set, split, key) - f["mu"]) / f["sd"]

    if readout == "rho":
        zf0, zf1 = z(h0_set, "train", "rho_summary"), z(h1_set, "train", "rho_summary")
        zt0, zt1 = z(h0_set, "test", "rho_summary"), z(h1_set, "test", "rho_summary")
    elif readout == "zsum":
        zf0 = z(h0_set, "train", "rho_summary") + z(h0_set, "train", "m")
        zf1 = z(h1_set, "train", "rho_summary") + z(h1_set, "train", "m")
        zt0 = z(h0_set, "test", "rho_summary") + z(h0_set, "test", "m")
        zt1 = z(h1_set, "test", "rho_summary") + z(h1_set, "test", "m")
    else:  # lda: ridge Fisher direction on the oriented, standardized FIT pair
        x0 = np.vstack([z(h0_set, "train", "rho_summary"), z(h0_set, "train", "m")]).T
        x1 = np.vstack([z(h1_set, "train", "rho_summary"), z(h1_set, "train", "m")]).T
        if len(x0) < 4 or len(x1) < 2:
            return None
        w_lda = np.linalg.solve(np.cov(x0.T) + RIDGE * np.eye(2), x1.mean(0) - x0.mean(0))
        zf0, zf1 = x0 @ w_lda, x1 @ w_lda
        zt0 = np.vstack([z(h0_set, "test", "rho_summary"), z(h0_set, "test", "m")]).T @ w_lda
        zt1 = np.vstack([z(h1_set, "test", "rho_summary"), z(h1_set, "test", "m")]).T @ w_lda
    return {
        "zf0": zf0,
        "zf1": zf1,
        "zt0": zt0,
        "zt1": zt1,
        "pt0": prompts(h0_set, "test"),
        "pt1": prompts(h1_set, "test"),
    }


def _nonoverlap_curve(h0, h1, ks, n_part, rng) -> dict[int, float | None]:
    """Partition-averaged AUC_K: each partition splits every class into floor(n/K) disjoint
    groups of K (remainder dropped) and scores group means; averaged over n_part partitions."""
    out: dict[int, float | None] = {}
    for k in ks:
        g0, g1 = len(h0) // k, len(h1) // k
        if min(g0, g1) < 1:
            out[k] = None
            continue
        vals = []
        for _ in range(n_part):
            p0 = rng.permutation(len(h0))[: g0 * k].reshape(g0, k)
            p1 = rng.permutation(len(h1))[: g1 * k].reshape(g1, k)
            vals.append(_two_auc(h0[p0].mean(1), h1[p1].mean(1)))
        out[k] = float(np.mean(vals))
    return out


def _fit_exponent(curve: dict) -> float | None:
    """Slope of log d_A(AUC_K) on log K over unsaturated K (0.5001 < AUC_K < 0.99); needs 2 K."""
    ks: list[int] = []
    ds: list[float] = []
    for k in K_GRID:
        a = curve.get(k)
        if a is None or not (0.5001 < a < 0.99):
            continue
        d = _dA(float(a))
        if d <= 0:
            continue
        ks.append(k)
        ds.append(d)
    if len(ks) < 2:
        return None
    return float(np.polyfit(np.log(ks), np.log(ds), 1)[0])


def _group_auc_wr(h0, h1, k, n_groups, rng) -> float:
    """AUC of n_groups mean-of-K groups drawn with replacement from each class."""
    i0 = rng.integers(0, len(h0), size=(n_groups, k))
    i1 = rng.integers(0, len(h1), size=(n_groups, k))
    return _two_auc(h0[i0].mean(1), h1[i1].mean(1))


def _pooled_block_stats(cells_scores, n_perm, n_boot, seed) -> dict:
    """Raw-block corrected pooled stats (blocks = (decoder, prompt); blur siblings together)."""
    blocks: dict[tuple, dict] = {}
    for (dec, _sb), sc in cells_scores.items():
        for p, zv in zip(sc["pt0"], sc["zt0"], strict=True):
            blocks.setdefault((dec, p), {"h0": [], "h1": []})["h0"].append(float(zv))
        for p, zv in zip(sc["pt1"], sc["zt1"], strict=True):
            blocks.setdefault((dec, p), {"h0": [], "h1": []})["h1"].append(float(zv))
    keys = sorted(blocks)
    h0_all = np.array([v for k in keys for v in blocks[k]["h0"]])
    h1_all = np.array([v for k in keys for v in blocks[k]["h1"]])
    obs = _two_auc(h0_all, h1_all)

    # Paired block permutation: each (decoder, prompt) block swaps its H0 and H1 rows together
    # with probability 1/2; one-sided p = (1 + #{AUC_perm >= AUC_obs}) / (n_perm + 1).
    rng = np.random.default_rng(seed)
    perm_ge = 0
    for _ in range(n_perm):
        h0p, h1p = [], []
        flips = rng.random(len(keys)) < 0.5
        for kk, fl in zip(keys, flips, strict=True):
            b = blocks[kk]
            (h0p if not fl else h1p).extend(b["h0"])
            (h1p if not fl else h0p).extend(b["h1"])
        if _two_auc(np.array(h0p), np.array(h1p)) >= obs:
            perm_ge += 1
    perm_p = (1.0 + perm_ge) / (n_perm + 1.0)

    # Outer bootstrap: resample blocks with replacement within each decoder (stratified);
    # inside a replicate the K-curve uses 1000 with-replacement groups per K.
    by_dec: dict[str, list] = {}
    for dec, p in keys:
        by_dec.setdefault(dec, []).append((dec, p))
    rb = np.random.default_rng(seed + 1)
    boots = {"auc1": [], "k16": [], "exp": []}
    for _ in range(n_boot):
        h0b, h1b = [], []
        for _dec, dkeys in by_dec.items():
            picks = rb.integers(0, len(dkeys), size=len(dkeys))
            for i in picks:
                b = blocks[dkeys[i]]
                h0b.extend(b["h0"])
                h1b.extend(b["h1"])
        h0b, h1b = np.array(h0b), np.array(h1b)
        boots["auc1"].append(_two_auc(h0b, h1b))
        curve = {k: _group_auc_wr(h0b, h1b, k, 1000, rb) for k in K_GRID}
        boots["k16"].append(curve[16])
        e = _fit_exponent(curve)
        if e is not None:
            boots["exp"].append(e)

    def ci(v):
        return [float(x) for x in np.percentile(v, [2.5, 97.5])] if v else None

    # Point curve on the full pooled sample with 8000 groups per K (own stream, seed + 2).
    rng2 = np.random.default_rng(seed + 2)
    point_curve = {k: _group_auc_wr(h0_all, h1_all, k, 8000, rng2) for k in K_GRID}
    return {
        "n_blocks": len(keys),
        "n_rows_h0": int(h0_all.size),
        "n_rows_h1": int(h1_all.size),
        "auc1": obs,
        "block_perm_p": perm_p,
        "pooled_curve_withrepl": {str(k): point_curve[k] for k in K_GRID},
        "k16_point": point_curve[16],
        "exp_point": _fit_exponent(point_curve),
        "auc1_ci95_blockboot": ci(boots["auc1"]),
        "k16_ci95_blockboot": ci(boots["k16"]),
        "exp_ci95_blockboot": ci(boots["exp"]),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default=None, help="dir of <decoder>/ p0 outputs (def outputs/p0)")
    ap.add_argument("--readouts", default="rho,zsum,lda")
    ap.add_argument("--contrast", default="w20_vs_1", choices=sorted(CONTRASTS))
    ap.add_argument("--n-part", type=int, default=2000, help="non-overlap partitions per cell/K")
    ap.add_argument("--n-perm", type=int, default=2000)
    ap.add_argument("--n-boot", type=int, default=500, help="raw-block outer bootstrap replicates")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    root = Path(args.root) if args.root else outputs_dir() / "p0"
    readouts = [r.strip() for r in args.readouts.split(",") if r.strip()]
    h1_set, h0_set = CONTRASTS[args.contrast]

    rows_by_dec: dict[str, list[dict]] = {}
    m_ok: dict[str, bool] = {}
    for dec_dir in sorted(p for p in root.iterdir() if (p / "p0_table.json").is_file()):
        rows = _load_p0_rows(dec_dir / "p0_table.json")
        for r in rows:
            r["decoder"] = dec_dir.name
            r["w"] = float(r["w"])
        if (dec_dir / "residual_v1.dat").is_file():
            # Server path: recompute m per row; store records pair with table rows by position.
            residual, _ = read_residual_v1(dec_dir)
            batch, _ = read_compact(dec_dir)
            if residual.shape[0] != len(rows) or batch.e_tex.shape[0] != len(rows):
                raise SystemExit(f"{dec_dir.name}: store/table length mismatch")
            for i, r in enumerate(rows):
                r["m"] = _m_from_residual(residual[i], batch.e_tex[i])
            m_ok[dec_dir.name] = True
        else:
            m_ok[dec_dir.name] = False
        rows_by_dec[dec_dir.name] = rows

    payload: dict = {
        "analysis": "within_config_curves",
        "contrast": args.contrast,
        "k_grid": K_GRID,
        "n_part": args.n_part,
        "protocol": "fixed-cell fixed-w non-overlapping mean-of-K (partition-averaged), FIT "
        "orientation/standardization per samecell_nonoverlap; pair readouts are audit "
        "reconstructions (ridge-LDA = FIT-H0 sample covariance + 1e-6 I, not deployed "
        "Ledoit-Wolf); FIT-predicted "
        "AUC_K=Phi(sqrt(K) d_fit/sqrt2); pooled stats re-reported with raw-generation blocks "
        "((decoder,prompt); the four post-decode blur siblings move together)",
        "readouts": {},
    }
    for readout in readouts:
        cells_scores: dict[tuple, dict] = {}
        for dec, rows in rows_by_dec.items():
            if readout != "rho" and not m_ok[dec]:
                print(f"[skip] {readout}/{dec}: residual store absent (m unavailable)")
                continue
            by_sb: dict[float, list[dict]] = {}
            for r in rows:
                by_sb.setdefault(float(r["sigma_blur"]), []).append(r)
            for sb, cr in sorted(by_sb.items()):
                sc = _cell_scores_full(cr, h1_set, h0_set, readout)
                if sc is not None:
                    cells_scores[(dec, sb)] = sc
        if not cells_scores:
            payload["readouts"][readout] = {"skipped": "no cells"}
            continue

        # One generator per readout, consumed in sorted cell order.
        rng = np.random.default_rng(args.seed)
        per_cell, exps, exps_alive = {}, [], []
        for key in sorted(cells_scores):
            sc = cells_scores[key]
            obs = _nonoverlap_curve(sc["zt0"], sc["zt1"], K_GRID, args.n_part, rng)
            # FIT deflection: mean oriented FIT-H1 score in FIT-H0 units (Eq. (14) prediction).
            d_fit = abs(float(np.mean(sc["zf1"])))
            pred = {k: float(norm.cdf(np.sqrt(k) * d_fit / np.sqrt(2.0))) for k in K_GRID}
            e_obs, e_pred = _fit_exponent(obs), _fit_exponent(pred)
            auc1 = obs.get(1)
            per_cell[f"{key[0]}/sb={key[1]:g}"] = {
                "auc1": auc1,
                "d_fit": d_fit,
                "obs": {str(k): obs[k] for k in K_GRID},
                "pred": {str(k): pred[k] for k in K_GRID},
                "exp_obs": e_obs,
                "exp_pred": e_pred,
                "n_test_h0": int(sc["zt0"].size),
                "n_test_h1": int(sc["zt1"].size),
            }
            if e_obs is not None:
                exps.append(e_obs)
                if auc1 is not None and auc1 >= 0.55:
                    exps_alive.append(e_obs)

        macro_obs = {
            k: float(
                np.mean(
                    [c["obs"][str(k)] for c in per_cell.values() if c["obs"][str(k)] is not None]
                )
            )
            for k in K_GRID
        }
        macro_pred = {
            k: float(np.mean([c["pred"][str(k)] for c in per_cell.values()])) for k in K_GRID
        }
        # Cell bootstrap of the median per-cell exponent (4000 replicates, seed + 7).
        rbc = np.random.default_rng(args.seed + 7)
        med_boot = (
            [float(np.median(rbc.choice(exps, size=len(exps), replace=True))) for _ in range(4000)]
            if exps
            else []
        )
        pooled = _pooled_block_stats(cells_scores, args.n_perm, args.n_boot, args.seed)

        entry = {
            "n_cells": len(per_cell),
            "per_cell": per_cell,
            "per_cell_exponent": {
                "median_all": float(np.median(exps)) if exps else None,
                "iqr_all": [float(x) for x in np.percentile(exps, [25, 75])] if exps else None,
                "n_all": len(exps),
                "median_informative": float(np.median(exps_alive)) if exps_alive else None,
                "n_informative": len(exps_alive),
                "median_ci95_cellboot": (
                    [float(x) for x in np.percentile(med_boot, [2.5, 97.5])] if med_boot else None
                ),
            },
            "macro_obs": {str(k): macro_obs[k] for k in K_GRID},
            "macro_pred": {str(k): macro_pred[k] for k in K_GRID},
            "macro_exp_obs": _fit_exponent(macro_obs),
            "macro_exp_pred_same_estimator": _fit_exponent(macro_pred),
            "pooled_block_corrected": pooled,
        }
        payload["readouts"][readout] = entry

        pc = entry["per_cell_exponent"]
        print(f"\n=== {readout} / {args.contrast} : within-configuration curves ===")
        print(f"{'cell':<24}{'AUC1':>7}{'K16obs':>8}{'K16pred':>9}{'exp_obs':>9}{'exp_pred':>9}")
        for cname, c in per_cell.items():
            eo = f"{c['exp_obs']:.3f}" if c["exp_obs"] is not None else "  --"
            ep = f"{c['exp_pred']:.3f}" if c["exp_pred"] is not None else "  --"
            print(
                f"{cname:<24}{c['obs']['1']:>7.3f}{c['obs']['16']:>8.3f}"
                f"{c['pred']['16']:>9.3f}{eo:>9}{ep:>9}"
            )
        print(
            f"per-cell exponent: median(all)={pc['median_all']}, IQR={pc['iqr_all']}, "
            f"median(informative AUC1>=0.55)={pc['median_informative']} "
            f"(n={pc['n_informative']}), cell-boot 95% CI {pc['median_ci95_cellboot']}"
        )
        print(f"macro curve obs {entry['macro_obs']} exp={entry['macro_exp_obs']}")
        print(
            f"macro curve PRED (same estimator) exp={entry['macro_exp_pred_same_estimator']} "
            "<- macro estimator deflation under an exact per-cell law"
        )
        print(
            f"pooled raw-block corrected: AUC1={pooled['auc1']:.4f} "
            f"perm_p={pooled['block_perm_p']:.5f} K16={pooled['k16_point']:.4f} "
            f"CI{pooled['k16_ci95_blockboot']} exp={pooled['exp_point']} "
            f"CI{pooled['exp_ci95_blockboot']} (blocks={pooled['n_blocks']})"
        )

    out = root / "within_config_curves.json"
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\n-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
