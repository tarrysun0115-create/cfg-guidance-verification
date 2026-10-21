"""Pair-readout battery: (rho, m) fusion readouts on the main grid and their selection-free audits.

Builds per-image values for the two-coordinate summary y(x) = (rho(x), m(x)) of Section III-C
(Eq. (8); the candidate ``diff_energy`` is m, the texture-masked, winsorized patch mean of
||r_cond,p - r_uncond,p||) and runs the existing single-channel machinery on them unchanged.
Candidates:

* ``rho``, ``diff_energy``, ``signed_diff`` (patch mean of sign(rho_p) * ||dr_p||), ``abs_rho``;
* ``fusion_zsum``      -- T_pair-zsum: FIT-H0-standardized coordinates summed with FIT signs;
* ``fusion_lda_diag``  -- diagonal LDA: standardized coordinates weighted by their FIT H1 shifts;
* ``fusion_lda_full``  -- T_pair-LDA, the deployed Pair-LDA head (Section III-D, Eq. (11)):
  weights Sigma^-1 mu, with Sigma the Ledoit-Wolf covariance of the standardized FIT-H0 pair and
  mu the mean standardized FIT-H1 shift; the cellwise additive constant of Eq. (11) is omitted;
* ``fusion_zsum_low3refit`` -- T_pair-zsum refitted on w in {1,2,3} vs {10,14,20}.

The three fusion heads are fitted per evaluation cell (decoder, sigma_blur) on the FIT split
(``split == "train"``) at w=20 vs w=1 (``_fit_stats_2d``) and then held fixed for all rows of
that cell. Each candidate is written into a virtual copy of every ``p0_table.json`` (only
``rho_summary`` replaced) and these runners are reused on the copies unchanged:
``fixed_h1_allcell`` (selection-free anchors for w=20 vs 1, {10,14,20} vs {1,2,3} and w=10 vs 3:
pooled/macro AUC1, TPR at 1%/5% FPR, all-cell label permutation), ``sqrtk_pool_curves``
(cross-cell K=16 bootstrap CI, prompt-disjoint check, leave-one-decoder-out curves),
``run_whitebox_lr`` (FPR operating map), ``cross_decoder`` (FPR transfer across decoders),
``pooled_robustness`` (alive-pool audit) and the PSF scorecard of ``feature_comparison``
(skipped when the sigma_calib artifacts are absent, e.g. under --smoke). A FIT-only selection
audit picks the best candidate by FIT AUC and reports its TEST AUC. Residualization against
CLIP and physical statistics is not run here (see pair_distinctness.py); for the
post-processing pool only the availability of residual stores is checked.

Produces:
    Table I      Pair-LDA rows w=10 vs 3 and {10,14,20} vs {1,2,3}: pooled AUC1, macro AUC1,
                 TPR@1%/5% (``fixed_h1_fusion_lda_full.json``:
                 ``variants/{w3_vs_w10,low3_vs_high10}/all_cell``).
    Section IV-B all-cell label-permutation p of the pair readout
                 (``fixed_h1_fusion_lda_full.json``: ``variants/w20/all_cell/auc1_perm_p``;
                 500 permutations, floor 2e-3).
    Table III    Pair-LDA leave-one-decoder-out range (``pair_readout_battery.json``:
                 ``candidates/fusion_lda_full/leave_decoder_out``).
    The per-cell fusion fit and virtual tables (``_compute_all_features``) are also rebuilt
    through this module by the other pair scripts (weak_contrast_rows.py,
    pair_variant_ci_matched.py, pair_ablation_k16.py, pair_distinctness.py,
    prompt_mismatch_pair.py, stored_score_battery.py, retarget_analyze.py,
    unified_baselines.py, baseline_incremental.py, renoising_group_compare.py,
    gpu_probe_analyze.py).

Inputs: ``outputs/p0/<decoder>/p0_table.json`` plus each decoder's residual feature store
(``residual_v1.dat``, ``compact_rho.npy``) in the same directory, read through
``candidate_readouts._per_image_features``, which aborts unless the recomputed rho reproduces
the stored ``rho_summary`` to 1e-3. The feature store is server-side and not shipped. CPU only;
no GPU, images or models. ``--smoke`` runs end to end on a synthetic store.

Outputs: ``outputs/p0/pair_readouts/`` (or ``--out``): ``pair_readout_battery.json``,
``fixed_h1_<candidate>.json``, ``fixed_h1_fusion_zsum_low3refit.json``,
``fusion_2d_diagnostics.json`` (per-cell FIT means, stds, shifts, H0 correlation, weights), and
per candidate ``<candidate>/curves/``, ``<candidate>/whitebox/``, ``<candidate>/robustness/``
and ``<candidate>/cross_decoder.json``. Figure writers are disabled during the reused runs.

Usage:
    uv run --no-sync python scripts/analysis/pair_readout_battery.py \\
        [--features-root outputs/p0] [--out outputs/p0/pair_readouts] [--seed 0]
    uv run --no-sync python scripts/analysis/pair_readout_battery.py --smoke
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import tempfile
from pathlib import Path

import numpy as np
from scipy import stats

from cfgid import viz
from cfgid.bounds.wstar_cov import ledoit_wolf_cov
from cfgid.numerics import finite_float
from cfgid.paths import outputs_dir
from cfgid.sweep.neff_scaling import _group_rows_by_cell, _load_p0_rows

FUSION_H1_W = 20
BASE_CANDIDATES = ("rho", "diff_energy", "signed_diff", "abs_rho")
FUSION_CANDIDATES = ("fusion_zsum", "fusion_lda_diag", "fusion_lda_full")
CANDIDATES = BASE_CANDIDATES + FUSION_CANDIDATES
ANCHOR_VARIANTS = {
    "w20": ((1,), (20,)),
    "low3_vs_high10": ((1, 2, 3), (10, 14, 20)),
    "w3_vs_w10": ((3,), (10,)),
}


def _load_sibling(name: str, subdir: str = ""):
    base = Path(__file__).parent if not subdir else Path(__file__).parents[2] / subdir
    path = base / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"{name}_reuse2", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_FM = _load_sibling("candidate_readouts")  # store loading + rho repro gate, smoke tree
_FH = _load_sibling("fixed_h1_allcell")
_XD = _load_sibling("cross_decoder")
_FC = _load_sibling("feature_comparison")
_WB = _load_sibling("run_whitebox_lr")
_PR = _load_sibling("pooled_robustness")
_SQ = _load_sibling("sqrtk_pool_curves", subdir="scripts/paper")


# --- stage 1: per-image candidate values -----------------------------------------------------


def _cell_key(r: dict) -> str:
    return f"{r['decoder']}/sb={float(r['sigma_blur']):g}"


def _fit_stats_2d(
    rows: list[dict],
    feats: dict[str, dict[str, float]],
    h0_ws: tuple[int, ...],
    h1_ws: tuple[int, ...],
) -> dict[str, dict]:
    """Per-cell 2-D FIT statistics on (rho, diff_energy): H0 mean/std per coordinate, H0
    correlation, standardized H1 shifts, and the three fusion weight vectors.

    Only FIT rows (``split == "train"``) enter. H0 = rows labelled h0 (when 1 is in ``h0_ws``)
    plus h1 rows whose w is in ``h0_ws``; H1 = h1 rows whose w is in ``h1_ws``. Cells with fewer
    than 8 rows per side are skipped. Weights act on the FIT-H0-standardized coordinates:
    ``w_zsum`` = FIT shift signs (T_pair-zsum), ``w_diag`` = the shifts themselves, ``w_full`` =
    LW-Sigma^-1 shift (T_pair-LDA).
    """
    pools: dict[str, dict[str, list[tuple[float, float]]]] = {}
    for r in rows:
        iid = str(r["image_id"])
        if iid not in feats["rho"] or r.get("split") != "train":
            continue
        w = int(r["w"])
        lab = str(r.get("label"))
        side = "h0" if (lab == "h0" and 1 in h0_ws) or (lab == "h1" and w in h0_ws) else (
            "h1" if lab == "h1" and w in h1_ws else ""
        )
        if not side:
            continue
        pools.setdefault(_cell_key(r), {}).setdefault(side, []).append(
            (feats["rho"][iid], feats["diff_energy"][iid])
        )
    out: dict[str, dict] = {}
    for key, sides in pools.items():
        h0 = np.asarray(sides.get("h0", []), dtype=np.float64)
        h1 = np.asarray(sides.get("h1", []), dtype=np.float64)
        if h0.shape[0] < 8 or h1.shape[0] < 8:
            continue
        m = h0.mean(axis=0)
        s = h0.std(axis=0, ddof=1)
        if not (np.all(np.isfinite(s)) and np.all(s > 0)):
            continue
        # standardize both classes by the FIT-H0 mean/std (ddof=1) of each coordinate
        z0 = (h0 - m) / s
        z1 = (h1 - m) / s
        shift = z1.mean(axis=0)  # standardized H1 shifts (d_r, d_d)
        corr = float(np.corrcoef(z0[:, 0], z0[:, 1])[0, 1]) if z0.shape[0] > 2 else 0.0
        w_zsum = np.array([np.sign(shift[0]) or 1.0, np.sign(shift[1]) or 1.0])  # 0 -> +1
        w_diag = shift.copy()
        # Ledoit-Wolf shrinkage covariance of the standardized FIT-H0 pair (Section III-D)
        cov, lw_shrink = ledoit_wolf_cov(z0)
        w_full = np.linalg.solve(cov, shift)
        out[key] = {
            "m": m.tolist(),
            "s": s.tolist(),
            "shift": shift.tolist(),
            "h0_corr": corr,
            "lw_shrinkage": float(lw_shrink),
            "w_zsum": w_zsum.tolist(),
            "w_diag": w_diag.tolist(),
            "w_full": w_full.tolist(),
        }
    return out


def _apply_fusion(
    rows: list[dict], feats: dict[str, dict[str, float]], stats2d: dict[str, dict], weight_key: str
) -> dict[str, float]:
    """Score every row of a fitted cell: weights . ((rho, m) - FIT-H0 mean) / FIT-H0 std."""
    fused: dict[str, float] = {}
    for r in rows:
        iid = str(r["image_id"])
        key = _cell_key(r)
        if iid not in feats["rho"] or key not in stats2d:
            continue
        st = stats2d[key]
        y = np.array([feats["rho"][iid], feats["diff_energy"][iid]])
        z = (y - np.asarray(st["m"])) / np.asarray(st["s"])
        fused[iid] = float(np.dot(np.asarray(st[weight_key]), z))
    return fused


def _compute_all_features(
    features_root: Path, work_root: Path
) -> tuple[dict[str, Path], dict, dict]:
    """Stage 1: per-image values for all candidates + the 2-D diagnostics + virtual tables."""
    roots = {name: work_root / name for name in CANDIDATES}
    roots["fusion_zsum_low3refit"] = work_root / "fusion_zsum_low3refit"
    diag2d: dict[str, dict] = {}
    counts: dict[str, int] = {}
    for dec_dir in sorted(d for d in features_root.glob("*/") if (d / "p0_table.json").exists()):
        rows = _load_p0_rows(dec_dir / "p0_table.json")
        feats = _FM._per_image_features(dec_dir, rows)  # rho/diff/signed/img_signed + repro gate
        feats["abs_rho"] = {i: abs(v) for i, v in feats["rho"].items()}
        # deployed heads are fitted at w=20 vs w=1; the low3 refit only feeds the refit-gap check
        st_w20 = _fit_stats_2d(rows, feats, (1,), (FUSION_H1_W,))
        st_low3 = _fit_stats_2d(rows, feats, (1, 2, 3), (10, 14, 20))
        diag2d[dec_dir.name] = st_w20
        feats["fusion_zsum"] = _apply_fusion(rows, feats, st_w20, "w_zsum")
        feats["fusion_lda_diag"] = _apply_fusion(rows, feats, st_w20, "w_diag")
        feats["fusion_lda_full"] = _apply_fusion(rows, feats, st_w20, "w_full")
        feats["fusion_zsum_low3refit"] = _apply_fusion(rows, feats, st_low3, "w_zsum")
        for name, root in roots.items():
            fmap = feats[name]
            clone = []
            for r in rows:
                iid = str(r["image_id"])
                if iid not in fmap:
                    continue
                rr = dict(r)
                rr["rho_summary"] = fmap[iid]
                clone.append(rr)
            counts[name] = counts.get(name, 0) + len(clone)
            out_dir = root / dec_dir.name
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / "p0_table.json").write_text(
                json.dumps(clone, allow_nan=False), encoding="utf-8"
            )
        print(f"[stage1] {dec_dir.name}: {len(rows)} rows -> {len(roots)} virtual tables")
    return roots, diag2d, counts


# --- audit helpers -----------------------------------------------------------------------------


def _dist_diagnostics(root: Path) -> dict:
    """KS vs N(0,1) on the pooled FIT-standardized held-out (TEST) null, plus moments."""
    rows: list[dict] = []
    for tp in sorted(root.glob("*/p0_table.json")):
        rows.extend(_load_p0_rows(tp))
    cells = _group_rows_by_cell(rows)
    recs = _FH.collect_fixed_h1_records(cells, (FUSION_H1_W,), min_n=8, h0_ws=(1,))
    if not recs:
        return {"n_cells": 0}
    _zf0, _zf1, zt0, zt1 = _FH._aligned_pools(recs)
    ks = stats.kstest(zt0, "norm")
    return {
        "n_cells": len(recs),
        "n_h0": int(zt0.size),
        "ks_D": float(ks.statistic),
        "ks_p": float(ks.pvalue),
        "h0_skew": float(stats.skew(zt0)),
        "h0_excess_kurtosis": float(stats.kurtosis(zt0)),
        "h1_skew": float(stats.skew(zt1)),
        "h0_h1_var_ratio": float(np.var(zt1, ddof=1) / np.var(zt0, ddof=1)),
    }


def _fit_only_selection_audit(roots: dict[str, Path]) -> dict:
    """Choose the best readout on FIT ONLY (w=20 vs 1), then report its TEST (EVAL) AUC."""
    from cfgid.detector.empirical import auc

    fit_auc: dict[str, float] = {}
    eval_auc: dict[str, float] = {}
    for name in CANDIDATES:
        rows: list[dict] = []
        for tp in sorted(roots[name].glob("*/p0_table.json")):
            rows.extend(_load_p0_rows(tp))
        cells = _group_rows_by_cell(rows)
        recs = _FH.collect_fixed_h1_records(cells, (FUSION_H1_W,), min_n=8, h0_ws=(1,))
        if not recs:
            continue
        zf0, zf1, zt0, zt1 = _FH._aligned_pools(recs)
        fit_auc[name] = float(
            auc(
                np.concatenate([zf1, zf0]),
                np.concatenate([np.ones(zf1.size, int), np.zeros(zf0.size, int)]),
            )
        )
        eval_auc[name] = float(
            auc(
                np.concatenate([zt1, zt0]),
                np.concatenate([np.ones(zt1.size, int), np.zeros(zt0.size, int)]),
            )
        )
    if not fit_auc:
        return {}
    winner = max(fit_auc, key=lambda k: fit_auc[k])
    return {
        "fit_auc": fit_auc,
        "eval_auc": eval_auc,
        "fit_selected": winner,
        "fit_selected_eval_auc": eval_auc[winner],
        "note": (
            "candidate chosen on FIT only; its EVAL row is the held-out number for the selected "
            "readout"
        ),
    }


def _psf_scorecard(features_root: Path, name: str, root: Path) -> dict:
    """PSF + detection scorecard via feature_comparison's own ``_scorecard`` (fail-soft)."""
    try:
        rows_all: list[dict] = []
        fmap: dict[str, float] = {}
        for tp in sorted(root.glob("*/p0_table.json")):
            rows = _load_p0_rows(tp)
            rows_all.extend(rows)
            for r in rows:
                fmap[str(r["image_id"])] = finite_float("v", r["rho_summary"])
        # the ORIGINAL rows (with real rho) provide splits/labels; values come from fmap
        orig_rows: list[dict] = []
        for tp in sorted(features_root.glob("*/p0_table.json")):
            orig_rows.extend(_load_p0_rows(tp))
        se_by = _FC._PGF._sigma_eff_by_cell(features_root)
        axis_modes = _FC._PGF._sigma_axis_modes(features_root, orig_rows, se_by)
        return _FC._scorecard(
            features_root,
            orig_rows,
            fmap,
            se_by,
            axis_modes,
            FUSION_H1_W,
            is_signed_channel=name in ("rho", "signed_diff") or name.startswith("fusion"),
            n_groups=2000,
            n_perm=100,
        )
    except (FileNotFoundError, KeyError, ValueError, SystemExit) as exc:
        return {"skipped": f"psf scorecard unavailable here: {exc}"}


def _postproc_store_preflight(features_root: Path) -> dict:
    """Preflight for the post-processing pool: does any robustness root carry residual stores?"""
    hits = []
    for pattern in ("../p0_robustness*", "../robustness*", "../p0_postproc*"):
        for d in features_root.glob(pattern):
            n = len(list(d.glob("*/residual_v1.dat")))
            hits.append({"root": str(d), "decoder_stores": n})
    return {
        "checked_patterns": ["p0_robustness*", "robustness*", "p0_postproc*"],
        "hits": hits,
        "note": "fusion post-processing column is pure reanalysis iff a robustness root carries "
        "residual_v1 stores; otherwise it stays rho-only and scoped.",
    }


def run(features_root: Path, out_dir: Path, work_root: Path, *, seed: int = 0) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    roots, diag2d, counts = _compute_all_features(features_root, work_root)
    (out_dir / "fusion_2d_diagnostics.json").write_text(
        json.dumps(diag2d, indent=2, allow_nan=False), encoding="utf-8"
    )

    viz_save = viz.save_figure
    viz.save_figure = lambda *a, **k: None  # type: ignore[assignment]
    xd_fig = _XD._figure
    _XD._figure = lambda *a, **k: None
    results: dict[str, dict] = {}
    try:
        for name in CANDIDATES:
            entry: dict = {"n_rows": counts.get(name)}
            # selection-free anchors on three contrasts (fixed_h1_allcell; 500 permutations)
            anchors_path = out_dir / f"fixed_h1_{name}.json"
            _FH.run(
                roots[name],
                anchors_path,
                variants=dict(ANCHOR_VARIANTS),
                n_groups=4000,
                n_perm=500,
                min_n=8,
                seed=seed,
            )
            a = json.loads(anchors_path.read_text(encoding="utf-8"))
            entry["anchors"] = {k: v["all_cell"] for k, v in a["variants"].items()}
            # K16 CI + prompt-disjoint + leave-decoder-out (sqrtk_pool_curves reuse)
            curves_dir = out_dir / name / "curves"
            curves_dir.mkdir(parents=True, exist_ok=True)
            rc = _SQ.main(
                [
                    "--features-root",
                    str(roots[name]),
                    "--out-dir",
                    str(curves_dir),
                    "--seed",
                    str(seed),
                ]
            )
            cj = curves_dir / "sqrtk_pool_curves.json"
            if rc == 0 and cj.exists():
                c = json.loads(cj.read_text(encoding="utf-8"))
                entry["k16"] = c.get("k16_all_cell")
                entry["leave_decoder_out"] = c.get("leave_decoder_out")
            # H0/H1 distribution diagnostics
            entry["dist"] = _dist_diagnostics(roots[name])
            # white-box FPR operating map
            wb_dir = out_dir / name / "whitebox"
            try:
                _WB.run_whitebox_lr(
                    roots[name],
                    wb_dir,
                    alphas=(0.1, 0.05, 0.01, 0.001),
                    n_eff_grid=(1, 4, 16),
                    n_groups=8000,
                    seed=seed,
                )
                wj = wb_dir / "whitebox_lr.json"
                if wj.exists():
                    recs_wb = json.loads(wj.read_text(encoding="utf-8"))
                    entry["whitebox_summary"] = {
                        "n_rows": len(recs_wb),
                        "n_fpr_controlled": sum(1 for r in recs_wb if r.get("fpr_controlled")),
                        "n_power_matched": sum(1 for r in recs_wb if r.get("power_matches")),
                    }
            except (ValueError, KeyError, SystemExit) as exc:
                entry["whitebox_summary"] = {"skipped": str(exc)}
            # cross-decoder FPR transfer
            _XD.run(
                roots[name],
                out_dir / name,
                alpha=0.01,
                alive_t=1.5,
                fpr_tol=0.01,
                n_groups=20000,
                seed=seed,
            )
            xj = out_dir / name / "cross_decoder.json"
            entry["transfer"] = _FM._transfer_summary(xj) if xj.exists() else None
            # alive-pool audit (leave-one-cell-out, threshold sweep)
            pr_dir = out_dir / name / "robustness"
            try:
                _PR.run(
                    roots[name],
                    pr_dir,
                    alive_t=1.5,
                    grid=(0.5, 1.0, 1.5, 2.0, 2.5),
                    n_perm=1000,
                    seed=seed,
                )
                pj = pr_dir / "pooled_robustness.json"
                if pj.exists():
                    p = json.loads(pj.read_text(encoding="utf-8"))
                    entry["alive_audit"] = {
                        k: p[k]
                        for k in ("n_alive_cells", "loco_min_auc", "threshold_sweep")
                        if k in p
                    }
            except (ValueError, KeyError, SystemExit) as exc:
                entry["alive_audit"] = {"skipped": str(exc)}
            # PSF scorecard (fail-soft without sigma_calib)
            entry["psf_scorecard"] = _psf_scorecard(features_root, name, roots[name])
            entry["sign_pattern"] = _FM._sign_pattern(roots[name])
            results[name] = entry
        # fixed-vs-refit gap for fusion under the low contrast
        refit_path = out_dir / "fixed_h1_fusion_zsum_low3refit.json"
        _FH.run(
            roots["fusion_zsum_low3refit"],
            refit_path,
            variants={"low3_vs_high10": ((1, 2, 3), (10, 14, 20))},
            n_groups=4000,
            n_perm=500,
            min_n=8,
            seed=seed,
        )
        refit = json.loads(refit_path.read_text(encoding="utf-8"))
        results["fusion_zsum_low3refit"] = {
            "anchors": {"low3_vs_high10": refit["variants"]["low3_vs_high10"]["all_cell"]}
        }
    finally:
        viz.save_figure = viz_save  # type: ignore[assignment]
        _XD._figure = xd_fig

    selection_audit = _fit_only_selection_audit(roots)
    postproc = _postproc_store_preflight(features_root)

    out = {
        "analysis": "pair_readout_battery",
        "seed": seed,
        "candidates": results,
        "fit_only_selection_audit": selection_audit,
        "postproc_store_preflight": postproc,
        "gates_note": (
            "Checks in this battery: anchors+CI+prompt-disjoint; H0 Gaussianity for "
            "tau(alpha,N_eff); matched FPR map; low-guidance definition (fixed-vs-refit). "
            "Distinctness/residualization is NOT run here (needs the CLIP checkpoint; see "
            "pair_distinctness.py). Prompt mismatch needs a wrong-prompt re-inversion (see "
            "prompt_mismatch_pair.py)."
        ),
    }
    out_path = out_dir / "pair_readout_battery.json"
    out_path.write_text(json.dumps(out, indent=2, allow_nan=False), encoding="utf-8")

    print("\n=== pair readout battery (w20 anchors | dist | whitebox | transfer) ===")
    for name, e in results.items():
        a = e.get("anchors", {}).get("w20")
        if not a or a.get("n_cells", 0) == 0:
            print(f"{name:22s} (no w20 anchors)")
            continue
        d = e.get("dist", {})
        k16 = e.get("k16") or {}
        wb = e.get("whitebox_summary") or {}
        tr = e.get("transfer") or {}
        ci = k16.get("bootstrap_ci95")
        print(
            f"{name:22s} AUC1={a['auc1_pooled_fitsign']:.3f} "
            f"K16={k16.get('group_auc', float('nan')):.3f}"
            f" CI={'-' if not ci else f'[{ci[0]:.3f},{ci[1]:.3f}]'}"
            f" disjoint={k16.get('prompt_disjoint_auc', float('nan'))}"
            f" TPR1%={a['tpr_at_1pct_fpr_k1']:.3f}"
            f" KSp={d.get('ks_p', float('nan')):.3f}"
            f" fprCtrl={wb.get('n_fpr_controlled', 'n/a')}/{wb.get('n_rows', 'n/a')}"
            f" xfer={tr.get('n_cross_controlled', 'n/a')}/{tr.get('n_cross_entries', 'n/a')}"
        )
    if selection_audit:
        print(
            f"\nFIT-only selection audit: picks {selection_audit['fit_selected']!r} "
            f"(FIT {selection_audit['fit_auc'][selection_audit['fit_selected']]:.3f}) -> "
            f"EVAL {selection_audit['fit_selected_eval_auc']:.3f}"
        )
    print(f"postproc store preflight: {postproc['hits'] or 'no robustness root found here'}")
    print(f"-> {out_path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="pair_readout_battery")
    ap.add_argument(
        "--features-root", default=None, help="dir of <decoder>/ stores (def outputs/p0)"
    )
    ap.add_argument("--out", default=None, help="output dir (def outputs/p0/pair_readouts)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smoke", action="store_true", help="local synthetic self-test")
    args = ap.parse_args(argv)
    if args.smoke:
        tmp = Path(tempfile.mkdtemp(prefix="pair_readouts_smoke_"))
        try:
            _FM._smoke_tree(tmp / "root", seed=args.seed)
            return run(tmp / "root", tmp / "out", tmp / "work", seed=args.seed)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
    features_root = Path(args.features_root) if args.features_root else (outputs_dir() / "p0")
    out_dir = Path(args.out) if args.out else (outputs_dir() / "p0" / "pair_readouts")
    work_root = Path(tempfile.mkdtemp(prefix="pair_readouts_work_"))
    try:
        return run(features_root, out_dir, work_root, seed=args.seed)
    finally:
        shutil.rmtree(work_root, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
