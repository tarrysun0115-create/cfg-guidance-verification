"""Raw-generation-block inference for the deployed readouts on 357 paired blocks (GPU-free).

Block-valid statistics of the deployed readouts, computed with the EXACT deployed estimator
(``_readout_common.fit_deployed``: FIT-H0 per-coordinate standardization, standardized FIT-H1
shift, per-cell Ledoit--Wolf H0 covariance, w = Sigma_LW^{-1} shift), not the audit ridge-LDA
reconstruction. Each raw generation yields four blur siblings, so rows are not independent.
Block construction and every statistic mirror ``within_config_curves._pooled_block_stats``
byte-for-byte:
* blocks = (decoder, prompt) pairs keeping the H0/H1 raw generations AND all blur siblings
  together (357 blocks; 1,428 held-out TEST rows per class);
* paired block-flip permutation test: each block's H0/H1 labels are swapped with probability
  1/2; p = (1 + #{AUC_b >= AUC_obs}) / (B + 1), B = --n-perm (2,000), seed --seed;
* decoder-stratified block bootstrap (--n-boot = 500 outer replicates, seed --seed + 1) for the
  AUC_1, K=16 and exponent CIs; each replicate rebuilds the cross-cell with-replacement AUC_K
  curve from 1,000 fresh groups per K and refits the exponent;
* point AUC_K curve from 8,000 with-replacement groups (seed --seed + 2).

Reports, per readout (rho self-check / zsum_dep / lda_dep):
* pooled row-level AUC_1 with block-bootstrap 95% CI and paired-block permutation p;
* block-valid mixture AUC_K curve, K=16 CI, and exponent CI;
* the same statistics per decoder.
Raw pooling concatenates the deployed scores unchanged. The ``lda_dep_stdpool`` entry applies
the standardized-pooling transformation behind Table I's pooled Pair-LDA AUC_1 (per cell,
FIT-H0 re-standardized and FIT-sign-aligned deployed scores) under the same block permutation
and bootstrap, so that pooled value gets its own block-valid CI.

GATES (FLAG, never silently pass):
* rho pooled AUC_1 must reproduce the selection-free Table I value (|delta| <= --rho-tol);
* lda_dep pooled AUC_1 is compared against the standardized-pooling Table I value (--pair-ref);
  a larger deviation is flagged REVIEW -- the raw-pooled number is reported as computed, never
  adjusted; the stdpool entry must reproduce --pair-ref.
The store loader itself aborts unless recomputed rho matches stored rho_summary within 1e-3.
The analysis additionally aborts unless all five decoders, 20 cells, 357 prompt blocks, and four
H0/H1 blur siblings per block are present exactly.

Produces: Table II (raw-pooled rho / T_pair-zsum / T_pair-LDA rows, the standardized-pooling
T_pair-LDA row, and the per-decoder T_pair-LDA rows: AUC_1, cross-cell AUC_16 and exponent with
block-bootstrap CIs; block permutation p) and the block counts of Section IV-B / Appendix B.
Inputs: main-grid outputs/p0/<dec>/p0_table.json + the server-side feature
store; CPU.
Outputs: outputs/p0/block_inference_deployed.json.
Usage:
  uv run --no-sync python scripts/analysis/block_inference_deployed.py

Strict JSON output (allow_nan=False). No torch, no re-inversion.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import within_config_curves as wcc
from _readout_common import (
    DEPLOYED_READOUTS,
    fit_deployed,
    load_residual_rows,
    score_deployed,
    write_json_strict,
)

from cfgid.paths import outputs_dir

CONTRAST = ("w20_vs_1", {20.0}, {1.0})
PUBLISHED_RHO = 0.662  # Table I: selection-free pooled rho AUC_1, w=20 vs 1
PUBLISHED_PAIR = 0.788  # Table I: standardized-pooling T_pair-LDA AUC_1, w=20 vs 1
EXPECTED_BLOCKS_BY_DECODER = {
    "hunyuandit": 51,
    "pixart_sigma": 51,
    "sd15": 102,
    "sd21": 51,
    "sdxl": 102,
}
EXPECTED_SIGMA_BLURS = frozenset({0.0, 8.0, 16.0, 32.0})


def _side_prompts(cell_rows: list[dict], w_set: set, split: str, side: str) -> list:
    """Prompt ids in the SAME order as _readout_common._side_matrix selects rows."""
    if side not in {"h0", "h1"}:
        raise ValueError(f"side must be h0/h1, got {side!r}")

    def selected(r: dict) -> bool:
        if r["split"] != split:
            return False
        label = r["label"]
        w = r["w"]
        if side == "h0":
            return (label == "h0" and 1.0 in w_set) or (label == "h1" and w in w_set)
        return label == "h1" and w in w_set

    return [r["prompt"] for r in cell_rows if selected(r)]


def _deployed_cells_scores(
    cells: dict[tuple[str, float], list[dict]],
    h1_set: set,
    h0_set: set,
    readout: str,
    *,
    stdpool: bool = False,
) -> dict[tuple[str, float], dict]:
    """Per-cell exact deployed TEST scores + aligned prompt ids, in _pooled_block_stats format.

    ``stdpool=True`` applies the standardized-pooling transformation on top of the deployed
    scores before pooling: per cell, z = sign * (score - mean(fit_h0_score)) / std(fit_h0_score,
    ddof=1) with sign = sign(mean(standardized FIT-H1 score)) -- the exact fixed_h1_allcell
    construction behind Table I's pooled AUC_1 values. This yields block statistics for the
    Table I estimand; ``stdpool=False`` pools raw deployed scores.
    """
    out: dict[tuple[str, float], dict] = {}
    for (dec, sb), cr in sorted(cells.items()):
        fit = fit_deployed(cr, h1_set, h0_set, readout)
        if fit is None:
            continue
        zt0 = score_deployed(cr, h0_set, "test", "h0", fit)
        zt1 = score_deployed(cr, h1_set, "test", "h1", fit)
        if stdpool:
            m0 = float(np.mean(fit.fit_h0))
            s0 = float(np.std(fit.fit_h0, ddof=1))
            if not (np.isfinite(s0) and s0 > 0.0):
                continue
            sign = float(np.sign(np.mean((fit.fit_h1 - m0) / s0)))
            if sign == 0.0:
                continue
            zt0 = sign * (zt0 - m0) / s0
            zt1 = sign * (zt1 - m0) / s0
        pt0 = _side_prompts(cr, h0_set, "test", "h0")
        pt1 = _side_prompts(cr, h1_set, "test", "h1")
        if len(pt0) != zt0.size or len(pt1) != zt1.size:
            raise SystemExit(
                f"[abort] {dec}/sb={sb:g}: prompt/score alignment broke "
                f"({len(pt0)}/{zt0.size} h0, {len(pt1)}/{zt1.size} h1)"
            )
        out[(dec, sb)] = {"zt0": zt0, "zt1": zt1, "pt0": pt0, "pt1": pt1}
    if not out:
        raise SystemExit(f"[abort] no cell produced a deployed fit for readout {readout!r}")
    return out


def _validate_deployed_structure(scores: dict[tuple[str, float], dict]) -> None:
    """Fail unless scores contain the exact 357 paired blocks and all four blur siblings."""
    expected_cells = {
        (decoder, sigma_blur)
        for decoder in EXPECTED_BLOCKS_BY_DECODER
        for sigma_blur in EXPECTED_SIGMA_BLURS
    }
    got_cells = set(scores)
    if got_cells != expected_cells:
        missing = sorted(expected_cells - got_cells)
        extra = sorted(got_cells - expected_cells)
        raise SystemExit(
            "[abort] deployed cell roster is not the exact 20-cell audit: "
            f"missing={missing}, extra={extra}"
        )

    for decoder, expected_n in EXPECTED_BLOCKS_BY_DECODER.items():
        reference_prompts: set | None = None
        for sigma_blur in sorted(EXPECTED_SIGMA_BLURS):
            score = scores[(decoder, sigma_blur)]
            p0 = list(score["pt0"])
            p1 = list(score["pt1"])
            if len(p0) != len(set(p0)) or len(p1) != len(set(p1)):
                raise SystemExit(
                    f"[abort] {decoder}/sb={sigma_blur:g}: duplicate prompt within a side"
                )
            prompts0, prompts1 = set(p0), set(p1)
            if prompts0 != prompts1 or len(prompts0) != expected_n:
                raise SystemExit(
                    f"[abort] {decoder}/sb={sigma_blur:g}: expected {expected_n} paired prompts, "
                    f"got h0={len(prompts0)}, h1={len(prompts1)}"
                )
            if reference_prompts is None:
                reference_prompts = prompts0
            elif prompts0 != reference_prompts:
                raise SystemExit(
                    f"[abort] {decoder}: blur siblings do not share the same raw prompt blocks"
                )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    ap.add_argument("--root", default=None, help="dir of <decoder>/ p0 outputs (def outputs/p0)")
    ap.add_argument("--n-perm", type=int, default=2000)
    ap.add_argument("--n-boot", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rho-tol", type=float, default=0.005)
    ap.add_argument("--pair-ref", type=float, default=PUBLISHED_PAIR)
    ap.add_argument("--rho-ref", type=float, default=PUBLISHED_RHO)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    if min(args.n_perm, args.n_boot) < 1:
        raise SystemExit("--n-perm and --n-boot must be positive")
    for name, value in (
        ("--rho-tol", args.rho_tol),
        ("--rho-ref", args.rho_ref),
        ("--pair-ref", args.pair_ref),
    ):
        if not math.isfinite(value) or (name == "--rho-tol" and value < 0.0):
            raise SystemExit(
                f"{name} must be finite" + (" and non-negative" if name == "--rho-tol" else "")
            )

    root = Path(args.root) if args.root else outputs_dir() / "p0"
    rows = load_residual_rows(root)
    cells: dict[tuple[str, float], list[dict]] = {}
    for r in rows:
        cells.setdefault((r["decoder"], r["sigma_blur"]), []).append(r)

    cname, h1_set, h0_set = CONTRAST
    flags: list[str] = []
    results: dict[str, dict] = {}
    for readout in DEPLOYED_READOUTS:
        print(f"[block_inference_deployed] scoring {readout}", flush=True)
        scores = _deployed_cells_scores(cells, h1_set, h0_set, readout)
        _validate_deployed_structure(scores)
        pooled = wcc._pooled_block_stats(scores, args.n_perm, args.n_boot, args.seed)
        per_decoder: dict[str, dict] = {}
        for dec in sorted({d for d, _ in scores}):
            sub = {k: v for k, v in scores.items() if k[0] == dec}
            per_decoder[dec] = wcc._pooled_block_stats(sub, args.n_perm, args.n_boot, args.seed)
        entry = {"pooled": pooled, "per_decoder": per_decoder}
        if readout == "rho":
            delta = float(pooled["auc1"] - args.rho_ref)
            entry["selfcheck"] = {
                "published": args.rho_ref,
                "delta": delta,
                "verdict": "PASS" if abs(delta) <= args.rho_tol else "FAIL",
            }
            if abs(delta) > args.rho_tol:
                flags.append(f"rho self-check FAIL: pooled {pooled['auc1']:.4f} vs published")
        if readout == "lda_dep":
            delta = float(pooled["auc1"] - args.pair_ref)
            entry["anchor_compare"] = {
                "published": args.pair_ref,
                "delta": delta,
                "verdict": "MATCH" if abs(delta) <= args.rho_tol else "REVIEW",
            }
            if abs(delta) > args.rho_tol:
                flags.append(
                    f"lda_dep pooled {pooled['auc1']:.4f} deviates from published "
                    f"{args.pair_ref:.3f} by {delta:+.4f} -- FLAG, kept as computed"
                )
        results[f"{readout}|{cname}"] = entry

    # Table I estimand under block inference: the SAME transformation that produces Table I's
    # pooled Pair-LDA AUC_1 (per-cell FIT-H0 re-standardized, FIT-sign-aligned deployed scores),
    # under the SAME block permutation/bootstrap, so that value gets its own block-valid CI
    # (Table II standardized-pooling row).
    print(
        "[block_inference_deployed] scoring lda_dep (published stdpool transformation)",
        flush=True,
    )
    scores_std = _deployed_cells_scores(cells, h1_set, h0_set, "lda_dep", stdpool=True)
    _validate_deployed_structure(scores_std)
    pooled_std = wcc._pooled_block_stats(scores_std, args.n_perm, args.n_boot, args.seed)
    delta_std = float(pooled_std["auc1"] - args.pair_ref)
    std_entry = {
        "pooled": pooled_std,
        "anchor_reproduction": {
            "published": args.pair_ref,
            "delta": delta_std,
            "verdict": "PASS" if abs(delta_std) <= args.rho_tol else "FAIL",
        },
    }
    if abs(delta_std) > args.rho_tol:
        flags.append(
            f"stdpool lda_dep {pooled_std['auc1']:.4f} fails to reproduce the published "
            f"anchor {args.pair_ref:.3f} (delta {delta_std:+.4f}) -- transformation mismatch"
        )
    results[f"lda_dep_stdpool|{cname}"] = std_entry

    payload = {
        "analysis": "block_inference_deployed",
        "protocol": "Exact deployed estimator on 357 raw-generation paired blocks; "
        "block construction and statistics mirror within_config_curves._pooled_block_stats",
        "contrast": cname,
        "n_perm": args.n_perm,
        "n_boot": args.n_boot,
        "seed": args.seed,
        "k_grid": wcc.K_GRID,
        "flags": flags,
        "results": results,
    }
    out = Path(args.out) if args.out else outputs_dir() / "p0" / "block_inference_deployed.json"
    write_json_strict(out, payload)
    print(f"[block_inference_deployed] wrote {out}")
    for name, entry in results.items():
        p = entry["pooled"]
        print(
            f"  {name:>16s}  auc1={p['auc1']:.4f} {p['auc1_ci95_blockboot']}  "
            f"perm_p={p['block_perm_p']:.4g}  k16={p['k16_point']:.4f} "
            f"{p['k16_ci95_blockboot']}  exp={p['exp_point']} {p['exp_ci95_blockboot']}"
        )
    for fl in flags:
        print(f"  FLAG: {fl}")
    if not flags:
        print("  gates: rho self-check PASS; lda_dep anchor MATCH")
    print("[done] block_inference_deployed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
