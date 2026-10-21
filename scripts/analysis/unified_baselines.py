"""Unified retargeted-baseline benchmark: re-noising, pair, rho and their fusion on one pool.

One benchmark, one split, one sample set: on the unblurred complete-feature subset of the main
grid (sigma_blur=0 cells, w=20 vs w=1), it aligns the re-noising baseline (per-cell FIT
ridge-LDA scalar of the 30-dim ``wangk_feats``, retarget_analyze's construction), the deployed
pair readout T_pair-LDA (``fusion_lda_full`` virtual tables) and rho (the real p0 tables) on
their COMMON image ids, each per-cell FIT-standardized and FIT-sign-oriented
(``baseline_incremental._aligned_map_split``). It reports TEST AUC1 for all three, a fusion of
re-noising and the pair (pooled 2-D ridge LDA fitted on the FIT rows, ridge 1e-6, exactly
baseline_incremental's recipe), and decoder-stratified prompt-block bootstrap 95% CIs
(``--n-boot`` 2000 replicates; within each decoder the (decoder, prompt) blocks of TEST images
are resampled with replacement) for every AUC and for the fusion-minus-component gaps.

Gates (abort on failure): the recomputed re-noising, pair and fusion AUCs and the FIT/TEST
image counts must reproduce ``outputs/p0/baseline_incremental.json`` within ``--gate-tol``;
rho's AUC is compared with ``--rho-ref`` (default 0.839, Table IV) and only flagged (not
aborted) if it deviates by more than 0.01.

Produces:
    Table IV (Section IV-B), AUC1 and bracketed CIs of the re-noising, rho and T_pair-LDA rows
    and the "Re-noising + pair fusion" row (``unified_baselines.json``:
    ``point/{base,rho,pair,fusion}`` and ``ci95_blockboot/{base,rho,pair,fusion}``; ``base`` is
    re-noising); also the standardized-pooling Pair-LDA value quoted in Table IX's note
    (``point/pair``).

Inputs: ``outputs/p0/retarget/retarget_wangk_*.jsonl`` (GPU-collected by
retarget_baselines.py; not shipped), ``outputs/p0/<decoder>/p0_table.json``, the server-side
residual feature store (``residual_v1.dat``, ``compact_rho.npy``) for the pair readout, and the
``outputs/p0/baseline_incremental.json`` from baseline_incremental.py (gate reference). CPU only.
The residualized column of Table IV is not recomputed here (see retarget_analyze.py
``--resid-vs-physical``).

Outputs: ``outputs/p0/unified_baselines.json`` (or ``--out``); strict JSON.

Usage:
    uv run --no-sync python scripts/analysis/unified_baselines.py [--root outputs/p0] \\
        [--retarget-dir outputs/p0/retarget]
"""

from __future__ import annotations

import argparse
import importlib.util
import math
import tempfile
from pathlib import Path

import numpy as np
from _readout_common import write_json_strict

from cfgid.detector.empirical import auc
from cfgid.numerics import finite_float, loads_strict_json
from cfgid.paths import outputs_dir
from cfgid.sweep.neff_scaling import _load_p0_rows


def _load_sibling(name: str):
    path = Path(__file__).with_name(f"{name}.py")
    spec = importlib.util.spec_from_file_location(f"{name}_reuse_ub", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_RA = _load_sibling("retarget_analyze")
_BI = _load_sibling("baseline_incremental")


def _boot_cis(stats: dict[str, list[float]]) -> dict[str, list[float] | None]:
    return {
        k: ([float(x) for x in np.percentile(v, [2.5, 97.5])] if v else None)
        for k, v in stats.items()
    }


def _assert_common_labels(
    maps: dict[str, dict[str, tuple[int, float]]], ids: list[str], split: str
) -> None:
    for image_id in ids:
        labels = {name: rows[image_id][0] for name, rows in maps.items()}
        if len(set(labels.values())) != 1:
            raise SystemExit(f"[abort] {split} label mismatch for {image_id}: {labels}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    ap.add_argument("--root", default=None, help="store root (def outputs/p0)")
    ap.add_argument("--retarget-dir", default=None)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gate-tol", type=float, default=1e-6)
    ap.add_argument("--rho-ref", type=float, default=0.839)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    if not math.isfinite(args.gate_tol) or args.gate_tol < 0.0:
        raise SystemExit("--gate-tol must be finite and non-negative")
    if args.n_boot < 1:
        raise SystemExit("--n-boot must be positive")
    if not math.isfinite(args.rho_ref) or not 0.0 <= args.rho_ref <= 1.0:
        raise SystemExit("--rho-ref must be finite and in [0, 1]")

    root = Path(args.root) if args.root else outputs_dir() / "p0"
    rdir = Path(args.retarget_dir) if args.retarget_dir else outputs_dir() / "p0" / "retarget"

    recs: list[dict] = []
    for f in sorted(rdir.glob("retarget_wangk_*.jsonl")):
        recs.extend(_RA._read_jsonl(f))
    if not recs:
        raise SystemExit(f"[abort] no retarget_wangk_*.jsonl under {rdir}")
    recs = [r for r in recs if float(r["sigma_blur"]) == 0.0]
    base_rows = _RA._lda_scalar_rows(recs, "wangk_feats")

    _PRB = _load_sibling("pair_readout_battery")
    with tempfile.TemporaryDirectory(prefix="unified_baselines_") as tmp:
        froots, _d, _c = _PRB._compute_all_features(root, Path(tmp))
        pair_rows: list[dict] = []
        for tp in sorted(froots["fusion_lda_full"].glob("*/p0_table.json")):
            pair_rows.extend(_load_p0_rows(tp))
    pair_rows = [r for r in pair_rows if float(r["sigma_blur"]) == 0.0]

    rho_rows: list[dict] = []
    for tp in sorted(root.glob("*/p0_table.json")):
        rho_rows.extend(_load_p0_rows(tp))
    rho_rows = [r for r in rho_rows if float(r["sigma_blur"]) == 0.0]

    maps = {
        name: (
            _BI._aligned_map_split(rows, "train"),
            _BI._aligned_map_split(rows, "test"),
        )
        for name, rows in (("base", base_rows), ("pair", pair_rows), ("rho", rho_rows))
    }
    fit_ids = sorted(set.intersection(*[set(maps[n][0]) for n in maps]))
    test_ids = sorted(set.intersection(*[set(maps[n][1]) for n in maps]))
    if len(fit_ids) < 50 or len(test_ids) < 50:
        raise SystemExit(
            f"[abort] too few common images (fit {len(fit_ids)}, test {len(test_ids)})"
        )
    _assert_common_labels({name: maps[name][0] for name in maps}, fit_ids, "FIT")
    _assert_common_labels({name: maps[name][1] for name in maps}, test_ids, "TEST")
    print(f"[pool] common unblurred subset: fit={len(fit_ids)}, test={len(test_ids)}")

    yf = np.array([maps["base"][0][i][0] for i in fit_ids], dtype=int)
    yt = np.array([maps["base"][1][i][0] for i in test_ids], dtype=int)
    s_fit = {n: np.array([maps[n][0][i][1] for i in fit_ids]) for n in maps}
    s_test = {n: np.array([maps[n][1][i][1] for i in test_ids]) for n in maps}

    # fusion: pooled 2-D ridge LDA on the aligned (re-noising, pair) FIT scores, applied to TEST
    xf = np.vstack([s_fit["base"], s_fit["pair"]]).T
    x0, x1 = xf[yf == 0], xf[yf == 1]
    w2 = np.linalg.solve(
        np.cov(np.concatenate([x0, x1]).T) + _BI.RIDGE * np.eye(2), x1.mean(0) - x0.mean(0)
    )
    s_test["fusion"] = np.vstack([s_test["base"], s_test["pair"]]).T @ w2

    point = {n: float(auc(s_test[n], yt)) for n in ("base", "pair", "rho", "fusion")}

    bi_path = root / "baseline_incremental.json"
    if bi_path.is_file():
        reference = loads_strict_json(bi_path.read_text(encoding="utf-8"))
        if not isinstance(reference, dict) or not isinstance(reference.get("point"), dict):
            raise SystemExit(f"[abort] {bi_path}: expected top-level point object")
        if reference.get("pool") != "sb0":
            raise SystemExit(
                f"[abort] {bi_path}: expected pool='sb0', got {reference.get('pool')!r}"
            )
        for key, got_count in (
            ("n_common_fit", len(fit_ids)),
            ("n_common_test", len(test_ids)),
        ):
            if reference.get(key) != got_count:
                raise SystemExit(
                    f"[abort] {bi_path}: {key}={reference.get(key)!r} != recomputed {got_count}"
                )
        want = reference["point"]
        for got_key, want_key in (
            ("base", "auc_base"),
            ("pair", "auc_pair"),
            ("fusion", "auc_combined"),
        ):
            refv = finite_float(f"baseline_incremental.point.{want_key}", want.get(want_key))
            if abs(refv - point[got_key]) > args.gate_tol:
                raise SystemExit(
                    f"[abort] gate FAILED {want_key}: {point[got_key]:.6f} vs shipped {refv}"
                )
        print(f"[gate] reproduced shipped incremental AUCs: {want}")
    else:
        raise SystemExit(f"[abort] {bi_path} absent; run baseline_incremental.py first")

    flags: list[str] = []
    rho_delta = point["rho"] - args.rho_ref
    if abs(point["rho"] - args.rho_ref) > 0.01:
        message = (
            f"[FLAG] rho raw AUC {point['rho']:.4f} deviates from the shipped reference "
            f"{args.rho_ref} by more than 0.01 -- investigate before quoting (not harmonized)"
        )
        flags.append(message)
        print(message)

    # decoder-stratified prompt-block bootstrap on TEST: within each decoder, resample its
    # (decoder, prompt) blocks with replacement; all four scores share each replicate
    blocks: dict[tuple, list[int]] = {}
    for idx, iid in enumerate(test_ids):
        blocks.setdefault(_BI._block_of(iid), []).append(idx)
    by_dec: dict[str, list] = {}
    for dec, p in sorted(blocks):
        by_dec.setdefault(dec, []).append((dec, p))
    rng = np.random.default_rng(args.seed)
    stats: dict[str, list[float]] = {
        k: []
        for k in (
            "base",
            "pair",
            "rho",
            "fusion",
            "fusion_minus_base",
            "fusion_minus_pair",
            "fusion_minus_rho",
            "fusion_minus_best",
            "pair_minus_rho",
        )
    }
    for _ in range(args.n_boot):
        take: list[int] = []
        for _dec, dkeys in by_dec.items():
            for i in rng.integers(0, len(dkeys), size=len(dkeys)):
                take.extend(blocks[dkeys[i]])
        ti = np.array(take)
        yb = yt[ti]
        if yb.min() == yb.max():
            continue
        vals = {n: float(auc(s_test[n][ti], yb)) for n in ("base", "pair", "rho", "fusion")}
        for n, v in vals.items():
            stats[n].append(v)
        stats["fusion_minus_base"].append(vals["fusion"] - vals["base"])
        stats["fusion_minus_pair"].append(vals["fusion"] - vals["pair"])
        stats["fusion_minus_rho"].append(vals["fusion"] - vals["rho"])
        stats["fusion_minus_best"].append(
            vals["fusion"] - max(vals[name] for name in ("base", "pair", "rho"))
        )
        stats["pair_minus_rho"].append(vals["pair"] - vals["rho"])
    if not stats["fusion"]:
        raise SystemExit("[abort] no valid bootstrap replicates")
    cis = _boot_cis(stats)
    best_single = max(("base", "pair", "rho"), key=point.__getitem__)
    point_differences = {
        "fusion_minus_base": point["fusion"] - point["base"],
        "fusion_minus_pair": point["fusion"] - point["pair"],
        "fusion_minus_rho": point["fusion"] - point["rho"],
        "fusion_minus_best": point["fusion"] - point[best_single],
        "pair_minus_rho": point["pair"] - point["rho"],
    }

    for n in ("base", "pair", "rho", "fusion"):
        print(f"[unified] {n:8s} AUC1={point[n]:.4f}  CI={cis[n]}")

    out = Path(args.out) if args.out else outputs_dir() / "p0" / "unified_baselines.json"
    write_json_strict(
        out,
        {
            "note": (
                "Unified benchmark: re-noising / pair / rho / fusion on the identical "
                "feature-complete unblurred main-grid subset and split, with "
                "decoder-stratified prompt-block bootstrap CIs. Residualized columns are "
                "NOT recomputed here (per-image battery features required); they remain "
                "sourced from the shipped retarget artifacts."
            ),
            "n_fit": len(fit_ids),
            "n_test": len(test_ids),
            "n_boot": args.n_boot,
            "n_boot_effective": len(stats["fusion"]),
            "seed": args.seed,
            "point": {k: round(v, 6) for k, v in point.items()},
            "point_differences": {k: round(v, 6) for k, v in point_differences.items()},
            "best_single_by_point": best_single,
            "ci95_blockboot": cis,
            "lda_weights_base_pair": [float(w) for w in w2],
            "results_pending": ["residualized columns (shipped artifacts remain the source)"],
            "baseline_gate": {
                "reference": str(bi_path),
                "tolerance": args.gate_tol,
                "n_common_fit": len(fit_ids),
                "n_common_test": len(test_ids),
            },
            "rho_reference_flag": {
                "reference": args.rho_ref,
                "tolerance": 0.01,
                "delta": rho_delta,
                "flagged": bool(flags),
            },
            "flags": flags,
        },
    )
    print(f"[unified] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
