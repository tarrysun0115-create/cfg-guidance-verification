"""Strongest-baseline vs deployed-pair incremental test on one pool and split (re-noising + pair).

On the unblurred complete-feature subset of the main grid (``--pool sb0``, the default) and its
FIT/TEST split, reports baseline-only, pair-only and baseline+pair combined TEST AUC1 with paired
bootstrap CIs of the gains. The baseline is the strongest retargeted family, the re-noising
features of Wang & Kalogeiton (retarget_baselines.py stage ``wangk``):

  1. baseline scalar = per-cell FIT ridge-LDA on the 30-dim ``wangk_feats``
                       (retarget_analyze's ``_lda_scalar_rows``);
  2. pair scalar     = the deployed T_pair-LDA (``fusion_lda_full``) rebuilt from the residual
                       store via pair_readout_battery;
  both per-cell FIT-H0-standardized and FIT-sign-oriented (``_aligned_map_split``), on the image
  ids common to both;
  3. combined        = pooled 2-D ridge LDA (ridge 1e-6) on the two aligned scalars, fitted on
                       FIT rows only and applied to TEST;
  4. paired prompt-block bootstrap on TEST (blocks = (decoder, prompt), resampled with
     replacement within each decoder; B = ``--n-boot``, default 2000, seed ``--seed``) for
     AUC(base), AUC(pair), AUC(combined) and the gains combined-pair, combined-base and
     base-pair.

Produces:
    Section IV-B (Table IV text): the fusion gain intervals over re-noising and over the pair
    (``baseline_incremental.json``: ``ci95/comb_minus_base``, ``ci95/comb_minus_pair``) and the
    re-noising/pair TEST score correlation (``point/pearson_base_pair_test``). Its point AUCs
    are the gate reference of unified_baselines.py and renoising_group_compare.py.

Inputs: ``retarget_wangk_*.jsonl`` (GPU-collected by retarget_baselines.py; not shipped),
``outputs/p0/<decoder>/p0_table.json`` and the server-side residual feature store
(``residual_v1.dat``, ``compact_rho.npy``). CPU only, no re-inversion. Note that the default
``--retarget-dir`` here is ``outputs/retarget`` while retarget_baselines.py writes to
``outputs/p0/retarget`` by default; pass the directory explicitly.

Outputs: ``<root>/baseline_incremental.json`` (default ``outputs/p0/baseline_incremental.json``).

Usage:
    uv run --no-sync python scripts/analysis/baseline_incremental.py \\
        --retarget-dir outputs/p0/retarget [--root outputs/p0] [--pool sb0]
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import tempfile
from pathlib import Path

import numpy as np

from cfgid.detector.empirical import auc
from cfgid.paths import outputs_dir
from cfgid.sweep.neff_scaling import _load_p0_rows

W_HIGH = 20
RIDGE = 1e-6
N_BOOT = 2000


def _load_sibling(name: str):
    path = Path(__file__).with_name(f"{name}.py")
    spec = importlib.util.spec_from_file_location(f"{name}_reuse_bi", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_RA = _load_sibling("retarget_analyze")


def _aligned_map_split(rows: list[dict], split: str) -> dict[str, tuple[int, float]]:
    """image_id -> (label01, aligned z) on the given split; per-cell FIT m/s + FIT sign.

    Same construction as retarget_analyze._aligned_map, generalized to also emit the FIT split
    (needed to fit the combined 2-D readout without touching TEST labels).
    """
    by_cell: dict[str, list[dict]] = {}
    for r in rows:
        by_cell.setdefault(f"{r['decoder']}/sb={float(r['sigma_blur']):g}", []).append(r)
    out: dict[str, tuple[int, float]] = {}
    for cell_rows in by_cell.values():
        h0f = [
            float(r["rho_summary"])
            for r in cell_rows
            if r["split"] == "train" and str(r["label"]) == "h0"
        ]
        h1f = [
            float(r["rho_summary"])
            for r in cell_rows
            if r["split"] == "train" and str(r["label"]) == "h1" and int(r["w"]) == W_HIGH
        ]
        if len(h0f) < 8 or len(h1f) < 8:
            continue
        m0, s0 = float(np.mean(h0f)), float(np.std(h0f, ddof=1))
        if not (np.isfinite(s0) and s0 > 0):
            continue
        sgn = 1.0 if (float(np.mean(h1f)) - m0) >= 0 else -1.0
        for r in cell_rows:
            if r["split"] != split:
                continue
            lab = str(r["label"])
            if lab == "h1" and int(r["w"]) != W_HIGH:
                continue
            if lab == "h0" and int(r["w"]) != 1:
                continue
            out[str(r["image_id"])] = (
                1 if lab == "h1" else 0,
                sgn * (float(r["rho_summary"]) - m0) / s0,
            )
    return out


def _block_of(image_id: str) -> tuple[str, str]:
    m = re.match(r"(.+?)_p(\d+)_s\d+_w", image_id)
    if m is None:
        raise ValueError(f"unparseable image_id {image_id!r}")
    return m.group(1), m.group(2)  # (decoder, prompt index)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--retarget-dir", default=None, help="dir of retarget_*.jsonl")
    ap.add_argument("--root", default=None, help="dir of <decoder>/ p0 outputs (def outputs/p0)")
    ap.add_argument("--pool", default="sb0", choices=["sb0", "mixed"])
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    root = Path(args.root) if args.root else outputs_dir() / "p0"
    rdir = Path(args.retarget_dir) if args.retarget_dir else outputs_dir() / "retarget"

    # 1. strongest baseline: wangk re-noising features -> per-cell FIT ridge-LDA scalar rows
    recs: list[dict] = []
    for f in sorted(rdir.glob("retarget_wangk_*.jsonl")):
        recs.extend(_RA._read_jsonl(f))
    if not recs:
        print(f"[skip] no retarget_wangk_*.jsonl under {rdir}")
        return 1
    if args.pool == "sb0":
        recs = [r for r in recs if float(r["sigma_blur"]) == 0.0]
    base_rows = _RA._lda_scalar_rows(recs, "wangk_feats")

    # 2. deployed pair scalar: fusion (rho, m) virtual tables from the residual store
    _PRB = _load_sibling("pair_readout_battery")
    work = Path(tempfile.mkdtemp(prefix="baseline_incremental_"))
    froots, _d, _c = _PRB._compute_all_features(root, work)
    pair_rows: list[dict] = []
    for tp in sorted(froots["fusion_lda_full"].glob("*/p0_table.json")):
        pair_rows.extend(_load_p0_rows(tp))
    if args.pool == "sb0":
        pair_rows = [r for r in pair_rows if float(r["sigma_blur"]) == 0.0]

    # 3. align on the common image set, FIT and TEST separately
    maps = {
        "base": (_aligned_map_split(base_rows, "train"), _aligned_map_split(base_rows, "test")),
        "pair": (_aligned_map_split(pair_rows, "train"), _aligned_map_split(pair_rows, "test")),
    }
    fit_ids = sorted(set(maps["base"][0]) & set(maps["pair"][0]))
    test_ids = sorted(set(maps["base"][1]) & set(maps["pair"][1]))
    if len(fit_ids) < 50 or len(test_ids) < 50:
        print(f"[skip] too few common images (fit {len(fit_ids)}, test {len(test_ids)})")
        return 1

    yf = np.array([maps["base"][0][i][0] for i in fit_ids], dtype=int)
    xf = np.vstack(
        [[maps["base"][0][i][1] for i in fit_ids], [maps["pair"][0][i][1] for i in fit_ids]]
    ).T
    yt = np.array([maps["base"][1][i][0] for i in test_ids], dtype=int)
    xt = np.vstack(
        [[maps["base"][1][i][1] for i in test_ids], [maps["pair"][1][i][1] for i in test_ids]]
    ).T

    # combined = pooled 2-D ridge-LDA fitted on FIT only
    x0, x1 = xf[yf == 0], xf[yf == 1]
    w2 = np.linalg.solve(
        np.cov(np.concatenate([x0, x1]).T) + RIDGE * np.eye(2), x1.mean(0) - x0.mean(0)
    )
    s_base, s_pair, s_comb = xt[:, 0], xt[:, 1], xt @ w2

    point = {
        "auc_base": _rounded(auc(s_base, yt)),
        "auc_pair": _rounded(auc(s_pair, yt)),
        "auc_combined": _rounded(auc(s_comb, yt)),
        "pearson_base_pair_test": float(np.corrcoef(s_base, s_pair)[0, 1]),
        "lda_weights_base_pair": [float(w) for w in w2],
    }

    # 4. paired prompt-block bootstrap on TEST
    blocks: dict[tuple, list[int]] = {}
    for idx, iid in enumerate(test_ids):
        blocks.setdefault(_block_of(iid), []).append(idx)
    bkeys = sorted(blocks)
    by_dec: dict[str, list] = {}
    for dec, p in bkeys:
        by_dec.setdefault(dec, []).append((dec, p))
    rng = np.random.default_rng(args.seed)
    stats: dict[str, list[float]] = {
        k: []
        for k in ("base", "pair", "comb", "comb_minus_pair", "comb_minus_base", "base_minus_pair")
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
        ab = float(auc(s_base[ti], yb))
        apair = float(auc(s_pair[ti], yb))
        ac = float(auc(s_comb[ti], yb))
        stats["base"].append(ab)
        stats["pair"].append(apair)
        stats["comb"].append(ac)
        stats["comb_minus_pair"].append(ac - apair)
        stats["comb_minus_base"].append(ac - ab)
        stats["base_minus_pair"].append(ab - apair)

    def ci(v):
        return [float(x) for x in np.percentile(v, [2.5, 97.5])] if v else None

    payload = {
        "analysis": "baseline_incremental",
        "pool": args.pool,
        "protocol": "wangk re-noising per-cell FIT ridge-LDA scalar vs deployed fusion pair "
        "scalar; combined = pooled FIT 2-D ridge-LDA; paired (decoder,prompt)-block bootstrap "
        f"B={args.n_boot} on the common held-out TEST images (n={len(test_ids)})",
        "n_common_fit": len(fit_ids),
        "n_common_test": len(test_ids),
        "n_blocks": len(bkeys),
        "point": point,
        "ci95": {k: ci(v) for k, v in stats.items()},
    }
    out = root / "baseline_incremental.json"
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"\n=== strongest-baseline incremental test ({args.pool}, n_test={len(test_ids)}) ===")
    for k in ("auc_base", "auc_pair", "auc_combined"):
        print(f"  {k:<14} = {point[k]:.4f}")
    for k in ("comb_minus_pair", "comb_minus_base", "base_minus_pair"):
        print(f"  {k:<16} 95% CI {payload['ci95'][k]}")
    print(f"  pearson(base,pair)={point['pearson_base_pair_test']:.3f}  blocks={len(bkeys)}")
    print(f"-> {out}")
    return 0


def _rounded(x) -> float:
    return float(x)


if __name__ == "__main__":
    raise SystemExit(main())
