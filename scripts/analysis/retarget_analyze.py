"""Analyze retargeted-baseline JSONLs into protocol-comparable detection numbers (CPU).

Consumes the per-(stage, decoder) JSONL checkpoints written by retarget_baselines.py and scores
every baseline feature under the selection-free fixed-H1 protocol of Table I (w=20 vs w=1,
every cell, per-cell FIT orientation, pooled TEST): scalar features (``ae_l1``, ``ae_l2``,
``ae_lpips``, ``dire_l1``, ``dire_l2``) are injected as virtual p0_table rows and oriented per
cell by their FIT sign; vector features (``wangk_feats``, 30-dim re-noising; ``xu_logits``,
13-dim attributor) are first reduced per cell to a scalar by a FIT-only ridge-LDA
(``_lda_scalar_rows``: per-coordinate standardization by the FIT mean/std of both classes,
pooled FIT covariance + 1e-3 I, weights Sigma^-1 (mu1 - mu0)). Both then run through the same
``collect_fixed_h1_records`` / ``_summarize`` machinery as the rho and pair anchors
(k_grid (1,2,4,8,16), n_groups 2000, seed 0): pooled AUC1, cross-cell K=16 group AUC and TPR at
1%/5% FPR at pooled TEST-H0 quantile thresholds.

Per baseline feature it also reports the paired complementarity with rho on the common TEST
images (``vs_rho``): the Pearson r of the aligned scores and both label-free residualized AUCs
(baseline | rho and rho | baseline; least-squares regression on the rows it corrects).

Options:
* ``--pool sb0``         restrict every baseline and rho to the unblurred (sigma_blur=0) cells
  (DIRE and re-noising are collected on those cells only).
* ``--probe-wangk``      the six-statistic physical battery (``rho_mechanism.physical_features``
  from the stored PNGs, per-cell FIT ridge-LDA) as a reference row, re-noising residualized on
  it, and a re-noising ablation by feature group (3 squared maps, low/mid/high timesteps).
* ``--resid-vs-physical`` every baseline, rho and (with the residual store) T_pair-LDA
  residualized against the physical battery's FIT-LDA score on the same images.

Produces:
    Table IV (Section IV-B), DIRE-style and re-noising rows, AUC1 and TPR@1%
      (``retarget_analysis.json``: ``results/dire/dire_l1`` and
      ``results/wangk/wangk_feats_fit_lda``; both are sigma_blur=0 only).
    Section IV-B, full-grid AEROBLADE-style and attributor AUC1 (``results/aeroblade/ae_l1``,
      ``results/xu/xu_logits_fit_lda``) and the score correlations of the AEROBLADE- and
      DIRE-style scores with rho (``vs_rho/pearson_r``).
    Table IV, unblurred-subset AEROBLADE, attributor and physical-battery rows with
      ``--pool sb0 --probe-wangk`` (output named ``retarget_analysis_sb0.json``), and the resid.
      column with ``--pool sb0 --resid-vs-physical`` (``retarget_analysis_resid.json``,
      ``results/resid_vs_physical/rows/<row>/auc_base_given_rho``, with the TPR@1% of the rho
      and T_pair-LDA rows under ``results/anchors_same_pool``); the re-noising vs
      physical-battery correlation is the ``pearson_r`` of the re-noising row there.

Inputs: ``outputs/p0/retarget/retarget_*.jsonl`` (GPU-collected by retarget_baselines.py; not
shipped) and ``outputs/p0/<decoder>/p0_table.json``. ``--probe-wangk`` and
``--resid-vs-physical`` also read the stored PNG images; the T_pair-LDA row of
``--resid-vs-physical`` needs the server-side residual feature store. No GPU.

Outputs: ``outputs/p0/retarget_analysis.json`` (or ``--out``).

Usage:
    uv run --no-sync python scripts/analysis/retarget_analyze.py
    uv run --no-sync python scripts/analysis/retarget_analyze.py --pool sb0 --probe-wangk \\
        --out outputs/p0/retarget_analysis_sb0.json
    uv run --no-sync python scripts/analysis/retarget_analyze.py --pool sb0 \\
        --resid-vs-physical --out outputs/p0/retarget_analysis_resid.json
    uv run --no-sync python scripts/analysis/retarget_analyze.py --smoke
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np

from cfgid.detector.empirical import auc
from cfgid.paths import outputs_dir
from cfgid.sweep.neff_scaling import _group_rows_by_cell, _load_p0_rows

W_HIGH = 20
K_GRID = (1, 2, 4, 8, 16)
N_GROUPS = 2000
SCALAR_FEATURES = {
    "aeroblade": ("ae_l1", "ae_l2", "ae_lpips"),
    "dire": ("dire_l1", "dire_l2"),
}
VECTOR_FEATURES = {"wangk": "wangk_feats", "xu": "xu_logits"}
LDA_RIDGE = 1e-3  # relative ridge on per-coordinate FIT-standardized features


def _load_sibling(name: str):
    path = Path(__file__).with_name(f"{name}.py")
    spec = importlib.util.spec_from_file_location(f"{name}_reuse_rt", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_FH = _load_sibling("fixed_h1_allcell")


def _read_jsonl(path: Path) -> list[dict]:
    with path.open() as fh:
        return [json.loads(line) for line in fh]


def _decoder_of(image_id: str) -> str:
    # image_id = f"{decoder}_p{pi}_..." with decoder names free of "_p<digit>"
    import re

    m = re.match(r"(.+?)_p\d+_s\d+_w", image_id)
    if m is None:
        raise ValueError(f"unparseable image_id {image_id!r}")
    return m.group(1)


def _virtual_rows(recs: list[dict], value_of) -> list[dict]:
    """Map JSONL records to minimal fixed_h1-compatible rows (rho_summary := feature)."""
    rows = []
    for r in recs:
        v = value_of(r)
        if v is None:
            continue
        rows.append(
            {
                "image_id": r["image_id"],
                "decoder": _decoder_of(r["image_id"]),
                "split": r["split"],
                "label": r["label"],
                "w": int(r["w"]),
                "sigma_blur": float(r["sigma_blur"]),
                "rho_summary": float(v),
                "qf": None,
            }
        )
    return rows


def _fixed_h1_summary(rows: list[dict]) -> dict:
    """Selection-free w=20 vs 1 summary of virtual rows: pooled AUC1, cross-cell K=16, TPR."""
    cells = _group_rows_by_cell(rows)
    recs = _FH.collect_fixed_h1_records(cells, (W_HIGH,), min_n=8, h0_ws=(1,))
    if not recs:
        return {"n_cells": 0}
    s = _FH._summarize(recs, k_grid=K_GRID, n_groups=N_GROUPS, n_perm=100, seed=0)
    _zf0, _zf1, zt0, zt1 = _FH._aligned_pools(recs)
    # descriptive TPR at pooled TEST-H0 quantile thresholds
    thr1 = float(np.quantile(zt0, 0.99))
    thr5 = float(np.quantile(zt0, 0.95))
    emp = np.asarray(s["auc_emp"], dtype=float)
    return {
        "n_cells": int(s["n_cells"]),
        "auc1": float(s["auc1_pooled_fitsign"]),
        "k16_group_auc": float(emp[-1]) if emp.size else None,
        "tpr_at_1pct": float(np.mean(zt1 >= thr1)),
        "tpr_at_5pct": float(np.mean(zt1 >= thr5)),
        "n_test_h0": int(zt0.size),
        "n_test_h1": int(zt1.size),
    }


def _lda_scalar_rows(recs: list[dict], key: str) -> list[dict]:
    """Per-cell FIT ridge-LDA on per-coordinate FIT-standardized vectors -> scalar rows."""
    by_cell: dict[tuple[str, float], list[dict]] = {}
    for r in recs:
        by_cell.setdefault((_decoder_of(r["image_id"]), float(r["sigma_blur"])), []).append(r)
    out_rows: list[dict] = []
    for (_dec, _sb), cell in by_cell.items():
        fit0 = [r[key] for r in cell if r["split"] == "train" and r["label"] == "h0"]
        fit1 = [
            r[key]
            for r in cell
            if r["split"] == "train" and r["label"] == "h1" and int(r["w"]) == W_HIGH
        ]
        if len(fit0) < 8 or len(fit1) < 8:
            continue
        f0, f1 = np.asarray(fit0, dtype=np.float64), np.asarray(fit1, dtype=np.float64)
        mu = np.concatenate([f0, f1]).mean(axis=0)
        sd = np.concatenate([f0, f1]).std(axis=0, ddof=1)
        sd[sd <= 0] = 1.0
        s0, s1 = (f0 - mu) / sd, (f1 - mu) / sd
        pooled = np.cov(np.concatenate([s0, s1]).T) + LDA_RIDGE * np.eye(s0.shape[1])
        wvec = np.linalg.solve(pooled, s1.mean(axis=0) - s0.mean(axis=0))
        for r in cell:
            z = (np.asarray(r[key], dtype=np.float64) - mu) / sd
            out_rows.append(
                {
                    "image_id": r["image_id"],
                    "decoder": _decoder_of(r["image_id"]),
                    "split": r["split"],
                    "label": r["label"],
                    "w": int(r["w"]),
                    "sigma_blur": float(r["sigma_blur"]),
                    "rho_summary": float(z @ wvec),
                    "qf": None,
                }
            )
    return out_rows


def _aligned_map(rows: list[dict]) -> dict[str, tuple[int, float]]:
    """image_id -> (label01, aligned z) on the test split; per-cell FIT m/s + FIT sign,
    mirroring the fixed_h1 alignment (same construction as stored_score_battery's stem stage)."""
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
            if r["split"] != "test":
                continue
            lab = str(r["label"])
            if lab == "h1" and int(r["w"]) != W_HIGH:
                continue
            out[str(r["image_id"])] = (
                1 if lab == "h1" else 0,
                sgn * (float(r["rho_summary"]) - m0) / s0,
            )
    return out


def _complementarity(base_rows: list[dict], rho_rows: list[dict]) -> dict | None:
    """Paired Pearson + two-way label-free residualized AUCs on common test images."""
    a = _aligned_map(base_rows)
    b = _aligned_map(rho_rows)
    common = sorted(set(a) & set(b))
    if len(common) < 50:
        return None
    lab = np.array([a[i][0] for i in common], dtype=int)
    x = np.array([a[i][1] for i in common])
    y = np.array([b[i][1] for i in common])
    # label-free linear residualization fitted on these same TEST rows
    xr = x - np.polyval(np.polyfit(y, x, 1), y)
    yr = y - np.polyval(np.polyfit(x, y, 1), x)
    return {
        "n_common": len(common),
        "pearson_r": float(np.corrcoef(x, y)[0, 1]),
        "auc_base": float(auc(x, lab)),
        "auc_rho": float(auc(y, lab)),
        "auc_base_given_rho": float(auc(xr, lab)),
        "auc_rho_given_base": float(auc(yr, lab)),
    }


def _physical_rows(recs: list[dict], features_root: Path) -> list[dict]:
    """Six-statistic physical battery (rho_mechanism.physical_features, the paper's
    canonical set) computed from the stored PNGs for the SAME images as `recs`,
    packed as a vector feature for the identical per-cell FIT-LDA path."""
    _RM = _load_sibling("rho_mechanism")
    out = []
    for r in recs:
        dec = _decoder_of(r["image_id"])
        png = features_root / dec / "images" / f"{r['image_id']}.png"
        feats = _RM.physical_features([png])
        vec = [float(feats[k][0]) for k in _RM.FEATURE_NAMES]
        rec = dict(r)
        rec["phys_feats"] = vec
        out.append(rec)
    return out


_WANGK_GROUPS = {
    "coord_e_opt2": [i for i in range(30) if i % 3 == 0],
    "coord_diff2": [i for i in range(30) if i % 3 == 1],
    "coord_e_uncond2": [i for i in range(30) if i % 3 == 2],
    "t_low(first3)": list(range(0, 9)),
    "t_mid(middle3)": list(range(9, 18)),
    "t_high(last4)": list(range(18, 30)),
}


def _probe_wangk(recs: list[dict], features_root: Path, rho_rows: list[dict]) -> dict:
    probe: dict = {}
    phys = _physical_rows(recs, features_root)
    phys_rows = _lda_scalar_rows(phys, "phys_feats")
    probe["physical_battery_lda"] = _fixed_h1_summary(phys_rows)
    wangk_rows = _lda_scalar_rows(recs, "wangk_feats")
    comp = _complementarity(wangk_rows, phys_rows)
    if comp:
        probe["wangk_vs_physical"] = comp  # auc_base_given_rho == wangk | physical here
    ablation = {}
    for name, idx in _WANGK_GROUPS.items():
        sliced = [dict(r, wangk_sub=[r["wangk_feats"][i] for i in idx]) for r in recs]
        rows = _lda_scalar_rows(sliced, "wangk_sub")
        ablation[name] = _fixed_h1_summary(rows).get("auc1")
    probe["dim_group_ablation_auc1"] = ablation
    return probe


def run(
    retarget_dir: Path,
    features_root: Path,
    out_path: Path,
    *,
    pool: str = "mixed",
    probe_wangk: bool = False,
    resid_vs_physical: bool = False,
) -> int:
    rho_rows: list[dict] = []
    for tp in sorted(features_root.glob("*/p0_table.json")):
        rho_rows.extend(_load_p0_rows(tp))
    if pool == "sb0":
        rho_rows = [r for r in rho_rows if float(r["sigma_blur"]) == 0.0]
    results: dict[str, dict] = {}

    def _stage_recs(stage: str) -> list[dict]:
        recs: list[dict] = []
        for f in sorted(retarget_dir.glob(f"retarget_{stage}_*.jsonl")):
            recs.extend(_read_jsonl(f))
        if pool == "sb0":
            recs = [r for r in recs if float(r["sigma_blur"]) == 0.0]
        return recs

    for stage, feats in SCALAR_FEATURES.items():
        recs = _stage_recs(stage)
        if not recs:
            results[stage] = {"skipped": "no JSONL"}
            continue
        results[stage] = {}
        for feat in feats:
            if feat not in recs[0]:
                results[stage][feat] = {"skipped": "field absent (e.g. lpips not installed)"}
                continue
            rows = _virtual_rows(recs, lambda r, f=feat: r.get(f))
            entry = _fixed_h1_summary(rows)
            comp = _complementarity(rows, rho_rows) if rho_rows else None
            if comp:
                entry["vs_rho"] = comp
            results[stage][feat] = entry

    for stage, key in VECTOR_FEATURES.items():
        recs = _stage_recs(stage)
        if not recs:
            results[stage] = {"skipped": "no JSONL"}
            continue
        rows = _lda_scalar_rows(recs, key)
        entry = _fixed_h1_summary(rows)
        comp = _complementarity(rows, rho_rows) if rho_rows else None
        if comp:
            entry["vs_rho"] = comp
        results[stage] = {f"{key}_fit_lda": entry}

    if probe_wangk:
        wrecs = _stage_recs("wangk")
        if wrecs:
            results["wangk_probe"] = _probe_wangk(wrecs, features_root, rho_rows)

    if resid_vs_physical:
        # one shared physical battery over the union of images seen by any stage
        seen: dict[str, dict] = {}
        for stage in (*SCALAR_FEATURES, *VECTOR_FEATURES):
            for r in _stage_recs(stage):
                seen.setdefault(r["image_id"], r)
        phys_rows = _lda_scalar_rows(
            _physical_rows(list(seen.values()), features_root), "phys_feats"
        )
        resid: dict[str, dict | None] = {}
        for stage, feats in SCALAR_FEATURES.items():
            recs = _stage_recs(stage)
            for feat in feats:
                if recs and feat in recs[0]:
                    rows = _virtual_rows(recs, lambda r, f=feat: r.get(f))
                    resid[feat] = _complementarity(rows, phys_rows)
        for stage, key in VECTOR_FEATURES.items():
            recs = _stage_recs(stage)
            if recs:
                resid[f"{key}_fit_lda"] = _complementarity(_lda_scalar_rows(recs, key), phys_rows)
        # rho on the same images (rho_rows already pool-filtered)
        common_ids = set(seen)
        rho_sub = [r for r in rho_rows if str(r["image_id"]) in common_ids]
        resid["rho"] = _complementarity(rho_sub, phys_rows)
        # T_pair-LDA needs the residual store (server): build its virtual rows via
        # pair_readout_battery
        try:
            import tempfile as _tf

            _PRB = _load_sibling("pair_readout_battery")
            work = Path(_tf.mkdtemp(prefix="retarget_fusion_"))
            froots, _d, _c = _PRB._compute_all_features(features_root, work)
            frows: list[dict] = []
            for tp in sorted(froots["fusion_lda_full"].glob("*/p0_table.json")):
                frows.extend(_load_p0_rows(tp))
            if pool == "sb0":
                frows = [r for r in frows if float(r["sigma_blur"]) == 0.0]
            frows = [r for r in frows if str(r["image_id"]) in common_ids]
            resid["fusion_lda_full"] = _complementarity(frows, phys_rows)
        except (Exception, SystemExit) as exc:  # store absent raises SystemExit deep in _FM
            resid["fusion_lda_full"] = {"skipped": f"residual store unavailable ({exc})"}
        anchors = {"rho": _fixed_h1_summary(rho_sub)}
        try:
            anchors["fusion_lda_full"] = _fixed_h1_summary(frows)
        except Exception as exc:
            anchors["fusion_lda_full"] = {"skipped": str(exc)}
        results["anchors_same_pool"] = anchors
        results["resid_vs_physical"] = {
            "note": "auc_base_given_rho = the row feature residualized against the six-statistic "
            "physical battery (label-free linear operator, same pool); auc_base = its raw AUC",
            "rows": resid,
        }

    payload = {
        "analysis": "retarget_analyze",
        "pool": pool,
        "protocol": "selection-free fixed-H1 (w=20 vs 1), per-cell FIT orientation / FIT "
        "ridge-LDA, aligned z on test; _summarize k_grid=(1,2,4,8,16), n_groups=2000, seed=0 "
        "-- same construction as the published 0.662/0.788 anchors",
        "results": results,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    print("\n=== retarget baselines under the fixed-H1 protocol ===")
    for stage, block in results.items():
        if "skipped" in block:
            print(f"{stage:10s} SKIPPED ({block['skipped']})")
            continue
        for feat, e in block.items():
            if not isinstance(e, dict) or "auc1" not in e:
                continue  # probe sub-blocks (complementarity / ablation) print via JSON only
            if "skipped" in e:
                print(f"{stage:10s} {feat:16s} skipped: {e['skipped']}")
                continue
            print(
                f"{stage:10s} {feat:16s} auc1={e['auc1']:.3f}  K16={e['k16_group_auc']:.3f}  "
                f"tpr1%={e['tpr_at_1pct']:.3f}  (cells={e['n_cells']})"
            )
    print(f"-> {out_path}")
    return 0


def _smoke() -> int:
    import tempfile

    tmp = Path(tempfile.mkdtemp(prefix="retarget_an_smoke_"))
    rt = tmp / "retarget"
    rt.mkdir(parents=True)
    rng = np.random.default_rng(0)
    recs_sc, recs_vec, rho_rows = [], [], []
    for dec in ("deca",):
        for pi in range(64):
            for split in ("train", "test"):
                for lab, w in (("h0", 1), ("h1", W_HIGH)):
                    iid = f"{dec}_p{pi}_s0_w{w}_{lab}_qfnone_sb0_{split}"
                    shift = 0.8 if lab == "h1" else 0.0
                    base = float(rng.normal(shift, 1.0))
                    recs_sc.append(
                        {
                            "image_id": iid,
                            "split": split,
                            "label": lab,
                            "w": w,
                            "sigma_blur": 0.0,
                            "dire_l1": base,
                            "dire_l2": base + float(rng.normal(0, 0.1)),
                        }
                    )
                    recs_vec.append(
                        {
                            "image_id": iid,
                            "split": split,
                            "label": lab,
                            "w": w,
                            "sigma_blur": 0.0,
                            "wangk_feats": [float(rng.normal(shift * 0.5, 1.0)) for _ in range(6)],
                        }
                    )
                    rho_rows.append(
                        {
                            "image_id": iid,
                            "decoder": dec,
                            "split": split,
                            "label": lab,
                            "w": w,
                            "sigma_blur": 0.0,
                            "rho_summary": base * 0.5 + float(rng.normal(0, 1.0)),
                            "qf": None,
                        }
                    )
    (rt / "retarget_dire_deca.jsonl").write_text("\n".join(json.dumps(r) for r in recs_sc) + "\n")
    (rt / "retarget_wangk_deca.jsonl").write_text("\n".join(json.dumps(r) for r in recs_vec) + "\n")
    feat_root = tmp / "features"
    (feat_root / "deca").mkdir(parents=True)
    (feat_root / "deca" / "p0_table.json").write_text(json.dumps(rho_rows))
    return run(rt, feat_root, tmp / "out.json")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="retarget_analyze")
    ap.add_argument("--retarget-dir", default=None, help="def outputs/p0/retarget")
    ap.add_argument("--features-root", default=None, help="def outputs/p0")
    ap.add_argument("--out", default=None, help="def outputs/p0/retarget_analysis.json")
    ap.add_argument(
        "--pool",
        choices=("mixed", "sb0"),
        default="mixed",
        help="sb0 = filter EVERY baseline to the sigma_blur=0 sub-pool "
        "(one apples-to-apples table; dire/wangk are sb0-only anyway)",
    )
    ap.add_argument(
        "--probe-wangk",
        action="store_true",
        help="wangk fairness probes: physical-battery baseline, wangk|physical "
        "residualization, 30-dim group ablation",
    )
    ap.add_argument(
        "--resid-vs-physical",
        action="store_true",
        help="per-row 'AUC | physical' column: every baseline, rho, and (with the residual "
        "store) T_pair-LDA residualized against the six-statistic battery on the same pool",
    )
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args(argv)
    if args.smoke:
        return _smoke()
    retarget_dir = (
        Path(args.retarget_dir) if args.retarget_dir else (outputs_dir() / "p0" / "retarget")
    )
    features_root = Path(args.features_root) if args.features_root else (outputs_dir() / "p0")
    out_path = Path(args.out) if args.out else (outputs_dir() / "p0" / "retarget_analysis.json")
    return run(
        retarget_dir,
        features_root,
        out_path,
        pool=args.pool,
        probe_wangk=args.probe_wangk,
        resid_vs_physical=args.resid_vs_physical,
    )


if __name__ == "__main__":
    raise SystemExit(main())
