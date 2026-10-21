"""sigma_eff PSF ordering for rho vs diff_energy, store-only and self-validating.

Role: helper for feature_comparison.py, which reuses ``_group_cells_with_feature``,
``_sigma_eff_by_cell``, ``_sigma_axis_modes`` and ``_feature_psf`` to compute every candidate's
PSF ordering for Table X (Section IV-E3). For one per-image feature map, ``_feature_psf`` takes
the selection-free cells (H1 fixed at w=20 vs w=1, FIT-H0 standardization and FIT sign, all
cells), converts each cell's TEST AUC to a deflection d = 2*Phi^-1(AUC) (cells at or below
chance dropped), fits a per-decoder power law d ~ x^-p along one shared axis x per decoder
(sigma_eff from master_table.json when every cell has it, else sqrt(sigma_core^2 +
sigma_blur^2), else sigma_blur + 1), and correlates p with the decoder's sigma_core over the
distinct-VAE decoders and over all decoders.

Run as a script, it applies this to rho and to the unsigned magnitude diff_energy (both
recomputed from the store with the texture mask, winsorization and patch mean of rho) and
prints the two orderings; its own JSON is not printed in the paper.

Inputs: ``outputs/p0/<decoder>/p0_table.json``, the server-side residual feature store
(``residual_v1.dat``, ``compact_rho.npy``), ``outputs/p0/master_table.json`` and
``outputs/sigma_calib/<decoder>.json``. CPU only. Self-checks: recomputed rho must reproduce the
stored ``rho_summary`` to 1e-3 and the fixed-H1 rho AUC must match ``--expect-rho-auc`` (0.662)
within ``--gate-tol`` (0.03).

Outputs: ``outputs/p0/psf_grounding_features.json`` (or ``--out``).

Usage:
    uv run --no-sync python scripts/analysis/psf_grounding_features.py [--features-root outputs/p0]
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np

from cfgid.channel.rho import image_rho_summary
from cfgid.detector.empirical import auc
from cfgid.featurestore import read_compact, read_residual_v1
from cfgid.numerics import auc_gate_passed, finite_float, finite_or_none, loads_strict_json
from cfgid.paths import outputs_dir
from cfgid.sweep.evidence import _require_row_store_alignment, _residual_energy_features
from cfgid.sweep.neff_scaling import _Cell, _load_p0_rows, filter_core_none_qf_rows

MASK_Q = 0.2
WINSOR = (0.01, 0.99)


def _load_sibling(name: str):
    path = Path(__file__).with_name(f"{name}.py")
    spec = importlib.util.spec_from_file_location(f"{name}_reuse", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_PSF = _load_sibling("psf_grounding")
_FH = _load_sibling("fixed_h1_allcell")  # collect_fixed_h1_records


def _image_summary(per_patch: np.ndarray, e_tex: np.ndarray) -> float:
    return image_rho_summary(per_patch, e_tex, mask_quantile=MASK_Q, winsor_quantiles=WINSOR)


def _group_cells_with_feature(rows: list[dict], feature: dict[str, float]) -> dict[str, _Cell]:
    """Group core rows (``filter_core_none_qf_rows``) into (decoder, sigma_blur) cells, using
    ``feature[image_id]`` as the score; train rows go to the FIT bucket, test rows to TEST."""
    cells: dict[str, _Cell] = {}
    for r in filter_core_none_qf_rows(rows):
        iid = str(r.get("image_id", ""))
        if iid not in feature:
            continue
        value = finite_float(f"feature[{iid}]", feature[iid])
        sb = float(r["sigma_blur"])
        w = int(r["w"])
        bucket = (
            "fit" if r.get("split") == "train" else ("test" if r.get("split") == "test" else None)
        )
        if bucket is None:
            continue
        cell = cells.setdefault(f"{r['decoder']}/sb={sb:g}", _Cell())
        cell.decoder = str(r["decoder"])
        cell.sigma_blur = sb
        label = str(r.get("label"))
        if label == "h0":
            cell.h0[bucket].append(value)
        elif label == "h1":
            cell.h1.setdefault(w, {"fit": [], "test": []})[bucket].append(value)
    return cells


def _sigma_eff_by_cell(root: Path) -> dict[tuple[str, float], float]:
    """(decoder, sigma_blur) -> sigma_eff from ``master_table.json`` (empty if absent)."""
    mt = root / "master_table.json"
    out: dict[tuple[str, float], float] = {}
    if mt.is_file():
        raw = loads_strict_json(mt.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError(f"{mt}: master_table.json must be an object")
        rows = raw.get("rows", [])
        if not isinstance(rows, list):
            raise ValueError(f"{mt}: master_table.rows must be a list")
        for r in rows:
            if not isinstance(r, dict):
                raise ValueError(f"{mt}: master_table rows must be objects")
            if r.get("sigma_eff") is not None:
                sb = finite_float("master_table.sigma_blur", r["sigma_blur"])
                se = finite_float("master_table.sigma_eff", r["sigma_eff"], positive=True)
                out[(str(r["decoder"]), sb)] = se
    return out


def _sigma_axis_modes(
    root: Path, rows: list[dict], se_by: dict[tuple[str, float], float]
) -> dict[str, str]:
    """One decay-axis mode per decoder, shared by every feature so their fits are comparable."""
    by_decoder: dict[str, set[float]] = {}
    for r in filter_core_none_qf_rows(rows):
        by_decoder.setdefault(str(r["decoder"]), set()).add(float(r["sigma_blur"]))
    modes: dict[str, str] = {}
    for dec, sbs in sorted(by_decoder.items()):
        score = finite_or_none(_PSF._sigma_core(root, dec), positive=True)
        if all((dec, sb) in se_by for sb in sbs):
            modes[dec] = "sigma_eff"
        elif score is not None:
            modes[dec] = "sigma_eff_reconstructed"
        else:
            modes[dec] = "sigma_blur+1"
    return modes


def _axis_values(
    root: Path,
    dec: str,
    clist: list[dict],
    se_by: dict[tuple[str, float], float],
    axis_modes: dict[str, str],
) -> tuple[list[float], str, float | None]:
    score = finite_or_none(_PSF._sigma_core(root, dec), positive=True)
    mode = axis_modes.get(dec, "sigma_blur+1")
    if mode == "sigma_eff":
        xs = [
            finite_float("sigma_eff", se_by[(dec, c["sigma_blur"])], positive=True) for c in clist
        ]
    elif mode == "sigma_eff_reconstructed" and score is not None:
        xs = [float(np.hypot(score, c["sigma_blur"])) for c in clist]
    else:
        mode = "sigma_blur+1"
        xs = [c["sigma_blur"] + 1.0 for c in clist]
    return xs, mode, score


def _fixed_h1_auc(cells: dict[str, _Cell], h1_w: int) -> float | None:
    recs = _FH.collect_fixed_h1_records(cells, (h1_w,), min_n=8)
    if not recs:
        return None
    _zf0, _zf1, zt0, zt1 = _FH._aligned_pools(recs)
    scores = np.concatenate([zt1, zt0])
    labels = np.concatenate([np.ones(zt1.size, int), np.zeros(zt0.size, int)])
    return finite_float("fixed-H1 AUC", auc(scores, labels))


def _feature_psf(
    root: Path,
    cells: dict[str, _Cell],
    h1_w: int,
    se_by: dict[tuple[str, float], float],
    axis_modes: dict[str, str],
) -> dict:
    """Per-decoder power-law decay of fixed-H1 cell AUC vs sigma_eff + Pearson(p, sigma_core)."""
    recs = _FH.collect_fixed_h1_records(cells, (h1_w,), min_n=8)
    by_decoder: dict[str, list[dict]] = {}
    for rec in recs:
        dec = rec.decoder
        sb = float(rec.key.split("sb=")[1].split("/")[0])
        auc = rec.cell_test_auc()
        d = _PSF._deflection(auc)  # 2*Phi^-1(AUC); None if <= chance
        if d is None:
            continue
        se = se_by.get((dec, sb))
        by_decoder.setdefault(dec, []).append(
            {"sigma_blur": sb, "sigma_eff": se, "auc": auc, "d": d}
        )
    per_decoder = []
    for dec, clist in sorted(by_decoder.items()):
        clist = sorted(clist, key=lambda c: c["sigma_blur"])
        xs, axis, score = _axis_values(root, dec, clist, se_by, axis_modes)
        fit = _PSF._powerlaw_fit(xs, [c["d"] for c in clist])
        p = finite_or_none(fit["p"])
        r2 = finite_or_none(fit["r2"])
        cell_details = [
            {
                "sigma_blur": finite_float("sigma_blur", c["sigma_blur"]),
                "sigma_eff": finite_or_none(c["sigma_eff"], positive=True),
                "axis_value": finite_float("axis_value", x, positive=True),
                "auc": finite_float("auc", c["auc"]),
                "deflection_d": finite_float("deflection_d", c["d"], positive=True),
            }
            for c, x in zip(clist, xs, strict=True)
        ]
        per_decoder.append(
            {
                "decoder": dec,
                "sigma_core": score,
                "axis": axis,
                "n_points": len(clist),
                "decay_exponent_p": p,
                "fit_r2": r2,
                "cells": cell_details,
            }
        )
    grounded = [
        d for d in per_decoder if d["sigma_core"] is not None and d["decay_exponent_p"] is not None
    ]
    distinct = [d for d in grounded if d["decoder"] not in _PSF._SHARED_VAE_DECODERS]

    def _pearson(ds: list[dict]) -> float | None:
        if len(ds) < 3:
            return None
        c = np.corrcoef([d["sigma_core"] for d in ds], [d["decay_exponent_p"] for d in ds])[0, 1]
        return float(c) if np.isfinite(c) else None

    return {
        "per_decoder": per_decoder,
        "pearson_all": _pearson(grounded),
        "pearson_distinct_vae": _pearson(distinct),
        "n_distinct_vae": len(distinct),
    }


def run(
    features_root: Path,
    out_path: Path,
    *,
    h1_w: int = 20,
    expect_rho_auc: float = 0.662,
    gate_tol: float = 0.03,
) -> int:
    decoder_dirs = sorted(d for d in features_root.glob("*/") if (d / "p0_table.json").exists())
    if not decoder_dirs:
        print(f"[skip] no <decoder>/p0_table.json under {features_root}")
        return 0
    rho_feat: dict[str, float] = {}
    diff_feat: dict[str, float] = {}
    rows_all: list[dict] = []
    for dec_dir in decoder_dirs:
        rows = _load_p0_rows(dec_dir / "p0_table.json")
        rows_all.extend(rows)
        for r in rows:
            rho_feat[str(r["image_id"])] = finite_float("rho_summary", r["rho_summary"])
        if not (dec_dir / "residual_v1.dat").exists() or not (dec_dir / "compact_rho.npy").exists():
            print(f"[skip] {dec_dir.name}: no feature store.")
            return 0
        batch, meta = read_compact(dec_dir)
        residual, _ = read_residual_v1(dec_dir)
        _require_row_store_alignment(rows, meta)
        ids = list(meta.image_ids) or [str(r["image_id"]) for r in rows]
        diff_p = _residual_energy_features(residual)
        max_rho_err = 0.0
        for i, iid in enumerate(ids):
            etex_i = batch.e_tex[i]
            diff_feat[iid] = _image_summary(diff_p[i], etex_i)
            max_rho_err = max(
                max_rho_err,
                finite_float(
                    "rho reproduction error",
                    abs(_image_summary(batch.rho[i], etex_i) - rho_feat[iid]),
                ),
            )
        if max_rho_err > 1e-3:
            raise SystemExit(f"rho reproduction off by {max_rho_err:.4g}; refusing.")

    se_by = _sigma_eff_by_cell(features_root)
    axis_modes = _sigma_axis_modes(features_root, rows_all, se_by)
    rho_cells = _group_cells_with_feature(rows_all, rho_feat)
    rho_auc = _fixed_h1_auc(rho_cells, h1_w)
    if not auc_gate_passed(rho_auc, expected=expect_rho_auc, tol=gate_tol):
        raise SystemExit(
            f"SELF-CHECK FAILED: rho fixed-H1 AUC {rho_auc} != "
            f"{expect_rho_auc}+/-{gate_tol}; refusing."
        )
    result = {
        "analysis": "psf_grounding_features",
        "h1_w": h1_w,
        "self_check": {
            "rho_auc1": rho_auc,
            "expected": expect_rho_auc,
            "tol": gate_tol,
            "passed": True,
        },
        "sigma_axis_mode_by_decoder": axis_modes,
        "features": {},
    }
    feature_cells = {
        "rho": rho_cells,
        "diff_energy": _group_cells_with_feature(rows_all, diff_feat),
    }
    for name, cells in feature_cells.items():
        result["features"][name] = _feature_psf(features_root, cells, h1_w, se_by, axis_modes)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")

    print("\nsigma_eff PSF-ordering: decay exponent p vs measured sigma_core (distinct-VAE)")
    for name in ("rho", "diff_energy"):
        f = result["features"][name]
        per = "  ".join(
            f"{d['decoder']}:p={d['decay_exponent_p']:.2f}@core{d['sigma_core']:.1f}"
            for d in f["per_decoder"]
            if d["sigma_core"] is not None and d["decay_exponent_p"] is not None
        )
        pd = f["pearson_distinct_vae"]
        print(f"  {name:<12} pearson(distinct-VAE)={pd if pd is None else f'{pd:+.3f}'}  [{per}]")
    print("  (rho high pearson but diff_energy low => PSF-ordering is a 2nd rho-specific property)")
    print(f"-> {out_path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="psf_grounding_features")
    ap.add_argument(
        "--features-root", default=None, help="dir of <decoder>/ stores (def: outputs/p0)"
    )
    ap.add_argument("--out", default=None, help="def: outputs/p0/psf_grounding_features.json")
    ap.add_argument("--expect-rho-auc", type=float, default=0.662)
    ap.add_argument("--gate-tol", type=float, default=0.03)
    args = ap.parse_args(argv)
    features_root = Path(args.features_root) if args.features_root else (outputs_dir() / "p0")
    out_path = (
        Path(args.out) if args.out else (outputs_dir() / "p0" / "psf_grounding_features.json")
    )
    return run(features_root, out_path, expect_rho_auc=args.expect_rho_auc, gate_tol=args.gate_tol)


if __name__ == "__main__":
    raise SystemExit(main())
