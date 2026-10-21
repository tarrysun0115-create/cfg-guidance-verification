"""Physical-statistics battery and the battery-residualized rho on the favorable subset.

Produces: no printed paper number directly (shared helper plus a standalone diagnostic).
``FEATURE_NAMES``, ``image_features``, ``physical_features`` and ``_residualize_multi`` define the
six-statistic physical battery used in Section IV-E1 / Table VIII (saturation, contrast,
high-frequency energy, colorfulness, blockiness, mean luminance) and are reused by
``pair_distinctness.py``, ``crossfit_resid.py``, ``retarget_analyze.py`` and the surrogate attack's
feature set (``cfgid.attack.rho_surrogate``, cross-checked in the tests). The ``rho_mechanism.json``
written here is a diagnostic and is not printed in the paper.

Inputs: ``outputs/p0/<decoder>/p0_table.json`` and the main-grid PNGs it references (not
shipped; written by the generation run). CPU only (numpy + Pillow); no re-inversion.

Outputs: ``outputs/p0/rho_mechanism.json``.

Usage: python scripts/analysis/rho_mechanism.py [--alive-t 1.5] [--n-perm 2000]

Question: is rho only measuring the familiar high-CFG over-saturation / texture collapse? On the
favorable-subset evaluation pool (selection imported from ``clip_vs_rho.py``, so the images match
the rho-vs-CLIP comparison) it reports:

  - a BATTERY of cheap per-image physical features computed from the stored PNGs: saturation,
    luminance contrast, high-frequency (Laplacian) energy, colorfulness (Hasler-Susstrunk), JPEG
    8x8 blockiness, mean luminance;
  - head-to-head AUC: aligned-rho vs each physical feature;
  - the BATTERY-residualized aligned-rho AUC + permutation p (``--n-perm`` draws, seed 100):
    regress rho_z on ALL physical features (label-free least squares with intercept), take the
    residual, and score it. A residual that still separates H1/H0 (perm p<0.05) means rho carries
    guidance signal beyond every tested physical correlate;
  - feature<->rho correlation per feature: which physical quantity best tracks the rho shift.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from operator import index
from pathlib import Path

import numpy as np

from cfgid.detector.empirical import auc
from cfgid.paths import outputs_dir
from cfgid.sweep.neff_scaling import _load_p0_rows

FEATURE_NAMES = ("saturation", "contrast", "highfreq", "colorfulness", "blockiness", "mean_luma")


def _positive_int(name: str, value: int) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    try:
        out = index(value)
    except TypeError as exc:
        raise ValueError(f"{name} must be a positive integer, got {value!r}") from exc
    if out < 1:
        raise ValueError(f"{name} must be >= 1, got {out}")
    return out


def _luma(img: np.ndarray) -> np.ndarray:
    return 0.299 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2]


def image_features(img: np.ndarray) -> dict[str, float]:
    """Cheap physical descriptors of one RGB image in [0,1], shape (H,W,3)."""
    img = np.asarray(img, dtype=np.float64)
    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError(f"image must be (H,W,3) RGB in [0,1], got shape {img.shape}")
    if min(img.shape[:2]) < 1:
        raise ValueError(f"image must have non-empty spatial dimensions, got shape {img.shape}")
    if not np.all(np.isfinite(img)):
        raise ValueError("image must contain only finite values")
    y = _luma(img)
    mx = img.max(axis=2)
    mn = img.min(axis=2)
    sat = np.where(mx > 1e-8, (mx - mn) / np.maximum(mx, 1e-8), 0.0)
    lap = -4.0 * y + np.roll(y, 1, 0) + np.roll(y, -1, 0) + np.roll(y, 1, 1) + np.roll(y, -1, 1)
    rg = img[..., 0] - img[..., 1]
    yb = 0.5 * (img[..., 0] + img[..., 1]) - img[..., 2]
    colorful = float(
        np.sqrt(rg.std() ** 2 + yb.std() ** 2) + 0.3 * np.sqrt(rg.mean() ** 2 + yb.mean() ** 2)
    )
    bh = np.abs(np.diff(y, axis=1))
    bv = np.abs(np.diff(y, axis=0))
    bh_block = float(bh[:, 7::8].mean()) if bh.shape[1] > 7 else 0.0
    bv_block = float(bv[7::8, :].mean()) if bv.shape[0] > 7 else 0.0
    bh_mean = float(bh.mean()) if bh.size else 0.0
    bv_mean = float(bv.mean()) if bv.size else 0.0
    blockiness = (bh_block - bh_mean) + (bv_block - bv_mean)
    return {
        "saturation": float(sat.mean()),
        "contrast": float(y.std()),
        "highfreq": float(lap[1:-1, 1:-1].var()) if min(y.shape) > 2 else 0.0,
        "colorfulness": colorful,
        "blockiness": float(blockiness),
        "mean_luma": float(y.mean()),
    }


def physical_features(paths: list[Path]) -> dict[str, np.ndarray]:
    """Per-image physical features for the eval PNGs; returns {feature_name: array(N,)}."""
    from PIL import Image  # local import: Pillow is only needed when the PNGs are read

    cols: dict[str, list[float]] = {name: [] for name in FEATURE_NAMES}
    for p in paths:
        arr = np.asarray(Image.open(p).convert("RGB"), dtype=np.float64) / 255.0
        feats = image_features(arr)
        for name in FEATURE_NAMES:
            cols[name].append(feats[name])
    return {name: np.asarray(vals, dtype=np.float64) for name, vals in cols.items()}


def _residualize_multi(x: np.ndarray, feature_matrix: np.ndarray) -> np.ndarray:
    """Residual of x after a label-free least-squares fit on [1 | features] (drops constant)."""
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    f = np.asarray(feature_matrix, dtype=np.float64)
    if f.ndim == 1:
        f = f.reshape(-1, 1)
    if f.ndim != 2 or f.shape[0] != x.size:
        raise ValueError(f"feature_matrix must be (N,P) matching x, got x={x.shape}, f={f.shape}")
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(f)):
        raise ValueError("residualization inputs must be finite")
    keep = [j for j in range(f.shape[1]) if np.std(f[:, j]) > 0.0]  # drop constant features
    cols = [np.ones(x.size)] + [f[:, j] for j in keep]
    design = np.ascontiguousarray(np.column_stack(cols))
    coef, *_ = np.linalg.lstsq(design, x, rcond=None)
    # errstate-suppress the spurious BLAS matmul FP flag (divide/invalid/overflow, platform-dep);
    # the finiteness assert below still fails loud on a genuine non-finite fit (no-silent-NaN).
    with np.errstate(all="ignore"):
        fitted = design @ coef
    if not np.all(np.isfinite(fitted)):
        raise ValueError("residualization produced non-finite fitted values")
    resid = x - fitted
    if not np.all(np.isfinite(resid)):
        raise ValueError("residualization produced non-finite residuals")
    return resid


def _perm_p(scores: np.ndarray, labels: np.ndarray, obs: float, *, n_perm: int, seed: int) -> float:
    n_perm = _positive_int("n_perm", n_perm)
    rng = np.random.default_rng(seed)
    perm = np.array([auc(scores, rng.permutation(labels)) for _ in range(n_perm)])
    return float((1.0 + np.sum(perm >= obs)) / (n_perm + 1.0))


def compare_rho_vs_baselines(
    rho_z: np.ndarray,
    features: dict[str, np.ndarray],
    labels: np.ndarray,
    *,
    n_perm: int,
    seed: int,
) -> dict:
    """rho vs each physical baseline + battery-residualized rho AUC/perm-p + feature<->rho corr."""
    rho_z = np.asarray(rho_z, dtype=np.float64).reshape(-1)
    labels = np.asarray(labels, dtype=int).reshape(-1)
    names = [n for n in FEATURE_NAMES if n in features]
    rho_auc = float(auc(rho_z, labels))
    per_feature: dict[str, dict] = {}
    for i, name in enumerate(names):
        f = np.asarray(features[name], dtype=np.float64).reshape(-1)
        f_auc = float(auc(f, labels))
        resid = _residualize_multi(rho_z, f)
        r_auc = float(auc(resid, labels))
        # correlation of the feature with rho (mechanism: which physical quantity tracks the shift).
        # Require BOTH series to vary: corrcoef with a constant vector is NaN, which would crash
        # json.dumps(allow_nan=False); 0.0 is the right degenerate value.
        corr = float(np.corrcoef(f, rho_z)[0, 1]) if (np.std(f) > 0 and np.std(rho_z) > 0) else 0.0
        per_feature[name] = {
            "feature_auc": f_auc,
            "rho_beats_feature": bool(rho_auc > f_auc),
            "rho_residualized_on_feature_auc": r_auc,
            "rho_residualized_on_feature_perm_p": _perm_p(
                resid, labels, r_auc, n_perm=n_perm, seed=seed + i
            ),
            "feature_rho_corr": corr,
        }
    mat = np.column_stack([np.asarray(features[n], dtype=np.float64).reshape(-1) for n in names])
    resid_all = _residualize_multi(rho_z, mat)
    resid_all_auc = float(auc(resid_all, labels))
    resid_all_p = _perm_p(resid_all, labels, resid_all_auc, n_perm=n_perm, seed=seed + 100)
    best_corr_feature = max(names, key=lambda n: abs(per_feature[n]["feature_rho_corr"]))
    return {
        "aligned_rho_auc": rho_auc,
        "rho_beats_all_features": bool(all(per_feature[n]["rho_beats_feature"] for n in names)),
        "rho_residualized_on_all_auc": resid_all_auc,
        "rho_residualized_on_all_perm_p": resid_all_p,
        "rho_adds_beyond_all_physical": bool(resid_all_p < 0.05 and resid_all_auc > 0.5),
        "dominant_rho_correlate": best_corr_feature,
        "dominant_rho_corr": per_feature[best_corr_feature]["feature_rho_corr"],
        "per_feature": per_feature,
    }


def _load_collect_alive_eval():
    """Import clip_vs_rho.collect_alive_eval so this reuses the IDENTICAL favorable-subset pool."""
    path = Path(__file__).with_name("clip_vs_rho.py")
    spec = importlib.util.spec_from_file_location("clip_vs_rho_for_mechanism", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.collect_alive_eval


def run(features_root: Path, out_dir: Path, *, alive_t: float, n_perm: int, seed: int) -> int:
    collect_alive_eval = _load_collect_alive_eval()
    rows_by_decoder: dict[str, tuple[Path, list[dict]]] = {}
    for table_path in sorted(features_root.glob("*/p0_table.json")):
        rows_by_decoder[table_path.parent.name] = (table_path.parent, _load_p0_rows(table_path))
    if not rows_by_decoder:
        print(f"[skip] no <decoder>/p0_table.json under {features_root}")
        return 0
    paths, _prompts, rho_z, labels, alive = collect_alive_eval(rows_by_decoder, alive_t=alive_t)
    if labels.size < 8 or labels.sum() == 0 or labels.sum() == labels.size:
        print(f"[skip] too few / single-class pooled eval images ({labels.size}); no comparison.")
        return 0
    feats = physical_features(paths)
    cmp = compare_rho_vs_baselines(rho_z, feats, labels, n_perm=n_perm, seed=seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "n_alive_cells": len(alive),
        "alive_cells": list(alive),
        "n_eval_images": int(labels.size),
        **cmp,
    }
    (out_dir / "rho_mechanism.json").write_text(
        json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8"
    )
    print(f"\nrho mechanism ({len(alive)} alive cells, {labels.size} eval imgs):")
    print(f"  aligned-rho AUC = {cmp['aligned_rho_auc']:.3f}")
    for name in (n for n in FEATURE_NAMES if n in cmp["per_feature"]):
        pf = cmp["per_feature"][name]
        beat = "rho>feat" if pf["rho_beats_feature"] else "feat>=rho"
        r_auc = pf["rho_residualized_on_feature_auc"]
        r_p = pf["rho_residualized_on_feature_perm_p"]
        print(
            f"  {name:12s} AUC={pf['feature_auc']:.3f} ({beat})  rho|feat AUC={r_auc:.3f} "
            f"p={r_p:.4f}  corr(feat,rho)={pf['feature_rho_corr']:+.3f}"
        )
    print(
        f"  rho | ALL physical: AUC={cmp['rho_residualized_on_all_auc']:.3f} "
        f"perm p={cmp['rho_residualized_on_all_perm_p']:.4f}"
    )
    print(
        f"  dominant rho correlate = {cmp['dominant_rho_correlate']} "
        f"(corr={cmp['dominant_rho_corr']:+.3f})"
    )
    if cmp["rho_adds_beyond_all_physical"]:
        print("  => rho carries CFG signal BEYOND every trivial physical baseline.")
    else:
        print("  => rho does NOT add beyond the physical battery -> it may BE the texture proxy.")
    print(f"-> {out_dir / 'rho_mechanism.json'}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="rho_mechanism")
    ap.add_argument(
        "--features-root", default=None, help="dir of <decoder>/p0_table.json (default: outputs/p0)"
    )
    ap.add_argument("--alive-t", type=float, default=1.5)
    ap.add_argument("--n-perm", type=int, default=2000)
    args = ap.parse_args(argv)
    features_root = Path(args.features_root) if args.features_root else (outputs_dir() / "p0")
    return run(
        features_root, outputs_dir() / "p0", alive_t=args.alive_t, n_perm=args.n_perm, seed=0
    )


if __name__ == "__main__":
    raise SystemExit(main())
