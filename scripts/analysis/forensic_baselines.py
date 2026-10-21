"""rho vs a CNN-free forensic toolbox on the favorable subset: ranking and held-out complementarity.

Produces: the CNN-free toolbox comparison of Section IV-E1 and the top-right bars of Fig. 1:
orientation-agnostic AUC_1 of the three descriptors and of rho on the favorable subset's
evaluation images (``compare.pool_auc.{spectral_slope,dct_benford,noise_kurtosis,rho}``), and the
fusion check (``compare.complementarity``: descriptor selected on the fitting half, LDA fusion with
rho, scoring-half gain and its 95% bootstrap interval ``gain_ci95`` from ``--n-boot`` 2,000
resamples). The Fig. 1 bars are drawn from this JSON.

Inputs: ``outputs/p0/<decoder>/p0_table.json`` and the main-grid PNGs it references (not
shipped; written by the generation run). CPU only (numpy + Pillow); no re-inversion, no model.

Outputs: ``outputs/p0/forensic_baselines.json`` (or ``--out-dir``) and
``outputs/figures/forensic_baselines.png``.

Usage: python scripts/analysis/forensic_baselines.py [--alive-t 1.5] [--n-boot 2000] [--seed 0]

``rho_mechanism.py`` asks whether rho is just a physical correlate of high CFG (oversaturation /
texture collapse) by residualizing it on a physical battery. This script asks a different question:
how does rho rank against standard CNN-free forensic descriptors, and does combining rho with the
best of them beat either one alone (i.e. is rho a distinct axis rather than a re-parameterization
of an existing descriptor)?

Three CNN-free forensic descriptors, each a scalar on one RGB image (computed from the stored PNGs,
each capturing DIFFERENT physics so the complementarity test is fair):

  * spectral_decay_slope -- log-log slope of the azimuthally-averaged radial power spectrum. The
    up-convolution / spectral-fingerprint family (Durall 2020, Dzanic 2020): synthesis perturbs the
    high-frequency spectral decay. (Frequency domain.)
  * dct_benford_chi2 -- chi-square divergence of the first-significant-digit distribution of 8x8
    block-DCT AC coefficients from Benford's law. A classic compression/forgery statistic; the
    relative magnitude floor makes it stable to overall brightness/contrast. (Coefficient domain.)
  * noise_residual_kurtosis -- excess kurtosis of an SRM high-pass residual: the non-Gaussianity of
    the generation noise, a steganalysis-flavored higher-order statistic (distinct from the mean
    residual ENERGY already in nuisance.srm). (Noise-residual / higher-moment domain.)

Two read-outs on the favorable-subset evaluation pool (selection imported from ``clip_vs_rho.py``,
so the images are identical to those of ``clip_vs_rho.py``, ``rho_mechanism.py`` and
``deep_probe.py``):

  - RANKING: orientation-agnostic AUC (max(a, 1-a), sign chosen on the evaluated pool) of
    aligned-rho and each forensic descriptor; rho's rank.
  - COMPLEMENTARITY: a leakage-controlled held-out test. Split the pool into a class-stratified
    TRAIN / TEST half (seed ``--seed``); standardize on TRAIN; pick the best forensic baseline by
    its TRAIN AUC (NOT test); fit a closed-form 2-feature Fisher LDA combination of
    [rho, best_forensic] on TRAIN; evaluate AUC on TEST. The gain = AUC(combo) - max(AUC(rho),
    AUC(best_forensic)) on TEST, with a TEST-set bootstrap CI (fixed fitted model, seed + 1).
    gain CI strictly > 0 => rho and the toolbox are complementary; gain ~ 0 => rho is either
    dominant-and-sufficient or redundant with the toolbox (the verdict says which).

Strict JSON (allow_nan=False); an undefined quantity (degenerate / underpowered split) is JSON null.
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

FORENSIC_NAMES = ("spectral_slope", "dct_benford", "noise_kurtosis")
# |DCT coef| below this FRACTION of the block-set's max magnitude is treated as a zero (excluded
# from the digit histogram). A *relative* floor makes the leading-digit law stable to overall
# brightness/contrast scaling -- exactly invariant under a power-of-10 rescale, and the membership
# set does not drift when the image is uniformly brightened/darkened.
_DIGIT_FLOOR_FRAC = 1e-6
_FLAT_EPS = 1e-12  # a near-constant image has no AC structure; report 0.0 not digitized noise
# Benford expected first-significant-digit probabilities P(d) = log10(1 + 1/d), d = 1..9.
_BENFORD = np.log10(1.0 + 1.0 / np.arange(1, 10, dtype=np.float64))
# SRM-style high-pass kernels for the noise residual (1st, 2nd order + the 3x3 "spam" kernel).
_HP_KERNELS = (
    np.array([[0.0, 0.0, 0.0], [0.0, -1.0, 1.0], [0.0, 0.0, 0.0]], dtype=np.float64),
    np.array([[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]], dtype=np.float64),
    np.array([[-1.0, 2.0, -1.0], [2.0, -4.0, 2.0], [-1.0, 2.0, -1.0]], dtype=np.float64) / 4.0,
)


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


def _gray(img: np.ndarray) -> np.ndarray:
    return 0.299 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2]


def _validate_rgb(img: np.ndarray) -> np.ndarray:
    img = np.asarray(img, dtype=np.float64)
    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError(f"image must be (H,W,3) RGB in [0,1], got shape {img.shape}")
    if min(img.shape[:2]) < 1:
        raise ValueError(f"image must have non-empty spatial dimensions, got shape {img.shape}")
    if not np.all(np.isfinite(img)):
        raise ValueError("image must contain only finite values")
    return img


def spectral_decay_slope(img: np.ndarray) -> float:
    """Slope of log(radial power) vs log(radius) of the azimuthally-averaged 2-D power spectrum.

    The up-convolution spectral fingerprint: synthesis alters how spectral power decays with
    frequency. We fit a single line through the radial profile (DC bin excluded) over integer-radius
    bins whose outer edge fits inside the largest centered circle, avoiding partial outer annuli and
    anisotropic corners. A flat / smooth image has a steep (very negative) slope; high-frequency
    content makes it shallower.
    """
    g = _gray(_validate_rgb(img))
    h, w = g.shape
    if min(h, w) < 4:
        return 0.0
    power = np.square(np.abs(np.fft.fftshift(np.fft.fft2(g))))
    yy, xx = np.ogrid[:h, :w]
    # Center on the fftshift DC location (index h//2, w//2) so the radius is exact for odd sizes.
    r = np.sqrt((yy - h // 2) ** 2 + (xx - w // 2) ** 2)
    r_int = r.astype(np.int64)
    r_limit = float(min(h, w)) / 2.0
    complete = (r_int > 0) & ((r_int.astype(np.float64) + 1.0) <= r_limit)
    if not bool(complete.any()):
        return 0.0
    r_max = int(r_int[complete].max())
    if r_max < 3:
        return 0.0
    bins = r_int[complete].ravel()
    tot = np.bincount(bins, weights=power[complete].ravel(), minlength=r_max + 1)
    cnt = np.bincount(bins, minlength=r_max + 1)
    radii = np.arange(1, r_max + 1)  # r=0 / DC already excluded
    prof = tot[1 : r_max + 1] / np.maximum(cnt[1 : r_max + 1], 1)
    mask = prof > 0.0
    if int(mask.sum()) < 3:
        return 0.0
    x = np.log(radii[mask].astype(np.float64))
    y = np.log(prof[mask])
    slope = float(np.polyfit(x, y, 1)[0])
    return slope if np.isfinite(slope) else 0.0


def dct_benford_chi2(img: np.ndarray) -> float:
    """Chi-square divergence of 8x8 block-DCT AC first-digit law from Benford's law.

    Natural-image block-DCT AC coefficients follow Benford's first-significant-digit distribution;
    synthesis / recompression perturbs it. The leading significant digit is read from the log10
    mantissa; coefficients below ``_DIGIT_FLOOR_FRAC`` of the per-image max magnitude are excluded
    (a relative floor -> stable to overall brightness). Returns sum_d (P(d) - Benford(d))^2 /
    Benford(d) over d = 1..9 (0.0 if no blocks or a flat image with no AC structure).
    """
    from scipy.fft import dctn

    g = _gray(_validate_rgb(img))
    h, w = g.shape
    bh, bw = h // 8, w // 8
    if bh < 1 or bw < 1:
        return 0.0
    blocks = g[: bh * 8, : bw * 8].reshape(bh, 8, bw, 8).transpose(0, 2, 1, 3).reshape(-1, 8, 8)
    coef = np.asarray(dctn(blocks, type=2, axes=(1, 2), norm="ortho"))
    mags = np.abs(coef.reshape(coef.shape[0], 64)[:, 1:]).ravel()  # drop DC at flat index 0
    mmax = float(mags.max()) if mags.size else 0.0
    if mmax <= _FLAT_EPS:  # constant / structureless block set -> no first-digit law to test
        return 0.0
    mags = mags[mags > _DIGIT_FLOOR_FRAC * mmax]
    if mags.size == 0:
        return 0.0
    mant = np.log10(mags)
    mant = mant - np.floor(mant)  # fractional part in [0, 1)
    digits = np.floor(np.power(10.0, mant)).astype(np.int64)  # leading digit in 1..9
    digits = np.clip(digits, 1, 9)
    hist = np.bincount(digits, minlength=10)[1:10].astype(np.float64)
    p = hist / hist.sum()
    chi2 = float(np.sum((p - _BENFORD) ** 2 / _BENFORD))
    return chi2 if np.isfinite(chi2) else 0.0


def noise_residual_kurtosis(img: np.ndarray) -> float:
    """Mean excess kurtosis of a small bank of SRM high-pass residuals (noise non-Gaussianity).

    Generation noise is not Gaussian the way sensor noise is; the residual's 4th standardized moment
    captures that, distinct from the mean residual ENERGY in nuisance.srm. Averaged over the kernel
    bank; a constant residual (zero variance) contributes 0.0.
    """
    from scipy.ndimage import convolve

    g = _gray(_validate_rgb(img))
    vals: list[float] = []
    for kernel in _HP_KERNELS:
        res = convolve(g, kernel, mode="reflect")
        sd = float(np.std(res))
        if not (np.isfinite(sd) and sd > 1e-12):
            vals.append(0.0)
            continue
        z = (res - float(np.mean(res))) / sd
        k = float(np.mean(z**4) - 3.0)  # excess kurtosis
        vals.append(k if np.isfinite(k) else 0.0)
    return float(np.mean(vals)) if vals else 0.0


FORENSICS = {
    "spectral_slope": spectral_decay_slope,
    "dct_benford": dct_benford_chi2,
    "noise_kurtosis": noise_residual_kurtosis,
}


def forensic_features(paths: list[Path]) -> dict[str, np.ndarray]:
    """Per-image forensic detector scores for the eval PNGs; returns {name: array(N,)}."""
    from PIL import Image  # local import: Pillow is only needed when the PNGs are read

    cols: dict[str, list[float]] = {name: [] for name in FORENSIC_NAMES}
    for p in paths:
        arr = np.asarray(Image.open(p).convert("RGB"), dtype=np.float64) / 255.0
        for name in FORENSIC_NAMES:
            cols[name].append(float(FORENSICS[name](arr)))
    out = {name: np.asarray(vals, dtype=np.float64) for name, vals in cols.items()}
    for name, arr in out.items():
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"forensic feature {name!r} produced a non-finite value")
    return out


def _oriented_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Orientation-agnostic AUC (a detector may separate in either direction)."""
    a = float(auc(scores, labels))
    return max(a, 1.0 - a)


def _train_oriented_auc(scores: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    """Return (sign, AUC) with the score direction selected on this split only."""
    a = float(auc(scores, labels))
    return (1.0, a) if a >= 0.5 else (-1.0, 1.0 - a)


def _stratified_split(labels: np.ndarray, *, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic stratified half/half split; both classes present in TRAIN and TEST."""
    rng = np.random.default_rng(seed)
    train: list[int] = []
    test: list[int] = []
    for cls in (0, 1):
        idx = np.where(labels == cls)[0]
        rng.shuffle(idx)
        cut = idx.size // 2
        train.extend(idx[:cut].tolist())
        test.extend(idx[cut:].tolist())
    return np.array(sorted(train), dtype=int), np.array(sorted(test), dtype=int)


def _lda_direction(x2: np.ndarray, y: np.ndarray, *, ridge: float = 1e-3) -> np.ndarray:
    """Closed-form 2-feature Fisher LDA direction (toward H1); ridge-stabilized covariance."""
    x1 = x2[y == 1]
    x0 = x2[y == 0]
    mu1 = x1.mean(axis=0)
    mu0 = x0.mean(axis=0)
    n = x2.shape[0]
    cov = (x1 - mu1).T @ (x1 - mu1) + (x0 - mu0).T @ (x0 - mu0)
    cov = cov / max(n - 2, 1) + ridge * np.eye(x2.shape[1])
    w = np.linalg.solve(cov, mu1 - mu0)
    return np.asarray(w, dtype=np.float64)


def complementarity_test(
    rho_z: np.ndarray,
    features: dict[str, np.ndarray],
    labels: np.ndarray,
    *,
    seed: int,
    n_boot: int,
) -> dict:
    """Held-out, leakage-controlled test of whether rho + the best forensic beat either alone.

    Best forensic is selected on the TRAIN half (no test leakage); the LDA combo is fit on TRAIN and
    scored on TEST; the gain over the best single TEST AUC gets a TEST-set bootstrap CI. The fitted
    model (best baseline + LDA direction + standardization) is FIXED during the bootstrap, which
    only resamples TEST to estimate the gain's sampling variability.
    """
    n_boot = _positive_int("n_boot", n_boot)
    rho_z = np.asarray(rho_z, dtype=np.float64).reshape(-1)
    labels = np.asarray(labels, dtype=int).reshape(-1)
    names = [n for n in FORENSIC_NAMES if n in features]
    if not names:
        raise ValueError("no forensic features supplied")
    tr, te = _stratified_split(labels, seed=seed)
    y_tr, y_te = labels[tr], labels[te]
    if (
        min(int(y_tr.sum()), int(y_te.sum())) == 0
        or y_tr.sum() == y_tr.size
        or y_te.sum() == y_te.size
    ):
        return {
            "usable": False,
            "reason": "a split half is single-class; pool too small/imbalanced",
        }

    # Standardize every candidate feature on TRAIN stats only.
    cand = {
        "rho": rho_z,
        **{n: np.asarray(features[n], dtype=np.float64).reshape(-1) for n in names},
    }
    std_full: dict[str, np.ndarray] = {}
    for nm, col in cand.items():
        mu, sd = float(col[tr].mean()), float(col[tr].std())
        std_full[nm] = (col - mu) / (sd if sd > 0 else 1.0)

    train_sign: dict[str, float] = {}
    train_auc: dict[str, float] = {}
    for nm in cand:
        sign, value = _train_oriented_auc(std_full[nm][tr], y_tr)
        train_sign[nm] = sign
        train_auc[nm] = value
    test_auc = {nm: float(auc(train_sign[nm] * std_full[nm][te], y_te)) for nm in cand}
    best_forensic = max(names, key=lambda n: train_auc[n])  # selected on TRAIN -> no winner's curse

    # LDA combination of [rho, best_forensic] fit on TRAIN.
    x_tr = np.column_stack([std_full["rho"][tr], std_full[best_forensic][tr]])
    x_te = np.column_stack([std_full["rho"][te], std_full[best_forensic][te]])
    w = _lda_direction(x_tr, y_tr)
    combo_te = x_te @ w
    combo_test_auc = float(auc(combo_te, y_te))  # LDA already orients toward H1
    best_single_test = max(test_auc["rho"], test_auc[best_forensic])
    gain = combo_test_auc - best_single_test

    # TEST-set bootstrap of the gain (fixed model). Single-class resamples are skipped.
    rng = np.random.default_rng(seed + 1)
    rho_te_std = std_full["rho"][te]
    base_te_std = std_full[best_forensic][te]
    gains: list[float] = []
    m = te.size
    for _ in range(n_boot):
        bi = rng.integers(0, m, m)
        yb = y_te[bi]
        if yb.sum() == 0 or yb.sum() == yb.size:
            continue
        cb = np.column_stack([rho_te_std[bi], base_te_std[bi]]) @ w
        ca = float(auc(cb, yb))
        ra = float(auc(train_sign["rho"] * rho_te_std[bi], yb))
        ba = float(auc(train_sign[best_forensic] * base_te_std[bi], yb))
        gains.append(ca - max(ra, ba))
    boot = np.array(gains, dtype=np.float64)
    ci_lo = float(np.quantile(boot, 0.025)) if boot.size else None
    ci_hi = float(np.quantile(boot, 0.975)) if boot.size else None
    p_gain_le_0 = float((1.0 + np.sum(boot <= 0.0)) / (boot.size + 1.0)) if boot.size else None

    complementary = bool(gain > 0.0 and ci_lo is not None and ci_lo > 0.0)
    return {
        "usable": True,
        "n_train": int(tr.size),
        "n_test": int(te.size),
        "best_forensic": best_forensic,
        "train_score_sign": {nm: float(train_sign[nm]) for nm in cand},
        "test_auc": {nm: float(test_auc[nm]) for nm in cand},
        "train_auc": {nm: float(train_auc[nm]) for nm in cand},
        "combo_test_auc": combo_test_auc,
        "best_single_test_auc": float(best_single_test),
        "gain": float(gain),
        "gain_ci95": [ci_lo, ci_hi],
        "gain_p_le_0": p_gain_le_0,
        "n_boot_used": int(boot.size),
        "rho_complementary_to_forensics": complementary,
    }


def compare_rho_vs_forensics(
    rho_z: np.ndarray,
    features: dict[str, np.ndarray],
    labels: np.ndarray,
    *,
    seed: int,
    n_boot: int,
) -> dict:
    """Full ranking (rho + forensics) on the pool + the held-out complementarity verdict."""
    rho_z = np.asarray(rho_z, dtype=np.float64).reshape(-1)
    labels = np.asarray(labels, dtype=int).reshape(-1)
    names = [n for n in FORENSIC_NAMES if n in features]
    pool_auc = {"rho": _oriented_auc(rho_z, labels)}
    for n in names:
        pool_auc[n] = _oriented_auc(np.asarray(features[n], dtype=np.float64).reshape(-1), labels)
    ranking = sorted(pool_auc.items(), key=lambda kv: kv[1], reverse=True)
    rho_rank = 1 + [k for k, _ in ranking].index("rho")
    comp = complementarity_test(rho_z, features, labels, seed=seed, n_boot=n_boot)
    return {
        "pool_auc": {k: float(v) for k, v in pool_auc.items()},
        "ranking": [{"detector": k, "pool_auc": float(v)} for k, v in ranking],
        "rho_rank": int(rho_rank),
        "rho_is_top_detector": bool(ranking[0][0] == "rho"),
        "complementarity": comp,
        "verdict": _verdict(rho_rank, ranking[0][0], comp),
    }


def _verdict(rho_rank: int, top: str, comp: dict) -> dict:
    """Does rho add a distinct forensic axis (held-out combo beats either piece)?"""
    if not comp.get("usable"):
        return {"label": "INCONCLUSIVE", "reason": comp.get("reason", "complementarity unusable")}
    if comp["rho_complementary_to_forensics"]:
        return {
            "label": "RHO_ADDS_DISTINCT_AXIS",
            "reason": "held-out [rho + best-forensic] LDA combo beats either alone (gain CI95 > 0) "
            "-> rho is a new axis, not a re-parameterization of an existing forensic detector",
        }
    if top == "rho":
        return {
            "label": "RHO_DOMINANT_BUT_NOT_COMPLEMENTARY",
            "reason": "rho is the top single detector but the combo does not significantly beat it "
            "-> rho is sufficient here; the forensic baselines add no held-out signal on top",
        }
    return {
        "label": "RHO_NOT_DISTINCT",
        "reason": "a forensic baseline matches/leads rho and the combo adds no held-out gain "
        "-> on this pool rho is not separable from the forensic toolbox",
    }


def _make_figure(payload: dict) -> None:
    plt = viz_pyplot()
    if plt is None:
        return
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.8))
    rank = payload["compare"]["ranking"]
    names = [r["detector"] for r in rank]
    vals = [r["pool_auc"] for r in rank]
    colors = ["#c0392b" if n == "rho" else "#7f8c8d" for n in names]
    axes[0].bar(range(len(names)), vals, color=colors)
    axes[0].set_xticks(range(len(names)))
    axes[0].set_xticklabels(names, rotation=30, ha="right", fontsize=8)
    axes[0].axhline(0.5, color="0.4", ls="--", lw=1.0)
    axes[0].set_ylabel("pool AUC (orientation-agnostic)")
    axes[0].set_title("rho vs forensic toolbox")
    comp = payload["compare"]["complementarity"]
    if comp.get("usable"):
        bars = {
            "rho": comp["test_auc"]["rho"],
            comp["best_forensic"]: comp["test_auc"][comp["best_forensic"]],
            "combo": comp["combo_test_auc"],
        }
        axes[1].bar(range(len(bars)), list(bars.values()), color=["#c0392b", "#7f8c8d", "#27ae60"])
        axes[1].set_xticks(range(len(bars)))
        axes[1].set_xticklabels(list(bars.keys()), fontsize=8)
        axes[1].axhline(0.5, color="0.4", ls="--", lw=1.0)
        axes[1].set_ylabel("held-out TEST AUC")
        axes[1].set_title(f"complementarity (gain={comp['gain']:+.3f})")
    viz_save(fig, "forensic_baselines")


def viz_pyplot():
    from cfgid import viz

    return viz.pyplot()


def viz_save(fig, name: str) -> None:
    from cfgid import viz

    viz.save_figure(fig, name)


def _load_collect_alive_eval():
    """Import clip_vs_rho.collect_alive_eval so this reuses the IDENTICAL favorable-subset pool."""
    path = Path(__file__).with_name("clip_vs_rho.py")
    spec = importlib.util.spec_from_file_location("clip_vs_rho_for_forensics", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.collect_alive_eval


def run(features_root: Path, out_dir: Path, *, alive_t: float, seed: int, n_boot: int) -> int:
    collect_alive_eval = _load_collect_alive_eval()
    rows_by_decoder: dict[str, tuple[Path, list[dict]]] = {}
    for table_path in sorted(features_root.glob("*/p0_table.json")):
        rows_by_decoder[table_path.parent.name] = (table_path.parent, _load_p0_rows(table_path))
    if not rows_by_decoder:
        print(f"[skip] no <decoder>/p0_table.json under {features_root}")
        return 0
    paths, _prompts, rho_z, labels, alive = collect_alive_eval(rows_by_decoder, alive_t=alive_t)
    if labels.size < 16 or labels.sum() == 0 or labels.sum() == labels.size:
        print(f"[skip] too few / single-class pooled eval images ({labels.size}); need >=16.")
        return 0
    feats = forensic_features(paths)
    compare = compare_rho_vs_forensics(rho_z, feats, labels, seed=seed, n_boot=n_boot)
    payload = {
        "n_alive_cells": len(alive),
        "alive_cells": list(alive),
        "n_eval_images": int(labels.size),
        "compare": compare,
        "note": (
            "rho vs CNN-free forensic detectors on the locked alive-eval pool (same pool as "
            "rho_mechanism.py). "
            "Complementarity is a leakage-controlled held-out test: best forensic picked on TRAIN, "
            "LDA combo fit on TRAIN, gain + bootstrap CI measured on TEST. NO re-inversion."
        ),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "forensic_baselines.json"
    out_path.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    _make_figure(payload)

    print(f"\nrho vs forensic toolbox ({len(alive)} alive cells, {labels.size} eval imgs):")
    for r in compare["ranking"]:
        tag = " <- rho" if r["detector"] == "rho" else ""
        print(f"  {r['detector']:16s} pool AUC={r['pool_auc']:.3f}{tag}")
    comp = compare["complementarity"]
    if comp.get("usable"):
        print(
            f"  held-out: rho={comp['test_auc']['rho']:.3f} "
            f"{comp['best_forensic']}={comp['test_auc'][comp['best_forensic']]:.3f} "
            f"combo={comp['combo_test_auc']:.3f}  gain={comp['gain']:+.3f} "
            f"CI95={comp['gain_ci95']}"
        )
    print(f"  VERDICT: {compare['verdict']['label']} ({compare['verdict']['reason']})")
    print(f"-> {out_path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="forensic_baselines",
        description="rho vs CNN-free forensic detectors (ranking + held-out complementarity)",
    )
    ap.add_argument(
        "--features-root", default=None, help="dir of <decoder>/p0_table.json (default: outputs/p0)"
    )
    ap.add_argument("--alive-t", type=float, default=1.5)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", default=None, help="default: outputs/p0")
    args = ap.parse_args(argv)
    features_root = Path(args.features_root) if args.features_root else (outputs_dir() / "p0")
    out_dir = Path(args.out_dir) if args.out_dir else (outputs_dir() / "p0")
    return run(features_root, out_dir, alive_t=args.alive_t, seed=args.seed, n_boot=args.n_boot)


if __name__ == "__main__":
    raise SystemExit(main())
