"""PSF grounding of the sigma_eff detectability decay: decay exponent vs measured sigma_core.

Asks whether the per-decoder loss of rho detectability along the degradation axis sigma_eff is
ordered by the decoder's MEASURED point-spread function (PSF) width: does a wider intrinsic PSF
(larger sigma_core) lose detectability FASTER as blur is added?

Per above-chance cell the Gaussian deflection is recovered as d = 2*Phi^-1(AUC1); cells with
AUC1 <= 0.5 give no positive deflection and are dropped. Per decoder a power-law decay
d ~ sigma_eff^-p is fitted on log-log axes; sigma_core is the calibrated sigma_eff at
sigma_blur=0 (outputs/sigma_calib). The PSF-grounding test is the Pearson correlation of p with
sigma_core across decoders, reported over all decoders and over the distinct-VAE decoders
(SD1.5, SD2.1, SDXL; PixArt-Sigma and HunyuanDiT reuse the SDXL VAE). With three distinct VAEs
and few cells per decoder this is a descriptive trend, not a law; no critical sigma_eff is
claimed.

Role: helper module. Its ``_deflection``, ``_powerlaw_fit``, ``_sigma_core`` and
``_SHARED_VAE_DECODERS`` are reused by psf_grounding_features.py and feature_comparison.py,
which build the PSF-order column of Table X (Section IV-E3); this script's own JSON is not
printed in the paper.

Inputs: ``outputs/p0/detector_eval.json`` (per-cell AUC), ``outputs/p0/master_table.json``
(sigma_eff per cell) and ``outputs/sigma_calib/<decoder>.json``. Pure numpy, CPU only.

Outputs: ``outputs/p0/psf_grounding.json`` (or ``--out``); strict JSON (allow_nan=False), an
undefined fit is JSON null.

Usage:
    uv run --no-sync python scripts/analysis/psf_grounding.py [--root outputs/p0]
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from cfgid.paths import outputs_dir

# AUC at/below chance gives a non-positive deflection -> excluded from the log fit.
_AUC_FLOOR = 0.5 + 1e-6

# pixart_sigma + hunyuandit reuse the SDXL VAE (models.yaml: pixart_sigma_sdxlvae; see
# channel/sigma_probe.py, channel/inversion/pixart_ddim.py). Their sigma_core matches
# SDXL's, so they are not independent PSF points and are excluded from the distinct-VAE
# (SD-series) correlation reported alongside the all-decoder one.
_SHARED_VAE_DECODERS = frozenset({"pixart_sigma", "hunyuandit"})


def _deflection(auc: float) -> float | None:
    """d = 2*Phi^-1(AUC); the Gaussian-location separation. None if AUC <= chance."""
    if not (np.isfinite(auc) and auc > _AUC_FLOOR):
        return None
    from statistics import NormalDist

    return 2.0 * NormalDist().inv_cdf(float(auc))


def _per_cell_rows(root: Path) -> list[dict]:
    """Per alive cell: (decoder, sigma_blur, sigma_eff, auc, d); sigma_eff from master_table."""
    det = json.loads((root / "detector_eval.json").read_text(encoding="utf-8"))
    cells = det.get("per_cell") or []
    se_by: dict[tuple[str, float], float] = {}  # (decoder, sigma_blur) -> sigma_eff
    mt_path = root / "master_table.json"
    if mt_path.is_file():
        mt = json.loads(mt_path.read_text(encoding="utf-8"))
        for r in mt.get("rows", []):
            if r.get("sigma_eff") is not None:
                se_by[(str(r["decoder"]), float(r["sigma_blur"]))] = float(r["sigma_eff"])
    out: list[dict] = []
    for c in cells:
        auc = c.get("auc") if c.get("auc") is not None else c.get("auc1")
        if auc is None:
            continue
        d = _deflection(float(auc))
        if d is None:
            continue
        dec = str(c.get("decoder", ""))
        sb = float(c.get("sigma_blur", 0.0))
        out.append(
            {
                "decoder": dec,
                "sigma_blur": sb,
                "sigma_eff": se_by.get((dec, sb)),
                "auc": float(auc),
                "d": float(d),
            }
        )
    return out


def _sigma_core(root: Path, decoder: str) -> float | None:
    """Decoder intrinsic PSF width = calibrated sigma at sigma_blur=0 (sigma_calib/<d>.json)."""
    p = root.parent / "sigma_calib" / f"{decoder}.json"
    if not p.is_file():
        return None
    sc = json.loads(p.read_text(encoding="utf-8"))
    # per_sigma_blur {sigma_blur: sigma_eff}; sb=0 entry is sigma_core. Fall back to sigma_hat.
    psb = sc.get("per_sigma_blur") or {}
    for k, v in psb.items():
        try:
            if abs(float(k)) < 1e-9:
                se = v.get("sigma_eff") if isinstance(v, dict) else v
                return float(se) if se is not None else None
        except (TypeError, ValueError):
            continue
    return float(sc["sigma_hat"]) if sc.get("sigma_hat") is not None else None


def _powerlaw_fit(x: list[float], y: list[float]) -> dict:
    """Fit ln(y) = ln(a) - p*ln(x) (power law y=a*x^-p). p None if < 2 points or degenerate."""
    xs = np.log(np.asarray(x, dtype=np.float64))
    ys = np.log(np.asarray(y, dtype=np.float64))
    n = xs.size
    if n < 2 or np.ptp(xs) <= 0:
        return {"p": None, "a": None, "r2": None, "n": int(n)}
    slope, intercept = np.polyfit(xs, ys, 1)
    yhat = slope * xs + intercept
    ss_res = float(np.sum((ys - yhat) ** 2))
    ss_tot = float(np.sum((ys - ys.mean()) ** 2))
    r2 = (1.0 - ss_res / ss_tot) if (ss_tot > 0 and n > 2) else None
    return {"p": float(-slope), "a": float(math.exp(intercept)), "r2": r2, "n": int(n)}


def _decoder_decay(root: Path, dec: str, cells: list[dict]) -> dict:
    """Power-law decay fit of d vs sigma_eff for one decoder (reconstruct sigma_eff if needed)."""
    cells = sorted(cells, key=lambda c: c["sigma_blur"])
    score = _sigma_core(root, dec)
    if all(c["sigma_eff"] is not None for c in cells):
        xs = [c["sigma_eff"] for c in cells]
        axis = "sigma_eff"
    elif score is not None:
        xs = [math.sqrt(score**2 + c["sigma_blur"] ** 2) for c in cells]
        axis = "sigma_eff(reconstructed)"
    else:
        xs = [c["sigma_blur"] + 1.0 for c in cells]
        axis = "sigma_blur+1"
    fit = _powerlaw_fit(xs, [c["d"] for c in cells])
    return {
        "decoder": dec,
        "sigma_core": score,
        "n_alive": len(cells),
        "axis": axis,
        "decay_exponent_p": fit["p"],
        "fit_r2": fit["r2"],
        "cells": [
            {
                "sigma_blur": c["sigma_blur"],
                "sigma_eff": c["sigma_eff"],
                "auc": c["auc"],
                "d": c["d"],
            }
            for c in cells
        ],
    }


def run(root: Path, out_path: Path) -> int:
    if not (root / "detector_eval.json").is_file():
        print(f"[skip] no detector_eval.json under {root}")
        return 0
    by_decoder: dict[str, list[dict]] = {}
    for r in _per_cell_rows(root):
        by_decoder.setdefault(r["decoder"], []).append(r)
    per_decoder = [_decoder_decay(root, dec, cells) for dec, cells in sorted(by_decoder.items())]

    # PSF-grounding: decay exponent p vs sigma_core across decoders that have BOTH.
    grounded = [
        d for d in per_decoder if d["sigma_core"] is not None and d["decay_exponent_p"] is not None
    ]
    grounded.sort(key=lambda d: d["sigma_core"])
    cores = [d["sigma_core"] for d in grounded]
    ps = [d["decay_exponent_p"] for d in grounded]
    pearson: float | None = None
    monotone: bool | None = None
    if len(grounded) >= 3:
        c = np.corrcoef(cores, ps)[0, 1]
        pearson = float(c) if np.isfinite(c) else None
        up = all(ps[i] <= ps[i + 1] for i in range(len(ps) - 1))
        down = all(ps[i] >= ps[i + 1] for i in range(len(ps) - 1))
        monotone = bool(up or down)

    # Distinct-VAE subset (SD-series scope): drop decoders that reuse the SDXL VAE, so the
    # correlation is over genuinely distinct VAEs. The all-decoder pearson above keeps the
    # shared-VAE replicate(s) as a robustness check: such a point sits at SDXL's sigma_core
    # with a different decay, lowering the all-decoder pearson without breaking the rank order.
    distinct = [d for d in grounded if d["decoder"] not in _SHARED_VAE_DECODERS]
    pearson_distinct: float | None = None
    if len(distinct) >= 3:
        dc = np.corrcoef(
            [d["sigma_core"] for d in distinct], [d["decay_exponent_p"] for d in distinct]
        )[0, 1]
        pearson_distinct = float(dc) if np.isfinite(dc) else None

    verdict = _verdict(grounded, pearson, monotone)
    payload = {
        "per_decoder": per_decoder,
        "psf_grounding": {
            "decoders_with_core_and_fit": [d["decoder"] for d in grounded],
            "sigma_core": cores,
            "decay_exponent_p": ps,
            "pearson_p_vs_sigma_core": pearson,
            "monotone": monotone,
            "n_decoders": len(grounded),
            "decoders_distinct_vae": [d["decoder"] for d in distinct],
            "pearson_distinct_vae": pearson_distinct,
            "n_distinct_vae": len(distinct),
        },
        "verdict": verdict,
        "note": (
            "d = 2*Phi^-1(AUC1) per alive cell; per-decoder power-law decay d ~ sigma_eff^-p; "
            "PSF-grounding = is p ordered by measured sigma_core (wider PSF -> faster "
            "collapse). A monotone trend grounds the sigma_eff order parameter in the PSF, "
            "but few decoders + thin sampling make it a SUPPORTED TREND, not a law; no "
            "critical sigma_eff* claimed. Residual radial-PSD grounding needs the residual store "
            "(server-side) and is a separate, deeper check. pearson_distinct_vae is the "
            "same correlation over the distinct-VAE decoders only (SD-series); the "
            "all-decoder pearson keeps the shared-SDXL-VAE replicate(s) as a robustness check."
        ),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")

    print("PSF-grounding of the sigma_eff detectability collapse:")
    for d in per_decoder:
        print(
            f"  {d['decoder']:12s} sigma_core={_fmt(d['sigma_core'])} n_alive={d['n_alive']} "
            f"decay p={_fmt(d['decay_exponent_p'])} (R2={_fmt(d['fit_r2'])}) [{d['axis']}]"
        )
    print(
        f"  p vs sigma_core: pearson={_fmt(pearson)} monotone={monotone} n={len(grounded)} "
        f"| distinct-VAE pearson={_fmt(pearson_distinct)} n={len(distinct)}"
    )
    print(f"  VERDICT: {verdict['label']} -- {verdict['reason']}")
    print(f"-> {out_path}")
    return 0


def _verdict(grounded: list[dict], pearson: float | None, monotone: bool | None) -> dict:
    if len(grounded) < 3:
        return {
            "label": "INSUFFICIENT",
            "reason": f"only {len(grounded)} decoders with sigma_core + a fit; need >=3",
        }
    if monotone and pearson is not None and pearson > 0.8:
        return {
            "label": "PSF_GROUNDED_TREND",
            "reason": (
                f"decay exponent p is monotone in measured sigma_core (pearson {pearson:.2f}) over "
                f"{len(grounded)} decoders -> wider PSF -> faster collapse; grounds the sigma_eff "
                "order parameter in the measured PSF. SUPPORTED TREND (few decoders), not a law."
            ),
        }
    return {
        "label": "NOT_GROUNDED",
        "reason": (
            f"decay exponent p not ordered by sigma_core (pearson {pearson}, monotone {monotone}) "
            "-> per-decoder collapse not explained by the measured PSF width; stays empirical."
        ),
    }


def _fmt(x: float | None) -> str:
    return "  —" if x is None else f"{x:.3f}"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="psf_grounding", description="PSF-grounding of the sigma_eff detectability collapse"
    )
    ap.add_argument("--root", default=None, help="dir with detector_eval.json (def: outputs/p0)")
    ap.add_argument("--out", default=None, help="output JSON (def: <root>/psf_grounding.json)")
    args = ap.parse_args(argv)
    root = Path(args.root) if args.root else (outputs_dir() / "p0")
    out_path = Path(args.out) if args.out else (root / "psf_grounding.json")
    return run(root, out_path)


if __name__ == "__main__":
    raise SystemExit(main())
