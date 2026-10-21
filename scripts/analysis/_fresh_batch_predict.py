"""FIT-only deployed-head fitting for fresh_batch_validation's preregistration stage.

Implements ``fresh_batch_validation.py --stage predict`` (Section IV-D). For each decoder it
rebuilds the (rho, m) pools of the main-grid sigma_blur=0 cell at w=20 vs w=1 from the FIT split
only (NULL, CAL and TEST rows never enter a pool), fits the deployed Pair-LDA head of Eq. (11)
(FIT-H0 per-coordinate standardization, Ledoit--Wolf FIT-H0 covariance, weights
Sigma^{-1} * standardized H1 shift), and freezes in ``frozen_<dec>.json``, before any image is
generated:

  * the head weights (m0, s0, wvec);
  * FIT-null quantile thresholds at alpha = 0.05 and 0.01 (the FIT-null rows of Table VI);
  * the FIT deflection S_eff (in FIT-H0 score standard deviations) and the Gaussian same-cell
    predictions AUC_K = Phi(sqrt(K) S_eff / sqrt(2)) for K in {1, 2, 4, 8, 16} (Eqs. (12) and
    (14); the predicted curves of Fig. 4);
  * the generation plan (prompt offset, seed base, n_prompts, n_shards, steps, w list);
  * ``frozen_id``: SHA-256 over the canonical JSON of all of the above.

If a prospective-calibration freeze (frozen_cal_<dec>.json from freeze_prospective_cal.py) is
present in the target directory, its digest is re-verified, its Pair-LDA head must equal this fit
within PROSPECTIVE_HEAD_TOL, and the digest is stored as ``prospective_cal_sha256`` in the frozen
artifact, so ``frozen_id`` (and therefore every generated row) is bound to that calibration.

Inputs: outputs/p0/<dec>/p0_table.json and the server-only feature store (compact rho/E_tex and
residual_v1); CPU. Refuses to overwrite an existing freeze, row file or summary.
Outputs: <fresh dir>/frozen_<dec>.json (outputs/fresh_batch by default, or $CFGID_FRESH_DIR).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from _fresh_batch_support import (
    ALPHAS,
    EXPECTED_CHANNEL,
    FROZEN_SCHEMA_VERSION,
    GROUP_K,
    MASK_Q,
    PROMPT_OFFSET,
    RHO_REPRODUCTION_TOL,
    SEED_BASE,
    W_H0,
    W_H1,
    WINSOR,
    finite_int,
    finite_vector,
    frozen_cal_sha256,
    frozen_id,
    frozen_path,
    m_masked_winsor,
    write_json,
)
from scipy.stats import norm

from cfgid.bounds.wstar_cov import ledoit_wolf_cov
from cfgid.channel.rho import image_rho_summary
from cfgid.featurestore import read_compact, read_residual_v1
from cfgid.numerics import finite_float, loads_strict_json
from cfgid.sweep.evidence import _require_row_store_alignment
from cfgid.sweep.neff_scaling import _load_p0_rows

PROSPECTIVE_HEAD_TOL = 1e-6


def _refuse_existing_fresh_artifacts(decoder: str) -> None:
    """A preregistration happens once: abort if any plan, row file or summary already exists."""
    out_dir = frozen_path(decoder).parent
    candidates = [
        frozen_path(decoder),
        out_dir / f"frozen_{decoder}.json.tmp",
        out_dir / f"{decoder}_summary.json",
        *sorted(out_dir.glob(f"{decoder}_rows*.jsonl")),
    ]
    existing = [path for path in candidates if path.exists()]
    if existing:
        raise SystemExit(
            f"{decoder}: refusing to refreeze over an existing preregistration/run; "
            f"move/remove first: {[str(path) for path in existing]}"
        )


def _prospective_cal_binding(
    decoder: str, *, n_prompts: int, n_shards: int
) -> tuple[str, dict[str, np.ndarray]] | None:
    """Validate and return an optional pre-generation calibration digest."""
    path = frozen_path(decoder).parent / f"frozen_cal_{decoder}.json"
    if not path.is_file():
        return None
    raw = loads_strict_json(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise SystemExit(f"{path}: prospective calibration freeze must be a JSON object")
    recorded = raw.get("freeze_sha256")
    if not isinstance(recorded, str) or recorded != frozen_cal_sha256(raw):
        raise SystemExit(f"{path}: prospective calibration sha256 is missing or invalid")
    if raw.get("schema") != "frozen_prospective_cal_v1" or raw.get("decoder") != decoder:
        raise SystemExit(f"{path}: prospective calibration schema/decoder mismatch")
    plan = raw.get("fresh_plan")
    expected = {
        "prompt_offset": PROMPT_OFFSET,
        "seed_base": SEED_BASE,
        "n_prompts": n_prompts,
        "n_shards": n_shards,
        "steps": EXPECTED_CHANNEL["steps"],
        "w_list": [W_H0, W_H1],
    }
    if plan != expected:
        raise SystemExit(f"{path}: prospective fresh_plan {plan!r} != requested plan {expected!r}")
    readouts = raw.get("readouts")
    if not isinstance(readouts, dict) or not isinstance(readouts.get("lda_dep"), dict):
        raise SystemExit(f"{path}: prospective calibration lacks readouts.lda_dep")
    weights = readouts["lda_dep"].get("weights")
    if not isinstance(weights, dict) or set(weights) != {"m0", "s0", "wvec"}:
        raise SystemExit(f"{path}: prospective lda_dep weights are missing or malformed")
    frozen_weights = {
        name: finite_vector(f"{path}: lda_dep.weights.{name}", weights[name], (2,))
        for name in ("m0", "s0", "wvec")
    }
    return recorded, frozen_weights


def _fit_pools(decoder: str, decoder_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    rows = _load_p0_rows(decoder_dir / "p0_table.json")
    if not (decoder_dir / "residual_v1.dat").is_file():
        raise SystemExit(f"{decoder}: residual store required for the frozen m coordinate")
    residual, residual_meta = read_residual_v1(decoder_dir)
    batch, compact_meta = read_compact(decoder_dir)
    if not compact_meta.image_ids or not residual_meta.image_ids:
        raise SystemExit(
            f"{decoder}: feature-store metadata lacks image_ids; positional fit is unsafe"
        )
    _require_row_store_alignment(rows, compact_meta)
    _require_row_store_alignment(rows, residual_meta)
    if len(set(compact_meta.image_ids)) != len(compact_meta.image_ids):
        raise SystemExit(f"{decoder}: feature-store metadata contains duplicate image_ids")
    if residual.shape[:2] != batch.e_tex.shape or batch.rho.shape != batch.e_tex.shape:
        raise SystemExit(f"{decoder}: residual/compact patch shapes disagree")
    if any(str(row.get("decoder", "")) != decoder for row in rows):
        raise SystemExit(f"{decoder}: p0_table contains rows stamped for another decoder")

    pools: dict[str, list[tuple[float, float]]] = {"h0": [], "h1": []}
    for i, row in enumerate(rows):
        # Nothing from TEST/null_adapt/calibration is even extracted into a feature pool.
        if row.get("split") != "train":
            continue
        if finite_float("sigma_blur", row.get("sigma_blur")) != 0.0:
            continue
        w = finite_int("w", row.get("w"))
        label = str(row.get("label", ""))
        # Same side rule as _readout_common._side_matrix: H0 = the w=1 rows (stored with label
        # "h0"), H1 = label "h1" rows at w=20.
        side = (
            "h0"
            if (label == "h0" and int(W_H0) == 1) or (label == "h1" and w == int(W_H0))
            else "h1"
            if label == "h1" and w == int(W_H1)
            else ""
        )
        if not side:
            continue
        # rho(x) of Eq. (4) is recomputed from the compact store and must match p0_table.
        rho = image_rho_summary(
            batch.rho[i], batch.e_tex[i], mask_quantile=MASK_Q, winsor_quantiles=WINSOR
        )
        stored_rho = finite_float("rho_summary", row.get("rho_summary"))
        if abs(rho - stored_rho) > RHO_REPRODUCTION_TOL:
            raise SystemExit(
                f"{decoder}: recomputed FIT rho differs from p0_table by "
                f"{abs(rho - stored_rho):.4g} (> {RHO_REPRODUCTION_TOL:g})"
            )
        m = m_masked_winsor(np.linalg.norm(residual[i], axis=1), batch.e_tex[i])
        pools[side].append((rho, m))
    return (
        np.asarray(pools["h0"], dtype=np.float64).reshape(-1, 2),
        np.asarray(pools["h1"], dtype=np.float64).reshape(-1, 2),
    )


def _deployed_stats(decoder: str, f0: np.ndarray, f1: np.ndarray) -> dict:
    """Deployed Pair-LDA head on FIT pools: standardization, LW weights, FIT-H0 scores, S_eff."""
    if f0.shape[0] < 8 or f1.shape[0] < 8:
        raise SystemExit(f"{decoder}: FIT pools too thin ({f0.shape[0]}/{f1.shape[0]})")
    if not (np.all(np.isfinite(f0)) and np.all(np.isfinite(f1))):
        raise SystemExit(f"{decoder}: FIT feature pools contain NaN/Inf")
    m0 = f0.mean(axis=0)
    s0 = f0.std(axis=0, ddof=1)
    if not (np.all(np.isfinite(s0)) and np.all(s0 > 0.0)):
        raise SystemExit(f"{decoder}: FIT-H0 coordinate scale is zero or non-finite")
    z0 = (f0 - m0) / s0
    shift = ((f1 - m0) / s0).mean(axis=0)
    cov, _ = ledoit_wolf_cov(z0)
    try:
        wvec = np.linalg.solve(cov, shift)
    except np.linalg.LinAlgError as exc:
        raise SystemExit(f"{decoder}: Ledoit-Wolf covariance solve failed: {exc}") from exc
    if not np.all(np.isfinite(wvec)):
        raise SystemExit(f"{decoder}: deployed Fisher weights contain NaN/Inf")
    # S_eff = (mean FIT-H1 score - mean FIT-H0 score) / sd(FIT-H0 score).
    sc0 = z0 @ wvec
    sc1 = ((f1 - m0) / s0) @ wvec
    score_scale = float(sc0.std(ddof=1))
    if not np.isfinite(score_scale) or score_scale <= 0.0:
        raise SystemExit(f"{decoder}: FIT-H0 deployed-score scale is zero or non-finite")
    s_eff = finite_float("s_eff_fit", (sc1.mean() - sc0.mean()) / score_scale, positive=True)
    return {"m0": m0, "s0": s0, "wvec": wvec, "sc0": sc0, "s_eff": s_eff}


def stage_predict(decoders: list[str], root: Path, n_prompts: int, n_shards: int) -> int:
    if not decoders:
        raise SystemExit("predict stage requires at least one decoder")
    if len(decoders) != len(set(decoders)):
        raise SystemExit("predict stage decoder list contains duplicates")
    finite_int("--n-prompts", n_prompts, positive=True)
    finite_int("--n-shards", n_shards, positive=True)
    for decoder in decoders:
        _refuse_existing_fresh_artifacts(decoder)
    bindings = {
        decoder: _prospective_cal_binding(decoder, n_prompts=n_prompts, n_shards=n_shards)
        for decoder in decoders
    }
    if any(binding is not None for binding in bindings.values()) and not all(
        binding is not None for binding in bindings.values()
    ):
        raise SystemExit(
            "prospective calibration freezes are only present for a subset of decoders"
        )
    frozen_by_decoder: dict[str, dict] = {}
    for decoder in decoders:
        f0, f1 = _fit_pools(decoder, root / decoder)
        stats = _deployed_stats(decoder, f0, f1)
        binding = bindings[decoder]
        if binding is not None:
            _digest, cal_weights = binding
            for name in ("m0", "s0", "wvec"):
                deviation = float(np.max(np.abs(stats[name] - cal_weights[name])))
                if deviation > PROSPECTIVE_HEAD_TOL:
                    raise SystemExit(
                        f"{decoder}: predict-stage lda head '{name}' differs from the "
                        f"prospective calibration freeze by {deviation:.3g} "
                        f"(> {PROSPECTIVE_HEAD_TOL:g})"
                    )
        s_eff = stats["s_eff"]
        frozen = {
            "schema_version": FROZEN_SCHEMA_VERSION,
            "decoder": decoder,
            "contrast": {"h0_w": W_H0, "h1_w": W_H1},
            "sigma_blur": 0,
            "channel": dict(EXPECTED_CHANNEL),
            "weights": {
                "m0": stats["m0"].tolist(),
                "s0": stats["s0"].tolist(),
                "wvec": stats["wvec"].tolist(),
            },
            # FIT-null thresholds: (1 - alpha) quantiles of the FIT-H0 deployed scores.
            "thresholds": {
                str(alpha): float(np.quantile(stats["sc0"], 1.0 - alpha)) for alpha in ALPHAS
            },
            "s_eff_fit": s_eff,
            "predicted_auc_k": {
                str(k): float(norm.cdf(np.sqrt(k) * s_eff / np.sqrt(2.0))) for k in GROUP_K
            },
            "plan": {
                "prompt_offset": PROMPT_OFFSET,
                "seed_base": SEED_BASE,
                "n_prompts": n_prompts,
                "n_shards": n_shards,
                "steps": EXPECTED_CHANNEL["steps"],
                "w_list": [W_H0, W_H1],
            },
            "fit_counts": {"h0": int(f0.shape[0]), "h1": int(f1.shape[0])},
        }
        if binding is not None:
            frozen["prospective_cal_sha256"] = binding[0]
        # Hash everything above; rows generated later carry this id (see validate_run_row).
        frozen["frozen_id"] = frozen_id(frozen)
        frozen_by_decoder[decoder] = frozen

    # Finish every FIT/binding check before writing any decoder plan. This avoids a partial
    # preregistration when a later decoder fails validation.
    for decoder, frozen in frozen_by_decoder.items():
        out = frozen_path(decoder)
        out.parent.mkdir(parents=True, exist_ok=True)
        write_json(out, frozen)
        print(
            f"[predict] {decoder}: S_eff(FIT)={frozen['s_eff_fit']:.3f}  predicted AUC_16="
            f"{frozen['predicted_auc_k']['16']:.3f}  -> {out}"
        )
    return 0
