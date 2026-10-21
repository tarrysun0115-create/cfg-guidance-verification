"""Frozen-plan, feature-representation, and checkpoint contracts for fresh_batch_validation.

Helper module (no CLI) shared by the preregistered fresh-batch scripts of Section IV-D:
fresh_batch_validation.py, _fresh_batch_predict.py, fresh_tail_compare.py,
freeze_prospective_cal.py and eval_prospective_cal.py. It holds:

  * the deployed channel constants every fresh row must be computed with (4x4 latent patches,
    denominator floor 1e-3, log ratio, E_tex mask quantile 0.2, winsorization [0.01, 0.99],
    capture fraction 0.5, 50 steps, 512^2), recorded as ``channel`` in every frozen artifact;
  * the m coordinate of Eq. (8) on the stored fp16-scaled residual representation, and the
    8-bit PNG round trip that main-grid images also went through before inversion;
  * the SHA-256 bindings: ``frozen_id`` hashes a frozen prediction plan (frozen_<dec>.json) and
    ``frozen_cal_sha256`` hashes a prospective-calibration freeze (frozen_cal_<dec>.json), both
    over canonical JSON (sorted keys, compact separators, finite numbers only) that excludes the
    digest field itself. Every generated row stores ``frozen_id``, so rows produced under another
    freeze are rejected;
  * strict JSONL row I/O with crash-safe resume, per-row validation, and the exact one-sided
    Clopper--Pearson upper bound used for realized FPR.

Prompt/seed plan: fresh prompts are ``_placeholder_prompts(offset + n)[offset:]``, and prompt i
uses seed ``seed_base + i`` for both its w=1 and its w=20 image. The defaults (offset 100000,
seed base 1000000) define the first fresh batch; the prospective-calibration batch overrides them
via CFGID_FRESH_DIR / CFGID_PROMPT_OFFSET / CFGID_SEED_BASE (outputs/fresh_cal_batch, 110000,
3000000; see README.md). Prompt ranges are disjoint from the main grid and from each other,
and each batch has its own seed base (Appendix B).
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
from scipy.stats import beta, norm

from cfgid.channel.rho import etex_keep_mask, winsorize
from cfgid.featurestore import dequantize_fp16_scaled, quantize_fp16_scaled
from cfgid.numerics import as_finite_float32, finite_float, loads_strict_json
from cfgid.paths import outputs_dir

PATCH = 4
FLOOR = 1e-3
LOG_RATIO = True
MASK_Q = 0.2
WINSOR = (0.01, 0.99)
CAPTURE_FRAC = 0.5
E_TEX_METHOD = "srm_highpass"
HEIGHT = WIDTH = 512
GROUP_K = [1, 2, 4, 8, 16]
ALPHAS = (0.05, 0.01)
# Env overrides let a SECOND preregistered batch (e.g. the prospective-calibration batch)
# reuse this pipeline without touching the first batch's artifacts. Keep the same env
# exported for EVERY stage (predict/run/summarize/eval) of that batch; the plan validator
# below enforces consistency and will abort on a mismatch.
PROMPT_OFFSET = int(os.environ.get("CFGID_PROMPT_OFFSET", "100000"))
SEED_BASE = int(os.environ.get("CFGID_SEED_BASE", "1000000"))
if any(
    os.environ.get(name) for name in ("CFGID_FRESH_DIR", "CFGID_PROMPT_OFFSET", "CFGID_SEED_BASE")
):
    print(
        f"[fresh-support] ENV OVERRIDE ACTIVE: fresh_dir="
        f"{os.environ.get('CFGID_FRESH_DIR', '<default>')} prompt_offset={PROMPT_OFFSET} "
        f"seed_base={SEED_BASE}"
    )
W_H0, W_H1 = 1.0, 20.0
FROZEN_SCHEMA_VERSION = 1
RHO_REPRODUCTION_TOL = 1e-3
FROZEN_CAL_SHA_KEY = "freeze_sha256"

EXPECTED_CHANNEL = {
    "patch": PATCH,
    "floor": FLOOR,
    "log_ratio": LOG_RATIO,
    "mask_q": MASK_Q,
    "winsor": list(WINSOR),
    "capture_frac": CAPTURE_FRAC,
    "e_tex_method": E_TEX_METHOD,
    "steps": 50,
    "height": HEIGHT,
    "width": WIDTH,
    "observation": "png_u8_roundtrip",
    "m_residual": "residual_feature_v1_fp16_scaled",
}


def m_masked_winsor(delta: np.ndarray, e_tex: np.ndarray) -> float:
    """m(x) of Eq. (8): mean of winsorized per-patch branch-difference norms over the E_tex mask."""
    delta = np.asarray(delta)
    e_tex = np.asarray(e_tex)
    if delta.ndim != 1 or e_tex.ndim != 1 or delta.shape != e_tex.shape:
        raise ValueError(
            f"delta/e_tex must be matching non-empty vectors; got {delta.shape}/{e_tex.shape}"
        )
    if delta.size == 0:
        raise ValueError("m coordinate received no patches")
    # M(x): keep patches at or above the image's MASK_Q texture-energy quantile (Eq. (3)).
    keep = etex_keep_mask(e_tex, MASK_Q)
    kept = delta[keep] if keep.any() else delta
    value = float(np.mean(winsorize(kept.astype(np.float32), WINSOR[0], WINSOR[1])))
    return finite_float("m", value)


def png_roundtrip(image: np.ndarray) -> np.ndarray:
    """Mirror run_p0._save_png without retaining the PNG."""
    arr = as_finite_float32("generated_image", image)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f"generated image must be HWC RGB; got {arr.shape}")
    if float(arr.min()) < 0.0 or float(arr.max()) > 1.0:
        raise ValueError("generated image must be in [0, 1]")
    u8 = np.clip(np.rint(arr * 255.0), 0.0, 255.0).astype(np.uint8)
    return u8.astype(np.float32) / 255.0


def stored_residual_roundtrip(residual: np.ndarray) -> np.ndarray:
    """Apply the residual_feature_v1 fp16-scaled representation used to fit m."""
    q, scale = quantize_fp16_scaled(residual)
    return dequantize_fp16_scaled(q, scale)


def cp_upper(successes: int, n: int, conf: float = 0.95) -> float:
    """Exact one-sided Clopper--Pearson upper bound (CP95 by default) on a binomial rate."""
    if n < 1 or not 0 <= successes <= n:
        raise ValueError(f"Clopper-Pearson requires 0 <= successes <= n; got {successes}/{n}")
    if not 0.0 < conf < 1.0:
        raise ValueError(f"confidence must be in (0,1), got {conf}")
    if successes == n:
        return 1.0
    value = beta.ppf(conf, successes + 1, n - successes)
    return finite_float("Clopper-Pearson upper bound", value)


def fresh_dir() -> Path:
    """Fresh-batch artifact dir; CFGID_FRESH_DIR points a second batch elsewhere."""
    override = os.environ.get("CFGID_FRESH_DIR")
    return Path(override) if override else outputs_dir() / "fresh_batch"


def frozen_path(decoder: str) -> Path:
    return fresh_dir() / f"frozen_{decoder}.json"


def rows_path(decoder: str, shard: int, n_shards: int) -> Path:
    suffix = f"_shard{shard}of{n_shards}" if n_shards > 1 else ""
    return fresh_dir() / f"{decoder}_rows{suffix}.jsonl"


def finite_int(name: str, value: object, *, positive: bool = False) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be an integer")
    number = finite_float(name, value)
    if not number.is_integer() or (positive and number < 1):
        want = "a positive integer" if positive else "an integer"
        raise ValueError(f"{name} must be {want}, got {value!r}")
    return int(number)


def frozen_id(payload: dict) -> str:
    """SHA-256 of the canonical JSON of a frozen plan, excluding the frozen_id field itself."""
    body = {key: value for key, value in payload.items() if key != "frozen_id"}
    encoded = json.dumps(
        body, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def frozen_cal_sha256(payload: dict) -> str:
    """Hash canonical finite JSON, excluding the self-referential digest field."""
    body = {key: value for key, value in payload.items() if key != FROZEN_CAL_SHA_KEY}
    encoded = json.dumps(
        body, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def write_json(path: Path, payload: dict) -> None:
    text = json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def finite_vector(name: str, value: object, shape: tuple[int, ...]) -> np.ndarray:
    try:
        arr = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if arr.shape != shape or not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must be finite with shape {shape}; got {arr.shape}")
    return arr


def load_frozen_path(path: Path, decoder: str) -> dict:
    """Load and fully validate one frozen artifact from an explicit path."""
    if not path.is_file():
        raise SystemExit(f"missing frozen predictions: {path} (run --stage predict first)")
    raw = loads_strict_json(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: frozen artifact must be a JSON object")
    frozen = dict(raw)
    if frozen.get("schema_version") != FROZEN_SCHEMA_VERSION:
        raise ValueError(f"{path}: unsupported frozen schema {frozen.get('schema_version')!r}")
    if frozen.get("decoder") != decoder:
        raise ValueError(f"{path}: decoder mismatch ({frozen.get('decoder')!r} != {decoder!r})")
    artifact_id = frozen.get("frozen_id")
    if not isinstance(artifact_id, str) or artifact_id != frozen_id(frozen):
        raise ValueError(f"{path}: frozen_id is missing or does not match the frozen contents")
    prospective_cal_sha256 = frozen.get("prospective_cal_sha256")
    if prospective_cal_sha256 is not None and (
        not isinstance(prospective_cal_sha256, str)
        or len(prospective_cal_sha256) != 64
        or any(c not in "0123456789abcdef" for c in prospective_cal_sha256)
    ):
        raise ValueError(f"{path}: prospective_cal_sha256 must be a lowercase sha256 digest")
    if frozen.get("channel") != EXPECTED_CHANNEL:
        raise ValueError(f"{path}: frozen channel does not match the deployed channel contract")
    if frozen.get("contrast") != {"h0_w": W_H0, "h1_w": W_H1}:
        raise ValueError(f"{path}: frozen contrast is not w={W_H1:g} versus w={W_H0:g}")
    if frozen.get("sigma_blur") != 0:
        raise ValueError(f"{path}: frozen sigma_blur must be zero")

    weights = frozen.get("weights")
    if not isinstance(weights, dict):
        raise ValueError(f"{path}: weights must be an object")
    finite_vector("weights.m0", weights.get("m0"), (2,))
    s0 = finite_vector("weights.s0", weights.get("s0"), (2,))
    if np.any(s0 <= 0.0):
        raise ValueError(f"{path}: weights.s0 must be strictly positive")
    finite_vector("weights.wvec", weights.get("wvec"), (2,))

    s_eff = finite_float("s_eff_fit", frozen.get("s_eff_fit"), positive=True)
    thresholds = frozen.get("thresholds")
    predicted = frozen.get("predicted_auc_k")
    if not isinstance(thresholds, dict) or set(thresholds) != {str(a) for a in ALPHAS}:
        raise ValueError(f"{path}: frozen threshold keys do not match {ALPHAS}")
    for alpha in ALPHAS:
        finite_float(f"threshold[{alpha}]", thresholds[str(alpha)])
    if not isinstance(predicted, dict) or set(predicted) != {str(k) for k in GROUP_K}:
        raise ValueError(f"{path}: predicted AUC keys do not match {GROUP_K}")
    # The frozen predictions must equal the Gaussian aggregation law
    # AUC_K = Phi(sqrt(K) * S_eff / sqrt(2)) of Eqs. (12) and (14) at the frozen S_eff.
    for k in GROUP_K:
        got = finite_float(f"predicted_auc_k[{k}]", predicted[str(k)])
        expected = float(norm.cdf(np.sqrt(k) * s_eff / np.sqrt(2.0)))
        if not (0.0 <= got <= 1.0 and np.isclose(got, expected, rtol=1e-12, atol=1e-12)):
            raise ValueError(f"{path}: predicted AUC for K={k} is inconsistent with s_eff_fit")

    plan = frozen.get("plan")
    if not isinstance(plan, dict):
        raise ValueError(f"{path}: plan must be an object")
    finite_int("plan.prompt_offset", plan.get("prompt_offset"))
    finite_int("plan.seed_base", plan.get("seed_base"))
    finite_int("plan.n_prompts", plan.get("n_prompts"), positive=True)
    finite_int("plan.n_shards", plan.get("n_shards"), positive=True)
    finite_int("plan.steps", plan.get("steps"), positive=True)
    if plan.get("prompt_offset") != PROMPT_OFFSET or plan.get("seed_base") != SEED_BASE:
        raise ValueError(f"{path}: frozen prompt/seed offsets do not match this script")
    if plan.get("w_list") != [W_H0, W_H1]:
        raise ValueError(f"{path}: frozen guidance list does not match the deployed contrast")
    if plan.get("steps") != EXPECTED_CHANNEL["steps"]:
        raise ValueError(f"{path}: frozen plan steps do not match the deployed channel")
    fit_counts = frozen.get("fit_counts")
    if not isinstance(fit_counts, dict) or set(fit_counts) != {"h0", "h1"}:
        raise ValueError(f"{path}: fit_counts must contain exactly h0/h1")
    if (
        min(finite_int(f"fit_counts.{key}", fit_counts[key], positive=True) for key in fit_counts)
        < 8
    ):
        raise ValueError(f"{path}: frozen FIT pools are too thin")
    return frozen


def load_frozen(decoder: str) -> dict:
    return load_frozen_path(frozen_path(decoder), decoder)


def _parse_object(text: str, context: str) -> dict:
    raw = loads_strict_json(text)
    if not isinstance(raw, dict):
        raise ValueError(f"{context}: expected a JSON object")
    return dict(raw)


def read_jsonl(path: Path, *, recover_trailing_partial: bool) -> list[dict]:
    """Read strict JSONL; only a syntactically truncated, unterminated tail is recoverable."""
    if not path.is_file():
        return []
    data = path.read_bytes()
    if not data:
        return []
    if not data.endswith(b"\n"):
        tail_start = data.rfind(b"\n") + 1
        try:
            _parse_object(data[tail_start:].decode("utf-8"), f"{path}: trailing line")
        except json.JSONDecodeError as exc:
            if not recover_trailing_partial:
                raise ValueError(
                    f"{path}: truncated trailing record; resume its run shard"
                ) from exc
            with path.open("r+b") as fh:
                fh.truncate(tail_start)
                fh.flush()
                os.fsync(fh.fileno())
            print(f"[run] recovered truncated trailing record in {path}")
            data = data[:tail_start]
        except UnicodeDecodeError as exc:
            raise ValueError(f"{path}: JSONL is not valid UTF-8") from exc
        else:
            if recover_trailing_partial:
                with path.open("ab") as fh:
                    fh.write(b"\n")
                    fh.flush()
                    os.fsync(fh.fileno())
                data += b"\n"
    rows: list[dict] = []
    for line_no, raw_line in enumerate(data.splitlines(), start=1):
        if not raw_line.strip():
            raise ValueError(f"{path}:{line_no}: blank JSONL record")
        try:
            rows.append(_parse_object(raw_line.decode("utf-8"), f"{path}:{line_no}"))
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise ValueError(f"{path}:{line_no}: malformed JSONL record") from exc
    return rows


def validate_run_row(row: dict, decoder: str, frozen: dict, shard: int, n_shards: int) -> str:
    """Check one generated row against the frozen plan and return its image_id.

    Requires the row's frozen_id, prompt range and shard (local index modulo n_shards), guidance
    level, seed (seed_base + local prompt index) and image_id to match the plan.
    """
    plan = frozen["plan"]
    if row.get("frozen_id") != frozen["frozen_id"]:
        raise ValueError("row was produced with a different frozen artifact")
    image_id = row.get("image_id")
    if not isinstance(image_id, str) or not image_id:
        raise ValueError("row image_id must be a non-empty string")
    prompt_idx = finite_int("row.prompt_idx", row.get("prompt_idx"))
    local_idx = prompt_idx - int(plan["prompt_offset"])
    if not 0 <= local_idx < int(plan["n_prompts"]):
        raise ValueError(f"row {image_id}: prompt_idx is outside the frozen prompt range")
    if local_idx % n_shards != shard:
        raise ValueError(f"row {image_id}: prompt belongs to the wrong shard")
    w = finite_float("row.w", row.get("w"))
    if w not in plan["w_list"]:
        raise ValueError(f"row {image_id}: w={w} is outside the frozen guidance list")
    seed = finite_int("row.seed", row.get("seed"))
    if seed != int(plan["seed_base"]) + local_idx:
        raise ValueError(f"row {image_id}: seed does not match its global prompt index")
    expected_id = f"{decoder}_fp{prompt_idx}_w{int(w)}"
    if image_id != expected_id:
        raise ValueError(f"row image_id {image_id!r} != expected {expected_id!r}")
    finite_float("row.rho", row.get("rho"))
    finite_float("row.m", row.get("m"))
    return image_id
