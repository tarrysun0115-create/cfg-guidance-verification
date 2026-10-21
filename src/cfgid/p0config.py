"""Typed loader for ``configs/p0.yaml`` — the PREREGISTERED main-grid (p0) protocol.

``p0`` names the preregistered main-grid protocol throughout the repo (``configs/p0.yaml``,
``outputs/p0/``, ``p0_table.json``). The file fixes the protocol constants of Sections III-B/C,
IV-A and Appendix B: the guidance grid w (w=1 is H0), the blur levels sigma_blur, T=50 inversion
steps with capture at 0.5T, the latent patch size, the E_tex mask quantile q, the winsorization
level kappa, the denominator floor, and the prompt-disjoint FIT/NULL/CAL/TEST splits (40/20/20/20,
seed 0), plus the Monte-Carlo TV settings and the preregistered gate thresholds.

The gate reads only this (frozen) config; downstream sweep / align / bounds / gate consume the
typed sections. ``qf`` entries normalize ``"none"`` -> ``None`` (uncompressed).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from omegaconf import OmegaConf
from omegaconf.errors import OmegaConfBaseException

from cfgid import paths
from cfgid.bounds.wstar_p0 import G_SIGMA_PROVENANCE, V_PROVENANCE

# The four canonical leakage splits train / null_adapt / calibration / test (the paper's FIT /
# NULL / CAL / TEST); split.fractions must name exactly these (sum ~= 1, all >= 0).
_CANONICAL_SPLITS = frozenset({"train", "null_adapt", "calibration", "test"})


@dataclass(frozen=True)
class PostprocOp:
    """One post-processing degradation applied to the decoded image just before inversion.

    ``kind`` selects the operator and ``param`` is its single scalar argument: JPEG quality 1-100
    for ``'jpeg'``, the downscale fraction in (0, 1] for ``'resize'``, the center-crop fraction in
    (0, 1] for ``'crop'``; ignored for ``'none'``. ``name`` is the row/cell tag the robustness
    analysis groups on. This is an OFF-gate axis used only by the (separate) robustness config; the
    preregistered p0.yaml leaves it absent, which normalizes to the single ``'none'`` op.
    """

    name: str
    kind: str  # 'none' | 'jpeg' | 'resize' | 'crop'
    param: float


@dataclass(frozen=True)
class GridConfig:
    decoders: tuple[str, ...]
    w: tuple[int, ...]
    qf: tuple[int | None, ...]  # None == uncompressed
    prompts_per_cell: int
    seeds_per_prompt: int
    gen_scheduler: str
    resolution: int  # generation/inversion image side (preregistered, shared by all decoders)
    sigma_blur: tuple[float, ...]  # controlled sigma_blur sweep: post-decode Gaussian widths (px)
    # OFF-gate one-at-a-time degradation axis (robustness config only); absent -> single 'none' op.
    postproc: tuple[PostprocOp, ...] = (PostprocOp("none", "none", 0.0),)


@dataclass(frozen=True)
class PatchConfig:
    patch_size: int
    inversion_steps: int
    capture_step_frac: float


@dataclass(frozen=True)
class RhoConfig:
    log_ratio: bool
    denom_floor: float
    low_etex_mask_quantile: float
    winsor_quantiles: tuple[float, float]


@dataclass(frozen=True)
class ETexConfig:
    method: str


@dataclass(frozen=True)
class BoundsConfig:
    delta: float
    n_eff_grid: tuple[int, ...]
    montecarlo_samples: int
    montecarlo_tol: float


@dataclass(frozen=True)
class ProvenanceConfig:
    v: str
    g_sigma: str


@dataclass(frozen=True)
class SplitConfig:
    group_key: str
    fractions: dict[str, float]
    seed: int


@dataclass(frozen=True)
class SigmaCalibConfig:
    sources: tuple[str, ...]
    min_r2: float
    max_rel_ci_width: float
    max_source_rel_gap: float


@dataclass(frozen=True)
class EvidenceConfig:
    """Evidence-wiring knobs (negative controls / sufficiency / sigma calibration)."""

    clip_baseline_enabled: bool
    clip_local_dir: str
    wrong_prompt_subset_qf: int | None
    sigma_calib: SigmaCalibConfig


@dataclass(frozen=True)
class GateConfig:
    predictive_only_pass: bool
    min_knee_wstar_slope: float
    slope_ci_level: float
    min_correlation: float
    negative_control_margin: float
    rho_sufficiency_max_auc_gap: float
    sigma_two_decoder_same_direction: bool
    bootstrap_resamples: int


@dataclass(frozen=True)
class P0Config:
    grid: GridConfig
    patch: PatchConfig
    rho: RhoConfig
    e_tex: ETexConfig
    bounds: BoundsConfig
    provenance: ProvenanceConfig
    split: SplitConfig
    evidence: EvidenceConfig
    gate: GateConfig
    raw: dict[str, Any] = field(default_factory=dict)


def _norm_qf(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, str) and value.lower() == "none":
        return None
    return int(value)


def _qf_levels(value: Any) -> tuple[int | None, ...]:
    """JPEG quality grid after ``none`` normalization; entries must be unique image-id keys."""
    qfs = tuple(_norm_qf(x) for x in value)
    if not qfs:
        raise ValueError("p0.yaml: grid.qf must not be empty")
    if len(set(qfs)) != len(qfs):
        dups = sorted(("none" if qf is None else str(qf)) for qf in set(qfs) if qfs.count(qf) > 1)
        raise ValueError(f"p0.yaml: grid.qf must have unique normalized values; duplicates {dups}")
    for qf in qfs:
        if qf is not None and not (1 <= qf <= 100):
            raise ValueError(f"p0.yaml: grid.qf entries must be 'none' or in [1, 100], got {qf}")
    return qfs


def _require(node: dict[str, Any], key: str) -> Any:
    if key not in node:
        raise KeyError(f"p0.yaml missing required section/key '{key}'")
    return node[key]


def _positive_multiple_of_8(value: Any) -> int:
    out = int(value)
    if out <= 0 or out % 8 != 0:
        raise ValueError(f"p0.yaml: grid.resolution must be a positive multiple of 8, got {out}")
    return out


def _w_levels(value: Any) -> tuple[int, ...]:
    """Guidance-strength grid: each w enters the image_id (``_w{w}_``) as a uniqueness key.

    sigma_blur is dedup-guarded by ``_sigma_blur_levels``; w must be too. A duplicated entry (e.g.
    [1, 2, 2]) would emit two H1 generations with the SAME image_id -> the second overwrites the
    first PNG/row, while ``_expected_result_count`` over-counts by the same amount so counts stay
    self-consistent and the duplicate cell slips SILENTLY into the regression. Reject dups LOUDLY.
    """
    ws = [int(x) for x in value]
    if not ws:
        raise ValueError("p0.yaml: grid.w must not be empty")
    if len(set(ws)) != len(ws):
        dups = sorted({w for w in ws if ws.count(w) > 1})
        raise ValueError(f"p0.yaml: grid.w must have unique values; duplicates {dups}")
    if 1 not in ws:
        raise ValueError("p0.yaml: grid.w must contain 1 (the H0 anchor)")
    bad = [w for w in ws if w < 1]
    if bad:
        raise ValueError(f"p0.yaml: grid.w entries must be >= 1, got {bad}")
    return tuple(ws)


def _sigma_blur_levels(value: Any) -> tuple[float, ...]:
    """Controlled sigma_blur sweep levels: finite, non-negative, unique, sorted, including 0.0.

    0.0 (the bare decoder PSF) is the sweep baseline and must always be present, so it is added if
    omitted. The levels are the post-decode Gaussian PSF widths (decoded pixels) run_p0 applies.

    Each level is also required to be ``%g``-round-trip-stable (``float(f"{s:g}") == s``): the level
    is used as a ``%g`` suffix in the image_id and as a ``%g`` JSON key in the sigma-calibration
    ``per_sigma_blur`` map, then looked up back by exact float. A level that loses precision under
    ``%g`` (e.g. 0.1+0.2) would later be silently dropped; we reject it LOUDLY at config load
    instead. (Stability also guarantees distinct ``%g`` keys: two distinct stable floats cannot
    share a key, since ``float(key)`` recovers each exactly.) Integer/few-decimal levels pass.
    """
    levels = sorted({0.0, *(float(x) for x in value)})
    for s in levels:
        if not (math.isfinite(s) and s >= 0.0):
            raise ValueError(f"p0.yaml: grid.sigma_blur must be finite and >= 0, got {s}")
        if float(f"{s:g}") != s:
            raise ValueError(
                f"p0.yaml: grid.sigma_blur level {s!r} is not %g-stable (round-trips to "
                f"{float(f'{s:g}')!r}); use a value with <= 6 significant figures."
            )
    return tuple(levels)


_POSTPROC_KINDS = frozenset({"none", "jpeg", "resize", "crop"})


def _postproc_ops(value: Any) -> tuple[PostprocOp, ...]:
    """Parse grid.postproc into PostprocOp tuple; absent/empty -> the single 'none' op.

    Each entry is a mapping ``{name, kind, param}``. 'none' needs no param; 'jpeg' needs an int
    quality in [1, 100]; 'resize'/'crop' need a fraction in (0, 1]. Names are unique (each tags a
    row). Kept OFF the preregistered p0.yaml gate: absent normalizes to a no-op so existing configs
    and the frozen gate are unaffected.
    """
    if value is None:
        return (PostprocOp("none", "none", 0.0),)
    if not isinstance(value, (list, tuple)):
        raise ValueError("p0.yaml: grid.postproc must be a list of {name, kind, param} mappings")
    ops: list[PostprocOp] = []
    for entry in value:
        if not isinstance(entry, dict):
            raise ValueError(f"p0.yaml: grid.postproc entry must be a mapping, got {entry!r}")
        name = str(entry["name"])
        kind = str(entry["kind"])
        if kind not in _POSTPROC_KINDS:
            raise ValueError(
                f"p0.yaml: grid.postproc '{name}' kind '{kind}' not in {sorted(_POSTPROC_KINDS)}"
            )
        param = float(entry.get("param", 0.0))
        if not math.isfinite(param):
            raise ValueError(f"p0.yaml: grid.postproc '{name}' param must be finite, got {param}")
        if kind == "jpeg" and not (param.is_integer() and 1 <= param <= 100):
            raise ValueError(
                f"p0.yaml: grid.postproc '{name}' jpeg param must be int in [1, 100], got {param}"
            )
        if kind in ("resize", "crop") and not (0.0 < param <= 1.0):
            raise ValueError(
                f"p0.yaml: grid.postproc '{name}' {kind} param must be in (0, 1], got {param}"
            )
        ops.append(PostprocOp(name=name, kind=kind, param=param))
    if not ops:
        return (PostprocOp("none", "none", 0.0),)
    names = [op.name for op in ops]
    if len(set(names)) != len(names):
        raise ValueError(f"p0.yaml: grid.postproc entry names must be unique, got {names}")
    return tuple(ops)


def _validate_provenance(provenance: dict[str, Any]) -> None:
    """Fail loud at the config boundary if a provenance track is not in the allowlist."""
    v = str(provenance["v"])
    if v not in V_PROVENANCE:
        raise ValueError(f"p0.yaml: provenance.v '{v}' not in allowed {sorted(V_PROVENANCE)}")
    g_sigma = str(provenance["g_sigma"])
    if g_sigma not in G_SIGMA_PROVENANCE:
        raise ValueError(
            f"p0.yaml: provenance.g_sigma '{g_sigma}' not in allowed {sorted(G_SIGMA_PROVENANCE)}"
        )


def _validate_split_fractions(fractions: dict[str, float]) -> None:
    """Fail loud at the config boundary if the leakage-split fractions are incoherent.

    Requires: the canonical split names present, every fraction non-negative, sum ~= 1 (abs tol
    1e-6). A drifted/typo'd split map would silently mis-partition the leakage-disjoint data.
    """
    names = set(fractions)
    if names != _CANONICAL_SPLITS:
        missing = sorted(_CANONICAL_SPLITS - names)
        extra = sorted(names - _CANONICAL_SPLITS)
        raise ValueError(
            f"p0.yaml: split.fractions must name exactly {sorted(_CANONICAL_SPLITS)}; "
            f"missing {missing}, unexpected {extra}"
        )
    negative = sorted(k for k, frac in fractions.items() if frac < 0.0)
    if negative:
        raise ValueError(f"p0.yaml: split.fractions must be non-negative; got negative {negative}")
    total = math.fsum(fractions.values())
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"p0.yaml: split.fractions must sum to 1.0 (abs tol 1e-6), got {total}")


def load_p0_config(path: Path | None = None) -> P0Config:
    cfg_path = Path(path) if path is not None else paths.configs_dir() / "p0.yaml"
    if not cfg_path.is_file():
        raise FileNotFoundError(f"p0.yaml not found at {cfg_path}")
    try:
        container = OmegaConf.to_container(OmegaConf.load(cfg_path), resolve=True)
    except (OmegaConfBaseException, yaml.YAMLError) as exc:
        raise ValueError(f"p0.yaml malformed: {exc}") from exc
    if not isinstance(container, dict):
        raise ValueError("p0.yaml malformed: expected a top-level mapping")

    data: dict[str, Any] = {str(k): v for k, v in container.items()}
    grid = _require(data, "grid")
    patch = _require(data, "patch")
    rho = _require(data, "rho")
    e_tex = _require(data, "e_tex")
    bounds = _require(data, "bounds")
    provenance = _require(data, "provenance")
    split = _require(data, "split")
    evidence = _require(data, "evidence")
    gate = _require(data, "gate")
    sigma_calib = _require(evidence, "sigma_calib")

    wq = tuple(float(x) for x in rho["winsor_quantiles"])
    if len(wq) != 2:
        raise ValueError("p0.yaml: rho.winsor_quantiles must have exactly 2 entries")

    _validate_provenance(provenance)
    split_fractions = {str(k): float(v) for k, v in dict(split["fractions"]).items()}
    _validate_split_fractions(split_fractions)

    return P0Config(
        grid=GridConfig(
            decoders=tuple(str(x) for x in grid["decoders"]),
            w=_w_levels(grid["w"]),
            qf=_qf_levels(grid["qf"]),
            prompts_per_cell=int(grid["prompts_per_cell"]),
            seeds_per_prompt=int(grid["seeds_per_prompt"]),
            gen_scheduler=str(grid["gen_scheduler"]),
            resolution=_positive_multiple_of_8(grid["resolution"]),
            sigma_blur=_sigma_blur_levels(grid["sigma_blur"]),
            postproc=_postproc_ops(grid.get("postproc")),
        ),
        patch=PatchConfig(
            patch_size=int(patch["patch_size"]),
            inversion_steps=int(patch["inversion_steps"]),
            capture_step_frac=float(patch["capture_step_frac"]),
        ),
        rho=RhoConfig(
            log_ratio=bool(rho["log_ratio"]),
            denom_floor=float(rho["denom_floor"]),
            low_etex_mask_quantile=float(rho["low_etex_mask_quantile"]),
            winsor_quantiles=(wq[0], wq[1]),
        ),
        e_tex=ETexConfig(method=str(e_tex["method"])),
        bounds=BoundsConfig(
            delta=float(bounds["delta"]),
            n_eff_grid=tuple(int(x) for x in bounds["n_eff_grid"]),
            montecarlo_samples=int(bounds["montecarlo_samples"]),
            montecarlo_tol=float(bounds["montecarlo_tol"]),
        ),
        provenance=ProvenanceConfig(
            v=str(provenance["v"]),
            g_sigma=str(provenance["g_sigma"]),
        ),
        split=SplitConfig(
            group_key=str(split["group_key"]),
            fractions=split_fractions,
            seed=int(split["seed"]),
        ),
        evidence=EvidenceConfig(
            clip_baseline_enabled=bool(evidence["clip_baseline_enabled"]),
            clip_local_dir=str(evidence["clip_local_dir"]),
            wrong_prompt_subset_qf=_norm_qf(evidence["wrong_prompt_subset_qf"]),
            sigma_calib=SigmaCalibConfig(
                sources=tuple(str(x) for x in sigma_calib["sources"]),
                min_r2=float(sigma_calib["min_r2"]),
                max_rel_ci_width=float(sigma_calib["max_rel_ci_width"]),
                max_source_rel_gap=float(sigma_calib["max_source_rel_gap"]),
            ),
        ),
        gate=GateConfig(
            predictive_only_pass=bool(gate["predictive_only_pass"]),
            min_knee_wstar_slope=float(gate["min_knee_wstar_slope"]),
            slope_ci_level=float(gate["slope_ci_level"]),
            min_correlation=float(gate["min_correlation"]),
            negative_control_margin=float(gate["negative_control_margin"]),
            rho_sufficiency_max_auc_gap=float(gate["rho_sufficiency_max_auc_gap"]),
            sigma_two_decoder_same_direction=bool(gate["sigma_two_decoder_same_direction"]),
            bootstrap_resamples=int(gate["bootstrap_resamples"]),
        ),
        raw=data,
    )
