"""Inversion-backend contract + pure (model-free) patch math.

Design: the numpy patch math (``patchify`` / ``rho_from_patch_residuals`` / ``pack_results``)
is separated from the torch/diffusers model glue so it is unit-testable without any weights.
The model glue (loading a pipeline, running the dual-branch inversion) is lazy-imported inside
methods and raises :class:`BackendUnavailableError` when weights or diffusers are absent.

``patchify`` defines the fixed row-major latent patch grid of Section III-B (16x16 patches of
4x4 latent pixels at 512^2). ``pack_results`` turns per-image results into the feature-store
arrays (``cfgid.featurestore``). The plain ratio below is used only by the smoke helpers::

    rho = ||r_cond|| / max(||r_uncond||, floor)

where ``r_cond`` / ``r_uncond`` are the conditional / unconditional score (noise-prediction)
residuals. The paper's statistic (guarded log ratio with the p0 floor, Eq. (2)) is
``cfgid.channel.rho.rho_patch``, applied by ``cfgid.channel.features.build_inversion_result``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from cfgid.config import ModelSpec
from cfgid.featurestore.schema import FeatureMetadata
from cfgid.featurestore.store import CompactBatch

if TYPE_CHECKING:  # heavy import only for type-checkers, never at runtime
    import torch

# Denominator floor of the smoke-only ``rho_from_patch_residuals``; the main grid uses
# ``rho.denom_floor`` from configs/p0.yaml (1e-3) through ``channel.rho.rho_patch``.
RHO_DENOM_FLOOR = 1e-6


class BackendUnavailableError(RuntimeError):
    """Raised when a backend cannot load (missing weights, missing diffusers, etc.)."""


def _finite_float32(name: str, values: np.ndarray) -> np.ndarray:
    try:
        out = np.asarray(values, dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if not np.all(np.isfinite(out)):
        raise ValueError(f"{name} contains non-finite values (NaN/Inf)")
    return out


def torch_dtype(name: str) -> torch.dtype:
    """Map a models.yaml dtype string to a torch dtype (torch imported lazily)."""
    import torch

    table = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "half": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
        "float": torch.float32,
    }
    key = name.lower()
    if key not in table:
        raise ValueError(f"unknown dtype '{name}' (expected one of {sorted(table)})")
    return table[key]


@dataclass
class InversionResult:
    """Per-image dual-branch result over the latent patch grid.

    ``residual`` is the full-dimensional per-patch residual (``r_cond - r_uncond`` flattened),
    persisted as ``residual_feature_v1``; ``rho_patch`` / ``e_tex_patch`` are the compact scalars.
    """

    rho_patch: np.ndarray  # (P,) float32
    e_tex_patch: np.ndarray  # (P,) float32
    residual: np.ndarray  # (P, D) float32
    sigma: float
    patch_grid: tuple[int, int]
    qf: float = float("nan")
    image_id: str = ""
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def n_patches(self) -> int:
        return int(self.rho_patch.shape[0])

    @property
    def residual_dim(self) -> int:
        return int(self.residual.shape[1])


@dataclass
class DualBranchResidual:
    """Raw dual-branch inversion output (model glue) before config-driven feature reduction.

    ``r_cond`` / ``r_uncond`` are the (C, H, W) noise-prediction residual maps captured at
    ``capture_step`` of the two deterministic inversions (cond prompt vs empty prompt) over the same
    source ``latent``. ``channel.features.build_inversion_result`` reduces these to an
    :class:`InversionResult` using the p0 rho / E_tex knobs.
    """

    r_cond: np.ndarray  # (C, H, W) float32
    r_uncond: np.ndarray  # (C, H, W) float32
    latent: np.ndarray  # (C, H, W) float32 (inversion-source latent; used for E_tex)
    capture_step: int
    meta: dict[str, Any] = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# pure patch math (no torch) — unit-testable without weights
# --------------------------------------------------------------------------- #
def infer_patch_grid(height: int, width: int, patch: int) -> tuple[int, int]:
    if height <= 0 or width <= 0:
        raise ValueError(f"latent dimensions must be positive, got {height}x{width}")
    if patch <= 0:
        raise ValueError(f"patch size must be positive, got {patch}")
    if height % patch != 0 or width % patch != 0:
        raise ValueError(f"latent {height}x{width} not divisible by patch size {patch}")
    return height // patch, width // patch


def patchify(latent: np.ndarray, grid: tuple[int, int]) -> np.ndarray:
    """(C, H, W) latent -> (P, C*ph*pw) flattened patches, P = rows*cols, row-major."""
    x0 = _finite_float32("latent", latent)
    if x0.ndim != 3:
        raise ValueError(f"latent must be (C, H, W); got {x0.shape}")
    c, h, w = x0.shape
    rows, cols = grid
    if rows <= 0 or cols <= 0:
        raise ValueError(f"grid dimensions must be positive, got {grid}")
    if h % rows != 0 or w % cols != 0:
        raise ValueError(f"latent {h}x{w} not divisible by grid {grid}")
    ph, pw = h // rows, w // cols
    x = x0.reshape(c, rows, ph, cols, pw)
    x = x.transpose(1, 3, 0, 2, 4)  # (rows, cols, C, ph, pw)
    return x.reshape(rows * cols, c * ph * pw)


def rho_from_patch_residuals(
    r_cond: np.ndarray, r_uncond: np.ndarray, floor: float = RHO_DENOM_FLOOR
) -> np.ndarray:
    """rho = ||r_cond|| / max(||r_uncond||, floor) per patch. Inputs (P, D)."""
    if floor <= 0.0 or not np.isfinite(floor):
        raise ValueError(f"floor must be positive and finite, got {floor}")
    rc = _finite_float32("r_cond", r_cond)
    ru = _finite_float32("r_uncond", r_uncond)
    if rc.ndim != 2 or ru.ndim != 2:
        raise ValueError(f"residuals must be (P, D); got {rc.shape} and {ru.shape}")
    if rc.shape != ru.shape:
        raise ValueError(f"r_cond shape {rc.shape} != r_uncond shape {ru.shape}")
    num = np.linalg.norm(rc, axis=1)
    den = np.linalg.norm(ru, axis=1)
    return (num / np.maximum(den, floor)).astype(np.float32)


def e_tex_from_patches(patches: np.ndarray) -> np.ndarray:
    """Smoke-grade local texture energy: per-patch variance. (P, D) -> (P,).

    Used only by the smoke helpers; the paper's E_tex,p (SRM high-pass energy, Eq. (3)) is
    ``cfgid.channel.e_tex.estimate_e_tex``.
    """
    p = _finite_float32("patches", patches)
    if p.ndim != 2:
        raise ValueError(f"patches must be (P, D); got {p.shape}")
    return np.var(p, axis=1).astype(np.float32)


def _validate_result(result: InversionResult) -> None:
    rho = _finite_float32("rho_patch", result.rho_patch)
    e_tex = _finite_float32("e_tex_patch", result.e_tex_patch)
    residual = _finite_float32("residual", result.residual)
    if rho.ndim != 1:
        raise ValueError(f"rho_patch must be (P,); got {rho.shape}")
    if e_tex.shape != rho.shape:
        raise ValueError(f"e_tex_patch shape {e_tex.shape} != rho_patch shape {rho.shape}")
    if residual.ndim != 2 or residual.shape[0] != rho.shape[0]:
        raise ValueError(f"residual must be (P, D) with P={rho.shape[0]}; got {residual.shape}")
    rows, cols = result.patch_grid
    if rows <= 0 or cols <= 0 or rows * cols != rho.shape[0]:
        raise ValueError(f"patch_grid {result.patch_grid} does not match P={rho.shape[0]}")
    if not np.isfinite(result.sigma):
        raise ValueError(f"sigma must be finite, got {result.sigma}")
    if not (np.isnan(result.qf) or np.isfinite(result.qf)):
        raise ValueError(f"qf must be finite or NaN, got {result.qf}")


def pack_results(
    results: list[InversionResult],
    *,
    model_name: str,
    branch_mode: str,
) -> tuple[CompactBatch, np.ndarray, FeatureMetadata]:
    """Pack per-image results into store-ready arrays + metadata (validates uniform shapes)."""
    if not results:
        raise ValueError("pack_results requires at least one InversionResult")
    for r in results:
        _validate_result(r)
    n_patches = results[0].n_patches
    residual_dim = results[0].residual_dim
    grid = results[0].patch_grid
    for r in results:
        if r.n_patches != n_patches or r.residual_dim != residual_dim or r.patch_grid != grid:
            raise ValueError("InversionResults have inconsistent patch grid / residual dim")

    rho = np.stack([r.rho_patch for r in results]).astype(np.float32)
    e_tex = np.stack([r.e_tex_patch for r in results]).astype(np.float32)
    sigma = np.array([r.sigma for r in results], dtype=np.float32)
    qf = np.array([r.qf for r in results], dtype=np.float32)
    residual = np.stack([r.residual for r in results]).astype(np.float32)
    image_ids = [r.image_id or f"img_{i}" for i, r in enumerate(results)]

    batch = CompactBatch(rho=rho, e_tex=e_tex, sigma=sigma, qf=qf)
    meta = FeatureMetadata(
        model_name=model_name,
        branch_mode=branch_mode,
        n_images=len(results),
        n_patches=n_patches,
        patch_grid=grid,
        residual_dim=residual_dim,
        image_ids=image_ids,
        sigma=float(np.mean(sigma)) if len(sigma) else None,
    )
    return batch, residual, meta


# --------------------------------------------------------------------------- #
# backend contract
# --------------------------------------------------------------------------- #
class InversionBackend(ABC):
    """Abstract dual-branch inversion backend (one concrete class per backbone family)."""

    backend_key: str = ""
    branch_mode: str = ""

    def __init__(self, spec: ModelSpec, device: torch.device) -> None:
        self.spec = spec
        self.device = device
        self._loaded = False
        self._pipe: Any = None

    @abstractmethod
    def load(self) -> None:
        """Load the frozen pipeline from ``spec.resolved_local_path()`` onto ``self.device``.

        Must raise :class:`BackendUnavailableError` if weights or diffusers are missing.
        """

    @abstractmethod
    def dual_branch_residual(
        self,
        image: np.ndarray,
        prompt: str,
        *,
        num_steps: int = 50,
        seed: int = 0,
        capture_step_frac: float = 0.5,
    ) -> DualBranchResidual:
        """Two deterministic inversions (cond prompt vs empty) -> per-step residual maps.

        The DDIM backends use the inverse update of Eq. (1); FLUX uses flow/RF inversion. Raw
        model glue: the config-driven reduction to rho / E_tex lives in
        ``channel.features.build_inversion_result``.
        """

    @abstractmethod
    def generate_image(
        self,
        prompt: str,
        *,
        w: float,
        seed: int,
        num_steps: int,
        height: int,
        width: int,
    ) -> np.ndarray:
        """Generate one image at guidance ``w`` (w=1 == H0). Returns (H, W, 3) float32 in [0, 1]."""

    @abstractmethod
    def invert(
        self,
        image: np.ndarray,
        prompt: str,
        *,
        num_steps: int = 50,
        seed: int = 0,
    ) -> InversionResult:
        """Convenience dual-branch inversion of ``image`` -> InversionResult (default knobs)."""

    def is_loaded(self) -> bool:
        return self._loaded

    def require_loaded(self) -> None:
        if not self._loaded:
            raise RuntimeError(f"{type(self).__name__}: call load() before use")

    def unload(self) -> None:
        self._pipe = None
        self._loaded = False
