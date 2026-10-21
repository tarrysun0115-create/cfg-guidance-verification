"""Config-driven reduction of a dual-branch residual to an InversionResult (pure, testable).

Separated from the model glue (``channel.inversion``): given the raw (C, H, W) cond/uncond
noise-prediction maps captured at the capture point (Section III-B) and the source latent z_0,
patchify them on the shared latent grid and compute, per patch, rho_p of Eq. (2) with the p0
knobs (denominator floor, log ratio; ``channel.rho.rho_patch``), E_tex,p of Eq. (3)
(``channel.e_tex``), and the full-dimensional difference r_cond,p - r_uncond,p, whose norm is
delta_p of Eq. (6). Called by the main-grid driver ``cfgid.sweep.run_p0`` and by the GPU
analysis scripts that re-invert images. No torch / no weights.
"""

from __future__ import annotations

import numpy as np

from cfgid.channel.e_tex import estimate_e_tex
from cfgid.channel.inversion.base import (
    DualBranchResidual,
    InversionResult,
    infer_patch_grid,
    patchify,
)
from cfgid.channel.rho import rho_patch
from cfgid.numerics import as_finite_float32


def build_inversion_result(
    dual: DualBranchResidual,
    *,
    patch_size: int,
    floor: float,
    log_ratio: bool,
    sigma: float,
    e_tex_method: str = "srm_highpass",
    qf: float = float("nan"),
    image_id: str = "",
) -> InversionResult:
    """Reduce raw (C, H, W) dual-branch residual maps to a per-patch InversionResult."""
    rc = as_finite_float32("r_cond", dual.r_cond)
    ru = as_finite_float32("r_uncond", dual.r_uncond)
    lat = as_finite_float32("latent", dual.latent)
    if rc.ndim != 3 or rc.shape != ru.shape:
        raise ValueError(f"r_cond/r_uncond must be matching (C, H, W); got {rc.shape}, {ru.shape}")
    # Row-major patch grid; at 512^2 with patch_size 4 the 64x64 latent gives 16x16 patches.
    grid = infer_patch_grid(rc.shape[1], rc.shape[2], patch_size)
    pc = patchify(rc, grid)
    pu = patchify(ru, grid)
    return InversionResult(
        rho_patch=rho_patch(pc, pu, floor=floor, log_ratio=log_ratio),
        e_tex_patch=estimate_e_tex(lat, patch_size, method=e_tex_method),
        # Stored as residual_feature_v1; delta_p = ||residual_p|| is computed downstream.
        residual=(pc - pu).astype(np.float32),
        sigma=sigma,
        patch_grid=grid,
        qf=qf,
        image_id=image_id,
        meta={"capture_step": dual.capture_step, **dual.meta},
    )
