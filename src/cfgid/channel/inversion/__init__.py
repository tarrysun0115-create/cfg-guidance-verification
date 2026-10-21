"""Dual-branch inversion backends (Section III-B; Appendix B, Table XV).

Each DDIM backend runs two deterministic inversions of the VAE-encoded query (conditioned on the
recorded prompt and on null text) with the update of Eq. (1), and captures both branches'
epsilon-parameterized noise predictions at the capture point (step 25 of 50):

  * sd_ddim          — SD1.5 epsilon-pred DDIM inverse                          (true_cfg_core)
  * sd_vpred_ddim    — SD2.1 v-pred DDIM inverse, captured as epsilon (v->eps)  (true_cfg_core)
  * sdxl_ddim        — SDXL epsilon-pred DDIM inverse                           (true_cfg_core)
  * pixart_ddim      — PixArt-Sigma epsilon-pred DDIM inverse (DiT)             (true_cfg_core)
  * hunyuan_dit_ddim — HunyuanDiT v-pred DDIM inverse, captured as epsilon (DiT) (true_cfg_core)
  * flux_flow        — FLUX flow/RF inversion (flux_specific_appendix); kept for completeness
                       but NOT used in the paper

DDIM inversion does NOT apply to FLUX (flow-matching); see ``flux_flow``.
"""

from cfgid.channel.inversion.base import (
    BackendUnavailableError,
    InversionBackend,
    InversionResult,
    e_tex_from_patches,
    infer_patch_grid,
    pack_results,
    patchify,
    rho_from_patch_residuals,
    torch_dtype,
)
from cfgid.channel.inversion.flux_flow import FluxFlowBackend
from cfgid.channel.inversion.hunyuan_dit_ddim import HunyuanDiTDDIMBackend
from cfgid.channel.inversion.pixart_ddim import PixArtDDIMBackend
from cfgid.channel.inversion.registry import (
    available_backends,
    backend_class,
    get_backend,
)
from cfgid.channel.inversion.sd_ddim import SD15DDIMBackend
from cfgid.channel.inversion.sd_vpred_ddim import SD21VPredDDIMBackend
from cfgid.channel.inversion.sdxl_ddim import SDXLDDIMBackend

__all__ = [
    "InversionBackend",
    "InversionResult",
    "BackendUnavailableError",
    "torch_dtype",
    "infer_patch_grid",
    "patchify",
    "rho_from_patch_residuals",
    "e_tex_from_patches",
    "pack_results",
    "SD15DDIMBackend",
    "SDXLDDIMBackend",
    "SD21VPredDDIMBackend",
    "FluxFlowBackend",
    "PixArtDDIMBackend",
    "HunyuanDiTDDIMBackend",
    "available_backends",
    "backend_class",
    "get_backend",
]
