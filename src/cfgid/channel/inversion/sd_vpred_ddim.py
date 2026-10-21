"""SD2.x v-prediction DDIM-inverse backend (SD2.1; Appendix B, Table XV).

SD2.1 is one of the three distinct-VAE decoders. It predicts v, not epsilon, but the rho channel
is defined on epsilon residuals (so rho stays HOMOGENEOUS across decoders; Section III-B). This
backend reuses the shared SD-family loader/inversion
(:class:`cfgid.channel.inversion.sd_ddim._SDInversionBackend`) and plugs in two hooks:

* ``_required_prediction_type = "v_prediction"`` -- the loader accepts the v-pred checkpoint
  instead of failing the epsilon guard.
* ``_to_epsilon`` -- converts the captured UNet v-output to epsilon via the standard
  diffusers identity ``eps = sqrt(alpha_t) * v + sqrt(1 - alpha_t) * z_t`` (alpha_t =
  ``alphas_cumprod[t]``), so the per-step rho residual is the same quantity sd15/sdxl produce.

The inverse SCHEDULER step still receives the raw v output (``DDIMInverseScheduler`` is configured
for ``v_prediction`` and does the v->x0/eps conversion internally for the trajectory); only the
captured rho residual is mapped to epsilon. ``_encode_branch`` is the single-text-encoder SD default
(SD2.1 uses one OpenCLIP-ViT-H/14 encoder), so no SDXL-style pooled/time-id override is needed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from cfgid.channel.inversion.sd_ddim import _SDInversionBackend

if TYPE_CHECKING:  # torch is a heavy runtime import; only needed for type-checkers here
    import torch


def v_to_epsilon(v: torch.Tensor, z_t: torch.Tensor, alpha_t: torch.Tensor) -> torch.Tensor:
    """v-prediction -> epsilon: ``eps = sqrt(alpha_t) * v + sqrt(1 - alpha_t) * z_t``.

    The standard diffusers v-parameterization identity (matches ``DDIMScheduler``'s internal
    ``pred_epsilon`` for ``prediction_type='v_prediction'``). ``alpha_t`` is the cumulative product
    of alphas at the current timestep; ``z_t`` is the current (noisy) latent. Pure tensor op.
    """
    import torch

    sqrt_alpha = torch.sqrt(alpha_t)
    sqrt_one_minus = torch.sqrt(torch.clamp(1.0 - alpha_t, min=0.0))
    return sqrt_alpha * v + sqrt_one_minus * z_t


class SD21VPredDDIMBackend(_SDInversionBackend):
    """SD2.x v-prediction DDIM-inverse backend (true-CFG core; epsilon-homogeneous rho)."""

    backend_key = "sd_vpred_ddim"
    branch_mode = "true_cfg_core"
    _pipeline_import = ("diffusers", "StableDiffusionPipeline")
    _required_prediction_type = "v_prediction"

    def _from_pretrained_kwargs(self) -> dict[str, Any]:
        # SD2.x ships a safety checker we neither need nor want for forensic features.
        return {"safety_checker": None, "requires_safety_checker": False}

    def _to_epsilon(
        self, model_output: torch.Tensor, z: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        import torch

        acp = self._pipe.inverse_scheduler.alphas_cumprod.to(device=z.device, dtype=torch.float32)
        alpha_t = acp[int(t)]
        return v_to_epsilon(model_output.to(torch.float32), z.to(torch.float32), alpha_t)
