"""FLUX flow-inversion backend: kept for completeness, NOT used in the paper.

FLUX.1-dev's guidance-distilled flow-matching pipeline lies outside the DDIM-invertible family and
is excluded from all reported experiments (Appendix B). This module stays because the backend
registry, the package exports and ``configs/models.yaml`` reference it; ``sweep.run_p0`` rejects
its ``flux_specific_appendix`` branch mode.

FLUX.1-dev is guidance-distilled: there is no SD-style cond/uncond split by default. diffusers'
``FluxPipeline`` does expose a real second branch via ``true_cfg_scale > 1`` + ``negative_prompt``,
so rho is computable — but the distilled ``guidance`` embedding enters both branches, so rho is
**NOT homogeneous** with SD. Hence ``branch_mode = flux_specific_appendix``: FLUX results would
stay a separate statistic and never enter the SD-homogeneous sigma–w* core regression. Inversion
is flow/ODE (rectified-flow reverse Euler), NOT DDIM.

Generation contract: the swept guidance ``w`` is realized via ``true_cfg_scale`` (genuine dual
transformer branch, ``noise = neg + w * (cond - neg)``); the distilled embedding is pinned to
:data:`FIXED_GUIDANCE` for every ``w`` so it acts as a constant nuisance, not a confound.
``w == 1.0`` means ``true_cfg_scale = 1.0`` (single branch, the H0 arm).

All diffusers call signatures below were verified against the installed diffusers 0.38.0 source
(``diffusers/pipelines/flux/pipeline_flux.py``, ``pipeline_flux_img2img.py``,
``schedulers/scheduling_flow_match_euler_discrete.py``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from cfgid.channel.inversion.base import (
    BackendUnavailableError,
    DualBranchResidual,
    InversionBackend,
    InversionResult,
    torch_dtype,
)
from cfgid.numerics import as_finite_float32

if TYPE_CHECKING:
    import torch

# Distilled-guidance embedding value, held CONSTANT across the entire w sweep (both branches,
# generation AND inversion) so the distilled mechanism is a constant nuisance while w itself is
# realized via the true_cfg_scale dual branch. 3.5 is the FluxPipeline default for FLUX.1-dev.
FIXED_GUIDANCE: float = 3.5

_INVERT_DEFER_MSG = (
    "FluxFlowBackend.invert (full-trajectory faithful inversion -> InversionResult with "
    "calibrated sigma) is not implemented; this backend provides only dual_branch_residual "
    "(single capture step). FLUX is not used in the paper; see README.md."
)

_SMOKE_DEFER_MSG = (
    "FluxFlowBackend.smoke_step is intentionally not implemented: a single off-schedule "
    "transformer forward on a random packed latent has no acceptance value for the flow "
    "backend. The FLUX smoke is the skip-guarded tiny-resolution dual_branch_residual."
)


# PEP 695 constraints are evaluated lazily (Python 3.12), so the TYPE_CHECKING-only torch
# import keeps this module importable without torch at runtime.
def reverse_flow_euler_step[ArrayT: (np.ndarray, torch.Tensor)](
    sample: ArrayT,
    velocity: ArrayT,
    sigma_current: float,
    sigma_next: float,
) -> ArrayT:
    """Exact algebraic inverse of the deterministic FlowMatchEuler forward update.

    Forward (``FlowMatchEulerDiscreteScheduler.step``, non-stochastic branch, installed
    diffusers 0.38.0): ``z_next = z + (sigma_next - sigma_current) * v`` with the sigma ladder
    walked DOWN (``sigma_current > sigma_next``).

    This inverse walks the ladder UP (toward noise): given ``z_next`` at ``sigma_next``,
    recover ``z`` at ``sigma_current``::

        z = z_next - (sigma_next - sigma_current) * v

    It is exact when ``v`` equals the forward step's velocity; the explicit inversion loop
    approximates that velocity by evaluating the transformer at the available (more-denoised)
    state with the SAME timestep — standard first-order rectified-flow inversion, the same
    discretization-error class as the forward Euler sampler.
    """
    if not (np.isfinite(sigma_current) and np.isfinite(sigma_next)):
        raise ValueError(f"sigmas must be finite, got ({sigma_current}, {sigma_next})")
    return sample - (sigma_next - sigma_current) * velocity


class FluxFlowBackend(InversionBackend):
    """FLUX.1-dev dual-branch flow backend (true_cfg_scale branch split, reverse-Euler ODE)."""

    backend_key = "flux_flow"
    branch_mode = "flux_specific_appendix"

    # CUDA GPUs below this total memory cannot hold the full bf16 FLUX pipeline (~34GB weights)
    # plus activations resident; model-level CPU offload is enabled instead (e.g. 40 GB GPUs).
    _OFFLOAD_BELOW_GB: float = 46.0

    def _should_offload(self) -> bool:
        if self.device.type != "cuda":
            return False  # MPS unified memory / CPU: keep fully resident
        import torch

        total_gb = torch.cuda.get_device_properties(self.device).total_memory / 1024**3
        return total_gb < self._OFFLOAD_BELOW_GB

    def load(self) -> None:
        if self._loaded:
            return
        if not self.spec.is_present():
            raise BackendUnavailableError(
                f"{self.spec.name}: weights missing at {self.spec.resolved_local_path()} "
                f"(missing: {self.spec.missing_files()}). FLUX.1-dev is gated; see README.md."
            )
        try:
            from diffusers import FluxPipeline
        except ImportError as exc:
            raise BackendUnavailableError(f"diffusers unavailable: {exc}") from exc

        pipe = FluxPipeline.from_pretrained(
            str(self.spec.resolved_local_path()),
            torch_dtype=torch_dtype(self.spec.dtype),
            local_files_only=True,
        )
        if self._should_offload():
            # FLUX bf16 weights total ~34GB (transformer 23.8G + T5 9.5G): fully resident does
            # NOT fit a 40 GB GPU once activations are added. Model-level CPU offload keeps only
            # the active component on the GPU (peak ~25GB; seconds of transfer per image; no
            # quality change). Accelerate hooks also fire on our direct component calls.
            pipe.enable_model_cpu_offload(device=self.device)
        else:
            # Large-unified-memory MPS hosts and >=48GB GPUs hold it fully resident.
            pipe = pipe.to(self.device)
        # The rho channel needs the distilled guidance input so FIXED_GUIDANCE can be pinned as
        # a constant nuisance. FLUX.1-schnell (guidance-free, guidance_embeds=False) can never
        # yield rho; fail loudly instead of silently dropping the guidance tensor.
        if not bool(pipe.transformer.config.guidance_embeds):
            raise BackendUnavailableError(
                f"{self.spec.name}: transformer.config.guidance_embeds is False (schnell-style "
                "guidance-free distillation). The flux_specific_appendix rho channel requires a "
                "guidance-embedded checkpoint (FLUX.1-dev); see README.md."
            )
        # reverse_flow_euler_step inverts only the deterministic branch of scheduler.step; a
        # stochastic-sampling scheduler config would silently break the inversion contract.
        if bool(pipe.scheduler.config.get("stochastic_sampling", False)):
            raise BackendUnavailableError(
                f"{self.spec.name}: scheduler config stochastic_sampling=True is not invertible "
                "by the deterministic reverse-Euler update; pin a deterministic "
                "FlowMatchEulerDiscreteScheduler config."
            )
        self._pipe = pipe
        self._loaded = True

    # ----------------------------------------------------------------------- #
    # generation
    # ----------------------------------------------------------------------- #
    def generate_image(
        self, prompt: str, *, w: float, seed: int, num_steps: int, height: int, width: int
    ) -> np.ndarray:
        self.require_loaded()
        import torch

        if not np.isfinite(w) or w < 1.0:
            # true_cfg_scale <= 1 silently degenerates to the single-branch arm inside the
            # pipeline; w < 1 would alias onto w == 1 and corrupt the sweep. Hard-fail instead.
            raise ValueError(f"w must be finite and >= 1 (w == 1 is the H0 arm), got {w}")
        if num_steps <= 0:
            raise ValueError(f"num_steps must be positive, got {num_steps}")
        # CPU generator + same seed for the H0 (w=1) / H1 (w>1) pair => differ ONLY in guidance.
        generator = torch.Generator(device="cpu").manual_seed(seed)
        with torch.no_grad():
            out = self._pipe(
                prompt=prompt,
                negative_prompt="",  # empty uncond branch, SD-analogous
                true_cfg_scale=float(w),  # the swept w; > 1 triggers the genuine second branch
                guidance_scale=FIXED_GUIDANCE,  # distilled embedding: constant nuisance
                num_inference_steps=num_steps,
                height=height,
                width=width,
                generator=generator,
                output_type="np",
            )
        return as_finite_float32("generated_image", out.images[0])

    # ----------------------------------------------------------------------- #
    # inversion
    # ----------------------------------------------------------------------- #
    @staticmethod
    def _capture_index(num_steps: int, capture_step_frac: float) -> int:
        # Mirrors _SDInversionBackend._capture_index (duplicated rather than importing a
        # private SD-family symbol into the FLUX module).
        if num_steps <= 0:
            raise ValueError(f"num_steps must be positive, got {num_steps}")
        if not (0.0 <= capture_step_frac <= 1.0):
            raise ValueError(f"capture_step_frac must be in [0, 1], got {capture_step_frac}")
        return max(0, min(num_steps - 1, int(capture_step_frac * num_steps)))

    def _encode_image_to_latent(self, image: np.ndarray) -> torch.Tensor:
        """Image (H, W, 3) in [0, 1] -> normalized FLUX VAE latent (1, C, H/8, W/8).

        Mirrors ``FluxImg2ImgPipeline._encode_vae_image`` (installed 0.38.0): the FLUX VAE
        latent is normalized as ``(z - shift_factor) * scaling_factor`` — NOT the plain
        ``scaling_factor`` scaling of SD. One deliberate deviation: img2img *samples* the
        posterior (``retrieve_latents`` sample mode); a forensic inversion needs a
        deterministic source, so we take ``latent_dist.mean``.
        """
        import torch

        img = as_finite_float32("image", image)
        if img.ndim != 3 or img.shape[2] != 3:
            raise ValueError(f"image must be (H, W, 3) in [0, 1]; got {img.shape}")
        if float(img.min()) < 0.0 or float(img.max()) > 1.0:
            raise ValueError("image must be in [0, 1]")
        multiple = int(self._pipe.vae_scale_factor) * 2  # 8x VAE + 2x2 latent packing
        if img.shape[0] % multiple != 0 or img.shape[1] % multiple != 0:
            raise ValueError(
                f"image dims must be divisible by {multiple} (VAE 8x + 2x2 packing); "
                f"got {img.shape[0]}x{img.shape[1]}"
            )
        vae = self._pipe.vae
        transformer_dtype = self._pipe.transformer.dtype
        original_vae_dtype = vae.dtype
        x = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)  # (1, 3, H, W)
        # Same upcast contract as the SD backends: only when the checkpoint asks for it.
        needs_upcast = vae.dtype == torch.float16 and bool(
            getattr(vae.config, "force_upcast", False)
        )
        try:
            if needs_upcast:
                vae.to(torch.float32)
            x = (x * 2.0 - 1.0).to(device=self.device, dtype=vae.dtype)
            with torch.no_grad():
                latent = vae.encode(x).latent_dist.mean
        finally:
            if needs_upcast:
                vae.to(original_vae_dtype)
        latent = (latent - vae.config.shift_factor) * vae.config.scaling_factor
        latent = latent.to(dtype=transformer_dtype)
        if not torch.all(torch.isfinite(latent.float())):
            raise BackendUnavailableError(
                f"{self.spec.name}: VAE encode produced non-finite latent"
            )
        return latent

    def _set_inversion_schedule(self, num_steps: int, image_seq_len: int) -> list[float]:
        """Configure the FlowMatchEuler schedule exactly as ``FluxPipeline.__call__`` step 5.

        Returns the post-shift sigma ladder as floats (length ``num_steps + 1``, descending,
        terminal 0.0 appended by ``set_timesteps``); ``scheduler.timesteps`` is set as a
        side effect.
        """
        from diffusers.pipelines.flux.pipeline_flux import calculate_shift

        scheduler = self._pipe.scheduler
        sigmas: np.ndarray | None = np.linspace(1.0, 1.0 / num_steps, num_steps)
        if hasattr(scheduler.config, "use_flow_sigmas") and scheduler.config.use_flow_sigmas:
            sigmas = None
        mu = calculate_shift(
            image_seq_len,
            scheduler.config.get("base_image_seq_len", 256),
            scheduler.config.get("max_image_seq_len", 4096),
            scheduler.config.get("base_shift", 0.5),
            scheduler.config.get("max_shift", 1.15),
        )
        scheduler.set_timesteps(num_steps, device=self.device, sigmas=sigmas, mu=mu)
        return [float(s) for s in scheduler.sigmas.cpu().tolist()]

    def _velocity(
        self,
        z: torch.Tensor,
        timestep: torch.Tensor,
        guidance: torch.Tensor,
        embeds: torch.Tensor,
        pooled: torch.Tensor,
        text_ids: torch.Tensor,
        img_ids: torch.Tensor,
        *,
        branch: str,
        step: int,
    ) -> torch.Tensor:
        """One transformer forward; kwargs mirror ``FluxPipeline.__call__``'s denoising call."""
        import torch

        transformer = self._pipe.transformer
        with torch.no_grad():
            velocity = transformer(
                hidden_states=z.to(transformer.dtype),
                timestep=timestep / 1000,  # pipeline convention: t / 1000 == sigma
                guidance=guidance,
                pooled_projections=pooled,
                encoder_hidden_states=embeds,
                txt_ids=text_ids,
                img_ids=img_ids,
                return_dict=False,
            )[0]
        if not torch.all(torch.isfinite(velocity)):
            raise BackendUnavailableError(
                f"{self.spec.name}: inversion diverged (non-finite {branch} velocity) "
                f"at inversion step {step}"
            )
        return velocity

    def _unpack_to_chw(self, packed: torch.Tensor, height: int, width: int) -> np.ndarray:
        # FluxPipeline._unpack_latents is a (stable, documented) staticmethod that undoes the
        # 2x2 sequence packing; height/width are PIXEL dims (it re-derives the latent dims).
        unpacked = self._pipe._unpack_latents(packed, height, width, self._pipe.vae_scale_factor)
        return as_finite_float32("unpacked_tensor", unpacked[0].float().cpu().numpy())

    def dual_branch_residual(
        self,
        image: np.ndarray,
        prompt: str,
        *,
        num_steps: int = 50,
        seed: int = 0,
        capture_step_frac: float = 0.5,
    ) -> DualBranchResidual:
        """Reverse-Euler flow inversion with a dual-branch velocity capture at one step.

        Direction: the forward sampler walks the sigma ladder DOWN (sigma_max -> 0); this
        inversion starts from the clean VAE latent at the appended terminal sigma 0 and walks
        the SAME ladder UP, applying :func:`reverse_flow_euler_step` (the exact algebraic
        inverse of the scheduler's deterministic update) with the velocity evaluated at the
        available state and the forward step's own timestep.

        Branch design: the trajectory is driven by the COND velocity alone; at the capture step
        both branches (cond prompt vs empty negative prompt) run on the SAME state ``z``, so the
        captured pair is an on-path cond residual plus an uncond probe at an identical state
        (unlike SD's two independent single-branch inversions — one trajectory halves compute
        and removes cross-trajectory state drift from the rho contrast). The distilled
        ``guidance`` tensor is FIXED_GUIDANCE in BOTH branches (constant nuisance; rho remains
        NOT homogeneous with SD, see module docstring).
        """
        self.require_loaded()
        import torch

        torch.manual_seed(seed)  # flow inversion is deterministic; seed any incidental sampling
        capture_idx = self._capture_index(num_steps, capture_step_frac)
        pipe = self._pipe
        transformer_dtype = pipe.transformer.dtype

        latent = self._encode_image_to_latent(image)  # (1, C, h_lat, w_lat), normalized
        num_channels = int(latent.shape[1])
        h_lat, w_lat = int(latent.shape[2]), int(latent.shape[3])
        # Pixel dims re-derived from the latent (h_lat = H / vae_scale_factor, exact because
        # _encode_image_to_latent enforces divisibility); used by _unpack_latents.
        height_px = h_lat * int(pipe.vae_scale_factor)
        width_px = w_lat * int(pipe.vae_scale_factor)
        # _pack_latents / _prepare_latent_image_ids are stable FluxPipeline staticmethods.
        packed = pipe._pack_latents(latent, 1, num_channels, h_lat, w_lat)
        img_ids = pipe._prepare_latent_image_ids(
            1, h_lat // 2, w_lat // 2, self.device, transformer_dtype
        )

        with torch.no_grad():
            cond_embeds, cond_pooled, cond_text_ids = pipe.encode_prompt(
                prompt=prompt, prompt_2=None, device=self.device, num_images_per_prompt=1
            )
            uncond_embeds, uncond_pooled, uncond_text_ids = pipe.encode_prompt(
                prompt="", prompt_2=None, device=self.device, num_images_per_prompt=1
            )
        # Same guidance tensor for BOTH branches, mirroring FluxPipeline.__call__.
        guidance = torch.full([1], FIXED_GUIDANCE, device=self.device, dtype=torch.float32)
        guidance = guidance.expand(packed.shape[0])

        sigma_list = self._set_inversion_schedule(num_steps, int(packed.shape[1]))
        timesteps = pipe.scheduler.timesteps  # length num_steps; t_i = sigma_i * 1000

        z = packed.to(torch.float32)  # fp32 state, like scheduler.step's internal upcast
        r_cond: np.ndarray | None = None
        r_uncond: np.ndarray | None = None
        for k in range(num_steps):
            i = num_steps - 1 - k  # scheduler index: z sits at sigma_{i+1}, stepping to sigma_i
            timestep = timesteps[i].expand(z.shape[0]).to(transformer_dtype)
            v_cond = self._velocity(
                z,
                timestep,
                guidance,
                cond_embeds,
                cond_pooled,
                cond_text_ids,
                img_ids,
                branch="cond",
                step=k,
            )
            if k == capture_idx:
                v_uncond = self._velocity(
                    z,
                    timestep,
                    guidance,
                    uncond_embeds,
                    uncond_pooled,
                    uncond_text_ids,
                    img_ids,
                    branch="uncond",
                    step=k,
                )
                r_cond = self._unpack_to_chw(v_cond, height_px, width_px)
                r_uncond = self._unpack_to_chw(v_uncond, height_px, width_px)
            # Continue the trajectory with the COND branch velocity (see docstring).
            z = reverse_flow_euler_step(
                z, v_cond.to(torch.float32), sigma_list[i], sigma_list[i + 1]
            )
            if not torch.all(torch.isfinite(z)):
                raise BackendUnavailableError(
                    f"{self.spec.name}: inversion diverged (non-finite state) at inversion step {k}"
                )
        if r_cond is None or r_uncond is None:
            raise RuntimeError(f"capture step {capture_idx} not reached in {num_steps} steps")

        meta: dict[str, Any] = {
            "num_steps": int(num_steps),
            "seed": int(seed),
            "capture_step_frac": float(capture_step_frac),
            "branch": "cond_vs_empty_neg",
            "fixed_guidance": FIXED_GUIDANCE,
        }
        return DualBranchResidual(
            r_cond=r_cond,
            r_uncond=r_uncond,
            latent=as_finite_float32("latent", latent[0].float().cpu().numpy()),
            capture_step=capture_idx,
            meta=meta,
        )

    # ----------------------------------------------------------------------- #
    # deferred surface
    # ----------------------------------------------------------------------- #
    def invert(
        self,
        image: np.ndarray,
        prompt: str,
        *,
        num_steps: int = 50,
        seed: int = 0,
    ) -> InversionResult:
        raise NotImplementedError(_INVERT_DEFER_MSG)

    def smoke_step(self, prompt: str, *, seed: int = 0) -> InversionResult:
        raise NotImplementedError(_SMOKE_DEFER_MSG)
