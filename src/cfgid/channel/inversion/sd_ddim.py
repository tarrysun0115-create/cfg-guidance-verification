"""SD1.5 (and shared SD-family) epsilon-prediction DDIM-inverse backend (Section III-B).

``_SDInversionBackend`` is shared by SD1.5 (here), SDXL (``sdxl_ddim``) and SD2.1
(``sd_vpred_ddim``). ``dual_branch_residual`` VAE-encodes the query to z_0 (posterior mean times
the VAE scaling factor), runs two independent single-conditioned DDIM inversions with the update
of Eq. (1) -- one with the recorded prompt, one with the empty prompt -- and captures each
branch's epsilon noise prediction at the capture step (``capture_step_frac`` 0.5 of 50 steps =
step 25). No CFG combination is formed inside the inversion, and any non-finite prediction or
encoded latent aborts with :class:`BackendUnavailableError` (the numerical guard of Section III-B).
``smoke_step`` is a single-forward plumbing check, not the paper's statistic.
Heavy diffusers imports are lazy; loading raises :class:`BackendUnavailableError` when
weights/diffusers are missing so the path stays import-safe without a download.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from cfgid.channel.inversion.base import (
    BackendUnavailableError,
    DualBranchResidual,
    InversionBackend,
    InversionResult,
    e_tex_from_patches,
    infer_patch_grid,
    patchify,
    rho_from_patch_residuals,
    torch_dtype,
)
from cfgid.numerics import as_finite_float32

if TYPE_CHECKING:
    import torch


class _SDInversionBackend(InversionBackend):
    """Shared SD-family loader + dual-branch smoke step.

    Subclasses set ``_pipeline_import`` and override ``_encode_branch`` for any
    family-specific conditioning (SDXL needs pooled text embeds + time ids).
    """

    # (module, class) resolved lazily, e.g. ("diffusers", "StableDiffusionPipeline").
    _pipeline_import: tuple[str, str] = ("diffusers", "StableDiffusionPipeline")
    # The rho channel assumes epsilon-prediction; the loader fails loudly on any other pred-type.
    # v-prediction subclasses set "v_prediction" and convert the UNet output to epsilon in
    # ``_to_epsilon`` (the captured residual must stay epsilon-homogeneous for rho).
    _required_prediction_type: str = "epsilon"

    def _from_pretrained_kwargs(self) -> dict[str, Any]:
        return {}

    def _to_epsilon(
        self, model_output: torch.Tensor, z: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        """UNet output -> epsilon residual for rho. Identity for epsilon-prediction; v-prediction
        subclasses override (eps = sqrt(alpha_t)*v + sqrt(1-alpha_t)*z_t)."""
        return model_output

    def _capture_epsilon(
        self, model_output: torch.Tensor, z: torch.Tensor, t: torch.Tensor, *, context: str
    ) -> torch.Tensor:
        """Validate raw UNet output and convert it to the epsilon residual used by rho."""
        import torch

        if not torch.all(torch.isfinite(model_output)):
            raise BackendUnavailableError(f"{self.spec.name}: non-finite model output ({context})")
        eps = self._to_epsilon(model_output, z, t)
        if not torch.all(torch.isfinite(eps)):
            raise BackendUnavailableError(
                f"{self.spec.name}: non-finite epsilon residual ({context})"
            )
        return eps

    def load(self) -> None:
        if self._loaded:
            return
        if not self.spec.is_present():
            raise BackendUnavailableError(
                f"{self.spec.name}: weights missing at {self.spec.resolved_local_path()} "
                f"(missing: {self.spec.missing_files()})"
            )
        if self.spec.scheduler != "DDIMScheduler":
            raise BackendUnavailableError(
                f"{self.spec.name}: scheduler '{self.spec.scheduler}' is not supported here "
                "(expected DDIMScheduler; configs/p0.yaml grid.gen_scheduler=ddim)"
            )
        try:
            from importlib import import_module

            from diffusers import DDIMInverseScheduler, DDIMScheduler
        except ImportError as exc:  # diffusers not installed
            raise BackendUnavailableError(f"diffusers unavailable: {exc}") from exc

        module = import_module(self._pipeline_import[0])
        pipe_cls = getattr(module, self._pipeline_import[1])
        kwargs: dict[str, Any] = {
            "torch_dtype": torch_dtype(self.spec.dtype),
            "local_files_only": True,
            "use_safetensors": True,
        }
        kwargs.update(self._from_pretrained_kwargs())
        if self.spec.variant:
            kwargs["variant"] = self.spec.variant
        pipe = pipe_cls.from_pretrained(str(self.spec.resolved_local_path()), **kwargs)
        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
        pipe = pipe.to(self.device)
        # Fail loudly on an unexpected pred-type rather than silently producing a wrong rho. The
        # base requires epsilon; a v-prediction subclass requires "v_prediction" and converts the
        # UNet output to epsilon in ``_to_epsilon`` so the captured residual stays homogeneous.
        pred_type = pipe.scheduler.config.get("prediction_type", "epsilon")
        if pred_type != self._required_prediction_type:
            raise BackendUnavailableError(
                f"{self.spec.name}: scheduler prediction_type='{pred_type}', expected "
                f"'{self._required_prediction_type}' for backend '{self.backend_key}'"
            )
        # Pre-build the inverse scheduler for the DDIM inversion trajectory (Eq. (1)).
        pipe.inverse_scheduler = DDIMInverseScheduler.from_config(pipe.scheduler.config)
        self._pipe = pipe
        self._loaded = True

    def _encode_branch(
        self, prompt: str, latent: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor] | None]:
        import torch

        # do_classifier_free_guidance=False -> the negative embeds are intentionally discarded:
        # the smoke runs each branch as its own single (non-CFG) forward, so cond vs uncond is
        # realized by calling this twice (prompt vs ""), not by an in-pass CFG combine.
        # no_grad: the text-encoder forward is inference-only (frozen backbone, no backprop through
        # the inversion chain); without it encode_prompt builds an autograd graph on the encoder.
        with torch.no_grad():
            prompt_embeds, _neg = self._pipe.encode_prompt(
                prompt,
                device=self.device,
                num_images_per_prompt=1,
                do_classifier_free_guidance=False,
            )
        return prompt_embeds, None

    def smoke_step(
        self,
        prompt: str,
        *,
        seed: int = 0,
        latent_size: int = 16,
        patch: int = 4,
        timestep: int = 500,
    ) -> InversionResult:
        """Single dual-branch UNet forward on a seeded random latent -> one InversionResult.

        This is a plumbing smoke (not the scientific rho): it exercises load -> dual-branch
        forward -> patchify -> rho/E_tex/residual -> feature store, end to end. ``latent_size``
        is a smoke-only knob (implied resolution is deliberately off-distribution); the
        inversion path does not inherit it as a default.
        """
        self.require_loaded()
        import torch

        unet = self._pipe.unet
        dtype = next(unet.parameters()).dtype
        in_channels = int(unet.config.in_channels)

        gen = torch.Generator().manual_seed(seed)
        latent = torch.randn(1, in_channels, latent_size, latent_size, generator=gen)
        latent = latent.to(device=self.device, dtype=dtype)
        t = torch.tensor([timestep], device=self.device, dtype=torch.long)

        cond_embeds, cond_added = self._encode_branch(prompt, latent)
        uncond_embeds, uncond_added = self._encode_branch("", latent)
        with torch.no_grad():
            model_cond = unet(
                latent, t, encoder_hidden_states=cond_embeds, added_cond_kwargs=cond_added
            ).sample
            model_uncond = unet(
                latent, t, encoder_hidden_states=uncond_embeds, added_cond_kwargs=uncond_added
            ).sample

        eps_cond = self._capture_epsilon(model_cond, latent, t, context="smoke cond branch")
        eps_uncond = self._capture_epsilon(model_uncond, latent, t, context="smoke uncond branch")
        rc = eps_cond[0].float().cpu().numpy()
        ru = eps_uncond[0].float().cpu().numpy()
        lat = latent[0].float().cpu().numpy()
        grid = infer_patch_grid(rc.shape[1], rc.shape[2], patch)
        pc = patchify(rc, grid)
        pu = patchify(ru, grid)
        pl = patchify(lat, grid)
        return InversionResult(
            rho_patch=rho_from_patch_residuals(pc, pu),
            e_tex_patch=e_tex_from_patches(pl),
            residual=(pc - pu).astype(np.float32),
            sigma=1.0,  # placeholder; calibrated sigma comes from channel.sigma_calib
            patch_grid=grid,
            qf=float("nan"),
            image_id="smoke_0",
            meta={"smoke": True, "timestep": int(timestep)},
        )

    @staticmethod
    def _capture_index(num_steps: int, capture_step_frac: float) -> int:
        # Capture point: inversion-loop index int(frac * num_steps); 0.5 * 50 = step 25.
        if num_steps <= 0:
            raise ValueError(f"num_steps must be positive, got {num_steps}")
        if not (0.0 <= capture_step_frac <= 1.0):
            raise ValueError(f"capture_step_frac must be in [0, 1], got {capture_step_frac}")
        return max(0, min(num_steps - 1, int(capture_step_frac * num_steps)))

    def _encode_image_to_latent(self, image: np.ndarray) -> torch.Tensor:
        import torch

        img = as_finite_float32("image", image)
        if img.ndim != 3 or img.shape[2] != 3:
            raise ValueError(f"image must be (H, W, 3) in [0, 1]; got {img.shape}")
        if float(img.min()) < 0.0 or float(img.max()) > 1.0:
            raise ValueError("image must be in [0, 1]")
        vae = self._pipe.vae
        original_vae_dtype = vae.dtype
        unet_dtype = next(self._pipe.unet.parameters()).dtype
        x = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)  # (1, 3, H, W)
        # SDXL's VAE is numerically unstable in fp16; upcast for the encode when the model asks.
        needs_upcast = vae.dtype == torch.float16 and bool(
            getattr(vae.config, "force_upcast", False)
        )
        try:
            if needs_upcast:
                vae.to(torch.float32)
            x = (x * 2.0 - 1.0).to(device=self.device, dtype=vae.dtype)
            # z_0 = encoder posterior MEAN (deterministic, no sampling) x VAE scaling factor.
            with torch.no_grad():
                latent = vae.encode(x).latent_dist.mean * vae.config.scaling_factor
        finally:
            if needs_upcast:
                vae.to(original_vae_dtype)
        latent = latent.to(dtype=unet_dtype)
        if not torch.all(torch.isfinite(latent)):
            raise BackendUnavailableError(
                f"{self.spec.name}: VAE encode produced non-finite latent"
            )
        return latent

    def _ddim_invert_capture(
        self, latent: torch.Tensor, prompt: str, num_steps: int, capture_idx: int
    ) -> np.ndarray:
        import torch

        # One single-conditioned DDIM inversion (prompt OR ""), t: 0 -> T in the noising direction.
        # DDIMInverseScheduler.step implements Eq. (1):
        #   z_{t+1} = sqrt(a_{t+1}/a_t) z_t
        #             + (sqrt(1 - a_{t+1}) - sqrt(a_{t+1}/a_t) sqrt(1 - a_t)) eps_theta(z_t, t, c),
        # with a_t = alphas_cumprod. The denoiser is evaluated once per step (no CFG combine).
        unet = self._pipe.unet
        scheduler = self._pipe.inverse_scheduler
        scheduler.set_timesteps(num_steps, device=self.device)
        z = latent.clone()
        embeds, added = self._encode_branch(prompt, z)
        captured: np.ndarray | None = None
        for i, t in enumerate(scheduler.timesteps):
            with torch.no_grad():
                # epsilon for the base; v for a v-prediction subclass. The inverse SCHEDULER step
                # consumes this raw output directly (it is configured for the model's pred-type and
                # converts internally); only the CAPTURED rho residual is mapped to epsilon.
                model_output = unet(
                    z, t, encoder_hidden_states=embeds, added_cond_kwargs=added
                ).sample
            if i == capture_idx:
                # The captured epsilon map is this branch's score residual r_branch (C, H, W).
                eps = self._capture_epsilon(model_output, z, t, context=f"inversion step {i}")
                captured = eps[0].float().cpu().numpy()
            elif not torch.all(torch.isfinite(model_output)):
                raise BackendUnavailableError(
                    f"{self.spec.name}: inversion diverged (non-finite residual) at step {i}"
                )
            z = scheduler.step(model_output, t, z).prev_sample
        if captured is None:
            raise RuntimeError(f"capture step {capture_idx} not reached in {num_steps} steps")
        return captured

    def dual_branch_residual(
        self,
        image: np.ndarray,
        prompt: str,
        *,
        num_steps: int = 50,
        seed: int = 0,
        capture_step_frac: float = 0.5,
    ) -> DualBranchResidual:
        self.require_loaded()
        import torch

        torch.manual_seed(seed)  # DDIM inversion is deterministic; seed any incidental sampling
        latent = self._encode_image_to_latent(image)
        capture_idx = self._capture_index(num_steps, capture_step_frac)
        # Two independent trajectories from the same z_0: each branch's prediction is read on its
        # own trajectory (Section III-B), conditioned on the prompt c and on null text "".
        r_cond = self._ddim_invert_capture(latent, prompt, num_steps, capture_idx)
        r_uncond = self._ddim_invert_capture(latent, "", num_steps, capture_idx)
        return DualBranchResidual(
            r_cond=r_cond,
            r_uncond=r_uncond,
            latent=latent[0].float().cpu().numpy(),
            capture_step=capture_idx,
            meta={
                "num_steps": int(num_steps),
                "seed": int(seed),
                "capture_step_frac": float(capture_step_frac),
            },
        )

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
        self.require_loaded()
        import torch

        # CPU generator + same seed for the H0 (w=1) / H1 (w>1) pair => differ ONLY in guidance.
        generator = torch.Generator(device="cpu").manual_seed(seed)
        with torch.no_grad():
            out = self._pipe(
                prompt=prompt,
                guidance_scale=float(w),
                num_inference_steps=num_steps,
                height=height,
                width=width,
                generator=generator,
                output_type="np",
            )
        return as_finite_float32("generated_image", out.images[0])

    def generate_image_with_trace(
        self,
        prompt: str,
        *,
        w: float,
        seed: int,
        num_steps: int,
        height: int,
        width: int,
        trace_step: int,
    ) -> tuple[np.ndarray, dict[str, np.ndarray | int | float]]:
        """generate_image + oracle-trace capture at generation step ``trace_step``.

        Captures the post-step latent (via the pipeline step callback) and, when ``w > 1``
        (CFG batch of 2: diffusers order is [uncond, cond]), the per-branch noise predictions
        via a UNet forward hook. These feed the inversion-artifact exclusion check
        (channel.oracle): the inversion-recovered rho must correlate with the oracle rho
        computed from the true generation trajectory.
        """
        self.require_loaded()
        import torch

        if not (0 <= trace_step < num_steps):
            raise ValueError(f"trace_step must be in [0, {num_steps}), got {trace_step}")
        trace: dict[str, np.ndarray | int | float] = {"gen_step": trace_step, "w": float(w)}
        do_cfg = float(w) > 1.0
        step_counter = {"i": -1}

        def _unet_hook(_module: Any, _inputs: tuple, output: Any) -> None:
            step_counter["i"] += 1
            if step_counter["i"] != trace_step or not do_cfg:
                return
            sample = output[0] if isinstance(output, tuple) else output.sample
            if len(_inputs) < 2:
                raise BackendUnavailableError(
                    f"{self.spec.name}: UNet hook did not expose sample/timestep for trace"
                )
            eps = self._capture_epsilon(
                sample,
                _inputs[0],
                _inputs[1],
                context=f"generation trace step {trace_step}",
            )
            if eps.shape[0] != 2:
                raise BackendUnavailableError(
                    f"{self.spec.name}: expected CFG trace batch of 2, got {eps.shape[0]}"
                )
            eps_uncond, eps_cond = eps.chunk(2)  # diffusers CFG batch order: [uncond, cond]
            trace["eps_uncond"] = eps_uncond[0].float().cpu().numpy()
            trace["eps_cond"] = eps_cond[0].float().cpu().numpy()

        def _on_step_end(_pipe: Any, step: int, timestep: Any, kwargs: dict) -> dict:
            if step == trace_step:
                trace["latent"] = kwargs["latents"][0].float().cpu().numpy()
                trace["timestep"] = int(timestep)
            return {}

        handle = self._pipe.unet.register_forward_hook(_unet_hook)
        try:
            generator = torch.Generator(device="cpu").manual_seed(seed)
            with torch.no_grad():
                out = self._pipe(
                    prompt=prompt,
                    guidance_scale=float(w),
                    num_inference_steps=num_steps,
                    height=height,
                    width=width,
                    generator=generator,
                    output_type="np",
                    callback_on_step_end=_on_step_end,
                )
        finally:
            handle.remove()
        img = as_finite_float32("generated_image", out.images[0])
        missing = {"latent", "timestep"} - set(trace)
        if do_cfg:
            missing |= {"eps_uncond", "eps_cond"} - set(trace)
        if missing:
            raise BackendUnavailableError(
                f"{self.spec.name}: generation trace missing required field(s) {sorted(missing)} "
                f"for backend '{self.backend_key}'"
            )
        return img, trace

    def invert(
        self,
        image: np.ndarray,
        prompt: str,
        *,
        num_steps: int = 50,
        seed: int = 0,
    ) -> InversionResult:
        # Convenience wrapper with default rho/E_tex knobs; the main-grid driver (sweep.run_p0)
        # instead calls dual_branch_residual() and channel.features.build_inversion_result() with
        # the p0.yaml knobs.
        from cfgid.channel.features import build_inversion_result

        dual = self.dual_branch_residual(image, prompt, num_steps=num_steps, seed=seed)
        return build_inversion_result(dual, patch_size=4, floor=1e-3, log_ratio=True, sigma=1.0)


class SD15DDIMBackend(_SDInversionBackend):
    backend_key = "sd_ddim"
    branch_mode = "true_cfg_core"
    _pipeline_import = ("diffusers", "StableDiffusionPipeline")

    def _from_pretrained_kwargs(self) -> dict[str, Any]:
        # SD1.5 ships a safety checker we neither need nor want for forensic features.
        return {"safety_checker": None, "requires_safety_checker": False}
