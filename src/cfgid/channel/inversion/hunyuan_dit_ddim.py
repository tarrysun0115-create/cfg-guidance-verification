"""Hunyuan-DiT (DiT) v-prediction DDIM-inverse backend (HunyuanDiT; Appendix B, Table XV).

Hunyuan-DiT (Tencent) is a multi-resolution diffusion TRANSFORMER conditioned on TWO text encoders
(a bilingual CLIP + an mT5, with 77/256-token budgets), trained with the **v-prediction**
objective. It is a DiT lineage different from PixArt-Sigma (different transformer design, 2D
rotary position embeds, dual text encoders, resolution/style micro-conditioning) and the second
non-SD decoder of the paper; it shares the SDXL VAE (Section IV-A).

rho is defined on EPSILON residuals (so it stays homogeneous with the SD1.5/SDXL epsilon channel and
PixArt). Hunyuan predicts v, so — exactly like ``sd_vpred_ddim`` — the captured per-step residual is
mapped v -> epsilon (``eps = sqrt(alpha_t) v + sqrt(1 - alpha_t) z_t``) while the inverse SCHEDULER
still consumes the raw v (``DDIMInverseScheduler`` configured for ``v_prediction`` does the
v->x0/eps conversion internally for the trajectory). DDPM scheduler -> DDIMScheduler swap mirrors
pixart/sd; the loader asserts ``prediction_type == 'v_prediction'`` (NOT epsilon).

This is a STANDALONE backend (it does NOT subclass the SD UNet backend) because the DiT forward
contract is entirely different from the UNet one — it reuses only the pure, weight-free patch math
from ``base`` and the ``v_to_epsilon`` identity from ``sd_vpred_ddim``. The denoising contract
mirrors diffusers ``HunyuanDiTPipeline.__call__`` exactly:

    out = transformer(z, t_expand,
                      encoder_hidden_states=clip_embeds, text_embedding_mask=clip_mask,
                      encoder_hidden_states_t5=mt5_embeds, text_embedding_mask_t5=mt5_mask,
                      image_meta_size=time_ids, style=style, image_rotary_emb=rotary,
                      return_dict=False)[0]
    v = out.chunk(2, dim=1)[0]   # learned variance: out_channels // 2 == latent_channels

Heavy diffusers/torch imports are lazy; loading raises BackendUnavailableError when weights or
diffusers are missing so the import path stays safe without a download.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

import numpy as np

from cfgid.channel.inversion.base import (
    BackendUnavailableError,
    DualBranchResidual,
    InversionBackend,
    InversionResult,
    infer_patch_grid,
    patchify,
    rho_from_patch_residuals,
    torch_dtype,
)
from cfgid.channel.inversion.sd_vpred_ddim import v_to_epsilon
from cfgid.numerics import as_finite_float32, loads_strict_json

if TYPE_CHECKING:
    import torch

# The base diffusers Hunyuan scaffold ships a HunyuanDiTPipeline model_index.
_ALLOWED_MODEL_INDEX_CLASSES = {"HunyuanDiTPipeline"}
# 1024-MS-class HunyuanDiT-v1.2: 4 latent channels in, 8 out (v + learned variance), patch_size 2.
_EXPECTED_TRANSFORMER_CONFIG = {"in_channels": 4, "patch_size": 2}
# CLIP (index 0) / mT5 (index 1) token budgets, matching HunyuanDiTPipeline.__call__.
_CLIP_MAX_TOKENS = 77
_MT5_MAX_TOKENS = 256


class HunyuanDiTDDIMBackend(InversionBackend):
    """Dual-branch v-prediction DDIM-inverse backend for Hunyuan-DiT (DiT + CLIP + mT5 + SD-VAE)."""

    backend_key = "hunyuan_dit_ddim"
    branch_mode = "true_cfg_core"

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
        self._validate_local_scaffold()
        try:
            from diffusers import DDIMInverseScheduler, DDIMScheduler, HunyuanDiTPipeline
        except ImportError as exc:
            raise BackendUnavailableError(f"diffusers unavailable: {exc}") from exc

        pipe = HunyuanDiTPipeline.from_pretrained(
            str(self.spec.resolved_local_path()),
            torch_dtype=torch_dtype(self.spec.dtype),
            local_files_only=True,
            use_safetensors=True,
            # The official v1.2 snapshot ships a StableDiffusionSafetyChecker; left enabled it would
            # blacken "flagged" generations -> corrupt generate_image + the oracle trace. Disabled.
            safety_checker=None,
            requires_safety_checker=False,
        )
        # Hunyuan ships a DDPMScheduler (v_prediction); swap to DDIM for a deterministic invertible
        # trajectory, preserving prediction_type. rho is epsilon-homogeneous (v -> epsilon capture).
        # Force timestep_spacing='leading' on BOTH schedulers: DDIMInverseScheduler.set_timesteps
        # rejects 'linspace', and 'leading' matches the SD/PixArt DDIM grids (cross-decoder + oracle
        # alignment). Force clip_sample=False: a checkpoint config carrying clip_sample=True would
        # clamp the inversion x0 to [-1,1] and silently corrupt rho. from_config drops source keys.
        pipe.scheduler = DDIMScheduler.from_config(
            pipe.scheduler.config, timestep_spacing="leading", clip_sample=False
        )
        pipe = pipe.to(self.device)
        pred_type = pipe.scheduler.config.get("prediction_type", "epsilon")
        if pred_type != "v_prediction":
            raise BackendUnavailableError(
                f"{self.spec.name}: scheduler prediction_type='{pred_type}', expected "
                f"'v_prediction' for backend '{self.backend_key}'"
            )
        pipe.inverse_scheduler = DDIMInverseScheduler.from_config(
            pipe.scheduler.config, timestep_spacing="leading", clip_sample=False
        )
        self._pipe = pipe
        if self._latent_channels() != self.spec.vae_channels:
            raise BackendUnavailableError(
                f"{self.spec.name}: VAE latent_channels={self._latent_channels()}, "
                f"expected models.yaml vae_channels={self.spec.vae_channels}"
            )
        self._loaded = True

    # ----------------------------------------------------------------- model glue

    def _read_local_json_object(self, rel_path: str) -> dict[str, object]:
        path = self.spec.resolved_local_path() / rel_path
        try:
            raw = loads_strict_json(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise BackendUnavailableError(
                f"{self.spec.name}: cannot read {rel_path}: {exc}"
            ) from exc
        if not isinstance(raw, dict):
            raise BackendUnavailableError(f"{self.spec.name}: {rel_path} must be a JSON object")
        return raw

    def _validate_local_scaffold(self) -> None:
        """Fail before loading weights if the local Hunyuan snapshot is not a v1.2 DiT pipeline."""
        model_index = self._read_local_json_object("model_index.json")
        class_name = model_index.get("_class_name")
        if class_name not in _ALLOWED_MODEL_INDEX_CLASSES:
            raise BackendUnavailableError(
                f"{self.spec.name}: model_index.json _class_name={class_name!r}; expected one of "
                f"{sorted(_ALLOWED_MODEL_INDEX_CLASSES)} for HunyuanDiTPipeline loading"
            )
        transformer_cfg = self._read_local_json_object("transformer/config.json")
        for key, expected in _EXPECTED_TRANSFORMER_CONFIG.items():
            got = transformer_cfg.get(key)
            if got != expected:
                raise BackendUnavailableError(
                    f"{self.spec.name}: transformer/config.json {key}={got!r}, expected "
                    f"{expected!r} for the Hunyuan-DiT v1.2 true-CFG core backend"
                )

    def _latent_channels(self) -> int:
        return int(self._pipe.vae.config.latent_channels)

    def _build_image_conditioning(
        self, height: int, width: int, dtype: torch.dtype
    ) -> dict[str, Any]:
        """The prompt-INDEPENDENT micro-conditioning HunyuanDiT's forward needs, mirroring the
        pipeline: 2D rotary position embedding, resolution time-ids, and the style index.

        Computed once per image size (identical across denoising steps and across the cond/uncond
        branches), so it threads unchanged through the inversion loop. Resolution-matched: original
        and target size both = (height, width), no crop -- so generation and inversion share one
        conditioning when generate_image passes the same sizes.
        """
        import torch
        from diffusers.models.embeddings import get_2d_rotary_pos_embed
        from diffusers.pipelines.hunyuandit.pipeline_hunyuandit import (
            get_resize_crop_region_for_grid,
        )

        transformer = self._pipe.transformer
        patch_size = int(transformer.config.patch_size)
        # The latent is height/8, then patchified by patch_size; an off-(8*patch_size) resolution
        # would make generation (which rounds) and inversion (raw dims) build mismatched grids. Fail
        # loud rather than silently mis-condition. (8*patch_size = 16 for the v1.2 patch_size=2.)
        step = 8 * patch_size
        if height % step != 0 or width % step != 0:
            raise ValueError(
                f"{self.spec.name}: height/width must be multiples of {step} (8*patch_size); "
                f"got {height}x{width}"
            )
        grid_height = height // 8 // patch_size
        grid_width = width // 8 // patch_size
        base_size = 512 // 8 // patch_size
        grid_crops_coords = get_resize_crop_region_for_grid((grid_height, grid_width), base_size)
        image_rotary_emb = get_2d_rotary_pos_embed(
            transformer.inner_dim // transformer.num_heads,
            grid_crops_coords,
            (grid_height, grid_width),
            device=self.device,
            output_type="pt",
        )
        # time-ids = [orig_h, orig_w, target_h, target_w, crop_top, crop_left]; resolution-matched.
        time_ids = torch.tensor(
            [[height, width, height, width, 0, 0]], dtype=dtype, device=self.device
        )
        style = torch.tensor([0], device=self.device)
        return {"image_rotary_emb": image_rotary_emb, "image_meta_size": time_ids, "style": style}

    def _encode_branch(self, prompt: str) -> dict[str, torch.Tensor]:
        """CLIP (index 0) + mT5 (index 1) embeds + masks for ONE branch (cond prompt OR empty).

        do_classifier_free_guidance=False so each branch is its own non-CFG encode (the dual-branch
        rho inverts cond and uncond as separate single-image forwards); negatives are discarded.
        """
        dtype = self._pipe.transformer.dtype
        clip_embeds, _, clip_mask, _ = self._pipe.encode_prompt(
            prompt=prompt,
            device=self.device,
            dtype=dtype,
            num_images_per_prompt=1,
            do_classifier_free_guidance=False,
            max_sequence_length=_CLIP_MAX_TOKENS,
            text_encoder_index=0,
        )
        mt5_embeds, _, mt5_mask, _ = self._pipe.encode_prompt(
            prompt=prompt,
            device=self.device,
            dtype=dtype,
            num_images_per_prompt=1,
            do_classifier_free_guidance=False,
            max_sequence_length=_MT5_MAX_TOKENS,
            text_encoder_index=1,
        )
        return {
            "encoder_hidden_states": clip_embeds,
            "text_embedding_mask": clip_mask,
            "encoder_hidden_states_t5": mt5_embeds,
            "text_embedding_mask_t5": mt5_mask,
        }

    def _v_output(
        self, z: torch.Tensor, t: torch.Tensor, branch: dict, img_cond: dict
    ) -> torch.Tensor:
        """One transformer forward -> the v-prediction (first chunk of the learned-variance output).

        Mirrors HunyuanDiTPipeline.__call__'s transformer call exactly; finite-guarded; under
        no_grad (no backprop through the inversion chain). Returns the raw v (NOT epsilon) -- the
        inverse scheduler (v_prediction) consumes v; the rho capture maps v -> epsilon separately.
        """
        import torch

        # Cast the timestep to the latent (transformer) float dtype, mirroring the pipeline's
        # `t_expand = ...to(dtype=latent_model_input.dtype)`. The inverse scheduler's timesteps are
        # INT64; HunyuanDiT2DModel.forward uses hidden_dtype=timestep.dtype, so an int64 timestep
        # would truncate the float sinusoidal time/size embeddings to 0 and break torch.cat(fp16,
        # int64, fp16) inside the size/style embedder -> a real-weights-only crash. (_alpha_t keeps
        # the ORIGINAL int64 t for alphas_cumprod indexing -- do not feed it the float-cast value.)
        timestep = t.expand(z.shape[0]) if t.ndim else t[None].expand(z.shape[0])
        timestep = timestep.to(dtype=z.dtype)
        with torch.no_grad():
            out = self._pipe.transformer(
                z,
                timestep,
                encoder_hidden_states=branch["encoder_hidden_states"],
                text_embedding_mask=branch["text_embedding_mask"],
                encoder_hidden_states_t5=branch["encoder_hidden_states_t5"],
                text_embedding_mask_t5=branch["text_embedding_mask_t5"],
                image_meta_size=img_cond["image_meta_size"],
                style=img_cond["style"],
                image_rotary_emb=img_cond["image_rotary_emb"],
                return_dict=False,
            )[0]
        if not torch.all(torch.isfinite(out)):
            raise BackendUnavailableError(
                f"{self.spec.name}: non-finite transformer output (inversion)"
            )
        latent_channels = self._latent_channels()
        if out.shape[1] == 2 * latent_channels:
            v = out.chunk(2, dim=1)[0]  # learned variance -> take the v-prediction half
        elif out.shape[1] == latent_channels:
            v = out
        else:
            raise BackendUnavailableError(
                f"{self.spec.name}: transformer output channels={out.shape[1]}; expected "
                f"{latent_channels} (v) or {2 * latent_channels} (v+variance)"
            )
        if not torch.all(torch.isfinite(v)):
            raise BackendUnavailableError(f"{self.spec.name}: non-finite v-prediction (inversion)")
        return v

    def _alpha_t(self, t: torch.Tensor) -> torch.Tensor:
        import torch

        acp = self._pipe.inverse_scheduler.alphas_cumprod.to(
            device=self.device, dtype=torch.float32
        )
        return acp[int(t)]

    @staticmethod
    def _capture_index(num_steps: int, capture_step_frac: float) -> int:
        if num_steps <= 0:
            raise ValueError(f"num_steps must be positive, got {num_steps}")
        if not 0.0 <= capture_step_frac <= 1.0:
            raise ValueError(f"capture_step_frac must be in [0, 1], got {capture_step_frac}")
        return max(0, min(num_steps - 1, int(capture_step_frac * num_steps)))

    def _ddim_invert_capture(
        self, latent: torch.Tensor, branch: dict, img_cond: dict, num_steps: int, capture_idx: int
    ) -> np.ndarray:
        import torch

        # One single-conditioned DDIM inversion (Eq. (1)); the scheduler step consumes raw v.
        scheduler = self._pipe.inverse_scheduler
        scheduler.set_timesteps(num_steps, device=self.device)
        z = latent.clone()
        captured: np.ndarray | None = None
        for i, t in enumerate(scheduler.timesteps):
            v = self._v_output(z, t, branch, img_cond)
            if i == capture_idx:
                # rho residual is EPSILON-homogeneous: convert the captured v at this latent/step.
                eps = v_to_epsilon(v.float(), z.float(), self._alpha_t(t))
                if not torch.all(torch.isfinite(eps)):
                    raise BackendUnavailableError(
                        f"{self.spec.name}: non-finite epsilon residual (capture step {i})"
                    )
                captured = eps[0].cpu().numpy()
            # The inverse scheduler (v_prediction) consumes the RAW v and does v->x0/eps internally.
            z = scheduler.step(v, t, z).prev_sample
            if not torch.all(torch.isfinite(z)):
                raise BackendUnavailableError(
                    f"{self.spec.name}: inversion diverged (non-finite latent) at step {i}"
                )
        if captured is None:
            raise RuntimeError(f"{self.spec.name}: capture step {capture_idx} never reached")
        return captured

    def _encode_image_to_latent(self, image: np.ndarray) -> torch.Tensor:
        import torch

        img = as_finite_float32("image", image)
        if img.ndim != 3 or img.shape[2] != 3:
            raise ValueError(f"image must be (H, W, 3) in [0, 1]; got {img.shape}")
        if float(img.min()) < 0.0 or float(img.max()) > 1.0:
            raise ValueError("image must be in [0, 1]")
        vae = self._pipe.vae
        original_vae_dtype = vae.dtype
        transformer_dtype = self._pipe.transformer.dtype
        x = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)  # (1, 3, H, W)
        # Upcast a fp16 force_upcast VAE for the encode (Hunyuan VAE is force_upcast=False, so this
        # is a no-op there, but the guard keeps the backend correct if a fp16 force_upcast VAE is
        # ever substituted -- matching sd_ddim/pixart).
        needs_upcast = vae.dtype == torch.float16 and bool(
            getattr(vae.config, "force_upcast", False)
        )
        try:
            if needs_upcast:
                vae.to(torch.float32)
            x = (x * 2.0 - 1.0).to(device=self.device, dtype=vae.dtype)
            with torch.no_grad():
                latent = vae.encode(x).latent_dist.mean * vae.config.scaling_factor
        finally:
            if needs_upcast:
                vae.to(original_vae_dtype)
        latent = latent.to(dtype=transformer_dtype)
        if not torch.all(torch.isfinite(latent)):
            raise BackendUnavailableError(
                f"{self.spec.name}: VAE encode produced non-finite latent"
            )
        return latent

    @contextmanager
    def _force_upcast_vae_for_decode(self) -> Iterator[None]:
        """Upcast a fp16 force_upcast VAE around the pipeline decode, then restore (no-op for the
        Hunyuan VAE, which is force_upcast=False; kept for parity + substitution safety)."""
        import torch

        vae = self._pipe.vae
        original_vae_dtype = getattr(vae, "dtype", None)
        needs_upcast = original_vae_dtype == torch.float16 and bool(
            getattr(vae.config, "force_upcast", False)
        )
        try:
            if needs_upcast:
                vae.to(torch.float32)
            yield
        finally:
            if needs_upcast:
                vae.to(original_vae_dtype)

    # ----------------------------------------------------------------- public API

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

        torch.manual_seed(seed)
        latent = self._encode_image_to_latent(image)
        height = int(image.shape[0])
        width = int(image.shape[1])
        img_cond = self._build_image_conditioning(height, width, self._pipe.transformer.dtype)
        capture_idx = self._capture_index(num_steps, capture_step_frac)
        cond = self._encode_branch(prompt)
        uncond = self._encode_branch("")
        r_cond = self._ddim_invert_capture(latent, cond, img_cond, num_steps, capture_idx)
        r_uncond = self._ddim_invert_capture(latent, uncond, img_cond, num_steps, capture_idx)
        return DualBranchResidual(
            r_cond=r_cond,
            r_uncond=r_uncond,
            latent=latent[0].float().cpu().numpy(),
            capture_step=capture_idx,
            meta={"num_steps": num_steps},
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

        generator = torch.Generator(device="cpu").manual_seed(seed)
        with self._force_upcast_vae_for_decode():
            with torch.no_grad():
                out = self._pipe(
                    prompt=prompt,
                    guidance_scale=float(w),
                    num_inference_steps=num_steps,
                    height=height,
                    width=width,
                    generator=generator,
                    original_size=(height, width),
                    target_size=(height, width),
                    crops_coords_top_left=(0, 0),
                    use_resolution_binning=False,
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
        self.require_loaded()
        import torch

        if not 0 <= trace_step < num_steps:
            raise ValueError(f"trace_step must be in [0, {num_steps}), got {trace_step}")
        trace: dict[str, np.ndarray | int | float] = {"gen_step": trace_step, "w": float(w)}
        do_cfg = float(w) > 1.0
        latent_channels = self._latent_channels()
        step_counter = {"i": -1}

        def _transformer_hook(_module: Any, _inputs: tuple, output: Any) -> None:
            step_counter["i"] += 1
            if step_counter["i"] != trace_step or not do_cfg:
                return
            sample = output[0] if isinstance(output, tuple) else output.sample
            # The transformer is called positionally as transformer(latent_model_input, t_expand,
            # ...), so _inputs[0] is the pre-step latent and _inputs[1] the timestep at THIS step --
            # both needed to map v -> epsilon at the right alpha_t (the callback's post-step latent
            # cannot supply the pre-step z the v was produced at).
            if len(_inputs) < 2:
                raise BackendUnavailableError(
                    f"{self.spec.name}: transformer hook did not expose latent+timestep for trace"
                )
            latent_input, timestep_input = _inputs[0], _inputs[1]
            if not torch.all(torch.isfinite(sample)):
                raise BackendUnavailableError(
                    f"{self.spec.name}: non-finite transformer output (generation trace)"
                )
            v = sample.chunk(2, dim=1)[0] if sample.shape[1] == 2 * latent_channels else sample
            if v.shape[0] != 2:
                raise BackendUnavailableError(
                    f"{self.spec.name}: expected CFG trace batch of 2, got {v.shape[0]}"
                )
            v_uncond, v_cond = v.chunk(2)  # CFG batch order: [uncond, cond]
            z_uncond, z_cond = latent_input.float().chunk(2)
            alpha_t = self._alpha_t(timestep_input.reshape(-1)[0])
            trace["eps_uncond"] = v_to_epsilon(v_uncond.float(), z_uncond, alpha_t)[0].cpu().numpy()
            trace["eps_cond"] = v_to_epsilon(v_cond.float(), z_cond, alpha_t)[0].cpu().numpy()

        def _on_step_end(_pipe: Any, step: int, timestep: Any, kwargs: dict) -> dict:
            if step == trace_step:
                trace["latent"] = kwargs["latents"][0].float().cpu().numpy()
                trace["timestep"] = int(timestep)
            return {}

        handle = self._pipe.transformer.register_forward_hook(_transformer_hook)
        try:
            generator = torch.Generator(device="cpu").manual_seed(seed)
            with self._force_upcast_vae_for_decode():
                with torch.no_grad():
                    out = self._pipe(
                        prompt=prompt,
                        guidance_scale=float(w),
                        num_inference_steps=num_steps,
                        height=height,
                        width=width,
                        generator=generator,
                        original_size=(height, width),
                        target_size=(height, width),
                        crops_coords_top_left=(0, 0),
                        use_resolution_binning=False,
                        output_type="np",
                        callback_on_step_end=_on_step_end,
                        callback_on_step_end_tensor_inputs=["latents"],
                    )
        finally:
            handle.remove()
        trace.pop("timestep_t", None)  # internal scratch, not persisted
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
        from cfgid.channel.features import build_inversion_result

        dual = self.dual_branch_residual(image, prompt, num_steps=num_steps, seed=seed)
        return build_inversion_result(dual, patch_size=4, floor=1e-3, log_ratio=True, sigma=1.0)

    # ----------------------------------------------------------------- smoke (weight-free shapes)

    def _patch_residual_smoke(self, r_cond: np.ndarray, r_uncond: np.ndarray) -> np.ndarray:
        grid = infer_patch_grid(r_cond.shape[1], r_cond.shape[2], 4)
        return rho_from_patch_residuals(patchify(r_cond, grid), patchify(r_uncond, grid))
