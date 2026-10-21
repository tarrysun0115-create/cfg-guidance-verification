"""PixArt-Sigma (DiT) epsilon-prediction DDIM-inverse backend (Appendix B, Table XV).

PixArt-Sigma is a transformer (DiT) denoiser conditioned on T5 text embeddings + a standard
4-channel SD-VAE (the SDXL VAE), trained with the epsilon (+ learned-sigma) objective -- so it is
DDIM-invertible exactly like the SD UNet family (Eq. (1)), and rho stays epsilon-homogeneous. It
is a different architecture (transformer vs UNet, T5 vs CLIP) and serves as the paper's
architecture-shift evaluation point (Section IV-A). Text conditioning uses the attention-masked
T5 encoder with a 300-token budget; the null branch encodes the empty string.

This is a STANDALONE backend (it does NOT subclass the SD UNet backend) so the working SD/SDXL/SD21
paths carry zero regression risk; it reuses only the pure, weight-free patch math from ``base``.
The denoising contract mirrors diffusers ``PixArtSigmaPipeline.__call__`` exactly:

    out = transformer(z, encoder_hidden_states=embeds, encoder_attention_mask=mask,
                      timestep=t_expanded,
                      added_cond_kwargs={"resolution": None, "aspect_ratio": None},
                      return_dict=False)[0]
    eps = out.chunk(2, dim=1)[0]   # learned sigma: out_channels // 2 == latent_channels

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
from cfgid.numerics import as_finite_float32, loads_strict_json

if TYPE_CHECKING:
    import torch

# Micro-conditioning for PixArt-Sigma: resolution/aspect_ratio embeddings are unused (None), exactly
# as diffusers' PixArtSigmaPipeline passes them.
_ADDED_COND_KWARGS = {"resolution": None, "aspect_ratio": None}
_ALLOWED_MODEL_INDEX_CLASSES = {"PixArtSigmaPipeline", "PixArtAlphaPipeline"}
_EXPECTED_TRANSFORMER_CONFIG = {"in_channels": 4, "out_channels": 8, "sample_size": 64}


class PixArtDDIMBackend(InversionBackend):
    """Dual-branch epsilon DDIM-inverse backend for PixArt-Sigma (DiT + T5 + SD-VAE)."""

    backend_key = "pixart_ddim"
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
            from diffusers import DDIMInverseScheduler, DDIMScheduler, PixArtSigmaPipeline
        except ImportError as exc:
            raise BackendUnavailableError(f"diffusers unavailable: {exc}") from exc

        pipe = PixArtSigmaPipeline.from_pretrained(
            str(self.spec.resolved_local_path()),
            torch_dtype=torch_dtype(self.spec.dtype),
            local_files_only=True,
            use_safetensors=True,
        )
        # PixArt-Sigma ships a DPM-Solver scheduler whose config has timestep_spacing='linspace',
        # which DDIMInverseScheduler.set_timesteps rejects (only 'leading'/'trailing'). Force
        # 'leading' on BOTH schedulers -- it matches the SD/SDXL DDIM default (so the generation and
        # inversion timestep grids stay consistent with the other core decoders + the oracle-trace
        # alignment), and is a supported inverse spacing. from_config also drops the DPM-only keys.
        pipe.scheduler = DDIMScheduler.from_config(
            pipe.scheduler.config, timestep_spacing="leading"
        )
        pipe = pipe.to(self.device)
        # The rho channel assumes epsilon-prediction; fail loudly on any other pred-type rather than
        # silently producing a wrong rho.
        pred_type = pipe.scheduler.config.get("prediction_type", "epsilon")
        if pred_type != "epsilon":
            raise BackendUnavailableError(
                f"{self.spec.name}: scheduler prediction_type='{pred_type}', expected 'epsilon' "
                f"for backend '{self.backend_key}'"
            )
        pipe.inverse_scheduler = DDIMInverseScheduler.from_config(
            pipe.scheduler.config, timestep_spacing="leading"
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
        """Fail before loading weights if the local PixArt scaffold cannot be the 512-MS backend.

        The official 512-MS repo is transformer-only. The runnable local snapshot must be assembled
        from the PixArt base components plus the 512-MS transformer. The base component repo
        still ships a PixArtAlphaPipeline model_index; explicit PixArtSigmaPipeline.from_pretrained
        accepts that scaffold, but a missing/other scaffold or a non-512 transformer must fail
        before a GPU job reaches the expensive path.
        """
        model_index = self._read_local_json_object("model_index.json")
        class_name = model_index.get("_class_name")
        if class_name not in _ALLOWED_MODEL_INDEX_CLASSES:
            raise BackendUnavailableError(
                f"{self.spec.name}: model_index.json _class_name={class_name!r}; expected one of "
                f"{sorted(_ALLOWED_MODEL_INDEX_CLASSES)} for explicit PixArtSigmaPipeline loading"
            )
        transformer_cfg = self._read_local_json_object("transformer/config.json")
        for key, expected in _EXPECTED_TRANSFORMER_CONFIG.items():
            got = transformer_cfg.get(key)
            if got != expected:
                raise BackendUnavailableError(
                    f"{self.spec.name}: transformer/config.json {key}={got!r}, "
                    f"expected {expected!r} "
                    "for the PixArt-Sigma 512-MS true-CFG core backend"
                )

    def _latent_channels(self) -> int:
        return int(self._pipe.vae.config.latent_channels)

    def _epsilon_from_transformer_output(
        self, out: torch.Tensor, *, expected_spatial: tuple[int, int], context: str
    ) -> torch.Tensor:
        import torch

        if not torch.all(torch.isfinite(out)):
            raise BackendUnavailableError(
                f"{self.spec.name}: non-finite transformer output ({context})"
            )
        if out.ndim != 4:
            raise BackendUnavailableError(
                f"{self.spec.name}: transformer output for {context} must be BCHW, "
                f"got {tuple(out.shape)}"
            )
        if tuple(out.shape[-2:]) != expected_spatial:
            raise BackendUnavailableError(
                f"{self.spec.name}: transformer output spatial {tuple(out.shape[-2:])} != "
                f"latent spatial {expected_spatial} ({context})"
            )
        latent_channels = self._latent_channels()
        if out.shape[1] == latent_channels:
            eps = out
        elif out.shape[1] == 2 * latent_channels:
            eps = out.chunk(2, dim=1)[0]  # learned sigma -> epsilon
        else:
            raise BackendUnavailableError(
                f"{self.spec.name}: transformer output channels={out.shape[1]} for {context}; "
                f"expected {latent_channels} (epsilon) or {2 * latent_channels} (epsilon+sigma)"
            )
        if not torch.all(torch.isfinite(eps)):
            raise BackendUnavailableError(
                f"{self.spec.name}: non-finite epsilon residual ({context})"
            )
        return eps

    def _encode_branch(self, prompt: str) -> tuple[torch.Tensor, torch.Tensor]:
        """T5 text embeds + attention mask for one branch (cond prompt OR empty).

        ``do_classifier_free_guidance=False`` -> the negative embeds are discarded: each branch is
        its own single (non-CFG) forward, so cond vs uncond is realized by calling this twice, NOT
        by an in-pass CFG combine. ``no_grad``: inference-only (frozen backbone: no backprop through
        the inversion chain).
        """
        import torch

        with torch.no_grad():
            embeds, mask, _neg_embeds, _neg_mask = self._pipe.encode_prompt(
                prompt,
                do_classifier_free_guidance=False,
                device=self.device,
                num_images_per_prompt=1,
                clean_caption=True,
                max_sequence_length=300,
            )
        return embeds, mask

    def _denoise(
        self, z: torch.Tensor, t: torch.Tensor, embeds: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """One transformer forward -> epsilon residual (learned-sigma chunked), finite-guarded.

        Mirrors PixArtSigmaPipeline: timestep broadcast to the batch, encoder_attention_mask passed,
        resolution/aspect_ratio None, and the learned-sigma split (out_channels // 2 == latent
        channels) taking the FIRST chunk as epsilon.
        """
        import torch

        timestep = t.expand(z.shape[0]) if t.ndim else t[None].expand(z.shape[0])
        with torch.no_grad():
            out = self._pipe.transformer(
                z,
                encoder_hidden_states=embeds,
                encoder_attention_mask=mask,
                timestep=timestep,
                added_cond_kwargs=_ADDED_COND_KWARGS,
                return_dict=False,
            )[0]
        return self._epsilon_from_transformer_output(
            out,
            expected_spatial=(int(z.shape[-2]), int(z.shape[-1])),
            context="inversion/generation branch",
        )

    def _encode_image_to_latent(self, image: np.ndarray) -> torch.Tensor:
        import torch

        img = as_finite_float32("image", image)
        if img.ndim != 3 or img.shape[2] != 3:
            raise ValueError(f"image must be (H, W, 3) in [0, 1]; got {img.shape}")
        if float(img.min()) < 0.0 or float(img.max()) > 1.0:
            raise ValueError("image must be in [0, 1]")
        vae = self._pipe.vae
        original_vae_dtype = vae.dtype
        transformer_dtype = next(self._pipe.transformer.parameters()).dtype
        x = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)  # (1, 3, H, W)
        # PixArt-Sigma reuses the SDXL VAE, which is numerically unstable in fp16 and ships
        # force_upcast=True; mirror sd_ddim/flux_flow and upcast to fp32 for the encode when the
        # model asks (a fp16 encode here yields non-finite or silently-corrupted inversion latents).
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
        """Upcast fp16 force_upcast VAEs around PixArt pipeline decode, then restore.

        diffusers' SDXL pipeline upcasts force_upcast VAEs before decode, but PixArtSigmaPipeline
        0.38.0 decodes with ``latents.to(self.vae.dtype)`` and has no equivalent guard. PixArt-Sigma
        reuses the SDXL VAE, so generation must apply the same protection as inversion encode.
        """
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

    @staticmethod
    def _capture_index(num_steps: int, capture_step_frac: float) -> int:
        if num_steps <= 0:
            raise ValueError(f"num_steps must be positive, got {num_steps}")
        if not 0.0 <= capture_step_frac <= 1.0:
            raise ValueError(f"capture_step_frac must be in [0, 1], got {capture_step_frac}")
        return max(0, min(num_steps - 1, int(capture_step_frac * num_steps)))

    def _ddim_invert_capture(
        self, latent: torch.Tensor, prompt: str, num_steps: int, capture_idx: int
    ) -> np.ndarray:
        import torch

        # One single-conditioned DDIM inversion (Eq. (1)); same loop as sd_ddim with the DiT.
        scheduler = self._pipe.inverse_scheduler
        scheduler.set_timesteps(num_steps, device=self.device)
        z = latent.clone()
        embeds, mask = self._encode_branch(prompt)
        captured: np.ndarray | None = None
        for i, t in enumerate(scheduler.timesteps):
            with torch.no_grad():
                eps = self._denoise(z, t, embeds, mask)  # already epsilon (learned-sigma chunked)
            if i == capture_idx:
                captured = eps[0].float().cpu().numpy()
            # The inverse scheduler consumes the epsilon residual (DDIMInverseScheduler configured
            # for prediction_type='epsilon'); for PixArt the chunked epsilon is the scheduler input.
            z = scheduler.step(eps, t, z).prev_sample
            if not torch.all(torch.isfinite(z)):
                raise BackendUnavailableError(
                    f"{self.spec.name}: inversion diverged (non-finite latent) at step {i}"
                )
        if captured is None:
            raise RuntimeError(f"capture step {capture_idx} not reached in {num_steps} steps")
        return captured

    # ----------------------------------------------------------------- contract

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
                    use_resolution_binning=False,  # keep requested h/w exact (no aspect-ratio bin)
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

        Captures the post-step latent and (when ``w > 1``: CFG batch [uncond, cond]) the per-branch
        epsilon via a transformer forward hook. The 8-channel learned-sigma output is split to
        epsilon (chunk over channels) before the [uncond, cond] batch split.
        """
        self.require_loaded()
        import torch

        if not 0 <= trace_step < num_steps:
            raise ValueError(f"trace_step must be in [0, {num_steps}), got {trace_step}")
        trace: dict[str, np.ndarray | int | float] = {"gen_step": trace_step, "w": float(w)}
        do_cfg = float(w) > 1.0
        step_counter = {"i": -1}

        def _transformer_hook(_module: Any, _inputs: tuple, output: Any) -> None:
            step_counter["i"] += 1
            if step_counter["i"] != trace_step or not do_cfg:
                return
            sample = output[0] if isinstance(output, tuple) else output.sample
            if not _inputs:
                raise BackendUnavailableError(
                    f"{self.spec.name}: transformer hook did not expose latent input for trace"
                )
            latent_input = _inputs[0]
            eps = self._epsilon_from_transformer_output(
                sample,
                expected_spatial=(int(latent_input.shape[-2]), int(latent_input.shape[-1])),
                context="generation trace",
            )
            if eps.shape[0] != 2:
                raise BackendUnavailableError(
                    f"{self.spec.name}: expected CFG trace batch of 2, got {eps.shape[0]}"
                )
            eps_uncond, eps_cond = eps.chunk(2)  # CFG batch order: [uncond, cond]
            trace["eps_uncond"] = eps_uncond[0].float().cpu().numpy()
            trace["eps_cond"] = eps_cond[0].float().cpu().numpy()

        def _callback(step: int, timestep: Any, latents: torch.Tensor) -> None:
            if step == trace_step:
                trace["latent"] = latents[0].float().cpu().numpy()
                if torch.is_tensor(timestep):
                    trace["timestep"] = int(timestep.detach().cpu().reshape(-1)[0].item())
                else:
                    trace["timestep"] = int(timestep)

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
                        use_resolution_binning=False,
                        output_type="np",
                        callback=_callback,
                        callback_steps=1,
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

    # ----------------------------------------------------------------- smoke (weight-free shapes)

    def _patch_residual_smoke(self, r_cond: np.ndarray, r_uncond: np.ndarray) -> np.ndarray:
        """Pure patch-math helper exercised by the unit test (no weights): rho on a 4-patch grid."""
        grid = infer_patch_grid(r_cond.shape[1], r_cond.shape[2], 4)
        return rho_from_patch_residuals(patchify(r_cond, grid), patchify(r_uncond, grid))
