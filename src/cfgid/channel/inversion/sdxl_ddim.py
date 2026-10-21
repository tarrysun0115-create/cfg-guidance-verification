"""SDXL epsilon-prediction DDIM-inverse backend (Section III-B; Appendix B, Table XV).

Reuses the shared SD-family loader and DDIM inversion of ``sd_ddim`` (Eq. (1)); only the branch
encoding differs: SDXL's UNet needs ``added_cond_kwargs`` (pooled text embeds + add_time_ids)
and both text encoders are used, with the second prompt defaulting to the first. add_time_ids
are built explicitly (original = target = 8x the latent size, no crop) to avoid depending on a
private pipeline helper. The main grid runs SDXL at 512^2, below its native 1024 (Section IV-A).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from cfgid.channel.inversion.sd_ddim import _SDInversionBackend

if TYPE_CHECKING:
    import torch


class SDXLDDIMBackend(_SDInversionBackend):
    backend_key = "sdxl_ddim"
    branch_mode = "true_cfg_core"
    _pipeline_import = ("diffusers", "StableDiffusionXLPipeline")

    def _encode_branch(
        self, prompt: str, latent: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor] | None]:
        import torch

        # no_grad: the text-encoder forward is inference-only (frozen backbone, no backprop through
        # the inversion chain); without it encode_prompt builds an autograd graph on the encoder.
        with torch.no_grad():
            prompt_embeds, _neg, pooled, _neg_pooled = self._pipe.encode_prompt(
                prompt=prompt,
                prompt_2=None,
                device=self.device,
                num_images_per_prompt=1,
                do_classifier_free_guidance=False,
            )
        # add_time_ids = [orig_h, orig_w, crop_top, crop_left, target_h, target_w]; latent is /8.
        h = int(latent.shape[-2]) * 8
        w = int(latent.shape[-1]) * 8
        add_time_ids = torch.tensor(
            [[h, w, 0, 0, h, w]], device=self.device, dtype=prompt_embeds.dtype
        )
        added = {"text_embeds": pooled, "time_ids": add_time_ids}
        return prompt_embeds, added
