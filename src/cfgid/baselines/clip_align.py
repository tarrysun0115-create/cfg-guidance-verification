"""CLIP-text-alignment negative-control baseline (needs a model; pure scoring).

Raising CFG also raises prompt adherence, so an alignment-only detector can separate H0/H1
without seeing any rho-channel trace. The gate requires rho to beat this baseline after
texture/semantic matching, else the verdict is inconclusive. This module only produces the
per-(image, prompt) cosine-similarity scores; AUC lives in ``cfgid.detector.empirical`` and
is computed by the caller. Heavy imports (transformers, torch, PIL) are lazy so the module
imports without weights; a missing checkpoint raises :class:`ClipUnavailableError` and the
driver reports the baseline as N/A (configs/p0.yaml ``evidence.clip_baseline_enabled``; the
paper's decision memo records it as disabled). The same scores are the CLIP prompt-adherence
detector of the Section IV-E1 comparisons (Table VIII), computed by
``scripts/analysis/neutral_pool_allcell.py`` and ``clip_vs_rho.py``.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from cfgid.numerics import as_finite_float32

if TYPE_CHECKING:
    import torch
    from PIL import Image


class ClipUnavailableError(RuntimeError):
    """Local CLIP checkpoint (or the transformers stack) is missing/unloadable."""


# transformers auto-detects the weight format; our checkpoint may ship pytorch_model.bin only,
# so we must NOT force safetensors at load time.
_WEIGHT_FILES = ("model.safetensors", "pytorch_model.bin")


def clip_available(clip_dir: Path) -> bool:
    """True iff ``config.json`` and at least one known weight file exist under ``clip_dir``."""
    clip_dir = Path(clip_dir)
    if not (clip_dir / "config.json").is_file():
        return False
    return any((clip_dir / name).is_file() for name in _WEIGHT_FILES)


def _to_pil(image: np.ndarray | Path) -> Image.Image:
    """One image -> RGB PIL: open paths as-is; convert float [0, 1] (H, W, 3) arrays to uint8."""
    from PIL import Image

    if isinstance(image, (str, Path)):
        return Image.open(image).convert("RGB")
    img = as_finite_float32("image", image)
    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError(f"image must be (H, W, 3) in [0, 1]; got shape {img.shape}")
    if float(img.min()) < 0.0 or float(img.max()) > 1.0:
        raise ValueError("image must be in [0, 1]")
    return Image.fromarray(np.round(img * 255.0).astype(np.uint8))


def clip_alignment_scores(
    images: list[np.ndarray] | list[Path],
    prompts: list[str],
    *,
    clip_dir: Path,
    device: torch.device,
    batch_size: int = 8,
) -> np.ndarray:
    """Per-pair CLIP image-text cosine similarity, float32 shape (N,).

    ``images`` are (H, W, 3) float arrays in [0, 1] or paths PIL can open; ``prompts`` pair up
    one-to-one. The checkpoint at ``clip_dir`` is loaded with ``local_files_only=True`` (no
    network); scores are NaN-guarded so non-finite embeddings can never propagate silently.
    """
    if len(images) != len(prompts):
        raise ValueError(f"got {len(images)} images but {len(prompts)} prompts")
    if not images:
        raise ValueError("images is empty")
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")
    if not clip_available(clip_dir):
        raise ClipUnavailableError(
            f"CLIP checkpoint missing at {clip_dir} "
            f"(need config.json + one of {list(_WEIGHT_FILES)})"
        )
    try:
        import torch
        from transformers import CLIPModel, CLIPProcessor
    except ImportError as exc:
        raise ClipUnavailableError(f"transformers/torch unavailable: {exc}") from exc

    # Typed as Any: transformers routes processor kwargs (padding/truncation/return_tensors)
    # through nested TypedDicts at runtime, which pyright cannot resolve statically.
    try:
        model: Any = CLIPModel.from_pretrained(str(clip_dir), local_files_only=True)
        # use_fast=True -> the torchvision fast image processor (default in transformers >=4.52).
        # The slow PIL processor does per-image CPU resize/normalize and starves the GPU forward
        # (GPU idle, CPU pegged); the fast path is the documented fix for that warning.
        processor: Any = CLIPProcessor.from_pretrained(
            str(clip_dir), local_files_only=True, use_fast=True
        )
    except (OSError, ValueError) as exc:  # incomplete/corrupt checkpoint directory
        raise ClipUnavailableError(f"failed to load CLIP from {clip_dir}: {exc}") from exc
    model = model.to(device)
    model.eval()

    sims: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(images), batch_size):
            pil_batch = [_to_pil(img) for img in images[start : start + batch_size]]
            text_batch = list(prompts[start : start + batch_size])
            inputs = processor(
                text=text_batch,
                images=pil_batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
            )
            inputs = {key: value.to(device) for key, value in inputs.items()}
            out = model(**inputs)
            # CLIPModel.forward returns normalized embeds; re-normalize anyway (idempotent)
            # so the score is a cosine similarity by construction, not by API fine print.
            img_emb = torch.nn.functional.normalize(out.image_embeds, dim=-1)
            txt_emb = torch.nn.functional.normalize(out.text_embeds, dim=-1)
            sims.append((img_emb * txt_emb).sum(dim=-1).float().cpu().numpy())
    return as_finite_float32("clip_alignment_scores", np.concatenate(sims))
