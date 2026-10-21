"""Paired generation (H0 w=1 vs H1 w>1, identical seed) and the JPEG compression operator.

H0/H1 differ ONLY in guidance strength (same prompt/seed/decoder/compression chain), so a detector
cannot latch onto source or compression differences (Appendix B: H0 and all H1 images of a prompt
share one seed). ``generate_pair`` is a convenience wrapper; the main-grid driver
``cfgid.sweep.run_p0`` calls the backends' ``generate_image`` directly. ``jpeg_compress``
round-trips through a real encoder (Pillow), exposing the bytes for DQT parsing; it is the JPEG
operator of the post-processing evaluation (Section IV-E6, Table XII). The main grid itself is
uncompressed (``configs/p0.yaml`` ``grid.qf: [none]``).
"""

from __future__ import annotations

import io
from dataclasses import dataclass

import numpy as np
from PIL import Image

from cfgid.channel.inversion.base import InversionBackend
from cfgid.numerics import as_finite_float32


@dataclass
class GeneratedPair:
    h0: np.ndarray  # (H, W, 3) float32 in [0, 1], guidance w=1
    h1: np.ndarray  # (H, W, 3) float32 in [0, 1], guidance w>1
    w: int
    seed: int
    prompt: str


def generate_pair(
    backend: InversionBackend,
    prompt: str,
    *,
    w: int,
    seed: int,
    num_steps: int,
    height: int,
    width: int,
) -> GeneratedPair:
    """Generate the paired H0 (w=1) and H1 (w) images for one prompt/seed."""
    if w <= 1:
        raise ValueError(f"paired H1 needs w>1, got {w}")
    common = {"seed": seed, "num_steps": num_steps, "height": height, "width": width}
    h0 = backend.generate_image(prompt, w=1.0, **common)
    h1 = backend.generate_image(prompt, w=float(w), **common)
    return GeneratedPair(h0=h0, h1=h1, w=w, seed=seed, prompt=prompt)


def jpeg_compress(image: np.ndarray, qf: int) -> tuple[np.ndarray, bytes]:
    """Round-trip an RGB image [0,1] through JPEG at quality ``qf`` -> (decoded[0,1], bytes)."""
    img = as_finite_float32("image", image)
    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError(f"image must be (H, W, 3); got {img.shape}")
    if not (1 <= qf <= 100):
        raise ValueError(f"qf must be in [1, 100], got {qf}")
    if float(img.min()) < -1e-6 or float(img.max()) > 1.0 + 1e-6:
        raise ValueError("image must be in [0, 1]")
    # 8-bit by truncation (astype), then encode; the decoded image is returned in [0, 1].
    u8 = np.clip(img * 255.0, 0.0, 255.0).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(u8).save(buf, format="JPEG", quality=int(qf))
    data = buf.getvalue()
    decoded = np.asarray(Image.open(io.BytesIO(data)).convert("RGB"), dtype=np.float32) / 255.0
    return decoded, data
