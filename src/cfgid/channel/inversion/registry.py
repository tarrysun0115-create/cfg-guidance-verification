"""Backend registry: ``ModelSpec.backend`` key -> concrete InversionBackend class.

Each decoder entry in ``configs/models.yaml`` names its backend key (Appendix B, Table XV).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from cfgid.channel.inversion.base import InversionBackend
from cfgid.channel.inversion.flux_flow import FluxFlowBackend
from cfgid.channel.inversion.hunyuan_dit_ddim import HunyuanDiTDDIMBackend
from cfgid.channel.inversion.pixart_ddim import PixArtDDIMBackend
from cfgid.channel.inversion.sd_ddim import SD15DDIMBackend
from cfgid.channel.inversion.sd_vpred_ddim import SD21VPredDDIMBackend
from cfgid.channel.inversion.sdxl_ddim import SDXLDDIMBackend
from cfgid.config import ModelSpec

if TYPE_CHECKING:
    import torch

_BACKENDS: dict[str, type[InversionBackend]] = {
    SD15DDIMBackend.backend_key: SD15DDIMBackend,
    SDXLDDIMBackend.backend_key: SDXLDDIMBackend,
    SD21VPredDDIMBackend.backend_key: SD21VPredDDIMBackend,
    FluxFlowBackend.backend_key: FluxFlowBackend,
    PixArtDDIMBackend.backend_key: PixArtDDIMBackend,
    HunyuanDiTDDIMBackend.backend_key: HunyuanDiTDDIMBackend,
}


def available_backends() -> tuple[str, ...]:
    return tuple(_BACKENDS)


def backend_class(key: str) -> type[InversionBackend]:
    if key not in _BACKENDS:
        raise KeyError(f"unknown inversion backend '{key}' (have {sorted(_BACKENDS)})")
    return _BACKENDS[key]


def get_backend(spec: ModelSpec, device: torch.device) -> InversionBackend:
    """Instantiate the backend for ``spec`` (does not load weights)."""
    return backend_class(spec.backend)(spec, device)
