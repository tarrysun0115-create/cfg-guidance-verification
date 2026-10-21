"""Layered on-disk feature store, written per decoder by the main-grid run (``cfgid.sweep.run_p0``).

Two schemas, on purpose:
  * compact (``compact_v1``): scalar per-patch stats (rho_p, E_tex,p of Section III-C) +
    per-image scalars (sigma, qf), stored as **float32** — these feed the theory side
    (TV / FPR), where fp16 quantization error would amplify.
  * residual_feature_v1: the full-dimensional per-patch inversion residual (the branch
    difference ``r_cond,p - r_uncond,p``, whose norm is delta_p of Eq. (6)), stored as **fp16 + a
    per-record float32 scale** (cheap on disk), so the pair readout and the feature ablations
    never have to re-run the expensive inversion.

``schema_version`` is written into the metadata of every feature directory and checked on read.
The store is large and server-side (not shipped); scripts that need it say so in their
docstrings, and README.md lists which scripts need it.
"""

from cfgid.featurestore.schema import (
    COMPACT_SCHEMA_VERSION,
    RESIDUAL_FEATURE_VERSION,
    FeatureMetadata,
)
from cfgid.featurestore.store import (
    CompactBatch,
    dequantize_fp16_scaled,
    quantize_fp16_scaled,
    read_compact,
    read_residual_v1,
    write_compact,
    write_residual_v1,
)

__all__ = [
    "COMPACT_SCHEMA_VERSION",
    "RESIDUAL_FEATURE_VERSION",
    "FeatureMetadata",
    "CompactBatch",
    "quantize_fp16_scaled",
    "dequantize_fp16_scaled",
    "write_compact",
    "read_compact",
    "write_residual_v1",
    "read_residual_v1",
]
