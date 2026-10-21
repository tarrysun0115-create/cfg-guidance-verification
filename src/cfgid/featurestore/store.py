"""Read/write the layered feature store (deterministic: same input -> same bytes).

Layout per feature directory:
  metadata.json            schema versions + provenance (sorted keys -> byte-stable)
  compact_rho.npy          (N, P) float32
  compact_e_tex.npy        (N, P) float32
  compact_sigma.npy        (N,)   float32
  compact_qf.npy           (N,)   float32  (NaN = uncompressed)
  residual_v1.dat          (N, P, D) float16 memmap  (per-patch r_cond - r_uncond, scaled)
  residual_v1_scale.npy    (N, P) float32   per-record scale

Plain ``.npy`` / raw memmap are used instead of ``.npz`` so the bytes are timestamp-free
and reproducible.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from cfgid.featurestore.schema import (
    COMPACT_SCHEMA_VERSION,
    RESIDUAL_FEATURE_VERSION,
    FeatureMetadata,
)
from cfgid.numerics import loads_strict_json

_METADATA_NAME = "metadata.json"
_RESIDUAL_DAT = "residual_v1.dat"
_RESIDUAL_SCALE = "residual_v1_scale.npy"
_COMPACT_FILES = {
    "rho": "compact_rho.npy",
    "e_tex": "compact_e_tex.npy",
    "sigma": "compact_sigma.npy",
    "qf": "compact_qf.npy",
}


@dataclass
class CompactBatch:
    """Per-image compact stats over the patch grid (float32 enforced on write)."""

    rho: np.ndarray  # (N, P) float32
    e_tex: np.ndarray  # (N, P) float32
    sigma: np.ndarray  # (N,) float32
    qf: np.ndarray  # (N,) float32, NaN for uncompressed

    def validate(self, n_images: int, n_patches: int) -> None:
        if self.rho.shape != (n_images, n_patches):
            raise ValueError(f"rho shape {self.rho.shape} != {(n_images, n_patches)}")
        if self.e_tex.shape != (n_images, n_patches):
            raise ValueError(f"e_tex shape {self.e_tex.shape} != {(n_images, n_patches)}")
        if self.sigma.shape != (n_images,):
            raise ValueError(f"sigma shape {self.sigma.shape} != {(n_images,)}")
        if self.qf.shape != (n_images,):
            raise ValueError(f"qf shape {self.qf.shape} != {(n_images,)}")
        _require_finite("rho", self.rho)
        _require_finite("e_tex", self.e_tex)
        _require_finite("sigma", self.sigma)
        _require_finite("qf", self.qf, allow_nan=True)


def _as_float32(name: str, values: np.ndarray) -> np.ndarray:
    try:
        return np.asarray(values, dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric") from exc


def _require_finite(name: str, values: np.ndarray, *, allow_nan: bool = False) -> None:
    arr = _as_float32(name, values)
    if allow_nan:
        bad = np.isinf(arr)
    else:
        bad = ~np.isfinite(arr)
    if np.any(bad):
        suffix = " (NaN is allowed only for qf=uncompressed)" if allow_nan else ""
        raise ValueError(f"{name} contains non-finite values{suffix}")


def _load_float32(path: Path, name: str) -> np.ndarray:
    arr = np.load(path)
    if arr.dtype != np.float32:
        raise ValueError(f"{name} must be stored as float32, got {arr.dtype}")
    return arr


def _load_residual_scale(path: Path, shape: tuple[int, int]) -> np.ndarray:
    scale = _load_float32(path, "residual scale")
    if scale.shape != shape:
        raise ValueError(f"residual scale shape {scale.shape} != {shape}")
    _require_finite("residual scale", scale)
    return scale


def _check_residual_dat_size(path: Path, shape: tuple[int, int, int]) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"residual memmap not found at {path}")
    expected = shape[0] * shape[1] * shape[2] * np.dtype(np.float16).itemsize
    actual = path.stat().st_size
    if actual != expected:
        raise ValueError(f"residual memmap size {actual} bytes != expected {expected} bytes")


# --------------------------------------------------------------------------- #
# fp16 + per-record scale quantization (used by residual_feature_v1)
# --------------------------------------------------------------------------- #
def quantize_fp16_scaled(residual: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Quantize (..., D) residual vectors to fp16 with a per-record float32 scale.

    Returns ``(q_fp16, scale_f32)`` where ``residual ~= q.astype(f32) * scale[..., None]``.
    Scaling by each record's max-abs keeps the full fp16 mantissa precision per vector.

    Raises ``ValueError`` on any non-finite input — a NaN/Inf would otherwise corrupt the
    whole record's scale and round-trip silently (a diverged high-CFG inversion must fail
    loudly, never become a silent NaN).
    """
    r = np.asarray(residual, dtype=np.float32)
    if not np.all(np.isfinite(r)):
        raise ValueError(
            "residual contains non-finite values (NaN/Inf); refusing to quantize "
            "(guard against silent high-CFG inversion divergence corruption)"
        )
    scale = np.max(np.abs(r), axis=-1)
    safe = np.where(scale > 0.0, scale, 1.0).astype(np.float32)
    q = (r / safe[..., None]).astype(np.float16)
    return q, scale.astype(np.float32)


def dequantize_fp16_scaled(q: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """Inverse of :func:`quantize_fp16_scaled`."""
    return q.astype(np.float32) * scale[..., None].astype(np.float32)


# --------------------------------------------------------------------------- #
# metadata
# --------------------------------------------------------------------------- #
def _write_metadata(out: Path, meta: FeatureMetadata) -> None:
    # allow_nan=False: NaN/Inf serialize to the bare token `NaN`/`Infinity`, which is invalid
    # JSON (RFC 8259) and would break the "same input -> same bytes" / external-reader contract.
    text = json.dumps(meta.to_dict(), sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False)
    (out / _METADATA_NAME).write_text(text + "\n", encoding="utf-8")


def _read_metadata(out: Path) -> FeatureMetadata:
    path = out / _METADATA_NAME
    if not path.is_file():
        raise FileNotFoundError(f"feature metadata not found at {path}")
    raw = loads_strict_json(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: feature metadata must be a JSON object")
    return FeatureMetadata.from_dict(raw)


def read_metadata(out_dir: str | Path) -> FeatureMetadata:
    return _read_metadata(Path(out_dir))


# --------------------------------------------------------------------------- #
# compact
# --------------------------------------------------------------------------- #
def write_compact(out_dir: str | Path, batch: CompactBatch, meta: FeatureMetadata) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    batch.validate(meta.n_images, meta.n_patches)
    np.save(out / _COMPACT_FILES["rho"], batch.rho.astype(np.float32, copy=False))
    np.save(out / _COMPACT_FILES["e_tex"], batch.e_tex.astype(np.float32, copy=False))
    np.save(out / _COMPACT_FILES["sigma"], batch.sigma.astype(np.float32, copy=False))
    np.save(out / _COMPACT_FILES["qf"], batch.qf.astype(np.float32, copy=False))
    _write_metadata(out, meta)


def read_compact(out_dir: str | Path) -> tuple[CompactBatch, FeatureMetadata]:
    out = Path(out_dir)
    meta = _read_metadata(out)
    if meta.compact_schema_version != COMPACT_SCHEMA_VERSION:
        raise ValueError(
            f"compact schema mismatch: file '{meta.compact_schema_version}' "
            f"!= reader '{COMPACT_SCHEMA_VERSION}'"
        )
    batch = CompactBatch(
        rho=_load_float32(out / _COMPACT_FILES["rho"], "rho"),
        e_tex=_load_float32(out / _COMPACT_FILES["e_tex"], "e_tex"),
        sigma=_load_float32(out / _COMPACT_FILES["sigma"], "sigma"),
        qf=_load_float32(out / _COMPACT_FILES["qf"], "qf"),
    )
    batch.validate(meta.n_images, meta.n_patches)
    return batch, meta


# --------------------------------------------------------------------------- #
# residual_feature_v1
# --------------------------------------------------------------------------- #
def write_residual_v1(out_dir: str | Path, residual: np.ndarray, meta: FeatureMetadata) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    r = np.asarray(residual, dtype=np.float32)
    if r.ndim != 3:
        raise ValueError(f"residual must be (N, P, D); got shape {r.shape}")
    n, p, d = r.shape
    if (n, p, d) != (meta.n_images, meta.n_patches, meta.residual_dim):
        raise ValueError(
            f"residual shape {(n, p, d)} != meta "
            f"{(meta.n_images, meta.n_patches, meta.residual_dim)}"
        )
    q, scale = quantize_fp16_scaled(r)
    mm = np.memmap(out / _RESIDUAL_DAT, dtype=np.float16, mode="w+", shape=(n, p, d))
    mm[:] = q
    mm.flush()
    del mm
    np.save(out / _RESIDUAL_SCALE, scale)
    _write_metadata(out, meta)


def read_residual_v1(out_dir: str | Path) -> tuple[np.ndarray, FeatureMetadata]:
    out = Path(out_dir)
    meta = _read_metadata(out)
    if meta.residual_schema_version != RESIDUAL_FEATURE_VERSION:
        raise ValueError(
            f"residual schema mismatch: file '{meta.residual_schema_version}' "
            f"!= reader '{RESIDUAL_FEATURE_VERSION}'"
        )
    shape = (meta.n_images, meta.n_patches, meta.residual_dim)
    _check_residual_dat_size(out / _RESIDUAL_DAT, shape)
    q = np.memmap(out / _RESIDUAL_DAT, dtype=np.float16, mode="r", shape=shape)
    scale = _load_residual_scale(out / _RESIDUAL_SCALE, shape[:2])
    return dequantize_fp16_scaled(np.asarray(q), scale), meta
