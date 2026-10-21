"""Feature-store schema definitions and metadata (schema_version lives here)."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from math import isfinite
from typing import Any

# Bump these (and migrate readers) on any on-disk layout change.
COMPACT_SCHEMA_VERSION = "compact_v1"
RESIDUAL_FEATURE_VERSION = "residual_feature_v1"
PATCH_ORDER_ROW_MAJOR = "row-major"


@dataclass
class FeatureMetadata:
    """Provenance + shape contract for one feature directory.

    Written to ``metadata.json`` and validated on read. ``patch_grid`` is (rows, cols) of
    the latent patch grid; ``n_patches`` must equal ``rows * cols``. Patch-index order is
    row-major over that grid, matching ``channel.inversion.base.patchify``.
    """

    model_name: str
    branch_mode: str
    n_images: int
    n_patches: int
    patch_grid: tuple[int, int]
    residual_dim: int
    patch_order: str = PATCH_ORDER_ROW_MAJOR
    image_ids: list[str] = field(default_factory=list)
    sigma: float | None = None
    qf_note: str = ""
    dqt: list[int] | None = None
    compact_schema_version: str = COMPACT_SCHEMA_VERSION
    residual_schema_version: str = RESIDUAL_FEATURE_VERSION
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        rows, cols = self.patch_grid
        if self.n_images <= 0:
            raise ValueError(f"n_images must be positive, got {self.n_images}")
        if self.n_patches <= 0:
            raise ValueError(f"n_patches must be positive, got {self.n_patches}")
        if self.residual_dim <= 0:
            raise ValueError(f"residual_dim must be positive, got {self.residual_dim}")
        if rows <= 0 or cols <= 0:
            raise ValueError(f"patch_grid dimensions must be positive, got {self.patch_grid}")
        if rows * cols != self.n_patches:
            raise ValueError(
                f"patch_grid {self.patch_grid} does not match n_patches {self.n_patches}"
            )
        if self.patch_order != PATCH_ORDER_ROW_MAJOR:
            raise ValueError(
                f"patch_order must be {PATCH_ORDER_ROW_MAJOR!r}, got {self.patch_order!r}"
            )
        if self.image_ids and len(self.image_ids) != self.n_images:
            raise ValueError(
                f"len(image_ids)={len(self.image_ids)} does not match n_images={self.n_images}"
            )
        if self.sigma is not None and not isfinite(float(self.sigma)):
            raise ValueError(f"sigma must be finite when present, got {self.sigma}")

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["patch_grid"] = [int(self.patch_grid[0]), int(self.patch_grid[1])]
        return d

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> FeatureMetadata:
        data = dict(raw)
        # On the on-disk path the *_schema_version keys MUST be present. The dataclass default is
        # for in-memory construction only; a versionless (stale) metadata.json that silently
        # inherited the current constants would let read_compact/read_residual_v1's version guard
        # see current==current and pass -- a stale artifact masquerading as current. So require them
        # (an EXPLICIT wrong version is still rejected downstream by the readers' guard).
        for key in ("compact_schema_version", "residual_schema_version"):
            if key not in raw:
                raise ValueError(
                    f"metadata.json missing required '{key}'; refusing to default a versionless "
                    "(stale) artifact to the current schema version"
                )
        pg = data.get("patch_grid", [0, 0])
        data["patch_grid"] = (int(pg[0]), int(pg[1]))
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})
