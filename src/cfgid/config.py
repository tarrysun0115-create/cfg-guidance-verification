"""Typed config loading for ``configs/models.yaml`` (OmegaConf-backed).

A ``ModelSpec`` is the frozen-backbone snapshot contract: pinning model id / revision /
dtype / scheduler / branch_mode guards against HF auto-downloads or diffusers default
changes silently polluting sigma / g_sigma / the feature distribution. The five decoders
used in the paper and their sources are listed in Table XV (Appendix B).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from omegaconf import OmegaConf
from omegaconf.errors import OmegaConfBaseException

from cfgid import paths

# How a model's rho relates to the SD cond/uncond rho:
#  - true_cfg_core: genuine cond/uncond branches, homogeneous with SD1.5/SDXL (core).
#  - flux_specific_appendix: guidance-distilled flow model (FLUX); rho is NOT homogeneous with
#    the core decoders. FLUX is outside the DDIM-invertible family and is not used in the paper.
VALID_BRANCH_MODES: tuple[str, ...] = ("true_cfg_core", "flux_specific_appendix")
VALID_BACKENDS: tuple[str, ...] = (
    "sd_ddim",
    "sdxl_ddim",
    "sd_vpred_ddim",
    "flux_flow",
    "pixart_ddim",
    "hunyuan_dit_ddim",
)

# A backend dictates its branch_mode; mismatched combos (e.g. flux_flow + true_cfg_core) are
# rejected so a FLUX entry can never be routed through the core (true_cfg_core) analysis.
_BACKEND_BRANCH_MODE: dict[str, str] = {
    "sd_ddim": "true_cfg_core",
    "sdxl_ddim": "true_cfg_core",
    "sd_vpred_ddim": "true_cfg_core",  # SD2.x v-pred: rho homogeneous after v->epsilon convert
    "flux_flow": "flux_specific_appendix",
    "pixart_ddim": "true_cfg_core",  # PixArt-Sigma DiT: epsilon-pred, rho-homogeneous core decoder
    "hunyuan_dit_ddim": "true_cfg_core",  # Hunyuan-DiT: v-pred->eps, rho-homogeneous core decoder
}


@dataclass(frozen=True)
class ModelSpec:
    name: str
    hf_repo_id: str
    revision: str
    local_dir: str
    dtype: str
    scheduler: str
    branch_mode: str
    backend: str
    license: str
    gated: bool
    vae_channels: int
    text_encoders: tuple[str, ...]
    expected_files: tuple[str, ...]
    variant: str | None = None
    notes: str = ""

    def resolved_local_path(self) -> Path:
        """Absolute path to the model snapshot under the (overridable) checkpoints root."""
        root = paths.checkpoints_dir()
        out = root / self.local_dir
        try:
            if not out.resolve().is_relative_to(root.resolve()):
                raise ValueError(
                    f"model '{self.name}' local_dir escapes checkpoints root: {self.local_dir}"
                )
        except OSError as exc:
            raise ValueError(f"model '{self.name}' local_dir resolution failed: {exc}") from exc
        return out

    def missing_files(self) -> list[str]:
        root = self.resolved_local_path()
        expected = _resolve_expected_files(self.name, root, self.expected_files)
        return [rel for rel, path in expected if not path.is_file()]

    def is_present(self) -> bool:
        return bool(self.expected_files) and not self.missing_files()


def _require(raw: dict[str, Any], key: str, model_name: str) -> Any:
    if key not in raw:
        raise KeyError(f"models.yaml: model '{model_name}' missing required key '{key}'")
    return raw[key]


def _validate_local_dir(model_name: str, value: Any) -> str:
    local_dir = str(value).strip()
    path = Path(local_dir)
    if not local_dir or path.is_absolute() or not path.parts or ".." in path.parts:
        raise ValueError(
            f"models.yaml: model '{model_name}' local_dir must be a relative path under "
            f"the checkpoints root, got '{value}'"
        )
    return local_dir


def _validate_expected_files(model_name: str, value: Any) -> tuple[str, ...]:
    if isinstance(value, str) or not isinstance(value, (list, tuple)):
        raise ValueError(
            f"models.yaml: model '{model_name}' expected_files must be a non-empty "
            "sequence of relative file paths"
        )
    out: list[str] = []
    for raw in value:
        rel = str(raw).strip()
        path = Path(rel)
        if not rel or path.is_absolute() or not path.parts or ".." in path.parts:
            raise ValueError(
                f"models.yaml: model '{model_name}' expected_files entries must be "
                f"relative paths under the model local_dir, got '{raw}'"
            )
        out.append(rel)
    if not out:
        raise ValueError(f"models.yaml: model '{model_name}' expected_files must not be empty")
    return tuple(out)


def _resolve_expected_files(
    model_name: str, root: Path, expected_files: tuple[str, ...]
) -> list[tuple[str, Path]]:
    files = _validate_expected_files(model_name, list(expected_files))
    resolved_root = root.resolve()
    out: list[tuple[str, Path]] = []
    for rel in files:
        candidate = root / rel
        try:
            if not candidate.resolve().is_relative_to(resolved_root):
                raise ValueError(
                    f"models.yaml: model '{model_name}' expected file escapes local_dir: {rel}"
                )
        except OSError as exc:
            raise ValueError(
                f"models.yaml: model '{model_name}' expected file resolution failed: {exc}"
            ) from exc
        out.append((rel, candidate))
    return out


def _to_spec(name: str, raw: dict[str, Any]) -> ModelSpec:
    branch_mode = str(_require(raw, "branch_mode", name))
    if branch_mode not in VALID_BRANCH_MODES:
        raise ValueError(
            f"models.yaml: model '{name}' invalid branch_mode '{branch_mode}' "
            f"(expected one of {VALID_BRANCH_MODES})"
        )
    backend = str(_require(raw, "backend", name))
    if backend not in VALID_BACKENDS:
        raise ValueError(
            f"models.yaml: model '{name}' invalid backend '{backend}' "
            f"(expected one of {VALID_BACKENDS})"
        )
    expected_branch = _BACKEND_BRANCH_MODE[backend]
    if branch_mode != expected_branch:
        raise ValueError(
            f"models.yaml: model '{name}' backend '{backend}' requires "
            f"branch_mode '{expected_branch}', got '{branch_mode}'"
        )
    variant = raw.get("variant")
    return ModelSpec(
        name=name,
        hf_repo_id=str(_require(raw, "hf_repo_id", name)),
        revision=str(raw.get("revision", "main")),
        local_dir=_validate_local_dir(name, _require(raw, "local_dir", name)),
        dtype=str(_require(raw, "dtype", name)),
        scheduler=str(_require(raw, "scheduler", name)),
        branch_mode=branch_mode,
        backend=backend,
        license=str(raw.get("license", "unknown")),
        gated=bool(raw.get("gated", False)),
        vae_channels=int(raw.get("vae_channels", 4)),
        text_encoders=tuple(str(x) for x in raw.get("text_encoders", [])),
        expected_files=_validate_expected_files(name, raw.get("expected_files", [])),
        variant=(str(variant) if variant is not None else None),
        notes=str(raw.get("notes", "")),
    )


def load_models_config(path: Path | None = None) -> dict[str, ModelSpec]:
    """Load ``models.yaml`` into ``{name: ModelSpec}`` (validates required fields)."""
    cfg_path = Path(path) if path is not None else paths.models_yaml_path()
    if not cfg_path.is_file():
        raise FileNotFoundError(f"models.yaml not found at {cfg_path}")
    try:
        container = OmegaConf.to_container(OmegaConf.load(cfg_path), resolve=True)
    except (OmegaConfBaseException, yaml.YAMLError) as exc:
        raise ValueError(f"models.yaml malformed: {exc}") from exc
    if not isinstance(container, dict):
        raise ValueError(f"models.yaml malformed: expected a mapping, got {type(container)}")
    models_node = container.get("models")
    if not isinstance(models_node, dict):
        raise ValueError("models.yaml must have a top-level 'models:' mapping of name -> spec")
    out: dict[str, ModelSpec] = {}
    for raw_name, raw_spec in models_node.items():
        if not isinstance(raw_spec, dict):
            raise ValueError(f"models.yaml: model '{raw_name}' must be a mapping")
        name = str(raw_name)
        out[name] = _to_spec(name, raw_spec)
    return out
