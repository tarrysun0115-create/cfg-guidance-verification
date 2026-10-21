"""In-repo path contract: all runtime artifacts live under the repo by default.

Roots: ``checkpoints/`` (model weights), ``data/``, ``.cache/`` and ``outputs/``.
Resolution priority for each root: env var ``CFGID_<NAME>_DIR`` > ``configs/paths.yaml``
entry > in-repo default. ``configs/paths.yaml`` is the single file a server edits to
relocate roots without ever touching Python.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import yaml
from omegaconf import OmegaConf
from omegaconf.errors import OmegaConfBaseException

# src/cfgid/paths.py -> parents: [0]=cfgid, [1]=src, [2]=<repo root>
REPO_ROOT: Path = Path(__file__).resolve().parents[2]

# Root name -> (env var, paths.yaml key, default path relative to REPO_ROOT).
_ROOTS: dict[str, tuple[str, str, str]] = {
    "checkpoints": ("CFGID_CHECKPOINTS_DIR", "checkpoints", "checkpoints"),
    "data": ("CFGID_DATA_DIR", "data", "data"),
    "cache": ("CFGID_CACHE_DIR", "cache", ".cache"),
    "outputs": ("CFGID_OUTPUTS_DIR", "outputs", "outputs"),
}


def _abs(value: str) -> Path:
    """Absolute path; relative values are anchored at the repo root, not the CWD."""
    p = Path(value).expanduser()
    return p if p.is_absolute() else (REPO_ROOT / p)


@lru_cache(maxsize=1)
def _paths_yaml() -> dict[str, str]:
    # Read from configs_dir() so the CFGID_CONFIGS_DIR override is honored consistently
    # everywhere (root resolution, doctor, models_yaml_path).
    yaml_path = configs_dir() / "paths.yaml"
    if not yaml_path.is_file():
        return {}
    try:
        container = OmegaConf.to_container(OmegaConf.load(yaml_path), resolve=True)
    except (OmegaConfBaseException, yaml.YAMLError) as exc:
        raise ValueError(f"paths.yaml malformed: {exc}") from exc
    if not isinstance(container, dict):
        return {}
    roots = container.get("roots", container)
    if not isinstance(roots, dict):
        return {}
    return {str(k): str(v) for k, v in roots.items() if v is not None}


def load_paths_yaml() -> dict[str, str]:
    """Parse ``configs/paths.yaml`` (the root overrides), raising ValueError if it is malformed.

    Always reads and parses the file regardless of any ``CFGID_*_DIR`` env override, so a
    malformed paths.yaml is detected by ``cfgid doctor`` even when every root is env-overridden
    (``_resolve_root`` short-circuits on the env var and would otherwise never trigger the parse).
    """
    return _paths_yaml()


def _resolve_root(name: str) -> Path:
    env_var, yaml_key, default_rel = _ROOTS[name]
    env_val = os.environ.get(env_var)
    if env_val:
        return _abs(env_val)
    yaml_val = _paths_yaml().get(yaml_key)
    if yaml_val:
        return _abs(yaml_val)
    return REPO_ROOT / default_rel


def checkpoints_dir() -> Path:
    return _resolve_root("checkpoints")


def data_dir() -> Path:
    return _resolve_root("data")


def cache_dir() -> Path:
    return _resolve_root("cache")


def outputs_dir() -> Path:
    return _resolve_root("outputs")


def configs_dir() -> Path:
    # configs/ holds paths.yaml itself, so it is resolved without reading that file.
    env_val = os.environ.get("CFGID_CONFIGS_DIR")
    return _abs(env_val) if env_val else (REPO_ROOT / "configs")


def models_yaml_path() -> Path:
    return configs_dir() / "models.yaml"


def paths_yaml_path() -> Path:
    return configs_dir() / "paths.yaml"


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def clear_cache() -> None:
    """Forget the cached paths.yaml read (used by tests that mutate config on disk)."""
    _paths_yaml.cache_clear()
