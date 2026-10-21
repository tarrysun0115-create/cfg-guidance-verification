"""``python -m cfgid.doctor`` — read-only preflight for a fresh checkout or a new machine.

Checks device negotiation, the Linux CUDA-build assertion, the paths.yaml/models.yaml
contracts + model-snapshot presence, and that runtime caches are pinned in-repo. Reports
``pass``/``warn``/``fail`` per check; only ``fail`` makes the exit code non-zero. Missing
weights are ``warn`` (the preflight passes without any downloads). Configuration only — never
mutates anything, and the fix is always to edit configs/env, never Python.

Usage: python -m cfgid.doctor [--quiet]
"""

from __future__ import annotations

import argparse
import os
import platform
from dataclasses import dataclass
from pathlib import Path

from cfgid import paths

_STATUS_ORDER = {"pass": 0, "warn": 1, "fail": 2}
_CACHE_ENV_VARS = (
    "HF_HOME",
    "TORCH_HOME",
    "XDG_CACHE_HOME",
    "MPLCONFIGDIR",
    "UV_CACHE_DIR",
    "RUFF_CACHE_DIR",
    "PYRIGHT_PYTHON_CACHE_DIR",
)


@dataclass
class Check:
    name: str
    status: str  # "pass" | "warn" | "fail"
    detail: str


def cuda_build_check(system: str, cuda_version: str | None, cuda_available: bool) -> Check:
    """Pure CUDA-build assertion (factored out so it is unit-testable without a GPU)."""
    if system != "Linux":
        return Check("cuda_build", "pass", f"not Linux ({system}); CUDA-build assertion N/A")
    if cuda_version is None:
        return Check(
            "cuda_build",
            "fail",
            "Linux but torch.version.cuda is None (CPU-only build) — install the CUDA wheel "
            "via the pytorch-cu124 index (see pyproject [[tool.uv.index]])",
        )
    if not cuda_available:
        return Check(
            "cuda_build",
            "warn",
            f"torch CUDA build {cuda_version} present, but torch.cuda.is_available() is False "
            "(driver/GPU not visible?)",
        )
    return Check("cuda_build", "pass", f"torch CUDA build {cuda_version}, GPU visible")


def _is_within(child: Path, parent: Path) -> bool:
    try:
        return child.expanduser().resolve().is_relative_to(parent.resolve())
    except (OSError, ValueError):
        return False


def _hf_token_present() -> bool:
    if os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"):
        return True
    hf_home = os.environ.get("HF_HOME")
    if not hf_home or not _is_within(Path(hf_home), paths.REPO_ROOT):
        return False
    try:
        from huggingface_hub import get_token
    except ImportError:
        return False
    return bool(get_token())


def _device_checks() -> list[Check]:
    try:
        import torch
    except ImportError as exc:
        return [Check("torch", "fail", f"torch not importable: {exc}")]

    from cfgid.device import get_device

    device = get_device()
    # getattr(torch, "version", ...) (not torch.version.cuda directly): the torch stubs do not
    # expose the `version` submodule as an attribute, which trips pyright.
    cuda_version = getattr(getattr(torch, "version", None), "cuda", None)
    return [
        Check("device", "pass", f"selected device: {device.type}"),
        cuda_build_check(platform.system(), cuda_version, torch.cuda.is_available()),
    ]


def _config_checks() -> list[Check]:
    checks: list[Check] = []
    # Root resolution is cached in cfgid.paths. Clear it before inspecting paths.yaml so a
    # long-lived process that changes CFGID_CONFIGS_DIR cannot carry stale roots into model checks.
    paths.clear_cache()
    paths_yaml = paths.paths_yaml_path()
    if not paths_yaml.is_file():
        checks.append(Check("paths.yaml", "warn", f"{paths_yaml} missing (using in-repo defaults)"))
    else:
        try:
            paths.load_paths_yaml()  # forces a parse of paths.yaml itself (NOT via an env-
            # overridable root: checkpoints_dir() short-circuits on CFGID_CHECKPOINTS_DIR and would
            # skip the parse, so a malformed paths.yaml would false-PASS preflight and only crash at
            # runtime when data/cache/outputs roots are resolved).
            checks.append(Check("paths.yaml", "pass", f"{paths_yaml} found"))
        except ValueError as exc:
            checks.append(Check("paths.yaml", "fail", f"{paths_yaml} malformed: {exc}"))

    models_yaml = paths.models_yaml_path()
    if not models_yaml.is_file():
        checks.append(Check("models.yaml", "fail", f"{models_yaml} missing"))
        return checks

    from cfgid.config import load_models_config

    try:
        models = load_models_config(models_yaml)
    except (ValueError, KeyError, FileNotFoundError) as exc:
        checks.append(Check("models.yaml", "fail", f"invalid: {exc}"))
        return checks

    checks.append(Check("models.yaml", "pass", f"{len(models)} models: {', '.join(models)}"))
    for name, spec in models.items():
        try:
            present = spec.is_present()
            local = spec.resolved_local_path()
            missing = spec.missing_files()
        except ValueError as exc:  # e.g. malformed paths.yaml during root resolution
            checks.append(Check(f"model:{name}", "fail", f"path resolution failed: {exc}"))
            continue
        if present:
            checks.append(
                Check(
                    f"model:{name}",
                    "pass",
                    f"present at {local} ({spec.branch_mode}/{spec.backend})",
                )
            )
            continue
        detail = f"weights missing at {local} (missing {missing})"
        if spec.gated and not _hf_token_present():
            detail += (
                "; gated — set HF_TOKEN and accept the license "
                "(see the Models section of README.md)"
            )
        checks.append(Check(f"model:{name}", "warn", detail))
    checks.extend(_aux_model_checks())
    return checks


def _aux_model_checks() -> list[Check]:
    """Auxiliary models the main-grid (p0) evidence uses (fail-soft: config error -> no check).

    The CLIP-text-alignment baseline is part of the preregistered negative-control SET when
    enabled; missing weights make analyze fail-safe (no pass), so surface it at preflight.
    """
    try:
        from cfgid.baselines.clip_align import clip_available
        from cfgid.p0config import load_p0_config

        evidence = load_p0_config().evidence
    except (FileNotFoundError, KeyError, ValueError, ImportError):
        return []  # optional preflight: a missing/malformed p0.yaml is reported elsewhere
    if not evidence.clip_baseline_enabled:
        return []
    clip_dir = paths.checkpoints_dir() / evidence.clip_local_dir
    if clip_available(clip_dir):
        return [Check("aux:clip", "pass", f"CLIP baseline present at {clip_dir}")]
    return [
        Check(
            "aux:clip",
            "warn",
            f"CLIP baseline ENABLED but weights missing at {clip_dir}; analyze will fail-safe "
            "(negative controls cannot support a pass) — download per the Models section of "
            "README.md",
        )
    ]


def _cache_checks() -> list[Check]:
    repo = paths.REPO_ROOT
    checks: list[Check] = []
    for var in _CACHE_ENV_VARS:
        val = os.environ.get(var)
        if not val:
            checks.append(
                Check(
                    f"env:{var}",
                    "warn",
                    "unset — run `source scripts/_env_paths.sh` to keep caches in-repo",
                )
            )
        elif _is_within(Path(val), repo):
            checks.append(Check(f"env:{var}", "pass", val))
        else:
            checks.append(Check(f"env:{var}", "warn", f"{val} is outside the repo"))
    return checks


def run_checks() -> list[Check]:
    return [*_device_checks(), *_config_checks(), *_cache_checks()]


def summarize(checks: list[Check]) -> str:
    worst = "pass"
    for check in checks:
        if _STATUS_ORDER[check.status] > _STATUS_ORDER[worst]:
            worst = check.status
    return worst


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="cfgid.doctor", description="open-box preflight")
    parser.add_argument("--quiet", action="store_true", help="only print the final verdict")
    args = parser.parse_args(argv)

    checks = run_checks()
    if not args.quiet:
        for check in checks:
            print(f"[{check.status.upper():4}] {check.name}: {check.detail}")
    worst = summarize(checks)
    print(f"\ncfgid doctor: {worst.upper()}")
    return 1 if worst == "fail" else 0


if __name__ == "__main__":
    raise SystemExit(main())
