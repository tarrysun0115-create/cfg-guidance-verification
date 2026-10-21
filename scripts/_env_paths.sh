# shellcheck shell=bash
# Pin every runtime download and cache inside the repository, so `rm -rf <repo>` removes all of it.
#
# The entry point scripts/run_p0 sources this file first. `python -m cfgid.doctor` warns
# when one of the cache variables below is unset or points outside the repository.
#
# Usage:  source scripts/_env_paths.sh
#
# Cross-shell repo-root detection: bash uses BASH_SOURCE, zsh uses ${(%):-%N}. Each shell
# only expands the branch it takes, so the "bad substitution" form is never evaluated.
_self="${BASH_SOURCE[0]:-${(%):-%N}}"
_REPO_ROOT="$(cd "$(dirname "$_self")/.." && pwd)"

export CFGID_REPO_ROOT="$_REPO_ROOT"

# Optional local secrets (HF_TOKEN for gated models, etc.). Never committed (.gitignore'd).
for _f in "$_REPO_ROOT/.env.local"; do
  if [ -f "$_f" ]; then
    set -a
    # shellcheck source=/dev/null
    . "$_f"
    set +a
  fi
done

# Pin cache vars after local secrets are loaded so `.env.local` cannot redirect runtime
# downloads/caches outside the repo.
export XDG_CACHE_HOME="$_REPO_ROOT/.cache"
export HF_HOME="$_REPO_ROOT/.cache/hf"
export TORCH_HOME="$_REPO_ROOT/.cache/torch"
export MPLCONFIGDIR="$_REPO_ROOT/.cache/mpl"
export UV_CACHE_DIR="$_REPO_ROOT/.cache/uv"
export RUFF_CACHE_DIR="$_REPO_ROOT/.cache/ruff"
export PYRIGHT_PYTHON_CACHE_DIR="$_REPO_ROOT/.cache/pyright"
# MPS ops without a kernel fall back to CPU instead of erroring (mirrors cfgid.device).
export PYTORCH_ENABLE_MPS_FALLBACK=1

mkdir -p \
  "$HF_HOME" \
  "$TORCH_HOME" \
  "$MPLCONFIGDIR" \
  "$XDG_CACHE_HOME" \
  "$UV_CACHE_DIR" \
  "$RUFF_CACHE_DIR" \
  "$PYRIGHT_PYTHON_CACHE_DIR"

unset _self _f _REPO_ROOT
