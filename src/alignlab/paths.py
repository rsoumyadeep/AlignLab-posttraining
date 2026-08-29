"""Machine-independent path resolution.

AlignLab runs in two very different places: a local Windows development
machine and a Linux GPU server. Hard-coding a path for either one would break
the other, and would make results non-portable. So every "where do things go"
question is answered here, from environment variables with local-friendly
defaults.

The four roots:

    ALIGNLAB_OUTPUT_ROOT   run directories: logs, resolved configs, manifests
    ALIGNLAB_CKPT_ROOT     model checkpoints (large; never in Git)
    ALIGNLAB_CACHE_ROOT    Hugging Face / dataset cache (large; never in Git)
    ALIGNLAB_RUN_NAME      optional explicit run name (otherwise generated)

Defaults are repository-relative, which is correct for the local machine and
deliberately WRONG for a shared server - on the server these must be pointed at
a large volume, which is why configs/env/server.yaml requires them explicitly
rather than silently inheriting a default that would fill a root partition.
"""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

# Environment variable names, centralised so tests and docs can refer to them
# without duplicating string literals.
ENV_VAR_ENV_NAME = "ALIGNLAB_ENV"
ENV_VAR_OUTPUT_ROOT = "ALIGNLAB_OUTPUT_ROOT"
ENV_VAR_CKPT_ROOT = "ALIGNLAB_CKPT_ROOT"
ENV_VAR_CACHE_ROOT = "ALIGNLAB_CACHE_ROOT"
ENV_VAR_RUN_NAME = "ALIGNLAB_RUN_NAME"


def repo_root() -> Path:
    """Return the repository root.

    Located by walking up from this file until a directory containing
    ``pyproject.toml`` is found. Falls back to three levels up (the layout
    src/alignlab/paths.py -> repo root) if no marker is found, which happens
    when the package is installed non-editable.
    """
    here = Path(__file__).resolve()
    for candidate in here.parents:
        if (candidate / "pyproject.toml").is_file():
            return candidate
    return here.parents[2]


def _root_from_env(var_name: str, default_subdir: str) -> Path:
    """Resolve one root directory from an environment variable.

    An unset variable falls back to ``<repo_root>/<default_subdir>``. That
    default is appropriate locally and is intentionally not appropriate on a
    shared server, where the value must be set explicitly.
    """
    raw = os.environ.get(var_name)
    if raw:
        return Path(raw).expanduser().resolve()
    return (repo_root() / default_subdir).resolve()


def output_root() -> Path:
    """Root directory for run outputs (logs, configs, manifests, metrics)."""
    return _root_from_env(ENV_VAR_OUTPUT_ROOT, "outputs")


def checkpoint_root() -> Path:
    """Root directory for model checkpoints."""
    return _root_from_env(ENV_VAR_CKPT_ROOT, "checkpoints")


def cache_root() -> Path:
    """Root directory for the Hugging Face / dataset cache."""
    return _root_from_env(ENV_VAR_CACHE_ROOT, ".cache")


def generate_run_name(prefix: str = "run") -> str:
    """Generate a sortable, human-readable run name.

    Format ``<prefix>-YYYYmmdd-HHMMSS``. Lexicographic order matches
    chronological order, which makes run directories easy to scan.
    ``ALIGNLAB_RUN_NAME`` overrides this entirely, which is what lets a
    resumed or re-attached job write into its original directory.
    """
    override = os.environ.get(ENV_VAR_RUN_NAME)
    if override:
        return override
    return f"{prefix}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"


def run_dir(run_name: str, create: bool = True) -> Path:
    """Return (and optionally create) the directory for a single run."""
    path = output_root() / run_name
    if create:
        path.mkdir(parents=True, exist_ok=True)
    return path


def describe_roots() -> dict[str, str]:
    """Return all resolved roots as strings, for logging and manifests.

    Recorded in every run manifest so that a result can always be traced back
    to the filesystem layout that produced it.
    """
    return {
        "repo_root": str(repo_root()),
        "output_root": str(output_root()),
        "checkpoint_root": str(checkpoint_root()),
        "cache_root": str(cache_root()),
    }
