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


def _resolve_root(
    configured: str | Path | None, var_name: str, default_subdir: str
) -> Path:
    """Resolve one storage root, in strict precedence order.

        1. ``configured``  - the value from the Hydra env config group
        2. the environment variable ``var_name``
        3. ``<repo_root>/<default_subdir>``

    BUG FIX (Phase 1C follow-up). Until this change the ``configured``
    parameter did not exist: the resolver consulted only the environment
    variable and the repo-relative default, so ``cfg.env.output_root`` and
    ``cfg.env.checkpoint_root`` were DECORATIVE. A run on the server appeared
    to honour them only because the configured values had been set to the same
    paths the repo-relative default already produced. Changing them in the
    config would have had no effect whatsoever.

    Rationale for config-first: the env config group is the project's declared
    source of truth for machine-specific storage, and a value written there
    must actually take effect. The environment variable remains a genuine
    fallback, which is what keeps ``configs/env/local.yaml`` - whose roots are
    deliberately empty strings - behaving exactly as before.

    An empty string counts as "not configured", so an empty config value falls
    through to the environment variable. That is what preserves the existing
    local behaviour.
    """
    if configured:
        return Path(configured).expanduser().resolve()
    raw = os.environ.get(var_name)
    if raw:
        return Path(raw).expanduser().resolve()
    return (repo_root() / default_subdir).resolve()


def output_root(configured: str | Path | None = None) -> Path:
    """Root directory for run outputs (logs, configs, manifests, metrics)."""
    return _resolve_root(configured, ENV_VAR_OUTPUT_ROOT, "outputs")


def checkpoint_root(configured: str | Path | None = None) -> Path:
    """Root directory for model checkpoints."""
    return _resolve_root(configured, ENV_VAR_CKPT_ROOT, "checkpoints")


def cache_root(configured: str | Path | None = None) -> Path:
    """Root directory for the Hugging Face / dataset cache."""
    return _resolve_root(configured, ENV_VAR_CACHE_ROOT, ".cache")


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


def run_dir(
    run_name: str, create: bool = True, configured: str | Path | None = None
) -> Path:
    """Return (and optionally create) the directory for a single run.

    ``configured`` is the ``cfg.env.output_root`` value; passing it is what
    makes the config authoritative rather than decorative.
    """
    path = output_root(configured) / run_name
    if create:
        path.mkdir(parents=True, exist_ok=True)
    return path


def configure_hf_cache(root: str | Path | None = None) -> dict[str, str]:
    """Point Hugging Face downloads at the AlignLab cache root.

    Declaring ``cache_root`` in a config achieves nothing on its own - the
    Hugging Face libraries read environment variables, not our config. Without
    this call a model download silently lands in ``~/.cache/huggingface``.

    That is not hypothetical. On the department server that default directory
    already holds 5.3 GB belonging to other projects, on a volume with under
    80 GB free, so an unconfigured download would both hide AlignLab's disk
    usage inside someone else's cache and consume space nobody attributed to
    us.

    Which variables are set, and why only these two:

        HF_HUB_CACHE       where model/tokenizer blobs are stored
        HF_DATASETS_CACHE  where dataset arrow files are stored

    ``HF_HOME`` is deliberately NOT set. It is the base directory for hub
    cache *and* stored credentials; moving it would relocate the token file
    too, so an existing login would stop resolving. Setting only the two cache
    variables moves the large files while leaving authentication exactly where
    the user configured it.

    An already-set variable is respected rather than overwritten: an operator
    who exported HF_HUB_CACHE deliberately outranks our config.

    Returns:
        The variables this call actually set (empty if all were already set).
    """
    base = Path(root).expanduser().resolve() if root else cache_root()

    wanted = {
        "HF_HUB_CACHE": str(base / "hub"),
        "HF_DATASETS_CACHE": str(base / "datasets"),
    }

    applied: dict[str, str] = {}
    for name, value in wanted.items():
        if os.environ.get(name):
            continue
        os.environ[name] = value
        applied[name] = value
    return applied


def describe_roots(
    output: str | Path | None = None,
    checkpoint: str | Path | None = None,
    cache: str | Path | None = None,
) -> dict[str, str]:
    """Return all resolved roots as strings, for logging and manifests.

    Recorded in every run manifest so a result can always be traced back to the
    filesystem layout that produced it.

    The configured values MUST be passed through here. Without them the
    manifest reports the environment-variable resolution while the run actually
    used the configured paths - which is precisely the discrepancy that
    exposed the decorative-config bug.
    """
    return {
        "repo_root": str(repo_root()),
        "output_root": str(output_root(output)),
        "checkpoint_root": str(checkpoint_root(checkpoint)),
        "cache_root": str(cache_root(cache)),
    }
