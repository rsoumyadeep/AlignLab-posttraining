"""Which machine are we on?

AlignLab has exactly two intended execution environments:

    "local"  - development machine: authoring, docs, fast CPU tests
    "server" - department GPU server: all real training

The environment name selects a Hydra config group (configs/env/<name>.yaml)
and is recorded in every run manifest, so a result can never be silently
mis-attributed to the wrong machine.

Resolution order:
    1. the ALIGNLAB_ENV environment variable (explicit wins, always)
    2. a CUDA-capable machine is assumed to be "server"
    3. otherwise "local"

Rule 2 is a convenience, not a guarantee. Being wrong means picking up the
wrong default paths, which the config system surfaces loudly rather than
silently mis-placing a checkpoint.
"""

from __future__ import annotations

import os
import platform
import socket

from alignlab.paths import ENV_VAR_ENV_NAME

LOCAL = "local"
SERVER = "server"
KNOWN_ENVIRONMENTS = (LOCAL, SERVER)


def hostname() -> str:
    """Return this machine's hostname (recorded in run manifests)."""
    try:
        return socket.gethostname()
    except OSError:  # pragma: no cover - essentially never fails
        return "unknown-host"


def _cuda_appears_available() -> bool:
    """Best-effort CUDA check that never raises and never requires torch.

    Imported lazily because this module is used by the config layer, which
    must stay importable in a stripped-down environment.
    """
    try:
        import torch
    except Exception:
        return False
    try:
        return bool(torch.cuda.is_available())
    except Exception:  # pragma: no cover - defensive
        return False


def detect_environment() -> str:
    """Return the environment name for this machine.

    Raises ValueError if ALIGNLAB_ENV holds an unknown value - a typo would
    otherwise select the wrong storage roots, so it must fail loudly.
    """
    explicit = os.environ.get(ENV_VAR_ENV_NAME)
    if explicit:
        name = explicit.strip().lower()
        if name not in KNOWN_ENVIRONMENTS:
            raise ValueError(
                f"{ENV_VAR_ENV_NAME}={explicit!r} is not a known environment. "
                f"Expected one of {KNOWN_ENVIRONMENTS}."
            )
        return name
    return SERVER if _cuda_appears_available() else LOCAL


def describe_platform() -> dict[str, str]:
    """Return a platform description for the run manifest."""
    return {
        "hostname": hostname(),
        "platform": platform.platform(),
        "system": platform.system(),
        "machine": platform.machine(),
        "processor": platform.processor() or "unknown",
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
    }
