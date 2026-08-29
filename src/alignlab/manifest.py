"""Run manifests - the reproducibility record.

PROJECT_INSTRUCTIONS section 24 requires that meaningful experiments record
model identifier and revision, tokenizer, dataset version, seed, configuration,
hyperparameters, hardware, software environment, checkpoint and evaluation
procedure. This module captures everything that is knowable at run start and
writes it beside the run.

The manifest exists to answer one question after the fact: which code, on
which machine, with which settings, produced this number? Because AlignLab
runs on two very different machines, "which machine" is not optional metadata -
a CPU fp32 result and an Ampere bf16 result are not interchangeable, and the
manifest is what stops them being tabulated as if they were.

Nothing here is inferred. Every field is read from the running process, or is
recorded as None. A field that could not be determined is never guessed.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from alignlab import __version__
from alignlab.device import describe_hardware
from alignlab.env_detect import describe_platform, detect_environment
from alignlab.logging_utils import get_logger
from alignlab.paths import describe_roots, repo_root

logger = get_logger(__name__)

MANIFEST_FILE_NAME = "run_manifest.json"

# Packages whose versions materially affect results. Recorded when importable;
# absent packages are recorded as None rather than omitted, so the manifest
# distinguishes "not installed" from "not checked".
_TRACKED_PACKAGES = (
    "torch",
    "numpy",
    "transformers",
    "datasets",
    "accelerate",
    "peft",
    "trl",
    "bitsandbytes",
    "hydra",
    "omegaconf",
    "wandb",
)


def _run_git(*args: str) -> str | None:
    """Run a git command in the repo, returning stripped stdout or None.

    Returns None rather than raising when git is missing or the directory is
    not a repository - a manifest is still worth writing without git metadata.
    """
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=str(repo_root()),
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        logger.debug("git %s failed: %s", " ".join(args), exc)
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip()


def git_info() -> dict[str, Any]:
    """Capture the git state that produced this run.

    The dirty flag matters more than the commit hash. A result produced from a
    dirty working tree is not reproducible from the recorded SHA, and the
    manifest must say so rather than imply a clean provenance.
    """
    commit = _run_git("rev-parse", "HEAD")
    status = _run_git("status", "--porcelain")
    return {
        "commit": commit,
        "branch": _run_git("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": bool(status) if status is not None else None,
        "dirty_files": status.splitlines() if status else [],
        "describe": _run_git("describe", "--always", "--dirty", "--tags"),
    }


def package_versions() -> dict[str, str | None]:
    """Record versions of the packages that affect results."""
    versions: dict[str, str | None] = {}
    for name in _TRACKED_PACKAGES:
        try:
            module = __import__(name)
        except Exception:
            versions[name] = None
        else:
            versions[name] = str(getattr(module, "__version__", "unknown"))
    return versions


def config_hash(config: dict[str, Any]) -> str:
    """Stable SHA-256 over a resolved configuration.

    Key order is normalised so that two structurally identical configs hash
    identically regardless of how they were composed. This hash is the
    mechanism behind Tier B (structural) cross-machine reproducibility: the
    local run and the server run should share it exactly, even though their
    floating-point outputs will not match.
    """
    payload = json.dumps(config, sort_keys=True, default=str, ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def relevant_env_vars() -> dict[str, str | None]:
    """Capture the environment variables that steer AlignLab.

    Deliberately an allowlist. Dumping os.environ would leak credentials into
    a file that is meant to be shareable.
    """
    names = (
        "ALIGNLAB_ENV",
        "ALIGNLAB_OUTPUT_ROOT",
        "ALIGNLAB_CKPT_ROOT",
        "ALIGNLAB_CACHE_ROOT",
        "ALIGNLAB_RUN_NAME",
        "HF_HOME",
        # HF_HUB_CACHE is the variable AlignLab actually sets (see
        # paths.configure_hf_cache); omitting it would leave the manifest
        # unable to say where a downloaded model came from.
        "HF_HUB_CACHE",
        "HF_DATASETS_CACHE",
        "TRANSFORMERS_CACHE",
        "WANDB_MODE",
        "WANDB_DIR",
        "CUDA_VISIBLE_DEVICES",
        "SLURM_JOB_ID",
        "SLURM_JOB_NAME",
        "SLURM_NNODES",
        "SLURM_NTASKS",
        "PYTHONHASHSEED",
        "CUBLAS_WORKSPACE_CONFIG",
        "OMP_NUM_THREADS",
    )
    return {name: os.environ.get(name) for name in names}


@dataclass
class RunManifest:
    """The complete provenance record for one run."""

    run_name: str
    created_at: str
    alignlab_version: str
    environment: str
    seed: int | None
    config: dict[str, Any] = field(default_factory=dict)
    config_sha256: str | None = None
    git: dict[str, Any] = field(default_factory=dict)
    platform: dict[str, str] = field(default_factory=dict)
    hardware: dict[str, Any] = field(default_factory=dict)
    packages: dict[str, str | None] = field(default_factory=dict)
    paths: dict[str, str] = field(default_factory=dict)
    env_vars: dict[str, str | None] = field(default_factory=dict)
    python_executable: str = ""
    notes: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def write(self, directory: str | Path) -> Path:
        """Write the manifest as JSON into a run directory."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / MANIFEST_FILE_NAME
        path.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=False, default=str),
            encoding="utf-8",
        )
        logger.info("Wrote run manifest: %s", path)
        return path


def capture_environment(
    run_name: str,
    config: dict[str, Any] | None = None,
    seed: int | None = None,
    notes: dict[str, Any] | None = None,
    roots: dict[str, str] | None = None,
) -> RunManifest:
    """Capture everything knowable about this run, right now.

    Args:
        run_name: identifier for the run.
        config: the fully resolved configuration (already a plain dict).
        seed: the master seed actually applied.
        notes: free-form extras, for example a deferred-work marker.
        roots: the storage roots the run ACTUALLY resolved, from
            paths.describe_roots(...) with the configured values passed in.
            Omitting this falls back to environment-variable resolution, which
            can disagree with what the run used - the exact discrepancy that
            exposed the decorative-config bug in Phase 1C.
    """
    config = config or {}
    return RunManifest(
        run_name=run_name,
        created_at=datetime.now(timezone.utc).isoformat(),
        alignlab_version=__version__,
        environment=detect_environment(),
        seed=seed,
        config=config,
        config_sha256=config_hash(config) if config else None,
        git=git_info(),
        platform=describe_platform(),
        hardware=describe_hardware().as_dict(),
        packages=package_versions(),
        paths=roots if roots is not None else describe_roots(),
        env_vars=relevant_env_vars(),
        python_executable=sys.executable,
        notes=notes or {},
    )


def load_manifest(path: str | Path) -> dict[str, Any]:
    """Read a manifest back from disk."""
    return json.loads(Path(path).read_text(encoding="utf-8"))
