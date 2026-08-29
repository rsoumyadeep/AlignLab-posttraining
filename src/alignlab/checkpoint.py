"""Checkpoint save, load and rotation.

Three properties matter, and each exists for a concrete failure this project
expects to hit on a shared, unscheduled GPU server:

1. ATOMICITY. A checkpoint is written to a temporary file and then renamed
   into place. os.replace is atomic within a filesystem, so a process killed
   mid-write leaves either the old complete checkpoint or the new complete
   one - never a truncated file that fails to load hours later. On a shared
   machine where a neighbour can trigger an OOM kill, this is not theoretical.

2. COMPLETENESS. A checkpoint that restores weights but not optimizer state,
   scheduler state, step counter and RNG state does not resume a run - it
   starts a subtly different one. All five are stored.

3. ROTATION. Long runs otherwise fill the disk. The server data volume was
   observed at 74% full and is shared with other users, so unbounded
   checkpointing is antisocial as well as risky.

Note on torch.load: weights_only=False is required because AlignLab
checkpoints legitimately contain non-tensor Python objects (RNG state tuples,
the config dict). That is safe here because we only ever load checkpoints this
project wrote. Never point load_checkpoint at an untrusted file.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from alignlab.logging_utils import get_logger
from alignlab.seeding import capture_rng_state, restore_rng_state

logger = get_logger(__name__)

CHECKPOINT_SUFFIX = ".pt"
LATEST_LINK_NAME = "latest.txt"
_TMP_SUFFIX = ".tmp"


@dataclass
class CheckpointPayload:
    """What was loaded back from disk."""

    step: int
    epoch: int
    model_state: dict[str, Any]
    optimizer_state: dict[str, Any] | None
    scheduler_state: dict[str, Any] | None
    rng_state: dict[str, Any] | None
    config: dict[str, Any] | None
    extra: dict[str, Any]


def save_checkpoint(
    directory: str | Path,
    step: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any | None = None,
    epoch: int = 0,
    config: dict[str, Any] | None = None,
    extra: dict[str, Any] | None = None,
    include_rng: bool = True,
    keep_last: int | None = 3,
) -> Path:
    """Write a checkpoint atomically and return its path.

    Args:
        directory: checkpoint directory (created if absent).
        step: global step, used in the filename and for ordering.
        model: the module whose state_dict is saved.
        optimizer / scheduler: saved when provided. Omitting the optimizer
            makes the checkpoint an inference artifact, not a resumable one.
        epoch, config, extra: recorded verbatim.
        include_rng: capture RNG state so a resume continues the same stream.
        keep_last: retain only the N most recent checkpoints. None disables
            rotation.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    payload: dict[str, Any] = {
        "step": step,
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict() if optimizer is not None else None,
        "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
        "rng_state": capture_rng_state() if include_rng else None,
        "config": config,
        "extra": extra or {},
        "format_version": 1,
    }

    final_path = directory / f"step-{step:09d}{CHECKPOINT_SUFFIX}"
    tmp_path = final_path.with_suffix(final_path.suffix + _TMP_SUFFIX)

    torch.save(payload, tmp_path)
    # Atomic within a filesystem; overwrites an existing target on POSIX and,
    # unlike os.rename, also on Windows.
    os.replace(tmp_path, final_path)

    # A plain text pointer rather than a symlink: symlinks need elevated
    # privileges on Windows, and this file must work identically on both
    # machines.
    (directory / LATEST_LINK_NAME).write_text(final_path.name, encoding="utf-8")

    logger.info("Saved checkpoint: %s", final_path)

    if keep_last is not None:
        _rotate(directory, keep_last)

    return final_path


def load_checkpoint(
    path: str | Path,
    model: torch.nn.Module | None = None,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any | None = None,
    restore_rng: bool = True,
    map_location: str | torch.device = "cpu",
) -> CheckpointPayload:
    """Load a checkpoint, optionally restoring state into live objects.

    map_location defaults to "cpu" deliberately: it makes a checkpoint written
    on the GPU server loadable on the local CPU machine for inspection, and
    avoids allocating GPU memory during load only to move tensors again.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"No checkpoint at {path}")

    # weights_only=False: AlignLab checkpoints contain RNG state tuples and the
    # config dict, which are not tensors. Only load checkpoints this project
    # wrote.
    raw = torch.load(path, map_location=map_location, weights_only=False)

    if model is not None:
        model.load_state_dict(raw["model_state"])
    if optimizer is not None and raw.get("optimizer_state") is not None:
        optimizer.load_state_dict(raw["optimizer_state"])
    if scheduler is not None and raw.get("scheduler_state") is not None:
        scheduler.load_state_dict(raw["scheduler_state"])
    if restore_rng and raw.get("rng_state") is not None:
        restore_rng_state(raw["rng_state"])

    logger.info("Loaded checkpoint: %s (step=%s)", path, raw.get("step"))

    return CheckpointPayload(
        step=raw.get("step", 0),
        epoch=raw.get("epoch", 0),
        model_state=raw["model_state"],
        optimizer_state=raw.get("optimizer_state"),
        scheduler_state=raw.get("scheduler_state"),
        rng_state=raw.get("rng_state"),
        config=raw.get("config"),
        extra=raw.get("extra", {}),
    )


def list_checkpoints(directory: str | Path) -> list[Path]:
    """Return checkpoints in the directory, oldest first."""
    directory = Path(directory)
    if not directory.is_dir():
        return []
    return sorted(directory.glob(f"step-*{CHECKPOINT_SUFFIX}"))


def latest_checkpoint(directory: str | Path) -> Path | None:
    """Return the most recent checkpoint, or None if there is none.

    Prefers the latest.txt pointer and falls back to filename ordering if the
    pointer is missing or stale - a run killed between the rename and the
    pointer write must still be resumable.
    """
    directory = Path(directory)
    pointer = directory / LATEST_LINK_NAME
    if pointer.is_file():
        candidate = directory / pointer.read_text(encoding="utf-8").strip()
        if candidate.is_file():
            return candidate
        logger.warning("Stale %s pointer; falling back to filename order", pointer)

    checkpoints = list_checkpoints(directory)
    return checkpoints[-1] if checkpoints else None


def _rotate(directory: Path, keep_last: int) -> None:
    """Delete all but the newest keep_last checkpoints."""
    if keep_last < 1:
        return
    checkpoints = list_checkpoints(directory)
    for stale in checkpoints[:-keep_last]:
        try:
            stale.unlink()
            logger.debug("Rotated out old checkpoint: %s", stale)
        except OSError as exc:  # pragma: no cover - filesystem dependent
            logger.warning("Could not remove %s: %s", stale, exc)
