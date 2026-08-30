"""Disk pre-flight guards.

Phase 1 identified "no pre-flight disk check in code" as an open prerequisite
and Phase 2 carried it forward. Phase 3 is where it stops being optional,
because Phase 3 is the first phase that writes multi-gigabyte files.

THE ACTUAL RISK, measured rather than imagined. The department server's /data
volume was 99% full with 98 GiB free on 2026-08-30, is shared with eight other
users, and holds every AlignLab artifact. The 2.9 GiB model download is not
what threatens it. A full-parameter AdamW checkpoint for a 1.5B model is:

    bf16 parameters          1.54e9 * 2  =  3.1 GB
    fp32 AdamW exp_avg       1.54e9 * 4  =  6.2 GB
    fp32 AdamW exp_avg_sq    1.54e9 * 4  =  6.2 GB
    fp32 master weights      1.54e9 * 4  =  6.2 GB   (mixed-precision optimisers)
                                            -------
                                            ~21.7 GB

and checkpoint rotation transiently holds two. That is the operation that can
exhaust the volume, and it fails halfway through a write, leaving a corrupt
checkpoint and an unhappy filesystem shared with eight other people.

DESIGN. Two independent things live here:

    estimate_*  - arithmetic, no filesystem access, unit-testable
    require_*   - the guard, which raises InsufficientStorage BEFORE the
                  expensive operation starts

The guard refuses by raising, never by silently continuing with a warning. A
warning in a log nobody reads is not a guard. Callers that genuinely want to
proceed anyway must pass an explicit override, which is recorded.

HEADROOM. Every check reserves headroom beyond the estimate. Estimates are
estimates; a shared volume can lose space to another user mid-run; and a
filesystem at 100% misbehaves in ways that are far worse than a clean refusal.
The default 10% (minimum 2 GiB) is a judgement call, not a measured optimum,
and is documented as such.
"""

from __future__ import annotations

import shutil
from dataclasses import asdict, dataclass
from pathlib import Path

from alignlab.logging_utils import get_logger

logger = get_logger(__name__)

GIB = 1024**3

# Reserve beyond the estimate. Not a measured optimum - a deliberate margin
# for estimate error and for other users of a shared volume.
DEFAULT_HEADROOM_FRACTION = 0.10
MIN_HEADROOM_BYTES = 2 * GIB

# Bytes per element, by dtype name. Only what AlignLab actually uses.
DTYPE_BYTES = {
    "float32": 4,
    "fp32": 4,
    "float16": 2,
    "fp16": 2,
    "bfloat16": 2,
    "bf16": 2,
    "int8": 1,
    "int4": 0.5,
}


class InsufficientStorage(RuntimeError):
    """Raised when an operation would risk exhausting the target volume."""


@dataclass(frozen=True)
class DiskStatus:
    """A point-in-time reading of one filesystem."""

    path: str
    total_bytes: int
    used_bytes: int
    free_bytes: int

    @property
    def percent_used(self) -> float:
        if self.total_bytes == 0:
            return 0.0
        return 100.0 * self.used_bytes / self.total_bytes

    def to_dict(self) -> dict[str, object]:
        data = asdict(self)
        data["percent_used"] = round(self.percent_used, 2)
        data["free_gib"] = round(self.free_bytes / GIB, 3)
        data["total_gib"] = round(self.total_bytes / GIB, 3)
        return data

    def __str__(self) -> str:
        return (
            f"{self.path}: {self.free_bytes / GIB:.1f} GiB free "
            f"of {self.total_bytes / GIB:.1f} GiB ({self.percent_used:.0f}% used)"
        )


def disk_status(path: str | Path) -> DiskStatus:
    """Read free/used/total for the filesystem containing ``path``.

    The path need not exist: the nearest existing ancestor is measured, which
    is what makes this usable as a pre-flight check for a directory the
    operation is about to create.
    """
    target = Path(path).expanduser().resolve()
    probe = target
    while not probe.exists():
        parent = probe.parent
        if parent == probe:  # reached the root without finding anything
            break
        probe = parent

    usage = shutil.disk_usage(probe)
    return DiskStatus(
        path=str(target),
        total_bytes=usage.total,
        used_bytes=usage.used,
        free_bytes=usage.free,
    )


def estimate_checkpoint_bytes(
    n_params: int,
    param_dtype: str = "bfloat16",
    optimizer: str | None = "adamw",
    master_weights: bool = True,
    optimizer_state_dtype: str = "float32",
) -> int:
    """Estimate one full-parameter checkpoint on disk, in bytes.

    This is ARITHMETIC, not a measurement. It is deliberately conservative:
    where a framework might or might not write a tensor, it is counted.

    HOW THE DEFAULT COMPARES TO REALITY - MEASURED, 2026-08-30. A real
    Qwen2.5-1.5B checkpoint written by TRL/transformers with bf16=True was:

        model.safetensors   3,087,467,144 B   = 2.00 bytes/param  (bf16)
        optimizer.pt        6,175,148,456 B   = 4.00 bytes/param
        total               9,262,615,600 B   = 6.00 bytes/param  = 8.63 GiB

    The default estimate for the same model is 20.13 GiB (14 bytes/param). It
    over-predicts by 2.33x, because the default assumes fp32 AdamW moments
    (8 bytes) plus an fp32 master copy (4 bytes), while this configuration
    stores both moments in bf16 (4 bytes total) and keeps no separate master
    copy - pure bf16 training rather than mixed precision with fp32 masters.

    THE OVER-PREDICTION IS KEPT AS THE DEFAULT ON PURPOSE. A disk guard that
    under-predicts fails halfway through writing a 9 GiB file on a shared
    volume; one that over-predicts refuses a run that would have fitted. Those
    are not symmetric costs. Callers who have MEASURED their own configuration
    can pass ``optimizer_state_dtype="bfloat16"`` and ``master_weights=False``
    to get the tighter figure.

    Args:
        n_params: trainable parameter count.
        param_dtype: storage dtype of the parameters themselves.
        optimizer: "adamw" (two fp32 moments per parameter), "sgd" (one fp32
            momentum buffer), or None for weights only.
        master_weights: whether a separate fp32 copy of the parameters is
            written, as mixed-precision optimisers keep. Counted by default
            because omitting it is the optimistic assumption, and an
            optimistic disk estimate is the useless kind.

    Returns:
        Estimated bytes. Compare against real measurements - Phase 3 records
        the actual figure beside this estimate.
    """
    if n_params < 0:
        raise ValueError(f"n_params must be non-negative, got {n_params}")

    per_param = float(DTYPE_BYTES.get(param_dtype, 4))

    state_bytes = float(DTYPE_BYTES.get(optimizer_state_dtype, 4))

    if optimizer is None:
        opt_per_param = 0.0
    elif optimizer.lower() in ("adamw", "adam"):
        opt_per_param = 2 * state_bytes  # exp_avg + exp_avg_sq
    elif optimizer.lower() == "sgd":
        opt_per_param = state_bytes  # momentum buffer
    else:
        raise ValueError(f"unknown optimizer {optimizer!r}")

    master_per_param = 4.0 if (master_weights and optimizer is not None) else 0.0

    return int(n_params * (per_param + opt_per_param + master_per_param))


def estimate_lora_checkpoint_bytes(
    n_trainable: int,
    param_dtype: str = "float32",
    optimizer: str | None = "adamw",
) -> int:
    """Estimate a LoRA adapter checkpoint.

    Separate from ``estimate_checkpoint_bytes`` only for readability: an
    adapter checkpoint stores no frozen base weights and no master copy of
    them, so the same arithmetic over the *trainable* count is the whole story.
    Phase 4 uses this to justify raising ``keep_last_checkpoints``.
    """
    return estimate_checkpoint_bytes(
        n_trainable,
        param_dtype=param_dtype,
        optimizer=optimizer,
        master_weights=False,
    )


def headroom_bytes(required_bytes: int, fraction: float = DEFAULT_HEADROOM_FRACTION) -> int:
    """Headroom to reserve on top of an estimate."""
    return max(int(required_bytes * fraction), MIN_HEADROOM_BYTES)


def require_free_space(
    path: str | Path,
    required_bytes: int,
    label: str = "operation",
    headroom_fraction: float = DEFAULT_HEADROOM_FRACTION,
    allow_override: bool = False,
) -> DiskStatus:
    """Refuse to proceed unless ``path``'s volume has room, plus headroom.

    Raises:
        InsufficientStorage: unless ``allow_override``, in which case the
            shortfall is logged at WARNING and the caller proceeds. The
            override exists so a human can make an informed decision; it is
            never the default, and it is always recorded.

    Returns:
        The DiskStatus that was checked, so callers can record it in a manifest.
    """
    status = disk_status(path)
    reserve = headroom_bytes(required_bytes, headroom_fraction)
    needed = required_bytes + reserve

    message = (
        f"{label}: needs ~{required_bytes / GIB:.2f} GiB "
        f"+ {reserve / GIB:.2f} GiB headroom = {needed / GIB:.2f} GiB; "
        f"{status}"
    )

    if status.free_bytes >= needed:
        logger.info("disk pre-flight OK - %s", message)
        return status

    shortfall = needed - status.free_bytes
    detail = f"INSUFFICIENT STORAGE - {message}; short by {shortfall / GIB:.2f} GiB"

    if allow_override:
        logger.warning("%s - PROCEEDING ANYWAY because override was requested", detail)
        return status

    raise InsufficientStorage(
        f"{detail}.\n"
        f"Refusing to start. Options: free space, reduce "
        f"train.keep_last_checkpoints, use LoRA (Phase 4) whose checkpoints are "
        f"orders of magnitude smaller, or re-run with an explicit override if "
        f"you have verified the risk."
    )


def require_free_space_for_checkpoints(
    path: str | Path,
    n_params: int,
    keep_last: int,
    param_dtype: str = "bfloat16",
    optimizer: str | None = "adamw",
    label: str = "training checkpoints",
    allow_override: bool = False,
) -> DiskStatus:
    """Guard a training run against its own checkpoint footprint.

    Budgets ``keep_last + 1`` checkpoints, not ``keep_last``. Rotation writes
    the new checkpoint before deleting the oldest - deleting first would risk
    destroying the only good checkpoint if the write then failed - so peak
    usage is one more than the retention setting.
    """
    per_checkpoint = estimate_checkpoint_bytes(
        n_params, param_dtype=param_dtype, optimizer=optimizer
    )
    peak = per_checkpoint * (max(keep_last, 0) + 1)
    return require_free_space(
        path,
        peak,
        label=(
            f"{label} ({keep_last}+1 x {per_checkpoint / GIB:.2f} GiB, "
            f"{n_params:,} params)"
        ),
        allow_override=allow_override,
    )
