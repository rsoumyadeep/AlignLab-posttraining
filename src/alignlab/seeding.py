"""Seeding and determinism.

Reproducibility in AlignLab is defined in three explicit tiers, because
claiming more than is achievable would be dishonest (PROJECT_INSTRUCTIONS
sections 15, 16 and 24):

    Tier A - BITWISE, same machine + same seed + same environment.
             Achievable and tested (tests/test_seeding.py).

    Tier B - STRUCTURAL, across machines. Same commit, same resolved config,
             same data version and same seed produce the same code path, the
             same tensor shapes and the same token accounting. Verified by
             comparing config hashes and manifests, NOT by comparing floats.

    Tier C - STATISTICAL, across machines or dtypes. Results are comparable in
             distribution, not identical. Any comparison between a CPU fp32 run
             and a GPU bf16 run is Tier C and must be labelled as such.

Bitwise agreement between the local Pascal/CPU machine and an Ampere GPU server
is NOT achievable, and this module does not pretend otherwise.
"""

from __future__ import annotations

import logging
import os
import random
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SeedReport:
    """What seeding actually did, recorded in the run manifest."""

    seed: int
    deterministic: bool
    cudnn_benchmark: bool
    cublas_workspace_config: str | None


def set_seed(seed: int, deterministic: bool = False) -> SeedReport:
    """Seed every RNG AlignLab can reach.

    Args:
        seed: the master seed. Applied to the stdlib random module, NumPy,
            torch CPU and all CUDA devices, and exported as PYTHONHASHSEED.
        deterministic: when True, additionally request deterministic algorithms
            from torch and cuDNN. OFF by default because it costs throughput
            and makes some ops raise rather than fall back to a
            non-deterministic kernel. That is a real trade, so it is an
            explicit flag rather than a silent default.

    Returns:
        A SeedReport describing exactly what was applied.

    Note on PYTHONHASHSEED: setting it here affects hash randomisation only for
    subprocesses spawned afterwards, not the already-running interpreter. It is
    set for the benefit of dataloader workers.
    """
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError(f"seed must be an int, got {type(seed).__name__}")

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    cublas_config: str | None = None
    if deterministic:
        # Required for deterministic CUDA matmuls on CUDA >= 10.2. Harmless on
        # CPU-only machines; set unconditionally so both environments agree.
        cublas_config = ":4096:8"
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = cublas_config
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True

    report = SeedReport(
        seed=seed,
        deterministic=deterministic,
        cudnn_benchmark=bool(torch.backends.cudnn.benchmark),
        cublas_workspace_config=cublas_config,
    )
    logger.debug("Seeded RNGs: %s", report)
    return report


def capture_rng_state() -> dict[str, Any]:
    """Snapshot every RNG state, for checkpointing.

    A checkpoint that restores weights but not RNG state does not truly resume:
    data order and dropout masks would diverge from the uninterrupted run.
    """
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: dict[str, Any]) -> None:
    """Restore RNG states captured by capture_rng_state.

    CUDA state is restored only if it was captured AND CUDA is available now,
    so a checkpoint written on the GPU server can still be inspected on the
    local CPU machine without raising.
    """
    if "python" in state:
        random.setstate(state["python"])
    if "numpy" in state:
        np.random.set_state(state["numpy"])
    if "torch" in state:
        torch.set_rng_state(_as_byte_tensor(state["torch"]))
    if "torch_cuda" in state and torch.cuda.is_available():
        try:
            torch.cuda.set_rng_state_all(
                [_as_byte_tensor(s) for s in state["torch_cuda"]]
            )
        except (RuntimeError, ValueError) as exc:
            # Device-count mismatch between the saving and loading machine.
            # Not fatal, but log loudly rather than silently pretend the
            # resume is exact.
            logger.warning(
                "Could not restore CUDA RNG state (device count mismatch?): %s", exc
            )


def _as_byte_tensor(value: Any) -> torch.Tensor:
    """torch RNG state must be a uint8 tensor on CPU."""
    tensor = value if isinstance(value, torch.Tensor) else torch.tensor(value)
    return tensor.cpu().to(torch.uint8)
