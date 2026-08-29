"""Structured configuration schema.

These dataclasses are registered with Hydra's ConfigStore so that composition
is type-checked. The point is that a typo fails loudly at compose time rather
than silently at training time - "learing_rate: 3e-4" in a YAML file would
otherwise be accepted, ignored, and quietly produce a run at the default
learning rate that nobody notices for hours.

The env group is the mechanism that keeps machine-specific paths out of the
source tree. configs/env/local.yaml carries real local defaults;
configs/env/server.yaml carries MISSING (???) for every path, because the
department server has not been probed and inventing a path would be a
fabrication. Hydra raises on an unresolved MISSING value, so the server config
cannot be used until someone fills it in with verified values.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from omegaconf import MISSING


@dataclass
class EnvConfig:
    """Machine-specific settings. One file per environment."""

    name: str = MISSING
    # Storage roots. Empty string means "fall back to the environment variable
    # or the repository-relative default" (see alignlab.paths).
    output_root: str = ""
    checkpoint_root: str = ""
    cache_root: str = ""
    # Device request passed to alignlab.device.resolve_device.
    device: str = "auto"
    # Dataloader workers. The server has 64 threads but is SHARED, so a polite
    # default belongs in the server config, not here.
    num_workers: int = 0
    # Whether this environment is expected to have a GPU. Used only to warn
    # when expectation and reality disagree - never to fake a capability.
    expect_cuda: bool = False


@dataclass
class TrainConfig:
    """Training hyperparameters.

    Phase 1 uses these only to drive the smoke test. Real training arrives in
    Phase 3; the fields exist now so the config surface is stable.
    """

    max_steps: int = 20
    batch_size: int = 8
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    grad_clip: float | None = 1.0
    log_every: int = 5
    checkpoint_every: int = 10
    keep_last_checkpoints: int = 3
    resume: bool = True


@dataclass
class LoggingConfig:
    """Console and file logging."""

    level: str = "INFO"
    file_level: str = "DEBUG"


@dataclass
class TrackingConfig:
    """Experiment tracking.

    Default is "noop" so that nothing in the foundation requires an account or
    a network connection. Whether the department server can reach
    api.wandb.ai is UNVERIFIED; "offline" plus a later "wandb sync" is the
    fallback if it cannot.
    """

    mode: str = "noop"  # noop | offline | online
    project: str = "alignlab"
    entity: str | None = None
    tags: list[str] = field(default_factory=list)


@dataclass
class ReproducibilityConfig:
    """Seeding and determinism.

    deterministic is False by default because it costs throughput. See
    alignlab.seeding for the Tier A/B/C definition of what reproducibility
    means across two dissimilar machines.
    """

    seed: int = 42
    deterministic: bool = False


@dataclass
class PreemptionConfig:
    """Graceful shutdown behaviour."""

    enabled: bool = True
    # Ask SLURM to requeue after checkpointing. Default False: whether the
    # server runs SLURM is UNKNOWN, and this path is NOT TESTED.
    requeue: bool = False


@dataclass
class AlignLabConfig:
    """Top-level configuration."""

    env: EnvConfig = field(default_factory=EnvConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    reproducibility: ReproducibilityConfig = field(
        default_factory=ReproducibilityConfig
    )
    preemption: PreemptionConfig = field(default_factory=PreemptionConfig)

    # Run identification. Empty means "generate a timestamped name".
    run_name: str = ""
    experiment: str = "smoke"
    extras: dict[str, Any] = field(default_factory=dict)


def register_configs() -> None:
    """Register the schema with Hydra's ConfigStore.

    Idempotent - safe to call from several entrypoints and from tests.
    """
    from hydra.core.config_store import ConfigStore

    store = ConfigStore.instance()
    store.store(name="alignlab_schema", node=AlignLabConfig)
