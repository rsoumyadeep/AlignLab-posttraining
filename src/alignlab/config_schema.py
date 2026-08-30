"""Structured configuration schema.

These dataclasses are registered with Hydra's ConfigStore so that composition
is type-checked. The point is that a typo fails loudly at compose time rather
than silently at training time - "learing_rate: 3e-4" in a YAML file would
otherwise be accepted, ignored, and quietly produce a run at the default
learning rate that nobody notices for hours.

The env group is the mechanism that keeps machine-specific paths out of the
source tree. configs/env/local.yaml leaves its roots empty, meaning "fall back
to the ALIGNLAB_* environment variables, then to repo-relative defaults".
configs/env/server.yaml carries real paths VERIFIED on the machine (they held
Hydra MISSING until the server was actually probed in Phase 1B).

Those configured values are authoritative: alignlab.paths resolves
config -> environment variable -> repo-relative default, in that order. Until
the Phase 1C follow-up they were decorative - the path layer consulted only the
environment, so a value written here had no effect.
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
    # Storage-aware default, not a universal requirement. A full-parameter SFT
    # checkpoint for a 1.5B model is an ESTIMATED ~15.5 GB (bf16 weights plus
    # two fp32 AdamW moment tensors), so keeping 3 would consume ~46 GB of the
    # department server's limited free space from a single run. LoRA
    # checkpoints are ~100x smaller and can safely raise this per experiment:
    #     python -m alignlab.train train.keep_last_checkpoints=3
    keep_last_checkpoints: int = 1
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
class ModelConfig:
    """The pretrained model to fine-tune.

    ``revision`` is not optional metadata. Hugging Face repository ``main``
    branches move, so a run pinned only by model id is not reproducible. Phase
    2 recorded the SHA; Phase 3 uses it for every load.
    """

    id: str = "Qwen/Qwen2.5-1.5B"
    revision: str = "8faed761d45a263340a0528343f099c05c9a4323"
    # bfloat16 on Ampere+, float32 on CPU. "auto" resolves from the device.
    dtype: str = "auto"
    attn_implementation: str = "sdpa"
    gradient_checkpointing: bool = True


@dataclass
class DataConfig:
    """Instruction dataset selection and subsampling."""

    name: str = "HuggingFaceH4/no_robots"
    revision: str = "main"
    train_split: str = "train_sft"
    eval_split: str = "test_sft"
    # None means "use the whole split". Subsampling is shuffled with the
    # global seed, never head-truncated - no_robots is grouped by category, so
    # taking the first N rows would silently bias the task mix.
    max_train: int | None = None
    max_eval: int | None = 200


@dataclass
class SFTHyperParams:
    """Supervised fine-tuning hyperparameters.

    Distinct from TrainConfig, which drives the Phase 1 smoke test. Merging
    them would force one set of defaults to be wrong for one of the two.
    """

    max_length: int = 1024
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 8
    learning_rate: float = 2.0e-5
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.03
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    num_train_epochs: float = 1.0
    # -1 means "use num_train_epochs". A positive value caps the run, which is
    # what keeps a first run on a shared, storage-constrained server short.
    max_steps: int = -1
    logging_steps: int = 5
    eval_steps: int = 50
    save_steps: int = 100
    save_total_limit: int = 1
    seed: int = 42
    # packing concatenates examples to fill the context. It raises throughput
    # and makes the loss mask far harder to verify by hand, so Phase 3 leaves
    # it OFF deliberately - see docs/phase3. This is a verifiability-over-speed
    # decision, recorded rather than defaulted into.
    packing: bool = False
    # None lets TRL auto-resolve from the dataset shape (True for
    # prompt/completion). Set explicitly only to demonstrate the broken arm.
    completion_only_loss: bool | None = None
    resume: bool = True


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


@dataclass
class SFTExperimentConfig:
    """Top-level configuration for the Phase 3 SFT entrypoint.

    A separate root from AlignLabConfig rather than an extension of it: the
    Phase 1 smoke test and a 1.5B fine-tune share storage, logging, tracking
    and seeding, but nothing else. Bolting model/data groups onto the smoke
    test's schema would make every Phase 1 config carry fields it must ignore.
    """

    env: EnvConfig = field(default_factory=EnvConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    sft: SFTHyperParams = field(default_factory=SFTHyperParams)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    reproducibility: ReproducibilityConfig = field(
        default_factory=ReproducibilityConfig
    )

    run_name: str = ""
    experiment: str = "sft"
    # Refuse to start unless the volume can hold the checkpoints this run will
    # write. Overridable, but never silently.
    storage_guard: bool = True
    allow_low_disk: bool = False
    extras: dict[str, Any] = field(default_factory=dict)


def register_configs() -> None:
    """Register the schema with Hydra's ConfigStore.

    Idempotent - safe to call from several entrypoints and from tests.
    """
    from hydra.core.config_store import ConfigStore

    store = ConfigStore.instance()
    store.store(name="alignlab_schema", node=AlignLabConfig)
    store.store(name="alignlab_sft_schema", node=SFTExperimentConfig)
