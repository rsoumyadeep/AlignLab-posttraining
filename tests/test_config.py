"""Hydra configuration composition.

Two things are being protected here:

1. Composition works and overrides apply. A config system that silently
   ignores an override is worse than no config system.

2. configs/env/server.yaml is structurally valid but has genuinely mandatory
   path values. It must COMPOSE (so the schema is checked and typos surface)
   while REFUSING to hand out an invented path. That is the mechanism stopping
   anyone from running on the server against fabricated directories before the
   machine has actually been probed.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import MissingMandatoryValue, OmegaConf

from alignlab.config_schema import AlignLabConfig, register_configs

CONFIG_DIR = str(Path(__file__).resolve().parents[1] / "configs")


def _compose(overrides: list[str] | None = None):
    register_configs()
    with initialize_config_dir(version_base=None, config_dir=CONFIG_DIR):
        return compose(config_name="config", overrides=overrides or [])


def test_default_config_composes() -> None:
    cfg = _compose()
    assert cfg.experiment == "smoke"
    assert cfg.reproducibility.seed == 42
    assert cfg.env.name == "local"


def test_local_env_is_cpu_and_expects_no_cuda() -> None:
    """Encodes the VERIFIED local reality: torch here is a CPU-only build."""
    cfg = _compose(["env=local"])
    assert cfg.env.device == "cpu"
    assert cfg.env.expect_cuda is False


def test_cli_override_applies() -> None:
    cfg = _compose(["train.max_steps=123", "reproducibility.seed=7"])
    assert cfg.train.max_steps == 123
    assert cfg.reproducibility.seed == 7


def test_tracking_group_switch() -> None:
    assert _compose().tracking.mode == "noop"
    assert _compose(["tracking=wandb_offline"]).tracking.mode == "offline"
    assert _compose(["tracking=wandb_online"]).tracking.mode == "online"


def test_server_env_composes_structurally() -> None:
    """The server config must be a valid document, not a broken one."""
    cfg = _compose(["env=server"])
    assert cfg.env.name == "server"
    assert cfg.env.expect_cuda is True
    assert cfg.env.device == "auto"


def test_server_paths_are_mandatory_and_unset() -> None:
    """The core guard: server paths raise rather than resolve to a guess.

    If this test ever fails because the values are populated, that is fine -
    but only once scripts/server_probe.sh has been run and the values are
    backed by real output. Never populate them to make this test pass.
    """
    cfg = _compose(["env=server"])
    for field in ("output_root", "checkpoint_root", "cache_root"):
        with pytest.raises(MissingMandatoryValue):
            _ = getattr(cfg.env, field)


def test_server_config_is_polite_about_shared_cpu() -> None:
    """64 threads exist on the server but it is shared; the default stays low."""
    cfg = _compose(["env=server"])
    assert cfg.env.num_workers <= 8


def test_schema_rejects_unknown_key() -> None:
    """A typo must fail at compose time, not be silently ignored at runtime."""
    with pytest.raises(Exception):
        _compose(["train.learing_rate=3e-4"])


def test_resolved_config_is_serialisable() -> None:
    """The manifest stores the resolved config, so it must convert cleanly."""
    cfg = _compose()
    resolved = OmegaConf.to_container(cfg, resolve=True)
    assert isinstance(resolved, dict)
    assert resolved["train"]["max_steps"] == 20


def test_schema_dataclass_defaults() -> None:
    """Defaults live in one place; the YAML should not silently disagree."""
    schema = AlignLabConfig()
    assert schema.reproducibility.seed == 42
    assert schema.reproducibility.deterministic is False
    assert schema.preemption.requeue is False
    assert schema.tracking.mode == "noop"
