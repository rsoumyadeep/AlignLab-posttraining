"""Regression tests: configured storage roots must ACTUALLY be used.

THE BUG (found by the Phase 1C storage audit, fixed in the Phase 1C follow-up).

``cfg.env.output_root`` and ``cfg.env.checkpoint_root`` were DECORATIVE.
``train.py`` called ``run_dir(run_name)`` and ``checkpoint_root()`` with no
arguments, and those functions consulted only the ALIGNLAB_* environment
variables and a repo-relative default. The configured values were never read.

It stayed invisible on both machines for the worst possible reason: the
configured server values had been set to exactly the paths the repo-relative
default already produced, so config and reality agreed by coincidence. Changing
a path in ``configs/env/server.yaml`` would have had no effect at all.

How it surfaced: comparing a real ``run_manifest.json`` against the config. The
manifest reported ``cache_root = <repo>/.cache`` while ``server.yaml`` said
``<repo>/.cache/huggingface``. A one-line disagreement in generated output -
not a failing test.

WHY IT MATTERS: the env config group is the entire mechanism for keeping
machine-specific paths out of the source tree. If it does not take effect then,
on a storage-constrained shared server, a checkpoint intended for one volume
lands silently on another - and the manifest records the wrong location, so the
mistake is not even auditable afterwards.

THE FIX: ``alignlab.paths`` resolves in strict order

    configured value  ->  environment variable  ->  repo-relative default

and ``train.py`` passes ``cfg.env.*`` through. An empty configured value falls
through, which preserves the existing local behaviour exactly.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from alignlab.config_schema import register_configs
from alignlab.manifest import MANIFEST_FILE_NAME
from alignlab.paths import (
    ENV_VAR_CKPT_ROOT,
    ENV_VAR_OUTPUT_ROOT,
    checkpoint_root,
    describe_roots,
    output_root,
    repo_root,
    run_dir,
)
from alignlab.train import train

CONFIG_DIR = str(Path(__file__).resolve().parents[1] / "configs")


def _config(overrides: list[str]):
    register_configs()
    with initialize_config_dir(version_base=None, config_dir=CONFIG_DIR):
        return compose(config_name="config", overrides=overrides)


# --------------------------------------------------------------------------
# precedence
# --------------------------------------------------------------------------


def test_configured_output_root_wins_over_env_var(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The regression: a configured value must beat the environment variable."""
    monkeypatch.setenv(ENV_VAR_OUTPUT_ROOT, str(tmp_path / "from_env"))
    configured = tmp_path / "from_config"

    assert output_root(str(configured)) == configured.resolve()
    # ...and without it, the environment variable still applies
    assert output_root() == (tmp_path / "from_env").resolve()


def test_configured_checkpoint_root_wins_over_env_var(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(ENV_VAR_CKPT_ROOT, str(tmp_path / "from_env"))
    configured = tmp_path / "from_config"

    assert checkpoint_root(str(configured)) == configured.resolve()
    assert checkpoint_root() == (tmp_path / "from_env").resolve()


def test_empty_configured_value_falls_through(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Empty string means "not configured" - this preserves local behaviour.

    configs/env/local.yaml deliberately leaves the roots empty.
    """
    monkeypatch.setenv(ENV_VAR_OUTPUT_ROOT, str(tmp_path / "from_env"))
    assert output_root("") == (tmp_path / "from_env").resolve()
    assert output_root(None) == (tmp_path / "from_env").resolve()


def test_run_dir_honours_configured_root(tmp_path: Path) -> None:
    configured = tmp_path / "cfg_outputs"
    path = run_dir("my-run", configured=str(configured))
    assert path == (configured / "my-run").resolve()
    assert path.is_dir()


def test_describe_roots_reports_configured_values(tmp_path: Path) -> None:
    """The manifest must record what the run USED, not what the env implies."""
    described = describe_roots(
        output=tmp_path / "o", checkpoint=tmp_path / "c", cache=tmp_path / "k"
    )
    assert described["output_root"] == str((tmp_path / "o").resolve())
    assert described["checkpoint_root"] == str((tmp_path / "c").resolve())
    assert described["cache_root"] == str((tmp_path / "k").resolve())


def test_paths_never_silently_use_unrelated_location(tmp_path: Path) -> None:
    """A configured root must never be quietly replaced by a machine default.

    This is the assertion that would have failed before the fix.
    """
    configured = tmp_path / "explicit"
    resolved = output_root(str(configured))

    assert resolved == configured.resolve()
    assert not str(resolved).startswith(str(repo_root()))


# --------------------------------------------------------------------------
# end-to-end: a real run must write where the config says
# --------------------------------------------------------------------------


@pytest.mark.slow
def test_run_writes_to_configured_roots(tmp_path: Path) -> None:
    """End-to-end proof, and the test that would actually catch a regression.

    Deliberately points the config at directories that are NOT the
    repo-relative defaults and NOT any environment variable, then asserts the
    artifacts landed there.
    """
    out = tmp_path / "cfg_out"
    ckpt = tmp_path / "cfg_ckpt"
    cache = tmp_path / "cfg_cache"

    cfg = _config(
        [
            "env=local",
            f"env.output_root={out.as_posix()}",
            f"env.checkpoint_root={ckpt.as_posix()}",
            f"env.cache_root={cache.as_posix()}",
            "run_name=wiring-e2e",
            "train.max_steps=6",
            "train.checkpoint_every=3",
            "logging.level=WARNING",
        ]
    )
    summary = train(cfg)

    # outputs went to the configured output root
    assert Path(summary["run_dir"]) == (out / "wiring-e2e").resolve()
    assert (out / "wiring-e2e" / MANIFEST_FILE_NAME).is_file()
    assert (out / "wiring-e2e" / "metrics.jsonl").is_file()

    # checkpoints went to the configured checkpoint root
    assert Path(summary["checkpoint_dir"]) == (ckpt / "wiring-e2e").resolve()
    assert list((ckpt / "wiring-e2e").glob("step-*.pt"))

    # ...and nothing leaked into the repo-relative defaults
    assert not (repo_root() / "outputs" / "wiring-e2e").exists()
    assert not (repo_root() / "checkpoints" / "wiring-e2e").exists()


@pytest.mark.slow
def test_manifest_records_the_roots_actually_used(tmp_path: Path) -> None:
    """The manifest must not disagree with reality.

    The original bug was DISCOVERED by exactly this disagreement, so it gets a
    test of its own.
    """
    out = tmp_path / "m_out"
    ckpt = tmp_path / "m_ckpt"
    cache = tmp_path / "m_cache"

    cfg = _config(
        [
            "env=local",
            f"env.output_root={out.as_posix()}",
            f"env.checkpoint_root={ckpt.as_posix()}",
            f"env.cache_root={cache.as_posix()}",
            "run_name=manifest-truth",
            "train.max_steps=3",
            "logging.level=WARNING",
        ]
    )
    summary = train(cfg)

    manifest = json.loads(
        (Path(summary["run_dir"]) / MANIFEST_FILE_NAME).read_text(encoding="utf-8")
    )
    assert manifest["paths"]["output_root"] == str(out.resolve())
    assert manifest["paths"]["checkpoint_root"] == str(ckpt.resolve())
    assert manifest["paths"]["cache_root"] == str(cache.resolve())
    # the HF cache must sit under the configured cache root, not ~/.cache
    assert manifest["env_vars"]["HF_HUB_CACHE"].startswith(str(cache.resolve()))


# --------------------------------------------------------------------------
# existing Phase 1 behaviour must be unchanged
# --------------------------------------------------------------------------


def test_local_env_behaviour_unchanged(isolated_roots: dict[str, Path]) -> None:
    """env=local leaves roots empty, so environment variables still drive it."""
    cfg = _config(["env=local"])
    assert cfg.env.output_root == ""
    assert output_root(cfg.env.output_root or None) == isolated_roots["output"].resolve()
    assert (
        checkpoint_root(cfg.env.checkpoint_root or None)
        == isolated_roots["checkpoint"].resolve()
    )


def test_server_config_roots_are_absolute_and_used() -> None:
    """The server config carries real paths, and they resolve to themselves."""
    cfg = _config(["env=server"])
    resolved = OmegaConf.to_container(cfg, resolve=True)
    configured = resolved["env"]["checkpoint_root"]

    assert configured.startswith("/")
    # On the POSIX server this resolves unchanged; on Windows a drive letter is
    # prepended, so compare the tail rather than the whole string.
    assert checkpoint_root(configured).as_posix().endswith(configured.lstrip("/"))


# --------------------------------------------------------------------------
# checkpoint retention default (Phase 1C decision)
# --------------------------------------------------------------------------


def test_default_retention_is_one() -> None:
    """Storage-aware default: keep_last_checkpoints defaults to 1.

    Rationale in docs/phase1/STORAGE_POLICY.md - a full-SFT checkpoint for a
    1.5B model is an ESTIMATED ~15.5 GB, so keeping 3 would consume ~46 GB from
    one run on a storage-constrained shared volume.
    """
    assert _config([]).train.keep_last_checkpoints == 1


def test_retention_is_overridable_per_experiment() -> None:
    """It is a DEFAULT, not a hard-coded limit.

    LoRA checkpoints are far smaller and can safely keep more.
    """
    assert _config(["train.keep_last_checkpoints=3"]).train.keep_last_checkpoints == 3
    assert _config(["train.keep_last_checkpoints=5"]).train.keep_last_checkpoints == 5


@pytest.mark.slow
def test_retention_default_actually_rotates_to_one(tmp_path: Path) -> None:
    """The default must take effect in a real run, not merely in the schema."""
    from alignlab.checkpoint import list_checkpoints

    out = tmp_path / "r_out"
    ckpt = tmp_path / "r_ckpt"
    cfg = _config(
        [
            "env=local",
            f"env.output_root={out.as_posix()}",
            f"env.checkpoint_root={ckpt.as_posix()}",
            "run_name=retention",
            "train.max_steps=9",
            "train.checkpoint_every=3",
            "logging.level=WARNING",
        ]
    )
    summary = train(cfg)

    remaining = list_checkpoints(Path(summary["checkpoint_dir"]))
    assert len(remaining) == 1, f"expected 1 checkpoint retained, got {remaining}"
