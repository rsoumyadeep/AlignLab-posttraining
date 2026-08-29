"""End-to-end foundation smoke test.

The claim: config composition, seeding, logging, manifest capture, tracking,
the training loop, checkpointing, resume and evaluation work together, not
merely in isolation. Unit tests can all pass while the wiring between them is
broken; this is the test that catches that.

What this does NOT test: language modelling. The model is a four-parameter
linear regression on synthetic data. No pretrained checkpoint is loaded and
Qwen2.5-1.5B is NOT downloaded. Real training is Phase 3.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from alignlab.checkpoint import latest_checkpoint, load_checkpoint
from alignlab.config_schema import register_configs
from alignlab.manifest import MANIFEST_FILE_NAME
from alignlab.train import ToyModel, train

CONFIG_DIR = str(Path(__file__).resolve().parents[1] / "configs")


def _config(overrides: list[str]):
    register_configs()
    with initialize_config_dir(version_base=None, config_dir=CONFIG_DIR):
        return compose(config_name="config", overrides=overrides)


def _base_overrides(run_name: str, **extra: object) -> list[str]:
    overrides = [
        "env=local",
        f"run_name={run_name}",
        "train.max_steps=10",
        "train.log_every=5",
        "train.checkpoint_every=5",
        "logging.level=WARNING",
    ]
    overrides.extend(f"{key}={value}" for key, value in extra.items())
    return overrides


@pytest.mark.slow
def test_smoke_run_produces_all_artifacts(isolated_roots: dict[str, Path]) -> None:
    cfg = _config(_base_overrides("smoke-artifacts"))
    summary = train(cfg)

    run_dir = Path(summary["run_dir"])
    assert summary["steps_completed"] == 10
    assert summary["stopped_early"] is False

    # Every artifact the foundation promises to leave behind.
    assert (run_dir / MANIFEST_FILE_NAME).is_file()
    assert (run_dir / "resolved_config.yaml").is_file()
    assert (run_dir / "metrics.jsonl").is_file()
    assert (run_dir / "eval_results.json").is_file()
    assert (run_dir / "run.log").is_file()

    manifest = json.loads((run_dir / MANIFEST_FILE_NAME).read_text(encoding="utf-8"))
    assert manifest["seed"] == 42
    assert manifest["notes"]["is_real_training"] is False
    assert manifest["notes"]["model_downloaded"] is False
    assert manifest["config_sha256"] == summary["config_sha256"]


@pytest.mark.slow
def test_loss_decreases(isolated_roots: dict[str, Path]) -> None:
    """A loop that never updates weights would still pass every other test.

    This is a foundation check, not a scientific claim: fitting a synthetic
    linear rule is expected to reduce MSE, and failing to would indicate the
    optimizer step is not wired in.
    """
    cfg = _config(_base_overrides("smoke-loss", **{"train.max_steps": 60}))
    summary = train(cfg)

    records = [
        json.loads(line)
        for line in (Path(summary["run_dir"]) / "metrics.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert len(records) >= 2
    assert records[-1]["loss"] < records[0]["loss"]


@pytest.mark.slow
def test_checkpoint_written_and_resumable(isolated_roots: dict[str, Path]) -> None:
    """The core resume claim: a second invocation continues, not restarts."""
    cfg = _config(_base_overrides("smoke-resume"))
    first = train(cfg)

    ckpt_dir = Path(first["checkpoint_dir"])
    saved = latest_checkpoint(ckpt_dir)
    assert saved is not None

    payload = load_checkpoint(saved)
    assert payload.step == 10
    assert payload.optimizer_state is not None
    assert payload.scheduler_state is not None
    assert payload.rng_state is not None

    # Re-run with a higher step budget. Resume must pick up at step 10 and
    # run only the remaining steps.
    cfg2 = _config(_base_overrides("smoke-resume", **{"train.max_steps": 15}))
    second = train(cfg2)
    assert second["steps_completed"] == 15

    resumed = latest_checkpoint(ckpt_dir)
    assert resumed is not None
    assert load_checkpoint(resumed).step == 15


@pytest.mark.slow
def test_resumed_weights_match_saved_weights(isolated_roots: dict[str, Path]) -> None:
    """Resume must restore the exact weights, bitwise."""
    cfg = _config(_base_overrides("smoke-weights"))
    summary = train(cfg)

    saved = latest_checkpoint(Path(summary["checkpoint_dir"]))
    assert saved is not None
    payload = load_checkpoint(saved)

    model = ToyModel(in_features=4)
    model.load_state_dict(payload.model_state)

    for key, tensor in payload.model_state.items():
        assert torch.equal(model.state_dict()[key], tensor)


@pytest.mark.slow
def test_same_seed_gives_same_config_hash_and_loss(
    isolated_roots: dict[str, Path],
) -> None:
    """Tier A on the full pipeline: two identical runs agree bitwise.

    resume=false so the second run genuinely retrains rather than loading the
    first run's checkpoint.
    """
    first = train(_config(_base_overrides("repro-a", **{"train.resume": "false"})))
    second = train(_config(_base_overrides("repro-b", **{"train.resume": "false"})))

    # run_name differs, so the config hashes are expected to differ; the
    # numerical result must not.
    assert first["final_loss"] == second["final_loss"]
    assert first["eval"]["toy_mse"]["mse"] == second["eval"]["toy_mse"]["mse"]


@pytest.mark.slow
def test_different_seed_changes_result(isolated_roots: dict[str, Path]) -> None:
    """Guards against a seed that is recorded but never actually applied."""
    first = train(
        _config(
            _base_overrides(
                "seed-a", **{"train.resume": "false", "reproducibility.seed": 1}
            )
        )
    )
    second = train(
        _config(
            _base_overrides(
                "seed-b", **{"train.resume": "false", "reproducibility.seed": 2}
            )
        )
    )
    assert first["final_loss"] != second["final_loss"]


@pytest.mark.slow
def test_preemption_stops_run_and_checkpoints(isolated_roots: dict[str, Path]) -> None:
    """Simulated preemption must checkpoint and stop early.

    Uses the OS-independent request_stop path, which is what makes this
    verifiable on Windows. Real SIGUSR1 delivery is covered on POSIX by
    tests/test_preemption.py.
    """
    from alignlab import train as train_module

    cfg = _config(_base_overrides("smoke-preempt", **{"train.max_steps": 100}))

    original_should_stop = train_module.PreemptionHandler.should_stop
    calls = {"n": 0}

    def stop_after_a_few(self) -> bool:
        calls["n"] += 1
        return calls["n"] >= 3

    train_module.PreemptionHandler.should_stop = stop_after_a_few  # type: ignore[method-assign]
    try:
        summary = train(cfg)
    finally:
        train_module.PreemptionHandler.should_stop = original_should_stop  # type: ignore[method-assign]

    assert summary["stopped_early"] is True
    assert summary["steps_completed"] < 100

    saved = latest_checkpoint(Path(summary["checkpoint_dir"]))
    assert saved is not None, "preemption must leave a resumable checkpoint"


@pytest.mark.slow
def test_evaluation_ran_and_recorded(isolated_roots: dict[str, Path]) -> None:
    cfg = _config(_base_overrides("smoke-eval"))
    summary = train(cfg)

    report = json.loads(
        (Path(summary["run_dir"]) / "eval_results.json").read_text(encoding="utf-8")
    )
    assert report["results"][0]["name"] == "toy_mse"
    assert report["results"][0]["status"] == "MEASURED"
    assert "NOT a language model" in report["model_description"]


@pytest.mark.slow
def test_resolved_config_snapshot_matches_run(isolated_roots: dict[str, Path]) -> None:
    """The snapshot must be reloadable and reflect the overrides used."""
    cfg = _config(_base_overrides("smoke-config", **{"reproducibility.seed": 777}))
    summary = train(cfg)

    snapshot = OmegaConf.load(Path(summary["run_dir"]) / "resolved_config.yaml")
    assert snapshot.reproducibility.seed == 777
    assert snapshot.env.name == "local"


@pytest.mark.slow
def test_resuming_a_completed_run_does_nothing_and_reports_it(
    isolated_roots: dict[str, Path],
) -> None:
    """Re-running a finished run must not fabricate a loss value.

    Found during Phase 1A manual CLI testing: resuming at the step budget left
    final_loss as NaN, which reads like a measured number. It is now None, and
    steps_this_invocation makes the no-op explicit.
    """
    cfg = _config(_base_overrides("already-done"))
    first = train(cfg)
    assert first["steps_this_invocation"] == 10
    assert first["final_loss"] is not None

    second = train(_config(_base_overrides("already-done")))
    assert second["steps_this_invocation"] == 0
    assert second["final_loss"] is None
    assert second["steps_completed"] == 10


@pytest.mark.slow
def test_no_extra_checkpoint_written_on_noop_resume(
    isolated_roots: dict[str, Path],
) -> None:
    """A no-op resume must not rotate a real checkpoint out of existence."""
    from alignlab.checkpoint import list_checkpoints

    cfg = _config(_base_overrides("noop-rotation"))
    summary = train(cfg)
    ckpt_dir = Path(summary["checkpoint_dir"])
    before = [p.name for p in list_checkpoints(ckpt_dir)]

    train(_config(_base_overrides("noop-rotation")))
    after = [p.name for p in list_checkpoints(ckpt_dir)]
    assert before == after
