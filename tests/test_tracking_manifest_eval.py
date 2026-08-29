"""Experiment tracking, run manifests and the evaluation harness.

No test here touches the network. The W&B offline test writes to a temporary
directory and is the strongest claim made about W&B in Phase 1A: ONLINE MODE
IS NOT TESTED, because it needs an API key and outbound access that has not
been established on either machine.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from alignlab.evaluation import EvalResult, load_report, run_evaluation
from alignlab.manifest import (
    MANIFEST_FILE_NAME,
    RunManifest,
    capture_environment,
    config_hash,
    git_info,
    load_manifest,
    package_versions,
    relevant_env_vars,
)
from alignlab.tracking import (
    MODE_NOOP,
    NoOpTracker,
    Tracker,
    build_tracker,
    write_metrics_jsonl,
)

# --------------------------------------------------------------------------
# tracking
# --------------------------------------------------------------------------


def test_noop_tracker_satisfies_protocol() -> None:
    assert isinstance(NoOpTracker(), Tracker)


def test_build_tracker_defaults_to_noop() -> None:
    tracker = build_tracker(mode=MODE_NOOP)
    assert isinstance(tracker, NoOpTracker)


def test_invalid_mode_raises() -> None:
    with pytest.raises(ValueError, match="tracking mode"):
        build_tracker(mode="tensorboard")


def test_noop_tracker_records_metrics_for_assertions() -> None:
    """The noop backend keeps metrics in memory so tests can inspect them."""
    tracker = NoOpTracker()
    tracker.log_config({"seed": 42})
    tracker.log_metrics({"loss": 0.5}, step=1)
    tracker.log_metrics({"loss": 0.4}, step=2)
    tracker.finish()

    assert tracker.logged_metrics == [({"loss": 0.5}, 1), ({"loss": 0.4}, 2)]


def test_metrics_jsonl_is_appended(tmp_path: Path) -> None:
    """A dependency-free local record that survives whatever the tracker does."""
    path = tmp_path / "metrics.jsonl"
    write_metrics_jsonl(path, {"loss": 1.0}, step=1)
    write_metrics_jsonl(path, {"loss": 0.5}, step=2)

    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert [r["step"] for r in records] == [1, 2]
    assert records[1]["loss"] == 0.5


@pytest.mark.slow
def test_wandb_offline_writes_locally(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Offline W&B must work with no credentials and no network.

    This is the fallback path if the department server turns out to have no
    outbound internet access, which is UNVERIFIED.
    """
    wandb = pytest.importorskip("wandb")
    monkeypatch.setenv("WANDB_MODE", "offline")
    monkeypatch.setenv("WANDB_DIR", str(tmp_path))
    monkeypatch.setenv("WANDB_SILENT", "true")

    tracker = build_tracker(
        mode="offline", project="alignlab-test", run_name="offline-test",
        run_dir=tmp_path, config={"seed": 42},
    )
    tracker.log_metrics({"loss": 0.25}, step=1)
    tracker.finish()

    assert (tmp_path / "wandb").is_dir()
    assert wandb is not None


# --------------------------------------------------------------------------
# manifest
# --------------------------------------------------------------------------


def test_config_hash_is_order_independent() -> None:
    """Two structurally identical configs must hash identically.

    This is the mechanism behind Tier B cross-machine reproducibility: the
    local run and the server run share this hash even though their floats
    will not match.
    """
    assert config_hash({"a": 1, "b": 2}) == config_hash({"b": 2, "a": 1})


def test_config_hash_detects_change() -> None:
    assert config_hash({"lr": 1e-3}) != config_hash({"lr": 3e-4})


def test_capture_environment_populates_required_fields() -> None:
    manifest = capture_environment(
        run_name="test-run", config={"seed": 42}, seed=42, notes={"phase": "1A"}
    )
    assert isinstance(manifest, RunManifest)
    assert manifest.run_name == "test-run"
    assert manifest.seed == 42
    assert manifest.config_sha256
    assert manifest.platform["hostname"]
    assert manifest.platform["python_version"]
    assert manifest.hardware["torch_version"]
    assert manifest.packages["torch"]
    assert manifest.environment in ("local", "server")
    assert manifest.python_executable
    assert manifest.notes["phase"] == "1A"


def test_manifest_records_git_provenance() -> None:
    """The dirty flag matters more than the SHA.

    A result from a dirty tree is not reproducible from the recorded commit,
    and the manifest must say so rather than imply clean provenance.
    """
    info = git_info()
    assert set(info) >= {"commit", "branch", "dirty", "describe"}
    if info["commit"] is not None:
        assert len(info["commit"]) == 40
        assert isinstance(info["dirty"], bool)


def test_manifest_round_trips_through_disk(tmp_path: Path) -> None:
    manifest = capture_environment(run_name="disk-test", config={"a": 1}, seed=1)
    path = manifest.write(tmp_path)

    assert path.name == MANIFEST_FILE_NAME
    loaded = load_manifest(path)
    assert loaded["run_name"] == "disk-test"
    assert loaded["config_sha256"] == manifest.config_sha256


def test_env_var_capture_is_an_allowlist(monkeypatch: pytest.MonkeyPatch) -> None:
    """Dumping os.environ would leak credentials into a shareable file."""
    monkeypatch.setenv("MY_SECRET_TOKEN", "hunter2")
    captured = relevant_env_vars()
    assert "MY_SECRET_TOKEN" not in captured
    assert "ALIGNLAB_ENV" in captured
    assert "CUDA_VISIBLE_DEVICES" in captured


def test_absent_packages_recorded_as_none() -> None:
    """Distinguishes "not installed" from "not checked"."""
    versions = package_versions()
    assert versions["torch"] is not None
    # transformers is NOT a Phase 1 dependency; it belongs to the train extra.
    assert "transformers" in versions


# --------------------------------------------------------------------------
# evaluation harness
# --------------------------------------------------------------------------


class _WorkingEvaluator:
    name = "working"

    def evaluate(self, model, **kwargs) -> EvalResult:
        return EvalResult(name=self.name, metrics={"score": 0.75}, n_examples=4)


class _BrokenEvaluator:
    name = "broken"

    def evaluate(self, model, **kwargs) -> EvalResult:
        raise RuntimeError("deliberate failure")


def test_evaluation_collects_results() -> None:
    report = run_evaluation(
        model=object(), evaluators=[_WorkingEvaluator()], run_name="eval-test"
    )
    assert len(report.results) == 1
    assert report.metric("working", "score") == 0.75
    assert report.results[0].status == "MEASURED"


def test_broken_evaluator_is_recorded_not_fatal() -> None:
    """One broken evaluator must not discard the metrics that did work.

    The failure is recorded as NOT_TESTED with the exception text - never as a
    zero, which would later be mistaken for a real measurement.
    """
    report = run_evaluation(
        model=object(),
        evaluators=[_WorkingEvaluator(), _BrokenEvaluator()],
        run_name="eval-test",
    )
    assert len(report.results) == 2

    broken = next(r for r in report.results if r.name == "broken")
    assert broken.status == "NOT_TESTED"
    assert broken.metrics == {}
    assert "RuntimeError" in broken.notes

    assert report.metric("working", "score") == 0.75


def test_missing_metric_returns_none() -> None:
    report = run_evaluation(
        model=object(), evaluators=[_WorkingEvaluator()], run_name="eval-test"
    )
    assert report.metric("working", "nonexistent") is None
    assert report.metric("nonexistent", "score") is None


def test_report_round_trips_through_disk(tmp_path: Path) -> None:
    report = run_evaluation(
        model=object(),
        evaluators=[_WorkingEvaluator()],
        run_name="eval-test",
        model_description="unit test object",
    )
    path = report.write(tmp_path)
    loaded = load_report(path)
    assert loaded["run_name"] == "eval-test"
    assert loaded["results"][0]["metrics"]["score"] == 0.75


def test_manifest_records_hf_cache_location(monkeypatch: pytest.MonkeyPatch) -> None:
    """The manifest must record WHERE a model cache lives.

    Phase 1C: HF_HUB_CACHE is the variable AlignLab actually sets, and it was
    initially missing from the allowlist - so a run could relocate its model
    cache without the manifest showing it.
    """
    monkeypatch.setenv("HF_HUB_CACHE", "/some/cache/hub")
    captured = relevant_env_vars()
    assert captured["HF_HUB_CACHE"] == "/some/cache/hub"
    assert "HF_DATASETS_CACHE" in captured
