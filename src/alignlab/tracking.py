"""Experiment tracking behind a narrow interface.

Three backends implement the same Tracker protocol:

    NoOpTracker   - discards everything. Default in tests, so the suite needs
                    no account, no network and no credentials.
    WandbTracker  - Weights and Biases, online or offline. Offline writes a
                    local run directory that can be uploaded later with
                    "wandb sync", which is the fallback if the department
                    server turns out to have no outbound internet access
                    (UNVERIFIED at the time of writing).

Why an abstraction rather than calling wandb directly: the server's network
access is unknown, so the training code must not hard-depend on a reachable
tracking service. The interface is deliberately tiny - four methods - to avoid
the premature abstraction that PROJECT_INSTRUCTIONS section 22 warns against.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from alignlab.logging_utils import get_logger

logger = get_logger(__name__)

MODE_NOOP = "noop"
MODE_OFFLINE = "offline"
MODE_ONLINE = "online"
VALID_MODES = (MODE_NOOP, MODE_OFFLINE, MODE_ONLINE)


@runtime_checkable
class Tracker(Protocol):
    """The complete tracking surface AlignLab is allowed to depend on."""

    def log_config(self, config: dict[str, Any]) -> None: ...
    def log_metrics(self, metrics: dict[str, Any], step: int | None = None) -> None: ...
    def log_artifact(self, path: str | Path, name: str | None = None) -> None: ...
    def finish(self) -> None: ...


class NoOpTracker:
    """A tracker that records nothing.

    Not a stub for missing functionality - it is the correct backend for unit
    tests and for any run whose metrics are not worth keeping.
    """

    mode = MODE_NOOP

    def __init__(self, **_: Any) -> None:
        self.logged_metrics: list[tuple[dict[str, Any], int | None]] = []

    def log_config(self, config: dict[str, Any]) -> None:
        logger.debug("NoOpTracker.log_config (%d keys)", len(config))

    def log_metrics(self, metrics: dict[str, Any], step: int | None = None) -> None:
        # Retained in memory so tests can assert on what a training loop
        # reported without needing a real backend.
        self.logged_metrics.append((dict(metrics), step))

    def log_artifact(self, path: str | Path, name: str | None = None) -> None:
        logger.debug("NoOpTracker.log_artifact %s", path)

    def finish(self) -> None:
        logger.debug("NoOpTracker.finish")


class WandbTracker:
    """Weights and Biases backend.

    wandb is imported lazily so that the package remains an optional
    dependency: a machine without it can still run the whole foundation using
    the noop backend.
    """

    def __init__(
        self,
        project: str,
        run_name: str,
        mode: str = MODE_OFFLINE,
        entity: str | None = None,
        tags: list[str] | None = None,
        run_dir: str | Path | None = None,
        config: dict[str, Any] | None = None,
    ) -> None:
        if mode not in (MODE_OFFLINE, MODE_ONLINE):
            raise ValueError(f"WandbTracker mode must be offline or online, got {mode!r}")

        try:
            import wandb
        except ImportError as exc:  # pragma: no cover - depends on install extras
            raise ImportError(
                "wandb is not installed. Install the 'tracking' extra, or select "
                "tracking mode 'noop'."
            ) from exc

        self._wandb = wandb
        self.mode = mode
        self.run = wandb.init(
            project=project,
            name=run_name,
            entity=entity,
            tags=tags or [],
            mode=mode,
            dir=str(run_dir) if run_dir is not None else None,
            config=config or {},
            reinit=True,
        )
        logger.info("W&B run started (mode=%s, name=%s)", mode, run_name)

    def log_config(self, config: dict[str, Any]) -> None:
        self.run.config.update(config, allow_val_change=True)

    def log_metrics(self, metrics: dict[str, Any], step: int | None = None) -> None:
        self.run.log(metrics, step=step)

    def log_artifact(self, path: str | Path, name: str | None = None) -> None:
        path = Path(path)
        artifact = self._wandb.Artifact(name or path.stem, type="run-artifact")
        if path.is_dir():
            artifact.add_dir(str(path))
        else:
            artifact.add_file(str(path))
        self.run.log_artifact(artifact)

    def finish(self) -> None:
        self.run.finish()
        logger.info("W&B run finished")


def build_tracker(
    mode: str = MODE_NOOP,
    project: str = "alignlab",
    run_name: str = "unnamed",
    entity: str | None = None,
    tags: list[str] | None = None,
    run_dir: str | Path | None = None,
    config: dict[str, Any] | None = None,
) -> Tracker:
    """Construct the tracker for the requested mode.

    Falls back to NoOpTracker with a warning if wandb is requested but not
    installed. A missing tracking package must never abort a training run that
    is otherwise fine - but the fallback is logged loudly so that nobody later
    wonders why a run is missing from the dashboard.
    """
    mode = (mode or MODE_NOOP).strip().lower()
    if mode not in VALID_MODES:
        raise ValueError(f"tracking mode must be one of {VALID_MODES}, got {mode!r}")

    if mode == MODE_NOOP:
        return NoOpTracker()

    try:
        return WandbTracker(
            project=project,
            run_name=run_name,
            mode=mode,
            entity=entity,
            tags=tags,
            run_dir=run_dir,
            config=config,
        )
    except ImportError:
        logger.warning(
            "Tracking mode %r was requested but wandb is unavailable. "
            "Falling back to the noop tracker - THIS RUN WILL NOT BE TRACKED.",
            mode,
        )
        return NoOpTracker()


def write_metrics_jsonl(path: str | Path, metrics: dict[str, Any], step: int) -> None:
    """Append one metrics record to a JSONL file.

    A local, dependency-free record that survives regardless of which tracking
    backend is active. If W&B is unreachable from the server, this file is
    still the evidence that a measurement actually happened.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {"step": step, **metrics}
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, default=str) + "\n")
