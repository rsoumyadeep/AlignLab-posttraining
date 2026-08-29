"""Preemption and graceful shutdown.

PROJECT_INSTRUCTIONS section 4 asks for SLURM/SIGUSR1 preemption handling
"where appropriate". Two facts shape this implementation:

1. signal.SIGUSR1 DOES NOT EXIST on Windows. VERIFIED locally: the available
   signals are SIGABRT, SIGBREAK, SIGFPE, SIGILL, SIGINT, SIGSEGV, SIGTERM.
   A module that unconditionally references signal.SIGUSR1 raises
   AttributeError at import time on the development machine.

2. Whether the department server runs SLURM is UNKNOWN. The system report
   contains no scheduler evidence, but that report captured no software
   inventory at all, so it is absence of evidence rather than evidence of
   absence.

The design that survives both facts: the DECISION LOGIC (a run should stop
soon, checkpoint, and optionally request a requeue) is separated from the
SIGNAL BINDING (which signals exist on this OS). The logic is therefore fully
testable on Windows by direct invocation, while the real signal path is
exercised on Linux with "kill -USR1 <pid>" - which works even with no
scheduler present.

Status of the SLURM requeue path: IMPLEMENTED, NOT TESTED. No SLURM
environment has been available to this project.
"""

from __future__ import annotations

import os
import signal
import subprocess
from dataclasses import dataclass, field
from typing import Callable

from alignlab.logging_utils import get_logger

logger = get_logger(__name__)

# Signals that mean "wrap up soon". SIGUSR1 is what SLURM sends when a job is
# submitted with --signal=USR1@<seconds>; SIGTERM is the generic termination
# request that both a scheduler and an ordinary kill will send.
_PREEMPTION_SIGNAL_NAMES = ("SIGUSR1", "SIGTERM", "SIGBREAK")


def available_preemption_signals() -> list[signal.Signals]:
    """Return the preemption-relevant signals this OS actually provides.

    Uses getattr rather than a hard reference so the module imports cleanly on
    Windows, where SIGUSR1 is absent.
    """
    found = []
    for name in _PREEMPTION_SIGNAL_NAMES:
        sig = getattr(signal, name, None)
        if sig is not None:
            found.append(sig)
    return found


def in_slurm_job() -> bool:
    """Whether this process appears to be inside a SLURM allocation.

    Detected from SLURM_JOB_ID, which SLURM exports into every job step.
    Returns False everywhere else, including on an unscheduled shared server.
    """
    return bool(os.environ.get("SLURM_JOB_ID"))


def slurm_job_id() -> str | None:
    """Return the SLURM job id if running under SLURM, else None."""
    return os.environ.get("SLURM_JOB_ID")


@dataclass
class PreemptionState:
    """Observable record of what the handler has seen and done."""

    preemption_requested: bool = False
    signal_received: str | None = None
    checkpoint_calls: int = 0
    requeue_attempted: bool = False
    requeue_succeeded: bool | None = None
    registered_signals: list[str] = field(default_factory=list)


class PreemptionHandler:
    """Catch a preemption signal, checkpoint once, optionally request requeue.

    Usage in a training loop::

        handler = PreemptionHandler(checkpoint_fn=save_now)
        handler.register()
        for step in range(max_steps):
            train_step()
            if handler.should_stop():
                handler.handle()     # checkpoints, maybe requeues
                break

    The handler deliberately does NOT checkpoint inside the signal handler
    itself. Signal handlers interrupt the interpreter at an arbitrary point,
    and writing a large file from one risks a corrupt checkpoint or a deadlock
    in the allocator. The handler only sets a flag; the training loop performs
    the write at a safe point. This is the single most important design
    decision in this module.
    """

    def __init__(
        self,
        checkpoint_fn: Callable[[], None] | None = None,
        requeue: bool = False,
    ) -> None:
        """
        Args:
            checkpoint_fn: called once when preemption is handled. Should save
                a resumable checkpoint.
            requeue: when True and running under SLURM, ask the scheduler to
                requeue this job after checkpointing.
        """
        self.checkpoint_fn = checkpoint_fn
        self.requeue = requeue
        self.state = PreemptionState()
        self._original_handlers: dict[int, object] = {}

    # -- signal binding ----------------------------------------------------

    def register(self) -> list[str]:
        """Install handlers for every preemption signal this OS supports.

        Returns the names of the signals actually registered. On Windows this
        is a strictly smaller set than on Linux, and the caller should log the
        returned list rather than assume what was installed.

        Signal handlers can only be installed from the main thread; a
        ValueError raised in a worker thread is caught and reported rather than
        allowed to kill the run.
        """
        registered: list[str] = []
        for sig in available_preemption_signals():
            try:
                self._original_handlers[int(sig)] = signal.getsignal(sig)
                signal.signal(sig, self._on_signal)
                registered.append(sig.name)
            except (ValueError, OSError, RuntimeError) as exc:
                logger.warning("Could not register handler for %s: %s", sig.name, exc)

        self.state.registered_signals = registered
        logger.info(
            "Preemption handler registered for: %s",
            ", ".join(registered) if registered else "(none available)",
        )
        return registered

    def unregister(self) -> None:
        """Restore the handlers that were in place before register()."""
        for signum, original in self._original_handlers.items():
            try:
                signal.signal(signum, original)  # type: ignore[arg-type]
            except (ValueError, OSError, RuntimeError, TypeError) as exc:
                logger.debug("Could not restore handler for signal %s: %s", signum, exc)
        self._original_handlers.clear()

    def _on_signal(self, signum: int, _frame: object) -> None:
        """Signal handler. Sets a flag and returns immediately.

        Performs no I/O beyond a log call, for the reasons in the class
        docstring.
        """
        try:
            name = signal.Signals(signum).name
        except ValueError:  # pragma: no cover - defensive
            name = str(signum)
        self.state.preemption_requested = True
        self.state.signal_received = name
        logger.warning(
            "Received %s - will checkpoint and stop at the next safe point", name
        )

    # -- decision logic (OS independent, fully testable on Windows) ---------

    def should_stop(self) -> bool:
        """Whether the training loop should wind up now."""
        return self.state.preemption_requested

    def request_stop(self, reason: str = "manual") -> None:
        """Trigger the preemption path without a real signal.

        This is what makes the logic testable on a platform that has no
        SIGUSR1, and is also useful as a manual early stop.
        """
        self.state.preemption_requested = True
        self.state.signal_received = reason
        logger.warning("Stop requested (%s)", reason)

    def handle(self) -> None:
        """Checkpoint once and, if configured and under SLURM, requeue.

        The checkpoint counter is recorded in PreemptionState so that a double
        save shows up in tests rather than passing silently.
        """
        if self.checkpoint_fn is not None:
            logger.info("Preemption: writing checkpoint before exit")
            self.checkpoint_fn()
            self.state.checkpoint_calls += 1
        else:
            logger.warning(
                "Preemption requested but no checkpoint_fn was provided - "
                "progress since the last checkpoint will be lost"
            )

        if self.requeue:
            self._requeue()

    def _requeue(self) -> None:
        """Ask SLURM to requeue this job.

        STATUS: IMPLEMENTED, NOT TESTED. No SLURM environment has been
        available to this project. Do not report this path as VERIFIED until
        it has actually run under a scheduler.
        """
        job_id = slurm_job_id()
        if not job_id:
            logger.info("Requeue requested but SLURM_JOB_ID is unset - skipping")
            return

        self.state.requeue_attempted = True
        try:
            result = subprocess.run(
                ["scontrol", "requeue", job_id],
                check=False,
                capture_output=True,
                text=True,
                timeout=30,
            )
            self.state.requeue_succeeded = result.returncode == 0
            if result.returncode == 0:
                logger.info("Requeued SLURM job %s", job_id)
            else:
                logger.error(
                    "scontrol requeue failed (rc=%s): %s",
                    result.returncode,
                    result.stderr.strip(),
                )
        except (OSError, subprocess.SubprocessError) as exc:
            self.state.requeue_succeeded = False
            logger.error("Could not invoke scontrol: %s", exc)

    # -- context manager ---------------------------------------------------

    def __enter__(self) -> "PreemptionHandler":
        self.register()
        return self

    def __exit__(self, *_exc: object) -> None:
        self.unregister()
