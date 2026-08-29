"""Preemption handling.

The design under test separates decision logic from signal binding, which is
what makes it verifiable on a machine with no SIGUSR1.

Coverage split, stated honestly:

  VERIFIED here  - the decision logic: flag setting, should_stop, one-shot
                   checkpointing, SLURM detection, handler registration and
                   restoration, and the real signal path where the platform
                   supports raising it.

  NOT TESTED     - the SLURM requeue path (no scheduler has been available),
                   and SIGUSR1 delivery on this Windows development machine
                   (the signal does not exist here). Both are exercised on the
                   server in Phase 1B, which has NOT been run.
"""

from __future__ import annotations

import os
import signal
import sys

import pytest

from alignlab.preemption import (
    PreemptionHandler,
    available_preemption_signals,
    in_slurm_job,
    slurm_job_id,
)

WINDOWS = sys.platform.startswith("win")


def test_available_signals_never_empty() -> None:
    """Every supported platform provides at least SIGTERM."""
    names = [s.name for s in available_preemption_signals()]
    assert names
    assert "SIGTERM" in names


def test_sigusr1_presence_matches_platform() -> None:
    """Documents the platform difference this module exists to absorb.

    SIGUSR1 is absent on Windows and present on POSIX. The handler must work
    either way, which is why it uses getattr rather than a hard reference.
    """
    names = [s.name for s in available_preemption_signals()]
    if WINDOWS:
        assert "SIGUSR1" not in names
    else:
        assert "SIGUSR1" in names


def test_import_does_not_raise_on_this_platform() -> None:
    """Regression guard: a hard signal.SIGUSR1 reference breaks import on Windows."""
    import alignlab.preemption as module

    assert module is not None


def test_initial_state_is_not_stopping() -> None:
    handler = PreemptionHandler()
    assert handler.should_stop() is False
    assert handler.state.signal_received is None
    assert handler.state.checkpoint_calls == 0


def test_request_stop_sets_flag() -> None:
    """The OS-independent path that makes the logic testable on Windows."""
    handler = PreemptionHandler()
    handler.request_stop(reason="unit-test")
    assert handler.should_stop() is True
    assert handler.state.signal_received == "unit-test"


def test_handle_invokes_checkpoint_once() -> None:
    calls: list[int] = []
    handler = PreemptionHandler(checkpoint_fn=lambda: calls.append(1))

    handler.request_stop()
    handler.handle()

    assert calls == [1]
    assert handler.state.checkpoint_calls == 1


def test_handle_without_checkpoint_fn_does_not_raise() -> None:
    """Losing progress is bad; crashing during shutdown is worse."""
    handler = PreemptionHandler(checkpoint_fn=None)
    handler.request_stop()
    handler.handle()
    assert handler.state.checkpoint_calls == 0


def test_register_returns_registered_names() -> None:
    handler = PreemptionHandler()
    try:
        registered = handler.register()
        assert registered
        assert registered == handler.state.registered_signals
    finally:
        handler.unregister()


def test_unregister_restores_previous_handler() -> None:
    """Leaving handlers installed would leak across tests and across runs."""
    original = signal.getsignal(signal.SIGTERM)
    handler = PreemptionHandler()
    handler.register()
    assert signal.getsignal(signal.SIGTERM) is not original
    handler.unregister()
    assert signal.getsignal(signal.SIGTERM) is original


def test_context_manager_registers_and_restores() -> None:
    original = signal.getsignal(signal.SIGTERM)
    with PreemptionHandler() as handler:
        assert handler.state.registered_signals
    assert signal.getsignal(signal.SIGTERM) is original


@pytest.mark.skipif(WINDOWS, reason="SIGUSR1 does not exist on Windows")
def test_real_sigusr1_sets_flag_posix_only() -> None:
    """The genuine signal path. Runs on Linux (including the server), skips here.

    On the department server this same test covers what SLURM would trigger
    with --signal=USR1@<seconds>, even though no scheduler is present.
    """
    calls: list[int] = []
    handler = PreemptionHandler(checkpoint_fn=lambda: calls.append(1))
    handler.register()
    try:
        os.kill(os.getpid(), signal.SIGUSR1)
        assert handler.should_stop() is True
        assert handler.state.signal_received == "SIGUSR1"
        handler.handle()
        assert calls == [1]
    finally:
        handler.unregister()


def test_slurm_detection_reads_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    assert in_slurm_job() is False
    assert slurm_job_id() is None

    monkeypatch.setenv("SLURM_JOB_ID", "123456")
    assert in_slurm_job() is True
    assert slurm_job_id() == "123456"


def test_requeue_skipped_outside_slurm(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without SLURM_JOB_ID the requeue is a no-op, not an error.

    This does NOT test a real requeue. Invoking scontrol has never been
    exercised by this project - that path remains IMPLEMENTED / NOT TESTED.
    """
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    handler = PreemptionHandler(checkpoint_fn=lambda: None, requeue=True)
    handler.request_stop()
    handler.handle()
    assert handler.state.requeue_attempted is False
    assert handler.state.requeue_succeeded is None
