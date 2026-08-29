"""Tier A reproducibility: bitwise identical results on the same machine.

The claim under test is narrow and therefore actually provable: same machine,
same seed, same environment produces bitwise identical draws. Cross-machine
bitwise agreement (Tier C territory) is NOT claimed and NOT tested, because it
is not achievable between a CPU x86 machine and an Ampere GPU.

The subprocess test matters more than the in-process one. Two calls to
set_seed inside one interpreter share module state, so agreeing there is weak
evidence. Two separate interpreter invocations agreeing is the real claim.
"""

from __future__ import annotations

import random
import subprocess
import sys
import textwrap

import numpy as np
import pytest
import torch

from alignlab.seeding import (
    SeedReport,
    capture_rng_state,
    restore_rng_state,
    set_seed,
)


def _draw() -> tuple[float, float, float]:
    """One draw from each RNG that AlignLab seeds."""
    return (
        random.random(),
        float(np.random.rand()),
        float(torch.rand(1).item()),
    )


def test_set_seed_returns_report() -> None:
    report = set_seed(123)
    assert isinstance(report, SeedReport)
    assert report.seed == 123
    assert report.deterministic is False


def test_set_seed_rejects_non_int() -> None:
    with pytest.raises(TypeError):
        set_seed("42")  # type: ignore[arg-type]


def test_set_seed_rejects_bool() -> None:
    """bool is a subclass of int; accepting it would silently seed with 0 or 1."""
    with pytest.raises(TypeError):
        set_seed(True)  # type: ignore[arg-type]


def test_same_seed_gives_identical_draws_in_process() -> None:
    set_seed(42)
    first = _draw()
    set_seed(42)
    second = _draw()
    assert first == second


def test_different_seeds_give_different_draws() -> None:
    """Guards against a seeding function that silently does nothing."""
    set_seed(1)
    first = _draw()
    set_seed(2)
    second = _draw()
    assert first != second


def test_deterministic_flag_sets_cublas_workspace(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    report = set_seed(7, deterministic=True)
    assert report.deterministic is True
    assert report.cublas_workspace_config == ":4096:8"
    assert report.cudnn_benchmark is False
    # Restore the non-deterministic default so later tests are unaffected.
    torch.use_deterministic_algorithms(False)


def test_rng_state_round_trip() -> None:
    """Capturing and restoring RNG state must reproduce the exact stream.

    This is what makes a resumed run continue the original data order rather
    than starting a subtly different one.
    """
    set_seed(99)
    state = capture_rng_state()
    expected = [_draw() for _ in range(3)]

    # Consume the streams so that a no-op restore could not pass by accident.
    for _ in range(10):
        _draw()

    restore_rng_state(state)
    actual = [_draw() for _ in range(3)]
    assert actual == expected


_SUBPROCESS_SCRIPT = textwrap.dedent(
    """
    import random, numpy as np, torch
    from alignlab.seeding import set_seed
    set_seed({seed})
    print(repr((random.random(), float(np.random.rand()), float(torch.rand(1).item()))))
    """
)


def _run_in_subprocess(seed: int) -> str:
    result = subprocess.run(
        [sys.executable, "-c", _SUBPROCESS_SCRIPT.format(seed=seed)],
        capture_output=True,
        text=True,
        timeout=180,
        check=True,
    )
    return result.stdout.strip()


def test_bitwise_reproducible_across_separate_processes() -> None:
    """Tier A, the real claim: two fresh interpreters agree bitwise.

    Uses repr() of the floats so the comparison is exact rather than
    approximate - "close enough" would not be bitwise.
    """
    first = _run_in_subprocess(4242)
    second = _run_in_subprocess(4242)
    assert first == second
    assert first != ""


def test_different_seeds_differ_across_processes() -> None:
    assert _run_in_subprocess(1) != _run_in_subprocess(2)
