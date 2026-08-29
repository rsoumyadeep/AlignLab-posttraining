"""Shared pytest fixtures.

Two rules the whole suite depends on:

1. No test touches the real output, checkpoint or cache roots. Every test that
   writes gets a tmp_path-backed set of roots via the isolated_roots fixture,
   so running the suite never pollutes the developer's outputs/ directory and
   never depends on state left by a previous run.

2. No test needs credentials or a network. Tracking defaults to the noop
   backend everywhere.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch

from alignlab.paths import (
    ENV_VAR_CACHE_ROOT,
    ENV_VAR_CKPT_ROOT,
    ENV_VAR_ENV_NAME,
    ENV_VAR_OUTPUT_ROOT,
    ENV_VAR_RUN_NAME,
)

_ALIGNLAB_VARS = (
    ENV_VAR_ENV_NAME,
    ENV_VAR_OUTPUT_ROOT,
    ENV_VAR_CKPT_ROOT,
    ENV_VAR_CACHE_ROOT,
    ENV_VAR_RUN_NAME,
)

# Hugging Face cache variables are PROCESS-GLOBAL and are set by
# paths.configure_hf_cache() during any train() call. Because that function
# deliberately respects an already-set value, one test running train() would
# otherwise pin the cache location for every later test in the same session -
# which is exactly how a later assertion started failing for a reason that had
# nothing to do with the code under test. Cleared per test.
_HF_VARS = ("HF_HUB_CACHE", "HF_DATASETS_CACHE", "HF_HOME", "TRANSFORMERS_CACHE")


@pytest.fixture(autouse=True)
def clean_alignlab_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Remove AlignLab and Hugging Face cache variables before every test.

    Autouse, because a variable left set in the developer's shell - or set by
    an earlier test in the same process - would otherwise silently change what
    the tests exercise.
    """
    for name in (*_ALIGNLAB_VARS, *_HF_VARS):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def isolated_roots(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Path]:
    """Point every AlignLab storage root at a temporary directory."""
    roots = {
        "output": tmp_path / "outputs",
        "checkpoint": tmp_path / "checkpoints",
        "cache": tmp_path / "cache",
    }
    for path in roots.values():
        path.mkdir(parents=True, exist_ok=True)

    monkeypatch.setenv(ENV_VAR_OUTPUT_ROOT, str(roots["output"]))
    monkeypatch.setenv(ENV_VAR_CKPT_ROOT, str(roots["checkpoint"]))
    monkeypatch.setenv(ENV_VAR_CACHE_ROOT, str(roots["cache"]))
    return roots


@pytest.fixture
def tiny_model() -> torch.nn.Module:
    """A small deterministic model for checkpoint and training tests."""
    torch.manual_seed(0)
    return torch.nn.Sequential(
        torch.nn.Linear(4, 8),
        torch.nn.ReLU(),
        torch.nn.Linear(8, 2),
    )


@pytest.fixture
def repo_root_path() -> Path:
    """The repository root, located from this test file."""
    return Path(__file__).resolve().parents[1]


def pytest_runtest_setup(item: pytest.Item) -> None:
    """Skip marked tests that cannot run in the current environment.

    Skips are visible in the -ra summary. A test that cannot run must SKIP,
    never silently pass - a green suite that quietly skipped the GPU tests
    would misrepresent what was actually verified.
    """
    if item.get_closest_marker("gpu") and not torch.cuda.is_available():
        pytest.skip("requires CUDA; torch.cuda.is_available() is False")

    if item.get_closest_marker("server"):
        pytest.skip("requires the department server; never runs locally")

    if item.get_closest_marker("network") and os.environ.get(
        "ALIGNLAB_ALLOW_NETWORK"
    ) != "1":
        pytest.skip("requires network; set ALIGNLAB_ALLOW_NETWORK=1 to enable")
