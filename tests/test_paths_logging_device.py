"""Path resolution, logging setup and device resolution.

Grouped because each is small, and because together they form the layer that
absorbs the difference between the two machines.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import pytest
import torch

from alignlab.device import AUTO, describe_hardware, resolve_device
from alignlab.env_detect import LOCAL, SERVER, detect_environment, hostname
from alignlab.logging_utils import LOG_FILE_NAME, get_logger, setup_logging
from alignlab.paths import (
    ENV_VAR_ENV_NAME,
    ENV_VAR_RUN_NAME,
    cache_root,
    checkpoint_root,
    describe_roots,
    generate_run_name,
    output_root,
    repo_root,
    run_dir,
)

# --------------------------------------------------------------------------
# paths
# --------------------------------------------------------------------------


def test_repo_root_contains_pyproject() -> None:
    assert (repo_root() / "pyproject.toml").is_file()


def test_roots_follow_env_vars(isolated_roots: dict[str, Path]) -> None:
    assert output_root() == isolated_roots["output"].resolve()
    assert checkpoint_root() == isolated_roots["checkpoint"].resolve()
    assert cache_root() == isolated_roots["cache"].resolve()


def test_roots_fall_back_to_repo_relative() -> None:
    """With no env vars set, defaults are repository-relative.

    Correct locally. Deliberately NOT correct on a shared server, which is why
    configs/env/server.yaml makes the paths mandatory instead of inheriting
    these.
    """
    assert output_root() == (repo_root() / "outputs").resolve()
    assert checkpoint_root() == (repo_root() / "checkpoints").resolve()


def test_run_dir_is_created(isolated_roots: dict[str, Path]) -> None:
    path = run_dir("my-run")
    assert path.is_dir()
    assert path.parent == isolated_roots["output"].resolve()


def test_run_name_is_sortable_and_prefixed() -> None:
    name = generate_run_name(prefix="sft")
    assert name.startswith("sft-")
    # "<prefix>-YYYYmmdd-HHMMSS"
    assert len(name.split("-")) == 3


def test_run_name_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    """Needed so a resumed job writes back into its original directory."""
    monkeypatch.setenv(ENV_VAR_RUN_NAME, "fixed-run-name")
    assert generate_run_name(prefix="ignored") == "fixed-run-name"


def test_describe_roots_returns_strings() -> None:
    described = describe_roots()
    assert set(described) == {
        "repo_root",
        "output_root",
        "checkpoint_root",
        "cache_root",
    }
    assert all(isinstance(v, str) for v in described.values())


# --------------------------------------------------------------------------
# environment detection
# --------------------------------------------------------------------------


def test_explicit_env_var_wins(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(ENV_VAR_ENV_NAME, "server")
    assert detect_environment() == SERVER
    monkeypatch.setenv(ENV_VAR_ENV_NAME, "LOCAL")
    assert detect_environment() == LOCAL


def test_unknown_env_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """A typo must not silently select the wrong storage roots."""
    monkeypatch.setenv(ENV_VAR_ENV_NAME, "sever")
    with pytest.raises(ValueError, match="not a known environment"):
        detect_environment()


def test_autodetect_matches_cuda_reality() -> None:
    detected = detect_environment()
    assert detected == (SERVER if torch.cuda.is_available() else LOCAL)


def test_hostname_is_non_empty() -> None:
    assert hostname()


# --------------------------------------------------------------------------
# logging
# --------------------------------------------------------------------------


def test_setup_logging_writes_file(tmp_path: Path) -> None:
    logger = setup_logging(level="DEBUG", log_dir=tmp_path)
    logger.info("hello from the test")
    for handler in logger.handlers:
        handler.flush()

    log_file = tmp_path / LOG_FILE_NAME
    assert log_file.is_file()
    assert "hello from the test" in log_file.read_text(encoding="utf-8")


def _alignlab_handlers(logger: logging.Logger) -> list[logging.Handler]:
    """Only the handlers this project installed.

    pytest attaches its own LogCaptureHandlers to every logger, so a raw
    len(logger.handlers) would be counting someone else's work. setup_logging
    deliberately leaves foreign handlers alone.
    """
    return [h for h in logger.handlers if getattr(h, "_alignlab_handler", False)]


def test_repeated_setup_does_not_duplicate_handlers(tmp_path: Path) -> None:
    """Duplicated handlers print every line N times, which looks like N runs."""
    first = setup_logging(level="INFO", log_dir=tmp_path)
    count_after_first = len(_alignlab_handlers(first))
    second = setup_logging(level="INFO", log_dir=tmp_path)
    third = setup_logging(level="INFO", log_dir=tmp_path)

    assert count_after_first == 2  # console + file
    assert len(_alignlab_handlers(second)) == count_after_first
    assert len(_alignlab_handlers(third)) == count_after_first


def test_no_log_dir_means_console_only() -> None:
    logger = setup_logging(level="INFO", log_dir=None)
    assert len(_alignlab_handlers(logger)) == 1


def test_setup_logging_leaves_foreign_handlers_alone() -> None:
    """Only our own handlers are removed - pytest's capture must survive."""
    logger = logging.getLogger("alignlab")
    foreign = logging.NullHandler()
    logger.addHandler(foreign)
    try:
        setup_logging(level="INFO")
        assert foreign in logger.handlers
    finally:
        logger.removeHandler(foreign)


def test_file_is_more_verbose_than_console(tmp_path: Path) -> None:
    """The console is for watching; the file is for diagnosing."""
    logger = setup_logging(level="WARNING", file_level="DEBUG", log_dir=tmp_path)
    logger.debug("debug detail")
    for handler in logger.handlers:
        handler.flush()
    assert "debug detail" in (tmp_path / LOG_FILE_NAME).read_text(encoding="utf-8")


def test_get_logger_namespaces_under_alignlab() -> None:
    assert get_logger("thing").name == "alignlab.thing"
    assert get_logger("alignlab.thing").name == "alignlab.thing"


def test_logger_does_not_propagate() -> None:
    """Prevents double-printing when a host application has a root handler."""
    logger = setup_logging(level="INFO")
    assert logger.propagate is False
    assert logging.getLogger("alignlab") is logger


# --------------------------------------------------------------------------
# device
# --------------------------------------------------------------------------


def test_resolve_cpu() -> None:
    assert resolve_device("cpu").type == "cpu"


def test_resolve_auto_matches_availability() -> None:
    device = resolve_device(AUTO)
    assert device.type == ("cuda" if torch.cuda.is_available() else "cpu")


@pytest.mark.skipif(
    torch.cuda.is_available(), reason="this asserts the CPU-only failure mode"
)
def test_explicit_cuda_raises_without_cuda() -> None:
    """Silent CPU fallback would produce numbers that are not comparable.

    On the GPU server this test skips and the gpu-marked tests run instead.
    """
    with pytest.raises(RuntimeError, match="cuda.is_available"):
        resolve_device("cuda")


def test_unknown_device_string_raises() -> None:
    with pytest.raises(ValueError):
        resolve_device("tpu")


def test_describe_hardware_reports_only_observed_facts() -> None:
    """On a CPU-only machine no GPU is described, even though one is installed.

    This machine has a GTX 1050 that the installed CPU-only torch build cannot
    use. The manifest must record what torch can actually use, not what is
    physically present.
    """
    info = describe_hardware()
    assert info.torch_version == torch.__version__
    assert info.cuda_available == torch.cuda.is_available()
    if not torch.cuda.is_available():
        assert info.devices == []
        assert info.device_count == 0
        assert info.bf16_supported is False
    assert isinstance(info.as_dict(), dict)


@pytest.mark.gpu
def test_gpu_is_described_when_present() -> None:
    """Runs on the server only. Skipped locally - never silently passed."""
    info = describe_hardware()
    assert info.cuda_available is True
    assert info.device_count >= 1
    assert info.devices[0]["name"]
    assert info.devices[0]["total_memory_gib"] > 0


# --------------------------------------------------------------------------
# Hugging Face cache wiring (Phase 1C storage policy)
# --------------------------------------------------------------------------


def test_configure_hf_cache_sets_expected_vars(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """cache_root must actually reach the HF libraries, not just the config."""
    from alignlab.paths import configure_hf_cache

    for name in ("HF_HUB_CACHE", "HF_DATASETS_CACHE", "HF_HOME"):
        monkeypatch.delenv(name, raising=False)

    applied = configure_hf_cache(tmp_path / "hf")

    assert applied["HF_HUB_CACHE"] == str((tmp_path / "hf" / "hub").resolve())
    assert applied["HF_DATASETS_CACHE"] == str((tmp_path / "hf" / "datasets").resolve())
    assert os.environ["HF_HUB_CACHE"] == applied["HF_HUB_CACHE"]


def test_configure_hf_cache_does_not_touch_hf_home(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """HF_HOME governs the token file; relocating it would break an existing login."""
    from alignlab.paths import configure_hf_cache

    for name in ("HF_HUB_CACHE", "HF_DATASETS_CACHE", "HF_HOME"):
        monkeypatch.delenv(name, raising=False)

    configure_hf_cache(tmp_path / "hf")
    assert "HF_HOME" not in os.environ


def test_configure_hf_cache_respects_existing_value(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An operator who exported the variable deliberately outranks our config."""
    from alignlab.paths import configure_hf_cache

    monkeypatch.setenv("HF_HUB_CACHE", "/deliberate/operator/choice")
    monkeypatch.delenv("HF_DATASETS_CACHE", raising=False)

    applied = configure_hf_cache(tmp_path / "hf")

    assert "HF_HUB_CACHE" not in applied
    assert os.environ["HF_HUB_CACHE"] == "/deliberate/operator/choice"
    assert "HF_DATASETS_CACHE" in applied


def test_configure_hf_cache_falls_back_to_cache_root(
    isolated_roots: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    """With no explicit root, it uses the configured ALIGNLAB_CACHE_ROOT."""
    from alignlab.paths import configure_hf_cache

    for name in ("HF_HUB_CACHE", "HF_DATASETS_CACHE"):
        monkeypatch.delenv(name, raising=False)

    applied = configure_hf_cache()
    assert applied["HF_HUB_CACHE"].startswith(str(isolated_roots["cache"].resolve()))
