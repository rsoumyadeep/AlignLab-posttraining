"""Structured logging.

Two sinks, always:

    console - human-readable, colourised via rich when available
    file    - complete record inside the run directory, so that a run that
              scrolled off a terminal or died inside a detached tmux session
              on the server can still be diagnosed afterwards

The single most important property here is idempotence. setup_logging may be
called more than once in a process (tests, notebooks, a resumed run). Without
care that produces duplicated handlers and every line printed N times, which
looks alarmingly like a training loop running N times. Handlers are therefore
tagged and removed before being re-added.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

_HANDLER_TAG = "_alignlab_handler"
LOG_FILE_NAME = "run.log"

_CONSOLE_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_FILE_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(filename)s:%(lineno)d | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def _clear_alignlab_handlers(logger: logging.Logger) -> None:
    """Remove handlers this module previously installed.

    Only our own handlers are touched. A handler installed by pytest, by a
    notebook host or by the user is left alone.
    """
    for handler in list(logger.handlers):
        if getattr(handler, _HANDLER_TAG, False):
            logger.removeHandler(handler)
            handler.close()


def _build_console_handler(level: int) -> logging.Handler:
    """Return a rich handler if rich is installed, else a plain stream handler."""
    try:
        from rich.logging import RichHandler
    except ImportError:
        handler: logging.Handler = logging.StreamHandler(stream=sys.stderr)
        handler.setFormatter(logging.Formatter(_CONSOLE_FORMAT, _DATE_FORMAT))
    else:
        handler = RichHandler(rich_tracebacks=True, show_path=False, markup=False)
        # rich supplies its own timestamp and level columns, so the format
        # string carries the message only.
        handler.setFormatter(logging.Formatter("%(name)s | %(message)s"))
    handler.setLevel(level)
    return handler


def setup_logging(
    level: str | int = "INFO",
    log_dir: Path | None = None,
    file_level: str | int = "DEBUG",
) -> logging.Logger:
    """Configure the AlignLab root logger.

    Args:
        level: console verbosity.
        log_dir: when given, a run.log file is written there at file_level.
            The file is intentionally more verbose than the console: the
            console is for watching, the file is for diagnosing.
        file_level: file verbosity.

    Returns:
        The configured "alignlab" logger.

    Safe to call repeatedly - previously installed AlignLab handlers are
    removed first, so messages are never duplicated.
    """
    console_level = logging.getLevelName(level) if isinstance(level, str) else level
    disk_level = (
        logging.getLevelName(file_level) if isinstance(file_level, str) else file_level
    )

    logger = logging.getLogger("alignlab")
    _clear_alignlab_handlers(logger)

    # The logger itself must pass through the most permissive of the two sinks;
    # each handler then filters down to its own level.
    logger.setLevel(min(console_level, disk_level))
    # Do not also emit through the root logger, which would double-print
    # whenever the host application has its own root handler.
    logger.propagate = False

    console = _build_console_handler(console_level)
    setattr(console, _HANDLER_TAG, True)
    logger.addHandler(console)

    if log_dir is not None:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(
            log_dir / LOG_FILE_NAME, mode="a", encoding="utf-8"
        )
        file_handler.setLevel(disk_level)
        file_handler.setFormatter(logging.Formatter(_FILE_FORMAT, _DATE_FORMAT))
        setattr(file_handler, _HANDLER_TAG, True)
        logger.addHandler(file_handler)

    return logger


def get_logger(name: str) -> logging.Logger:
    """Return a child logger under the alignlab namespace.

    Modules should call get_logger(__name__) so that every AlignLab log line
    inherits the configuration installed by setup_logging.
    """
    if name.startswith("alignlab"):
        return logging.getLogger(name)
    return logging.getLogger(f"alignlab.{name}")
