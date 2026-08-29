"""Guard: no machine-specific absolute paths in the source tree.

This is the test that keeps the two-environment model honest. The moment a
Windows drive letter or a server home directory is written into a source file,
the project stops being portable and results stop being attributable. It is
much easier to prevent than to find later.

Scanned: src/, tests/, configs/, scripts/. Not scanned: documentation, which
legitimately quotes real paths when reporting what was observed on a machine.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

SCANNED_DIRS = ("src", "tests", "configs", "scripts")
SCANNED_SUFFIXES = (".py", ".yaml", ".yml", ".sh", ".toml")

# Patterns that indicate a hard-coded machine-specific location.
FORBIDDEN = {
    "windows_drive": re.compile(r"[A-Za-z]:[\\/]{1,2}(?:Users|projects|Program)", re.I),
    "server_home": re.compile(r"/data/home/\w+"),
    "user_home_abs": re.compile(r"/home/(?!<)\w+/"),
    "cuda_install_path": re.compile(r"/usr/local/cuda-[\d.]+"),
}

# Lines that legitimately mention such a path in prose. A comment describing
# what was observed on a machine is documentation, not a dependency - the rule
# is that it must not be usable as a value.
ALLOW_MARKERS = ("#", "//", '"""', "observed", "VERIFIED", "example", "e.g.")


# configs/env/ is the ONE place machine-specific values are allowed to live -
# that is the entire purpose of the env config group. Scanning it would forbid
# exactly the thing the design exists to permit.
#
# Phase 1B note: this exclusion was added when configs/env/server.yaml was
# populated with verified server paths. Before that the file held only Hydra
# MISSING placeholders, so the broader scan passed by accident rather than by
# design. The guard remains fully in force for src/, tests/, scripts/ and the
# rest of configs/ - and test_machine_specific_paths_are_confined_to_env_group
# below asserts that configs/env/ really is the only exception.
EXCLUDED_FROM_SCAN = ("configs/env",)


def _is_excluded(path: Path, repo_root: Path) -> bool:
    rel = path.relative_to(repo_root).as_posix()
    return any(rel.startswith(prefix) for prefix in EXCLUDED_FROM_SCAN)


def _source_files(repo_root: Path, apply_exclusions: bool = True) -> list[Path]:
    files: list[Path] = []
    for directory in SCANNED_DIRS:
        base = repo_root / directory
        if not base.is_dir():
            continue
        for path in base.rglob("*"):
            if not (path.is_file() and path.suffix in SCANNED_SUFFIXES):
                continue
            if apply_exclusions and _is_excluded(path, repo_root):
                continue
            files.append(path)
    return files


def _is_prose(line: str) -> bool:
    """Whether a line is a comment or docstring rather than an assignment."""
    stripped = line.strip()
    return any(stripped.startswith(m) or m in stripped for m in ALLOW_MARKERS)


def test_source_files_exist(repo_root_path: Path) -> None:
    """Sanity: the scan must actually be looking at something."""
    files = _source_files(repo_root_path)
    assert len(files) > 10, f"expected a populated source tree, found {len(files)} files"


@pytest.mark.parametrize("pattern_name", sorted(FORBIDDEN))
def test_no_hardcoded_absolute_paths(repo_root_path: Path, pattern_name: str) -> None:
    pattern = FORBIDDEN[pattern_name]
    offenders: list[str] = []

    for path in _source_files(repo_root_path):
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:  # pragma: no cover - defensive
            continue
        for number, line in enumerate(text.splitlines(), start=1):
            if pattern.search(line) and not _is_prose(line):
                rel = path.relative_to(repo_root_path)
                offenders.append(f"{rel}:{number}: {line.strip()}")

    assert not offenders, (
        f"Hard-coded machine-specific path ({pattern_name}) found. Route it "
        "through alignlab.paths or a config env group instead:\n"
        + "\n".join(offenders)
    )


def test_machine_specific_paths_are_confined_to_env_group(repo_root_path: Path) -> None:
    """Machine-specific paths may exist ONLY under configs/env/.

    The complement of the exclusion above. Without this, adding a directory to
    EXCLUDED_FROM_SCAN would silently widen the hole. Here the scan runs with
    exclusions OFF, and every hit must be inside configs/env/.
    """
    stray: list[str] = []

    for path in _source_files(repo_root_path, apply_exclusions=False):
        rel = path.relative_to(repo_root_path).as_posix()
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:  # pragma: no cover - defensive
            continue
        for number, line in enumerate(text.splitlines(), start=1):
            if _is_prose(line):
                continue
            for pattern in FORBIDDEN.values():
                if pattern.search(line) and not rel.startswith("configs/env"):
                    stray.append(f"{rel}:{number}: {line.strip()}")

    assert not stray, (
        "Machine-specific paths must live only in the configs/env/ group:\n"
        + "\n".join(stray)
    )


def test_storage_roots_come_from_env_vars() -> None:
    """The path layer must read the documented environment variables."""
    source = (Path(__file__).resolve().parents[1] / "src/alignlab/paths.py").read_text(
        encoding="utf-8"
    )
    for var in (
        "ALIGNLAB_OUTPUT_ROOT",
        "ALIGNLAB_CKPT_ROOT",
        "ALIGNLAB_CACHE_ROOT",
    ):
        assert var in source
