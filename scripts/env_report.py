"""Print a factual report of the current AlignLab environment.

Run this on either machine to see what is actually available. Every line is an
observation from the running process - nothing here is inferred, and hardware
that torch cannot use is not reported as usable.

    python scripts/env_report.py
    python scripts/env_report.py --json
"""

from __future__ import annotations

import argparse
import json
import signal
import sys

from alignlab import __version__
from alignlab.device import describe_hardware
from alignlab.env_detect import describe_platform, detect_environment
from alignlab.manifest import git_info, package_versions, relevant_env_vars
from alignlab.paths import describe_roots
from alignlab.preemption import available_preemption_signals, in_slurm_job


def collect() -> dict:
    """Gather every observable environment fact."""
    return {
        "alignlab_version": __version__,
        "environment": detect_environment(),
        "platform": describe_platform(),
        "hardware": describe_hardware().as_dict(),
        "packages": package_versions(),
        "paths": describe_roots(),
        "env_vars": relevant_env_vars(),
        "signals": {
            "preemption_signals": [s.name for s in available_preemption_signals()],
            "has_sigusr1": hasattr(signal, "SIGUSR1"),
            "in_slurm_job": in_slurm_job(),
        },
        "git": git_info(),
        "python_executable": sys.executable,
    }


def _print_section(title: str, rows: dict) -> None:
    print(f"\n--- {title} ---")
    width = max((len(str(k)) for k in rows), default=0)
    for key, value in rows.items():
        print(f"  {str(key):<{width}} : {value}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="emit JSON")
    args = parser.parse_args()

    data = collect()

    if args.json:
        print(json.dumps(data, indent=2, default=str))
        return 0

    print("=" * 70)
    print(f"AlignLab environment report (version {data['alignlab_version']})")
    print(f"Detected environment: {data['environment']}")
    print("=" * 70)

    _print_section("Platform", data["platform"])
    _print_section("Paths", data["paths"])
    _print_section("Git", {k: v for k, v in data["git"].items() if k != "dirty_files"})

    hardware = data["hardware"]
    _print_section(
        "Hardware",
        {
            "device": hardware["device"],
            "cuda_available": hardware["cuda_available"],
            "device_count": hardware["device_count"],
            "torch_version": hardware["torch_version"],
            "torch_cuda_version": hardware["torch_cuda_version"],
            "bf16_supported": hardware["bf16_supported"],
        },
    )
    if hardware["devices"]:
        for device in hardware["devices"]:
            _print_section(f"GPU {device['index']}", device)
    else:
        print("\n  No CUDA device is usable by the installed torch build.")
        print("  (A GPU may be physically present; torch cannot use it.)")

    _print_section("Signals", data["signals"])
    _print_section("Packages", data["packages"])
    _print_section("AlignLab env vars", data["env_vars"])

    print("\nEvery line above is an observation, not an assumption.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
