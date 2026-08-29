"""Device resolution and hardware description.

The local machine has a 4 GB Pascal GPU that PyTorch cannot use (the installed
build is CPU-only) and the department server has Ampere GPUs. Code must not
assume either. Every device decision goes through resolve_device, and the
resulting hardware description is recorded in the run manifest so a measured
number can always be attributed to the hardware that produced it.

No fabrication rule: describe_hardware reports only what torch actually
returns. If CUDA is unavailable it says so - it never falls back to describing
a GPU it merely believes is installed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

from alignlab.logging_utils import get_logger

logger = get_logger(__name__)

AUTO = "auto"


@dataclass
class DeviceInfo:
    """Resolved device plus the capability facts that matter for training."""

    device: str
    cuda_available: bool
    device_count: int
    torch_version: str
    torch_cuda_version: str | None
    bf16_supported: bool
    devices: list[dict[str, Any]] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "device": self.device,
            "cuda_available": self.cuda_available,
            "device_count": self.device_count,
            "torch_version": self.torch_version,
            "torch_cuda_version": self.torch_cuda_version,
            "bf16_supported": self.bf16_supported,
            "devices": self.devices,
        }


def resolve_device(requested: str = AUTO) -> torch.device:
    """Resolve a device string to a concrete torch.device.

    Args:
        requested: "auto", "cpu", "cuda", or an explicit "cuda:N".

    "auto" selects CUDA when it is genuinely available and CPU otherwise.
    An explicit "cuda" request on a machine without CUDA raises rather than
    silently degrading to CPU - a training run that quietly falls back to CPU
    would produce numbers that are not comparable to the GPU runs they would be
    tabulated beside.
    """
    requested = (requested or AUTO).strip().lower()

    if requested == AUTO:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if requested == "cpu":
        return torch.device("cpu")

    if requested.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError(
                f"Device {requested!r} was requested but torch.cuda.is_available() "
                f"is False. Installed torch is {torch.__version__} "
                f"(torch.version.cuda={torch.version.cuda}). Refusing to fall back "
                "to CPU silently, because CPU results are not comparable to GPU "
                "results. Pass device=cpu explicitly if CPU is intended."
            )
        index = 0
        if ":" in requested:
            index = int(requested.split(":", 1)[1])
        if index >= torch.cuda.device_count():
            raise RuntimeError(
                f"Device {requested!r} was requested but only "
                f"{torch.cuda.device_count()} CUDA device(s) are visible."
            )
        return torch.device(requested)

    raise ValueError(f"Unrecognised device string: {requested!r}")


def bf16_supported() -> bool:
    """Whether bfloat16 is usable on the current CUDA device.

    False on CPU-only machines and on pre-Ampere GPUs. This is the flag that
    decides whether mixed-precision training is worth enabling at all.
    """
    if not torch.cuda.is_available():
        return False
    try:
        return bool(torch.cuda.is_bf16_supported())
    except Exception:  # pragma: no cover - older torch without the helper
        return False


def describe_hardware() -> DeviceInfo:
    """Describe the hardware torch can actually see.

    Reports observed facts only. On a CPU-only machine devices is empty - it
    does not describe a GPU that torch cannot use.
    """
    cuda_available = torch.cuda.is_available()
    devices: list[dict[str, Any]] = []

    if cuda_available:
        for index in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(index)
            devices.append(
                {
                    "index": index,
                    "name": props.name,
                    "total_memory_gib": round(props.total_memory / 1024**3, 2),
                    "compute_capability": f"{props.major}.{props.minor}",
                    "multi_processor_count": props.multi_processor_count,
                }
            )

    return DeviceInfo(
        device=str(resolve_device(AUTO)),
        cuda_available=cuda_available,
        device_count=torch.cuda.device_count() if cuda_available else 0,
        torch_version=torch.__version__,
        torch_cuda_version=torch.version.cuda,
        bf16_supported=bf16_supported(),
        devices=devices,
    )
