"""E15 - verify bitsandbytes actually works on this machine, before relying on it.

WHY A SCRIPT AND NOT A `pip install`. Phase 4 needs 4-bit quantization for
QLoRA. The department server has **no `nvcc` and no root**, which already made
`flash-attn` unbuildable in Phase 2. If bitsandbytes needed to compile CUDA
kernels from source, QLoRA would be blocked here and the honest response would
be to say so, not to substitute something else quietly.

It does not need to compile: modern bitsandbytes ships prebuilt `.so` kernels
per CUDA minor version. This script proves that the shipped kernel matches this
machine's CUDA and that it executes.

AN IMPORT IS NOT A VERIFICATION. `import bitsandbytes` succeeds in plenty of
broken installations - wrong kernel, CPU fallback, silently degraded paths. So
this script does four things an import cannot:

  1. reports WHICH native library actually loaded
  2. runs real quantize/dequantize CUDA kernels and measures the error
  3. compares a Linear4bit forward against a bf16 nn.Linear
  4. measures the actual memory ratio rather than assuming 4x

HYPOTHESES, recorded before running:

  H1  the loaded native library matches torch's CUDA version (12.4)
  H2  NF4 round-trip error is small but NOT zero - quantization is lossy
  H3  NF4 beats FP4 on normally-distributed data. This is the QLoRA paper's
      central claim about NF4: it is information-theoretically optimal for
      data with a normal distribution, which is what neural network weights
      approximately are. FP4 spaces its levels differently.
  H4  4-bit storage is exactly 4x smaller than bf16 (2 bytes -> 0.5 bytes)

Run (server):
    python scripts/experiments/e15_bitsandbytes_environment.py
"""

from __future__ import annotations

import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import torch

from alignlab.paths import repo_root

EVIDENCE = repo_root() / "docs" / "phase4"


def environment_facts() -> dict:
    """Everything a later reader needs to know this run is comparable to theirs."""
    facts = {
        "recorded_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": platform.node(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "cuda_available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count(),
        "bf16_supported": torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False,
        "devices": [],
    }
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        facts["devices"].append(
            {
                "index": i,
                "name": p.name,
                "compute_capability": f"{p.major}.{p.minor}",
                "total_memory_gib": round(p.total_memory / 1024**3, 2),
            }
        )
    for package in ("bitsandbytes", "peft", "transformers", "trl", "accelerate", "triton"):
        try:
            module = __import__(package)
            facts[package] = getattr(module, "__version__", "unknown")
        except Exception as exc:  # noqa: BLE001 - record the failure, do not hide it
            facts[package] = f"IMPORT FAILED: {type(exc).__name__}: {exc}"
    return facts


def main() -> int:
    print("=" * 78)
    print("E15 - bitsandbytes environment verification")
    print("=" * 78)

    facts = environment_facts()
    for key in ("hostname", "python", "torch", "torch_cuda", "cudnn", "bf16_supported"):
        print(f"  {key:<18} {facts[key]}")
    for d in facts["devices"]:
        print(f"  gpu{d['index']:<15} {d['name']} cc{d['compute_capability']} "
              f"{d['total_memory_gib']} GiB")
    for package in ("bitsandbytes", "peft", "triton"):
        print(f"  {package:<18} {facts[package]}")

    if not torch.cuda.is_available():
        print("\nNO CUDA - bitsandbytes cannot be verified here. This script is "
              "server-only by design.")
        return 2

    import bitsandbytes as bnb
    from bitsandbytes.functional import dequantize_4bit, quantize_4bit

    # ------------------------------------------------------- H1: which kernel?
    print("\n--- H1: which native library actually loaded ---")
    import bitsandbytes.cextension as cext

    lib = getattr(cext, "lib", None)
    lib_name = getattr(lib, "_name", None) or str(type(lib).__name__)
    print(f"  {lib_name}")
    expected = f"cuda{torch.version.cuda.replace('.', '')}"
    h1 = expected in str(lib_name)
    print(f"  torch CUDA is {torch.version.cuda} -> expecting '{expected}' in the name")
    print(f"  H1 holds: {h1}")
    if not h1:
        print("  WARNING: kernel/toolkit mismatch. Do NOT trust QLoRA numbers from "
              "this machine until resolved.")

    device = torch.device("cuda")

    # ------------------------------------- H2/H3: real quantization round-trip
    print("\n--- H2/H3: NF4 and FP4 round-trip on N(0,1) data, 4096x4096 bf16 ---")
    torch.manual_seed(0)
    x = torch.randn(4096, 4096, device=device, dtype=torch.bfloat16)
    original_bytes = x.numel() * x.element_size()

    quant = {}
    for qtype in ("nf4", "fp4"):
        packed, state = quantize_4bit(x, quant_type=qtype, blocksize=64)
        restored = dequantize_4bit(packed, state, quant_type=qtype, blocksize=64)
        err = (x.float() - restored.float()).abs()
        rel = float(err.mean() / x.float().abs().mean())
        packed_bytes = packed.numel() * packed.element_size()
        quant[qtype] = {
            "mean_abs_err": float(err.mean()),
            "max_abs_err": float(err.max()),
            "relative_err": rel,
            "packed_bytes": packed_bytes,
        }
        print(f"  {qtype}: mean|err| {err.mean():.5f}  max|err| {err.max():.5f}  "
              f"rel {rel:.4f}  packed {packed_bytes:,} B")

    h2 = 0.0 < quant["nf4"]["mean_abs_err"] < 1.0
    h3 = quant["nf4"]["relative_err"] < quant["fp4"]["relative_err"]
    print(f"\n  H2 holds (error small but non-zero): {h2}")
    print(f"  H3 holds (NF4 beats FP4 on normal data): {h3}")
    print(f"     NF4 rel {quant['nf4']['relative_err']:.4f} vs "
          f"FP4 rel {quant['fp4']['relative_err']:.4f} "
          f"({100 * (1 - quant['nf4']['relative_err'] / quant['fp4']['relative_err']):.1f}% better)")
    print("     NF4's levels are placed at the quantiles of a normal distribution,")
    print("     which is what makes it suited to weights. This is a MEASUREMENT of")
    print("     that claim on random normal data, NOT on real weights.")

    # ------------------------------------------- Linear4bit vs a bf16 Linear
    print("\n--- Linear4bit forward vs bf16 nn.Linear (same weights) ---")
    torch.manual_seed(1)
    reference = torch.nn.Linear(2048, 2048, bias=False).to(device, torch.bfloat16)
    inputs = torch.randn(8, 2048, device=device, dtype=torch.bfloat16)
    expected_out = reference(inputs)

    q4 = bnb.nn.Linear4bit(
        2048, 2048, bias=False, compute_dtype=torch.bfloat16, quant_type="nf4"
    )
    q4.weight = bnb.nn.Params4bit(
        reference.weight.data.clone(), requires_grad=False, quant_type="nf4"
    )
    q4 = q4.to(device)
    actual_out = q4(inputs)

    diff = (expected_out.float() - actual_out.float()).abs()
    forward_rel = float(diff.mean() / expected_out.float().abs().mean())
    print(f"  max|diff| {diff.max():.5f}  mean|diff| {diff.mean():.5f}  rel {forward_rel:.4f}")
    print("  The forward pass DEQUANTIZES to compute_dtype before the matmul -")
    print("  the arithmetic is bf16; only the STORAGE is 4-bit.")

    # --------------------------------------------------------- H4: memory ratio
    print("\n--- H4: storage ratio ---")
    w = reference.weight.data
    bf16_bytes = w.numel() * w.element_size()
    packed = q4.weight.data
    packed_bytes = packed.numel() * packed.element_size()
    ratio = bf16_bytes / packed_bytes
    print(f"  bf16 weight  {bf16_bytes:,} B")
    print(f"  4-bit packed {packed_bytes:,} B   ratio {ratio:.2f}x")
    h4 = abs(ratio - 4.0) < 0.01
    print(f"  H4 holds (exactly 4x): {h4}")
    print("  NOTE: this counts the packed weights ONLY. The quantization state")
    print("  (absmax scales, one per 64-element block) is extra, so a whole")
    print("  quantized MODEL is not 4x smaller - measured properly in E18.")

    print("\n" + "=" * 78)
    print("VERDICT")
    print("=" * 78)
    checks = {"H1 kernel matches torch CUDA": h1,
              "H2 error small but non-zero  ": h2,
              "H3 NF4 beats FP4             ": h3,
              "H4 storage ratio is 4x       ": h4}
    for name, ok in checks.items():
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    usable = h1 and h2 and h4
    print(f"\n  bitsandbytes USABLE for QLoRA on this machine: {usable}")
    if not usable:
        print("  STOP. Do not proceed to QLoRA; report the failure instead.")

    payload = {
        "environment": facts,
        "native_library": str(lib_name),
        "quantization": quant,
        "linear4bit_forward_relative_err": forward_rel,
        "storage_ratio": ratio,
        "hypotheses": {"H1": h1, "H2": h2, "H3": h3, "H4": h4},
        "usable": usable,
    }
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    out = EVIDENCE / "e15_bitsandbytes_environment.json"
    out.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print(f"\n  evidence: {out.relative_to(repo_root())}")
    return 0 if usable else 1


if __name__ == "__main__":
    raise SystemExit(main())
