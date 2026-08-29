"""E10 - SDPA backends: same maths, different memory.

OBJECTIVE   Test the central Flash Attention claim empirically: that an
            IO-aware attention kernel computes the SAME function as the naive
            one while using far less memory.

HYPOTHESIS  torch's F.scaled_dot_product_attention dispatches to several
            backends. MATH materialises the [B,H,T,T] score matrix; EFFICIENT
            and FLASH do not. Therefore:
              * all backends agree within float tolerance (the maths is exact)
              * MATH peak memory grows QUADRATICALLY with sequence length
              * FLASH/EFFICIENT peak memory grows roughly LINEARLY
              * the gap widens with T

WHAT THIS IS NOT. This is PyTorch's implementation of the idea, not the
reference FlashAttention-2 kernel. flash-attn cannot be built on the department
server (nvcc absent, no root - verified in Phase 1B). So this measures the
CONCEPT through torch, and the paper's own numbers are NOT reproduced here.

CONFIG      B=4, H=16, d_k=64, dtype bf16 on CUDA (FLASH requires fp16/bf16),
            T in {512, 1024, 2048, 4096}, causal.

Run on the server: CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/experiments/e10_sdpa_backends.py
"""
from __future__ import annotations

import time

import torch
import torch.nn.functional as F

from alignlab.seeding import set_seed

SEED, B, H, D_K = 20260829, 4, 16, 64
LENGTHS = (512, 1024, 2048, 4096)
WARMUP, ITERS = 3, 10


def measure(backend, name, q, k, v):
    """Return (ms, peak MiB, output) or None if the backend refuses."""
    from torch.nn.attention import sdpa_kernel

    try:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        base = torch.cuda.memory_allocated()

        with sdpa_kernel(backend):
            with torch.no_grad():
                for _ in range(WARMUP):
                    out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()

                times = []
                for _ in range(ITERS):
                    t0 = time.perf_counter()
                    out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
                    torch.cuda.synchronize()
                    times.append((time.perf_counter() - t0) * 1000)

        peak = (torch.cuda.max_memory_allocated() - base) / 1024**2
        times.sort()
        return times[len(times) // 2], peak, out.float()
    except Exception as exc:
        print(f"    {name}: unavailable ({type(exc).__name__}: {str(exc)[:70]})")
        return None


def main() -> None:
    if not torch.cuda.is_available():
        print("E10 requires CUDA. torch.cuda.is_available() is False - SKIPPED.")
        print(f"torch {torch.__version__}")
        return

    from torch.nn.attention import SDPBackend

    set_seed(SEED)
    props = torch.cuda.get_device_properties(0)
    dtype = torch.bfloat16

    print("=" * 92)
    print("E10 - SDPA backends: same maths, different memory")
    print("=" * 92)
    print(f"torch      : {torch.__version__}")
    print(f"gpu        : {props.name} | {props.total_memory/1024**3:.2f} GiB | "
          f"cc {props.major}.{props.minor}")
    print(f"cuda       : {torch.version.cuda} | cudnn {torch.backends.cudnn.version()}")
    print(f"dtype      : {dtype} | B={B} H={H} d_k={D_K} | causal | "
          f"median of {ITERS} iters | seed {SEED}\n")

    backends = [
        (SDPBackend.MATH, "MATH"),
        (SDPBackend.EFFICIENT_ATTENTION, "EFFICIENT"),
        (SDPBackend.FLASH_ATTENTION, "FLASH"),
    ]

    print(f"{'T':>6} {'backend':>11} {'ms':>9} {'peak MiB':>10} {'MiB/T':>9} "
          f"{'max|diff vs MATH|':>19}")
    print("-" * 92)

    for T in LENGTHS:
        q = torch.randn(B, H, T, D_K, device="cuda", dtype=dtype)
        k = torch.randn(B, H, T, D_K, device="cuda", dtype=dtype)
        v = torch.randn(B, H, T, D_K, device="cuda", dtype=dtype)

        reference = None
        for backend, name in backends:
            result = measure(backend, name, q, k, v)
            if result is None:
                continue
            ms, peak, out = result
            if name == "MATH":
                reference = out
                diff = "-"
            else:
                diff = f"{float((out - reference).abs().max()):.3e}" if reference is not None else "n/a"
            print(f"{T:>6} {name:>11} {ms:>9.3f} {peak:>10.1f} {peak/T:>9.4f} {diff:>19}")

        del q, k, v
        torch.cuda.empty_cache()
        print()

    print("=" * 92)
    print("Read two things:")
    print("  1. the diff column - all backends compute the SAME function")
    print("  2. 'MiB/T' - for MATH this GROWS with T (peak is quadratic);")
    print("     for FLASH/EFFICIENT it should stay roughly constant (linear peak)")
    print("=" * 92)


if __name__ == "__main__":
    main()
