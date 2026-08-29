"""E7 - LayerNorm vs RMSNorm: parameters, cost, and what is actually dropped.

OBJECTIVE   Quantify what removing the mean-subtraction buys and costs.

HYPOTHESIS  RMSNorm halves the parameters (no bias) and should be somewhat
            faster, because it needs ONE reduction over the feature axis
            (mean of squares) instead of TWO (mean, then variance). It must
            NOT centre its output, and on already-centred input the two should
            coincide.

FINDING THAT CORRECTED THE TIMING CLAIM (preserved deliberately).
            The first version of this experiment compared only our educational
            LayerNorm against our educational RMSNorm, and reported RMSNorm as
            3.5-4.4x faster. That number is an ARTEFACT of comparing two
            unfused implementations. torch.nn.LayerNorm is a fused kernel and
            is ~3x FASTER than our unfused RMSNorm (ratio ~0.30-0.33). The
            fused-LayerNorm column below exists so that the honest comparison
            cannot be omitted.

            What survives: RMSNorm does algorithmically LESS WORK and has HALF
            the parameters. What does NOT survive: any claim that RMSNorm is
            faster than LayerNorm in production. Implementation quality
            dominates the algorithmic difference at this scale.

CONFIG      B=8, T=512, d_model in {512, 1024, 4096}; float32 CPU / bf16 CUDA.

Run: python scripts/experiments/e7_norm_comparison.py
"""
from __future__ import annotations

import time

import torch

from alignlab.models.normalization import LayerNorm, RMSNorm
from alignlab.seeding import set_seed

SEED, B, T, WARMUP, ITERS = 20260829, 8, 512, 5, 30
D_MODELS = (512, 1024, 4096)


def bench(module, x) -> float:
    with torch.no_grad():
        for _ in range(WARMUP):
            module(x)
        if x.is_cuda:
            torch.cuda.synchronize()
        times = []
        for _ in range(ITERS):
            t0 = time.perf_counter()
            module(x)
            if x.is_cuda:
                torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000)
    times.sort()
    return times[len(times) // 2]


def main() -> None:
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    print("=" * 78)
    print("E7 - LayerNorm vs RMSNorm")
    print("=" * 78)
    print(f"torch {torch.__version__} | device {device} | dtype {dtype}")
    print(f"B={B} T={T} | median of {ITERS} iters | seed={SEED}\n")

    print(f"{'d_model':>8} {'LN par':>8} {'RMS par':>8} {'oursLN':>8} "
          f"{'fusedLN':>8} {'oursRMS':>8} {'vs oursLN':>10} {'vs fusedLN':>11}")
    print("-" * 78)

    for d in D_MODELS:
        ln = LayerNorm(d).to(device=device, dtype=dtype).eval()
        rms = RMSNorm(d).to(device=device, dtype=dtype).eval()
        x = torch.randn(B, T, d, device=device, dtype=dtype)

        fused = torch.nn.LayerNorm(d).to(device=device, dtype=dtype).eval()

        ln_p = sum(p.numel() for p in ln.parameters())
        rms_p = sum(p.numel() for p in rms.parameters())
        ln_t = bench(ln, x)
        fused_t = bench(fused, x)
        rms_t = bench(rms, x)

        print(f"{d:>8} {ln_p:>8,} {rms_p:>8,} {ln_t:>8.3f} {fused_t:>8.3f} "
              f"{rms_t:>8.3f} {ln_t / rms_t:>9.2f}x {fused_t / rms_t:>10.2f}x")

    # what is actually dropped: centring
    x = torch.randn(4, 16, 512) + 10.0
    ln_mean = float(LayerNorm(512)(x).mean())
    rms_mean = float(RMSNorm(512)(x).mean())
    print(f"\ninput offset by +10.0:")
    print(f"  LayerNorm  output mean = {ln_mean:+.6f}   (centred)")
    print(f"  RMSNorm    output mean = {rms_mean:+.6f}   (offset SURVIVES)")
    print("\nThat surviving offset IS the difference. RMSNorm normalises")
    print("magnitude only; whether the lost centring matters is empirical.")
    print("=" * 78)


if __name__ == "__main__":
    main()
