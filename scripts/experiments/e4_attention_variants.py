"""E4 - MHA vs MQA vs GQA: parameters, KV cache, memory and latency.

OBJECTIVE
    Quantify what is actually saved by sharing key/value heads, and show WHERE
    the saving appears - which is not where people usually assume.

HYPOTHESIS
    Let H = query heads, H_kv = key/value heads, d_k = d_model / H.
      * PARAMETERS: W_Q and W_O are fixed at d_model^2 each; W_K and W_V are
        d_model * H_kv * d_k each. So fewer KV heads shrink parameters, but
        only the W_K/W_V half - never below the W_Q/W_O floor.
      * KV CACHE: 2 * B * H_kv * T * d_k * bytes. LINEAR in H_kv, so MQA cuts
        it by exactly a factor of H. This is the dominant win.
      * COMPUTE: after repeat_kv the attention matmuls are IDENTICAL in shape,
        so FLOPs are essentially unchanged. Latency should therefore be broadly
        similar - GQA is a MEMORY optimisation, not a FLOP optimisation.

CONFIGURATION
    d_model=1024, H=16, so d_k=64. H_kv in {16, 8, 4, 2, 1}.
    Latency: B=4, T=512, float32 on CPU / bfloat16 on CUDA.
    KV-cache arithmetic quoted at B=1, T=4096, bf16 (2 bytes).

Run:  python scripts/experiments/e4_attention_variants.py
On the server: CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/.../e4_...py
"""

from __future__ import annotations

import time

import torch

from alignlab.models.attention import MultiHeadAttention, causal_mask
from alignlab.seeding import set_seed

SEED = 20260829
D_MODEL, N_HEADS = 1024, 16
D_K = D_MODEL // N_HEADS
KV_HEAD_VALUES = (16, 8, 4, 2, 1)

BATCH, SEQ = 4, 512
CACHE_BATCH, CACHE_SEQ, CACHE_BYTES = 1, 4096, 2  # bf16

WARMUP, ITERS = 3, 10


def kind(n_kv: int) -> str:
    if n_kv == N_HEADS:
        return "MHA"
    return "MQA" if n_kv == 1 else "GQA"


def kv_cache_bytes(n_kv: int) -> int:
    """2 (K and V) * B * H_kv * T * d_k * bytes_per_element."""
    return 2 * CACHE_BATCH * n_kv * CACHE_SEQ * D_K * CACHE_BYTES


def benchmark(layer: MultiHeadAttention, device: torch.device, dtype: torch.dtype):
    """Return (median latency ms, peak CUDA memory MiB or None)."""
    layer = layer.to(device=device, dtype=dtype).eval()
    x = torch.randn(BATCH, SEQ, D_MODEL, device=device, dtype=dtype)
    mask = causal_mask(SEQ, device=device)

    with torch.no_grad():
        for _ in range(WARMUP):
            layer(x, mask=mask)

    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    times = []
    with torch.no_grad():
        for _ in range(ITERS):
            start = time.perf_counter()
            layer(x, mask=mask)
            if device.type == "cuda":
                torch.cuda.synchronize()
            times.append((time.perf_counter() - start) * 1000.0)

    peak = (
        torch.cuda.max_memory_allocated() / 1024**2 if device.type == "cuda" else None
    )
    times.sort()
    return times[len(times) // 2], peak


def main() -> None:
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    print("=" * 86)
    print("E4 - MHA vs MQA vs GQA")
    print("=" * 86)
    print(f"torch      : {torch.__version__}")
    print(f"device     : {device}")
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(0)
        print(f"gpu        : {props.name} | {props.total_memory / 1024**3:.2f} GiB "
              f"| cc {props.major}.{props.minor}")
    print(f"dtype      : {dtype}")
    print(f"d_model={D_MODEL} n_heads={N_HEADS} d_k={D_K}")
    print(f"latency    : B={BATCH}, T={SEQ}, median of {ITERS} iters")
    print(f"kv cache   : B={CACHE_BATCH}, T={CACHE_SEQ}, {CACHE_BYTES} bytes/elt")
    print()

    header = (
        f"{'variant':>8} {'H_kv':>5} {'params':>12} {'vs MHA':>8} "
        f"{'KV cache MiB':>13} {'vs MHA':>8} {'latency ms':>11}"
    )
    if device.type == "cuda":
        header += f" {'peak MiB':>10}"
    print(header)
    print("-" * len(header))

    baseline_params = None
    baseline_cache = None

    for n_kv in KV_HEAD_VALUES:
        layer = MultiHeadAttention(D_MODEL, N_HEADS, n_kv_heads=n_kv)
        params = sum(p.numel() for p in layer.parameters())
        cache = kv_cache_bytes(n_kv)

        if baseline_params is None:
            baseline_params, baseline_cache = params, cache

        latency, peak = benchmark(layer, device, dtype)

        row = (
            f"{kind(n_kv):>8} {n_kv:>5} {params:>12,} "
            f"{params / baseline_params:>7.2%} "
            f"{cache / 1024**2:>13.1f} {cache / baseline_cache:>7.2%} "
            f"{latency:>11.2f}"
        )
        if peak is not None:
            row += f" {peak:>10.1f}"
        print(row)

        del layer
        if device.type == "cuda":
            torch.cuda.empty_cache()

    print()
    print("=" * 86)
    print("Expected reading:")
    print("  * params fall, but only the W_K/W_V half - W_Q and W_O are a floor")
    print("  * KV cache falls LINEARLY with H_kv: MQA is 1/H of MHA")
    print("  * latency is broadly similar - after repeat_kv the matmuls have the")
    print("    same shape, so GQA is a MEMORY win, not a FLOP win")
    print("=" * 86)


if __name__ == "__main__":
    main()
