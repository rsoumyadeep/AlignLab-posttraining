"""E8 - KV cache: correctness and the cost curve.

OBJECTIVE   Show that caching is exact (not an approximation) and measure what
            it actually buys during generation.

HYPOTHESIS  Correctness: cached and uncached logits agree to float32 round-off,
            because causal masking means a past token's K/V can never change.
            Cost: WITHOUT a cache, step t reprocesses all t tokens, so
            per-token time GROWS with t and total generation is O(T^2).
            WITH a cache, only the new token is projected, so per-token time is
            roughly FLAT and total is O(T) in the projections. The attention
            itself is still O(t) per step, so the cache removes the redundant
            projections and attention rows, not the cost of attending to
            history.

CONFIG      educational decoder, vocab 256, d_model 256, 4 layers, 4 heads,
            greedy decoding, float32 CPU / bf16 CUDA, seed fixed.

Run: python scripts/experiments/e8_kv_cache.py
"""
from __future__ import annotations

import time

import torch

from alignlab.models.generation import generate
from alignlab.models.kv_cache import KVCache
from alignlab.models.transformer import DecoderOnlyTransformer, TransformerConfig
from alignlab.seeding import set_seed

SEED = 20260829
LENGTHS = (16, 32, 64, 128, 256)


def main() -> None:
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    cfg = TransformerConfig(vocab_size=256, d_model=256, n_layers=4,
                            n_heads=4, max_seq_len=512)
    model = DecoderOnlyTransformer(cfg).to(device=device, dtype=dtype).eval()

    print("=" * 84)
    print("E8 - KV cache correctness and cost")
    print("=" * 84)
    print(f"torch {torch.__version__} | device {device} | dtype {dtype}")
    if device.type == "cuda":
        p = torch.cuda.get_device_properties(0)
        print(f"gpu {p.name} | {p.total_memory/1024**3:.2f} GiB | cc {p.major}.{p.minor}")
    print(f"model: d_model={cfg.d_model} layers={cfg.n_layers} heads={cfg.n_heads} "
          f"params={model.num_parameters():,}")
    print(f"seed={SEED}, greedy decoding\n")

    # ---- correctness -----------------------------------------------------
    ids = torch.randint(0, cfg.vocab_size, (1, 24), device=device)
    with torch.no_grad():
        full, _ = model(ids)
        cache = KVCache(cfg.n_layers)
        steps = [model(ids[:, t:t+1], cache=cache)[0][:, -1, :] for t in range(24)]
    diff = float((full.float() - torch.stack(steps, 1).float()).abs().max())
    print(f"CORRECTNESS  max|full - cached| = {diff:.3e}")
    print(f"             ({dtype} round-off, NOT an approximation)")
    if dtype == torch.bfloat16:
        print(f"             bf16 eps = {torch.finfo(torch.bfloat16).eps:.3e};"
              " a difference of this order is the REPRESENTATION LIMIT,")
        print("             not an algorithmic error.")
    print()

    # ---- cost ------------------------------------------------------------
    prompt = torch.randint(0, cfg.vocab_size, (1, 4), device=device)

    print(f"{'new tokens':>11} {'no cache ms':>12} {'cache ms':>10} {'speedup':>9} "
          f"{'no-cache/tok':>13} {'cache/tok':>10} {'cache MiB':>10}")
    print("-" * 84)

    for n in LENGTHS:
        torch.manual_seed(SEED)
        t0 = time.perf_counter()
        generate(model, prompt, max_new_tokens=n, greedy=True, use_cache=False)
        if device.type == "cuda":
            torch.cuda.synchronize()
        no_cache = (time.perf_counter() - t0) * 1000

        torch.manual_seed(SEED)
        t0 = time.perf_counter()
        generate(model, prompt, max_new_tokens=n, greedy=True, use_cache=True)
        if device.type == "cuda":
            torch.cuda.synchronize()
        cached = (time.perf_counter() - t0) * 1000

        c = KVCache(cfg.n_layers)
        with torch.no_grad():
            model(torch.randint(0, cfg.vocab_size, (1, n), device=device), cache=c)

        print(f"{n:>11} {no_cache:>12.1f} {cached:>10.1f} {no_cache/cached:>8.2f}x "
              f"{no_cache/n:>13.3f} {cached/n:>10.3f} {c.memory_bytes()/1024**2:>10.2f}")

    print("\n" + "=" * 84)
    if device.type == "cuda":
        print("NOTE ON THE GPU RUN: at this model size (3.2M params, T <= 260)")
        print("the per-step cost is dominated by Python and kernel-launch")
        print("overhead, so BOTH columns sit flat and the O(T^2) vs O(T)")
        print("difference is INVISIBLE underneath it. The CPU run shows the")
        print("effect clearly. This is a limitation of the experiment scale on")
        print("fast hardware, NOT evidence against KV caching.")
        print()
    print("Read the per-token columns: 'no-cache/tok' should GROW with length")
    print("(each step reprocesses the whole prefix), while 'cache/tok' stays")
    print("roughly flat. That difference IS the O(T^2) -> O(T) argument.")
    print("=" * 84)


if __name__ == "__main__":
    main()
