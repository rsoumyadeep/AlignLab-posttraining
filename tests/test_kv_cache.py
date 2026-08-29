"""KV cache: correctness first, then the cost model.

THE CENTRAL CLAIM. A KV cache is NOT an approximation. Causal masking
guarantees a past token's K/V can never change when a new token is appended,
so there is nothing to recompute. Cached and uncached generation must produce
the SAME logits - and that is what these tests assert.

Tolerance is float32 round-off (1e-5), not zero: the cached path concatenates
tensors and runs matmuls of different shapes, so the reduction order differs.
That is Tier C from Phase 1C, at a much smaller scale.
"""

from __future__ import annotations

import pytest
import torch

from alignlab.models.generation import generate
from alignlab.models.kv_cache import KVCache
from alignlab.models.shapes import assert_shape
from alignlab.models.transformer import DecoderOnlyTransformer, TransformerConfig

VOCAB, D_MODEL, LAYERS, HEADS = 64, 64, 2, 4
RTOL, ATOL = 1e-5, 1e-5


def make_model(**overrides) -> DecoderOnlyTransformer:
    base = dict(
        vocab_size=VOCAB, d_model=D_MODEL, n_layers=LAYERS, n_heads=HEADS, max_seq_len=64
    )
    base.update(overrides)
    torch.manual_seed(0)
    return DecoderOnlyTransformer(TransformerConfig(**base)).eval()


# ==========================================================================
# the cache object
# ==========================================================================


def test_cache_starts_empty() -> None:
    cache = KVCache(LAYERS)
    assert cache.length(0) == 0
    assert cache.get(0) == (None, None)


def test_cache_grows_by_one_per_step() -> None:
    cache = KVCache(1)
    for step in range(1, 5):
        k = torch.randn(1, 2, 1, 8)
        cached_k, _ = cache.update(0, k, k.clone())
        assert cache.length(0) == step
        assert_shape(cached_k, (1, 2, step, 8), "cached k")


def test_cache_layers_are_independent() -> None:
    cache = KVCache(3)
    cache.update(0, torch.randn(1, 2, 5, 8), torch.randn(1, 2, 5, 8))
    assert cache.length(0) == 5
    assert cache.length(1) == 0


def test_cache_reset_clears_everything() -> None:
    cache = KVCache(2)
    cache.update(0, torch.randn(1, 2, 4, 8), torch.randn(1, 2, 4, 8))
    cache.reset()
    assert cache.length(0) == 0
    assert cache.memory_bytes() == 0


def test_cache_rejects_bad_layer_index() -> None:
    with pytest.raises(IndexError):
        KVCache(2).update(5, torch.randn(1, 1, 1, 8), torch.randn(1, 1, 1, 8))


def test_measured_memory_matches_the_formula() -> None:
    """The cost model in the docs must match reality."""
    cache = KVCache(2)
    for layer in range(2):
        k = torch.randn(1, 4, 10, 16)  # [B, n_kv, T, d]
        cache.update(layer, k, k.clone())

    predicted = KVCache.predicted_bytes(
        batch=1, n_kv_heads=4, seq_len=10, head_dim=16, n_layers=2, bytes_per_element=4
    )
    assert cache.memory_bytes() == predicted


def test_cache_bytes_scale_with_kv_heads() -> None:
    """The GQA/MQA payoff, measured on the real object rather than a formula."""
    def bytes_for(n_kv: int) -> int:
        cache = KVCache(1)
        k = torch.randn(1, n_kv, 32, 16)
        cache.update(0, k, k.clone())
        return cache.memory_bytes()

    assert bytes_for(8) == 8 * bytes_for(1)
    assert bytes_for(4) == 4 * bytes_for(1)


# ==========================================================================
# CORRECTNESS: cached == uncached
# ==========================================================================


def test_cached_incremental_forward_matches_full_forward() -> None:
    """THE test. Feed one token at a time with a cache; compare against a
    single full-sequence forward pass.
    """
    model = make_model()
    ids = torch.randint(0, VOCAB, (2, 10))

    with torch.no_grad():
        full_logits, _ = model(ids)

        cache = KVCache(model.cfg.n_layers)
        stepwise = []
        for t in range(ids.shape[1]):
            logits, _ = model(ids[:, t : t + 1], cache=cache)
            stepwise.append(logits[:, -1, :])
        cached_logits = torch.stack(stepwise, dim=1)

    diff = float((full_logits - cached_logits).abs().max())
    print(f"\n  max|full - cached| = {diff:.3e}  (tolerance {ATOL:.1e})")
    torch.testing.assert_close(full_logits, cached_logits, rtol=RTOL, atol=ATOL)


def test_prompt_then_incremental_matches_full() -> None:
    """The realistic pattern: process the prompt in one pass, then step."""
    model = make_model()
    ids = torch.randint(0, VOCAB, (1, 12))
    split = 7

    with torch.no_grad():
        full, _ = model(ids)

        cache = KVCache(model.cfg.n_layers)
        model(ids[:, :split], cache=cache)  # prompt fills the cache
        tail = []
        for t in range(split, ids.shape[1]):
            logits, _ = model(ids[:, t : t + 1], cache=cache)
            tail.append(logits[:, -1, :])
        cached_tail = torch.stack(tail, dim=1)

    torch.testing.assert_close(full[:, split:], cached_tail, rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("n_kv_heads", [1, 2, 4])
def test_cache_correct_for_every_attention_variant(n_kv_heads: int) -> None:
    """GQA/MQA must not break cache correctness."""
    model = make_model(n_kv_heads=n_kv_heads)
    ids = torch.randint(0, VOCAB, (1, 8))

    with torch.no_grad():
        full, _ = model(ids)
        cache = KVCache(model.cfg.n_layers)
        steps = [model(ids[:, t : t + 1], cache=cache)[0][:, -1, :] for t in range(8)]

    torch.testing.assert_close(full, torch.stack(steps, dim=1), rtol=RTOL, atol=ATOL)


def test_greedy_generation_identical_with_and_without_cache() -> None:
    """End-to-end: the cache must not change what is generated.

    Greedy is used because it is deterministic, so any difference is a real
    bug rather than sampling noise.
    """
    model = make_model()
    prompt = torch.randint(0, VOCAB, (1, 4))

    cached = generate(model, prompt, max_new_tokens=12, greedy=True, use_cache=True)
    uncached = generate(model, prompt, max_new_tokens=12, greedy=True, use_cache=False)

    assert torch.equal(cached, uncached), (
        f"cache changed the output:\n  cached   {cached.tolist()}\n"
        f"  uncached {uncached.tolist()}"
    )


def test_rope_offset_is_what_makes_the_cache_correct() -> None:
    """Regression guard for THE classic KV-cache bug.

    If RoPE were applied at position 0 for every cached step instead of at the
    true absolute position, generation would still run and produce fluent-
    looking output - with silently wrong positional structure. Forcing the
    offset to 0 must visibly break the equivalence, which proves the passing
    test above is actually testing the offset.
    """
    model = make_model()
    ids = torch.randint(0, VOCAB, (1, 6))

    with torch.no_grad():
        full, _ = model(ids)

        cache = KVCache(model.cfg.n_layers)
        broken = []
        original_length = KVCache.length
        try:
            KVCache.length = lambda self, layer_idx=0: 0  # force offset 0
            for t in range(6):
                logits, _ = model(ids[:, t : t + 1], cache=cache)
                broken.append(logits[:, -1, :])
        finally:
            KVCache.length = original_length

    broken_logits = torch.stack(broken, dim=1)
    assert not torch.allclose(full, broken_logits, rtol=RTOL, atol=ATOL), (
        "forcing the RoPE offset to 0 did NOT change the result - so the "
        "equivalence test is not actually exercising the offset"
    )
