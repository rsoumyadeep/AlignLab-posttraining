# The KV Cache

**WP6.** Follows directly from the generation asymmetry in
[[phase2-decoder-only-and-generation]] §5.

---

## 1. The redundancy

Generation feeds the model its own output:

```
step 1:  [a]           → predict b
step 2:  [a, b]        → predict c
step 3:  [a, b, c]     → predict d
```

At step 3, K and V for tokens *a* and *b* are recomputed — and they are
**provably identical** to what step 2 computed, because the model is causal and
nothing after a position can affect it.

Generating T tokens costs `Σ O(t) = O(T²)` key/value projections for work that
is entirely redundant.

## 2. The fix

Keep K and V for every position already processed. Each step projects only the
**one** new token, appends its K/V, and attends the single new query over the
whole cached history:

```
without cache:   Q [B,H,t,d]   K,V [B,H,t,d]     ← everything recomputed
with cache:      Q [B,H,1,d]   K,V [B,H,t,d]     ← only the new token projected
```

## 3. Why this is exact, not an approximation

This is the point most explanations skip.

**Causal masking guarantees** that the representation at position *i* depends
only on positions ≤ *i*. A past token's K and V therefore *cannot* change when
a new token is appended. The cache recomputes nothing because **there is nothing
to recompute**.

So cached and uncached generation must produce **identical** logits — a
falsifiable claim, and the one the tests assert.

*Measured (E8):* `max|full − cached| = 8.345e-07`, i.e. float32 round-off. The
residual difference comes from concatenation and differently-shaped matmuls
changing the reduction order — the Tier C phenomenon from Phase 1C at small
scale, not an algorithmic difference.

*End-to-end:* greedy generation with and without the cache produces
**bit-identical token sequences** (`torch.equal`).

---

## 4. Experiment E8 — the cost curve

`scripts/experiments/e8_kv_cache.py`. Educational decoder, 3.2M params,
d_model=256, 4 layers, greedy, CPU float32, seed 20260829.

| new tokens | no cache | cache | speedup | **no-cache ms/token** | **cache ms/token** | cache MiB |
|---|---|---|---|---|---|---|
| 16 | 98.5 ms | 89.4 ms | 1.10× | 6.159 | 5.587 | 0.12 |
| 32 | 203.2 ms | 172.6 ms | 1.18× | 6.350 | 5.393 | 0.25 |
| 64 | 535.4 ms | 351.5 ms | 1.52× | 8.365 | 5.492 | 0.50 |
| 128 | 1423.8 ms | 718.6 ms | 1.98× | 11.123 | 5.614 | 1.00 |
| 256 | 3973.2 ms | 1340.8 ms | **2.96×** | **15.520** | **5.238** | 2.00 |

**Read the per-token columns — that is the whole argument.** Without a cache the
per-token cost *grows* 6.16 → 15.52 ms (2.5×) because each step reprocesses the
entire prefix. With a cache it stays **flat** at ~5.4 ms. The speedup therefore
grows without bound as generation lengthens.

**What E8 proves.** Caching is exact to round-off, and converts a per-token cost
that grows with position into one that does not.

**What E8 does NOT prove.**
- Not an asymptotic measurement — 256 tokens on a 3.2M-parameter toy model.
- The speedup is *understated* versus production: at this tiny size, per-step
  Python and kernel-launch overhead (~5 ms/token, the flat floor) dominates.
  With a real model the compute fraction is far larger and the cache matters
  more.
- CPU float32 only; the GPU run is E8-GPU on the server.

### A precise claim about complexity

The cache removes redundant **projections**, not the cost of attending to
history. Per step the new query must still attend over all *t* keys, so
attention remains O(t) per step and O(T²) in total. What becomes O(T) is the
projection work — which is why the flat per-token line is flat *at 5.4 ms* and
not at zero.

---

## 5. The cost: memory

```
KV cache bytes = 2 · B · n_kv_heads · T · head_dim · bytes · n_layers
```

*Verified:* `test_measured_memory_matches_the_formula` compares the formula
against the real object's byte count — they match exactly.

Linear in `n_kv_heads`, which is precisely why GQA and MQA exist (WP2). At long
context and large batch this term dominates inference memory, and it is the
reason a served model's maximum batch size is usually a KV-cache limit rather
than a weights limit.

*Verified:* 8 KV heads use exactly 8× the bytes of 1.

---

## 6. The bug this design is built to avoid

**RoPE must be applied at the token's absolute position.** During cached
decoding the new token sits at index `past_len`, not 0.

Get it wrong and generation still runs, still produces fluent-looking text, and
is silently wrong — every token thinks it is at position 0, destroying all
relative-position structure. Nothing crashes.

*Guarded by* `test_rope_offset_is_what_makes_the_cache_correct`, which forces
the offset to 0 and asserts the result **does** change. Without that guard, the
passing equivalence test would not prove the offset is being exercised at all.

A second detail: keys are cached **after** rotation. Caching pre-rotation keys
would require re-rotating the whole history every step — defeating the purpose.

---

## My Understanding

> **[USER CHECKPOINT — DEFERRED — USER EXPLAIN-BACK REQUIRED]**
>
> **Checkpoint 9 — Explain KV caching.**
> What exactly is cached and why is it safe to reuse? Write the per-step shapes
> with and without a cache. Why is this *exact* rather than an approximation —
> which property of the architecture guarantees it? Give the memory formula and
> explain why it, not the weights, usually limits serving batch size.
>
> Then write the incremental-decoding loop as pseudocode from memory.
>
> And, without looking:
> - A colleague's cached generation is fluent but subtly worse than uncached.
>   Nothing errors. What is your first hypothesis?
> - Does the KV cache make generation O(T) overall? Answer precisely.
> - Why cache the keys *after* applying RoPE rather than before?

**Related:** [[phase2-decoder-only-and-generation]] ·
[[phase2-multihead-and-variants]] · [[phase2-flash-attention-concepts]]
