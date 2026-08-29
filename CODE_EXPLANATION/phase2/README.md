# Phase 2 — Code Explanation

Describes the code that **actually exists** at commit `a62d8d4` on
`phase-1-foundation`. If this document and the source disagree, the source is
right and this is stale.

Theory lives in `STUDY_WITH_CLAUDE/phase2/`; this file is the *implementation*.

---

## 1. Files

| File | Responsibility |
|---|---|
| `models/shapes.py` | `assert_shape` — makes a shape claim executable |
| `models/attention.py` | scaled dot-product attention + `MultiHeadAttention` (MHA/MQA/GQA), `causal_mask` |
| `models/attention_numpy.py` | NumPy reference, same algorithm |
| `models/attention_pure.py` | pure-Python reference — lists and `math` only |
| `models/positional.py` | learned absolute, sinusoidal, RoPE, ALiBi |
| `models/normalization.py` | educational `LayerNorm`, `RMSNorm` |
| `models/feedforward.py` | classic `FeedForward`, `SwiGLU` |
| `models/kv_cache.py` | `KVCache` + cost model |
| `models/transformer.py` | `TransformerConfig`, `CausalSelfAttention`, `DecoderBlock`, `DecoderOnlyTransformer` |
| `models/generation.py` | greedy / temperature / top-k / top-p / sampling / beam search |
| `scripts/experiments/` | E1, E2, E4, E7, E8, E9, E10, `tiny_lm.py`, WP9 reconciliation |
| `scripts/fetch_qwen_config.py` | metadata-only download with allowlist + denylist |

---

## 2. Data flow

```
input_ids [B,T]
   │ nn.Embedding
   ▼
x [B,T,d_model] ──┐
   │              │  × n_layers (DecoderBlock)
   │   ┌──────────┴───────────────────────────────┐
   │   │ RMSNorm ─► CausalSelfAttention ─► + x    │  residual
   │   │   │  w_q/w_k/w_v ─► split heads          │
   │   │   │  RoPE(q,k, offset=cache.length)      │
   │   │   │  cache.update(layer_idx, k, v)       │
   │   │   │  repeat_kv (GQA/MQA)                 │
   │   │   │  sdpa(q,k,v, causal_mask) ─► w_o     │
   │   │ RMSNorm ─► SwiGLU ─► + h                 │  residual
   │   └──────────────────────────────────────────┘
   ▼
final RMSNorm ─► lm_head (tied) ─► logits [B,T,V]
                                      │
                                      └─► cross_entropy(targets) ─► loss
```

---

## 3. Equation → code

```
Q = X W_Q                self.w_q(x).view(B,T,H,d_k).transpose(1,2)
K = X W_K                self.w_k(x).view(B,T,H_kv,d_k).transpose(1,2)
V = X W_V                self.w_v(x).view(B,T,H_kv,d_k).transpose(1,2)

Q Kᵀ                     query @ key.transpose(-2, -1)          [.., T_q, T_k]
÷ √d_k                   scores * (1.0 / math.sqrt(d_k))
mask                     scores.masked_fill(~mask, -inf)        BEFORE softmax
softmax over keys        torch.softmax(scores, dim=-1)          dim=-1 == keys
· V                      weights @ value                        [.., T_q, d_v]
concat heads             .transpose(1,2).contiguous().view(B,T,H*d_v)
W_O                      self.w_o(merged)
```

Each line of that table is one line of `scaled_dot_product_attention` /
`MultiHeadAttention.forward`, in order.

---

## 4. Key functions

### `attention.scaled_dot_product_attention(q, k, v, mask, scale, ...)`
Works for any number of leading batch dims, so it serves both `[B,T,d]` and
`[B,H,T,d]` callers. Returns `(context, weights)` — the weights are returned
because they are the most useful diagnostic object in the architecture.
`scale=1.0` disables scaling, which is what E1 uses.

### `attention.causal_mask(seq_len, device, key_len)`
`True` means **allowed**. When `key_len > seq_len` the queries are taken to be
the **last** `seq_len` positions — the KV-cache case, where one new query faces
many cached keys. Raises if `key_len < seq_len`.

### `MultiHeadAttention._repeat_kv(x, n_rep)`
`repeat_interleave`, **not** `repeat` — each KV head must serve a *contiguous*
group of query heads. `repeat` would pair query heads with the wrong group and
train to a worse model without ever erroring.

### `positional.RotaryPositionalEmbedding.forward(q, k, offset)`
`offset` is the absolute position of the first token — `cache.length(layer)`
during cached decoding. Uses the **rotate-half** convention (Llama/Qwen), not
interleave-adjacent; the two are equivalent up to a permutation, and mixing them
is silent.

### `kv_cache.KVCache.update(layer_idx, key, value)`
Grows by `torch.cat`. Production caches pre-allocate and write into slices,
avoiding a reallocation per token; concatenation is used here because it makes
the mechanism obvious. Keys must arrive **already rotated**.

### `generation.top_p_filter(logits, p)`
Sort → cumulative softmax → keep through the crossing token → **unsort by
`scatter(-1, sorted_idx, ...)`**. See §6 for the bug this line used to contain.

---

## 5. Design decisions

**No `einops`.** Explicit `view`/`transpose`/`contiguous`. `einops` would hide
exactly the shape mechanics this phase teaches. Cost: verbosity, accepted.

**`F.scaled_dot_product_attention` as a comparison target only.** Used in tests
(agreement to 3.576e-07) and in E10, never as a substitute in the educational
path.

**Modern-LLM-shaped integrated model.** Pre-norm, RMSNorm, RoPE, SwiGLU,
GQA-capable, no biases. The 2017 variants exist as separate modules. Rejected:
building the 2017 original as the integrated model — it would be historically
faithful and would not resemble what Phase 3 fine-tunes.
**Known divergence from Qwen:** Qwen uses attention **QKV bias**; we do not
(WP9).

**Educational `LayerNorm` alongside `torch.nn.LayerNorm`.** Ours is tested
*against* torch's, which pins the biased-variance and eps-inside-sqrt details.

**Causal masking is not configurable.** There is deliberately no
`use_causal_mask=False` flag. E2 disables it by monkey-patching in the
experiment script, so production code has no switch that can silently turn off
causality.

**Pure Python is unbatched.** Batching adds bookkeeping and teaches nothing new.

---

## 6. Bugs found, and why the tests missed them

### 6a. `top_p_filter` unsorted with the wrong permutation

Used `scatter(-1, sorted_idx.argsort(), ...)` — the *inverse* permutation,
correct for a `gather`, wrong for a `scatter`. Surviving logits were written to
the **wrong vocabulary positions**, so the nucleus held arbitrary tokens.

Found by **E9**, not by a test: top-p=0.9 produced 0.000 grammaticality and a
log-probability of −674.86 against −25 for plain sampling.

**Why five existing top-p tests missed it:** every one used
**already-descending** logits, where `sorted_idx` is the identity and
`argsort(identity)` is also the identity — so the bug was invisible. *A test can
be hand-computed, correct, and still structurally blind.*

Fixed to `scatter(-1, sorted_idx, ...)`; three regression tests added using
deliberately unsorted input.

### 6b. E8 reported the wrong dtype

The correctness line hard-coded "float32 round-off" while the script uses bf16
on CUDA — the GPU run printed `9.766e-03` labelled float32. Now prints the real
dtype and the bf16 epsilon.

### 6c. Test arithmetic error in the KV-cache byte test

Asserted `gqa_bytes == 4 * mqa_bytes` for `n_kv` 2 vs 1, which is a factor of 2.
The test was wrong, not the code.

### 6d. Tiny-LM `max_seq_len` equalled `block_size`

Generating 96 tokens from a model whose positional capacity was the 32-token
training window raised. Separated the two.

---

## 7. Edge cases

| Case | Handling | Tested |
|---|---|---|
| `T_q ≠ T_k` (cache, cross-attn) | supported throughout | ✓ |
| mask before vs after softmax | `-inf` before; rows still sum to 1 | ✓ |
| fully masked row | pure Python raises; torch would give NaN | ✓ |
| large logits | `torch.softmax` subtracts the row max | ✓ |
| `transpose` then `view` | `.contiguous()` in `_merge_heads` | ✓ |
| odd `head_dim` for RoPE | raises | ✓ |
| RoPE table too short | rebuilt on demand | ✓ |
| `n_heads % n_kv_heads ≠ 0` | raises | ✓ |
| sequence > `max_seq_len` | raises | ✓ |
| beam search batch > 1 | raises | ✓ |
| `temperature = 0` | explicit argmax, no division | ✓ |
| top-p emptying the nucleus | crossing token kept | ✓ |
| bf16 in RMSNorm | computed in float32 | ✓ |

---

## 8. Test map

| File | Covers | Tests |
|---|---|---|
| `test_attention.py` | shapes, softmax axis, causality, scaling, tied Q/K, stability, gradients | 22 |
| `test_attention_equivalence.py` | three implementations vs each other and vs `F.sdpa` | 14 |
| `test_attention_variants.py` | MHA/MQA/GQA, `repeat_interleave`, accounting | 24 |
| `test_positional.py` | sinusoidal formula + rotation identity, RoPE relative property, ALiBi | 24 |
| `test_normalization_ffn.py` | LayerNorm vs torch, RMSNorm non-centring, SwiGLU gating, SiLU | 21 |
| `test_transformer.py` | end-to-end causality, ln(V) init, weight tying, variants | 18 |
| `test_kv_cache.py` | cached == uncached, RoPE-offset guard, memory formula | 14 |
| `test_generation.py` | hand-computed decoding, top-p regression | 29 |

**Local: 288 passed, 2 skipped. Server: 289 passed, 1 skipped.**

**Related:** [[phase2-attention-from-first-principles]] · [[phase2-kv-cache]] ·
[[phase2-qwen-reconciliation]]
