# Multi-Head Attention, and Why Modern LLMs Share KV Heads

**WP2.** Builds directly on [[phase2-attention-from-first-principles]].

---

## 1. Why more than one head

One softmax gives **one** distribution per query. A single head must therefore
average one kind of relationship at a time, but language needs several at once —
subject–verb agreement, coreference, local n-grams, topic.

MHA projects into H subspaces of width `d_k = d_model / H`, attends
independently in each, concatenates, and mixes with `W_O`.

**The point that is usually missed:** with `d_k = d_model/H` the parameter count
is *identical* to one head of width `d_model`. `W_Q` is `[d_model, H·d_k] =
[d_model, d_model]` either way. Heads do not buy **capacity** — they buy
**specialisation**: several distinct relationships represented at once instead
of being averaged into one.

`W_O` is load-bearing. Without it the concatenated head outputs never mix, and
each output dimension would depend on exactly one head.

---

## 2. The three variants are one number apart

| Variant | Q heads | KV heads | Idea |
|---|---|---|---|
| **MHA** | H | H | every query head owns its K/V |
| **GQA** | H | H/G | groups of query heads share a K/V head |
| **MQA** | H | 1 | all query heads share one K/V head |

Everything else — scores, scaling, mask, softmax, `W_O` — is identical. In code
it is one constructor argument plus one `repeat_interleave`.

### Shapes, H=8, d_model=64, d_k=8

```
                              MHA           GQA(H_kv=2)      MQA
W_Q            [d_model, H·d_k]  [64,64]      [64,64]        [64,64]
W_K        [d_model, H_kv·d_k]   [64,64]      [64,16]        [64,8]
W_V        [d_model, H_kv·d_k]   [64,64]      [64,16]        [64,8]
W_O            [H·d_k, d_model]  [64,64]      [64,64]        [64,64]

Q          [B, H,    T, d_k]  [B,8,T,8]     [B,8,T,8]      [B,8,T,8]
K (proj)   [B, H_kv, T, d_k]  [B,8,T,8]     [B,2,T,8]      [B,1,T,8]
K (after repeat_kv)           [B,8,T,8]     [B,8,T,8]      [B,8,T,8]
scores     [B, H,    T, T]    [B,8,T,T]     [B,8,T,T]      [B,8,T,T]
```

**Note the last two rows.** After `repeat_kv`, every variant runs the *same*
attention matmuls on the *same* shapes. The saving is never in the score
computation — it is in the **projections** and, above all, in the **KV cache**.

`repeat_interleave` (not `repeat`) matters: it gives `[kv0, kv0, kv1, kv1]`, so
each KV head serves a *contiguous* group of query heads. `repeat` would give
`[kv0, kv1, kv0, kv1]` and silently pair query heads with the wrong group — a
bug that trains to a mediocre model rather than crashing.
*Pinned by* `test_repeat_kv_expands_groups_correctly`.

---

## 3. Why GQA exists: the KV cache

During generation with a cache, K and V for every past position are held in
memory:

```
KV cache bytes = 2 · B · H_kv · T · d_k · bytes_per_element
                 ↑                ↑
                K and V      NOT H — this is the whole point
```

Linear in `H_kv`. MQA cuts it by exactly a factor of H.

At `T = 4096`, `d_k = 64`, bf16, one layer, batch 1:

| Variant | KV cache | vs MHA |
|---|---|---|
| MHA (H_kv=16) | 16.0 MiB | 100% |
| GQA (H_kv=8) | 8.0 MiB | 50% |
| GQA (H_kv=4) | 4.0 MiB | 25% |
| GQA (H_kv=2) | 2.0 MiB | 12.5% |
| MQA (H_kv=1) | 1.0 MiB | **6.25% = 1/16** |

Multiply by layer count and batch size and this becomes the dominant term in
inference memory. That is why GQA is near-universal in modern LLMs: it keeps
most of MHA's quality while making long-context, large-batch serving affordable.

**GQA is the compromise.** MQA's single KV head is a real quality cost; GQA
recovers most of it while keeping most of the memory win. Qwen2.5 uses GQA —
verified against the actual config in WP9, not assumed here.

---

## 4. Experiment E4 — measured

`scripts/experiments/e4_attention_variants.py`, CPU, float32, torch 2.6.0+cpu,
d_model=1024, H=16, d_k=64; latency at B=4, T=512, median of 10 iterations.

| Variant | H_kv | params | vs MHA | KV cache | vs MHA | latency |
|---|---|---|---|---|---|---|
| MHA | 16 | 4,194,304 | 100.00% | 16.0 MiB | 100.00% | 160.4 ms |
| GQA | 8 | 3,145,728 | 75.00% | 8.0 MiB | 50.00% | 134.8 ms |
| GQA | 4 | 2,621,440 | 62.50% | 4.0 MiB | 25.00% | 126.9 ms |
| GQA | 2 | 2,359,296 | 56.25% | 2.0 MiB | 12.50% | 119.1 ms |
| MQA | 1 | 2,228,224 | 53.12% | 1.0 MiB | **6.25%** | 117.4 ms |

**Parameters bottom out at 53%, not 0%** — `W_Q` and `W_O` are a floor of
`2·d_model² = 2,097,152`; MQA adds only `2·d_model·d_k = 131,072` on top.
Exactly matches the measurement.

> ### A prediction I got wrong, kept on the record
>
> I predicted **"latency broadly similar — GQA is a memory optimisation, not a
> FLOP optimisation."** Measured: MQA was **27% faster** than MHA. Not similar.
>
> Diagnosis (FLOP accounting, `B=4, H=16, d_model=1024`):
>
> | T | attention GF | projections GF (MHA) | MQA/MHA total |
> |---|---|---|---|
> | 512 | 4.29 | **17.18** | **0.625** |
> | 2048 | 68.72 | 68.72 | 0.766 |
> | 8192 | 1099.51 | 274.88 | 0.906 |
>
> At T=512 the **projections dominate attention 4:1**, and shrinking `W_K`/`W_V`
> removes ~37% of total FLOPs — which is the right order for the 27% measured.
>
> The claim was not wrong in kind, only in *regime*. Attention is O(T²) and
> projections O(T), so as context grows attention dominates and the ratio
> climbs toward 1 (0.906 by T=8192). At the long contexts where GQA actually
> matters, my original statement becomes true. At T=512 it was false.
>
> **The generalisable lesson:** "attention is the expensive part" is a claim
> about a *regime*, not a universal truth, and at modest sequence lengths the
> projections are the larger cost.

**What E4 proves.** Parameter and KV-cache savings are exactly linear in `H_kv`
and match the arithmetic; MQA cuts the cache by precisely `1/H`.

**What E4 does NOT prove.** Nothing about **quality** — no model was trained, so
this says nothing about whether GQA degrades output. That is the entire
substance of the MQA-vs-GQA tradeoff and it is *not* measured here. It also
measures CPU float32 latency, which is not a proxy for GPU bf16 serving;
the GPU run is E4-GPU on the server.

---

## 5. Choosing H_kv

| Question | Answer |
|---|---|
| Quality-first, short context, memory ample | MHA |
| Serving long contexts / large batches | GQA, typically H/4 or H/8 |
| Extreme memory pressure, quality tolerant | MQA |

The choice is made by whether **KV cache memory** or **quality** binds first —
and, per E4, at short sequence lengths there is a compute saving too.

---

## My Understanding

> **[USER CHECKPOINT — DEFERRED — USER EXPLAIN-BACK REQUIRED]**
>
> **Checkpoint 2 — Explain every tensor shape in MHA.**
> Start from `X: [B, T, d_model]` and narrate: Q, K, V before and after the
> head split, QKᵀ, scores, weights, context, merged, output. Say which axis
> softmax runs over and why. Then explain what `.contiguous()` is for.
>
> **Checkpoint 6 — Compare MHA, MQA and GQA.**
> For each: Q-head count, KV-head count, KV-cache size, parameter count,
> compute implications. Then: why do modern LLMs use GQA rather than MHA or
> MQA? What exactly does MQA give up? At what sequence length does the
> *compute* argument for GQA stop mattering, and why?

**Related:** [[phase2-attention-from-first-principles]] · [[phase2-kv-cache]] ·
[[phase2-code-explanation]] · [[interview-phase2-attention]]
