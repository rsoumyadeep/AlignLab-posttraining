# Flash Attention — Concepts

**WP7.** Deliberately **not** a reimplementation. Writing a competitive fused
CUDA kernel would teach GPU programming, not attention, and the educational
value is in understanding *why* it works.

---

## 1. The bottleneck is not FLOPs

The intuitive story — "attention is O(T²), so it is slow because of arithmetic" —
is **incomplete**, and getting this right is the whole insight.

A modern GPU can do far more arithmetic per second than it can move bytes.
Attention's `[B, H, T, T]` score matrix is the problem not because computing it
costs FLOPs, but because **materialising it in memory** costs bandwidth.

### The naive memory pattern

Standard attention writes and re-reads the T×T matrix repeatedly:

```
1. S = QKᵀ            write  [B,H,T,T]  to HBM
2. S = S / √d_k       read   [B,H,T,T], write it back
3. S = S + mask       read   [B,H,T,T], write it back
4. P = softmax(S)     read   [B,H,T,T], write it back
5. O = PV             read   [B,H,T,T]
```

Five round trips through the largest object in the computation. At T=4096, one
head's score matrix is 16.7M entries — **33 MB in bf16, per head, per layer**.
That does not fit in on-chip SRAM, so every step goes to HBM.

**The memory is quadratic too, not just the compute** — and it is the memory
traffic that dominates the wall clock.

## 2. The GPU memory hierarchy

| | Capacity | Bandwidth |
|---|---|---|
| **SRAM** (on-chip, per SM) | ~100s of KB | very high |
| **HBM** (device memory) | tens of GB | high, but ~an order of magnitude slower |

Classic memory hierarchy: small and fast versus large and slow. An algorithm
that repeatedly streams a large intermediate through HBM is **memory-bound** —
the arithmetic units idle while waiting for data.

## 3. The idea: tiling, and never materialising S

Flash Attention splits Q, K and V into **blocks small enough to fit in SRAM**,
and for each block pair computes its partial contribution to the output *inside
SRAM*, accumulating as it goes. **The full T×T matrix is never written to HBM.**

```
naive:   compute all of S in HBM  ─►  softmax over all of S  ─►  multiply by V
flash:   for each block of K,V:
             load into SRAM
             compute this block's scores
             update a RUNNING softmax and a RUNNING output accumulator
         never materialise the full S
```

The obstacle is that **softmax needs a global normaliser** — the denominator
sums over *all* keys, which you do not have while processing one block. Flash
Attention solves this with an **online softmax**: keep a running maximum and a
running sum, and rescale the accumulated output whenever a new block reveals a
larger maximum. Algebraically this is the same identity that makes the
max-subtraction trick valid (`softmax(x) = softmax(x − c)`), applied
incrementally.

> **Honesty note.** The tiling algorithm and the online-softmax recurrence are in
> the paper *body*, which was **not read** (see `LEARNING_RESOURCES`). The
> description above is reasoned from the abstract's "tiling to reduce the number
> of memory reads/writes between HBM and SRAM" plus the standard online-softmax
> identity. It is **not** a faithful account of their exact recurrence.

## 4. The claim that matters most: it is EXACT

**Flash Attention computes the same function as standard attention.** Not an
approximation, not a low-rank sketch, not a sparsity pattern.

This is the paper's own word — the title is *"Fast and Memory-Efficient **Exact**
Attention with IO-Awareness"*, and FlashAttention-2's abstract says the speedup
comes "**with no approximation**".

*(Block-sparse FlashAttention, a separate extension, **is** approximate. The base
algorithm is not.)*

**Why this is the interview-critical point.** Most efficiency work in ML trades
quality for speed. Flash Attention does not: identical outputs (to floating-point
reassociation), obtained by reorganising *memory access*. If asked "does Flash
Attention change your model's outputs?", the answer is **no** — and being able to
say *why* (it is a scheduling change, not a mathematical one) is what separates
understanding from having read a blog post.

The corollary: you can enable it on an already-trained model without retraining,
and you cannot blame a quality regression on it.

## 5. Reported numbers — from the abstracts we actually read

**FlashAttention** (arXiv 2205.14135): memory linear rather than quadratic;
15% end-to-end on BERT-large (len 512); **3× on GPT-2** (len 1K); 2.4× on
long-range arena. Enabled Path-X at 16K (61.4%) and Path-256 at 64K (63.1%) —
sequence lengths previously infeasible.

**FlashAttention-2** (arXiv 2307.08691): FlashAttention-1 reaches only
**25–40%** of theoretical maximum FLOPs/s; FA-2 improves work partitioning to
reach **50–73% on A100**, roughly **2×** faster.

That 25–40% figure is itself instructive: even the memory-optimised kernel was
leaving most of the GPU idle, and the remaining gap was *scheduling* — thread
block occupancy and warp-level communication — not algorithm.

**These are A100 numbers.** Our server has **RTX A6000** — same Ampere
generation, different bandwidth and SRAM budget. They should not be assumed to
transfer.

## 6. What we can and cannot measure here

`flash-attn` cannot be built on the department server: `nvcc` is absent
(verified in Phase 1B) and installing a CUDA toolkit needs root, which the
account does not have.

Instead, experiment **E10** compares PyTorch's own
`F.scaled_dot_product_attention` backends via `torch.nn.attention.sdpa_kernel`:
`MATH` (the naive materialising path), `EFFICIENT` (memory-efficient), and
`FLASH`. This measures the *same idea* through PyTorch's implementation.

**What E10 can show:** that the backends produce the same values within
tolerance, and that peak memory differs sharply.
**What E10 cannot show:** the reference FlashAttention-2 kernel's performance,
or the paper's numbers. It is a different implementation.

---

## My Understanding

> **[USER CHECKPOINT — DEFERRED — USER EXPLAIN-BACK REQUIRED]**
>
> **Checkpoint 10 — Explain why Flash Attention is useful.**
> Describe the naive memory pattern and say which object is quadratic. Explain
> the HBM/SRAM distinction and what "memory-bound" means. Describe tiling and
> why softmax makes it non-trivial. Then answer the question that matters:
> **does Flash Attention change the model's output?** Justify your answer.
>
> Then, without looking:
> - A colleague says "Flash Attention is an approximation that trades a little
>   accuracy for speed." Correct them precisely.
> - Why did FlashAttention-1 reach only 25–40% of peak FLOPs/s, and what did
>   FA-2 change?
> - You enable Flash Attention and quality drops. Is Flash Attention the cause?
> - Why can attention be memory-bound when its FLOP count is quadratic?

**Related:** [[phase2-attention-from-first-principles]] · [[phase2-kv-cache]] ·
[[phase2-learning-resources]]
