# Learning Resources — Phase 2

**Status change.** Phases 1A–1C accessed **no** external resources and said so.
Phase 2 is the first phase in which resources were **actually inspected**.

**Scope of what was inspected, stated precisely:** for the arXiv papers, the
**abstract pages** (`arxiv.org/abs/...`) were fetched and read — title, authors,
and abstract. **The full PDFs were NOT read.** Every claim recorded below is
therefore an *abstract-level* claim. Where a paper's detailed method or
experimental table would be needed to settle a question, that is marked
explicitly as **NOT VERIFIED**.

All fetched on **2026-08-29**.

---

## 1. Attention Is All You Need

```
Resource:        Attention Is All You Need
Type:            Paper (abstract page)
URL:             https://arxiv.org/abs/1706.03762
Authors:         Vaswani, Shazeer, Parmar, Uszkoreit, Jones, Gomez, Kaiser, Polosukhin
Topic:           The Transformer; attention replacing recurrence
Access Status:   PARTIALLY INSPECTED - abstract page only, full PDF NOT read
```

**Claims actually obtained**
- Architecture "based solely on attention mechanisms, dispensing with
  recurrence and convolutions entirely".
- **28.4 BLEU** on WMT 2014 EN→DE, "over 2 BLEU" above prior best including
  ensembles.
- **41.8 BLEU** on WMT 2014 EN→FR, single-model state of the art, after
  **3.5 days on eight GPUs**.
- Claims models are "more parallelizable and requiring significantly less time
  to train".

**What this supports in our documentation.** The parallelism argument in
`08_architecture_families` — the abstract states parallelisability explicitly as
a motivation, which is exactly the "why Transformers over RNNs" claim.

**Limitations / NOT VERIFIED**
- The `√d_k` derivation is in the paper body (§3.2.1), **not** the abstract. Our
  variance derivation in `01_attention` was reasoned independently and verified
  empirically by E1 — it is **not** cited to this paper.
- The sinusoidal PE formula and its rotation property are likewise body content,
  not read. Our derivation in `03_positional_encoding` is our own.
- The 2017 model is post-norm with LayerNorm and sinusoidal PE; our integrated
  model deliberately differs (WP5). Nothing here validates that choice.

> ## Own Words
> **[USER CHECKPOINT — DEFERRED — USER MUST WRITE THIS]**

---

## 2. RoFormer / RoPE

```
Resource:        RoFormer: Enhanced Transformer with Rotary Position Embedding
Type:            Paper (abstract page)
URL:             https://arxiv.org/abs/2104.09864
Authors:         Su, Lu, Pan, Murtadha, Wen, Liu
Access Status:   PARTIALLY INSPECTED - abstract page only
```

**Claims actually obtained**
- RoPE "encodes the absolute position with a rotation matrix and meanwhile
  incorporates the explicit **relative position dependency** in self-attention
  formulation."
- Three claimed properties: **flexibility of sequence length**; **decaying
  inter-token dependency with increasing relative distance**; **compatibility
  with linear self-attention**.

**What this supports.** The abstract's "absolute position via rotation, relative
dependency in attention" is precisely the property our experiment **E5** measured
independently (inner product invariant at fixed gap, spread 2.4e-07). Paper claim
and our measurement agree.

**Limitations / NOT VERIFIED**
- The **decay** property was not measured by us and is **not verified here**.
- The rotate-half vs interleave convention is an implementation detail not in
  the abstract; our note about it comes from the implementation, not the paper.
- `rope_theta` scaling for context extension is **not** an abstract claim.

> ## Own Words
> **[USER CHECKPOINT — DEFERRED — USER MUST WRITE THIS]**

---

## 3. ALiBi

```
Resource:        Train Short, Test Long: Attention with Linear Biases Enables
                 Input Length Extrapolation
Type:            Paper (abstract page)
URL:             https://arxiv.org/abs/2108.12409
Authors:         Press, Smith, Lewis
Access Status:   PARTIALLY INSPECTED - abstract page only
```

**Claims actually obtained**
- ALiBi "does not add positional embeddings to word embeddings; instead, it
  **biases query-key attention scores with a penalty that is proportional to
  their distance**." — confirms our implementation's mechanism.
- A **1.3B** model trained at length **1024** extrapolates to **2048**, matching
  the perplexity of a sinusoidal model *trained* at 2048, while training
  **11% faster** and using **11% less memory**.
- Notes an "inductive bias towards recency".

**What this supports.** Our RoPE-vs-ALiBi comparison in `03_positional_encoding`
claims ALiBi extrapolates well. The abstract supports this with a concrete
1024→2048 result.

**Limitations / NOT VERIFIED**
- The per-head **slope schedule** (geometric, ratio 2^(−8/n)) is body content,
  **not** in the abstract. Our implementation follows the widely-used
  formulation; it is **not verified against the paper**.
- 1024→2048 is a 2× extrapolation. Our doc says ALiBi extrapolates "well" —
  supported at 2×, **not** verified at larger factors.

> ## Own Words
> **[USER CHECKPOINT — DEFERRED — USER MUST WRITE THIS]**

---

## 4. GQA

```
Resource:        GQA: Training Generalized Multi-Query Transformer Models from
                 Multi-Head Checkpoints
Type:            Paper (abstract page)
URL:             https://arxiv.org/abs/2305.13245
Authors:         Ainslie, Lee-Thorp, de Jong, Zemlyanskiy, Lebrón, Sanghai
Access Status:   PARTIALLY INSPECTED - abstract page only
```

**Claims actually obtained**
- "MQA... **can lead to quality degradation**" — this is the paper's own framing,
  and it is the load-bearing justification for GQA existing at all.
- Uptraining an existing MHA checkpoint to MQA costs **5% of original
  pre-training compute**.
- "uptrained GQA achieves **quality close to multi-head attention** with
  **comparable speed to MQA**."

**What this supports — and a gap it fills.** Our **E4** measured parameter and
KV-cache savings but explicitly recorded that it proves **nothing about
quality**. This abstract is the source for the quality claim our own experiment
could not make. The two are complementary: we measured the cost, the paper
asserts the quality.

**Limitations / NOT VERIFIED**
- "Quality close to" and "comparable speed" are **qualitative** in the abstract;
  no numbers were obtained. We cannot quantify the MQA quality penalty.
- The choice of *how many* KV groups is body content, not read.

> ## Own Words
> **[USER CHECKPOINT — DEFERRED — USER MUST WRITE THIS]**

---

## 5. RMSNorm

```
Resource:        Root Mean Square Layer Normalization
Type:            Paper (abstract page)
URL:             https://arxiv.org/abs/1910.07467
Authors:         Biao Zhang, Rico Sennrich
Access Status:   PARTIALLY INSPECTED - abstract page only
```

**Claims actually obtained**
- The central hypothesis, verbatim: "we hypothesize that **re-centering
  invariance in LayerNorm is dispensable**".
- RMSNorm gives "re-scaling invariance property and implicit learning rate
  adaptation ability".
- "achieves comparable performance against LayerNorm but **reduces the running
  time by 7%~64%** on different models".
- Also proposes **pRMSNorm**, estimating RMS from p% of inputs — which we did
  not implement and had not encountered.

**Directly relevant to our E7 retraction.** The paper claims a **7–64% speedup**;
our measurement found our RMSNorm **slower** than `torch.nn.LayerNorm`. These are
not in conflict once the baseline is stated: the paper's 2019 comparison is
against a *non-fused* LayerNorm (largely in RNNs), whereas our comparison was
against a **fused** kernel. This is exactly the point E7's retraction makes —
implementation quality dominates the algorithmic difference. The paper's claim is
about the *algorithm*; ours is about *today's kernels*.

**Limitations / NOT VERIFIED**
- We did **not** verify the 7–64% range, nor which architectures it covers.
- The "implicit learning rate adaptation" claim is **not verified** by us at all.
- Whether dropping re-centring is harmless for *quality* is asserted by the
  paper's experiments, which we did not read and did not reproduce.

> ## Own Words
> **[USER CHECKPOINT — DEFERRED — USER MUST WRITE THIS]**

---

## 6. GLU Variants / SwiGLU

```
Resource:        GLU Variants Improve Transformer
Type:            Paper (abstract page)
URL:             https://arxiv.org/abs/2002.05202
Author:          Noam Shazeer
Access Status:   PARTIALLY INSPECTED - abstract page only
```

**Claims actually obtained**
- GLUs are "the component-wise product of two linear projections, one of which
  is first passed through a sigmoid function"; variants substitute other
  functions.
- "we test these variants in the feed-forward sublayers of the Transformer... and
  find that **some of them yield quality improvements** over the typically-used
  ReLU or GELU activations."

**Confirms a claim we already made honestly.** `04_normalization_and_ffn` states
that the paper "offers no principled explanation for why gating helps — it is an
empirical result". The abstract is purely empirical: it reports testing and
finding improvements, with **no** theoretical mechanism. Our characterisation was
correct and is now **sourced** rather than asserted.

**Limitations / NOT VERIFIED**
- The **8/3 width convention** is **not** in the abstract. Our derivation of it
  from parameter-count equivalence is our own arithmetic (verified by test), not
  a paper citation — and WP9 shows Qwen does **not** follow it anyway.
- Which specific variants won, and by how much, is body content not read.

> ## Own Words
> **[USER CHECKPOINT — DEFERRED — USER MUST WRITE THIS]**

---

## 7. FlashAttention

```
Resource:        FlashAttention: Fast and Memory-Efficient Exact Attention with
                 IO-Awareness
Type:            Paper (abstract page)
URL:             https://arxiv.org/abs/2205.14135
Authors:         Dao, Fu, Ermon, Rudra, Ré
Access Status:   PARTIALLY INSPECTED - abstract page only
```

**Claims actually obtained**
- The missing principle is **IO-awareness** — "accounting for reads and writes
  between levels of GPU memory".
- Uses **tiling** to reduce reads/writes between **HBM** and on-chip **SRAM**.
- **"IO-aware exact attention algorithm"** — the word *exact* is the paper's own.
  Block-sparse FlashAttention is the *approximate* extension.
- Speedups: **15%** end-to-end on BERT-large (len 512) vs the MLPerf 1.1 record;
  **3×** on GPT-2 (len 1K); **2.4×** on long-range arena (1K–4K).
- Enables longer context: 0.7 better perplexity on GPT-2; **61.4%** on Path-X
  (16K) and **63.1%** on Path-256 (64K).

**The single most important claim for us:** attention is **exact**. Our
`07_flash_attention_concepts` insists Flash Attention changes *memory
behaviour*, not the mathematical result — the paper's own word "exact" is the
source.

**Limitations / NOT VERIFIED**
- The tiling algorithm and online-softmax recurrence are body content, **not
  read**. Our conceptual description is reasoned, not sourced in detail.
- We did **not** reproduce any speedup number.

> ## Own Words
> **[USER CHECKPOINT — DEFERRED — USER MUST WRITE THIS]**

---

## 8. FlashAttention-2

```
Resource:        FlashAttention-2: Faster Attention with Better Parallelism and
                 Work Partitioning
Type:            Paper (abstract page)
URL:             https://arxiv.org/abs/2307.08691
Author:          Tri Dao
Access Status:   PARTIALLY INSPECTED - abstract page only
```

**Claims actually obtained**
- FlashAttention-1 gives "**linear instead of quadratic**" memory and 2–4×
  runtime speedup over optimised baselines, "**with no approximation**".
- But FlashAttention-1 reaches only **25–40%** of theoretical maximum FLOPs/s.
- Three fixes: fewer non-matmul FLOPs; parallelise across thread blocks even for
  a single head; better warp-level work distribution.
- Result: **~2×** over FlashAttention-1, reaching **50–73%** of theoretical
  maximum FLOPs/s **on A100**.

**Note on our hardware.** These numbers are for **A100**. The department server
has **RTX A6000** — same Ampere generation, different memory bandwidth and SRAM
budget. The percentages above should **not** be assumed to transfer.

**Limitations / NOT VERIFIED**
- Nothing here was reproduced. Our E10 measures PyTorch's SDPA backends, which
  is *not* the same as the reference FlashAttention-2 kernel.

> ## Own Words
> **[USER CHECKPOINT — DEFERRED — USER MUST WRITE THIS]**

---

## 9. Qwen2.5-1.5B model card

```
Resource:        Qwen/Qwen2.5-1.5B model card
Type:            Model card (HTML page) + config.json at a pinned revision
URL:             https://huggingface.co/Qwen/Qwen2.5-1.5B
Revision:        8faed761d45a263340a0528343f099c05c9a4323
Access Status:   ACTUALLY INSPECTED - model card read; config.json,
                 generation_config.json and tokenizer_config.json DOWNLOADED
                 (8,050 bytes total). NO weights downloaded.
```

**Claims actually obtained**
- **Base model, not instruction-tuned.** Verbatim: *"We do not recommend using
  base language models for conversations. Instead, you can apply post-training,
  e.g., SFT, RLHF, continued pretraining, etc., on this model."* — which is
  precisely AlignLab's scope.
- Architecture listed: **RoPE, SwiGLU, RMSNorm, Attention QKV bias**, tied word
  embeddings, **GQA**.
- **1.54B total / 1.31B non-embedding**, **28 layers**, **12 Q heads / 2 KV
  heads**, context **32,768**.

**A precise independent check.** Our parameter arithmetic from `config.json`
alone produced **1,543,656,960 total** and **1,310,281,728 in the blocks** —
matching the card's 1.54B / 1.31B. Since we derived those from first principles
(WP1–WP5 component formulas) and the card states them independently, this is a
genuine validation of our architectural understanding.

**Two discrepancies recorded rather than smoothed over**
1. `config.json` says `max_position_embeddings = 131072`; the card says context
   **32,768**. These differ by 4×. Not resolved here — likely the distinction
   between positional capacity and validated context, but that is a hypothesis,
   **NOT VERIFIED**.
2. The card lists **"Attention QKV bias"**. Our `CausalSelfAttention` uses
   `bias=False` throughout. **Our model differs from Qwen here** — see WP9.

> ## Own Words
> **[USER CHECKPOINT — DEFERRED — USER MUST WRITE THIS]**

---

## Register discipline (unchanged from Phase 1)

1. A status changes **only** when the resource is actually opened.
2. "PARTIALLY INSPECTED — abstract only" is used honestly and often here. Eight
   of nine entries are abstract-level. Claims requiring the paper body are marked
   **NOT VERIFIED**.
3. No claim in the Phase 2 documentation is attributed to a paper unless it
   appears in the text actually read. The `√d_k` derivation, the sinusoidal
   rotation identity and the 8/3 width are **our own** reasoning, verified by
   experiment, and are labelled as such.

**Still NOT INSPECTED from the Phase 1 queue:** Hydra docs, PyTorch
reproducibility notes, PyTorch DDP notes, W&B docs, SLURM docs, LoRA, QLoRA and
DPO papers (Phases 4 and 6).
