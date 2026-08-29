# WP9 — Reconciling Our Model With Qwen2.5-1.5B

**Run deliberately last.** Reading the real config *before* building would have
taught us to copy numbers. Reading it *after* makes it a **test**: we predicted
GQA, RMSNorm, SwiGLU, RoPE and weight tying from first principles — does the
actual configuration agree?

**Scope.** `config.json`, `generation_config.json`, `tokenizer_config.json` —
**8,050 bytes total**, at pinned revision
`8faed761d45a263340a0528343f099c05c9a4323`. **No weights.** The download script
enforces an allowlist and a weight-extension denylist, and asserts no weight file
was written.

---

## 1. What we predicted, and what is actually there

| Component | We built | Qwen2.5-1.5B actually uses | ✓ |
|---|---|---|---|
| Attention | GQA-capable | **GQA: 12 Q heads / 2 KV heads (6:1)** | ✓ |
| Normalization | RMSNorm, eps 1e-6 | `rms_norm_eps = 1e-06` | ✓ |
| FFN activation | SwiGLU (SiLU) | `hidden_act = silu` | ✓ |
| Positional | RoPE | RoPE | ✓ |
| Weight tying | tied by default | `tie_word_embeddings = True` | ✓ |
| head_dim | power of two | 1536/12 = **128** | ✓ |
| Norm placement | pre-norm | pre-norm | ✓ |

Six of seven architectural choices match. The educational model is the same
*species* as the model Phase 3 will fine-tune — which was the point of building
it that way (WP5).

## 2. Where we were wrong

### 2a. The SwiGLU width is **not** 8/3

```
hidden_size        1536
intermediate_size  8960
ratio              5.833 ×          ← we predicted 8/3 = 2.667 ×
8/3 rule would be  4096             ← actual is 2.19× that
```

`04_normalization_and_ffn` derived 8/3 as the width making the three-matrix
gated form parameter-equivalent to a classic 4× FFN, and said real models "round
it to a hardware-friendly multiple". **That is wrong for this model** — 8960 is
not a rounding of 4096, it is more than double it.

The consequence is visible in the parameter budget:

```
attention / layer     5,505,024
FFN       / layer    41,287,680        ← 88.2% of the block
```

**The FFN holds 88% of each block, not the ~2/3 the textbook description
suggests.** Qwen2.5-1.5B is deliberately FFN-heavy.

**What I can and cannot say about why.** I can say the 8/3 rule is a
*parameter-equivalence* convention, not a law, and that this model departs from
it substantially. I *cannot* say why Qwen chose 5.83× — that would require the
Qwen technical report, which was **not read**. Recording it as an observed fact
with an unexplained cause is the honest position.

### 2b. `rope_theta` is 1,000,000, not 10,000

Our `RotaryPositionalEmbedding` defaults to `base = 10000.0`, the original RoPE
value. Qwen2.5-1.5B uses **1,000,000 — 100× larger**.

This is exactly the knob `03_positional_encoding` flagged: raising the base
lengthens the low-frequency wavelengths, so positional phases evolve more slowly
and remain distinguishable at much longer distances. It is the mechanism behind
`max_position_embeddings = 131072`.

Our default is not *wrong* — it is the paper's value — but it is the wrong value
for a long-context model, and Phase 3 must take it from the config rather than
from our default.

### 2c. Qwen uses attention QKV **bias**; we do not

The model card lists **"Attention QKV bias"** among the architecture components.
Our `CausalSelfAttention` builds all four projections with `bias=False`.

A real difference, discovered only by reading the card. Most modern models drop
biases; Qwen2 retains them on Q, K and V. Noted rather than "fixed" — the
educational model is not required to be a Qwen clone, but Phase 3 must not assume
they match.

## 3. An independent check that our understanding is right

Computing the parameter count from the config using **our own component
formulas** (WP1–WP5):

```
embedding (tied)      233,373,696   15.1%
attention / layer       5,505,024
FFN       / layer      41,287,680   88.2% of a block
per layer              46,795,776
× 28 layers         1,310,281,728   84.9%
TOTAL               1,543,656,960   1.54 B
```

The model card independently states **1.54B total / 1.31B non-embedding**.

**These match.** We derived them from the architecture; the card reports them
from the real model. That agreement is a genuine validation that the component
formulas in WP1–WP5 are correct — a stronger check than any single unit test,
because it exercises every component's parameter accounting at once.

## 4. GQA's payoff at Qwen's real dimensions

Applying the WP2 KV-cache formula with Qwen's actual numbers (28 layers,
head_dim 128, bf16, batch 1):

| Context | GQA (2 KV heads) | MHA would be (12) | Saving |
|---|---|---|---|
| 4,096 | 0.11 GiB | 0.66 GiB | **6×** |
| 32,768 | 0.88 GiB | 5.25 GiB | **6×** |
| 131,072 | **3.50 GiB** | 21.00 GiB | **6×** |

At full context, GQA is the difference between a 3.5 GiB cache and a 21 GiB one —
on a 48 GiB A6000, with ~3.1 GiB of weights, that is the difference between
comfortable and impossible. **This is what GQA is for**, in the concrete.

## 5. Other recorded facts

- **28 layers**, `torch_dtype = bfloat16`
- `bos_token_id == eos_token_id == 151643` — a base model with no distinct BOS
- `sliding_window = 131072` but `use_sliding_window = False` — **configured but
  disabled**; do not assume sliding-window attention is active
- `vocab_size = 151936` — tying saves 233M parameters, **13.1%** of the untied
  total
- **A discrepancy, unresolved:** `config.json` says
  `max_position_embeddings = 131072`; the model card says context **32,768**.
  A 4× difference. Plausibly positional capacity versus validated context, but
  that is a **hypothesis, NOT VERIFIED**.

## 6. What this means for Phase 3

- It is a **base** model. The card: *"We do not recommend using base language
  models for conversations. Instead, you can apply post-training, e.g., SFT,
  RLHF, continued pretraining"* — exactly AlignLab's scope.
- Weights are **3.1 GB in bf16** — still not downloaded.
- Read `rope_theta`, `intermediate_size` and the QKV-bias setting **from the
  config**, never from our defaults.
- Pin this revision in the run manifest.

---

## My Understanding

> **[USER CHECKPOINT — DEFERRED — USER EXPLAIN-BACK REQUIRED]**
>
> - Qwen2.5-1.5B has 12 query heads and 2 KV heads. What is that called, what is
>   the group size, and what does it save at 32k context?
> - Its `intermediate_size` is 5.83× `hidden_size`, not the 8/3 convention. What
>   *is* the 8/3 rule actually derived from, and what does departing from it cost
>   or buy?
> - Why is `rope_theta = 1e6` rather than 1e4, and what does that have to do with
>   a 131k context window?
> - From `hidden_size=1536`, `n_heads=12`, `n_kv=2`, `intermediate=8960`,
>   `layers=28`, `vocab=151936`, tied embeddings — compute the parameter count.
>   You should get 1.54B.

**Related:** [[phase2-multihead-and-variants]] · [[phase2-normalization-and-ffn]]
· [[phase2-positional-encoding]] · [[phase2-learning-resources]]
