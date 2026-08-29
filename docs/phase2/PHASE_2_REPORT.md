# Phase 2 Report — Transformer Understanding

**Date:** 2026-08-29 · **Branch:** `phase-1-foundation`
**Machines:** local (Windows, CPU) + `csrslave` (2 × RTX A6000)

---

## 1. What Was Built

**Educational implementations** (`src/alignlab/models/`, 11 modules):

| Module | Contents |
|---|---|
| `shapes.py` | `assert_shape` — shape claims made executable |
| `attention.py` | scaled dot-product attention, `MultiHeadAttention` (MHA/MQA/GQA), `causal_mask` |
| `attention_numpy.py` | NumPy reference |
| `attention_pure.py` | **pure Python** — lists + `math` only, incl. multi-head by slicing |
| `positional.py` | learned absolute, sinusoidal, **RoPE**, **ALiBi** |
| `normalization.py` | educational `LayerNorm`, `RMSNorm` |
| `feedforward.py` | classic FFN, **SwiGLU** |
| `kv_cache.py` | `KVCache` + cost model |
| `transformer.py` | `TransformerConfig`, `CausalSelfAttention`, `DecoderBlock`, `DecoderOnlyTransformer` |
| `generation.py` | greedy, temperature, top-k, top-p, sampling pipeline, beam search |

**Experiments:** E1, E2, E4, E7, E8, E9, E10 + `tiny_lm.py` + WP9 reconciliation.
**Scripts:** `fetch_qwen_config.py` (allowlist/denylist enforced).
**Tiny educational LM:** 334,080 params, synthetic grammar, no downloads.
**Documentation:** all five folders + this report.

## 2. What Was Verified

- **Three attention implementations agree to machine epsilon** (§3)
- Attention weights form a distribution over **keys**; output is a **convex
  combination** of values
- **Zero** attention weight to future positions; changing the future leaves
  earlier outputs bit-identical
- Masked rows still sum to 1 (mask before softmax)
- Tied `W_Q=W_K` produces a **symmetric** score matrix; independent ones do not
- Head split/merge round-trips exactly
- **`GQA(n_kv_heads = n_heads)` is bit-identical to MHA**
- `repeat_interleave` groups query heads contiguously
- Sinusoidal PE matches a **scalar transcription** of the published formula; the
  fixed-offset rotation identity holds at four positions for every frequency
- **RoPE's relative-position property** (E5); RoPE preserves norms; `offset`
  shifts positions correctly
- ALiBi: zero diagonal, monotone penalty, geometric distinct slopes, 0 params
- Our LayerNorm matches `torch.nn.LayerNorm` to 1e-6
- RMSNorm does **not** centre; coincides with LayerNorm on centred input
- SwiGLU gating is **multiplicative**; SiLU passes gradient for negatives with a
  minimum measured in (−1.5, −1.0)
- SwiGLU's 8/3 width is parameter-equivalent to a 4× classic FFN
- **Full-model causality**; untrained loss ≈ ln(V); weight tying saves exactly
  `V·d_model`; gradients reach block 0
- **Cached == uncached logits**, and greedy generation is **bit-identical**
- **The RoPE-offset guard**: forcing offset to 0 *does* change results
- Decoding strategies against **hand-computed** cases
- **Qwen reconciliation**: 6 of 7 architectural predictions confirmed; our
  parameter arithmetic reproduces **1.54B / 1.31B** exactly

## 3. What Was Measured

| Measurement | Value |
|---|---|
| **Three-implementation agreement** | torch–numpy 3.331e-16 · pure–numpy 1.110e-16 · torch–pure 4.441e-16 (tol 1e-12) |
| Ours vs `F.scaled_dot_product_attention` | 3.576e-07 (fp32, tol 1e-5) |
| **E1** unscaled, d_k 4→1024 | logit std 1.85→31.50 (≈√d_k) · max prob 0.42→0.992 · entropy 1.78→0.028 · **Jacobian mass 0.708→0.014** |
| E1 scaled | all four metrics flat over 256× |
| **E2** masked vs unmasked | train loss **0.2907 vs 0.0088 — 33.2× lower when leaking** |
| **E4** CPU, H_kv 16→1 | params 100%→53.12% · **KV cache 100%→6.25%** · latency 160.4→117.4 ms |
| E4 GPU | latency **flat** 0.86→0.81 ms |
| **E5** RoPE, fixed gap | spread **2.384e-07** across positions 0–100 |
| **E7** vs fused LayerNorm | RMSNorm **0.28×** (CPU), **0.17–0.22×** (GPU) |
| E7 centring | LN mean −0.000000 · RMS mean **+0.995010** |
| **E8** CPU | correctness 8.345e-07 · no-cache/token **6.16→15.52 ms** · cached **flat ~5.4 ms** · speedup 2.96× |
| E8 GPU | 1.05× — **overhead-bound**, see §7 |
| **E9** decoding | greedy distinct-4 0.257 / grammatical 0.750 / logprob −18.71 · top-p 0.9 → 0.566 / **0.863** / −23.30 · temp 2.0 → 0.667 / **0.323** / −80.67 |
| **E10** GPU, T=4096 bf16 | MATH **82.397 ms / 9568.3 MiB** · FLASH **1.478 ms / 65.0 MiB** → **56× faster, 147× less memory**; diff 1.562e-02 = 2× bf16 eps |
| E10 MiB/T | MATH 0.369→2.336 (quadratic) · FLASH **0.0159 constant** (linear) |
| **Tiny LM** | 334,080 params · 2.87→0.27 loss vs ln(17)=2.8332 |
| **Qwen2.5-1.5B** | 12 Q / 2 KV heads · d_ff/d_model **5.83×** · rope_theta **1e6** · computed **1,543,656,960** params |

## 4. Experiment Results — interpretation

**E1** turns the √d_k variance argument into a curve, with `logit_std` tracking
`√d_k` almost exactly. **E2** is the strongest single result: removing the causal
mask makes loss **33× better** and generation useless — the clearest possible
demonstration that a loss drop can be leakage. **E5** confirms RoPE's algebra to
the float32 floor. **E8** shows the per-token cost flattening. **E10** shows
"same maths, different memory" with a 147× memory gap and a linear-vs-quadratic
`MiB/T` signature.

## 5. Unexpected Findings

1. **The GPU contradicted the CPU on E4.** On CPU, MQA was 27% faster and my
   "latency broadly similar" prediction was wrong. On GPU latency *is* flat — so
   the prediction holds there and fails here. **The regime depends on hardware as
   well as sequence length**, a stronger statement than either run alone.
2. **A fused LayerNorm beats an unfused RMSNorm by ~3.5×.** Implementation
   quality dominates the algorithmic difference by an order of magnitude.
3. **Qwen's FFN is 88.2% of each block**, not the ~2/3 the textbook implies — and
   `d_ff/d_model = 5.83`, not 8/3.
4. **`rope_theta = 1,000,000`**, 100× the paper's value.
5. **Qwen uses attention QKV bias**; we do not. Found only in the model card.
6. **Greedy had the *highest* log-probability and the *worst* diversity** in E9 —
   likelihood is not quality.
7. **The RMSNorm paper's 7–64% speedup claim is not contradicted by our result**
   once baselines are compared: theirs was unfused, ours fused.
8. `sliding_window` is configured but **disabled** in Qwen2.5-1.5B.

## 6. Bugs / Corrections

**6a. `top_p_filter` unsorted with the wrong permutation** — used
`sorted_idx.argsort()` (inverse permutation; right for `gather`, wrong for
`scatter`), so surviving logits landed at **wrong vocabulary positions**.
*Found by E9*, not by a test: top-p gave 0.000 grammaticality and logprob −674.86.
**Why five tests missed it:** all used **already-descending** logits, where the
permutation is the identity. Fixed; three regression tests added with unsorted
input.

**6b. E8 reported the wrong dtype** — hard-coded "float32 round-off" while using
bf16 on CUDA. Now prints the real dtype and bf16 epsilon.

**6c. E1's `grad_norm` was a confounded metric** — predicted to shrink, actually
rose then fell. Replaced by `jacobian_mass`; the bad metric is **kept** in the
script.

**6d. E7's original comparison was an artefact** — measured only against our own
unfused LayerNorm. Fused baseline now permanent.

**6e. Test arithmetic error** in the KV-cache byte test (factor 4 vs 2).

**6f. Tiny-LM `max_seq_len` equalled `block_size`**, so generation raised.

## 7. Resources Actually Inspected

**Nine resources, on 2026-08-29.** For the eight arXiv papers, **abstract pages
only — the full PDFs were NOT read**, so every recorded claim is abstract-level.

Attention Is All You Need · RoFormer/RoPE · ALiBi · GQA · RMSNorm · GLU Variants
· FlashAttention · FlashAttention-2 · Qwen2.5-1.5B model card *(fully read, plus
config downloaded)*.

Three findings changed our docs: FlashAttention's own title says **exact**; GLU
Variants is **purely empirical**, confirming our honesty note; RMSNorm's
7–64% claim reconciles with our E7 retraction once baselines are compared.

**Explicitly NOT sourced to any paper:** the √d_k derivation, the sinusoidal
rotation identity, the 8/3 width — our own reasoning, verified by experiment.

## 8. Qwen Reconciliation

Metadata only — **8,050 bytes**, revision
`8faed761d45a263340a0528343f099c05c9a4323`, **no weights**.

**Confirmed (6/7):** GQA 12/2, RMSNorm, SwiGLU/SiLU, RoPE, weight tying,
head_dim 128, pre-norm.
**Wrong (3):** FFN width 5.83× not 8/3; `rope_theta` 1e6 not 1e4; Qwen uses QKV
bias, we do not.
**Validation:** our component formulas give **1,543,656,960 / 1,310,281,728** —
matching the card's 1.54B / 1.31B independently.
**Unresolved:** config says 131072 max positions, card says 32768 context.
Hypothesis recorded, **NOT VERIFIED**.

## 9. Test Results

| | Local | Server |
|---|---|---|
| Passed | **288** | **289** |
| Failed / Errors | 0 / 0 | 0 / 0 |
| Skipped | 2 | 1 |
| Time | ~25 s | ~13 s |

Phase 2 added **166 tests** (122 → 288).

## 10. Git / Synchronization State

```
Branch : phase-1-foundation   (local + server)
HEAD   : a62d8d4881059fbfde9112fffe44b712a0a54cda  (before this report)
Remote : git@github.com:rsoumyadeep/AlignLab-posttraining.git  [PRIVATE]
Phase 2 commits: 139328e WP1 · 8d23c15 WP2 · 2a7005b WP3 · 95f826e WP4 ·
                 177b08b WP5+WP6 · 58bac80 tiny LM/E2/E9 · 0a8b757 WP7-WP9 ·
                 febed6e E8 fix · a62d8d4 GPU evidence (committed FROM server)
Merged   : none; main remains at d21c070
Working tree: clean on both machines
```
No history rewritten, no force-push. Phase 1 commits untouched.

## 11. Explain-Back Checkpoints Still Deferred

**ALL TWELVE remain `DEFERRED — USER EXPLAIN-BACK REQUIRED`.** None is complete;
documentation existing does not complete them.

| # | Checkpoint | Location |
|---|---|---|
| 1 | Derive and explain self-attention | `01_attention` |
| 2 | Every tensor shape in MHA | `02_multihead` |
| 3 | Masked attention, narrating dimensions *(Drill 5)* | `01_attention` |
| 4 | Derive sinusoidal PE | `03_positional` |
| 5 | RoPE mathematically and intuitively | `03_positional` |
| 6 | Compare MHA/MQA/GQA | `02_multihead` |
| 7 | Draw a decoder-only Transformer *(Drill 3)* | `05_decoder_only` |
| 8 | Autoregressive generation token-by-token | `05_decoder_only` |
| 9 | Explain KV caching | `06_kv_cache` |
| 10 | Why Flash Attention is useful | `07_flash` |
| 11 | RMSNorm vs LayerNorm | `04_norm_ffn` |
| 12 | SwiGLU | `04_norm_ffn` |

Plus **Drill 4** (attention in three languages, unaided), nine **Own Words**
entries in `LEARNING_RESOURCES`, and the `INTERVIEW_DEFENSE` explain-back.

## 12. Limitations

1. No model was **trained** beyond a 334k-parameter toy on synthetic grammar.
2. **No quality comparison** between MHA/MQA/GQA, or between positional schemes.
3. E8's GPU run is **overhead-bound** and does not reproduce the CPU result.
4. E10 measures **PyTorch's** SDPA backends, **not** the reference
   FlashAttention-2 kernel (`flash-attn` unbuildable — no `nvcc`, no root).
5. Papers read at **abstract level only**.
6. Pure-Python attention is unbatched and single-head-per-call by design.
7. No multi-GPU work; DDP untouched.
8. RoPE long-context extrapolation **not tested**.
9. ALiBi slope schedule **not verified against the paper**.
10. Tokenization studied only as character-level in the toy LM — **no real
    tokenizer** was exercised.
11. Our model **differs from Qwen** (QKV bias, FFN width, `rope_theta` default).

## 13. What Phase 2 Does **Not** Establish

- That any of these components **improves quality** — no controlled training.
- That the educational model is competitive, or trainable at scale.
- Any claim about Qwen's *behaviour* — only its **configuration** was read.
- That our implementations are **efficient**; they are written to be read.
- Anything about SFT, LoRA, QLoRA or DPO.
- That the three implementations are **correct in formulation** — only that they
  agree with each other and satisfy the stated properties.

## 14. Phase 3 Prerequisites

**Blocking**
1. Download Qwen2.5-1.5B **weights** (~3.1 GB bf16) at the pinned revision —
   still **not downloaded**.
2. Install the `train` extra (transformers, datasets, accelerate, peft, trl) —
   **not installed**.
3. Choose and download an instruction dataset — **none downloaded**.
4. Read `rope_theta`, `intermediate_size` and the QKV-bias setting **from the
   config**, never from our defaults.

**Carried forward from Phase 1**
5. `/data` at 99% (90 GB free after cleanup); `keep_last_checkpoints` defaults
   to 1.
6. W&B online **NOT TESTED**; offline verified.
7. Dataloader worker RNG **not captured** — matters at `num_workers: 4`.
8. No pre-flight disk check in code.

**New from Phase 2**
9. **Loss masking must be explicitly verified before PEFT** (instructions §3) —
   the E2 methodology (compare arms, inspect generation, distrust a loss drop)
   is directly reusable.
10. A dataset **fingerprint** for the run manifest.
11. Decide whether to align our attention with Qwen's QKV bias, or keep the
    educational model deliberately distinct.

---

**PHASE 2 STATUS: IMPLEMENTATION, EXPERIMENTS, VERIFICATION AND DOCUMENTATION
COMPLETE — USER EXPLAIN-BACK CHECKPOINTS DEFERRED.**

Per §30 and the explain-back protocol, Phase 2 is **not** fully complete until
the twelve checkpoints are answered by the USER. Everything achievable without
the USER is done and verified.
