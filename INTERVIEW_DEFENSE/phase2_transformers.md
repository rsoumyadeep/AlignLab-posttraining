# Interview Defense — Phase 2: Transformers

Reasoning questions, whiteboard prompts, live-coding tasks, and named traps.
Model answers are given only for **factual** material; the explain-back
checkpoints stay empty.

---

## A. Whiteboard prompts

**W1.** Draw a decoder-only Transformer from token IDs to logits. Annotate every
tensor shape. Mark where the causal mask enters and where the residual adds sit
relative to the norms.

**W2.** Draw the attention chain for `B=2, T=6, d_model=16, H=4`. Label the one
object that is `T×T` and say why it matters.

**W3.** Derive `Var(q·k) = d_k` and explain what follows for softmax.

**W4.** Derive the sinusoidal fixed-offset rotation identity.

**W5.** Draw MHA, GQA and MQA side by side with head counts and KV-cache sizes.

**W6.** Diagram the naive attention memory pattern, marking each HBM round trip.

---

## B. Live coding

**C1.** Masked attention from scratch, narrating every shape aloud *(Drill 5)*.
**C2.** Multi-head attention in NumPy, no torch *(Drill 4)*.
**C3.** `top_p_filter` from scratch. *Watch the unsort.*
**C4.** KV-cache incremental decoding loop, pseudocode from memory.
**C5.** RMSNorm in five lines; then say what LayerNorm adds.

---

## C. Conceptual questions with model answers

### Q1. Why divide by √d_k?
For unit-variance `q`, `k`, the dot product sums `d_k` independent terms each of
variance 1, so `Var(q·k) = d_k` and logits have std `√d_k`. As `d_k` grows,
softmax saturates toward one-hot; its Jacobian carries a factor `a(1−a)` which
then vanishes, so attention stops learning.

**Our measurement (E1), d_k 4→1024, unscaled:** logit std `1.85→31.50` (tracking
`√d_k` = 2,4,8,16,32), max prob `0.42→0.992`, entropy `1.78→0.028` out of
ln 16 = 2.77, and softmax Jacobian mass collapsing **49×** (0.708→0.014). Scaled:
every metric flat across a 256× range.

**Trap:** "to normalise the values". No — it normalises the *variance of the
logits*, and the failure mode is gradient vanishing, not scale.

### Q2. What breaks if `W_Q = W_K`?
`scores = (XW)(XW)ᵀ` is a **Gram matrix**: symmetric, so `score(i,j) =
score(j,i)`. Attention is forced to be mutually reciprocal — if "it" attends to
"the trophy", "the trophy" must attend equally back. Wrong prior for language,
where reference is directional. It also has a dominant diagonal, biasing every
token toward itself. *Verified:* `test_tied_qk_produces_symmetric_scores`.

### Q3. Why does the mask go before softmax?
Masking after leaves the row summing to **less than 1**, by an amount varying
with position — an unintended positional signal, and no longer a convex
combination. **Measured:** masking after gave row sum **0.3348** versus **1.0**
before.

### Q4. Why softmax over the key axis?
Each query needs a distribution over positions summing to 1, so the output is a
weighted average of values. Softmaxing over queries normalises each *key's*
budget across queries — a different question, and no convex combination.

### Q5. Do multiple heads add capacity?
**No.** With `d_k = d_model/H` the parameter count is identical to one head of
width `d_model`. Heads buy **specialisation** — several relationships
represented simultaneously instead of averaged into one.
**Trap:** "more heads = more capacity."

### Q6. MHA vs MQA vs GQA — and what does GQA give up?
One number: KV-head count. KV cache `= 2·B·H_kv·T·d_k·bytes·layers`, **linear in
H_kv**, so MQA cuts it by exactly `1/H`.
**Measured (E4):** params `100%→53.12%` (floor is `W_Q`+`W_O`); cache
`100%→6.25%` at H=16.
The **GQA paper's own framing** is that "MQA can lead to quality degradation",
and GQA achieves "quality close to multi-head attention with comparable speed to
MQA", uptrained with 5% of pretraining compute.
**Honest boundary:** our experiment measured **cost only** — it proves nothing
about quality. The quality claim is the paper's, not ours.

### Q7. Why does RoPE give relative position?
Rotations satisfy `R(a)ᵀR(b) = R(b−a)`, so `⟨R(m)q, R(n)k⟩ = qᵀR(n−m)k`.
Attention consumes exactly that inner product, so absolute position **cancels**.
**Measured (E5):** at fixed gap 3, inner products at m = 0,5,20,57,100 agree to a
spread of **2.4e-07** — the float32 floor.
RoPE also preserves norms (rotations are orthogonal), so it does not
de-calibrate the `1/√d_k` scaling.

### Q8. RoPE vs ALiBi?
RoPE rotates the *vectors*; ALiBi biases the *scores* with `−m_h·(i−j)`. ALiBi is
cruder — monotone decay, no phase structure — and extrapolates better *because*
of it: a linear penalty still means something at unseen lengths. The ALiBi paper
reports a 1.3B model trained at 1024 extrapolating to 2048, matching a
sinusoidal model trained at 2048, while training 11% faster with 11% less memory.

### Q9. RMSNorm vs LayerNorm?
RMSNorm drops the mean subtraction and the bias. The paper's hypothesis,
verbatim: *"re-centering invariance in LayerNorm is dispensable."*
**Measured:** with input offset +10, LayerNorm output mean `−0.000000`, RMSNorm
`+0.995010` — the surviving offset *is* the difference. On already-centred input
they coincide to 1e-3.
**Trap, and I fell into it:** the paper claims a 7–64% speedup, so it is tempting
to say RMSNorm is faster. Our measurement against torch's **fused** LayerNorm
found RMSNorm at **0.28×** — the LayerNorm is ~3.5× faster. Not a contradiction:
the paper's baseline was unfused. **Implementation quality dominates the
algorithmic difference.** What survives is "fewer parameters, less algorithmic
work" — never "faster".

### Q10. Why the 8/3 width in SwiGLU?
Gating needs three matrices instead of two. `3·d·(8/3 d) = 8d² = 2·d·(4d)`, so
8/3 is exactly the width making the gated form parameter-equivalent.
**But it is a convention, not a law:** Qwen2.5-1.5B uses `8960/1536 = 5.83×`,
**2.19× the 8/3 width**, putting **88.2%** of each block in the FFN.
**Honest boundary:** I do not know *why* Qwen chose that — it would need the
technical report, which I did not read.

### Q11. Does Flash Attention change the output?
**No.** The title says *exact*; FA-2 says "with no approximation". It reorganises
**memory access** — tiling so the `T×T` matrix never reaches HBM — not the maths.
Block-sparse FlashAttention *is* approximate; the base algorithm is not.
**Measured (E10, RTX A6000, bf16, T=4096):** MATH `82.397 ms / 9568.3 MiB`;
FLASH `1.478 ms / 65.0 MiB` — **147× less peak memory, 56× faster**, with
`max|diff| = 1.562e-02`, which is 2× the bf16 epsilon (7.812e-03), i.e. the
representation limit. Peak memory per token: MATH `0.369→2.336` (quadratic),
FLASH pinned at `0.0159` (linear).
**Corollary:** you can enable it on a trained model without retraining, and you
cannot blame a quality regression on it.

### Q12. Is the KV cache an approximation?
**No.** Causal masking guarantees a past token's K/V cannot change when a token
is appended, so there is nothing to recompute.
**Measured (E8, CPU):** `max|full − cached| = 8.345e-07`; greedy generation is
**bit-identical** with and without the cache. Per-token cost without a cache
grows `6.16→15.52 ms`; with a cache it stays flat at ~5.4 ms.
**Precisely:** the cache removes redundant **projections**. Each new query still
attends over all `t` keys, so attention stays O(t) per step. Overall generation
is *not* O(T).

### Q13. Why did Transformers replace RNNs?
**Parallelism** is decisive: an RNN's `h_t = f(h_{t−1}, x_t)` forces T sequential
steps in training; a Transformer computes all positions in one matmul, and the
causal mask is what makes that *valid*. Also O(1) path length between any two
tokens versus O(distance), and no fixed-size state bottleneck.
**What was given up:** RNNs have O(1) state and O(1) per-token inference;
Transformers have O(T²) attention and an O(T) cache. KV caching, GQA and Flash
Attention all exist to claw that back.

---

## D. Traps, named

| Trap | The correction |
|---|---|
| "More heads = more capacity" | Same parameter count; heads buy specialisation |
| "√d_k normalises the values" | It normalises logit *variance*; failure is gradient vanishing |
| "Flash Attention is an approximation" | It is **exact**; only block-sparse is approximate |
| "The KV cache makes generation O(T)" | Projections become O(T); attention stays O(T²) |
| "RMSNorm is faster" | Not against a fused LayerNorm — measured 0.28× |
| "GQA is a compute optimisation" | Primarily **memory**; the compute win depends on regime |
| "Loss dropped a lot — great" | **Measured: 33.2× lower loss from a leak** that broke generation |
| "d_ff is always 8/3·d_model" | Qwen uses 5.83× |
| "Beam search improves generation" | Higher likelihood, blander text; greedy scored **highest** log-prob in E9 and was the most repetitive |
| "Sinusoidal PE is relative" | It is **absolute** with a linearly-recoverable offset; RoPE is relative |

---

## E. Follow-ups that separate depth from familiarity

1. Your GQA has `n_kv_heads=1`. What is it called, and what did you lose?
2. Cached generation is fluent but slightly worse than uncached. Nothing errors.
   First hypothesis? *(RoPE offset)*
3. Why is `.contiguous()` needed before `view` in `_merge_heads`?
4. `softmax(scores, dim=1)` on a `[B,H,T,T]` tensor — what did you just do?
5. Why is `rope_theta` 1,000,000 in Qwen and 10,000 in the paper?
6. Your top-p tests all pass but nucleus sampling produces garbage. Where do you
   look? *(This actually happened — the tests used pre-sorted logits, so an
   inverse-permutation bug in the unsort was invisible.)*
7. Why does `cross_entropy` take logits rather than probabilities?
8. Would enabling Flash Attention change your eval numbers?

---

## F. "What does this prove / not prove"

| Result | Proves | Does **not** prove |
|---|---|---|
| E1 scaling | logit std tracks √d_k; saturation is real at init | that a *trained* model fails without it; that √d_k is optimal |
| E3 three implementations agree to 1e-16 | no transpose/axis/mask bug | that the *formulation* is right — three copies of one misunderstanding would also agree |
| E4 variants | cost savings exactly linear in `H_kv` | anything about **quality** — nothing was trained |
| E5 RoPE | our implementation has the relative property | that RoPE beats sinusoidal or ALiBi |
| E7 norms | RMSNorm halves params; does not centre | that it is faster (measured: the opposite) or trains as well |
| E8 KV cache | exact to round-off; per-token cost flattens | asymptotics; GPU run was overhead-bound at 1.05× |
| E9 decoding | measured behaviour differences | anything about a real LM — 334k params, synthetic grammar |
| E10 SDPA | same values, 147× less memory | the reference FlashAttention-2 kernel, or the paper's numbers |
| E2 leakage | masking is required; low loss can mean leakage | quantitative transfer to real models |
| WP9 param match | our component formulas are right | that our model *is* Qwen — QKV bias and FFN width differ |

---

## Explain Back

> **[USER CHECKPOINT — DEFERRED — USER EXPLAIN-BACK REQUIRED]**
>
> Answer Q1, Q6, Q11 and Q12 aloud before re-reading. Then attempt the ones with
> no written answer:
> - Both RoPE and ALiBi are relative and parameter-free. When would you choose
>   ALiBi, and what are you accepting?
> - You must serve a 1.5B model at 128k context on one 48 GB GPU. Walk through
>   the memory budget and say which decisions matter most.
> - A colleague reports 40% faster training after swapping LayerNorm for
>   RMSNorm. What do you ask before believing it?
> - Design an experiment that would prove causal masking is required — without
>   looking at how E2 did it.

**Related:** [[phase2-attention-from-first-principles]] ·
[[phase2-multihead-and-variants]] · [[phase2-kv-cache]] ·
[[phase2-flash-attention-concepts]] · [[phase2-learning-resources]]
