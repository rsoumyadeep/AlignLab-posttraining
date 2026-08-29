# Attention From First Principles

**WP1.** Theory → intuition → mathematics → derivation → implementation → verification → experiment.
Implementation details live in `CODE_EXPLANATION/phase2/`; this file is the *why*.

---

## 1. The problem attention solves

To predict the next token, a model must combine information from earlier tokens.
Two older answers, and why each fails:

**Fixed window (n-gram, CNN).** Look at the last *n* tokens. Fails because the
relevant token may be 400 positions back — "The **trophy** doesn't fit in the
suitcase because **it** is too large."

**Recurrence (RNN/LSTM).** Compress all history into a fixed-size state. Two
problems: the state is a bottleneck (everything must fit in ~1000 numbers), and
positions must be processed **sequentially**, so training cannot parallelise
across the sequence.

**Attention's answer:** let every position look directly at every other
position, and *learn* which ones matter. No fixed window, no compression
bottleneck, and — crucially — all positions computed **in parallel**, because
the operation is one big matrix multiply.

That last point is why Transformers displaced RNNs, and it is an argument about
*hardware utilisation* as much as modelling.

---

## 2. Intuition: a soft dictionary lookup

A Python dict is a hard lookup: `d[key]` returns one value, for an exact match.

Attention is the soft version:

| Hard lookup | Attention |
|---|---|
| one query | a query **vector** |
| exact key match | **similarity** to every key |
| returns one value | returns a **weighted average** of all values |
| match is 0 or 1 | match is a probability from softmax |

So each token asks a question (**query**), every token advertises what it has
(**key**), the match scores become a probability distribution, and the answer is
a blend of what the matched tokens offer (**values**).

The soft version is what makes it differentiable — and therefore learnable.

---

## 3. Why Q, K and V are three *separate* projections

A token plays three distinct roles at once:

- as a **query** — "what am I looking for?"
- as a **key** — "what do I offer to others?"
- as a **value** — "what do I actually contribute if selected?"

```
Q = X W_Q        [B, T, d_model] @ [d_model, d_k]
K = X W_K
V = X W_V
```

**What breaks if W_Q = W_K.** Then

```
scores = (X W)(X W)ᵀ
```

which is a **Gram matrix** — necessarily **symmetric**: `score(i,j) = score(j,i)`.
Attention would be forced to be mutually reciprocal: if "it" attends strongly to
"the trophy", then "the trophy" must attend equally strongly back. That is the
wrong prior for language, where reference is directional. A Gram matrix also has
a dominant diagonal (a vector's largest inner product is usually with itself),
biasing every token towards attending to itself.

*Verified:* `test_tied_qk_produces_symmetric_scores` — tied projections give a
symmetric score matrix; independent ones do not.

**Why V is separate from K.** The features that make a token *findable* need not
be the features worth *retrieving*. A key might encode "I am a plural noun" so
a verb can find it; the value carries the semantic content. Tying them forces
one vector to be both the index and the payload.

---

## 4. The equation, derived

```
Attention(Q, K, V) = softmax( Q Kᵀ / √d_k ) V
```

Built up one piece at a time:

**Step 1 — similarity.** How well does query *i* match key *j*? Use the inner
product, contracting over the **feature** axis:

```
score(i,j) = q_i · k_j = Σ_{f=1..d_k} q_i[f] · k_j[f]
```

All pairs at once: `S = Q Kᵀ`, shape `[T_q, T_k]`.

**Step 2 — scale.** Divide by `√d_k`. Derived in §5.

**Step 3 — mask.** Set disallowed entries to `-∞`. Derived in §6.

**Step 4 — normalise.** Softmax over the **key** axis, so each query gets a
probability distribution over positions:

```
a(i,j) = exp(S(i,j)) / Σ_{j'} exp(S(i,j'))     with  Σ_j a(i,j) = 1
```

**Step 5 — retrieve.** Weighted average of the values:

```
out_i = Σ_j a(i,j) · v_j
```

Because the weights are non-negative and sum to 1, `out_i` is a **convex
combination** of the value vectors — it lies inside their convex hull. That is a
strong, testable property (`test_output_is_a_convex_combination_of_values`), and
it fails immediately if the softmax axis is wrong.

---

## 5. Deriving the √d_k factor

**Setup.** Suppose the components of `q` and `k` are independent with mean 0 and
variance 1. Then

```
q · k = Σ_{f=1..d_k} q_f k_f
```

Each term has mean `E[q_f k_f] = E[q_f]E[k_f] = 0` and variance
`Var(q_f k_f) = E[q_f²]E[k_f²] = 1`. The terms are independent, so variances add:

```
Var(q·k) = d_k        ⇒  std(q·k) = √d_k
```

**Consequence.** Logits with standard deviation `√d_k` spread wider as `d_k`
grows. Softmax of widely-spread logits saturates towards one-hot. And where
softmax is one-hot, its Jacobian

```
∂a_i/∂s_j = a_i(δ_ij − a_j)
```

vanishes — every term carries a factor `a(1−a)`, which is ~0 when `a≈1` or
`a≈0`. Saturated attention therefore **stops learning**.

**Fix.** Divide by `√d_k` to restore unit variance, independent of `d_k`.

### Experiment E1 — measured, not asserted

`scripts/experiments/e1_scaling.py`, T=16, float64, seed 20260829.
Uniform-distribution entropy for T=16 is ln 16 = 2.7726 nats.

**UNSCALED**

| d_k | logit_std | √d_k | max_prob | entropy | jac_mass |
|---|---|---|---|---|---|
| 4 | 1.85 | 2 | 0.422 | 1.777 | 0.708 |
| 16 | 3.81 | 4 | 0.649 | 1.051 | 0.488 |
| 64 | 7.41 | 8 | 0.800 | 0.523 | 0.273 |
| 256 | 16.23 | 16 | 0.948 | 0.131 | 0.077 |
| 1024 | 31.50 | 32 | **0.992** | **0.028** | **0.014** |

**SCALED by 1/√d_k**

| d_k | logit_std | max_prob | entropy | jac_mass |
|---|---|---|---|---|
| 4 | 0.927 | 0.231 | 2.400 | 0.872 |
| 16 | 0.953 | 0.219 | 2.417 | 0.881 |
| 64 | 0.927 | 0.220 | 2.415 | 0.879 |
| 256 | 1.015 | 0.268 | 2.311 | 0.853 |
| 1024 | 0.984 | 0.243 | 2.373 | 0.874 |

**What this proves.** `logit_std` tracks `√d_k` almost exactly (1.85≈2, 3.81≈4,
7.41≈8, 16.2≈16, 31.5≈32), confirming the variance derivation empirically. Without
scaling, attention becomes effectively one-hot by `d_k=1024` (max_prob 0.992,
entropy 0.028 out of 2.77) and the softmax Jacobian mass collapses **49×**
(0.708→0.014). With scaling, all four metrics stay flat across a 256× range of
`d_k`.

**What this does NOT prove.** It does not show that a *trained* model fails
without scaling — no model was trained here. It measures the behaviour at
initialisation with i.i.d. Gaussian inputs; real Q/K come from learned
projections of correlated embeddings, which could in principle adapt their
scale. It also says nothing about whether `√d_k` is *optimal* — only that
unscaled is bad and `1/√d_k` restores unit variance.

> **A prediction I got wrong, kept on the record.** The original hypothesis said
> `grad_norm` would shrink as `d_k` grew. It does not — measured, it goes
> 6.06 → 13.0 → 85.1 → 201 → 109, rising then falling. The metric is
> **confounded**: with `scale=1.0` the backward pass also carries the un-shrunk
> logit magnitudes, so a growing gradient scale masks the loss of softmax
> responsiveness. `jac_mass` (= Σ p(1−p), the softmax Jacobian's diagonal mass)
> was added afterwards as the unconfounded measure, and it behaves exactly as
> predicted. The wrong metric is kept in the script rather than deleted —
> choosing a confounded metric is a more common research failure than a wrong
> hypothesis.

---

## 6. Why the mask, and why *before* the softmax

**Why mask at all.** Training computes the loss at *every* position in parallel:
position *t* predicts token *t+1*. Without a mask, position *t* can read
position *t+1* — the answer it is being asked for. The task collapses into
copying, training loss plummets, and the model learns nothing that transfers to
generation, where future tokens genuinely do not exist.

This is also a trap worth internalising: **a sudden dramatic loss drop is
evidence of leakage, not of success** (PROJECT_INSTRUCTIONS §18 — stop and
investigate). Measured in E2.

**Why before, not after.** Suppose we softmaxed first and zeroed the future
weights afterwards. The surviving weights would then sum to less than 1 — and by
a *different* amount at each position (position 0 would lose the most). The
output would be shrunk toward zero by a position-dependent factor: a positional
signal nobody intended, and not a convex combination any more.

Adding `−∞` **before** softmax makes `exp(−∞) = 0`, so masked entries are exactly
zero *after* normalisation while the surviving weights still sum to 1.

*Verified:* `test_masked_rows_still_sum_to_one`, and the stronger
`test_output_at_position_t_ignores_later_tokens` — rewrite the future half of the
input and the earlier outputs must not move at all.

---

## 7. Why softmax over the *key* axis

Each query needs a distribution over the positions it may attend to, summing to
1, so the output is a weighted average.

Softmaxing over the **query** axis instead would make each *key's* attention sum
to 1 across queries — "how is this key's attention budget divided among
queries?" — which answers a question nobody asked, and no longer produces a
convex combination of values.

*Verified:* `test_weights_form_a_distribution_over_keys` (rows sum to 1) plus
the convex-hull test. If the axis were wrong, columns would sum to 1 and both
fail.

---

## 8. Why multiple heads

One softmax produces **one** distribution per query. A single head can therefore
average one kind of relationship at a time — but language needs several at once:
subject–verb agreement, coreference, local n-gram context, topic.

Multi-head attention projects into H lower-dimensional subspaces
(`d_k = d_model / H`), attends independently in each, concatenates, and mixes
with `W_O`.

**The point most explanations miss:** with `d_k = d_model/H`, the parameter count
is *identical* to one head of width `d_model`. Multiple heads do not buy
capacity — they buy **specialisation**: the ability to represent several
distinct relationships simultaneously rather than being forced to average them
into one.

`W_O` is not decorative: without it the concatenated head outputs would never
mix, and each output dimension would depend on only one head.

---

## 9. Tensor shapes — the full chain

Narrate these aloud (Drill 5). `B=2, T=6, d_model=16, H=4 ⇒ d_k=4`:

```
X          [B, T, d_model]     [2, 6, 16]   input
                │
   ┌────────────┼────────────┐
   ▼            ▼            ▼
Q=XW_Q      K=XW_K       V=XW_V           [2, 6, 16]  after projection
   │            │            │
   └── view(B,T,H,d_k).transpose(1,2) ──┐  the "head split" is just a
                                        │  reinterpretation of the feature axis
Q          [B, H, T, d_k]    [2, 4, 6, 4]
K          [B, H, T, d_k]    [2, 4, 6, 4]
V          [B, H, T, d_v]    [2, 4, 6, 4]
   │
   ▼  Q @ Kᵀ        [B,H,T,d_k] @ [B,H,d_k,T]
scores     [B, H, T_q, T_k]  [2, 4, 6, 6]   ← the only T×T object
   │  ÷ √d_k, mask, softmax(dim=-1)
weights    [B, H, T_q, T_k]  [2, 4, 6, 6]   rows sum to 1
   │  @ V           [B,H,T,T] @ [B,H,T,d_v]
context    [B, H, T, d_v]    [2, 4, 6, 4]
   │  transpose(1,2).contiguous().view(B,T,H*d_v)
merged     [B, T, d_model]   [2, 6, 16]
   │  @ W_O
output     [B, T, d_model]   [2, 6, 16]     ← same shape as X (residual works)
```

Two things to be able to say instantly:
- **`scores` is `[B,H,T,T]`** — quadratic in sequence length. This single fact
  explains the memory wall, KV caching, and Flash Attention.
- **output shape == input shape** — which is what lets the block sit in a
  residual stream.

`.contiguous()` before `view` is mandatory: `transpose` returns a non-contiguous
view and `view` refuses to reinterpret it. This is the most common bug in
hand-written attention.

---

## 10. Three implementations, cross-verified (E3)

Same algorithm in PyTorch, NumPy, and pure Python (lists + `math` only).

Measured, float64, tolerance `atol=1e-12` declared *before* implementation:

```
max|torch − numpy| = 3.331e-16
max|pure  − numpy| = 1.110e-16
max|torch − pure | = 4.441e-16
max|ours − F.scaled_dot_product_attention| = 3.576e-07   (float32, tol 1e-5)
```

All at machine epsilon (~2.2e-16). **No tolerance was loosened.**

**What this proves.** The three agree to the limit of float64 precision, so a
wrong transpose, a softmax over the wrong axis, or a mask applied after
normalisation is ruled out — such a bug would show up far above 1e-16.

**What it does NOT prove.** That the *algorithm* is the right one — three
implementations of the same misunderstanding would still agree. Correctness of
the formulation rests on the derivations above and the property tests, not on
mutual agreement.

---

## 11. Attention alone has no sense of order (E6)

Permuting the input tokens permutes the output identically:

```
MHA(X)[perm] == MHA(X[perm])       (verified to rtol 1e-5)
```

Attention is **permutation-equivariant**. It cannot distinguish "dog bites man"
from "man bites dog". Two fixes exist, and they answer different questions:

- **causal masking** breaks the symmetry between past and future (§6) — but
  gives no notion of *distance*;
- **positional encoding** injects order information — WP3.

Meeting this fact *before* studying positional encoding is deliberate: PE then
arrives as the answer to a problem you have already seen, not a formula to
memorise.

---

## My Understanding

> **[USER CHECKPOINT — DEFERRED — USER EXPLAIN-BACK REQUIRED]**
>
> **Checkpoint 1 — Derive and explain self-attention.**
> From `X: [B, T, d_model]`, derive `softmax(QKᵀ/√d_k)V` in your own words.
> Then: why are `W_Q`, `W_K`, `W_V` separate? What *specifically* goes wrong
> if `W_Q = W_K`? Why is softmax over the key axis? Why does the mask go
> before softmax rather than after?

## Explain Back

> **[USER CHECKPOINT — DEFERRED — USER EXPLAIN-BACK REQUIRED]**
>
> **Checkpoint 3 — Implement masked attention while narrating tensor
> dimensions** (Drill 5). Write it from scratch and say every shape aloud as
> you go: X, Q, K, V, QKᵀ, scores, weights, context, merged, output.
>
> Then answer without looking:
> - A colleague removes the causal mask and reports training loss dropped from
>   4.1 to 0.3. What do you say?
> - You have `d_k = 4096` and no scaling. What does the attention distribution
>   look like, and what happens to gradients?
> - Why is `.contiguous()` needed before `view` in `_merge_heads`?

**Related:** [[phase2-code-explanation]] · [[phase2-multihead-and-variants]] ·
[[interview-phase2-attention]]
