# Normalization and the Feed-Forward Block

**WP4.** The two components that sit between attention layers.

---

## 1. Why normalise

A residual stack adds contributions layer after layer, so activation magnitudes
drift upward with depth. Downstream layers then see inputs at a scale their
weights were not initialised for, gradients scale badly, and training
destabilises. Normalisation pins the scale at every layer — it is what makes
deep stacks trainable at all.

**Why not BatchNorm.** BatchNorm normalises across the **batch** axis, so each
example's output depends on the other examples beside it. For language models
that is fatal:

- sequences have different lengths, so padding pollutes the statistics;
- batch size varies between training and serving;
- at generation time the batch is often **1** — there are no batch statistics.

LayerNorm and RMSNorm normalise across the **feature** axis of each token
independently, so a token's output depends only on itself.
*Verified:* `test_layernorm_normalises_per_token_not_per_batch` — adding an
unrelated second batch element leaves the first output bit-identical.

---

## 2. LayerNorm

```
μ   = mean(x)                       over the last axis
σ²  = mean((x − μ)²)                BIASED estimator (÷N, not ÷(N−1))
y   = γ · (x − μ)/√(σ² + ε) + β
```

Two details that are easy to get subtly wrong:

- **Biased variance.** `torch.nn.LayerNorm` uses ÷N. Using the unbiased ÷(N−1)
  form produces a small systematic mismatch against every reference — big
  enough to fail a tolerance check while looking like "just numerics".
- **ε inside the square root.** `√(σ² + ε)`, not `√σ² + ε`. When σ² is tiny,
  adding ε *after* the root barely changes the denominator; the standard form
  bounds it away from zero.

*Verified:* our implementation matches `torch.nn.LayerNorm` to `1e-6`, which
pins both details at once.

Parameters: `2 · d_model` (gain γ and bias β).

---

## 3. RMSNorm

```
y = γ · x / √(mean(x²) + ε)
```

**Exactly what it drops:** the mean subtraction (re-centring), and usually the
bias. It keeps only the rescaling.

The original claim is that LayerNorm's benefit comes almost entirely from
re-**scaling**, not re-**centring** — so the mean subtraction is doing little
work and can go.

**What it buys**
- One reduction over the feature axis instead of two.
- Half the parameters: `d_model` instead of `2 · d_model`.
- A simpler backward pass.

**What it costs**
- The output is **no longer mean-zero**. Any constant offset in `x` survives.

*Verified, and this is the defining property:*

```
input offset by +10.0
  LayerNorm output mean = -0.000000   ← centred
  RMSNorm   output mean = +0.995010   ← offset SURVIVES
```

And the complement — on already-centred input the two **coincide** to `1e-3`
(`test_rmsnorm_equals_layernorm_on_already_centred_input`), which shows
precisely that re-centring is the *only* thing RMSNorm gives up.

**Precision note.** RMSNorm computes in float32 even under bf16 autocast.
Squaring bf16 activations loses precision badly, and a normalisation layer is
exactly where such error would propagate everywhere. Llama and Qwen do the same.

---

## 4. Experiment E7 — and the claim it forced me to retract

`scripts/experiments/e7_norm_comparison.py`, CPU float32, B=8, T=512.

| d_model | LN params | RMS params | ours LN | **fused LN** | ours RMS | vs ours LN | **vs fused LN** |
|---|---|---|---|---|---|---|---|
| 512 | 1,024 | 512 | 15.32 ms | **1.24 ms** | 4.49 ms | 3.42× | **0.28×** |
| 1024 | 2,048 | 1,024 | 30.35 ms | **2.14 ms** | 7.71 ms | 3.93× | **0.28×** |
| 4096 | 8,192 | 4,096 | 121.61 ms | **9.02 ms** | 31.30 ms | 3.89× | **0.29×** |

> ### A result I had to retract mid-experiment
>
> The first version of E7 compared only *our* LayerNorm against *our* RMSNorm
> and reported RMSNorm as **3.5–4.4× faster**. I nearly wrote that down as the
> finding.
>
> It is an **artefact**. Both were unfused educational implementations. Against
> `torch.nn.LayerNorm` — a fused kernel — our RMSNorm runs at **0.28×**, i.e.
> the *LayerNorm* is about 3.5× **faster**.
>
> **What survives:** RMSNorm does algorithmically less work and has exactly half
> the parameters. Both are verified.
> **What does not survive:** any claim that RMSNorm is *faster* than LayerNorm.
> At this scale **implementation quality dominates the algorithmic difference by
> an order of magnitude.**
>
> The fused column is now permanently in the script so the flattering
> comparison cannot be reported alone.
>
> **The generalisable lesson:** benchmarking your implementation against your
> *own* naive baseline measures your baseline, not the idea. Always include the
> optimised reference.

**What E7 proves.** RMSNorm halves the parameters; it does not centre;
on centred input it coincides with LayerNorm.

**What E7 does NOT prove.** That RMSNorm is faster in practice (measured: the
opposite, against a fused kernel), that it trains as well, or that the lost
centring is harmless. The field's answer to the last point is by adoption, not
by anything measured here.

---

## 5. The feed-forward block

Attention **mixes** information across positions but is, given the weights, a
linear combination of values — no position-wise nonlinearity of its own. The
FFN supplies that, processing each position **independently**.

The usual division of labour: **attention routes information, the FFN processes
it.** In most Transformers the FFN holds roughly two thirds of the parameters.

*Verified:* `test_feedforward_is_position_wise` — running the FFN on one
position alone gives the same result as running it on the full sequence and
slicing.

```
classic   FFN(x) = W₂ · act(W₁x + b₁) + b₂        d_ff = 4·d_model
```

`4×` is convention from the 2017 paper, not a derivation.

---

## 6. SwiGLU

```
SwiGLU(x) = W_down( Swish(W_gate x) ⊙ (W_up x) )
Swish(z)  = z · sigmoid(z)                          (= SiLU)
```

**Two ideas combined.**

**1. Swish instead of ReLU.** ReLU is exactly zero for negative inputs, so those
units receive *no gradient* — the dying-ReLU problem. Swish is smooth and only
asymptotically zero, so gradient keeps flowing. It is also **non-monotonic**,
dipping below zero with a minimum near `z ≈ −1.28`.
*Verified:* `test_silu_is_smooth_and_nonmonotonic` — gradient is non-zero for
all negative inputs, and the minimum is measured in (−1.5, −1.0).

**2. A gating branch.** Two projections multiplied elementwise, one gated by
Swish. This is **multiplicative** interaction, which a plain MLP cannot express
at the same width.
*Verified:* `test_swiglu_gating_is_multiplicative` — zeroing `W_gate` zeroes the
output entirely, whatever `W_up` does. A sum could not do that.

### Why the odd 8/3 width

Gating costs a third matrix. To keep parameters comparable to a classic FFN at
`d_ff = 4·d_model`, the hidden width is scaled by 2/3:

```
classic: 2 · d_model · (4·d_model)      = 8·d_model²
SwiGLU:  3 · d_model · (8/3·d_model)    = 8·d_model²
```

**So `8/3` is not mystical — it is exactly the width that makes the
three-matrix gated form cost the same as the two-matrix form.** *Verified* at
d=768: classic is exactly `8d²`, SwiGLU within 1% (integer truncation).

Real models round `d_ff` to a hardware-friendly multiple, so Qwen's
`intermediate_size` will not be exactly `8/3 · hidden_size`. The actual value is
read from the config in WP9.

**An honest limit.** The GLU-variants paper offers no principled explanation for
*why* gating helps — it is an empirical result, reported as such. Claiming a
clean theoretical justification would overstate what is known.

---

## My Understanding

> **[USER CHECKPOINT — DEFERRED — USER EXPLAIN-BACK REQUIRED]**
>
> **Checkpoint 11 — RMSNorm vs LayerNorm.**
> Write both formulas. What exactly does RMSNorm drop? What does that buy and
> what does it cost? Why is BatchNorm unusable for language models — give three
> reasons. Why is ε inside the square root?
>
> **Checkpoint 12 — SwiGLU.**
> Write the formula. Why Swish over ReLU? What does the gate branch add that a
> plain MLP cannot express? Derive the 8/3 width from the parameter-count
> requirement. What does the FFN do that attention cannot?
>
> Then, without looking:
> - Someone benchmarks their hand-written RMSNorm against their hand-written
>   LayerNorm, finds it 4× faster, and puts it in a paper. What is wrong?
> - Under bf16 autocast, why compute the norm in float32?

**Related:** [[phase2-attention-from-first-principles]] ·
[[phase2-decoder-only]] · [[phase2-code-explanation]]
