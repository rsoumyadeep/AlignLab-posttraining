# The Decoder-Only Transformer, and Autoregressive Generation

**WP5.** Assembles WP1–WP4 into a working language model.

---

## 1. The full path

```
token IDs        [B, T]                integers in [0, V)
    │  nn.Embedding                    a lookup, not a matmul
embeddings       [B, T, d_model]
    │
    │   ┌────────── × n_layers ──────────┐
    │   │  h = x + Attn(Norm(x))         │  ← residual, pre-norm
    │   │  y = h + FFN(Norm(h))          │  ← residual, pre-norm
    │   └────────────────────────────────┘
    │
final norm       [B, T, d_model]
    │  LM head: Linear(d_model → V)
logits           [B, T, V]
```

Draw this from memory (Checkpoint 7). Two things to be able to say immediately:

- **Every block is shape-preserving** — `[B,T,d_model]` in, `[B,T,d_model]` out.
  That is what makes the residual stream possible.
- **The only `T×T` object is the attention score matrix.** Everything else is
  linear in T. That single fact explains the memory wall, KV caching, and Flash
  Attention.

Our configuration is **modern-LLM-shaped**, not the 2017 original:

| | 2017 | AlignLab (and Llama/Qwen) |
|---|---|---|
| Norm placement | post-norm | **pre-norm** |
| Norm | LayerNorm | **RMSNorm** |
| Position | sinusoidal, added | **RoPE, in attention** |
| FFN | ReLU MLP | **SwiGLU** |
| Attention | MHA | **GQA-capable** |
| Biases | yes | **none** |

The 2017 variants all exist as separate modules for comparison. The integrated
model is faithful to what Phase 3 will actually fine-tune.

---

## 2. Why pre-norm

```
post-norm (2017):  x = Norm(x + Sublayer(x))     ← normalisation ON the highway
pre-norm  (now):   x = x + Sublayer(Norm(x))     ← highway untouched
```

Post-norm puts a normalisation **on the residual path itself**, so the identity
shortcut is rescaled at every layer and gradients degrade with depth — the
original Transformer needed learning-rate warmup to train at all.

Pre-norm normalises the sublayer *input* and leaves an unobstructed additive
path from the embedding all the way to the final norm, so gradients reach early
layers directly.

*Verified:* `test_gradients_reach_the_first_layer` — every parameter in block 0
of a 4-layer model receives a finite, non-zero gradient.

**Why a final norm is then needed.** With pre-norm, nothing normalises the last
block's output — the residual stream has been accumulating unnormalised
additions the whole way up. A final norm fixes the scale the LM head sees.

---

## 3. Weight tying

The input embedding and the output projection are the **same matrix**. Both map
between token identity and representation space, just in opposite directions.

*Verified:* `test_weight_tying_saves_exactly_vocab_times_dmodel` — saves exactly
`V × d_model` parameters. For a real vocabulary (152k for Qwen) that is a large
fraction of a small model.

---

## 4. A sanity check worth knowing

A randomly initialised LM should be roughly **uniform** over the vocabulary, so
its cross-entropy loss should start near **ln(V)**.

*Verified:* `test_untrained_loss_is_near_ln_vocab` — with V=1000, loss is within
0.5 of ln(1000) = 6.91.

This is the first thing to check when a training run misbehaves. Loss starting
far *below* ln(V) means the logits are not uniform at init — often a leak, a bad
initialisation, or a label misalignment.

---

## 5. Autoregressive generation

```
prompt = [t1, t2, t3]
step 1: forward(prompt)   → logits [B,3,V] → take logits[:, -1] → t4
step 2: forward([…,t4])   → take the last position              → t5
```

**Only the last position's logits matter.** The forward pass predicts at every
position, but positions `0…T−2` are predicting tokens we already have. In
*training* all of them contribute to the loss — which is what makes training
parallel over positions. In *generation* all but the last are discarded.

### Training and inference are fundamentally different

| | Training | Generation |
|---|---|---|
| Passes | **one**, parallel over T | **T**, sequential |
| Conditions on | the **true** prefix (teacher forcing) | the model's **own** output |
| Cost | one forward | T forwards |

That asymmetry is **exposure bias**: at training time the model never sees its
own mistakes, so a model with excellent teacher-forced loss can still generate
poorly once errors start compounding. It is also why the causal mask matters so
much — it is what makes the single parallel training pass a valid simulation of
T sequential steps.

---

## 6. Decoding strategies

All are pure functions of the logits, so they are testable against
hand-computed cases with no model at all.

**Greedy** — `argmax`. Deterministic, and tends to repeat: the
highest-probability continuation of a repeated phrase is usually more of the
same, and there is no mechanism to escape.

**Temperature** — divide logits by T before softmax. `T→0` approaches argmax,
`T=1` is the model's own distribution, `T→∞` approaches uniform. It rescales
**confidence without changing the ranking** — *verified*, the argsort is
identical for every `T > 0`.

**Top-k** — keep the k highest, mask the rest. Removes the long tail, where most
degenerate samples come from: individually each tail token is unlikely, but
collectively they hold real mass. **Limitation:** k is *fixed*. When the model is
confident, k=50 admits 49 bad options; when uncertain, it may cut good ones off.

**Top-p (nucleus)** — the adaptive answer. Keep the smallest set whose
cumulative probability ≥ p.

*Hand-computed test:* with softmax `[0.5, 0.3, 0.2]` and `p=0.7`, cumulative
runs 0.5, 0.8 — the **second** token crosses the threshold and is **kept**, so
the nucleus is `{0,1}`. Dropping the crossing token would retain less than p
and, for a peaked distribution, could empty the set entirely. *Verified:* even
`p=0.001` keeps at least one token.

*Verified adaptivity:* a peaked distribution keeps **1** token at p=0.9; a flat
one keeps many. Top-k would keep exactly k in both.

**Order matters.** Temperature is applied *first*, because it changes the
probabilities that top-k and top-p then threshold. Filtering on one distribution
and sampling from another would be a subtle, silent bug.

**Beam search** — keep the `beam_width` best partial sequences, so a locally
worse token that leads somewhere better can survive. Scores are **summed
log-probabilities**: a product of hundreds of probabilities underflows float32
almost immediately. A **length penalty** is needed because every extra token
adds a negative log-prob, biasing raw beam search toward short output.

*Verified:* beam width 1 reduces exactly to greedy; beam search scores ≥ greedy
on total log-probability.

**When beam search is wrong.** It suits translation and summarisation, where one
high-probability answer is wanted. For open-ended generation it produces bland,
repetitive text — the highest-probability continuation is usually the least
interesting one.

---

## My Understanding

> **[USER CHECKPOINT — DEFERRED — USER EXPLAIN-BACK REQUIRED]**
>
> **Checkpoint 7 — Draw a decoder-only Transformer.**
> On a whiteboard, from memory, annotating every tensor shape. Mark where the
> causal mask enters, where the residual adds are relative to the norms, and
> which single object is `T×T`.
>
> **Checkpoint 8 — Explain autoregressive generation token-by-token.**
> Walk through generating 3 tokens from a 4-token prompt. Which logits are used
> and which discarded? Why does training take one pass and generation T? What
> is exposure bias and why does it follow from that asymmetry?
>
> Then, without looking:
> - Why pre-norm rather than post-norm? What breaks with post-norm at depth?
> - Your untrained model reports loss 0.4 with a vocabulary of 50,000. What is
>   wrong?
> - Why must temperature be applied before top-k rather than after?
> - Your top-p implementation drops the crossing token instead of keeping it.
>   Give a concrete distribution where this breaks.

**Related:** [[phase2-attention-from-first-principles]] ·
[[phase2-normalization-and-ffn]] · [[phase2-kv-cache]]
