# Positional Encoding: Absolute vs Relative

**WP3.** Motivated by the permutation-equivariance result in
[[phase2-attention-from-first-principles]] §11.

---

## 1. The problem

Attention is **permutation-equivariant**: permute the input tokens and the
output permutes identically. Verified — `MHA(X)[perm] == MHA(X[perm])`.

So attention alone cannot distinguish *"dog bites man"* from *"man bites dog"*.
Position must be injected from outside. Causal masking breaks past/future
symmetry, but gives no notion of **distance**.

## 2. The organising axis

| | **Absolute** | **Relative** |
|---|---|---|
| Depends on | position index *i* | distance *i − j* |
| Says | "I am token 5" | "you are 3 behind me" |
| Applied | once, to embeddings | inside attention, every layer |
| Examples | learned, sinusoidal | RoPE, ALiBi |

Language regularities are mostly **relative** — an adjective modifies a nearby
noun regardless of where the sentence begins. And relative schemes extrapolate:
a learned embedding for position 5000 that was never trained is simply
undefined.

| Method | Type | Learned | Params | Extrapolates |
|---|---|---|---|---|
| Learned absolute | absolute | yes | `max_len × d_model` | **no — hard limit** |
| Sinusoidal | absolute | no | 0 | in principle |
| **RoPE** | relative | no | 0 | moderately |
| **ALiBi** | relative | no | 0 | well |

---

## 3. Learned absolute embeddings

A lookup table indexed by position (GPT-2's choice). Simple, effective, and
with one hard limitation: the table has `max_seq_len` rows, so position
`max_seq_len` has **no embedding at all**.

*Verified:* `test_learned_pe_hard_fails_beyond_max_len` — it raises. Not
"degrades gracefully"; undefined. This is the concrete reason absolute learned
embeddings cannot extrapolate, and it is why the field moved on.

---

## 4. Sinusoidal encoding — derived

```
PE(pos, 2i)   = sin(pos / base^(2i/d_model))
PE(pos, 2i+1) = cos(pos / base^(2i/d_model))          base = 10000
```

### The derivation

We want a *fixed* function of position with three properties.

**(1) Unique per position.** A single sine repeats every period, so it cannot
distinguish `pos` from `pos + 2π/w`. Fix: use **many** frequencies, in a
geometric ladder from `w=1` down to `w=1/10000`. This behaves like a binary
counter written in continuous values — fast dimensions flip quickly and
distinguish neighbours, slow dimensions distinguish distant positions.

*Verified:* `test_sinusoidal_positions_are_unique` — minimum pairwise distance
over 128 positions stays well above zero.

**(2) Bounded.** sin/cos ∈ [−1, 1], so the encoding cannot swamp the token
embedding it is added to. *Verified:* `test_sinusoidal_is_bounded`.

Contrast: raw `pos` as a feature would grow without bound, and `pos/max_len`
would change meaning whenever `max_len` changed.

**(3) Relative offsets linearly recoverable — the elegant part.**

For a fixed offset *k*, the pair `(sin(w·pos), cos(w·pos))` at `pos+k` is a
**rotation by w·k** of the pair at `pos`:

```
[sin(w(pos+k))]   [ cos(wk)   sin(wk) ] [sin(w·pos)]
[cos(w(pos+k))] = [ −sin(wk)  cos(wk) ] [cos(w·pos)]
```

Using the angle-addition identities:
`sin(a+b) = sin a cos b + cos a sin b`, `cos(a+b) = cos a cos b − sin a sin b`.

The matrix depends on **k but not on pos**. So one linear map implements "shift
by k" *uniformly across the whole sequence* — which is exactly the kind of
operation a learned linear projection can represent.

*Verified numerically:* `test_sinusoidal_offset_is_a_fixed_linear_rotation`
checks this rotation identity at positions 0, 3, 17, 40 for every frequency.

### The seed of RoPE

Property (3) is a rotation observation, and RoPE is what you get by taking it
seriously. The difference is **where it is applied**:

- **Sinusoidal** *adds* this structure to the embedding — the relative
  information must then survive `W_Q`/`W_K`, which is not guaranteed.
- **RoPE** *rotates Q and K directly* — so the relative structure is exact, and
  cannot be washed out by the projections.

---

## 5. RoPE — intuition, mathematics, rotation, relativity

### Intuition

Don't *add* a position vector. **Rotate** each query and key by an angle
proportional to its position. Split the head dimension into 2-D pairs; rotate
pair *p* of a vector at position *m* by angle `m·θ_p`, with
`θ_p = base^(−2p/d)` — the same geometric ladder as sinusoidal.

### The mathematics

Rotation matrices satisfy `R(a)ᵀ R(b) = R(b − a)`. Therefore

```
⟨R(m)q, R(n)k⟩ = qᵀ R(m)ᵀ R(n) k = qᵀ R(n − m) k
```

**Attention consumes exactly this inner product.** So the score between
positions *m* and *n* depends on `(m − n)` alone — **absolute position
cancels**. Relative positioning emerges from the geometry; nothing is learned
and nothing is added.

### Experiment E5 — measured

Fix a gap of 3 and slide both positions along the sequence. If the theory
holds, the inner product must not move.

```
positions (m+3, m) for m = 0, 5, 20, 57, 100
inner products: -0.5877403, -0.5877404, -0.5877402, -0.5877402, -0.5877404
spread = 2.384e-07
```

**Spread 2.4e-07 — the float32 round-off floor.** Absolute position cancels
exactly, as derived.

*Guard:* `test_rope_inner_product_changes_with_different_distance` — a
*different* gap gives a materially different product, so E5 cannot pass by the
function being trivially constant.

**What E5 proves.** Our implementation has the relative-position property, so
the pairing convention and signs are right.

**What E5 does NOT prove.** That RoPE is *better* than sinusoidal or ALiBi — no
model was trained. It also says nothing about long-context extrapolation, which
is where the interesting RoPE failure modes live.

### Two consequences

- **Norm preservation.** Rotations are orthogonal, so RoPE does not change
  `‖q‖`. If it did, the attention logit scale would shift and silently
  de-calibrate the `1/√d_k` factor from WP1. *Verified:* `test_rope_preserves_norm`.
- **`offset` matters.** During cached generation the new token sits at position
  `past_len`, not 0. Getting `offset` wrong is *the* classic KV-cache bug: the
  model still produces fluent text, with subtly wrong positional structure.
  *Verified:* `test_rope_offset_shifts_positions`.

### A convention trap worth knowing

Two layouts exist: **rotate-half** (`x1, x2 = split(x); (−x2, x1)`), used by
Llama/Qwen, and **interleave-adjacent**, used in the original paper. They are
equivalent up to a fixed permutation of the head dimension. What matters is
that **Q and K use the same one**. Mixing them does not crash — the model just
trains worse.

`base` (often `rope_theta`) is a real knob: raising it lengthens low-frequency
wavelengths and underpins several context-extension methods. Qwen2.5's actual
value is read from the config in WP9, not guessed here.

---

## 6. ALiBi — bias the scores, touch nothing else

No positional vectors at all. Add a distance-proportional penalty directly to
the attention scores:

```
score(i, j) += −m_h · (i − j)
```

`m_h` is a per-head slope from a fixed geometric sequence. Nearby tokens are
penalised less, so each head acquires a characteristic **range**: small slopes
look far, large slopes look locally.

*Verified:* zero on the diagonal (a token is distance 0 from itself); penalty
strictly increases with distance; slopes are geometric and distinct per head;
**zero learned parameters**.

### RoPE vs ALiBi

| | RoPE | ALiBi |
|---|---|---|
| Acts on | Q and K (the vectors) | the scores directly |
| Mechanism | rotation → phase structure | monotone linear penalty |
| Expressiveness | richer | cruder: decay only |
| Extrapolation | moderate | **strong** |
| Params | 0 | 0 |

ALiBi is deliberately the blunter instrument, and extrapolates *because* of it:
a linear penalty still means something at lengths never trained on, whereas
rotations at unseen positions land in phase regions the model has never seen.

---

## 7. Where each acts in the network

```
tokens ─► embedding ─┬─► [+ learned PE]      ◄── ABSOLUTE, once
                     └─► [+ sinusoidal PE]   ◄── ABSOLUTE, once
                              │
                    ┌─────────▼─────────┐
                    │  for every layer: │
                    │    Q, K ◄── RoPE  │   ◄── RELATIVE, every layer
                    │    scores ◄ ALiBi │   ◄── RELATIVE, every layer
                    └───────────────────┘
```

Absolute schemes modify the **residual stream**; relative schemes leave it
untouched and act **inside** attention. *Verified:*
`test_absolute_schemes_change_the_representation_relative_ones_do_not`.

---

## My Understanding

> **[USER CHECKPOINT — DEFERRED — USER EXPLAIN-BACK REQUIRED]**
>
> **Checkpoint 4 — Derive sinusoidal positional encoding.**
> From scratch: what three properties are we asking for? Why many frequencies
> rather than one? Why bounded? Derive the rotation identity for a fixed offset
> *k* and explain why it matters that the matrix does not depend on `pos`.
>
> **Checkpoint 5 — Explain RoPE mathematically and intuitively.**
> Give the intuition in one sentence. Then derive why
> `⟨R(m)q, R(n)k⟩` depends only on `m − n`. Why does RoPE preserve norms, and
> why would it be a problem if it did not? What is `offset` for, and what
> breaks if it is wrong during cached generation?
>
> Then, without looking:
> - Why did the field move from learned absolute to relative encodings?
> - RoPE and ALiBi are both relative and both parameter-free. When would you
>   pick ALiBi?
> - Sinusoidal PE also has a rotation property. Why is RoPE's version exact
>   while sinusoidal's is only "linearly recoverable"?

**Related:** [[phase2-attention-from-first-principles]] ·
[[phase2-code-explanation]] · [[interview-phase2-attention]]
