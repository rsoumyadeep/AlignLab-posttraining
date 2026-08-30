# Phase 4 — LoRA, SVD, and PCA

**Status:** engineering complete, USER explain-back DEFERRED.

---

## 1. LoRA in one equation

A full fine-tune learns an update to every weight:

```
W' = W + ΔW            ΔW has the same shape as W:  d_out × d_in
```

LoRA factorises the update and freezes the base:

```
ΔW = B A               B: d_out × r,   A: r × d_in,   r ≪ min(d_out, d_in)

W'x = Wx + (α/r) · B A x
```

Parameters drop from `d_out·d_in` to `r·(d_out + d_in)`.

**Measured on Qwen2.5-1.5B** (E16): adapting the four attention projections at
`r=16` gives **4,358,144** trainable parameters — **0.28%** of the model's
1,543,714,304.

### Why the scaling is α/r and not α

Changing `r` should not force you to re-tune the learning rate. `A` is
initialised from a zero-mean distribution and `B` from zeros, so the variance
of `BAx` grows roughly linearly in `r`; dividing by `r` keeps the update's
magnitude roughly `r`-independent. `α` is then a single "how much adaptation"
knob, decoupled from capacity.

The common convention `α = 2r` makes the scale exactly 2 at every rank — which
is what makes a rank sweep *not* also a learning-rate sweep. AlignLab uses it.

### Why B = 0 and A random — the only workable choice

| init | ΔW at step 0 | gradients | verdict |
|---|---|---|---|
| both zero | 0 | `dL/dA ∝ B = 0`, `dL/dB ∝ A = 0` | **dead saddle point** — never moves |
| both random | ≠ 0 | fine | **silently perturbs** the pretrained model |
| **B=0, A random** | **0** | `dL/dA = 0`, `dL/dB ≠ 0` | starts exactly at the base model, and moves |

All three are asserted in `tests/test_lora.py`, including a test that
deliberately zeroes `A` to demonstrate the dead saddle point rather than merely
claiming it.

---

## 2. SVD — the machinery

Any real matrix factorises as

```
A = U Σ Vᵀ      U, V orthonormal;  Σ = diag(σ₁ ≥ σ₂ ≥ … ≥ 0)
```

**Eckart–Young:** the best rank-`r` approximation in Frobenius norm is the
truncated SVD `A_r = U_r Σ_r V_rᵀ`, and its error is

```
‖A − A_r‖_F = sqrt( Σ_{i>r} σᵢ² )
```

**This is why "explained energy" accumulates SQUARED singular values.** Using
raw ones is the most common error in write-ups on this topic. Executed
(`examples/svd_pca_examples.py`, §3): at `r=4` the raw cumulative fraction says
**0.5911** while the squared says **0.8322**, and only the squared one predicts
the measured reconstruction error `sqrt(1 − 0.8322) = 0.4096` — matching the
observed `0.409598` exactly.

Also executed (§2), the identity holds to machine precision at every rank:

| r | ‖A − A_r‖_F | sqrt(tail sum of σ²) | agree |
|---:|---:|---:|:--:|
| 1 | 33.390762 | 33.390762 | ✓ |
| 5 | 27.020925 | 27.020925 | ✓ |
| 20 | 8.661172 | 8.661172 | ✓ |

And optimality is not merely asserted: the truncated rank-5 SVD's error is
**27.020925**, against **74.302855** for the best of 20 random rank-5
factorisations of the same matrix.

---

## 3. PCA — the same machinery, applied to data

PCA maximises variance under an orthonormality constraint. Solving it gives the
eigenvectors of the covariance matrix, ordered by eigenvalue.

**The connection to SVD.** For centred data `X_c` (n × d):

```
C = X_cᵀ X_c / (n − 1)        covariance
X_c = U Σ Vᵀ                   SVD
⟹ C = V (Σ² / (n − 1)) Vᵀ
```

So:

| PCA object | SVD object |
|---|---|
| principal directions | right singular vectors `V` |
| eigenvalues of `C` | `σᵢ² / (n − 1)` |
| explained variance ratio | `σᵢ² / Σσ²` |
| projection onto `k` components | `U_k Σ_k` |

**Executed** (§4): eigenvalues of the covariance
`[7.0631, 1.3734, 0.0108, 0.0105, 0.0092]` are **identical** to `S²/(n−1)` from
the SVD, and the principal directions agree with the right singular vectors to
`|cos| = 1.0000000000` (up to sign, which is arbitrary in both).

### Centring is what makes it PCA

**Executed** (§5), same data with a large mean offset:

```
singular values, UNCENTRED : [936.98, 20.50, 2.19]
singular values, CENTRED   : [ 21.14, 13.93, 1.77]
```

Uncentred, the first direction just points at the mean and dominates
everything — the ratio of first to second is 45.7 versus 1.5 centred. **PCA
without centring measures where the data *is*, not how it *varies*.**

**Why this does not apply to ΔW.** `ΔW` is a *difference of two weight
matrices*, not a cloud of data points. There is no sample mean to remove, so an
SVD of `ΔW` is already the right object and no centring step exists. Conflating
"SVD of a weight update" with "PCA of a dataset" is a category error, even
though the linear algebra is shared.

---

## 4. Choosing a LoRA rank — what actually happened

Phase 3 produced a **real** ΔW: `W_sft − W_base` from a full-parameter fine-tune.
E17 measured its spectrum, with two controls (a same-norm Gaussian matrix, and
the pretrained weight itself).

### The prediction, and its failure

> **H2 (recorded before running):** r ≤ 64 captures ≥ 50% of ΔW's energy.

**Measured, layer 13 `q_proj` (1536×1536):**

| rank | energy | noise control | recon. error | LoRA params | compression |
|---:|---:|---:|---:|---:|---:|
| 1 | 3.33% | 0.26% | 0.9832 | 3,072 | 768× |
| 8 | 7.80% | 2.01% | 0.9602 | 24,576 | 96× |
| **16** | **10.69%** | 3.95% | **0.9450** | 49,152 | 48× |
| 64 | **22.95%** | 14.57% | 0.8778 | 196,608 | 12× |
| 256 | 54.53% | 46.92% | 0.6743 | 786,432 | 3× |

**H2 is DISPROVED.** r=64 captures 22.95%, not ≥50%. Reaching 90% needs rank
**~730 of 1536**.

Rank needed for 90% energy, across modules:

| module | shape | r@90% | r@90% / max |
|---|---|---:|---:|
| `layers.13.self_attn.q_proj` | (1536, 1536) | 730 | 0.475 |
| `layers.13.self_attn.o_proj` | (1536, 1536) | 726 | 0.473 |
| `layers.27.self_attn.q_proj` | (1536, 1536) | 624 | 0.406 |
| `layers.13.self_attn.v_proj` | (256, 1536) | 196 | 0.766 |
| `layers.13.mlp.gate_proj` | (8960, 1536) | 1196 | 0.779 |

And the noise control needs **783** where ΔW needs **730** — real structure,
but a **7% margin**, not an order of magnitude.

### Why this does not sink LoRA

> LoRA's premise is **not** "the ΔW full fine-tuning produces is low-rank".
> It is "**there exists** a low-rank ΔW achieving comparable task performance".

Unconstrained gradient descent has no reason to produce a low-rank update —
every loss-reducing direction gets some update, so the solution spreads across
the whole spectrum. Rank ~730 measures **how unconstrained the optimiser was**,
not how much rank the task needs.

### The break-even that makes the naive procedure absurd

**Executed** (§6): for a 1536×1536 matrix the break-even rank is **768**.

```
r=730 :  730 × (1536+1536) = 2,242,560   vs dense 2,359,296   →  saves 4.9%
r=16  :   16 × (1536+1536) =    49,152   vs dense 2,359,296   →  saves 97.9%
```

Worse, for the narrow GQA projections (256×1536, dense 393,216) rank 256 costs
**458,752** — **more than the dense update**. "Low-rank" is only low-rank below
the break-even.

So "SVD the update and read off the rank at 90% energy" returns 730, which
saves 4.9% and which nobody uses. **The procedure is not imprecise; it is
wrong.**

### Six reasons it is a heuristic, not a rule

1. **It measures a solution, not a requirement.** An upper bound at best.
2. **Energy ≠ usefulness.** A tiny singular direction may carry the task signal.
3. **It is post-hoc and circular** — it needs the full fine-tune LoRA exists to avoid.
4. **The matrices disagree** — different modules want different ranks; a global `r` is already a compromise.
5. **Task and data scale dominate** rank more than the model does.
6. **Precision-limited.** ΔW is a difference of bf16 tensors; its tail sits near the floor. The SVD runs in float64 to keep round-off out, but the *input* is bf16.

---

## 5. What actually determines the rank

- **Sweep it against validation loss under a fixed budget.** It is a
  hyperparameter; cost is linear in `r` and tiny (0.28% at r=16).
- **Spend the first extra parameters on WIDENING the target set**, not raising
  `r`. E16 measured: attention-only at r=16 is 4.36M trainable; all-linear at
  r=16 is 18.46M and covers 84.9% of the model's weights instead of 10.0%.
- **Use the spectrum as a sanity check and a relative ranking** — square
  attention projections concentrate better than MLP here, which is a reason to
  prefer them when the budget is tight.

---

## My Understanding

> **[USER CHECKPOINT — DEFERRED — USER EXPLAIN-BACK REQUIRED]**
>
> **Drill 1 — LoRA via SVD.** How does the singular-value spectrum of a weight
> update help justify a LoRA rank? Then, having seen E17: explain why it mostly
> *doesn't*, and what the measurement actually tells you.
>
> Without looking:
> - r=16 captures 10.69% of the real ΔW's energy, yet r=16 adapters work. Resolve that.
> - What is the break-even rank for a 1536×1536 matrix, and why does it matter?
> - Why is a same-norm random control necessary before claiming a spectrum decays?

## Explain Back

> **[USER CHECKPOINT — DEFERRED — USER EXPLAIN-BACK REQUIRED]**
>
> **Drill 2 — PCA.** Derive PCA from variance maximisation: the constrained
> optimisation, the Lagrangian, why the answer is an eigenvector problem.
> Then relate eigenvalues to singular values, explain what centring does, and
> say why no centring step applies to ΔW.
>
> **Checkpoint 16 — LoRA initialisation.** Why B=0 and A random? What exactly
> goes wrong with both-zero, and with both-random? Write the two gradients.

**Related:** [[phase4-code-explanation]] · [[pytorch-svd-and-quantization]] ·
[[interview-phase4-peft]] · [[phase3-sft-and-loss-masking]]
