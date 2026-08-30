# Interview Defence — Phase 4, LoRA / QLoRA / rank selection

Questions that test reasoning. Every number is measured and traceable to a
script in this repository.

---

## Q1 (Tier 1). "How would you choose the LoRA rank for a new task and model?"

**Do not answer "use SVD."** Here is why, from a measurement.

Phase 3 gave me a *real* weight update: `ΔW = W_sft − W_base` from a
full-parameter fine-tune of Qwen2.5-1.5B. I took its SVD (E17).

**Measured, layer 13 `q_proj` (1536×1536):**

| rank | cumulative energy | recon. error | LoRA params |
|---:|---:|---:|---:|
| 16 | **10.69%** | 0.9450 | 49,152 |
| 64 | 22.95% | 0.8778 | 196,608 |
| 730 (=90% energy) | 90% | ~0.32 | 2,242,560 |

Against the dense update's 2,359,296 parameters, rank 730 saves **4.9%**. The
break-even rank for a 1536×1536 matrix is **768** — above that a "low-rank"
factorisation costs *more* than what it approximates.

So the naive procedure, applied honestly, recommends a rank nobody would use.

**The resolution.** LoRA's premise is not "the ΔW full fine-tuning produces is
low-rank". It is "**there exists** a low-rank ΔW that achieves comparable task
performance". Unconstrained gradient descent has no incentive to be low-rank —
every loss-reducing direction gets some update — so rank ~730 measures how
unconstrained the optimiser was, not how much rank the task needs.

**What I would actually do:**
1. Treat rank as a hyperparameter and sweep it against validation loss under a
   fixed budget. Cost is linear in `r` and tiny (r=16 attention-only is 0.28%
   of this model).
2. Spend the first extra parameters on **widening the target module set**, not
   raising `r`. Measured (E16): attention-only at r=16 is 4.36M trainable and
   covers 10.0% of the model's weights; all-linear at r=16 is 18.46M and covers
   **84.9%**.
3. Use the spectrum as a **sanity check and relative ranking** — square
   attention projections concentrate better here than MLP matrices, which is a
   reason to prefer them when the budget is tight.

**Follow-up: "so the SVD is useless?"** No — but it needs a control. ΔW needs
rank 730 for 90% energy; a *same-norm Gaussian matrix* needs 783. There is real
structure, just only a 7% margin. Without the control, "the spectrum decays"
would be an observation about every matrix, not about this one.

---

## Q2 (Tier 1). "Derive PCA, and connect it to LoRA."

**PCA.** Maximise variance of a projection under `‖w‖ = 1`:

```
max_w  wᵀ C w   s.t.  wᵀw = 1        C = X_cᵀX_c / (n−1)
```

Lagrangian `wᵀCw − λ(wᵀw − 1)`; setting the gradient to zero gives
`Cw = λw`. The principal directions are eigenvectors of the covariance and the
eigenvalues are the explained variances.

**The SVD connection.** For centred `X_c = UΣVᵀ`:

```
C = X_cᵀX_c/(n−1) = V (Σ²/(n−1)) Vᵀ
```

so principal directions = right singular vectors, and eigenvalues = `σᵢ²/(n−1)`.

**Verified numerically** (executed): covariance eigenvalues
`[7.0631, 1.3734, 0.0108, 0.0105, 0.0092]` are *identical* to `S²/(n−1)`, and
directions match to `|cos| = 1.0000000000`.

**Centring is what makes it PCA.** Executed, with a large mean offset: singular
values `[936.98, 20.50, 2.19]` uncentred vs `[21.14, 13.93, 1.77]` centred.
Uncentred, the first direction just points at the mean.

**The honest connection to LoRA — and its limit.** Both use the SVD and both
are low-rank approximation. But **ΔW is not a data matrix**: it is a difference
of two weight matrices, not a cloud of samples. There is no sample mean, so no
centring step applies, and "explained variance" becomes "explained Frobenius
energy". Calling the SVD of ΔW "PCA" is a category error even though the linear
algebra is shared. I would say they are the *same tool*, not the same method.

**Follow-up: "why squared singular values?"** Eckart–Young: the best rank-`r`
approximation is the truncated SVD and its error is `sqrt(Σ_{i>r} σᵢ²)`.
Verified to machine precision at r = 1, 5, 10, 20, 29. Accumulating *raw*
singular values is the most common error in this analysis — at r=4 the raw
curve says 0.59 while the squared says 0.83, and only the squared one predicts
the measured reconstruction error 0.4096.

---

## Q3 (Tier 1). "What exactly makes QLoRA different from ordinary LoRA?"

**One thing: how the frozen base is stored.** The adapters are identical, the
optimizer is identical, the loss is identical.

| | LoRA | QLoRA |
|---|---|---|
| frozen base storage | bf16 | **4-bit NF4** |
| adapter | bf16, trainable | bf16, trainable |
| matmul arithmetic | bf16 | **bf16** |

**"4-bit" is a storage format, not an arithmetic mode.** Every forward
dequantizes the needed block back to `compute_dtype` and does an ordinary bf16
matmul. Measured (E15): a `Linear4bit` forward differs from a bf16 `nn.Linear`
with the same weights by relative error **0.0910** — that is quantization error
in the *stored weights*, not reduced-precision arithmetic.

**Three mechanisms, per the paper's abstract, and what we used:**
1. **NF4** — levels at the quantiles of a normal distribution. Measured
   NF4 rel. error **0.0912** vs FP4 **0.1210**, so **24.6% better** on normal
   data. Used.
2. **Double quantization** — quantizes the per-block absmax constants too,
   ~0.4 bits/param. Used.
3. **Paged optimizers** — for memory spikes. **Not used**: peak VRAM was
   2.60 GiB against 47.5 GiB available, so there was nothing to page.

**The cost.** Measured on matched smoke runs: LoRA peak **4.44 GiB** at 13.99 s;
QLoRA peak **2.60 GiB** at 16.25 s. QLoRA trades ~16% throughput for ~41% less
memory, because it pays a dequantization on every matmul.

**Follow-up: "why does freezing save so much memory?"** Not mainly the
adapter's size — it is that a frozen parameter has **no optimizer state**. For
AdamW that is two fp32 moments per parameter. QLoRA compounds it: 4-bit storage
*and* no moments for 99.7% of parameters.

**Follow-up: "do gradients touch the base?"** They flow *through* it but not
*to* it. `requires_grad=False` removes a parameter from the optimizer, not from
the computation graph — which is why activations still dominate memory and
gradient checkpointing still helps.

---

## Q4. "You implemented LoRA yourself. Why is B initialised to zero?"

Because it is the only initialisation that is both *safe* and *trainable*.

| init | ΔW at step 0 | gradients | outcome |
|---|---|---|---|
| both zero | 0 | `dL/dA ∝ B = 0`, `dL/dB ∝ A = 0` | **dead saddle** — never moves |
| both random | ≠ 0 | fine | **silently perturbs** the pretrained model |
| **B=0, A random** | **0** | `dL/dA = 0`, `dL/dB ≠ 0` | starts at the base model, and moves |

`B` moves first, then `A` follows once `B` is non-zero.

I test all three, including a test that deliberately zeroes `A` to *demonstrate*
the dead saddle rather than assert it. And I verify the headline property
directly: the adapted model's output is **bitwise identical** to the base
model's at initialisation.

---

## Q5. "How do you know your LoRA implementation is correct?"

46 unit tests prove it is *internally consistent* — which proves only that it
does what I think LoRA is. If my reading of the equation were wrong (scaling in
the wrong place, `B` and `A` transposed) every test would still pass.

So I compared it against `peft` (E18), on a real tiny Qwen2, with adapter
weights copied across:

| check | result |
|---|---|
| same modules adapted | 8 vs 8, identical set |
| same trainable count | 7,168 vs 7,168 |
| logits with matched weights | **max diff 0.000e+00** |
| both == base at init | both exactly 0 |
| merged weights vs `merge_and_unload()` | **max diff 0.000e+00** |

Exact agreement. That is what turns "self-consistent" into "correct".

**What it does not show:** that peft's *training-time* behaviour matches — I
set dropout to 0 precisely to remove that variable.

---

## Q6. "LoRA claims 10,000× fewer trainable parameters. Do you see that?"

**No — I measure 354×**, and the difference is instructive rather than a
contradiction.

| configuration | trainable | ratio |
|---|---:|---:|
| attention, r=16 | 4,358,144 | **354×** |
| q/v only, r=4 | 544,768 | 2,834× |
| attention+MLP, r=16 | 18,464,768 | 84× |

Dense update cost grows as `d²` while adapter cost grows as `r·d`, so the ratio
scales roughly with model width. The paper's figure is for GPT-3 175B — about
100× wider than Qwen2.5-1.5B — at a small rank on two matrices.

**Quoting "10,000×" for a 1.5B model would be wrong.** It is exactly the sort
of number that gets repeated without checking, and checking took one division.

---

## Q7. "Merging — what does it buy and what does it cost?"

Folding `ΔW = (α/r)BA` into `W` makes the layer an ordinary `nn.Linear`, so
there is **zero additional inference latency** — LoRA's advantage over adapter
architectures that add depth.

Verified: merged output matches adapter output to 2.682e-07, and merged weights
match peft's exactly.

**What it costs:** you lose the ability to swap adapters cheaply. Unmerged, one
frozen base serves many tasks by switching a ~17 MiB adapter. Merged, you have
a full 2.9 GiB model per task. The right choice depends on whether you serve
one task or many.

**Subtlety:** `unmerge()` is exact only to float tolerance — subtracting is not
bitwise reversible. My test asserts a tolerance and says why.

---

## Q8 (challenge). "Your SVD result seems to undermine LoRA. Defend it."

It undermines a *popular explanation* of LoRA, not LoRA.

The claim "ΔW is low-rank, therefore factorise it" is not what the measurement
supports: the real ΔW needs ~730 of 1536 for 90% energy and beats noise by only
7%. If that claim were LoRA's foundation, LoRA should not work at r=16.

What actually holds is weaker and sufficient: a low-rank update **suffices**
even though the unconstrained one is not low-rank. There is no contradiction
because LoRA never has to reproduce full fine-tuning's ΔW — it optimises within
the low-rank family directly and finds its own solution there.

**What would change my mind:** if a LoRA run at r=16 landed far worse than full
fine-tuning on matched settings, the low-rank family would be too small and the
premise would be in trouble. That is measured in this phase, not assumed.

---

## Q9 (challenge). "What's the weakest part of your Phase 4 evidence?"

Three things, in order.

1. **Single seed, single run per arm.** No variance estimate, so a small gap
   between arms is not distinguishable from noise.
2. **The learning rate is confounded with the arm.** LoRA conventionally wants
   a higher LR than full FT because adapters start at zero. Holding it fixed
   makes the comparison about LR sensitivity; changing it breaks the control.
   I ran LoRA at *both* 2e-4 and 2e-5 so both readings exist — but that is a
   mitigation, not a solution.
3. **In-distribution evaluation only.** Perplexity on no_robots' own test
   split. "Better" is partly circular.

Also: no rank sweep was run, so r=16 is a starting point, not a tuned value —
which is exactly the thing Q1 says you should sweep.

---

## Explain Back

> **[USER CHECKPOINT — DEFERRED — USER EXPLAIN-BACK REQUIRED]**
>
> **Drill 1** (LoRA via SVD) and **Drill 2** (PCA derivation) are the two Tier-1
> drills this phase covers. Answer Q1, Q2, Q3 and Q6 aloud without notes, then
> write two questions this document does not answer and answer those.

**Related:** [[phase4-lora-svd-and-pca]] · [[phase4-code-explanation]] ·
[[pytorch-quantization-and-lowrank]] · [[interview-phase3-sft]]
