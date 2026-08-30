# Phase 6 — DPO: theory, implementation, and a negative result

**Status:** engineering complete, USER explain-back DEFERRED.

---

## 1. The derivation

RLHF maximises reward under a KL constraint against a frozen reference:

```
max_π  E_{y~π}[ r(x,y) ]  −  β · KL( π(·|x) ‖ π_ref(·|x) )        (paper Eq. 3)
```

This has a closed form (paper Eq. 4, **read directly from the HTML full text**):

```
π_r(y|x) = (1/Z(x)) · π_ref(y|x) · exp( (1/β) r(x,y) )
Z(x)     = Σ_y π_ref(y|x) exp( (1/β) r(x,y) )
```

The paper notes: *"it is still expensive to estimate the partition function
Z(x), which makes this representation hard to utilize in practice. However, we
can rearrange Eq. 4 to express the reward function in terms of its
corresponding optimal policy."*

Solving for the reward:

```
r(x,y) = β · log[ π*(y|x) / π_ref(y|x) ] + β · log Z(x)
```

Substituting into Bradley-Terry, **`log Z(x)` cancels** — it depends only on
`x`, and both responses share `x`:

```
L_DPO = − log σ( β·[log π_θ(y_w|x) − log π_ref(y_w|x)]
                − β·[log π_θ(y_l|x) − log π_ref(y_l|x)] )
```

> **The cancellation is why both responses must share a prompt.** Phase 5's E21
> verified that on 3000/3000 rows precisely because this step depends on it.

**Notice the shape:** this is the *same* Bradley-Terry loss a reward model is
trained with (Phase 5 §3), with the reward difference replaced by a log-ratio
difference. The reward model has been reparameterised into the policy.

### Provenance

The derivation above was written from standing knowledge in Phase 5. For Phase
6 the paper's HTML full text was fetched and **Section 3 and Appendix B were
read in full**; Sections 1, 2, 5, 6, 7 and Appendices A.1–A.3, A.5, A.6 were
**not**. Equations 3 and 4 are quoted from the text. The rearrangement and
cancellation steps are ours, and were **verified numerically**, not cited.

---

## 2. Verified against the paper's own code

Appendix B gives a reference implementation verbatim:

```python
pi_logratios  = pi_yw_logps - pi_yl_logps
ref_logratios = ref_yw_logps - ref_yl_logps
losses = -F.logsigmoid(beta * (pi_logratios - ref_logratios))
rewards = beta * (pi_logps - ref_logps).detach()
```

Ours groups the terms differently — per-response implicit rewards, then their
difference — which makes the implicit reward a first-class loggable quantity:

```
r_w = β·(log π_w − log π_ref,w)
r_l = β·(log π_l − log π_ref,l)
L   = −logsigmoid(r_w − r_l)
```

Algebraically identical, and **asserted numerically to 1e-12 at β ∈ {0.01, 0.1,
0.5, 1.0}**. Our grouping was chosen *before* the appendix was read, so this is
a genuine independent check.

---

## 3. What β controls

**Executed** (`PYTORCH_CONCEPTS/examples/dpo_mechanics_examples.py`), log-ratio
difference fixed at 4 nats so the logit is exactly `4β`:

| β | logit | loss | dL/dlogit |
|---:|---:|---:|---:|
| 0.001 | 0.0040 | 0.691149 | −0.499000 |
| 0.1 | 0.4000 | 0.513015 | −0.401312 |
| 0.5 | 2.0000 | 0.126928 | −0.119203 |
| 2.0 | 8.0000 | 0.000335 | −0.000335 |

Small β keeps the *logit* gradient near its maximum but multiplies the
*parameter* gradient by β too, so the policy barely moves. Large β saturates
the sigmoid and the gradient vanishes. β trades **how far the policy may move**
against **how much signal survives once it has**.

---

## 4. The experiment — pre-registered

β ∈ {0.01, 0.1, 0.5}, SUM objective, fixed evaluation split, identical budget
per arm. Committed at 13:15:12, **before any DPO code existed**
(`docs/phase6/BETA_PREREGISTRATION.md`).

Policy and reference both start from the Phase 3 SFT checkpoint. Both
pre-flight checks passed on every run: reference frozen (0 trainable, eval
mode), and implicit reward **exactly 0** at initialisation with loss
`0.6931471824645996 = log 2`.

---

## 5. The result

| model | **SUM (primary)** | MEAN (diag) | reward acc | margin | KL median | gen len | im_end |
|---|---:|---:|---:|---:|---:|---:|---:|
| SFT baseline | **46.7%** | 58.7% | — | 0.0000 | 0.0000 | 93.8 | 4/4 |
| DPO β=0.01 | **46.7%** | 58.7% | 51.1% | −0.0001 | 0.0008 | 95.8 | 4/4 |
| DPO β=0.1 | **46.7%** | 58.7% | 51.1% | +0.0041 | 0.0008 | 96.0 | 4/4 |
| DPO β=0.5 | **46.7%** | 58.7% | 54.3% | +0.0090 | 0.0008 | 95.8 | 4/4 |

**Preference accuracy did not change in any arm — not by a single pair out of
184.**

### Hypothesis outcomes

| | prediction | outcome |
|---|---|---|
| **H1** | all β reduce loss below `log 2` | **NOT SUPPORTED** — logged losses drifted up, but they are single-pair noise (§7); the eval margins moved only for β ≥ 0.1 |
| **H2** | SUM accuracy exceeds 47.2% for some β | **DISPROVED** — unchanged at 46.7% everywhere |
| **H3** | KL monotone decreasing in β | **NOT SUPPORTED** — all three at 0.0008, indistinguishable |
| **H4** | length increases for some β | **DISPROVED in the sweep** (+2%); **CONFIRMED post-hoc** (+30%) |
| **H5** | SUM and MEAN may diverge | **NOT OBSERVED in the sweep**; observed post-hoc (SUM flat, MEAN +0.5p) |

### The direction *was* right

β=0.1 raised `log π(chosen)` by +0.0199 and lowered `log π(rejected)` by
−0.0206. The reward margin is **monotone in β**: −0.0001, +0.0041, +0.0090.
The mechanism works — the magnitude is the problem.

### The decisive number

**KL(policy ‖ SFT reference) median = 0.0008**, against Phase 5's
**KL(SFT ‖ base) median = 0.2044**.

> The policy moved roughly **250× less** than one epoch of SFT moved from the
> base model. On sequences of ~270 tokens, a ±0.02-nat shift cannot flip a
> ranking — which is exactly what the unchanged accuracy shows.

This is a **training-budget** result, not evidence that DPO does not work: 116
optimiser steps × 16 pairs = 1,856 pairs seen, at lr 5e-7. The paper uses batch
64 with RMSprop at 1e-6.

### The post-hoc diagnostic — NOT pre-registered

One run at **10x the learning rate** (5e-6, beta=0.1, everything else
identical), launched *after* the sweep's results were seen and labelled
`[POST-HOC]` everywhere so it cannot displace the primary result:

| | SUM | MEAN | reward acc | margin | gen len |
|---|---:|---:|---:|---:|---:|
| SFT baseline | 46.7% | 58.7% | - | 0.0000 | 93.8 |
| beta=0.1, lr 5e-7 | 46.7% | 58.7% | 51.1% | +0.0041 | 96.0 |
| **[POST-HOC] lr 5e-6** | **46.7%** | **59.2%** | **65.2%** | **+0.0700** | **122.0** |

The policy moved **~49x further** (log pi(chosen) +0.9773 vs +0.0199), the
margin grew **17x**, reward accuracy reached 65.2% - and **SUM accuracy still
did not move a single pair**, while **generated length rose 30%**.

**H4 is confirmed here**, having been disproved in the sweep. The length trap
Phase 5 predicted appears exactly when training has enough signal to exploit it.

---

## 5b. Why SUM accuracy cannot move - the arithmetic

**Measured on the evaluation set:**

| | chosen | rejected |
|---|---:|---:|
| mean summed log-prob | -291.12 | -261.65 |
| mean tokens | 271.7 | 242.2 |
| **mean per-token log-prob** | **-1.0713** | **-1.0803** |

**Per token, the chosen response is genuinely more likely.** The SUM comparison
inverts that verdict purely because chosen carries 29.5 more tokens:

```
SUM gap (chosen - rejected)      = -29.47 nats
explained by length alone        = -31.90 nats
residual once length is removed  =  +2.43 nats   (chosen is BETTER)
```

> The 46.7% SUM baseline is a **length artifact ~29 nats deep**. MEAN removes
> the length term and reports 58.7% - which is exactly the 11-point swing
> Phase 5 measured on identical models and data.

To flip the *average* pair under SUM, the policy must shift the gap by ~29
nats:

| run | relative movement | shortfall |
|---|---:|---:|
| beta=0.1, lr 5e-7 | +0.0405 nats | **728x short** |
| [POST-HOC] lr 5e-6 | +0.7004 nats | **42x short** |

Even the 10x run is 42x short. The primary metric is **dominated by a property
of the data, not of the model** - the same lesson as Phase 3's masked loss,
Phase 4's stop-token gap, and Phase 5's baseline swing.

The pre-registration anticipated this outcome in §7: *"Training loss falls but
SUM preference accuracy does not exceed 47.2% → DPO optimised its objective
without improving the measured preference."* The measured case is stronger —
DPO barely optimised its objective at all.

---

## 6. What was NOT confounded

Worth stating, because Phase 5 predicted a length trap:

- **Length did not blow up.** +2 tokens, 1.02×. H4 disproved.
- **Stop-token behaviour preserved.** 4/4 emit `<|im_end|>` in every arm — the
  capability Phase 4 found LoRA could not learn is intact.
- **Qualitative outputs are near-identical** — one phrase differs between the
  baseline and the DPO models.

So there is no hidden degradation being masked by a flat metric. The models
simply barely changed.

---

## 7. A reporting weakness found in our own logs

`dpo_summary.json["history"]` records the stats of the **last pair** in each
gradient-accumulation window, not a window average. With one pair per forward,
a ±0.01 margin there is **noise, not trend**.

This is why §5 leads with the eval-set figures over 184 pairs and treats the
logged loss trajectory as unreliable. Found while reading the sweep logs; the
training code was **deliberately not patched mid-sweep**, because β=0.1 and
β=0.5 had to run the same code as β=0.01 for the arms to be comparable.

**A second metric caveat.** `reward_accuracy` reads "0.0000 → 0.5109". That is
**not** a 51-point gain: at baseline every margin is *exactly* zero, so a
`margin > 0` test counts none of them. The baseline is degenerate, not terrible.

---

## My Understanding

> **[USER CHECKPOINT — DEFERRED — USER EXPLAIN-BACK REQUIRED]**
>
> **Checkpoint 20 — Derive the DPO loss.** From `π* ∝ π_ref·exp(r/β)`, solve
> for `r`, substitute into Bradley-Terry, and show where `log Z(x)` cancels.
> Name the step that requires both responses to share a prompt.
>
> Without looking:
> - Why is the loss exactly `log 2` at initialisation, and why is that a useful
>   check rather than a triviality?
> - β=0.01 and β=0.5 gave reward margins of −0.0001 and +0.0090. Explain the
>   ordering from the loss's algebra.
> - KL from the reference was 0.0008 while SFT-from-base was 0.2044. What does
>   that ratio tell you, and what does it *not* tell you?

## Explain Back

> **[USER CHECKPOINT — DEFERRED — USER EXPLAIN-BACK REQUIRED]**
>
> **Checkpoint 21 — Interpret a null result.** Preference accuracy was
> identical across all β. Argue the case that DPO failed here, then the case
> that the experiment was underpowered, and say what single measurement best
> distinguishes them.
>
> **Checkpoint 22 — β.** What does β trade off? Why does small β both *keep*
> the logit gradient large and *reduce* how far the policy moves?

**Related:** [[phase6-code-explanation]] · [[interview-phase6-dpo]] ·
[[phase5-rlhf-ppo-and-dpo]] · [[pytorch-dpo-mechanics]]
