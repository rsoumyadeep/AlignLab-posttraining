# Phase 5 — Preference Learning: RLHF, PPO, and why DPO

**Status:** engineering complete, USER explain-back DEFERRED.

**Scope label:** the PPO material in §4–§6 is **CONCEPTUAL — NOT IMPLEMENTED**.
No PPO training system exists in AlignLab and none is planned. The preference
data infrastructure (§2, §8) and the measurements (§3, §7) are **IMPLEMENTED
and MEASURED**.

---

## 1. The pipeline, and where each phase sits

```
   pretraining                    OUTSIDE ALIGNLAB
        │
        ▼
   Qwen2.5-1.5B  ──────────────►  base / reference model
        │
        │  SFT on demonstrations                     PHASE 3 ✅ IMPLEMENTED
        ▼
   π_SFT  ─────────────────────►  the policy DPO starts from,
        │                          AND the reference π_ref it is compared to
        │
        ├──── RLHF branch ────────────────────────── CONCEPTUAL, NOT IMPLEMENTED
        │        preference data → reward model r_φ
        │        → PPO(policy, reference, value, reward) → aligned π
        │
        └──── DPO branch ─────────────────────────── PHASE 6, NOT YET RUN
                 preference data → DPO loss → aligned π
```

The two branches consume **the same preference data** and aim at **the same
optimum**. They differ in what they build in between.

---

## 2. Preference data — IMPLEMENTED

A preference example is a prompt with two candidate responses and a label:

```
x        prompt
y_w      chosen   ("w" = win)
y_l      rejected ("l" = lose)
```

`alignlab.preference` loads, validates, audits and fingerprints these.

**Measured on `HuggingFaceH4/ultrafeedback_binarized` (E21):**

| property | measured |
|---|---|
| raw train rows | 61,135 (`train_prefs`, **not** `train`) |
| score **ties** (chosen == rejected) | **11.9%** train, 13.6% eval |
| inverted (chosen < rejected) | 0.0% |
| **identical** chosen/rejected text | 14 in 3,000 |
| empty responses | 2 in 3,000 |
| prompt shared by both responses | **3000/3000** |
| chosen longer, characters | 55.8% (1291 vs 1132) |
| chosen longer, **tokens** | **56.5%** (283.7 vs 241.6) |

Three of these are filtered. **The length bias cannot be** — it is a property
of the data, and §7 shows it changes a headline number by 11 points.

> **The 11.9% tie rate matters conceptually, not just practically.**
> Bradley-Terry assumes a genuine ordering. A tie says the annotator saw no
> difference, yet the pair is presented as "prefer y_w". Dropping ties is
> defensible; `drop_ties` therefore defaults to **False** so the choice is an
> explicit, recorded decision rather than a silent default.

---

## 3. The reward model — CONCEPTUAL (not trained here)

A reward model `r_φ(x, y)` maps a prompt/response pair to a scalar. It is
usually the SFT model with the language-modelling head replaced by a scalar
head, trained on the **Bradley-Terry** likelihood:

```
P(y_w ≻ y_l | x) = σ( r_φ(x,y_w) − r_φ(x,y_l) )

loss = − log σ( r_φ(x,y_w) − r_φ(x,y_l) )
```

`alignlab.preference.bradley_terry_{probability,loss}` implement exactly this
and are unit-tested. **No reward model was trained** — the functions exist so
the concept has a runnable referent.

Two properties worth having in your hands rather than only in prose, both
asserted in tests:

- **Only the difference is identified.** `r(·)+c` gives identical
  probabilities for every `c`. Preference data cannot pin down an absolute
  reward scale, which is why a reward model's raw output is not interpretable
  on its own and why RM scores are not comparable across training runs.
- **It saturates.** A large margin gives probability ≈ 1 and a nearly flat
  gradient, so confidently-ordered pairs stop contributing signal.

---

## 4. PPO — CONCEPTUAL, NOT IMPLEMENTED

RLHF-with-PPO needs **four** models in memory:

| model | role | trained? |
|---|---|---|
| **policy** `π_θ` | generates; the thing being aligned | yes |
| **reference** `π_ref` | frozen SFT model; anchors the KL penalty | no |
| **reward** `r_φ` | scores completed generations | no (pre-trained) |
| **value** `V_ψ` | estimates expected return, for advantages | yes |

The objective, per token:

```
maximise  E[ r_φ(x,y) − β · KL( π_θ(·|x) ‖ π_ref(·|x) ) ]
```

optimised with PPO's clipped surrogate:

```
ratio  = π_θ(a|s) / π_θ_old(a|s)
L_clip = E[ min( ratio · Â , clip(ratio, 1−ε, 1+ε) · Â ) ]
```

The clip is what makes it *proximal*: it removes the incentive to move the
policy far in one update, because beyond `1±ε` the objective stops improving.

**Why this is hard in practice** — and this is the honest answer to "why not
PPO":

1. **Four models, two of them training.** Memory and complexity multiply.
2. **It is on-policy.** Every update needs *fresh generations*, so training
   contains a full generation loop — slow, and a second source of bugs
   (sampling settings, truncation, EOS handling).
3. **Reward hacking.** The policy optimises `r_φ`, not human preference.
   `r_φ` is an imperfect proxy, and the optimiser will find its flaws. The KL
   penalty exists mostly to slow this down.
4. **Many interacting hyperparameters**: `β`, clip `ε`, value coefficient, GAE
   `λ`, epochs per batch, generation length.
5. **The reward model is a separate artifact** to train, validate and version.

---

## 5. The KL constraint — CONCEPTUAL, but MEASURED here

Without a constraint, maximising a learned reward drives the policy to whatever
degenerate text the reward model happens to score highly. The penalty
`β · KL(π_θ ‖ π_ref)` keeps it near a model that already produces sensible
language.

**The direction is not arbitrary.** `KL(policy ‖ reference)` is large when the
policy puts mass where the reference puts none — which is exactly what reward
hacking looks like in distribution space. The reverse direction would not
penalise that.

**Measured (E22), giving the constraint a scale on real models:**

| policy | KL(policy ‖ base), mean | median | max |
|---|---:|---:|---:|
| full SFT | 0.6547 | **0.2044** | 7.3349 |
| LoRA @2e-4 | 0.6233 | 0.2008 | — |
| QLoRA @2e-4 | 0.5564 | 0.1589 | — |

The distribution is **heavily right-skewed** — the mean is ~3× the median — so
the median is the number to quote: one epoch of SFT moved a typical token's
distribution by about **0.20 nats**. That is the order of magnitude a penalty
coefficient would have to be chosen against.

LoRA and QLoRA moved *less*, consistent with changing 0.28% of parameters and
being unable to touch the output projection (Phase 4's E20).

### The estimator PPO actually uses

Exact KL needs the full vocabulary distribution at every position — `[B, T,
151936]` here, too large to hold for a PPO batch. So implementations estimate
it from sampled tokens:

```
k1 = log π(y) − log π_ref(y)        unbiased, high variance
k3 = exp(−k1) − 1 + k1              biased low, lower variance, ≥ 0
```

`alignlab.logprobs.approximate_kl` implements `k3`. **"PPO penalises KL" and
"PPO penalises an estimator of KL with different properties" are different
statements, and the second is true.**

---

## 6. DPO's key idea — why the reward model disappears

The RLHF objective has a **closed-form optimum**:

```
π*(y|x) ∝ π_ref(y|x) · exp( r(x,y) / β )
```

Rearranging for the reward:

```
r(x,y) = β · log[ π*(y|x) / π_ref(y|x) ] + β log Z(x)
```

Substituting into Bradley-Terry, **`log Z(x)` cancels** because it depends only
on `x` and both responses share `x`:

```
L_DPO = − log σ(  β·[ log π_θ(y_w|x) − log π_ref(y_w|x) ]
                − β·[ log π_θ(y_l|x) − log π_ref(y_l|x) ] )
```

**The reward model is gone.** It has been *reparameterised* into the policy
itself — hence the paper's subtitle, *"Your Language Model is Secretly a Reward
Model"*.

Note the shape: this is the **same Bradley-Terry loss from §3**, with the
reward difference replaced by a difference of log-ratios. That single
observation is the most useful thing to have ready when asked how DPO relates
to reward modelling.

### What DPO removes

| RLHF/PPO | DPO |
|---|---|
| reward model (train, validate, version) | **gone** |
| value model | **gone** |
| on-policy generation loop | **gone** |
| PPO hyperparameters (ε, λ, value coef, …) | **gone** |
| 4 models in memory | **2** (policy + frozen reference) |
| RL machinery | a **classification loss** |

### What DPO assumes — and gives up

1. **Bradley-Terry is the right preference model.** Ties violate it (11.9%
   here), and it assumes transitive, consistent annotators.
2. **The closed-form optimum is reachable** by optimising the reparameterised
   objective directly — exact in theory, approximate under finite data and SGD.
3. **It is off-policy.** DPO only ever sees the *fixed* preference dataset. It
   never generates, so it never gets feedback on **its own current outputs** —
   which is precisely what PPO's generation loop provides.
4. **No reusable reward model.** PPO's `r_φ` can score arbitrary new
   completions, rank candidates, or drive best-of-n. DPO produces none.
5. **`β` still matters.** DPO does not eliminate the KL trade-off; it absorbs
   it into a single coefficient.

### When PPO is still preferable

- You want a **reward model as an artifact** (rejection sampling, best-of-n,
  evaluation, monitoring).
- Your reward is **programmatic** rather than preference-derived (unit tests
  pass, a verifier accepts) — there are no preference pairs to run DPO on.
- You need **on-policy** correction of the model's *own* current failures,
  which a fixed dataset cannot supply.
- You want to **continue improving** past what a static preference set covers.

---

## 7. The measurement that changes how Phase 6 must be read

**E22 asked: before any DPO, does the SFT model already prefer the chosen
response?**

| comparison | prefers chosen |
|---|---:|
| **SUM of log-probs** (the published DPO objective) | **51/108 = 47.2%** |
| MEAN per token (length-normalised) | **63/108 = 58.3%** |

**Identical models. Identical data. An 11.1-point swing — below chance versus
comfortably above it.**

The cause is length, not quality. Chosen responses are 56.5% longer in tokens
(§2), and a summed log-probability is more negative for a longer sequence
*simply for being longer*. The SUM comparison is therefore biased **against**
the chosen response, hard enough to push it under 50%.

**Consequences for Phase 6:**

1. The baseline is **47.2%**, not 58.3%, if the published objective is used.
   Quoting the length-normalised figure while training on the summed one would
   understate DPO's improvement.
2. "DPO raised preference accuracy from 47% to X%" needs the length caveat —
   part of any gain may be the model learning to **lengthen**, not improve.
3. Phase 6 **must record which variant it optimises**. They are different
   objectives that share a name.

### The reference-model wiring check

With `policy == reference`, the implicit reward must be **exactly 0**.
Measured: `0.000e+00` on every sequence. This costs nothing and catches a
reference plumbed to the wrong checkpoint — a bug that otherwise appears as a
run that trains happily while optimising the wrong thing.

---

## 8. Data flow — what Phase 6 will actually execute

```
preference pair (x, y_w, y_l)
   │  apply_chat_template  →  prompt text, + chosen / + rejected
   ▼
tokenize                        prompt tokens MUST be a prefix of both
   │                            (E21: 0 violations in 400 sampled pairs)
   ▼
labels: −100 on prompt, ids on completion        [alignlab.masking]
   │
   ├── π_θ    (trainable)  ──►  log π_θ(y_w|x),  log π_θ(y_l|x)
   └── π_ref  (frozen)     ──►  log π_ref(y_w|x), log π_ref(y_l|x)
   │                            [alignlab.logprobs.sequence_logprobs]
   ▼
implicit rewards  β·(log π_θ − log π_ref)        [logprob_ratio]
   ▼
L = − log σ( r_w − r_l )                          ← PHASE 6, NOT YET WRITTEN
```

Everything above the last line is **implemented and tested** in Phase 5. The
DPO loss itself belongs to Phase 6.

---

## My Understanding

> **[USER CHECKPOINT — DEFERRED — USER EXPLAIN-BACK REQUIRED]**
>
> **Checkpoint 17 — Derive DPO from the RLHF objective.** Start from
> `π* ∝ π_ref · exp(r/β)`, solve for `r`, substitute into Bradley-Terry, and
> show why `log Z(x)` cancels. Say exactly which step needs both responses to
> share a prompt.
>
> Without looking:
> - Why can preference data never identify an absolute reward scale?
> - Why is `KL(π ‖ π_ref)` the right direction rather than the reverse?
> - The pre-DPO baseline is 47.2% by SUM and 58.3% by MEAN. Explain the gap,
>   and say which you would report and why.

## Explain Back

> **[USER CHECKPOINT — DEFERRED — USER EXPLAIN-BACK REQUIRED]**
>
> **Checkpoint 18 — RLHF vs DPO, end to end.** Draw both pipelines. Name every
> model each needs, which are trained, and what each contributes. Then argue
> both sides: a case where DPO is clearly right, and a case where PPO is.
>
> **Checkpoint 19 — The KL constraint.** What does it prevent? What is the
> measured KL between our SFT model and the base? Why is the median the right
> statistic here rather than the mean?

**Related:** [[phase5-code-explanation]] · [[interview-phase5-rlhf-dpo]] ·
[[phase3-sft-and-loss-masking]] · [[phase4-lora-svd-and-pca]]
