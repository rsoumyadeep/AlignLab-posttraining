# Interview Defence — Phase 5: RLHF, PPO, and DPO vs PPO

Every number below is measured in this repository and traceable to a script.
Where material is conceptual it says so — claiming a PPO implementation this
project does not have would be the fastest way to lose an interview.

**Honesty boundary, stated once:** AlignLab has **no PPO implementation and no
trained reward model**. It has preference-data infrastructure, log-probability
and KL arithmetic, and measurements on real models. Say that plainly if asked.

---

## Q1 (priority). "Why DPO instead of PPO?"

**Short answer.** DPO reaches the same optimum with a classification loss
instead of an RL loop, removing three of the four models and the entire
generation loop. For a fixed preference dataset and a small team, that is a
large reduction in cost and in ways to be wrong — and it is why I chose it.

**The substance.** RLHF-with-PPO needs four models:

| model | trained? | what it's for |
|---|---|---|
| policy `π_θ` | **yes** | the thing being aligned |
| reference `π_ref` | no | anchors the KL penalty |
| reward `r_φ` | no (separately trained) | scores generations |
| value `V_ψ` | **yes** | advantage estimation |

DPO needs **two**: policy and frozen reference. It also removes:

- the **on-policy generation loop** — PPO must sample fresh completions every
  update, which is slow and a second surface for bugs (sampling settings,
  truncation, EOS handling — the last of which Phase 4 showed is easy to get
  wrong);
- the **reward model** as a separate artifact to train, validate, version;
- PPO's hyperparameters — clip `ε`, GAE `λ`, value coefficient, epochs/batch.

**The mechanism.** The RLHF objective has a closed-form optimum
`π* ∝ π_ref · exp(r/β)`. Solve for `r`, substitute into Bradley-Terry, and
`log Z(x)` cancels because both responses share the prompt. What remains is:

```
L_DPO = −log σ( β·[log π_θ(y_w|x) − log π_ref(y_w|x)]
              − β·[log π_θ(y_l|x) − log π_ref(y_l|x)] )
```

**Notice the shape**: it is the *same* Bradley-Terry loss a reward model is
trained with, with the reward difference replaced by a log-ratio difference.
The reward model has been reparameterised into the policy — the paper's
subtitle is *"Your Language Model is Secretly a Reward Model"*.

---

## Q2. "What does PPO actually optimize?"

Per token, maximise reward minus a KL penalty against the frozen reference:

```
E[ r_φ(x,y) − β · KL( π_θ(·|x) ‖ π_ref(·|x) ) ]
```

via the clipped surrogate:

```
ratio  = π_θ(a|s) / π_θ_old(a|s)
L_clip = E[ min( ratio·Â , clip(ratio, 1−ε, 1+ε)·Â ) ]
```

The clip is what makes it *proximal*: past `1±ε` the objective stops improving,
so there is no incentive to move far in a single update.

**CONCEPTUAL — NOT IMPLEMENTED here.**

---

## Q3. "Why does PPO need a reward model at all?"

Because RL needs a scalar reward for **arbitrary** generations, and human
preferences only exist for the *pairs someone labelled*. PPO generates new
completions every step — nobody has labelled those. The reward model is a
learned stand-in that generalises the preference labels to unseen text.

**This is also PPO's central weakness.** The policy optimises `r_φ`, not human
preference. `r_φ` is an imperfect proxy and the optimiser is very good at
finding its flaws — reward hacking. The KL penalty exists largely to slow that
down.

DPO sidesteps it by never generating: it only ever scores completions that
*came with labels*. That is a genuine advantage and also exactly what it gives
up (Q8).

---

## Q4. "What is the reference model for, in each?"

**PPO:** the anchor for the KL penalty — the "don't drift" target.

**DPO:** it defines the *implicit reward*. `β·(log π_θ − log π_ref)` is DPO's
notion of how good a response is. Without `π_ref`, maximising `log π_θ(y_w)`
would just push probability mass onto chosen strings with no counterweight.

**Both are the frozen SFT checkpoint** in standard practice.

**A check worth quoting.** With `policy == reference`, DPO's implicit reward
must be **exactly zero** for every sequence. I measured `0.000e+00` (E22). It
costs nothing and catches a reference plumbed to the wrong checkpoint — a bug
that otherwise shows up as a run that trains happily while optimising the wrong
objective.

---

## Q5. "Why is KL control needed, and which direction?"

**Needed** because maximising a *learned* reward without constraint drives the
policy toward whatever degenerate text the reward model happens to score
highly. The constraint keeps it near a model that already produces sensible
language.

**Direction:** `KL(policy ‖ reference)`, not the reverse. That direction is
large when the policy puts mass where the reference puts none — precisely what
reward hacking looks like in distribution space. The reverse would not penalise
it. KL is not symmetric, and my tests assert that rather than assume it.

**A number, so this isn't hand-waving.** I measured KL between our Phase 3 SFT
model and the base (E22):

| policy | mean | **median** | max |
|---|---:|---:|---:|
| full SFT | 0.6547 | **0.2044** | 7.3349 |
| LoRA | 0.6233 | 0.2008 | — |
| QLoRA | 0.5564 | 0.1589 | — |

Quote the **median**: the distribution is heavily right-skewed (mean ≈ 3×
median), so one epoch of SFT moved a typical token's distribution by about
**0.20 nats**. That is the scale a `β` would be chosen against.

**Follow-up I'd expect: "how is KL computed in practice?"** Not exactly. Exact
KL needs the full vocabulary at every position — `[B, T, 151936]` for this
model, too large for a PPO batch. Implementations use a sampled estimator,
usually `k3 = exp(−k1) − 1 + k1` where `k1 = log π − log π_ref`: biased low,
much lower variance, always non-negative. So "PPO penalises KL" and "PPO
penalises an *estimator* of KL with different properties" are different
statements, and the second is the true one.

---

## Q6. "How does preference data enter DPO?"

As `(x, y_w, y_l)` triples. For each, four log-probabilities are computed — two
models × two responses — and combined into the loss.

**Three things that must hold, which I checked rather than assumed:**

1. **Both responses share the prompt.** Otherwise the comparison is
   meaningless — and `log Z(x)` only cancels because `x` is shared. Measured:
   3000/3000.
2. **Prompt tokens are a genuine token-prefix of both full sequences.** DPO
   scores completion tokens only. Phase 3 found 1.5% of no_robots rows
   violating this because BPE merged the template's trailing newline into a
   completion starting with whitespace. Measured here after stripping: **0
   violations in 400 sampled pairs, on both responses**.
3. **Only completion tokens are scored.** Prompt tokens are shared, so they
   cancel in the *difference* — but not in a raw log-probability, a perplexity,
   or a KL.

---

## Q7. "What does DPO assume?"

1. **Bradley-Terry is the right preference model** — a consistent, transitive
   ordering. **11.9% of UltraFeedback pairs are score ties**, where the
   annotator saw no difference yet the pair still says "prefer `y_w`".
2. **The closed-form optimum is reachable** by optimising the reparameterised
   objective — exact in theory, approximate under finite data and SGD.
3. **The preference data covers what matters.** DPO never generates, so it
   never sees its own current failures.
4. **The reference is a sensible anchor** — normally the SFT model.

---

## Q8. "What does DPO give up?"

- **No reusable reward model.** PPO's `r_φ` can score arbitrary completions,
  drive best-of-n or rejection sampling, and serve as an evaluator/monitor.
  DPO produces no such artifact.
- **Off-policy only.** It optimises against a fixed dataset and gets no
  feedback on its own outputs. PPO's generation loop is a cost *and* a
  capability.
- **No continued improvement** beyond what the preference set covers.
- **`β` still matters.** DPO does not remove the KL trade-off; it absorbs it
  into one coefficient.

---

## Q9. "When is PPO still the right call?"

- You want the **reward model as a deliverable** (best-of-n, rejection
  sampling, monitoring, evaluation).
- The reward is **programmatic** — unit tests pass, a verifier accepts, a
  compiler succeeds. There are no preference pairs, so DPO does not apply.
- You need **on-policy** correction of the model's *own* live failure modes,
  which a static dataset cannot supply.
- You have the engineering capacity and want headroom past what a fixed
  preference set covers.

---

## Q10 (challenge). "Give me a measurement that would change how you read a DPO result."

Yes — I have one, and it is the most useful thing I found in Phase 5.

**Before any DPO, does our SFT model already prefer the chosen response?**

| comparison | prefers chosen |
|---|---:|
| **SUM of log-probs** — the published objective | **51/108 = 47.2%** |
| MEAN per token — length-normalised | **63/108 = 58.3%** |

Identical models, identical data, **11.1-point swing**, and the two straddle
chance.

**The cause is length, not quality.** Chosen responses are 56.5% longer in
tokens, and a summed log-probability is more negative for a longer sequence
*simply for being longer*. The SUM comparison is biased **against** chosen,
hard enough to push it below 50%.

**Why this changes the reading:**

1. The baseline is **47.2%**, not 58.3%, if you train the published objective.
   Quoting the length-normalised number as your baseline would **understate**
   DPO's improvement.
2. "DPO raised preference accuracy from 47% to X%" needs the length caveat —
   part of any gain may be the model learning to be **longer**, not better.
3. You must **say which variant you optimised**. They are different objectives
   sharing a name.

This is the same lesson as Phase 3's masked-loss finding and Phase 4's
stop-token finding: **a single headline number can be dominated by a property
of the data rather than of the model.**

---

## Q11 (challenge). "You haven't implemented PPO. Isn't your comparison hollow?"

Partly, and I'd rather say so than overclaim.

**What I can defend from measurement:** the KL constraint's scale on real
models; that the reference wiring produces an exactly-zero implicit reward at
init; the preference data's tie rate, length bias, and prefix consistency; and
the pre-DPO baseline including its length confound.

**What is conceptual only:** PPO's objective, the value model, the reward
model, reward hacking, and the four-model memory argument. I have not measured
PPO's instability — I have read the DPO abstract's claim that RLHF is "complex
and often unstable" and I have not verified it.

**What I would do to close the gap:** train a small reward model on this same
preference data — the Bradley-Terry loss is already implemented and tested —
and measure its accuracy and calibration. That is the cheapest step toward a
non-hollow comparison, and it stops short of a full PPO loop, which is not
warranted here.

---

## Explain Back

> **[USER CHECKPOINT — DEFERRED — USER EXPLAIN-BACK REQUIRED]**
>
> Answer Q1, Q3, Q5 and Q10 aloud without notes. Then derive the DPO loss from
> the RLHF optimum on a whiteboard, saying at which step the shared prompt is
> required. Finally, write two questions this document does not answer, and
> answer them.

**Related:** [[phase5-rlhf-ppo-and-dpo]] · [[phase5-code-explanation]] ·
[[interview-phase4-peft]] · [[interview-phase3-sft]]
