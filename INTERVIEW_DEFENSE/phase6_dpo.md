# Interview Defence — Phase 6: DPO

Every number is measured in this repository. Where a result is null, it is
presented as null — a phase whose headline is "nothing changed" is more
defensible than one that finds an improvement it cannot support.

---

## Q1. "Derive the DPO loss."

Start from the KL-constrained RLHF objective:

```
max_π  E[ r(x,y) ] − β·KL( π(·|x) ‖ π_ref(·|x) )
```

Its optimum is the reference tilted by the exponentiated reward:

```
π*(y|x) = (1/Z(x))·π_ref(y|x)·exp( r(x,y)/β ),   Z(x) = Σ_y π_ref(y|x)exp(r/β)
```

Solve for the reward:

```
r(x,y) = β·log[ π*(y|x)/π_ref(y|x) ] + β·log Z(x)
```

Substitute into Bradley-Terry, `P(y_w ≻ y_l) = σ(r_w − r_l)`. **`log Z(x)`
depends only on `x`, and both responses share `x`, so it cancels:**

```
L = −log σ( β·[log π_θ(y_w|x) − log π_ref(y_w|x)]
          − β·[log π_θ(y_l|x) − log π_ref(y_l|x)] )
```

**The step to name:** the cancellation *requires* both responses to answer the
same prompt. I verified that on the actual data — 3000/3000 rows — precisely
because the derivation depends on it.

**Follow-up: "how do you know your implementation is right?"** I checked it
against the paper's own Appendix B reference code. Mine groups the terms as
per-response implicit rewards then their difference; theirs as log-ratios then
their difference. Algebraically identical, and I assert it **numerically to
1e-12** at β ∈ {0.01, 0.1, 0.5, 1.0}. My grouping was written before I read the
appendix, so it is a genuine independent check.

---

## Q2. "Why is the loss exactly `log 2` at initialisation?"

Because policy and reference start as the *same* checkpoint, so both implicit
rewards are exactly 0, the sigmoid argument is 0, and `−log σ(0) = −log 0.5 =
log 2 = 0.6931471805599453`.

**This is the most valuable free check in the whole pipeline.** It catches a
reference pointed at the wrong checkpoint, and a sum/mean mismatch between the
two branches — bugs that otherwise train happily while optimising the wrong
objective.

My trainer *refuses to start* if it fails. Measured on the real run:
`max_abs_chosen_reward = 0.0`, `loss = 0.6931471824645996`.

**And it is structural, not numerical** — it holds for any log-probabilities,
however large, so it is a reliable check rather than a coincidence. A reference
off by 0.1 nats on a single sequence is detected immediately.

---

## Q3. "What does β do?"

β multiplies the log-ratio difference *inside* the sigmoid, setting how large a
gap is needed to saturate the loss.

**Measured**, log-ratio difference fixed at 4 nats so the logit is exactly `4β`:

| β | logit | loss | dL/dlogit |
|---:|---:|---:|---:|
| 0.001 | 0.0040 | 0.691149 | −0.499000 |
| 0.1 | 0.4000 | 0.513015 | −0.401312 |
| 0.5 | 2.0000 | 0.126928 | −0.119203 |
| 2.0 | 8.0000 | 0.000335 | −0.000335 |

The subtlety worth having ready: **small β keeps the gradient w.r.t. the
*logit* near its maximum (−0.5), yet moves the policy less** — because β also
multiplies the *parameter* gradient. Large β saturates and the gradient
vanishes. β trades *how far the policy may move* against *how much signal
survives once it has*.

**My sweep confirms the ordering.** Reward margin was monotone in β:
−0.0001 (β=0.01), +0.0041 (β=0.1), +0.0090 (β=0.5).

---

## Q4. "How did you choose β? And how do I know you didn't pick it afterwards?"

**I pre-registered it.** `docs/phase6/BETA_PREREGISTRATION.md` was committed at
13:15:12, before any DPO code existed — the git history is the evidence. It
fixed β ∈ {0.01, 0.1, 0.5}, the primary metric, the evaluation split, and five
hypotheses, and stated in advance what would count as a negative result.

Only afterwards did I read the paper's Appendix B, which uses **β = 0.1 by
default and β = 0.5 for TL;DR summarization**. Both are in my set. That is
corroboration, and I explicitly did *not* change the sweep — it was already
running.

---

## Q5. "What did DPO do to your model?"

**Essentially nothing, and I can tell you exactly why.**

| model | SUM (primary) | MEAN (diag) | margin | KL median | gen len |
|---|---:|---:|---:|---:|---:|
| SFT baseline | **46.7%** | 58.7% | 0.0000 | 0.0000 | 93.8 |
| β=0.01 | 46.7% | 58.7% | −0.0001 | 0.0008 | 95.8 |
| β=0.1 | 46.7% | 58.7% | +0.0041 | 0.0008 | 96.0 |
| β=0.5 | 46.7% | 58.7% | +0.0090 | 0.0008 | 95.8 |

Preference accuracy did not change by a **single pair out of 184** in any arm.

**The direction was right** — β=0.1 raised `log π(chosen)` and lowered
`log π(rejected)`. **The magnitude was the problem**, and the decisive number is
the KL: **0.0008** from the reference, against **0.2044** for Phase 5's
SFT-from-base. The policy moved roughly **250× less** than one epoch of SFT
did. On ~270-token sequences a ±0.02-nat shift cannot flip a ranking.

**So this is a training-budget result, not evidence DPO fails**: 116 steps × 16
pairs = 1,856 pairs at lr 5e-7, against the paper's batch-64 RMSprop at 1e-6.

**What I would not say:** "DPO didn't work." What I *can* say is that at this
budget the policy did not move enough to change the metric, and I have the KL
measurement that distinguishes those two claims.

---

## Q6. "Which objective did you optimise — sum or mean log-probability?"

**SUM**, the published objective, and I pre-registered that choice.

It matters more than it sounds. Phase 5 measured the pre-DPO baseline at
**47.2% under SUM and 58.3% under MEAN** — identical models, identical data, an
11-point swing that straddles chance. The cause is length: UltraFeedback's
chosen responses are **56.5% longer in tokens**, and a summed log-probability
is more negative for a longer sequence *simply for being longer*.

So the two are **different objectives sharing a name**. Reporting the MEAN
figure as a published-objective baseline would understate any improvement. My
code computes both every time and labels MEAN a diagnostic.

---

## Q7. "Phase 5 said length would be a trap. Was it?"

I predicted (H4, pre-registered) that response length would **increase** — the
SUM objective is length-sensitive and chosen responses are longer, so "be
longer" is a gradient direction DPO can exploit.

**H4 was disproved.** Generated length went 93.8 → ~95.8 tokens: **+2%, ratio
1.02×.** Stop-token behaviour is intact (4/4 emit `<|im_end|>` in every arm) and
qualitative outputs are near-identical.

That is a *consistent* null, not a lucky one: the model barely moved at all, so
it had no opportunity to exploit the length direction. **The trap is still
real** — it just needs a run with enough budget to spring it. Which is exactly
what I would watch for next.

---

## Q8. "What is the reference model for, and how do you keep it honest?"

It defines the implicit reward: `β·(log π_θ − log π_ref)` is DPO's notion of
how good a response is. Without it, maximising `log π_θ(y_w)` would just push
mass onto chosen strings with no counterweight.

**Three things I enforce, all tested:**
1. **Zero trainable parameters.** A trainable reference does not crash — it
   *drifts toward the policy*, the implicit rewards shrink, and the run
   optimises progressively less.
2. **`eval()` mode.** Dropout would make its log-probabilities stochastic and
   the rewards noisy.
3. **Zero implicit reward at init** (Q2).

Memory-wise a frozen model costs its **weights only** — no gradients, no
optimizer state. Measured: policy side 11.50 GiB vs reference 2.88 GiB, a 4×
ratio. Peak VRAM 18.56 GiB, against ~14.4 GiB for the Phase 3 full SFT.

---

## Q9. "How does the preference data enter, and what did you have to clean?"

As `(x, y_w, y_l)` triples. Four log-probabilities per pair — two models × two
responses.

**What the Phase 5 audit found and what I did about it:**

| finding | measured | treatment |
|---|---|---|
| score **ties** | 11.9% | **kept** (`drop_ties=False`), recorded as an explicit decision |
| identical chosen/rejected | 14 in 3,000 | **filtered** — log-ratio is 0 by construction |
| empty responses | 2 in 3,000 | **filtered** |
| length imbalance | chosen 56.5% longer | **cannot be filtered** — measured and carried forward |
| over-length pairs | 131/2000 train | **dropped, not truncated** |

**Why dropped, not truncated:** truncation removes tail tokens, which for a
prompt/completion pair is exactly what DPO scores — and it removes *different
amounts* from chosen and rejected, making their log-probabilities incomparable.

Ties are the interesting one: Bradley-Terry assumes a genuine ordering, and a
tie says the annotator saw none. Dropping 11.9% is defensible, which is exactly
why it must be a recorded decision rather than a silent default.

---

## Q10 (challenge). "Your headline result is that nothing happened. Why should I be impressed?"

Because the null is **characterised**, not just reported.

I can tell you the policy moved 250× less than SFT did (KL 0.0008 vs 0.2044),
that the direction was correct and monotone in β, that length did *not* blow
up, that stop-token behaviour survived, and that both safety checks passed on
every run. That is a diagnosis, not a shrug.

I also pre-registered what would count as a negative result *before* running,
so I could not reinterpret it afterwards — and the outcome landed in a category
the pre-registration had already named.

**What would make me doubt myself:** if the loss had fallen sharply while
accuracy stayed flat, I would suspect the metric. It didn't — the loss barely
moved either, and the KL confirms the policy barely moved. The story is
internally consistent.

**The honest next step** is more budget, not a different algorithm. I ran one
post-hoc diagnostic at 10× the learning rate specifically to test that, and
labelled it as post-hoc so it cannot masquerade as a pre-registered result.

---

## Q11 (challenge). "What would you do differently?"

1. **More budget.** 1,856 pairs is small. The paper uses batch 64 and far more
   steps. This is the single biggest limitation.
2. **A learning-rate sweep**, pre-registered alongside β. I swept β and held LR
   fixed at a conventional value — and LR turned out to be the binding
   constraint.
3. **Log window averages, not the last pair.** My per-step history records one
   pair, so its margin is noise. The eval-set numbers over 184 pairs carry the
   result.
4. **Multiple seeds.** One run per arm; I cannot separate a small effect from
   noise.
5. **Read IPO/KTO/ORPO** — they exist substantially to address the tie rate and
   length bias I measured, and I have not read them.

---

## Explain Back

> **[USER CHECKPOINT — DEFERRED — USER EXPLAIN-BACK REQUIRED]**
>
> Answer Q1, Q3, Q5 and Q10 aloud without notes. Derive the loss on a
> whiteboard and name the step requiring a shared prompt. Then argue both
> sides of Q10 — that the null is a real finding, and that it is merely an
> underpowered experiment — and say which single measurement decides it.

**Related:** [[phase6-dpo-theory-and-results]] · [[phase6-code-explanation]] ·
[[interview-phase5-rlhf-dpo]] · [[pytorch-dpo-mechanics]]
