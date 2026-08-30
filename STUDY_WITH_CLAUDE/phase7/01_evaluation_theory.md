# Phase 7 — Evaluation: what metrics actually measure

**Status:** engineering complete, USER explain-back DEFERRED.

---

## 1. The question evaluation exists to answer

> *"How do you know the model actually got better?"*

Four phases of this project answered "we don't, yet" in four different ways.
Phase 7 turns those four failures into an evaluation subsystem that makes the
same mistakes hard to repeat.

---

## 2. The four lessons, and what each one cost

### Phase 3 — a loss is a mean over a token *population*

Measured on the base model, same forward pass:

| region | tokens counted | mean CE |
|---|---:|---:|
| prompt (masked out) | 25 | **6.3314** |
| completion (trained) | 7 | **3.6295** |
| everything | 32 | 5.7403 |

I predicted the prompt would be *easier*. It was **harder** — the base model
had never seen ChatML, so the templated prompt was unfamiliar.

> **The lesson is stronger than the prediction was.** An unmasked loss is not
> reliably higher *or* lower. It is a mean over a **different population of
> tokens**, so it is not comparable to a masked loss in either direction.

**Encoded as:** `PerplexityResult` carries its `region` and its `n_tokens`, and
`comparable_to()` returns False across regions **even when the numbers are
identical** — there is a test for exactly that.

### Phase 4 — perplexity did not reveal an unusable model

| model | completion ppl | emitted `<\|im_end\|>` |
|---|---:|---:|
| base | 8.541 | 0/4 |
| **full SFT** | **7.199** | **4/4** |
| LoRA r=16 | 7.449 | **0/4** |
| QLoRA r=16 | 7.511 | **0/4** |

Perplexity separated the arms by **3.5%**. Stop-token behaviour separated them
**completely**. All three PEFT models wrote fluent, on-topic answers and then
ran to the token cap forever.

The mechanism was structural: emitting a rare token means moving its *logit*,
which `lm_head` produces — and `lm_head` is tied to the embedding, so it was
excluded from LoRA's target set. Full fine-tuning moved that matrix by relative
**0.0136**, the largest relative change of any matrix measured in this project.

> **A likelihood metric cannot see a behaviour it does not score.**

**Encoded as:** `TerminationStats` separates *terminated* from *emitted a stop
token* (a model can stop by exhausting its budget), and the dashboard reports
stop rate beside perplexity in every comparison.

### Phase 5 — the same comparison, two answers

| comparison | prefers chosen |
|---|---:|
| **SUM** of log-probs (the published DPO objective) | **47.2%** |
| MEAN per token | **58.3%** |

Identical models. Identical data. An **11.1-point swing that straddles chance**.

**Encoded as:** `PreferenceStats` computes both, always, and its `to_dict()`
cannot emit one without the other or without the token counts.

### Phase 6 — the arithmetic underneath

| | chosen | rejected |
|---|---:|---:|
| mean summed log-prob | −291.12 | −261.65 |
| mean tokens | 271.7 | 242.2 |
| **per-token log-prob** | **−1.0713** | **−1.0803** |

**Per token, the chosen response is genuinely more likely.** The SUM comparison
inverts that verdict purely because chosen carries 29.5 more tokens:

```
SUM gap                      = −29.47 nats
explained by length alone    = −31.90 nats
residual once length removed =  +2.43 nats   ← chosen is BETTER
```

To flip the average pair under SUM, a policy must shift the gap by ~29 nats.
DPO achieved 0.04 at the pre-registered LR (**728× short**) and 0.70 at 10×
(**42× short**).

> **The metric was measuring length, not quality.**

**Encoded as:** `PreferenceStats.length_attribution()` performs exactly this
decomposition on any preference set, and the dashboard prints it for every
model.

---

## 3. Perplexity, precisely

```
PPL = exp( mean negative log-likelihood per token )
```

Three things must be stated or the number is meaningless:

1. **Which tokens.** Completion-only and full-sequence are different
   quantities. Phase 3 measured them differing by a factor of 1.7 in nats.
2. **Which denominator.** Token-weighted (total NLL ÷ total tokens), not a mean
   of per-example means — the latter weights a 5-token answer like a 500-token
   one.
3. **Which data.** Perplexity on a model's own training distribution measures
   *fit to that distribution*. Every number in this project is
   **in-distribution**, and that makes "better" partly circular.

`perplexity()` raises on a negative NLL, because that cannot arise from a
correct computation and almost always means a sign error.

---

## 4. Uncertainty: Wilson, not the textbook interval

The normal approximation `p ± z·√(p(1−p)/n)` produces bounds outside [0, 1] and
has poor coverage at small n — exactly where this project lives: 184 preference
pairs, 6 generation prompts, ~12 judge comparisons.

**Wilson score intervals** stay inside [0, 1] and behave at small n. Verified
against a known value: 50/100 gives **[0.4038, 0.5962]**.

The subsystem reports an interval for every rate and calls a difference
**"NOT resolvable at this sample size"** when the intervals overlap.
`difference_is_resolvable()` deliberately returns **no p-value** — it answers
one narrow question: is the gap larger than the uncertainty?

> Overlapping intervals do **not** prove no difference. This is a conservative
> test, and it is the honest alternative to inventing significance.

Two real cases, both asserted in tests: Phase 4's 0/4 vs 4/4 stop gap **is**
resolvable even at n=4; Phase 6's identical 86/184 preference rates are **not**.

---

## 5. LLM-as-judge, and why it is not ground truth

### Position bias is measured, not assumed away

Every pair is judged **twice**, in both presentation orders, and the swapped
verdict is mapped back to the original labels. Then:

- **consistent** → the judge picked the same answer both times → a real vote
- **inconsistent** → it picked by position → the verdict is position, not
  preference

A judge that always picks whichever answer it sees first therefore agrees with
itself on **no** pair and yields **zero** decided verdicts — not a spurious
100% win rate. There is a test using a scripted position-biased judge that
asserts exactly this.

`position_bias_rate` is a first-class reported number.

### What is excluded, and why

Ties and position-flips are **counted, not split as half-wins**. They carry no
preference information; folding them in as 0.5 manufactures precision.
Unparseable replies return `"unparsed"` rather than being guessed as ties.

### The judge's own limitations, recorded in the JSON

The judge is **Qwen2.5-7B-Instruct** — about 4.5× the models it judges, but:

- **small by frontier standards**;
- **from the same model family** as the models under test, a known bias risk —
  a judge may favour text resembling its own training distribution;
- **never validated against human preference** in this project.

Those limitations are attached to every `JudgeResult` object, not just written
in prose.

---

## 6. Why there is no single quality score

`evals/report.py` contains no aggregate score and no function that produces
one. A test asserts the module exposes nothing matching `"score"`.

The reason is empirical, not stylistic. In each of Phases 3–6, two metrics
disagreed, and **the disagreement was the finding**:

| phase | metric A said | metric B said |
|---|---|---|
| 3 | loss improved 33× | generation was destroyed |
| 4 | perplexity within 3.5% | stop behaviour 0/4 vs 4/4 |
| 5 | 47.2% preference | 58.3% preference |
| 6 | accuracy unchanged | reward margin moved, length +30% |

A weighted average would have erased all four.

---

## 7. Evaluation leakage

Every number in this project is **in-distribution**:

- SFT perplexity is measured on no_robots' test split — the dataset SFT trained
  on;
- preference accuracy on UltraFeedback's test split — the dataset DPO trained
  on.

Test *splits*, so there is no example-level leakage. But **distribution-level**
overlap remains, and it makes "better" partly circular: the model was optimised
toward exactly the distribution it is scored on.

The honest framing: these metrics measure **fit to the training distribution**,
which is what one epoch of that data should buy. They are **not** evidence of
general assistant quality, and Phase 7's dashboard says so in its own output.

---

## My Understanding

> **[USER CHECKPOINT — DEFERRED — USER EXPLAIN-BACK REQUIRED]**
>
> **Checkpoint 23 — Why isn't perplexity enough?**
> Give the Phase 4 numbers from memory and explain the mechanism. Then answer:
> what class of failure can a likelihood metric never reveal, and why?
>
> Without looking:
> - Two models report completion perplexity 7.20 and 7.45. What must you check
>   before saying the first is better?
> - Why is a mean of per-example means the wrong denominator?
> - Perplexity on the training distribution's test split — what does it measure,
>   and what does it not?

## Explain Back

> **[USER CHECKPOINT — DEFERRED — USER EXPLAIN-BACK REQUIRED]**
>
> **Checkpoint 24 — Design an evaluation for an aligned model.**
> Which metrics, on which data, with what uncertainty, and what would each one
> fail to catch? Justify every inclusion by naming the failure it detects.
>
> **Checkpoint 25 — LLM-as-judge.**
> Explain position bias and the two-order protocol. Why are ties and flips
> excluded rather than split? Give three reasons this project's judge is not
> ground truth.

**Related:** [[phase7-code-explanation]] · [[interview-phase7-evaluation]] ·
[[phase6-dpo-theory-and-results]] · [[phase4-lora-svd-and-pca]]
