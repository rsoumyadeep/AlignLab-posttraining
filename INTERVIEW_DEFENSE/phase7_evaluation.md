# Interview Defence — Phase 7: Evaluation

Every number is measured in this repository. The strongest material here is not
a good result — it is four cases where a metric said one thing and the model
was doing another.

---

## Q1. "How do you know your fine-tuned model got better?"

**I mostly know it didn't — on the metrics I have — and I can tell you exactly
what each metric can and cannot see.**

That is the honest answer, and it is more defensible than a win rate.

What I measured for SFT against the base model:

| | base | SFT |
|---|---:|---:|
| completion perplexity | 8.541 | **7.199** |
| emitted `<\|im_end\|>` | 0/4 | **4/4** |
| generation | degenerate repetition (`-unstyled` ×N), answered an English prompt in Chinese | coherent, correctly formatted, terminates |

**The perplexity gain is 15.7% and it is the weaker evidence.** It measures fit
to no_robots' own test split — the distribution SFT trained on. The *strong*
evidence is behavioural: the model learned to stop, which is binary,
user-visible, and invisible to perplexity.

**What I will not say:** "SFT made a better assistant." I have no
out-of-distribution evaluation, so that claim is not supported.

---

## Q2. "Why isn't perplexity enough?"

I have a measured case, not an argument.

Phase 4, four models on the same held-out data:

| model | completion ppl | emitted `<\|im_end\|>` |
|---|---:|---:|
| full SFT | 7.199 | **4/4** |
| LoRA r=16 | 7.449 | **0/4** |
| QLoRA r=16 | 7.511 | **0/4** |

Perplexity separated them by **3.5%**. Stop-token behaviour separated them
**completely**. The PEFT models wrote fluent, on-topic answers and then ran to
the token cap forever — unusable, and perplexity said "fine".

**The mechanism, which is the part worth knowing:** emitting a rare token means
moving that token's *logit*, produced by `lm_head` — which is tied to the
embedding and therefore excluded from LoRA's target set. Full fine-tuning moved
that matrix by relative 0.0136, the largest relative change of any matrix in the
project. LoRA structurally could not reach it.

**Generalised:** a likelihood metric scores the tokens in your dataset. It
cannot see a behaviour those tokens do not exercise. Termination, refusal,
format compliance and tool-call validity all fall in that gap.

---

## Q3. "Why can preference accuracy be misleading?"

Because I watched it invert.

Same models, same data, two ways of aggregating token log-probabilities:

| | prefers chosen |
|---|---:|
| **SUM** (the published DPO objective) | **47.2%** |
| MEAN per token | **58.3%** |

An 11-point swing straddling chance. Then the arithmetic:

| | chosen | rejected |
|---|---:|---:|
| summed log-prob | −291.12 | −261.65 |
| tokens | 271.7 | 242.2 |
| **per-token** | **−1.0713** | **−1.0803** |

**Per token, chosen is genuinely more likely.** The SUM comparison inverts that
purely because chosen carries 29.5 more tokens:

```
SUM gap                      = −29.47 nats
explained by length alone    = −31.90 nats
residual once length removed =  +2.43 nats   ← chosen is BETTER
```

**The metric was measuring length.** And it is ~29 nats deep, so DPO would need
to shift the gap by 29 nats to flip the average pair — it managed 0.04 at the
pre-registered LR and 0.70 at 10×.

**What I do about it:** my `PreferenceStats` object cannot serialise SUM without
MEAN, or either without token counts, and `length_attribution()` runs that
decomposition automatically. The failure mode is designed out, not remembered.

---

## Q4. "How would you evaluate an aligned model?"

Four layers, each justified by a failure it catches:

1. **Likelihood** — completion-only *and* full-sequence perplexity, each
   labelled with its token region. *Catches:* distribution fit. *Misses:*
   everything behavioural (Q2).
2. **Behavioural** — termination, stop-token emission, empty output, repetition
   (`distinct-2`), format checks. *Catches:* unusable models that score well.
   *Misses:* whether the content is correct.
3. **Preference** — SUM and MEAN, always with lengths. *Catches:* ranking
   ability. *Misses:* anything the preference data doesn't cover, and it is
   length-confounded (Q3).
4. **Judged** — pairwise LLM-as-judge with position randomisation. *Catches:*
   holistic quality differences. *Misses:* ground truth (Q5).

**Plus, on every layer:** a Wilson interval, and an explicit "NOT resolvable at
this sample size" when intervals overlap.

**And no aggregate score.** In each of Phases 3–6 two metrics disagreed and the
disagreement was the finding; a weighted average would have erased all four. My
report module contains no function that produces one, and a test asserts it.

---

## Q5. "What are the limitations of LLM-as-judge?"

Five, and I handle the first structurally.

**1. Position bias.** A judge tends to favour whichever answer it sees first.
So I judge every pair **twice**, in both orders, and map the swapped verdict
back to the original labels:

- same answer both times → a real vote
- different → the verdict was *position*, not preference → **excluded**

A judge that always picks the first answer therefore agrees with itself on no
pair and produces **zero decided verdicts**, not a spurious 100% win rate.
That's asserted by a test with a scripted position-biased judge, and
`position_bias_rate` is reported as a first-class number.

**2. It is not ground truth.** My judge has never been validated against human
preference in this project. No agreement study was run.

**3. Family bias.** My judge is Qwen2.5-7B-Instruct — the **same family** as the
models under test. A judge may favour text resembling its own training
distribution. This is recorded in every result's JSON, not just in prose.

**4. Capability.** 7B is ~4.5× the models it judges, but small by frontier
standards. It may simply be unable to tell two mediocre answers apart.

**5. Manufactured precision.** Ties and position-flips carry no preference
information. I count them separately rather than splitting them as half-wins —
halves would inflate the sample and narrow the interval dishonestly.

**Follow-up I'd expect: "what would make it trustworthy?"** A human agreement
study on a subset, a judge from a different family, and a sample size where the
interval actually excludes 0.5. I have none of the three.

---

## Q6. "How do you avoid evaluation leakage?"

**I don't fully, and I say so.**

There is no *example-level* leakage — everything is measured on held-out test
splits. But there is **distribution-level** overlap: SFT perplexity is measured
on no_robots' test split, and no_robots is what SFT trained on; preference
accuracy is on UltraFeedback's test split, which is what DPO trained on.

So the model was optimised toward exactly the distribution it is scored on.
"Better" is partly circular, and my dashboard's own output says these numbers
measure **fit to the training distribution**, not general quality.

**What proper mitigation looks like:** a held-out set from a different source,
prompts written after the training data was frozen, and a task the training
distribution doesn't cover. That is the top item on my Phase 8 list.

**The leakage I did control:** dataset fingerprints. Every run records a sha256
of the exact rows, so I can prove the evaluation set was not the training set —
and Phase 3 caught a real bug that way, when a stale `datasets` cache silently
served pre-fix rows and the fingerprint changing is what proved the fix landed.

---

## Q7. "How would you compare two fine-tuned models fairly?"

**Hold everything fixed except the thing under test, and say plainly what you
couldn't hold fixed.**

What I fix: base model + revision (pinned SHA), dataset + fingerprint,
objective, `max_length`, batch geometry, epochs, seed, evaluation set, decoding
(greedy — sampling adds a second source of variation to a comparison trying to
isolate one), and hardware.

**What I could not fix, in Phase 4:** the learning rate. LoRA conventionally
needs a higher LR than full fine-tuning because its adapters start at zero.
Holding it fixed measures LR sensitivity; changing it breaks the control.

**So I ran both** — LoRA at 2e-5 (matched) and 2e-4 (conventional) — and
reported both. That mattered: the LR effect was 0.032 nats against a
LoRA-vs-full-FT gap of 0.051, so reporting only the matched arm would have
**overstated LoRA's cost by about half**.

**And then uncertainty.** Every rate gets a Wilson interval. Two real verdicts
from my own data:

- Phase 4's 0/4 vs 4/4 stop gap: **resolvable** even at n=4.
- Phase 6's identical 86/184 preference rates: **not resolvable** — and the
  report says exactly that rather than reporting "+0.0%".

---

## Q8 (challenge). "Your evaluation found almost nothing improved. Isn't that a failed project?"

No — it is a working measurement apparatus reporting honestly.

The project *did* find improvements: SFT cut completion perplexity 15.7% and
took stop-token emission from 0/4 to 4/4, which is the difference between an
unusable model and a usable one. What it did not find is DPO improving a
length-dominated metric at an under-powered budget — and I can quantify both
the budget shortfall (KL 0.0008 vs SFT's 0.2044, ~250×) and the metric's
length dependence (~29 nats).

**The alternative would have been worse.** With one arbitrary quality score I
would have reported four "improvements" and missed: a 33× loss drop that was
causal leakage, three models that never stop, an 11-point swing from
aggregation choice, and a metric measuring token count.

**What would change my assessment:** an out-of-distribution evaluation showing
the gains are real, or a properly-powered DPO run. Both are Phase 8 work, and
neither is claimed now.

---

## Q9 (challenge). "What's the weakest part of your evaluation?"

1. **Everything is in-distribution.** The single biggest gap (Q6).
2. **Sample sizes are small** — 184 preference pairs, 6 generation prompts, ~12
   judge comparisons. Most differences are genuinely unresolvable, and the
   report says so rather than hiding it.
3. **The judge is unvalidated and same-family** (Q5).
4. **One seed everywhere.** No variance estimate across runs.
5. **No task-specific correctness checks** — no unit tests for code answers, no
   factuality checking.
6. **`distinct-2` is a crude repetition proxy**, not a fluency measure.
7. **Greedy decoding only.** Real deployments sample, and behaviour can differ.

---

## Explain Back

> **[USER CHECKPOINT — DEFERRED — USER EXPLAIN-BACK REQUIRED]**
>
> Answer Q1, Q2, Q3 and Q5 aloud without notes. Then take Q8 and argue both
> sides: that a project whose evaluation finds mostly nulls is failing, and that
> it is working correctly. Say which evidence decides it.

**Related:** [[phase7-evaluation-theory]] · [[phase7-code-explanation]] ·
[[interview-phase6-dpo]] · [[interview-phase4-peft]]
