# Interview Defence — The Whole Project

The per-phase files defend individual decisions. This one defends **AlignLab as
a body of work**, and it exists because the hardest questions are not about
LoRA or DPO — they are about what a project full of null results is worth.

---

## Q1. "In one minute: what is AlignLab and what did you find?"

A complete post-training pipeline for Qwen2.5-1.5B — SFT, LoRA, QLoRA, DPO and
an evaluation subsystem — built from first principles and verified against the
reference libraries.

**What I found is mostly what didn't happen.** SFT worked and is the one large
behavioural win: stop-token emission **0/6 → 6/6**, perplexity **−16.2%**. After
that, LoRA and QLoRA cut trainable parameters to **0.28%** and never learned to
stop. DPO changed **nothing** measurable under a preregistered budget. And my
own LLM judge couldn't confirm the SFT result it was pointed at.

The interesting part is the diagnosis in each case, and that **five stated
hypotheses were disproved** — including the standard explanation of why LoRA
works.

---

## Q2. "Why start from Qwen2.5-1.5B base rather than Instruct?"

Because the headline result would otherwise have been pre-baked.

An Instruct model has already been through somebody else's SFT and alignment. If
I started there, "the model learned to emit `<|im_end|>`" would have been true
before I ran anything, and I'd be measuring my fine-tuning on top of an unknown
one — no clean baseline, no attributable delta.

Starting from base is what makes **0/6 → 6/6** a measurement. The base model
genuinely cannot do it: it produces `.DrawString(...)` loops, `-unstyled`
repeated 128 times, and answers an English cooking prompt in Chinese.

**1.5B specifically** because the binding constraint was disk on a shared,
100%-full volume, not ideology — and because it fits full fine-tuning in
~16.8 GiB, which meant I could compare full FT against PEFT rather than assume
PEFT was necessary.

---

## Q3 (challenge). "Your project's headline results are mostly null. What did you actually prove?"

I'd push back on "prove" — but here is what the evidence establishes:

1. **SFT substantially changed termination behaviour.** 0/6 → 6/6, the only
   comparison of 24 whose Wilson intervals are disjoint.
2. **PEFT's cost is small and its ceiling is real.** 0.28% of parameters, 3.6%
   perplexity, and complete failure at a behaviour that mattered.
3. **A widely-repeated explanation of LoRA is not supported by the update I
   measured** (E17).
4. **The SUM preference metric is dominated by length in this dataset** —
   demonstrated on five models including an untrained one.
5. **DPO at 250× under-budget KL changes nothing**, and I can quantify both the
   shortfall and the ~29-nat gap it would have had to close.

What I did **not** prove: that any model is better in general. Every number is
in-distribution.

**And the null results are load-bearing, not consolation.** A pipeline that
produces four confident improvements is easy to build if you don't look
carefully. I looked carefully and found that in every phase from 3 to 6, two
metrics disagreed — and each disagreement was the real finding.

---

## Q4. "What's the single most interesting thing you found?"

**That the SUM preference metric inverts its own verdict, on every model,
including the untrained base.**

Per token, chosen responses are genuinely more likely — `−1.0713` vs `−1.0803`
for SFT. But chosen responses carry **29.5 more tokens**, and the published DPO
objective compares *summed* log-probabilities. So:

```
SUM gap                      = -29.47 nats
explained by length alone    = -31.90 nats
residual once length removed =  +2.43 nats   <- chosen is BETTER
```

Positive residual in all five rows. **The base model has had no preference
training at all and shows the same inversion**, which is what proves it's the
metric and not the training.

The consequence for Phase 6 is concrete: to flip the average pair under SUM, a
policy must move the gap ~29 nats. My run moved it **0.04**.

---

## Q5 (challenge). "Isn't 'the metric was wrong' just an excuse for DPO not working?"

It would be, if I'd said it *after* the run. Three things stop that:

1. **The measurement predates the result.** Phase 5's E22 found the SFT model
   preferred chosen only **47.2%** by SUM — below chance — *before* any DPO code
   existed. Phase 6 was entered knowing the metric started underwater.
2. **β was pre-registered**, committed at 13:15:12 before the DPO
   implementation, along with the criteria for declaring a negative result. I
   could not have picked β to flatter the outcome.
3. **The post-hoc run is labelled POST-HOC** and is not allowed to replace the
   preregistered result.

And I don't claim the metric explains everything. The KL shortfall is a separate,
independent problem: **0.0008 against SFT's 0.2044**. The policy barely moved.
Both are true, and neither is "DPO doesn't work."

---

## Q6. "What would you have done differently?"

Three things, in order of how much they'd have changed the project:

1. **Held out an out-of-distribution evaluation set from day one.** Every number
   I have measures fit to a training distribution. This is the single biggest
   structural weakness and it was baked in early.
2. **Budgeted DPO by KL rather than by epochs.** I fixed the training length and
   discovered afterwards that the policy had moved 250× less than SFT did. A KL
   target would have surfaced that on the first run instead of the last.
3. **Run the MLP-target LoRA arm.** It's 18 minutes and it converts my best
   hypothesis (E20 reachability) into a result. I deferred it and it's still
   deferred — which is the honest state, but it's the thing I'd most want back.

What I would **not** change: building the evaluation subsystem to refuse an
aggregate score. That decision caught four disagreements that a single quality
number would have averaged away.

---

## Q7. "How do I know your numbers are real?"

Every run writes a manifest with the git SHA **and dirty flag**, config hash,
dataset and eval fingerprints (sha256 over the actual rows), hardware, library
versions, dtype and seed. Evidence files — dashboards, run logs, summaries —
are committed.

And it's checked mechanically, not asserted:

```bash
python -m alignlab.provenance     # exit 1 if any applicable field is missing
```

Run across all 18 real artefacts, it reports **one** gap: `eval-full-001`'s
per-model `dataset_fingerprint`, a bug I found and fixed, with the post-fix run
clean. **I did not back-fill it** — reconstructing provenance after the fact is
worse than lacking it.

**Where I'd expect scepticism, and where you'd be right:** single seeds
everywhere. I have sampling uncertainty on every rate, and **no training
variance at all**. The 3.6% PEFT perplexity gap has no error bar across runs.

---

## Q8 (challenge). "You wrote your own LoRA and DPO. Why not just use the libraries?"

For the pipeline I *did* use them — TRL for SFT, `peft` for the PEFT arms. The
from-scratch implementations exist to be **checked against** those libraries,
and the checking found things:

- Our LoRA vs `peft`: agrees in fp32 — and that comparison is what exposed that
  **merging in bf16 is 12,460× less exact**, which changed the evaluation
  protocol to load adapters unmerged.
- Our DPO loss: verified to give exactly `ln 2 = 0.6931471805599453` at
  initialisation and exactly `0.000e+00` implicit reward when π = π_ref. Those
  are now **pre-flight assertions that refuse to train** if violated.

Writing it twice is how you find out that the library's convenient default
isn't what you assumed. Writing it once and trusting it is how Phase 3's
completion-only-loss flag would have silently resolved to `False`.

---

## Q9 (challenge). "Your LLM judge failed. Why report it at all?"

Because the failure is the most informative thing the judge produced.

Base emits `-unstyled` 128 times. SFT writes a correct email. **The judge could
not resolve that comparison at 95% confidence** — 2 of 6 pairs flipped when I
swapped the presentation order, leaving 4 decided verdicts and an interval of
[0.301, 0.954].

That's the two-order protocol working. A single-order judge would have handed me
a clean 6-pair win rate while a third of its verdicts were decided by position.
I'd have reported a number that was one-third noise and never known.

It also calibrates against the literature: MT-Bench measured **GPT-4 at 65%
self-consistency and Claude-v1 at 23.8%**. A 7B judge showing position
sensitivity is expected, not anomalous.

**So the honest conclusion is: my judge cannot serve as ground truth in this
experiment**, and I say that rather than quoting a win rate.

---

## Q10. "What's the engineering you're proudest of?"

The parts that **refuse to run**.

- SFT will not take an optimiser step until the loss mask has been verified
  against an independent computation on a real batch, the prompt prefix is
  consistent across 300 sampled rows, and the truncation audit has run.
- DPO will not train unless the reference model is frozen and the implicit
  reward is exactly zero at initialisation.
- Training will not start if the checkpoints wouldn't fit on disk.
- The evaluation report contains **no function** that can produce an aggregate
  score, and a test asserts it.

Every one of those exists because something went wrong first. E2 measured a
**33.2× lower** loss from a leaking causal mask — a masking bug looks exactly
like spectacular progress, and the only defence is a gate that runs before you
get to see the number.

---

## Q11. "What's the weakest part of the whole project?"

**Everything is in-distribution.** SFT perplexity is measured on the dataset SFT
trained on; preference accuracy on the dataset DPO trained on. Held-out splits,
so no example-level leakage — but the models were optimised toward exactly the
distribution they're scored on, which makes "better" partly circular.

Second: **single seeds**, so no result has a training-variance estimate.

Third: **the E20 confirmation was never run**, which means the project's best
explanation for its most interesting failure remains **NOT CONFIRMED**.

I'd rather state those three plainly than let someone find them.

---

## Explain Back

> **[USER CHECKPOINT — DEFERRED — USER EXPLAIN-BACK REQUIRED]**
>
> Answer Q1, Q3, Q4 and Q9 aloud without notes.
>
> Then take Q5 and argue it from the other side: make the strongest possible
> case that "the metric was length-sensitive" IS an excuse, and say what
> evidence would settle it.
>
> Finally: someone claims "AlignLab shows LoRA is nearly as good as full
> fine-tuning." Correct them using only numbers from this repository.

**Related:** [[project-narrative]] · [[limitations]] · [[experiment-registry]] ·
[[interview-phase7-evaluation]] · [[interview-phase4-peft]]
