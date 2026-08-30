# AlignLab — What Was Actually Discovered

A post-training pipeline was built end to end: SFT, LoRA, QLoRA, DPO and an
evaluation subsystem, all on `Qwen/Qwen2.5-1.5B` at a pinned revision.

**That is the least interesting part of the project.** The pipeline worked. What
makes the work worth reading is the eight findings below, five of which are
things that turned out **not** to be true, and one of which is a popular
explanation of LoRA that the measured data does not support.

Every number here traces to a committed experiment. Nothing is estimated.

---

## 1. SFT substantially changed termination behaviour

This is the project's one large, unambiguous, behavioural win.

| | base | after SFT |
|---|---|---|
| emitted `<\|im_end\|>` | **0/6** | **6/6** |
| completion perplexity | 8.824 | 7.398 (**−16.2%**) |
| distinct-2 | 0.523 | **0.839** |
| generation | `.DrawString(...)` loops; `-unstyled` × ~128 to the cap; one English prompt answered in Chinese | correct, formatted, terminates |

**Of the 24 pairwise comparisons in the Phase 7 evaluation, only 3 were
resolvable at 95% confidence — and all three were stop-token rates.** The
strongest evidence in the project is behavioural, not a likelihood number.

The base model's failure mode matters: it did *terminate* twice out of six, but
it emitted the ChatML turn terminator **zero** times — it hit `<|endoftext|>`
instead. A single "did it finish" metric would have scored it 33% and hidden
the thing SFT actually taught.

---

## 2. LoRA and QLoRA saved a great deal, and did not learn to stop

| | trainable params | peak VRAM | completion ppl | emitted `<\|im_end\|>` |
|---|---:|---:|---:|---:|
| full SFT | 1,543,714,304 | — | 7.398 | **6/6** |
| LoRA r=16 | **4,358,144** (0.28%) | 5,073,415,168 B | 7.665 | **0/6** |
| QLoRA r=16 | **4,358,144** | **3,095,107,584 B** | 7.733 | **0/6** |

The savings are real: **0.28% of the parameters**, and QLoRA at ~61% of LoRA's
peak VRAM. The perplexity cost is small — **3.6%**.

**And all three PEFT arms were unusable**, because they never stopped. On every
one of six prompts they ran to the 256-token cap (min = max = median = 256).
What they emit past the answer is the diagnostic:

```
Dear [Name], ... Best regards, [Your Name] комф
You are a helpful assistant.TRGL
You are a helpful assistant.TRGL     [to the cap]
```

They fall back into the chat template. With no `<|im_end|>` in their learned
behaviour, the likeliest continuation after a finished answer is the next
ChatML turn — so the model emits the system prompt back at you. `distinct-2`
ranks them last of all five models (LoRA **0.449**, QLoRA 0.572).

> **A likelihood metric separated these models by 3.6%. Usability separated
> them completely.**

---

## 3. E17 challenged the standard explanation of why LoRA works

The usual account: fine-tuning updates are intrinsically low-rank, so a
low-rank parameterisation loses little.

We measured the actual ΔW from full fine-tuning and did an SVD on it.

**H2 — "the real ΔW is approximately low-rank" — was DISPROVED.** The measured
update is not low-rank.

**LoRA still worked.** The popular *explanation* for why did not survive
contact with the data.

This is the single most important negative result in the project, and it
constrains what the project may claim. No AlignLab document says "LoRA works
because fine-tuning updates are low rank," because our own measurement
contradicts it.

Related, from the same phase: explained energy in an SVD uses **squared**
singular values (H1, holds), and the break-even rank where LoRA stops saving
parameters is `r* = (d_in·d_out)/(d_in+d_out)`.

---

## 4. E20 named a better hypothesis — and stopped short of claiming it

If rank is not the explanation, what limited the PEFT arms?

**Reachability.** Emitting a rare token means moving that token's *logit*,
which `lm_head` produces. In Qwen2.5-1.5B `lm_head` is **tied to the embedding**
and therefore sits **outside LoRA's target set** (`q_proj, k_proj, v_proj,
o_proj`). Full fine-tuning moved that matrix by relative **0.0136** — the
largest relative change of any matrix measured in the project.

> **LoRA's ceiling is set by which matrices it can reach, not only by rank. A
> behaviour whose mechanism lives outside the target set is unreachable at any
> rank.**

**Status: NOT CONFIRMED.** The evidence is consistent with the hypothesis; it
is not proof. The decisive test — a LoRA variant adapting the MLP, or an untied
head — is a single ~18-minute run. It is **deferred, not skipped**, and it
remains the most valuable outstanding experiment in the project.

Saying "consistent with" instead of "shows" is the difference between the two
sentences that matter here.

---

## 5. SUM preference metrics are strongly length-sensitive in this dataset

The published DPO objective compares **summed** log-probabilities. On
UltraFeedback, chosen responses are **56.5% longer** in tokens, and summed
log-probability is a sum over tokens.

Phase 7 ran the decomposition on all five models at once:

| model | chosen/token | rejected/token | SUM gap | explained by length | residual |
|---|---:|---:|---:|---:|---:|
| **base (untrained)** | −1.1470 | −1.1607 | −30.57 | −34.28 | **+3.71** |
| SFT | −1.0713 | −1.0803 | −29.47 | −31.90 | **+2.43** |
| LoRA r=16 | −1.1001 | −1.1098 | −30.13 | −32.78 | **+2.65** |
| QLoRA r=16 | −1.0924 | −1.1005 | −30.30 | −32.50 | **+2.20** |
| DPO β=0.1 | −1.0712 | −1.0804 | −29.43 | −31.91 | **+2.48** |

**Every model prefers the chosen response per token. Every model's SUM
comparison inverts that verdict.** The residual is positive in all five rows —
including the **untrained base model**, which has had no preference training of
any kind.

The same comparison, aggregated two ways, on identical models and data:
**SUM 47.2% vs MEAN 58.3%** — an 11.1-point swing straddling chance.

> The inversion is not caused by any training stage. **It is a property of the
> metric.**

---

## 6. DPO did not change preference accuracy under the preregistered budget

β ∈ {0.01, 0.1, 0.5} was **pre-registered before any DPO code existed**
(`docs/phase6/BETA_PREREGISTRATION.md`, committed 13:15:12), so β could not be
selected after seeing results.

| arm | SUM accuracy | MEAN | KL from reference |
|---|---|---|---:|
| SFT (baseline) | 0.4674 (86/184) | 0.5870 | 0.000000 |
| DPO β=0.01 | 0.4674 (86/184) | 0.5870 | 0.000801 |
| DPO β=0.1 | 0.4674 (86/184) | 0.5870 | 0.000799 |
| DPO β=0.5 | 0.4674 (86/184) | 0.5870 | 0.000803 |

**Byte-identical, not merely close.**

> **Under the preregistered configuration and training budget, DPO did not
> change SUM preference accuracy on the evaluation set.**

**This is not "DPO does not work."** It is a statement about this configuration
and this budget, and we can quantify why:

- The SUM metric carries a **~29-nat** length gap. Flipping the average pair
  requires shifting it by ~29 nats; the run achieved **0.04** — **728× short**.
- KL from the reference was **0.0008**, against SFT's **0.2044** — roughly
  **250× under-budget**. The policy barely moved.

A **post-hoc** 10× learning-rate run (labelled POST-HOC, not covered by the
preregistration) shifted the gap by 0.70 nats — still **42× short** — and
produced ~30% longer output, which is the length effect reappearing.

Phase 7 replicated the null independently, on a different evaluation path with
a different eval subset: DPO's perplexity moved in the **4th decimal**, greedy
generation differed on 4 of 6 prompts, and **no aggregate metric changed at
all**.

---

## 7. The LLM judge showed substantial position sensitivity

Every pair is judged **twice**, in both presentation orders. Agreement is a
vote; disagreement means the verdict was *position*, not preference, and the
pair is **excluded** rather than counted as half a win.

| | base vs SFT | SFT vs DPO |
|---|---|---|
| **position-inconsistent** | **2 of 6 (33.3%)** | 1 of 6 (16.7%) |
| decided | 4 | 2 |
| win rate (B) | 0.750 **[0.301, 0.954]** | 0.500 [0.095, 0.905] |
| verdict | **NOT resolvable at n=4** | **NOT resolvable at n=2** |

**Base emits `-unstyled` 128 times; SFT writes a correct email; the judge still
could not resolve the comparison.** On the cooking prompt — base answered in
Chinese — it replied `A` in **both** orders: it picked whichever it saw first,
twice.

That is the protocol working. A one-order judge would have reported a clean
6-pair win rate while a third of its verdicts were decided by presentation
order.

**Calibration from the literature** (MT-Bench, arXiv 2306.05685, Table 2, read
directly): GPT-4 is self-consistent on only **65.0%** of pairs, Claude-v1 on
**23.8%**. A high bias rate from a 7B judge is the expected result. We claim no
comparison with those figures — n=6.

**Therefore the judge cannot be treated as ground truth from this experiment.**
It was never validated against human preference here, and it is from the
**same model family** as the models it judges.

*(One unplanned check passed: two SFT-vs-DPO pairs were byte-identical and a
third differed by two words; the judge returned `TIE` on all three, in both
orders, unprompted. Weak evidence — but it was not designed in.)*

---

## 8. Metrics disagree, and the disagreement is the finding

In **every** phase from 3 to 6, two metrics pointed in different directions:

| phase | metric A said | metric B said | what was true |
|---|---|---|---|
| 3 | loss improved **33.2×** | generation was destroyed | causal-mask leakage |
| 4 | perplexity within **3.5%** | stop behaviour **0/4 vs 4/4** | the PEFT models were unusable |
| 5 | preference **47.2%** | preference **58.3%** | aggregation choice, not quality |
| 6 | accuracy unchanged | reward margin moved, length **+30%** | the budget was 250× short |

**So AlignLab produces no aggregate quality score, and contains no function that
could produce one** — a test asserts the report module exposes nothing matching
`"score"`. A weighted average would have erased all four findings.

Instead, the design encodes each disagreement so it cannot recur silently:

- `PerplexityResult` carries its **token region**, and `comparable_to()` returns
  False across regions *even for identical values*.
- `PreferenceStats` cannot serialise SUM without MEAN, or either without token
  counts; `length_attribution()` runs the decomposition automatically.
- `TerminationStats` keeps *terminated* and *emitted a stop token* separate.
- Every rate carries a **Wilson interval**, and `difference_is_resolvable()`
  returns **no p-value** — it answers one narrow question and says "NOT
  resolvable at this sample size" when the answer is no.

---

## The theme

The pipeline was the easy part. What the project is actually about is that
**five stated hypotheses were disproved**, a **popular explanation of LoRA did
not survive measurement**, a **metric turned out to be measuring token count**,
and the **evaluation instrument was too weak to confirm the project's own
clearest result** — and every one of those is written down, with its original
wording, rather than quietly corrected after the fact.

**Related:** [[experiment-registry]] · [[limitations]] · [[phase7-report]]
