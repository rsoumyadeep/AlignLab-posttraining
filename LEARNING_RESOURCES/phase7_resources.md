# Phase 7 — Resources Actually Inspected

Honest audit per PROJECT_INSTRUCTIONS §9. **All inspections 2026-08-30.**

---

## Resource 1 — the one that mattered

```
Resource:        Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena
Type:            paper (arXiv), abstract + HTML full text
URL / Identifier: arxiv.org/abs/2306.05685 · arxiv.org/html/2306.05685v4
                 (247,563 bytes fetched)
Access Status:   PARTIALLY INSPECTED - abstract in full, plus the position-bias
                 section and Table 2 read directly. The paper was NOT read end
                 to end; the MT-Bench construction, Chatbot Arena analysis and
                 appendices were NOT read.
```

**Claims obtained, quoted:**

- The biases they identify: *"position, verbosity, and self-enhancement
  biases, as well as limited reasoning ability."*
- Their consistency protocol — **identical to the one implemented here**:
  *"Consistency is the percentage of cases where a judge gives consistent
  results when swapping the order of two assistants."*
- **Table 2**, read verbatim:

  | judge | prompt | consistency | biased→first | biased→second | error |
  |---|---|---:|---:|---:|---:|
  | Claude-v1 | default | **23.8%** | 75.0% | 0.0% | 1.2% |
  | GPT-3.5 | default | 46.2% | 50.0% | 1.2% | 2.5% |
  | **GPT-4** | default | **65.0%** | 30.0% | 5.0% | 0.0% |
  | GPT-4 | rename | 66.2% | 28.7% | 5.0% | 0.0% |

- *"The position bias can be very significant. Only GPT-4 outputs consistent
  results in more than 60% of cases."*
- On agreement with humans: *"strong LLM judges like GPT-4 can match both
  controlled and crowdsourced human preferences well, achieving over 80%
  agreement, the same level of agreement between humans."*

### What this changed

**It validated the design after the fact, and it recalibrated my expectations.**

The two-order swap protocol in `evals/judge.py` was written before this paper
was read; the paper's "Consistency" is the same quantity my
`position_bias_rate` complements (`consistency = 1 − position_bias_rate`). That
is independent convergence on the method, not a copy.

**But the numbers are sobering, and they set the prior for our own run:**
GPT-4 — far larger than anything here — is consistent on only **65%** of pairs.
Claude-v1 on **23.8%**. Our judge is Qwen2.5-7B-Instruct, so a **low consistency
rate should be expected, not treated as a bug**, and a high position-bias rate
in our results is corroborated by the literature rather than anomalous.

**Two of their three named biases apply directly to this project:**

- **Verbosity bias** — judges favour longer answers. Phases 5 and 6 measured
  length dominating a preference metric (chosen responses 56.5% longer; a
  ~29-nat SUM gap), and the post-hoc DPO run generated **30% longer** output.
  So a judge preferring the DPO model could be measuring length *again*, by a
  different route. Our judge prompt explicitly instructs *"Ignore which answer
  is longer - length is not quality"*, which is a mitigation, **not a fix** —
  we did not verify it works.
- **Self-enhancement bias** — judges favour their own outputs. Our judge is
  Qwen2.5-7B-Instruct and the models under test are Qwen2.5-1.5B derivatives:
  **same family**. This is the closest thing to self-enhancement bias short of
  self-judging, and it is recorded in every `JudgeResult`.

**What we did NOT adopt:** their "rename" prompt variant, which improved
Claude-v1's consistency from 23.8% to 56.2%. Worth testing; **not tested here**.

**What we cannot claim:** the >80% human-agreement result is for **GPT-4**. It
says nothing about a 7B judge, and **no human agreement study was run in this
project**.

---

## Resource 2

```
Resource:        HuggingFaceH4/no_robots  and
                 HuggingFaceH4/ultrafeedback_binarized
Type:            datasets
Access Status:   ACTUALLY INSPECTED - carried forward from Phases 3 and 5,
                 unchanged, same fingerprints
Claims Obtained: no_robots test split used for perplexity (IN-DISTRIBUTION for
                 the SFT model); ultrafeedback test_prefs for preference
                 accuracy (IN-DISTRIBUTION for the DPO models). 11.9% ties,
                 chosen 56.5% longer in tokens.
Limitations:     both are the distributions the models were trained on, so
                 every number here measures fit to training distribution.
                 Neither dataset CARD was read.
```

---

## Resource 3

```
Resource:        Qwen/Qwen2.5-7B-Instruct  (the judge)
Type:            model
Access Status:   ACTUALLY INSPECTED - downloaded and executed
Claims Obtained (measured by us): 14.20 GiB across 11 files at revision main;
                 159 s to download; runs in bf16 on one A6000.
Limitations:     revision pinned to "main", NOT a commit SHA - a deviation from
                 this project's practice everywhere else, and it means a future
                 rerun could get a different judge. Recorded as a limitation.
                 Its agreement with human preference is UNVERIFIED here.
```

---

## Resource 4

```
Resource:        Wilson score interval for a binomial proportion
Type:            statistical method
Access Status:   NOT READ FROM A SOURCE - implemented from standing knowledge
                 and VERIFIED NUMERICALLY instead
What Was Verified: the 95% interval for 50/100 is [0.4038, 0.5962]; intervals
                 stay within [0,1] at 0/10, 10/10 and 1/200 where the normal
                 approximation escapes; width decreases monotonically with n.
Limitations:     Wilson (1927) was NOT consulted. The implementation is
                 asserted on the strength of the numerical checks in
                 tests/test_eval_metrics.py, not on a citation. No coverage
                 simulation was run.
```

---

## NOT INSPECTED

| Resource | Why it would matter |
|---|---|
| MT-Bench paper, full text | their prompt designs, the Chatbot Arena analysis, the reference-guided judging variant |
| AlpacaEval / length-controlled AlpacaEval | length-controlled win rates address exactly the verbosity bias we measured |
| HELM, lm-evaluation-harness | standard evaluation suites; ours is bespoke and narrower |
| Wilson (1927); Brown, Cai & DasGupta (2001) | the coverage properties we assert numerically |
| Any out-of-distribution benchmark | the single biggest gap in this evaluation |

**No claim in this repository rests on any of these.** In particular we make
**no claim** that our judge agrees with human preference, and **no claim** that
our metrics generalise beyond the training distributions.

**The gap that matters most for Phase 8:** length-controlled evaluation.
AlpacaEval's length-controlled variant exists precisely because of the bias
Phases 5–7 kept measuring. **NOT INSPECTED.**

---

## Own Words

> ## Own Words
> **[USER CHECKPOINT — DEFERRED — USER MUST WRITE THIS]**
> One entry per resource above, after reading it yourself.

**Related:** [[phase7-evaluation-theory]] · [[phase6-resources]] ·
[[interview-phase7-evaluation]]
