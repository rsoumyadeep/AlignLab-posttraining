# Phase 7 — `eval-full-001` measured results

Every number here is copied from `eval-full-001_dashboard.json` /
`.txt` in this directory, produced by one run of `alignlab.evaluate` on
`csrslave` (NVIDIA RTX A6000, 47.53 GiB, torch 2.6.0+cu124, bf16).

```
run_name         eval-full-001
git_commit       2e2a51f1a2dc5c251ddb7c79b62a847261510bfa   git_dirty: false
timestamp        2026-08-30T09:29:16Z
base model       Qwen/Qwen2.5-1.5B @ 8faed761d45a263340a0528343f099c05c9a4323
perplexity set   HuggingFaceH4/no_robots test, 150 examples, 27,127 completion tokens
preference set   HuggingFaceH4/ultrafeedback_binarized test_prefs, 200 rows -> 184 usable
                 eval_fingerprint b3bc775a6db554cc757152219fc9ffb2f54a016c6dd69635847df970224a7748
generation       6 fixed prompts, greedy, max_new_tokens 256, seed 42
judge            Qwen/Qwen2.5-7B-Instruct @ main (NOT SHA-pinned - see limitations)
```

---

## 1. Per-model metrics

| model | ppl[compl] | ppl[full] | pref SUM | pref MEAN | stop | gen len | distinct-2 |
|---|---:|---:|---:|---:|---:|---:|---:|
| base | 8.824 | 14.993 | 45.7% | 61.4% | 0/6 | 212.3 | 0.523 |
| **SFT** | **7.398** | **12.284** | 46.7% | 58.7% | **6/6** | 117.3 | **0.839** |
| LoRA r=16 | 7.665 | 13.600 | 46.7% | 60.9% | 0/6 | 256.0 | 0.449 |
| QLoRA r=16 | 7.733 | 13.732 | 46.7% | 61.4% | 0/6 | 256.0 | 0.572 |
| DPO b=0.1 | 7.398 | 12.287 | 46.7% | 58.7% | **6/6** | 118.2 | 0.831 |

**No aggregate score is produced.** The dashboard has no function that emits
one, and a test asserts it.

---

## 2. The four findings

### 2a. SFT is the only training stage that changed anything visible

base → SFT: completion perplexity **8.824 → 7.398 (−16.2%)**, stop-token rate
**0/6 → 6/6 (RESOLVED)**, generated length **212.3 → 117.3**, distinct-2
**0.523 → 0.839**.

The base model's failures are qualitative and severe. Verbatim, greedy, from
the JSON:

```
prompt: Write a two-sentence summary of why the sky appears blue.
base  : .DrawString("Sky appears blue due to Rayleigh scattering, ...", 10, 100);
        .DrawString("Sky appears blue due to Rayleigh scattering, ...", 10, 100);
        [repeats to the 256-token cap]

prompt: Write a short, friendly email declining a meeting invitation.
base  : -unstyled
        -unstyled
        [x ~128, to the cap]

prompt: List three practical tips for someone learning to cook.
base  : [answers in Chinese]
```

SFT answers all six correctly, in English, and stops.

### 2b. The PEFT arms reproduce Phase 4's termination failure — and the new
### generations show *why* it is worse than "runs long"

LoRA and QLoRA hit the 256-token cap on **6/6** prompts (min = max = median =
256). Their answers begin correctly and then do not stop. What they emit after
the answer is new information:

```
LoRA r=16, email prompt:
  Dear [Name], ... Best regards, [Your Name] комф
  You are a helpful assistant.Can ...

QLoRA r=16, email prompt:
  Dear [Name], ... Best regards, [Your Name] комф
  You are a helpful assistant.TRGL
  You are a helpful assistant.TRGL
  You are a helpful assistant.TRGL ...
```

**They fall back into the chat template.** Having never learned to emit
`<|im_end|>`, the most likely continuation after a finished answer is the next
ChatML turn — so the model regurgitates the system prompt. That is exactly the
behaviour predicted by Phase 4's structural explanation (`lm_head` is tied to
the embedding and therefore outside LoRA's target set), now visible in the text
rather than only in a rate.

distinct-2 ranks them lowest of all five models: **LoRA 0.449**, QLoRA 0.572.

### 2c. DPO changed the weights, and changed no metric

| | SFT | DPO b=0.1 | Δ |
|---|---:|---:|---:|
| ppl[completion] | 7.3984 | 7.3978 | **−0.0006** |
| ppl[full] | 12.2839 | 12.2866 | +0.0027 |
| preference SUM | 0.4674 | 0.4674 | **+0.0000** |
| preference MEAN | 0.5870 | 0.5870 | **+0.0000** |
| stop rate | 6/6 | 6/6 | 0 |
| generated length | 117.3 | 118.2 | +0.8 |

**The weights did change** — perplexity moved in the 4th decimal, and greedy
generation differs on **4 of 6** prompts (e.g. SFT: *"because of the scattering
of sunlight by the atmosphere"*; DPO: *"because of Rayleigh scattering"*); the
other 2 are byte-identical. But on every aggregate metric the change is nil.

This is an **independent replication of the Phase 6 null result**, on a
different evaluation path, with a different eval subset, run by different code.
The preregistered Phase 6 conclusion stands unchanged:

> Under the preregistered configuration and training budget, DPO did not change
> SUM preference accuracy on the evaluation set.

Phase 6's measured explanation also holds here: the KL budget was ~250× smaller
than SFT's, and the SUM metric carries a ~30-nat length gap.

### 2d. The SUM/MEAN inversion reproduces on all five models

| model | chosen/token | rejected/token | SUM gap | explained by length | residual |
|---|---:|---:|---:|---:|---:|
| base | −1.1470 | −1.1607 | −30.57 | −34.28 | **+3.71** |
| SFT | −1.0713 | −1.0803 | −29.47 | −31.90 | **+2.43** |
| LoRA r=16 | −1.1001 | −1.1098 | −30.13 | −32.78 | **+2.65** |
| QLoRA r=16 | −1.0924 | −1.1005 | −30.30 | −32.50 | **+2.20** |
| DPO b=0.1 | −1.0712 | −1.0804 | −29.43 | −31.91 | **+2.48** |

**Every model — including the untrained base — prefers the chosen response per
token, and every model's SUM comparison inverts that verdict.** The residual is
positive in all five rows; the length term is larger than the gap in all five.

This is the strongest form of the Phase 5/6 finding yet measured. The
inversion is not a property of any training stage. It is a property of the
metric.

---

## 3. LLM-as-judge

| | base vs SFT | SFT vs DPO b=0.1 |
|---|---|---|
| pairs | 6 | 6 |
| a / b / tie | 1 / 3 / 0 | 1 / 1 / 3 |
| **inconsistent (excluded)** | **2** | **1** |
| decided | 4 | 2 |
| position bias rate | **33.3%** | 16.7% |
| win rate (B, decided only) | 0.750 [0.301, 0.954] | 0.500 [0.095, 0.905] |
| verdict | **NOT resolvable at n=4** | **NOT resolvable at n=2** |
| unparsed replies | 0 | 0 |

### 3a. The judge did not confirm what every other metric shows

Base produces `.DrawString(...)` loops and `-unstyled` repeated to the cap;
SFT produces correct English answers. The judge still failed to resolve the
comparison at 95% confidence, because **2 of 6 pairs were position-flips and
were excluded**, leaving n=4.

**That is the protocol working, not failing.** A one-order judge would have
reported a clean win rate over 6 pairs and hidden the fact that a third of its
verdicts were determined by presentation order.

The excluded pairs are instructive. On the cooking prompt (base answered in
Chinese, SFT in English) the judge replied `A` in **both** orders — that is,
it picked whichever answer it saw first, twice. Position, not preference.

### 3b. An accidental positive control that passed

Two of the six SFT-vs-DPO pairs produced **byte-identical** answers, and a
third differed by two words. The judge returned `TIE` on all three, **in both
presentation orders**. It was not asked to; nothing in the prompt mentions
identical inputs.

A judge answering at random would not do this. It is weak evidence — n=3, one
judge — but it is evidence the judge is reading the answers, and it was not
designed in.

### 3c. What the position-bias rates do and do not say

The MT-Bench paper (arXiv 2306.05685, Table 2) reports GPT-4 consistent on
**65.0%** of pairs and Claude-v1 on **23.8%**. Our rates imply consistency of
66.7% and 83.3%.

**We do not claim our 7B judge matches GPT-4.** Two reasons:

1. **n = 6 per comparison.** A Wilson interval on 4/6 is enormous.
2. The 83.3% figure is **inflated by construction** — two of those six pairs
   had byte-identical answers and a third differed by two words, and a judge is
   trivially consistent when both options are the same text. The base-vs-SFT figure (33.3% bias on six genuinely
   different pairs) is the only one measuring anything.

---

## 4. Reconciliation with the Phase 4 numbers

The Phase 4 report gives base 8.541 and SFT 7.199. This run gives **8.824** and
**7.398**. Both are correct; they are means over **different token
populations**:

| | Phase 4 (E19) | Phase 7 (eval-full-001) |
|---|---:|---:|
| completion tokens | 38,831 (full test split) | 27,127 (`max_examples: 150`) |
| base | 8.5408 | 8.8241 |
| SFT | 7.1992 | 7.3984 |
| LoRA r=16 @2e-4 | 7.4491 | 7.6653 |
| QLoRA r=16 @2e-4 | 7.5112 | 7.7329 |

**This is Phase 3's own lesson recurring across phases** — a perplexity is a
mean over a token population, and changing the population changes the number.
Neither figure supersedes the other, and quoting them side by side without the
denominator would be exactly the error the `region` field was added to prevent.

**What replicates is the comparison, which is what matters:**

| | Phase 4 | Phase 7 |
|---|---:|---:|
| base → SFT | −15.7% | −16.2% |
| SFT → LoRA | +3.5% | +3.6% |
| SFT → QLoRA | +4.3% | +4.5% |

The relative structure is stable to within 0.5 percentage points across two
different eval subsets. The Phase 4 conclusion — perplexity separates the arms
by ~3.5% while stop behaviour separates them completely — holds, and the
stop-behaviour gap reproduced at 0/6 vs 6/6.

**No Phase 3–6 document has been edited.** The earlier numbers stand as
measured; this section records the relationship.

---

## 5. Bug found by this run

`dataset_fingerprint` was **null in all five provenance records**, and a single
`eval_fingerprint` field named the preference rows while sitting beside the
perplexity numbers as well — so a reader would attribute the perplexity to the
wrong dataset.

Fixed in `evaluate.py`: each metric family now records its own dataset, split,
row count, rows dropped and sha256, under `extras.fingerprints`. Four tests in
`TestProvenanceFingerprints` assert that no field is silently null for an
enabled family and that a disabled family reports `NOT_MEASURED` rather than
`None`.

**The dashboard in this directory is the pre-fix artefact and is left as
produced.** The run is still fully traceable — its top-level `protocol` block
carries both fingerprints separately:

```
lm_eval_fingerprint  f92fdec349b9c8ed92fb5aacefc23a7cbcbb1ec733da1c662c911e4c58b71fae   no_robots test, 150
eval_fingerprint     b3bc775a6db554cc757152219fc9ffb2f54a016c6dd69635847df970224a7748   ultrafeedback test_prefs, 200
```

and both are logged in `eval-full-001_run.log`. What was missing is the
**per-model** provenance record, which named only one of them and left
`dataset_fingerprint` null.

---

## 6. What this run establishes, and what it does not

**Establishes:**

- SFT taught termination — 0/6 → 6/6, intervals disjoint, RESOLVED.
- SFT improved in-distribution completion perplexity by 16.2%.
- LoRA and QLoRA did not learn to terminate, and degrade into template
  regurgitation when they run past the answer.
- DPO b=0.1 changed no aggregate metric — replicating Phase 6 independently.
- The SUM preference metric is length-dominated on all five models, base
  included.

**Does not establish:**

- That any model is *generally* better. Every metric is measured on the
  distribution the models trained on.
- That SFT beats base *in the judge's opinion* — n=4 decided, interval includes
  0.5.
- Anything about the judge's agreement with human preference. No human study
  was run.
- Any out-of-distribution behaviour whatsoever.

**Related:** [[phase7-evaluation-theory]] · [[phase7-code-explanation]] ·
[[interview-phase7-evaluation]] · [[phase7-resources]]
