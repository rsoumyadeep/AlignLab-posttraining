# Phase 5 Report — Preference Learning / RLHF

**Date:** 2026-08-30 · **Branch:** `phase-5-preference`
**Machine:** `csrslave`, 1 × RTX A6000, bf16
**Model:** `Qwen/Qwen2.5-1.5B` @ `8faed761d45a263340a0528343f099c05c9a4323` (PINNED)

**No DPO was trained. No PPO was implemented. No reward model was trained.**

---

## 1. What Was Built

| Component | Status |
|---|---|
| `src/alignlab/preference.py` — loading, validation, audit, fingerprint, Bradley-Terry | **IMPLEMENTED** |
| `src/alignlab/logprobs.py` — sequence log-probs, log-ratios, exact KL, `k3` estimator | **IMPLEMENTED** |
| `tests/test_logprobs_and_preference.py` — 55 tests | **IMPLEMENTED** |
| E21 — preference data readiness | **MEASURED** |
| E22 — reference model, KL, pre-DPO baseline | **MEASURED** |
| `PYTORCH_CONCEPTS/examples/logprob_and_kl_examples.py` | **EXECUTED** |
| PPO training loop | **CONCEPTUAL — NOT IMPLEMENTED** |
| Reward model training | **CONCEPTUAL — NOT IMPLEMENTED** |
| Value model | **CONCEPTUAL — NOT IMPLEMENTED** |
| DPO loss | **NOT IMPLEMENTED — Phase 6** |

Tests: **405 → 460** (458 passed / 2 skipped local).

---

## 2. What Was Verified

- **Split names**, against the live builder: the preference splits are
  `train_prefs` / `test_prefs`, **not** `train`/`test`. Phase 3 shipped a bug
  from guessing these; this time they were checked.
- **Schema**: `prompt, prompt_id, chosen, rejected, messages, score_chosen,
  score_rejected`. A missing column raises rather than adapting silently.
- **Chosen and rejected share their prompt** — 3000/3000 raw, 0 mismatches in
  2,000 filtered. This is the property that makes `log Z(x)` cancel in the DPO
  derivation, so it is load-bearing, not cosmetic.
- **Prompt tokens are a genuine token-prefix of both full sequences** —
  **0 violations in 400 sampled pairs, on both responses**, after whitespace
  stripping. Phase 3 found 1.5% of no_robots rows violating this.
- **The reference-model wiring**: with `policy == reference`, DPO's implicit
  reward is **exactly `0.000e+00`** on every sequence checked.
- **Reproducibility**: two independent loads produce identical train and eval
  fingerprints.
- **Log-probability arithmetic**: the shift, sum-vs-mean, and the contributing-
  token count are all unit-tested, including the case that produced a wrong
  number in Phase 3's E12.
- **KL properties**: zero at equality, non-negative, **not symmetric**, and
  monotone in policy displacement — asserted, not assumed.
- **Bradley-Terry**: only differences are identified; the loss saturates;
  `loss(0,0) = log 2`.

---

## 3. What Was Measured

### Preference data — `HuggingFaceH4/ultrafeedback_binarized` (E21)

| property | value |
|---|---:|
| raw `train_prefs` rows | 61,135 |
| raw `test_prefs` rows | 2,000 |
| **score ties** (chosen == rejected) | **11.9%** train, 13.6% eval |
| inverted (chosen < rejected) | **0.0%** |
| **identical** chosen/rejected text | 14 in 3,000 |
| empty responses | 2 in 3,000 |
| multi-turn | 0 |
| score margin, mean / median | 1.882 / 1.000 |
| chosen longer, **characters** | 55.8% (1291 vs 1132) |
| chosen longer, **tokens** | **56.5%** (283.7 vs 241.6) |
| prompt tokens, mean | 176.0 |
| pairs over `max_length=1024` | 26/400 (6.5%) |
| train fingerprint | `1e48de32d0d5ece3b345948b8727cdfd…` |
| eval fingerprint | `a84516e89aa3015fec1d22c042c1b8d4…` |

### KL and the pre-DPO baseline (E22, 108 usable pairs)

| policy | KL(policy ‖ base) mean | **median** | max |
|---|---:|---:|---:|
| full SFT | 0.6547 | **0.2044** | 7.3349 |
| LoRA @2e-4 | 0.6233 | 0.2008 | — |
| QLoRA @2e-4 | 0.5564 | 0.1589 | — |

The distribution is **heavily right-skewed** (mean ≈ 3× median), so the median
is the number to quote: one epoch of SFT moved a typical token's distribution
by ~**0.20 nats**.

| model | prefers chosen, **SUM** | prefers chosen, **MEAN** |
|---|---:|---:|
| full SFT | **51/108 = 47.2%** | **63/108 = 58.3%** |
| LoRA @2e-4 | 50/108 = 46.3% | 62/108 = 57.4% |
| QLoRA @2e-4 | 50/108 = 46.3% | 62/108 = 57.4% |

### Executed examples

`k1` has **194×** the variance of `k3`. `k3` shows 11 negatives in 200,000
float32 samples (min −2.980e−08) and **0** in float64.

---

## 4. Hypothesis Outcomes

**E21:** H1 (shared prompt) **HOLDS** · H2 (token-prefix, both responses)
**HOLDS** · H3 (length bias in tokens) **HOLDS** · H4 (reproducible)
**HOLDS**. → **Preference data READY for Phase 6.**

**E22:**

| | outcome |
|---|---|
| H1 KL(SFT‖base) > 0 | **HOLDS**, with a caveat the criterion hid (see §6b) |
| H2 KL(LoRA) < KL(SFT) | **HOLDS** — 0.6233 < 0.6547 |
| H3 implicit reward = 0 when π == π_ref | **HOLDS EXACTLY** — `0.000e+00` |
| H4 SFT already prefers chosen >50% by SUM | **DISPROVED — 47.2%, below chance** |
| H5 MEAN favours chosen more than SUM | **HOLDS** — 58.3% vs 47.2% |

---

## 5. The Result That Changes How Phase 6 Must Be Read

**H4's failure is the most consequential Phase 5 finding.**

By the **summed** log-probability — the published DPO objective — the SFT model
prefers the chosen response on only **47.2%** of pairs. **Below chance.**
Length-normalising the same comparison, on identical models and identical data,
gives **58.3%**. An **11.1-point swing**.

**The cause is length, not quality.** Chosen responses are 56.5% longer in
tokens, and a summed log-probability is more negative for a longer sequence
*simply for being longer*. The SUM comparison is biased **against** chosen —
hard enough to push it under 50%.

**Three consequences for Phase 6:**

1. **The baseline is 47.2%, not 58.3%**, if Phase 6 trains the published
   objective. Quoting the length-normalised figure as the baseline while
   training the summed one would **understate** DPO's improvement.
2. **"DPO raised preference accuracy from 47% to X%" needs the length caveat.**
   Part of any gain may be the model learning to be *longer*, not better.
3. **Phase 6 must record which variant it optimises.** They are different
   objectives that share a name.

This is the same lesson as Phase 3's masked-loss finding and Phase 4's
stop-token finding: **a single headline number can be dominated by a property
of the data rather than of the model.**

---

## 6. Bugs / Corrections

**6a. A test that mis-indexed the shift — caught on first run.**
`test_values_match_a_hand_computed_log_softmax` expected `picked[0] == 0` for a
leading `IGNORE_INDEX`. Wrong: a leading ignore is **dropped by the shift, not
masked**. **The module was right; the test was wrong** — precisely the
off-by-one `token_logprobs` exists to centralise.

**6b. E22's H1 criterion could not test what it claimed.** The hypothesis said
KL would be "small"; the code tested `0 < mean < 1`. It passed, but the
distribution is heavily right-skewed (mean 0.6547, median 0.2044, max 7.3349),
so the mean is not the typical value and "small" was doing unearned work. Now
reports all three and states the skew.

**6c. A claim of mine that the executed example falsified.** I wrote that `k3`
is *always* non-negative; the check printed `False`. Working it out:
`f(x) = e^{−x} − 1 + x` has `f(0)=0` and `f'(x) = 1 − e^{−x}`, so `x=0` is its
minimum and `f(x) ≥ 0` for all real `x` — it **is** non-negative
mathematically. The 11 float32 negatives in 200,000 samples are catastrophic
cancellation at tiny `|k1|` (magnitude ~3e−08) and vanish in float64. Practical
consequence recorded: **do not assert `kl >= 0` strictly in float32.**

**6d. A storage estimate that measured the wrong quantity.** Pre-flight
estimated **≤620 MiB** — the *download* size from the Hub API. Actual disk was
**+1.1 GiB**, because parquet expands into an arrow cache. Not wrong so much as
measuring a different thing. Future pre-flights should budget for the
decompressed form.

---

## 7. Resources Actually Inspected

**Four papers, PARTIALLY INSPECTED** — abstracts fetched programmatically, full
PDFs **NOT** read: DPO (2305.18290), InstructGPT (2203.02155), PPO
(1707.06347), Deep RL from Human Preferences (1706.03741).

**Two ACTUALLY INSPECTED by execution:** the UltraFeedback dataset (schema,
splits, 3,000-row content audit) and our own Phase 3/4 checkpoints.

**Stated plainly in the audit:** the **PPO clipped-surrogate formula** and the
**DPO derivation** in our study note were written from standing knowledge, **not
taken from those papers** — the abstracts do not contain them. Both are labelled
CONCEPTUAL, and the interview material says so.

**NOT INSPECTED:** all four papers in full; Bradley & Terry (1952); TRL's
`DPOTrainer` and `PPOTrainer` source; IPO/KTO/ORPO. **No claim here rests on
them.** IPO/KTO/ORPO are flagged as a gap worth closing before Phase 6, since
they exist largely to address the two data problems Phase 5 measured.

**Claims we did NOT verify:** DPO's "stable, performant, computationally
lightweight" — we ran no DPO. InstructGPT's 1.3B-beats-175B — not reproduced.

---

## 8. Documentation Created

| Folder | File |
|---|---|
| `STUDY_WITH_CLAUDE/` | `phase5/01_rlhf_ppo_and_dpo.md` — RLHF, reward modelling, PPO, KL, the DPO derivation, comparisons |
| `CODE_EXPLANATION/` | `phase5/README.md` — data flow, equation→code, design decisions, bugs, test map |
| `PYTORCH_CONCEPTS/` | `pytorch-logprobs-and-kl.md` + executed example |
| `LEARNING_RESOURCES/` | `phase5_resources.md` |
| `INTERVIEW_DEFENSE/` | `phase5_rlhf_dpo.md` — 11 questions, DPO-vs-PPO as the priority artifact |

---

## 9. Conceptual vs Implemented

| Component | Status |
|---|---|
| Preference dataset representation | **IMPLEMENTED** |
| Chosen/rejected handling | **IMPLEMENTED** |
| Bradley-Terry probability & loss | **IMPLEMENTED** (unit-tested; **not** a reward model) |
| Sequence log-probabilities, log-ratios | **IMPLEMENTED** |
| Exact KL + `k3` estimator | **IMPLEMENTED** |
| Reference-model handling | **IMPLEMENTED + VERIFIED** |
| Dataset fingerprinting | **IMPLEMENTED** |
| Reward model (trained) | **CONCEPTUAL — NOT IMPLEMENTED** |
| Value model | **CONCEPTUAL — NOT IMPLEMENTED** |
| PPO objective / clipped surrogate | **CONCEPTUAL — NOT IMPLEMENTED** |
| KL-regularised RL loop | **CONCEPTUAL — NOT IMPLEMENTED** |
| DPO loss & training | **NOT IMPLEMENTED — Phase 6** |

---

## 10. Limitations

1. **No PPO implementation**, so every PPO claim is conceptual. PPO's
   "instability" is read from an abstract, **not measured**.
2. **No reward model trained.** Bradley-Terry exists as tested functions only.
3. **108 usable pairs** for the KL and baseline measurements — small, one seed,
   bf16, **no confidence intervals**. The 47.2% vs 58.3% gap is large enough to
   be robust to that; the LoRA-vs-QLoRA KL differences are **not**.
4. **KL measured against the BASE model**, not the SFT model Phase 6 will use
   as `π_ref`. It gives the constraint a scale; it is not the Phase 6 quantity.
5. **Ties not dropped** (`drop_ties=False`). 11.9% of pairs carry no genuine
   preference signal.
6. **Length bias not addressed**, only measured.
7. **Dataset card not read** — annotation methodology and score provenance are
   not established here.
8. **Only one preference dataset** inspected in depth.
9. **DPO's derivation verified for internal consistency only**, not against the
   paper.

---

## 11. Git / Synchronization State

```
Branch : phase-5-preference   (local + GitHub + server, same commit)
Base   : 08bdf19 (end of Phase 4, on phase-4-peft)
Remote : git@github.com:rsoumyadeep/AlignLab-posttraining.git  [PRIVATE]
Merged : none. main d21c070; phase-1-foundation 71ed4e6;
         phase-3-sft d5cda3a; phase-4-peft 08bdf19
Working tree: clean on both machines
```

No history rewritten, no force-push. **Phase 4 not reopened**; the MLP-target
LoRA experiment remains **NOT CONFIRMED — DEFERRED** exactly as recorded.
**`checkpoint-295` (8.7 GiB) and `final/` (2.9 GiB) preserved** — verified
present. `/data`: **77 GiB free** (−1.1 GiB for the preference dataset).

---

## 12. Deferred USER Explain-Backs

Phase 5 adds **Checkpoints 17–19** (derive DPO from the RLHF optimum; RLHF vs
DPO end to end; the KL constraint), plus `PYTORCH_CONCEPTS` My Understanding,
`LEARNING_RESOURCES` × 6 Own Words, and the `INTERVIEW_DEFENSE` Explain Back.

**Running total: 23 checkpoints + 39 Own Words + Drills 3–5.** All marked
`DEFERRED — USER EXPLAIN-BACK REQUIRED`. None answered on the USER's behalf.

---

## 13. Phase 6 Prerequisites

**Ready:**

1. **Preference data verified READY** — schema, shared prompts, token-prefix
   consistency, fingerprints, reproducibility (E21).
2. **Reference-model handling verified** — implicit reward exactly 0 at
   initialisation (E22).
3. **Log-probability infrastructure implemented and tested** —
   `sequence_logprobs`, `logprob_ratio`, both sum and mean.
4. **Pre-DPO baseline recorded**: 47.2% (SUM) / 58.3% (MEAN).
5. **Bradley-Terry implemented** — the DPO loss is this same loss with
   `β·(log-ratio difference)` in place of the reward difference.
6. **Storage**: 77 GiB free. A DPO run needs policy + frozen reference in
   memory; a LoRA-based DPO would keep the checkpoint small.

**Decisions Phase 6 must make explicitly and record:**

7. **SUM or MEAN.** Not cosmetic — §5. The published objective is SUM.
8. **`β`.** Not yet chosen; no sweep has been run.
9. **`π_ref` = the Phase 3 SFT checkpoint** (standard practice), not the base.
10. **Drop ties or not.** 11.9% of pairs; currently kept.
11. **Full-parameter DPO or LoRA-DPO.** Phase 4 measured the trade-offs;
    note E20's finding that LoRA could not reach the output projection.
12. **`max_length`** — 6.5% of pairs exceed 1024, and truncation removes
    exactly the completion tokens DPO scores.

**Still NOT TESTED:** TRL's `DPOTrainer` (source not read), adapter resume
end-to-end, W&B online.

---

**PHASE 5 STATUS: INFRASTRUCTURE, MEASUREMENTS, VERIFICATION AND DOCUMENTATION
COMPLETE — CONCEPTUAL COMPONENTS LABELLED AS SUCH — USER EXPLAIN-BACK
CHECKPOINTS DEFERRED BY INSTRUCTION.**

**DPO NOT TRAINED. PPO NOT IMPLEMENTED. Phase 6 NOT STARTED.**
