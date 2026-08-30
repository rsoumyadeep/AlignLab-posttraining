# Phase 6 — Resources Actually Inspected

Honest audit per PROJECT_INSTRUCTIONS §9. **All inspections 2026-08-30.**

**Status upgrade from Phase 5, stated plainly.** In Phase 5 the DPO paper was
recorded as PARTIALLY INSPECTED — abstract only — and the derivation in
`STUDY_WITH_CLAUDE/phase5` was explicitly labelled as written from standing
knowledge and **not** cited to the paper. For Phase 6 the paper's **HTML full
text was fetched and specific sections were read**, so parts of that record can
now be upgraded. The upgrade is section-by-section, not wholesale: the sections
listed below were read, the rest were not.

---

## Resource 1

```
Resource:        Direct Preference Optimization: Your Language Model is
                 Secretly a Reward Model
Type:            paper (arXiv), HTML full text
URL / Identifier: arxiv.org/html/2305.18290v3   (407,412 bytes fetched)
Access Status:   PARTIALLY INSPECTED - specific sections read in full;
                 the paper was NOT read end to end
What Was Inspected:
  - the complete section/appendix heading list
  - Section 3 Preliminaries: Eq. 3 (the KL-constrained RLHF objective) and
    Eq. 4 (the closed-form optimum) with the surrounding text
  - Appendix B: DPO Implementation Details and Hyperparameters, including the
    reference PyTorch code, IN FULL
  - Appendix A.4 heading and the opening of the gradient derivation
What Was NOT Inspected:
  - Sections 1, 2, 5, 6, 7 (introduction, related work, theory, experiments,
    discussion)
  - Appendices A.1, A.2, A.3, A.5, A.6 - the derivations themselves
  - Appendix C and D
```

**Claims obtained, quoted from the text:**

- Eq. 4, the closed-form optimum:
  `π_r(y|x) = (1/Z(x)) π_ref(y|x) exp((1/β) r(x,y))`, where
  `Z(x) = Σ_y π_ref(y|x) exp((1/β) r(x,y))` is the partition function.
  *"See Appendix A.1 for a complete derivation."*
- *"it is still expensive to estimate the partition function Z(x), which makes
  this representation hard to utilize in practice. However, we can rearrange
  Eq. 4 to express the reward function in terms of its corresponding optimal
  policy"*
- Eq. 3, on β: *"where β is a parameter controlling the deviation from the base
  reference policy π_ref, namely the initial SFT model π_SFT. In practice, the
  language model policy π_θ is also initialized to π_SFT."*
- On why the constraint matters: *"it prevents the model from deviating too far
  from the distribution on which the reward model is accurate, as well as
  maintaining the generation diversity and preventing mode-collapse"*
- **Appendix B reference implementation, verbatim:**
  ```python
  pi_logratios  = pi_yw_logps - pi_yl_logps
  ref_logratios = ref_yw_logps - ref_yl_logps
  losses = -F.logsigmoid(beta * (pi_logratios - ref_logratios))
  rewards = beta * (pi_logps - ref_logps).detach()
  ```
- **Hyperparameters:** *"Unless noted otherwise, we use β = 0.1, batch size of
  64 and the RMSprop optimizer with a learning rate of 1e-6 by default. We
  linearly warmup the learning rate from 0 to 1e-6 over 150 steps. For TL;DR
  summarization, we use β = 0.5."*

### What this changed, and what it did not

**Verified against our implementation.** `tests/test_dpo.py::TestMatchesPaper
ReferenceImplementation` asserts our loss equals the appendix's reference code
to **1e-12** at β ∈ {0.01, 0.1, 0.5, 1.0}. Our grouping (per-response implicit
rewards, then their difference) differs from theirs (log-ratios, then their
difference) but is algebraically identical — and ours was written **before** the
appendix was read.

**Corroboration of the pre-registration, NOT a change to it.** The paper's
default β = 0.1 and its TL;DR β = 0.5 are both in our pre-registered
{0.01, 0.1, 0.5}. That set was committed at 13:15:12, before any DPO code
existed and before this appendix was read; the sweep was already running when
it was. Nothing was adjusted.

**A difference we did not adopt:** the paper uses RMSprop at lr 1e-6 with batch
64. We use AdamW at 5e-7 with an effective batch of 16 — same order of
magnitude for the learning rate, different optimizer and batch. This is a
**deviation**, recorded rather than hidden, and it means our results are not a
replication of theirs.

**Something read but NOT resolved.** The rendered Eq. 21 in Appendix A.4 shows
the gradient's sigmoid argument with `y_l` first and `y_w` second — the reverse
of Eq. 7's ordering. This may be an HTML rendering artefact, a sign convention
internal to that derivation, or an erratum. **We did not resolve it**, and
nothing in our implementation depends on it: our loss is verified against
Appendix B's code, not against Eq. 21.

---

## Resource 2

```
Resource:        HuggingFaceH4/ultrafeedback_binarized
Type:            dataset
Access Status:   ACTUALLY INSPECTED - carried forward from Phase 5, unchanged
Claims Obtained: 61,135 train_prefs rows; 11.9% score ties; 0% inverted;
                 14/3000 identical chosen-rejected pairs; 2/3000 empty;
                 3000/3000 share a prompt; chosen longer in 56.5% of pairs by
                 tokens (283.7 vs 241.6 mean)
Limitations:     the dataset CARD was still not read; annotation methodology
                 and score provenance remain unestablished.
```

The dataset, filtering and seed are **unchanged from Phase 5**, so the
fingerprints match. Changing any of them would change the fingerprint, which
the manifest records.

---

## Resource 3

```
Resource:        Our own Phase 3 SFT checkpoint and Phase 5 measurements
Type:            artifacts, inspected by execution
Access Status:   ACTUALLY INSPECTED - measured
Claims Obtained: pre-DPO preference baseline re-measured in-process at
                 46.74% SUM (Phase 5 measured 47.2% on a different subsample -
                 consistent); reference-frozen and zero-implicit-reward checks
                 both pass inside the DPO code path
                 (max_abs_reward = 0.0, loss = 0.6931471824645996 = log 2)
Limitations:     one seed; bf16; a single dataset.
```

---

## NOT INSPECTED

| Resource | Why it would matter |
|---|---|
| DPO paper §5 (theory) | the Lemmas and Theorem 1 behind the reparameterisation |
| DPO paper §6 (experiments) | their empirical claims — stability, win rates |
| Appendices A.1/A.2 | the derivations themselves; ours is still independent |
| TRL `DPOTrainer` source | not used, and not read |
| IPO / KTO / ORPO | variants addressing the tie and length issues we measured |
| Bradley & Terry (1952) | the original preference model |
| RSO, SLiC-HF | alternative preference-optimisation objectives |

**No claim in this repository rests on any of these.** In particular, we make
**no claim about DPO's stability or win rates** — those are §6 results we did
not read and did not reproduce.

**The gap that matters most before Phase 7:** IPO and KTO exist substantially to
address the two data problems we measured (11.9% ties, 56.5% length bias).
Reading them would place our length findings in context. **NOT INSPECTED.**

---

## Own Words

> ## Own Words
> **[USER CHECKPOINT — DEFERRED — USER MUST WRITE THIS]**
> One entry per resource above, after reading it yourself.

**Related:** [[phase6-dpo-theory-and-results]] · [[phase5-resources]] ·
[[interview-phase6-dpo]]
